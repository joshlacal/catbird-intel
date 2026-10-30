#!/usr/bin/env python3
"""Oct 22 Local For You Preview — Offline Check & S17 Preview Candidate Bank.

Implements Whole-Network Signals Plan §3.3 (lines 314-320) & Preregistration §9.8 / A1:
1. Build pre-Oct-22 InterestedIn (top-10 communities, social_proof >= 2, weights 2.0 follow +
   1.0 like + 2.0 repost per V8(i), excluding courtesy/self likes and heavy followers >= 3,000)
   for Oct 22 light engagers using ONLY Oct 19-21 netcup-export + out-full-20261024/edges.bin.
2. Filter 2026-10-22/posts.parquet to clean original posts (is_deleted == 0, reply_parent == '',
   reply_root == '', no post self_labels) by S13-eligible, live-ok producers NOT in
   hide_from_recs_union.tsv (Amendment A1 posts-only exclusion) or base_safety_exclude_dids.txt.
3. Score every candidate post x in its author's KnownFor communities c by the exact §3.3 residual:
     residual(x, c) = N_c(x) - p_c * N_all(x)
   where N_all(x) is distinct light non-courtesy Oct 22 engagers (with pre-Oct-22 InterestedIn)
   and N_c(x) is the subset whose InterestedIn includes c.
4. Run exact leave-one-out offline check on 10,000 sampled active Oct 22 users (with >=1 real
   Oct 22 like on a non-followed eligible producer's clean original post):
   Verify Recall@50 (<=2 posts/author, <=5% high-H community cap) beats BOTH:
     (a) global_popularity (N_all,-u(x) across all non-followed eligible originals), AND
     (b) community_popularity_without_residual (both raw post popularity in InterestedIn
         communities without the residual AND author follower popularity in InterestedIn
         communities), with 2,000-resample user-bootstrap 95% CI > 0.
5. Emit ~/catbird-lake-local/s16-s17-20261030/oct22_preview_posts.parquet (top-60 per community,
   <=2 per author) and preview_oct22_report.json.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import pathlib
import re
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

POL_RE = re.compile(
    r"\b(trump|biden|kamala|harris|vance|walz|desantis|obama|pelosi|schumer|mcconnell|aoc|"
    r"bernie|sanders|maga|democrat|democrats|republican|republicans|gop|dnc|rnc|liberal|"
    r"liberals|conservative|conservatives|leftist|right-wing|far-right|fascist|fascism|"
    r"authoritarian|antifa|woke|dei|scotus|supreme court|congress|senate|filibuster|impeach|"
    r"ballot|midterms|electoral|polling|pollster|gerrymander|white house|doj|ice raids|"
    r"deportation|tariffs|zionist|intifada|netanyahu|hamas|idf|gaza|kremlin|putin|zelensky|"
    r"nato|starmer|tories|labour party)\b",
    re.I,
)


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def lower_bound_u32(col: np.ndarray, vals: np.ndarray) -> np.ndarray:
    n = len(col)
    lo = np.zeros(len(vals), dtype=np.int64)
    hi = np.full(len(vals), n, dtype=np.int64)
    for _ in range(34):
        act = lo < hi
        if not act.any():
            break
        mid = (lo + hi) >> 1
        less = col[np.minimum(mid, n - 1)].astype(np.int64) < vals
        lo = np.where(act & less, mid + 1, lo)
        hi = np.where(act & ~less, mid, hi)
    return lo


def boot_diff_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 20261030) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    out = np.empty(n_boot, dtype=np.float64)
    for b0 in range(0, n_boot, 100):
        nb = min(100, n_boot - b0)
        idx = rng.integers(0, n, size=(nb, n))
        out[b0 : b0 + nb] = diff[idx].mean(axis=1)
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def select_top_k_with_caps(
    post_ids: np.ndarray,
    scores: np.ndarray,
    post_author: np.ndarray,
    post_comm0: np.ndarray,
    high_h_comms: set[int] | None,
    k: int = 50,
    max_per_author: int = 2,
    max_high_h: int = 2,
) -> list[int]:
    """Select top-k posts by descending score (deterministic tiebreak by post_id), <=2/author, <=5% high-H."""
    if len(post_ids) == 0:
        return []
    # Pre-take top min(len, 600) via argpartition for speed, then exact sort
    if len(post_ids) > 600:
        part_idx = np.argpartition(-scores, 600)[:600]
        sub_p = post_ids[part_idx]
        sub_s = scores[part_idx]
    else:
        sub_p = post_ids
        sub_s = scores
    ord_i = np.lexsort((sub_p, -sub_s))
    chosen: list[int] = []
    acnt: dict[int, int] = {}
    hh_cnt = 0
    for pid_x in sub_p[ord_i].tolist():
        au = int(post_author[pid_x])
        if acnt.get(au, 0) >= max_per_author:
            continue
        if high_h_comms:
            c0 = int(post_comm0[pid_x])
            if c0 in high_h_comms:
                if hh_cnt >= max_high_h:
                    continue
                hh_cnt += 1
        acnt[au] = acnt.get(au, 0) + 1
        chosen.append(pid_x)
        if len(chosen) >= k:
            break
    return chosen


def main() -> None:
    t0 = time.time()
    lake = pathlib.Path.home() / "catbird-lake-local"
    out_dir = lake / "s16-s17-20261030"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand5_full_dir = lake / "s3/matched/cand5-full-matched"
    s5_dir = lake / "s5-sweep-20261030"
    s13_dir = lake / "s13-elig-20261030"
    s14_dir = lake / "s14-hostility-20261030"
    gb_dir = lake / "graph-build-v3/out-full-20261024"
    nc_dir = lake / "netcup-export"

    # 1. Load producer universe, KnownFor (cand5-full-matched), S13 eligibility, and post-author exclusions
    full_dids: list[str] = []
    full_fols_list: list[int] = []
    with open(cand5_full_dir / "producers.tsv") as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            full_dids.append(parts[1])
            full_fols_list.append(int(parts[2]))
    n_prod = len(full_dids)
    prod_followers = np.array(full_fols_list, dtype=np.float64)
    did2pid = {d: i for i, d in enumerate(full_dids)}

    purge_dids = {ln.strip() for ln in open(s5_dir / "purge_dids.txt") if ln.startswith("did:")}
    base_excl = {ln.strip() for ln in open(s13_dir / "base_safety_exclude_dids.txt") if ln.startswith("did:")}
    rule2_excl = {ln.strip() for ln in open(s5_dir / "eligibility_rule2_exclude_dids.txt") if ln.startswith("did:")}
    hide_dids = {
        ln.split("\t")[0].strip()
        for ln in open(lake / "s5-declarations-20261027/hide_from_recs_union.tsv")
        if ln.startswith("did:")
    }

    # Live profile check status from s5-sweep-20261030/profiles.parquet
    t_prof = pq.read_table(
        s5_dir / "profiles.parquet",
        columns=[
            "did",
            "found",
            "handle",
            "no_unauthenticated",
            "any_mod_label",
            "hard_label",
            "adult_graphic",
            "self_labels",
            "repo_status",
        ],
    )
    bad_live_dids: set[str] = set()
    ok_live_dids: set[str] = set()
    for r in t_prof.to_pylist():
        d = r["did"]
        is_ok = (
            bool(r["found"])
            and not bool(r["no_unauthenticated"])
            and not bool(r["any_mod_label"])
            and not bool(r["hard_label"])
            and not bool(r["adult_graphic"])
            and len(r["self_labels"] or []) == 0
            and r["handle"] != "handle.invalid"
            and r["repo_status"] == "active"
        )
        if is_ok:
            ok_live_dids.add(d)
        else:
            bad_live_dids.add(d)

    t_s13 = pq.read_table(s13_dir / "s13_prod_eligibility_and_scores.parquet", columns=["pid", "eligible"])
    s13_elig = t_s13["eligible"].to_numpy().astype(bool)

    post_author_excl = purge_dids | base_excl | rule2_excl | hide_dids | bad_live_dids
    allowed_post_pid = np.array(
        [bool(s13_elig[p]) and (full_dids[p] not in post_author_excl) for p in range(n_prod)],
        dtype=bool,
    )

    # KnownFor CSR / fixed 3-slot table per producer (rank 0, 1, 2)
    n_comm = 8500
    kf_comm = np.full((n_prod, 3), -1, dtype=np.int32)
    kf_weight = np.zeros((n_prod, 3), dtype=np.float32)
    t_kf = pq.read_table(cand5_full_dir / "known_for.parquet", columns=["pid", "community", "score", "rank"])
    for p_i, c_i, sc_f, rk_i in zip(
        t_kf["pid"].to_numpy().tolist(),
        t_kf["community"].to_numpy().tolist(),
        t_kf["score"].to_numpy().tolist(),
        t_kf["rank"].to_numpy().tolist(),
    ):
        if 0 <= p_i < n_prod and 0 <= rk_i < 3 and 0 <= c_i < n_comm:
            kf_comm[p_i, rk_i] = int(c_i)
            kf_weight[p_i, rk_i] = float(sc_f)

    # Also check community flags (flagged follow-back / follow-train communities)
    memb_full = np.load(cand5_full_dir / "membership.npy").astype(np.int64)
    fl_full = np.load(cand5_full_dir / "community_flags.npz")
    raw_flagged_full = fl_full["follow_back"].astype(bool) | fl_full["follow_train"].astype(bool)
    comm_flagged = np.zeros(n_comm, dtype=bool)
    for r in pq.read_table(cand5_full_dir / "communities.parquet", columns=["community", "top_dids"]).to_pylist():
        raws = [int(memb_full[did2pid[d]]) for d in r["top_dids"] if d in did2pid]
        if raws:
            maj = collections.Counter(raws).most_common(1)[0][0]
            if maj >= 0 and raw_flagged_full[maj]:
                comm_flagged[int(r["community"])] = True

    allowed_post_pid &= (kf_comm[:, 0] >= 0) & (~comm_flagged[np.maximum(kf_comm[:, 0], 0)]) & (~raw_flagged_full[np.maximum(memb_full, 0)])
    print(f"[oct22] allowed S13-eligible non-hfr post authors={int(allowed_post_pid.sum()):,} ({time.time()-t0:.1f}s)", flush=True)

    # Load high-H pairs (H_prod_cand5_full.parquet)
    t_h = pq.read_table(s14_dir / "H_prod_cand5_full.parquet", columns=["c_actor", "c_target", "is_high_top1pct"])
    high_h_by_actor: dict[int, set[int]] = collections.defaultdict(set)
    for ca, ct, hi in zip(t_h["c_actor"].to_pylist(), t_h["c_target"].to_pylist(), t_h["is_high_top1pct"].to_pylist()):
        if hi:
            high_h_by_actor[int(ca)].add(int(ct))

    # 2. Read clean 2026-10-22 original posts by allowed_post_pid
    tp22 = pq.read_table(
        nc_dir / "2026-10-22/posts.parquet",
        columns=["did", "rkey", "rev", "seq", "is_deleted", "reply_parent", "reply_root", "self_labels", "text", "politicalness", "heat"],
    )
    # Deduplicate (did, rkey) latest (rev, seq) and honour deletes on Oct 22
    latest_p22: dict[tuple[str, str], tuple[int, int, int, str, str, int, str, float, float]] = {}
    for did_s, rkey_s, rev_i, seq_i, del_i, rp_s, rr_s, sl_l, txt_s, pol_f, heat_f in zip(
        tp22["did"].to_pylist(),
        tp22["rkey"].to_pylist(),
        tp22["rev"].to_numpy().tolist(),
        tp22["seq"].to_numpy().tolist(),
        tp22["is_deleted"].to_numpy().tolist(),
        tp22["reply_parent"].to_pylist(),
        tp22["reply_root"].to_pylist(),
        tp22["self_labels"].to_pylist(),
        tp22["text"].to_pylist(),
        tp22["politicalness"].to_numpy().tolist(),
        tp22["heat"].to_numpy().tolist(),
    ):
        p_idx = did2pid.get(did_s, -1)
        if p_idx < 0 or not allowed_post_pid[p_idx]:
            continue
        k = (did_s, rkey_s)
        ver = (int(rev_i or 0), int(seq_i or 0))
        prev = latest_p22.get(k)
        if prev is None or ver >= (prev[0], prev[1]):
            latest_p22[k] = (
                ver[0],
                ver[1],
                int(del_i or 0),
                rp_s or "",
                rr_s or "",
                len(sl_l or []),
                txt_s or "",
                float(pol_f or 0.0),
                float(heat_f or 0.0),
            )
    del tp22

    post_keys: list[tuple[str, str]] = []
    post_author_list: list[int] = []
    post_pol_list: list[float] = []
    post_heat_list: list[float] = []
    post_pol_re_list: list[bool] = []
    for (did_s, rkey_s), (_rv, _sq, del_i, rp_s, rr_s, n_sl, txt_s, pol_f, heat_f) in sorted(latest_p22.items()):
        if del_i == 0 and rp_s == "" and rr_s == "" and n_sl == 0:
            post_keys.append((did_s, rkey_s))
            post_author_list.append(did2pid[did_s])
            post_pol_list.append(pol_f)
            post_heat_list.append(heat_f)
            post_pol_re_list.append(bool(POL_RE.search(txt_s)))
    del latest_p22

    n_posts = len(post_keys)
    post_key_to_id = {k: i for i, k in enumerate(post_keys)}
    post_author = np.array(post_author_list, dtype=np.int32)
    post_comm0 = kf_comm[post_author, 0]
    post_pol = np.array(post_pol_list, dtype=np.float32)
    post_heat = np.array(post_heat_list, dtype=np.float32)
    post_pol_re = np.array(post_pol_re_list, dtype=bool)
    print(f"[oct22] clean Oct 22 original posts by eligible authors={n_posts:,} ({time.time()-t0:.1f}s)", flush=True)

    # 3. Scan 2026-10-22/engagements.parquet: identify light Oct 22 engagers & collect Oct 22 candidate-post engagements
    te22 = pq.read_table(
        nc_dir / "2026-10-22/engagements.parquet",
        columns=["kind", "did", "rkey", "is_deleted", "subject_did", "subject_collection", "subject_rkey"],
    )
    # Track unlikes/deletes by (did, rkey) on Oct 22
    del_e22: set[tuple[str, str]] = set()
    e22_is_del = te22["is_deleted"].to_numpy()
    if (e22_is_del != 0).any():
        t_del = te22.filter(pc.not_equal(te22["is_deleted"], 0))
        for d_s, rk_s in zip(t_del["did"].to_pylist(), t_del["rkey"].to_pylist()):
            del_e22.add((d_s, rk_s))

    t_act22 = te22.filter(
        pc.and_(
            pc.equal(te22["is_deleted"], 0),
            pc.equal(te22["subject_collection"], "app.bsky.feed.post"),
        )
    )
    del te22

    # First count total Oct 22 post engagements per did to filter heavy/bot engagers (>500/day) and purged DIDs
    e22_dids_col = t_act22["did"].to_pylist()
    e22_rkeys_col = t_act22["rkey"].to_pylist()
    e22_sdids_col = t_act22["subject_did"].to_pylist()
    e22_srkeys_col = t_act22["subject_rkey"].to_pylist()
    e22_kinds_col = t_act22["kind"].to_pylist()
    del t_act22

    tot_oct22_by_user: dict[str, int] = collections.Counter()
    cand_eng_raw: list[tuple[str, int, bool]] = []  # (engager_did, post_id, is_like)
    for d_s, rk_s, sd_s, srk_s, kd_b in zip(e22_dids_col, e22_rkeys_col, e22_sdids_col, e22_srkeys_col, e22_kinds_col):
        if (d_s, rk_s) in del_e22 or d_s in purge_dids or d_s == sd_s:
            continue
        tot_oct22_by_user[d_s] += 1
        pid_x = post_key_to_id.get((sd_s, srk_s), -1)
        if pid_x >= 0:
            is_lk = (kd_b == b"like" or kd_b == "like")
            cand_eng_raw.append((d_s, pid_x, is_lk))
    del e22_dids_col, e22_rkeys_col, e22_sdids_col, e22_srkeys_col, e22_kinds_col, del_e22

    # Keep users who engaged with >=1 candidate post AND had <= 500 total Oct 22 engagements
    oct22_user_dids = sorted({d_s for d_s, _, _ in cand_eng_raw if 1 <= tot_oct22_by_user[d_s] <= 500})
    u22_map = {d: i for i, d in enumerate(oct22_user_dids)}
    n_u22 = len(oct22_user_dids)
    print(f"[oct22] candidate-post Oct 22 active users (<=500/day)={n_u22:,}, raw cand eng={len(cand_eng_raw):,} ({time.time()-t0:.1f}s)", flush=True)

    # 4. Map oct22_user_dids -> follower_weights.tsv uid & slice edges.bin follows (< 3,000 producer follows = light engager)
    u22_to_uid = np.full(n_u22, -1, dtype=np.int64)
    with open(gb_dir / "follower_weights.tsv") as fh:
        for ln in fh:
            tab1 = ln.find("\t")
            tab2 = ln.find("\t", tab1 + 1)
            d_s = ln[tab1 + 1 : tab2]
            ui = u22_map.get(d_s)
            if ui is not None:
                u22_to_uid[ui] = int(ln[:tab1])

    mapped_mask = u22_to_uid >= 0
    mapped_u_idx = np.flatnonzero(mapped_mask)
    mapped_uids = u22_to_uid[mapped_u_idx]
    ord_uid = np.argsort(mapped_uids)
    sorted_uids = mapped_uids[ord_uid]
    sorted_u_idx = mapped_u_idx[ord_uid]

    edges_mm = np.memmap(gb_dir / "edges.bin", dtype=np.dtype([("f", "<u4"), ("p", "<u4")]), mode="r")
    starts = lower_bound_u32(edges_mm["f"], sorted_uids)
    ends = lower_bound_u32(edges_mm["f"], sorted_uids + 1)
    deg_fol = ends - starts

    # Light engager definition (plan §0 glossary line 51): follows < 3,000 producers
    is_light_u22 = np.ones(n_u22, dtype=bool)
    for u_i, dg in zip(sorted_u_idx.tolist(), deg_fol.tolist()):
        if dg >= 3000:
            is_light_u22[u_i] = False

    # Store follow ranges per u22 so we can project InterestedIn AND test non-followed status in O(1)
    u22_fol_start = np.zeros(n_u22, dtype=np.int64)
    u22_fol_end = np.zeros(n_u22, dtype=np.int64)
    u22_fol_start[sorted_u_idx] = starts
    u22_fol_end[sorted_u_idx] = ends
    print(
        f"[oct22] mapped in follower_weights={int(mapped_mask.sum()):,}/{n_u22:,}, "
        f"light (<3000 follows)={int(is_light_u22.sum()):,} ({time.time()-t0:.1f}s)",
        flush=True,
    )

    # 5. Collect pre-Oct-22 (Oct 19, 20, 21 ONLY) non-courtesy likes (1.0) & reposts (2.0) for u22 users
    pre_eng_by_u: list[dict[int, float]] = [{} for _ in range(n_u22)]
    for day in ("2026-10-19", "2026-10-20", "2026-10-21"):
        # Map reply posts (did, rkey) -> reply_parent author did for courtesy-like exclusion
        tp_d = pq.read_table(nc_dir / day / "posts.parquet", columns=["did", "rkey", "reply_parent"])
        rp_col = tp_d["reply_parent"].to_pylist()
        reply_parent_author: dict[tuple[str, str], str] = {}
        for d_s, rk_s, rp_s in zip(tp_d["did"].to_pylist(), tp_d["rkey"].to_pylist(), rp_col):
            if rp_s and rp_s.startswith("at://"):
                parts = rp_s.split("/")
                if len(parts) >= 3:
                    reply_parent_author[(d_s, rk_s)] = parts[2]
        del tp_d, rp_col

        te_d = pq.read_table(
            nc_dir / day / "engagements.parquet",
            columns=["kind", "did", "rkey", "is_deleted", "subject_did", "subject_collection", "subject_rkey"],
        )
        del_ed: set[tuple[str, str]] = set()
        if (te_d["is_deleted"].to_numpy() != 0).any():
            td = te_d.filter(pc.not_equal(te_d["is_deleted"], 0))
            for d_s, rk_s in zip(td["did"].to_pylist(), td["rkey"].to_pylist()):
                del_ed.add((d_s, rk_s))
        ta_d = te_d.filter(
            pc.and_(
                pc.equal(te_d["is_deleted"], 0),
                pc.equal(te_d["subject_collection"], "app.bsky.feed.post"),
            )
        )
        del te_d

        # Deduplicate distinct (u_i, pid, srk, is_repost) within the day
        seen_day: set[tuple[int, int, str, bool]] = set()
        for d_s, rk_s, sd_s, srk_s, kd_b in zip(
            ta_d["did"].to_pylist(),
            ta_d["rkey"].to_pylist(),
            ta_d["subject_did"].to_pylist(),
            ta_d["subject_rkey"].to_pylist(),
            ta_d["kind"].to_pylist(),
        ):
            ui = u22_map.get(d_s)
            if ui is None or not is_light_u22[ui]:
                continue
            p_t = did2pid.get(sd_s, -1)
            if p_t < 0 or d_s == sd_s or (d_s, rk_s) in del_ed:
                continue
            is_rp = (kd_b == b"repost" or kd_b == "repost")
            if not is_rp and reply_parent_author.get((sd_s, srk_s)) == d_s:
                continue  # courtesy like on direct reply to user
            key4 = (ui, p_t, srk_s, is_rp)
            if key4 in seen_day:
                continue
            seen_day.add(key4)
            dct = pre_eng_by_u[ui]
            dct[p_t] = dct.get(p_t, 0.0) + (2.0 if is_rp else 1.0)
        del ta_d, del_ed, reply_parent_author, seen_day
        print(f"[oct22] processed pre-Oct-22 engagements for {day} ({time.time()-t0:.1f}s)", flush=True)

    # 6. Compute top-10 InterestedIn communities (social_proof >= 2 distinct producers) for each light u22 user
    TOP_II = 10
    u_ii_comm = np.full((n_u22, TOP_II), -1, dtype=np.int32)
    u_ii_score = np.zeros((n_u22, TOP_II), dtype=np.float32)
    p_col_mm = edges_mm["p"]

    for u_i in range(n_u22):
        if not is_light_u22[u_i]:
            continue
        st_e, en_e = int(u22_fol_start[u_i]), int(u22_fol_end[u_i])
        pe = pre_eng_by_u[u_i]
        if st_e == en_e and not pe:
            continue
        # Combine producer weights: 2.0 per follow + Oct 19-21 like/repost weight
        pw: dict[int, float] = {}
        if en_e > st_e:
            for p_f in p_col_mm[st_e:en_e].tolist():
                pw[int(p_f)] = 2.0
        for p_e, w_e in pe.items():
            pw[p_e] = pw.get(p_e, 0.0) + w_e
        if len(pw) < 2:
            continue

        pids_arr = np.fromiter(pw.keys(), dtype=np.int32, count=len(pw))
        wts_arr = np.fromiter(pw.values(), dtype=np.float32, count=len(pw))
        cs = kf_comm[pids_arr]  # (M, 3)
        ks = kf_weight[pids_arr]  # (M, 3)

        c_raw: dict[int, float] = {}
        c_cnt: dict[int, int] = {}
        for slot in range(3):
            c_col = cs[:, slot]
            w_col = wts_arr * ks[:, slot]
            valid = (c_col >= 0) & (w_col > 0)
            if not valid.any():
                continue
            for c_v, w_v in zip(c_col[valid].tolist(), w_col[valid].tolist()):
                c_raw[c_v] = c_raw.get(c_v, 0.0) + w_v
                c_cnt[c_v] = c_cnt.get(c_v, 0) + 1

        elig_c = [(c_v, sc_v) for c_v, sc_v in c_raw.items() if c_cnt.get(c_v, 0) >= 2 and not comm_flagged[c_v]]
        if not elig_c:
            continue
        elig_c.sort(key=lambda x: (-x[1], x[0]))
        top_c = elig_c[:TOP_II]
        tot_sc = sum(x[1] for x in top_c)
        for j, (c_v, sc_v) in enumerate(top_c):
            u_ii_comm[u_i, j] = c_v
            u_ii_score[u_i, j] = sc_v / max(tot_sc, 1e-6)

    del pre_eng_by_u
    has_ii = u_ii_comm[:, 0] >= 0
    print(f"[oct22] computed pre-Oct-22 InterestedIn: {int(has_ii.sum()):,}/{n_u22:,} users have valid InterestedIn ({time.time()-t0:.1f}s)", flush=True)

    # 7. Aggregate Oct 22 distinct light engagers per candidate post: N_all[x] and N_c[x, slot]
    # Deduplicate (u_i, post_id), keeping track of whether u_i LIKED post_id
    pair_liked: dict[tuple[int, int], bool] = {}
    for d_s, pid_x, is_lk in cand_eng_raw:
        ui = u22_map.get(d_s)
        if ui is None or not has_ii[ui]:
            continue
        k2 = (ui, pid_x)
        if is_lk:
            pair_liked[k2] = True
        elif k2 not in pair_liked:
            pair_liked[k2] = False
    del cand_eng_raw

    N_all = np.zeros(n_posts, dtype=np.int32)
    N_c = np.zeros((n_posts, 3), dtype=np.int32)
    tot_in_c = np.zeros(n_comm, dtype=np.int64)
    u_oct22_all_posts: list[list[int]] = [[] for _ in range(n_u22)]
    u_oct22_liked_posts: list[list[int]] = [[] for _ in range(n_u22)]

    # Prebuild fast per-user top-10 set
    u_ii_sets: list[set[int]] = [
        {int(c) for c in u_ii_comm[u_i] if c >= 0} if has_ii[u_i] else set()
        for u_i in range(n_u22)
    ]

    for (ui, pid_x), liked_flag in pair_liked.items():
        N_all[pid_x] += 1
        u_oct22_all_posts[ui].append(pid_x)
        if liked_flag:
            u_oct22_liked_posts[ui].append(pid_x)
        u_set = u_ii_sets[ui]
        for c_v in u_set:
            tot_in_c[c_v] += 1
        au = int(post_author[pid_x])
        for s in range(3):
            c_p = int(kf_comm[au, s])
            if c_p >= 0 and c_p in u_set:
                N_c[pid_x, s] += 1
    tot_all_eng = int(N_all.sum())
    p_c_global = tot_in_c.astype(np.float64) / max(tot_all_eng, 1)
    print(f"[oct22] aggregated post counts: total distinct light engagements on cand posts={tot_all_eng:,} ({time.time()-t0:.1f}s)", flush=True)

    # Pre-index candidate posts by KnownFor community c -> (post_ids_c, slot_c, kf_w_c)
    comm_posts_list: list[list[tuple[int, int, float]]] = [[] for _ in range(n_comm)]
    for pid_x in range(n_posts):
        au = int(post_author[pid_x])
        for s in range(3):
            c_p = int(kf_comm[au, s])
            w_p = float(kf_weight[au, s])
            if c_p >= 0 and w_p > 0:
                comm_posts_list[c_p].append((pid_x, s, w_p))
    comm_p_ids = [np.array([t[0] for t in lst], dtype=np.int32) for lst in comm_posts_list]
    comm_p_slot = [np.array([t[1] for t in lst], dtype=np.int8) for lst in comm_posts_list]
    comm_p_kfw = [np.array([t[2] for t in lst], dtype=np.float32) for lst in comm_posts_list]
    del comm_posts_list

    # Pre-sort top 400 global posts by N_all for O(1) leave-one-out global_popularity comparator
    glob_ord = np.lexsort((np.arange(n_posts, dtype=np.int32), -N_all))
    glob_top_pool = glob_ord[:1000].astype(np.int32)

    # 8. Identify eligible active Oct 22 evaluation users (have >=1 real Oct 22 like on a NON-FOLLOWED producer's clean original)
    eval_candidates: list[tuple[int, set[int], set[int]]] = []
    for ui in range(n_u22):
        if not has_ii[ui] or not u_oct22_liked_posts[ui]:
            continue
        st_e, en_e = int(u22_fol_start[ui]), int(u22_fol_end[ui])
        fol_set = set(p_col_mm[st_e:en_e].tolist()) if en_e > st_e else set()
        nonfol_likes = {px for px in u_oct22_liked_posts[ui] if int(post_author[px]) not in fol_set}
        if nonfol_likes:
            eval_candidates.append((ui, fol_set, nonfol_likes))

    rng = np.random.default_rng(20261030)
    MAX_EVAL_U = 10000
    if len(eval_candidates) > MAX_EVAL_U:
        sel_idx = np.sort(rng.choice(len(eval_candidates), size=MAX_EVAL_U, replace=False))
        eval_sample = [eval_candidates[i] for i in sel_idx.tolist()]
    else:
        eval_sample = eval_candidates
    n_eval = len(eval_sample)
    print(f"[oct22] total users with >=1 non-followed Oct 22 like={len(eval_candidates):,}; evaluating sample n={n_eval:,}", flush=True)

    rec_residual = np.zeros(n_eval, dtype=np.float64)
    rec_global_pop = np.zeros(n_eval, dtype=np.float64)
    rec_comm_post_pop_no_res = np.zeros(n_eval, dtype=np.float64)
    rec_comm_author_pop_no_res = np.zeros(n_eval, dtype=np.float64)

    for idx_e, (ui, fol_set, nonfol_likes) in enumerate(eval_sample):
        u_all_set = set(u_oct22_all_posts[ui])
        k_u = len(u_all_set)
        denom_u = max(tot_all_eng - k_u, 1)

        # High-H target communities toward user's top-3 InterestedIn communities
        top3_c = [int(c) for c in u_ii_comm[ui, :3] if c >= 0]
        hh_targets: set[int] = set()
        for c_pick in top3_c:
            hh_targets |= high_h_by_actor.get(c_pick, set())

        # A) Global popularity comparator (leave-one-out on glob_top_pool, filtering followed authors)
        gp_ids = np.array([px for px in glob_top_pool.tolist() if int(post_author[px]) not in fol_set], dtype=np.int32)
        gp_scores = np.array([float(N_all[px] - (1 if px in u_all_set else 0)) for px in gp_ids.tolist()], dtype=np.float64)
        top50_glob = select_top_k_with_caps(gp_ids, gp_scores, post_author, post_comm0, None, k=50, max_per_author=2)
        rec_global_pop[idx_e] = sum(1 for px in top50_glob if px in nonfol_likes) / len(nonfol_likes)

        # B) Candidate posts in user's top-10 InterestedIn communities (non-followed authors)
        acc_res: dict[int, float] = {}
        acc_post_pop: dict[int, float] = {}
        acc_auth_pop: dict[int, float] = {}

        for j in range(TOP_II):
            c_v = int(u_ii_comm[ui, j])
            if c_v < 0:
                break
            ii_w = float(u_ii_score[ui, j])
            pids_c = comm_p_ids[c_v]
            if len(pids_c) == 0:
                continue
            slots_c = comm_p_slot[c_v]
            kfw_c = comm_p_kfw[c_v]
            p_c_u = max(int(tot_in_c[c_v]) - k_u, 0) / denom_u

            n_all_c = N_all[pids_c].astype(np.float64)
            n_in_c = N_c[pids_c, slots_c].astype(np.float64)
            # Leave-one-out adjustment on posts user ui engaged with on Oct 22
            if u_all_set:
                for loc_idx, px in enumerate(pids_c.tolist()):
                    if px in u_all_set:
                        n_all_c[loc_idx] -= 1.0
                        n_in_c[loc_idx] -= 1.0

            res_vec = n_in_c - p_c_u * n_all_c
            # Keep only non-followed authors
            au_c = post_author[pids_c]
            for loc_idx, px in enumerate(pids_c.tolist()):
                au = int(au_c[loc_idx])
                if au in fol_set:
                    continue
                w_comb = ii_w * float(kfw_c[loc_idx])
                # Residual ranking (plan §3.3): community-excess engagement weighted by InterestedIn x KnownFor
                r_val = float(res_vec[loc_idx])
                if r_val > 0.0:
                    acc_res[px] = acc_res.get(px, 0.0) + w_comb * r_val
                elif px not in acc_res:
                    acc_res[px] = w_comb * r_val
                # Community post popularity WITHOUT residual: raw global post popularity N_all,-u(x) inside II communities
                acc_post_pop[px] = max(acc_post_pop.get(px, 0.0), float(n_all_c[loc_idx]))
                # Community author popularity WITHOUT residual: author follower count weighted by II x KnownFor
                acc_auth_pop[px] = acc_auth_pop.get(px, 0.0) + w_comb * float(prod_followers[au])

        if acc_res:
            c_ids_arr = np.fromiter(acc_res.keys(), dtype=np.int32, count=len(acc_res))
            res_sc_arr = np.fromiter(acc_res.values(), dtype=np.float64, count=len(acc_res))
            top50_res = select_top_k_with_caps(
                c_ids_arr, res_sc_arr, post_author, post_comm0, hh_targets, k=50, max_per_author=2, max_high_h=2
            )
            rec_residual[idx_e] = sum(1 for px in top50_res if px in nonfol_likes) / len(nonfol_likes)

            pp_ids_arr = np.fromiter(acc_post_pop.keys(), dtype=np.int32, count=len(acc_post_pop))
            pp_sc_arr = np.fromiter(acc_post_pop.values(), dtype=np.float64, count=len(acc_post_pop))
            top50_pp = select_top_k_with_caps(
                pp_ids_arr, pp_sc_arr, post_author, post_comm0, hh_targets, k=50, max_per_author=2, max_high_h=2
            )
            rec_comm_post_pop_no_res[idx_e] = sum(1 for px in top50_pp if px in nonfol_likes) / len(nonfol_likes)

            ap_ids_arr = np.fromiter(acc_auth_pop.keys(), dtype=np.int32, count=len(acc_auth_pop))
            ap_sc_arr = np.fromiter(acc_auth_pop.values(), dtype=np.float64, count=len(acc_auth_pop))
            top50_ap = select_top_k_with_caps(
                ap_ids_arr, ap_sc_arr, post_author, post_comm0, hh_targets, k=50, max_per_author=2, max_high_h=2
            )
            rec_comm_author_pop_no_res[idx_e] = sum(1 for px in top50_ap if px in nonfol_likes) / len(nonfol_likes)

    diff_vs_glob = rec_residual - rec_global_pop
    ci_vs_glob = boot_diff_ci(diff_vs_glob)
    diff_vs_comm_post = rec_residual - rec_comm_post_pop_no_res
    ci_vs_comm_post = boot_diff_ci(diff_vs_comm_post)
    diff_vs_comm_auth = rec_residual - rec_comm_author_pop_no_res
    ci_vs_comm_auth = boot_diff_ci(diff_vs_comm_auth)

    offline_pass = bool(ci_vs_glob[0] > 0.0 and ci_vs_comm_post[0] > 0.0 and ci_vs_comm_auth[0] > 0.0)
    print(
        f"[oct22] OFFLINE CHECK (n={n_eval:,}): "
        f"residual={rec_residual.mean():.6f} | global_pop={rec_global_pop.mean():.6f} (diff={diff_vs_glob.mean():+.6f}, CI95={ci_vs_glob}) | "
        f"comm_post_pop_no_res={rec_comm_post_pop_no_res.mean():.6f} (diff={diff_vs_comm_post.mean():+.6f}, CI95={ci_vs_comm_post}) | "
        f"comm_author_pop_no_res={rec_comm_author_pop_no_res.mean():.6f} (diff={diff_vs_comm_auth.mean():+.6f}, CI95={ci_vs_comm_auth}) | "
        f"PASS={offline_pass}",
        flush=True,
    )

    # 9. Emit top-60 Oct 22 preview candidate posts per community (<=2 per author) for S17 live getPosts hydration
    rows_out: list[dict] = []
    for c_v in range(n_comm):
        if comm_flagged[c_v]:
            continue
        pids_c = comm_p_ids[c_v]
        if len(pids_c) == 0:
            continue
        slots_c = comm_p_slot[c_v]
        kfw_c = comm_p_kfw[c_v]
        n_all_c = N_all[pids_c].astype(np.float64)
        n_in_c = N_c[pids_c, slots_c].astype(np.float64)
        exp_c = p_c_global[c_v] * n_all_c
        raw_res_c = n_in_c - exp_c
        # Weight by KnownFor share; gently break zero-engagement ties by KnownFor share & post order
        w_res_c = kfw_c.astype(np.float64) * raw_res_c
        top_c_posts = select_top_k_with_caps(pids_c, w_res_c, post_author, post_comm0, None, k=60, max_per_author=2)
        loc_map = {int(px): idx_l for idx_l, px in enumerate(pids_c.tolist())}
        for rk_out, px in enumerate(top_c_posts):
            li = loc_map[px]
            d_s, rkey_s = post_keys[px]
            rows_out.append(
                {
                    "community": int(c_v),
                    "rank": int(rk_out),
                    "uri": f"at://{d_s}/app.bsky.feed.post/{rkey_s}",
                    "did": d_s,
                    "rkey": rkey_s,
                    "kf_rank": int(slots_c[li]),
                    "kf_score": float(kfw_c[li]),
                    "n_all": int(N_all[px]),
                    "n_comm": int(N_c[px, int(slots_c[li])]),
                    "expected_comm": round(float(exp_c[li]), 4),
                    "residual": round(float(raw_res_c[li]), 4),
                    "weighted_residual": round(float(w_res_c[li]), 4),
                    "politicalness": round(float(post_pol[px]), 4),
                    "heat": round(float(post_heat[px]), 4),
                    "pol_keyword": bool(post_pol_re[px]),
                }
            )

    pq_out = out_dir / "oct22_preview_posts.parquet"
    pq.write_table(pa.Table.from_pylist(rows_out), pq_out, compression="zstd")

    rep = {
        "step": "S16/S17 Local For You Preview — Oct 22 Offline Check & Candidate Bank",
        "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md §3.3 (lines 314-320)",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md §9.8, Amendment A1",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t0, 2),
        "caveats": [
            "Exposure bias: observational Oct 22 likes reflect what Bluesky timelines/feeds surfaced on 2026-10-22.",
            "Follow snapshot post-dates Oct 22 by ~2 days (2026-10-24 fold), so any Oct 22 non-followed like followed by an Oct 22-24 follow is excluded from nonfol_likes rather than leaking.",
        ],
        "counts": {
            "eligible_non_hfr_post_authors": int(allowed_post_pid.sum()),
            "clean_oct22_original_posts": n_posts,
            "oct22_active_cand_users_le500": n_u22,
            "light_users_lt3000_follows": int(is_light_u22.sum()),
            "users_with_pre_oct22_interested_in": int(has_ii.sum()),
            "total_light_engagements_on_cand_posts": tot_all_eng,
            "users_with_nonfollowed_oct22_likes": len(eval_candidates),
            "evaluated_users_sampled": n_eval,
            "preview_bank_rows_written": len(rows_out),
            "communities_with_preview_posts": len({r["community"] for r in rows_out}),
        },
        "offline_check_recall_at_50": {
            "pass": offline_pass,
            "residual_recall50": float(rec_residual.mean()),
            "global_popularity_recall50": float(rec_global_pop.mean()),
            "community_post_popularity_no_residual_recall50": float(rec_comm_post_pop_no_res.mean()),
            "community_author_popularity_no_residual_recall50": float(rec_comm_author_pop_no_res.mean()),
            "diff_vs_global_popularity": {
                "mean": float(diff_vs_glob.mean()),
                "ci95": ci_vs_glob,
                "pass_ci_gt_0": bool(ci_vs_glob[0] > 0.0),
            },
            "diff_vs_community_post_popularity_no_residual": {
                "mean": float(diff_vs_comm_post.mean()),
                "ci95": ci_vs_comm_post,
                "pass_ci_gt_0": bool(ci_vs_comm_post[0] > 0.0),
            },
            "diff_vs_community_author_popularity_no_residual": {
                "mean": float(diff_vs_comm_auth.mean()),
                "ci95": ci_vs_comm_auth,
                "pass_ci_gt_0": bool(ci_vs_comm_auth[0] > 0.0),
            },
        },
        "outputs": {
            "oct22_preview_posts_parquet": str(pq_out),
            "oct22_preview_posts_sha256": sha256_file(pq_out),
        },
    }
    (out_dir / "preview_oct22_report.json").write_text(json.dumps(rep, indent=2))
    print(f"[oct22] DONE in {time.time()-t0:.1f}s -> {out_dir / 'preview_oct22_report.json'}", flush=True)


if __name__ == "__main__":
    main()
