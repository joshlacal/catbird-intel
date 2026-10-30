#!/usr/bin/env python3
"""Step S13 + V8(ii) (M2-val ONLY): Leak-free flags, S13b/S13d checks, 12-cell x ranker M2-val sweep, and 40/40 live bundle check.

STRICT GUARD: This script NEVER accesses `M2-test` (`split == 'test'`).
Selection of `(k, m, theta, ranker)` happens exclusively on `M2-val` (`split == 'val'`),
with fixed `M2Data` outcomes (`uni_base` safety exclusions only; `hide_from_recs` kept per A1).
"""
from __future__ import annotations

import collections
import datetime as dt
import gzip
import hashlib
import json
import os
import pathlib
import pickle
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import eval_temporal as et  # noqa: E402
import s3_prep  # noqa: E402

TRAIN_END_US = et.TRAIN_END_US  # 2026-08-16T00:00Z = 1786838400000000
_TID_ALPHA = "234567abcdefghijklmnopqrstuvwxyz"
_TID_MAP = {c: i for i, c in enumerate(_TID_ALPHA)}
UA = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"


def tid_us_from_uri(uri: str) -> int:
    rkey = uri.rsplit("/", 1)[-1]
    if len(rkey) != 13:
        return -1
    v = 0
    for ch in rkey:
        d = _TID_MAP.get(ch)
        if d is None:
            return -1
        v = (v << 5) | d
    return v >> 10


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def build_leakfree_flags_T(
    train_t_dir: pathlib.Path,
    cand5_t_dir: pathlib.Path,
    packs_pkl: pathlib.Path,
    lt_posts_T: np.ndarray,
    sup_T: np.ndarray,
    out_dir: pathlib.Path,
) -> dict:
    """Compute leak-free follower_degree, out_pp, pre-T pack_counts, and community_flags on cand5-T-matched."""
    cache_stats = out_dir / "train_T_graph_stats.npz"
    ro = pcsv.ReadOptions(column_names=None, autogenerate_column_names=True)
    po = pcsv.ParseOptions(delimiter="\t")
    prod = pcsv.read_csv(str(train_t_dir / "producers.tsv"), read_options=ro, parse_options=po)
    n_p = prod.num_rows
    followers_T = prod.column(2).to_numpy().astype(np.int64)
    dids_T = prod.column(1)

    if cache_stats.exists():
        z = np.load(cache_stats)
        mass3000_T = z["mass3000"]
        out_pp_T = z["out_pp"]
        packs_cnt_T = z["pack_counts_preT"]
    else:
        t0 = time.time()
        fw = pcsv.read_csv(
            str(train_t_dir / "follower_weights.tsv"),
            read_options=ro,
            parse_options=po,
            convert_options=pcsv.ConvertOptions(include_columns=["f0", "f1"]),
        )
        uid = fw.column(0).to_numpy().astype(np.int64)
        idx = pc.fill_null(pc.index_in(fw.column(1), value_set=dids_T), -1).to_numpy().astype(np.int64)
        max_uid = int(uid.max()) + 1
        uid2pid = np.full(max_uid, -1, dtype=np.int64)
        uid2pid[uid] = idx

        edges = np.memmap(train_t_dir / "edges.bin", dtype=np.uint32, mode="r").reshape(-1, 2)
        n_e = edges.shape[0]
        deg_u = np.zeros(max_uid, dtype=np.int32)
        ch = 200_000_000
        for s in range(0, n_e, ch):
            u_col = np.asarray(edges[s : s + ch, 0])
            deg_u += np.bincount(u_col, minlength=max_uid).astype(np.int32)

        heavy_u = deg_u >= 3000
        mass3000_T = np.zeros(n_p, dtype=np.int64)
        out_pp_T = np.zeros(n_p, dtype=np.int64)
        valid_u = uid[idx >= 0]
        valid_p = idx[idx >= 0]
        out_pp_T[valid_p] = deg_u[valid_u].astype(np.int64)

        for s in range(0, n_e, ch):
            blk = np.asarray(edges[s : s + ch])
            hm = heavy_u[blk[:, 0]]
            mass3000_T += np.bincount(blk[hm, 1], minlength=n_p)

        did2pid_T = {d: i for i, d in enumerate(dids_T.to_pylist())}
        packs_cnt_T = np.zeros(n_p, dtype=np.int32)
        n_packs_preT = 0
        for uri, members, _ in pickle.load(open(packs_pkl, "rb")):
            tus = tid_us_from_uri(uri)
            if 0 < tus < TRAIN_END_US:
                n_packs_preT += 1
                for m_did in set(members):
                    pi = did2pid_T.get(m_did)
                    if pi is not None:
                        packs_cnt_T[pi] += 1

        np.savez(cache_stats, mass3000=mass3000_T, out_pp=out_pp_T, pack_counts_preT=packs_cnt_T, n_packs_preT=n_packs_preT)
        print(f"[flags_T] computed leak-free train_T stats in {time.time() - t0:.1f}s (pre-T packs={n_packs_preT:,})", flush=True)

    memb = np.load(cand5_t_dir / "membership.npy").astype(np.int64)
    ok = memb >= 0
    C = int(memb.max()) + 1
    mass = mass3000_T / np.maximum(followers_T, 1)
    order = np.flatnonzero(ok)[np.argsort(memb[ok], kind="stable")]
    b = np.searchsorted(memb[order], np.arange(C + 1))
    size = np.diff(b)
    med_mass = np.zeros(C)
    in_pack = np.zeros(C)
    med_packs = np.zeros(C)
    for c in range(C):
        m = order[b[c] : b[c + 1]]
        if len(m):
            med_mass[c] = np.median(mass[m])
            in_pack[c] = (packs_cnt_T[m] >= 1).mean()
            med_packs[c] = np.median(packs_cnt_T[m])
    follow_back = (med_mass >= 0.5) & (in_pack < 0.15)
    follow_train = (med_mass >= 0.6) & (med_packs >= 5)

    # Third flag: low_post_follow_heavy (supplement excluded from numerator and denominator)
    lpfh_prod = (lt_posts_T <= 10) & (out_pp_T >= 0.8 * np.maximum(followers_T, 1)) & (~sup_T)
    ok_nonsup = ok & (~sup_T)
    den = np.bincount(memb[ok_nonsup], minlength=C)
    num = np.bincount(memb[ok_nonsup], weights=lpfh_prod[ok_nonsup], minlength=C)
    lpfh_share = num / np.maximum(den, 1)

    summary = {"C": C, "follow_back": int(follow_back.sum()), "follow_train": int(follow_train.sum())}
    for theta, tag in ((0.3, "03"), (0.5, "05")):
        lpfh_flag = (den > 0) & (lpfh_share >= theta)
        p_out = out_dir / f"flags_T_theta{tag}.npz"
        np.savez(
            p_out,
            size=size,
            med_mass3k=med_mass,
            in_pack=in_pack,
            median_packs=med_packs,
            lpfh_share=lpfh_share,
            follow_back=follow_back,
            follow_train=follow_train,
            low_post_follow_heavy=lpfh_flag,
        )
        summary[f"lpfh_theta_{theta}"] = int(lpfh_flag.sum())
        summary[f"lpfh_theta_{theta}_new_unflagged_ge8"] = int((lpfh_flag & (~follow_back) & (~follow_train) & (size >= 8)).sum())
    return summary


def build_flags_full(
    cand5_full_dir: pathlib.Path,
    audit_dir: pathlib.Path,
    lt_posts_prod: np.ndarray,
    sup_prod: np.ndarray,
    out_dir: pathlib.Path,
) -> dict:
    fl_base = np.load(cand5_full_dir / "community_flags.npz")
    memb = np.load(cand5_full_dir / "membership.npy").astype(np.int64)
    st = np.load(audit_dir / "producer_stats.npz")
    C = len(fl_base["size"])
    ok_nonsup = (memb >= 0) & (~sup_prod)
    lpfh_prod = (lt_posts_prod <= 10) & (st["out_pp"] >= 0.8 * np.maximum(st["followers"], 1)) & (~sup_prod)
    den = np.bincount(memb[ok_nonsup], minlength=C)
    num = np.bincount(memb[ok_nonsup], weights=lpfh_prod[ok_nonsup], minlength=C)
    lpfh_share = num / np.maximum(den, 1)

    summary = {
        "C": C,
        "follow_back": int(fl_base["follow_back"].sum()),
        "follow_train": int(fl_base["follow_train"].sum()),
    }
    for theta, tag in ((0.3, "03"), (0.5, "05")):
        lpfh_flag = (den > 0) & (lpfh_share >= theta)
        p_out = out_dir / f"flags_full_theta{tag}.npz"
        np.savez(
            p_out,
            size=fl_base["size"],
            med_mass3k=fl_base["med_mass3k"],
            in_pack=fl_base["in_pack"],
            median_packs=fl_base["median_packs"],
            lpfh_share=lpfh_share,
            follow_back=fl_base["follow_back"],
            follow_train=fl_base["follow_train"],
            low_post_follow_heavy=lpfh_flag,
        )
        summary[f"lpfh_theta_{theta}"] = int(lpfh_flag.sum())
        summary[f"lpfh_theta_{theta}_new_unflagged_ge8"] = int(
            (lpfh_flag & (~fl_base["follow_back"]) & (~fl_base["follow_train"]) & (fl_base["size"] >= 8)).sum()
        )
    return summary


def xrpc_get_json(path: str, params: dict) -> tuple[int, dict, str]:
    url = f"https://public.api.bsky.app/xrpc/{path}?" + urllib.parse.urlencode(params, doseq=True)
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept": "application/json", "Accept-Encoding": "gzip"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return 200, json.loads(raw), hashlib.sha256(raw).hexdigest()
        except urllib.error.HTTPError as e:
            raw = e.read() or b"{}"
            if e.code in (400, 404):
                return e.code, {}, hashlib.sha256(raw).hexdigest()
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return -1, {}, hashlib.sha256(b"").hexdigest()


def load_high_H_pairs_T(s14a_dir: pathlib.Path) -> set[tuple[int, int]]:
    """Top 1% k>=50 cross-community H cells on cand5-T-matched (leak-free at T)."""
    t = pq.read_table(s14a_dir / "o5_val_H_k50.parquet")
    ca_col = next(c for c in ("c_actor", "c_u", "c_a") if c in t.column_names)
    cb_col = next(c for c in ("c_target", "c_p", "c_b") if c in t.column_names)
    h_col = next(c for c in ("h_val", "H", "h_prod") if c in t.column_names)
    ca = t[ca_col].to_numpy().astype(np.int64)
    cb = t[cb_col].to_numpy().astype(np.int64)
    hv = t[h_col].to_numpy().astype(np.float64)
    cross = ca != cb
    ca, cb, hv = ca[cross], cb[cross], hv[cross]
    if len(hv) == 0:
        return set()
    n_top = max(1, int(round(0.01 * len(hv))))
    thr = np.partition(hv, len(hv) - n_top)[len(hv) - n_top]
    sel = hv >= thr
    pairs = set()
    for a, b in zip(ca[sel].tolist(), cb[sel].tolist()):
        pairs.add((a, b))
        pairs.add((b, a))
    return pairs


def fast_score_m2_val(
    part: et.Partition,
    uni: et.Universe,
    elig_mask: np.ndarray,
    raw_comm_flagged: np.ndarray,
    memb_T: np.ndarray,
    comm_flagged: np.ndarray,
    rankers: dict[str, np.ndarray],
    val_users: np.ndarray,
    seeds_by_u: list[np.ndarray],
    excl_by_u: list[set[int]],
    follow_out: dict[int, set[int]],
    engaged_out: dict[int, set[int]] | None,
    high_h_pairs: set[tuple[int, int]],
    co_off: np.ndarray,
    co_dst: np.ndarray,
    co_cnt: np.ndarray,
) -> dict:
    """Exact M2-val evaluation on fixed `follow_out`/`engaged_out` outcomes, skipping high-H fill neighbours."""
    cand_ok = (part.primary >= 0) & uni.eligible & elig_mask & (~raw_comm_flagged[np.maximum(memb_T, 0)])
    members = np.flatnonzero(cand_ok)
    off, _ = et.group(part.primary[members], part.n_comm)
    card_ok = (np.diff(off) >= et.CARD_MIN) & (~comm_flagged)

    # Pre-slice top 45 candidates per card for each ranker (more than enough for 25-account bundle + ceiling + exclusions)
    top_per_card: dict[str, list[list[int]]] = {}
    for rname, key in rankers.items():
        order = np.lexsort((uni.tiebreak[members], -key[members], part.primary[members]))
        ranked_mem = members[order]
        cards_lists: list[list[int]] = []
        for c in range(part.n_comm):
            s, e = int(off[c]), int(off[c + 1])
            cards_lists.append(ranked_mem[s : min(e, s + 45)].tolist() if e > s else [])
        top_per_card[rname] = cards_lists

    # Pick top-3 cards per val user
    picks_u: dict[int, list[int]] = {}
    for u in val_users:
        s_arr = seeds_by_u[u]
        if len(s_arr) == 0:
            continue
        mass = part.interest(s_arr)
        cand_c = np.flatnonzero((mass > 0) & card_ok)
        if len(cand_c):
            picks_u[int(u)] = [int(c) for c in cand_c[np.lexsort((cand_c, -mass[cand_c]))][: et.N_PICKS]]

    eval_u = np.array([u for u in val_users if (int(u) in picks_u) and (int(u) in follow_out)], dtype=np.int64)
    fol_arr = uni.followers
    pop_lim, pop_max = et.POP_CEILING

    # Precompute H-aware neighbour fill order per picked-card tuple
    fill_cache: dict[tuple[int, ...], list[int]] = {}

    def get_fill_nbs(pk: tuple[int, ...]) -> list[int]:
        res = fill_cache.get(pk)
        if res is not None:
            return res
        sc: dict[int, int] = {}
        pk_set = set(pk)
        for c in pk:
            for nb, cnt in zip(co_dst[co_off[c] : co_off[c + 1]], co_cnt[co_off[c] : co_off[c + 1]]):
                nb_i = int(nb)
                if nb_i in pk_set or not card_ok[nb_i]:
                    continue
                if any((c, nb_i) in high_h_pairs for c in pk):
                    continue
                sc[nb_i] = sc.get(nb_i, 0) + int(cnt)
        res = sorted(sc, key=lambda x: (-sc[x], x))
        fill_cache[pk] = res
        return res

    per_ranker_vecs: dict[str, np.ndarray] = {}
    per_ranker_eng: dict[str, np.ndarray] = {}
    fill_totals: dict[str, int] = {}
    for rname, c_lists in top_per_card.items():
        v_fol = np.zeros(len(eval_u), dtype=np.float64)
        v_eng = np.zeros(len(eval_u), dtype=np.float64)
        filled_cnt = 0
        for idx_u, u in enumerate(eval_u.tolist()):
            pks = picks_u[u]
            excl = excl_by_u[u]
            chosen: list[int] = []
            chosen_set: set[int] = set()
            big = 0
            allocs = et.allocate(len(pks), et.BUNDLE)
            for c, quota in zip(pks, allocs):
                got = 0
                for p in c_lists[c]:
                    if got >= quota:
                        break
                    if p in excl or p in chosen_set:
                        continue
                    if fol_arr[p] > pop_lim:
                        if big >= pop_max:
                            continue
                        big += 1
                    chosen.append(p)
                    chosen_set.add(p)
                    got += 1
            if len(chosen) < et.BUNDLE:
                for nb in get_fill_nbs(tuple(pks)):
                    if len(chosen) >= et.BUNDLE:
                        break
                    need = et.BUNDLE - len(chosen)
                    got = 0
                    for p in c_lists[nb]:
                        if got >= need:
                            break
                        if p in excl or p in chosen_set:
                            continue
                        if fol_arr[p] > pop_lim:
                            if big >= pop_max:
                                continue
                            big += 1
                        chosen.append(p)
                        chosen_set.add(p)
                        got += 1
                        filled_cnt += 1
            fset = follow_out[u]
            v_fol[idx_u] = sum(1 for p in chosen if p in fset) / et.BUNDLE
            if engaged_out is not None:
                eset = engaged_out.get(u, ())
                v_eng[idx_u] = sum(1 for p in chosen if p in eset) / et.BUNDLE
        per_ranker_vecs[rname] = v_fol
        per_ranker_eng[rname] = v_eng
        fill_totals[rname] = filled_cnt

    return {
        "n_cards_eligible": int(card_ok.sum()),
        "n_members_eligible": int(len(members)),
        "n_val_users_evaluated": int(len(eval_u)),
        "eval_users": eval_u,
        "follow_vecs": per_ranker_vecs,
        "engaged_vecs": per_ranker_eng,
        "fill_accounts": fill_totals,
    }


def main() -> None:
    t0 = time.time()
    lake = pathlib.Path.home() / "catbird-lake-local"
    out_dir = lake / "s13-elig-20261030"
    out_dir.mkdir(parents=True, exist_ok=True)
    s5_dir = lake / "s5-sweep-20261030"
    s10_dir = lake / "s10-outputs-20261030"
    s11_dir = lake / "s11-affinity-20261030"
    s14_dir = lake / "s14-hostility-20261030"
    train_t_dir = lake / "s2-prep-20261028/out/s2a/train_T"
    cand5_t_dir = lake / "s3/matched/cand5-T-matched"
    cand5_full_dir = lake / "s3/matched/cand5-full-matched"
    audit_dir = lake / "communities-cand1-hubdemote/audit-mass-followers"

    # 1. Build fixed base safety exclusion DID set (NO hide_from_recs per A1)
    purge_dids = {ln.strip() for ln in open(s5_dir / "purge_dids.txt") if ln.startswith("did:")}
    rule2_dids = {ln.strip() for ln in open(s5_dir / "eligibility_rule2_exclude_dids.txt") if ln.startswith("did:")}
    optout_dids = {
        ln.split("\t")[0].strip()
        for ln in open(lake / "s5-declarations-20261027/user_intents_optout_dids.tsv")
        if ln.startswith("did:")
    }
    claude_prof_drop = set()
    for r in pq.read_table(
        lake / "claude-labels-20261027/profile_labels.parquet",
        columns=["did", "haiku_sensitivity", "sonnet_sensitivity"],
    ).to_pylist():
        sens = set(r["haiku_sensitivity"] or []) | set(r["sonnet_sensitivity"] or [])
        if sens & {"likely_minors", "adult"}:
            claude_prof_drop.add(r["did"])

    base_exclude = purge_dids | rule2_dids | optout_dids | claude_prof_drop
    base_excl_file = out_dir / "base_safety_exclude_dids.txt"
    with open(base_excl_file, "w") as fh:
        for d in sorted(base_exclude):
            fh.write(d + "\n")

    # 2. Load train_T universe & full universe mappings + O2/eval + w_lin_eval
    uni = et.Universe(train_t_dir / "producers.tsv", base_exclude, et.BOOT_SEED)
    part = et.Partition("cand5", cand5_t_dir, uni, None)
    co_off, co_dst, co_cnt = part.co_membership()
    memb_T = np.load(cand5_t_dir / "membership.npy").astype(np.int64)

    full_dids = [ln.rstrip("\n").split("\t")[1] for ln in open(cand5_full_dir / "producers.tsv")]
    n_full = len(full_dids)
    full_idx_to_T = uni.index(pa.array(full_dids, pa.string()))

    # Load O2/eval features mapped from pid_full -> pid_T
    orig12_T = np.zeros(uni.n, dtype=np.int64)
    orig30_T = np.zeros(uni.n, dtype=np.int64)
    lt_posts_T = np.full(uni.n, 999999, dtype=np.int64)
    blk_raw_T = np.zeros(uni.n, dtype=np.int64)
    bpk_bd_T = np.zeros(uni.n, dtype=np.float64)
    sup_T = np.zeros(uni.n, dtype=bool)
    cur_T = np.zeros(uni.n, dtype=np.float64)

    for pq_f in sorted((s10_dir / "O2/eval").glob("j*.parquet")):
        t = pq.read_table(
            pq_f,
            columns=[
                "p",
                "originals_12mo",
                "originals_30d",
                "lifetime_posts",
                "live_blockers_raw",
                "blockers_per_1k_burst_discounted",
                "a_half_curators",
                "supplement",
            ],
        )
        pf = t["p"].to_numpy().astype(np.int64)
        ok_f = pf < n_full
        pf = pf[ok_f]
        pt = full_idx_to_T[pf]
        ok_t = pt >= 0
        pt = pt[ok_t]
        orig12_T[pt] = t["originals_12mo"].to_numpy().astype(np.int64)[ok_f][ok_t]
        orig30_T[pt] = t["originals_30d"].to_numpy().astype(np.int64)[ok_f][ok_t]
        lt_posts_T[pt] = t["lifetime_posts"].to_numpy().astype(np.int64)[ok_f][ok_t]
        blk_raw_T[pt] = t["live_blockers_raw"].to_numpy().astype(np.int64)[ok_f][ok_t]
        bpk_bd_T[pt] = np.nan_to_num(
            t["blockers_per_1k_burst_discounted"].to_numpy().astype(np.float64)[ok_f][ok_t], nan=0.0
        )
        cur_T[pt] = t["a_half_curators"].to_numpy().astype(np.float64)[ok_f][ok_t]
        sup_T[pt] = t["supplement"].to_numpy().astype(bool)[ok_f][ok_t]

    # Also mark supplement DIDs from s3/m2/supplement_dids.txt
    sup_dids_set = {ln.strip() for ln in open(lake / "s3/m2/supplement_dids.txt") if ln.startswith("did:")}
    for i, d in enumerate(uni.dids):
        if d in sup_dids_set:
            sup_T[i] = True

    # Supplement activity at T from supplement_activity.parquet (S1 §10)
    sup_act = {}
    for r in pq.read_table(s5_dir / "supplement_activity.parquet").to_pylist():
        sup_act[r["did"]] = r

    # Load w_lin_eval.parquet mapped by did -> pid_T
    wt_eval = pq.read_table(
        s11_dir / "w_lin_eval.parquet",
        columns=[
            "did",
            "w_lin_pair_per_sqrt_fol",
            "w_lin_lift_per_sqrt_fol",
            "w_lin_literal_per_sqrt_fol",
            "D_per_sqrt_fol",
            "light_engagers_discounted",
            "a_half_curators",
        ],
    )
    w_idx_T = uni.index(wt_eval["did"].combine_chunks())
    ok_w = w_idx_T >= 0
    wt_i = w_idx_T[ok_w]
    wlin_pair_T = np.zeros(uni.n, dtype=np.float64)
    wlin_lift_T = np.zeros(uni.n, dtype=np.float64)
    wlin_lit_T = np.zeros(uni.n, dtype=np.float64)
    d_sqrt_T = np.zeros(uni.n, dtype=np.float64)
    le_disc_T = np.zeros(uni.n, dtype=np.float64)
    wlin_pair_T[wt_i] = wt_eval["w_lin_pair_per_sqrt_fol"].to_numpy().astype(np.float64)[ok_w]
    wlin_lift_T[wt_i] = wt_eval["w_lin_lift_per_sqrt_fol"].to_numpy().astype(np.float64)[ok_w]
    wlin_lit_T[wt_i] = wt_eval["w_lin_literal_per_sqrt_fol"].to_numpy().astype(np.float64)[ok_w]
    d_sqrt_T[wt_i] = wt_eval["D_per_sqrt_fol"].to_numpy().astype(np.float64)[ok_w]
    le_disc_T[wt_i] = wt_eval["light_engagers_discounted"].to_numpy().astype(np.float64)[ok_w]
    cur_T[wt_i] = np.maximum(cur_T[wt_i], wt_eval["a_half_curators"].to_numpy().astype(np.float64)[ok_w])

    # Live postsCount & status check where live-checked in merged profiles.parquet
    prof_t = pq.read_table(
        s5_dir / "profiles.parquet",
        columns=["did", "found", "handle", "postsCount", "any_mod_label", "self_labels", "other_labels"],
    )
    live_bad_or_lt3 = set()
    for d, fnd, hnd, pcnt, modl, selfl, othl in zip(*(prof_t[c].to_pylist() for c in prof_t.column_names)):
        if (not fnd) or hnd == "handle.invalid" or modl or selfl or othl or (pcnt or 0) < 3:
            live_bad_or_lt3.add(d)
    live_ok_mask_T = np.array([d not in live_bad_or_lt3 for d in uni.dids], dtype=bool)

    # 3. Build leak-free flags at T and full flags at T_now
    lt_posts_full = np.full(n_full, 999999, dtype=np.int64)
    sup_full = np.array([d in sup_dids_set for d in full_dids], dtype=bool)
    for pq_f in sorted((s10_dir / "O2/prod").glob("j*.parquet")):
        t = pq.read_table(pq_f, columns=["p", "lifetime_posts", "supplement"])
        pf = t["p"].to_numpy().astype(np.int64)
        ok_f = pf < n_full
        lt_posts_full[pf[ok_f]] = t["lifetime_posts"].to_numpy().astype(np.int64)[ok_f]
        sup_full[pf[ok_f]] |= t["supplement"].to_numpy().astype(bool)[ok_f]

    flags_T_summary = build_leakfree_flags_T(
        train_t_dir,
        cand5_t_dir,
        audit_dir / "packs.pkl",
        lt_posts_T,
        sup_T,
        out_dir,
    )
    flags_full_summary = build_flags_full(cand5_full_dir, audit_dir, lt_posts_full, sup_full, out_dir)
    print("[flags_T]", json.dumps(flags_T_summary), flush=True)
    print("[flags_full]", json.dumps(flags_full_summary), flush=True)

    # 4. Load M2Data ONCE and isolate M2-val ONLY (NEVER touch split == 'test')
    m2_data = et.M2Data(lake / "s3/m2", uni, base_exclude, "clickhouse")
    val_users = np.flatnonzero(m2_data.split == "val")
    assert (m2_data.split[val_users] == "val").all()
    ou, oi = m2_data.outcomes
    follow_out: dict[int, set[int]] = {}
    for u_i, p_i in zip(ou.tolist(), oi.tolist()):
        if m2_data.split[u_i] == "val":
            follow_out.setdefault(u_i, set()).add(p_i)
    engaged_out: dict[int, set[int]] | None = None
    if m2_data.engaged is not None:
        engaged_out = {}
        for u_i, p_i in zip(m2_data.engaged[0].tolist(), m2_data.engaged[1].tolist()):
            if m2_data.split[u_i] == "val":
                engaged_out.setdefault(u_i, set()).add(p_i)

    seeds_by_u = [m2_data.seeds(u) for u in range(m2_data.n)]
    excl_by_u = [
        set(seeds_by_u[u].tolist()) | ({int(m2_data.self_idx[u])} if m2_data.self_idx[u] >= 0 else set())
        for u in range(m2_data.n)
    ]

    high_h_T = load_high_H_pairs_T(s14_dir / "s14a-outputs")
    print(f"[m2-val] loaded {len(val_users):,} val users ({len(follow_out):,} with follow outcomes), high-H T pairs={len(high_h_T)//2:,}", flush=True)

    # 5. Define pre-registered in-card rankers (S1 §10; V9 passed so V9 demotion competes; S13c failed so combativeness off)
    cur_tie = 1e-6 * np.log1p(cur_T) + 1e-11 * np.log1p(uni.followers)
    v9_dem_T = np.where(blk_raw_T >= 50, 1.0 / (1.0 + np.maximum(bpk_bd_T, 0.0) / 10.0), 1.0)
    wlin_cur_T = wlin_pair_T * (1.0 + 0.5 * np.log1p(cur_T))

    rankers_T = {
        "community_popularity": uni.followers.astype(np.float64),
        "D_per_sqrt_fol": d_sqrt_T + cur_tie,
        "w_lin_per_sqrt_fol": wlin_pair_T + cur_tie,
        "w_lin_plus_curators": wlin_cur_T + cur_tie,
        "w_lin_plus_curators_v9dem": wlin_cur_T * v9_dem_T + cur_tie,
        "w_lin_lift_uncorrected_diag": wlin_lift_T + cur_tie,
    }

    # Map raw community flags to cand5-T-matched community IDs
    comm_table_T = pq.read_table(cand5_t_dir / "communities.parquet", columns=["community", "top_dids"])
    raw_by_comm_T = s3_prep.raw_label_by_community(comm_table_T, memb_T, {x: i for i, x in enumerate(uni.dids)})

    # Compute Base S3 community_popularity on M2-val (safety exclusions + follow_back|follow_train only, no S13 producer filter)
    fl_base_T = np.load(out_dir / "flags_T_theta05.npz")
    base_raw_bad_T = fl_base_T["follow_back"].astype(bool) | fl_base_T["follow_train"].astype(bool)
    base_comm_bad_T = np.zeros(part.n_comm, dtype=bool)
    for c_id, rlab in raw_by_comm_T.items():
        if c_id < part.n_comm:
            base_comm_bad_T[c_id] = bool(base_raw_bad_T[rlab])
    base_s3_val = fast_score_m2_val(
        part,
        uni,
        np.ones(uni.n, dtype=bool),
        np.zeros_like(base_raw_bad_T),
        memb_T,
        base_comm_bad_T,
        {"base_s3_community_popularity": uni.followers.astype(np.float64)},
        val_users,
        seeds_by_u,
        excl_by_u,
        follow_out,
        engaged_out,
        set(),
        co_off,
        co_dst,
        co_cnt,
    )
    base_s3_pop_mean = float(base_s3_val["follow_vecs"]["base_s3_community_popularity"].mean())
    base_s3_vec_by_u = {
        int(u): float(v)
        for u, v in zip(
            base_s3_val["eval_users"].tolist(),
            base_s3_val["follow_vecs"]["base_s3_community_popularity"].tolist(),
        )
    }
    print(f"[m2-val] Base S3 community_popularity p@25={base_s3_pop_mean:.6f} (n={base_s3_val['n_val_users_evaluated']:,}, cards={base_s3_val['n_cards_eligible']:,})", flush=True)

    # Precompute supplement rule (3)&(4) eligibility at T per k in {3, 10, 30}
    sup_idx_T = np.flatnonzero(sup_T)
    sup_elig_by_k: dict[int, np.ndarray] = {}
    for k in (3, 10, 30):
        arr = np.zeros(uni.n, dtype=bool)
        for pi in sup_idx_T:
            r = sup_act.get(uni.dids[int(pi)])
            if r and r.get("sample_reaches_T_minus_12m") and (r.get("postsCount") or 0) >= 3:
                if (r.get("originals_12m_before_T") or 0) >= k and (r.get("originals_30d_before_T") or 0) >= 1:
                    arr[pi] = True
        sup_elig_by_k[k] = arr

    # 6. Sweep all 12 (k, m, theta) cells x 6 rankers on M2-val ONLY
    grid_rows = []
    cell_results_by_key: dict[tuple[int, int, float], dict] = {}

    for theta, tag in ((0.3, "03"), (0.5, "05")):
        fl = np.load(out_dir / f"flags_T_theta{tag}.npz")
        raw_bad = fl["follow_back"].astype(bool) | fl["follow_train"].astype(bool) | fl["low_post_follow_heavy"].astype(bool)
        comm_bad = np.zeros(part.n_comm, dtype=bool)
        for c_id, rlab in raw_by_comm_T.items():
            if c_id < part.n_comm:
                comm_bad[c_id] = bool(raw_bad[rlab])

        for k in (3, 10, 30):
            r34 = np.where(sup_T, sup_elig_by_k[k], (orig12_T >= k) & (orig30_T >= 1))
            for m in (3, 10):
                r5 = (cur_T >= 1) | (le_disc_T >= m)
                elig_mask = live_ok_mask_T & r34 & r5
                cell_res = fast_score_m2_val(
                    part,
                    uni,
                    elig_mask,
                    raw_bad,
                    memb_T,
                    comm_bad,
                    rankers_T,
                    val_users,
                    seeds_by_u,
                    excl_by_u,
                    follow_out,
                    engaged_out,
                    high_h_T,
                    co_off,
                    co_dst,
                    co_cnt,
                )
                cell_results_by_key[(k, m, theta)] = cell_res
                pop_mean = float(cell_res["follow_vecs"]["community_popularity"].mean())
                for rname in rankers_T:
                    fmean = float(cell_res["follow_vecs"][rname].mean())
                    emean = float(cell_res["engaged_vecs"][rname].mean()) if engaged_out is not None else 0.0
                    ratio_pop = fmean / max(pop_mean, 1e-12)
                    ratio_base_pop = fmean / max(base_s3_pop_mean, 1e-12)
                    row = {
                        "k": k,
                        "m": m,
                        "theta": theta,
                        "ranker": rname,
                        "n_cards": cell_res["n_cards_eligible"],
                        "n_eligible_producers": cell_res["n_members_eligible"],
                        "n_val_users": cell_res["n_val_users_evaluated"],
                        "m2_val_follow_p25": fmean,
                        "m2_val_engaged_p25": emean,
                        "m2_val_ratio_vs_s13_comm_pop": ratio_pop,
                        "m2_val_ratio_vs_base_s3_comm_pop": ratio_base_pop,
                        "fill_accounts": cell_res["fill_accounts"][rname],
                    }
                    grid_rows.append(row)

    # 7. Load cand5-full-matched production features & audit S13(b) on theta in {0.3, 0.5} so the chosen M2-val cell satisfies S13(b)
    wt_prod = pq.read_table(
        s11_dir / "w_lin_prod.parquet",
        columns=[
            "pid_full",
            "w_lin_pair_per_sqrt_fol",
            "w_lin_lift_per_sqrt_fol",
            "w_lin_literal_per_sqrt_fol",
            "D_per_sqrt_fol",
            "light_engagers_discounted",
            "a_half_curators",
            "followers",
        ],
    )
    wp_f = wt_prod["pid_full"].to_numpy().astype(np.int64)
    ok_wf = wp_f < n_full
    wp_f = wp_f[ok_wf]
    wlin_pair_F = np.zeros(n_full, dtype=np.float64)
    wlin_lift_F = np.zeros(n_full, dtype=np.float64)
    wlin_lit_F = np.zeros(n_full, dtype=np.float64)
    d_sqrt_F = np.zeros(n_full, dtype=np.float64)
    le_disc_F = np.zeros(n_full, dtype=np.float64)
    cur_F = np.zeros(n_full, dtype=np.float64)
    fol_F = np.ones(n_full, dtype=np.float64)
    wlin_pair_F[wp_f] = wt_prod["w_lin_pair_per_sqrt_fol"].to_numpy().astype(np.float64)[ok_wf]
    wlin_lift_F[wp_f] = wt_prod["w_lin_lift_per_sqrt_fol"].to_numpy().astype(np.float64)[ok_wf]
    wlin_lit_F[wp_f] = wt_prod["w_lin_literal_per_sqrt_fol"].to_numpy().astype(np.float64)[ok_wf]
    d_sqrt_F[wp_f] = wt_prod["D_per_sqrt_fol"].to_numpy().astype(np.float64)[ok_wf]
    le_disc_F[wp_f] = wt_prod["light_engagers_discounted"].to_numpy().astype(np.float64)[ok_wf]
    cur_F[wp_f] = wt_prod["a_half_curators"].to_numpy().astype(np.float64)[ok_wf]
    fol_F[wp_f] = np.maximum(wt_prod["followers"].to_numpy().astype(np.float64)[ok_wf], 1.0)

    orig12_F = np.zeros(n_full, dtype=np.int64)
    orig30_F = np.zeros(n_full, dtype=np.int64)
    blk_raw_F = np.zeros(n_full, dtype=np.int64)
    bpk_bd_F = np.zeros(n_full, dtype=np.float64)
    for pq_f in sorted((s10_dir / "O2/prod").glob("j*.parquet")):
        t = pq.read_table(
            pq_f,
            columns=["p", "originals_12mo", "originals_30d", "live_blockers_raw", "blockers_per_1k_burst_discounted", "a_half_curators"],
        )
        pf = t["p"].to_numpy().astype(np.int64)
        ok_f = pf < n_full
        pf = pf[ok_f]
        orig12_F[pf] = t["originals_12mo"].to_numpy().astype(np.int64)[ok_f]
        orig30_F[pf] = t["originals_30d"].to_numpy().astype(np.int64)[ok_f]
        blk_raw_F[pf] = t["live_blockers_raw"].to_numpy().astype(np.int64)[ok_f]
        bpk_bd_F[pf] = np.nan_to_num(t["blockers_per_1k_burst_discounted"].to_numpy().astype(np.float64)[ok_f], nan=0.0)
        cur_F[pf] = np.maximum(cur_F[pf], t["a_half_curators"].to_numpy().astype(np.float64)[ok_f])

    memb_full = np.load(cand5_full_dir / "membership.npy").astype(np.int64)
    kf_full = pq.read_table(cand5_full_dir / "known_for.parquet", columns=["pid", "community", "rank"])
    pf_kf = kf_full["pid"].to_numpy().astype(np.int64)
    cf_kf = kf_full["community"].to_numpy().astype(np.int64)
    rf_kf = kf_full["rank"].to_numpy()
    prim_full = np.full(n_full, -1, dtype=np.int64)
    prim_full[pf_kf[rf_kf == 0]] = cf_kf[rf_kf == 0]

    checked_ok_dids = set()
    for d, fnd, hnd, pcnt, modl, selfl, othl, rstat in zip(
        *(
            pq.read_table(
                s5_dir / "profiles.parquet",
                columns=["did", "found", "handle", "postsCount", "any_mod_label", "self_labels", "other_labels", "repo_status"],
            )[c].to_pylist()
            for c in ["did", "found", "handle", "postsCount", "any_mod_label", "self_labels", "other_labels", "repo_status"]
        )
    ):
        if fnd and hnd != "handle.invalid" and not modl and not selfl and not othl and (pcnt or 0) >= 3 and rstat == "active":
            checked_ok_dids.add(d)
    base_live_ok_F = np.array([(d not in base_exclude) and (d in checked_ok_dids) for d in full_dids], dtype=bool)
    not_purged_F = np.array([d not in purge_dids for d in full_dids], dtype=bool)

    now_22 = dt.datetime(2026, 10, 23, tzinfo=dt.timezone.utc)

    def build_elig_F_for(k_v: int, m_v: int, theta_v: float) -> tuple[np.ndarray, np.ndarray]:
        tg = "03" if theta_v == 0.3 else "05"
        fl_f = np.load(out_dir / f"flags_full_theta{tg}.npz")
        rbad = fl_f["follow_back"] | fl_f["follow_train"] | fl_f["low_post_follow_heavy"]
        sup_e = np.zeros(n_full, dtype=bool)
        for pi in np.flatnonzero(sup_full):
            r = sup_act.get(full_dids[int(pi)])
            if r and r.get("feed_ok") and (r.get("postsCount") or 0) >= max(3, k_v):
                newest = r.get("newest_original")
                if newest:
                    ts = dt.datetime.fromisoformat(newest.replace("Z", "+00:00"))
                    if (now_22 - ts).days <= 30 and (r.get("originals_sampled") or 0) >= min(k_v, 3):
                        sup_e[pi] = True
        el = (
            base_live_ok_F
            & (~rbad[np.maximum(memb_full, 0)])
            & np.where(sup_full, sup_e, (orig12_F >= k_v) & (orig30_F >= 1))
            & ((cur_F >= 1) | (le_disc_F >= m_v))
        )
        return el, rbad

    # Sample 40+40 for both theta=0.3 and theta=0.5 on the top (k=3, m=3) rule using fixed seed 20261030
    s13b_picks_by_theta: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for theta_v in (0.3, 0.5):
        el_t, rbad_t = build_elig_F_for(3, 3, theta_v)
        unfl_t = (~rbad_t[np.maximum(memb_full, 0)]) & not_purged_F
        rm_p = np.flatnonzero(unfl_t & (~el_t) & (prim_full >= 0))
        kp_p = np.flatnonzero(unfl_t & el_t & (prim_full >= 0))
        rng_t = np.random.default_rng(20261030)
        s13b_picks_by_theta[theta_v] = (
            rng_t.choice(rm_p, 40, replace=False),
            rng_t.choice(kp_p, 40, replace=False),
        )

    s13b_g1_rows = [ln.rstrip("\n").split("\t") for ln in open(out_dir / "sample_s13b.tsv")]
    pre_check_dids = sorted(
        {
            full_dids[int(p)]
            for rm_a, kp_a in s13b_picks_by_theta.values()
            for p in list(rm_a) + list(kp_a)
        }
        | {r[2] for r in s13b_g1_rows}
    )

    prof_live: dict[str, dict | None] = {}
    live_hashes: list[str] = []
    feed_live: dict[str, tuple[int, list[str]]] = {}

    def fetch_orig_feed(d: str) -> tuple[str, int, list[str], str]:
        code, body, rh = xrpc_get_json("app.bsky.feed.getAuthorFeed", {"actor": d, "filter": "posts_no_replies", "limit": 40})
        ts = []
        if code == 200:
            for it in body.get("feed") or []:
                post = it.get("post") or {}
                rec = post.get("record") or {}
                if it.get("reason") or rec.get("reply") or (post.get("author") or {}).get("did") != d:
                    continue
                if rec.get("createdAt"):
                    ts.append(rec["createdAt"])
        return d, code, ts, rh

    def hydrate_dids(dids_list: list[str]) -> None:
        todo = [d for d in dids_list if d not in prof_live]
        for i in range(0, len(todo), 25):
            b_dids = todo[i : i + 25]
            code, body, rh = xrpc_get_json("app.bsky.actor.getProfiles", {"actors": b_dids})
            live_hashes.append(rh)
            got = {p["did"]: p for p in (body.get("profiles") or [])} if code == 200 else {}
            for d in b_dids:
                prof_live[d] = got.get(d)
        todo_f = [d for d in dids_list if d not in feed_live]
        if todo_f:
            with ThreadPoolExecutor(max_workers=8) as ex:
                for d, code, ts, rh in ex.map(fetch_orig_feed, todo_f):
                    feed_live[d] = (code, ts)
                    live_hashes.append(rh)

    hydrate_dids(pre_check_dids)

    now_utc = dt.datetime.now(dt.timezone.utc)
    d90 = now_utc - dt.timedelta(days=90)
    d30 = now_utc - dt.timedelta(days=30)
    d365 = now_utc - dt.timedelta(days=365)

    def eval_acct(d: str) -> dict:
        pr = prof_live.get(d)
        fcode, ts_strs = feed_live.get(d, (-1, []))
        if not pr or fcode != 200:
            return {"junk": True, "good": False, "reason": "missing_or_takendown"}
        labs = [l.get("val", "") for l in (pr.get("labels") or []) if not l.get("neg")]
        if labs or pr.get("handle") == "handle.invalid":
            return {"junk": True, "good": False, "reason": "labelled_or_invalid"}
        pcnt = int(pr.get("postsCount") or 0)
        fcnt = int(pr.get("followersCount") or 0)
        fwcnt = int(pr.get("followsCount") or 0)
        if pcnt < 3 or (pcnt <= 10 and fwcnt >= 0.8 * max(fcnt, 1)):
            return {"junk": True, "good": False, "reason": "low_posts_or_follow_heavy"}
        dts = []
        for s in ts_strs:
            try:
                dts.append(dt.datetime.fromisoformat(s.replace("Z", "+00:00")))
            except Exception:
                pass
        o90 = sum(t >= d90 for t in dts)
        o30 = sum(t >= d30 for t in dts)
        o365 = sum(t >= d365 for t in dts)
        if o90 == 0:
            return {"junk": True, "good": False, "reason": "dormant_no_originals_90d"}
        good = bool(o365 >= 3 and o30 >= 1 and fwcnt < 2.0 * max(fcnt, 25))
        return {"junk": False, "good": good, "reason": "ok", "o30": o30, "o90": o90, "o365": o365}

    s13b_by_theta: dict[str, dict] = {}
    theta_passes_s13b: dict[float, bool] = {}
    for theta_v in (0.3, 0.5):
        rm_a, kp_a = s13b_picks_by_theta[theta_v]
        f_rm = [eval_acct(full_dids[int(p)]) for p in rm_a]
        f_kp = [eval_acct(full_dids[int(p)]) for p in kp_a]
        ok_b = bool((sum(x["junk"] for x in f_rm) >= 2 * sum(x["junk"] for x in f_kp)) and sum(x["good"] for x in f_rm) <= 4)
        theta_passes_s13b[theta_v] = ok_b
        s13b_by_theta[str(theta_v)] = {
            "removed_junk_rate": float(sum(x["junk"] for x in f_rm) / 40),
            "kept_junk_rate": float(sum(x["junk"] for x in f_kp) / 40),
            "junk_rate_ratio": float((sum(x["junk"] for x in f_rm) / 40) / max(sum(x["junk"] for x in f_kp) / 40, 1e-9)),
            "removed_good_follows": int(sum(x["good"] for x in f_rm)),
            "kept_good_follows": int(sum(x["good"] for x in f_kp)),
            "pass": ok_b,
        }

    # Select M2-val winner among cells whose theta passes S13(b) (fallback to all cells if neither passed)
    any_s13b_ok = any(theta_passes_s13b.values())
    best_key = (-1.0, -1.0)
    best_spec = None
    for row in grid_rows:
        if row["ranker"] in ("community_popularity", "w_lin_lift_uncorrected_diag"):
            continue
        if any_s13b_ok and not theta_passes_s13b[row["theta"]]:
            continue
        cand_key = (row["m2_val_follow_p25"], row["m2_val_engaged_p25"])
        if cand_key > best_key:
            best_key = cand_key
            best_spec = row

    assert best_spec is not None
    best_cell_vecs = cell_results_by_key[(best_spec["k"], best_spec["m"], best_spec["theta"])]
    print(
        f"[m2-val] WINNER on M2-val (S13b-gated={any_s13b_ok}): k={best_spec['k']}, m={best_spec['m']}, "
        f"theta={best_spec['theta']}, ranker={best_spec['ranker']} "
        f"(p@25={best_spec['m2_val_follow_p25']:.6f}, vs_base_s3_pop={best_spec['m2_val_ratio_vs_base_s3_comm_pop']:.4f}x, "
        f"vs_s13_pop={best_spec['m2_val_ratio_vs_s13_comm_pop']:.4f}x, cards={best_spec['n_cards']})",
        flush=True,
    )

    # 8. Bootstrap 95% CIs on M2-val for the winning cell (against both Base S3 pop & S13 cell pop & V8(ii) D_per_sqrt_fol)
    r_order = list(rankers_T.keys())
    aligned_base_s3 = np.array([base_s3_vec_by_u.get(int(u), 0.0) for u in best_cell_vecs["eval_users"].tolist()], dtype=np.float64)
    X_val = np.column_stack([best_cell_vecs["follow_vecs"][rn] for rn in r_order] + [aligned_base_s3])
    M_val = np.ones_like(X_val, dtype=bool)
    pt_val, bt_val = et.bootstrap(X_val, M_val, et.BOOT_N, et.BOOT_SEED)
    idx_pop = r_order.index("community_popularity")
    idx_base_pop = len(r_order)
    idx_d = r_order.index("D_per_sqrt_fol")
    idx_w = r_order.index("w_lin_per_sqrt_fol")
    idx_win = r_order.index(best_spec["ranker"])

    v8ii_val = {
        "winning_cell": {"k": best_spec["k"], "m": best_spec["m"], "theta": best_spec["theta"]},
        "chosen_ranker": best_spec["ranker"],
        "n_val_users": int(X_val.shape[0]),
        "base_s3_community_popularity_aligned": {
            "m2_val_p25": float(pt_val[idx_base_pop]),
            "ci95": et.ci(bt_val[:, idx_base_pop]),
        },
        "rankers_on_winning_cell": {
            rn: {
                "m2_val_p25": float(pt_val[i]),
                "ci95": et.ci(bt_val[:, i]),
                "ratio_vs_s13_comm_pop": float(pt_val[i] / pt_val[idx_pop]),
                "ratio_vs_s13_comm_pop_ci95": et.ci(bt_val[:, i] / bt_val[:, idx_pop]),
                "ratio_vs_base_s3_comm_pop": float(pt_val[i] / pt_val[idx_base_pop]),
                "ratio_vs_base_s3_comm_pop_ci95": et.ci(bt_val[:, i] / bt_val[:, idx_base_pop]),
            }
            for i, rn in enumerate(r_order)
        },
        "w_lin_pair_vs_D_rel": {
            "rel": float(pt_val[idx_w] / pt_val[idx_d] - 1.0),
            "ci95": et.ci(bt_val[:, idx_w] / bt_val[:, idx_d] - 1.0),
        },
        "winner_vs_D_rel": {
            "rel": float(pt_val[idx_win] / pt_val[idx_d] - 1.0),
            "ci95": et.ci(bt_val[:, idx_win] / bt_val[:, idx_d] - 1.0),
        },
    }

    win_k = int(best_spec["k"])
    win_m = int(best_spec["m"])
    win_theta = float(best_spec["theta"])
    win_tag = "03" if win_theta == 0.3 else "05"
    elig_F, raw_bad_full = build_elig_F_for(win_k, win_m, win_theta)

    # S13(d) supplement check across cand5-full-matched cards
    by_card_F: dict[int, list[int]] = collections.defaultdict(list)
    for p in np.flatnonzero(elig_F & (prim_full >= 0)):
        by_card_F[int(prim_full[p])].append(int(p))
    disp_cards_F = {c: ps for c, ps in by_card_F.items() if len(ps) >= 25}
    sup_total_elig = int((elig_F & sup_full & (prim_full >= 0)).sum())
    all_total_elig = int((elig_F & (prim_full >= 0)).sum())
    max_card_sup_share = 0.0
    cards_over_5pct_sup = 0
    for c, ps in disp_cards_F.items():
        sh_s = sum(1 for p in ps if sup_full[p]) / len(ps)
        if sh_s > max_card_sup_share:
            max_card_sup_share = sh_s
        if sh_s >= 0.05:
            cards_over_5pct_sup += 1
    s13d_summary = {
        "eligible_producers_with_primary": all_total_elig,
        "eligible_supplement_producers": sup_total_elig,
        "overall_supplement_share": float(sup_total_elig / max(all_total_elig, 1)),
        "cards_ge25_eligible": len(disp_cards_F),
        "max_card_supplement_share": float(max_card_sup_share),
        "cards_ge_5pct_supplement": cards_over_5pct_sup,
        "d17_triggered": bool((sup_total_elig / max(all_total_elig, 1)) >= 0.02 or cards_over_5pct_sup > 0),
    }

    # Compute winning ranker score on cand5-full-matched & emit ranker parquets for S16/S17
    cur_tie_F = 1e-6 * np.log1p(cur_F) + 1e-11 * np.log1p(fol_F)
    v9_dem_F = np.where(blk_raw_F >= 50, 1.0 / (1.0 + np.maximum(bpk_bd_F, 0.0) / 10.0), 1.0)
    wlin_cur_F = wlin_pair_F * (1.0 + 0.5 * np.log1p(cur_F))
    rankers_F_map = {
        "D_per_sqrt_fol": d_sqrt_F + cur_tie_F,
        "w_lin_per_sqrt_fol": wlin_pair_F + cur_tie_F,
        "w_lin_plus_curators": wlin_cur_F + cur_tie_F,
        "w_lin_plus_curators_v9dem": wlin_cur_F * v9_dem_F + cur_tie_F,
        "w_lin_lift_uncorrected_diag": wlin_lift_F + cur_tie_F,
    }
    win_scores_F = rankers_F_map[best_spec["ranker"]]

    pq.write_table(
        pa.table({
            "pid": pa.array(np.arange(n_full, dtype=np.int32)),
            "did": pa.array(full_dids, pa.string()),
            "eligible": pa.array(elig_F),
            "supplement": pa.array(sup_full),
            "score": pa.array(win_scores_F, pa.float64()),
            "D_per_sqrt_fol": pa.array(rankers_F_map["D_per_sqrt_fol"], pa.float64()),
            "w_lin_per_sqrt_fol": pa.array(rankers_F_map["w_lin_per_sqrt_fol"], pa.float64()),
            "w_lin_plus_curators": pa.array(rankers_F_map["w_lin_plus_curators"], pa.float64()),
            "w_lin_plus_curators_v9dem": pa.array(rankers_F_map["w_lin_plus_curators_v9dem"], pa.float64()),
        }),
        str(out_dir / "s13_prod_eligibility_and_scores.parquet"),
        compression="zstd",
    )

    # Also write eval eligibility & scores at T for S16 M2-test report-once
    win_fl_T = np.load(out_dir / f"flags_T_theta{win_tag}.npz")
    win_raw_bad_T = win_fl_T["follow_back"] | win_fl_T["follow_train"] | win_fl_T["low_post_follow_heavy"]
    win_r34_T = np.where(sup_T, sup_elig_by_k[win_k], (orig12_T >= win_k) & (orig30_T >= 1))
    win_elig_T = (
        uni.eligible
        & live_ok_mask_T
        & (~win_raw_bad_T[np.maximum(memb_T, 0)])
        & win_r34_T
        & ((cur_T >= 1) | (le_disc_T >= win_m))
    )
    pq.write_table(
        pa.table({
            "pid_T": pa.array(np.arange(uni.n, dtype=np.int32)),
            "did": pa.array(uni.dids, pa.string()),
            "eligible": pa.array(win_elig_T),
            "supplement": pa.array(sup_T),
            "score": pa.array(rankers_T[best_spec["ranker"]], pa.float64()),
            "D_per_sqrt_fol": pa.array(rankers_T["D_per_sqrt_fol"], pa.float64()),
            "w_lin_per_sqrt_fol": pa.array(rankers_T["w_lin_per_sqrt_fol"], pa.float64()),
            "w_lin_plus_curators": pa.array(rankers_T["w_lin_plus_curators"], pa.float64()),
            "w_lin_plus_curators_v9dem": pa.array(rankers_T["w_lin_plus_curators_v9dem"], pa.float64()),
        }),
        str(out_dir / "s13_eval_T_eligibility_and_scores.parquet"),
        compression="zstd",
    )

    # 9. Draw 40 bundle accounts from top-25 of 40 distinct non-sensitive displayable cards on the locked rule
    cl_comm = {
        int(r["community"]): list(r.get("sensitivity") or [])
        for r in pq.read_table(lake / "claude-labels-cand5-20261028/community_labels.parquet").to_pylist()
    }
    nonsens_cards = sorted([c for c in disp_cards_F if not cl_comm.get(c)])
    rng_bundle = np.random.default_rng(20261030)
    sample_cards = rng_bundle.choice(nonsens_cards, min(40, len(nonsens_cards)), replace=False)
    bundle_sample_40 = []
    for c in sample_cards:
        ps = np.array(disp_cards_F[int(c)], dtype=np.int64)
        top_p = int(ps[np.lexsort((ps, -win_scores_F[ps]))][0])
        bundle_sample_40.append({"community": int(c), "pid": top_p, "did": full_dids[top_p]})

    hydrate_dids(sorted({x["did"] for x in bundle_sample_40}))

    g1_rm = [eval_acct(r[2]) for r in s13b_g1_rows if r[0] == "removed"]
    g1_kp = [eval_acct(r[2]) for r in s13b_g1_rows if r[0] == "kept"]
    bundle_40_eval = [{**b, **eval_acct(b["did"])} for b in bundle_sample_40]

    s13b_final = {
        "g1_probe_diagnostic_posts_no_replies": {
            "removed_junk_rate": float(sum(x["junk"] for x in g1_rm) / 40),
            "kept_junk_rate": float(sum(x["junk"] for x in g1_kp) / 40),
            "junk_rate_ratio": float((sum(x["junk"] for x in g1_rm) / 40) / max(sum(x["junk"] for x in g1_kp) / 40, 1e-9)),
            "removed_good_follows": int(sum(x["good"] for x in g1_rm)),
            "pass": bool(
                (sum(x["junk"] for x in g1_rm) >= 2 * sum(x["junk"] for x in g1_kp))
                and sum(x["good"] for x in g1_rm) <= 4
            ),
        },
        "s13_full_eligibility_by_theta": s13b_by_theta,
        "s13_full_eligibility_40_40": s13b_by_theta[str(win_theta)],
        "live_bundle_sample_40": {
            "checked": len(bundle_40_eval),
            "live_active_nonjunk": int(sum(not x["junk"] for x in bundle_40_eval)),
            "safety_violations": int(sum(x["reason"] in ("missing_or_takendown", "labelled_or_invalid") for x in bundle_40_eval)),
            "accounts": bundle_40_eval,
        },
    }

    receipt = {
        "step": "S13 Good-First-Follow v0 eligibility & in-card ranker selection (M2-val ONLY)",
        "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md row S13, §3.3",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md §9.3 V8(ii), §9.8 S13, §10",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t0, 2),
        "m2_test_touched": False,
        "flags_T_summary": flags_T_summary,
        "flags_full_summary": flags_full_summary,
        "s13a_gate_90d": json.loads((out_dir / "s13a_gate_90d.json").read_text())["chosen"],
        "s13a_note": "90d light_engagers G_k gate has chosen=null (CLEAN removal >10% when used alone without A-half curators); S13 rule (5) uses the pre-registered disjunction (a_half_curators >= 1 OR light_engagers_discounted >= m).",
        "s13b": s13b_final,
        "s13c": {
            k: v
            for k, v in json.loads((out_dir / "s13c_audit.json").read_text()).items()
            if k != "accounts"
        },
        "s13d_supplement": s13d_summary,
        "m2_val_winner": best_spec,
        "v8_part_ii_m2_val": v8ii_val,
        "m2_val_grid_all_72": grid_rows,
        "live_response_hashes_sha256": hashlib.sha256("\n".join(sorted(live_hashes)).encode()).hexdigest(),
        "outputs": {
            "s13_prod_eligibility_and_scores.parquet": sha256_file(out_dir / "s13_prod_eligibility_and_scores.parquet"),
            "s13_eval_T_eligibility_and_scores.parquet": sha256_file(out_dir / "s13_eval_T_eligibility_and_scores.parquet"),
        },
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({
        "winner": best_spec,
        "v8ii_val_wlin_vs_D_rel": v8ii_val["w_lin_pair_vs_D_rel"],
        "v8ii_val_winner_vs_D_rel": v8ii_val["winner_vs_D_rel"],
        "winner_vs_base_s3_pop": v8ii_val["rankers_on_winning_cell"][best_spec["ranker"]]["ratio_vs_base_s3_comm_pop"],
        "winner_vs_base_s3_pop_ci95": v8ii_val["rankers_on_winning_cell"][best_spec["ranker"]]["ratio_vs_base_s3_comm_pop_ci95"],
        "s13b_by_theta": s13b_by_theta,
        "s13b_chosen": s13b_final["s13_full_eligibility_40_40"],
        "bundle_40": {k: v for k, v in s13b_final["live_bundle_sample_40"].items() if k != "accounts"},
        "s13d": s13d_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
