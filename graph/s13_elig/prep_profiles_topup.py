#!/usr/bin/env python3
"""Step S13a + S5(i) cand5/O7 serving-universe diff for live getProfiles top-up.

1. Evaluates S13a: reruns the pre-registered producer-quality gate (DECISIONS 2026-10-25:
   G_k and G_k+R for k in {1, 2, 5, 10}, >=70% FB removed and <=10% CLEAN removed)
   unchanged on 90-day O2/prod `light_engagers_discounted` over `cand5-full-matched`.
2. Draws the seeded 40 removed + 40 kept unflagged producer sample (`sample_s13b.tsv`, seed=7)
   and the top-100 combative (k >= 50 targets) + 100 size-matched unflagged control sample
   (`sample_s13c.tsv`, seed=20261030).
3. Builds the comprehensive `cand5-full-matched` + `O7` (`partition_map == 'cand5'`) serving
   universe (`cand5_serving_universe.tsv`, `cand5_card_packs.tsv`), diffs against Oct 28
   `profiles.parquet` (excluding `purge_dids.txt`), and writes `topup_dids.txt` for `appview.py`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand5-dir", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s3/matched/cand5-full-matched")
    ap.add_argument("--audit-dir", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/communities-cand1-hubdemote/audit-mass-followers")
    ap.add_argument("--s10-dir", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s10-outputs-20261030")
    ap.add_argument("--s5-oct28", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s5-sweep-20261028")
    ap.add_argument("--s5-oct30", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s5-sweep-20261030")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s13-elig-20261030")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    purge_path = args.s5_oct30 / "purge_dids.txt"
    purge_dids = {ln.strip() for ln in open(purge_path) if ln.startswith("did:")}

    # 1. Load cand5-full-matched producers, membership, flags, and audit arrays
    dids = []
    with open(args.cand5_dir / "producers.tsv") as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            dids.append(parts[1])
    n_prod = len(dids)
    not_purged = np.array([d not in purge_dids for d in dids], dtype=bool)

    fd = np.load(args.audit_dir / "follower_degree.npz")
    st = np.load(args.audit_dir / "producer_stats.npz")
    packs_cnt = np.load(args.audit_dir / "pack_counts.npy")
    followers_st = np.maximum(st["followers"], 1)
    mass = fd["mass3000"] / followers_st
    mutual = st["recip_out"] / followers_st

    memb = np.load(args.cand5_dir / "membership.npy").astype(np.int64)
    fl = np.load(args.cand5_dir / "community_flags.npz")
    fb_c, ft_c, size_c = fl["follow_back"], fl["follow_train"], fl["size"]
    fb = fb_c[memb] & not_purged
    ft = ft_c[memb] & not_purged
    clean = (packs_cnt >= 1) & (mass < 0.4) & (~fb_c[memb]) & (~ft_c[memb]) & not_purged
    other = (~fb_c[memb]) & (~ft_c[memb]) & not_purged

    # 2. Load 90-day O2/prod + lu_discounted_prod
    le_disc = np.zeros(n_prod, dtype=np.float64)
    d_disc = np.zeros(n_prod, dtype=np.float64)
    lu_t = pq.read_table(args.s10_dir / "O2/lu_discounted_prod.parquet")
    lu_p = lu_t["p"].to_numpy().astype(np.int64)
    le_disc[lu_p] = lu_t["light_engagers_discounted"].to_numpy().astype(np.float64)
    d_disc[lu_p] = lu_t["D_discounted"].to_numpy().astype(np.float64)

    aud = np.zeros(n_prod, dtype=np.float64)
    fol_o2 = np.ones(n_prod, dtype=np.float64)
    cur_o2 = np.zeros(n_prod, dtype=np.float64)
    comb_o2 = np.full(n_prod, np.nan, dtype=np.float64)
    sup_o2 = np.zeros(n_prod, dtype=np.uint8)
    orig30 = np.zeros(n_prod, dtype=np.int64)
    orig12 = np.zeros(n_prod, dtype=np.int64)

    for p_pq in sorted((args.s10_dir / "O2/prod").glob("j*.parquet")):
        t = pq.read_table(
            p_pq,
            columns=["p", "audience", "followers", "a_half_curators", "combativeness", "supplement", "originals_30d", "originals_12mo"],
        )
        pids = t["p"].to_numpy().astype(np.int64)
        aud[pids] = t["audience"].to_numpy().astype(np.float64)
        fol_o2[pids] = np.maximum(t["followers"].to_numpy().astype(np.float64), 1.0)
        cur_o2[pids] = t["a_half_curators"].to_numpy().astype(np.float64)
        comb_vals = [float("nan") if v is None else float(v) for v in t["combativeness"].to_pylist()]
        comb_o2[pids] = np.array(comb_vals, dtype=np.float64)
        sup_o2[pids] = t["supplement"].to_numpy().astype(np.uint8)
        orig30[pids] = t["originals_30d"].to_numpy().astype(np.int64)
        orig12[pids] = t["originals_12mo"].to_numpy().astype(np.int64)

    # 3. Evaluate S13a gate unchanged on 90d light_engagers_discounted
    rules = {}
    for k in (1, 2, 5, 10):
        rules[f"G{k}"] = le_disc >= k
        rules[f"G{k}+R"] = (le_disc >= k) & (mutual < 0.6)

    res = {}
    for name, ok in rules.items():
        drop = ~ok
        res[name] = {
            "removed_fb": float(drop[fb].mean()),
            "removed_clean": float(drop[clean].mean()),
            "removed_ft": float(drop[ft].mean()),
            "removed_other": float(drop[other].mean()),
            "removed_all_nonpurged": int(drop[not_purged].sum()),
            "eligible_all_nonpurged": int((ok & not_purged).sum()),
        }
        res[name]["pass"] = bool(res[name]["removed_fb"] >= 0.70 and res[name]["removed_clean"] <= 0.10)

    passing = [n for n in rules if res[n]["pass"]]
    chosen = min(passing, key=lambda n: res[n]["removed_clean"]) if passing else None
    probe = chosen or min(rules, key=lambda n: max(0.70 - res[n]["removed_fb"], 0.0) + max(res[n]["removed_clean"] - 0.10, 0.0))

    C = len(size_c)
    drop_probe = ~rules[probe]
    rm_share = np.bincount(memb[not_purged], weights=drop_probe[not_purged], minlength=C) / np.maximum(
        np.bincount(memb[not_purged], minlength=C), 1
    )
    cand_rm = np.flatnonzero((~fb_c) & (~ft_c) & (rm_share >= 0.5) & (size_c >= 8))
    largest_rm = cand_rm[np.argsort(-size_c[cand_rm])][:20]

    # S13b sample: 40 removed + 40 kept unflagged non-purged producers (seed=7, matching producer_quality.py)
    rng_b = np.random.default_rng(7)
    pick_rm = rng_b.choice(np.flatnonzero(drop_probe & other), 40, replace=False)
    pick_kp = rng_b.choice(np.flatnonzero((~drop_probe) & other), 40, replace=False)
    with open(args.out_dir / "sample_s13b.tsv", "w") as out_f:
        for grp, picks in (("removed", pick_rm), ("kept", pick_kp)):
            for p in picks:
                out_f.write(f"{grp}\t{int(p)}\t{dids[int(p)]}\t{le_disc[int(p)]:.2f}\t{mutual[int(p)]:.4f}\n")

    # S13c sample: top-100 combative (non-null combativeness -> >=50 distinct targets, unflagged, non-purged)
    # + 100 follower-decile-matched unflagged eligible controls with non-top combativeness
    comb_valid = np.isfinite(comb_o2) & other & rules[probe]
    comb_idx = np.flatnonzero(comb_valid)
    top100_comb = comb_idx[np.lexsort((comb_idx, -comb_o2[comb_idx]))][:100]
    top100_set = set(top100_comb.tolist())
    ctrl_pool = np.array([i for i in np.flatnonzero(other & rules[probe]) if i not in top100_set], dtype=np.int64)
    fol_edges = np.quantile(fol_o2[other & rules[probe]], np.linspace(0.1, 0.9, 9))
    comb_dec = np.digitize(fol_o2[top100_comb], fol_edges)
    ctrl_dec = np.digitize(fol_o2[ctrl_pool], fol_edges)
    rng_c = np.random.default_rng(20261030)
    ctrl100 = []
    for dec in range(10):
        need = int((comb_dec == dec).sum())
        if need > 0:
            sub = ctrl_pool[ctrl_dec == dec]
            chosen_sub = rng_c.choice(sub, size=min(need, len(sub)), replace=False)
            ctrl100.extend(chosen_sub.tolist())
    with open(args.out_dir / "sample_s13c.tsv", "w") as out_f:
        for grp, picks in (("top_combative", top100_comb), ("control", ctrl100)):
            for p in picks:
                cv = comb_o2[int(p)]
                cv_s = "null" if np.isnan(cv) else f"{cv:.6f}"
                out_f.write(f"{grp}\t{int(p)}\t{dids[int(p)]}\t{cv_s}\t{int(fol_o2[int(p)])}\n")

    s13a_gate = {
        "step": "S13a_90d_producer_quality_gate",
        "preregistration": "docs/DECISIONS.md 2026-10-25 & S1 §9.8 S13(a): >=70% FB removed and <=10% CLEAN removed",
        "populations_nonpurged": {
            "FB": int(fb.sum()),
            "CLEAN": int(clean.sum()),
            "FT": int(ft.sum()),
            "producers_nonpurged": int(not_purged.sum()),
            "supplement_producers_in_O2": int(sup_o2[not_purged].sum()),
        },
        "rules": res,
        "chosen": chosen,
        "probe_rule_for_checks": probe,
        "unflagged_communities_mostly_removed": int(len(cand_rm)),
        "largest_unflagged_mostly_removed": [
            {"raw_community": int(c), "size": int(size_c[c]), "removed_share": float(rm_share[c])}
            for c in largest_rm
        ],
    }
    (args.out_dir / "s13a_gate_90d.json").write_text(json.dumps(s13a_gate, indent=2))
    print(f"[s13a] chosen={chosen}, probe={probe}, supplement={int(sup_o2[not_purged].sum())}")
    for n, r in res.items():
        print(f"  {n:<7} rm_FB={r['removed_fb']:6.2%} rm_CLEAN={r['removed_clean']:6.2%} rm_FT={r['removed_ft']:6.2%} pass={r['pass']}")

    # 4. Build exact S5(i) cand5 + O7 serving universe (matching serving_universe.py + W_lin + D + O7)
    wlin_prod_path = pathlib.Path.home() / "catbird-lake-local/s11-affinity-20261030/w_lin_prod.parquet"
    wlin_lift_sqrt = np.zeros(n_prod, dtype=np.float64)
    if wlin_prod_path.exists():
        wt = pq.read_table(wlin_prod_path, columns=["pid_full", "w_lin_lift_per_sqrt_fol"])
        wlin_lift_sqrt[wt["pid_full"].to_numpy().astype(np.int64)] = wt["w_lin_lift_per_sqrt_fol"].to_numpy().astype(np.float64)

    kf = pq.read_table(args.cand5_dir / "known_for.parquet", columns=["pid", "did", "community", "rank"]).to_pandas()
    prim = kf[(kf["rank"] == 0) & (~kf["did"].isin(purge_dids))].copy()
    prim["flagged_base"] = (fb_c | ft_c)[memb[prim["pid"].values]]
    cand = prim[~prim["flagged_base"]].copy()
    sz = cand.groupby("community").size()
    disp = set(sz[sz >= 25].index)  # exact S5(i) / S1 §10 displayable community threshold (>=25 base-rule candidates)
    c2 = cand[cand["community"].isin(disp)].copy()
    pids_c2 = c2["pid"].values.astype(np.int64)
    sq_f = np.sqrt(fol_o2[pids_c2])
    c2["r_le"] = le_disc[pids_c2] / sq_f
    c2["r_d"] = d_disc[pids_c2] / sq_f
    c2["r_wlin"] = wlin_lift_sqrt[pids_c2]
    c2["r_cur"] = cur_o2[pids_c2]
    c2["r_fol"] = fol_o2[pids_c2]

    reasons: dict[str, set[str]] = collections.defaultdict(set)
    for col_r, head_n, tag in (
        ("r_le", 50, "top50_le_sqrtfol"),
        ("r_d", 50, "top50_D_sqrtfol"),
        ("r_wlin", 50, "top50_wlin_sqrtfol"),
        ("r_cur", 25, "top25_curators"),
    ):
        for d in c2.sort_values([col_r, "r_fol", "did"], ascending=[False, False, True]).groupby("community").head(head_n)["did"]:
            reasons[d].add(tag)

    cm = pq.read_table(args.cand5_dir / "communities.parquet", columns=["community", "top_dids"]).to_pandas()
    for c_id, lst in zip(cm["community"], cm["top_dids"]):
        if c_id in disp:
            for d in lst:
                if d not in purge_dids:
                    reasons[d].add("card_top_dids")

    did2c = dict(zip(c2["did"], c2["community"]))
    best_packs = collections.defaultdict(list)
    for uri, mem, _ in pickle.load(open(args.audit_dir / "packs.pkl", "rb")):
        cr = uri[5:].split("/")[0]
        c_id = did2c.get(cr)
        if c_id is None or len(mem) < 5 or cr in purge_dids:
            continue
        best_packs[c_id].append((-sum(did2c.get(x) == c_id for x in mem) / len(mem), -len(mem), uri, mem, cr))

    chosen_packs = []
    for c_id, lst in best_packs.items():
        lst.sort(key=lambda x: x[:3])
        for share, neg_n, uri, mem, cr in lst[:2]:
            chosen_packs.append((c_id, uri, abs(share), -neg_n))
            if cr not in purge_dids:
                reasons[cr].add("pack_creator")
            for m_did in mem:
                if m_did not in purge_dids:
                    reasons[m_did].add("pack_member")

    # Add O7 feed creators (`partition_map == 'cand5'`)
    o7 = pq.read_table(args.s10_dir / "O7/O7.parquet").to_pandas()
    o7_c5 = o7[o7["partition_map"] == "cand5"]
    for f_uri in o7_c5["feed_uri"].unique():
        cr_did = f_uri[5:].split("/")[0]
        if cr_did not in purge_dids:
            reasons[cr_did].add("o7_feed_creator")

    # Add S13b and S13c sample DIDs
    for p in list(pick_rm) + list(pick_kp):
        reasons[dids[int(p)]].add("s13b_sample")
    for p in list(top100_comb) + list(ctrl100):
        reasons[dids[int(p)]].add("s13c_sample")

    # Add non-purged S5 positive-control accounts so top-up sweep also re-verifies controls
    ctrl_json = json.loads((args.s5_oct28 / "controls/label_and_onboarding_controls.json").read_text())
    ctrl_dids = set()
    for acc in ctrl_json.get("accounts", []):
        if acc["did"] not in purge_dids:
            ctrl_dids.add(acc["did"])
    for h_rec in json.loads((args.s5_oct28 / "controls/hide_from_recs_live_control.json").read_text()):
        if h_rec["did"] not in purge_dids:
            ctrl_dids.add(h_rec["did"])

    uni_tsv = args.s5_oct30 / "cand5_serving_universe.tsv"
    with open(uni_tsv, "w") as fh:
        for d in sorted(reasons):
            fh.write(f"{d}\t{','.join(sorted(reasons[d]))}\n")

    packs_tsv = args.s5_oct30 / "cand5_card_packs.tsv"
    with open(packs_tsv, "w") as fh:
        for c_id, u, s, n in sorted(chosen_packs):
            fh.write(f"{c_id}\t{u}\t{s:.4f}\t{n}\n")

    # Diff against Oct 28 profiles.parquet
    existing = set(pq.read_table(args.s5_oct28 / "profiles.parquet", columns=["did"])["did"].to_pylist())
    missing_uni = sorted((set(reasons.keys()) - existing) | ctrl_dids)
    topup_txt = args.s5_oct30 / "topup_dids.txt"
    with open(topup_txt, "w") as fh:
        for d in missing_uni:
            fh.write(d + "\n")

    summary = {
        "displayable_communities_superset": len(disp),
        "cand5_o7_universe_dids": len(reasons),
        "already_in_oct28_profiles": len(set(reasons.keys()) & existing),
        "missing_universe_dids": len(set(reasons.keys()) - existing),
        "control_dids_included": len(ctrl_dids),
        "topup_dids_total": len(missing_uni),
        "estimated_getprofiles_batches_25": (len(missing_uni) + 24) // 25,
        "cand5_packs_candidates": len(chosen_packs),
        "o7_cand5_feed_rows": int(len(o7_c5)),
        "o7_cand5_unique_feeds": int(o7_c5["feed_uri"].nunique()),
    }
    (args.out_dir / "topup_prep_summary.json").write_text(json.dumps(summary, indent=2))
    print("[topup]", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
