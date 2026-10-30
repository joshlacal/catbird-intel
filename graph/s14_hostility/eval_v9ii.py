#!/usr/bin/env python3
"""Evaluate V9(ii) (S1 §9.3 V9) across all labelled communities in cand5-full-matched.

Rule (S1 §9.3 V9):
  Over all labelled communities, excluded ones included: among producers the demotion
  would move out of their card's top 25 vs kept producers (same blocker-count decile),
    (a) share of blockers who blocked in a burst or subscribe to a list-block covering them,
    (b) share of blockers in their top-3 blocker communities;
  sensitive-class share of removals reported alongside.
  Bar: (i) Rate ratio CI excluding 1 [already PASS in s14_v9i_report.json: 7.2189];
       (ii) for demoted producers, (a) and (b) each <= 1.5x the kept producers' values.
  If it fails: Demotion off in v0.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from s6picker import keywords  # noqa: E402


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def eval_demotion_variant(
    name: str,
    base_score: np.ndarray,
    demoted_score: np.ndarray,
    primary_comm: np.ndarray,
    cand_mask: np.ndarray,
    n_blockers: np.ndarray,
    share_a: np.ndarray,
    share_b: np.ndarray,
    share_b_assigned: np.ndarray,
    comm_is_labelled: set[int],
    comm_sens_classes: dict[int, list[str]],
    seed: int = 20261028,
    n_boot: int = 2000,
) -> dict:
    """Compare demoted (in base top-25, out of demoted top-25) vs kept (in both top-25)
    among k>=50 blocker producers across ALL labelled communities (excluded ones included),
    matched on blocker-count decile."""
    by_comm: dict[int, list[int]] = collections.defaultdict(list)
    for p in np.flatnonzero(cand_mask & (primary_comm >= 0)):
        c = int(primary_comm[p])
        if c in comm_is_labelled:
            by_comm[c].append(int(p))

    demoted_pids = []
    kept_pids = []
    demoted_comm = []
    for c, pids in sorted(by_comm.items()):
        if len(pids) <= 25:
            # No producer can move out of top 25 if card has <=25 candidates, though kept k>=50 still exist
            arr = np.array(pids, dtype=np.int64)
            top_base = set(arr.tolist())
            top_dem = set(arr.tolist())
        else:
            arr = np.array(pids, dtype=np.int64)
            ord_base = arr[np.lexsort((arr, -base_score[arr]))][:25]
            ord_dem = arr[np.lexsort((arr, -demoted_score[arr]))][:25]
            top_base = set(ord_base.tolist())
            top_dem = set(ord_dem.tolist())
        for p in top_base:
            if n_blockers[p] >= 50:
                if p not in top_dem:
                    demoted_pids.append(p)
                    demoted_comm.append(c)
                else:
                    kept_pids.append(p)

    dem_arr = np.array(demoted_pids, dtype=np.int64)
    kep_arr = np.array(kept_pids, dtype=np.int64)
    all_k50_top25 = np.concatenate([dem_arr, kep_arr])
    if len(dem_arr) == 0 or len(kep_arr) == 0:
        return {"variant": name, "n_demoted_k50": int(len(dem_arr)), "n_kept_k50": int(len(kep_arr)), "pass": False}

    # Blocker-count deciles across all top-25 k>=50 producers
    edges = np.quantile(n_blockers[all_k50_top25], np.linspace(0.1, 0.9, 9))
    dec_dem = np.digitize(n_blockers[dem_arr], edges)
    dec_kep = np.digitize(n_blockers[kep_arr], edges)

    # Decile-matched kept weights so kept distribution over blocker-count deciles matches demoted
    dec_counts_dem = np.bincount(dec_dem, minlength=10).astype(np.float64)
    dec_counts_kep = np.bincount(dec_kep, minlength=10).astype(np.float64)
    w_kep = np.zeros(len(kep_arr), dtype=np.float64)
    for d in range(10):
        if dec_counts_dem[d] > 0 and dec_counts_kep[d] > 0:
            w_kep[dec_kep == d] = dec_counts_dem[d] / dec_counts_kep[d]
    if w_kep.sum() == 0:
        w_kep[:] = 1.0

    def ratio_stats(vals_dem: np.ndarray, vals_kep: np.ndarray) -> dict:
        m_dem = float(vals_dem.mean())
        m_kep_matched = float(np.average(vals_kep, weights=w_kep))
        m_kep_unweighted = float(vals_kep.mean())
        ratio_matched = m_dem / max(m_kep_matched, 1e-12)
        ratio_unweighted = m_dem / max(m_kep_unweighted, 1e-12)
        rng = np.random.default_rng(seed)
        boots = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            id_d = rng.integers(0, len(vals_dem), len(vals_dem))
            id_k = rng.integers(0, len(vals_kep), len(vals_kep))
            wk = w_kep[id_k]
            mk = float(np.average(vals_kep[id_k], weights=wk)) if wk.sum() > 0 else float(vals_kep[id_k].mean())
            boots[b] = float(vals_dem[id_d].mean()) / max(mk, 1e-12)
        ci95 = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        return {
            "demoted_mean": m_dem,
            "kept_decile_matched_mean": m_kep_matched,
            "kept_unweighted_mean": m_kep_unweighted,
            "ratio_decile_matched": ratio_matched,
            "ratio_decile_matched_ci95": ci95,
            "ratio_unweighted": ratio_unweighted,
            "le_1_5x": bool(ratio_matched <= 1.5),
        }

    res_a = ratio_stats(share_a[dem_arr], share_a[kep_arr])
    res_b = ratio_stats(share_b[dem_arr], share_b[kep_arr])
    res_b_assigned = ratio_stats(share_b_assigned[dem_arr], share_b_assigned[kep_arr])

    # Sensitive-class share of removals
    sens_counter = collections.Counter()
    any_sens = 0
    for c in demoted_comm:
        cls = comm_sens_classes.get(c, [])
        if cls:
            any_sens += 1
            for cl in cls:
                sens_counter[cl] += 1

    return {
        "variant": name,
        "communities_evaluated": len(by_comm),
        "n_demoted_k50": int(len(dem_arr)),
        "n_kept_k50": int(len(kep_arr)),
        "part_a_burst_or_listblock": res_a,
        "part_b_top3_blocker_communities": res_b,
        "part_b_top3_assigned_diagnostic": res_b_assigned,
        "sensitive_class_share_of_removals": {
            "demoted_in_any_sensitive_card": any_sens,
            "share_any_sensitive": float(any_sens / len(dem_arr)),
            "by_class_counts": dict(sorted(sens_counter.items())),
            "by_class_shares": {k: float(v / len(dem_arr)) for k, v in sorted(sens_counter.items())},
        },
        "pass": bool(res_a["le_1_5x"] and res_b["le_1_5x"]),
    }


def main() -> None:
    lake = pathlib.Path.home() / "catbird-lake-local"
    cand5_dir = lake / "s3/matched/cand5-full-matched"
    s10_dir = lake / "s10-outputs-20261030"
    s11_dir = lake / "s11-affinity-20261030"
    s14_dir = lake / "s14-hostility-20261030"
    s5_dir = lake / "s5-sweep-20261030"

    # Load producers
    dids = [ln.rstrip("\n").split("\t")[1] for ln in open(cand5_dir / "producers.tsv")]
    n_prod = len(dids)
    purge_dids = {ln.strip() for ln in open(s5_dir / "purge_dids.txt") if ln.startswith("did:")}
    not_purged = np.array([d not in purge_dids for d in dids], dtype=bool)

    # Primary community in cand5-full-matched (across ALL labelled communities, excluded ones included!)
    kf = pq.read_table(cand5_dir / "known_for.parquet", columns=["pid", "community", "rank"])
    pids_kf = kf["pid"].to_numpy().astype(np.int64)
    comm_kf = kf["community"].to_numpy().astype(np.int64)
    rank_kf = kf["rank"].to_numpy()
    primary_comm = np.full(n_prod, -1, dtype=np.int64)
    r0 = rank_kf == 0
    primary_comm[pids_kf[r0]] = comm_kf[r0]

    # Load all labelled communities (Opus + list-name keyword screen)
    cl_t = pq.read_table(lake / "claude-labels-cand5-20261028/community_labels.parquet").to_pylist()
    list_names_all: dict[int, list[str]] = {}
    ln_path = cand5_dir / "audit-quality/community_list_names.tsv"
    if ln_path.exists():
        for ln in open(ln_path):
            parts = ln.rstrip("\n").split("\t")
            if parts[0] != "community":
                list_names_all[int(parts[0])] = parts[6].split("; ") if len(parts) > 6 else []

    comm_is_labelled = set()
    comm_sens_classes: dict[int, list[str]] = {}
    for row in cl_t:
        c = int(row["community"])
        comm_is_labelled.add(c)
        kw = {k for nm in list_names_all.get(c, [])[:6] for k in keywords.classes(nm)}
        sens = sorted(set(row.get("sensitivity") or []) | kw)
        comm_sens_classes[c] = sens

    # Load W_lin_prod & D_prod & O2/prod blockers_per_1k_burst_discounted
    wt = pq.read_table(
        s11_dir / "w_lin_prod.parquet",
        columns=["pid_full", "w_lin_lift_per_sqrt_fol", "D_per_sqrt_fol", "light_engagers_discounted", "a_half_curators"],
    )
    w_lin = np.zeros(n_prod, dtype=np.float64)
    d_sqrt = np.zeros(n_prod, dtype=np.float64)
    le_disc = np.zeros(n_prod, dtype=np.float64)
    cur = np.zeros(n_prod, dtype=np.float64)
    wp = wt["pid_full"].to_numpy().astype(np.int64)
    w_lin[wp] = wt["w_lin_lift_per_sqrt_fol"].to_numpy().astype(np.float64)
    d_sqrt[wp] = wt["D_per_sqrt_fol"].to_numpy().astype(np.float64)
    le_disc[wp] = wt["light_engagers_discounted"].to_numpy().astype(np.float64)
    cur[wp] = wt["a_half_curators"].to_numpy().astype(np.float64)

    bpk_bd = np.zeros(n_prod, dtype=np.float64)
    orig30 = np.zeros(n_prod, dtype=np.int64)
    orig12 = np.zeros(n_prod, dtype=np.int64)
    for pq_f in sorted((s10_dir / "O2/prod").glob("j*.parquet")):
        t = pq.read_table(pq_f, columns=["p", "blockers_per_1k_burst_discounted", "originals_30d", "originals_12mo"])
        pp = t["p"].to_numpy().astype(np.int64)
        ok_p = pp < n_prod
        pp = pp[ok_p]
        bpk_bd[pp] = np.nan_to_num(t["blockers_per_1k_burst_discounted"].to_numpy().astype(np.float64)[ok_p], nan=0.0)
        orig30[pp] = t["originals_30d"].to_numpy().astype(np.int64)[ok_p]
        orig12[pp] = t["originals_12mo"].to_numpy().astype(np.int64)[ok_p]

    # Load v9ii_producers_now.parquet
    v9t = pq.read_table(s14_dir / "s14a-outputs/v9ii_producers_now.parquet")
    n_blk = np.zeros(n_prod, dtype=np.int64)
    sh_a = np.zeros(n_prod, dtype=np.float64)
    sh_b = np.zeros(n_prod, dtype=np.float64)
    sh_b_asg = np.zeros(n_prod, dtype=np.float64)
    vp = v9t["p"].to_numpy().astype(np.int64)
    ok_v = vp < n_prod
    vp = vp[ok_v]
    n_blk[vp] = v9t["n_blockers"].to_numpy().astype(np.int64)[ok_v]
    sh_a[vp] = v9t["share_burst_or_lblk"].to_numpy().astype(np.float64)[ok_v]
    sh_b[vp] = v9t["share_top3_comm"].to_numpy().astype(np.float64)[ok_v]
    sh_b_asg[vp] = v9t["share_top3_comm_assigned"].to_numpy().astype(np.float64)[ok_v]

    # Candidate pool in each labelled community (active non-purged producers with >=3 12mo & >=1 30d originals & (cur>=1 or le>=3))
    cand_mask = not_purged & (orig12 >= 3) & (orig30 >= 1) & ((cur >= 1) | (le_disc >= 3))

    # Standard smooth multiplicative blockers/1k demotion (applied only when n_blockers >= 50 per S1 §10:
    # `1 / (1 + bpk_bd / scale)` for scale in {10.0, 25.0} on both W_lin/sqrt(fol) and D/sqrt(fol))
    k50 = n_blk >= 50
    dem_factor_10 = np.where(k50, 1.0 / (1.0 + np.maximum(bpk_bd, 0.0) / 10.0), 1.0)
    dem_factor_25 = np.where(k50, 1.0 / (1.0 + np.maximum(bpk_bd, 0.0) / 25.0), 1.0)

    variants = {}
    for vname, base_s, fac in (
        ("w_lin_demote_scale10", w_lin, dem_factor_10),
        ("w_lin_demote_scale25", w_lin, dem_factor_25),
        ("D_sqrt_demote_scale10", d_sqrt, dem_factor_10),
    ):
        variants[vname] = eval_demotion_variant(
            vname,
            base_s,
            base_s * fac,
            primary_comm,
            cand_mask,
            n_blk,
            sh_a,
            sh_b,
            sh_b_asg,
            comm_is_labelled,
            comm_sens_classes,
        )

    s14_v9i = json.loads((s14_dir / "s14_v9i_report.json").read_text())
    primary_v = variants["w_lin_demote_scale10"]
    v9_overall_pass = bool(s14_v9i["v9_part_i"]["pass"] and primary_v["pass"])

    report = {
        "step": "V9(ii) brigade/pile-on audit & combined V9 decision",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md §9.3 V9, §10",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs": {
            "v9ii_producers_now.parquet": sha256_file(s14_dir / "s14a-outputs/v9ii_producers_now.parquet"),
            "community_labels.parquet": sha256_file(lake / "claude-labels-cand5-20261028/community_labels.parquet"),
            "w_lin_prod.parquet": sha256_file(s11_dir / "w_lin_prod.parquet"),
            "s14_v9i_report.json": sha256_file(s14_dir / "s14_v9i_report.json"),
        },
        "v9_part_i_pass": bool(s14_v9i["v9_part_i"]["pass"]),
        "v9_part_i_rate_ratio": s14_v9i["v9_part_i"]["top_vs_bottom_burst_discounted"]["rate_ratio"],
        "v9_part_i_ci95": s14_v9i["v9_part_i"]["top_vs_bottom_burst_discounted"]["rate_ratio_ci95"],
        "primary_variant": "w_lin_demote_scale10",
        "v9_part_ii_pass": bool(primary_v["pass"]),
        "v9_overall_pass": v9_overall_pass,
        "note_on_usage": (
            "Per S1 §10: blockers-per-1,000 demotion is 'off by default' and eligible to test on M2-val ONLY if V9 passes."
            if v9_overall_pass
            else "V9 failed part (ii); blockers-per-1,000 demotion is strictly OFF in v0."
        ),
        "variants": variants,
    }
    out_p = s14_dir / "v9ii_report.json"
    out_p.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "v9_part_i_pass": report["v9_part_i_pass"],
        "v9_part_ii_pass": report["v9_part_ii_pass"],
        "v9_overall_pass": report["v9_overall_pass"],
        "primary_a_ratio": primary_v["part_a_burst_or_listblock"]["ratio_decile_matched"],
        "primary_b_ratio": primary_v["part_b_top3_blocker_communities"]["ratio_decile_matched"],
        "n_demoted_k50": primary_v["n_demoted_k50"],
        "n_kept_k50": primary_v["n_kept_k50"],
        "sensitive_share": primary_v["sensitive_class_share_of_removals"]["share_any_sensitive"],
    }, indent=2))


if __name__ == "__main__":
    main()
