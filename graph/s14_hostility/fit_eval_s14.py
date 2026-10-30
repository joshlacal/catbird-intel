#!/usr/bin/env python3
"""Mac-side S14 H-matrix validation & fit + V9(i) evaluation (`s14a-20261030a`):

1. Evaluates `S14` (`plan row S14`, `S1 §9.8`):
   - High-`H` cells (`stratum == 1`, top 1% of `k >= 50` pre-`T` block cells by size-matched `H_val`
     on frozen train-only `cand5-T-matched`) vs size-matched low-`H` cells (`stratum == 0` `<50`
     blocks, and `stratum IN (0, 4)` including bottom-50% of `k>=50`) on 30-day post-`T` cross-follow
     outcomes in `(T, Sep 18]`.
   - 2,000-resample user-bucket bootstrap (`ub = 0..99`) 95% CI of the rate ratio, plus `>= 100`
     outcome events power floor.
2. If `S14` passes (and also stored cleanly as an inspectable table regardless, marked `enabled_in_v0`
   per the gate), builds the production `8,500 x 8,500` sparse compatibility/hostility table
   `H_prod_cand5_full.parquet` (`blocks-only` per `V7`, `k >= 50`) using the identical size-matched
   formula on `cand5-full-matched` at `T_now`.
3. Evaluates `V9(i)` (`plan §4.8 V9`, `S1 §9.3`):
   - Top vs bottom tercile of producer `blockers_per_1k` (burst-discounted, list-blocks excluded,
     `>= 50` blockers at `T`) on follower->producer blocks within 30 days of non-bulk follows in
     `(T, T + 30d]`, with 2,000-resample user-bucket bootstrap 95% CI of the rate ratio.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def assign_deciles(items: list[tuple[int, int]]) -> dict[int, int]:
    s_items = sorted(items, key=lambda x: (x[1], x[0]))
    n = max(1, len(s_items))
    return {cid: min(9, (idx * 10) // n) for idx, (cid, _) in enumerate(s_items)}


def bootstrap_rate_ratio(
    nh: np.ndarray,
    yh: np.ndarray,
    nl: np.ndarray,
    yl: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 20261030,
) -> dict:
    """Cluster bootstrap over the 100 user-hash buckets."""
    rh = float(yh.sum() / max(nh.sum(), 1e-12))
    rl = float(yl.sum() / max(nl.sum(), 1e-12))
    rr = float(rh / max(rl, 1e-12))
    diff = float(rh - rl)

    rng = np.random.default_rng(seed)
    m = len(nh)
    idx = rng.integers(0, m, size=(n_boot, m))
    b_rh = yh[idx].sum(axis=1) / np.maximum(nh[idx].sum(axis=1), 1e-12)
    b_rl = yl[idx].sum(axis=1) / np.maximum(nl[idx].sum(axis=1), 1e-12)
    b_rr = b_rh / np.maximum(b_rl, 1e-12)
    b_diff = b_rh - b_rl
    ci_rr = [float(np.quantile(b_rr, 0.025)), float(np.quantile(b_rr, 0.975))]
    ci_diff = [float(np.quantile(b_diff, 0.025)), float(np.quantile(b_diff, 0.975))]
    return {
        "high_follows": float(nh.sum()),
        "high_outcomes": float(yh.sum()),
        "high_rate": rh,
        "low_weighted_follows": float(nl.sum()),
        "low_weighted_outcomes": float(yl.sum()),
        "low_rate": rl,
        "rate_ratio": rr,
        "rate_ratio_ci95": ci_rr,
        "rate_diff": diff,
        "rate_diff_ci95": ci_diff,
        "ci_excludes_1_above": bool(ci_rr[0] > 1.0),
    }


def build_prod_H(s14a_dir: pathlib.Path, out_pq: pathlib.Path) -> dict:
    comm_tb = pq.read_table(s14a_dir / "o5_prod_comm_stats_now.parquet")
    blk_tb = pq.read_table(s14a_dir / "o5_prod_cand5_matched_blocks_k50.parquet")

    s_act: dict[int, int] = {}
    s_tgt: dict[int, int] = {}
    for c, np_v, sf_v, nu_v in zip(
        comm_tb["c"].to_pylist(),
        comm_tb["n_producers"].to_pylist(),
        comm_tb["sum_followers_now"].to_pylist(),
        comm_tb["n_users_now"].to_pylist(),
    ):
        cid = int(c)
        s_act[cid] = max(int(nu_v), int(np_v), 1)
        s_tgt[cid] = max(int(sf_v), int(np_v), 1)

    d_act = assign_deciles(list(s_act.items()))
    d_tgt = assign_deciles(list(s_tgt.items()))

    act_sum = [0.0] * 10
    tgt_sum = [0.0] * 10
    same_sum = [0.0] * 100
    for c in s_act:
        da, dtg = d_act[c], d_tgt[c]
        act_sum[da] += float(s_act[c])
        tgt_sum[dtg] += float(s_tgt[c])
        same_sum[da * 10 + dtg] += float(s_act[c]) * float(s_tgt[c])
    exp_denom = [
        max(act_sum[da] * tgt_sum[dtg] - same_sum[da * 10 + dtg], 1.0)
        for da in range(10) for dtg in range(10)
    ]

    # Use w_block_pairs_out_now summed by actor decile x target decile approximation from k>=50 + community out/in totals
    wb_out_dec = [0.0] * 10
    wb_in_dec = [0.0] * 10
    for c, wbo, wbi in zip(
        comm_tb["c"].to_pylist(),
        comm_tb["w_block_pairs_out_now"].to_pylist(),
        comm_tb["w_block_pairs_in_now"].to_pylist(),
    ):
        cid = int(c)
        wb_out_dec[d_act[cid]] += float(wbo)
        wb_in_dec[d_tgt[cid]] += float(wbi)
    tot_wb = max(sum(wb_out_dec), 1.0)
    r_base = [
        (wb_out_dec[da] * (wb_in_dec[dtg] / tot_wb)) / exp_denom[da * 10 + dtg]
        for da in range(10) for dtg in range(10)
    ]
    g_rate = tot_wb / max(sum(exp_denom), 1.0)

    rows = []
    for ca, cb, na, wb in zip(
        blk_tb["c_actor"].to_pylist(),
        blk_tb["c_target"].to_pylist(),
        blk_tb["n_actors"].to_pylist(),
        blk_tb["w_pairs_burst_disc"].to_pylist(),
    ):
        ca_i, cb_i, na_i, wb_f = int(ca), int(cb), int(na), float(wb)
        if ca_i == cb_i or na_i < 50:
            continue
        sc = d_act[ca_i] * 10 + d_tgt[cb_i]
        rate = r_base[sc] if r_base[sc] > 0 else g_rate
        exp_wb = max(rate * float(s_act[ca_i]) * float(s_tgt[cb_i]), 1e-6)
        hv = wb_f / exp_wb
        rows.append((hv, ca_i, cb_i, na_i, wb_f, exp_wb))

    rows.sort(key=lambda x: (-x[0], x[1], x[2]))
    n_k50 = len(rows)
    n_top1 = max(1, round(0.01 * n_k50)) if n_k50 > 0 else 0

    out_tb = pa.table({
        "c_actor": pa.array([r[1] for r in rows], type=pa.int32()),
        "c_target": pa.array([r[2] for r in rows], type=pa.int32()),
        "n_actors": pa.array([r[3] for r in rows], type=pa.uint64()),
        "w_pairs_burst_disc": pa.array([r[4] for r in rows], type=pa.float64()),
        "expected_w_pairs": pa.array([r[5] for r in rows], type=pa.float64()),
        "h_prod": pa.array([r[0] for r in rows], type=pa.float64()),
        "is_high_top1pct": pa.array([i < n_top1 for i in range(n_k50)], type=pa.bool_()),
    })
    tmp = out_pq.with_name(out_pq.name + ".tmp")
    pq.write_table(out_tb, tmp, compression="zstd")
    os.replace(tmp, out_pq)
    return {
        "k50_cross_cells": n_k50,
        "high_top1pct_cells": n_top1,
        "top1pct_threshold_h": float(rows[n_top1 - 1][0]) if n_top1 > 0 else None,
        "file": str(out_pq),
        "bytes": out_pq.stat().st_size,
        "sha256": sha256_file(out_pq),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s14a-dir", default="~/catbird-lake-local/s14-hostility-20261030/s14a-outputs")
    ap.add_argument("--out-dir", default="~/catbird-lake-local/s14-hostility-20261030")
    args = ap.parse_args()

    s14a_dir = pathlib.Path(args.s14a_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    strata_tb = pq.read_table(s14a_dir / "o5_val_s14_strata_summary.parquet")
    boot_tb = pq.read_table(s14a_dir / "o5_val_s14_boot_buckets.parquet")
    v9i_tb = pq.read_table(s14a_dir / "v9i_terciles.parquet")
    v9i_boot_tb = pq.read_table(s14a_dir / "v9i_terciles_boot.parquet")

    nh = boot_tb["nh_follows"].to_numpy().astype(np.float64)
    yh_any = boot_tb["yh_any"].to_numpy().astype(np.float64)
    yh_ug = boot_tb["yh_ug"].to_numpy().astype(np.float64)

    cmp_vs_low0_any = bootstrap_rate_ratio(
        nh, yh_any,
        boot_tb["wl0_follows"].to_numpy().astype(np.float64),
        boot_tb["yl0_any"].to_numpy().astype(np.float64),
    )
    cmp_vs_low04_any = bootstrap_rate_ratio(
        nh, yh_any,
        boot_tb["wl04_follows"].to_numpy().astype(np.float64),
        boot_tb["yl04_any"].to_numpy().astype(np.float64),
    )
    cmp_vs_low4_any = bootstrap_rate_ratio(
        nh, yh_any,
        boot_tb["wl4_follows"].to_numpy().astype(np.float64),
        boot_tb["yl4_any"].to_numpy().astype(np.float64),
    )
    cmp_vs_low0_ug = bootstrap_rate_ratio(
        nh, yh_ug,
        boot_tb["wl0_follows"].to_numpy().astype(np.float64),
        boot_tb["yl0_ug"].to_numpy().astype(np.float64),
    )

    total_high_outcomes = int(round(float(yh_any.sum())))
    power_floor_met = total_high_outcomes >= 100
    # Primary registered S14 gate: top-1% of k>=50 cells vs size-matched low-H pairs (`stratum IN (0,4)` and `stratum == 0`),
    # >= 100 outcome events in high-H, and 95% rate-ratio CI > 1.0.
    s14_pass = bool(power_floor_met and cmp_vs_low04_any["ci_excludes_1_above"] and cmp_vs_low0_any["ci_excludes_1_above"])

    # V9(i): top tercile (2) vs bottom tercile (0) of burst-discounted blockers/1k at T
    v9_boot_df = v9i_boot_tb.to_pydict()
    v9i_results = {}
    for kind in ("burst_discounted", "raw"):
        n_top = np.zeros(100, dtype=np.float64)
        y_top = np.zeros(100, dtype=np.float64)
        n_bot = np.zeros(100, dtype=np.float64)
        y_bot = np.zeros(100, dtype=np.float64)
        for k, tc, ub, nf, nb in zip(
            v9_boot_df["kind"], v9_boot_df["tercile"], v9_boot_df["ub"], v9_boot_df["n_follows"], v9_boot_df["n_blocked_30d"]
        ):
            if k != kind:
                continue
            if int(tc) == 2:
                n_top[int(ub)] += float(nf)
                y_top[int(ub)] += float(nb)
            elif int(tc) == 0:
                n_bot[int(ub)] += float(nf)
                y_bot[int(ub)] += float(nb)
        v9i_results[kind] = bootstrap_rate_ratio(n_top, y_top, n_bot, y_bot)

    v9i_pass = bool(v9i_results["burst_discounted"]["ci_excludes_1_above"])

    h_prod_rep = build_prod_H(s14a_dir, out_dir / "H_prod_cand5_full.parquet")
    h_prod_rep["enabled_in_v0"] = s14_pass

    report = {
        "step": "S14 H-matrix validation & V9(i) post-T follow-to-block check",
        "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md §3.1 S14, §4.8 V9",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md §9.3 V9, §9.8 S14",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "s14": {
            "pass": s14_pass,
            "decision": "H enabled in v0 (bundle suggestions, KnownFor fill, <=5% preview cap)" if s14_pass else "H off in v0 (underpowered or CI includes 1)",
            "high_H_outcome_events": total_high_outcomes,
            "power_floor_ge_100_met": power_floor_met,
            "strata_summary": strata_tb.to_pylist(),
            "primary_high_top1pct_vs_sizematched_low04_any": cmp_vs_low04_any,
            "high_top1pct_vs_sizematched_low_lt50_any": cmp_vs_low0_any,
            "high_top1pct_vs_sizematched_bottom50_k50_any": cmp_vs_low4_any,
            "high_top1pct_vs_sizematched_low_lt50_up_or_gate_only": cmp_vs_low0_ug,
        },
        "v9_part_i": {
            "pass": v9i_pass,
            "tercile_rows": v9i_tb.to_pylist(),
            "top_vs_bottom_burst_discounted": v9i_results["burst_discounted"],
            "top_vs_bottom_raw_diagnostic": v9i_results["raw"],
        },
        "h_prod_cand5_full": h_prod_rep,
    }
    (out_dir / "s14_v9i_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({
        "s14_pass": s14_pass,
        "s14_high_outcomes": total_high_outcomes,
        "s14_rr_low04": round(cmp_vs_low04_any["rate_ratio"], 4),
        "s14_rr_ci95": [round(x, 4) for x in cmp_vs_low04_any["rate_ratio_ci95"]],
        "v9i_pass": v9i_pass,
        "v9i_rr": round(v9i_results["burst_discounted"]["rate_ratio"], 4),
        "v9i_rr_ci95": [round(x, 4) for x in v9i_results["burst_discounted"]["rate_ratio_ci95"]],
    }, indent=1))


if __name__ == "__main__":
    main()
