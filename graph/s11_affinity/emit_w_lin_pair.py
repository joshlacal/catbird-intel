#!/usr/bin/env python3
"""Add exact zero-interaction-referenced pair-evidence `w_lin_pair` to `w_lin_{eval,prod}.parquet`.

Why this exists:
In Task A (`S1 §3–§4`), `eta(u, p) = theta0 + beta*o_ref + theta_pair*x_pair(u,p) + theta_user*x_user(u) + theta_prod*x_prod(p)`.
In `fit_and_eval.py`, `w_lin_lift` subtracted `eta(x_null=0)` instead of the pair's zero-interaction baseline
`eta_0(u, p) = theta0 + beta*o_ref + theta_pair*x_pair_0 + theta_user*x_user(u) + theta_prod*x_prod(p)`, leaving
`theta_PORIG * log(1+orig90) * S0 = -0.4223 * log(1+orig90) * S0` and `+2.9327 * S_DVP` inside `w_lin_lift`
(which made prolific original-post producers strongly negative, down to -3.76M).

`w_lin_pair(p) = idf_p * sum_{u in aud(p)} a_u * sum_{k in pair+eng} theta_k * (x_k(u,p) - x_{k,0})`
is the exact linear sum of pair-interaction log-odds lifts over zero interaction, using the identical frozen S11 R2 (`noqn`, `alpha=0.5`) weights.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

NON_PAIR_FEATS = {"DU", "DVP", "HEAVY", "MISS", "PFOL", "PORIG", "POSH", "PCUR", "PBLK", "PBLKN"}


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def compute_pair_wlin(o2_dir: pathlib.Path, weights: dict[str, float], n_actors: float, n_full: int) -> np.ndarray:
    pair_feats = [k for k, v in weights.items() if k not in NON_PAIR_FEATS and v != 0.0]
    out = np.zeros(n_full, dtype=np.float64)
    for pq_f in sorted(o2_dir.glob("j*.parquet")):
        cols = ["p", "audience", "S0__a05", "S0__a10"] + [f"{fn}__a05" for fn in pair_feats]
        aux_cols = []
        for fn in pair_feats:
            if fn.endswith("_DSL") or fn.endswith("_DSF"):
                bt = fn[:-4]
                for s in ("L14", "L45", "L120", "ND", "DP", "C7"):
                    ca = f"{bt}_{s}__a05"
                    if ca not in cols and ca not in aux_cols:
                        aux_cols.append(ca)
        t = pq.read_table(pq_f, columns=cols + aux_cols)
        pf = t["p"].to_numpy().astype(np.int64)
        ok_p = pf < n_full
        aud = t["audience"].to_numpy().astype(np.float64)
        s0_raw = t["S0__a05"].to_numpy().astype(np.float64)
        s0_a10 = t["S0__a10"].to_numpy().astype(np.float64)

        s0 = s0_raw.copy()
        bad_s0 = ~np.isfinite(s0)
        if bad_s0.any():
            fin_both = np.isfinite(s0_raw) & np.isfinite(s0_a10) & (s0_raw >= 20000)
            r05 = float(np.median(s0_raw[fin_both] / np.maximum(s0_a10[fin_both], 1.0))) if fin_both.any() else 1.86
            u10 = bad_s0 & np.isfinite(s0_a10)
            s0[u10] = np.clip(r05 * s0_a10[u10], 65504.0, aud[u10])
            sb = ~np.isfinite(s0)
            s0[sb] = np.clip(0.32 * aud[sb], 65504.0, np.maximum(aud[sb], 65504.0))

        def clean_aux(cn: str) -> np.ndarray:
            a = t[cn].to_numpy().astype(np.float64)
            bd = ~np.isfinite(a)
            if bd.any():
                fh = np.isfinite(a) & (s0 >= 10000) & (s0 <= 60000)
                rt = float(np.median(a[fh] / np.maximum(s0[fh], 1.0))) if fh.any() else 0.5
                a = a.copy()
                a[bd] = np.maximum(65504.0, rt * s0[bd])
            return a

        acc = np.zeros(len(pf), dtype=np.float64)
        for fn in pair_feats:
            w = float(weights[fn])
            sk = t[f"{fn}__a05"].to_numpy().astype(np.float64)
            bd = ~np.isfinite(sk)
            if fn.endswith("_DSL") or fn.endswith("_DSF"):
                if bd.any():
                    bt = fn[:-4]
                    Xb = np.column_stack([clean_aux(f"{bt}_{s}__a05") for s in ("L14", "L45", "L120", "ND", "DP", "C7")])
                    ff = (~bd) & np.isfinite(s0_raw) & (s0_raw > 20.0) & (s0_raw <= 700.0)
                    if int(ff.sum()) >= 100:
                        yr = np.clip(90.0 * s0[ff] - sk[ff], 0.0, None)
                        cf, *_ = np.linalg.lstsq(Xb[ff], yr, rcond=None)
                        rh = np.clip(Xb[bd] @ cf, 0.0, 90.0 * s0[bd])
                    else:
                        rh = 0.0
                    sk = sk.copy()
                    sk[bd] = 90.0 * s0[bd] - rh
                delta = sk - 90.0 * s0
            else:
                if bd.any():
                    sk = clean_aux(f"{fn}__a05")
                delta = sk
            acc += w * delta

        idf_p = np.log1p(n_actors / (1.0 + aud))
        vals = acc * idf_p
        out[pf[ok_p]] = vals[ok_p]
    return out


def update_parquet(path: pathlib.Path, w_pair_by_pid: np.ndarray) -> dict:
    tbl = pq.read_table(path)
    pids = tbl["pid_full"].to_numpy().astype(np.int64)
    fol = np.maximum(tbl["followers"].to_numpy().astype(np.float64), 1.0)
    wp = w_pair_by_pid[pids]
    wp_sqrt = wp / np.sqrt(fol)

    cols = {c: tbl[c] for c in tbl.column_names}
    cols["w_lin_pair"] = pa.array(wp, pa.float64())
    cols["w_lin_pair_per_sqrt_fol"] = pa.array(wp_sqrt, pa.float64())
    cols["score"] = pa.array(wp_sqrt, pa.float64())

    new_tbl = pa.table(cols)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(new_tbl, tmp, compression="zstd")
    os.replace(tmp, path)
    return {
        "rows": int(new_tbl.num_rows),
        "w_lin_pair_p50": float(np.median(wp)),
        "w_lin_pair_p99": float(np.percentile(wp, 99)),
        "sha256": sha256_file(path),
    }


def main() -> None:
    lake = pathlib.Path.home() / "catbird-lake-local"
    s10 = lake / "s10-outputs-20261030"
    s11 = lake / "s11-affinity-20261030"
    n_full = sum(1 for _ in open(lake / "s3/matched/cand5-full-matched/producers.tsv"))
    wj = json.loads((s11 / "weights.json").read_text())
    n_actors = float(json.loads((s10 / "O1/summary.json").read_text())["o1"]["actors"]["||"])

    wp_eval = compute_pair_wlin(s10 / "O2/eval", wj["preT_model"]["weights"], n_actors, n_full)
    wp_prod = compute_pair_wlin(s10 / "O2/prod", wj["prod_model"]["weights"], n_actors, n_full)

    r_eval = update_parquet(s11 / "w_lin_eval.parquet", wp_eval)
    r_prod = update_parquet(s11 / "w_lin_prod.parquet", wp_prod)

    rec_path = s11 / "receipt.json"
    rec = json.loads(rec_path.read_text())
    rec["w_lin_pair_correction"] = {
        "eval": r_eval,
        "prod": r_prod,
        "note": "Exact zero-interaction-referenced pair-evidence lift (subtracts passive user/producer baseline eta_0(u,p) so prolific original posting PORIG is not negatively weighted by S0).",
    }
    rec_path.write_text(json.dumps(rec, indent=2))
    print(json.dumps({"eval": r_eval, "prod": r_prod}, indent=2))


if __name__ == "__main__":
    main()
