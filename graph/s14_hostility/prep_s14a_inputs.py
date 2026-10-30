#!/usr/bin/env python3
"""Prepare privacy-clean, DID-free inputs on the Mac for R-S14a (`s14a-20261030a`):

1. Hashes non-purged `cand5-T-matched/known_for.parquet` (`0..8,224`) and
   `cand5-full-matched/known_for.parquet` (`0..8,499`) by `h = city_hash64(did)`,
   asserting 0 hash collisions and emitting ZERO DID columns:
     - `kf_cand5_T.parquet`: `(h uint64, community int32, score float32, rank uint8)`
     - `kf_cand5_full.parquet`: `(h uint64, community int32, score float32, rank uint8)`
2. Reads purged `O2/eval/j*.parquet` (`T = 2026-08-15`) for producers with `live_blockers_raw >= 50`
   and emits integer-only `prod_t_blockers.parquet`:
     - `(p uint32, followers_T uint32, live_blockers_T uint32, blk_1k_raw_T float32,
         blk_1k_burst_disc_T float32, tercile_burst_disc uint8, tercile_raw uint8)`
3. Writes `prep_receipt.json` with SHA-256 checksums and counts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "vmpass"))
from cityhash import city_hash64  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def atomic_write_parquet(tb: pa.Table, out_path: pathlib.Path) -> None:
    tmp = out_path.with_name(out_path.name + ".tmp")
    pq.write_table(tb, tmp, compression="zstd")
    os.replace(tmp, out_path)


def prep_kf(src_pq: pathlib.Path, out_pq: pathlib.Path, purge_dids: set[str]) -> dict:
    tb = pq.read_table(src_pq, columns=["did", "community", "score", "rank"])
    dids = tb["did"].to_pylist()
    comms = tb["community"].to_pylist()
    scores = tb["score"].to_pylist()
    ranks = tb["rank"].to_pylist()

    did_to_h: dict[str, int] = {}
    h_to_did: dict[int, str] = {}
    out_h: list[int] = []
    out_c: list[int] = []
    out_s: list[float] = []
    out_r: list[int] = []
    dropped_purge = 0

    for d, c, sc, rk in zip(dids, comms, scores, ranks):
        if d in purge_dids:
            dropped_purge += 1
            continue
        hv = did_to_h.get(d)
        if hv is None:
            hv = city_hash64(d)
            prev = h_to_did.get(hv)
            if prev is not None and prev != d:
                raise RuntimeError(f"city_hash64 collision between {prev} and {d}: {hv}")
            did_to_h[d] = hv
            h_to_did[hv] = d
        out_h.append(hv)
        out_c.append(int(c))
        out_s.append(float(sc))
        out_r.append(int(rk))

    out_tb = pa.table({
        "h": pa.array(out_h, type=pa.uint64()),
        "community": pa.array(out_c, type=pa.int32()),
        "score": pa.array(out_s, type=pa.float32()),
        "rank": pa.array(out_r, type=pa.uint8()),
    })
    assert "did" not in out_tb.column_names
    atomic_write_parquet(out_tb, out_pq)
    r0_count = sum(1 for r in out_r if r == 0)
    return {
        "source": str(src_pq),
        "source_sha256": sha256_file(src_pq),
        "rows_in": len(dids),
        "rows_dropped_purge": dropped_purge,
        "rows_out": len(out_h),
        "distinct_producers": len(did_to_h),
        "rank0_producers": r0_count,
        "n_communities": (max(out_c) + 1) if out_c else 0,
        "out_file": str(out_pq),
        "out_bytes": out_pq.stat().st_size,
        "out_sha256": sha256_file(out_pq),
    }


def equal_terciles(vals: np.ndarray, pids: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Deterministic equal-count terciles (0=bottom, 1=mid, 2=top), tie-broken by pid."""
    order = np.lexsort((pids, vals))
    n = len(vals)
    t = np.zeros(n, dtype=np.uint8)
    c1 = n // 3
    c2 = (2 * n) // 3
    t[order[c1:c2]] = 1
    t[order[c2:]] = 2
    cuts = [float(vals[order[0]]), float(vals[order[c1]]), float(vals[order[c2]]), float(vals[order[-1]])]
    return t, cuts


def prep_prod_t_blockers(o2_eval_dir: pathlib.Path, out_pq: pathlib.Path) -> dict:
    files = sorted(glob.glob(str(o2_eval_dir / "j*.parquet")))
    if len(files) != 16:
        raise RuntimeError(f"Expected 16 O2/eval parquets in {o2_eval_dir}, found {len(files)}")
    cols = ["p", "followers", "live_blockers_raw", "blockers_per_1k", "blockers_per_1k_burst_discounted"]
    tb = pa.concat_tables([pq.read_table(f, columns=cols) for f in files])
    p = tb["p"].to_numpy().astype(np.uint32)
    fol = tb["followers"].to_numpy().astype(np.uint32)
    blk = tb["live_blockers_raw"].to_numpy().astype(np.uint64)
    b1k_raw = tb["blockers_per_1k"].to_numpy(zero_copy_only=False)
    b1k_bd = tb["blockers_per_1k_burst_discounted"].to_numpy(zero_copy_only=False)

    keep = blk >= 50
    p = p[keep]
    fol = fol[keep]
    blk_u32 = np.minimum(blk[keep], (1 << 32) - 1).astype(np.uint32)
    b1k_raw = np.asarray(b1k_raw[keep], dtype=np.float32)
    b1k_bd = np.asarray(b1k_bd[keep], dtype=np.float32)

    order = np.argsort(p, kind="stable")
    p, fol, blk_u32, b1k_raw, b1k_bd = p[order], fol[order], blk_u32[order], b1k_raw[order], b1k_bd[order]
    assert len(np.unique(p)) == len(p)

    terc_bd, cuts_bd = equal_terciles(b1k_bd, p)
    terc_raw, cuts_raw = equal_terciles(b1k_raw, p)

    out_tb = pa.table({
        "p": pa.array(p, type=pa.uint32()),
        "followers_T": pa.array(fol, type=pa.uint32()),
        "live_blockers_T": pa.array(blk_u32, type=pa.uint32()),
        "blk_1k_raw_T": pa.array(b1k_raw, type=pa.float32()),
        "blk_1k_burst_disc_T": pa.array(b1k_bd, type=pa.float32()),
        "tercile_burst_disc": pa.array(terc_bd, type=pa.uint8()),
        "tercile_raw": pa.array(terc_raw, type=pa.uint8()),
    })
    atomic_write_parquet(out_tb, out_pq)
    return {
        "o2_eval_dir": str(o2_eval_dir),
        "producers_ge50_blockers_at_T": int(len(p)),
        "burst_discounted_tercile_cutpoints": cuts_bd,
        "raw_tercile_cutpoints": cuts_raw,
        "tercile_counts": [int((terc_bd == k).sum()) for k in (0, 1, 2)],
        "out_file": str(out_pq),
        "out_bytes": out_pq.stat().st_size,
        "out_sha256": sha256_file(out_pq),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--purge-dids", default="~/catbird-lake-local/s5-sweep-20261030/purge_dids.txt")
    ap.add_argument("--cand5-t-kf", default="~/catbird-lake-local/s3/matched/cand5-T-matched/known_for.parquet")
    ap.add_argument("--cand5-full-kf", default="~/catbird-lake-local/s3/matched/cand5-full-matched/known_for.parquet")
    ap.add_argument("--o2-eval-dir", default="~/catbird-lake-local/s10-outputs-20261030/O2/eval")
    ap.add_argument("--out-dir", default="~/catbird-lake-local/s14-hostility-20261030/inputs")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    purge_path = pathlib.Path(args.purge_dids).expanduser().resolve()
    purge_dids = {ln.strip() for ln in open(purge_path) if ln.startswith("did:")}

    rep_t = prep_kf(
        pathlib.Path(args.cand5_t_kf).expanduser().resolve(),
        out_dir / "kf_cand5_T.parquet",
        purge_dids,
    )
    rep_full = prep_kf(
        pathlib.Path(args.cand5_full_kf).expanduser().resolve(),
        out_dir / "kf_cand5_full.parquet",
        purge_dids,
    )
    rep_blk = prep_prod_t_blockers(
        pathlib.Path(args.o2_eval_dir).expanduser().resolve(),
        out_dir / "prod_t_blockers.parquet",
    )

    receipt = {
        "step": "S14a local DID-free input preparation",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purge_dids": {"path": str(purge_path), "count": len(purge_dids), "sha256": sha256_file(purge_path)},
        "kf_cand5_T": rep_t,
        "kf_cand5_full": rep_full,
        "prod_t_blockers": rep_blk,
    }
    (out_dir / "prep_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({
        "kf_T_producers": rep_t["distinct_producers"],
        "kf_full_producers": rep_full["distinct_producers"],
        "prod_t_blockers_ge50": rep_blk["producers_ge50_blockers_at_T"],
    }, indent=1))


if __name__ == "__main__":
    main()
