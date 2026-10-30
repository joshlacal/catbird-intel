#!/usr/bin/env python3
"""Purge S12m vector units and local input units/O8 shards under §4.9.

Scans:
  1. S12m output vectors: `~/catbird-lake-local/s12mm-colab-s12m-20261029a/vectors/units/u-*.parquet`
     (and updates corresponding `ledger/u-*.json` sha256 + row count while recording pre-purge hashes)
  2. S12m local input units: `~/catbird-lake-local/s12m-units-s9/units/*.parquet`, `manifest.json`, `record_dids.txt`
  3. S12m local O8 shards (if present): `~/catbird-lake-local/s12m-inputs/o8/*.parquet`

Drops every row whose producer integer `p` is in `purge_pids` OR whose URI author DID is in `purge_dids`.
Runs `mmlib.audit_output_table` on every rewritten vector unit and verifies 0 purged rows remain across all files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mmlib as L  # noqa: E402


def load_purge_sets(purge_txt: pathlib.Path, producers_tsv: pathlib.Path) -> tuple[set[str], set[int]]:
    purge_dids = {line.strip() for line in open(purge_txt) if line.startswith("did:")}
    purge_pids: set[int] = set()
    with open(producers_tsv) as fh:
        for line in fh:
            pid_s, did, _ = line.rstrip("\n").split("\t", 2)
            if did in purge_dids:
                purge_pids.add(int(pid_s))
    return purge_dids, purge_pids


def row_bad_mask(t: pa.Table, purge_pids_arr: pa.Array, purge_dids_arr: pa.Array) -> pa.ChunkedArray:
    bad = pc.is_in(t["p"], value_set=purge_pids_arr)
    parts = pc.split_pattern(t["uri"], "/")
    dids = pc.list_element(parts, 2)
    return pc.or_(bad, pc.is_in(dids, value_set=purge_dids_arr))


def purge_parquet_dir(files: list[pathlib.Path], purge_pids_arr: pa.Array, purge_dids_arr: pa.Array, *, audit_vectors: bool = False) -> dict:
    files_scanned = 0
    files_rewritten = 0
    rows_before = 0
    rows_removed = 0
    rows_after = 0
    rewritten_details = []

    for f in files:
        files_scanned += 1
        meta = pq.read_table(f, columns=["p", "uri"])
        n0 = meta.num_rows
        rows_before += n0
        bad = row_bad_mask(meta, purge_pids_arr, purge_dids_arr)
        n_bad = int(pc.sum(pc.cast(bad, pa.int64())).as_py() or 0)
        if n_bad == 0:
            rows_after += n0
            continue

        pre_sha = L.sha256_file(f)
        full = pq.read_table(f)
        kept = full.filter(pc.invert(row_bad_mask(full, purge_pids_arr, purge_dids_arr)))
        tmp = f.with_name(f.name + ".tmp")
        pq.write_table(kept, tmp, compression="zstd")
        if audit_vectors:
            aud = L.audit_output_table(tmp)
            if not aud["ok"]:
                raise SystemExit(f"audit_output_table failed on rewritten {f}: {aud}")
        os.replace(tmp, f)
        post_sha = L.sha256_file(f)
        files_rewritten += 1
        rows_removed += n0 - kept.num_rows
        rows_after += kept.num_rows
        rewritten_details.append({
            "file": f.name,
            "rows_before": n0,
            "rows_removed": n0 - kept.num_rows,
            "rows_after": kept.num_rows,
            "pre_sha256": pre_sha,
            "post_sha256": post_sha,
            "post_bytes": f.stat().st_size,
        })

    # Post-verification pass over every file
    rem_total = 0
    for f in files:
        meta = pq.read_table(f, columns=["p", "uri"])
        bad = row_bad_mask(meta, purge_pids_arr, purge_dids_arr)
        rem_total += int(pc.sum(pc.cast(bad, pa.int64())).as_py() or 0)
    if rem_total != 0:
        raise SystemExit(f"Verification failed: {rem_total} purged rows remain!")

    return {
        "files_scanned": files_scanned,
        "files_rewritten": files_rewritten,
        "rows_before": rows_before,
        "rows_removed": rows_removed,
        "rows_after": rows_after,
        "verified_remaining_purged_rows": rem_total,
        "rewritten_files": rewritten_details,
    }


def update_vector_ledger_and_units_manifest(
    run_dir: pathlib.Path,
    vec_rep: dict,
    units_dir: pathlib.Path | None,
    units_rep: dict | None,
    purge_dids: set[str],
    purge_sha: str,
) -> None:
    ledger_dir = run_dir / "ledger"
    by_name = {d["file"]: d for d in vec_rep["rewritten_files"]}
    for fname, d in by_name.items():
        # fname = u-bXXXX-NNNN-00000.parquet -> unit = bXXXX-NNNN
        unit = fname[len("u-") : -len("-00000.parquet")]
        lp = ledger_dir / f"{unit}.json"
        if lp.exists():
            rec = json.loads(lp.read_text())
            rec.setdefault("purge_history", []).append({
                "purged_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "purge_dids_sha256": purge_sha,
                "pre_sha256": d["pre_sha256"],
                "pre_rows": d["rows_before"],
                "rows_removed": d["rows_removed"],
            })
            rec["sha256"] = d["post_sha256"]
            rec["rows"] = d["rows_after"]
            rec["bytes"] = d["post_bytes"]
            tmp = lp.with_name(lp.name + ".tmp")
            tmp.write_text(json.dumps(rec, indent=1) + "\n")
            os.replace(tmp, lp)

    if units_dir is not None and units_rep is not None and (units_dir / "manifest.json").exists():
        man_path = units_dir / "manifest.json"
        man = json.loads(man_path.read_text())
        u_by_name = {d["file"]: d for d in units_rep["rewritten_files"]}
        if u_by_name:
            for u in man.get("units", []):
                fn = pathlib.Path(u["file"]).name
                if fn in u_by_name:
                    uf = units_dir / u["file"]
                    ut = pq.read_table(uf, columns=["image_cids"])
                    lens = pc.fill_null(pc.list_value_length(ut["image_cids"]), 0)
                    u["rows"] = u_by_name[fn]["rows_after"]
                    u["bytes"] = u_by_name[fn]["post_bytes"]
                    u["sha256"] = u_by_name[fn]["post_sha256"]
                    u["image_posts"] = int(pc.sum(pc.cast(pc.greater(lens, 0), pa.int64())).as_py() or 0)
                    u["images"] = int(pc.sum(lens).as_py() or 0)
            man["totals"] = {
                "units": len(man["units"]),
                "posts": sum(u["rows"] for u in man["units"]),
                "image_posts": sum(u["image_posts"] for u in man["units"]),
                "images": sum(u["images"] for u in man["units"]),
                "bytes": sum(u["bytes"] for u in man["units"]),
            }
            man["last_purged_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            man["last_purge_dids_sha256"] = purge_sha
            tmp = man_path.with_name("manifest.json.tmp")
            tmp.write_text(json.dumps(man, indent=1) + "\n")
            os.replace(tmp, man_path)

        rd = units_dir / "record_dids.txt"
        if rd.exists():
            lines = [ln for ln in open(rd) if ln.strip() not in purge_dids]
            tmp = rd.with_name("record_dids.txt.tmp")
            tmp.write_text("".join(lines))
            os.replace(tmp, rd)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="~/catbird-lake-local/s12mm-colab-s12m-20261029a")
    ap.add_argument("--units-dir", default="~/catbird-lake-local/s12m-units-s9")
    ap.add_argument("--o8-dir", default="~/catbird-lake-local/s12m-inputs/o8")
    ap.add_argument("--purge-dids", required=True)
    ap.add_argument("--producers-tsv", default="~/catbird-lake-local/graph-build-v3/out-full-20261024/producers.tsv")
    ap.add_argument("--receipt-out", default="")
    args = ap.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    units_dir = pathlib.Path(args.units_dir).expanduser().resolve() if args.units_dir else None
    o8_dir = pathlib.Path(args.o8_dir).expanduser().resolve() if args.o8_dir else None
    purge_txt = pathlib.Path(args.purge_dids).expanduser().resolve()
    producers_tsv = pathlib.Path(args.producers_tsv).expanduser().resolve()
    receipt_out = pathlib.Path(args.receipt_out).expanduser().resolve() if args.receipt_out else (run_dir / "purge_receipt_20261030.json")

    purge_sha = L.sha256_file(purge_txt)
    purge_dids, purge_pids = load_purge_sets(purge_txt, producers_tsv)
    purge_pids_arr = pa.array(sorted(purge_pids), type=pa.int64())
    purge_dids_arr = pa.array(sorted(purge_dids), type=pa.string())

    vec_files = sorted((run_dir / "vectors/units").glob("u-*.parquet"))
    if not vec_files:
        raise SystemExit(f"no vector units under {run_dir / 'vectors/units'}")
    vec_rep = purge_parquet_dir(vec_files, purge_pids_arr, purge_dids_arr, audit_vectors=True)

    units_rep = None
    if units_dir and (units_dir / "units").exists():
        u_files = sorted((units_dir / "units").glob("*.parquet"))
        if u_files:
            units_rep = purge_parquet_dir(u_files, purge_pids_arr, purge_dids_arr, audit_vectors=False)

    o8_rep = None
    if o8_dir and o8_dir.exists():
        o8_files = sorted(o8_dir.glob("*.parquet"))
        if o8_files:
            o8_rep = purge_parquet_dir(o8_files, purge_pids_arr, purge_dids_arr, audit_vectors=False)

    update_vector_ledger_and_units_manifest(run_dir, vec_rep, units_dir, units_rep, purge_dids, purge_sha)

    receipt = {
        "step": "S12m vector and local slab unit purge (§4.9)",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purge_dids_file": str(purge_txt),
        "purge_dids_sha256": purge_sha,
        "purge_dids_count": len(purge_dids),
        "purge_producer_pids_count": len(purge_pids),
        "vectors_units": vec_rep,
        "input_units": units_rep,
        "local_o8": o8_rep,
    }
    receipt_out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({
        "vectors_files_rewritten": vec_rep["files_rewritten"],
        "vectors_rows_removed": vec_rep["rows_removed"],
        "vectors_rows_after": vec_rep["rows_after"],
        "input_units_rows_removed": units_rep["rows_removed"] if units_rep else None,
        "local_o8_rows_removed": o8_rep["rows_removed"] if o8_rep else None,
    }, indent=1))


if __name__ == "__main__":
    main()
