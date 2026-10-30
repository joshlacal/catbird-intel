#!/usr/bin/env python3
"""S10 local pull + SHA-256 verification + §4.9 integer/DID purge + D12 positive-control hash retirement.

Pulls from `us-west1` GCS (`~3.6 GiB` egress vs `$1.84` cap):
  - `s5p` outputs (`purge_uids.parquet`, `eval_sample_uh.parquet`, `receipt.json`, purged `O13/{m1,m2}_engagement.parquet`)
  - `s9` metadata (`o3b_scales.json`, `rules.json`)
  - `s9` outputs (`O1`, `O2`, `O3a`, `O3b`, `O3c`, `O4`, `O5-prod`, `O6`, `O7`, `O9`, `O12/random_nonfollower_quotes.parquet`)

Verifies every pulled file's pre-purge SHA-256 against `s9-receipt/receipt.json` and `s5p/receipt.json`,
applies `purge_uids.parquet` + `purge_dids.txt` atomically (`*.tmp` + `os.replace`), verifies 0 purged rows
remain across every file, moves `s9-receipt/i00..i15.parquet` to `_trash/s9-positive-control-hashes/` (D12),
and writes `s10_receipt.json`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def gs_cp(sources: list[str], dest: pathlib.Path, *, project: str, recursive: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gcloud", "storage", "cp", "--quiet", f"--project={project}"]
    if recursive:
        cmd.append("--recursive")
    cmd += [*sources, str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcloud storage cp failed ({r.returncode}): {r.stderr[-1200:]}")


def uri_did_array(col: pa.ChunkedArray) -> pa.ChunkedArray:
    parts = pc.split_pattern(col, "/")
    return pc.list_element(parts, 2)


def purge_table_file(
    path: pathlib.Path,
    purge_uids_u32: pa.Array,
    purge_dids_str: pa.Array,
) -> dict:
    pre_sha = sha256_file(path)
    pre_bytes = path.stat().st_size
    tb = pq.read_table(path)
    n0 = tb.num_rows
    cols = set(tb.column_names)

    bad_exprs = []
    for c in ("u", "p", "q"):
        if c in cols:
            bad_exprs.append(pc.is_in(pc.cast(tb[c], pa.uint32()), value_set=purge_uids_u32))
    for c in ("did", "subject"):
        if c in cols:
            bad_exprs.append(pc.is_in(tb[c], value_set=purge_dids_str))
    for c in ("uri", "feed_uri"):
        if c in cols:
            bad_exprs.append(pc.is_in(uri_did_array(tb[c]), value_set=purge_dids_str))

    if not bad_exprs or n0 == 0:
        return {
            "rows_before": n0,
            "rows_removed": 0,
            "rows_after": n0,
            "pre_sha256": pre_sha,
            "post_sha256": pre_sha,
            "post_bytes": pre_bytes,
        }

    bad = bad_exprs[0]
    for e in bad_exprs[1:]:
        bad = pc.or_(bad, e)
    n_bad = int(pc.sum(pc.cast(bad, pa.int64())).as_py() or 0)
    if n_bad > 0:
        kept = tb.filter(pc.invert(bad))
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(kept, tmp, compression="zstd")
        os.replace(tmp, path)

    # Re-verify 0 purged rows remain
    tb_chk = pq.read_table(path)
    rem = 0
    for c in ("u", "p", "q"):
        if c in cols and tb_chk.num_rows > 0:
            rem += int(pc.sum(pc.cast(pc.is_in(pc.cast(tb_chk[c], pa.uint32()), value_set=purge_uids_u32), pa.int64())).as_py() or 0)
    for c in ("did", "subject"):
        if c in cols and tb_chk.num_rows > 0:
            rem += int(pc.sum(pc.cast(pc.is_in(tb_chk[c], value_set=purge_dids_str), pa.int64())).as_py() or 0)
    for c in ("uri", "feed_uri"):
        if c in cols and tb_chk.num_rows > 0:
            rem += int(pc.sum(pc.cast(pc.is_in(uri_did_array(tb_chk[c]), value_set=purge_dids_str), pa.int64())).as_py() or 0)
    if rem != 0:
        raise RuntimeError(f"Post-purge verification failed on {path}: {rem} purged rows remain!")

    return {
        "rows_before": n0,
        "rows_removed": n_bad,
        "rows_after": n0 - n_bad,
        "pre_sha256": pre_sha,
        "post_sha256": sha256_file(path),
        "post_bytes": path.stat().st_size,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default="catbird-436600")
    ap.add_argument("--plan30", default="gs://catbird-436600-plan-30d")
    ap.add_argument("--plan60", default="gs://catbird-436600-plan-60d")
    ap.add_argument("--s9-run-id", default="s9-20261028a")
    ap.add_argument("--s5p-run-id", default="s5p-20261030a")
    ap.add_argument("--s9-receipt", default="~/catbird-lake-local/s9-receipt/receipt.json")
    ap.add_argument("--purge-dids", default="~/catbird-lake-local/s5-sweep-20261030/purge_dids.txt")
    ap.add_argument("--out-dir", default="~/catbird-lake-local/s10-outputs-20261030")
    ap.add_argument("--trash-pc-dir", default="~/catbird-lake-local/_trash/s9-positive-control-hashes")
    args = ap.parse_args(argv)

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    s9_rec_path = pathlib.Path(args.s9_receipt).expanduser().resolve()
    purge_txt = pathlib.Path(args.purge_dids).expanduser().resolve()
    trash_pc = pathlib.Path(args.trash_pc_dir).expanduser().resolve()

    s9_rec = json.loads(s9_rec_path.read_text())
    expected_s9: dict[str, dict] = {}
    for entry in s9_rec["outputs"]:
        p = entry["path"]
        if p.startswith("out30/"):
            rel = p[len("out30/"):]
        elif p.startswith("out60/"):
            rel = p[len("out60/"):]
        else:
            continue
        expected_s9[rel] = entry

    # 1. Pull S5p outputs + purged O13 + S9 inputs metadata
    s5p_local = out_dir / "s5p"
    s5p_local.mkdir(parents=True, exist_ok=True)
    if not (s5p_local / "receipt.json").exists():
        gs_cp([f"{args.plan30}/outputs/{args.s5p_run_id}/*"], s5p_local, project=args.project)

    o13_local = out_dir / "O13"
    o13_local.mkdir(parents=True, exist_ok=True)
    if not (o13_local / "m1_engagement.parquet").exists() or not (o13_local / "m2_engagement.parquet").exists():
        gs_cp([f"{args.plan60}/outputs/{args.s5p_run_id}/O13/*.parquet"], o13_local, project=args.project)

    meta_local = out_dir / "meta"
    meta_local.mkdir(parents=True, exist_ok=True)
    for mfn in ("o3b_scales.json", "rules.json"):
        if not (meta_local / mfn).exists():
            gs_cp([f"{args.plan30}/inputs/{args.s9_run_id}/{mfn}"], meta_local / mfn, project=args.project)

    # 2. Pull S9 deliverables (O1..O7, O9, O12/random_nonfollower_quotes.parquet)
    pull_specs = [
        (f"{args.plan30}/outputs/{args.s9_run_id}/O1/*", out_dir / "O1", False),
        (f"{args.plan30}/outputs/{args.s9_run_id}/O2", out_dir, True),
        (f"{args.plan30}/outputs/{args.s9_run_id}/O3a", out_dir, True),
        (f"{args.plan30}/outputs/{args.s9_run_id}/O3b", out_dir, True),
        (f"{args.plan30}/outputs/{args.s9_run_id}/O3c", out_dir, True),
        (f"{args.plan30}/outputs/{args.s9_run_id}/O4", out_dir, True),
        (f"{args.plan30}/outputs/{args.s9_run_id}/O5-prod", out_dir, True),
        (f"{args.plan30}/outputs/{args.s9_run_id}/O6", out_dir, True),
        (f"{args.plan30}/outputs/{args.s9_run_id}/O7", out_dir, True),
        (f"{args.plan60}/outputs/{args.s9_run_id}/O9", out_dir, True),
        (
            f"{args.plan30}/outputs/{args.s9_run_id}/O12/random_nonfollower_quotes.parquet",
            out_dir / "O12/random_nonfollower_quotes.parquet",
            False,
        ),
    ]
    marker_pulled = out_dir / "_raw_pull_verified.json"
    if not marker_pulled.exists():
        (out_dir / "O1").mkdir(parents=True, exist_ok=True)
        (out_dir / "O12").mkdir(parents=True, exist_ok=True)
        for src_uri, dst_path, rec_flag in pull_specs:
            gs_cp([src_uri], dst_path, project=args.project, recursive=rec_flag)

        # Verify pre-purge SHA-256 against s9-receipt/receipt.json & s5p/receipt.json
        s5p_rec = json.loads((s5p_local / "receipt.json").read_text())
        assert sha256_file(s5p_local / "purge_uids.parquet") == s5p_rec["purge_uids"]["sha256"], "purge_uids sha mismatch"
        assert sha256_file(s5p_local / "eval_sample_uh.parquet") == s5p_rec["eval_sample_uh"]["sha256"], "eval_sample_uh sha mismatch"
        for k in ("m1_engagement", "m2_engagement"):
            got = sha256_file(o13_local / f"{k}.parquet")
            want = s5p_rec["o13_purged"][k]["post_sha256"]
            assert got == want, f"O13 {k} sha mismatch: {got} != {want}"

        verified_s9_files = 0
        s10_prefixes = ("O2/", "O3a/", "O3b/", "O3c/", "O4/", "O5-prod/", "O6/", "O7/", "O9/", "O12/random_nonfollower_quotes.parquet")
        for rel, exp in sorted(expected_s9.items()):
            if not rel.startswith(s10_prefixes):
                continue
            lf = out_dir / rel
            if not lf.exists():
                raise RuntimeError(f"Missing expected S10 file {lf}")
            got = sha256_file(lf)
            if got != exp["sha256"]:
                raise RuntimeError(f"Pre-purge SHA-256 mismatch on {rel}: got {got}, expected {exp['sha256']}")
            verified_s9_files += 1

        marker_pulled.write_text(json.dumps({
            "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "verified_s9_files": verified_s9_files,
        }, indent=1) + "\n")

    # 3. Apply integer (`purge_uids.parquet`) + DID (`purge_dids.txt`) purges locally across all pulled parquets
    pu_tb = pq.read_table(s5p_local / "purge_uids.parquet", columns=["uid"])
    purge_uids_u32 = pc.cast(pu_tb["uid"], pa.uint32()).combine_chunks()
    purge_dids = sorted({ln.strip() for ln in open(purge_txt) if ln.startswith("did:")})
    purge_dids_str = pa.array(purge_dids, type=pa.string())

    parquet_files = sorted(
        p for p in out_dir.rglob("*.parquet")
        if "s5p" not in p.parts and not p.name.startswith("._")
    )
    group_stats: dict[str, dict] = {}
    file_details: dict[str, dict] = {}
    total_removed = 0
    files_rewritten = 0
    total_bytes = 0

    for pf in parquet_files:
        rel = str(pf.relative_to(out_dir))
        grp = rel.split("/", 1)[0]
        res = purge_table_file(pf, purge_uids_u32, purge_dids_str)
        file_details[rel] = res
        total_bytes += res["post_bytes"]
        if res["rows_removed"] > 0:
            files_rewritten += 1
            total_removed += res["rows_removed"]
        gs = group_stats.setdefault(grp, {"files": 0, "rows_before": 0, "rows_removed": 0, "rows_after": 0, "bytes": 0})
        gs["files"] += 1
        gs["rows_before"] += res["rows_before"]
        gs["rows_removed"] += res["rows_removed"]
        gs["rows_after"] += res["rows_after"]
        gs["bytes"] += res["post_bytes"]

    # 4. Move S9 positive-control hashes (`s9-receipt/i00..i15.parquet`) to `_trash` per D12 (never rm)
    pc_moved = []
    s9_rec_dir = s9_rec_path.parent
    for pcf in sorted(s9_rec_dir.glob("i*.parquet")):
        trash_pc.mkdir(parents=True, exist_ok=True)
        dst = trash_pc / pcf.name
        shutil.move(str(pcf), str(dst))
        pc_moved.append(pcf.name)

    # 5. Also place purged O13 engagement parquets into ~/catbird-lake-local/s3/{m1,m2}/ if present
    s3_root = pathlib.Path("~/catbird-lake-local/s3").expanduser()
    s3_linked = {}
    if s3_root.exists():
        for sub, fn in (("m1", "m1_engagement.parquet"), ("m2", "m2_engagement.parquet")):
            d = s3_root / sub
            if d.exists():
                target = d / fn
                if target.exists():
                    # Move any old copy to _trash before replacing
                    tr = pathlib.Path("~/catbird-lake-local/_trash/s3-pre-s10").expanduser()
                    tr.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(target), str(tr / f"{sub}_{fn}"))
                shutil.copy2(o13_local / fn, target)
                s3_linked[f"{sub}/{fn}"] = str(target)

    receipt = {
        "step": "S10 pull outputs to Mac, SHA-256 audit, §4.9 local purge, and D12 positive-control hash retirement",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t0, 1),
        "s9_run_id": args.s9_run_id,
        "s5p_run_id": args.s5p_run_id,
        "purge_uids_count": len(pu_tb),
        "purge_dids_count": len(purge_dids),
        "parquet_files_audited": len(parquet_files),
        "parquet_files_rewritten": files_rewritten,
        "total_rows_removed_locally": total_removed,
        "total_pulled_parquet_bytes": total_bytes,
        "total_pulled_parquet_gib": round(total_bytes / (1 << 30), 3),
        "d12_positive_control_hashes_moved_to_trash": {
            "trash_dir": str(trash_pc),
            "files_moved": pc_moved,
            "remaining_in_s9_receipt": len(list(s9_rec_dir.glob("i*.parquet"))),
        },
        "s3_engagement_updated": s3_linked,
        "groups": group_stats,
        "files": file_details,
    }
    rec_out = out_dir / "s10_receipt.json"
    rec_out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({
        "parquet_files_audited": len(parquet_files),
        "parquet_files_rewritten": files_rewritten,
        "total_rows_removed_locally": total_removed,
        "total_gib": receipt["total_pulled_parquet_gib"],
        "pc_hashes_moved": len(pc_moved),
    }, indent=1))


if __name__ == "__main__":
    main()
