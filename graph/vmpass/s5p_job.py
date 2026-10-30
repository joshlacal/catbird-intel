#!/usr/bin/env python3
"""R-S5p GCE Spot job (whole-network plan §4.9, pre-S10 purge join):

Runs on a short-lived self-deleting Spot VM in `us-west1` using stdlib Python + `clickhouse local`:
  1. Joins Fri Oct 30 `purge_dids.txt` against `gs://catbird-436600-plan-60d/didmap/s9-20261028a/didmap.parquet`.
  2. Emits integer-only `purge_uids.parquet` (`uid, is_prod, part, s9_purge, oct30_purge`) — zero DIDs.
  3. Emits integer+hash-only `eval_sample_uh.parquet` (`uid, h, is_prod, m1, o13m1, m2c`) for non-purged
     M1/M2 evaluation cohort accounts (`m1=1 OR o13m1=1 OR m2c=1`) — zero DIDs, asserts 0 hash collisions.
  4. Purges `O13/m1_engagement.parquet` and `O13/m2_engagement.parquet` (`did` or `subject` in cumulative
     purge set = `purge_dids.txt` U `didmap.purge=1`), verifies 0 purged DIDs remain, and writes them to
     `plan-60d/outputs/<run_id>/O13/` with `--custom-time` set for the 60-day lifecycle rule.
  5. Writes `receipt.json` with SHA-256 checksums and exact before/after row counts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import time


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run_cmd(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd[:4])}\nstderr: {r.stderr[-1200:]}")
    return r.stdout


def ch_query(sql: str, max_mem: int = 20_000_000_000) -> str:
    prelude = (
        "SET session_timezone = 'UTC';\n"
        "SET output_format_parquet_compression_method = 'zstd';\n"
        "SET engine_file_truncate_on_insert = 1;\n"
        f"SET max_memory_usage = {max_mem};\n"
        f"SET max_bytes_before_external_group_by = {max_mem // 3};\n"
        f"SET max_bytes_before_external_sort = {max_mem // 3};\n"
    )
    return run_cmd(["clickhouse", "local", "--multiquery", "-q", prelude + sql])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="/mnt/work")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--s9-run-id", default="s9-20261028a")
    ap.add_argument("--plan30", default="gs://catbird-436600-plan-30d")
    ap.add_argument("--plan60", default="gs://catbird-436600-plan-60d")
    args = ap.parse_args()

    t0 = time.time()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out_dir = work / "out"
    out_o13 = out_dir / "O13"
    out_o13.mkdir(parents=True, exist_ok=True)

    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    max_mem = max(4_000_000_000, int(ram * 0.72))

    # 1. Stage inputs from us-west1 GCS ($0 intra-region)
    purge_txt = work / "purge_dids.txt"
    didmap_pq = work / "didmap.parquet"
    m1_in = work / "m1_engagement_in.parquet"
    m2_in = work / "m2_engagement_in.parquet"

    if not purge_txt.exists():
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan30}/inputs/{args.run_id}/purge_dids.txt", str(purge_txt)])
    run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan60}/didmap/{args.s9_run_id}/didmap.parquet", str(didmap_pq)])
    run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan60}/outputs/{args.s9_run_id}/O13/m1_engagement.parquet", str(m1_in)])
    run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan60}/outputs/{args.s9_run_id}/O13/m2_engagement.parquet", str(m2_in)])

    t_staged = time.time()

    purge_uids_pq = out_dir / "purge_uids.parquet"
    eval_uh_pq = out_dir / "eval_sample_uh.parquet"
    m1_out = out_o13 / "m1_engagement.parquet"
    m2_out = out_o13 / "m2_engagement.parquet"
    all_purge_tsv = work / "all_purge_dids.tsv"

    # 2. Build purge_uids.parquet, eval_sample_uh.parquet, and combined purge DID set
    sql_dm = f"""
    CREATE TABLE oct30 ENGINE = MergeTree ORDER BY did AS
    SELECT DISTINCT trimBoth(did) AS did
    FROM file('{purge_txt}', TSV, 'did String')
    WHERE startsWith(did, 'did:');

    INSERT INTO FUNCTION file('{purge_uids_pq}', Parquet)
    SELECT
        d.uid AS uid,
        d.is_prod AS is_prod,
        d.part AS part,
        d.purge AS s9_purge,
        toUInt8(o.did != '') AS oct30_purge
    FROM file('{didmap_pq}', Parquet) AS d
    LEFT JOIN oct30 AS o ON o.did = d.did
    WHERE d.purge = 1 OR o.did != ''
    ORDER BY uid;

    INSERT INTO FUNCTION file('{eval_uh_pq}', Parquet)
    SELECT
        d.uid AS uid,
        d.h AS h,
        d.is_prod AS is_prod,
        d.m1 AS m1,
        d.o13m1 AS o13m1,
        d.m2c AS m2c
    FROM file('{didmap_pq}', Parquet) AS d
    LEFT ANTI JOIN oct30 AS o ON o.did = d.did
    WHERE d.purge = 0 AND (d.m1 = 1 OR d.o13m1 = 1 OR d.m2c = 1)
    ORDER BY uid;

    INSERT INTO FUNCTION file('{all_purge_tsv}', TSV)
    SELECT DISTINCT did FROM (
        SELECT did FROM oct30
        UNION ALL
        SELECT did FROM file('{didmap_pq}', Parquet) WHERE purge = 1
    );
    """
    ch_query(sql_dm, max_mem=max_mem)

    # Verify eval_sample_uh has 0 hash collisions and no DID column
    uh_stats_raw = ch_query(
        f"SELECT count(), uniqExact(h), countIf(m1=1), countIf(o13m1=1), countIf(m2c=1) "
        f"FROM file('{eval_uh_pq}', Parquet) FORMAT TSV",
        max_mem=max_mem,
    ).strip().split("\t")
    uh_rows, uh_uniq_h, uh_m1, uh_o13m1, uh_m2c = [int(x) for x in uh_stats_raw]
    if uh_rows != uh_uniq_h:
        raise RuntimeError(f"cityHash64 collision in eval_sample_uh: rows={uh_rows} != uniq_h={uh_uniq_h}")

    pu_stats_raw = ch_query(
        f"SELECT count(), countIf(is_prod=1), countIf(s9_purge=1), countIf(oct30_purge=1), countIf(s9_purge=0 AND oct30_purge=1) "
        f"FROM file('{purge_uids_pq}', Parquet) FORMAT TSV",
        max_mem=max_mem,
    ).strip().split("\t")
    pu_total, pu_prod, pu_s9, pu_oct30, pu_delta = [int(x) for x in pu_stats_raw]

    # 3. Filter O13 m1_engagement and m2_engagement against all_purge_dids.tsv
    o13_rep = {}
    for label, src_pq, dst_pq in (("m1_engagement", m1_in, m1_out), ("m2_engagement", m2_in, m2_out)):
        n_before = int(ch_query(f"SELECT count() FROM file('{src_pq}', Parquet) FORMAT TSV", max_mem=max_mem).strip())
        sql_o13 = f"""
        CREATE TABLE pset ENGINE = Join(ANY, LEFT, did) AS
        SELECT did, toUInt8(1) AS bad FROM file('{all_purge_tsv}', TSV, 'did String');

        INSERT INTO FUNCTION file('{dst_pq}', Parquet)
        SELECT did, subject, created_us, kind
        FROM file('{src_pq}', Parquet)
        WHERE joinGet('pset', 'bad', did) = 0 AND joinGet('pset', 'bad', subject) = 0
        ORDER BY did, subject, created_us, kind;
        """
        ch_query(sql_o13, max_mem=max_mem)
        chk_raw = ch_query(
            f"""
            CREATE TABLE pset ENGINE = Join(ANY, LEFT, did) AS
            SELECT did, toUInt8(1) AS bad FROM file('{all_purge_tsv}', TSV, 'did String');
            SELECT count(), countIf(joinGet('pset', 'bad', did) = 1 OR joinGet('pset', 'bad', subject) = 1)
            FROM file('{dst_pq}', Parquet) FORMAT TSV
            """,
            max_mem=max_mem,
        ).strip().split("\t")
        n_after, n_bad_rem = int(chk_raw[0]), int(chk_raw[1])
        if n_bad_rem != 0:
            raise RuntimeError(f"Purged DIDs remain in {label}: {n_bad_rem}")
        o13_rep[label] = {
            "rows_before": n_before,
            "rows_removed": n_before - n_after,
            "rows_after": n_after,
            "verified_remaining_purged": n_bad_rem,
            "pre_sha256": sha256_file(src_pq),
            "post_sha256": sha256_file(dst_pq),
            "post_bytes": dst_pq.stat().st_size,
        }

    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt = {
        "step": "R-S5p Spot purge-ID join and O13 purge (§4.9)",
        "run_id": args.run_id,
        "s9_run_id": args.s9_run_id,
        "written_utc": now_iso,
        "stage_seconds": round(t_staged - t0, 2),
        "wall_seconds": round(time.time() - t0, 2),
        "inputs": {
            "purge_dids_txt": {"sha256": sha256_file(purge_txt), "bytes": purge_txt.stat().st_size},
            "s9_didmap_parquet": {"sha256": sha256_file(didmap_pq), "bytes": didmap_pq.stat().st_size},
        },
        "purge_uids": {
            "total_uids": pu_total,
            "producer_uids": pu_prod,
            "s9_purge_uids": pu_s9,
            "oct30_matched_uids": pu_oct30,
            "newly_purged_since_s9": pu_delta,
            "sha256": sha256_file(purge_uids_pq),
            "bytes": purge_uids_pq.stat().st_size,
        },
        "eval_sample_uh": {
            "rows": uh_rows,
            "unique_h": uh_uniq_h,
            "m1_rows": uh_m1,
            "o13m1_rows": uh_o13m1,
            "m2c_rows": uh_m2c,
            "sha256": sha256_file(eval_uh_pq),
            "bytes": eval_uh_pq.stat().st_size,
        },
        "o13_purged": o13_rep,
    }
    rec_path = out_dir / "receipt.json"
    rec_path.write_text(json.dumps(receipt, indent=1) + "\n")

    # 4. Upload to plan-30d and plan-60d (with custom-time on plan-60d)
    run_cmd([
        "gcloud", "storage", "cp", "--quiet",
        str(purge_uids_pq), str(eval_uh_pq), str(rec_path),
        f"{args.plan30}/outputs/{args.run_id}/",
    ])
    run_cmd([
        "gcloud", "storage", "cp", "--quiet", f"--custom-time={now_iso}",
        str(m1_out), str(m2_out),
        f"{args.plan60}/outputs/{args.run_id}/O13/",
    ])
    print("S5P_DONE", json.dumps(receipt["purge_uids"]), flush=True)


if __name__ == "__main__":
    main()
