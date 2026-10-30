#!/usr/bin/env python3
"""Assemble status.tsv, purge_dids_live.txt, purge_dids.txt, and receipt.json for an S5(vii) sweep rerun.

Follows the exact S1 §11 & DECISIONS 2026-10-28 S5(vii) rules:
  1. Verifies the 1,000-DID PDS completeness check (`absent_check_pds.jsonl`) clears the <= 5% active bar.
  2. Builds `status.tsv` (`did\\tfalse\\tstatus`, sorted by DID, active=false rows only; requires
     `status_list_complete: false` in vmpass rules):
       - every row in `listing_pds.tsv.zst` with `active == 'false'`
       - every DID in `ourdata_absent_pds.tsv` (`status = 'absent'`)
       - every row in `thirdparty_relay_status.tsv` where HTTP 200 returned `active == 'false'`
         (overrides PDS `active == 'true'`, catching relay-throttled/takendown third-party repos)
  3. Builds `purge_dids_live.txt` = `status.tsv` DIDs + serving-universe DIDs absent from `listing_pds.tsv.zst`.
  4. Builds `purge_dids.txt` = cumulative union of `purge_dids_live.txt` and any `--prior-purge` file,
     so no account scheduled for purge under an earlier 7-day SLA is ever left unpurged.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def parse_absent_check(jsonl_path: pathlib.Path) -> dict:
    rows = [json.loads(line) for line in open(jsonl_path) if line.strip()]
    if not rows:
        raise SystemExit(f"empty absent check file: {jsonl_path}")
    active_pds = 0
    active_relay = 0
    for r in rows:
        ps = r.get("pds_status")
        if isinstance(ps, list) and len(ps) == 2 and ps[0] == 200 and isinstance(ps[1], dict) and ps[1].get("active") is True:
            active_pds += 1
        rs = r.get("relay")
        if isinstance(rs, list) and len(rs) == 2 and rs[0] == 200 and isinstance(rs[1], dict) and rs[1].get("active") is True:
            active_relay += 1
    rate = active_pds / len(rows)
    if rate > 0.05:
        raise SystemExit(f"PDS listing failed S1 completeness bar: {active_pds}/{len(rows)} ({rate:.1%}) > 5%")
    return {
        "sample": len(rows),
        "active_on_own_pds": active_pds,
        "relay_says_active": active_relay,
        "verdict": f"complete ({rate:.2%} <= 5%)",
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="Sweep directory (expects work/ inside)")
    ap.add_argument("--universe-tsv", action="append", default=[], help="Serving universe TSV(s) whose absent DIDs join purge_dids")
    ap.add_argument("--prior-purge", default="", help="Prior sweep purge_dids.txt for cumulative union")
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out_dir).expanduser().resolve()
    work = out / "work"

    comp = parse_absent_check(work / "absent_check_pds.jsonl")

    # Relay overrides (`200\tfalse\t<status>`).
    relay_overrides: dict[str, str] = {}
    relay_counts = collections.Counter()
    tp_path = work / "thirdparty_relay_status.tsv"
    if tp_path.exists():
        for line in open(tp_path):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            did, code, act, st = parts[:4]
            relay_counts[(code, act, st)] += 1
            if code == "200" and act == "false":
                relay_overrides[did] = st or "inactive"

    ov_tsv = work / "_relay_overrides.tsv"
    with open(ov_tsv, "w") as fh:
        for did, st in sorted(relay_overrides.items()):
            fh.write(f"{did}\t{st}\n")

    uni_dids: set[str] = set()
    for up in args.universe_tsv:
        p = pathlib.Path(up).expanduser()
        if p.exists():
            for line in open(p):
                if line.startswith("did:"):
                    uni_dids.add(line.split("\t", 1)[0].strip())
    uni_tsv = work / "_universe_dids.tsv"
    with open(uni_tsv, "w") as fh:
        for did in sorted(uni_dids):
            fh.write(f"{did}\n")

    # Use ClickHouse local to merge listing_pds (active=false), ourdata_absent_pds, and relay_overrides -> status.tsv,
    # and find universe DIDs absent from listing_pds.
    status_tmp = out / "status.tsv.tmp"
    status_out = out / "status.tsv"
    uni_absent_out = work / "_universe_absent.txt"

    sql_status = f"""
    SELECT did, 'false' AS active, argMax(status, prio) AS st
    FROM (
        SELECT did, if(status = 'none', '', status) AS status, 1 AS prio
        FROM file('{work}/listing_pds.tsv.zst', TSV, 'did String, active String, status String, nh UInt32, n_active UInt32, conflict UInt8, host String')
        WHERE active = 'false'
        UNION ALL
        SELECT did, 'absent' AS status, 2 AS prio
        FROM file('{work}/ourdata_absent_pds.tsv', TSV, 'did String, srcs Array(String)')
        UNION ALL
        SELECT did, status, 3 AS prio
        FROM file('{ov_tsv}', TSV, 'did String, status String')
    )
    GROUP BY did
    ORDER BY did
    INTO OUTFILE '{status_tmp}' TRUNCATE FORMAT TSV
    """
    subprocess.run(["clickhouse", "local", "--max_memory_usage", "14000000000", "-q", sql_status], check=True)
    os.replace(status_tmp, status_out)

    if uni_dids:
        sql_uni = f"""
        SELECT u.did
        FROM file('{uni_tsv}', TSV, 'did String') AS u
        LEFT ANTI JOIN (
            SELECT did FROM file('{work}/listing_pds.tsv.zst', TSV, 'did String, active String, status String, nh UInt32, n_active UInt32, conflict UInt8, host String')
        ) AS l ON l.did = u.did
        ORDER BY u.did
        INTO OUTFILE '{uni_absent_out}' TRUNCATE FORMAT TSV
        """
        subprocess.run(["clickhouse", "local", "--max_memory_usage", "14000000000", "-q", sql_uni], check=True)
        uni_absent = {l.strip() for l in open(uni_absent_out) if l.startswith("did:")}
    else:
        uni_absent = set()

    status_dids: set[str] = set()
    status_classes = collections.Counter()
    with open(status_out) as fh:
        for line in fh:
            d, _a, s = line.rstrip("\n").split("\t")
            status_dids.add(d)
            status_classes[s or "empty"] += 1

    live_purge = status_dids | uni_absent
    live_tmp = out / "purge_dids_live.txt.tmp"
    live_out = out / "purge_dids_live.txt"
    with open(live_tmp, "w") as fh:
        for d in sorted(live_purge):
            fh.write(d + "\n")
    os.replace(live_tmp, live_out)

    prior_dids: set[str] = set()
    if args.prior_purge:
        pp = pathlib.Path(args.prior_purge).expanduser()
        if pp.exists():
            prior_dids = {l.strip() for l in open(pp) if l.startswith("did:")}

    union_purge = live_purge | prior_dids
    purge_tmp = out / "purge_dids.txt.tmp"
    purge_out = out / "purge_dids.txt"
    with open(purge_tmp, "w") as fh:
        for d in sorted(union_purge):
            fh.write(d + "\n")
    os.replace(purge_tmp, purge_out)

    # Also emit cumulative status.tsv rows if prior_dids has DIDs that reactivated?
    # Note: vmpass uses status.tsv with status_list_complete=false, while S5p/Mac purges read purge_dids.txt directly.
    run_jsons = sorted((work / "pdslist").glob("run-*.json"))
    pds_summary = {}
    if run_jsons:
        rj = json.loads(run_jsons[-1].read_text())
        pds_summary = {
            "wall_s": round(rj.get("wall_s", 0), 1),
            "stats": rj.get("stats", {}),
            "hosts_total": len(rj.get("hosts", {})),
            "hosts_ok": sum(1 for v in rj.get("hosts", {}).values() if v.get("done")),
            "hosts_failed": sum(1 for v in rj.get("hosts", {}).values() if v.get("failed")),
            "repos_rows": sum(v.get("rows", 0) for v in rj.get("hosts", {}).values()),
        }

    receipt = {
        "step": "S5 (vii) per-PDS account-status and purge sweep rerun",
        "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md §3.1 S5, §4.9",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md §11",
        "wall_times_utc": {
            "pds_listrepos_start": (work / "pdslist.start").read_text().strip() if (work / "pdslist.start").exists() else None,
            "pds_listrepos_end": (work / "pdslist.end").read_text().strip() if (work / "pdslist.end").exists() else None,
            "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
        "pds_listrepos": pds_summary,
        "completeness_check_pds_listing": comp,
        "thirdparty_relay_override": {
            "checked": sum(relay_counts.values()),
            "relay_inactive_overrides": len(relay_overrides),
            "breakdown": {f"{c}/{a}/{s}": n for (c, a, s), n in sorted(relay_counts.items())},
        },
        "status_and_purge": {
            "status_tsv_rows": len(status_dids),
            "status_tsv_classes": dict(status_classes.most_common()),
            "status_tsv_format": "did<TAB>false<TAB>status, no header, active=false rows only; requires status_list_complete=false",
            "universe_absent_not_in_status_tsv": len(uni_absent - status_dids),
            "purge_dids_live": len(live_purge),
            "prior_purge_dids": len(prior_dids),
            "new_purge_dids_vs_prior": len(live_purge - prior_dids) if prior_dids else None,
            "prior_only_kept_for_monotonic_sla": len(prior_dids - live_purge) if prior_dids else 0,
            "purge_dids_cumulative_union": len(union_purge),
        },
        "outputs": {
            name: {
                "sha256": sha256_file(out / name),
                "bytes": (out / name).stat().st_size,
                "rows": sum(1 for _ in open(out / name)),
            }
            for name in ("status.tsv", "purge_dids_live.txt", "purge_dids.txt")
        },
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt["status_and_purge"], indent=1))


if __name__ == "__main__":
    main()
