#!/usr/bin/env python3
"""Build receipt.json for whole-network plan step S2 from chq.sh logs, input and output hashes.

    s2_receipt.py S2_OUT RECEIPT_JSON [--hash-fold]

For every logged run in S2_OUT/logs (NAME.cmd present) the receipt holds the exact command, the
sha256 of the statements file it ran, its raw stdout, its stderr tail, /usr/bin/time's wall time
and peak RSS, and the disk samples. Inputs are hashed with sha256 (the fold DB's table files only
with --hash-fold, which reads ~110 GB); every file under S2_OUT/out is hashed. Gate results are
computed from the raw outputs, never typed in.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys

HOME = pathlib.Path.home()
LAKE = HOME / "catbird-lake-local"
NETCUP_DAYS = ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]
NETCUP_FILES = ["engagements.parquet", "posts.parquet", "blocks.parquet", "manifest.json"]
OTHER_INPUTS = [
    LAKE / "graph-build-v3/out-full-20261024/edges.bin",
    LAKE / "graph-build-v3/out-full-20261024/producers.tsv",
    LAKE / "graph-build-v3/out-full-20261024/follower_weights.tsv",
    LAKE / "graph-build-v3/out-full-20261024/stats.tsv",
    LAKE / "graph-build-v3/fold.sql",
    LAKE / "communities-cand1-hubdemote/producers.tsv",
    LAKE / "communities-cand1-hubdemote/membership.npy",
]
FOLD_TABLES = {
    "follow_events": "5d9/5d9614eb-f5d2-4c4c-bcd7-39ef07fc219a",
    "follows": "205/205eb761-5236-4660-947d-c39b5ed13f98",
    "producers": "ce7/ce7ca214-7f65-455f-abe7-b0b2ebbc2b3d",
    "floors": "44c/44c52c46-3bf5-4804-bfe8-af7b4e6ff94f",
    "inactive": "99d/99dcd9ce-49fa-4059-96cc-2734161305d7",
    "followers_ok": "204/204b3613-6b20-4db0-91ca-82388c6aef48",
    "snapshots": "72f/72fd9fa8-74c5-4024-843a-84b9d505c9ac",
    "coverage": "b7d/b7d6a4cf-1d72-4341-b837-b9e599390372",
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_many(paths: list[pathlib.Path]) -> dict[str, dict]:
    paths = [p for p in paths if p.is_file()]
    with cf.ThreadPoolExecutor(8) as ex:
        digests = list(ex.map(sha256, paths))
    return {str(p): {"sha256": d, "bytes": p.stat().st_size} for p, d in zip(paths, digests)}


def parse_tsv_blocks(text: str) -> list[dict]:
    """chq outputs are TSVWithNames blocks; a header row starts with 'step' or 'metric'."""
    blocks, header = [], None
    for line in text.splitlines():
        cols = line.split("\t")
        if cols and cols[0] in ("step", "metric"):
            header = cols
            continue
        if header and len(cols) == len(header):
            blocks.append(dict(zip(header, cols)))
    return blocks


def run_record(log_dir: pathlib.Path, name: str) -> dict:
    p = lambda ext: log_dir / f"{name}.{ext}"
    rec = {"name": name, "command": p("cmd").read_text().strip(),
           "statements_file": str(p("sql")), "statements_sha256": sha256(p("sql"))}
    out = p("out").read_text() if p("out").exists() else ""
    rec["stdout_raw"] = out
    rec["stdout_sha256"] = hashlib.sha256(out.encode()).hexdigest()
    rec["rows"] = parse_tsv_blocks(out)
    err = p("err").read_text() if p("err").exists() else ""
    rec["stderr_exceptions"] = [l for l in err.splitlines() if "Exception" in l]
    m = re.search(r"([\d.]+) real", err)
    rss = re.search(r"(\d+)\s+maximum resident set size", err)
    foot = re.search(r"(\d+)\s+peak memory footprint", err)
    rec["wall_s"] = float(m.group(1)) if m else None
    rec["peak_rss_bytes"] = int(rss.group(1)) if rss else None
    rec["peak_footprint_bytes"] = int(foot.group(1)) if foot else None
    if p("disk").exists():
        disk = p("disk").read_text().splitlines()
        frees = [int(x) for x in re.findall(r"free_gib=(\d+)", "\n".join(disk))]
        rec["disk_free_gib_min"] = min(frees) if frees else None
        rec["disk_start_end"] = [disk[0], disk[-1]] if disk else []
    return rec


def wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 5), round(c + h, 5)]


def main(out_dir: str, receipt_path: str, hash_fold: bool) -> None:
    base = pathlib.Path(out_dir)
    logs = base / "logs"
    names = sorted(p.stem for p in logs.glob("*.cmd"))
    runs = {n: run_record(logs, n) for n in names}
    for py in sorted(logs.glob("*.pyout")):
        runs[py.stem] = {"name": py.stem, "command": (logs / f"{py.stem}.pycmd").read_text().strip(),
                         "stdout_raw": py.read_text()}

    inputs = hash_many([LAKE / "netcup-export" / d / f for d in NETCUP_DAYS for f in NETCUP_FILES] + OTHER_INPUTS)
    fold = {}
    if hash_fold:
        for table, rel in FOLD_TABLES.items():
            files = sorted(p for p in (LAKE / "graph-build-v3/ch/store" / rel).rglob("*") if p.is_file())
            digests = hash_many(files)
            agg = hashlib.sha256("".join(f"{k}\t{v['sha256']}\n" for k, v in sorted(digests.items())).encode()).hexdigest()
            fold[table] = {"dir": str(LAKE / "graph-build-v3/ch/store" / rel), "files": len(files),
                           "bytes": sum(v["bytes"] for v in digests.values()), "manifest_sha256": agg}
    outputs = hash_many(sorted(p for p in (base / "out").rglob("*") if p.is_file()))

    tree = {}
    for tag in ("before", "after"):
        f = logs / f"ch_tree_{tag}.txt"
        if f.exists():
            tree[tag] = sha256(f)
    receipt = {
        "step": "S2 local prep, whole-network signals plan",
        "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md §3.1 S2",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md §9.1",
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "host_clickhouse": subprocess.run(["clickhouse", "--version"], capture_output=True, text=True).stdout.strip(),
        "fold_db_untouched": {"tree_listing_sha256": tree,
                              "unchanged": bool(tree.get("before") and tree.get("before") == tree.get("after")),
                              "how": "find graph-build-v3/ch | stat name,size,mtime,mode | sort, before and after every run"},
        "inputs": inputs,
        "fold_tables": fold,
        "outputs": outputs,
        "runs": runs,
    }
    gates = {}
    gates.update(gate_2b(runs))
    gates.update(gate_2a(runs))
    receipt["gates"] = gates
    receipt["h_tests"] = h_tests(runs)
    pathlib.Path(receipt_path).write_text(json.dumps(receipt, indent=1))
    print(json.dumps(gates, indent=1))


def first(runs: dict, run: str, step: str, key: str = "step") -> dict | None:
    for row in runs.get(run, {}).get("rows", []):
        if row.get(key) == step:
            return row
    return None


def gate_2b(runs: dict) -> dict:
    out = {}
    pa = first(runs, "s2b_repro", "parent_author_like", "metric")
    rb = first(runs, "s2b_repro", "reply_back", "metric")
    fs_raw = runs.get("s2b_follow_share", {}).get("stdout_raw")
    fs = json.loads(fs_raw) if fs_raw else None
    for key, val, exp in (("2b_parent_author_like", pa and int(pa["parent_author_liked_n"]) / int(pa["replies"]), 0.492),
                          ("2b_reply_back", rb and int(rb["replied_back_n"]) / int(rb["replies"]), 0.280),
                          ("2b_follower_share", fs and fs["follower_share_unrounded"], 0.597)):
        out[key] = {"measured": None if val is None else round(val, 5), "expected": exp, "tolerance": 0.01,
                    "pass": None if val is None else abs(val - exp) <= 0.01}
    return out


def gate_2a(runs: dict) -> dict:
    ca = first(runs, "s2a_train", "check_a")
    tr = first(runs, "s2a_train", "train")
    deg = first(runs, "s2a_train", "deg")
    out = {}
    if ca:
        out["2a_live_snapshot_producer_edges"] = {
            "measured": int(ca["producer_edges_snap"]), "expected": 2036276732,
            "pass": int(ca["producer_edges_snap"]) == 2036276732,
            "producers_measured": int(ca["producers_snap"]), "producers_fold": int(ca["fold_producers"]),
            "producers_matching_fold_exactly": int(ca["producers_matching_fold_exactly"])}
    if deg:
        out["2a_live_snapshot_edges_all"] = {"measured": int(deg["distinct_pairs_snap"]), "expected": 2647488881,
                                             "pass": int(deg["distinct_pairs_snap"]) == 2647488881}
    if tr:
        out["2a_max_train_tid_before_T"] = {"max_train_c_us": int(tr["max_train_c_us"]), "max_train_tid_time": tr["max_train_tid_time"],
                                            "T_boundary_c_us": 1786838400000000, "pass": int(tr["max_train_c_us"]) < 1786838400000000}
    return out


def h_tests(runs: dict) -> dict:
    res = {}
    for row in runs.get("s2c_heuristics", {}).get("rows", []):
        step = row.get("step")
        if step in ("H1", "H2", "H5"):
            n = int(row["pairs"])
            k = int(row.get("liked_top_level_within_48h") or row.get("same_community") or row.get("followed_within_48h_live") or 0)
            row = dict(row, wilson95=wilson(k, n))
            res.setdefault(step, []).append(row)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--hash-fold" in sys.argv[3:])
