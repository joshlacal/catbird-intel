#!/usr/bin/env python3
"""Finish the existing archive-only fold locally, preserving stage evidence.

Never ships to netcup. The already-running fold used live archive globs, so the
receipt explicitly cannot claim a frozen consumed input inventory. Downstream
artifacts are SHA-256 pinned. A failed stage stops the pipeline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-pid", type=int, required=True)
    parser.add_argument("--lake", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local")
    parser.add_argument("--rayon-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    out = args.lake / "graph-build/out"
    logs = args.lake / "logs"
    status_path = out / "pipeline_status.json"
    status = {"started_at": stamp(), "graph_version": "phase1-v1", "stage": "waiting_for_fold",
              "fold_pid": args.fold_pid, "stages": []}
    if status_path.exists():
        if not args.resume:
            raise SystemExit("Pipeline status exists; use --resume (no duplicate driver)")
        status = json.loads(status_path.read_text())
        previous_pid = status.get("driver_pid")
        if previous_pid:
            result = subprocess.run(["ps", "-p", str(previous_pid), "-o", "command="], capture_output=True, text=True)
            if "run_phase1_v1.py" in result.stdout:
                raise SystemExit(f"Previous driver PID {previous_pid} is still running")
    status["driver_pid"] = os.getpid()

    def save() -> None:
        status["updated_at"] = stamp()
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    def stage(name: str, command: list[str], artifacts: list[pathlib.Path], env: dict | None = None) -> None:
        log = logs / f"phase1-v1-{name}.log"
        previous = next((r for r in reversed(status["stages"]) if r["stage"] == name and r["returncode"] == 0), None)
        if previous:
            for path, digest in previous["artifacts"].items():
                if not pathlib.Path(path).exists() or sha256(pathlib.Path(path)) != digest:
                    raise RuntimeError(f"Sealed artifact changed: {path}")
            print(f"{stamp()} verified completed {name}; skipping", flush=True)
            return
        # Preserve unsealed remnants before retrying a failed stage.
        remnants = [p for p in [log, *artifacts] if p.exists()]
        if remnants:
            archive = args.lake / "_quarantine" / "phase1-v1-retry" / str(time.time_ns())
            archive.mkdir(parents=True)
            for path in remnants:
                path.rename(archive / path.name)
        status["stage"] = name
        save()
        print(f"{stamp()} starting {name}; log={log}", flush=True)
        start = time.monotonic()
        with log.open("xb") as fh:
            result = subprocess.run(command, cwd=repo, env=env, stdout=fh, stderr=subprocess.STDOUT)
        receipt = {"stage": name, "command": command, "returncode": result.returncode,
                   "seconds": round(time.monotonic() - start, 3), "log": str(log), "finished_at": stamp()}
        if result.returncode == 0:
            receipt["artifacts"] = {str(path): sha256(path) for path in artifacts}
        status["stages"].append(receipt)
        save()
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); inspect {log}")
        print(f"{stamp()} completed {name}", flush=True)

    save()
    try:
        fold_command = subprocess.run(["ps", "-p", str(args.fold_pid), "-o", "command="], capture_output=True, text=True).stdout.strip()
        if fold_command and ("clickhouse local" not in fold_command or str(args.lake / "graph-build/fold.sql") not in fold_command):
            raise RuntimeError("Fold PID does not identify the expected ClickHouse fold process")
        if fold_command:
            status["observed_fold_command"] = fold_command
            save()
        while True:
            try:
                os.kill(args.fold_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(15)
        for name in ["producers.tsv", "edges.bin", "stats.tsv"]:
            if not (out / name).is_file() or not (out / name).stat().st_size:
                raise RuntimeError(f"Fold stopped without nonempty {name}; do not continue")
        if (out / "edges.bin").stat().st_size % 8:
            raise RuntimeError("Fold edges.bin has a truncated RowBinary record")
        lines = (out / "stats.tsv").read_text().strip().splitlines()
        if len(lines) != 2:
            raise RuntimeError("Fold stats.tsv is not a complete header and result row")
        stats = dict(zip(lines[0].split("\t"), map(int, lines[1].split("\t"))))
        required = {"follow_records", "follow_edges", "followers", "inactive_accounts", "producers", "producer_edges"}
        if stats.keys() != required or not all(stats[k] > 0 for k in required - {"inactive_accounts"}):
            raise RuntimeError(f"Incomplete or empty fold stats: {stats}")
        if (out / "edges.bin").stat().st_size // 8 != stats["producer_edges"]:
            raise RuntimeError("Binary edge count disagrees with fold stats")
        n_producers = follower_sum = 0
        with (out / "producers.tsv").open() as fh:
            for line in fh:
                pid, did, count = line.rstrip("\n").split("\t")
                if int(pid) != n_producers or not did.startswith("did:"):
                    raise RuntimeError("Producer IDs are not dense or DID is invalid")
                n_producers += 1
                follower_sum += int(count)
        if n_producers != stats["producers"] or follower_sum != stats["producer_edges"]:
            raise RuntimeError("Producer row/follower counts disagree with fold stats")
        fold_sql = args.lake / "graph-build/fold.sql"
        sql = fold_sql.read_text()
        expected_glob = str(args.lake / "{archive-graph,gce-archive-graph}/events/events-*.parquet")
        if expected_glob not in sql or "supplemental" in sql.lower():
            raise RuntimeError("Fold SQL is not the expected archive-only input")
        provenance = {
            "graph_version": "phase1-v1", "created_at": stamp(),
            "supplemental_edges_excluded": True,
            "exclusion_evidence": "fold SQL reads only archive events-*.parquet; supplemental file quarantined outside both directories",
            "fold_sql_sha256": sha256(fold_sql), "fold_sql": sql,
            "observed_fold_command": status.get("observed_fold_command"),
            "fold_stats": stats,
            "exact_consumed_shard_inventory_available": False,
            "input_limitation": "fold began before inventory was frozen; decoder appended shards during execution",
            "artifacts": {name: {"bytes": (out / name).stat().st_size, "sha256": sha256(out / name)}
                          for name in ["producers.tsv", "edges.bin", "stats.tsv"]},
        }
        provenance_path = out / "input_provenance.json"
        if provenance_path.exists():
            sealed = json.loads(provenance_path.read_text())
            if sealed["artifacts"] != provenance["artifacts"] or sealed["fold_sql_sha256"] != provenance["fold_sql_sha256"]:
                raise RuntimeError("Fold inputs changed since the sealed provenance receipt")
        else:
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        env = {**os.environ, "RAYON_NUM_THREADS": str(args.rayon_threads), "CARGO_BUILD_JOBS": "2",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
        stage("simgraph", ["cargo", "run", "--release", "--manifest-path", "graph/simgraph/Cargo.toml",
                          "--", "--dir", str(out)], [out / "sim.bin"], env)
        stage("cluster", [sys.executable, "graph/cluster.py", "--dir", str(out),
                          "--resolution", "0", "--max-communities", "30000",
                          "--target-size", "100", "--size-objective", "community", "--search-steps", "7"],
              [out / "known_for.parquet", out / "communities.parquet", out / "cluster_stats.json"], env)
        stage("evaluation", [sys.executable, "graph/eval_phase1.py", "--graph-dir", str(out),
                             "--events-dir", str(args.lake / "archive-graph/events"),
                             "--events-dir", str(args.lake / "gce-archive-graph/events")],
              [out / "phase1_eval_report.json", out / "communities_labeled.parquet", out / "evaluation_input_manifest.json"], env)
        pairs_dir = args.lake / "drive-sync-staging/training-banks"
        pair_command = [sys.executable, "ops/export_pairs_v1.py", "--graph-dir", str(out)]
        if (pairs_dir / "phase5_live_source_posts_v1.manifest.json").exists():
            pair_command.extend(["--source-posts", str(pairs_dir / "phase5_live_source_posts_v1.parquet")])
        stage("pairs", pair_command,
              [pairs_dir / "phase5_social_encoder_pairs_v1.parquet", pairs_dir / "phase5_social_encoder_pairs_v1.manifest.json"], env)
        status["stage"] = "complete_local_unqualified"
        status["completed_at"] = stamp()
        status["qualification"] = "Measured graph and pair outputs; no production ship, no human review or A Blue Start comparison"
        save()
        print(json.dumps(status, indent=2), flush=True)
    except Exception as exc:
        status["stage_failed"] = status["stage"]
        status["stage"] = "failed"
        status["error"] = str(exc)
        save()
        raise


if __name__ == "__main__":
    main()
