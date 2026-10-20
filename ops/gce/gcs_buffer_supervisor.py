#!/usr/bin/env python3
"""Own one bounded collector/buffer run; stop collection on any unsafe exit."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def utc_timestamp(value: str) -> float:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline must have a timezone")
    return parsed.timestamp()


def collector_may_start(status: dict, free_bytes: int, segment_count: int) -> bool:
    return (
        status.get("pending_sealed") == 0
        and free_bytes >= 100 * 1024**3
        and segment_count <= 3
    )


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as out:
        json.dump(data, out, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/data/gcs-buffer"))
    parser.add_argument("--segments-dir", type=Path, default=Path("/data/jetstream/segments"))
    parser.add_argument("--receipt-script", type=Path,
                        default=Path("/opt/catbird-offloader/backfill/make_receipt.mjs"))
    parser.add_argument("--source-epoch", default="buffer-admission-20261020")
    parser.add_argument("--poweroff-on-exit", action="store_true")
    args = parser.parse_args()
    deadline = utc_timestamp(args.deadline)
    if deadline <= time.time():
        raise RuntimeError("authorized deadline has already expired")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.work_dir / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_service = subprocess.run(
        ["systemctl", "is-active", "--quiet", "catbird-gce-offloader.service"],
        check=False,
    )
    if old_service.returncode == 0:
        raise RuntimeError("old Drive offloader is active")
    container = subprocess.check_output(
        ["docker", "inspect", "js", "--format", "{{.State.Status}} {{.HostConfig.RestartPolicy.Name}}"],
        text=True,
    ).strip()
    if container != "exited no":
        raise RuntimeError(f"expected stopped collector with restart disabled; found {container!r}")

    stop_requested = False

    def stop_handler(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("gcs_archive_buffer.py")),
        "--bucket", args.bucket, "--source-id", "gce-selfhost-001",
        "--source-epoch", args.source_epoch, "--segments-dir", str(args.segments_dir),
        "--work-dir", str(args.work_dir), "--receipt-script", str(args.receipt_script),
        "--max-bytes", str(1024**4), "--run-hours", str((deadline-time.time())/3600),
        "--poll-seconds", "5",
    ]
    child = subprocess.Popen(command)
    started_collector = False
    started_at = time.time()
    reason = "unknown"
    state = {}
    try:
        while True:
            now = time.time()
            free_bytes = shutil.disk_usage(args.segments_dir).free
            segment_count = sum(1 for _ in args.segments_dir.glob("seg_*.jss"))
            status_path = args.work_dir / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            fresh = status_path.exists() and status_path.stat().st_mtime >= started_at
            state = {
                "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "deadline": args.deadline, "collector_started": started_collector,
                "free_disk_bytes": free_bytes, "local_segments": segment_count,
                "buffer_status": status, "buffer_status_from_this_run": fresh,
            }
            atomic_json(args.work_dir / "supervisor_status.json", state)
            if stop_requested:
                reason = "operator_or_service_stop"
                break
            if now >= deadline - 180:
                reason = "authorized_deadline"
                break
            if child.poll() is not None:
                reason = f"buffer_exited_{child.returncode}"
                break
            if free_bytes < 20 * 1024**3:
                reason = "disk_free_below_20GiB"
                break
            if not started_collector and fresh and collector_may_start(status, free_bytes, segment_count):
                subprocess.run(["docker", "start", "js"], check=True)
                started_collector = True
                print(json.dumps({"event": "collector_started_after_verified_backlog"}), flush=True)
            if started_collector:
                running = subprocess.check_output(
                    ["docker", "inspect", "js", "--format", "{{.State.Running}}"], text=True,
                ).strip()
                if running != "true":
                    reason = "collector_exited"
                    break
            time.sleep(10)
    finally:
        # Stop the writer before its uploader. Never restart automatically.
        subprocess.run(["docker", "stop", "--time", "120", "js"], check=False)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state.update(stop_reason=reason, stopped_at=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(args.work_dir / "supervisor_status.json", state)
        print(json.dumps({"event": "bounded_run_stopped", "reason": reason}), flush=True)
        os.sync()
        lock.close()
        if args.poweroff_on_exit:
            subprocess.run(["systemctl", "poweroff"], check=True)
    return 0 if reason in ("authorized_deadline", "operator_or_service_stop") else 1


if __name__ == "__main__":
    raise SystemExit(main())
