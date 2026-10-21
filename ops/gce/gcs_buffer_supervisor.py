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
import threading
import time
import uuid

TIB = 1024 ** 4
MAX_BYTES = 3 * TIB
MAX_RUN_SECONDS = 72 * 3600
SHUTDOWN_RESERVE_SECONDS = 150


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
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Watchdog:
    """The parent owns elapsed time; a blocked uploader cannot extend the run."""

    def __init__(self, deadline: float, stale_seconds: float, run_id: str):
        self.stop_at = time.monotonic() + deadline - time.time() - SHUTDOWN_RESERVE_SECONDS
        self.last_heartbeat = time.monotonic()
        self.stale_seconds, self.run_id = stale_seconds, run_id
        self.revision = None

    def observe(self, status: dict, revision) -> bool:
        fresh = status.get("run_id") == self.run_id and revision is not None
        if fresh and revision != self.revision:
            self.revision = revision
            self.last_heartbeat = time.monotonic()
        return fresh

    def stop_reason(self, wall_now: float, deadline: float) -> str | None:
        # Also stop on a forward wall-clock jump; a backward jump cannot extend
        # the monotonic limit calculated at launch.
        if wall_now >= deadline - SHUTDOWN_RESERVE_SECONDS or time.monotonic() >= self.stop_at:
            return "authorized_deadline"
        if time.monotonic() - self.last_heartbeat >= self.stale_seconds:
            return "buffer_status_stale"
        return None


def status_snapshot(path: Path) -> tuple[dict, tuple | None]:
    try:
        with path.open() as stream:
            info = os.fstat(stream.fileno())
            status = json.load(stream)
        return status if isinstance(status, dict) else {}, (info.st_ino, info.st_mtime_ns, info.st_size)
    except (OSError, ValueError):
        return {}, None


def stop_runtime(child) -> list[str]:
    """Bound each stop step, including an unresponsive Docker daemon/child."""
    errors = []
    try:
        result = subprocess.run(["docker", "stop", "--time", "60", "js"], check=False, timeout=65)
        if result.returncode:
            errors.append(f"collector_stop_exit_{result.returncode}")
    except (OSError, subprocess.TimeoutExpired) as error:
        errors.append(f"collector_stop_{type(error).__name__}")
    if errors:
        try:
            subprocess.run(["docker", "kill", "js"], check=False, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(f"collector_kill_{type(error).__name__}")
    if child is not None:
        # The verifier inherits this private process group. Stop it too, so a
        # replacement run cannot overlap a surviving verifier/uploader.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except OSError as error:
            if not isinstance(error, ProcessLookupError):
                errors.append(f"buffer_terminate_{type(error).__name__}")
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # An exited leader does not imply its verifier exited. Always finish
        # private-group cleanup; no source/checkpoint deletion occurs here.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except OSError as error:
            if not isinstance(error, ProcessLookupError):
                errors.append(f"buffer_kill_{type(error).__name__}")
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            errors.append("buffer_kill_timeout")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/data/gcs-buffer"))
    parser.add_argument("--segments-dir", type=Path, default=Path("/data/jetstream/segments"))
    parser.add_argument("--receipt-script", type=Path,
                        default=Path("/opt/catbird-offloader/backfill/make_receipt.mjs"))
    parser.add_argument("--source-epoch", default="buffer-admission-20261020")
    parser.add_argument("--max-bytes", type=int, default=TIB)
    parser.add_argument("--stale-seconds", type=float, default=900)
    parser.add_argument("--poweroff-on-exit", action="store_true")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.work_dir / "supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        # A duplicate process must never stop the valid owner or its VM.
        lock.close()
        raise
    try:
        return supervise(args)
    finally:
        # Expired configuration, boot/preflight errors, and failed state writes
        # all reach the backstop once this process owns the exclusive lock.
        try:
            if args.poweroff_on_exit:
                subprocess.run(["systemctl", "poweroff"], check=True, timeout=10)
        finally:
            lock.close()


def supervise(args) -> int:
    deadline = utc_timestamp(args.deadline)
    remaining = deadline - time.time()
    if not SHUTDOWN_RESERVE_SECONDS < remaining <= MAX_RUN_SECONDS:
        raise ValueError("deadline must leave more than 150 seconds and be within 72 hours")
    if not 0 < args.max_bytes <= MAX_BYTES or not 0 < args.stale_seconds <= 3600:
        raise ValueError("max-bytes must be <=3TiB and stale-seconds must be <=3600; both positive")
    if 'live_segments' in args.segments_dir.resolve().parts:
        raise ValueError("bootstrap live_segments must remain local for collector merge")
    old_service = subprocess.run(
        ["systemctl", "is-active", "--quiet", "catbird-gce-offloader.service"],
        check=False,
        timeout=10,
    )
    if old_service.returncode == 0:
        raise RuntimeError("old Drive offloader is active")
    container = subprocess.check_output(
        ["docker", "inspect", "js", "--format", "{{.State.Status}} {{.HostConfig.RestartPolicy.Name}}"],
        text=True,
        timeout=10,
    ).strip()
    if container not in ("exited no", "created no"):
        raise RuntimeError(f"expected stopped collector with restart disabled; found {container!r}")

    stop_requested = threading.Event()

    def stop_handler(_signum, _frame):
        stop_requested.set()

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    run_id = uuid.uuid4().hex
    run_dir = args.work_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    # Preserve the prior process/boot observations before the shared status
    # filenames are replaced. No source, checkpoint, or journal is reset.
    for name in ("supervisor_status.json", "status.json"):
        previous = args.work_dir / name
        if previous.exists():
            shutil.copy2(previous, run_dir / ("previous_" + name))
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    boot_id = boot_path.read_text().strip() if boot_path.exists() else "unavailable"
    atomic_json(run_dir / "start.json", {
        "run_id": run_id, "boot_id": boot_id, "pid": os.getpid(),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "deadline": args.deadline, "max_bytes": args.max_bytes,
        "segments_dir": str(args.segments_dir.resolve()), "source_epoch": args.source_epoch,
        "admission_accounting": "cumulative_observed_generations_no_refund",
    })
    watchdog = Watchdog(deadline, args.stale_seconds, run_id)
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("gcs_archive_buffer.py")),
        "--bucket", args.bucket, "--source-id", "gce-selfhost-001",
        "--source-epoch", args.source_epoch, "--segments-dir", str(args.segments_dir),
        "--work-dir", str(args.work_dir), "--receipt-script", str(args.receipt_script),
        "--max-bytes", str(args.max_bytes), "--run-hours", str((deadline-time.time())/3600),
        "--run-id", run_id,
        "--poll-seconds", "5",
    ]
    child = None
    started_collector = False
    reason = "unknown"
    state = {}
    try:
        child = subprocess.Popen(command, start_new_session=True)
        while True:
            now = time.time()
            reason = "operator_or_service_stop" if stop_requested.is_set() else watchdog.stop_reason(now, deadline)
            if reason:
                break
            free_bytes = shutil.disk_usage(args.segments_dir).free
            segment_count = sum(1 for _ in args.segments_dir.glob("seg_*.jss"))
            status_path = args.work_dir / "status.json"
            status, revision = status_snapshot(status_path)
            fresh = watchdog.observe(status, revision)
            state = {
                "run_id": run_id, "boot_id": boot_id,
                "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "deadline": args.deadline, "collector_started": started_collector,
                "free_disk_bytes": free_bytes, "local_segments": segment_count,
                "buffer_status": status, "buffer_status_from_this_run": fresh,
                "max_bytes": args.max_bytes, "stale_seconds": args.stale_seconds,
            }
            atomic_json(args.work_dir / "supervisor_status.json", state)
            if child.poll() is not None:
                reason = f"buffer_exited_{child.returncode}"
                break
            if free_bytes < 20 * 1024**3:
                reason = "disk_free_below_20GiB"
                break
            if not started_collector and fresh and status.get("reason") == "ready" and collector_may_start(status, free_bytes, segment_count):
                subprocess.run(["docker", "start", "js"], check=True, timeout=20)
                started_collector = True
                print(json.dumps({"event": "collector_started_after_verified_backlog"}), flush=True)
            if started_collector:
                running = subprocess.check_output(
                    ["docker", "inspect", "js", "--format", "{{.State.Running}}"], text=True, timeout=10,
                ).strip()
                if running != "true":
                    reason = "collector_exited"
                    break
            stop_requested.wait(5)
    except Exception as error:
        reason = f"supervisor_error_{type(error).__name__}"
        state["error"] = str(error)
    finally:
        # Stop the writer before its uploader. Never restart automatically.
        state.update(run_id=run_id, boot_id=boot_id, stop_errors=stop_runtime(child),
                     stop_reason=reason, stopped_at=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(args.work_dir / "supervisor_status.json", state)
        atomic_json(run_dir / "stop.json", state)
        print(json.dumps({"event": "bounded_run_stopped", "reason": reason}), flush=True)
        subprocess.run(["sync"], check=False, timeout=5)
    return 0 if reason in ("authorized_deadline", "operator_or_service_stop") else 1


if __name__ == "__main__":
    raise SystemExit(main())
