#!/usr/bin/env python3
"""Refresh bounded retained snapshots while a pinned writer exclusion holds.

This is a supervisor of the reviewed snapshot selector and serial runner. It
does not itself send Drive requests. An owner-authored, hash-pinned standing
envelope supplies the nonlocal manual/Colab/Netcup exclusion promise; fresh
local checks supply only facts this VM can observe. Any ambiguity stops.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import uuid

from drive_transfer_authority import TransferAuthority
from gcs_drive_drain import Conflict
from gcs_drive_retained_overflow import (
    BUCKET, DRIVE_ROOT, CAPTURE_SEGMENTS_DIR, CAPTURE_RESTART_FLOOR_BYTES,
)
from gcs_drive_retained_runner import STOP_AT, _account_gate, _arm_parent_death


SCHEMA = "catbird.drive.retained-continuous.v1"
ENVELOPE_SCHEMA = "catbird.drive.retained-standing-exclusion.v1"
SNAPSHOT_SCHEMA = "catbird.drive.retained-snapshot.v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
UNIT_NAMES = ("catbird-gce-offloader.service", "catbird-gcs-drive-drain.service",
              "catbird-gcs-drive-drain.timer")
WRITER_MARKERS = (b"gce_segment_offloader.py", b"gcs_drive_drain.py",
                  b"gcs_drive_diagnostic.py", b"gcs_drive_retained_overflow.py",
                  b"gcs_drive_retained_runner.py", b"upload_repair_v1.py",
                  b"gdrive_lake_sync.py", b"rclone")
MAX_CYCLES = 5000  # >72 hours at one empty scan per minute; STOP still wins.
IDLE_SECONDS = 60
MIN_WINDOW_SECONDS = 1290  # 20-minute pair, runner margin, and clock slack.
MAX_STANDING_SECONDS = 3600  # A reviewed nonlocal assertion expires within one hour.
CONTINUOUS_ROOT = Path("/data/gcs-drive-retained-continuous")
MAX_CACHE_BYTES = 4 * (1 << 30)
SNAPSHOT_RESERVATION_BYTES = 16 * (1 << 20)  # <=9 receipts + manifests at 1 MiB each.


class WindowClosing(Conflict):
    """The reviewed standing or capture window ended without a child outcome."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _strict_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise Conflict("Continuous JSON contains duplicate keys")
        value[key] = item
    return value


def _bad_constant(_value):
    raise Conflict("Continuous JSON contains nonfinite number")


def _read_json(path: Path, limit: int) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise Conflict("Continuous input is missing, symlinked, or oversized")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_strict_pairs,
                           parse_constant=_bad_constant)
    except (ValueError, TypeError, UnicodeError) as error:
        raise Conflict("Continuous JSON is malformed") from error
    if not isinstance(value, dict):
        raise Conflict("Continuous JSON must be an object")
    return value, raw


def _finite(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise Conflict("Invalid " + label)
    return float(value)


def _file_sha256(path: Path, limit: int) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise Conflict("Pinned local file is missing, symlinked, or oversized")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def standing_envelope(path: Path, expected_sha256: str, *, authority_db: Path,
                      oauth_file: Path, account_id: str, writer_id: str,
                      now: float, require_root: bool = True) -> tuple[dict, str]:
    """Verify immutable owner attestation, pinned credentials, and time scope.

    The envelope attests nonlocal holds; this function does not claim to
    observe another host or a browser. The worker independently checks the
    live Drive permission ID through its paced authority before media sends.
    """
    if not SHA256.fullmatch(expected_sha256):
        raise Conflict("Standing envelope digest is invalid")
    if require_root and (os.geteuid() != 0 or path.stat().st_uid != 0 or
                         stat.S_IMODE(path.stat().st_mode) & 0o077):
        raise Conflict("Standing envelope must be root-owned and private")
    envelope, raw = _read_json(path, 16 * 1024)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise Conflict("Standing envelope bytes differ from pinned review")
    if (envelope.get("schema") != ENVELOPE_SCHEMA or
        envelope.get("account_id") != account_id or
        envelope.get("writer_id") != writer_id or
        envelope.get("drive_root") != DRIVE_ROOT or
        envelope.get("bucket") != BUCKET or
        envelope.get("authority_db") != str(authority_db.resolve(strict=True)) or
        not isinstance(envelope.get("owner_reference"), str) or
        not envelope["owner_reference"].strip()):
        raise Conflict("Standing envelope identity differs")
    issued = _finite(envelope.get("issued_at"), "standing issue time")
    identity_at = _finite(envelope.get("account_identity_verified_at"),
                          "account verification time")
    if issued > now or identity_at > issued or now - identity_at > 72 * 3600:
        raise Conflict("Standing account verification is missing or stale")
    untils = tuple(_finite(envelope.get(key), key) for key in (
        "manual_colab_hold_until", "netcup_nonwriter_hold_until",
        "other_writer_hold_until"))
    # A distant timestamp in a remote-writer attestation cannot silently
    # authorize an unattended multi-day run. Renewal needs a new reviewed
    # file/digest and a fresh supervisor start.
    exclusion_until = min(*untils, STOP_AT, issued + MAX_STANDING_SECONDS)
    if exclusion_until < now + MIN_WINDOW_SECONDS:
        raise Conflict("Standing nonlocal exclusion cannot cover another pair")
    pinned_oauth = envelope.get("oauth_file_sha256")
    if not isinstance(pinned_oauth, str) or not SHA256.fullmatch(pinned_oauth):
        raise Conflict("Standing OAuth fingerprint is invalid")
    if _file_sha256(oauth_file, 64 * 1024) != pinned_oauth:
        raise Conflict("Existing OAuth bytes differ from verified account binding")
    return envelope, expected_sha256


def local_writer_preflight(*, unit_state=None, proc_cmdlines=None) -> dict:
    """Observe known VM writer paths without changing units or processes."""
    def live_unit(name):
        result = subprocess.run(
            ["systemctl", "show", name, "--no-pager", "--property=ActiveState",
             "--property=UnitFileState"], capture_output=True, timeout=5, check=False)
        if result.returncode or len(result.stdout) > 4096:
            raise Conflict("Legacy writer unit state is unavailable")
        fields = dict(line.split("=", 1) for line in result.stdout.decode("ascii").splitlines()
                      if "=" in line)
        return fields
    unit_state = unit_state or live_unit
    report = {}
    for name in UNIT_NAMES:
        fields = unit_state(name)
        if (fields.get("ActiveState") not in ("inactive", "failed") or
            fields.get("UnitFileState") not in ("disabled", "masked", "static", "not-found")):
            raise Conflict("A legacy Drive writer unit or timer is not held")
        report[name] = (fields["ActiveState"], fields["UnitFileState"])
    if proc_cmdlines is None:
        def proc_cmdlines():
            for path in Path("/proc").glob("[0-9]*/cmdline"):
                try:
                    with path.open("rb") as source:
                        yield source.read(4097)
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise Conflict("Process inventory is unreadable") from error
    observed = 0
    for cmdline in proc_cmdlines():
        observed += 1
        if observed > 32768 or len(cmdline) > 4096:
            raise Conflict("Process inventory exceeds bound")
        if any(marker in cmdline for marker in WRITER_MARKERS):
            raise Conflict("Another known Drive writer process is running")
    return {"legacy_units": report, "processes_inspected": observed}


def review_evidence(candidate: dict, *, account_id: str, writer_id: str,
                    envelope_sha256: str, local_report: dict,
                    exclusion_until: float, now: float) -> dict:
    """Bind fresh observed gates to one manifest and an owner attestation."""
    basis = {"run_id": candidate["release_id"], "writer_id": writer_id,
             "account_id": account_id,
             "reviewed_manifest_sha256": candidate["manifest_sha256"],
             "observed_at": now, "exclusion_valid_until": exclusion_until,
             "history_basis": "unproven", "account_identity_verified": True,
             "all_writers_excluded": True, "quota_rejection_accepted": True,
             "standing_envelope_sha256": envelope_sha256,
             "local_writer_preflight": local_report}
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**basis, "evidence_sha256": hashlib.sha256(raw).hexdigest()}


def _write_new(path: Path, raw: bytes) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise Conflict("Continuous artifact path is symlinked")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _event(path: Path, run_id: str, state: str, **details) -> None:
    """Append one fsynced full status row; never replace earlier evidence."""
    if path.is_symlink() or path.parent.is_symlink():
        raise Conflict("Continuous status path is symlinked")
    record = {"schema": SCHEMA, "run_id": run_id, "state": state,
              "updated_epoch": time.time(), **details}
    raw = json.dumps(record, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode() + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "ab", buffering=0) as output:
        output.write(raw)
        os.fsync(output.fileno())


def check_snapshot_cache(work_root: Path, capture_dir: Path) -> dict:
    """Bound all retained supervisor artifacts before and during selection."""
    if (work_root.is_symlink() or capture_dir.is_symlink() or
        not work_root.is_dir() or not capture_dir.is_dir() or
        os.stat(work_root).st_dev != os.stat(capture_dir).st_dev):
        raise Conflict("Snapshot cache and capture filesystem identity differs")
    total, visited = 0, 0
    def unreadable(error):
        raise Conflict("Snapshot cache subtree is unreadable") from error
    for directory, dirs, files in os.walk(work_root, followlinks=False,
                                           onerror=unreadable):
        visited += len(dirs) + len(files)
        if visited > 100_000:
            raise Conflict("Snapshot cache inventory exceeds bound")
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise Conflict("Snapshot cache contains symlink")
        for name in files:
            meta = (Path(directory) / name).lstat()
            if not stat.S_ISREG(meta.st_mode):
                raise Conflict("Snapshot cache contains nonregular file")
            total += meta.st_size
            if total + SNAPSHOT_RESERVATION_BYTES > MAX_CACHE_BYTES:
                raise Conflict("Cumulative snapshot cache ceiling reached")
    disk = os.statvfs(capture_dir)
    free = disk.f_bavail * disk.f_frsize
    if free - SNAPSHOT_RESERVATION_BYTES < CAPTURE_RESTART_FLOOR_BYTES:
        raise Conflict("Snapshot cache would breach capture restart headroom")
    return {"snapshot_cache_bytes": total, "capture_free_bytes": free}


def _run_bounded(command: list[str], *, timeout: float,
                 monitor=None) -> tuple[int, bytes]:
    if not sys.platform.startswith("linux") or timeout <= 0:
        raise Conflict("Continuous child needs bounded Linux execution")
    parent = os.getpid()
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             start_new_session=True, close_fds=True,
                             preexec_fn=lambda: _arm_parent_death(parent))
    try:
        end = time.monotonic() + timeout
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                output, _ = child.communicate(timeout=min(5, remaining))
                if monitor is not None:
                    monitor()
                break
            except subprocess.TimeoutExpired:
                if monitor is not None:
                    monitor()
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        raise
    finally:
        if child.stdout is not None:
            child.stdout.close()
    if len(output) > 256 * 1024:
        raise Conflict("Continuous child output exceeds bound")
    return child.returncode, output


def _snapshot(*, account_id: str, writer_id: str, authority_db: Path,
              work: Path, monitor=None, timeout: float = 330) -> tuple[dict, bytes]:
    command = [sys.executable, "-B", str(Path(__file__).with_name("gcs_drive_retained_queue.py")),
               "--account-id", account_id, "--writer-id", writer_id,
               "--authority-db", str(authority_db), "--work-dir", str(work),
               "--max-scan", "5000", "--max-pairs", "8",
               "--max-reserved-bytes", str(8 * 320 * (1 << 20)),
               "--max-run-seconds", "1200", "--max-snapshot-seconds", "300"]
    code, output = _run_bounded(command, timeout=min(330, timeout), monitor=monitor)
    if code != 0:
        raise Conflict("Bounded GCS snapshot failed")
    try:
        snapshot = json.loads(output, object_pairs_hook=_strict_pairs,
                              parse_constant=_bad_constant)
    except (ValueError, UnicodeError) as error:
        raise Conflict("Bounded GCS snapshot output is malformed") from error
    if (not isinstance(snapshot, dict) or snapshot.get("schema") != SNAPSHOT_SCHEMA or
        not isinstance(snapshot.get("candidates"), list) or
        len(snapshot["candidates"]) > 8):
        raise Conflict("Bounded GCS snapshot schema differs")
    return snapshot, output.strip()


def _last_status(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * (1 << 20):
        raise Conflict("Runner status is missing or oversized")
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise Conflict("Runner status has a torn final row")
    try:
        item = json.loads(raw.splitlines()[-1], object_pairs_hook=_strict_pairs,
                          parse_constant=_bad_constant)
    except (ValueError, UnicodeError) as error:
        raise Conflict("Runner status final row is malformed") from error
    if not isinstance(item, dict) or item.get("schema") != "catbird.drive.retained-runner.v1":
        raise Conflict("Runner status final row differs")
    return item


def _runner(*, snapshot_path: Path, snapshot_sha256: str, snapshot_dir: Path,
            authority_db: Path, evidence_dir: Path, account_id: str,
            writer_id: str, oauth_file: Path, status_file: Path,
            monotonic_stop: float | None = None) -> tuple[int, dict]:
    command = [sys.executable, "-B", str(Path(__file__).with_name("gcs_drive_retained_runner.py")),
               "--snapshot-json", str(snapshot_path), "--snapshot-sha256", snapshot_sha256,
               "--snapshot-dir", str(snapshot_dir), "--authority-db", str(authority_db),
               "--evidence-dir", str(evidence_dir), "--account-id", account_id,
               "--writer-id", writer_id, "--oauth-file", str(oauth_file),
               "--status-file", str(status_file)]
    remaining = STOP_AT - time.time() - 30
    if monotonic_stop is not None:
        remaining = min(remaining, monotonic_stop - time.monotonic() - 30)
    code, _ = _run_bounded(command, timeout=min(8 * 1200 + 60, remaining))
    return code, _last_status(status_file)


def run_cycles(*, envelope_path: Path, envelope_sha256: str, authority_db: Path,
               oauth_file: Path, account_id: str, writer_id: str, work_root: Path,
               snapshot_fn=_snapshot, runner_fn=_runner,
               local_check=local_writer_preflight, clock=time.time,
               monotonic=time.monotonic, sleep=time.sleep,
               capture_dir: Path = CAPTURE_SEGMENTS_DIR,
               cache_check=check_snapshot_cache,
               require_root: bool = True,
               max_cycles: int = MAX_CYCLES) -> dict:
    """Continue successful bounded snapshots; stop on ambiguous results."""
    if (not sys.platform.startswith("linux") or type(max_cycles) is not int or
        not 0 < max_cycles <= MAX_CYCLES or work_root.is_symlink() or
        not work_root.is_dir() or work_root.resolve() != CONTINUOUS_ROOT or
        authority_db.is_symlink() or
        not authority_db.is_file()):
        raise Conflict("Continuous runtime or authority root is invalid")
    canonical = authority_db.resolve(strict=True)
    lock_path = canonical.with_name(canonical.name + ".continuous.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    closed = 0
    supervisor_id = uuid.uuid4().hex
    status_path = work_root / "continuous_status.jsonl"
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started_wall, started_monotonic = clock(), monotonic()
        initial_envelope, _ = standing_envelope(
            envelope_path, envelope_sha256, authority_db=canonical,
            oauth_file=oauth_file, account_id=account_id, writer_id=writer_id,
            now=started_wall, require_root=require_root)
        standing_wall_stop = min(
            initial_envelope["manual_colab_hold_until"],
            initial_envelope["netcup_nonwriter_hold_until"],
            initial_envelope["other_writer_hold_until"],
            initial_envelope["issued_at"] + MAX_STANDING_SECONDS)
        standing_monotonic_stop = started_monotonic + max(
            0, standing_wall_stop - started_wall)
        capture_monotonic_stop = started_monotonic + max(0, STOP_AT - started_wall)
        monotonic_stop = min(capture_monotonic_stop, standing_monotonic_stop)
        def remaining() -> float:
            return min(STOP_AT - clock(), standing_wall_stop - clock(),
                       monotonic_stop - monotonic())
        def stop_reason() -> str:
            standing_left = min(standing_wall_stop - clock(),
                                standing_monotonic_stop - monotonic())
            if standing_left < MIN_WINDOW_SECONDS:
                return "standing_exclusion_approaching"
            return "capture_stop_approaching"
        _event(status_path, supervisor_id, "starting", pairs_closed=0)
        for _cycle in range(max_cycles):
            now = clock()
            if remaining() < MIN_WINDOW_SECONDS:
                reason = stop_reason()
                _event(status_path, supervisor_id, "stopped",
                       reason=reason, pairs_closed=closed)
                return {"state": "stopped", "reason": reason,
                        "pairs_closed": closed}
            envelope, digest = standing_envelope(
                envelope_path, envelope_sha256, authority_db=canonical,
                oauth_file=oauth_file, account_id=account_id, writer_id=writer_id,
                now=now, require_root=require_root)
            account = TransferAuthority.inspect_retained_account(
                canonical, writer_id=writer_id, account_id=account_id)
            _account_gate(account, now)
            cycle_dir = work_root / ("cycle-" + uuid.uuid4().hex)
            cycle_dir.mkdir(mode=0o700)
            snapshot_dir = cycle_dir / "snapshot"
            evidence_dir = cycle_dir / "evidence"
            snapshot_dir.mkdir(mode=0o700)
            evidence_dir.mkdir(mode=0o700)
            cache_check(work_root, capture_dir)
            def monitor_snapshot():
                cache_check(work_root, capture_dir)
                if remaining() < MIN_WINDOW_SECONDS:
                    raise WindowClosing(stop_reason())
            snapshot, snapshot_raw = snapshot_fn(account_id=account_id, writer_id=writer_id,
                                                 authority_db=canonical, work=snapshot_dir,
                                                 monitor=monitor_snapshot,
                                                 timeout=min(330, remaining() - MIN_WINDOW_SECONDS))
            monitor_snapshot()
            candidates = snapshot.get("candidates")
            if (snapshot.get("schema") != SNAPSHOT_SCHEMA or not isinstance(candidates, list)
                    or len(candidates) > 8 or any(
                        not isinstance(row, dict) or
                        not re.fullmatch(r"[0-9a-f]{32}", str(row.get("release_id", ""))) or
                        not SHA256.fullmatch(str(row.get("manifest_sha256", "")))
                        for row in candidates)):
                raise Conflict("Continuous snapshot candidate identity differs")
            if not candidates:
                _event(status_path, supervisor_id, "idle", pairs_closed=closed,
                       reason="no_current_live_candidate")
                sleep(min(IDLE_SECONDS, max(0, remaining() - MIN_WINDOW_SECONDS)))
                continue
            snapshot_path = snapshot_dir / "snapshot.json"
            _write_new(snapshot_path, snapshot_raw)
            # Snapshot reads can consume their full five-minute watchdog.
            # Recheck the standing file, pinned OAuth bytes, and local writers
            # *after* selection so the observation time below is honest.
            observed = clock()
            if remaining() < MIN_WINDOW_SECONDS:
                raise WindowClosing(stop_reason())
            envelope, digest = standing_envelope(
                envelope_path, envelope_sha256, authority_db=canonical,
                oauth_file=oauth_file, account_id=account_id, writer_id=writer_id,
                now=observed, require_root=require_root)
            local_report = local_check()
            account = TransferAuthority.inspect_retained_account(
                canonical, writer_id=writer_id, account_id=account_id)
            _account_gate(account, observed)
            exclusion_until = min(envelope["manual_colab_hold_until"],
                                  envelope["netcup_nonwriter_hold_until"],
                                  envelope["other_writer_hold_until"], STOP_AT,
                                  envelope["issued_at"] + MAX_STANDING_SECONDS)
            for candidate in candidates:
                evidence = review_evidence(candidate, account_id=account_id,
                                           writer_id=writer_id, envelope_sha256=digest,
                                           local_report=local_report,
                                           exclusion_until=exclusion_until, now=clock())
                _write_new(evidence_dir / (candidate["release_id"] + ".json"),
                           json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode())
            if remaining() < MIN_WINDOW_SECONDS:
                raise WindowClosing(stop_reason())
            status_file = cycle_dir / "runner_status.jsonl"
            _event(status_path, supervisor_id, "batch_running", pairs_closed=closed,
                   runner_status_path=str(status_file),
                   snapshot_sha256=hashlib.sha256(snapshot_raw).hexdigest())
            code, result = runner_fn(snapshot_path=snapshot_path,
                                     snapshot_sha256=hashlib.sha256(snapshot_raw).hexdigest(),
                                     snapshot_dir=snapshot_dir, authority_db=canonical,
                                     evidence_dir=evidence_dir, account_id=account_id,
                                     writer_id=writer_id, oauth_file=oauth_file,
                                     status_file=status_file,
                                     monotonic_stop=monotonic_stop)
            finished = result.get("pairs_closed")
            if type(finished) is not int or not 0 <= finished <= len(candidates):
                raise Conflict("Runner closed-pair count differs from snapshot")
            if code == 0 and result.get("state") == "completed" and finished != len(candidates):
                raise Conflict("Runner completed without all reviewed pairs")
            closed += finished
            if code == 0 and result.get("state") == "completed":
                _event(status_path, supervisor_id, "batch_closed", pairs_closed=closed,
                       runner_status_path=str(status_file))
                continue
            if code == 75 and result.get("state") == "awaiting_fresh_evidence":
                account = TransferAuthority.inspect_retained_account(
                    canonical, writer_id=writer_id, account_id=account_id)
                _account_gate(account, clock())
                _event(status_path, supervisor_id, "refreshing_evidence",
                       pairs_closed=closed, runner_status_path=str(status_file))
                continue  # New immutable snapshot/evidence directory; no replay.
            _event(status_path, supervisor_id, "stopped", pairs_closed=closed,
                   reason=result.get("stop_reason") or "runner_failed")
            return {"state": "stopped", "reason": result.get("stop_reason") or "runner_failed",
                    "pairs_closed": closed}
        _event(status_path, supervisor_id, "stopped", pairs_closed=closed,
               reason="cycle_bound_reached")
        return {"state": "stopped", "reason": "cycle_bound_reached",
                "pairs_closed": closed}
    except WindowClosing as error:
        _event(status_path, supervisor_id, "stopped", pairs_closed=closed,
               reason=error.reason)
        return {"state": "stopped", "reason": error.reason,
                "pairs_closed": closed}
    except BaseException as error:
        with contextlib.suppress(OSError, Conflict):
            _event(status_path, supervisor_id, "stopped", pairs_closed=closed,
                   reason=type(error).__name__)
        raise
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standing-envelope", type=Path, required=True)
    parser.add_argument("--standing-sha256", required=True)
    parser.add_argument("--authority-db", type=Path, required=True)
    parser.add_argument("--oauth-file", type=Path, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--writer-id", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_cycles(envelope_path=args.standing_envelope,
                            envelope_sha256=args.standing_sha256,
                            authority_db=args.authority_db, oauth_file=args.oauth_file,
                            account_id=args.account_id, writer_id=args.writer_id,
                            work_root=args.work_root)
        print(json.dumps({"event": "retained_continuous_result", **result}, sort_keys=True))
        return 0 if result["reason"] == "capture_stop_approaching" else 75
    except Exception as error:
        print(json.dumps({"event": "retained_continuous_stopped",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
