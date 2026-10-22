#!/usr/bin/env python3
"""Bounded, serial runner for reviewed retained GCS-to-Drive snapshots.

One child performs one retained pair. The shared authority, not this runner,
meters Drive requests and persists pacing, cooldown, and uncertain charges.
No failed child or nonterminal pair is automatically retried.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid

from drive_transfer_authority import TransferAuthority
from gcs_drive_drain import Conflict, PILOT_CHARGE_LIMIT
from gcs_drive_retained_overflow import (
    BUCKET, CAPTURE_RESTART_FLOOR_BYTES, CAPTURE_SEGMENTS_DIR, DRIVE_ROOT,
    FINAL_STATUS, _canonical, load_overflow_manifest, pair_key,
)
from retained_transfer_authority import GLOBAL_ROLLING_CAP_BYTES


COLLECTOR_STOP_AT = dt.datetime(2026, 10, 24, 10, 10, tzinfo=dt.timezone.utc).timestamp()
# Decoupled from collector stop so a downsized n2-standard-2 VM can drain the
# remaining GCS buffer backlog after the collector finishes.
STOP_AT = dt.datetime(2026, 10, 31, 10, 10, tzinfo=dt.timezone.utc).timestamp()
SCHEMA = "catbird.drive.retained-runner.v1"
SNAPSHOT_SCHEMA = "catbird.drive.retained-snapshot.v1"
RELEASE_SCHEMA = "catbird.drive.retained-cache-release.v1"
STAGE_RELEASE_APPROVAL_REF = "josh-authorized-2026-10-22-verified-local-stage-release"
MAX_PAIRS = 8
MAX_CONCURRENT_PAIRS = 4
DEFAULT_CONCURRENT_PAIRS = 3
MAX_RESERVED_BYTES = MAX_PAIRS * PILOT_CHARGE_LIMIT
MAX_STAGED_BYTES = 32 * (1 << 30)
DATA_ROOT = Path("/data")
MAPPING_RESERVATION_BYTES = 16 * 1024
CONTROL_STAGING_BYTES = 16 * (1 << 20)
STOP_MARGIN_SECONDS = 60
MIN_LAUNCH_SECONDS = 390
HEARTBEAT_SECONDS = 10
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID = re.compile(r"[0-9a-f]{32}\Z")


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Conflict("Runner JSON has duplicate keys")
        result[key] = value
    return result


def _bad_constant(_value):
    raise Conflict("Runner JSON has nonfinite number")


def _read_json(path: Path, limit: int) -> tuple[dict, bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise Conflict("Runner input is missing, symlinked, or too large")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object,
                           parse_constant=_bad_constant)
    except (ValueError, UnicodeError, TypeError) as error:
        raise Conflict("Runner JSON is malformed") from error
    if not isinstance(value, dict):
        raise Conflict("Runner JSON must be an object")
    return value, raw


def _under(path: Path, parent: Path) -> bool:
    return path.is_absolute() and path.parent == parent and not path.is_symlink()


def load_snapshot(path: Path, digest: str, snapshot_dir: Path,
                  account_id: str) -> tuple[dict, ...]:
    if not SHA256.fullmatch(digest):
        raise Conflict("Reviewed snapshot SHA-256 is malformed")
    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
        raise Conflict("Snapshot directory is missing or symlinked")
    snapshot, raw = _read_json(path, 128 * 1024)
    if hashlib.sha256(raw).hexdigest() != digest or snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise Conflict("Snapshot bytes or schema differ from reviewed packet")
    candidates = snapshot.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > MAX_PAIRS:
        raise Conflict("Snapshot pair count exceeds bound")
    total, keys, runs, sources, selected = 0, set(), set(), set(), []
    for item in candidates:
        if not isinstance(item, dict):
            raise Conflict("Snapshot candidate is malformed")
        key, run_id = item.get("pair_key"), item.get("release_id")
        claimed_hash, claimed_bytes = item.get("manifest_sha256"), item.get("reserved_bytes")
        if (not isinstance(key, str) or not SHA256.fullmatch(key)
                or not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id)
                or not isinstance(claimed_hash, str) or not SHA256.fullmatch(claimed_hash)
                or type(claimed_bytes) is not int or not 0 < claimed_bytes <= PILOT_CHARGE_LIMIT):
            raise Conflict("Snapshot candidate identity or charge is invalid")
        manifest_path = Path(item.get("manifest_path", ""))
        if not _under(manifest_path, snapshot_dir):
            raise Conflict("Snapshot manifest is outside exact snapshot directory")
        manifest, parsed_hash = load_overflow_manifest(
            manifest_path, release_id=run_id, account_id=account_id)
        source = (manifest["bucket"], manifest["payload"]["object"],
                  manifest["payload"]["generation"])
        if (parsed_hash != claimed_hash or pair_key(manifest) != key
                or manifest["max_charged_bytes"] != claimed_bytes
                or manifest["bucket"] != BUCKET or key in keys or run_id in runs
                or source in sources):
            raise Conflict("Snapshot manifest or duplicate source differs")
        keys.add(key)
        runs.add(run_id)
        sources.add(source)
        total += claimed_bytes
        if total > MAX_RESERVED_BYTES:
            raise Conflict("Snapshot cumulative reservation exceeds bound")
        selected.append({"pair_key": key, "run_id": run_id, "manifest_path": manifest_path,
                         "manifest_sha256": claimed_hash, "manifest": manifest,
                         "reserved_bytes": claimed_bytes, "source": source})
    return tuple(selected)


def _stage_bytes(data_root: Path) -> int:
    """Count retained work trees without following links; fail on uninspectable trees."""
    total, visited = 0, 0
    def unreadable(_error):
        raise Conflict("Retained staging tree is unreadable")
    for root in data_root.glob("gcs-drive-retained-*"):
        if root.is_symlink() or not root.is_dir():
            raise Conflict("Retained staging root is not a directory")
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=unreadable):
            visited += len(dirs) + len(files)
            if visited > 100_000:
                raise Conflict("Retained staging scan exceeded bound")
            for name in dirs:
                if (Path(directory) / name).is_symlink():
                    raise Conflict("Retained staging tree contains symlink")
            for name in files:
                meta = (Path(directory) / name).lstat()
                if not stat.S_ISREG(meta.st_mode):
                    raise Conflict("Retained staging tree contains nonregular file")
                total += meta.st_size
                if total > MAX_STAGED_BYTES:
                    raise Conflict("Retained staging ceiling reached")
    return total


def check_disk(candidate: dict, *, data_root: Path, capture_dir: Path,
               max_staged_bytes: int = MAX_STAGED_BYTES) -> dict:
    if (type(max_staged_bytes) is not int or not 0 < max_staged_bytes <= MAX_STAGED_BYTES
            or data_root.is_symlink() or not data_root.is_dir()
            or capture_dir.is_symlink() or not capture_dir.is_dir()
            or os.stat(data_root).st_dev != os.stat(capture_dir).st_dev):
        raise Conflict("Runner capture and stage filesystem identity differs")
    size = _stage_bytes(data_root)
    manifest = candidate["manifest"]
    reserve = 2 * (manifest["payload"]["bytes"] + manifest["receipt"]["bytes"])
    reserve += CONTROL_STAGING_BYTES
    free = shutil.disk_usage(capture_dir).free
    if (size + reserve > max_staged_bytes or
            free - reserve < CAPTURE_RESTART_FLOOR_BYTES):
        raise Conflict("Retained staging would breach capture restart headroom or ceiling")
    return {"staged_bytes": size, "reserved_stage_bytes": reserve,
            "free_before_bytes": free, "free_after_reservation_bytes": free - reserve}


def check_disk_wave(wave: list[dict] | tuple[dict, ...], *, data_root: Path,
                    capture_dir: Path, max_staged_bytes: int = MAX_STAGED_BYTES) -> dict:
    first = check_disk(wave[0], data_root=data_root, capture_dir=capture_dir,
                       max_staged_bytes=max_staged_bytes)
    if len(wave) == 1 or "staged_bytes" not in first or "free_before_bytes" not in first:
        return first
    size = first["staged_bytes"]
    free = first["free_before_bytes"]
    reserve = sum(
        2 * (item["manifest"]["payload"]["bytes"] + item["manifest"]["receipt"]["bytes"])
        + CONTROL_STAGING_BYTES
        for item in wave
    )
    if (size + reserve > max_staged_bytes or
            free - reserve < CAPTURE_RESTART_FLOOR_BYTES):
        raise Conflict("Retained staging would breach capture restart headroom or ceiling")
    return {"staged_bytes": size, "reserved_stage_bytes": reserve,
            "free_before_bytes": free, "free_after_reservation_bytes": free - reserve}


def _account_gate(account: dict, now: float) -> None:
    if (account.get("diagnostic_imported") is not True
            or account.get("admission_mode") != "retained_pair_observation"
            or account.get("history_proven") is not False
            or account.get("nonterminal_run_id") is not None
            or account.get("nonterminal_status") is not None
            or account.get("unresolved_attempts") != 0
            or account.get("mapping_partition_remaining_bytes", -1) < MAPPING_RESERVATION_BYTES
            or account.get("cooldown_until", math.inf) > now
            or account.get("last_clock", math.inf) > now):
        raise Conflict("Retained account has unresolved, cooldown, partition, or clock stop")


def _candidate_gate(candidate: dict, rows: tuple[dict, ...]) -> None:
    for row in rows:
        if row.get("status") not in ("closed", "diagnostic_verified_unqualified"):
            raise Conflict("Earlier retained pair is nonterminal")
        source = (row.get("bucket"), row.get("payload_object"), row.get("payload_generation"))
        if (row.get("run_id") == candidate["run_id"]
                or row.get("pair_key") == candidate["pair_key"]
                or source == candidate["source"]):
            raise Conflict("Snapshot source was already admitted; refresh snapshot")


def _evidence(path: Path, candidate: dict, account_id: str, writer_id: str,
              deadline: float, now: float) -> dict:
    evidence, _ = _read_json(path, 32 * 1024)
    expected = {"run_id": candidate["run_id"], "writer_id": writer_id,
                "account_id": account_id,
                "reviewed_manifest_sha256": candidate["manifest_sha256"],
                "history_basis": "unproven", "account_identity_verified": True,
                "all_writers_excluded": True, "quota_rejection_accepted": True}
    if (any(evidence.get(key) != value for key, value in expected.items())
            or not SHA256.fullmatch(str(evidence.get("evidence_sha256", "")))
            or type(evidence.get("observed_at")) not in (int, float)
            or type(evidence.get("exclusion_valid_until")) not in (int, float)
            or not math.isfinite(evidence["observed_at"])
            or not math.isfinite(evidence["exclusion_valid_until"])
            or not 0 <= now - evidence["observed_at"] <= 300
            or evidence["exclusion_valid_until"] < deadline):
        raise Conflict("Fresh owner evidence does not cover this exact pair and deadline")
    return evidence


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_release_evidence(data_root: Path, run_id: str, record: dict) -> None:
    root = data_root / "catbird-retained-cache-evidence"
    if root.is_symlink():
        raise Conflict("Cache evidence directory is symlinked")
    existed = root.exists()
    root.mkdir(mode=0o700, exist_ok=True)
    if not existed:
        _fsync_dir(data_root)
    log_path = root / f"release-{run_id}.jsonl"
    if log_path.is_symlink():
        raise Conflict("Cache release evidence log is symlinked")
    raw = _canonical(record) + b"\n"
    fd = os.open(log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        if os.write(fd, raw) != len(raw):
            raise Conflict("Cache release evidence append was short")
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(root)


def release_closed_pair_stage(
    run_id: str,
    pair_key_str: str,
    mapping_sha256: str,
    *,
    data_root: Path,
    approval_reference: str = STAGE_RELEASE_APPROVAL_REF,
) -> dict:
    """Release verified local stage duplicates for a durably closed pair.

    Requires:
    - Exact 32-hex run_id, 64-hex pair_key, and 64-hex mapping_sha256 from closed SQLite row.
    - Nonblocking exclusive lock on work/overflow.lock.
    - Terminal journal.json (`offloaded_retained_unqualified`) with matching mapping_sha256,
      index_generation > 0, and 3 unique pinned Drive IDs.
    - Stage files match journal payload_digest and receipt_digest (size, sha256, md5, single link).
    - Appends durable intent and completion evidence outside /data/gcs-drive-retained-*.
    """
    if (not RUN_ID.fullmatch(str(run_id or ""))
            or not SHA256.fullmatch(str(pair_key_str or ""))
            or not SHA256.fullmatch(str(mapping_sha256 or ""))):
        raise Conflict("Closed stage release identity is malformed")
    work = data_root / ("gcs-drive-retained-" + run_id)
    stage = work / "stage" / pair_key_str
    journal_path = work / "journal.json"
    lock_path = work / "overflow.lock"
    if not stage.exists() or not journal_path.exists() or not lock_path.exists():
        return {"run_id": run_id, "pair_key": pair_key_str, "released": [],
                "bytes_released": 0, "state": "already_absent"}
    if work.is_symlink() or (work / "stage").is_symlink() or stage.is_symlink():
        raise Conflict("Closed stage release directory is symlinked")
    existing_names = [entry.name for entry in stage.iterdir()]
    if not existing_names:
        return {"run_id": run_id, "pair_key": pair_key_str, "released": [],
                "bytes_released": 0, "state": "already_empty"}
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise Conflict("Retained worker or recovery owns the stage") from error
        journal, _ = _read_json(journal_path, 1 << 20)
        if (journal.get("bucket") != BUCKET or journal.get("root") != DRIVE_ROOT
                or not isinstance(journal.get("segments"), dict)):
            raise Conflict("Closed stage journal namespace differs")
        entry = journal["segments"].get(pair_key_str)
        if (not isinstance(entry, dict) or entry.get("key") != pair_key_str
                or entry.get("release_id") != run_id
                or entry.get("status") != FINAL_STATUS
                or entry.get("mapping_sha256") != mapping_sha256
                or type(entry.get("index_generation")) is not int
                or entry["index_generation"] <= 0):
            raise Conflict("Closed stage journal is not terminal with verified mapping")
        ids = {kind: entry.get("drive", {}).get(kind, {}).get("id")
               for kind in ("folder", "payload", "receipt")}
        if any(not isinstance(val, str) or not val for val in ids.values()) or len(set(ids.values())) != 3:
            raise Conflict("Closed stage journal lacks three unique Drive IDs")
        payload_name = entry.get("payload_name") or Path(entry["payload_ref"]["name"]).parent.name
        receipt_name = entry.get("receipt_name") or (payload_name + ".receipt.json")
        expected_map = {"payload": payload_name, "receipt": receipt_name}
        if not set(existing_names).issubset(set(expected_map.values())):
            raise Conflict("Closed stage contains unexpected file")
        stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        released = []
        bytes_released = 0
        release_id = hashlib.sha256(f"{run_id}:{pair_key_str}:{mapping_sha256}".encode("ascii")).hexdigest()
        try:
            _append_release_evidence(data_root, run_id, {
                "schema": RELEASE_SCHEMA, "event": "cache_release_start",
                "release_id": release_id, "approval_reference": approval_reference,
                "run_id": run_id, "pair_key": pair_key_str,
                "mapping_sha256": mapping_sha256,
                "index_generation": entry["index_generation"],
                "drive_ids": ids, "at": time.time(),
            })
            for kind in ("payload", "receipt"):
                fname = expected_map[kind]
                if fname not in existing_names:
                    continue
                digest = entry.get(kind + "_digest")
                if not isinstance(digest, dict):
                    raise Conflict("Closed stage journal lacks media digest")
                file_fd = os.open(fname, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=stage_fd)
                try:
                    st = os.fstat(file_fd)
                    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size != digest.get("size"):
                        raise Conflict("Closed stage file is not a matching single-link regular file")
                    sha = hashlib.sha256()
                    md5 = hashlib.md5(usedforsecurity=False)
                    count = 0
                    while block := os.read(file_fd, 1 << 20):
                        sha.update(block)
                        md5.update(block)
                        count += len(block)
                    if (count != digest.get("size")
                            or sha.hexdigest() != digest.get("sha256")
                            or md5.hexdigest() != digest.get("md5")):
                        raise Conflict("Closed stage file bytes differ from verified journal digest")
                finally:
                    os.close(file_fd)
                _append_release_evidence(data_root, run_id, {
                    "schema": RELEASE_SCHEMA, "event": "cache_release_file_intent",
                    "release_id": release_id, "kind": kind,
                    "path": str(stage / fname), "inode": st.st_ino,
                    "bytes": count, "sha256": digest["sha256"], "md5": digest["md5"],
                    "at": time.time(),
                })
                os.unlink(fname, dir_fd=stage_fd)
                os.fsync(stage_fd)
                _append_release_evidence(data_root, run_id, {
                    "schema": RELEASE_SCHEMA, "event": "cache_release_file_removed",
                    "release_id": release_id, "kind": kind,
                    "path": str(stage / fname), "bytes": count,
                    "at": time.time(),
                })
                released.append(kind)
                bytes_released += count
            _append_release_evidence(data_root, run_id, {
                "schema": RELEASE_SCHEMA, "event": "cache_release_complete",
                "release_id": release_id, "released": released,
                "bytes_released": bytes_released,
                "creation_retained_local": True,
                "current_cache_state": "released", "at": time.time(),
            })
        finally:
            os.close(stage_fd)
        return {"run_id": run_id, "pair_key": pair_key_str, "released": released,
                "bytes_released": bytes_released, "state": "released"}
    finally:
        os.close(lock_fd)


def reclaim_closed_stages(
    rows: tuple[dict, ...],
    *,
    data_root: Path,
    approval_reference: str = STAGE_RELEASE_APPROVAL_REF,
) -> dict:
    """Release local stage files for all closed pairs in the ledger before disk check."""
    pairs_released = 0
    bytes_released = 0
    for row in rows:
        if row.get("status") != "closed":
            continue
        r_id = row.get("run_id")
        p_key = row.get("pair_key")
        m_sha = row.get("mapping_sha256")
        if (isinstance(r_id, str) and RUN_ID.fullmatch(r_id)
                and isinstance(p_key, str) and SHA256.fullmatch(p_key)
                and isinstance(m_sha, str) and SHA256.fullmatch(m_sha)):
            res = release_closed_pair_stage(
                r_id, p_key, m_sha, data_root=data_root,
                approval_reference=approval_reference)
            if res["released"]:
                pairs_released += 1
                bytes_released += res["bytes_released"]
    return {"pairs_released": pairs_released, "bytes_released": bytes_released}


class Status:
    def __init__(self, path: Path, run_id: str, snapshot_sha256: str):
        if path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
            raise Conflict("Runner status directory is unavailable or symlinked")
        self.path = path
        self.base = {"schema": SCHEMA, "run_id": run_id,
                     "snapshot_sha256": snapshot_sha256, "state": "starting",
                     "pairs_closed": 0, "charged_bytes_observed": None,
                     "last_pair": None, "last_mapping_sha256": None,
                     "cooldown_until": None, "stop_reason": None}
        self.update()

    def update(self, **changes) -> None:
        self.base.update(changes)
        self.base["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.base["updated_epoch"] = time.time()
        self.base["heartbeat_age_seconds_at_write"] = 0
        raw = b"\n" + json.dumps(self.base, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        created = not self.path.exists()
        fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            if os.write(fd, raw) != len(raw):
                raise Conflict("Runner status append was short")
            os.fsync(fd)
        finally:
            os.close(fd)
        if created:
            _fsync_dir(self.path.parent)


def _descendants(pid: int) -> set[int]:
    found, pending = set(), [pid]
    while pending and len(found) < 1024:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        try:
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            children = []
        pending.extend(int(child) for child in children)
    if pending:
        raise Conflict("Runner child tree exceeded kill bound")
    return found


def _arm_parent_death(expected_parent_pid: int) -> None:
    """Executed in the Linux child before exec; lost runner kills supervisor."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        os._exit(127)
    # The parent can die between fork and prctl; refuse to exec in that case.
    if os.getppid() != expected_parent_pid:
        os._exit(127)


def _kill_child_tree(child: subprocess.Popen) -> None:
    try:
        descendants = _descendants(child.pid)
    except Conflict:
        # At minimum stop the original session even if /proc is malformed.
        descendants = {child.pid}
    groups = set()
    for pid in descendants:
        with contextlib.suppress(ProcessLookupError):
            groups.add(os.getpgid(pid))
    groups.discard(os.getpgrp())
    for group in groups:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
    for pid in descendants:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        child.wait(timeout=2)
    if child.poll() is None:
        raise Conflict("Runner could not reap timed-out worker")


def _run_child(command: list[str], *, deadline: float, status: Status,
               log_path: Path, runner_stop_monotonic: float,
               capture_dir: Path, inflight_allowance: int,
               stop_at: float | None = None) -> int:
    effective_stop = STOP_AT if stop_at is None else stop_at
    if not sys.platform.startswith("linux"):
        raise Conflict("Retained runner watchdog requires Linux")
    if log_path.exists() or log_path.is_symlink():
        raise Conflict("Runner output log already exists")
    with open(log_path, "xb", buffering=0) as output:
        parent_pid = os.getpid()
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True,
                                 preexec_fn=lambda: _arm_parent_death(parent_pid))
        try:
            cutoff = min(runner_stop_monotonic - STOP_MARGIN_SECONDS / 2,
                         time.monotonic() + max(0, deadline - time.time()) - 5)
            status.update(state="transferring", child_pid=child.pid)
            while child.poll() is None:
                if time.monotonic() >= cutoff or time.time() >= effective_stop - 30:
                    _kill_child_tree(child)
                    raise Conflict("Retained child exceeded pair or capture STOP bound")
                try:
                    free = shutil.disk_usage(capture_dir).free
                except OSError as error:
                    _kill_child_tree(child)
                    raise Conflict("Capture filesystem became unavailable") from error
                if free - inflight_allowance < CAPTURE_RESTART_FLOOR_BYTES:
                    _kill_child_tree(child)
                    raise Conflict("In-flight retained transfer would breach capture free floor")
                status.update(capture_free_bytes=free)
                time.sleep(max(0, min(HEARTBEAT_SECONDS, cutoff - time.monotonic())))
            return child.returncode
        except BaseException:
            if child.poll() is None:
                _kill_child_tree(child)
            raise
        finally:
            status.update(child_pid=None)


def _run_children_concurrent(
    wave_tasks: list[dict], *, deadline: float, status: Status,
    runner_stop_monotonic: float, capture_dir: Path,
    stop_at: float | None = None, child_runner=_run_child,
) -> list[int]:
    if len(wave_tasks) == 1:
        task = wave_tasks[0]
        return [child_runner(
            task["command"], deadline=deadline, status=status,
            log_path=task["log_path"], runner_stop_monotonic=runner_stop_monotonic,
            capture_dir=capture_dir, inflight_allowance=task["inflight_allowance"],
        )]
    if child_runner is not _run_child:
        with ThreadPoolExecutor(max_workers=len(wave_tasks)) as pool:
            futures = [
                pool.submit(
                    child_runner, task["command"], deadline=deadline, status=status,
                    log_path=task["log_path"], runner_stop_monotonic=runner_stop_monotonic,
                    capture_dir=capture_dir, inflight_allowance=task["inflight_allowance"],
                )
                for task in wave_tasks
            ]
            return [f.result() for f in futures]
    effective_stop = STOP_AT if stop_at is None else stop_at
    if not sys.platform.startswith("linux"):
        raise Conflict("Retained runner watchdog requires Linux")
    for task in wave_tasks:
        if task["log_path"].exists() or task["log_path"].is_symlink():
            raise Conflict("Runner output log already exists")
    outputs = []
    children = []
    parent_pid = os.getpid()
    try:
        for task in wave_tasks:
            out = open(task["log_path"], "xb", buffering=0)
            outputs.append(out)
            child = subprocess.Popen(
                task["command"], stdout=out, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
                preexec_fn=lambda: _arm_parent_death(parent_pid),
            )
            children.append(child)
        cutoff = min(runner_stop_monotonic - STOP_MARGIN_SECONDS / 2,
                     time.monotonic() + max(0, deadline - time.time()) - 5)
        status.update(state="transferring", child_pid=children[0].pid,
                      child_pids=[c.pid for c in children])
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None, 0) for c in children):
                for c in children:
                    if c.poll() is None:
                        _kill_child_tree(c)
                break
            if time.monotonic() >= cutoff or time.time() >= effective_stop - 30:
                for c in children:
                    if c.poll() is None:
                        _kill_child_tree(c)
                raise Conflict("Retained child exceeded pair or capture STOP bound")
            try:
                free = shutil.disk_usage(capture_dir).free
            except OSError as error:
                for c in children:
                    if c.poll() is None:
                        _kill_child_tree(c)
                raise Conflict("Capture filesystem became unavailable") from error
            active_inflight = sum(
                task["inflight_allowance"]
                for task, c in zip(wave_tasks, children)
                if c.poll() is None
            )
            if free - active_inflight < CAPTURE_RESTART_FLOOR_BYTES:
                for c in children:
                    if c.poll() is None:
                        _kill_child_tree(c)
                raise Conflict("In-flight retained transfer would breach capture free floor")
            status.update(capture_free_bytes=free)
            time.sleep(max(0, min(HEARTBEAT_SECONDS, cutoff - time.monotonic())))
        return [c.returncode if c.returncode is not None else 1 for c in children]
    except BaseException:
        for c in children:
            if c.poll() is None:
                _kill_child_tree(c)
        raise
    finally:
        for out in outputs:
            with contextlib.suppress(OSError):
                out.close()
        status.update(child_pid=None, child_pids=None)


def _child_stop_reason(code: int, log_path: Path) -> str:
    reason = "pair_exit_" + str(code)
    if code == 75:
        with contextlib.suppress(OSError, ValueError, Conflict):
            if log_path.stat().st_size <= 64 * 1024:
                for line in log_path.read_text().splitlines()[-4:]:
                    record = json.loads(line, object_pairs_hook=_strict_object,
                                        parse_constant=_bad_constant)
                    if (isinstance(record, dict) and
                            record.get("event") == "overflow_drive_http_stop" and
                            record.get("http_status") in (403, 429)):
                        reason = "drive_http_" + str(record["http_status"])
    return reason


def run_snapshot(*, snapshot_json: Path, snapshot_sha256: str, snapshot_dir: Path,
                 authority_db: Path, evidence_dir: Path, account_id: str,
                 writer_id: str, oauth_file: Path, status_file: Path,
                 data_root: Path = Path("/data"), capture_dir: Path = CAPTURE_SEGMENTS_DIR,
                 max_staged_bytes: int = MAX_STAGED_BYTES,
                 max_concurrent_pairs: int = 1,
                 stop_at: float | None = None,
                 child_runner=_run_child) -> dict:
    if not sys.platform.startswith("linux"):
        raise Conflict("Retained runner requires Linux")
    if (type(max_concurrent_pairs) is not int or
            not 1 <= max_concurrent_pairs <= MAX_CONCURRENT_PAIRS):
        raise Conflict("Runner max_concurrent_pairs must be 1..4")
    effective_stop = STOP_AT if stop_at is None else stop_at
    if type(effective_stop) not in (int, float) or not math.isfinite(effective_stop) or effective_stop <= 0:
        raise Conflict("Runner stop_at epoch is invalid")
    if (not account_id or not writer_id or evidence_dir.is_symlink()
            or not evidence_dir.is_dir() or data_root.resolve() != DATA_ROOT):
        raise Conflict("Runner identity, evidence directory, or data root differs")
    if oauth_file.is_symlink() or not oauth_file.is_file():
        raise Conflict("Existing OAuth file is unavailable")
    candidates = load_snapshot(snapshot_json, snapshot_sha256, snapshot_dir, account_id)
    if not candidates:
        raise Conflict("Empty current-live snapshot is not historical coverage proof")
    seconds = effective_stop - time.time()
    if seconds < MIN_LAUNCH_SECONDS:
        raise Conflict("Capture STOP has insufficient time for another retained pair")
    monotonic_stop = time.monotonic() + seconds
    runner_id = uuid.uuid4().hex
    if authority_db.is_symlink() or not authority_db.is_file():
        raise Conflict("Canonical retained authority ledger is unavailable")
    canonical_db = authority_db.resolve(strict=True)
    lock_path = canonical_db.with_name(canonical_db.name + ".runner.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = Status(status_file, runner_id, snapshot_sha256)
        try:
            index = 0
            while index < len(candidates):
                wave = list(candidates[index:index + max_concurrent_pairs])
                now = time.time()
                duration = max(c["manifest"]["max_duration_seconds"] for c in wave)
                required_window = max(MIN_LAUNCH_SECONDS, duration + STOP_MARGIN_SECONDS + 30)
                if (effective_stop - now < required_window
                        or monotonic_stop - time.monotonic() < required_window):
                    status.update(state="stopped", stop_reason="capture_stop_approaching")
                    break
                account = TransferAuthority.inspect_retained_account(
                    authority_db, writer_id=writer_id, account_id=account_id)
                _account_gate(account, now)
                if account.get("mapping_partition_remaining_bytes", -1) < MAPPING_RESERVATION_BYTES * len(wave):
                    raise Conflict("Retained account has unresolved, cooldown, partition, or clock stop")
                known_rolling = account.get("known_local_rolling_bytes", 0)
                while len(wave) > 1 and known_rolling + sum(c["reserved_bytes"] for c in wave) > GLOBAL_ROLLING_CAP_BYTES:
                    wave.pop()
                if known_rolling + sum(c["reserved_bytes"] for c in wave) > GLOBAL_ROLLING_CAP_BYTES:
                    raise Conflict("Retained pair would exceed 700 GB rolling cap")
                rows = TransferAuthority.list_retained_pairs(
                    authority_db, writer_id=writer_id, account_id=account_id)
                for candidate in wave:
                    _candidate_gate(candidate, rows)
                reclaim_closed_stages(rows, data_root=data_root)
                disk = check_disk_wave(wave, data_root=data_root, capture_dir=capture_dir,
                                       max_staged_bytes=max_staged_bytes)
                deadline = min(now + duration, effective_stop - STOP_MARGIN_SECONDS,
                               now + monotonic_stop - time.monotonic() - STOP_MARGIN_SECONDS)
                if deadline - now <= 330:
                    raise Conflict("Insufficient reviewed transfer window before capture STOP")
                wave_tasks = []
                missing_evidence_pair = None
                for candidate in wave:
                    review_path = evidence_dir / (candidate["run_id"] + ".json")
                    if not _under(review_path, evidence_dir):
                        raise Conflict("Fresh evidence path differs")
                    try:
                        _evidence(review_path, candidate, account_id, writer_id, deadline, now)
                    except Conflict:
                        missing_evidence_pair = candidate["pair_key"]
                        break
                    run_work = data_root / ("gcs-drive-retained-" + candidate["run_id"])
                    log_path = data_root / ("retained-run-" + runner_id + "-" + candidate["run_id"] + ".log")
                    command = [sys.executable, "-B", str(Path(__file__).with_name("gcs_drive_retained_overflow.py")),
                               "--overflow-manifest", str(candidate["manifest_path"]),
                               "--run-id", candidate["run_id"], "--account-id", account_id,
                               "--writer-id", writer_id, "--authority-db", str(authority_db),
                               "--review-evidence", str(review_path),
                               "--deadline-epoch", str(deadline), "--work-dir", str(run_work),
                               "--oauth-file", str(oauth_file), "--once", "--retain-gcs",
                               "--quota-observation"]
                    if max_concurrent_pairs > 1:
                        command.extend(["--max-concurrent-runs", str(max_concurrent_pairs)])
                    wave_tasks.append({
                        "candidate": candidate,
                        "command": command,
                        "log_path": log_path,
                        "inflight_allowance": (
                            candidate["manifest"]["payload"]["bytes"]
                            + candidate["manifest"]["receipt"]["bytes"]
                            + CONTROL_STAGING_BYTES
                        ),
                    })
                if not wave_tasks:
                    status.update(state="awaiting_fresh_evidence",
                                  stop_reason="fresh_pair_evidence_missing",
                                  last_pair=missing_evidence_pair)
                    break
                status.update(state="admitting_pair",
                              last_pair=wave_tasks[-1]["candidate"]["pair_key"],
                              child_deadline_epoch=deadline, disk=disk,
                              cooldown_until=account["cooldown_until"],
                              charged_bytes_observed=account["known_local_rolling_bytes"])
                codes = _run_children_concurrent(
                    wave_tasks, deadline=deadline, status=status,
                    runner_stop_monotonic=monotonic_stop, capture_dir=capture_dir,
                    stop_at=effective_stop, child_runner=child_runner,
                )
                for task, code in zip(wave_tasks, codes):
                    if code != 0:
                        reason = _child_stop_reason(code, task["log_path"])
                        status.update(state="stopped", stop_reason=reason)
                        raise Conflict("Retained pair exited without verified closure")
                account_after = TransferAuthority.inspect_retained_account(
                    authority_db, writer_id=writer_id, account_id=account_id)
                _account_gate(account_after, time.time())
                for task in wave_tasks:
                    candidate = task["candidate"]
                    row = TransferAuthority.inspect_retained_pair(
                        authority_db, writer_id=writer_id, account_id=account_id,
                        run_id=candidate["run_id"], pair_key=candidate["pair_key"],
                        manifest_sha256=candidate["manifest_sha256"])
                    if (row.get("status") != "closed"
                            or not SHA256.fullmatch(str(row.get("mapping_sha256") or ""))):
                        raise Conflict("Child exited without durable shared mapping commit")
                    release_closed_pair_stage(
                        candidate["run_id"], candidate["pair_key"], row["mapping_sha256"],
                        data_root=data_root)
                    status.update(state="pair_closed", pairs_closed=status.base["pairs_closed"] + 1,
                                  last_pair=candidate["pair_key"],
                                  last_mapping_sha256=row["mapping_sha256"],
                                  charged_bytes_observed=account_after["known_local_rolling_bytes"],
                                  cooldown_until=account_after["cooldown_until"])
                if missing_evidence_pair is not None:
                    status.update(state="awaiting_fresh_evidence",
                                  stop_reason="fresh_pair_evidence_missing",
                                  last_pair=missing_evidence_pair)
                    break
                index += len(wave_tasks)
            else:
                status.update(state="completed", stop_reason="snapshot_exhausted")
        except BaseException as error:
            status.update(state="stopped", stop_reason=status.base.get("stop_reason") or type(error).__name__)
            raise
        return dict(status.base)
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-json", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--authority-db", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--writer-id", required=True)
    parser.add_argument("--oauth-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, default=Path("/data/gcs-buffer/retained_runner_status.jsonl"))
    parser.add_argument("--max-staged-bytes", type=int, default=MAX_STAGED_BYTES)
    parser.add_argument("--max-concurrent-pairs", type=int,
                        default=DEFAULT_CONCURRENT_PAIRS, choices=(1, 2, 3, 4))
    parser.add_argument("--stop-at-epoch", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        result = run_snapshot(snapshot_json=args.snapshot_json,
                              snapshot_sha256=args.snapshot_sha256, snapshot_dir=args.snapshot_dir,
                              authority_db=args.authority_db, evidence_dir=args.evidence_dir,
                              account_id=args.account_id, writer_id=args.writer_id,
                              oauth_file=args.oauth_file, status_file=args.status_file,
                              max_staged_bytes=args.max_staged_bytes,
                              max_concurrent_pairs=args.max_concurrent_pairs,
                              stop_at=args.stop_at_epoch)
        print(json.dumps({"event": "retained_runner_result", "state": result["state"],
                          "pairs_closed": result["pairs_closed"],
                          "stop_reason": result["stop_reason"]}, sort_keys=True))
        return 0 if result["state"] == "completed" else 75
    except Exception as error:
        print(json.dumps({"event": "retained_runner_stopped",
                          "error_type": type(error).__name__}), file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
