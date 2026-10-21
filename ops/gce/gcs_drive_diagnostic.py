#!/usr/bin/env python3
"""One-pair Drive diagnostic in an isolated app-created folder; offline v3 draft.

This path never qualifies the archive shard, releases GCS, or cleans local files.
It requires an existing, independently evidenced transfer authority release.
No release initialization, retry loop, VM start, or OAuth scope change lives here.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import sys
import time
import urllib.error
import urllib.parse

from drive_transfer_authority import DIAGNOSTIC_QUOTA_OBSERVATION, TransferAuthority
from gcs_drive_drain import (Conflict, Drain, FOLDER, GCS, PILOT_PAYLOAD_LIMIT,
                             RECEIPT_LIMIT, SafeDrive, gce_region, matches_drive)

BUCKET = "catbird-436600-jetstream-buffer-20261020"
SOURCE_ID = "gce-selfhost-001"
SOURCE_EPOCH = "buffer-admission-20261020"
DRIVE_ROOT = "16Zd7n4pwTSDMDsnP9chX2Ci0r0ahEy45"
PAYLOAD = {
    "object": "sources/gce-selfhost-001/buffer-admission-20261020/segments/0000000e/"
              "seg_00000002u5.jss/5417c6724312db880c07bf1cb514e54e1d735d6d1ce347c45e283bd2f59457aa.jss",
    "generation": 1792518093053185,
    "bytes": 268980444,
    "sha256": "5417c6724312db880c07bf1cb514e54e1d735d6d1ce347c45e283bd2f59457aa",
}
RECEIPT = {
    "object": PAYLOAD["object"] + ".g1792518093053185.receipt.json",
    "generation": 1792518094289877,
    "bytes": 1093,
    "sha256": "fe4bc8db1aceb0c4868d9c1c82866925784b48be31db91d5301d3f1f51945c44",
}
CONTROL_ALLOWANCE = 1 << 20
TOTAL_CAP = PAYLOAD["bytes"] + RECEIPT["bytes"] + CONTROL_ALLOWANCE
MAX_DURATION_SECONDS = 20 * 60
WATCHDOG_MARGIN_SECONDS = 2
FINAL_STATUS = "diagnostic_verified_unqualified"


def build_diagnostic_manifest(*, release_id: str, account_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{32}", release_id):
        raise Conflict("Diagnostic release ID must be a unique lowercase UUID hex value")
    if not isinstance(account_id, str) or not account_id or len(account_id) > 256 or any(
            character.isspace() for character in account_id):
        raise Conflict("Diagnostic account ID is invalid")
    return {
        "schema": "catbird.drive.diagnostic.v1", "release_id": release_id,
        "account_id": account_id, "bucket": BUCKET, "source_id": SOURCE_ID,
        "source_epoch": SOURCE_EPOCH, "drive_root": DRIVE_ROOT,
        "payload": dict(PAYLOAD), "receipt": dict(RECEIPT),
        "folder_name": "catbird-drive-diagnostic-" + release_id,
        "control_allowance_bytes": CONTROL_ALLOWANCE,
        "max_charged_bytes": TOTAL_CAP, "max_duration_seconds": MAX_DURATION_SECONDS,
        "retain_gcs": True, "retain_local": True, "archive_qualified": False,
    }


def load_diagnostic_manifest(path: Path, *, release_id: str, account_id: str) -> tuple[dict, str]:
    """Bind the exact approved pair, isolated child name, and diagnostic limits."""
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema") != "catbird.drive.diagnostic.v1":
        raise Conflict("Unsupported diagnostic manifest")
    expected = build_diagnostic_manifest(release_id=release_id, account_id=account_id)
    if set(manifest) != set(expected):
        raise Conflict("Diagnostic manifest fields are not exact")
    for key, value in expected.items():
        if key == "max_duration_seconds":
            continue
        if type(value) is bool:
            matches = manifest.get(key) is value
        else:
            matches = manifest.get(key) == value and type(manifest.get(key)) is type(value)
        if not matches:
            raise Conflict("Diagnostic manifest mismatch: " + key)
    duration = manifest.get("max_duration_seconds")
    if type(duration) is not int or not 0 < duration <= MAX_DURATION_SECONDS:
        raise Conflict("Diagnostic duration exceeds 20 minutes")
    if "drive_parent" in manifest:
        raise Conflict("Diagnostic manifest must not name an archive shard parent")
    return manifest, hashlib.sha256(raw).hexdigest()


def _folder_matches(info: dict, *, folder_id: str, name: str, root: str) -> bool:
    return (info.get("id") == folder_id and info.get("name") == name
            and info.get("mimeType") == FOLDER and info.get("trashed") is False
            and info.get("parents") == [root])


def create_folder_by_pinned_id(drive, *, folder_id: str, name: str, root: str) -> dict:
    """One metered POST; the caller has already journaled the generated ID."""
    body = json.dumps({"id": folder_id, "name": name, "parents": [root],
                       "mimeType": FOLDER}, sort_keys=True, separators=(",", ":")).encode()
    url = "https://www.googleapis.com/drive/v3/files?fields=" + urllib.parse.quote(
        "id,name,mimeType,parents,trashed")
    with drive.request(url, "POST", body, {"Content-Type": "application/json"},
                       kind="diagnostic_folder_create") as response:
        return json.load(response)


class DriveHttpStop(RuntimeError):
    def __init__(self, status: int, reason: str | None):
        self.status = status
        self.reason = reason


class OAuthRefreshBlocked(RuntimeError):
    """Credential refresh failed before a metered Drive API dispatch."""


class _TokenProxy:
    def __init__(self, auth):
        self.auth = auth

    def token(self):
        try:
            return self.auth.token()
        except urllib.error.HTTPError as error:
            error.close()
            raise OAuthRefreshBlocked from None


def _bounded_drive_reason(error: urllib.error.HTTPError) -> str | None:
    try:
        body = json.loads(error.read(16 * 1024))
        detail = body.get("error", {})
        reasons = detail.get("errors", [])
        reason = reasons[0].get("reason") if isinstance(reasons, list) and reasons else None
        if not reason:
            reason = detail.get("status")
        if isinstance(reason, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", reason):
            return reason
    except (AttributeError, TypeError, ValueError, OSError):
        pass
    return None


class DiagnosticSafeDrive(SafeDrive):
    """Identify Drive API quota/permission responses without tagging OAuth/GCS."""

    def __init__(self, oauth: Path, authority: TransferAuthority):
        super().__init__(oauth, authority)
        self.auth = _TokenProxy(self.auth)

    def request(self, *args, **kwargs):
        try:
            return super().request(*args, **kwargs)
        except urllib.error.HTTPError as error:
            if error.code in (403, 429):
                reason = _bounded_drive_reason(error)
                error.close()
                raise DriveHttpStop(error.code, reason) from None
            raise


class DiagnosticDrain(Drain):
    """Exact source pair -> pinned diagnostic child IDs, with no archive promotion."""

    def __init__(self, cloud, drive, authority, work: Path, manifest: dict, manifest_sha256: str):
        super().__init__(cloud, drive, work, DRIVE_ROOT, SOURCE_ID, SOURCE_EPOCH,
                         max_segment_bytes=PILOT_PAYLOAD_LIMIT,
                         pilot_manifest=manifest, manifest_sha256=manifest_sha256,
                         authority=authority)

    def _safe_stage(self, key: str, name: str) -> tuple[Path, Path]:
        stage = self.work / "stage" / key
        if stage.is_symlink() or stage.parent.is_symlink():
            raise Conflict("Diagnostic staging directory is symlinked")
        stage.mkdir(parents=True, exist_ok=True)
        receipt_path, payload_path = stage / (name + ".receipt.json"), stage / name
        for path in (receipt_path, payload_path):
            if path.is_symlink() or path.with_name(path.name + ".partial").is_symlink():
                raise Conflict("Diagnostic staging artifact is symlinked")
        return receipt_path, payload_path

    def _verify_folder(self, entry: dict) -> None:
        folder_id = entry["drive"]["folder"]["id"]
        info = self.drive.metadata(folder_id)
        if not _folder_matches(info, folder_id=folder_id,
                               name=self.pilot_manifest["folder_name"], root=self.root):
            raise Conflict("Pinned diagnostic folder identity differs")

    def _pin_ids(self, entry: dict) -> None:
        for kind in ("folder", "payload", "receipt"):
            if not entry["drive"][kind].get("id"):
                entry["drive"][kind]["id"] = self.drive.reserve_id()
                self.save()  # Durable before any create/send.
        ids = [entry["drive"][kind]["id"] for kind in ("folder", "payload", "receipt")]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 3:
            raise Conflict("Diagnostic Drive IDs are missing or duplicated")
        entry["status"] = "ids_pinned"
        self.save()

    def _ensure_folder(self, entry: dict) -> None:
        status = entry["status"]
        if status == "folder_create_inflight":
            # A lost reply might mean the folder exists. Query only the pinned
            # ID; never POST again, allocate another ID, or infer from a name.
            try:
                self._verify_folder(entry)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    raise Conflict("Pinned diagnostic folder not visible after uncertain create; no retry") from error
                raise
            entry["status"] = "folder_reconciled_unsettled"
            self.save()
            raise Conflict("Pinned diagnostic folder exists, but uncertain create charge needs review")
        if status == "ids_pinned":
            entry["status"] = "folder_create_inflight"
            self.save()
            info = create_folder_by_pinned_id(
                self.drive, folder_id=entry["drive"]["folder"]["id"],
                name=self.pilot_manifest["folder_name"], root=self.root)
            if info.get("id") != entry["drive"]["folder"]["id"]:
                raise Conflict("Diagnostic folder create returned a different ID")
            entry["status"] = "folder_create_ack"
            self.save()
        if entry["status"] in ("folder_create_ack", "folder_verified"):
            self._verify_folder(entry)
            entry["status"] = "folder_verified"
            self.save()
            return
        raise Conflict("Diagnostic folder state requires manual reconciliation")

    def run_diagnostic(self) -> dict:
        manifest = self.pilot_manifest
        expected_receipt = manifest["receipt"]
        key = hashlib.sha256(f"{expected_receipt['object']}:{expected_receipt['generation']}".encode()).hexdigest()
        if set(self.state["segments"]) - {key}:
            raise Conflict("Diagnostic work directory contains another pair")
        entry = self.state["segments"].get(key)
        if entry:
            if (entry.get("diagnostic_release_id") != manifest["release_id"]
                    or entry.get("manifest_sha256") != self.manifest_sha256
                    or entry.get("archive_qualified") is not False):
                raise Conflict("Diagnostic journal identity differs")
            if entry.get("status") in ("file_upload_inflight", "folder_reconciled_unsettled", FINAL_STATUS):
                raise Conflict("Diagnostic replay requires review; no automatic retry")
            if entry.get("status") not in ("source_verified", "ids_pinned", "folder_create_inflight",
                                           "folder_create_ack", "folder_verified"):
                raise Conflict("Diagnostic journal state is not resumable")
        self.authority.assert_active(60)
        ref = self.cloud.lookup(expected_receipt["object"], expected_receipt["generation"])
        if (ref is None or ref.name != expected_receipt["object"]
                or ref.generation != expected_receipt["generation"]
                or ref.size != expected_receipt["bytes"]
                or ref.metadata.get("sha256") != expected_receipt["sha256"]):
            raise Conflict("Pinned diagnostic receipt metadata differs")
        name = ref.metadata.get("segment_name", "")
        if name != "seg_00000002u5.jss":
            raise Conflict("Pinned diagnostic segment name differs")
        receipt_path, payload_path = self._safe_stage(key, name)
        receipt_digest = self.download_checked(ref, receipt_path,
                                               expected_receipt["sha256"], RECEIPT_LIMIT)
        receipt, payload_ref = self.parse_receipt(ref, receipt_path)
        expected_payload = manifest["payload"]
        if (payload_ref.name != expected_payload["object"]
                or payload_ref.generation != expected_payload["generation"]
                or payload_ref.size != expected_payload["bytes"]
                or payload_ref.metadata.get("sha256") != expected_payload["sha256"]
                or receipt.get("sha256") != expected_payload["sha256"]
                or receipt.get("name") != name):
            raise Conflict("Pinned diagnostic payload differs")
        try:
            md5 = base64.b64decode(receipt["gcs"]["md5_hash"], validate=True)
        except (KeyError, ValueError) as error:
            raise Conflict("Diagnostic receipt payload MD5 is invalid") from error
        if len(md5) != 16:
            raise Conflict("Diagnostic receipt payload MD5 length differs")
        payload_digest = {"size": payload_ref.size, "sha256": expected_payload["sha256"],
                          "md5": md5.hex()}
        if self.download_checked(payload_ref, payload_path,
                                 expected_payload["sha256"], PILOT_PAYLOAD_LIMIT) != payload_digest:
            raise Conflict("Diagnostic payload bytes differ from receipt")
        self.authority.reserve_pair(key, payload_ref.size + ref.size,
                                    manifest["control_allowance_bytes"], {
            "manifest_sha256": self.manifest_sha256,
            "diagnostic_folder_name": manifest["folder_name"],
            "drive_root": self.root,
            "payload_object": payload_ref.name,
            "payload_generation": payload_ref.generation,
            "receipt_object": ref.name,
            "receipt_generation": ref.generation,
        })
        if entry is None:
            entry = {"key": key, "status": "source_verified", "archive_qualified": False,
                     "diagnostic_release_id": manifest["release_id"],
                     "manifest_sha256": self.manifest_sha256,
                     "payload_ref": vars(payload_ref), "receipt_ref": vars(ref),
                     "payload_name": name, "receipt_name": receipt_path.name,
                     "payload_digest": payload_digest, "receipt_digest": receipt_digest,
                     "drive": {"folder": {}, "payload": {}, "receipt": {}}}
            self.state["segments"][key] = entry
            self.save()
        elif (entry.get("payload_digest") != payload_digest
              or entry.get("receipt_digest") != receipt_digest
              or entry.get("payload_ref") != vars(payload_ref)
              or entry.get("receipt_ref") != vars(ref)):
            raise Conflict("Diagnostic source journal identity differs")
        self.drive.verify_account(self.authority.account_id)
        root_info = self.drive.metadata(self.root)
        if (root_info.get("id") != self.root or root_info.get("mimeType") != FOLDER
                or root_info.get("trashed") is not False):
            raise Conflict("Approved diagnostic root is not visible as a folder")
        if entry["status"] == "source_verified":
            self._pin_ids(entry)
        else:
            ids = [entry["drive"][kind].get("id") for kind in ("folder", "payload", "receipt")]
            if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 3:
                raise Conflict("Diagnostic journal lacks three pinned Drive IDs")
        self._ensure_folder(entry)
        folder_id = entry["drive"]["folder"]["id"]
        entry["status"] = "file_upload_inflight"
        self.save()  # Any interruption stops replay before another create/send.
        for kind, path in (("payload", payload_path), ("receipt", receipt_path)):
            expected = entry[kind + "_digest"]
            uploaded = self.drive.ensure_file(path, folder_id, expected,
                                              entry["drive"][kind], self.save)
            pinned_id = entry["drive"][kind]["id"]
            if uploaded.get("id") != pinned_id or not matches_drive(
                    uploaded, path.name, expected, folder_id):
                raise Conflict("Diagnostic upload differs from pinned ID or bytes")
            self.save()
        for kind in ("payload", "receipt"):
            pinned_id = entry["drive"][kind]["id"]
            info = self.drive.metadata(pinned_id)
            if info.get("id") != pinned_id or not matches_drive(
                    info, entry[kind + "_name"], entry[kind + "_digest"], folder_id):
                raise Conflict("Diagnostic pair cannot be verified by pinned IDs")
        self.authority.assert_active(0)
        # Complete only when every attempt is settled. An uncertain folder
        # creation remains charged and cannot be closed by this path.
        self.authority.close_pair(key)
        entry["status"] = FINAL_STATUS
        entry["archive_qualified"] = False
        self.save()
        return {"status": FINAL_STATUS, "archive_qualified": False,
                "folder_id": folder_id, "payload_id": entry["drive"]["payload"]["id"],
                "receipt_id": entry["drive"]["receipt"]["id"]}


def release_watchdog_cutoff(db_path: Path, release_id: str, duration: int) -> float:
    """Read the existing release deadline without opening or binding its writer lease.

    The child performs the full authority validation. This read merely gives the
    independent parent process a shorter monotonic kill timer for active I/O.
    """
    if type(duration) is not int or not 0 < duration <= MAX_DURATION_SECONDS:
        raise Conflict("Diagnostic watchdog duration is invalid")
    uri = "file:" + urllib.parse.quote(str(db_path.resolve()), safe="/") + "?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=1)) as db:
        row = db.execute("SELECT started_at,deadline_at FROM release WHERE id=?",
                         (release_id,)).fetchone()
    if row is None:
        raise Conflict("Diagnostic release deadline is unavailable")
    started, deadline = row
    monotonic_at_read = time.monotonic()
    now = time.time()
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in (started, deadline, now)):
        raise Conflict("Diagnostic release clock is invalid")
    if not 0 < deadline - started <= duration or not started <= now < deadline:
        raise Conflict("Diagnostic release duration or deadline differs")
    budget = min(deadline - now, float(duration)) - WATCHDOG_MARGIN_SECONDS
    if budget <= 0:
        raise Conflict("Diagnostic release has no safe watchdog budget")
    return monotonic_at_read + budget


def _arm_parent_death(parent_pid: int) -> None:
    """On GCE Linux, stop the worker if its supervisor is killed outright."""
    if not sys.platform.startswith("linux"):
        raise Conflict("Diagnostic process guard requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise Conflict("Linux parent-death signal could not be armed")
    if os.getppid() != parent_pid:
        raise Conflict("Diagnostic supervisor exited before guard was armed")


def _stop_worker_group(child: int, *, child_reaped: bool) -> None:
    """Stop even descendants if the worker forked, then reap the direct child."""
    group_error = None
    try:
        os.killpg(child, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        group_error = error
    # Also cover the tiny interval before the child calls setsid().
    if not child_reaped:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)
        with contextlib.suppress(ChildProcessError):
            os.waitpid(child, 0)
    if group_error is not None:
        raise Conflict("Diagnostic worker process group could not be stopped") from group_error


def supervise_worker(worker, cutoff_monotonic: float) -> int:
    """SIGKILL an in-flight worker before the absolute release deadline.

    The independent parent's monotonic clock keeps running during a trickling
    HTTP body or blocking SDK call. SIGKILL deliberately leaves local/GCS copies
    and any precharged but unacknowledged Drive request unresolved for review.
    """
    if not math.isfinite(cutoff_monotonic) or cutoff_monotonic <= time.monotonic():
        raise Conflict("Diagnostic watchdog cutoff is invalid or expired")
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt("Diagnostic supervisor interrupted")

    previous_term = signal.signal(signal.SIGTERM, interrupt)
    previous_hup = signal.signal(signal.SIGHUP, interrupt)
    child = None
    cleaned = False
    try:
        parent_pid = os.getpid()
        child = os.fork()
        if child == 0:
            try:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGHUP, signal.SIG_DFL)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
                os.setsid()  # Descendants share a worker-owned process group.
                _arm_parent_death(parent_pid)
                # Backup if the supervisor exits unexpectedly. The parent's
                # monotonic timer remains the hard stop for a trickling socket.
                signal.setitimer(signal.ITIMER_REAL,
                                 max(0.001, cutoff_monotonic - time.monotonic()))
                code = worker()
            except BaseException:
                os._exit(1)
            os._exit(code if isinstance(code, int) and 0 <= code <= 255 else 1)
        while True:
            if time.monotonic() >= cutoff_monotonic:
                cleaned = True
                _stop_worker_group(child, child_reaped=False)
                print(json.dumps({"event": "drive_diagnostic_watchdog_stop",
                                  "archive_qualified": False, "history_proven": False,
                                  "admission_mode": DIAGNOSTIC_QUOTA_OBSERVATION}),
                      file=sys.stderr, flush=True)
                return 124
            pid, status = os.waitpid(child, os.WNOHANG)
            if pid == child:
                cleaned = True
                _stop_worker_group(child, child_reaped=True)
                if time.monotonic() >= cutoff_monotonic:
                    print(json.dumps({"event": "drive_diagnostic_watchdog_stop",
                                      "archive_qualified": False, "history_proven": False,
                                      "admission_mode": DIAGNOSTIC_QUOTA_OBSERVATION}),
                          file=sys.stderr, flush=True)
                    return 124
                return os.waitstatus_to_exitcode(status)
            remaining = cutoff_monotonic - time.monotonic()
            if remaining <= 0:
                cleaned = True
                _stop_worker_group(child, child_reaped=False)
                print(json.dumps({"event": "drive_diagnostic_watchdog_stop",
                                  "archive_qualified": False, "history_proven": False,
                                  "admission_mode": DIAGNOSTIC_QUOTA_OBSERVATION}),
                      file=sys.stderr, flush=True)
                return 124
            time.sleep(min(remaining, 0.05))
    except BaseException:
        # The supervisor must not leave a transfer child running if interrupted.
        if child is not None and child > 0 and not cleaned:
            _stop_worker_group(child, child_reaped=False)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGHUP, previous_hup)


def _run_worker(args, manifest: dict, manifest_sha256: str) -> int:
    authority = TransferAuthority.open_existing(
        args.authority_db, release_id=args.release_id, writer_id=args.writer_id,
        account_id=args.account_id, manifest_sha256=manifest_sha256,
        allowed_mode=DIAGNOSTIC_QUOTA_OBSERVATION)
    if authority.admission_mode != DIAGNOSTIC_QUOTA_OBSERVATION:
        raise Conflict("Diagnostic authority is not in quota-observation mode")
    authority.validate_manifest_limits(manifest["max_charged_bytes"],
                                       manifest["max_duration_seconds"])
    authority.assert_active(300)
    region = gce_region()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / "diagnostic.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cloud = GCS(BUCKET)
        if region != "us-west1" or cloud.bucket.location.lower() != region:
            raise Conflict("GCE and bucket must both be single-region us-west1")
        for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
            with contextlib.suppress(FileNotFoundError, PermissionError):
                if b"gce_segment_offloader.py" in cmdline.read_bytes():
                    raise Conflict("Stop legacy Drive offloader before diagnostic")
        result = DiagnosticDrain(cloud, DiagnosticSafeDrive(args.oauth_file, authority), authority,
                                 args.work_dir, manifest, manifest_sha256).run_diagnostic()
        print(json.dumps({"event": "drive_diagnostic", **result,
                          "admission_mode": DIAGNOSTIC_QUOTA_OBSERVATION,
                          "history_proven": False}), flush=True)
    return 0


def _run_worker_guarded(args, manifest: dict, manifest_sha256: str) -> int:
    try:
        return _run_worker(args, manifest, manifest_sha256)
    except DriveHttpStop as error:
        message = {"event": "drive_http_stop", "http_status": error.status,
                   "archive_qualified": False, "history_proven": False,
                   "admission_mode": DIAGNOSTIC_QUOTA_OBSERVATION}
        if error.reason:
            message["drive_reason"] = error.reason
        print(json.dumps(message), flush=True)
        return 75
    except Exception as error:
        if isinstance(error, urllib.error.HTTPError):
            error.close()
        # Never print HTTP error text: it can include a resumable session token.
        print(json.dumps({"event": "drive_diagnostic_preserved", "error_type": type(error).__name__,
                          "archive_qualified": False, "history_proven": False,
                          "admission_mode": DIAGNOSTIC_QUOTA_OBSERVATION}),
              file=sys.stderr, flush=True)
        return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--authority-db", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--writer-id", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--oauth-file", type=Path,
                        default=Path("/root/.config/catbird-drive/oauth.json"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--retain-gcs", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--quota-observation", action="store_true")
    args = parser.parse_args(argv)
    if not args.once or not args.retain_gcs or not args.quota_observation:
        parser.error("Diagnostic requires --once, --retain-gcs, and explicit --quota-observation")
    if (not args.work_dir.resolve().is_relative_to(Path("/data"))
            or args.work_dir.name != "gcs-drive-diagnostic-" + args.release_id):
        parser.error("Diagnostic work directory must be a release-specific child of /data")
    try:
        manifest, manifest_sha256 = load_diagnostic_manifest(
            args.diagnostic_manifest, release_id=args.release_id, account_id=args.account_id)
        cutoff = release_watchdog_cutoff(args.authority_db, args.release_id,
                                         manifest["max_duration_seconds"])
        return supervise_worker(lambda: _run_worker_guarded(args, manifest, manifest_sha256), cutoff)
    except Exception as error:
        # Never print HTTP error text: it can include a resumable session token.
        print(json.dumps({"event": "drive_diagnostic_preserved", "error_type": type(error).__name__,
                          "archive_qualified": False, "history_proven": False,
                          "admission_mode": DIAGNOSTIC_QUOTA_OBSERVATION}),
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
