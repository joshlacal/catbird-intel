"""Separate, charged account authority for one reviewed receipt-only recovery.

The original retained run and HTTP 500 attempt are immutable. This authority
has its own short lease because the original run's 20-minute deadline cannot
be extended or replayed. It shares the canonical account pacing clocks.
No function here sends a network request or generates a new Drive file ID.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import stat
import time
import uuid
import urllib.parse
import re

from drive_transfer_authority import (
    AuthorityBlocked, PACING_SECONDS, QUOTA_COOLDOWN_SECONDS,
    WINDOW_SECONDS, _PROCESS_NONCE, _canonical, _connect, _finite_nonnegative,
    _required_identity, _valid_digest,
)
from retained_transfer_authority import (
    GLOBAL_ROLLING_CAP_BYTES, RetainedObservationAuthority, _boot_id,
    _reviewed_manifest,
)


MAX_RECOVERY_SECONDS = 1200
MAX_CONTROL_BYTES = 1 << 20
MAX_RECEIPT_BYTES = 1 << 20
RECOVERY_SCHEMA = "catbird.drive.receipt-recovery-review.v1"
_DRIVE_UPLOAD_URL = (
    "https://www.googleapis.com/upload/drive/v3/files"
    "?uploadType=resumable&fields=id"
)


def _operation_key(method: str, kind: str, target_sha: str,
                   content_range: str, body: bytes) -> str:
    return hashlib.sha256("\0".join((
        method, kind, target_sha, content_range,
        hashlib.sha256(body).hexdigest(),
    )).encode()).hexdigest()


def _exact_stage_bytes(path: str, size: int, expected_sha: str) -> bytes:
    target = Path(path)
    if not target.is_absolute() or target.is_symlink() or not _valid_digest(expected_sha):
        raise AuthorityBlocked("Pinned receipt stage path is invalid")
    before = target.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != size:
        raise AuthorityBlocked("Pinned receipt stage identity differs")
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
            opened.st_ctime_ns) != (before.st_dev, before.st_ino, before.st_size,
                                    before.st_mtime_ns, before.st_ctime_ns):
            raise AuthorityBlocked("Pinned receipt stage changed")
        data = os.read(fd, size + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    latest = target.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size,
                             item.st_mtime_ns, item.st_ctime_ns)
    if (len(data) != size or identity(before) != identity(after) or
        identity(before) != identity(latest) or
        hashlib.sha256(data).hexdigest() != expected_sha):
        raise AuthorityBlocked("Pinned receipt stage bytes differ")
    return data


def _ensure_schema(db: sqlite3.Connection) -> None:
    """Explicit administrator migration; ordinary inspection never creates it."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS retained_receipt_recovery (
            id TEXT PRIMARY KEY, original_run_id TEXT NOT NULL,
            original_attempt_id TEXT NOT NULL UNIQUE,
            pair_key TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
            receipt_id TEXT NOT NULL, session_sha256 TEXT NOT NULL,
            started_at REAL NOT NULL, deadline_at REAL NOT NULL,
            budget_bytes INTEGER NOT NULL, reserved_remaining INTEGER NOT NULL,
            evidence_json TEXT NOT NULL, process_host TEXT NOT NULL,
            process_pid INTEGER NOT NULL, process_nonce TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('open','closed','blocked')),
            probe_json TEXT, probe_sha256 TEXT,
            id_absence_json TEXT, id_absence_sha256 TEXT,
            new_session_url TEXT,
            new_session_sha256 TEXT, permit_key TEXT, permit_kind TEXT,
            permit_method TEXT, permit_target_sha256 TEXT,
            permit_bytes INTEGER, permit_used INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS retained_receipt_recovery_attempt (
            id TEXT PRIMARY KEY, recovery_id TEXT NOT NULL
                REFERENCES retained_receipt_recovery(id),
            at REAL NOT NULL, slot_at REAL NOT NULL, charge_through REAL NOT NULL,
            timeout_s REAL NOT NULL, dispatched_at REAL,
            dispatch_monotonic REAL, bytes INTEGER NOT NULL,
            kind TEXT NOT NULL, method TEXT NOT NULL,
            target_category TEXT NOT NULL, target_sha256 TEXT NOT NULL,
            operation_key TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('unresolved','settled')),
            http_status INTEGER
        );
        CREATE TABLE IF NOT EXISTS retained_reconciliation (
            attempt_id TEXT PRIMARY KEY REFERENCES retained_attempt(id),
            run_id TEXT NOT NULL REFERENCES retained_run(id),
            pair_key TEXT NOT NULL, proof_json TEXT NOT NULL,
            proof_sha256 TEXT NOT NULL, applied_at REAL NOT NULL
        );
    """)


class RecoveryAuthority:
    """SafeDrive-compatible request authority restricted to one pinned receipt."""

    def __init__(self, db: sqlite3.Connection, db_path: Path, recovery_id: str,
                 *, account_id: str, writer_id: str, clock=time.time,
                 monotonic_clock=time.monotonic):
        self.db = db
        self.db_path = str(db_path.resolve())
        self.release_id = recovery_id
        self.account_id = account_id
        self.writer_id = writer_id
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self.process_host = socket.gethostname()
        self.process_pid = os.getpid()
        self.process_nonce = _PROCESS_NONCE

    @classmethod
    def open_new(cls, db_path: Path, *, recovery_id: str, original_run_id: str,
                 original_attempt_id: str, account_id: str, writer_id: str,
                 receipt_id: str, session_url: str, evidence: dict,
                 deadline_at: float, clock=time.time,
                 monotonic_clock=time.monotonic) -> "RecoveryAuthority":
        """Reserve a new charged lease only after the old process is stopped.

        Evidence truth, including exclusion of other account writers, must be
        reviewed by the sole operator. This method validates its binding and
        keeps the previous run, attempt, bytes and HTTP status untouched.
        """
        for label, value in (("recovery ID", recovery_id),
                             ("original run ID", original_run_id),
                             ("attempt ID", original_attempt_id),
                             ("account ID", account_id), ("writer ID", writer_id),
                             ("receipt ID", receipt_id)):
            _required_identity(value, label)
        from gcs_drive_retained_receipt_transport import _session_url
        _session_url(session_url)
        now = _finite_nonnegative(clock(), "recovery clock")
        deadline = _finite_nonnegative(deadline_at, "recovery deadline")
        if not now < deadline <= now + MAX_RECOVERY_SECONDS:
            raise AuthorityBlocked("Receipt recovery exceeds 20-minute lease")
        if (not isinstance(evidence, dict) or
            evidence.get("schema") != RECOVERY_SCHEMA or
            evidence.get("account_id") != account_id or
            evidence.get("writer_id") != writer_id or
            evidence.get("original_run_id") != original_run_id or
            evidence.get("original_attempt_id") != original_attempt_id or
            evidence.get("receipt_id") != receipt_id or
            evidence.get("session_sha256") != hashlib.sha256(session_url.encode()).hexdigest() or
            evidence.get("all_writers_excluded") is not True or
            evidence.get("account_identity_verified") is not True or
            evidence.get("permitted_branch") not in (
                "probe_only", "partial_exact_suffix", "expired_same_id_restart",
                "receipt_only_auto", "payload_tail_and_receipt_auto") or
            evidence.get("receipt_stage_path") is None or
            not isinstance(evidence.get("folder_id"), str) or
            not evidence["folder_id"] or
            not isinstance(evidence.get("payload_id"), str) or
            not evidence["payload_id"] or
            not isinstance(evidence.get("receipt_name"), str) or
            not re.fullmatch(r"seg_[0-9a-z]{10}\.jss\.receipt\.json",
                             evidence["receipt_name"]) or
            evidence.get("exclusion_valid_until", 0) < deadline or
            not isinstance(evidence.get("owner_reference"), str) or
            not evidence["owner_reference"] or
            not now - 300 <= evidence.get("observed_at", -1) <= now):
            raise AuthorityBlocked("Receipt recovery review evidence is missing or stale")
        path = Path(db_path)
        if not path.is_file() or path.is_symlink():
            raise AuthorityBlocked("Canonical retained ledger is unavailable")
        # Full semantic inspection precedes the explicit migration.
        RetainedObservationAuthority.inspect_account(path, writer_id=writer_id,
                                                      account_id=account_id, clock=clock)
        db = _connect(path, create=False)
        authority = cls(db, path, recovery_id, account_id=account_id,
                        writer_id=writer_id, clock=clock,
                        monotonic_clock=monotonic_clock)
        try:
            _ensure_schema(db)
            db.execute("BEGIN IMMEDIATE")
            account = db.execute("SELECT * FROM retained_account WHERE id=1").fetchone()
            run = db.execute("SELECT * FROM retained_run WHERE id=?",
                             (original_run_id,)).fetchone()
            attempt = db.execute("SELECT * FROM retained_attempt WHERE id=?",
                                 (original_attempt_id,)).fetchone()
            if account is None or run is None or attempt is None:
                raise AuthorityBlocked("Receipt recovery ledger identity is absent")
            if account["boot_id"] is not None and account["boot_id"] != _boot_id():
                raise AuthorityBlocked("Retained boot changed before receipt recovery")
            manifest, digest, pair_key = _reviewed_manifest(
                run["manifest_raw"], run["id"], account_id)
            receipt = manifest["receipt"]
            payload = manifest["payload"]
            is_payload_tail = (
                evidence.get("permitted_branch") == "payload_tail_and_receipt_auto"
            )
            original_receipt_body = _exact_stage_bytes(
                evidence["receipt_stage_path"], receipt["bytes"],
                receipt["sha256"],
            )
            if is_payload_tail:
                if (not isinstance(evidence.get("payload_name"), str) or
                    not re.fullmatch(r"seg_[0-9a-z]{10}\.jss",
                                     evidence["payload_name"]) or
                    evidence["receipt_name"] !=
                        evidence["payload_name"] + ".receipt.json" or
                    not isinstance(evidence.get("payload_stage_path"), str) or
                    not evidence["payload_stage_path"].startswith("/") or
                    Path(evidence["payload_stage_path"]).name !=
                        evidence["payload_name"] or
                    evidence.get("payload_sha256") != payload["sha256"] or
                    not 0 < attempt["bytes"] <= min(8 << 20, payload["bytes"])):
                    raise AuthorityBlocked(
                        "Payload-tail recovery stage or size differs"
                    )
                payload_body = _exact_stage_bytes(
                    evidence["payload_stage_path"], payload["bytes"],
                    payload["sha256"],
                )
                payload_offset = payload["bytes"] - attempt["bytes"]
                if attempt["operation_key"] != _operation_key(
                    "PUT", "media_chunk", attempt["target_sha256"],
                    f"bytes {payload_offset}-{payload['bytes']-1}/{payload['bytes']}",
                    payload_body[payload_offset:],
                ):
                    raise AuthorityBlocked("Original payload-tail PUT operation differs")
                budget = (
                    payload["bytes"] + receipt["bytes"] + (2 * MAX_CONTROL_BYTES)
                )
                legacy_budget = (
                    attempt["bytes"] + receipt["bytes"] + MAX_CONTROL_BYTES
                )
                attempt_bytes_ok = True
            else:
                if attempt["operation_key"] != _operation_key(
                    "PUT", "media_chunk", attempt["target_sha256"],
                    f"bytes 0-{receipt['bytes']-1}/{receipt['bytes']}",
                    original_receipt_body,
                ):
                    raise AuthorityBlocked("Original receipt PUT operation differs")
                budget = receipt["bytes"] + MAX_CONTROL_BYTES
                legacy_budget = budget
                attempt_bytes_ok = attempt["bytes"] == receipt["bytes"]
            old_host, old_pid = account["process_host"], account["process_pid"]
            if old_host != authority.process_host or type(old_pid) is not int or old_pid <= 0:
                raise AuthorityBlocked("Prior writer process cannot be proven stopped")
            try:
                os.kill(old_pid, 0)
            except ProcessLookupError:
                pass
            except (OSError, PermissionError) as error:
                raise AuthorityBlocked("Prior writer process state is uncertain") from error
            else:
                raise AuthorityBlocked("Prior writer process may still run")
            open_recoveries = db.execute(
                "SELECT * FROM retained_receipt_recovery WHERE status!='closed'"
            ).fetchall()
            if (account["account_id"] != account_id or account["writer_id"] != writer_id or
                account["db_path"] != str(path.resolve()) or
                now < account["last_clock"] or account["cooldown_until"] > now or
                run["status"] != "active" or run["manifest_sha256"] != digest or
                run["pair_key"] != pair_key or
                attempt["run_id"] != original_run_id or
                attempt["status"] != "unresolved" or attempt["http_status"] != 500 or
                attempt["kind"] != "media_chunk" or attempt["method"] != "PUT" or
                not attempt_bytes_ok or
                attempt["target_sha256"] != hashlib.sha256(session_url.encode()).hexdigest() or
                receipt["bytes"] > MAX_RECEIPT_BYTES or
                evidence.get("manifest_sha256") != digest or
                evidence.get("pair_key") != pair_key or
                evidence.get("receipt_sha256") != receipt["sha256"] or
                not evidence.get("receipt_stage_path", "").startswith("/") or
                Path(evidence["receipt_stage_path"]).name !=
                    evidence["receipt_name"] or
                len(open_recoveries) > 1):
                raise AuthorityBlocked("Only the exact unresolved receipt may enter recovery")
            if len(open_recoveries) == 1:
                existing = open_recoveries[0]
                prior_attempts = db.execute(
                    "SELECT * FROM retained_receipt_recovery_attempt "
                    "WHERE recovery_id=? ORDER BY at,id", (existing["id"],),
                ).fetchall()
                probe_target_bytes = (
                    payload["bytes"] if is_payload_tail else receipt["bytes"]
                )
                expected_probe_key = _operation_key(
                    "PUT", "resumable_status", existing["session_sha256"],
                    f"bytes */{probe_target_bytes}", b"",
                )
                if (existing["status"] != "open" or
                    existing["original_run_id"] != original_run_id or
                    existing["original_attempt_id"] != original_attempt_id or
                    existing["pair_key"] != pair_key or
                    existing["manifest_sha256"] != digest or
                    existing["receipt_id"] != receipt_id or
                    existing["session_sha256"] != attempt["target_sha256"] or
                    existing["budget_bytes"] not in (budget, legacy_budget) or
                    existing["reserved_remaining"] != existing["budget_bytes"] or
                    existing["id_absence_json"] is not None or
                    existing["new_session_url"] is not None or
                    existing["permit_used"] != 1 or
                    existing["permit_key"] is not None or
                    existing["process_host"] != authority.process_host or
                    existing["process_pid"] != old_pid or
                    len(prior_attempts) != 1 or
                    prior_attempts[0]["kind"] != "resumable_status" or
                    prior_attempts[0]["method"] != "PUT" or
                    prior_attempts[0]["bytes"] != 0 or
                    prior_attempts[0]["target_sha256"] != existing["session_sha256"] or
                    prior_attempts[0]["operation_key"] != expected_probe_key or
                    prior_attempts[0]["dispatched_at"] is None or
                    prior_attempts[0]["http_status"] not in (404, 410)):
                    raise AuthorityBlocked("Existing receipt recovery lease cannot be adopted")
                authority.release_id = existing["id"]
                probe_record = {
                    "state": "expired", "acknowledged": 0,
                    "http_status": prior_attempts[0]["http_status"],
                    "session_sha256": existing["session_sha256"],
                    "observed_at": now,
                }
                probe_raw = _canonical(probe_record)
                probe_digest = hashlib.sha256(probe_raw.encode()).hexdigest()
                if prior_attempts[0]["status"] != "settled":
                    db.execute(
                        "UPDATE retained_receipt_recovery_attempt "
                        "SET status='settled',charge_through=MAX(charge_through,?) "
                        "WHERE id=?", (now, prior_attempts[0]["id"]),
                    )
                db.execute(
                    "UPDATE retained_receipt_recovery SET "
                    "started_at=MIN(started_at,?),deadline_at=?,evidence_json=?,"
                    "process_host=?,process_pid=?,process_nonce=?,"
                    "probe_json=?,probe_sha256=? WHERE id=?", (
                        now, min(existing["started_at"] + MAX_RECOVERY_SECONDS, deadline),
                        _canonical(evidence), authority.process_host,
                        authority.process_pid, authority.process_nonce,
                        probe_raw, probe_digest, existing["id"],
                    ),
                )
            else:
                rolling = db.execute(
                    "SELECT COALESCE(SUM(bytes),0) FROM retained_attempt "
                    "WHERE status='unresolved' OR charge_through>?",
                    (now - WINDOW_SECONDS,),
                ).fetchone()[0]
                rolling += db.execute(
                    "SELECT COALESCE(SUM(bytes),0) FROM retained_import "
                    "WHERE charge_through>?", (now - WINDOW_SECONDS,),
                ).fetchone()[0]
                rolling += db.execute(
                    "SELECT COALESCE(SUM(reserved_remaining),0) FROM retained_run "
                    "WHERE status IN ('active','verified')",
                ).fetchone()[0]
                if rolling + budget > GLOBAL_ROLLING_CAP_BYTES:
                    raise AuthorityBlocked("Recovery reservation exceeds account upload cap")
                db.execute("INSERT INTO retained_receipt_recovery "
                           "(id,original_run_id,original_attempt_id,pair_key,"
                           "manifest_sha256,receipt_id,session_sha256,started_at,"
                           "deadline_at,budget_bytes,reserved_remaining,evidence_json,"
                           "process_host,process_pid,process_nonce,status) "
                           "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    recovery_id, original_run_id, original_attempt_id, pair_key, digest,
                    receipt_id, hashlib.sha256(session_url.encode()).hexdigest(),
                    now, deadline, budget, budget, _canonical(evidence),
                    authority.process_host, authority.process_pid, authority.process_nonce,
                    "open",
                ))
            db.execute("UPDATE retained_account SET process_host=?,process_pid=?,"
                       "process_nonce=?,last_clock=? WHERE id=1", (
                           authority.process_host, authority.process_pid,
                           authority.process_nonce, now))
            db.execute("COMMIT")
            return authority
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            db.close()
            raise

    def close(self) -> None:
        self.db.close()

    @property
    def permitted_branch(self) -> str:
        row = self._recovery()
        return json.loads(row["evidence_json"])["permitted_branch"]

    def _recovery(self) -> sqlite3.Row:
        if os.getpid() != self.process_pid:
            raise AuthorityBlocked("Receipt recovery handle crossed a process fork")
        row = self.db.execute(
            "SELECT * FROM retained_receipt_recovery WHERE id=?", (self.release_id,)
        ).fetchone()
        if row is None or row["status"] != "open" or (
            row["process_host"], row["process_pid"], row["process_nonce"]
        ) != (self.process_host, self.process_pid, self.process_nonce):
            raise AuthorityBlocked("Receipt recovery lease is not active")
        return row

    def _account(self) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM retained_account WHERE id=1").fetchone()
        if (row is None or row["account_id"] != self.account_id or
            row["writer_id"] != self.writer_id or row["db_path"] != self.db_path or
            (row["process_host"], row["process_pid"], row["process_nonce"]) !=
            (self.process_host, self.process_pid, self.process_nonce)):
            raise AuthorityBlocked("Receipt recovery lost exclusive account claim")
        return row

    def _now(self, account: sqlite3.Row) -> float:
        now = _finite_nonnegative(self.clock(), "receipt recovery clock")
        if now < account["last_clock"]:
            raise AuthorityBlocked("Receipt recovery wall clock regressed")
        return now

    def _active(self, timeout_s: float = 0, *, slot_at: float | None = None
                ) -> tuple[sqlite3.Row, sqlite3.Row, float]:
        row, account = self._recovery(), self._account()
        now = self._now(account)
        timeout = _finite_nonnegative(timeout_s, "receipt recovery timeout")
        if (now >= row["deadline_at"] or
            max(now, slot_at if slot_at is not None else now) + timeout >=
                row["deadline_at"] or
            account["cooldown_until"] > now or
            json.loads(row["evidence_json"])["exclusion_valid_until"] <
                now + timeout):
            raise AuthorityBlocked("Receipt recovery lease or cooldown blocks request")
        return row, account, now

    def assert_active(self, timeout_s: float = 0, *, slot_at: float | None = None) -> None:
        _, _, now = self._active(timeout_s, slot_at=slot_at)
        if slot_at is not None and now < slot_at:
            raise AuthorityBlocked("Receipt recovery request slot has not arrived")

    def _probe_from_row(self, row: sqlite3.Row) -> dict | None:
        if row["probe_json"] is None:
            return None
        try:
            probe = json.loads(row["probe_json"])
        except (TypeError, ValueError) as error:
            raise AuthorityBlocked("Persisted receipt probe is malformed") from error
        if (hashlib.sha256(row["probe_json"].encode()).hexdigest() !=
            row["probe_sha256"]):
            raise AuthorityBlocked("Persisted receipt probe hash differs")
        return {**probe, "proof_sha256": row["probe_sha256"]}

    def probe_outcome(self) -> dict | None:
        return self._probe_from_row(self._recovery())

    def new_session(self) -> dict | None:
        row = self._recovery()
        url = row["new_session_url"]
        if url is None:
            return None
        if hashlib.sha256(url.encode()).hexdigest() != row["new_session_sha256"]:
            raise AuthorityBlocked("Private receipt session fingerprint changed")
        return {"url": url, "session_sha256": row["new_session_sha256"]}

    def prepare_request(self, url: str, method: str, kind: str,
                        data: bytes | None, headers: dict | None) -> None:
        """Pin exactly one next SafeDrive request before OAuth or dispatch."""
        body = data or b""
        if type(body) is not bytes or not isinstance(headers, dict):
            raise AuthorityBlocked("Receipt recovery request shape differs")
        content_range = next((value for key, value in headers.items()
                              if key.lower() == "content-range"), "")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row, _, now = self._active()
            evidence = json.loads(row["evidence_json"])
            old = self.db.execute("SELECT * FROM retained_run WHERE id=?",
                                  (row["original_run_id"],)).fetchone()
            old_attempt = self.db.execute(
                "SELECT * FROM retained_attempt WHERE id=?",
                (row["original_attempt_id"],),
            ).fetchone()
            if old is None or old["status"] != "active" or old_attempt is None:
                raise AuthorityBlocked("Original retained pair state changed")
            manifest, digest, key = _reviewed_manifest(
                old["manifest_raw"], old["id"], self.account_id)
            receipt = manifest["receipt"]
            payload = manifest["payload"]
            if (digest != row["manifest_sha256"] or key != row["pair_key"] or
                row["permit_used"] != 1 or row["permit_key"] is not None):
                raise AuthorityBlocked("Receipt recovery operation is already pending")
            old_sha = row["session_sha256"]
            probe = self._probe_from_row(row)
            is_payload_tail = (
                evidence.get("permitted_branch") == "payload_tail_and_receipt_auto"
            )
            if is_payload_tail:
                payload_offset = payload["bytes"] - old_attempt["bytes"]
                media_attempts = self.db.execute(
                    "SELECT * FROM retained_receipt_recovery_attempt "
                    "WHERE recovery_id=? AND kind='media_chunk' ORDER BY at,id",
                    (self.release_id,),
                ).fetchall()
                starts = self.db.execute(
                    "SELECT * FROM retained_receipt_recovery_attempt "
                    "WHERE recovery_id=? AND kind='resumable_start' ORDER BY at,id",
                    (self.release_id,),
                ).fetchall()
                absence = (
                    json.loads(row["id_absence_json"])
                    if row["id_absence_json"] is not None else None
                )
                url_sha = hashlib.sha256(url.encode()).hexdigest()
                if kind == "resumable_status":
                    if (probe is not None or method != "PUT" or body or
                        url_sha != old_sha or
                        self.db.execute(
                            "SELECT COUNT(*) FROM retained_receipt_recovery_attempt "
                            "WHERE recovery_id=? AND kind='resumable_status'",
                            (self.release_id,),
                        ).fetchone()[0] or
                        content_range != f"bytes */{payload['bytes']}"):
                        raise AuthorityBlocked(
                            "Only original payload-session empty status probe is allowed"
                        )
                elif kind == "media_chunk":
                    if (probe is None or method != "PUT" or
                        now - probe["observed_at"] > 300):
                        raise AuthorityBlocked(
                            "Payload-tail/receipt media continuation is not authorized"
                        )
                    if url_sha == old_sha:
                        if (probe["state"] != "partial" or
                            probe["acknowledged"] != payload_offset or
                            row["new_session_sha256"] is not None or
                            media_attempts):
                            raise AuthorityBlocked(
                                "Payload-tail media continuation state differs"
                            )
                        source = _exact_stage_bytes(
                            evidence["payload_stage_path"],
                            payload["bytes"], payload["sha256"],
                        )
                        if (content_range !=
                            f"bytes {payload_offset}-{payload['bytes']-1}/{payload['bytes']}" or
                            body != source[payload_offset:]):
                            raise AuthorityBlocked(
                                "Payload-tail continuation bytes or range differ"
                            )
                    elif (probe["state"] == "expired" and not media_attempts and
                          len(starts) == 1 and
                          isinstance(absence, dict) and
                          absence.get("payload_id") == evidence["payload_id"] and
                          "receipt_id" not in absence and
                          row["new_session_sha256"] is not None and
                          url_sha == row["new_session_sha256"]):
                        source = _exact_stage_bytes(
                            evidence["payload_stage_path"],
                            payload["bytes"], payload["sha256"],
                        )
                        if (content_range !=
                            f"bytes 0-{payload['bytes']-1}/{payload['bytes']}" or
                            body != source):
                            raise AuthorityBlocked(
                                "Payload full restart bytes or range differ"
                            )
                    else:
                        payload_done = (
                            (probe["state"] in ("complete", "expired") and
                             not media_attempts and len(starts) == 1) or
                            (probe["state"] == "partial" and
                             len(media_attempts) == 1 and len(starts) == 1 and
                             media_attempts[0]["status"] == "settled" and
                             media_attempts[0]["http_status"] in (200, 201) and
                             media_attempts[0]["target_sha256"] == old_sha) or
                            (probe["state"] == "expired" and
                             len(media_attempts) == 1 and len(starts) == 2 and
                             all(s["status"] == "settled" and
                                 s["http_status"] in (200, 201)
                                 for s in starts) and
                             media_attempts[0]["status"] == "settled" and
                             media_attempts[0]["http_status"] in (200, 201) and
                             media_attempts[0]["bytes"] == payload["bytes"] and
                             media_attempts[0]["target_sha256"] != old_sha)
                        )
                        if (not payload_done or
                            not isinstance(absence, dict) or
                            absence.get("receipt_id") != row["receipt_id"] or
                            row["new_session_sha256"] is None or
                            url_sha != row["new_session_sha256"]):
                            raise AuthorityBlocked(
                                "Receipt session after payload-tail recovery is not ready"
                            )
                        source = _exact_stage_bytes(
                            evidence["receipt_stage_path"],
                            receipt["bytes"], receipt["sha256"],
                        )
                        if (content_range !=
                            f"bytes 0-{receipt['bytes']-1}/{receipt['bytes']}" or
                            body != source):
                            raise AuthorityBlocked(
                                "Receipt media bytes or range differ"
                            )
                elif kind == "resumable_start":
                    if (probe is not None and probe["state"] == "expired" and
                        not starts and not media_attempts and
                        isinstance(absence, dict) and
                        absence.get("payload_id") == evidence["payload_id"] and
                        "receipt_id" not in absence and
                        row["new_session_sha256"] is None):
                        expected_body = _canonical({
                            "id": evidence["payload_id"],
                            "name": evidence["payload_name"],
                            "parents": [evidence["folder_id"]],
                        }).encode()
                        if (now - probe["observed_at"] > 300 or
                            method != "POST" or url != _DRIVE_UPLOAD_URL or
                            content_range or body != expected_body):
                            raise AuthorityBlocked(
                                "Same-ID payload session start differs"
                            )
                    else:
                        expected_body = _canonical({
                            "id": row["receipt_id"],
                            "name": evidence["receipt_name"],
                            "parents": [evidence["folder_id"]],
                        }).encode()
                        payload_done = (
                            probe is not None and (
                                (probe["state"] in ("complete", "expired") and
                                 not media_attempts and not starts) or
                                (probe["state"] == "partial" and
                                 len(media_attempts) == 1 and not starts and
                                 media_attempts[0]["status"] == "settled" and
                                 media_attempts[0]["http_status"] in (200, 201) and
                                 media_attempts[0]["target_sha256"] == old_sha) or
                                (probe["state"] == "expired" and
                                 len(media_attempts) == 1 and len(starts) == 1 and
                                 starts[0]["status"] == "settled" and
                                 starts[0]["http_status"] in (200, 201) and
                                 media_attempts[0]["status"] == "settled" and
                                 media_attempts[0]["http_status"] in (200, 201) and
                                 media_attempts[0]["bytes"] == payload["bytes"] and
                                 row["new_session_sha256"] is None and
                                 isinstance(absence, dict) and
                                 isinstance(absence.get("payload_absence"), dict))
                            )
                        )
                        if (not payload_done or
                            now - probe["observed_at"] > 300 or
                            not isinstance(absence, dict) or
                            absence.get("receipt_id") != row["receipt_id"] or
                            row["new_session_sha256"] is not None or
                            method != "POST" or url != _DRIVE_UPLOAD_URL or
                            content_range or body != expected_body):
                            raise AuthorityBlocked(
                                "Same-ID receipt session start after payload-tail differs"
                            )
                else:
                    raise AuthorityBlocked(
                        "Recovery may send only receipt/payload-tail status/start/media"
                    )
            elif kind == "resumable_status":
                if (probe is not None or method != "PUT" or body or
                    hashlib.sha256(url.encode()).hexdigest() != old_sha or
                    self.db.execute("SELECT COUNT(*) FROM retained_receipt_recovery_attempt "
                                    "WHERE recovery_id=? AND kind='resumable_status'",
                                    (self.release_id,)).fetchone()[0] or
                    content_range != f"bytes */{receipt['bytes']}"):
                    raise AuthorityBlocked("Only original-session empty status probe is allowed")
            elif kind == "media_chunk":
                if (probe is None or method != "PUT" or
                    evidence["permitted_branch"] == "probe_only" or
                    now - probe["observed_at"] > 60 or
                    self.db.execute("SELECT COUNT(*) FROM retained_receipt_recovery_attempt "
                                    "WHERE recovery_id=? AND kind='media_chunk'",
                                    (self.release_id,)).fetchone()[0]):
                    raise AuthorityBlocked("Receipt-only media continuation is not authorized")
                source = _exact_stage_bytes(evidence["receipt_stage_path"],
                                            receipt["bytes"], receipt["sha256"])
                if probe["state"] == "partial":
                    offset = probe["acknowledged"]
                    expected_url_sha = old_sha
                    if evidence["permitted_branch"] not in (
                        "partial_exact_suffix", "receipt_only_auto"):
                        raise AuthorityBlocked("Partial receipt branch differs")
                elif probe["state"] == "expired":
                    offset = 0
                    expected_url_sha = row["new_session_sha256"]
                    if (evidence["permitted_branch"] not in (
                        "expired_same_id_restart", "receipt_only_auto") or
                        expected_url_sha is None):
                        raise AuthorityBlocked("New receipt session was not persisted")
                else:
                    raise AuthorityBlocked("Completed receipt must not be resent")
                if (hashlib.sha256(url.encode()).hexdigest() != expected_url_sha or
                    content_range != f"bytes {offset}-{receipt['bytes']-1}/{receipt['bytes']}" or
                    body != source[offset:]):
                    raise AuthorityBlocked("Receipt continuation bytes or range differ")
            elif kind == "resumable_start":
                expected_body = _canonical({
                    "id": row["receipt_id"], "name": evidence["receipt_name"],
                    "parents": [evidence["folder_id"]],
                }).encode()
                if (probe is None or probe["state"] != "expired" or
                    now - probe["observed_at"] > 60 or
                    evidence["permitted_branch"] not in (
                        "expired_same_id_restart", "receipt_only_auto") or
                    row["id_absence_json"] is None or
                    row["new_session_sha256"] is not None or
                    method != "POST" or url != _DRIVE_UPLOAD_URL or
                    content_range or body != expected_body):
                    raise AuthorityBlocked("Same-ID receipt session start differs")
            else:
                raise AuthorityBlocked("Recovery may send only receipt status/start/media")
            target_sha = hashlib.sha256(url.encode()).hexdigest()
            key_hash = _operation_key(method, kind, target_sha,
                                      content_range, body)
            self.db.execute("UPDATE retained_receipt_recovery SET permit_key=?,"
                            "permit_kind=?,permit_method=?,permit_target_sha256=?,"
                            "permit_bytes=?,permit_used=0 WHERE id=?", (
                                key_hash, kind, method, target_sha, len(body),
                                self.release_id,
                            ))

    def start_request(self, body_bytes: int, timeout_s: float, kind: str, *,
                      method: str, target_category: str, target_sha256: str,
                      operation_key: str) -> tuple[str, float]:
        max_timeout = 300 if kind == "media_chunk" else 60
        if (type(body_bytes) is not int or body_bytes < 0 or
            not 0 < _finite_nonnegative(timeout_s, "request timeout") <= max_timeout or
            kind not in ("resumable_status", "resumable_start", "media_chunk") or
            target_category != kind or method not in ("PUT", "POST") or
            not _valid_digest(target_sha256) or not _valid_digest(operation_key)):
            raise AuthorityBlocked("Receipt recovery request is not permitted")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row, account, now = self._active(timeout_s)
            slot = max(now, account["next_slot"])
            self._active(timeout_s, slot_at=slot)
            if (row["permit_used"] != 0 or row["permit_key"] != operation_key or
                row["permit_kind"] != kind or row["permit_method"] != method or
                row["permit_target_sha256"] != target_sha256 or
                row["permit_bytes"] != body_bytes or
                body_bytes > row["reserved_remaining"]):
                raise AuthorityBlocked("Receipt recovery request differs from pinned operation")
            known = self.db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_attempt "
                "WHERE status='unresolved' OR charge_through>?",
                (now - WINDOW_SECONDS,),
            ).fetchone()[0]
            known += self.db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_receipt_recovery_attempt "
                "WHERE status='unresolved' OR charge_through>?",
                (now - WINDOW_SECONDS,),
            ).fetchone()[0]
            known += self.db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_import WHERE charge_through>?",
                (now - WINDOW_SECONDS,),
            ).fetchone()[0]
            reserved = self.db.execute(
                "SELECT COALESCE(SUM(reserved_remaining),0) FROM retained_run "
                "WHERE status IN ('active','verified')",
            ).fetchone()[0]
            reserved += self.db.execute(
                "SELECT COALESCE(SUM(reserved_remaining),0) "
                "FROM retained_receipt_recovery WHERE status='open'",
            ).fetchone()[0]
            if known + reserved > GLOBAL_ROLLING_CAP_BYTES:
                raise AuthorityBlocked("Receipt recovery exceeds account upload cap")
            attempt_id = uuid.uuid4().hex
            self.db.execute(
                "INSERT INTO retained_receipt_recovery_attempt VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    attempt_id, self.release_id, now, slot, slot + timeout_s,
                    timeout_s, None, None, body_bytes, kind, method, kind,
                    target_sha256, operation_key, "unresolved", None,
                ))
            self.db.execute(
                "UPDATE retained_receipt_recovery SET "
                "reserved_remaining=reserved_remaining-?,permit_key=NULL,"
                "permit_kind=NULL,permit_method=NULL,permit_target_sha256=NULL,"
                "permit_bytes=NULL,permit_used=1 WHERE id=?",
                (body_bytes, self.release_id),
            )
            self.db.execute(
                "UPDATE retained_account SET last_clock=?,next_slot=? WHERE id=1",
                (now, slot + PACING_SECONDS),
            )
            return attempt_id, slot

    def _pending_attempt(self, attempt_id: str) -> sqlite3.Row:
        _required_identity(attempt_id, "receipt recovery attempt ID")
        attempt = self.db.execute(
            "SELECT * FROM retained_receipt_recovery_attempt "
            "WHERE id=? AND recovery_id=?", (attempt_id, self.release_id),
        ).fetchone()
        if (attempt is None or attempt["status"] != "unresolved" or
            attempt["dispatched_at"] is not None):
            raise AuthorityBlocked("Receipt recovery attempt is not awaiting dispatch")
        return attempt

    def seconds_until_dispatch(self, attempt_id: str) -> float:
        attempt = self._pending_attempt(attempt_id)
        row, account, now = self._active(attempt["timeout_s"],
                                         slot_at=attempt["slot_at"])
        mono = _finite_nonnegative(self.monotonic_clock(), "monotonic clock")
        wall_wait = max(0.0, attempt["slot_at"] - now)
        if account["last_dispatch_wall"] is not None:
            wall_wait = max(wall_wait,
                            account["last_dispatch_wall"] + PACING_SECONDS - now)
        mono_wait = 0.0
        if account["last_dispatch_monotonic"] is not None:
            if mono < account["last_dispatch_monotonic"]:
                raise AuthorityBlocked("Receipt recovery monotonic clock regressed")
            mono_wait = max(
                0.0, account["last_dispatch_monotonic"] + PACING_SECONDS - mono,
            )
        wait = max(wall_wait, mono_wait)
        if now + wait + attempt["timeout_s"] >= row["deadline_at"]:
            raise AuthorityBlocked("Receipt recovery dispatch cannot fit lease")
        return wait

    def mark_dispatch(self, attempt_id: str) -> None:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            attempt = self._pending_attempt(attempt_id)
            _, account, now = self._active(attempt["timeout_s"],
                                            slot_at=attempt["slot_at"])
            mono = _finite_nonnegative(self.monotonic_clock(), "monotonic clock")
            if (now < attempt["slot_at"] or
                account["last_dispatch_wall"] is not None and
                now + 1e-9 < account["last_dispatch_wall"] + PACING_SECONDS or
                account["last_dispatch_monotonic"] is not None and
                mono + 1e-9 < account["last_dispatch_monotonic"] + PACING_SECONDS):
                raise AuthorityBlocked("Receipt recovery violates global request pacing")
            self.db.execute(
                "UPDATE retained_receipt_recovery_attempt SET "
                "dispatched_at=?,dispatch_monotonic=?,"
                "charge_through=MAX(charge_through,?) WHERE id=?",
                (now, mono, now + attempt["timeout_s"], attempt_id),
            )
            self.db.execute(
                "UPDATE retained_account SET last_clock=?,"
                "next_slot=MAX(next_slot,?),last_dispatch_wall=?,"
                "last_dispatch_monotonic=? WHERE id=1",
                (now, now + PACING_SECONDS, now, mono),
            )

    def finish_request(self, attempt_id: str, status_code: int | None) -> None:
        if status_code is not None and (
            type(status_code) is not int or not 100 <= status_code <= 599
        ):
            raise AuthorityBlocked("Receipt recovery HTTP status is invalid")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            _, account, now = self._active()
            attempt = self.db.execute(
                "SELECT * FROM retained_receipt_recovery_attempt "
                "WHERE id=? AND recovery_id=?", (attempt_id, self.release_id),
            ).fetchone()
            if (attempt is None or attempt["dispatched_at"] is None or
                attempt["http_status"] is not None):
                raise AuthorityBlocked("Receipt recovery attempt result changed")
            settled = (status_code == 308 or
                       status_code is not None and 200 <= status_code < 300 or
                       attempt["kind"] == "resumable_status" and status_code in (404, 410))
            self.db.execute(
                "UPDATE retained_receipt_recovery_attempt SET status=?,"
                "http_status=?,charge_through=MAX(charge_through,?) WHERE id=?",
                ("settled" if settled else "unresolved", status_code, now,
                 attempt_id),
            )
            if status_code in (403, 429):
                self.db.execute(
                    "UPDATE retained_account SET cooldown_until=MAX(cooldown_until,?) "
                    "WHERE id=1", (now + QUOTA_COOLDOWN_SECONDS,),
                )
            self.db.execute(
                "UPDATE retained_account SET last_clock=? WHERE id=1", (now,),
            )

    def record_probe(self, outcome: dict) -> str:
        """Append the exact first status result; never infer media success."""
        if not isinstance(outcome, dict):
            raise AuthorityBlocked("Receipt status proof is missing")
        state = outcome.get("state")
        http = outcome.get("http_status")
        offset = outcome.get("acknowledged")
        if (state, http) not in (
            ("complete", 200), ("complete", 201),
            ("partial", 308), ("expired", 404), ("expired", 410),
        ) or type(offset) is not int:
            raise AuthorityBlocked("Receipt status result is not terminal")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row, account, now = self._active()
            if row["probe_json"] is not None:
                raise AuthorityBlocked("Receipt status was already recorded")
            evidence = json.loads(row["evidence_json"])
            old = self.db.execute("SELECT manifest_raw FROM retained_run WHERE id=?",
                                  (row["original_run_id"],)).fetchone()
            old_attempt = self.db.execute(
                "SELECT * FROM retained_attempt WHERE id=?",
                (row["original_attempt_id"],),
            ).fetchone()
            if old is None or old_attempt is None:
                raise AuthorityBlocked("Original retained manifest disappeared")
            manifest, _, _ = _reviewed_manifest(
                old["manifest_raw"], row["original_run_id"], self.account_id)
            if evidence.get("permitted_branch") == "payload_tail_and_receipt_auto":
                size = manifest["payload"]["bytes"]
                expected_id = evidence["payload_id"]
                expected_offset = size - old_attempt["bytes"]
                if (state == "expired" and offset != 0 or
                    state == "complete" and offset != size or
                    state == "partial" and offset != expected_offset or
                    state == "complete" and outcome.get("metadata") !=
                        {"id": expected_id}):
                    raise AuthorityBlocked(
                        "Payload status offset or ID differs"
                    )
            else:
                size = manifest["receipt"]["bytes"]
                if (state == "complete" and offset != size or
                    state == "partial" and not 0 <= offset < size or
                    state == "expired" and offset != 0 or
                    state == "complete" and outcome.get("metadata") !=
                        {"id": row["receipt_id"]}):
                    raise AuthorityBlocked("Receipt status offset or ID differs")
            attempts = self.db.execute(
                "SELECT * FROM retained_receipt_recovery_attempt "
                "WHERE recovery_id=? ORDER BY at,id", (self.release_id,),
            ).fetchall()
            if (len(attempts) != 1 or attempts[0]["kind"] != "resumable_status" or
                attempts[0]["status"] != "settled" or
                attempts[0]["http_status"] != http or
                attempts[0]["bytes"] != 0):
                raise AuthorityBlocked("Receipt status lacks its charged request")
            record = {
                "state": state, "acknowledged": offset, "http_status": http,
                "session_sha256": row["session_sha256"], "observed_at": now,
            }
            raw = _canonical(record)
            digest = hashlib.sha256(raw.encode()).hexdigest()
            self.db.execute(
                "UPDATE retained_receipt_recovery SET probe_json=?,probe_sha256=? "
                "WHERE id=?", (raw, digest, self.release_id),
            )
            self.db.execute("UPDATE retained_account SET last_clock=? WHERE id=1",
                            (now,))
            return digest

    def record_id_absence(self, evidence: dict) -> str:
        if not isinstance(evidence, dict):
            raise AuthorityBlocked("Exact-ID absence evidence missing")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row, account, now = self._active()
            review = json.loads(row["evidence_json"])
            probe = self._probe_from_row(row)
            is_payload_tail = (
                review.get("permitted_branch") == "payload_tail_and_receipt_auto"
            )
            probe_ok = (
                probe is not None and (
                    probe["state"] == "expired" or
                    (is_payload_tail and probe["state"] in ("partial", "complete"))
                )
            )
            max_probe_age = 300 if is_payload_tail else 60
            common_ok = (
                probe_ok and
                now - probe["observed_at"] <= max_probe_age and
                evidence.get("folder_id") == review["folder_id"] and
                evidence.get("account_id") == self.account_id and
                evidence.get("http_status") == 404 and
                type(evidence.get("observed_at")) in (int, float) and
                0 <= now - evidence["observed_at"] <= 60 and
                isinstance(evidence.get("operator_reference"), str) and
                bool(evidence["operator_reference"])
            )
            if not common_ok:
                raise AuthorityBlocked("Exact-ID absence is not fresh or pinned")
            clear_session = False
            if (is_payload_tail and probe["state"] == "expired" and
                row["id_absence_json"] is None and
                evidence.get("payload_id") == review["payload_id"] and
                "receipt_id" not in evidence):
                record = dict(evidence)
            elif evidence.get("receipt_id") == row["receipt_id"]:
                if row["id_absence_json"] is None:
                    record = dict(evidence)
                elif is_payload_tail and probe["state"] == "expired":
                    prior = json.loads(row["id_absence_json"])
                    starts = self.db.execute(
                        "SELECT * FROM retained_receipt_recovery_attempt "
                        "WHERE recovery_id=? AND kind='resumable_start' ORDER BY at,id",
                        (self.release_id,),
                    ).fetchall()
                    media = self.db.execute(
                        "SELECT * FROM retained_receipt_recovery_attempt "
                        "WHERE recovery_id=? AND kind='media_chunk' ORDER BY at,id",
                        (self.release_id,),
                    ).fetchall()
                    if (not isinstance(prior, dict) or
                        prior.get("payload_id") != review["payload_id"] or
                        "receipt_id" in prior or
                        row["new_session_sha256"] is None or
                        len(starts) != 1 or len(media) != 1 or
                        starts[0]["status"] != "settled" or
                        starts[0]["http_status"] not in (200, 201) or
                        media[0]["status"] != "settled" or
                        media[0]["http_status"] not in (200, 201) or
                        media[0]["target_sha256"] != row["new_session_sha256"]):
                        raise AuthorityBlocked(
                            "Receipt absence requires completed payload restart"
                        )
                    record = {
                        **evidence,
                        "payload_absence": prior,
                        "payload_new_session_sha256": row["new_session_sha256"],
                    }
                    clear_session = True
                else:
                    raise AuthorityBlocked("Exact-ID absence was already recorded")
            else:
                raise AuthorityBlocked("Exact-ID absence is not fresh or pinned")
            raw = _canonical(record)
            digest = hashlib.sha256(raw.encode()).hexdigest()
            if clear_session:
                self.db.execute(
                    "UPDATE retained_receipt_recovery SET id_absence_json=?,"
                    "id_absence_sha256=?,new_session_url=NULL,"
                    "new_session_sha256=NULL WHERE id=?",
                    (raw, digest, self.release_id),
                )
            else:
                self.db.execute(
                    "UPDATE retained_receipt_recovery SET id_absence_json=?,"
                    "id_absence_sha256=? WHERE id=?",
                    (raw, digest, self.release_id),
                )
            self.db.execute("UPDATE retained_account SET last_clock=? WHERE id=1",
                            (now,))
            return digest

    def record_new_session(self, url: str) -> str:
        from gcs_drive_retained_receipt_transport import _session_url
        url = _session_url(url)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row, account, now = self._active()
            review = json.loads(row["evidence_json"])
            probe = self._probe_from_row(row)
            is_payload_tail = (
                review.get("permitted_branch") == "payload_tail_and_receipt_auto"
            )
            probe_ok = (
                probe is not None and (
                    probe["state"] == "expired" or
                    (is_payload_tail and probe["state"] in ("partial", "complete"))
                )
            )
            max_probe_age = 300 if is_payload_tail else 60
            digest = hashlib.sha256(url.encode()).hexdigest()
            if (not probe_ok or
                now - probe["observed_at"] > max_probe_age or
                row["id_absence_json"] is None or
                row["new_session_url"] is not None or
                digest == row["session_sha256"]):
                raise AuthorityBlocked("New receipt session is not fresh or distinct")
            absence = json.loads(row["id_absence_json"])
            expected_starts = (
                2 if (is_payload_tail and
                      isinstance(absence, dict) and
                      isinstance(absence.get("payload_absence"), dict))
                else 1
            )
            if (expected_starts == 2 and
                digest == absence.get("payload_new_session_sha256")):
                raise AuthorityBlocked("New receipt session reuses payload session")
            attempts = self.db.execute(
                "SELECT * FROM retained_receipt_recovery_attempt "
                "WHERE recovery_id=? AND kind='resumable_start' ORDER BY at,id",
                (self.release_id,),
            ).fetchall()
            if (len(attempts) != expected_starts or
                any(item["status"] != "settled" or
                    item["http_status"] not in (200, 201)
                    for item in attempts)):
                raise AuthorityBlocked("New receipt session lacks settled start")
            self.db.execute(
                "UPDATE retained_receipt_recovery SET new_session_url=?,"
                "new_session_sha256=? WHERE id=?", (url, digest, self.release_id),
            )
            self.db.execute("UPDATE retained_account SET last_clock=? WHERE id=1",
                            (now,))
            return digest

    def apply_verified_proof(self, proof: dict) -> str:
        """Resolve the *effective* uncertainty after full external readbacks.

        The original HTTP 500 row is never changed, and its bytes remain
        charged. This commits one append-only forensic proof and marks only
        the exact old run verified. Journal and GCS mapping remain separate
        crash-safe no-send steps.
        """
        if not isinstance(proof, dict):
            raise AuthorityBlocked("Full receipt recovery proof is missing")
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row, account, now = self._active()
            review = json.loads(row["evidence_json"])
            probe = self._probe_from_row(row)
            if (probe is None or
                proof.get("attempt_id") != row["original_attempt_id"] or
                proof.get("run_id") != row["original_run_id"] or
                proof.get("pair_key") != row["pair_key"] or
                proof.get("manifest_sha256") != row["manifest_sha256"] or
                proof.get("receipt_drive_id") != row["receipt_id"] or
                proof.get("account_id") != self.account_id or
                proof.get("writer_id") != self.writer_id or
                not _valid_digest(proof.get("mapping_sha256")) or
                type(proof.get("applied_at")) not in (int, float) or
                proof["applied_at"] > now or now - proof["applied_at"] > 60):
                raise AuthorityBlocked("Receipt recovery proof identity differs")
            original = self.db.execute(
                "SELECT * FROM retained_attempt WHERE id=?",
                (row["original_attempt_id"],),
            ).fetchone()
            run = self.db.execute(
                "SELECT * FROM retained_run WHERE id=?",
                (row["original_run_id"],),
            ).fetchone()
            if (original is None or run is None or
                original["status"] != "unresolved" or
                original["http_status"] != 500 or
                run["status"] != "active"):
                raise AuthorityBlocked("Original HTTP 500 state changed")
            attempts = self.db.execute(
                "SELECT * FROM retained_receipt_recovery_attempt "
                "WHERE recovery_id=? ORDER BY at,id", (self.release_id,),
            ).fetchall()
            if any(item["status"] != "settled" for item in attempts):
                raise AuthorityBlocked("Unresolved receipt recovery request blocks proof")
            if review.get("permitted_branch") == "payload_tail_and_receipt_auto":
                media = [item for item in attempts if item["kind"] == "media_chunk"]
                starts = [item for item in attempts if item["kind"] == "resumable_start"]
                absence = (
                    json.loads(row["id_absence_json"])
                    if row["id_absence_json"] is not None else None
                )
                has_payload_restart = (
                    isinstance(absence, dict) and
                    isinstance(absence.get("payload_absence"), dict)
                )
                expected_starts = 2 if has_payload_restart else 1
                expected_media_count = (
                    2 if (probe["state"] == "partial" or has_payload_restart)
                    else 1
                )
                if (probe["state"] not in ("partial", "complete", "expired") or
                    (has_payload_restart and probe["state"] != "expired") or
                    row["new_session_sha256"] is None or
                    not isinstance(absence, dict) or
                    absence.get("receipt_id") != row["receipt_id"] or
                    len(starts) != expected_starts or
                    any(item["http_status"] not in (200, 201) for item in starts) or
                    len(media) != expected_media_count or
                    any(item["http_status"] not in (200, 201) for item in media)):
                    raise AuthorityBlocked(
                        "Payload-tail and receipt completion results are absent"
                    )
            elif probe["state"] == "complete":
                if (len(attempts) != 1 or
                    attempts[0]["kind"] != "resumable_status" or
                    attempts[0]["http_status"] not in (200, 201)):
                    raise AuthorityBlocked("Completed original session result differs")
            else:
                media = [item for item in attempts if item["kind"] == "media_chunk"]
                if (len(media) != 1 or media[0]["http_status"] not in (200, 201) or
                    (probe["state"] == "expired" and
                     (row["new_session_sha256"] is None or
                      row["id_absence_json"] is None or
                      len([item for item in attempts if item["kind"] ==
                           "resumable_start"]) != 1))):
                    raise AuthorityBlocked("Receipt-only completion result is absent")
            raw = _canonical(proof)
            digest = hashlib.sha256(raw.encode()).hexdigest()
            self.db.execute(
                "INSERT INTO retained_reconciliation VALUES (?,?,?,?,?,?)", (
                    row["original_attempt_id"], row["original_run_id"],
                    row["pair_key"], raw, digest, proof["applied_at"],
                ))
            companion_probes = self.db.execute(
                "SELECT a.* FROM retained_attempt a "
                "LEFT JOIN retained_reconciliation r ON r.attempt_id=a.id "
                "WHERE a.run_id=? AND a.status='unresolved' "
                "AND a.kind='resumable_status' AND a.bytes=0 "
                "AND a.target_sha256=? "
                "AND a.http_status IN (404,410,500,502,503,504) "
                "AND r.attempt_id IS NULL",
                (row["original_run_id"], original["target_sha256"]),
            ).fetchall()
            for item in companion_probes:
                companion_proof = {
                    **proof,
                    "attempt_id": item["id"],
                    "parent_attempt_id": row["original_attempt_id"],
                    "original_http_status": item["http_status"],
                    "original_charge_bytes": 0,
                    "operation_key": item["operation_key"],
                }
                companion_raw = _canonical(companion_proof)
                companion_digest = hashlib.sha256(
                    companion_raw.encode()
                ).hexdigest()
                self.db.execute(
                    "INSERT INTO retained_reconciliation VALUES (?,?,?,?,?,?)", (
                        item["id"], row["original_run_id"], row["pair_key"],
                        companion_raw, companion_digest, proof["applied_at"],
                    ))
            manifest, _, _ = _reviewed_manifest(
                run["manifest_raw"], run["id"], self.account_id)
            validator = RetainedObservationAuthority(
                self.db, Path(self.db_path), run_id=run["id"],
                writer_id=self.writer_id, account_id=self.account_id,
                manifest=manifest, manifest_sha256=row["manifest_sha256"],
                pair_key=row["pair_key"], boot_id=account["boot_id"],
                clock=self.clock, monotonic_clock=self.monotonic_clock,
                recovery_only=True,
            )
            if row["original_attempt_id"] not in validator._reconciled_attempt_ids():
                raise AuthorityBlocked("Full-media reconciliation proof was not accepted")
            self.db.execute(
                "UPDATE retained_run SET status='verified',mapping_sha256=? "
                "WHERE id=?", (proof["mapping_sha256"], run["id"]),
            )
            self.db.execute(
                "UPDATE retained_receipt_recovery SET status='closed' WHERE id=?",
                (self.release_id,),
            )
            self.db.execute(
                "UPDATE retained_account SET last_clock=? WHERE id=1", (now,),
            )
            validator._validate_all()
            return digest
