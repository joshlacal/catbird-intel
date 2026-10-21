"""Offline-first, fail-closed authority for one bounded GCS-to-Drive pilot.

This is a review draft, not a source of Google account upload history. An
operator must independently establish account identity, complete recent-use
history (or account-wide quiescence), and exclusion of every other writer.
``initialize_release`` records that evidence; ``open_existing`` never creates
an empty history or transfers an expired lease to another writer.

The default 700 GB decimal planning bound is below the cited Workspace
750 GB decimal reference, but that reference's applicability to a personal
account is unverified. Provider rejection remains authoritative.

The first release deliberately supports one writer, one pair and one local
SQLite file. Separate hosts or accounts must use a genuinely shared authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import socket
import sys
import time
import uuid
from typing import Callable, Iterable

ROLLING_CAP_BYTES = 700 * (1 << 30)
# Conservative reference only: the Workspace 750 GB/day figure is decimal.
# Its applicability to the account in use must be established separately.
PROVIDER_REFERENCE_BYTES = 750_000_000_000
DEFAULT_PROVIDER_BOUND_BYTES = 700_000_000_000
PILOT_CAP_BYTES = 320 * (1 << 20)
MAX_PAYLOAD_BYTES = 300 * (1 << 20)
WINDOW_SECONDS = 24 * 60 * 60
PACING_SECONDS = 0.55
MAX_RUN_SECONDS = 20 * 60
QUOTA_COOLDOWN_SECONDS = 60 * 60
SCHEMA_VERSION = 2
_PROCESS_NONCE = uuid.uuid4().hex
STRICT_HISTORY = "strict_history"
DIAGNOSTIC_QUOTA_OBSERVATION = "diagnostic_quota_observation"
DIAGNOSTIC_PAIR_CAP_BYTES = 270_030_113
DIAGNOSTIC_PAYLOAD_BYTES = 268_980_444
DIAGNOSTIC_RECEIPT_BYTES = 1_093
DIAGNOSTIC_CONTROL_BYTES = 1 << 20
DIAGNOSTIC_PAYLOAD_OBJECT = (
    "sources/gce-selfhost-001/buffer-admission-20261020/segments/0000000e/"
    "seg_00000002u5.jss/5417c6724312db880c07bf1cb514e54e1d735d6d1ce347c45e283bd2f59457aa.jss"
)
DIAGNOSTIC_RECEIPT_OBJECT = DIAGNOSTIC_PAYLOAD_OBJECT + ".g1792518093053185.receipt.json"
DIAGNOSTIC_PAIR_KEY = hashlib.sha256(
    f"{DIAGNOSTIC_RECEIPT_OBJECT}:1792518094289877".encode()).hexdigest()


class AuthorityBlocked(RuntimeError):
    """The caller has no authority to start another Drive request."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _required_identity(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or any(c.isspace() for c in value):
        raise AuthorityBlocked(f"Invalid {field}")
    return value


def _valid_digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuthorityBlocked(f"Invalid {field}")
    number = float(value)
    if not (0 <= number < float("inf")):
        raise AuthorityBlocked(f"Invalid {field}")
    return number


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AuthorityBlocked(f"Invalid {field}")
    return value


def _validate_evidence(evidence: dict, account_id: str, now: float, deadline_at: float,
                       provider_bound_bytes: int) -> None:
    """Check packet shape; truth of external evidence must be reviewed outside this module."""
    if not isinstance(evidence, dict) or evidence.get("account_id") != account_id:
        raise AuthorityBlocked("Account-wide provenance is missing or mismatched")
    if evidence.get("account_identity_verified") is not True or evidence.get("all_writers_excluded") is not True:
        raise AuthorityBlocked("Account identity or writer exclusion is unproven")
    if evidence.get("provider_bound_bytes") != provider_bound_bytes or evidence.get("provider_bound_basis") not in (
        "account_verified", "conservative_workspace_reference"
    ):
        raise AuthorityBlocked("Provider-bound provenance is missing or mismatched")
    if (evidence["provider_bound_basis"] == "conservative_workspace_reference" and
        provider_bound_bytes > DEFAULT_PROVIDER_BOUND_BYTES):
        raise AuthorityBlocked("Unverified provider bound exceeds planning margin")
    if _finite_nonnegative(evidence.get("exclusion_valid_until"), "exclusion end") < deadline_at:
        raise AuthorityBlocked("Writer exclusion does not cover the release")
    observed = _finite_nonnegative(evidence.get("observed_at"), "evidence observation time")
    if observed > now or now - observed > 300:
        raise AuthorityBlocked("Release evidence is not fresh")
    basis = evidence.get("history_basis")
    if basis == "reconciled":
        if not all(evidence.get(k) is True for k in (
            "all_writers_included", "all_attempts_included", "uncertain_attempts_included"
        )):
            raise AuthorityBlocked("Recent upload history is incomplete")
    elif basis == "quiescence":
        quiet_since = _finite_nonnegative(evidence.get("quiet_since"), "quiet since")
        last_send = _finite_nonnegative(evidence.get("latest_possible_send_at"), "latest possible send")
        if quiet_since > now - WINDOW_SECONDS or last_send > quiet_since:
            raise AuthorityBlocked("Account-wide 24-hour quiescence is unproven")
        if evidence.get("no_outstanding_sessions") is not True:
            raise AuthorityBlocked("Outstanding session status is unproven")
    else:
        raise AuthorityBlocked("Unknown history provenance")
    if not _valid_digest(evidence.get("evidence_sha256")):
        raise AuthorityBlocked("Evidence artifact digest is missing")
    if not _valid_digest(evidence.get("history_sha256")):
        raise AuthorityBlocked("Imported history digest is missing")
    if evidence.get("history_time_basis") != "latest_possible_send_or_completion":
        raise AuthorityBlocked("Imported charge timestamps lack a conservative time basis")


def _validate_observation_evidence(evidence: dict, account_id: str, now: float,
                                   deadline_at: float) -> None:
    """Validate diagnostic packet shape without claiming upload history proof."""
    if not isinstance(evidence, dict) or evidence.get("account_id") != account_id:
        raise AuthorityBlocked("Diagnostic account provenance is missing")
    if (evidence.get("account_identity_verified") is not True or
        evidence.get("all_writers_excluded") is not True or
        evidence.get("history_basis") != "unproven" or
        evidence.get("quota_rejection_accepted") is not True):
        raise AuthorityBlocked("Diagnostic account, writer, or unproven-history evidence is missing")
    if _finite_nonnegative(evidence.get("exclusion_valid_until"), "exclusion end") < deadline_at:
        raise AuthorityBlocked("Diagnostic writer exclusion does not cover release")
    observed = _finite_nonnegative(evidence.get("observed_at"), "evidence observation time")
    if observed > now or now - observed > 300:
        raise AuthorityBlocked("Diagnostic evidence is not fresh")
    if not _valid_digest(evidence.get("evidence_sha256")):
        raise AuthorityBlocked("Diagnostic evidence artifact digest is missing")


def _validate_diagnostic_manifest(raw: bytes, release_id: str, account_id: str) -> tuple[str, int]:
    if not isinstance(raw, bytes) or not raw or len(raw) > (1 << 20):
        raise AuthorityBlocked("Diagnostic manifest bytes are missing or too large")
    if re.fullmatch(r"[0-9a-f]{32}", release_id) is None:
        raise AuthorityBlocked("Diagnostic release ID must be lowercase UUID hex")
    try:
        manifest = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise AuthorityBlocked("Diagnostic manifest cannot be decoded") from error
    payload = {"object": DIAGNOSTIC_PAYLOAD_OBJECT, "generation": 1792518093053185,
               "bytes": DIAGNOSTIC_PAYLOAD_BYTES,
               "sha256": "5417c6724312db880c07bf1cb514e54e1d735d6d1ce347c45e283bd2f59457aa"}
    receipt = {"object": DIAGNOSTIC_RECEIPT_OBJECT,
               "generation": 1792518094289877, "bytes": DIAGNOSTIC_RECEIPT_BYTES,
               "sha256": "fe4bc8db1aceb0c4868d9c1c82866925784b48be31db91d5301d3f1f51945c44"}
    expected = {
        "schema": "catbird.drive.diagnostic.v1", "release_id": release_id,
        "account_id": account_id, "bucket": "catbird-436600-jetstream-buffer-20261020",
        "source_id": "gce-selfhost-001", "source_epoch": "buffer-admission-20261020",
        "drive_root": "16Zd7n4pwTSDMDsnP9chX2Ci0r0ahEy45", "payload": payload,
        "receipt": receipt, "folder_name": "catbird-drive-diagnostic-" + release_id,
        "control_allowance_bytes": DIAGNOSTIC_CONTROL_BYTES,
        "max_charged_bytes": DIAGNOSTIC_PAIR_CAP_BYTES,
        "retain_gcs": True, "retain_local": True, "archive_qualified": False,
    }
    if not isinstance(manifest, dict) or set(manifest) != set(expected) | {"max_duration_seconds"}:
        raise AuthorityBlocked("Diagnostic manifest fields are not exact")
    for key, value in expected.items():
        if manifest.get(key) != value or type(manifest.get(key)) is not type(value):
            raise AuthorityBlocked("Diagnostic manifest is not the pinned segment 3677 pair")
    duration = manifest["max_duration_seconds"]
    if type(duration) is not int or not 0 < duration <= MAX_RUN_SECONDS:
        raise AuthorityBlocked("Diagnostic manifest duration exceeds 20 minutes")
    return hashlib.sha256(raw).hexdigest(), duration


def _diagnostic_pair_metadata(release_id: str, manifest_sha256: str) -> dict:
    return {
        "manifest_sha256": manifest_sha256,
        "diagnostic_folder_name": "catbird-drive-diagnostic-" + release_id,
        "drive_root": "16Zd7n4pwTSDMDsnP9chX2Ci0r0ahEy45",
        "payload_object": DIAGNOSTIC_PAYLOAD_OBJECT,
        "payload_generation": 1792518093053185,
        "receipt_object": DIAGNOSTIC_RECEIPT_OBJECT,
        "receipt_generation": 1792518094289877,
    }


def _connect(path: Path, *, create: bool) -> sqlite3.Connection:
    mode = "rwc" if create else "rw"
    # URI escaping is performed by pathlib so spaces and URL punctuation in a
    # local path cannot alter the sqlite mode.
    uri = path.resolve().as_uri().replace("file://", "file:") + f"?mode={mode}"
    connection = None
    try:
        connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise AuthorityBlocked("Corrupt authority database")
        return connection
    except (sqlite3.Error, OSError, AuthorityBlocked) as error:
        if connection is not None:
            connection.close()
        raise AuthorityBlocked("Authority database unavailable or corrupt") from error


class TransferAuthority:
    """One-pair admission, request pacing and conservative upload accounting."""

    def __init__(self, connection: sqlite3.Connection, db_path: Path, release_id: str, writer_id: str,
                 account_id: str, manifest_sha256: str, clock: Callable[[], float],
                 monotonic_clock: Callable[[], float]):
        self.db = connection
        self.db_path = str(db_path.resolve())
        self.release_id = release_id
        self.writer_id = writer_id
        self.account_id = account_id
        self.manifest_sha256 = manifest_sha256
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self.process_pid = os.getpid()
        self.process_host = socket.gethostname()
        self.process_nonce = _PROCESS_NONCE

    @classmethod
    def initialize_retained_observation(
        cls, db_path: Path, *, writer_id: str, account_id: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Create a separate, opt-in durable ledger for reviewed retained pairs."""
        from retained_transfer_authority import RetainedObservationAuthority
        RetainedObservationAuthority.initialize(
            db_path, writer_id=writer_id, account_id=account_id, clock=clock)

    @classmethod
    def import_retained_legacy_diagnostic(
        cls, db_path: Path, *, result_bytes: bytes, writer_id: str,
        account_id: str, clock: Callable[[], float] = time.time,
    ) -> None:
        """Import the exact reviewed segment-3677 result into the retained ledger."""
        from retained_transfer_authority import RetainedObservationAuthority
        RetainedObservationAuthority.import_legacy_result(
            db_path, result_bytes=result_bytes, writer_id=writer_id,
            account_id=account_id, clock=clock)

    @classmethod
    def list_retained_pairs(cls, db_path: Path, *, writer_id: str,
                            account_id: str) -> tuple[dict, ...]:
        from retained_transfer_authority import RetainedObservationAuthority
        return RetainedObservationAuthority.list_pairs(
            db_path, writer_id=writer_id, account_id=account_id)

    @classmethod
    def inspect_retained_pair(
        cls, db_path: Path, *, writer_id: str, account_id: str,
        run_id: str, pair_key: str, manifest_sha256: str,
    ) -> dict:
        from retained_transfer_authority import RetainedObservationAuthority
        return RetainedObservationAuthority.inspect_pair(
            db_path, writer_id=writer_id, account_id=account_id, run_id=run_id,
            pair_key=pair_key, manifest_sha256=manifest_sha256)

    @classmethod
    def inspect_retained_account(
        cls, db_path: Path, *, writer_id: str, account_id: str,
        clock: Callable[[], float] = time.time,
    ) -> dict:
        """Read validated account-wide retained stop gates without a writer claim."""
        from retained_transfer_authority import RetainedObservationAuthority
        return RetainedObservationAuthority.inspect_account(
            db_path, writer_id=writer_id, account_id=account_id, clock=clock)

    @classmethod
    def open_retained_run(
        cls, db_path: Path, *, run_id: str, writer_id: str, account_id: str,
        manifest_bytes: bytes, evidence: dict, deadline_at: float,
        clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        boot_id: str | None = None, recovery_only: bool = False,
    ):
        """Begin one reviewed pair, or recover only its pending index commit."""
        from retained_transfer_authority import RetainedObservationAuthority
        return RetainedObservationAuthority.open_run(
            db_path, run_id=run_id, writer_id=writer_id, account_id=account_id,
            manifest_bytes=manifest_bytes, evidence=evidence, deadline_at=deadline_at,
            clock=clock, monotonic_clock=monotonic_clock, boot_id=boot_id,
            recovery_only=recovery_only,
        )

    @classmethod
    def initialize_release(
        cls, db_path: Path, *, release_id: str, writer_id: str, account_id: str,
        manifest_sha256: str, evidence: dict, prior_charges: Iterable[dict],
        deadline_at: float, pilot_budget_bytes: int = PILOT_CAP_BYTES,
        provider_bound_bytes: int = DEFAULT_PROVIDER_BOUND_BYTES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize a strict-history release; this remains the default mode."""
        cls._initialize_release(
            db_path, release_id=release_id, writer_id=writer_id,
            account_id=account_id, manifest_sha256=manifest_sha256,
            evidence=evidence, prior_charges=prior_charges,
            deadline_at=deadline_at, pilot_budget_bytes=pilot_budget_bytes,
            provider_bound_bytes=provider_bound_bytes, clock=clock,
            admission_mode=STRICT_HISTORY,
        )

    @classmethod
    def initialize_diagnostic_quota_observation(
        cls, db_path: Path, *, release_id: str, writer_id: str, account_id: str,
        manifest_bytes: bytes, evidence: dict, deadline_at: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initialize only the exact isolated segment-3677 quota diagnostic.

        This records that account-wide history is unproven. It authorizes no
        archive qualification and never invents an empty account history.
        """
        digest, manifest_duration = _validate_diagnostic_manifest(
            manifest_bytes, release_id, account_id)
        now = _finite_nonnegative(clock(), "clock")
        deadline = _finite_nonnegative(deadline_at, "deadline")
        if not now < deadline <= now + manifest_duration:
            raise AuthorityBlocked("Diagnostic deadline exceeds pinned manifest duration")
        cls._initialize_release(
            db_path, release_id=release_id, writer_id=writer_id,
            account_id=account_id, manifest_sha256=digest, evidence=evidence,
            prior_charges=(), deadline_at=deadline, pilot_budget_bytes=DIAGNOSTIC_PAIR_CAP_BYTES,
            provider_bound_bytes=DEFAULT_PROVIDER_BOUND_BYTES, clock=clock,
            admission_mode=DIAGNOSTIC_QUOTA_OBSERVATION,
        )

    @classmethod
    def _initialize_release(
        cls, db_path: Path, *, release_id: str, writer_id: str, account_id: str,
        manifest_sha256: str, evidence: dict, prior_charges: Iterable[dict],
        deadline_at: float, pilot_budget_bytes: int = PILOT_CAP_BYTES,
        provider_bound_bytes: int = DEFAULT_PROVIDER_BOUND_BYTES,
        clock: Callable[[], float] = time.time,
        admission_mode: str,
    ) -> None:
        """Record an independently reviewed release packet in a *new* database.

        This setup API cannot itself establish that the supplied account-wide
        evidence is true. Each prior charge must carry
        ``latest_possible_send_at`` (latest possible provider acceptance or
        completion), ``bytes``, and ``unresolved``. A request creation time is
        too early to start its 24-hour aging. The live drain never calls setup.
        """
        path = Path(db_path)
        if path.exists():
            raise AuthorityBlocked("Existing authority database must not be replaced")
        for field, value in (("release ID", release_id), ("writer ID", writer_id), ("account ID", account_id)):
            _required_identity(value, field)
        if not _valid_digest(manifest_sha256):
            raise AuthorityBlocked("Invalid manifest digest")
        now = _finite_nonnegative(clock(), "clock")
        deadline = _finite_nonnegative(deadline_at, "deadline")
        if not now < deadline <= now + MAX_RUN_SECONDS:
            raise AuthorityBlocked("Release duration exceeds 20 minutes")
        if not 0 < _positive_int(pilot_budget_bytes, "pilot budget") <= PILOT_CAP_BYTES:
            raise AuthorityBlocked("Pilot budget exceeds 320 MiB")
        if not 0 < _positive_int(provider_bound_bytes, "provider bound") <= PROVIDER_REFERENCE_BYTES:
            raise AuthorityBlocked("Provider bound exceeds conservative reference")
        if admission_mode == STRICT_HISTORY:
            _validate_evidence(evidence, account_id, now, deadline, provider_bound_bytes)
        elif admission_mode == DIAGNOSTIC_QUOTA_OBSERVATION:
            _validate_observation_evidence(evidence, account_id, now, deadline)
            if pilot_budget_bytes != DIAGNOSTIC_PAIR_CAP_BYTES:
                raise AuthorityBlocked("Diagnostic pair cap mismatch")
        else:
            raise AuthorityBlocked("Unknown admission mode")
        entries = list(prior_charges)
        normalized = []
        for charge in entries:
            if not isinstance(charge, dict):
                raise AuthorityBlocked("Invalid imported upload charge")
            # The rolling window must start at the *latest possible* provider
            # acceptance/completion, never request creation. An uncertain send
            # remains charged indefinitely regardless of this timestamp.
            at = _finite_nonnegative(charge.get("latest_possible_send_at"), "latest possible send time")
            amount = _positive_int(charge.get("bytes"), "charge bytes")
            if at > now or type(charge.get("unresolved")) is not bool:
                raise AuthorityBlocked("Invalid imported upload charge")
            normalized.append((at, amount, int(charge["unresolved"])))
        normalized.sort()
        history_hash = hashlib.sha256(_canonical(normalized).encode()).hexdigest()
        if admission_mode == STRICT_HISTORY and evidence.get("history_sha256") != history_hash:
            raise AuthorityBlocked("Imported history does not match provenance")
        if admission_mode == DIAGNOSTIC_QUOTA_OBSERVATION and normalized:
            raise AuthorityBlocked("Diagnostic mode cannot import purported complete history")
        db = _connect(path, create=True)
        try:
            db.executescript("""
                CREATE TABLE release (
                    id TEXT PRIMARY KEY, writer_id TEXT NOT NULL, account_id TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL, evidence_json TEXT NOT NULL,
                    evidence_digest TEXT NOT NULL, pilot_budget_bytes INTEGER NOT NULL,
                    provider_bound_bytes INTEGER NOT NULL,
                    started_at REAL NOT NULL, deadline_at REAL NOT NULL,
                    lease_expires_at REAL NOT NULL, last_clock REAL NOT NULL,
                    next_slot REAL NOT NULL, cooldown_until REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK(status IN ('active','closed')),
                    admission_mode TEXT NOT NULL CHECK(admission_mode IN (
                        'strict_history','diagnostic_quota_observation')),
                    process_host TEXT, process_pid INTEGER, process_nonce TEXT,
                    db_path TEXT NOT NULL
                );
                CREATE TABLE history_charge (
                    id INTEGER PRIMARY KEY, at REAL NOT NULL, bytes INTEGER NOT NULL,
                    unresolved INTEGER NOT NULL CHECK(unresolved IN (0,1))
                );
                CREATE TABLE pair (
                    pair_key TEXT PRIMARY KEY, release_id TEXT NOT NULL REFERENCES release(id),
                    media_bytes INTEGER NOT NULL, control_allowance INTEGER NOT NULL,
                    reserved_remaining INTEGER NOT NULL, metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('open','closed'))
                );
                CREATE TABLE attempt (
                    id TEXT PRIMARY KEY, release_id TEXT NOT NULL REFERENCES release(id),
                    pair_key TEXT NOT NULL REFERENCES pair(pair_key),
                    at REAL NOT NULL, slot_at REAL NOT NULL, charge_through REAL NOT NULL,
                    timeout_s REAL NOT NULL, dispatched_at REAL, dispatch_monotonic REAL,
                    bytes INTEGER NOT NULL,
                    kind TEXT NOT NULL, method TEXT NOT NULL, target_category TEXT NOT NULL,
                    target_sha256 TEXT NOT NULL, operation_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('unresolved','settled')), http_status INTEGER
                );
            """)
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO release VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,?)", (
                release_id, writer_id, account_id, manifest_sha256, _canonical(evidence),
                evidence["evidence_sha256"], pilot_budget_bytes, provider_bound_bytes,
                now, deadline, deadline,
                now, now, 0, "active", admission_mode, str(path.resolve()),
            ))
            db.executemany("INSERT INTO history_charge(at,bytes,unresolved) VALUES (?,?,?)", normalized)
            db.execute("COMMIT")
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()
        # Setup deliberately leaves the writer process unbound. Only the first
        # authorized open claims it; an expired/dead process never auto-takes over.

    @classmethod
    def open_existing(
        cls, db_path: Path, *, release_id: str, writer_id: str, account_id: str,
        manifest_sha256: str, clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        allowed_mode: str = STRICT_HISTORY,
    ) -> "TransferAuthority":
        if allowed_mode not in (STRICT_HISTORY, DIAGNOSTIC_QUOTA_OBSERVATION):
            raise AuthorityBlocked("Unknown requested admission mode")
        for field, value in (("release ID", release_id), ("writer ID", writer_id), ("account ID", account_id)):
            _required_identity(value, field)
        if not _valid_digest(manifest_sha256):
            raise AuthorityBlocked("Invalid manifest digest")
        path = Path(db_path)
        if not path.is_file():
            raise AuthorityBlocked("Missing authority database is not empty history")
        db = _connect(path, create=False)
        authority = cls(db, path, release_id, writer_id, account_id, manifest_sha256,
                        clock, monotonic_clock)
        authority._allowed_mode = allowed_mode
        try:
            authority._claim_process()
            authority.assert_active()
        except Exception:
            db.close()
            raise
        return authority

    def close(self) -> None:
        self.db.close()

    @property
    def admission_mode(self) -> str:
        return self._row()["admission_mode"]

    def _claim_process(self) -> None:
        if os.getpid() != self.process_pid:
            raise AuthorityBlocked("Authority handle cannot cross a process fork")
        try:
            self._begin()
            row = self.db.execute("SELECT * FROM release WHERE id=?", (self.release_id,)).fetchone()
            if row is None or (row["writer_id"], row["account_id"], row["manifest_sha256"]) != (
                self.writer_id, self.account_id, self.manifest_sha256
            ) or row["db_path"] != self.db_path:
                raise AuthorityBlocked("Release identity mismatch")
            if row["admission_mode"] != self._allowed_mode:
                raise AuthorityBlocked("Admission mode mismatch before writer claim")
            binding = (row["process_host"], row["process_pid"], row["process_nonce"])
            expected = (self.process_host, self.process_pid, self.process_nonce)
            if binding == (None, None, None):
                self.db.execute("UPDATE release SET process_host=?,process_pid=?,process_nonce=? WHERE id=?",
                                (*expected, self.release_id))
            elif binding != expected:
                raise AuthorityBlocked("Writer process differs; explicit reconciliation required")
            self.db.execute("COMMIT")
        except (AuthorityBlocked, sqlite3.Error) as error:
            self._rollback()
            if isinstance(error, AuthorityBlocked):
                raise
            raise AuthorityBlocked("Writer process claim unavailable") from error

    def _row(self, *, allow_closed: bool = False) -> sqlite3.Row:
        if os.getpid() != self.process_pid:
            raise AuthorityBlocked("Authority handle cannot cross a process fork")
        try:
            row = self.db.execute("SELECT * FROM release WHERE id=?", (self.release_id,)).fetchone()
        except sqlite3.Error as error:
            raise AuthorityBlocked("Authority state unavailable or corrupt") from error
        if row is None:
            raise AuthorityBlocked("Release record missing")
        if (row["writer_id"], row["account_id"], row["manifest_sha256"]) != (
            self.writer_id, self.account_id, self.manifest_sha256
        ) or row["db_path"] != self.db_path:
            raise AuthorityBlocked("Account, writer lease or manifest identity mismatch")
        if row["admission_mode"] != self._allowed_mode:
            raise AuthorityBlocked("Admission mode mismatch")
        if (row["process_host"], row["process_pid"], row["process_nonce"]) != (
            self.process_host, self.process_pid, self.process_nonce
        ):
            raise AuthorityBlocked("Writer process binding mismatch")
        try:
            evidence = json.loads(row["evidence_json"])
            if row["admission_mode"] == STRICT_HISTORY:
                _validate_evidence(evidence, self.account_id, float(row["started_at"]),
                                   float(row["deadline_at"]), int(row["provider_bound_bytes"]))
            elif row["admission_mode"] == DIAGNOSTIC_QUOTA_OBSERVATION:
                _validate_observation_evidence(evidence, self.account_id,
                                               float(row["started_at"]), float(row["deadline_at"]))
                if row["pilot_budget_bytes"] != DIAGNOSTIC_PAIR_CAP_BYTES:
                    raise AuthorityBlocked("Diagnostic pair cap changed")
            else:
                raise AuthorityBlocked("Unknown stored admission mode")
        except (ValueError, TypeError, KeyError) as error:
            raise AuthorityBlocked("Release provenance is corrupt") from error
        if row["evidence_digest"] != evidence["evidence_sha256"]:
            raise AuthorityBlocked("Release provenance digest mismatch")
        imported = [tuple(charge) for charge in self.db.execute(
            "SELECT at,bytes,unresolved FROM history_charge ORDER BY at,bytes,unresolved"
        ).fetchall()]
        if row["admission_mode"] == STRICT_HISTORY:
            if hashlib.sha256(_canonical(imported).encode()).hexdigest() != evidence["history_sha256"]:
                raise AuthorityBlocked("Imported upload history changed or is missing")
        elif imported:
            raise AuthorityBlocked("Diagnostic history must remain explicitly unproven")
        if row["status"] != "active" and not allow_closed:
            raise AuthorityBlocked("Release is closed")
        if not 0 < row["pilot_budget_bytes"] <= PILOT_CAP_BYTES:
            raise AuthorityBlocked("Invalid pilot cap")
        if not 0 < row["provider_bound_bytes"] <= PROVIDER_REFERENCE_BYTES:
            raise AuthorityBlocked("Invalid provider bound")
        if not 0 < row["deadline_at"] - row["started_at"] <= MAX_RUN_SECONDS:
            raise AuthorityBlocked("Invalid release duration")
        if (row["lease_expires_at"] != row["deadline_at"] or
            not row["started_at"] <= row["last_clock"] or
            not row["started_at"] <= row["next_slot"] or
            row["cooldown_until"] < 0):
            raise AuthorityBlocked("Release clock or lease state is inconsistent")
        pairs = self.db.execute("SELECT * FROM pair").fetchall()
        if len(pairs) > 1:
            raise AuthorityBlocked("One-pair release contains multiple pairs")
        if pairs:
            pair = pairs[0]
            initial = pair["media_bytes"] + pair["control_allowance"]
            try:
                metadata = json.loads(pair["metadata_json"])
            except (TypeError, ValueError) as error:
                raise AuthorityBlocked("Pair provenance is corrupt") from error
            if (pair["release_id"] != self.release_id or
                metadata.get("manifest_sha256") != self.manifest_sha256 or
                not 0 < pair["media_bytes"] <= MAX_PAYLOAD_BYTES + (1 << 20) or
                not 0 < pair["control_allowance"] <= (1 << 20) or
                initial > row["pilot_budget_bytes"] or
                not 0 <= pair["reserved_remaining"] <= initial):
                raise AuthorityBlocked("Pair reservation provenance is inconsistent")
            if row["admission_mode"] == DIAGNOSTIC_QUOTA_OBSERVATION and (
                pair["pair_key"] != DIAGNOSTIC_PAIR_KEY or
                pair["media_bytes"] != DIAGNOSTIC_PAYLOAD_BYTES + DIAGNOSTIC_RECEIPT_BYTES or
                pair["control_allowance"] != DIAGNOSTIC_CONTROL_BYTES or
                metadata != _diagnostic_pair_metadata(self.release_id, self.manifest_sha256)
            ):
                raise AuthorityBlocked("Diagnostic pair is not the pinned segment 3677 pair")
            spent = self.db.execute("SELECT COALESCE(SUM(bytes),0) FROM attempt WHERE pair_key=?",
                                    (pair["pair_key"],)).fetchone()[0]
            if (pair["status"] == "open" and spent + pair["reserved_remaining"] != initial or
                pair["status"] == "closed" and (pair["reserved_remaining"] != 0 or spent > initial)):
                raise AuthorityBlocked("Attempt journal and reservation do not reconcile")
        attempts = self.db.execute("SELECT * FROM attempt").fetchall()
        if attempts and not pairs:
            raise AuthorityBlocked("Attempt exists without a reserved pair")
        dispatches = []
        monotonic_dispatches = []
        for attempt in attempts:
            if (attempt["release_id"] != self.release_id or
                attempt["pair_key"] != pairs[0]["pair_key"] or
                not isinstance(attempt["bytes"], int) or attempt["bytes"] < 0 or
                attempt["method"] not in ("GET", "POST", "PUT", "PATCH", "DELETE") or
                not _valid_digest(attempt["target_sha256"]) or
                not _valid_digest(attempt["operation_key"])):
                raise AuthorityBlocked("Malformed request identity or byte charge")
            _required_identity(attempt["id"], "attempt ID")
            _required_identity(attempt["kind"], "request kind")
            _required_identity(attempt["target_category"], "target category")
            at = _finite_nonnegative(attempt["at"], "attempt creation")
            slot = _finite_nonnegative(attempt["slot_at"], "attempt slot")
            timeout = _finite_nonnegative(attempt["timeout_s"], "attempt timeout")
            charge_through = _finite_nonnegative(attempt["charge_through"], "attempt charge window")
            if (timeout <= 0 or at < row["started_at"] or at > row["last_clock"] or
                slot < at or slot + timeout >= row["deadline_at"] or
                charge_through < slot + timeout):
                raise AuthorityBlocked("Malformed request timing or charge window")
            dispatch = attempt["dispatched_at"]
            dispatch_mono = attempt["dispatch_monotonic"]
            if (dispatch is None) != (dispatch_mono is None):
                raise AuthorityBlocked("Actual dispatch clocks disagree")
            if dispatch is not None:
                dispatch = _finite_nonnegative(dispatch, "actual dispatch")
                dispatch_mono = _finite_nonnegative(dispatch_mono, "monotonic dispatch")
                if (dispatch < slot or dispatch > row["last_clock"] or
                    dispatch + timeout >= row["deadline_at"] or
                    charge_through < dispatch + timeout):
                    raise AuthorityBlocked("Actual dispatch or charge window is inconsistent")
                dispatches.append(dispatch)
                monotonic_dispatches.append(dispatch_mono)
            http_status = attempt["http_status"]
            if http_status is not None and (
                not isinstance(http_status, int) or not 100 <= http_status <= 599
            ):
                raise AuthorityBlocked("Malformed HTTP result")
            settled_http = http_status == 308 or http_status is not None and 200 <= http_status < 300
            if (attempt["status"] == "settled" and not settled_http or
                attempt["status"] == "unresolved" and settled_http):
                raise AuthorityBlocked("Attempt result and charge status disagree")
        dispatches.sort()
        if any(b - a < PACING_SECONDS - 1e-9 for a, b in zip(dispatches, dispatches[1:])):
            raise AuthorityBlocked("Persisted request starts violate pacing")
        monotonic_dispatches.sort()
        if any(b - a < PACING_SECONDS - 1e-9 for a, b in zip(monotonic_dispatches, monotonic_dispatches[1:])):
            raise AuthorityBlocked("Persisted monotonic request starts violate pacing")
        if dispatches and row["next_slot"] < dispatches[-1] + PACING_SECONDS - 1e-9:
            raise AuthorityBlocked("Persisted next slot is earlier than actual dispatch")
        return row

    def _now(self, row: sqlite3.Row) -> float:
        now = _finite_nonnegative(self.clock(), "clock")
        if now < row["last_clock"]:
            raise AuthorityBlocked("Clock regressed; no upload admitted")
        return now

    def _active(self, row: sqlite3.Row, now: float, timeout_s: float = 0,
                *, slot_at: float | None = None) -> None:
        timeout = _finite_nonnegative(timeout_s, "request timeout")
        start = max(now, slot_at if slot_at is not None else now)
        if now >= row["lease_expires_at"] or now >= row["deadline_at"]:
            raise AuthorityBlocked("Writer lease or run deadline expired")
        if start + timeout >= min(row["deadline_at"], row["lease_expires_at"]):
            raise AuthorityBlocked("Request could exceed run deadline")
        if row["cooldown_until"] > now:
            raise AuthorityBlocked("Provider quota cooldown active")

    def assert_active(self, timeout_s: float = 0, *, slot_at: float | None = None) -> None:
        row = self._row()
        now = self._now(row)
        if slot_at is not None and now < _finite_nonnegative(slot_at, "assigned request slot"):
            raise AuthorityBlocked("Assigned request slot has not arrived")
        self._active(row, now, timeout_s, slot_at=slot_at)

    def validate_manifest_limits(self, max_charged_bytes: int, max_duration_seconds: int) -> None:
        """Bind the stored release to the pinned manifest's smaller limits."""
        byte_limit = _positive_int(max_charged_bytes, "manifest byte limit")
        duration = _positive_int(max_duration_seconds, "manifest duration")
        if byte_limit > PILOT_CAP_BYTES or duration > MAX_RUN_SECONDS:
            raise AuthorityBlocked("Manifest exceeds bounded pilot limits")
        row = self._row()
        if row["pilot_budget_bytes"] > byte_limit or row["deadline_at"] - row["started_at"] > duration:
            raise AuthorityBlocked("Release budget or duration exceeds pinned manifest")
        self.assert_active()

    def _begin(self) -> None:
        try:
            self.db.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as error:
            raise AuthorityBlocked("Cannot lock authority ledger") from error

    def _rollback(self) -> None:
        if self.db.in_transaction:
            self.db.execute("ROLLBACK")

    def _usage(self, now: float) -> int:
        threshold = now - WINDOW_SECONDS
        history = self.db.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM history_charge WHERE unresolved=1 OR at>?", (threshold,)
        ).fetchone()[0]
        attempts = self.db.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM attempt WHERE status='unresolved' OR charge_through>?", (threshold,)
        ).fetchone()[0]
        reserved = self.db.execute("SELECT COALESCE(SUM(reserved_remaining),0) FROM pair WHERE status='open'").fetchone()[0]
        return int(history) + int(attempts) + int(reserved)

    def reserve_pair(self, pair_key: str, media_bytes: int, control_allowance: int, metadata: dict) -> None:
        _required_identity(pair_key, "pair key")
        media = _positive_int(media_bytes, "media bytes")
        control = _positive_int(control_allowance, "control allowance")
        if media > MAX_PAYLOAD_BYTES + (1 << 20):
            raise AuthorityBlocked("Pair media exceeds bounded payload plus receipt")
        if control > (1 << 20):
            raise AuthorityBlocked("Pilot control allowance exceeds 1 MiB")
        if not isinstance(metadata, dict) or metadata.get("manifest_sha256") != self.manifest_sha256:
            raise AuthorityBlocked("Pair does not match pinned manifest")
        amount = media + control
        try:
            self._begin()
            row = self._row()
            now = self._now(row)
            self._active(row, now)
            if row["admission_mode"] == DIAGNOSTIC_QUOTA_OBSERVATION and (
                pair_key != DIAGNOSTIC_PAIR_KEY or
                media != DIAGNOSTIC_PAYLOAD_BYTES + DIAGNOSTIC_RECEIPT_BYTES or
                control != DIAGNOSTIC_CONTROL_BYTES or
                metadata != _diagnostic_pair_metadata(self.release_id, self.manifest_sha256)
            ):
                raise AuthorityBlocked("Diagnostic reservation differs from pinned pair")
            existing = self.db.execute("SELECT * FROM pair").fetchall()
            if existing:
                if len(existing) != 1 or existing[0]["pair_key"] != pair_key or existing[0]["metadata_json"] != _canonical(metadata):
                    raise AuthorityBlocked("One-pair release already has another reservation")
                if existing[0]["media_bytes"] != media or existing[0]["control_allowance"] != control:
                    raise AuthorityBlocked("Pinned pair size changed")
                if existing[0]["status"] != "open":
                    raise AuthorityBlocked("Pair already closed")
                self.db.execute("UPDATE release SET last_clock=? WHERE id=?", (now, self.release_id))
                self.db.execute("COMMIT")
                return
            effective_cap = min(ROLLING_CAP_BYTES, row["provider_bound_bytes"])
            if (amount > row["pilot_budget_bytes"] or
                row["admission_mode"] == STRICT_HISTORY and self._usage(now) + amount > effective_cap):
                raise AuthorityBlocked("Pilot or rolling account-wide byte cap exceeded")
            self.db.execute("INSERT INTO pair VALUES (?,?,?,?,?,?,?)", (
                pair_key, self.release_id, media, control, amount, _canonical(metadata), "open",
            ))
            self.db.execute("UPDATE release SET last_clock=? WHERE id=?", (now, self.release_id))
            self.db.execute("COMMIT")
        except (AuthorityBlocked, sqlite3.Error) as error:
            self._rollback()
            if isinstance(error, AuthorityBlocked):
                raise
            raise AuthorityBlocked("Pair reservation could not be persisted") from error

    def start_request(self, body_bytes: int, timeout_s: float, kind: str, *,
                      method: str, target_category: str, target_sha256: str,
                      operation_key: str) -> tuple[str, float]:
        if isinstance(body_bytes, bool) or not isinstance(body_bytes, int) or body_bytes < 0:
            raise AuthorityBlocked("Invalid request body length")
        _required_identity(kind, "request kind")
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            raise AuthorityBlocked("Invalid HTTP method")
        _required_identity(target_category, "target category")
        if not _valid_digest(target_sha256) or not _valid_digest(operation_key):
            raise AuthorityBlocked("Request target or operation digest missing")
        if _finite_nonnegative(timeout_s, "request timeout") <= 0:
            raise AuthorityBlocked("Request timeout must be positive")
        try:
            self._begin()
            row = self._row()
            now = self._now(row)
            pair = self.db.execute("SELECT * FROM pair WHERE release_id=?", (self.release_id,)).fetchone()
            if pair is None or pair["status"] != "open":
                raise AuthorityBlocked("Pair must be reserved before any Drive request")
            slot = max(now, float(row["next_slot"]))
            self._active(row, now, timeout_s, slot_at=slot)
            if body_bytes > pair["reserved_remaining"]:
                raise AuthorityBlocked("Request would exceed pre-send reservation")
            # Moving bytes from reserve to unresolved charge keeps total
            # admission unchanged, including across crashes and lost replies.
            attempt_id = uuid.uuid4().hex
            self.db.execute("UPDATE pair SET reserved_remaining=reserved_remaining-? WHERE pair_key=?",
                            (body_bytes, pair["pair_key"]))
            self.db.execute("INSERT INTO attempt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                attempt_id, self.release_id, pair["pair_key"], now, slot, slot + timeout_s,
                timeout_s, None, None, body_bytes,
                kind, method, target_category, target_sha256, operation_key,
                "unresolved", None,
            ))
            self.db.execute("UPDATE release SET last_clock=?, next_slot=? WHERE id=?",
                            (now, slot + PACING_SECONDS, self.release_id))
            if (row["admission_mode"] == STRICT_HISTORY and
                self._usage(now) > min(ROLLING_CAP_BYTES, row["provider_bound_bytes"])):
                raise AuthorityBlocked("Account-wide cap inconsistent after reservation")
            self.db.execute("COMMIT")
            return attempt_id, slot
        except (AuthorityBlocked, sqlite3.Error) as error:
            self._rollback()
            if isinstance(error, AuthorityBlocked):
                raise
            raise AuthorityBlocked("Request attempt could not be persisted") from error

    def mark_dispatch(self, attempt_id: str) -> None:
        """Persist the actual request start immediately before HTTP dispatch.

        The reserved slot prevents an early start; this second transaction
        prevents a *late* start from letting the next request start too soon.
        A crash before this mark leaves the precharged attempt unresolved.
        """
        _required_identity(attempt_id, "attempt ID")
        try:
            self._begin()
            row = self._row()
            now = self._now(row)
            mono_now = _finite_nonnegative(self.monotonic_clock(), "monotonic clock")
            attempt = self.db.execute("SELECT * FROM attempt WHERE id=? AND release_id=?",
                                      (attempt_id, self.release_id)).fetchone()
            if attempt is None or attempt["status"] != "unresolved" or attempt["dispatched_at"] is not None:
                raise AuthorityBlocked("Request attempt cannot be dispatched again")
            if now < attempt["slot_at"]:
                raise AuthorityBlocked("Assigned request slot has not arrived")
            self._active(row, now, attempt["timeout_s"], slot_at=attempt["slot_at"])
            latest = self.db.execute(
                "SELECT MAX(dispatched_at),MAX(dispatch_monotonic) FROM attempt WHERE release_id=? AND id<>?",
                (self.release_id, attempt_id),
            ).fetchone()
            if ((latest[0] is not None and now < latest[0] + PACING_SECONDS) or
                (latest[1] is not None and mono_now < latest[1] + PACING_SECONDS)):
                raise AuthorityBlocked("Actual request starts would violate persisted pacing")
            self.db.execute("UPDATE attempt SET dispatched_at=?,dispatch_monotonic=?,"
                            "charge_through=MAX(charge_through,?) WHERE id=?",
                            (now, mono_now, now + attempt["timeout_s"], attempt_id))
            self.db.execute("UPDATE release SET last_clock=?,next_slot=MAX(next_slot,?) WHERE id=?",
                            (now, now + PACING_SECONDS, self.release_id))
            self.db.execute("COMMIT")
        except (AuthorityBlocked, sqlite3.Error) as error:
            self._rollback()
            if isinstance(error, AuthorityBlocked):
                raise
            raise AuthorityBlocked("Actual request start could not be persisted") from error

    def finish_request(self, attempt_id: str, status_code: int | None) -> None:
        _required_identity(attempt_id, "attempt ID")
        if status_code is not None and (isinstance(status_code, bool) or not isinstance(status_code, int)):
            raise AuthorityBlocked("Invalid response status")
        try:
            self._begin()
            row = self._row()
            now = self._now(row)
            attempt = self.db.execute("SELECT * FROM attempt WHERE id=? AND release_id=?",
                                      (attempt_id, self.release_id)).fetchone()
            if attempt is None:
                raise AuthorityBlocked("Unknown request attempt")
            if attempt["status"] == "settled":
                if attempt["http_status"] != status_code:
                    raise AuthorityBlocked("Attempt result changed")
            elif status_code is not None and (200 <= status_code < 300 or status_code == 308):
                self.db.execute("UPDATE attempt SET status='settled',http_status=?,"
                                "charge_through=MAX(charge_through,?) WHERE id=?",
                                (status_code, now, attempt_id))
            else:
                # A provider rejection or transport failure cannot prove that
                # a body was never accepted; keep this charge unresolved.
                self.db.execute("UPDATE attempt SET http_status=?,"
                                "charge_through=MAX(charge_through,?) WHERE id=?",
                                (status_code, now, attempt_id))
            if status_code in (403, 429):
                self.db.execute("UPDATE release SET cooldown_until=MAX(cooldown_until,?) WHERE id=?",
                                (now + QUOTA_COOLDOWN_SECONDS, self.release_id))
            self.db.execute("UPDATE release SET last_clock=? WHERE id=?", (now, self.release_id))
            self.db.execute("COMMIT")
        except (AuthorityBlocked, sqlite3.Error) as error:
            self._rollback()
            if isinstance(error, AuthorityBlocked):
                raise
            raise AuthorityBlocked("Request result could not be persisted") from error

    def close_pair(self, pair_key: str) -> None:
        """Settle only unused reservation after independent pair verification."""
        try:
            self._begin()
            row = self._row(allow_closed=True)
            now = self._now(row)
            pair = self.db.execute("SELECT * FROM pair WHERE pair_key=? AND release_id=?",
                                   (pair_key, self.release_id)).fetchone()
            if pair is None:
                raise AuthorityBlocked("Unknown pair")
            if pair["status"] == "closed":
                self.db.execute("COMMIT")
                return
            unresolved = self.db.execute("SELECT COUNT(*) FROM attempt WHERE pair_key=? AND status='unresolved'",
                                         (pair_key,)).fetchone()[0]
            if unresolved:
                raise AuthorityBlocked("Unresolved sends block reservation settlement")
            self.db.execute("UPDATE pair SET reserved_remaining=0,status='closed' WHERE pair_key=?", (pair_key,))
            self.db.execute("UPDATE release SET last_clock=?,status='closed' WHERE id=?", (now, self.release_id))
            self.db.execute("COMMIT")
        except (AuthorityBlocked, sqlite3.Error) as error:
            self._rollback()
            if isinstance(error, AuthorityBlocked):
                raise
            raise AuthorityBlocked("Pair settlement could not be persisted") from error

    def snapshot(self) -> dict:
        """Small, nonsecret test/review view of persisted balances."""
        row = self._row()
        now = self._now(row)
        pair = self.db.execute("SELECT * FROM pair").fetchone()
        return {
            "admission_mode": row["admission_mode"],
            "history_proven": row["admission_mode"] == STRICT_HISTORY,
            "rolling_reserved_and_charged_bytes": self._usage(now),
            "effective_rolling_cap_bytes": (min(ROLLING_CAP_BYTES, row["provider_bound_bytes"])
                                            if row["admission_mode"] == STRICT_HISTORY else None),
            "pair_reserved_remaining": pair["reserved_remaining"] if pair else 0,
            "unresolved_attempts": self.db.execute(
                "SELECT COUNT(*) FROM attempt WHERE status='unresolved'"
            ).fetchone()[0],
            "next_slot": row["next_slot"],
            "cooldown_until": row["cooldown_until"],
            "status": row["status"],
        }


def read_existing_oauth_permission_id(oauth_file: Path) -> str:
    """Read-only Drive account preflight using the existing GCE OAuth client.

    DriveClient may refresh an access token in memory; it does not rewrite the
    OAuth file. This helper never logs a token or follows a redirect.
    """
    import urllib.request
    from gce_segment_offloader import DriveClient

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, message, headers, newurl):
            raise AuthorityBlocked("Drive account preflight redirected")

    try:
        token = DriveClient(Path(oauth_file)).token()
        request = urllib.request.Request(
            "https://www.googleapis.com/drive/v3/about?fields=user(permissionId)",
            headers={"Authorization": "Bearer " + token}, method="GET")
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
            about = json.load(response)
        permission_id = about.get("user", {}).get("permissionId")
        return _required_identity(permission_id, "Drive permission ID")
    except Exception as error:
        raise AuthorityBlocked("Drive account preflight unavailable") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a bounded, offline Drive authority release")
    command = parser.add_subparsers(dest="command", required=True)
    preflight = command.add_parser("preflight-account", help="Read Drive permissionId with existing OAuth")
    preflight.add_argument("--oauth-file", type=Path, required=True)
    setup = command.add_parser("initialize-diagnostic", help="Persist an offline quota observation release")
    setup.add_argument("--authority-db", type=Path, required=True)
    setup.add_argument("--manifest", type=Path, required=True)
    setup.add_argument("--evidence-json", type=Path, required=True)
    setup.add_argument("--release-id", required=True)
    setup.add_argument("--writer-id", required=True)
    setup.add_argument("--account-id", required=True)
    setup.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight-account":
            print(json.dumps({"permissionId": read_existing_oauth_permission_id(args.oauth_file)}))
            return 0
        evidence = json.loads(args.evidence_json.read_text())
        TransferAuthority.initialize_diagnostic_quota_observation(
            args.authority_db, release_id=args.release_id, writer_id=args.writer_id,
            account_id=args.account_id, manifest_bytes=args.manifest.read_bytes(),
            evidence=evidence, deadline_at=args.deadline_epoch,
        )
        print(json.dumps({"event": "diagnostic_authority_initialized",
                          "admission_mode": DIAGNOSTIC_QUOTA_OBSERVATION,
                          "history_proven": False}))
        return 0
    except (AuthorityBlocked, OSError, ValueError) as error:
        parser.exit(2, f"blocked: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
