"""Opt-in, local SQLite authority for reviewed retained GCS-to-Drive pairs.

One account/writer/database carries pacing, cooldown and observed upload charges
through successive one-pair runs. Account-wide prior history is explicitly
unproven; observation mode stops on provider rejection rather than inferring
headroom from this incomplete local history.
Only terminal, index-committed runs permit the next pair. A stopped active run
or uncertain send needs manual reconciliation; a verified/index-pending run
may be reopened solely to publish or confirm its immutable mapping.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import time
from typing import Callable, Iterator
import uuid

from drive_transfer_authority import (
    AuthorityBlocked, DIAGNOSTIC_PAYLOAD_OBJECT, MAX_RUN_SECONDS,
    PACING_SECONDS, PILOT_CAP_BYTES, QUOTA_COOLDOWN_SECONDS,
    WINDOW_SECONDS, _PROCESS_NONCE, _canonical, _connect,
    _finite_nonnegative, _positive_int, _required_identity, _valid_digest,
)

LEGACY_DIAGNOSTIC_SOURCE_SHA256 = "28a10e6fffe4308310b1ef0ca6a523df401821b3164fc99466d1058f4493a005"
LEGACY_DIAGNOSTIC_RELEASE_ID = "edef464c6ef8489ba798c22b257da7b4"
LEGACY_DIAGNOSTIC_ATTEMPTS = 51
LEGACY_DIAGNOSTIC_CHARGED_BYTES = 268_982_001
LEGACY_DIAGNOSTIC_BUCKET = "catbird-436600-jetstream-buffer-20261020"
LEGACY_DIAGNOSTIC_PAYLOAD_GENERATION = 1792518093053185
MAPPING_PARTITION_BYTES = 33_554_432
MAPPING_MAX_BODY_BYTES = 16_384
GLOBAL_ROLLING_CAP_BYTES = 700_000_000_000
MAX_CONCURRENT_RUNS = 4
RECONCILIATION_SCHEMA = "catbird.drive.retained-receipt-500-reconciliation.v1"
STATUS_PROBE_RECONCILIATION_SCHEMA = (
    "catbird.drive.retained-status-probe-reconciliation.v1"
)


def _boot_id() -> str:
    """Linux boot identity; uncertain boot identity blocks new sends."""
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError as error:
        raise AuthorityBlocked("Boot identity unavailable for retained writer") from error
    return _required_identity(value, "boot identity")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise AuthorityBlocked("Duplicate retained manifest key")
        result[key] = value
    return result


def _reviewed_manifest(raw: bytes, run_id: str, account_id: str) -> tuple[dict, str, str]:
    if not isinstance(raw, bytes) or not raw or len(raw) > (1 << 20):
        raise AuthorityBlocked("Retained manifest bytes are missing or too large")
    try:
        manifest = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, TypeError) as error:
        raise AuthorityBlocked("Retained manifest cannot be decoded") from error
    try:
        from gcs_drive_retained_overflow import pair_key, validate_overflow_manifest
        validate_overflow_manifest(manifest, release_id=run_id, account_id=account_id)
        key = pair_key(manifest)
    except Exception as error:
        raise AuthorityBlocked("Retained manifest is not a pinned one-pair packet") from error
    return manifest, hashlib.sha256(raw).hexdigest(), key


def _review_evidence(evidence: dict, *, run_id: str, writer_id: str, account_id: str,
                     manifest_sha256: str, now: float, deadline_at: float) -> None:
    if not isinstance(evidence, dict) or any(evidence.get(name) != expected for name, expected in (
        ("run_id", run_id), ("writer_id", writer_id), ("account_id", account_id),
        ("reviewed_manifest_sha256", manifest_sha256), ("history_basis", "unproven"),
    )):
        raise AuthorityBlocked("Retained review is not bound to this run, writer and manifest")
    if (evidence.get("account_identity_verified") is not True or
        evidence.get("all_writers_excluded") is not True or
        evidence.get("quota_rejection_accepted") is not True):
        raise AuthorityBlocked("Retained account, writer exclusion or quota consent is unverified")
    if _finite_nonnegative(evidence.get("exclusion_valid_until"), "exclusion end") < deadline_at:
        raise AuthorityBlocked("Retained writer exclusion ends before run deadline")
    observed = _finite_nonnegative(evidence.get("observed_at"), "review observation")
    if observed > now or now - observed > 300:
        raise AuthorityBlocked("Retained review evidence is stale")
    if not _valid_digest(evidence.get("evidence_sha256")):
        raise AuthorityBlocked("Retained review evidence digest is missing")


def _expected_metadata(manifest: dict, digest: str) -> dict:
    return {
        "manifest_sha256": digest, "folder_name": manifest["folder_name"],
        "drive_root": manifest["drive_root"],
        "payload_object": manifest["payload"]["object"],
        "payload_generation": manifest["payload"]["generation"],
        "receipt_object": manifest["receipt"]["object"],
        "receipt_generation": manifest["receipt"]["generation"],
    }


def _validated_legacy_result(raw: bytes, *, expected_sha256: str,
                             expected_release: str, expected_attempts: int,
                             expected_bytes: int, expected_account: str) -> dict:
    """Validate a complete, externally reviewed diagnostic result, without I/O."""
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AuthorityBlocked("Legacy diagnostic evidence digest differs")
    try:
        result = json.loads(raw, object_pairs_hook=_unique_object)
        release = result["release"]
        descriptor = result["descriptor"]
        attempts = result["attempts"]
    except (ValueError, TypeError, KeyError) as error:
        raise AuthorityBlocked("Legacy diagnostic evidence is malformed") from error
    if (not isinstance(release, dict) or not isinstance(descriptor, dict) or
        not isinstance(attempts, list) or len(attempts) != expected_attempts or
        result.get("charged_bytes") != expected_bytes or
        result.get("unresolved_attempts") != 0 or
        result.get("history_proven") is not False or
        result.get("archive_qualified") is not False or
        release.get("id") != expected_release or
        release.get("status") != "closed" or
        release.get("admission_mode") != "diagnostic_quota_observation" or
        descriptor.get("release_id") != expected_release or
        release.get("account_id") != expected_account or
        descriptor.get("account_id") != expected_account or
        not _valid_digest(descriptor.get("manifest_sha256"))):
        raise AuthorityBlocked("Legacy diagnostic identity or outcome differs")
    total = 0
    last_dispatch = 0.0
    charge_through = 0.0
    ids = set()
    for attempt in attempts:
        if not isinstance(attempt, dict) or attempt.get("status") != "settled":
            raise AuthorityBlocked("Legacy diagnostic contains an uncertain attempt")
        key = attempt.get("id")
        if not isinstance(key, str) or key in ids:
            raise AuthorityBlocked("Legacy diagnostic attempt ID is missing or repeated")
        ids.add(key)
        body = attempt.get("bytes")
        if type(body) is not int or body < 0:
            raise AuthorityBlocked("Legacy diagnostic byte charge is malformed")
        total += body
        at = _finite_nonnegative(attempt.get("at"), "legacy attempt start")
        slot = _finite_nonnegative(attempt.get("slot_at"), "legacy request slot")
        dispatch = _finite_nonnegative(attempt.get("dispatched_at"), "legacy actual dispatch")
        through = _finite_nonnegative(attempt.get("charge_through"), "legacy charge-through time")
        if not at <= slot <= dispatch <= through:
            raise AuthorityBlocked("Legacy diagnostic attempt timing is inconsistent")
        http = attempt.get("http_status")
        if type(http) is not int or not (200 <= http < 300 or http == 308):
            raise AuthorityBlocked("Legacy diagnostic settled HTTP result differs")
        if attempt.get("method") not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            raise AuthorityBlocked("Legacy diagnostic HTTP method is malformed")
        last_dispatch = max(last_dispatch, dispatch)
        charge_through = max(charge_through, through)
    if total != expected_bytes:
        raise AuthorityBlocked("Legacy diagnostic attempt bytes do not sum to charge")
    entries = result.get("entries")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise AuthorityBlocked("Legacy diagnostic source entry is missing")
    entry = entries[0]
    payload_ref = entry.get("payload_ref")
    if (entry.get("status") != "diagnostic_verified_unqualified" or
        entry.get("archive_qualified") is not False or
        not isinstance(payload_ref, dict) or
        payload_ref.get("name") != DIAGNOSTIC_PAYLOAD_OBJECT or
        payload_ref.get("generation") != LEGACY_DIAGNOSTIC_PAYLOAD_GENERATION or
        payload_ref.get("size") != 268_980_444):
        raise AuthorityBlocked("Legacy diagnostic source identity differs")
    next_slot = _finite_nonnegative(release.get("next_slot"), "legacy next request slot")
    cooldown = _finite_nonnegative(release.get("cooldown_until"), "legacy quota cooldown")
    started = _finite_nonnegative(release.get("started_at"), "legacy start")
    deadline = _finite_nonnegative(release.get("deadline_at"), "legacy deadline")
    if (not started < deadline or last_dispatch > deadline or
        next_slot < last_dispatch + PACING_SECONDS - 1e-6):
        raise AuthorityBlocked("Legacy diagnostic release timing is inconsistent")
    return {
        "source_sha256": expected_sha256,
        "release_id": expected_release,
        "manifest_sha256": descriptor["manifest_sha256"],
        "account_id": expected_account,
        "writer_id": _required_identity(descriptor.get("writer_id"), "legacy writer ID"),
        "bytes": total, "charge_through": charge_through,
        "last_dispatch_wall": last_dispatch,
        "next_slot": next_slot, "cooldown_until": cooldown,
        "bucket": LEGACY_DIAGNOSTIC_BUCKET,
        "payload_object": payload_ref["name"],
        "payload_generation": payload_ref["generation"],
    }


def _read_only_connection(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise AuthorityBlocked("Retained ledger is missing")
    db = None
    try:
        uri = path.resolve().as_uri().replace("file://", "file:") + "?mode=ro"
        db = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise AuthorityBlocked("Retained ledger integrity check failed")
        return db
    except Exception as error:
        if db is not None:
            db.close()
        if isinstance(error, sqlite3.Error):
            raise AuthorityBlocked("Retained ledger read is unavailable") from error
        raise


class RetainedObservationAuthority:
    """One active reviewed run on a durable single-account ledger."""

    def __init__(self, db: sqlite3.Connection, path: Path, *, run_id: str,
                 writer_id: str, account_id: str, manifest: dict,
                 manifest_sha256: str, pair_key: str, boot_id: str,
                 clock: Callable[[], float], monotonic_clock: Callable[[], float],
                 recovery_only: bool, max_concurrent_runs: int = 1):
        self.db = db
        self.db_path = str(path.resolve())
        self.release_id = run_id
        self.writer_id = writer_id
        self.account_id = account_id
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.expected_pair_key = pair_key
        self.boot_id = boot_id
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self.recovery_only = recovery_only
        self.max_concurrent_runs = max_concurrent_runs
        self.process_host = socket.gethostname()
        self.process_pid = os.getpid()
        self.process_nonce = _PROCESS_NONCE

    @classmethod
    def initialize(cls, db_path: Path, *, writer_id: str, account_id: str,
                   clock: Callable[[], float] = time.time) -> None:
        path = Path(db_path)
        if path.exists():
            raise AuthorityBlocked("Existing retained ledger must not be replaced")
        _required_identity(writer_id, "writer ID")
        _required_identity(account_id, "account ID")
        now = _finite_nonnegative(clock(), "clock")
        db = _connect(path, create=True)
        try:
            db.executescript("""
                CREATE TABLE retained_account (
                    id INTEGER PRIMARY KEY CHECK(id=1), writer_id TEXT NOT NULL,
                    account_id TEXT NOT NULL, db_path TEXT NOT NULL, boot_id TEXT,
                    process_host TEXT, process_pid INTEGER, process_nonce TEXT,
                    last_clock REAL NOT NULL, next_slot REAL NOT NULL,
                    cooldown_until REAL NOT NULL, last_dispatch_wall REAL,
                    last_dispatch_monotonic REAL
                );
                CREATE TABLE retained_run (
                    id TEXT PRIMARY KEY, manifest_sha256 TEXT NOT NULL,
                    manifest_raw BLOB NOT NULL, pair_key TEXT NOT NULL UNIQUE,
                    media_bytes INTEGER NOT NULL, control_allowance INTEGER NOT NULL,
                    budget_bytes INTEGER NOT NULL, metadata_json TEXT,
                    reserved INTEGER NOT NULL CHECK(reserved IN (0,1)),
                    reserved_remaining INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        'active','verified','index_pending','closed')),
                    mapping_sha256 TEXT, evidence_json TEXT NOT NULL,
                    started_at REAL NOT NULL, deadline_at REAL NOT NULL
                );
                CREATE TABLE retained_attempt (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES retained_run(id),
                    at REAL NOT NULL, slot_at REAL NOT NULL, charge_through REAL NOT NULL,
                    timeout_s REAL NOT NULL, dispatched_at REAL, dispatch_monotonic REAL,
                    bytes INTEGER NOT NULL, kind TEXT NOT NULL, method TEXT NOT NULL,
                    target_category TEXT NOT NULL, target_sha256 TEXT NOT NULL,
                    operation_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('unresolved','settled')),
                    http_status INTEGER
                );
                CREATE TABLE retained_import (
                    source_sha256 TEXT PRIMARY KEY, release_id TEXT NOT NULL UNIQUE,
                    manifest_sha256 TEXT NOT NULL, original_writer_id TEXT NOT NULL,
                    account_id TEXT NOT NULL, bytes INTEGER NOT NULL,
                    charge_through REAL NOT NULL, last_dispatch_wall REAL NOT NULL,
                    next_slot REAL NOT NULL, cooldown_until REAL NOT NULL,
                    bucket TEXT NOT NULL, payload_object TEXT NOT NULL,
                    payload_generation INTEGER NOT NULL
                );
                CREATE TABLE retained_mapping_reservation (
                    run_id TEXT PRIMARY KEY REFERENCES retained_run(id),
                    pair_key TEXT NOT NULL UNIQUE, mapping_sha256 TEXT NOT NULL,
                    object_name TEXT NOT NULL, bytes INTEGER NOT NULL,
                    reservation_id TEXT NOT NULL UNIQUE
                );
                CREATE TABLE retained_reconciliation (
                    attempt_id TEXT PRIMARY KEY REFERENCES retained_attempt(id),
                    run_id TEXT NOT NULL REFERENCES retained_run(id),
                    pair_key TEXT NOT NULL, proof_json TEXT NOT NULL,
                    proof_sha256 TEXT NOT NULL, applied_at REAL NOT NULL
                );
                CREATE TABLE retained_run_claim (
                    run_id TEXT PRIMARY KEY REFERENCES retained_run(id),
                    process_host TEXT NOT NULL,
                    process_pid INTEGER NOT NULL,
                    process_nonce TEXT NOT NULL,
                    max_concurrent_runs INTEGER NOT NULL CHECK(max_concurrent_runs BETWEEN 1 AND 4)
                );
            """)
            db.execute("INSERT INTO retained_account VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?)", (
                writer_id, account_id, str(path.resolve()), None, None, None, None,
                now, now, 0, None, None,
            ))
        finally:
            db.close()

    def _has_run_claim_table(self) -> bool:
        return self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retained_run_claim'"
        ).fetchone() is not None

    def _ensure_run_claim_table(self) -> None:
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS retained_run_claim (
                run_id TEXT PRIMARY KEY REFERENCES retained_run(id),
                process_host TEXT NOT NULL,
                process_pid INTEGER NOT NULL,
                process_nonce TEXT NOT NULL,
                max_concurrent_runs INTEGER NOT NULL CHECK(max_concurrent_runs BETWEEN 1 AND 4)
            )
        """)

    def _run_claim(self, run_id: str) -> sqlite3.Row | None:
        if not self._has_run_claim_table():
            return None
        return self.db.execute(
            "SELECT * FROM retained_run_claim WHERE run_id=?", (run_id,),
        ).fetchone()

    def _claim_process_alive(self, claim: sqlite3.Row) -> bool:
        if (claim["process_host"] != self.process_host or
            type(claim["process_pid"]) is not int or claim["process_pid"] <= 0):
            return False
        if claim["process_pid"] == self.process_pid:
            return True
        try:
            os.kill(claim["process_pid"], 0)
        except (ProcessLookupError, PermissionError, OSError):
            return False
        return True

    def _blocking_unresolved_count(self, now: float, active_run_ids: set[str]) -> int:
        resolved = self._reconciled_attempt_ids()
        count = 0
        for row in self.db.execute("SELECT * FROM retained_attempt WHERE status='unresolved'"):
            if row["id"] in resolved:
                continue
            if (row["run_id"] in active_run_ids and
                row["http_status"] is None and
                now < row["charge_through"]):
                continue
            count += 1
        if self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND "
            "name='retained_receipt_recovery_attempt'").fetchone():
            count += self.db.execute(
                "SELECT COUNT(*) FROM retained_receipt_recovery_attempt "
                "WHERE status='unresolved'").fetchone()[0]
        return count

    @classmethod
    def import_legacy_result(
        cls, db_path: Path, *, result_bytes: bytes, writer_id: str,
        account_id: str, clock: Callable[[], float] = time.time,
    ) -> None:
        """Import the reviewed segment-3677 charge once, without claiming full history.

        The source digest, release, attempt count and charged bytes are pinned in
        this module. `charge_through` is the latest conservative charge horizon,
        not the request creation time. The original result/database is untouched.
        """
        path = Path(db_path)
        _required_identity(writer_id, "writer ID")
        account_id = _required_identity(account_id, "account ID")
        evidence = _validated_legacy_result(
            result_bytes, expected_sha256=LEGACY_DIAGNOSTIC_SOURCE_SHA256,
            expected_release=LEGACY_DIAGNOSTIC_RELEASE_ID,
            expected_attempts=LEGACY_DIAGNOSTIC_ATTEMPTS,
            expected_bytes=LEGACY_DIAGNOSTIC_CHARGED_BYTES,
            expected_account=account_id,
        )
        values = (
            evidence["source_sha256"], evidence["release_id"],
            evidence["manifest_sha256"], evidence["writer_id"], account_id,
            evidence["bytes"], evidence["charge_through"],
            evidence["last_dispatch_wall"], evidence["next_slot"],
            evidence["cooldown_until"],
            evidence["bucket"], evidence["payload_object"],
            evidence["payload_generation"],
        )
        now = _finite_nonnegative(clock(), "clock")
        if now < evidence["last_dispatch_wall"]:
            raise AuthorityBlocked("Import clock predates legacy dispatch")
        if not path.is_file():
            raise AuthorityBlocked("Retained ledger must be initialized before import")
        db = _connect(path, create=False)
        try:
            db.execute("BEGIN IMMEDIATE")
            account = db.execute("SELECT * FROM retained_account WHERE id=1").fetchone()
            if (account is None or account["writer_id"] != writer_id or
                account["account_id"] != account_id or
                account["db_path"] != str(path.resolve()) or
                now < account["last_clock"]):
                raise AuthorityBlocked("Legacy import ledger identity or clock differs")
            prior = db.execute("SELECT * FROM retained_import").fetchall()
            if prior:
                if len(prior) != 1 or tuple(prior[0]) != values:
                    raise AuthorityBlocked("Conflicting legacy diagnostic import")
            else:
                if db.execute("SELECT COUNT(*) FROM retained_run").fetchone()[0]:
                    raise AuthorityBlocked("Legacy diagnostic must be imported before retained runs")
                db.execute("INSERT INTO retained_import VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
                db.execute(
                    "UPDATE retained_account SET last_clock=?,next_slot=MAX(next_slot,?),"
                    "cooldown_until=MAX(cooldown_until,?) WHERE id=1",
                    (now, evidence["next_slot"], evidence["cooldown_until"]),
                )
            db.execute("COMMIT")
        except Exception as error:
            if db.in_transaction:
                db.execute("ROLLBACK")
            if isinstance(error, sqlite3.Error):
                raise AuthorityBlocked("Legacy import ledger unavailable or corrupt") from error
            raise
        finally:
            db.close()

    @classmethod
    def _inspect(cls, db_path: Path, *, writer_id: str, account_id: str
                 ) -> tuple[sqlite3.Connection, "RetainedObservationAuthority"]:
        """Open and semantically validate a ledger without claiming a process."""
        path = Path(db_path)
        db = _read_only_connection(path)
        try:
            account = db.execute("SELECT * FROM retained_account WHERE id=1").fetchone()
            if (account is None or account["writer_id"] != writer_id or
                account["account_id"] != account_id or
                account["db_path"] != str(path.resolve())):
                raise AuthorityBlocked("Retained inspection account or writer differs")
            authority = cls(
                db, path, run_id="inspection", writer_id=writer_id,
                account_id=account_id, manifest={}, manifest_sha256="",
                pair_key="", boot_id=account["boot_id"] or "inspection",
                clock=time.time, monotonic_clock=time.monotonic,
                recovery_only=True,
            )
            authority._validate_all()
            if db.execute("SELECT COUNT(*) FROM retained_import").fetchone()[0] != 1:
                raise AuthorityBlocked("Retained inspection lacks legacy charge import")
            return db, authority
        except Exception:
            db.close()
            raise

    @classmethod
    def list_pairs(cls, db_path: Path, *, writer_id: str,
                   account_id: str) -> tuple[dict, ...]:
        """List retained states plus an unqualified diagnostic source skip marker."""
        db, _ = cls._inspect(db_path, writer_id=writer_id, account_id=account_id)
        try:
            imported = db.execute("SELECT * FROM retained_import").fetchone()
            result = [dict(
                run_id=imported["release_id"], pair_key=None,
                manifest_sha256=imported["manifest_sha256"],
                status="diagnostic_verified_unqualified", mapping_sha256=None,
                bucket=imported["bucket"],
                payload_object=imported["payload_object"],
                payload_generation=imported["payload_generation"],
            )]
            for row in db.execute("SELECT * FROM retained_run ORDER BY started_at,id"):
                manifest, _, _ = _reviewed_manifest(
                    row["manifest_raw"], row["id"], account_id)
                result.append(dict(
                    run_id=row["id"], pair_key=row["pair_key"],
                    manifest_sha256=row["manifest_sha256"], status=row["status"],
                    mapping_sha256=row["mapping_sha256"],
                    bucket=manifest["bucket"],
                    payload_object=manifest["payload"]["object"],
                    payload_generation=manifest["payload"]["generation"],
                ))
            return tuple(result)
        finally:
            db.close()

    @classmethod
    def inspect_pair(cls, db_path: Path, *, writer_id: str, account_id: str,
                     run_id: str, pair_key: str, manifest_sha256: str) -> dict:
        """Read the terminal mapping state for no-upload local cache recovery."""
        db, _ = cls._inspect(db_path, writer_id=writer_id, account_id=account_id)
        try:
            row = db.execute("SELECT * FROM retained_run WHERE id=?", (run_id,)).fetchone()
            if (row is None or row["pair_key"] != pair_key or
                row["manifest_sha256"] != manifest_sha256):
                raise AuthorityBlocked("Retained pair inspection identity mismatch")
            return dict(run_id=run_id, pair_key=pair_key,
                        manifest_sha256=manifest_sha256, status=row["status"],
                        mapping_sha256=row["mapping_sha256"])
        finally:
            db.close()

    @classmethod
    def inspect_account(cls, db_path: Path, *, writer_id: str, account_id: str,
                        clock: Callable[[], float] = time.time) -> dict:
        """Read validated stop gates without claiming the account writer lease."""
        db, authority = cls._inspect(db_path, writer_id=writer_id, account_id=account_id)
        try:
            account = db.execute("SELECT * FROM retained_account WHERE id=1").fetchone()
            now = _finite_nonnegative(clock(), "inspection clock")
            if now < account["last_clock"]:
                raise AuthorityBlocked("Retained inspection clock regressed")
            used = db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_mapping_reservation"
            ).fetchone()[0]
            raw_unresolved = db.execute(
                "SELECT COUNT(*) FROM retained_attempt WHERE status='unresolved'"
            ).fetchone()[0]
            recovery_raw = (db.execute(
                "SELECT COUNT(*) FROM retained_receipt_recovery_attempt "
                "WHERE status='unresolved'").fetchone()[0] if db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                    "name='retained_receipt_recovery_attempt'").fetchone() else 0)
            unresolved = authority._effective_unresolved_count()
            nonterminals = db.execute(
                "SELECT id,status FROM retained_run WHERE status!='closed' ORDER BY started_at,id"
            ).fetchall()
            nonterminal = nonterminals[0] if nonterminals else None
            return {
                "admission_mode": "retained_pair_observation",
                "history_proven": False,
                "diagnostic_imported": True,
                "cooldown_until": account["cooldown_until"],
                "next_slot": account["next_slot"],
                "last_clock": account["last_clock"],
                "unresolved_attempts": unresolved,
                "forensic_raw_unresolved_attempts": raw_unresolved + recovery_raw,
                "mapping_partition_bytes": MAPPING_PARTITION_BYTES,
                "mapping_reserved_total_bytes": used,
                "mapping_partition_remaining_bytes": MAPPING_PARTITION_BYTES - used,
                "known_local_rolling_bytes": authority._known_usage(now),
                "nonterminal_run_id": nonterminal["id"] if nonterminal else None,
                "nonterminal_status": nonterminal["status"] if nonterminal else None,
                "active_run_count": len(nonterminals),
                "nonterminal_run_ids": [row["id"] for row in nonterminals],
            }
        finally:
            db.close()

    @classmethod
    def open_run(
        cls, db_path: Path, *, run_id: str, writer_id: str, account_id: str,
        manifest_bytes: bytes, evidence: dict, deadline_at: float,
        clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        boot_id: str | None = None, recovery_only: bool = False,
        max_concurrent_runs: int = 1,
    ) -> "RetainedObservationAuthority":
        _required_identity(run_id, "run ID")
        _required_identity(writer_id, "writer ID")
        _required_identity(account_id, "account ID")
        if type(max_concurrent_runs) is not int or not 1 <= max_concurrent_runs <= MAX_CONCURRENT_RUNS:
            raise AuthorityBlocked("Retained max_concurrent_runs must be 1..4")
        if recovery_only and max_concurrent_runs != 1:
            raise AuthorityBlocked("Retained recovery_only requires single-run mode")
        manifest, digest, key = _reviewed_manifest(manifest_bytes, run_id, account_id)
        now = _finite_nonnegative(clock(), "clock")
        deadline = _finite_nonnegative(deadline_at, "deadline")
        if not now < deadline <= now + min(MAX_RUN_SECONDS, manifest["max_duration_seconds"]):
            raise AuthorityBlocked("Retained run exceeds reviewed 20-minute deadline")
        _review_evidence(evidence, run_id=run_id, writer_id=writer_id,
                         account_id=account_id, manifest_sha256=digest,
                         now=now, deadline_at=deadline)
        boot = _required_identity(boot_id if boot_id is not None else _boot_id(), "boot identity")
        path = Path(db_path)
        if not path.is_file():
            raise AuthorityBlocked("Missing retained ledger cannot imply empty history")
        db = _connect(path, create=False)
        authority = cls(db, path, run_id=run_id, writer_id=writer_id,
                        account_id=account_id, manifest=manifest,
                        manifest_sha256=digest, pair_key=key, boot_id=boot,
                        clock=clock, monotonic_clock=monotonic_clock,
                        recovery_only=recovery_only,
                        max_concurrent_runs=max_concurrent_runs)
        try:
            with authority._transaction():
                account = authority._account(require_process=False)
                if now < account["last_clock"]:
                    raise AuthorityBlocked("Retained ledger clock regressed")
                if account["boot_id"] is not None and account["boot_id"] != boot and not recovery_only:
                    raise AuthorityBlocked("Retained ledger boot changed; reconciliation required")
                if max_concurrent_runs > 1:
                    authority._ensure_run_claim_table()
                authority._validate_all()
                if db.execute("SELECT COUNT(*) FROM retained_import").fetchone()[0] != 1:
                    raise AuthorityBlocked("Reviewed legacy diagnostic charge is not imported")
                imported = db.execute("SELECT * FROM retained_import").fetchone()
                if (manifest["bucket"], manifest["payload"]["object"],
                    manifest["payload"]["generation"]) == (
                    imported["bucket"], imported["payload_object"],
                    imported["payload_generation"]):
                    raise AuthorityBlocked("Verified diagnostic payload must not be recopied")
                prior = db.execute("SELECT * FROM retained_run ORDER BY started_at DESC").fetchall()
                same = next((r for r in prior if r["id"] == run_id), None)
                if recovery_only:
                    if (same is None or same["manifest_sha256"] != digest or
                        same["pair_key"] != key or same["status"] not in
                        ("verified", "index_pending") or
                        authority._effective_unresolved_count(run_id)):
                        raise AuthorityBlocked("No verified retained pair is eligible for no-upload recovery")
                    if any(r["id"] != run_id and r["status"] != "closed" for r in prior):
                        raise AuthorityBlocked("Another retained run remains open")
                    prior_host, prior_pid, prior_nonce = (
                        account["process_host"], account["process_pid"],
                        account["process_nonce"])
                    if prior_host is not None or prior_pid is not None or prior_nonce is not None:
                        if (prior_host != authority.process_host or
                            type(prior_pid) is not int or prior_pid <= 0):
                            raise AuthorityBlocked("Prior recovery writer cannot be proven stopped")
                        try:
                            os.kill(prior_pid, 0)
                        except ProcessLookupError:
                            pass  # Same-host prior PID is no longer running.
                        except (PermissionError, OSError) as error:
                            raise AuthorityBlocked("Prior recovery writer state is uncertain") from error
                        else:
                            raise AuthorityBlocked("Prior recovery writer may still be running")
                else:
                    if same is not None or any(r["pair_key"] == key for r in prior):
                        raise AuthorityBlocked("Retained run or source pair replay")
                    if any(
                        (old_manifest["bucket"], old_manifest["payload"]["object"],
                         old_manifest["payload"]["generation"]) ==
                        (manifest["bucket"], manifest["payload"]["object"],
                         manifest["payload"]["generation"])
                        for old_manifest, _, _ in (
                            _reviewed_manifest(r["manifest_raw"], r["id"], account_id)
                            for r in prior)
                    ):
                        raise AuthorityBlocked("Retained payload generation already has a run")
                    nonterminal_runs = [r for r in prior if r["status"] != "closed"]
                    if max_concurrent_runs == 1:
                        if nonterminal_runs:
                            raise AuthorityBlocked("Prior retained pair is not index committed")
                        if authority._effective_unresolved_count():
                            raise AuthorityBlocked("Unresolved prior Drive send blocks next pair")
                    else:
                        if len(nonterminal_runs) >= max_concurrent_runs:
                            raise AuthorityBlocked("Prior retained pair is not index committed")
                        for open_r in nonterminal_runs:
                            claim = authority._run_claim(open_r["id"])
                            if (claim is None or
                                claim["max_concurrent_runs"] != max_concurrent_runs or
                                open_r["deadline_at"] <= now or
                                not authority._claim_process_alive(claim)):
                                raise AuthorityBlocked("Prior retained pair is not index committed")
                        if authority._blocking_unresolved_count(
                            now, {r["id"] for r in nonterminal_runs}
                        ):
                            raise AuthorityBlocked("Unresolved prior Drive send blocks next pair")
                    if account["cooldown_until"] > now:
                        raise AuthorityBlocked("Provider quota cooldown remains active")
                    mapping_used = db.execute(
                        "SELECT COALESCE(SUM(bytes),0) FROM retained_mapping_reservation"
                    ).fetchone()[0]
                    unreserved_open = sum(
                        1 for r in nonterminal_runs
                        if not db.execute(
                            "SELECT 1 FROM retained_mapping_reservation WHERE run_id=?",
                            (r["id"],),
                        ).fetchone()
                    )
                    if mapping_used + (unreserved_open + 1) * MAPPING_MAX_BODY_BYTES > MAPPING_PARTITION_BYTES:
                        raise AuthorityBlocked("Fixed GCS mapping partition cannot cover a new pair")
                    db.execute("INSERT INTO retained_run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        run_id, digest, manifest_bytes, key,
                        manifest["payload"]["bytes"] + manifest["receipt"]["bytes"],
                        manifest["control_allowance_bytes"], manifest["max_charged_bytes"],
                        None, 0, 0, "active", None, _canonical(evidence), now, deadline,
                    ))
                    if max_concurrent_runs > 1:
                        db.execute(
                            "INSERT OR REPLACE INTO retained_run_claim VALUES (?,?,?,?,?)",
                            (run_id, authority.process_host, authority.process_pid,
                             authority.process_nonce, max_concurrent_runs),
                        )
                    elif authority._has_run_claim_table():
                        db.execute("DELETE FROM retained_run_claim WHERE run_id=?", (run_id,))
                db.execute(
                    "UPDATE retained_account SET boot_id=COALESCE(boot_id,?),process_host=?,"
                    "process_pid=?,process_nonce=?,last_clock=? WHERE id=1",
                    (boot, authority.process_host, authority.process_pid,
                     authority.process_nonce, now),
                )
            authority._run(allow_recovery=True)
            return authority
        except Exception:
            db.close()
            raise

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        try:
            self.db.execute("BEGIN IMMEDIATE")
            yield
            self.db.execute("COMMIT")
        except Exception as error:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            if isinstance(error, sqlite3.Error):
                raise AuthorityBlocked("Retained authority ledger unavailable or corrupt") from error
            raise

    def _account(self, *, require_process: bool = True) -> sqlite3.Row:
        if os.getpid() != self.process_pid:
            raise AuthorityBlocked("Retained authority handle cannot cross a process fork")
        account = self.db.execute("SELECT * FROM retained_account WHERE id=1").fetchone()
        if (account is None or account["writer_id"] != self.writer_id or
            account["account_id"] != self.account_id or account["db_path"] != self.db_path):
            raise AuthorityBlocked("Retained account, writer or ledger path changed")
        if require_process:
            claim = self._run_claim(self.release_id)
            if claim is not None:
                if ((claim["process_host"], claim["process_pid"], claim["process_nonce"]) !=
                    (self.process_host, self.process_pid, self.process_nonce) or
                    account["process_host"] != self.process_host or
                    account["process_pid"] is None or
                    account["process_nonce"] is None):
                    raise AuthorityBlocked("Retained writer process claim changed")
            elif (
                account["process_host"], account["process_pid"], account["process_nonce"]
            ) != (self.process_host, self.process_pid, self.process_nonce):
                raise AuthorityBlocked("Retained writer process claim changed")
        if account["boot_id"] is not None and account["boot_id"] != self.boot_id and not self.recovery_only:
            raise AuthorityBlocked("Retained boot identity changed")
        if (account["next_slot"] < 0 or account["last_clock"] < 0 or
            account["cooldown_until"] < 0):
            raise AuthorityBlocked("Retained account clock state is malformed")
        return account

    def _reconciled_state(self) -> tuple[set[str], dict[str, int]]:
        """Validate append-only proof rows without altering the original HTTP result.

        Older ledgers have no table until a reviewed administrator performs the
        explicit one-time migration. Merely opening one cannot resolve a send.
        """
        exists = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retained_reconciliation'"
        ).fetchone()
        if exists is None:
            return set(), {}
        columns = tuple(row[1] for row in self.db.execute(
            "PRAGMA table_info(retained_reconciliation)"))
        if columns != ("attempt_id", "run_id", "pair_key", "proof_json",
                       "proof_sha256", "applied_at"):
            raise AuthorityBlocked("Retained reconciliation table differs")
        resolved: set[str] = set()
        credits_by_run: dict[str, int] = {}
        for row in self.db.execute("SELECT * FROM retained_reconciliation"):
            try:
                proof = json.loads(row["proof_json"], object_pairs_hook=_unique_object)
            except (ValueError, TypeError, UnicodeError) as error:
                raise AuthorityBlocked("Retained reconciliation proof is malformed") from error
            schema = proof.get("schema") if isinstance(proof, dict) else None
            if (not isinstance(proof, dict) or
                _canonical(proof) != row["proof_json"] or
                not _valid_digest(row["proof_sha256"]) or
                hashlib.sha256(row["proof_json"].encode()).hexdigest() != row["proof_sha256"] or
                schema not in (RECONCILIATION_SCHEMA,
                               STATUS_PROBE_RECONCILIATION_SCHEMA) or
                proof.get("attempt_id") != row["attempt_id"] or
                proof.get("run_id") != row["run_id"] or
                proof.get("pair_key") != row["pair_key"] or
                _finite_nonnegative(row["applied_at"], "reconciliation time") !=
                    proof.get("applied_at")):
                raise AuthorityBlocked("Retained reconciliation record differs")
            attempt = self.db.execute("SELECT * FROM retained_attempt WHERE id=?",
                                      (row["attempt_id"],)).fetchone()
            run = self.db.execute("SELECT * FROM retained_run WHERE id=?",
                                  (row["run_id"],)).fetchone()
            if attempt is None or run is None:
                raise AuthorityBlocked("Retained reconciliation is orphaned")
            manifest, digest, key = _reviewed_manifest(run["manifest_raw"], run["id"],
                                                       self.account_id)
            receipt = manifest["receipt"]
            payload = manifest["payload"]
            if schema == STATUS_PROBE_RECONCILIATION_SCHEMA:
                probe_id = proof.get("probe_attempt_id")
                probe = (self.db.execute("SELECT * FROM retained_attempt WHERE id=?",
                                         (probe_id,)).fetchone()
                         if isinstance(probe_id, str) else None)
                ack = proof.get("acknowledged_bytes")
                committed = proof.get("committed_bytes")
                credit = proof.get("uncommitted_retry_bytes")
                is_status_attempt = (
                    attempt["kind"] == "resumable_status" and
                    attempt["target_category"] == "resumable_status" and
                    attempt["bytes"] == 0 and committed == 0 and credit == 0
                )
                is_media_attempt = (
                    attempt["kind"] == "media_chunk" and
                    attempt["target_category"] == "media_chunk" and
                    0 < attempt["bytes"] <= (8 << 20) and
                    (probe["http_status"] not in (200, 201) or
                     committed == attempt["bytes"])
                )
                if (attempt["run_id"] != run["id"] or run["pair_key"] != key or
                    row["pair_key"] != key or attempt["status"] != "unresolved" or
                    attempt["http_status"] not in (500, 502, 503, 504) or
                    not (is_media_attempt or is_status_attempt) or
                    attempt["method"] != "PUT" or attempt["dispatched_at"] is None or
                    probe is None or probe["id"] == attempt["id"] or
                    probe["run_id"] != run["id"] or probe["status"] != "settled" or
                    probe["http_status"] not in (200, 201, 308) or
                    probe["kind"] != "resumable_status" or
                    probe["target_category"] != "resumable_status" or
                    probe["method"] != "PUT" or probe["bytes"] != 0 or
                    probe["dispatched_at"] is None or
                    probe["dispatched_at"] < attempt["dispatched_at"] or
                    probe["target_sha256"] != attempt["target_sha256"] or
                    proof.get("manifest_sha256") != digest or
                    proof.get("account_id") != self.account_id or
                    proof.get("writer_id") != self.writer_id or
                    proof.get("original_http_status") != attempt["http_status"] or
                    proof.get("original_charge_bytes") != attempt["bytes"] or
                    proof.get("probe_http_status") != probe["http_status"] or
                    proof.get("target_sha256") != attempt["target_sha256"] or
                    proof.get("operation_key") != attempt["operation_key"] or
                    type(ack) is not int or type(committed) is not int or
                    type(credit) is not int or
                    not 0 <= ack <= max(payload["bytes"], receipt["bytes"]) or
                    not 0 <= committed <= attempt["bytes"] or
                    committed > ack or
                    credit != attempt["bytes"] - committed):
                    raise AuthorityBlocked(
                        "Retained status-probe reconciliation attempt identity differs"
                    )
                resolved.add(row["attempt_id"])
                credits_by_run[run["id"]] = credits_by_run.get(run["id"], 0) + credit
                continue
            is_payload_tail = (
                proof.get("permitted_branch") == "payload_tail_and_receipt_auto"
            )
            parent_id = proof.get("parent_attempt_id")
            parent = (
                self.db.execute(
                    "SELECT * FROM retained_attempt WHERE id=?",
                    (parent_id,),
                ).fetchone()
                if isinstance(parent_id, str) and parent_id != attempt["id"]
                else None
            )
            parent_reconciled = (
                self.db.execute(
                    "SELECT 1 FROM retained_reconciliation WHERE attempt_id=?",
                    (parent_id,),
                ).fetchone()
                is not None
                if parent is not None
                else False
            )
            is_companion_status_probe = (
                attempt["kind"] == "resumable_status" and
                attempt["target_category"] == "resumable_status" and
                attempt["bytes"] == 0 and
                attempt["http_status"] in (404, 410, 500, 502, 503, 504) and
                parent is not None and
                parent_reconciled and
                parent["run_id"] == run["id"] and
                parent["status"] == "unresolved" and
                parent["http_status"] == 500 and
                parent["kind"] == "media_chunk" and
                parent["target_category"] == "media_chunk" and
                parent["method"] == "PUT" and
                parent["target_sha256"] == attempt["target_sha256"] and
                parent["dispatched_at"] is not None and
                attempt["dispatched_at"] is not None and
                attempt["dispatched_at"] >= parent["dispatched_at"] and
                proof.get("original_http_status") == attempt["http_status"] and
                proof.get("original_charge_bytes") == 0 and
                (proof.get("payload_session_sha256") == attempt["target_sha256"]
                 if is_payload_tail else True)
            )
            expected_bytes_ok = (
                (0 < attempt["bytes"] <= min(8 << 20, payload["bytes"]) and
                 proof.get("payload_session_sha256") == attempt["target_sha256"])
                if is_payload_tail else attempt["bytes"] == receipt["bytes"]
            )
            primary_media_ok = (
                attempt["http_status"] == 500 and
                attempt["kind"] == "media_chunk" and
                attempt["target_category"] == "media_chunk" and
                expected_bytes_ok and
                proof.get("original_http_status") == 500 and
                proof.get("original_charge_bytes") == attempt["bytes"]
            )
            if (attempt["run_id"] != run["id"] or run["pair_key"] != key or
                row["pair_key"] != key or attempt["status"] != "unresolved" or
                attempt["method"] != "PUT" or attempt["dispatched_at"] is None or
                not (primary_media_ok or is_companion_status_probe) or
                proof.get("manifest_sha256") != digest or
                proof.get("account_id") != self.account_id or
                proof.get("writer_id") != self.writer_id or
                proof.get("target_sha256") != attempt["target_sha256"] or
                proof.get("operation_key") != attempt["operation_key"] or
                proof.get("receipt_ref") != receipt or
                not _valid_digest(proof.get("mapping_sha256")) or
                not _valid_digest(proof.get("journal_sha256"))):
                raise AuthorityBlocked("Retained reconciliation attempt identity differs")
            for kind in ("payload", "receipt"):
                expected = manifest[kind]
                local = proof.get(kind + "_local")
                gcs = proof.get(kind + "_gcs")
                drive = proof.get(kind + "_drive")
                if not all(isinstance(item, dict) for item in (local, gcs, drive)):
                    raise AuthorityBlocked("Retained reconciliation media proof is absent")
                for observed in (local, gcs, drive):
                    if (observed.get("bytes") != expected["bytes"] or
                        observed.get("sha256") != expected["sha256"] or
                        not isinstance(observed.get("md5"), str) or
                        len(observed["md5"]) != 32 or
                        any(c not in "0123456789abcdef" for c in observed["md5"])):
                        raise AuthorityBlocked("Retained reconciliation media digest differs")
                if (len({local["md5"], gcs["md5"], drive["md5"]}) != 1 or
                    gcs.get("object") != expected["object"] or
                    gcs.get("generation") != expected["generation"] or
                    drive.get("media_readback") is not True or
                    drive.get("account_id") != self.account_id or
                    drive.get("parent_id") != proof.get("folder_id") or
                    drive.get("trashed") is not False or
                    drive.get("id") != proof.get(kind + "_drive_id")):
                    raise AuthorityBlocked("Retained reconciliation source or Drive identity differs")
            if (not isinstance(proof.get("folder_id"), str) or
                not proof["folder_id"] or
                proof.get("payload_drive_id") == proof.get("receipt_drive_id") or
                not isinstance(proof.get("operator_reference"), str) or
                not proof["operator_reference"].strip()):
                raise AuthorityBlocked("Retained reconciliation operator or IDs are absent")
            resolved.add(row["attempt_id"])
        return resolved, credits_by_run

    def _reconciled_attempt_ids(self) -> set[str]:
        return self._reconciled_state()[0]

    def _status_probe_reconciled_credits(self, run_id: str) -> int:
        return self._reconciled_state()[1].get(run_id, 0)

    def _effective_unresolved_count(self, run_id: str | None = None) -> int:
        resolved = self._reconciled_attempt_ids()
        if run_id is None:
            attempts = self.db.execute(
                "SELECT id FROM retained_attempt WHERE status='unresolved'")
        else:
            attempts = self.db.execute(
                "SELECT id FROM retained_attempt WHERE run_id=? AND status='unresolved'",
                (run_id,))
        count = sum(row["id"] not in resolved for row in attempts)
        if self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND "
            "name='retained_receipt_recovery_attempt'").fetchone():
            if run_id is None:
                count += self.db.execute(
                    "SELECT COUNT(*) FROM retained_receipt_recovery_attempt "
                    "WHERE status='unresolved'").fetchone()[0]
            else:
                count += self.db.execute(
                    "SELECT COUNT(*) FROM retained_receipt_recovery_attempt a "
                    "JOIN retained_receipt_recovery r ON r.id=a.recovery_id "
                    "WHERE r.original_run_id=? AND a.status='unresolved'",
                    (run_id,)).fetchone()[0]
        return count

    def _validate_recovery_rows(self) -> None:
        """Include receipt-only recovery in the canonical account authority."""
        tables = {row[0] for row in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND "
            "name IN ('retained_receipt_recovery',"
            "'retained_receipt_recovery_attempt')")}
        if not tables:
            return
        if tables != {"retained_receipt_recovery",
                      "retained_receipt_recovery_attempt"}:
            raise AuthorityBlocked("Partial receipt recovery schema")
        recoveries = self.db.execute(
            "SELECT * FROM retained_receipt_recovery").fetchall()
        if sum(row["status"] == "open" for row in recoveries) > 1:
            raise AuthorityBlocked("Multiple receipt recovery leases are open")
        for row in recoveries:
            run = self.db.execute(
                "SELECT * FROM retained_run WHERE id=?",
                (row["original_run_id"],),
            ).fetchone()
            attempt = self.db.execute(
                "SELECT * FROM retained_attempt WHERE id=?",
                (row["original_attempt_id"],),
            ).fetchone()
            if run is None or attempt is None:
                raise AuthorityBlocked("Receipt recovery is orphaned")
            manifest, digest, key = _reviewed_manifest(
                run["manifest_raw"], run["id"], self.account_id)
            receipt = manifest["receipt"]
            payload = manifest["payload"]
            try:
                evidence = json.loads(row["evidence_json"],
                                      object_pairs_hook=_unique_object)
            except (ValueError, TypeError) as error:
                raise AuthorityBlocked("Receipt recovery review malformed") from error
            is_payload_tail = (
                isinstance(evidence, dict) and
                evidence.get("permitted_branch") == "payload_tail_and_receipt_auto"
            )
            expected_budgets = (
                {
                    payload["bytes"] + receipt["bytes"] + (2 << 20),
                    attempt["bytes"] + receipt["bytes"] + (1 << 20),
                }
                if is_payload_tail else {receipt["bytes"] + (1 << 20)}
            )
            attempt_bytes_ok = (
                0 < attempt["bytes"] <= min(8 << 20, payload["bytes"])
                if is_payload_tail else attempt["bytes"] == receipt["bytes"]
            )
            if (not isinstance(evidence, dict) or
                _canonical(evidence) != row["evidence_json"] or
                evidence.get("schema") !=
                    "catbird.drive.receipt-recovery-review.v1" or
                evidence.get("account_id") != self.account_id or
                evidence.get("writer_id") != self.writer_id or
                evidence.get("original_run_id") != run["id"] or
                evidence.get("original_attempt_id") != attempt["id"] or
                evidence.get("receipt_id") != row["receipt_id"] or
                evidence.get("session_sha256") != row["session_sha256"] or
                evidence.get("manifest_sha256") != digest or
                evidence.get("pair_key") != key or
                attempt["run_id"] != run["id"] or
                attempt["status"] != "unresolved" or
                attempt["http_status"] != 500 or
                attempt["kind"] != "media_chunk" or
                attempt["method"] != "PUT" or
                not attempt_bytes_ok or
                attempt["target_sha256"] != row["session_sha256"] or
                row["manifest_sha256"] != digest or row["pair_key"] != key or
                row["budget_bytes"] not in expected_budgets or
                not 0 <= row["reserved_remaining"] <= row["budget_bytes"] or
                not 0 < row["deadline_at"] - row["started_at"] <= 1200 or
                row["status"] not in ("open", "closed", "blocked")):
                raise AuthorityBlocked("Receipt recovery binding or charge changed")
            attempts = self.db.execute(
                "SELECT * FROM retained_receipt_recovery_attempt "
                "WHERE recovery_id=?", (row["id"],),
            ).fetchall()
            if sum(item["bytes"] for item in attempts) + row["reserved_remaining"] != row["budget_bytes"]:
                raise AuthorityBlocked("Receipt recovery byte reservation changed")
            for item in attempts:
                settled = (item["http_status"] == 308 or
                           item["http_status"] is not None and
                           200 <= item["http_status"] < 300 or
                           item["kind"] == "resumable_status" and
                           item["http_status"] in (404, 410))
                legacy_unadopted_410 = (
                    row["status"] == "open" and
                    row["probe_json"] is None and
                    len(attempts) == 1 and
                    item["kind"] == "resumable_status" and
                    item["http_status"] == 410 and
                    item["bytes"] == 0 and
                    item["status"] == "unresolved"
                )
                if (item["kind"] not in (
                    "resumable_status", "resumable_start", "media_chunk") or
                    item["method"] not in ("PUT", "POST") or
                    item["target_category"] != item["kind"] or
                    not _valid_digest(item["target_sha256"]) or
                    not _valid_digest(item["operation_key"]) or
                    type(item["bytes"]) is not int or item["bytes"] < 0 or
                    item["status"] not in ("settled", "unresolved") or
                    ((item["status"] == "settled") != bool(settled) and
                     not legacy_unadopted_410) or
                    not row["started_at"] <= item["at"] <= item["slot_at"] or
                    item["slot_at"] + item["timeout_s"] >= row["deadline_at"] or
                    item["charge_through"] < item["slot_at"] + item["timeout_s"]):
                    raise AuthorityBlocked("Receipt recovery attempt is malformed")
            if row["status"] == "closed":
                proof = self.db.execute(
                    "SELECT * FROM retained_reconciliation "
                    "WHERE attempt_id=?", (row["original_attempt_id"],),
                ).fetchone()
                if proof is None or run["status"] == "active":
                    raise AuthorityBlocked("Closed receipt recovery lacks accepted proof")

    def _validate_all(self) -> None:
        """Semantic check beyond SQLite integrity before every admission."""
        account = self._account(require_process=False)
        imported = self.db.execute("SELECT * FROM retained_import").fetchall()
        if len(imported) > 1:
            raise AuthorityBlocked("Multiple legacy diagnostic imports")
        for item in imported:
            if (item["source_sha256"] != LEGACY_DIAGNOSTIC_SOURCE_SHA256 or
                item["release_id"] != LEGACY_DIAGNOSTIC_RELEASE_ID or
                item["account_id"] != self.account_id or
                item["bytes"] != LEGACY_DIAGNOSTIC_CHARGED_BYTES or
                item["bucket"] != LEGACY_DIAGNOSTIC_BUCKET or
                item["payload_object"] != DIAGNOSTIC_PAYLOAD_OBJECT or
                item["payload_generation"] != LEGACY_DIAGNOSTIC_PAYLOAD_GENERATION or
                not _valid_digest(item["manifest_sha256"]) or
                not item["original_writer_id"] or
                _finite_nonnegative(item["charge_through"], "import charge horizon") <
                    _finite_nonnegative(item["last_dispatch_wall"], "import dispatch") or
                _finite_nonnegative(item["next_slot"], "import request slot") <
                    item["last_dispatch_wall"] + PACING_SECONDS - 1e-6 or
                _finite_nonnegative(item["cooldown_until"], "import cooldown") < 0 or
                account["next_slot"] < item["next_slot"] or
                account["cooldown_until"] < item["cooldown_until"]):
                raise AuthorityBlocked("Imported diagnostic charge or pacing changed")
        if ((account["last_dispatch_wall"] is None) !=
            (account["last_dispatch_monotonic"] is None)):
            raise AuthorityBlocked("Retained persisted dispatch clocks disagree")
        if account["last_dispatch_wall"] is not None:
            wall = _finite_nonnegative(account["last_dispatch_wall"], "last dispatch wall clock")
            _finite_nonnegative(account["last_dispatch_monotonic"], "last dispatch monotonic clock")
            if (wall > account["last_clock"] or
                account["next_slot"] < wall + PACING_SECONDS - 1e-9):
                raise AuthorityBlocked("Retained persisted next slot is before last dispatch")
        runs = self.db.execute("SELECT * FROM retained_run").fetchall()
        nonterminal_runs = [run for run in runs if run["status"] != "closed"]
        if self._has_run_claim_table():
            claims = {
                row["run_id"]: row
                for row in self.db.execute("SELECT * FROM retained_run_claim").fetchall()
            }
            by_id = {run["id"]: run for run in runs}
            for claimed_id, claim in claims.items():
                claimed_run = by_id.get(claimed_id)
                if (claimed_run is None or claimed_run["status"] == "closed" or
                    type(claim["max_concurrent_runs"]) is not int or
                    not 1 <= claim["max_concurrent_runs"] <= MAX_CONCURRENT_RUNS or
                    not isinstance(claim["process_host"], str) or not claim["process_host"] or
                    type(claim["process_pid"]) is not int or claim["process_pid"] <= 0 or
                    not isinstance(claim["process_nonce"], str) or not claim["process_nonce"]):
                    raise AuthorityBlocked("Retained run claim is orphaned or malformed")
        else:
            claims = {}
        if len(nonterminal_runs) > 1:
            if len(nonterminal_runs) > MAX_CONCURRENT_RUNS:
                raise AuthorityBlocked("Multiple retained runs are nonterminal")
            for open_run in nonterminal_runs:
                claim = claims.get(open_run["id"])
                if (claim is None or
                    claim["max_concurrent_runs"] < 2 or
                    len(nonterminal_runs) > claim["max_concurrent_runs"]):
                    raise AuthorityBlocked("Multiple retained runs are nonterminal")
        payload_identities = set()
        for run in runs:
            manifest, digest, key = _reviewed_manifest(run["manifest_raw"], run["id"], self.account_id)
            source = (manifest["bucket"], manifest["payload"]["object"],
                      manifest["payload"]["generation"])
            if source in payload_identities:
                raise AuthorityBlocked("Retained ledger repeats a payload generation")
            payload_identities.add(source)
            if (run["manifest_sha256"] != digest or run["pair_key"] != key or
                run["media_bytes"] != manifest["payload"]["bytes"] + manifest["receipt"]["bytes"] or
                run["control_allowance"] != manifest["control_allowance_bytes"] or
                run["budget_bytes"] != manifest["max_charged_bytes"] or
                not 0 < run["deadline_at"] - run["started_at"] <= manifest["max_duration_seconds"]):
                raise AuthorityBlocked("Retained run manifest or limits changed")
            evidence = json.loads(run["evidence_json"])
            _review_evidence(evidence, run_id=run["id"], writer_id=self.writer_id,
                             account_id=self.account_id, manifest_sha256=digest,
                             now=run["started_at"], deadline_at=run["deadline_at"])
            if (run["reserved"] not in (0, 1) or
                not 0 <= run["reserved_remaining"] <= run["budget_bytes"]):
                raise AuthorityBlocked("Retained reservation is malformed")
            if run["reserved"]:
                if json.loads(run["metadata_json"]) != _expected_metadata(manifest, digest):
                    raise AuthorityBlocked("Retained pair metadata changed")
            elif run["metadata_json"] is not None or run["reserved_remaining"] != 0:
                raise AuthorityBlocked("Unreserved retained pair has a charge")
            attempts = self.db.execute("SELECT * FROM retained_attempt WHERE run_id=?", (run["id"],)).fetchall()
            spent = 0
            for attempt in attempts:
                spent += self._validate_attempt(attempt, run)
            net_spent = spent - self._status_probe_reconciled_credits(run["id"])
            if net_spent < 0:
                raise AuthorityBlocked("Retained pair status-probe credit exceeds spend")
            if run["status"] in ("active", "verified") and run["reserved"]:
                if net_spent + run["reserved_remaining"] != run["budget_bytes"]:
                    raise AuthorityBlocked("Retained pair charges do not reconcile")
            elif run["status"] in ("index_pending", "closed"):
                if (not run["reserved"] or run["reserved_remaining"] != 0 or
                    not _valid_digest(run["mapping_sha256"]) or net_spent > run["budget_bytes"]):
                    raise AuthorityBlocked("Retained index state or charge is malformed")
                if self._effective_unresolved_count(run["id"]):
                    raise AuthorityBlocked("Index pending with uncertain Drive attempt")
            elif spent:
                raise AuthorityBlocked("Unreserved retained pair has attempts")
            if run["status"] in ("verified", "index_pending", "closed") and not _valid_digest(run["mapping_sha256"]):
                raise AuthorityBlocked("Verified retained mapping digest missing")
        mappings = self.db.execute("SELECT * FROM retained_mapping_reservation").fetchall()
        if sum(item["bytes"] for item in mappings) > MAPPING_PARTITION_BYTES:
            raise AuthorityBlocked("Retained mapping partition exceeded")
        by_run = {run["id"]: run for run in runs}
        for item in mappings:
            run = by_run.get(item["run_id"])
            reviewed = (_reviewed_manifest(run["manifest_raw"], run["id"], self.account_id)[0]
                        if run is not None else None)
            if (run is None or run["status"] not in ("index_pending", "closed") or
                item["pair_key"] != run["pair_key"] or
                item["mapping_sha256"] != run["mapping_sha256"] or
                item["object_name"] != reviewed["index_object"] or
                not isinstance(item["object_name"], str) or
                not item["object_name"] or len(item["object_name"]) > 1024 or
                any(ord(c) < 32 for c in item["object_name"]) or
                type(item["bytes"]) is not int or
                not 0 < item["bytes"] <= MAPPING_MAX_BODY_BYTES or
                not _valid_digest(item["reservation_id"])):
                raise AuthorityBlocked("Retained mapping reservation is malformed")
        if any(run["status"] == "closed" and not any(
            item["run_id"] == run["id"] for item in mappings) for run in runs):
            raise AuthorityBlocked("Closed retained pair has no mapping reservation")
        self._validate_recovery_rows()
        all_attempts = self.db.execute("SELECT * FROM retained_attempt").fetchall()
        if len(all_attempts) != sum(self.db.execute(
            "SELECT COUNT(*) FROM retained_attempt WHERE run_id=?", (run["id"],)).fetchone()[0]
            for run in runs):
            raise AuthorityBlocked("Orphan retained attempt")
        if self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND "
            "name='retained_receipt_recovery_attempt'").fetchone():
            recovery_attempts = self.db.execute(
                "SELECT * FROM retained_receipt_recovery_attempt").fetchall()
            all_attempts += recovery_attempts
        dispatched = sorted(
            (a for a in all_attempts if a["dispatch_monotonic"] is not None),
            key=lambda a: (a["dispatched_at"], a["dispatch_monotonic"]),
        )
        for a, b in zip(dispatched, dispatched[1:]):
            owner_a = (
                ("run", a["run_id"])
                if "run_id" in a.keys()
                else ("recovery", a["recovery_id"])
            )
            owner_b = (
                ("run", b["run_id"])
                if "run_id" in b.keys()
                else ("recovery", b["recovery_id"])
            )
            if b["dispatched_at"] - a["dispatched_at"] < PACING_SECONDS - 1e-6:
                raise AuthorityBlocked("Retained actual starts violate monotonic pacing")
            if (
                owner_a == owner_b and
                b["dispatch_monotonic"] - a["dispatch_monotonic"] < PACING_SECONDS - 1e-9
            ):
                raise AuthorityBlocked("Retained actual starts violate monotonic pacing")

    @staticmethod
    def _validate_attempt(attempt: sqlite3.Row, run: sqlite3.Row) -> int:
        amount = attempt["bytes"]
        if (not isinstance(amount, int) or amount < 0 or
            attempt["method"] not in ("GET", "POST", "PUT", "PATCH", "DELETE") or
            not _valid_digest(attempt["target_sha256"]) or
            not _valid_digest(attempt["operation_key"])):
            raise AuthorityBlocked("Malformed retained request identity or charge")
        _required_identity(attempt["kind"], "request kind")
        _required_identity(attempt["target_category"], "request category")
        at = _finite_nonnegative(attempt["at"], "attempt start")
        slot = _finite_nonnegative(attempt["slot_at"], "request slot")
        through = _finite_nonnegative(attempt["charge_through"], "charge-through time")
        timeout = _finite_nonnegative(attempt["timeout_s"], "request timeout")
        if (timeout <= 0 or not run["started_at"] <= at <= slot or
            slot + timeout >= run["deadline_at"] or through < slot + timeout):
            raise AuthorityBlocked("Malformed retained attempt timing")
        dispatch = attempt["dispatched_at"]
        mono = attempt["dispatch_monotonic"]
        if (dispatch is None) != (mono is None):
            raise AuthorityBlocked("Retained dispatch clocks disagree")
        if dispatch is not None:
            dispatch = _finite_nonnegative(dispatch, "actual dispatch")
            _finite_nonnegative(mono, "monotonic dispatch")
            if (dispatch < slot or dispatch + timeout >= run["deadline_at"] or
                through < dispatch + timeout):
                raise AuthorityBlocked("Malformed retained dispatch timing")
        http = attempt["http_status"]
        if http is not None and (not isinstance(http, int) or not 100 <= http <= 599):
            raise AuthorityBlocked("Malformed retained HTTP status")
        settled = http == 308 or http is not None and 200 <= http < 300
        legacy_or_settled_status_probe = (
            attempt["kind"] == "resumable_status" and
            amount == 0 and
            http in (404, 410)
        )
        if not legacy_or_settled_status_probe and (
            attempt["status"] == "settled" and not settled or
            attempt["status"] == "unresolved" and settled):
            raise AuthorityBlocked("Retained attempt settlement is inconsistent")
        return amount

    def _run(self, *, allow_recovery: bool = False) -> tuple[sqlite3.Row, sqlite3.Row]:
        account = self._account()
        self._validate_all()
        run = self.db.execute("SELECT * FROM retained_run WHERE id=?", (self.release_id,)).fetchone()
        if (run is None or run["manifest_sha256"] != self.manifest_sha256 or
            run["pair_key"] != self.expected_pair_key):
            raise AuthorityBlocked("Retained run binding changed")
        if self.recovery_only and not allow_recovery:
            raise AuthorityBlocked("Recovery-only handle cannot send Drive requests")
        if run["status"] != "active" and not allow_recovery:
            raise AuthorityBlocked("Retained pair is no longer active for Drive requests")
        return account, run

    def _now(self, account: sqlite3.Row) -> float:
        now = _finite_nonnegative(self.clock(), "clock")
        if now < account["last_clock"]:
            raise AuthorityBlocked("Retained ledger clock regressed")
        return now

    def _active(self, account: sqlite3.Row, run: sqlite3.Row, now: float,
                timeout_s: float = 0, slot_at: float | None = None) -> None:
        timeout = _finite_nonnegative(timeout_s, "request timeout")
        if now >= run["deadline_at"] or account["cooldown_until"] > now:
            raise AuthorityBlocked("Retained run deadline or quota cooldown blocks Drive")
        if max(now, slot_at if slot_at is not None else now) + timeout >= run["deadline_at"]:
            raise AuthorityBlocked("Retained request could exceed run deadline")

    def assert_active(self, timeout_s: float = 0, *, slot_at: float | None = None) -> None:
        account, run = self._run()
        now = self._now(account)
        if slot_at is not None and now < _finite_nonnegative(slot_at, "request slot"):
            raise AuthorityBlocked("Retained request slot has not arrived")
        self._active(account, run, now, timeout_s, slot_at)

    def validate_manifest_limits(self, max_charged_bytes: int, max_duration_seconds: int) -> None:
        account, run = self._run()
        if (_positive_int(max_charged_bytes, "manifest byte limit") != run["budget_bytes"] or
            _positive_int(max_duration_seconds, "manifest duration") <
            run["deadline_at"] - run["started_at"]):
            raise AuthorityBlocked("Retained run differs from reviewed manifest limits")
        self._active(account, run, self._now(account))

    def _known_usage(self, now: float) -> int:
        charges = self.db.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM retained_attempt "
            "WHERE status='unresolved' OR charge_through>?", (now - WINDOW_SECONDS,),
        ).fetchone()[0]
        reserved = self.db.execute(
            "SELECT COALESCE(SUM(reserved_remaining),0) FROM retained_run "
            "WHERE status IN ('active','verified')", 
        ).fetchone()[0]
        imported = self.db.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM retained_import WHERE charge_through>?",
            (now - WINDOW_SECONDS,),
        ).fetchone()[0]
        recovery = (self.db.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM retained_receipt_recovery_attempt "
            "WHERE status='unresolved' OR charge_through>?",
            (now - WINDOW_SECONDS,),
        ).fetchone()[0] if self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND "
            "name='retained_receipt_recovery_attempt'").fetchone() else 0)
        recovery_reserved = (self.db.execute(
            "SELECT COALESCE(SUM(reserved_remaining),0) "
            "FROM retained_receipt_recovery WHERE status='open'",
        ).fetchone()[0] if self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND "
            "name='retained_receipt_recovery'").fetchone() else 0)
        return (int(charges) + int(reserved) + int(imported) +
                int(recovery) + int(recovery_reserved))

    def reserve_pair(self, pair_key: str, media_bytes: int, control_allowance: int,
                     metadata: dict) -> None:
        with self._transaction():
            account, run = self._run()
            now = self._now(account)
            self._active(account, run, now)
            if (pair_key != self.expected_pair_key or
                media_bytes != run["media_bytes"] or
                control_allowance != run["control_allowance"] or
                metadata != _expected_metadata(self.manifest, self.manifest_sha256)):
                raise AuthorityBlocked("Retained reservation differs from reviewed pair")
            if run["reserved"]:
                if run["metadata_json"] != _canonical(metadata):
                    raise AuthorityBlocked("Retained reservation changed on replay")
                return
            if run["budget_bytes"] > PILOT_CAP_BYTES:
                raise AuthorityBlocked("Retained pair exceeds reviewed pilot cap")
            if self._known_usage(now) + run["budget_bytes"] > GLOBAL_ROLLING_CAP_BYTES:
                raise AuthorityBlocked("Known account charges exceed 700 GiB rolling cap")
            self.db.execute(
                "UPDATE retained_run SET reserved=1,reserved_remaining=?,metadata_json=? WHERE id=?",
                (run["budget_bytes"], _canonical(metadata), self.release_id),
            )
            self.db.execute("UPDATE retained_account SET last_clock=? WHERE id=1", (now,))

    def start_request(self, body_bytes: int, timeout_s: float, kind: str, *,
                      method: str, target_category: str, target_sha256: str,
                      operation_key: str) -> tuple[str, float]:
        if type(body_bytes) is not int or body_bytes < 0:
            raise AuthorityBlocked("Invalid retained request body length")
        _required_identity(kind, "request kind")
        _required_identity(target_category, "target category")
        if (method not in ("GET", "POST", "PUT", "PATCH", "DELETE") or
            not _valid_digest(target_sha256) or not _valid_digest(operation_key) or
            _finite_nonnegative(timeout_s, "request timeout") <= 0):
            raise AuthorityBlocked("Retained request identity or timeout invalid")
        with self._transaction():
            account, run = self._run()
            now = self._now(account)
            slot = max(now, account["next_slot"])
            self._active(account, run, now, timeout_s, slot)
            if not run["reserved"] or body_bytes > run["reserved_remaining"]:
                raise AuthorityBlocked("Retained request exceeds pre-send reservation")
            if self._known_usage(now) > GLOBAL_ROLLING_CAP_BYTES:
                raise AuthorityBlocked("Known account charges exceed 700 GiB rolling cap")
            attempt_id = uuid.uuid4().hex
            self.db.execute("UPDATE retained_run SET reserved_remaining=reserved_remaining-? WHERE id=?",
                            (body_bytes, self.release_id))
            self.db.execute("INSERT INTO retained_attempt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                attempt_id, self.release_id, now, slot, slot + timeout_s,
                timeout_s, None, None, body_bytes, kind, method, target_category,
                target_sha256, operation_key, "unresolved", None,
            ))
            self.db.execute("UPDATE retained_account SET last_clock=?,next_slot=? WHERE id=1",
                            (now, slot + PACING_SECONDS))
            return attempt_id, slot

    def seconds_until_dispatch(self, attempt_id: str) -> float:
        """Read the wait for one precharged attempt; zero means ready to recheck.

        This does not dispatch, settle, or reserve another attempt. The caller
        must keep its process dispatch lock, sleep, then use mark_dispatch on
        this same ID immediately before its one network send.
        """
        _required_identity(attempt_id, "attempt ID")
        account, run = self._run()
        now = self._now(account)
        mono = _finite_nonnegative(self.monotonic_clock(), "monotonic clock")
        attempt = self.db.execute(
            "SELECT * FROM retained_attempt WHERE id=? AND run_id=?",
            (attempt_id, self.release_id),
        ).fetchone()
        if (attempt is None or attempt["status"] != "unresolved" or
            attempt["dispatched_at"] is not None):
            raise AuthorityBlocked("Retained attempt is not awaiting dispatch")
        self._active(account, run, now, attempt["timeout_s"], attempt["slot_at"])
        wall_wait = max(0.0, attempt["slot_at"] - now)
        if account["last_dispatch_wall"] is not None:
            wall_wait = max(wall_wait,
                            account["last_dispatch_wall"] + PACING_SECONDS - now)
        mono_wait = 0.0
        if account["last_dispatch_monotonic"] is not None:
            if mono < account["last_dispatch_monotonic"]:
                raise AuthorityBlocked("Retained monotonic clock regressed")
            mono_wait = max(0.0,
                            account["last_dispatch_monotonic"] + PACING_SECONDS - mono)
        wait = max(wall_wait, mono_wait)
        if now + wait + attempt["timeout_s"] >= run["deadline_at"]:
            raise AuthorityBlocked("Retained dispatch wait cannot fit run deadline")
        return wait

    def mark_dispatch(self, attempt_id: str) -> None:
        _required_identity(attempt_id, "attempt ID")
        with self._transaction():
            account, run = self._run()
            now = self._now(account)
            mono = _finite_nonnegative(self.monotonic_clock(), "monotonic clock")
            attempt = self.db.execute(
                "SELECT * FROM retained_attempt WHERE id=? AND run_id=?",
                (attempt_id, self.release_id),
            ).fetchone()
            if (attempt is None or attempt["status"] != "unresolved" or
                attempt["dispatched_at"] is not None or now < attempt["slot_at"]):
                raise AuthorityBlocked("Retained attempt cannot be dispatched")
            self._active(account, run, now, attempt["timeout_s"], attempt["slot_at"])
            if ((account["last_dispatch_wall"] is not None and
                 now < account["last_dispatch_wall"] + PACING_SECONDS) or
                (account["last_dispatch_monotonic"] is not None and
                 mono < account["last_dispatch_monotonic"] + PACING_SECONDS)):
                raise AuthorityBlocked("Retained actual dispatch would violate global pacing")
            self.db.execute("UPDATE retained_attempt SET dispatched_at=?,dispatch_monotonic=?,"
                            "charge_through=MAX(charge_through,?) WHERE id=?",
                            (now, mono, now + attempt["timeout_s"], attempt_id))
            self.db.execute("UPDATE retained_account SET last_clock=?,"
                            "next_slot=MAX(next_slot,?),last_dispatch_wall=?,"
                            "last_dispatch_monotonic=? WHERE id=1",
                            (now, now + PACING_SECONDS, now, mono))

    def finish_request(self, attempt_id: str, status_code: int | None) -> None:
        _required_identity(attempt_id, "attempt ID")
        if status_code is not None and (type(status_code) is not int or not 100 <= status_code <= 599):
            raise AuthorityBlocked("Invalid retained HTTP result")
        with self._transaction():
            account, _ = self._run()
            now = self._now(account)
            attempt = self.db.execute(
                "SELECT * FROM retained_attempt WHERE id=? AND run_id=?",
                (attempt_id, self.release_id),
            ).fetchone()
            if attempt is None:
                raise AuthorityBlocked("Unknown retained request attempt")
            if attempt["status"] == "settled" and attempt["http_status"] != status_code:
                raise AuthorityBlocked("Retained attempt result changed")
            settled = (
                status_code == 308 or
                status_code is not None and 200 <= status_code < 300 or
                attempt["kind"] == "resumable_status" and
                attempt["bytes"] == 0 and
                status_code in (404, 410)
            )
            self.db.execute("UPDATE retained_attempt SET status=?,http_status=?,"
                            "charge_through=MAX(charge_through,?) WHERE id=?",
                            ("settled" if settled else "unresolved", status_code, now, attempt_id))
            if not settled and self._has_run_claim_table():
                self.db.execute(
                    "UPDATE retained_run_claim SET max_concurrent_runs=1 WHERE run_id=?",
                    (self.release_id,),
                )
            if status_code in (403, 429):
                self.db.execute("UPDATE retained_account SET cooldown_until=MAX(cooldown_until,?) WHERE id=1",
                                (now + QUOTA_COOLDOWN_SECONDS,))
            self.db.execute("UPDATE retained_account SET last_clock=? WHERE id=1", (now,))

    def reconcile_attempt_by_status_probe(
        self, *, failed_attempt_id: str, probe_attempt_id: str,
        acknowledged_bytes: int, committed_bytes: int,
    ) -> str:
        """Reconcile a transient 5xx media_chunk attempt after a settled status probe.

        The original retained_attempt row and its byte charge are never rewritten.
        An append-only proof row records the settled status probe and credits any
        uncommitted retry bytes back to the active run's pair budget so the retry
        chunk can be reserved without double-charging the per-pair cap, while the
        700 GiB rolling account cap still conservatively counts all wire bytes.
        """
        _required_identity(failed_attempt_id, "failed attempt ID")
        _required_identity(probe_attempt_id, "probe attempt ID")
        if failed_attempt_id == probe_attempt_id:
            raise AuthorityBlocked("Status probe reconciliation requires distinct attempts")
        with self._transaction():
            account, run = self._run()
            now = self._now(account)
            if not run["reserved"]:
                raise AuthorityBlocked("Status probe reconciliation requires reserved pair")
            failed = self.db.execute(
                "SELECT * FROM retained_attempt WHERE id=? AND run_id=?",
                (failed_attempt_id, self.release_id),
            ).fetchone()
            probe = self.db.execute(
                "SELECT * FROM retained_attempt WHERE id=? AND run_id=?",
                (probe_attempt_id, self.release_id),
            ).fetchone()
            prior = self.db.execute(
                "SELECT 1 FROM retained_reconciliation WHERE attempt_id=?",
                (failed_attempt_id,),
            ).fetchone()
            max_media = max(self.manifest["payload"]["bytes"],
                            self.manifest["receipt"]["bytes"])
            if (prior is not None or failed is None or probe is None or
                failed["status"] != "unresolved" or
                failed["http_status"] not in (500, 502, 503, 504) or
                failed["kind"] != "media_chunk" or
                failed["target_category"] != "media_chunk" or
                failed["method"] != "PUT" or failed["dispatched_at"] is None or
                not 0 < failed["bytes"] <= (8 << 20) or
                probe["status"] != "settled" or
                probe["http_status"] not in (200, 201, 308) or
                probe["kind"] != "resumable_status" or
                probe["target_category"] != "resumable_status" or
                probe["method"] != "PUT" or probe["bytes"] != 0 or
                probe["dispatched_at"] is None or
                probe["dispatched_at"] < failed["dispatched_at"] or
                probe["target_sha256"] != failed["target_sha256"] or
                type(acknowledged_bytes) is not int or
                type(committed_bytes) is not int or
                not 0 <= acknowledged_bytes <= max_media or
                not 0 <= committed_bytes <= failed["bytes"] or
                committed_bytes > acknowledged_bytes or
                (probe["http_status"] in (200, 201) and
                 committed_bytes != failed["bytes"])):
                raise AuthorityBlocked(
                    "Transient 5xx media attempt cannot be reconciled by status probe"
                )
            uncommitted_retry_bytes = failed["bytes"] - committed_bytes
            proof = {
                "schema": STATUS_PROBE_RECONCILIATION_SCHEMA,
                "attempt_id": failed_attempt_id,
                "probe_attempt_id": probe_attempt_id,
                "run_id": self.release_id,
                "pair_key": self.expected_pair_key,
                "manifest_sha256": self.manifest_sha256,
                "account_id": self.account_id,
                "writer_id": self.writer_id,
                "original_http_status": failed["http_status"],
                "original_charge_bytes": failed["bytes"],
                "probe_http_status": probe["http_status"],
                "target_sha256": failed["target_sha256"],
                "operation_key": failed["operation_key"],
                "acknowledged_bytes": acknowledged_bytes,
                "committed_bytes": committed_bytes,
                "uncommitted_retry_bytes": uncommitted_retry_bytes,
                "applied_at": now,
            }
            raw = _canonical(proof)
            digest = hashlib.sha256(raw.encode()).hexdigest()
            self.db.execute(
                "INSERT INTO retained_reconciliation VALUES (?,?,?,?,?,?)",
                (failed_attempt_id, self.release_id, self.expected_pair_key,
                 raw, digest, now),
            )
            intermediate_probes = self.db.execute(
                "SELECT a.* FROM retained_attempt a LEFT JOIN "
                "retained_reconciliation r ON r.attempt_id=a.id "
                "WHERE a.run_id=? AND a.target_sha256=? AND a.status='unresolved' "
                "AND a.kind='resumable_status' AND a.bytes=0 AND r.attempt_id IS NULL",
                (self.release_id, failed["target_sha256"]),
            ).fetchall()
            for item in intermediate_probes:
                if (item["http_status"] in (500, 502, 503, 504) and
                    item["dispatched_at"] is not None and
                    failed["dispatched_at"] <= item["dispatched_at"] <=
                        probe["dispatched_at"]):
                    p_proof = {
                        "schema": STATUS_PROBE_RECONCILIATION_SCHEMA,
                        "attempt_id": item["id"],
                        "probe_attempt_id": probe_attempt_id,
                        "run_id": self.release_id,
                        "pair_key": self.expected_pair_key,
                        "manifest_sha256": self.manifest_sha256,
                        "account_id": self.account_id,
                        "writer_id": self.writer_id,
                        "original_http_status": item["http_status"],
                        "original_charge_bytes": 0,
                        "probe_http_status": probe["http_status"],
                        "target_sha256": item["target_sha256"],
                        "operation_key": item["operation_key"],
                        "acknowledged_bytes": acknowledged_bytes,
                        "committed_bytes": 0,
                        "uncommitted_retry_bytes": 0,
                        "applied_at": now,
                    }
                    p_raw = _canonical(p_proof)
                    p_digest = hashlib.sha256(p_raw.encode()).hexdigest()
                    self.db.execute(
                        "INSERT INTO retained_reconciliation VALUES (?,?,?,?,?,?)",
                        (item["id"], self.release_id, self.expected_pair_key,
                         p_raw, p_digest, now),
                    )
            self.db.execute(
                "UPDATE retained_run SET reserved_remaining=reserved_remaining+? "
                "WHERE id=?",
                (uncommitted_retry_bytes, self.release_id),
            )
            if (self._has_run_claim_table() and self.max_concurrent_runs > 1 and
                self._effective_unresolved_count(self.release_id) == 0):
                self.db.execute(
                    "UPDATE retained_run_claim SET max_concurrent_runs=? WHERE run_id=?",
                    (self.max_concurrent_runs, self.release_id),
                )
            self.db.execute(
                "UPDATE retained_account SET last_clock=? WHERE id=1", (now,),
            )
            self._validate_all()
            return digest

    def mark_pair_verified(self, pair_key: str, mapping_sha256: str) -> None:
        if not _valid_digest(mapping_sha256):
            raise AuthorityBlocked("Verified mapping digest missing")
        with self._transaction():
            account, run = self._run()
            now = self._now(account)
            if pair_key != self.expected_pair_key or not run["reserved"]:
                raise AuthorityBlocked("Verified retained pair differs from reservation")
            if self._effective_unresolved_count(self.release_id):
                raise AuthorityBlocked("Unresolved sends block retained verification")
            self.db.execute("UPDATE retained_run SET status='verified',mapping_sha256=? WHERE id=?",
                            (mapping_sha256, self.release_id))
            self.db.execute("UPDATE retained_account SET last_clock=? WHERE id=1", (now,))

    def close_pair(self, pair_key: str) -> None:
        """Move verified pair to index_pending; next pair stays blocked."""
        with self._transaction():
            account, run = self._run(allow_recovery=True)
            now = self._now(account)
            if pair_key != self.expected_pair_key or run["status"] != "verified":
                raise AuthorityBlocked("Retained pair is not verified for index publication")
            if self._effective_unresolved_count(self.release_id):
                raise AuthorityBlocked("Unresolved sends block retained closure")
            self.db.execute("UPDATE retained_run SET status='index_pending',reserved_remaining=0 WHERE id=?",
                            (self.release_id,))
            self.db.execute("UPDATE retained_account SET last_clock=? WHERE id=1", (now,))

    def reserve_index_create(self, pair_key: str, mapping_sha256: str,
                             object_name: str, body_bytes: int) -> str:
        """Durably charge a create attempt to the fixed GCS mapping partition.

        An ambiguous outcome never refunds the reservation. Exact replay uses
        the original reservation ID and does not charge twice.
        """
        if (not _valid_digest(mapping_sha256) or
            not isinstance(object_name, str) or not object_name or
            len(object_name) > 1024 or any(ord(c) < 32 for c in object_name) or
            object_name != self.manifest["index_object"] or
            type(body_bytes) is not int or
            not 0 < body_bytes <= MAPPING_MAX_BODY_BYTES):
            raise AuthorityBlocked("Invalid mapping create identity or size")
        with self._transaction():
            account, run = self._run(allow_recovery=True)
            now = self._now(account)
            if (run["status"] != "index_pending" or pair_key != self.expected_pair_key or
                mapping_sha256 != run["mapping_sha256"]):
                raise AuthorityBlocked("Mapping create is not bound to pending verified pair")
            prior = self.db.execute(
                "SELECT * FROM retained_mapping_reservation WHERE run_id=?",
                (self.release_id,),
            ).fetchone()
            if prior is not None:
                if (prior["pair_key"], prior["mapping_sha256"],
                    prior["object_name"], prior["bytes"]) != (
                    pair_key, mapping_sha256, object_name, body_bytes):
                    raise AuthorityBlocked("Mapping create replay differs from reservation")
                return prior["reservation_id"]
            used = self.db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_mapping_reservation",
            ).fetchone()[0]
            if used + body_bytes > MAPPING_PARTITION_BYTES:
                raise AuthorityBlocked("Fixed GCS mapping partition is full")
            reservation_id = uuid.uuid4().hex + uuid.uuid4().hex
            self.db.execute(
                "INSERT INTO retained_mapping_reservation VALUES (?,?,?,?,?,?)",
                (self.release_id, pair_key, mapping_sha256, object_name,
                 body_bytes, reservation_id),
            )
            self.db.execute("UPDATE retained_account SET last_clock=? WHERE id=1", (now,))
            return reservation_id

    def mark_index_committed(self, pair_key: str, mapping_sha256: str) -> None:
        """Called only after a create-only shared index commit is durably verified."""
        with self._transaction():
            account, run = self._run(allow_recovery=True)
            now = self._now(account)
            reservation = self.db.execute(
                "SELECT * FROM retained_mapping_reservation WHERE run_id=?",
                (self.release_id,),
            ).fetchone()
            if (pair_key != self.expected_pair_key or run["status"] != "index_pending" or
                run["mapping_sha256"] != mapping_sha256 or reservation is None or
                reservation["pair_key"] != pair_key or
                reservation["mapping_sha256"] != mapping_sha256):
                raise AuthorityBlocked("Shared index commit does not match verified mapping")
            self.db.execute("UPDATE retained_run SET status='closed' WHERE id=?", (self.release_id,))
            remaining_claim = None
            if self._has_run_claim_table():
                self.db.execute("DELETE FROM retained_run_claim WHERE run_id=?", (self.release_id,))
                remaining_claim = self.db.execute(
                    "SELECT process_host,process_pid,process_nonce "
                    "FROM retained_run_claim ORDER BY rowid LIMIT 1"
                ).fetchone()
            if remaining_claim is not None:
                self.db.execute(
                    "UPDATE retained_account SET process_host=?,process_pid=?,"
                    "process_nonce=?,last_clock=? WHERE id=1",
                    (remaining_claim["process_host"], remaining_claim["process_pid"],
                     remaining_claim["process_nonce"], now),
                )
            else:
                self.db.execute("UPDATE retained_account SET process_host=NULL,process_pid=NULL,"
                                "process_nonce=NULL,last_clock=? WHERE id=1", (now,))

    def pair_status(self, pair_key: str, manifest_sha256: str) -> str:
        self._validate_all()
        run = self.db.execute("SELECT * FROM retained_run WHERE id=?", (self.release_id,)).fetchone()
        if (run is None or pair_key != self.expected_pair_key or
            manifest_sha256 != self.manifest_sha256 or run["pair_key"] != pair_key or
            run["manifest_sha256"] != manifest_sha256):
            raise AuthorityBlocked("Retained pair status identity mismatch")
        return run["status"]

    def snapshot(self) -> dict:
        account = self._account(require_process=False)
        self._validate_all()
        run = self.db.execute("SELECT * FROM retained_run WHERE id=?", (self.release_id,)).fetchone()
        now = self._now(account)
        return {
            "admission_mode": "retained_pair_observation", "history_proven": False,
            "known_local_rolling_bytes": self._known_usage(now),
            "effective_rolling_cap_bytes": None,
            "mapping_reserved_total_bytes": self.db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_mapping_reservation",
            ).fetchone()[0],
            "mapping_partition_bytes": MAPPING_PARTITION_BYTES,
            "pair_reserved_remaining": run["reserved_remaining"],
            "pair_status": run["status"], "next_slot": account["next_slot"],
            "cooldown_until": account["cooldown_until"],
            "unresolved_attempts": self._effective_unresolved_count(),
            "forensic_raw_unresolved_attempts": self.db.execute(
                "SELECT COUNT(*) FROM retained_attempt WHERE status='unresolved'").fetchone()[0],
        }

    def close(self) -> None:
        self.db.close()


def _offline_setup_cli() -> int:
    """Initialize/import a ledger offline; transfer and credentials are absent."""
    import argparse

    parser = argparse.ArgumentParser(description="Offline retained-observation ledger setup")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("initialize", "import-legacy"):
        command = commands.add_parser(action)
        command.add_argument("--db", type=Path, required=True)
        command.add_argument("--writer-id", required=True)
        command.add_argument("--account-id", required=True)
        if action == "import-legacy":
            command.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "initialize":
            RetainedObservationAuthority.initialize(
                args.db, writer_id=args.writer_id, account_id=args.account_id)
            print("Retained observation ledger initialized; legacy charge import is required.")
        else:
            RetainedObservationAuthority.import_legacy_result(
                args.db, result_bytes=args.result.read_bytes(),
                writer_id=args.writer_id, account_id=args.account_id)
            print("Pinned diagnostic charge import confirmed; account history remains unproven.")
    except (AuthorityBlocked, OSError) as error:
        parser.exit(2, f"Offline ledger setup blocked: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_offline_setup_cli())
