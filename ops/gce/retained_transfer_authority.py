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
import bisect
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
CRASH_ADOPTION_RECONCILIATION_SCHEMA = (
    "catbird.drive.retained-crash-adoption-reconciliation.v1"
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
                 recovery_only: bool, max_concurrent_runs: int = 1,
                 resume_open: bool = False):
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
        self.resume_open = resume_open
        self.process_host = socket.gethostname()
        self.process_pid = os.getpid()
        self.process_nonce = _PROCESS_NONCE
        self._closed_fingerprint: dict | None = None
        self._guarded_total_changes: int | None = None
        self._guarded_data_version: int | None = None
        self._last_validated_payload_identities: set[tuple[str, str, int]] = set()
        self._validation_stats: dict[str, int] = {
            "fast_checks": 0,
            "fast_hits": 0,
            "fallbacks": 0,
            "total_changes": 0,
            "data_version": 0,
            "cheap_fingerprint": 0,
            "uninitialized": 0,
            "full_validate_calls": 0,
            "open_run_full_validates": 0,
            "close_pair_full_validates": 0,
        }

    def validation_stats(self) -> dict[str, int]:
        return dict(self._validation_stats)

    def _db_data_version(self) -> int:
        return int(self.db.execute("PRAGMA data_version").fetchone()[0])

    def _sync_db_guards(self) -> None:
        self._guarded_total_changes = self.db.total_changes
        self._guarded_data_version = self._db_data_version()

    def _ensure_validation_indexes(self) -> None:
        existing = {
            row[0]
            for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name IN ('idx_retained_run_status',"
                "'idx_retained_attempt_run_id',"
                "'idx_retained_reconciliation_run_id')"
            ).fetchall()
        }
        if "idx_retained_run_status" not in existing:
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_retained_run_status "
                "ON retained_run(status)"
            )
        if "idx_retained_attempt_run_id" not in existing:
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_retained_attempt_run_id "
                "ON retained_attempt(run_id)"
            )
        if "idx_retained_reconciliation_run_id" not in existing:
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_retained_reconciliation_run_id "
                "ON retained_reconciliation(run_id)"
            )

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
                CREATE INDEX idx_retained_run_status ON retained_run(status);
                CREATE INDEX idx_retained_attempt_run_id ON retained_attempt(run_id);
                CREATE INDEX idx_retained_reconciliation_run_id ON retained_reconciliation(run_id);
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
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _has_run_adoption_table(self) -> bool:
        return self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retained_run_adoption'"
        ).fetchone() is not None

    def _ensure_run_adoption_table(self) -> None:
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS retained_run_adoption (
                adoption_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES retained_run(id),
                prior_started_at REAL NOT NULL,
                prior_deadline_at REAL NOT NULL,
                prior_evidence_json TEXT NOT NULL,
                adopted_at REAL NOT NULL,
                deadline_at REAL NOT NULL,
                evidence_json TEXT NOT NULL
            )
        """)

    def _run_adoptions(self, run_id: str) -> list[sqlite3.Row]:
        if not self._has_run_adoption_table():
            return []
        return self.db.execute(
            "SELECT * FROM retained_run_adoption WHERE run_id=? ORDER BY adopted_at, rowid",
            (run_id,),
        ).fetchall()

    def _blocking_unresolved_count(self, now: float, active_run_ids: set[str]) -> int:
        resolved = self._reconciled_attempt_ids()
        count = 0
        for row in self.db.execute("SELECT * FROM retained_attempt WHERE status='unresolved'"):
            if row["id"] in resolved:
                continue
            if row["run_id"] in active_run_ids:
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
        if not path.is_file():
            raise AuthorityBlocked("Retained ledger must be initialized before import")
        db = _connect(path, create=False)
        try:
            db.execute("BEGIN IMMEDIATE")
            now = _finite_nonnegative(clock(), "clock")
            if now < evidence["last_dispatch_wall"]:
                raise AuthorityBlocked("Import clock predates legacy dispatch")
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
    def inspect_open_run(cls, db_path: Path, *, writer_id: str, account_id: str,
                         run_id: str) -> dict:
        """Read nonterminal run state and whether its prior owner process is stopped."""
        db, authority = cls._inspect(db_path, writer_id=writer_id, account_id=account_id)
        try:
            account = db.execute("SELECT * FROM retained_account WHERE id=1").fetchone()
            row = db.execute("SELECT * FROM retained_run WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise AuthorityBlocked("Retained open-run inspection target is absent")
            manifest, digest, key = _reviewed_manifest(
                row["manifest_raw"], row["id"], account_id)
            claim = authority._run_claim(run_id)
            owner_alive = False
            host_ok = True
            for p_host, p_pid in (
                ((claim["process_host"], claim["process_pid"]),)
                if claim is not None else ()
            ) + (
                ((account["process_host"], account["process_pid"]),)
                if (account["process_host"] is not None or
                    account["process_pid"] is not None) else ()
            ):
                if p_host != authority.process_host or type(p_pid) is not int or p_pid <= 0:
                    host_ok = False
                    break
                if p_pid == authority.process_pid:
                    owner_alive = True
                    break
                try:
                    os.kill(p_pid, 0)
                except ProcessLookupError:
                    pass
                except (PermissionError, OSError):
                    owner_alive = True
                    break
                else:
                    owner_alive = True
                    break
            reconciled_ids = authority._reconciled_attempt_ids()
            recoverable_unresolved = 0
            if row["status"] == "active":
                for att in db.execute(
                    "SELECT * FROM retained_attempt WHERE run_id=? AND status='unresolved'",
                    (run_id,),
                ).fetchall():
                    if att["id"] in reconciled_ids:
                        continue
                    if (
                        att["dispatched_at"] is None or
                        (att["kind"] != "media_chunk" and att["http_status"] is None and att["bytes"] <= 4096) or
                        (att["kind"] == "media_chunk" and att["target_category"] == "media_chunk" and
                         att["method"] == "PUT" and 0 < att["bytes"] <= (8 << 20) and
                         att["dispatched_at"] is not None and
                         att["http_status"] in (None, 500, 502, 503, 504)) or
                        (att["kind"] == "resumable_status" and att["target_category"] == "resumable_status" and
                         att["method"] == "PUT" and att["bytes"] == 0 and
                         att["dispatched_at"] is not None and
                         att["http_status"] in (None, 500, 502, 503, 504))
                    ):
                        recoverable_unresolved += 1
            return {
                "run_id": row["id"],
                "pair_key": key,
                "manifest_sha256": digest,
                "manifest_raw": bytes(row["manifest_raw"]),
                "manifest": manifest,
                "status": row["status"],
                "started_at": row["started_at"],
                "deadline_at": row["deadline_at"],
                "owner_host_matches": host_ok,
                "owner_alive": owner_alive,
                "recoverable_unresolved_attempts": recoverable_unresolved,
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
        max_concurrent_runs: int = 1, resume_open: bool = False,
    ) -> "RetainedObservationAuthority":
        _required_identity(run_id, "run ID")
        _required_identity(writer_id, "writer ID")
        _required_identity(account_id, "account ID")
        if type(max_concurrent_runs) is not int or not 1 <= max_concurrent_runs <= MAX_CONCURRENT_RUNS:
            raise AuthorityBlocked("Retained max_concurrent_runs must be 1..4")
        if (recovery_only or resume_open) and max_concurrent_runs != 1:
            raise AuthorityBlocked("Retained recovery_only requires single-run mode")
        if recovery_only and resume_open:
            raise AuthorityBlocked("Cannot combine recovery_only and resume_open")
        manifest, digest, key = _reviewed_manifest(manifest_bytes, run_id, account_id)
        deadline = _finite_nonnegative(deadline_at, "deadline")
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
                        max_concurrent_runs=max_concurrent_runs,
                        resume_open=resume_open)
        try:
            with authority._transaction():
                now = _finite_nonnegative(clock(), "clock")
                if not now < deadline <= now + min(MAX_RUN_SECONDS, manifest["max_duration_seconds"]):
                    raise AuthorityBlocked("Retained run exceeds reviewed 20-minute deadline")
                _review_evidence(evidence, run_id=run_id, writer_id=writer_id,
                                 account_id=account_id, manifest_sha256=digest,
                                 now=now, deadline_at=deadline)
                account = authority._account(
                    require_process=False, allow_boot_change=True
                )
                if now < account["last_clock"]:
                    raise AuthorityBlocked("Retained ledger clock regressed")
                if (account["boot_id"] is not None and account["boot_id"] != boot and
                    not (recovery_only or resume_open)):
                    has_open_run = (
                        db.execute(
                            "SELECT 1 FROM retained_run WHERE status != 'closed' LIMIT 1"
                        ).fetchone() is not None
                    )
                    if has_open_run or authority._effective_unresolved_count() > 0:
                        raise AuthorityBlocked(
                            "Retained ledger boot changed; reconciliation required"
                        )
                if max_concurrent_runs > 1:
                    authority._ensure_run_claim_table()
                authority._ensure_validation_indexes()
                authority._validation_stats["open_run_full_validates"] += 1
                authority._validate_all(allow_boot_change=True)
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

                def _assert_prior_process_stopped() -> None:
                    claim = authority._run_claim(run_id)
                    checks = []
                    if claim is not None:
                        checks.append((
                            claim["process_host"], claim["process_pid"],
                            claim["process_nonce"],
                        ))
                    if (account["process_host"] is not None or
                        account["process_pid"] is not None or
                        account["process_nonce"] is not None):
                        checks.append((
                            account["process_host"], account["process_pid"],
                            account["process_nonce"],
                        ))
                    for prior_host, prior_pid, _prior_nonce in checks:
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

                def _assert_other_open_runs_stopped() -> int:
                    other_open = [
                        r for r in prior
                        if r["id"] != run_id and r["status"] != "closed"
                    ]
                    if not other_open:
                        return 1
                    total_open = len(other_open) + 1
                    if total_open > MAX_CONCURRENT_RUNS:
                        raise AuthorityBlocked("Another retained run remains open")
                    for other_r in other_open:
                        other_claim = authority._run_claim(other_r["id"])
                        if (other_claim is None or
                            other_claim["max_concurrent_runs"] < total_open or
                            other_claim["process_host"] != authority.process_host or
                            type(other_claim["process_pid"]) is not int or
                            other_claim["process_pid"] <= 0 or
                            authority._claim_process_alive(other_claim)):
                            raise AuthorityBlocked("Another retained run remains open")
                    return total_open

                if recovery_only:
                    if (same is None or same["manifest_sha256"] != digest or
                        same["pair_key"] != key or same["status"] not in
                        ("verified", "index_pending") or
                        authority._effective_unresolved_count(run_id)):
                        raise AuthorityBlocked("No verified retained pair is eligible for no-upload recovery")
                    total_open = _assert_other_open_runs_stopped()
                    _assert_prior_process_stopped()
                    if authority._has_run_claim_table():
                        if total_open > 1:
                            db.execute(
                                "INSERT OR REPLACE INTO retained_run_claim VALUES (?,?,?,?,?)",
                                (run_id, authority.process_host, authority.process_pid,
                                 authority.process_nonce, max(total_open, max_concurrent_runs)),
                            )
                        else:
                            db.execute("DELETE FROM retained_run_claim WHERE run_id=?", (run_id,))
                elif resume_open:
                    if (same is None or same["manifest_sha256"] != digest or
                        same["pair_key"] != key or
                        same["status"] not in ("active", "verified", "index_pending")):
                        raise AuthorityBlocked("No open retained pair is eligible for crash recovery")
                    total_open = _assert_other_open_runs_stopped()
                    if account["cooldown_until"] > now:
                        raise AuthorityBlocked("Provider quota cooldown remains active")
                    _assert_prior_process_stopped()
                    if authority._has_run_claim_table():
                        if total_open > 1:
                            db.execute(
                                "INSERT OR REPLACE INTO retained_run_claim VALUES (?,?,?,?,?)",
                                (run_id, authority.process_host, authority.process_pid,
                                 authority.process_nonce, max(total_open, max_concurrent_runs)),
                            )
                        else:
                            db.execute("DELETE FROM retained_run_claim WHERE run_id=?", (run_id,))
                    authority._ensure_run_adoption_table()
                    existing_adoptions = authority._run_adoptions(run_id)
                    if existing_adoptions:
                        last_ad = existing_adoptions[-1]
                        p_start = last_ad["adopted_at"]
                        p_dead = last_ad["deadline_at"]
                        p_ev = last_ad["evidence_json"]
                    else:
                        p_start = same["started_at"]
                        p_dead = same["deadline_at"]
                        p_ev = same["evidence_json"]
                    db.execute(
                        "INSERT INTO retained_run_adoption VALUES (?,?,?,?,?,?,?,?)",
                        (uuid.uuid4().hex, run_id, p_start, p_dead, p_ev,
                         now, deadline, _canonical(evidence)),
                    )
                    db.execute(
                        "UPDATE retained_run SET deadline_at=? WHERE id=?",
                        (deadline, run_id),
                    )
                    reconciled_ids = authority._reconciled_attempt_ids(run_id)
                    for att in db.execute(
                        "SELECT * FROM retained_attempt WHERE run_id=? AND status='unresolved'",
                        (run_id,),
                    ).fetchall():
                        if att["id"] in reconciled_ids:
                            continue
                        if att["dispatched_at"] is None:
                            credit = int(att["bytes"])
                        elif att["kind"] != "media_chunk" and att["http_status"] is None and att["bytes"] <= 4096:
                            credit = 0
                        else:
                            continue
                        proof = {
                            "schema": CRASH_ADOPTION_RECONCILIATION_SCHEMA,
                            "attempt_id": att["id"],
                            "run_id": run_id,
                            "pair_key": key,
                            "manifest_sha256": digest,
                            "account_id": account_id,
                            "writer_id": writer_id,
                            "original_http_status": att["http_status"],
                            "original_charge_bytes": att["bytes"],
                            "dispatched": att["dispatched_at"] is not None,
                            "uncommitted_retry_bytes": credit,
                            "applied_at": now,
                        }
                        p_raw = _canonical(proof)
                        p_digest = hashlib.sha256(p_raw.encode()).hexdigest()
                        db.execute(
                            "INSERT INTO retained_reconciliation VALUES (?,?,?,?,?,?)",
                            (att["id"], run_id, key, p_raw, p_digest, now),
                        )
                        if credit > 0:
                            db.execute(
                                "UPDATE retained_run SET reserved_remaining=reserved_remaining+? WHERE id=?",
                                (credit, run_id),
                            )
                else:
                    if same is not None or any(r["pair_key"] == key for r in prior):
                        raise AuthorityBlocked("Retained run or source pair replay")
                    if (
                        manifest["bucket"], manifest["payload"]["object"],
                        manifest["payload"]["generation"],
                    ) in authority._last_validated_payload_identities:
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
                                claim["process_host"] != authority.process_host or
                                (not authority._claim_process_alive(claim) and
                                 abs(open_r["deadline_at"] - deadline) > 5.0)):
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
                boot_changed = (
                    account["boot_id"] is not None and
                    account["boot_id"] != boot
                )
                db.execute(
                    "UPDATE retained_account SET boot_id=?,"
                    "process_host=?,process_pid=?,process_nonce=?,last_clock=?,"
                    "last_dispatch_wall=CASE WHEN ? THEN NULL ELSE last_dispatch_wall END,"
                    "last_dispatch_monotonic=CASE WHEN ? THEN NULL ELSE last_dispatch_monotonic END "
                    "WHERE id=1",
                    (boot,
                     authority.process_host, authority.process_pid,
                     authority.process_nonce, now,
                     1 if boot_changed else 0,
                     1 if boot_changed else 0),
                )
            authority._run(allow_recovery=True, fast_per_request=True)
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
            self._sync_db_guards()
        except Exception as error:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            if isinstance(error, sqlite3.Error):
                raise AuthorityBlocked("Retained authority ledger unavailable or corrupt") from error
            raise

    def _account(
        self, *, require_process: bool = True, allow_boot_change: bool = False
    ) -> sqlite3.Row:
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
        if (account["boot_id"] is not None and account["boot_id"] != self.boot_id and
            not (allow_boot_change or self.recovery_only or self.resume_open)):
            raise AuthorityBlocked("Retained boot identity changed")
        if (account["next_slot"] < 0 or account["last_clock"] < 0 or
            account["cooldown_until"] < 0):
            raise AuthorityBlocked("Retained account clock state is malformed")
        return account

    def _reconciled_state(
        self, *, run_ids: set[str] | None = None
    ) -> tuple[set[str], dict[str, int]]:
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
        if run_ids is not None:
            if not run_ids:
                return resolved, credits_by_run
            ordered_ids = tuple(sorted(run_ids))
            placeholders = ",".join("?" for _ in ordered_ids)
            rows = self.db.execute(
                f"SELECT * FROM retained_reconciliation WHERE run_id IN ({placeholders})",
                ordered_ids,
            )
        else:
            rows = self.db.execute("SELECT * FROM retained_reconciliation")
        for row in rows:
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
                               STATUS_PROBE_RECONCILIATION_SCHEMA,
                               CRASH_ADOPTION_RECONCILIATION_SCHEMA) or
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
            if schema == CRASH_ADOPTION_RECONCILIATION_SCHEMA:
                credit = proof.get("uncommitted_retry_bytes")
                undispatched = (
                    attempt["dispatched_at"] is None and
                    proof.get("dispatched") is False and
                    credit == attempt["bytes"]
                )
                control_interrupted = (
                    attempt["dispatched_at"] is not None and
                    proof.get("dispatched") is True and
                    attempt["http_status"] is None and
                    attempt["kind"] != "media_chunk" and
                    attempt["bytes"] <= 4096 and
                    credit == 0
                )
                if (attempt["run_id"] != run["id"] or run["pair_key"] != key or
                    row["pair_key"] != key or attempt["status"] != "unresolved" or
                    attempt["http_status"] is not None or
                    not (undispatched or control_interrupted) or
                    proof.get("manifest_sha256") != digest or
                    proof.get("account_id") != self.account_id or
                    proof.get("writer_id") != self.writer_id or
                    proof.get("original_http_status") is not None or
                    proof.get("original_charge_bytes") != attempt["bytes"] or
                    type(credit) is not int):
                    raise AuthorityBlocked(
                        "Retained crash-adoption reconciliation attempt identity differs"
                    )
                resolved.add(row["attempt_id"])
                credits_by_run[run["id"]] = credits_by_run.get(run["id"], 0) + credit
                continue
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
                    attempt["http_status"] not in (None, 500, 502, 503, 504) or
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

    def _reconciled_attempt_ids(self, run_id: str | None = None) -> set[str]:
        return self._reconciled_state(
            run_ids={run_id} if run_id is not None else None
        )[0]

    def _status_probe_reconciled_credits(self, run_id: str) -> int:
        return self._reconciled_state(run_ids={run_id})[1].get(run_id, 0)

    def _effective_unresolved_count(self, run_id: str | None = None) -> int:
        resolved = self._reconciled_attempt_ids(run_id)
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

    def _build_closed_fingerprint(
        self,
        *,
        imported: list[sqlite3.Row],
        runs: list[sqlite3.Row],
        all_attempts: list[sqlite3.Row],
        mappings: list[sqlite3.Row],
        adoptions_by_rowid: list[sqlite3.Row],
        reconciliations_by_rowid: list[sqlite3.Row],
        recoveries_by_rowid: list[sqlite3.Row],
        recovery_attempts_by_rowid: list[sqlite3.Row],
        reviewed_by_run: dict[str, dict],
        only_closed_run_rowids: frozenset[int] | None = None,
    ) -> dict:
        closed_runs = [
            r for r in runs
            if r["status"] == "closed" and (
                only_closed_run_rowids is None or
                int(r["rowid"]) in only_closed_run_rowids
            )
        ]
        closed_run_rowids = frozenset(int(r["rowid"]) for r in closed_runs)
        closed_run_ids = {r["id"] for r in closed_runs}
        closed_attempts = [
            a for a in all_attempts if a["run_id"] in closed_run_ids
        ]
        closed_mappings = [
            m for m in mappings if m["run_id"] in closed_run_ids
        ]
        closed_adoptions = [
            ad for ad in adoptions_by_rowid if ad["run_id"] in closed_run_ids
        ]
        closed_reconciliations = [
            rec for rec in reconciliations_by_rowid
            if rec["run_id"] in closed_run_ids
        ]
        closed_recoveries = [
            rec for rec in recoveries_by_rowid
            if rec["original_run_id"] in closed_run_ids
        ]
        closed_recovery_ids = {rec["id"] for rec in closed_recoveries}
        closed_recovery_attempts = [
            ra for ra in recovery_attempts_by_rowid
            if ra["recovery_id"] in closed_recovery_ids
        ]

        hasher = hashlib.sha256()
        hasher.update(b"retained_import\n")
        for item in imported:
            hasher.update(repr(tuple(item)).encode("utf-8"))
            hasher.update(b"\n")
        hasher.update(b"retained_run\n")
        for r in closed_runs:
            hasher.update(repr((
                r["rowid"], r["id"], r["manifest_sha256"],
                len(r["manifest_raw"]), r["pair_key"], r["media_bytes"],
                r["control_allowance"], r["budget_bytes"], r["metadata_json"],
                r["reserved"], r["reserved_remaining"], r["status"],
                r["mapping_sha256"], r["evidence_json"], r["started_at"],
                r["deadline_at"],
            )).encode("utf-8"))
            hasher.update(bytes(r["manifest_raw"]))
            hasher.update(b"\n")
        hasher.update(b"retained_attempt\n")
        for a in closed_attempts:
            hasher.update(repr(tuple(a)).encode("utf-8"))
            hasher.update(b"\n")
        hasher.update(b"retained_mapping_reservation\n")
        for m in closed_mappings:
            hasher.update(repr(tuple(m)).encode("utf-8"))
            hasher.update(b"\n")
        hasher.update(b"retained_run_adoption\n")
        for ad in closed_adoptions:
            hasher.update(repr(tuple(ad)).encode("utf-8"))
            hasher.update(b"\n")
        hasher.update(b"retained_reconciliation\n")
        for rec in closed_reconciliations:
            hasher.update(repr(tuple(rec)).encode("utf-8"))
            hasher.update(b"\n")
        hasher.update(b"retained_receipt_recovery\n")
        for rec in closed_recoveries:
            hasher.update(repr(tuple(rec)).encode("utf-8"))
            hasher.update(b"\n")
        hasher.update(b"retained_receipt_recovery_attempt\n")
        for ra in closed_recovery_attempts:
            hasher.update(repr(tuple(ra)).encode("utf-8"))
            hasher.update(b"\n")

        closed_payload_identities = frozenset(
            (
                reviewed_by_run[r["id"]]["bucket"],
                reviewed_by_run[r["id"]]["payload"]["object"],
                reviewed_by_run[r["id"]]["payload"]["generation"],
            )
            for r in closed_runs
        )
        closed_pair_keys = frozenset(r["pair_key"] for r in closed_runs)
        closed_mapping_bytes = sum(int(m["bytes"]) for m in closed_mappings)

        closed_dispatched_ats = tuple(sorted(
            float(a["dispatched_at"])
            for a in (closed_attempts + closed_recovery_attempts)
            if a["dispatch_monotonic"] is not None
        ))

        closed_unresolved_attempt_bytes = sum(
            int(a["bytes"]) for a in closed_attempts if a["status"] == "unresolved"
        ) + sum(
            int(ra["bytes"]) for ra in closed_recovery_attempts
            if ra["status"] == "unresolved"
        )
        settled_pairs = sorted(
            (float(a["charge_through"]), int(a["bytes"]))
            for a in (closed_attempts + closed_recovery_attempts)
            if a["status"] == "settled" and int(a["bytes"]) > 0
        )
        charge_throughs = [pt[0] for pt in settled_pairs]
        prefix_bytes = [0]
        running = 0
        for _, b_val in settled_pairs:
            running += b_val
            prefix_bytes.append(running)

        return {
            "closed_run_count": len(closed_runs),
            "closed_run_rowids": closed_run_rowids,
            "max_closed_run_rowid": int(closed_runs[-1]["rowid"]) if closed_runs else 0,
            "closed_attempt_count": len(closed_attempts),
            "max_closed_attempt_rowid": int(closed_attempts[-1]["rowid"]) if closed_attempts else 0,
            "max_attempt_rowid": int(all_attempts[-1]["rowid"]) if all_attempts else 0,
            "sha256": hasher.hexdigest(),
            "closed_payload_identities": closed_payload_identities,
            "closed_pair_keys": closed_pair_keys,
            "closed_mapping_bytes": closed_mapping_bytes,
            "closed_dispatched_ats": closed_dispatched_ats,
            "closed_unresolved_attempt_bytes": closed_unresolved_attempt_bytes,
            "closed_settled_charge_throughs": charge_throughs,
            "closed_settled_prefix_bytes": prefix_bytes,
        }

    def _validate_account_and_import(self, account: sqlite3.Row) -> None:
        imported = self.db.execute("SELECT rowid, * FROM retained_import ORDER BY rowid").fetchall()
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

    def _validate_all(
        self,
        *,
        allow_boot_change: bool = False,
        refresh_closed_fingerprint: bool = False,
    ) -> None:
        """Semantic check beyond SQLite integrity before every admission."""
        self._validation_stats["full_validate_calls"] += 1
        account = self._account(
            require_process=False, allow_boot_change=allow_boot_change
        )
        imported = self.db.execute("SELECT rowid, * FROM retained_import ORDER BY rowid").fetchall()
        self._validate_account_and_import(account)
        runs = self.db.execute("SELECT rowid, * FROM retained_run ORDER BY rowid").fetchall()
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
        adoptions_by_run: dict[str, list[sqlite3.Row]] = {}
        adoptions_by_rowid: list[sqlite3.Row] = []
        if self._has_run_adoption_table():
            for ad_row in self.db.execute(
                "SELECT rowid, * FROM retained_run_adoption ORDER BY adopted_at, rowid"
            ).fetchall():
                adoptions_by_run.setdefault(ad_row["run_id"], []).append(ad_row)
                adoptions_by_rowid.append(ad_row)
            adoptions_by_rowid.sort(key=lambda r: r["rowid"])
        all_attempts = self.db.execute("SELECT rowid, * FROM retained_attempt ORDER BY rowid").fetchall()
        attempts_by_run: dict[str, list[sqlite3.Row]] = {}
        for attempt_row in all_attempts:
            attempts_by_run.setdefault(attempt_row["run_id"], []).append(attempt_row)
        reconciled_state: tuple[set[str], dict[str, int]] | None = None
        unresolved_recovery_by_run: dict[str, int] | None = None
        reviewed_by_run: dict[str, dict] = {}
        payload_identities: set[tuple[str, str, int]] = set()
        for run in runs:
            manifest, digest, key = _reviewed_manifest(run["manifest_raw"], run["id"], self.account_id)
            reviewed_by_run[run["id"]] = manifest
            source = (manifest["bucket"], manifest["payload"]["object"],
                      manifest["payload"]["generation"])
            if source in payload_identities:
                raise AuthorityBlocked("Retained ledger repeats a payload generation")
            payload_identities.add(source)
            adoptions = adoptions_by_run.get(run["id"], ())
            initial_deadline = (
                adoptions[0]["prior_deadline_at"] if adoptions else run["deadline_at"]
            )
            if (run["manifest_sha256"] != digest or run["pair_key"] != key or
                run["media_bytes"] != manifest["payload"]["bytes"] + manifest["receipt"]["bytes"] or
                run["control_allowance"] != manifest["control_allowance_bytes"] or
                run["budget_bytes"] != manifest["max_charged_bytes"] or
                not 0 < initial_deadline - run["started_at"] <= manifest["max_duration_seconds"]):
                raise AuthorityBlocked("Retained run manifest or limits changed")
            evidence = json.loads(run["evidence_json"])
            _review_evidence(evidence, run_id=run["id"], writer_id=self.writer_id,
                             account_id=self.account_id, manifest_sha256=digest,
                             now=run["started_at"], deadline_at=initial_deadline)
            prev_deadline = initial_deadline
            for ad in adoptions:
                if (ad["run_id"] != run["id"] or
                    ad["prior_deadline_at"] != prev_deadline or
                    ad["adopted_at"] < run["started_at"] or
                    not 0 < ad["deadline_at"] - ad["adopted_at"] <= manifest["max_duration_seconds"]):
                    raise AuthorityBlocked("Retained run adoption window is malformed")
                ad_evidence = json.loads(ad["evidence_json"])
                _review_evidence(ad_evidence, run_id=run["id"], writer_id=self.writer_id,
                                 account_id=self.account_id, manifest_sha256=digest,
                                 now=ad["adopted_at"], deadline_at=ad["deadline_at"])
                prev_deadline = ad["deadline_at"]
            if adoptions and prev_deadline != run["deadline_at"]:
                raise AuthorityBlocked("Retained run deadline differs from latest adoption")
            if (run["reserved"] not in (0, 1) or
                not 0 <= run["reserved_remaining"] <= run["budget_bytes"]):
                raise AuthorityBlocked("Retained reservation is malformed")
            if run["reserved"]:
                if json.loads(run["metadata_json"]) != _expected_metadata(manifest, digest):
                    raise AuthorityBlocked("Retained pair metadata changed")
            elif run["metadata_json"] is not None or run["reserved_remaining"] != 0:
                raise AuthorityBlocked("Unreserved retained pair has a charge")
            max_deadline = max(
                [run["deadline_at"]] +
                [ad["prior_deadline_at"] for ad in adoptions] +
                [ad["deadline_at"] for ad in adoptions]
            )
            attempts = attempts_by_run.get(run["id"], ())
            spent = 0
            for attempt in attempts:
                spent += self._validate_attempt(attempt, run, max_deadline=max_deadline)
            if reconciled_state is None:
                reconciled_state = self._reconciled_state()
            reconciled_ids, reconciled_credits_by_run = reconciled_state
            net_spent = spent - reconciled_credits_by_run.get(run["id"], 0)
            if net_spent < 0:
                raise AuthorityBlocked("Retained pair status-probe credit exceeds spend")
            if run["status"] in ("active", "verified") and run["reserved"]:
                if net_spent + run["reserved_remaining"] != run["budget_bytes"]:
                    raise AuthorityBlocked("Retained pair charges do not reconcile")
            elif run["status"] in ("index_pending", "closed"):
                if (not run["reserved"] or run["reserved_remaining"] != 0 or
                    not _valid_digest(run["mapping_sha256"]) or net_spent > run["budget_bytes"]):
                    raise AuthorityBlocked("Retained index state or charge is malformed")
                if unresolved_recovery_by_run is None:
                    unresolved_recovery_by_run = {}
                    if self.db.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                        "name='retained_receipt_recovery_attempt'"
                    ).fetchone():
                        for rec_row in self.db.execute(
                            "SELECT r.original_run_id AS run_id, COUNT(*) AS cnt "
                            "FROM retained_receipt_recovery_attempt a "
                            "JOIN retained_receipt_recovery r ON r.id=a.recovery_id "
                            "WHERE a.status='unresolved' "
                            "GROUP BY r.original_run_id"
                        ).fetchall():
                            unresolved_recovery_by_run[rec_row["run_id"]] = rec_row["cnt"]
                unresolved_for_run = (
                    sum(
                        att["status"] == "unresolved" and att["id"] not in reconciled_ids
                        for att in attempts
                    ) + unresolved_recovery_by_run.get(run["id"], 0)
                )
                if unresolved_for_run:
                    raise AuthorityBlocked("Index pending with uncertain Drive attempt")
            elif spent:
                raise AuthorityBlocked("Unreserved retained pair has attempts")
            if run["status"] in ("verified", "index_pending", "closed") and not _valid_digest(run["mapping_sha256"]):
                raise AuthorityBlocked("Verified retained mapping digest missing")
        mappings = self.db.execute(
            "SELECT rowid, * FROM retained_mapping_reservation ORDER BY rowid"
        ).fetchall()
        if sum(item["bytes"] for item in mappings) > MAPPING_PARTITION_BYTES:
            raise AuthorityBlocked("Retained mapping partition exceeded")
        by_run = {run["id"]: run for run in runs}
        for item in mappings:
            run = by_run.get(item["run_id"])
            reviewed = reviewed_by_run.get(run["id"]) if run is not None else None
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
        mapped_run_ids = {item["run_id"] for item in mappings}
        if any(run["status"] == "closed" and run["id"] not in mapped_run_ids for run in runs):
            raise AuthorityBlocked("Closed retained pair has no mapping reservation")
        self._validate_recovery_rows()
        if len(all_attempts) != sum(len(attempts_by_run.get(run["id"], ())) for run in runs):
            raise AuthorityBlocked("Orphan retained attempt")
        recoveries_by_rowid: list[sqlite3.Row] = []
        recovery_attempts_by_rowid: list[sqlite3.Row] = []
        if self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND "
            "name='retained_receipt_recovery_attempt'").fetchone():
            recoveries_by_rowid = self.db.execute(
                "SELECT rowid, * FROM retained_receipt_recovery ORDER BY rowid"
            ).fetchall()
            recovery_attempts_by_rowid = self.db.execute(
                "SELECT rowid, * FROM retained_receipt_recovery_attempt ORDER BY rowid"
            ).fetchall()
        dispatched = sorted(
            (
                a for a in (all_attempts + recovery_attempts_by_rowid)
                if a["dispatch_monotonic"] is not None
            ),
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
            same_adoption_epoch = True
            if owner_a == owner_b and owner_a[0] == "run":
                if any(
                    a["dispatched_at"] <= ad["adopted_at"] <= b["dispatched_at"]
                    for ad in adoptions_by_run.get(owner_a[1], ())
                ):
                    same_adoption_epoch = False
            if (
                owner_a == owner_b and
                same_adoption_epoch and
                b["dispatch_monotonic"] - a["dispatch_monotonic"] < PACING_SECONDS - 1e-9
            ):
                raise AuthorityBlocked("Retained actual starts violate monotonic pacing")

        reconciliations_by_rowid: list[sqlite3.Row] = []
        if self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retained_reconciliation'"
        ).fetchone():
            reconciliations_by_rowid = self.db.execute(
                "SELECT rowid, * FROM retained_reconciliation ORDER BY rowid"
            ).fetchall()

        current_fp = self._build_closed_fingerprint(
            imported=imported,
            runs=runs,
            all_attempts=all_attempts,
            mappings=mappings,
            adoptions_by_rowid=adoptions_by_rowid,
            reconciliations_by_rowid=reconciliations_by_rowid,
            recoveries_by_rowid=recoveries_by_rowid,
            recovery_attempts_by_rowid=recovery_attempts_by_rowid,
            reviewed_by_run=reviewed_by_run,
        )
        prev_fp = self._closed_fingerprint
        if prev_fp is not None:
            prev_rowids = prev_fp.get("closed_run_rowids", frozenset())
            curr_rowids = current_fp["closed_run_rowids"]
            if (
                current_fp["closed_run_count"] < prev_fp["closed_run_count"] or
                not prev_rowids.issubset(curr_rowids)
            ):
                raise AuthorityBlocked("Closed retained history fingerprint changed")
            if curr_rowids == prev_rowids:
                if (
                    current_fp["closed_run_count"],
                    current_fp["max_closed_run_rowid"],
                    current_fp["closed_attempt_count"],
                    current_fp["max_closed_attempt_rowid"],
                    current_fp["sha256"],
                ) != (
                    prev_fp["closed_run_count"],
                    prev_fp["max_closed_run_rowid"],
                    prev_fp["closed_attempt_count"],
                    prev_fp["max_closed_attempt_rowid"],
                    prev_fp["sha256"],
                ):
                    raise AuthorityBlocked("Closed retained history fingerprint changed")
            elif prev_fp["closed_run_count"] > 0:
                prefix_fp = self._build_closed_fingerprint(
                    imported=imported,
                    runs=runs,
                    all_attempts=all_attempts,
                    mappings=mappings,
                    adoptions_by_rowid=adoptions_by_rowid,
                    reconciliations_by_rowid=reconciliations_by_rowid,
                    recoveries_by_rowid=recoveries_by_rowid,
                    recovery_attempts_by_rowid=recovery_attempts_by_rowid,
                    reviewed_by_run=reviewed_by_run,
                    only_closed_run_rowids=prev_rowids,
                )
                if (
                    prefix_fp["closed_run_count"],
                    prefix_fp["max_closed_run_rowid"],
                    prefix_fp["closed_attempt_count"],
                    prefix_fp["max_closed_attempt_rowid"],
                    prefix_fp["sha256"],
                ) != (
                    prev_fp["closed_run_count"],
                    prev_fp["max_closed_run_rowid"],
                    prev_fp["closed_attempt_count"],
                    prev_fp["max_closed_attempt_rowid"],
                    prev_fp["sha256"],
                ):
                    raise AuthorityBlocked("Closed retained history fingerprint changed")

        self._closed_fingerprint = current_fp
        self._last_validated_payload_identities = payload_identities
        self._sync_db_guards()

    def _validate_open_and_fingerprint(
        self, *, allow_boot_change: bool = False
    ) -> None:
        """Per-request O(1) validation of open runs, their attempts, and the closed-history fingerprint."""
        self._validation_stats["fast_checks"] += 1
        fp = self._closed_fingerprint
        if (
            fp is None or
            self._guarded_total_changes is None or
            self._guarded_data_version is None or
            not _valid_digest(fp.get("sha256"))
        ):
            self._validation_stats["fallbacks"] += 1
            self._validation_stats["uninitialized"] += 1
            self._validate_all(allow_boot_change=allow_boot_change)
            return

        if self.db.total_changes != self._guarded_total_changes:
            self._validation_stats["fallbacks"] += 1
            self._validation_stats["total_changes"] += 1
            self._validate_all(allow_boot_change=allow_boot_change)
            return

        if self._db_data_version() != self._guarded_data_version:
            self._validation_stats["fallbacks"] += 1
            self._validation_stats["data_version"] += 1
            self._validate_all(allow_boot_change=allow_boot_change)
            return

        cheap_row = self.db.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM retained_run WHERE status='closed'), "
            "(SELECT COALESCE(MAX(rowid), 0) FROM retained_run WHERE status='closed'), "
            "(SELECT COALESCE(MAX(rowid), 0) FROM retained_attempt)"
        ).fetchone()
        if (
            cheap_row is None or
            int(cheap_row[0]) != fp["closed_run_count"] or
            int(cheap_row[1]) != fp["max_closed_run_rowid"] or
            int(cheap_row[2]) != fp["max_attempt_rowid"]
        ):
            self._validation_stats["fallbacks"] += 1
            self._validation_stats["cheap_fingerprint"] += 1
            self._validate_all(allow_boot_change=allow_boot_change)
            return

        account = self._account(
            require_process=False, allow_boot_change=allow_boot_change
        )
        self._validate_account_and_import(account)

        nonterminal_runs = self.db.execute(
            "SELECT rowid, * FROM retained_run "
            "WHERE status IN ('active','verified','index_pending') "
            "ORDER BY rowid"
        ).fetchall()
        nonterminal_by_id = {run["id"]: run for run in nonterminal_runs}

        if self._has_run_claim_table():
            claims = {
                row["run_id"]: row
                for row in self.db.execute("SELECT * FROM retained_run_claim").fetchall()
            }
            for claimed_id, claim in claims.items():
                claimed_run = nonterminal_by_id.get(claimed_id)
                if (
                    claimed_run is None or
                    type(claim["max_concurrent_runs"]) is not int or
                    not 1 <= claim["max_concurrent_runs"] <= MAX_CONCURRENT_RUNS or
                    not isinstance(claim["process_host"], str) or not claim["process_host"] or
                    type(claim["process_pid"]) is not int or claim["process_pid"] <= 0 or
                    not isinstance(claim["process_nonce"], str) or not claim["process_nonce"]
                ):
                    raise AuthorityBlocked("Retained run claim is orphaned or malformed")
        else:
            claims = {}

        if len(nonterminal_runs) > 1:
            if len(nonterminal_runs) > MAX_CONCURRENT_RUNS:
                raise AuthorityBlocked("Multiple retained runs are nonterminal")
            for open_run in nonterminal_runs:
                claim = claims.get(open_run["id"])
                if (
                    claim is None or
                    claim["max_concurrent_runs"] < 2 or
                    len(nonterminal_runs) > claim["max_concurrent_runs"]
                ):
                    raise AuthorityBlocked("Multiple retained runs are nonterminal")

        reconciled_ids, reconciled_credits_by_run = self._reconciled_state(
            run_ids=set(nonterminal_by_id.keys())
        )
        open_payload_identities: set[tuple[str, str, int]] = set()
        open_pair_keys: set[str] = set()
        adoptions_by_run: dict[str, list[sqlite3.Row]] = {}
        open_attempts_all: list[sqlite3.Row] = []
        open_mapping_bytes = 0

        for run in nonterminal_runs:
            manifest, digest, key = _reviewed_manifest(
                run["manifest_raw"], run["id"], self.account_id
            )
            source = (
                manifest["bucket"],
                manifest["payload"]["object"],
                manifest["payload"]["generation"],
            )
            if (
                source in open_payload_identities or
                source in fp["closed_payload_identities"] or
                key in open_pair_keys or
                key in fp["closed_pair_keys"]
            ):
                raise AuthorityBlocked("Retained ledger repeats a payload generation")
            open_payload_identities.add(source)
            open_pair_keys.add(key)

            adoptions = self._run_adoptions(run["id"])
            adoptions_by_run[run["id"]] = adoptions
            initial_deadline = (
                adoptions[0]["prior_deadline_at"] if adoptions else run["deadline_at"]
            )
            if (
                run["manifest_sha256"] != digest or
                run["pair_key"] != key or
                run["media_bytes"] != manifest["payload"]["bytes"] + manifest["receipt"]["bytes"] or
                run["control_allowance"] != manifest["control_allowance_bytes"] or
                run["budget_bytes"] != manifest["max_charged_bytes"] or
                not 0 < initial_deadline - run["started_at"] <= manifest["max_duration_seconds"]
            ):
                raise AuthorityBlocked("Retained run manifest or limits changed")
            evidence = json.loads(run["evidence_json"])
            _review_evidence(
                evidence,
                run_id=run["id"],
                writer_id=self.writer_id,
                account_id=self.account_id,
                manifest_sha256=digest,
                now=run["started_at"],
                deadline_at=initial_deadline,
            )
            prev_deadline = initial_deadline
            for ad in adoptions:
                if (
                    ad["run_id"] != run["id"] or
                    ad["prior_deadline_at"] != prev_deadline or
                    ad["adopted_at"] < run["started_at"] or
                    not 0 < ad["deadline_at"] - ad["adopted_at"] <= manifest["max_duration_seconds"]
                ):
                    raise AuthorityBlocked("Retained run adoption window is malformed")
                ad_evidence = json.loads(ad["evidence_json"])
                _review_evidence(
                    ad_evidence,
                    run_id=run["id"],
                    writer_id=self.writer_id,
                    account_id=self.account_id,
                    manifest_sha256=digest,
                    now=ad["adopted_at"],
                    deadline_at=ad["deadline_at"],
                )
                prev_deadline = ad["deadline_at"]
            if adoptions and prev_deadline != run["deadline_at"]:
                raise AuthorityBlocked("Retained run deadline differs from latest adoption")
            if (
                run["reserved"] not in (0, 1) or
                not 0 <= run["reserved_remaining"] <= run["budget_bytes"]
            ):
                raise AuthorityBlocked("Retained reservation is malformed")
            if run["reserved"]:
                if json.loads(run["metadata_json"]) != _expected_metadata(manifest, digest):
                    raise AuthorityBlocked("Retained pair metadata changed")
            elif run["metadata_json"] is not None or run["reserved_remaining"] != 0:
                raise AuthorityBlocked("Unreserved retained pair has a charge")

            max_deadline = max(
                [run["deadline_at"]] +
                [ad["prior_deadline_at"] for ad in adoptions] +
                [ad["deadline_at"] for ad in adoptions]
            )
            attempts = self.db.execute(
                "SELECT rowid, * FROM retained_attempt WHERE run_id=? ORDER BY rowid",
                (run["id"],),
            ).fetchall()
            open_attempts_all.extend(attempts)
            spent = 0
            for attempt in attempts:
                spent += self._validate_attempt(attempt, run, max_deadline=max_deadline)
            net_spent = spent - reconciled_credits_by_run.get(run["id"], 0)
            if net_spent < 0:
                raise AuthorityBlocked("Retained pair status-probe credit exceeds spend")
            if run["status"] in ("active", "verified") and run["reserved"]:
                if net_spent + run["reserved_remaining"] != run["budget_bytes"]:
                    raise AuthorityBlocked("Retained pair charges do not reconcile")
            elif run["status"] == "index_pending":
                if (
                    not run["reserved"] or
                    run["reserved_remaining"] != 0 or
                    not _valid_digest(run["mapping_sha256"]) or
                    net_spent > run["budget_bytes"]
                ):
                    raise AuthorityBlocked("Retained index state or charge is malformed")
                unresolved_for_run = sum(
                    att["status"] == "unresolved" and att["id"] not in reconciled_ids
                    for att in attempts
                )
                if unresolved_for_run:
                    raise AuthorityBlocked("Index pending with uncertain Drive attempt")
            elif spent:
                raise AuthorityBlocked("Unreserved retained pair has attempts")
            if run["status"] in ("verified", "index_pending") and not _valid_digest(run["mapping_sha256"]):
                raise AuthorityBlocked("Verified retained mapping digest missing")

            mapping_row = self.db.execute(
                "SELECT * FROM retained_mapping_reservation WHERE run_id=?",
                (run["id"],),
            ).fetchone()
            if mapping_row is not None:
                if (
                    run["status"] != "index_pending" or
                    mapping_row["pair_key"] != run["pair_key"] or
                    mapping_row["mapping_sha256"] != run["mapping_sha256"] or
                    mapping_row["object_name"] != manifest["index_object"] or
                    not isinstance(mapping_row["object_name"], str) or
                    not mapping_row["object_name"] or
                    len(mapping_row["object_name"]) > 1024 or
                    any(ord(c) < 32 for c in mapping_row["object_name"]) or
                    type(mapping_row["bytes"]) is not int or
                    not 0 < mapping_row["bytes"] <= MAPPING_MAX_BODY_BYTES or
                    not _valid_digest(mapping_row["reservation_id"])
                ):
                    raise AuthorityBlocked("Retained mapping reservation is malformed")
                open_mapping_bytes += int(mapping_row["bytes"])

        if fp["closed_mapping_bytes"] + open_mapping_bytes > MAPPING_PARTITION_BYTES:
            raise AuthorityBlocked("Retained mapping partition exceeded")

        dispatched = sorted(
            (a for a in open_attempts_all if a["dispatch_monotonic"] is not None),
            key=lambda a: (a["dispatched_at"], a["dispatch_monotonic"]),
        )
        closed_dispatched_ats = fp["closed_dispatched_ats"]
        for a in dispatched:
            d_at = float(a["dispatched_at"])
            idx = bisect.bisect_left(closed_dispatched_ats, d_at)
            if idx > 0 and d_at - closed_dispatched_ats[idx - 1] < PACING_SECONDS - 1e-6:
                raise AuthorityBlocked("Retained actual starts violate monotonic pacing")
            if idx < len(closed_dispatched_ats) and closed_dispatched_ats[idx] - d_at < PACING_SECONDS - 1e-6:
                raise AuthorityBlocked("Retained actual starts violate monotonic pacing")
        for a, b in zip(dispatched, dispatched[1:]):
            if b["dispatched_at"] - a["dispatched_at"] < PACING_SECONDS - 1e-6:
                raise AuthorityBlocked("Retained actual starts violate monotonic pacing")
            same_adoption_epoch = True
            if a["run_id"] == b["run_id"]:
                if any(
                    a["dispatched_at"] <= ad["adopted_at"] <= b["dispatched_at"]
                    for ad in adoptions_by_run.get(a["run_id"], ())
                ):
                    same_adoption_epoch = False
            if (
                a["run_id"] == b["run_id"] and
                same_adoption_epoch and
                b["dispatch_monotonic"] - a["dispatch_monotonic"] < PACING_SECONDS - 1e-9
            ):
                raise AuthorityBlocked("Retained actual starts violate monotonic pacing")
        self._validation_stats["fast_hits"] += 1

    @staticmethod
    def _validate_attempt(attempt: sqlite3.Row, run: sqlite3.Row, *,
                          max_deadline: float | None = None) -> int:
        deadline_limit = run["deadline_at"] if max_deadline is None else max_deadline
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
            slot + timeout >= deadline_limit or through < slot + timeout):
            raise AuthorityBlocked("Malformed retained attempt timing")
        dispatch = attempt["dispatched_at"]
        mono = attempt["dispatch_monotonic"]
        if (dispatch is None) != (mono is None):
            raise AuthorityBlocked("Retained dispatch clocks disagree")
        if dispatch is not None:
            dispatch = _finite_nonnegative(dispatch, "actual dispatch")
            _finite_nonnegative(mono, "monotonic dispatch")
            if (dispatch < slot or dispatch + timeout >= deadline_limit or
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

    def _run(
        self,
        *,
        allow_recovery: bool = False,
        fast_per_request: bool = False,
    ) -> tuple[sqlite3.Row, sqlite3.Row]:
        account = self._account()
        if fast_per_request:
            self._validate_open_and_fingerprint()
        else:
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
        account, run = self._run(fast_per_request=True)
        now = self._now(account)
        if slot_at is not None and now < _finite_nonnegative(slot_at, "request slot"):
            raise AuthorityBlocked("Retained request slot has not arrived")
        self._active(account, run, now, timeout_s, slot_at)

    def validate_manifest_limits(self, max_charged_bytes: int, max_duration_seconds: int) -> None:
        account, run = self._run(fast_per_request=True)
        adoptions = self._run_adoptions(run["id"])
        window_start = adoptions[-1]["adopted_at"] if adoptions else run["started_at"]
        if (_positive_int(max_charged_bytes, "manifest byte limit") != run["budget_bytes"] or
            _positive_int(max_duration_seconds, "manifest duration") <
            run["deadline_at"] - window_start):
            raise AuthorityBlocked("Retained run differs from reviewed manifest limits")
        self._active(account, run, self._now(account))

    def _known_usage(self, now: float) -> int:
        cutoff = now - WINDOW_SECONDS
        fp = self._closed_fingerprint
        if (
            fp is not None and
            self._guarded_total_changes is not None and
            self._guarded_data_version is not None and
            self.db.total_changes == self._guarded_total_changes and
            self._db_data_version() == self._guarded_data_version and
            _valid_digest(fp.get("sha256"))
        ):
            cts = fp["closed_settled_charge_throughs"]
            pbytes = fp["closed_settled_prefix_bytes"]
            idx = bisect.bisect_right(cts, cutoff)
            closed_settled_within_window = pbytes[-1] - pbytes[idx]
            open_charges = self.db.execute(
                "SELECT COALESCE(SUM(a.bytes),0) "
                "FROM retained_attempt a "
                "JOIN retained_run r ON r.id=a.run_id "
                "WHERE r.status IN ('active','verified','index_pending') "
                "AND (a.status='unresolved' OR a.charge_through>?)",
                (cutoff,),
            ).fetchone()[0]
            open_recovery = (self.db.execute(
                "SELECT COALESCE(SUM(a.bytes),0) "
                "FROM retained_receipt_recovery_attempt a "
                "JOIN retained_receipt_recovery r ON r.id=a.recovery_id "
                "JOIN retained_run rr ON rr.id=r.original_run_id "
                "WHERE rr.status IN ('active','verified','index_pending') "
                "AND (a.status='unresolved' OR a.charge_through>?)",
                (cutoff,),
            ).fetchone()[0] if self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                "name='retained_receipt_recovery_attempt'").fetchone() else 0)
            charges_and_recovery = (
                int(fp["closed_unresolved_attempt_bytes"]) +
                int(closed_settled_within_window) +
                int(open_charges) +
                int(open_recovery)
            )
        else:
            charges = self.db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_attempt "
                "WHERE status='unresolved' OR charge_through>?", (cutoff,),
            ).fetchone()[0]
            recovery = (self.db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_receipt_recovery_attempt "
                "WHERE status='unresolved' OR charge_through>?",
                (cutoff,),
            ).fetchone()[0] if self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND "
                "name='retained_receipt_recovery_attempt'").fetchone() else 0)
            charges_and_recovery = int(charges) + int(recovery)

        reserved = self.db.execute(
            "SELECT COALESCE(SUM(reserved_remaining),0) FROM retained_run "
            "WHERE status IN ('active','verified')",
        ).fetchone()[0]
        imported = self.db.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM retained_import WHERE charge_through>?",
            (cutoff,),
        ).fetchone()[0]
        recovery_reserved = (self.db.execute(
            "SELECT COALESCE(SUM(reserved_remaining),0) "
            "FROM retained_receipt_recovery WHERE status='open'",
        ).fetchone()[0] if self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND "
            "name='retained_receipt_recovery'").fetchone() else 0)
        return (charges_and_recovery + int(reserved) + int(imported) +
                int(recovery_reserved))

    def reserve_pair(self, pair_key: str, media_bytes: int, control_allowance: int,
                     metadata: dict) -> None:
        with self._transaction():
            account, run = self._run(fast_per_request=True)
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
            account, run = self._run(fast_per_request=True)
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
            cursor = self.db.execute("INSERT INTO retained_attempt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                attempt_id, self.release_id, now, slot, slot + timeout_s,
                timeout_s, None, None, body_bytes, kind, method, target_category,
                target_sha256, operation_key, "unresolved", None,
            ))
            if self._closed_fingerprint is not None and cursor.lastrowid is not None:
                self._closed_fingerprint["max_attempt_rowid"] = max(
                    int(self._closed_fingerprint["max_attempt_rowid"]),
                    int(cursor.lastrowid),
                )
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
        account, run = self._run(fast_per_request=True)
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
            account, run = self._run(fast_per_request=True)
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
            account, _ = self._run(fast_per_request=True)
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
            if status_code in (403, 429):
                self.db.execute("UPDATE retained_account SET cooldown_until=MAX(cooldown_until,?) WHERE id=1",
                                (now + QUOTA_COOLDOWN_SECONDS,))
            self.db.execute("UPDATE retained_account SET last_clock=? WHERE id=1", (now,))

    def unresolved_session_media_attempts(self, session_url: str) -> list[dict]:
        """Return unreconciled media_chunk attempts for a resumable session URL."""
        if (not isinstance(session_url, str) or not session_url or
                len(session_url) > 4096 or any(c.isspace() for c in session_url)):
            raise AuthorityBlocked("Invalid resumable session URL")
        self._run(fast_per_request=True)
        target_sha256 = hashlib.sha256(session_url.encode("utf-8")).hexdigest()
        resolved = self._reconciled_attempt_ids(self.release_id)
        rows = self.db.execute(
            "SELECT * FROM retained_attempt WHERE run_id=? AND target_sha256=? "
            "AND kind='media_chunk' ORDER BY at, rowid",
            (self.release_id, target_sha256),
        ).fetchall()
        settled_prior_bytes = 0
        out: list[dict] = []
        for row in rows:
            if row["status"] == "settled":
                settled_prior_bytes += int(row["bytes"])
            elif row["id"] not in resolved:
                out.append({
                    "attempt_id": row["id"],
                    "bytes": int(row["bytes"]),
                    "http_status": row["http_status"],
                    "dispatched_at": row["dispatched_at"],
                    "settled_prior_bytes": settled_prior_bytes,
                })
        return out

    def reconcile_attempt_by_status_probe(
        self, *, failed_attempt_id: str, probe_attempt_id: str,
        acknowledged_bytes: int, committed_bytes: int,
    ) -> str:
        """Reconcile a transient 5xx or crash-interrupted session attempt after a settled status probe.

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
            account, run = self._run(fast_per_request=True)
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
            is_status_attempt = (
                failed is not None and
                failed["kind"] == "resumable_status" and
                failed["target_category"] == "resumable_status" and
                failed["bytes"] == 0 and committed_bytes == 0
            )
            is_media_attempt = (
                failed is not None and
                failed["kind"] == "media_chunk" and
                failed["target_category"] == "media_chunk" and
                0 < failed["bytes"] <= (8 << 20) and
                (probe is None or probe["http_status"] not in (200, 201) or
                 committed_bytes == failed["bytes"])
            )
            if (prior is not None or failed is None or probe is None or
                failed["status"] != "unresolved" or
                failed["http_status"] not in (None, 500, 502, 503, 504) or
                not (is_media_attempt or is_status_attempt) or
                failed["method"] != "PUT" or failed["dispatched_at"] is None or
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
                committed_bytes > acknowledged_bytes):
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
                if (item["http_status"] in (None, 500, 502, 503, 504) and
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
            self.db.execute(
                "UPDATE retained_account SET last_clock=? WHERE id=1", (now,),
            )
            self._validate_open_and_fingerprint()
            return digest

    def mark_pair_verified(self, pair_key: str, mapping_sha256: str) -> None:
        if not _valid_digest(mapping_sha256):
            raise AuthorityBlocked("Verified mapping digest missing")
        with self._transaction():
            account, run = self._run(fast_per_request=True)
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
            self._validation_stats["close_pair_full_validates"] += 1
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
            account, run = self._run(allow_recovery=True, fast_per_request=True)
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
            used = (
                int(self._closed_fingerprint["closed_mapping_bytes"])
                + int(
                    self.db.execute(
                        "SELECT COALESCE(SUM(m.bytes),0) "
                        "FROM retained_mapping_reservation m "
                        "JOIN retained_run r ON r.id=m.run_id "
                        "WHERE r.status!='closed'"
                    ).fetchone()[0]
                )
                if self._closed_fingerprint is not None
                else int(
                    self.db.execute(
                        "SELECT COALESCE(SUM(bytes),0) FROM retained_mapping_reservation"
                    ).fetchone()[0]
                )
            )
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
            account, run = self._run(allow_recovery=True, fast_per_request=True)
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
            self._closed_fingerprint = None

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
