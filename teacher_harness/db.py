"""Transactional SQLite ledger for run state, requests, batches, and budget enforcement."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
from typing import Any, Iterator

from .config import (
    BudgetConstraintError,
    ConfigurationMismatchError,
    HarnessConfig,
    assert_no_credentials,
    canonical_sha256,
)

RESERVATION_HOLDING_STATUSES = (
    "reserved",
    "submitting",
    "submitted",
    "submission_ambiguous",
    "unresolved",
)

TERMINAL_RELEASE_STATUSES = (
    "succeeded",
    "refused",
    "truncated",
    "invalid",
    "errored",
    "expired",
    "canceled",
)


class ProcessLockError(RuntimeError):
    """Raised when another process holds the single-writer harness lock."""


class ArtifactIntegrityError(RuntimeError):
    """Raised when interrupted .partial files or modified immutable shards are found."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "v1_initial_ledger_and_requests",
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            run_id TEXT NOT NULL,
            generation_id TEXT NOT NULL,
            config_bundle_sha256 TEXT NOT NULL,
            input_manifest_sha256 TEXT NOT NULL,
            rubric_sha256 TEXT NOT NULL,
            schema_sha256 TEXT NOT NULL,
            pricing_sha256 TEXT NOT NULL,
            budget_halted INTEGER NOT NULL DEFAULT 0,
            halt_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS posts (
            uri TEXT PRIMARY KEY,
            cid TEXT,
            author_did TEXT NOT NULL,
            author_split_bucket INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            post_type TEXT NOT NULL,
            langs_json TEXT NOT NULL,
            language_in_scope INTEGER NOT NULL,
            unscored_reason TEXT,
            text TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            text_only_context_sha256 TEXT NOT NULL,
            context_enriched_sha256 TEXT NOT NULL,
            has_local_parent_or_quote_text INTEGER NOT NULL,
            near_dup_fingerprint TEXT NOT NULL,
            thread_component_id TEXT NOT NULL,
            split TEXT NOT NULL,
            leakage_reason TEXT,
            eligible_for_labeling INTEGER NOT NULL,
            eligibility_snapshot_json TEXT NOT NULL,
            sampling_stratum TEXT,
            sampling_method TEXT,
            inclusion_probability REAL,
            selection_priority_key TEXT,
            moderation_join_json TEXT NOT NULL,
            raw_row_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS requests (
            custom_id TEXT PRIMARY KEY,
            request_sha256 TEXT NOT NULL,
            uri TEXT NOT NULL,
            stage TEXT NOT NULL,
            model_tier TEXT NOT NULL,
            teacher_model TEXT NOT NULL,
            observation_mode TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            status TEXT NOT NULL,
            counted_input_tokens INTEGER NOT NULL,
            token_count_source TEXT NOT NULL,
            reserved_input_tokens INTEGER NOT NULL,
            reserved_output_tokens INTEGER NOT NULL,
            reserved_microdollars INTEGER NOT NULL,
            batch_id TEXT,
            provider_batch_id TEXT,
            returned_model TEXT,
            stop_reason TEXT,
            stop_details_json TEXT,
            actual_input_tokens INTEGER,
            actual_output_tokens INTEGER,
            actual_billed_microdollars INTEGER,
            response_sha256 TEXT,
            raw_shard_path TEXT,
            parsed_output_json TEXT,
            has_insufficient_component INTEGER,
            error_type TEXT,
            error_message TEXT,
            simulated INTEGER NOT NULL DEFAULT 0,
            wire_request_json TEXT NOT NULL,
            identity_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(uri, stage, teacher_model, observation_mode, attempt)
        );

        CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            client_submission_id TEXT NOT NULL UNIQUE,
            stage TEXT NOT NULL,
            model_tier TEXT NOT NULL,
            teacher_model TEXT NOT NULL,
            request_list_sha256 TEXT NOT NULL,
            request_count INTEGER NOT NULL,
            reserved_microdollars INTEGER NOT NULL,
            settled_microdollars INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            provider_batch_id TEXT,
            transport_mode TEXT NOT NULL,
            request_manifest_path TEXT NOT NULL,
            request_manifest_sha256 TEXT NOT NULL,
            raw_results_shard_path TEXT,
            raw_results_sha256 TEXT,
            ambiguity_reason TEXT,
            submitted_at TEXT,
            ended_at TEXT,
            reconciled_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS budget_ledger (
            entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            stage TEXT NOT NULL,
            batch_id TEXT,
            custom_id TEXT,
            entry_type TEXT NOT NULL,
            delta_reserved_microdollars INTEGER NOT NULL,
            delta_settled_microdollars INTEGER NOT NULL,
            delta_contingency_microdollars INTEGER NOT NULL,
            total_reserved_after INTEGER NOT NULL,
            total_settled_after INTEGER NOT NULL,
            total_contingency_after INTEGER NOT NULL,
            stage_reserved_after INTEGER NOT NULL,
            stage_settled_after INTEGER NOT NULL,
            note TEXT NOT NULL,
            prev_entry_sha256 TEXT NOT NULL,
            entry_sha256 TEXT NOT NULL
        );
        """,
    ),
    (
        2,
        "v2_routing_and_stage_gates",
        """
        CREATE TABLE IF NOT EXISTS routing_decisions (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            uri TEXT NOT NULL,
            target_stage TEXT NOT NULL,
            model_tier TEXT NOT NULL,
            routing_bucket TEXT NOT NULL,
            priority_key TEXT NOT NULL,
            rationale_json TEXT NOT NULL,
            blind_prompt_verified INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            UNIQUE(uri, target_stage, model_tier)
        );

        CREATE TABLE IF NOT EXISTS stage_gate_evidence (
            wave_name TEXT PRIMARY KEY,
            approved INTEGER NOT NULL,
            schema_valid_rate REAL NOT NULL,
            human_agreement_alpha REAL,
            evidence_receipt_json TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        );
        """,
    ),
]


@contextmanager
def acquire_writer_lock(run_dir: Path, command_name: str = "harness") -> Iterator[Path]:
    """Acquire an exclusive non-blocking fcntl file lock for the run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "writer.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            existing = ""
            try:
                existing = os.read(fd, 4096).decode("utf-8", errors="replace")
            except OSError:
                pass
            raise ProcessLockError(
                f"Run directory {run_dir} is locked by another process: {existing.strip()}"
            ) from exc

        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        payload = json.dumps({
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "command": command_name,
            "acquired_at": utc_now_iso(),
        }) + "\n"
        os.write(fd, payload.encode("utf-8"))
        os.fsync(fd)
        yield lock_path
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


class HarnessDB:
    """Transactional SQLite store for run/request/budget state."""

    def __init__(self, run_dir: Path, config: HarnessConfig):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.run_dir / "state.sqlite3"
        self.config = config
        self._apply_migrations()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _apply_migrations(self) -> None:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                );
                """
            )
            applied = {
                int(row["version"])
                for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
            }
            for version, name, sql_script in MIGRATIONS:
                if version not in applied:
                    conn.executescript(sql_script)
                    conn.execute("BEGIN IMMEDIATE;")
                    conn.execute(
                        "INSERT OR IGNORE INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                        (version, name, utc_now_iso()),
                    )
                    conn.execute("COMMIT;")
            if conn.in_transaction:
                conn.execute("COMMIT;")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def check_no_partial_artifacts(self) -> None:
        """Refuse automatic resume if any interrupted .partial file exists in run_dir."""
        partials = sorted(self.run_dir.rglob(".*.partial"))
        if partials:
            names = [str(p.relative_to(self.run_dir)) for p in partials]
            raise ArtifactIntegrityError(
                f"Interrupted .partial artifact writes found ({names}); "
                "preserve/move them via recover_partial_artifacts() before resuming."
            )

    def recover_partial_artifacts(self) -> list[str]:
        """Move any interrupted .partial artifacts into run_dir/recovery/ for diagnosis."""
        recovery_dir = self.run_dir / "recovery"
        moved: list[str] = []
        for partial in sorted(self.run_dir.rglob(".*.partial")):
            if recovery_dir in partial.parents:
                continue
            recovery_dir.mkdir(parents=True, exist_ok=True)
            dest = recovery_dir / f"{partial.name}.{int(datetime.now(timezone.utc).timestamp() * 1000)}"
            os.replace(partial, dest)
            moved.append(str(dest.relative_to(self.run_dir)))
        return moved

    def verify_immutable_shards(self) -> None:
        """Verify that every persisted batch request manifest and response shard matches its recorded SHA-256."""
        from real_pipeline.common import sha256 as file_sha256

        conn = self.connect()
        try:
            rows = conn.execute("SELECT * FROM batches").fetchall()
            for row in rows:
                req_path = self.run_dir / row["request_manifest_path"]
                if not req_path.exists() or file_sha256(req_path) != row["request_manifest_sha256"]:
                    raise ArtifactIntegrityError(
                        f"Batch {row['batch_id']} request manifest missing or SHA-256 changed: {req_path}"
                    )
                if row["raw_results_shard_path"] and row["raw_results_sha256"]:
                    res_path = self.run_dir / row["raw_results_shard_path"]
                    if not res_path.exists() or file_sha256(res_path) != row["raw_results_sha256"]:
                        raise ArtifactIntegrityError(
                            f"Batch {row['batch_id']} raw results shard missing or SHA-256 changed: {res_path}"
                        )
        finally:
            conn.close()

    def init_or_verify_run(
        self,
        *,
        run_id: str,
        generation_id: str,
        input_manifest_sha256: str,
    ) -> dict[str, Any]:
        """Initialize run metadata or verify exact configuration match on resume."""
        self.check_no_partial_artifacts()
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute("SELECT * FROM run_metadata WHERE id = 1").fetchone()
            now = utc_now_iso()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO run_metadata (
                        id, run_id, generation_id, config_bundle_sha256,
                        input_manifest_sha256, rubric_sha256, schema_sha256, pricing_sha256,
                        budget_halted, halt_reason, created_at, updated_at
                    ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
                    """,
                    (
                        run_id,
                        generation_id,
                        self.config.bundle_sha256,
                        input_manifest_sha256,
                        self.config.rubric_sha256,
                        self.config.schema_sha256,
                        self.config.pricing_sha256,
                        now,
                        now,
                    ),
                )
                conn.execute("COMMIT;")
                return self.get_run_metadata()

            mismatches: list[str] = []
            if row["generation_id"] != generation_id:
                mismatches.append(f"generation_id ({row['generation_id']} != {generation_id})")
            if row["input_manifest_sha256"] != input_manifest_sha256:
                mismatches.append("input_manifest_sha256 changed")
            if row["rubric_sha256"] != self.config.rubric_sha256:
                mismatches.append("rubric_sha256 changed")
            if row["schema_sha256"] != self.config.schema_sha256:
                mismatches.append("schema_sha256 changed")
            if row["pricing_sha256"] != self.config.pricing_sha256:
                mismatches.append("pricing_sha256 changed")
            if row["config_bundle_sha256"] != self.config.bundle_sha256:
                mismatches.append("config_bundle_sha256 changed")

            conn.execute("COMMIT;")
            if mismatches:
                raise ConfigurationMismatchError(
                    "Refusing to resume run with incompatible configuration/generation: "
                    + ", ".join(mismatches)
                )
            return dict(row)
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def verify_existing_run_config(self) -> dict[str, Any]:
        """Verify that an existing run directory matches the current config bundle."""
        self.check_no_partial_artifacts()
        self.verify_immutable_shards()
        conn = self.connect()
        try:
            row = conn.execute("SELECT * FROM run_metadata WHERE id = 1").fetchone()
            if row is None:
                raise ValueError(f"Run directory {self.run_dir} is not initialized; run 'prepare' first")
            if (
                row["rubric_sha256"] != self.config.rubric_sha256
                or row["schema_sha256"] != self.config.schema_sha256
                or row["pricing_sha256"] != self.config.pricing_sha256
                or row["config_bundle_sha256"] != self.config.bundle_sha256
            ):
                raise ConfigurationMismatchError(
                    "Existing run state does not match current rubric/schema/pricing configuration"
                )
            return dict(row)
        finally:
            conn.close()

    def get_run_metadata(self) -> dict[str, Any]:
        conn = self.connect()
        try:
            row = conn.execute("SELECT * FROM run_metadata WHERE id = 1").fetchone()
            if row is None:
                raise ValueError("Run metadata not initialized")
            return dict(row)
        finally:
            conn.close()

    def store_prepared_posts(self, posts: list[dict[str, Any]]) -> None:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            for p in posts:
                assert_no_credentials(p, f"post:{p['uri']}")
                raw_copy = {
                    k: v
                    for k, v in p.items()
                    if k not in ("eligibility_snapshot", "moderation_join")
                }
                conn.execute(
                    """
                    INSERT OR REPLACE INTO posts (
                        uri, cid, author_did, author_split_bucket, created_at,
                        post_type, langs_json, language_in_scope, unscored_reason,
                        text, text_sha256, text_only_context_sha256, context_enriched_sha256,
                        has_local_parent_or_quote_text, near_dup_fingerprint, thread_component_id,
                        split, leakage_reason, eligible_for_labeling, eligibility_snapshot_json,
                        sampling_stratum, sampling_method, inclusion_probability,
                        selection_priority_key, moderation_join_json, raw_row_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p["uri"],
                        p.get("cid"),
                        p["author_did"],
                        int(p["author_split_bucket"]),
                        p["created_at"],
                        p["post_type"],
                        json.dumps(p["langs"], ensure_ascii=False),
                        1 if p["language_in_scope"] else 0,
                        p.get("unscored_reason"),
                        p["text"],
                        p["text_sha256"],
                        p["text_only_context_sha256"],
                        p["context_enriched_sha256"],
                        1 if p["has_local_parent_or_quote_text"] else 0,
                        p["near_dup_fingerprint"],
                        p["thread_component_id"],
                        p["split"],
                        p.get("leakage_reason"),
                        1 if p["eligible_for_labeling"] else 0,
                        json.dumps(p["eligibility_snapshot"], sort_keys=True),
                        p.get("sampling_stratum"),
                        p.get("sampling_method"),
                        p.get("inclusion_probability"),
                        p.get("selection_priority_key"),
                        json.dumps(p["moderation_join"], sort_keys=True),
                        json.dumps(raw_copy, sort_keys=True, ensure_ascii=False),
                    ),
                )
            conn.execute("COMMIT;")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def load_posts(
        self,
        *,
        split: str | None = None,
        eligible_only: bool = False,
        selected_only: bool = False,
    ) -> list[dict[str, Any]]:
        conn = self.connect()
        try:
            clauses: list[str] = []
            params: list[Any] = []
            if split is not None:
                clauses.append("split = ?")
                params.append(split)
            if eligible_only:
                clauses.append("eligible_for_labeling = 1")
            if selected_only:
                clauses.append("sampling_stratum IS NOT NULL")
            where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = conn.execute(f"SELECT * FROM posts {where_sql} ORDER BY uri ASC", params).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                d = json.loads(r["raw_row_json"])
                d["eligibility_snapshot"] = json.loads(r["eligibility_snapshot_json"])
                d["moderation_join"] = json.loads(r["moderation_join_json"])
                d["sampling_stratum"] = r["sampling_stratum"]
                d["sampling_method"] = r["sampling_method"]
                d["inclusion_probability"] = r["inclusion_probability"]
                d["selection_priority_key"] = r["selection_priority_key"]
                out.append(d)
            return out
        finally:
            conn.close()

    def store_planned_requests(self, requests_list: list[dict[str, Any]]) -> dict[str, int]:
        """Insert planned requests idempotently without reserving budget yet."""
        conn = self.connect()
        inserted = 0
        skipped_existing = 0
        try:
            conn.execute("BEGIN IMMEDIATE;")
            now = utc_now_iso()
            for req in requests_list:
                assert_no_credentials(req["wire_request"], f"request:{req['custom_id']}")
                existing = conn.execute(
                    "SELECT custom_id, request_sha256, status FROM requests WHERE custom_id = ?",
                    (req["custom_id"],),
                ).fetchone()
                if existing is not None:
                    if existing["request_sha256"] != req["request_sha256"]:
                        raise ConfigurationMismatchError(
                            f"Custom ID {req['custom_id']} already exists with different request_sha256"
                        )
                    skipped_existing += 1
                    continue

                # Also check if an active or ambiguous request already exists for (uri, stage, teacher_model, observation_mode, attempt)
                dup = conn.execute(
                    """
                    SELECT custom_id, request_sha256 FROM requests
                    WHERE uri = ? AND stage = ? AND teacher_model = ? AND observation_mode = ? AND attempt = ?
                    """,
                    (
                        req["uri"],
                        req["stage"],
                        req["model_id"],
                        req["observation_mode"],
                        int(req["attempt"]),
                    ),
                ).fetchone()
                if dup is not None:
                    if dup["request_sha256"] != req["request_sha256"]:
                        raise ConfigurationMismatchError(
                            f"Request slot for {req['uri']} ({req['stage']}/{req['model_id']}) changed identity"
                        )
                    skipped_existing += 1
                    continue

                res = req["reservation"]
                conn.execute(
                    """
                    INSERT INTO requests (
                        custom_id, request_sha256, uri, stage, model_tier, teacher_model,
                        observation_mode, attempt, status, counted_input_tokens,
                        token_count_source, reserved_input_tokens, reserved_output_tokens,
                        reserved_microdollars, wire_request_json, identity_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        req["custom_id"],
                        req["request_sha256"],
                        req["uri"],
                        req["stage"],
                        req["model_tier"],
                        req["model_id"],
                        req["observation_mode"],
                        int(req["attempt"]),
                        int(res["counted_input_tokens"]),
                        req["token_count_source"],
                        int(res["reserved_input_tokens"]),
                        int(res["reserved_output_tokens"]),
                        int(res["reserved_microdollars"]),
                        json.dumps(req["wire_request"], sort_keys=True, ensure_ascii=False),
                        json.dumps(req["identity"], sort_keys=True, ensure_ascii=False),
                        now,
                        now,
                    ),
                )
                inserted += 1
            conn.execute("COMMIT;")
            return {"inserted": inserted, "skipped_existing": skipped_existing}
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def _compute_totals_in_tx(self, conn: sqlite3.Connection, stage: str | None = None) -> dict[str, int]:
        placeholders = ",".join("?" for _ in RESERVATION_HOLDING_STATUSES)
        total_reserved = int(
            conn.execute(
                f"SELECT COALESCE(SUM(reserved_microdollars), 0) FROM requests WHERE status IN ({placeholders})",
                RESERVATION_HOLDING_STATUSES,
            ).fetchone()[0]
        )
        total_settled = int(
            conn.execute(
                "SELECT COALESCE(SUM(actual_billed_microdollars), 0) FROM requests WHERE actual_billed_microdollars IS NOT NULL"
            ).fetchone()[0]
        )
        last_ledger = conn.execute(
            "SELECT total_contingency_after, entry_sha256 FROM budget_ledger ORDER BY entry_id DESC LIMIT 1"
        ).fetchone()
        total_contingency = int(last_ledger["total_contingency_after"]) if last_ledger else 0
        prev_sha = str(last_ledger["entry_sha256"]) if last_ledger else ("0" * 64)

        stage_reserved = 0
        stage_settled = 0
        if stage is not None:
            stage_reserved = int(
                conn.execute(
                    f"SELECT COALESCE(SUM(reserved_microdollars), 0) FROM requests WHERE stage = ? AND status IN ({placeholders})",
                    (stage, *RESERVATION_HOLDING_STATUSES),
                ).fetchone()[0]
            )
            stage_settled = int(
                conn.execute(
                    "SELECT COALESCE(SUM(actual_billed_microdollars), 0) FROM requests WHERE stage = ? AND actual_billed_microdollars IS NOT NULL",
                    (stage,),
                ).fetchone()[0]
            )
        return {
            "total_reserved": total_reserved,
            "total_settled": total_settled,
            "total_contingency": total_contingency,
            "stage_reserved": stage_reserved,
            "stage_settled": stage_settled,
            "prev_entry_sha256": prev_sha,
        }

    def _append_ledger_in_tx(
        self,
        conn: sqlite3.Connection,
        *,
        stage: str,
        batch_id: str | None,
        custom_id: str | None,
        entry_type: str,
        delta_reserved: int,
        delta_settled: int,
        delta_contingency: int,
        note: str,
    ) -> None:
        totals = self._compute_totals_in_tx(conn, stage)
        now = utc_now_iso()
        new_contingency = totals["total_contingency"] + delta_contingency
        payload = {
            "timestamp": now,
            "stage": stage,
            "batch_id": batch_id,
            "custom_id": custom_id,
            "entry_type": entry_type,
            "delta_reserved_microdollars": delta_reserved,
            "delta_settled_microdollars": delta_settled,
            "delta_contingency_microdollars": delta_contingency,
            "total_reserved_after": totals["total_reserved"],
            "total_settled_after": totals["total_settled"],
            "total_contingency_after": new_contingency,
            "stage_reserved_after": totals["stage_reserved"],
            "stage_settled_after": totals["stage_settled"],
            "note": note,
            "prev_entry_sha256": totals["prev_entry_sha256"],
        }
        entry_sha = canonical_sha256(payload)
        conn.execute(
            """
            INSERT INTO budget_ledger (
                timestamp, stage, batch_id, custom_id, entry_type,
                delta_reserved_microdollars, delta_settled_microdollars, delta_contingency_microdollars,
                total_reserved_after, total_settled_after, total_contingency_after,
                stage_reserved_after, stage_settled_after, note, prev_entry_sha256, entry_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                stage,
                batch_id,
                custom_id,
                entry_type,
                delta_reserved,
                delta_settled,
                delta_contingency,
                totals["total_reserved"],
                totals["total_settled"],
                new_contingency,
                totals["stage_reserved"],
                totals["stage_settled"],
                note,
                totals["prev_entry_sha256"],
                entry_sha,
            ),
        )

    def reserve_batch_for_submission(
        self,
        *,
        stage: str,
        model_tier: str,
        custom_ids: list[str],
        transport_mode: str,
        request_manifest_relpath: str,
        request_manifest_sha256: str,
    ) -> dict[str, Any]:
        """Atomically verify ceilings, reserve budget, and persist batch in 'submitting' state BEFORE API call."""
        if not custom_ids:
            raise ValueError("Cannot reserve an empty batch")
        ceilings = self.config.pricing["ceilings_microdollars"]
        if stage not in ceilings["stages"]:
            raise ValueError(f"Unknown budget stage: {stage!r}")

        total_cash_ceiling = int(ceilings["total_cash_ceiling"])
        gen_ceiling = int(ceilings["generation_allocation_ceiling"])
        stage_ceiling = int(ceilings["stages"][stage])
        outstanding_limit = int(ceilings["outstanding_liability_limits"][stage])

        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            meta = conn.execute("SELECT * FROM run_metadata WHERE id = 1").fetchone()
            if meta is None:
                raise ValueError("Run metadata not initialized")
            if int(meta["budget_halted"]) != 0:
                raise BudgetConstraintError(
                    f"Submissions are halted due to budget/ledger anomaly: {meta['halt_reason']}"
                )

            # Verify no ambiguous or unresolved batches exist for this stage before creating a new batch
            ambiguous = conn.execute(
                "SELECT batch_id, status FROM batches WHERE status IN ('submitting', 'submission_ambiguous', 'unresolved')"
            ).fetchall()
            if ambiguous:
                ids = [f"{r['batch_id']}({r['status']})" for r in ambiguous]
                raise BudgetConstraintError(
                    f"Cannot submit new batch while ambiguous/unresolved batches exist: {ids}. Run 'reconcile' first."
                )

            req_rows = []
            for cid in custom_ids:
                r = conn.execute("SELECT * FROM requests WHERE custom_id = ?", (cid,)).fetchone()
                if r is None:
                    raise ValueError(f"Request {cid} not found in database")
                if r["stage"] != stage or r["model_tier"] != model_tier:
                    raise ValueError(f"Request {cid} stage/model mismatch ({r['stage']}/{r['model_tier']})")
                if r["status"] != "planned":
                    raise BudgetConstraintError(
                        f"Request {cid} has status {r['status']!r}; only 'planned' requests can be reserved and submitted"
                    )
                req_rows.append(r)

            new_reservation = sum(int(r["reserved_microdollars"]) for r in req_rows)
            totals = self._compute_totals_in_tx(conn, stage)

            # Check 1: Outstanding liability limit ($2 in pilot, $5 afterward)
            if totals["stage_reserved"] + new_reservation > outstanding_limit:
                raise BudgetConstraintError(
                    f"Stage '{stage}' outstanding liability would be "
                    f"{totals['stage_reserved'] + new_reservation} microdollars, exceeding "
                    f"outstanding limit {outstanding_limit} microdollars"
                )

            # Check 2: Stage ceiling ($8 pilot, $80 haiku_bulk, $70 sonnet_review, $22 opus_review)
            if totals["stage_settled"] + totals["stage_reserved"] + new_reservation > stage_ceiling:
                raise BudgetConstraintError(
                    f"Stage '{stage}' settled ({totals['stage_settled']}) + outstanding ({totals['stage_reserved']}) "
                    f"+ new reservation ({new_reservation}) exceeds stage ceiling ({stage_ceiling} microdollars)"
                )

            # Check 3: Generation allocation ceiling ($180) and Total cash ceiling ($200)
            if totals["total_settled"] + totals["total_reserved"] + new_reservation > gen_ceiling:
                raise BudgetConstraintError(
                    f"Total settled ({totals['total_settled']}) + outstanding ({totals['total_reserved']}) "
                    f"+ new reservation ({new_reservation}) exceeds generation allocation ceiling ({gen_ceiling} microdollars)"
                )
            if (
                totals["total_settled"]
                + totals["total_reserved"]
                + totals["total_contingency"]
                + new_reservation
                > total_cash_ceiling
            ):
                raise BudgetConstraintError(
                    f"Total liability exceeds $200 cash ceiling ({total_cash_ceiling} microdollars)"
                )

            req_list_sha = canonical_sha256(sorted(custom_ids))
            batch_id = f"batch_{stage}_{model_tier}_{req_list_sha[:12]}"
            client_submission_id = f"sub_{batch_id}_{hashlib.sha256(req_list_sha.encode()).hexdigest()[:8]}"
            teacher_model = str(req_rows[0]["teacher_model"])
            now = utc_now_iso()

            conn.execute(
                """
                INSERT INTO batches (
                    batch_id, client_submission_id, stage, model_tier, teacher_model,
                    request_list_sha256, request_count, reserved_microdollars, settled_microdollars,
                    status, provider_batch_id, transport_mode, request_manifest_path,
                    request_manifest_sha256, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 'submitting', NULL, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    client_submission_id,
                    stage,
                    model_tier,
                    teacher_model,
                    req_list_sha,
                    len(custom_ids),
                    new_reservation,
                    transport_mode,
                    request_manifest_relpath,
                    request_manifest_sha256,
                    now,
                    now,
                ),
            )

            for cid in custom_ids:
                conn.execute(
                    """
                    UPDATE requests
                    SET status = 'submitting', batch_id = ?, updated_at = ?
                    WHERE custom_id = ?
                    """,
                    (batch_id, now, cid),
                )

            self._append_ledger_in_tx(
                conn,
                stage=stage,
                batch_id=batch_id,
                custom_id=None,
                entry_type="reserve",
                delta_reserved=new_reservation,
                delta_settled=0,
                delta_contingency=0,
                note=f"Reserved {len(custom_ids)} requests for {stage}/{model_tier}",
            )
            conn.execute("COMMIT;")
            return {
                "batch_id": batch_id,
                "client_submission_id": client_submission_id,
                "stage": stage,
                "model_tier": model_tier,
                "teacher_model": teacher_model,
                "request_count": len(custom_ids),
                "reserved_microdollars": new_reservation,
                "request_list_sha256": req_list_sha,
            }
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def mark_batch_submitted(self, batch_id: str, provider_batch_id: str) -> None:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            now = utc_now_iso()
            conn.execute(
                """
                UPDATE batches
                SET status = 'submitted', provider_batch_id = ?, submitted_at = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (provider_batch_id, now, now, batch_id),
            )
            conn.execute(
                """
                UPDATE requests
                SET status = 'submitted', provider_batch_id = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (provider_batch_id, now, batch_id),
            )
            conn.execute("COMMIT;")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def mark_batch_submission_ambiguous(self, batch_id: str, reason: str) -> None:
        """Record that external batch creation failed/timed out after Phase-A reservation."""
        assert_no_credentials(reason, "ambiguity_reason")
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            now = utc_now_iso()
            conn.execute(
                """
                UPDATE batches
                SET status = 'submission_ambiguous', ambiguity_reason = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (reason, now, batch_id),
            )
            conn.execute(
                """
                UPDATE requests
                SET status = 'submission_ambiguous', error_message = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (reason, now, batch_id),
            )
            conn.execute("COMMIT;")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def mark_batch_unresolved(self, batch_id: str, reason: str) -> None:
        """Keep reservation locked when ambiguous submission cannot be matched to provider state."""
        assert_no_credentials(reason, "unresolved_reason")
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            now = utc_now_iso()
            conn.execute(
                """
                UPDATE batches
                SET status = 'unresolved', ambiguity_reason = ?, updated_at = ?
                WHERE batch_id = ?
                """,
                (reason, now, batch_id),
            )
            conn.execute(
                """
                UPDATE requests
                SET status = 'unresolved', error_message = ?, updated_at = ?
                WHERE batch_id = ? AND status IN ('submitting', 'submission_ambiguous', 'submitted')
                """,
                (reason, now, batch_id),
            )
            conn.execute("COMMIT;")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def settle_batch_results(
        self,
        *,
        batch_id: str,
        raw_results_relpath: str,
        raw_results_sha256: str,
        reconciled_items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Atomically persist reconciled request outcomes, verify hashes, and settle/hold reservations."""
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            batch_row = conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
            if batch_row is None:
                raise ValueError(f"Batch {batch_id} not found")
            stage = str(batch_row["stage"])
            now = utc_now_iso()

            expected_reqs = {
                str(r["custom_id"]): dict(r)
                for r in conn.execute("SELECT * FROM requests WHERE batch_id = ?", (batch_id,)).fetchall()
            }
            seen_cids: set[str] = set()
            batch_settled_total = 0
            overrun_total = 0
            status_counts: dict[str, int] = {}

            for item in reconciled_items:
                cid = str(item["custom_id"])
                if cid not in expected_reqs:
                    raise ValueError(f"Unexpected custom_id {cid} in batch {batch_id} results")
                if cid in seen_cids:
                    raise ValueError(f"Duplicate custom_id {cid} in batch {batch_id} results")
                seen_cids.add(cid)

                req_row = expected_reqs[cid]
                outcome_status = str(item["status"])
                status_counts[outcome_status] = status_counts.get(outcome_status, 0) + 1

                reserved_micro = int(req_row["reserved_microdollars"])
                actual_in = item.get("actual_input_tokens")
                actual_out = item.get("actual_output_tokens")
                billed_micro = int(item.get("actual_billed_microdollars") or 0)
                response_sha = item.get("response_sha256")
                if outcome_status in TERMINAL_RELEASE_STATUSES and not response_sha:
                    raise ValueError(f"Terminal request {cid} must have a response_sha256 before releasing reservation")

                overrun = max(0, billed_micro - reserved_micro)
                if overrun > 0:
                    overrun_total += overrun

                if outcome_status in TERMINAL_RELEASE_STATUSES:
                    batch_settled_total += billed_micro

                conn.execute(
                    """
                    UPDATE requests
                    SET status = ?,
                        returned_model = ?,
                        stop_reason = ?,
                        stop_details_json = ?,
                        actual_input_tokens = ?,
                        actual_output_tokens = ?,
                        actual_billed_microdollars = ?,
                        response_sha256 = ?,
                        raw_shard_path = ?,
                        parsed_output_json = ?,
                        has_insufficient_component = ?,
                        error_type = ?,
                        error_message = ?,
                        simulated = ?,
                        updated_at = ?
                    WHERE custom_id = ?
                    """,
                    (
                        outcome_status,
                        item.get("returned_model"),
                        item.get("stop_reason"),
                        json.dumps(item["stop_details"], sort_keys=True) if item.get("stop_details") is not None else None,
                        actual_in,
                        actual_out,
                        billed_micro if outcome_status in TERMINAL_RELEASE_STATUSES else None,
                        response_sha,
                        raw_results_relpath,
                        json.dumps(item["parsed_output"], sort_keys=True, ensure_ascii=False)
                        if item.get("parsed_output") is not None
                        else None,
                        1 if item.get("has_insufficient_component") else (0 if item.get("has_insufficient_component") is False else None),
                        item.get("error_type"),
                        item.get("error_message"),
                        1 if item.get("simulated") else 0,
                        now,
                        cid,
                    ),
                )

                if outcome_status in TERMINAL_RELEASE_STATUSES:
                    self._append_ledger_in_tx(
                        conn,
                        stage=stage,
                        batch_id=batch_id,
                        custom_id=cid,
                        entry_type="settle" if billed_micro > 0 else "release_unbilled",
                        delta_reserved=-reserved_micro,
                        delta_settled=billed_micro,
                        delta_contingency=overrun,
                        note=f"Settled {cid} as {outcome_status} (billed={billed_micro}, reserved={reserved_micro})",
                    )

            # Any missing custom_ids in an ended batch become 'unresolved' and KEEP their reservations!
            missing_cids = sorted(set(expected_reqs.keys()) - seen_cids)
            for m_cid in missing_cids:
                status_counts["unresolved"] = status_counts.get("unresolved", 0) + 1
                conn.execute(
                    """
                    UPDATE requests
                    SET status = 'unresolved',
                        error_type = 'missing_in_batch_results',
                        error_message = 'Custom ID missing from provider batch results; reservation held',
                        updated_at = ?
                    WHERE custom_id = ?
                    """,
                    (now, m_cid),
                )

            batch_final_status = "unresolved" if missing_cids or status_counts.get("unresolved", 0) > 0 else "reconciled"
            conn.execute(
                """
                UPDATE batches
                SET status = ?,
                    settled_microdollars = ?,
                    raw_results_shard_path = ?,
                    raw_results_sha256 = ?,
                    ended_at = COALESCE(ended_at, ?),
                    reconciled_at = ?,
                    updated_at = ?
                WHERE batch_id = ?
                """,
                (
                    batch_final_status,
                    batch_settled_total,
                    raw_results_relpath,
                    raw_results_sha256,
                    now,
                    now,
                    now,
                    batch_id,
                ),
            )

            if overrun_total > 0:
                halt_msg = (
                    f"Billed cost exceeded reservation by {overrun_total} microdollars on batch {batch_id}; "
                    "charged to contingency reserve and halted new submissions pending review."
                )
                conn.execute(
                    "UPDATE run_metadata SET budget_halted = 1, halt_reason = ?, updated_at = ? WHERE id = 1",
                    (halt_msg, now),
                )
                self._append_ledger_in_tx(
                    conn,
                    stage=stage,
                    batch_id=batch_id,
                    custom_id=None,
                    entry_type="halt",
                    delta_reserved=0,
                    delta_settled=0,
                    delta_contingency=0,
                    note=halt_msg,
                )

            conn.execute("COMMIT;")
            return {
                "batch_id": batch_id,
                "batch_status": batch_final_status,
                "status_counts": status_counts,
                "missing_custom_ids": missing_cids,
                "settled_microdollars": batch_settled_total,
                "overrun_microdollars": overrun_total,
            }
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def create_retry_requests_for_eligible_failures(
        self,
        *,
        stage: str,
        max_attempts: int = 2,
    ) -> list[str]:
        """Create fresh-attempt 'planned' requests ONLY for eligible transient failures (errored_transient, expired)."""
        conn = self.connect()
        created_cids: list[str] = []
        try:
            conn.execute("BEGIN IMMEDIATE;")
            rows = conn.execute(
                """
                SELECT * FROM requests
                WHERE stage = ?
                  AND (
                      status = 'expired'
                      OR (status = 'errored' AND error_type IN ('server_error', 'overloaded_error', 'rate_limit_error', 'errored_transient'))
                  )
                ORDER BY custom_id ASC
                """,
                (stage,),
            ).fetchall()
            now = utc_now_iso()
            for r in rows:
                next_attempt = int(r["attempt"]) + 1
                if next_attempt > max_attempts:
                    continue
                already = conn.execute(
                    """
                    SELECT custom_id FROM requests
                    WHERE uri = ? AND stage = ? AND teacher_model = ? AND observation_mode = ? AND attempt = ?
                    """,
                    (r["uri"], r["stage"], r["teacher_model"], r["observation_mode"], next_attempt),
                ).fetchone()
                if already is not None:
                    continue

                from .schemas import make_custom_id

                new_cid = make_custom_id(
                    stage=r["stage"],
                    model_tier=r["model_tier"],
                    observation_mode=r["observation_mode"],
                    uri=r["uri"],
                    attempt=next_attempt,
                    request_sha256=r["request_sha256"],
                )
                wire_req = json.loads(r["wire_request_json"])
                wire_req["custom_id"] = new_cid

                conn.execute(
                    """
                    INSERT INTO requests (
                        custom_id, request_sha256, uri, stage, model_tier, teacher_model,
                        observation_mode, attempt, status, counted_input_tokens,
                        token_count_source, reserved_input_tokens, reserved_output_tokens,
                        reserved_microdollars, wire_request_json, identity_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_cid,
                        r["request_sha256"],
                        r["uri"],
                        r["stage"],
                        r["model_tier"],
                        r["teacher_model"],
                        r["observation_mode"],
                        next_attempt,
                        int(r["counted_input_tokens"]),
                        r["token_count_source"],
                        int(r["reserved_input_tokens"]),
                        int(r["reserved_output_tokens"]),
                        int(r["reserved_microdollars"]),
                        json.dumps(wire_req, sort_keys=True, ensure_ascii=False),
                        r["identity_json"],
                        now,
                        now,
                    ),
                )
                created_cids.append(new_cid)
            conn.execute("COMMIT;")
            return created_cids
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK;")
            raise
        finally:
            conn.close()

    def get_budget_summary(self) -> dict[str, Any]:
        conn = self.connect()
        try:
            ceilings = self.config.pricing["ceilings_microdollars"]
            meta = conn.execute("SELECT * FROM run_metadata WHERE id = 1").fetchone()
            totals = self._compute_totals_in_tx(conn)
            per_stage: dict[str, Any] = {}
            for stg, stg_ceiling in ceilings["stages"].items():
                st_totals = self._compute_totals_in_tx(conn, stg)
                out_limit = int(ceilings["outstanding_liability_limits"][stg])
                per_stage[stg] = {
                    "ceiling_microdollars": int(stg_ceiling),
                    "outstanding_limit_microdollars": out_limit,
                    "settled_microdollars": st_totals["stage_settled"],
                    "outstanding_reserved_microdollars": st_totals["stage_reserved"],
                    "remaining_stage_budget_microdollars": int(stg_ceiling)
                    - st_totals["stage_settled"]
                    - st_totals["stage_reserved"],
                }
            ledger_count = int(conn.execute("SELECT COUNT(*) FROM budget_ledger").fetchone()[0])
            return {
                "total_cash_ceiling_microdollars": int(ceilings["total_cash_ceiling"]),
                "generation_allocation_ceiling_microdollars": int(ceilings["generation_allocation_ceiling"]),
                "contingency_reserve_ceiling_microdollars": int(ceilings["contingency_reserve"]),
                "total_settled_microdollars": totals["total_settled"],
                "total_outstanding_reserved_microdollars": totals["total_reserved"],
                "total_contingency_used_microdollars": totals["total_contingency"],
                "remaining_total_cash_microdollars": int(ceilings["total_cash_ceiling"])
                - totals["total_settled"]
                - totals["total_reserved"]
                - totals["total_contingency"],
                "budget_halted": bool(meta["budget_halted"]) if meta else False,
                "halt_reason": meta["halt_reason"] if meta else None,
                "ledger_entries": ledger_count,
                "stages": per_stage,
            }
        finally:
            conn.close()

    def verify_ledger_hash_chain(self) -> bool:
        conn = self.connect()
        try:
            rows = conn.execute("SELECT * FROM budget_ledger ORDER BY entry_id ASC").fetchall()
            prev_sha = "0" * 64
            for r in rows:
                if r["prev_entry_sha256"] != prev_sha:
                    raise ArtifactIntegrityError(f"Ledger entry {r['entry_id']} prev_entry_sha256 mismatch")
                payload = {
                    "timestamp": r["timestamp"],
                    "stage": r["stage"],
                    "batch_id": r["batch_id"],
                    "custom_id": r["custom_id"],
                    "entry_type": r["entry_type"],
                    "delta_reserved_microdollars": int(r["delta_reserved_microdollars"]),
                    "delta_settled_microdollars": int(r["delta_settled_microdollars"]),
                    "delta_contingency_microdollars": int(r["delta_contingency_microdollars"]),
                    "total_reserved_after": int(r["total_reserved_after"]),
                    "total_settled_after": int(r["total_settled_after"]),
                    "total_contingency_after": int(r["total_contingency_after"]),
                    "stage_reserved_after": int(r["stage_reserved_after"]),
                    "stage_settled_after": int(r["stage_settled_after"]),
                    "note": r["note"],
                    "prev_entry_sha256": r["prev_entry_sha256"],
                }
                if canonical_sha256(payload) != r["entry_sha256"]:
                    raise ArtifactIntegrityError(f"Ledger entry {r['entry_id']} entry_sha256 tampered")
                prev_sha = r["entry_sha256"]
            return True
        finally:
            conn.close()

    def record_stage_gate_evidence(
        self,
        *,
        wave_name: str,
        approved: bool,
        schema_valid_rate: float,
        human_agreement_alpha: float | None,
        evidence_receipt: dict[str, Any],
    ) -> None:
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT OR REPLACE INTO stage_gate_evidence (
                    wave_name, approved, schema_valid_rate, human_agreement_alpha,
                    evidence_receipt_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    wave_name,
                    1 if approved else 0,
                    float(schema_valid_rate),
                    float(human_agreement_alpha) if human_agreement_alpha is not None else None,
                    json.dumps(evidence_receipt, sort_keys=True),
                    utc_now_iso(),
                ),
            )
            conn.execute("COMMIT;")
        finally:
            conn.close()

    def check_expansion_gate(self, target_wave: str) -> dict[str, Any]:
        """Verify that expanding beyond pilot requires recorded review/gate evidence."""
        if target_wave == "pilot":
            return {"allowed": True, "wave": "pilot", "prerequisite": None}
        prereq_map = {
            "wave_25k": "pilot",
            "wave_150k": "wave_25k",
            "wave_500k": "wave_150k",
        }
        if target_wave not in prereq_map:
            raise ValueError(f"Unknown expansion wave: {target_wave!r}")
        prereq = prereq_map[target_wave]
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT * FROM stage_gate_evidence WHERE wave_name = ?", (prereq,)
            ).fetchone()
            if row is None or int(row["approved"]) != 1:
                return {
                    "allowed": False,
                    "wave": target_wave,
                    "prerequisite": prereq,
                    "reason": f"Expansion to {target_wave} requires recorded approved gate evidence for {prereq}",
                }
            if float(row["schema_valid_rate"]) < 0.99:
                return {
                    "allowed": False,
                    "wave": target_wave,
                    "prerequisite": prereq,
                    "reason": f"Prerequisite {prereq} schema_valid_rate ({row['schema_valid_rate']:.4f}) < 0.99 gate",
                }
            return {
                "allowed": True,
                "wave": target_wave,
                "prerequisite": prereq,
                "evidence": json.loads(row["evidence_receipt_json"]),
            }
        finally:
            conn.close()
