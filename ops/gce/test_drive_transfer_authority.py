"""Synthetic-only checks for the review-draft Drive transfer authority."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from drive_transfer_authority import (
    AuthorityBlocked, MAX_RUN_SECONDS, PACING_SECONDS, PILOT_CAP_BYTES,
    DEFAULT_PROVIDER_BOUND_BYTES, PROVIDER_REFERENCE_BYTES,
    ROLLING_CAP_BYTES, TransferAuthority, WINDOW_SECONDS,
    STRICT_HISTORY, DIAGNOSTIC_QUOTA_OBSERVATION,
    DIAGNOSTIC_PAIR_CAP_BYTES, DIAGNOSTIC_PAYLOAD_BYTES, DIAGNOSTIC_RECEIPT_BYTES,
    DIAGNOSTIC_CONTROL_BYTES, DIAGNOSTIC_PAYLOAD_OBJECT, DIAGNOSTIC_RECEIPT_OBJECT,
    DIAGNOSTIC_PAIR_KEY,
    main, read_existing_oauth_permission_id,
)


class FakeClock:
    def __init__(self, value=1_000_000.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class AuthorityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "authority.sqlite"
        self.clock = FakeClock()
        self.monotonic = self.clock
        self.manifest = "a" * 64
        self.account = "synthetic-account"
        self.writer = "synthetic-unique-writer"
        self.release = "synthetic-release"

    def evidence(self, prior_charges=(), **changes):
        normalized = sorted((float(c["latest_possible_send_at"]), c["bytes"], int(c["unresolved"]))
                            for c in prior_charges)
        value = {
            "account_id": self.account,
            "account_identity_verified": True,
            "all_writers_excluded": True,
            "exclusion_valid_until": self.clock() + MAX_RUN_SECONDS,
            "observed_at": self.clock(),
            "history_basis": "reconciled",
            "history_time_basis": "latest_possible_send_or_completion",
            "all_writers_included": True,
            "all_attempts_included": True,
            "uncertain_attempts_included": True,
            "evidence_sha256": "b" * 64,
            "provider_bound_bytes": DEFAULT_PROVIDER_BOUND_BYTES,
            "provider_bound_basis": "conservative_workspace_reference",
            "history_sha256": hashlib.sha256(json.dumps(
                normalized, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest(),
        }
        value.update(changes)
        return value

    def create(self, *, prior_charges=(), evidence=None, budget=PILOT_CAP_BYTES,
               provider_bound=DEFAULT_PROVIDER_BOUND_BYTES):
        TransferAuthority.initialize_release(
            self.path, release_id=self.release, writer_id=self.writer,
            account_id=self.account, manifest_sha256=self.manifest,
            evidence=self.evidence(prior_charges) if evidence is None else evidence,
            prior_charges=prior_charges, deadline_at=self.clock() + MAX_RUN_SECONDS,
            pilot_budget_bytes=budget, provider_bound_bytes=provider_bound,
            clock=self.clock,
        )
        return self.open()

    def open(self, allowed_mode=STRICT_HISTORY):
        result = TransferAuthority.open_existing(
            self.path, release_id=self.release, writer_id=self.writer,
            account_id=self.account, manifest_sha256=self.manifest, clock=self.clock,
            monotonic_clock=self.monotonic, allowed_mode=allowed_mode,
        )
        self.addCleanup(result.close)
        return result

    def reserve(self, authority, *, key="one", media=100, control=10):
        authority.reserve_pair(key, media, control, {"manifest_sha256": self.manifest})

    def request(self, authority, body_bytes, timeout_s, kind, *, method="PUT", range_key="none"):
        operation_key = hashlib.sha256(f"{kind}:{range_key}".encode()).hexdigest()
        return authority.start_request(
            body_bytes, timeout_s, kind, method=method, target_category=kind,
            target_sha256=hashlib.sha256(b"synthetic-target").hexdigest(),
            operation_key=operation_key,
        )

    def diagnostic_manifest(self):
        payload = {
            "object": DIAGNOSTIC_PAYLOAD_OBJECT, "generation": 1792518093053185,
            "bytes": DIAGNOSTIC_PAYLOAD_BYTES,
            "sha256": "5417c6724312db880c07bf1cb514e54e1d735d6d1ce347c45e283bd2f59457aa",
        }
        receipt = {
            "object": DIAGNOSTIC_RECEIPT_OBJECT, "generation": 1792518094289877,
            "bytes": DIAGNOSTIC_RECEIPT_BYTES,
            "sha256": "fe4bc8db1aceb0c4868d9c1c82866925784b48be31db91d5301d3f1f51945c44",
        }
        return {
            "schema": "catbird.drive.diagnostic.v1", "release_id": self.release,
            "account_id": self.account, "bucket": "catbird-436600-jetstream-buffer-20261020",
            "source_id": "gce-selfhost-001", "source_epoch": "buffer-admission-20261020",
            "drive_root": "16Zd7n4pwTSDMDsnP9chX2Ci0r0ahEy45",
            "payload": payload, "receipt": receipt,
            "folder_name": "catbird-drive-diagnostic-" + self.release,
            "control_allowance_bytes": DIAGNOSTIC_CONTROL_BYTES,
            "max_charged_bytes": DIAGNOSTIC_PAIR_CAP_BYTES,
            "max_duration_seconds": MAX_RUN_SECONDS,
            "retain_gcs": True, "retain_local": True, "archive_qualified": False,
        }

    def observation_evidence(self, **changes):
        value = {
            "account_id": self.account, "account_identity_verified": True,
            "all_writers_excluded": True,
            "exclusion_valid_until": self.clock() + MAX_RUN_SECONDS,
            "observed_at": self.clock(), "evidence_sha256": "b" * 64,
            "history_basis": "unproven", "quota_rejection_accepted": True,
        }
        value.update(changes)
        return value

    def initialize_observation(self, *, evidence=None, manifest=None, deadline_seconds=MAX_RUN_SECONDS):
        self.release = "c" * 32
        document = self.diagnostic_manifest() if manifest is None else manifest
        raw = json.dumps(document, sort_keys=True).encode()
        self.manifest = hashlib.sha256(raw).hexdigest()
        TransferAuthority.initialize_diagnostic_quota_observation(
            self.path, release_id=self.release, writer_id=self.writer,
            account_id=self.account, manifest_bytes=raw,
            evidence=self.observation_evidence() if evidence is None else evidence,
            deadline_at=self.clock() + deadline_seconds, clock=self.clock,
        )

    def create_observation(self, *, evidence=None, manifest=None, deadline_seconds=MAX_RUN_SECONDS):
        self.initialize_observation(evidence=evidence, manifest=manifest,
                                    deadline_seconds=deadline_seconds)
        return self.open(allowed_mode=DIAGNOSTIC_QUOTA_OBSERVATION)

    def reserve_observation(self, authority):
        authority.reserve_pair(
            DIAGNOSTIC_PAIR_KEY,
            DIAGNOSTIC_PAYLOAD_BYTES + DIAGNOSTIC_RECEIPT_BYTES,
            DIAGNOSTIC_CONTROL_BYTES,
            {
                "manifest_sha256": self.manifest,
                "diagnostic_folder_name": "catbird-drive-diagnostic-" + self.release,
                "drive_root": "16Zd7n4pwTSDMDsnP9chX2Ci0r0ahEy45",
                "payload_object": DIAGNOSTIC_PAYLOAD_OBJECT,
                "payload_generation": 1792518093053185,
                "receipt_object": DIAGNOSTIC_RECEIPT_OBJECT,
                "receipt_generation": 1792518094289877,
            },
        )

    def test_missing_or_corrupt_state_never_implies_zero_history(self):
        with self.assertRaises(AuthorityBlocked):
            self.open()
        self.path.write_bytes(b"not a SQLite database")
        with self.assertRaises(AuthorityBlocked):
            self.open()

    def test_explicit_account_and_writer_history_provenance_required(self):
        for altered in (
            {"account_identity_verified": False},
            {"all_writers_excluded": False},
            {"all_attempts_included": False},
            {"uncertain_attempts_included": False},
            {"evidence_sha256": ""},
            {"observed_at": self.clock() - 301},
        ):
            with self.subTest(altered=altered), self.assertRaises(AuthorityBlocked):
                TransferAuthority.initialize_release(
                    self.path, release_id=self.release, writer_id=self.writer,
                    account_id=self.account, manifest_sha256=self.manifest,
                    evidence=self.evidence(**altered), prior_charges=(),
                    deadline_at=self.clock() + MAX_RUN_SECONDS, clock=self.clock,
                )

    def test_quiescence_requires_full_day_and_no_outstanding_sessions(self):
        basis = {"history_basis": "quiescence", "quiet_since": self.clock() - WINDOW_SECONDS,
                 "latest_possible_send_at": self.clock() - WINDOW_SECONDS,
                 "no_outstanding_sessions": True}
        authority = self.create(evidence=self.evidence(**basis))
        self.reserve(authority)
        self.assertEqual(authority.snapshot()["rolling_reserved_and_charged_bytes"], 110)

    def test_exact_rolling_cap_and_overlap(self):
        old = {"latest_possible_send_at": self.clock() - 10, "bytes": DEFAULT_PROVIDER_BOUND_BYTES - 110,
               "unresolved": False}
        authority = self.create(prior_charges=[old])
        self.reserve(authority)
        self.assertEqual(authority.snapshot()["rolling_reserved_and_charged_bytes"], DEFAULT_PROVIDER_BOUND_BYTES)
        self.assertEqual(authority.snapshot()["effective_rolling_cap_bytes"], DEFAULT_PROVIDER_BOUND_BYTES)
        self.assertGreater(ROLLING_CAP_BYTES, PROVIDER_REFERENCE_BYTES)
        self.assertEqual(PROVIDER_REFERENCE_BYTES - DEFAULT_PROVIDER_BOUND_BYTES, 50_000_000_000)
        with self.assertRaises(AuthorityBlocked):
            self.reserve(authority, key="other", media=1, control=1)

    def test_reservation_one_byte_above_cap_denied(self):
        old = {"latest_possible_send_at": self.clock() - 10, "bytes": DEFAULT_PROVIDER_BOUND_BYTES - 109,
               "unresolved": False}
        authority = self.create(prior_charges=[old])
        with self.assertRaises(AuthorityBlocked):
            self.reserve(authority)
        self.assertEqual(authority.snapshot()["pair_reserved_remaining"], 0)

    def test_provider_bound_can_only_be_lowered_with_matching_provenance(self):
        lower = 1_000
        authority = self.create(provider_bound=lower,
                                evidence=self.evidence(provider_bound_bytes=lower,
                                                       provider_bound_basis="account_verified"))
        self.assertEqual(authority.snapshot()["effective_rolling_cap_bytes"], lower)
        self.reserve(authority)
        with self.assertRaises(AuthorityBlocked):
            TransferAuthority.initialize_release(
                Path(self.tmp.name) / "unsafe.sqlite", release_id="other", writer_id=self.writer,
                account_id=self.account, manifest_sha256=self.manifest,
                evidence=self.evidence(provider_bound_bytes=PROVIDER_REFERENCE_BYTES + 1),
                prior_charges=(), deadline_at=self.clock() + MAX_RUN_SECONDS,
                provider_bound_bytes=PROVIDER_REFERENCE_BYTES + 1, clock=self.clock,
            )

    def test_manifest_cannot_widen_stored_release_budget_or_duration(self):
        authority = self.create(budget=1_000)
        authority.validate_manifest_limits(1_000, MAX_RUN_SECONDS)
        with self.assertRaises(AuthorityBlocked):
            authority.validate_manifest_limits(999, MAX_RUN_SECONDS)
        with self.assertRaises(AuthorityBlocked):
            authority.validate_manifest_limits(1_000, MAX_RUN_SECONDS - 1)
        with self.assertRaises(AuthorityBlocked):
            authority.validate_manifest_limits(PILOT_CAP_BYTES + 1, MAX_RUN_SECONDS)

    def test_resolved_old_charges_age_out_but_uncertain_do_not(self):
        old = {"latest_possible_send_at": self.clock() - WINDOW_SECONDS + 1,
               "bytes": DEFAULT_PROVIDER_BOUND_BYTES - 109, "unresolved": False}
        authority = self.create(prior_charges=[old])
        with self.assertRaises(AuthorityBlocked):
            self.reserve(authority)
        self.clock.advance(2)
        self.reserve(authority)
        self.assertEqual(authority.snapshot()["rolling_reserved_and_charged_bytes"], 110)

    def test_concurrent_reservations_have_only_one_winner(self):
        authority = self.create()
        authority.close()

        def contender(key):
            opened = TransferAuthority.open_existing(
                self.path, release_id=self.release, writer_id=self.writer,
                account_id=self.account, manifest_sha256=self.manifest,
                clock=self.clock, monotonic_clock=self.monotonic,
            )
            try:
                self.reserve(opened, key=key)
                return "admitted"
            except AuthorityBlocked:
                return "blocked"
            finally:
                opened.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(contender, ("one", "two")))
        self.assertCountEqual(results, ["admitted", "blocked"])
        self.assertEqual(self.open().snapshot()["rolling_reserved_and_charged_bytes"], 110)

    def test_reservation_and_ambiguous_attempt_survive_reopen(self):
        authority = self.create()
        self.reserve(authority)
        authority.close()
        reopened = self.open()
        self.assertEqual(reopened.snapshot()["pair_reserved_remaining"], 110)
        attempt, _ = self.request(reopened, 40, 20, "media")
        reopened.close()  # Crash before HTTP response is settled.
        recovered = self.open()
        self.assertEqual(recovered.snapshot()["unresolved_attempts"], 1)
        self.assertEqual(recovered.snapshot()["rolling_reserved_and_charged_bytes"], 110)
        with self.assertRaises(AuthorityBlocked):
            recovered.close_pair("one")
        recovered.finish_request(attempt, 308)
        self.assertEqual(recovered.snapshot()["unresolved_attempts"], 0)

    def test_repeated_media_body_consumes_new_reservation_and_is_blocked(self):
        authority = self.create()
        self.reserve(authority)
        first, _ = self.request(authority, 100, 20, "media")
        # Lost response remains charged; an automatic replay cannot reuse it.
        with self.assertRaises(AuthorityBlocked):
            self.request(authority, 100, 20, "media")
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 1)
        self.assertEqual(authority.snapshot()["pair_reserved_remaining"], 10)
        authority.finish_request(first, None)

    def test_imported_history_digest_detects_missing_rows(self):
        charge = {"latest_possible_send_at": self.clock() - 10, "bytes": 20, "unresolved": False}
        authority = self.create(prior_charges=[charge])
        authority.db.execute("DELETE FROM history_charge")
        with self.assertRaises(AuthorityBlocked):
            authority.assert_active()

    def test_missing_attempt_row_fails_closed(self):
        authority = self.create()
        self.reserve(authority)
        attempt, _ = self.request(authority, 20, 20, "media")
        authority.db.execute("DELETE FROM attempt WHERE id=?", (attempt,))
        with self.assertRaises(AuthorityBlocked):
            authority.assert_active()

    def test_pacing_persists_across_reopen_and_has_no_burst(self):
        authority = self.create()
        self.reserve(authority)
        first, slot1 = self.request(authority, 0, 20, "lookup")
        authority.finish_request(first, 200)
        second, slot2 = self.request(authority, 0, 20, "status")
        authority.finish_request(second, 308)
        authority.close()
        third, slot3 = self.request(self.open(), 0, 20, "verify")
        self.assertGreaterEqual(slot2 - slot1, PACING_SECONDS - 1e-9)
        self.assertGreaterEqual(slot3 - slot2, PACING_SECONDS - 1e-9)
        self.assertEqual(self.open().snapshot()["unresolved_attempts"], 1)
        self.open().finish_request(third, 200)

    def test_post_wait_check_refuses_an_early_or_regressed_slot(self):
        authority = self.create()
        self.reserve(authority)
        first, _ = self.request(authority, 0, 20, "lookup")
        authority.finish_request(first, 200)
        second, future_slot = self.request(authority, 0, 20, "verify")
        with self.assertRaises(AuthorityBlocked):
            authority.assert_active(20, slot_at=future_slot)
        self.clock.advance(PACING_SECONDS)
        authority.assert_active(20, slot_at=future_slot)
        authority.finish_request(second, 200)

    def test_late_dispatch_moves_next_persisted_slot(self):
        authority = self.create()
        self.reserve(authority)
        first, assigned = self.request(authority, 10, 20, "media")
        self.assertEqual(assigned, self.clock())
        self.clock.advance(5)
        authority.mark_dispatch(first)
        authority.finish_request(first, 200)
        authority.close()
        second, next_slot = self.request(self.open(), 0, 20, "verify")
        self.assertGreaterEqual(next_slot, self.clock() + PACING_SECONDS - 1e-9)
        with self.assertRaises(AuthorityBlocked):
            self.open().mark_dispatch(second)  # Cannot dispatch before its slot.
        self.clock.advance(PACING_SECONDS)
        self.open().mark_dispatch(second)

    def test_assigned_slot_plus_timeout_cannot_cross_deadline(self):
        authority = self.create()
        self.reserve(authority)
        authority.db.execute("UPDATE release SET next_slot=? WHERE id=?",
                             (self.clock() + MAX_RUN_SECONDS - 10, self.release))
        with self.assertRaises(AuthorityBlocked):
            self.request(authority, 1, 20, "media")
        self.assertEqual(authority.snapshot()["pair_reserved_remaining"], 110)

    def test_quota_error_cools_down_and_retains_uncertain_bytes(self):
        authority = self.create()
        self.reserve(authority)
        attempt, _ = self.request(authority, 20, 20, "control")
        authority.finish_request(attempt, 429)
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 1)
        with self.assertRaises(AuthorityBlocked):
            self.request(authority, 0, 20, "lookup")

    def test_clock_regression_and_foreign_writer_fail_closed(self):
        authority = self.create()
        self.reserve(authority)
        self.clock.advance(-1)
        with self.assertRaises(AuthorityBlocked):
            self.request(authority, 0, 20, "lookup")
        self.clock.advance(1)
        with self.assertRaises(AuthorityBlocked):
            TransferAuthority.open_existing(
                self.path, release_id=self.release, writer_id="different-writer",
                account_id=self.account, manifest_sha256=self.manifest, clock=self.clock,
            )

    def test_pair_settlement_releases_only_unused_reservation(self):
        authority = self.create()
        self.reserve(authority)
        attempt, _ = self.request(authority, 20, 20, "media")
        authority.finish_request(attempt, 200)
        authority.close_pair("one")
        self.assertEqual(authority.db.execute("SELECT reserved_remaining FROM pair").fetchone()[0], 0)
        self.assertEqual(authority.db.execute("SELECT SUM(bytes) FROM attempt").fetchone()[0], 20)
        authority.close_pair("one")  # Identical verified closure is idempotent.
        with self.assertRaises(AuthorityBlocked):
            self.request(authority, 1, 20, "media")

    def test_setup_process_hands_unbound_ledger_to_one_different_writer_process(self):
        TransferAuthority.initialize_release(
            self.path, release_id=self.release, writer_id=self.writer,
            account_id=self.account, manifest_sha256=self.manifest,
            evidence=self.evidence(), prior_charges=(),
            deadline_at=self.clock() + MAX_RUN_SECONDS, clock=self.clock,
        )
        self.assertIsNone(self._raw_binding()[0])
        self.assertEqual(self._child_open(), 0)
        self.assertIsNotNone(self._raw_binding()[0])
        with self.assertRaises(AuthorityBlocked):
            self.open()

    def _raw_binding(self):
        import sqlite3
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute("SELECT process_host,process_pid,process_nonce FROM release").fetchone()

    def _child_open(self):
        script = (
            "import sys; from pathlib import Path; "
            "from drive_transfer_authority import TransferAuthority; "
            "a=TransferAuthority.open_existing(Path(sys.argv[1]),"
            "release_id='synthetic-release',writer_id='synthetic-unique-writer',"
            "account_id='synthetic-account',manifest_sha256='a'*64,"
            "clock=lambda:1000000.0); a.assert_active(); a.close()"
        )
        child_dir = Path(self.tmp.name) / "other_work_dir"
        child_dir.mkdir(exist_ok=True)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                           PYTHONPATH=str(Path(__file__).resolve().parent))
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.path)], cwd=child_dir,
            env=environment, capture_output=True, text=True, check=False,
        )
        return result.returncode

    def test_same_writer_id_in_other_process_cannot_claim_live_or_expired_lease(self):
        authority = self.create()
        self.assertNotEqual(self._child_open(), 0)
        self.clock.advance(MAX_RUN_SECONDS + 1)
        self.assertNotEqual(self._child_open(), 0)
        with self.assertRaises(AuthorityBlocked):
            authority.assert_active()

    def test_two_connections_revalidate_actual_dispatch_spacing(self):
        first_connection = self.create()
        self.reserve(first_connection)
        second_connection = self.open()
        older, _ = self.request(first_connection, 0, 20, "lookup")
        self.clock.advance(PACING_SECONDS)
        newer, _ = self.request(second_connection, 0, 20, "status")
        second_connection.mark_dispatch(newer)
        self.clock.advance(0.05)
        with self.assertRaises(AuthorityBlocked):
            first_connection.mark_dispatch(older)
        self.clock.advance(PACING_SECONDS - 0.05)
        first_connection.mark_dispatch(older)

    def test_forward_wall_jump_cannot_bypass_monotonic_pacing(self):
        self.monotonic = FakeClock(self.clock())
        authority = self.create()
        self.reserve(authority)
        first, _ = self.request(authority, 0, 20, "lookup")
        authority.mark_dispatch(first)
        self.clock.advance(100)
        second, _ = self.request(authority, 0, 20, "status")
        with self.assertRaises(AuthorityBlocked):
            authority.mark_dispatch(second)
        self.monotonic.advance(PACING_SECONDS)
        authority.mark_dispatch(second)

    def test_malformed_attempt_row_blocks_even_when_aggregate_bytes_balance(self):
        authority = self.create()
        self.reserve(authority)
        first, _ = self.request(authority, 20, 20, "media")
        second, _ = self.request(authority, 20, 20, "media", range_key="second")
        authority.db.execute("UPDATE attempt SET bytes=-1 WHERE id=?", (first,))
        authority.db.execute("UPDATE pair SET reserved_remaining=91 WHERE pair_key='one'")
        with self.assertRaises(AuthorityBlocked):
            authority.assert_active()
        authority.db.execute("UPDATE attempt SET bytes=20 WHERE id=?", (first,))
        authority.db.execute("UPDATE pair SET reserved_remaining=70 WHERE pair_key='one'")
        for column, bad in (
            ("method", "BAD"), ("target_sha256", "not-a-digest"),
            ("operation_key", "not-a-digest"), ("at", -1),
            ("slot_at", -1), ("timeout_s", -1),
            ("charge_through", -1), ("http_status", 999),
            ("status", "settled"),
        ):
            with self.subTest(column=column):
                good = authority.db.execute(f"SELECT {column} FROM attempt WHERE id=?", (first,)).fetchone()[0]
                authority.db.execute(f"UPDATE attempt SET {column}=? WHERE id=?", (bad, first))
                with self.assertRaises(AuthorityBlocked):
                    authority.assert_active()
                authority.db.execute(f"UPDATE attempt SET {column}=? WHERE id=?", (good, first))
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 2)

    def test_observation_mode_is_explicit_and_does_not_claim_history_proof(self):
        authority = self.create_observation()
        self.assertEqual(authority.admission_mode, DIAGNOSTIC_QUOTA_OBSERVATION)
        self.assertFalse(authority.snapshot()["history_proven"])
        self.assertIsNone(authority.snapshot()["effective_rolling_cap_bytes"])
        with self.assertRaises(AuthorityBlocked):
            self.open()  # Archive caller uses the strict default.
        authority.validate_manifest_limits(DIAGNOSTIC_PAIR_CAP_BYTES, MAX_RUN_SECONDS)
        self.reserve_observation(authority)
        self.assertEqual(authority.snapshot()["pair_reserved_remaining"], DIAGNOSTIC_PAIR_CAP_BYTES)

    def test_wrong_mode_open_does_not_claim_unbound_observation_writer(self):
        self.initialize_observation()
        self.assertIsNone(self._raw_binding()[0])
        with self.assertRaises(AuthorityBlocked):
            self.open()
        self.assertIsNone(self._raw_binding()[0])
        self.open(allowed_mode=DIAGNOSTIC_QUOTA_OBSERVATION)
        self.assertIsNotNone(self._raw_binding()[0])

    def test_strict_history_release_rejects_observation_caller(self):
        self.create()
        with self.assertRaises(AuthorityBlocked):
            self.open(allowed_mode=DIAGNOSTIC_QUOTA_OBSERVATION)

    def test_observation_requires_exact_manifest_and_duration(self):
        self.release = "c" * 32
        for alteration in (
            {"max_charged_bytes": DIAGNOSTIC_PAIR_CAP_BYTES + 1},
            {"max_duration_seconds": MAX_RUN_SECONDS + 1},
            {"archive_qualified": True},
            {"drive_parent": {"shard": 0, "id": "wrong"}},
            {"payload": {**self.diagnostic_manifest()["payload"], "bytes": 1}},
        ):
            with self.subTest(alteration=alteration), self.assertRaises(AuthorityBlocked):
                self.create_observation(manifest={**self.diagnostic_manifest(), **alteration})
        with self.assertRaises(AuthorityBlocked):
            self.create_observation(deadline_seconds=MAX_RUN_SECONDS + 1)

    def test_observation_requires_verified_account_exclusion_and_explicit_unproven_history(self):
        for alteration in (
            {"account_identity_verified": False}, {"all_writers_excluded": False},
            {"history_basis": "reconciled"}, {"quota_rejection_accepted": False},
            {"observed_at": self.clock() - 301},
            {"exclusion_valid_until": self.clock() + MAX_RUN_SECONDS - 1},
        ):
            with self.subTest(alteration=alteration), self.assertRaises(AuthorityBlocked):
                self.create_observation(evidence=self.observation_evidence(**alteration))

    def test_observation_rejects_other_pair_and_retains_ambiguous_charge(self):
        authority = self.create_observation()
        with self.assertRaises(AuthorityBlocked):
            authority.reserve_pair("other", 100, 1, {"manifest_sha256": self.manifest})
        self.reserve_observation(authority)
        attempt, _ = self.request(authority, 100, 20, "media")
        authority.finish_request(attempt, None)
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 1)
        self.assertEqual(authority.snapshot()["rolling_reserved_and_charged_bytes"],
                         DIAGNOSTIC_PAIR_CAP_BYTES)
        with self.assertRaises(AuthorityBlocked):
            authority.close_pair(DIAGNOSTIC_PAIR_KEY)

    def test_observation_429_cools_down_and_imported_history_is_not_claimed(self):
        authority = self.create_observation()
        self.reserve_observation(authority)
        attempt, _ = self.request(authority, 0, 20, "about", method="GET")
        authority.mark_dispatch(attempt)
        authority.finish_request(attempt, 429)
        with self.assertRaises(AuthorityBlocked):
            self.request(authority, 0, 20, "retry")
        authority.db.execute(
            "INSERT INTO history_charge(at,bytes,unresolved) VALUES (?,?,?)",
            (self.clock(), 1, 0),
        )
        with self.assertRaises(AuthorityBlocked):
            authority.assert_active()

    def test_offline_diagnostic_initializer_cli_persists_unbound_mode(self):
        self.clock.value = time.time()
        self.release = "c" * 32
        manifest_file = Path(self.tmp.name) / "manifest.json"
        evidence_file = Path(self.tmp.name) / "evidence.json"
        manifest_file.write_text(json.dumps(self.diagnostic_manifest()))
        evidence_file.write_text(json.dumps(self.observation_evidence()))
        output = io.StringIO()
        with redirect_stdout(output):
            result = main([
                "initialize-diagnostic", "--authority-db", str(self.path),
                "--manifest", str(manifest_file), "--evidence-json", str(evidence_file),
                "--release-id", self.release, "--writer-id", self.writer,
                "--account-id", self.account,
                "--deadline-epoch", str(self.clock() + 600),
            ])
        self.assertEqual(result, 0)
        self.assertFalse(json.loads(output.getvalue())["history_proven"])
        self.assertIsNone(self._raw_binding()[0])

    def test_account_preflight_helper_uses_existing_oauth_and_only_returns_id(self):
        fake_module = types.ModuleType("gce_segment_offloader")

        class FakeDriveClient:
            def __init__(self, oauth):
                self.oauth = oauth

            def token(self):
                return "synthetic-token-never-output"

        class FakeOpener:
            def open(self, request, timeout):
                self.request = request
                self.timeout = timeout
                return io.BytesIO(b'{"user":{"permissionId":"synthetic-permission-id"}}')

        opener = FakeOpener()
        fake_module.DriveClient = FakeDriveClient
        with mock.patch.dict(sys.modules, {"gce_segment_offloader": fake_module}), \
             mock.patch("urllib.request.build_opener", return_value=opener):
            result = read_existing_oauth_permission_id(Path(self.tmp.name) / "oauth.json")
        self.assertEqual(result, "synthetic-permission-id")
        self.assertEqual(opener.request.full_url,
                         "https://www.googleapis.com/drive/v3/about?fields=user(permissionId)")
        self.assertEqual(opener.request.get_method(), "GET")
        self.assertEqual(opener.timeout, 30)


if __name__ == "__main__":
    unittest.main()
