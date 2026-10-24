"""Synthetic-only tests for the opt-in retained-copy account authority."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from drive_transfer_authority import AuthorityBlocked, DIAGNOSTIC_PAYLOAD_OBJECT, TransferAuthority
import gcs_drive_retained_overflow as overflow_module
from gcs_drive_retained_overflow import INDEX_PREFIX, pair_key, validate_overflow_manifest
import retained_transfer_authority as retained_module
from retained_transfer_authority import _validated_legacy_result


class FakeClock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, amount: float) -> None:
        self.value += amount


class RetainedAuthorityTests(unittest.TestCase):
    LEGACY_RELEASE_ID = "synthetic-legacy-release"
    LEGACY_ATTEMPTS = 2
    LEGACY_BYTES = 12

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "shared-account.sqlite"
        self.account_id = "synthetic-account"
        self.writer_id = "synthetic-single-writer"
        self.clock = FakeClock(time.time())
        self.monotonic = FakeClock(time.monotonic())
        TransferAuthority.initialize_retained_observation(
            self.db_path, writer_id=self.writer_id, account_id=self.account_id)
        self.clock.value = time.time() + 1
        self.monotonic.value = time.monotonic() + 1
        legacy_result = self.synthetic_legacy_result()
        self.legacy_raw = json.dumps(legacy_result, sort_keys=True).encode()
        self.legacy_digest = hashlib.sha256(self.legacy_raw).hexdigest()
        for name, value in (
            ("LEGACY_DIAGNOSTIC_SOURCE_SHA256", self.legacy_digest),
            ("LEGACY_DIAGNOSTIC_RELEASE_ID", self.LEGACY_RELEASE_ID),
            ("LEGACY_DIAGNOSTIC_ATTEMPTS", self.LEGACY_ATTEMPTS),
            ("LEGACY_DIAGNOSTIC_CHARGED_BYTES", self.LEGACY_BYTES),
        ):
            binding = patch.object(retained_module, name, value)
            binding.start()
            self.addCleanup(binding.stop)
        retained_module.RetainedObservationAuthority.import_legacy_result(
            self.db_path, result_bytes=self.legacy_raw, writer_id=self.writer_id,
            account_id=self.account_id, clock=self.clock)

    def synthetic_legacy_result(self, *, release_id: str | None = None) -> dict:
        release_id = release_id or self.LEGACY_RELEASE_ID
        first = self.clock() - 100
        second = first + 0.55
        return {
            "release": {
                "id": release_id, "status": "closed", "account_id": self.account_id,
                "admission_mode": "diagnostic_quota_observation",
                "started_at": first - 10, "deadline_at": self.clock() + 100,
                "next_slot": second + 0.55, "cooldown_until": 0,
            },
            "descriptor": {
                "release_id": release_id, "account_id": self.account_id,
                "writer_id": self.writer_id, "manifest_sha256": "a" * 64,
            },
            "attempts": [
                {"id": "legacy-1", "bytes": 5, "at": first, "slot_at": first,
                 "dispatched_at": first, "charge_through": first + 1,
                 "http_status": 200, "method": "PUT", "status": "settled"},
                {"id": "legacy-2", "bytes": 7, "at": second, "slot_at": second,
                 "dispatched_at": second, "charge_through": second + 1,
                 "http_status": 308, "method": "PUT", "status": "settled"},
            ],
            "charged_bytes": self.LEGACY_BYTES, "unresolved_attempts": 0,
            "history_proven": False, "archive_qualified": False,
            "entries": [{
                "status": "diagnostic_verified_unqualified",
                "archive_qualified": False,
                "payload_ref": {
                    "name": DIAGNOSTIC_PAYLOAD_OBJECT,
                    "generation": retained_module.LEGACY_DIAGNOSTIC_PAYLOAD_GENERATION,
                    "size": 268_980_444,
                },
            }],
        }

    def child_env(self) -> dict[str, str]:
        return dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                    PYTHONPATH=str(Path(__file__).resolve().parent),
                    SYNTHETIC_LEGACY_DIGEST=self.legacy_digest,
                    SYNTHETIC_LEGACY_RELEASE=self.LEGACY_RELEASE_ID,
                    SYNTHETIC_LEGACY_ATTEMPTS=str(self.LEGACY_ATTEMPTS),
                    SYNTHETIC_LEGACY_BYTES=str(self.LEGACY_BYTES))

    @staticmethod
    def child_patch_code() -> str:
        return (
            "import os; import retained_transfer_authority as rta; "
            "rta.LEGACY_DIAGNOSTIC_SOURCE_SHA256=os.environ['SYNTHETIC_LEGACY_DIGEST']; "
            "rta.LEGACY_DIAGNOSTIC_RELEASE_ID=os.environ['SYNTHETIC_LEGACY_RELEASE']; "
            "rta.LEGACY_DIAGNOSTIC_ATTEMPTS=int(os.environ['SYNTHETIC_LEGACY_ATTEMPTS']); "
            "rta.LEGACY_DIAGNOSTIC_CHARGED_BYTES=int(os.environ['SYNTHETIC_LEGACY_BYTES']); "
        )

    def manifest(self, index: int, *, release_id: str | None = None) -> dict:
        release_id = release_id or f"{index:032x}"
        payload_body = f"tiny-payload-{index}".encode()
        receipt_body = f"tiny-receipt-{index}".encode()
        payload_hash = hashlib.sha256(payload_body).hexdigest()
        receipt_hash = hashlib.sha256(receipt_body).hexdigest()
        payload_object = (
            "sources/gce-selfhost-001/buffer-admission-20261020/segments/"
            f"{index // 256:08x}/seg_{index:010x}.jss/{payload_hash}.jss"
        )
        # For single-digit indices, hex and base36 segment spellings coincide.
        payload = {"object": payload_object, "generation": index + 10,
                   "bytes": len(payload_body), "sha256": payload_hash}
        receipt = {"object": payload_object + f".g{index + 10}.receipt.json",
                   "generation": index + 100, "bytes": len(receipt_body),
                   "sha256": receipt_hash}
        manifest = {
            "schema": "catbird.drive.retained-overflow.v1",
            "release_id": release_id,
            "account_id": self.account_id,
            "bucket": "catbird-436600-jetstream-buffer-20261020",
            "source_id": "gce-selfhost-001",
            "source_epoch": "buffer-admission-20261020",
            "drive_root": "16Zd7n4pwTSDMDsnP9chX2Ci0r0ahEy45",
            "payload": payload,
            "receipt": receipt,
            "control_allowance_bytes": 1 << 20,
            "max_charged_bytes": len(payload_body) + len(receipt_body) + (1 << 20),
            "max_duration_seconds": 20 * 60,
            "retain_gcs": True,
            "retain_local": True,
            "archive_qualified": False,
        }
        manifest["folder_name"] = "catbird-overflow-" + pair_key(manifest)[:32]
        manifest["index_bucket"] = manifest["bucket"]
        manifest["index_object"] = INDEX_PREFIX + pair_key(manifest) + ".json"
        validate_overflow_manifest(manifest, release_id=release_id,
                                   account_id=self.account_id)
        return manifest

    @staticmethod
    def raw_and_digest(manifest: dict) -> tuple[bytes, str]:
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        return raw, hashlib.sha256(raw).hexdigest()

    def evidence(self, manifest: dict, digest: str, *, observed_at: float | None = None,
                 **changes) -> dict:
        value = {
            "run_id": manifest["release_id"],
            "writer_id": self.writer_id,
            "account_id": self.account_id,
            "reviewed_manifest_sha256": digest,
            "account_identity_verified": True,
            "all_writers_excluded": True,
            "exclusion_valid_until": self.clock() + 20 * 60,
            "observed_at": self.clock() if observed_at is None else observed_at,
            "evidence_sha256": "e" * 64,
            "history_basis": "unproven",
            "quota_rejection_accepted": True,
        }
        value.update(changes)
        return value

    def open_run(self, index: int, *, manifest: dict | None = None,
                 evidence: dict | None = None, boot_id: str = "synthetic-boot",
                 recovery_only: bool = False, max_concurrent_runs: int = 1):
        manifest = self.manifest(index) if manifest is None else manifest
        raw, digest = self.raw_and_digest(manifest)
        evidence = self.evidence(manifest, digest) if evidence is None else evidence
        authority = TransferAuthority.open_retained_run(
            self.db_path, run_id=manifest["release_id"], writer_id=self.writer_id,
            account_id=self.account_id, manifest_bytes=raw, evidence=evidence,
            deadline_at=self.clock() + 20 * 60, clock=self.clock,
            monotonic_clock=self.monotonic, boot_id=boot_id,
            recovery_only=recovery_only,
            max_concurrent_runs=max_concurrent_runs,
        )
        self.addCleanup(authority.close)
        return authority, manifest, digest

    @staticmethod
    def reserve(authority, manifest: dict, digest: str) -> None:
        authority.reserve_pair(
            pair_key(manifest), manifest["payload"]["bytes"] + manifest["receipt"]["bytes"],
            manifest["control_allowance_bytes"], {
                "manifest_sha256": digest,
                "folder_name": manifest["folder_name"],
                "drive_root": manifest["drive_root"],
                "payload_object": manifest["payload"]["object"],
                "payload_generation": manifest["payload"]["generation"],
                "receipt_object": manifest["receipt"]["object"],
                "receipt_generation": manifest["receipt"]["generation"],
            })

    @staticmethod
    def request(authority, body_bytes: int = 1, kind: str = "media_chunk"):
        return authority.start_request(
            body_bytes, 20, kind, method="PUT", target_category=kind,
            target_sha256="a" * 64, operation_key="b" * 64)

    @staticmethod
    def finish_index(authority, manifest: dict, digest: str) -> str:
        key = pair_key(manifest)
        mapping_digest = hashlib.sha256((key + digest).encode()).hexdigest()
        authority.mark_pair_verified(key, mapping_digest)
        authority.close_pair(key)
        authority.reserve_index_create(key, mapping_digest, INDEX_PREFIX + key + ".json", 64)
        authority.mark_index_committed(key, mapping_digest)
        return mapping_digest

    def test_two_reviewed_pairs_share_one_ledger_pacing_and_known_charges(self):
        first, first_manifest, first_digest = self.open_run(1)
        self.assertFalse(first.snapshot()["history_proven"])
        self.reserve(first, first_manifest, first_digest)
        self.assertEqual(first.snapshot()["known_local_rolling_bytes"],
                         self.LEGACY_BYTES + first_manifest["max_charged_bytes"])
        first_attempt, first_slot = self.request(first, 5)
        first.mark_dispatch(first_attempt)
        first.finish_request(first_attempt, 200)
        first_usage = first.snapshot()["known_local_rolling_bytes"]
        self.finish_index(first, first_manifest, first_digest)

        second, second_manifest, second_digest = self.open_run(2)
        self.reserve(second, second_manifest, second_digest)
        second_attempt, second_slot = self.request(second, 7)
        self.assertGreaterEqual(second_slot - first_slot, 0.55 - 1e-6)
        self.assertGreaterEqual(second.snapshot()["known_local_rolling_bytes"], first_usage)
        with self.assertRaises(AuthorityBlocked):
            second.mark_dispatch(second_attempt)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        second.mark_dispatch(second_attempt)
        second.finish_request(second_attempt, 200)

    def test_review_packet_and_manifest_mutation_fail_closed(self):
        manifest = self.manifest(1)
        raw, digest = self.raw_and_digest(manifest)
        for changed in (
            {"reviewed_manifest_sha256": "0" * 64},
            {"writer_id": "other-writer"},
            {"account_identity_verified": False},
            {"all_writers_excluded": False},
            {"history_basis": "reconciled"},
            {"quota_rejection_accepted": False},
            {"observed_at": self.clock() - 301},
            {"exclusion_valid_until": self.clock() + 10},
        ):
            with self.subTest(changed=changed), self.assertRaises(AuthorityBlocked):
                TransferAuthority.open_retained_run(
                    self.db_path, run_id=manifest["release_id"], writer_id=self.writer_id,
                    account_id=self.account_id, manifest_bytes=raw,
                    evidence=self.evidence(manifest, digest, **changed),
                    deadline_at=self.clock() + 1200, clock=self.clock,
                    monotonic_clock=self.monotonic, boot_id="synthetic-boot")
        mutated = {**manifest, "payload": {**manifest["payload"], "bytes": 1}}
        with self.assertRaises(AuthorityBlocked):
            TransferAuthority.open_retained_run(
                self.db_path, run_id=manifest["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=self.raw_and_digest(mutated)[0],
                evidence=self.evidence(manifest, digest),
                deadline_at=self.clock() + 1200, clock=self.clock,
                monotonic_clock=self.monotonic, boot_id="synthetic-boot")

    def test_exact_pair_reservation_and_precharged_retry_budget(self):
        authority, manifest, digest = self.open_run(1)
        with self.assertRaises(AuthorityBlocked):
            self.request(authority)
        with self.assertRaises(AuthorityBlocked):
            authority.reserve_pair(pair_key(manifest), 1, 1, {
                "manifest_sha256": digest,
            })
        self.reserve(authority, manifest, digest)
        full_budget = manifest["max_charged_bytes"]
        attempt, _ = self.request(authority, full_budget)
        authority.finish_request(attempt, None)  # Lost response: keep full charge.
        with self.assertRaises(AuthorityBlocked):
            self.request(authority, 1)
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 1)

    def test_forward_wall_clock_jump_cannot_bypass_monotonic_request_pacing(self):
        self.monotonic.value = 1_000_000_000.0  # Exercise a large monotonic epoch.
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        first, _ = self.request(authority)
        authority.mark_dispatch(first)
        authority.finish_request(first, 200)
        self.clock.advance(10)
        self.monotonic.advance(0.1)
        second, _ = self.request(authority)
        with self.assertRaises(AuthorityBlocked):
            authority.mark_dispatch(second)
        # Use times strictly on either side of 0.55 s. Adding 0.1 + 0.45
        # to a large platform monotonic epoch can round one ULP below 0.55.
        self.monotonic.advance(0.449)
        with self.assertRaises(AuthorityBlocked):
            authority.mark_dispatch(second)
        self.monotonic.advance(0.002)
        authority.mark_dispatch(second)
        authority.finish_request(second, 200)

    def test_precharged_attempt_waits_for_monotonic_floor_without_second_charge(self):
        self.monotonic.value = 1_000_000_000.0
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        first, _ = self.request(authority)
        authority.mark_dispatch(first)
        authority.finish_request(first, 200)
        self.clock.advance(10)  # Wall time alone cannot release the next send.
        self.monotonic.advance(0.1)
        second, _ = self.request(authority)
        self.assertGreater(authority.seconds_until_dispatch(second), 0.449)
        self.monotonic.advance(0.449)
        self.assertGreater(authority.seconds_until_dispatch(second), 0)
        self.monotonic.advance(0.002)
        self.assertEqual(authority.seconds_until_dispatch(second), 0)
        authority.mark_dispatch(second)
        authority.finish_request(second, 200)
        db = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM retained_attempt").fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT SUM(bytes) FROM retained_attempt").fetchone()[0], 2)
        finally:
            db.close()

    def test_precharged_attempt_cannot_wait_past_deadline(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        first, _ = self.request(authority)
        authority.mark_dispatch(first)
        authority.finish_request(first, 200)
        self.clock.advance(1198.8)
        self.monotonic.advance(0.1)
        second, _ = authority.start_request(
            1, 1, "media_chunk", method="PUT", target_category="media_chunk",
            target_sha256="a" * 64, operation_key="b" * 64)
        with self.assertRaisesRegex(AuthorityBlocked, "cannot fit run deadline"):
            authority.seconds_until_dispatch(second)
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 1)

    def test_semantically_corrupt_attempt_does_not_reduce_known_charge(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        attempt, _ = self.request(authority, 5)
        authority.mark_dispatch(attempt)
        authority.finish_request(attempt, 200)
        db = sqlite3.connect(self.db_path)
        try:
            db.execute("UPDATE retained_attempt SET bytes=-1 WHERE id=?", (attempt,))
            db.commit()
        finally:
            db.close()
        with self.assertRaises(AuthorityBlocked):
            authority.snapshot()

    def test_synthetic_legacy_result_validator_rejects_incomplete_or_changed_charge(self):
        release_id = "synthetic-legacy-release"
        base = {
            "release": {
                "id": release_id, "status": "closed", "account_id": self.account_id,
                "admission_mode": "diagnostic_quota_observation", "started_at": 90,
                "deadline_at": 200, "next_slot": 101.1, "cooldown_until": 0,
            },
            "descriptor": {
                "release_id": release_id, "account_id": self.account_id,
                "writer_id": self.writer_id, "manifest_sha256": "a" * 64,
            },
            "attempts": [
                {"id": "attempt-1", "bytes": 5, "at": 100, "slot_at": 100,
                 "dispatched_at": 100, "charge_through": 100.5,
                 "http_status": 200, "method": "PUT", "status": "settled"},
                {"id": "attempt-2", "bytes": 7, "at": 100.5, "slot_at": 100.55,
                 "dispatched_at": 100.55, "charge_through": 101,
                 "http_status": 308, "method": "PUT", "status": "settled"},
            ],
            "charged_bytes": 12, "unresolved_attempts": 0,
            "history_proven": False, "archive_qualified": False,
            "entries": [{
                "status": "diagnostic_verified_unqualified",
                "archive_qualified": False,
                "payload_ref": {
                    "name": DIAGNOSTIC_PAYLOAD_OBJECT,
                    "generation": retained_module.LEGACY_DIAGNOSTIC_PAYLOAD_GENERATION,
                    "size": 268_980_444,
                },
            }],
        }

        def validate(result: dict) -> dict:
            raw = json.dumps(result, sort_keys=True).encode()
            return _validated_legacy_result(
                raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
                expected_release=release_id, expected_attempts=2,
                expected_bytes=12, expected_account=self.account_id)

        parsed = validate(base)
        self.assertEqual(parsed["bytes"], 12)
        self.assertEqual(parsed["charge_through"], 101)
        self.assertEqual(parsed["next_slot"], 101.1)
        for change in ("wrong-total", "unresolved", "duplicate", "history", "identity"):
            damaged = json.loads(json.dumps(base))
            if change == "wrong-total":
                damaged["charged_bytes"] = 11
            elif change == "unresolved":
                damaged["attempts"][1]["status"] = "unresolved"
            elif change == "duplicate":
                damaged["attempts"][1]["id"] = "attempt-1"
            elif change == "history":
                damaged["history_proven"] = True
            else:
                damaged["descriptor"]["release_id"] = "another-release"
            with self.subTest(change=change), self.assertRaises(AuthorityBlocked):
                validate(damaged)
        raw = json.dumps(base).encode()
        with self.assertRaises(AuthorityBlocked):
            _validated_legacy_result(raw, expected_sha256="0" * 64,
                                     expected_release=release_id, expected_attempts=2,
                                     expected_bytes=12, expected_account=self.account_id)
        with self.assertRaises(AuthorityBlocked):
            _validated_legacy_result(raw, expected_sha256=hashlib.sha256(raw).hexdigest(),
                                     expected_release=release_id, expected_attempts=3,
                                     expected_bytes=12, expected_account=self.account_id)

    def test_legacy_import_is_idempotent_and_conflicting_source_is_rejected(self):
        importer = retained_module.RetainedObservationAuthority.import_legacy_result
        importer(self.db_path, result_bytes=self.legacy_raw, writer_id=self.writer_id,
                 account_id=self.account_id, clock=self.clock)
        other = self.synthetic_legacy_result(release_id="another-synthetic-release")
        other_raw = json.dumps(other, sort_keys=True).encode()
        with patch.object(retained_module, "LEGACY_DIAGNOSTIC_SOURCE_SHA256",
                          hashlib.sha256(other_raw).hexdigest()), patch.object(
                              retained_module, "LEGACY_DIAGNOSTIC_RELEASE_ID",
                              "another-synthetic-release"):
            with self.assertRaises(AuthorityBlocked):
                importer(self.db_path, result_bytes=other_raw, writer_id=self.writer_id,
                         account_id=self.account_id, clock=self.clock)
        db = sqlite3.connect(self.db_path)
        try:
            imported = db.execute("SELECT source_sha256, bytes FROM retained_import").fetchall()
        finally:
            db.close()
        self.assertEqual(imported, [(self.legacy_digest, self.LEGACY_BYTES)])

    def test_account_inspection_is_read_only_and_reports_durable_stop_gates(self):
        def process_claim() -> tuple:
            db = sqlite3.connect(self.db_path)
            try:
                return db.execute(
                    "SELECT process_host,process_pid,process_nonce "
                    "FROM retained_account WHERE id=1").fetchone()
            finally:
                db.close()

        inspect = lambda: TransferAuthority.inspect_retained_account(
            self.db_path, writer_id=self.writer_id, account_id=self.account_id,
            clock=self.clock)
        unclaimed = process_claim()
        self.assertEqual(unclaimed, (None, None, None))
        initial = inspect()
        self.assertEqual(process_claim(), unclaimed)
        self.assertTrue(initial["diagnostic_imported"])
        self.assertFalse(initial["history_proven"])
        self.assertEqual(initial["cooldown_until"], 0)
        self.assertGreater(initial["next_slot"], 0)
        self.assertEqual(initial["unresolved_attempts"], 0)
        self.assertEqual(initial["known_local_rolling_bytes"], self.LEGACY_BYTES)
        self.assertEqual(initial["mapping_reserved_total_bytes"], 0)
        self.assertEqual(initial["mapping_partition_remaining_bytes"],
                         initial["mapping_partition_bytes"])
        self.assertIsNone(initial["nonterminal_run_id"])
        with self.assertRaises(AuthorityBlocked):
            TransferAuthority.inspect_retained_account(
                self.db_path, writer_id=self.writer_id,
                account_id="wrong-account", clock=self.clock)
        self.assertEqual(process_claim(), unclaimed)

        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        attempt, slot = self.request(authority, 5)
        claimed = process_claim()
        self.assertNotEqual(claimed, unclaimed)
        admitted = inspect()
        self.assertEqual(process_claim(), claimed)
        self.assertEqual(admitted["unresolved_attempts"], 1)
        self.assertGreaterEqual(admitted["next_slot"], slot + 0.55 - 1e-6)
        self.assertEqual(admitted["nonterminal_run_id"], manifest["release_id"])
        self.assertEqual(admitted["nonterminal_status"], "active")

        authority.mark_dispatch(attempt)
        authority.finish_request(attempt, 200)
        key = pair_key(manifest)
        mapping_digest = hashlib.sha256(b"inspection-mapping").hexdigest()
        authority.mark_pair_verified(key, mapping_digest)
        authority.close_pair(key)
        authority.reserve_index_create(key, mapping_digest,
                                       manifest["index_object"], 123)
        pending = inspect()
        self.assertEqual(process_claim(), claimed)
        self.assertEqual(pending["unresolved_attempts"], 0)
        self.assertEqual(pending["mapping_reserved_total_bytes"], 123)
        self.assertEqual(pending["mapping_partition_remaining_bytes"],
                         pending["mapping_partition_bytes"] - 123)
        self.assertEqual(pending["nonterminal_status"], "index_pending")

    def test_index_pending_blocks_next_pair_and_all_drive_requests(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        key = pair_key(manifest)
        mapping_digest = hashlib.sha256(b"synthetic-mapping").hexdigest()
        authority.mark_pair_verified(key, mapping_digest)
        with self.assertRaises(AuthorityBlocked):
            self.request(authority)
        authority.close_pair(key)
        self.assertEqual(authority.pair_status(key, digest), "index_pending")
        with self.assertRaises(AuthorityBlocked):
            self.open_run(2)
        with self.assertRaises(AuthorityBlocked):
            authority.mark_index_committed(key, mapping_digest)
        object_name = INDEX_PREFIX + key + ".json"
        with self.assertRaises(AuthorityBlocked):
            authority.reserve_index_create(key, mapping_digest, object_name + ".wrong", 64)
        reservation_id = authority.reserve_index_create(key, mapping_digest, object_name, 64)
        self.assertEqual(authority.reserve_index_create(key, mapping_digest, object_name, 64),
                         reservation_id)
        with self.assertRaises(AuthorityBlocked):
            authority.reserve_index_create(key, mapping_digest, object_name + ".wrong", 64)
        with self.assertRaises(AuthorityBlocked):
            authority.mark_index_committed(key, "0" * 64)
        authority.mark_index_committed(key, mapping_digest)
        self.assertEqual(authority.pair_status(key, digest), "closed")
        self.open_run(2)

    def test_pending_run_blocks_recovery_while_prior_process_is_alive(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        key = pair_key(manifest)
        mapping_digest = hashlib.sha256(b"mapping-ready").hexdigest()
        authority.mark_pair_verified(key, mapping_digest)
        authority.close_pair(key)
        authority.close()  # Closing a handle does not prove its process died.
        with self.assertRaises(AuthorityBlocked):
            self.open_run(1)
        with self.assertRaises(AuthorityBlocked):
            self.open_run(1, manifest=manifest, recovery_only=True)

    def test_pending_run_recovery_after_owner_process_exits_never_sends(self):
        manifest = self.manifest(1)
        raw, digest = self.raw_and_digest(manifest)
        evidence = self.evidence(manifest, digest)
        key = pair_key(manifest)
        mapping_digest = hashlib.sha256(b"mapping-ready").hexdigest()
        manifest_file = Path(self.temp.name) / "manifest.json"
        evidence_file = Path(self.temp.name) / "evidence.json"
        manifest_file.write_bytes(raw)
        evidence_file.write_text(json.dumps(evidence))
        script = self.child_patch_code() + (
            "import json,sys; from pathlib import Path; "
            "from drive_transfer_authority import TransferAuthority; "
            "a=TransferAuthority.open_retained_run(Path(sys.argv[1]),run_id=sys.argv[2],"
            "writer_id=sys.argv[3],account_id=sys.argv[4],manifest_bytes=Path(sys.argv[5]).read_bytes(),"
            "evidence=json.loads(Path(sys.argv[6]).read_text()),deadline_at=float(sys.argv[7]),"
            "clock=lambda:float(sys.argv[8]),monotonic_clock=lambda:float(sys.argv[9]),"
            "boot_id='synthetic-boot'); "
            "a.reserve_pair(sys.argv[10],int(sys.argv[11]),int(sys.argv[12]),json.loads(sys.argv[13])); "
            "a.mark_pair_verified(sys.argv[10],sys.argv[14]); a.close_pair(sys.argv[10]); "
            "a.reserve_index_create(sys.argv[10],sys.argv[14],sys.argv[15],64); a.close()"
        )
        metadata = {
            "manifest_sha256": digest,
            "folder_name": manifest["folder_name"],
            "drive_root": manifest["drive_root"],
            "payload_object": manifest["payload"]["object"],
            "payload_generation": manifest["payload"]["generation"],
            "receipt_object": manifest["receipt"]["object"],
            "receipt_generation": manifest["receipt"]["generation"],
        }
        env = self.child_env()
        child = subprocess.run(
            [sys.executable, "-c", script, str(self.db_path), manifest["release_id"],
             self.writer_id, self.account_id, str(manifest_file), str(evidence_file),
             str(self.clock() + 1200), str(self.clock()), str(self.monotonic()),
             key, str(manifest["payload"]["bytes"] + manifest["receipt"]["bytes"]),
             str(manifest["control_allowance_bytes"]), json.dumps(metadata), mapping_digest,
             INDEX_PREFIX + key + ".json"],
            env=env, capture_output=True, text=True, check=False)
        self.assertEqual(child.returncode, 0, child.stderr)
        with self.assertRaises(AuthorityBlocked):
            self.open_run(2)
        recovered, _, _ = self.open_run(1, manifest=manifest, recovery_only=True)
        self.assertEqual(recovered.pair_status(key, digest), "index_pending")
        with self.assertRaises(AuthorityBlocked):
            self.request(recovered)
        self.assertTrue(recovered.reserve_index_create(
            key, mapping_digest, INDEX_PREFIX + key + ".json", 64))
        recovered.mark_index_committed(key, mapping_digest)
        self.open_run(2)

    def test_active_crash_uncertain_send_and_quota_cooldown_block_next_run(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        attempt, _ = self.request(authority, 5)
        authority.mark_dispatch(attempt)
        authority.finish_request(attempt, 429)
        self.assertGreater(authority.snapshot()["cooldown_until"], self.clock())
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 1)
        authority.close()
        with self.assertRaises(AuthorityBlocked):
            self.open_run(2)
        with self.assertRaises(AuthorityBlocked):
            self.open_run(1, manifest=manifest, recovery_only=True)

    def test_pair_identity_replay_and_writer_account_mismatch(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        self.finish_index(authority, manifest, digest)
        with self.assertRaises(AuthorityBlocked):
            self.open_run(1, manifest=manifest)
        replay = self.manifest(1, release_id=f"{3:032x}")
        with self.assertRaises(AuthorityBlocked):
            self.open_run(3, manifest=replay)
        raw, digest2 = self.raw_and_digest(self.manifest(2))
        with self.assertRaises(AuthorityBlocked):
            TransferAuthority.open_retained_run(
                self.db_path, run_id=f"{2:032x}", writer_id="other-writer",
                account_id=self.account_id, manifest_bytes=raw,
                evidence=self.evidence(self.manifest(2), digest2),
                deadline_at=self.clock() + 1200, clock=self.clock,
                monotonic_clock=self.monotonic, boot_id="synthetic-boot")

    def test_new_receipt_generation_cannot_recopy_existing_payload_generation(self):
        authority, first, digest = self.open_run(1)
        self.reserve(authority, first, digest)
        self.finish_index(authority, first, digest)

        listed = TransferAuthority.list_retained_pairs(
            self.db_path, writer_id=self.writer_id, account_id=self.account_id)
        self.assertEqual(len(listed), 2)
        self.assertEqual(listed[0]["status"], "diagnostic_verified_unqualified")
        self.assertEqual(listed[0]["payload_object"], DIAGNOSTIC_PAYLOAD_OBJECT)
        self.assertEqual(listed[1]["status"], "closed")
        self.assertEqual(listed[1]["bucket"], first["bucket"])
        self.assertEqual(listed[1]["payload_object"], first["payload"]["object"])
        self.assertEqual(listed[1]["payload_generation"], first["payload"]["generation"])

        second = self.manifest(2)
        second["payload"] = dict(first["payload"])
        second["receipt"]["object"] = (
            first["payload"]["object"] +
            f".g{first['payload']['generation']}.receipt.json")
        second["receipt"]["generation"] = first["receipt"]["generation"] + 1
        second["max_charged_bytes"] = (
            second["payload"]["bytes"] + second["receipt"]["bytes"] +
            second["control_allowance_bytes"])
        second["folder_name"] = "catbird-overflow-" + pair_key(second)[:32]
        second["index_object"] = INDEX_PREFIX + pair_key(second) + ".json"
        validate_overflow_manifest(second, release_id=second["release_id"],
                                   account_id=self.account_id)
        self.assertEqual(pair_key(second), pair_key(first))
        with self.assertRaises(AuthorityBlocked):
            self.open_run(2, manifest=second)  # Fresh evidence is derived from these bytes.

    def test_imported_diagnostic_payload_is_skipped_by_retained_admission(self):
        manifest = self.manifest(1)
        generation = retained_module.LEGACY_DIAGNOSTIC_PAYLOAD_GENERATION
        manifest["payload"] = {
            "object": DIAGNOSTIC_PAYLOAD_OBJECT,
            "generation": generation,
            "bytes": 268_980_444,
            "sha256": DIAGNOSTIC_PAYLOAD_OBJECT.rsplit("/", 1)[-1][:-4],
        }
        manifest["receipt"]["object"] = (
            DIAGNOSTIC_PAYLOAD_OBJECT + f".g{generation}.receipt.json")
        manifest["max_charged_bytes"] = (
            manifest["payload"]["bytes"] + manifest["receipt"]["bytes"] +
            manifest["control_allowance_bytes"])
        manifest["folder_name"] = "catbird-overflow-" + pair_key(manifest)[:32]
        manifest["index_object"] = INDEX_PREFIX + pair_key(manifest) + ".json"

        # The public manifest gate rejects it first. Patch only that duplicate
        # gate to exercise the ledger's independent imported-source exclusion.
        with patch.object(overflow_module, "COMPLETED_DIAGNOSTIC_PAYLOAD",
                          {"object": "other", "generation": -1}):
            validate_overflow_manifest(manifest, release_id=manifest["release_id"],
                                       account_id=self.account_id)
            with self.assertRaisesRegex(AuthorityBlocked,
                                        "Verified diagnostic payload must not be recopied"):
                self.open_run(1, manifest=manifest)

    def test_boot_change_or_wall_clock_regression_blocks_new_upload_run(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        with self.assertRaises(AuthorityBlocked):
            self.open_run(2, boot_id="different-boot")
        self.finish_index(authority, manifest, digest)
        self.clock.advance(-1)
        with self.assertRaises(AuthorityBlocked):
            self.open_run(2)

    def test_missing_corrupt_ledger_and_foreign_path_never_mean_empty_state(self):
        absent = Path(self.temp.name) / "missing.sqlite"
        manifest = self.manifest(1)
        raw, digest = self.raw_and_digest(manifest)
        with self.assertRaises(AuthorityBlocked):
            TransferAuthority.open_retained_run(
                absent, run_id=manifest["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw,
                evidence=self.evidence(manifest, digest), deadline_at=self.clock() + 1200,
                clock=self.clock, monotonic_clock=self.monotonic, boot_id="synthetic-boot")
        corrupt = Path(self.temp.name) / "corrupt.sqlite"
        corrupt.write_bytes(b"not sqlite")
        with self.assertRaises(AuthorityBlocked):
            TransferAuthority.open_retained_run(
                corrupt, run_id=manifest["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw,
                evidence=self.evidence(manifest, digest), deadline_at=self.clock() + 1200,
                clock=self.clock, monotonic_clock=self.monotonic, boot_id="synthetic-boot")

    def test_closed_handoff_to_second_process_uses_same_database(self):
        first, manifest, digest = self.open_run(1)
        self.reserve(first, manifest, digest)
        self.finish_index(first, manifest, digest)
        next_manifest = self.manifest(2)
        raw, next_digest = self.raw_and_digest(next_manifest)
        manifest_file = Path(self.temp.name) / "next-manifest.json"
        evidence_file = Path(self.temp.name) / "next-evidence.json"
        manifest_file.write_bytes(raw)
        evidence_file.write_text(json.dumps(self.evidence(next_manifest, next_digest)))
        child_cwd = Path(self.temp.name) / "other-work-directory"
        child_cwd.mkdir()
        script = self.child_patch_code() + (
            "import json,sys; from pathlib import Path; "
            "from drive_transfer_authority import TransferAuthority; "
            "a=TransferAuthority.open_retained_run(Path(sys.argv[1]),run_id=sys.argv[2],"
            "writer_id=sys.argv[3],account_id=sys.argv[4],manifest_bytes=Path(sys.argv[5]).read_bytes(),"
            "evidence=json.loads(Path(sys.argv[6]).read_text()),deadline_at=float(sys.argv[7]),"
            "clock=lambda:float(sys.argv[8]),monotonic_clock=lambda:float(sys.argv[9]),"
            "boot_id='synthetic-boot'); a.assert_active(); a.close()"
        )
        env = self.child_env()
        child = subprocess.run(
            [sys.executable, "-c", script, str(self.db_path), next_manifest["release_id"],
             self.writer_id, self.account_id, str(manifest_file), str(evidence_file),
             str(self.clock() + 601), str(self.clock() + 1), str(self.monotonic() + 1)],
            cwd=child_cwd, env=env, capture_output=True, text=True, check=False)
        self.assertEqual(child.returncode, 0, child.stderr)
        with self.assertRaises(AuthorityBlocked):
            self.open_run(3)

    def test_status_probe_reconciles_transient_5xx_media_and_intermediate_probe(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        payload_size = manifest["payload"]["bytes"]
        failed_attempt, _ = self.request(authority, payload_size, "media_chunk")
        authority.mark_dispatch(failed_attempt)
        authority.finish_request(failed_attempt, 503)
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 1)

        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        probe_502, _ = self.request(authority, 0, "resumable_status")
        authority.mark_dispatch(probe_502)
        authority.finish_request(probe_502, 502)
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 2)

        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        probe_308, _ = self.request(authority, 0, "resumable_status")
        authority.mark_dispatch(probe_308)
        authority.finish_request(probe_308, 308)

        proof_sha = authority.reconcile_attempt_by_status_probe(
            failed_attempt_id=failed_attempt,
            probe_attempt_id=probe_308,
            acknowledged_bytes=5,
            committed_bytes=5,
        )
        self.assertEqual(len(proof_sha), 64)
        snap = authority.snapshot()
        self.assertEqual(snap["unresolved_attempts"], 0)
        self.assertEqual(snap["forensic_raw_unresolved_attempts"], 2)
        inspected = TransferAuthority.inspect_retained_account(
            self.db_path, writer_id=self.writer_id, account_id=self.account_id,
            clock=self.clock)
        self.assertEqual(inspected["unresolved_attempts"], 0)
        self.assertEqual(inspected["forensic_raw_unresolved_attempts"], 2)

        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        retry_attempt, _ = self.request(authority, payload_size - 5, "media_chunk")
        authority.mark_dispatch(retry_attempt)
        authority.finish_request(retry_attempt, 200)
        self.finish_index(authority, manifest, digest)
        self.open_run(2)

    def test_multiple_status_probe_reconciliations_in_single_run_preserve_exact_budget(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        payload_size = manifest["payload"]["bytes"]
        receipt_size = manifest["receipt"]["bytes"]

        # 1st transient 5xx on payload chunk (partial commit of 4 bytes)
        fail_1, _ = self.request(authority, payload_size, "media_chunk")
        authority.mark_dispatch(fail_1)
        authority.finish_request(fail_1, 503)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        probe_1, _ = self.request(authority, 0, "resumable_status")
        authority.mark_dispatch(probe_1)
        authority.finish_request(probe_1, 308)
        authority.reconcile_attempt_by_status_probe(
            failed_attempt_id=fail_1,
            probe_attempt_id=probe_1,
            acknowledged_bytes=4,
            committed_bytes=4,
        )
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 0)

        # Retry remaining payload bytes (payload_size - 4), which ALSO hits transient 502 (0 additional bytes committed)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        fail_2, _ = self.request(authority, payload_size - 4, "media_chunk")
        authority.mark_dispatch(fail_2)
        authority.finish_request(fail_2, 502)
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 1)

        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        probe_2, _ = self.request(authority, 0, "resumable_status")
        authority.mark_dispatch(probe_2)
        authority.finish_request(probe_2, 308)
        authority.reconcile_attempt_by_status_probe(
            failed_attempt_id=fail_2,
            probe_attempt_id=probe_2,
            acknowledged_bytes=4,
            committed_bytes=0,
        )
        snap = authority.snapshot()
        self.assertEqual(snap["unresolved_attempts"], 0)
        self.assertEqual(snap["forensic_raw_unresolved_attempts"], 2)

        # Final payload retry + receipt upload succeed within exact pair reservation
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        payload_done, _ = self.request(authority, payload_size - 4, "media_chunk")
        authority.mark_dispatch(payload_done)
        authority.finish_request(payload_done, 200)

        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        receipt_done, _ = self.request(authority, receipt_size, "media_chunk")
        authority.mark_dispatch(receipt_done)
        authority.finish_request(receipt_done, 200)

        self.finish_index(authority, manifest, digest)
        self.open_run(2)

    def test_zero_byte_resumable_status_404_and_410_settle_in_finish_request(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)
        for code in (404, 410):
            probe, _ = self.request(authority, 0, "resumable_status")
            authority.mark_dispatch(probe)
            authority.finish_request(probe, code)
            self.assertEqual(authority.snapshot()["unresolved_attempts"], 0)
            self.clock.advance(0.55)
            self.monotonic.advance(0.55)
        media, _ = self.request(authority, manifest["payload"]["bytes"], "media_chunk")
        authority.mark_dispatch(media)
        authority.finish_request(media, 410)
        self.assertEqual(authority.snapshot()["unresolved_attempts"], 1)

    def test_controlled_reboot_monotonic_reset_does_not_collide_with_prior_boot_uptime(self):
        self.monotonic.value = 2500.0
        first, manifest_1, digest_1 = self.open_run(1, boot_id="boot-1")
        self.reserve(first, manifest_1, digest_1)
        a1, _ = self.request(first, manifest_1["payload"]["bytes"], "media_chunk")
        first.mark_dispatch(a1)
        first.finish_request(a1, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        a2, _ = self.request(first, manifest_1["receipt"]["bytes"], "media_chunk")
        first.mark_dispatch(a2)
        first.finish_request(a2, 200)
        self.finish_index(first, manifest_1, digest_1)

        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "UPDATE retained_account SET boot_id='boot-2', process_host=NULL, "
                "process_pid=NULL, process_nonce=NULL, last_dispatch_wall=NULL, "
                "last_dispatch_monotonic=NULL WHERE id=1"
            )
        self.clock.advance(60.0)
        self.monotonic.value = 2500.0  # Exact same uptime as boot-1's first dispatch
        second, manifest_2, digest_2 = self.open_run(2, boot_id="boot-2")
        self.reserve(second, manifest_2, digest_2)
        b1, _ = self.request(second, manifest_2["payload"]["bytes"], "media_chunk")
        second.mark_dispatch(b1)
        second.finish_request(b1, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        b2, _ = self.request(second, manifest_2["receipt"]["bytes"], "media_chunk")
        second.mark_dispatch(b2)
        second.finish_request(b2, 200)
        self.finish_index(second, manifest_2, digest_2)
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "UPDATE retained_account SET boot_id='boot-3', process_host=NULL, "
                "process_pid=NULL, process_nonce=NULL, last_dispatch_wall=NULL, "
                "last_dispatch_monotonic=NULL WHERE id=1"
            )
        self.clock.advance(60.0)
        # Uptime on boot-3 is 0.15s higher than boot-2's final dispatch (2500.55),
        # which previously triggered a false-positive monotonic pacing error.
        self.monotonic.value = 2500.70
        third, manifest_3, digest_3 = self.open_run(3, boot_id="boot-3")
        self.reserve(third, manifest_3, digest_3)
        c1, _ = self.request(third, manifest_3["payload"]["bytes"], "media_chunk")
        third.mark_dispatch(c1)
        third.finish_request(c1, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        c2, _ = self.request(third, manifest_3["receipt"]["bytes"], "media_chunk")
        third.mark_dispatch(c2)
        third.finish_request(c2, 200)
        self.finish_index(third, manifest_3, digest_3)

        # Verify within-run monotonic pacing violation is still caught by _validate_all.
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "UPDATE retained_attempt SET dispatch_monotonic=? WHERE id=?",
                (2500.80, c2),
            )
        with self.assertRaisesRegex(AuthorityBlocked, "violate monotonic pacing"):
            self.open_run(4, boot_id="boot-3")

    def test_rolling_cap_is_strictly_below_750_gb_decimal(self):
        self.assertEqual(retained_module.GLOBAL_ROLLING_CAP_BYTES, 700_000_000_000)
        self.assertLess(retained_module.GLOBAL_ROLLING_CAP_BYTES, 750_000_000_000)

    def test_concurrent_runs_share_single_writer_ledger_pacing_and_charges(self):
        run1, m1, d1 = self.open_run(1, max_concurrent_runs=3)
        run2, m2, d2 = self.open_run(2, max_concurrent_runs=3)
        run3, m3, d3 = self.open_run(3, max_concurrent_runs=3)
        with self.assertRaisesRegex(AuthorityBlocked, "not index committed"):
            self.open_run(4, max_concurrent_runs=3)

        self.reserve(run1, m1, d1)
        self.reserve(run2, m2, d2)
        self.reserve(run3, m3, d3)
        expected_rolling = (
            self.LEGACY_BYTES +
            m1["max_charged_bytes"] +
            m2["max_charged_bytes"] +
            m3["max_charged_bytes"]
        )
        self.assertEqual(run3.snapshot()["known_local_rolling_bytes"], expected_rolling)

        a1, slot1 = self.request(run1, m1["payload"]["bytes"], "media_chunk")
        a2, slot2 = self.request(run2, m2["payload"]["bytes"], "media_chunk")
        a3, slot3 = self.request(run3, m3["payload"]["bytes"], "media_chunk")
        self.assertGreaterEqual(slot2 - slot1, 0.55 - 1e-6)
        self.assertGreaterEqual(slot3 - slot2, 0.55 - 1e-6)

        run1.mark_dispatch(a1)
        run1.finish_request(a1, 200)
        with self.assertRaises(AuthorityBlocked):
            run2.mark_dispatch(a2)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        run2.mark_dispatch(a2)
        run2.finish_request(a2, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        run3.mark_dispatch(a3)
        run3.finish_request(a3, 200)

        for authority, manifest, digest in ((run1, m1, d1), (run2, m2, d2), (run3, m3, d3)):
            self.clock.advance(0.55)
            self.monotonic.advance(0.55)
            r_att, _ = self.request(authority, manifest["receipt"]["bytes"], "media_chunk")
            authority.mark_dispatch(r_att)
            authority.finish_request(r_att, 200)
            self.finish_index(authority, manifest, digest)

        inspected = TransferAuthority.inspect_retained_account(
            self.db_path, writer_id=self.writer_id, account_id=self.account_id,
            clock=self.clock)
        self.assertIsNone(inspected["nonterminal_run_id"])
        self.assertEqual(inspected["active_run_count"], 0)
        self.assertEqual(inspected["unresolved_attempts"], 0)

    def test_non_settled_result_does_not_block_sibling_pairs_or_own_5xx_recovery(self):
        run1, m1, d1 = self.open_run(1, max_concurrent_runs=3)
        self.reserve(run1, m1, d1)
        session1 = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=s1"
        session1_sha = hashlib.sha256(session1.encode("utf-8")).hexdigest()
        fail_att, _ = run1.start_request(
            m1["payload"]["bytes"], 20, "media_chunk", method="PUT",
            target_category="media_chunk", target_sha256=session1_sha,
            operation_key="b" * 64,
        )
        run1.mark_dispatch(fail_att)
        run1.finish_request(fail_att, 503)

        # Sibling run2 is NOT blocked by run1's non-settled 503 attempt
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        run2, m2, d2 = self.open_run(2, max_concurrent_runs=3)
        self.reserve(run2, m2, d2)
        p2, _ = self.request(run2, m2["payload"]["bytes"], "media_chunk")
        run2.mark_dispatch(p2)
        run2.finish_request(p2, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        r2, _ = self.request(run2, m2["receipt"]["bytes"], "media_chunk")
        run2.mark_dispatch(r2)
        run2.finish_request(r2, 200)
        self.finish_index(run2, m2, d2)

        # And run1 can still perform its own 5xx status probe recovery and close
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        probe_att, _ = run1.start_request(
            0, 20, "resumable_status", method="PUT",
            target_category="resumable_status", target_sha256=session1_sha,
            operation_key="c" * 64,
        )
        run1.mark_dispatch(probe_att)
        run1.finish_request(probe_att, 200)
        run1.reconcile_attempt_by_status_probe(
            failed_attempt_id=fail_att, probe_attempt_id=probe_att,
            acknowledged_bytes=m1["payload"]["bytes"],
            committed_bytes=m1["payload"]["bytes"],
        )
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        r1, _ = self.request(run1, m1["receipt"]["bytes"], "media_chunk")
        run1.mark_dispatch(r1)
        run1.finish_request(r1, 200)
        self.finish_index(run1, m1, d1)

        inspected = TransferAuthority.inspect_retained_account(
            self.db_path, writer_id=self.writer_id, account_id=self.account_id,
            clock=self.clock)
        self.assertIsNone(inspected["nonterminal_run_id"])
        self.assertEqual(inspected["unresolved_attempts"], 0)

    def test_crash_open_run_adoption_and_session_probe_reconciliation(self):
        dead_pid = 999999
        m1 = self.manifest(1)
        raw1, d1 = self.raw_and_digest(m1)
        ev1 = self.evidence(m1, d1)
        self.monotonic.value = 82247.5
        with patch.object(retained_module.os, "getpid", return_value=dead_pid):
            run1 = TransferAuthority.open_retained_run(
                self.db_path, run_id=m1["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw1, evidence=ev1,
                deadline_at=self.clock() + 20 * 60, clock=self.clock,
                monotonic_clock=self.monotonic, boot_id="synthetic-boot",
            )
            self.addCleanup(run1.close)
            self.reserve(run1, m1, d1)
            session1 = (
                "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
                "&fields=id%2Cname%2Csize%2Cmd5Checksum%2CmimeType%2Cparents%2Ctrashed"
                "&upload_id=" + ("A" * 160) + "&session_crd=" + ("B" * 140)
            )
            self.assertGreater(len(session1), 360)
            session1_sha = hashlib.sha256(session1.encode("utf-8")).hexdigest()
            crash_att, _ = run1.start_request(
                m1["payload"]["bytes"], 20, "media_chunk", method="PUT",
                target_category="media_chunk", target_sha256=session1_sha,
                operation_key="d" * 64,
            )
            run1.mark_dispatch(crash_att)
            # Also pre-charge an undispatched chunk right before reboot/crash
            self.clock.advance(0.55)
            self.monotonic.advance(0.55)
            undispatched_att, _ = run1.start_request(
                10, 20, "media_chunk", method="PUT",
                target_category="media_chunk", target_sha256=session1_sha,
                operation_key="a" * 64,
            )
            # Simulate crash/reboot: close sqlite connection without finish_request
            run1.close()

        with patch.object(retained_module.os, "kill", side_effect=ProcessLookupError):
            open_info = TransferAuthority.inspect_open_retained_run(
                self.db_path, writer_id=self.writer_id,
                account_id=self.account_id, run_id=m1["release_id"],
            )
            self.assertEqual(open_info["status"], "active")
            self.assertTrue(open_info["owner_host_matches"])
            self.assertFalse(open_info["owner_alive"])
            self.assertEqual(open_info["recoverable_unresolved_attempts"], 2)

            self.clock.advance(60.0)
            # Kernel monotonic clock resets to a lower value on the new boot
            self.monotonic.value = 12.0
            ev_resumed = self.evidence(m1, d1, observed_at=self.clock())
            resumed = TransferAuthority.open_retained_run(
                self.db_path, run_id=m1["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw1, evidence=ev_resumed,
                deadline_at=self.clock() + 20 * 60, clock=self.clock,
                monotonic_clock=self.monotonic, boot_id="synthetic-boot-after-reboot",
                resume_open=True,
            )
            self.addCleanup(resumed.close)
        pending = resumed.unresolved_session_media_attempts(session1)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["attempt_id"], crash_att)

        probe_att, _ = resumed.start_request(
            0, 20, "resumable_status", method="PUT",
            target_category="resumable_status", target_sha256=session1_sha,
            operation_key="e" * 64,
        )
        self.assertEqual(resumed.seconds_until_dispatch(probe_att), 0.0)
        resumed.mark_dispatch(probe_att)
        resumed.finish_request(probe_att, 308)
        resumed.reconcile_attempt_by_status_probe(
            failed_attempt_id=crash_att, probe_attempt_id=probe_att,
            acknowledged_bytes=0, committed_bytes=0,
        )
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        retry_att, _ = resumed.start_request(
            m1["payload"]["bytes"], 20, "media_chunk", method="PUT",
            target_category="media_chunk", target_sha256=session1_sha,
            operation_key="f" * 64,
        )
        resumed.mark_dispatch(retry_att)
        resumed.finish_request(retry_att, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        r_att, _ = self.request(resumed, m1["receipt"]["bytes"], "media_chunk")
        resumed.mark_dispatch(r_att)
        resumed.finish_request(r_att, 200)
        self.finish_index(resumed, m1, d1)

        inspected = TransferAuthority.inspect_retained_account(
            self.db_path, writer_id=self.writer_id, account_id=self.account_id,
            clock=self.clock)
        self.assertIsNone(inspected["nonterminal_run_id"])
        self.assertEqual(inspected["unresolved_attempts"], 0)

    def test_clean_reboot_opens_next_pair_while_open_pair_requires_adoption(self):
        self.monotonic.value = 50000.0
        run1, m1, d1 = self.open_run(1, boot_id="boot-1")
        self.reserve(run1, m1, d1)
        p1, _ = self.request(run1, m1["payload"]["bytes"], "media_chunk")
        run1.mark_dispatch(p1)
        run1.finish_request(p1, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        r1, _ = self.request(run1, m1["receipt"]["bytes"], "media_chunk")
        run1.mark_dispatch(r1)
        run1.finish_request(r1, 200)
        self.finish_index(run1, m1, d1)

        with sqlite3.connect(self.db_path) as db:
            acct_before = db.execute(
                "SELECT boot_id, last_dispatch_wall, last_dispatch_monotonic "
                "FROM retained_account WHERE id=1"
            ).fetchone()
        self.assertEqual(acct_before[0], "boot-1")
        self.assertIsNotNone(acct_before[1])
        self.assertIsNotNone(acct_before[2])

        # Clean reboot with zero open pairs and zero unresolved attempts:
        # monotonic clock resets lower than prior boot, next pair opens cleanly.
        self.clock.advance(30.0)
        self.monotonic.value = 5.0
        run2, m2, d2 = self.open_run(2, boot_id="boot-2")
        with sqlite3.connect(self.db_path) as db:
            acct_after_clean_boot = db.execute(
                "SELECT boot_id, last_dispatch_wall, last_dispatch_monotonic "
                "FROM retained_account WHERE id=1"
            ).fetchone()
        self.assertEqual(acct_after_clean_boot[0], "boot-2")
        self.assertIsNone(acct_after_clean_boot[1])
        self.assertIsNone(acct_after_clean_boot[2])

        self.reserve(run2, m2, d2)
        p2, _ = self.request(run2, m2["payload"]["bytes"], "media_chunk")
        self.assertEqual(run2.seconds_until_dispatch(p2), 0.0)
        run2.mark_dispatch(p2)
        run2.finish_request(p2, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        r2, _ = self.request(run2, m2["receipt"]["bytes"], "media_chunk")
        run2.mark_dispatch(r2)
        run2.finish_request(r2, 200)
        self.finish_index(run2, m2, d2)

        # Reboot while pair 3 is open: normal open_run for pair 3 or pair 4 is refused
        # and adoption (resume_open=True) is still required.
        dead_pid = 999998
        m3 = self.manifest(3)
        raw3, d3 = self.raw_and_digest(m3)
        ev3 = self.evidence(m3, d3)
        with patch.object(retained_module.os, "getpid", return_value=dead_pid):
            run3 = TransferAuthority.open_retained_run(
                self.db_path, run_id=m3["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw3, evidence=ev3,
                deadline_at=self.clock() + 20 * 60, clock=self.clock,
                monotonic_clock=self.monotonic, boot_id="boot-2",
            )
            self.addCleanup(run3.close)
            self.reserve(run3, m3, d3)
            run3.close()

        self.clock.advance(30.0)
        self.monotonic.value = 2.0
        with self.assertRaisesRegex(AuthorityBlocked, "boot changed; reconciliation required"):
            self.open_run(4, boot_id="boot-3")
        with self.assertRaisesRegex(AuthorityBlocked, "boot changed; reconciliation required"):
            self.open_run(3, manifest=m3, boot_id="boot-3")

        with patch.object(retained_module.os, "kill", side_effect=ProcessLookupError):
            ev3_resumed = self.evidence(m3, d3, observed_at=self.clock())
            adopted3 = TransferAuthority.open_retained_run(
                self.db_path, run_id=m3["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw3, evidence=ev3_resumed,
                deadline_at=self.clock() + 20 * 60, clock=self.clock,
                monotonic_clock=self.monotonic, boot_id="boot-3",
                resume_open=True,
            )
            self.addCleanup(adopted3.close)
        p3, _ = self.request(adopted3, m3["payload"]["bytes"], "media_chunk")
        self.assertEqual(adopted3.seconds_until_dispatch(p3), 0.0)
        adopted3.mark_dispatch(p3)
        adopted3.finish_request(p3, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        r3, _ = self.request(adopted3, m3["receipt"]["bytes"], "media_chunk")
        adopted3.mark_dispatch(r3)
        adopted3.finish_request(r3, 200)
        self.finish_index(adopted3, m3, d3)

        # Even if all runs are marked closed, any unresolved attempt still refuses
        # a boot change on normal open_run.
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT INTO retained_attempt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("unresolved-on-closed", m3["release_id"], self.clock(),
                 self.clock(), self.clock() + 20, 20.0, None, None, 0,
                 "resumable_status", "PUT", "resumable_status", "a" * 64,
                 "b" * 64, "unresolved", 500),
            )
        self.clock.advance(30.0)
        self.monotonic.value = 1.0
        with self.assertRaisesRegex(AuthorityBlocked, "boot changed; reconciliation required"):
            self.open_run(4, boot_id="boot-4")

    def test_validate_all_hoisted_checks_preserve_all_failure_modes(self):
        # Build a multi-run ledger with closed runs, an adopted run, and a reconciled attempt.
        run1, m1, d1 = self.open_run(1, boot_id="boot-1")
        self.reserve(run1, m1, d1)
        p1, _ = self.request(run1, m1["payload"]["bytes"], "media_chunk")
        run1.mark_dispatch(p1)
        run1.finish_request(p1, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        r1, _ = self.request(run1, m1["receipt"]["bytes"], "media_chunk")
        run1.mark_dispatch(r1)
        run1.finish_request(r1, 200)
        self.finish_index(run1, m1, d1)

        dead_pid = 999997
        m2 = self.manifest(2)
        raw2, d2 = self.raw_and_digest(m2)
        ev2 = self.evidence(m2, d2)
        with patch.object(retained_module.os, "getpid", return_value=dead_pid):
            run2 = TransferAuthority.open_retained_run(
                self.db_path, run_id=m2["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw2, evidence=ev2,
                deadline_at=self.clock() + 20 * 60, clock=self.clock,
                monotonic_clock=self.monotonic, boot_id="boot-1",
            )
            self.addCleanup(run2.close)
            self.reserve(run2, m2, d2)
            undispatched_p2, _ = self.request(run2, m2["payload"]["bytes"], "media_chunk")
            run2.close()

        self.clock.advance(10.0)
        self.monotonic.advance(10.0)
        with patch.object(retained_module.os, "kill", side_effect=ProcessLookupError):
            ev2_resumed = self.evidence(m2, d2, observed_at=self.clock())
            adopted2 = TransferAuthority.open_retained_run(
                self.db_path, run_id=m2["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw2, evidence=ev2_resumed,
                deadline_at=self.clock() + 20 * 60, clock=self.clock,
                monotonic_clock=self.monotonic, boot_id="boot-1",
                resume_open=True,
            )
            self.addCleanup(adopted2.close)
        self.clock.advance(1.0)
        self.monotonic.advance(1.0)
        p2, _ = self.request(adopted2, m2["payload"]["bytes"], "media_chunk")
        adopted2.mark_dispatch(p2)
        adopted2.finish_request(p2, 200)
        self.clock.advance(0.55)
        self.monotonic.advance(0.55)
        r2, _ = self.request(adopted2, m2["receipt"]["bytes"], "media_chunk")
        adopted2.mark_dispatch(r2)
        adopted2.finish_request(r2, 200)
        self.finish_index(adopted2, m2, d2)

        def assert_mutated_ledger_fails(sql_setup, expected_regex):
            copy_path = self.db_path.parent / "mutated.sqlite"
            src = sqlite3.connect(self.db_path)
            dst = sqlite3.connect(copy_path)
            src.backup(dst)
            src.close()
            dst.execute(
                "UPDATE retained_account SET db_path=? WHERE id=1",
                (str(copy_path.resolve()),),
            )
            sql_setup(dst)
            dst.commit()
            dst.close()
            try:
                with self.assertRaisesRegex(AuthorityBlocked, expected_regex):
                    TransferAuthority.inspect_retained_account(
                        copy_path,
                        writer_id=self.writer_id,
                        account_id=self.account_id,
                        clock=self.clock,
                    )
            finally:
                copy_path.unlink(missing_ok=True)

        # 0. Reconciliation manifest mismatch & duplicate payload generation across runs
        dup_m2 = self.manifest(1)
        dup_m2["release_id"] = m2["release_id"]
        dup_raw2, dup_d2 = self.raw_and_digest(dup_m2)
        assert_mutated_ledger_fails(
            lambda db: db.execute(
                "UPDATE retained_run SET manifest_raw=?, manifest_sha256=? WHERE id=?",
                (dup_raw2, dup_d2, m2["release_id"]),
            ),
            "Retained crash-adoption reconciliation attempt identity differs",
        )
        dup_m1 = self.manifest(2)
        dup_m1["release_id"] = m1["release_id"]
        dup_raw1, dup_d1 = self.raw_and_digest(dup_m1)
        assert_mutated_ledger_fails(
            lambda db: (
                db.execute("PRAGMA foreign_keys=OFF"),
                db.execute(
                    "UPDATE retained_run SET manifest_raw=?, manifest_sha256=?, pair_key='dup-key-1' WHERE id=?",
                    (dup_raw1, dup_d1, m1["release_id"]),
                ),
            ),
            "Retained ledger repeats a payload generation|Retained run manifest or limits changed",
        )

        # 1. Adoption initial deadline limit, adoption window malformed & adoption deadline mismatch
        assert_mutated_ledger_fails(
            lambda db: db.execute(
                "UPDATE retained_run_adoption SET prior_deadline_at = prior_deadline_at + 5"
            ),
            "Retained run manifest or limits changed",
        )
        assert_mutated_ledger_fails(
            lambda db: db.execute(
                "UPDATE retained_run_adoption SET adopted_at = prior_started_at - 5"
            ),
            "Retained run adoption window is malformed",
        )
        assert_mutated_ledger_fails(
            lambda db: db.execute(
                "UPDATE retained_run SET deadline_at = deadline_at + 1 WHERE id=?",
                (m2["release_id"],),
            ),
            "Retained run deadline differs from latest adoption",
        )

        # 2. Reconciled credit / spend & unresolved attempt on closed run
        assert_mutated_ledger_fails(
            lambda db: db.execute(
                "DELETE FROM retained_reconciliation WHERE attempt_id=?",
                (undispatched_p2,),
            ),
            "Index pending with uncertain Drive attempt|Retained index state or charge is malformed",
        )
        assert_mutated_ledger_fails(
            lambda db: db.execute(
                "UPDATE retained_attempt SET status='unresolved', http_status=500 WHERE id=?",
                (r1,),
            ),
            "Index pending with uncertain Drive attempt",
        )

        # 3. Mapping reservation object_name mismatch & missing mapping reservation
        assert_mutated_ledger_fails(
            lambda db: db.execute(
                "UPDATE retained_mapping_reservation SET object_name='wrong/object.json' WHERE run_id=?",
                (m2["release_id"],),
            ),
            "Retained mapping reservation is malformed",
        )
        assert_mutated_ledger_fails(
            lambda db: db.execute(
                "DELETE FROM retained_mapping_reservation WHERE run_id=?",
                (m2["release_id"],),
            ),
            "Closed retained pair has no mapping reservation",
        )

        # 4. Orphan retained attempt
        assert_mutated_ledger_fails(
            lambda db: (
                db.execute("PRAGMA foreign_keys=OFF"),
                db.execute(
                    "INSERT INTO retained_attempt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("orphan-att", "nonexistent-run", self.clock() - 50,
                     self.clock() - 50, self.clock() - 30, 20.0, None, None, 0,
                     "drive_api", "GET", "drive_api", "a" * 64,
                     "b" * 64, "settled", 200),
                ),
            ),
            "Orphan retained attempt",
        )

        # 5. Monotonic pacing violation within the same adoption epoch
        assert_mutated_ledger_fails(
            lambda db: db.execute(
                "UPDATE retained_attempt SET dispatch_monotonic = "
                "(SELECT dispatch_monotonic FROM retained_attempt WHERE id=?) + 0.1 "
                "WHERE id=?",
                (p2, r2),
            ),
            "Retained actual starts violate monotonic pacing",
        )

    def test_concurrent_open_run_reads_clock_after_begin_immediate_lock(self):
        import threading

        t0 = self.clock()
        m1 = self.manifest(1)
        raw1, d1 = self.raw_and_digest(m1)
        ev1 = self.evidence(m1, d1, observed_at=t0)

        m2 = self.manifest(2)
        raw2, d2 = self.raw_and_digest(m2)
        ev2 = self.evidence(m2, d2, observed_at=t0)

        # Call 2 enters open_run() first while shared_now == t0 (the earlier clock)
        # and waits to acquire BEGIN IMMEDIATE until Call 1 has acquired BEGIN IMMEDIATE,
        # advanced shared_now to t0 + 0.6, and committed last_clock = t0 + 0.6.
        shared_now = [t0]
        call2_clock_samples: list[float] = []
        call2_waiting_on_lock = threading.Event()
        call1_committed = threading.Event()
        call2_opened = threading.Event()
        call1_closed = threading.Event()
        results: dict[str, object] = {}

        orig_connect = retained_module._connect

        def instrumented_connect(path: Path, *, create: bool = False):
            conn = orig_connect(path, create=create)
            if threading.current_thread().name == "open-run-2":
                call2_waiting_on_lock.set()
                # Wait until Call 1 has committed last_clock = t0 + 0.6 before Call 2
                # acquires BEGIN IMMEDIATE inside _transaction().
                call1_committed.wait(timeout=10.0)
            return conn

        def clock_for_call1() -> float:
            if shared_now[0] < t0 + 0.6:
                shared_now[0] = t0 + 0.6
            return shared_now[0]

        def clock_for_call2() -> float:
            # If open_run() had sampled clock() before _connect() / BEGIN IMMEDIATE,
            # this would have been called while shared_now[0] == t0 (< t0 + 0.6) and
            # would have failed with "Retained ledger clock regressed".
            val = shared_now[0]
            call2_clock_samples.append(val)
            return val

        def run_call2():
            try:
                auth2 = TransferAuthority.open_retained_run(
                    self.db_path, run_id=m2["release_id"], writer_id=self.writer_id,
                    account_id=self.account_id, manifest_bytes=raw2, evidence=ev2,
                    deadline_at=t0 + 20 * 60, clock=clock_for_call2,
                    monotonic_clock=self.monotonic, boot_id="boot-race",
                    max_concurrent_runs=2,
                )
                try:
                    results["auth2_opened"] = True
                    call2_opened.set()
                    call1_closed.wait(timeout=10.0)
                    shared_now[0] += 0.6
                    self.monotonic.advance(0.6)
                    self.reserve(auth2, m2, d2)
                    p2, _ = self.request(auth2, m2["payload"]["bytes"], "media_chunk")
                    auth2.mark_dispatch(p2)
                    auth2.finish_request(p2, 200)
                    shared_now[0] += 0.6
                    self.monotonic.advance(0.6)
                    r2, _ = self.request(auth2, m2["receipt"]["bytes"], "media_chunk")
                    auth2.mark_dispatch(r2)
                    auth2.finish_request(r2, 200)
                    self.finish_index(auth2, m2, d2)
                finally:
                    auth2.close()
            except Exception as exc:
                results["err2"] = exc
                call2_opened.set()

        with patch.object(retained_module, "_connect", side_effect=instrumented_connect):
            t2 = threading.Thread(target=run_call2, name="open-run-2")
            t2.start()
            self.assertTrue(call2_waiting_on_lock.wait(timeout=10.0))
            # Call 2 has already passed pre-lock setup and is about to enter _transaction(),
            # and must not have sampled its clock before acquiring BEGIN IMMEDIATE.
            self.assertEqual(call2_clock_samples, [])

            auth1 = TransferAuthority.open_retained_run(
                self.db_path, run_id=m1["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw1, evidence=ev1,
                deadline_at=t0 + 20 * 60, clock=clock_for_call1,
                monotonic_clock=self.monotonic, boot_id="boot-race",
                max_concurrent_runs=2,
            )
            self.addCleanup(auth1.close)
            call1_committed.set()
            self.assertTrue(call2_opened.wait(timeout=10.0))

            shared_now[0] += 0.6
            self.monotonic.advance(0.6)
            self.reserve(auth1, m1, d1)
            p1, _ = self.request(auth1, m1["payload"]["bytes"], "media_chunk")
            auth1.mark_dispatch(p1)
            auth1.finish_request(p1, 200)
            shared_now[0] += 0.6
            self.monotonic.advance(0.6)
            r1, _ = self.request(auth1, m1["receipt"]["bytes"], "media_chunk")
            auth1.mark_dispatch(r1)
            auth1.finish_request(r1, 200)
            self.finish_index(auth1, m1, d1)
            call1_closed.set()
            t2.join(timeout=10.0)

        self.assertNotIn("err2", results, f"Call 2 failed unexpectedly: {results.get('err2')}")
        self.assertTrue(results.get("auth2_opened"))
        self.assertGreaterEqual(call2_clock_samples[0], t0 + 0.6)

        # Verify the clock-regression guard inside _transaction() remains intact.
        m3 = self.manifest(3)
        raw3, d3 = self.raw_and_digest(m3)
        ev3 = self.evidence(m3, d3, observed_at=t0)
        with self.assertRaisesRegex(AuthorityBlocked, "Retained ledger clock regressed"):
            TransferAuthority.open_retained_run(
                self.db_path, run_id=m3["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw3, evidence=ev3,
                deadline_at=t0 + 20 * 60, clock=lambda: t0,
                monotonic_clock=self.monotonic, boot_id="boot-race",
                max_concurrent_runs=2,
            )

    def test_mid_run_closed_attempt_tampering_detected_at_next_request_or_close(self) -> None:
        auth1, m1, d1 = self.open_run(1)
        self.reserve(auth1, m1, d1)
        a1, _ = self.request(auth1, 5, "media_chunk")
        auth1.mark_dispatch(a1)
        auth1.finish_request(a1, 200)
        self.finish_index(auth1, m1, d1)

        self.clock.advance(0.6)
        self.monotonic.advance(0.6)
        auth2, m2, d2 = self.open_run(2)
        self.reserve(auth2, m2, d2)

        conn = sqlite3.connect(self.db_path)
        try:
            closed_attempt = conn.execute(
                "SELECT rowid, id, operation_key, http_status, bytes FROM retained_attempt WHERE run_id = ? ORDER BY rowid ASC LIMIT 1",
                (m1["release_id"],),
            ).fetchone()
            self.assertIsNotNone(closed_attempt)
            closed_rowid, _closed_attempt_id, orig_op_key, orig_http, orig_bytes = closed_attempt
        finally:
            conn.close()

        def exec_outside(sql: str, params: tuple) -> None:
            c = sqlite3.connect(self.db_path)
            try:
                c.execute(sql, params)
                c.commit()
            finally:
                c.close()

        def restore_closed_attempt(*, assert_still_active: bool = True) -> None:
            exec_outside(
                "UPDATE retained_attempt SET status = 'settled', http_status = ?, bytes = ?, operation_key = ? WHERE rowid = ?",
                (orig_http, orig_bytes, orig_op_key, closed_rowid),
            )
            if assert_still_active:
                auth2.assert_active()

        # 1. Semantic edit to a closed run's attempt row mid-run (e.g. status='unresolved')
        # must be detected at assert_active and start_request.
        exec_outside(
            "UPDATE retained_attempt SET status = 'unresolved', http_status = 500 WHERE rowid = ?",
            (closed_rowid,),
        )
        with self.assertRaisesRegex(AuthorityBlocked, "Index pending with uncertain Drive attempt"):
            auth2.assert_active()
        with self.assertRaisesRegex(AuthorityBlocked, "Index pending with uncertain Drive attempt"):
            self.request(auth2, 5, "media_chunk")
        restore_closed_attempt()

        # 2. Subtle edit to a closed run's attempt row mid-run that keeps status='settled'
        # and valid http_status (200 -> 201) or valid hex operation_key, testing that the
        # SHA-256 closed-history fingerprint catches tampering on every per-request call
        # and on close_pair.
        self.clock.advance(0.6)
        self.monotonic.advance(0.6)
        p_attempt, _ = self.request(auth2, 5, "media_chunk")

        # 2a. Detected at seconds_until_dispatch
        exec_outside("UPDATE retained_attempt SET http_status = 201 WHERE rowid = ?", (closed_rowid,))
        with self.assertRaisesRegex(AuthorityBlocked, "Closed retained history fingerprint changed"):
            auth2.seconds_until_dispatch(p_attempt)
        restore_closed_attempt()

        # 2b. Detected at mark_dispatch (also test same-connection mutation via auth2.db)
        auth2.db.execute("UPDATE retained_attempt SET operation_key = ? WHERE rowid = ?", ("c" * 64, closed_rowid))
        auth2.db.commit()
        with self.assertRaisesRegex(AuthorityBlocked, "Closed retained history fingerprint changed"):
            auth2.mark_dispatch(p_attempt)
        restore_closed_attempt()

        auth2.mark_dispatch(p_attempt)

        # 2c. Detected at finish_request
        exec_outside("UPDATE retained_attempt SET http_status = 201 WHERE rowid = ?", (closed_rowid,))
        with self.assertRaisesRegex(AuthorityBlocked, "Closed retained history fingerprint changed"):
            auth2.finish_request(p_attempt, 200)
        restore_closed_attempt()
        auth2.finish_request(p_attempt, 200)

        # Mark verified so we can test detection at close_pair
        key2 = pair_key(m2)
        mapping_digest2 = hashlib.sha256((key2 + d2).encode()).hexdigest()
        auth2.mark_pair_verified(key2, mapping_digest2)

        # 2d. Detected at close_pair
        exec_outside("UPDATE retained_attempt SET http_status = 201 WHERE rowid = ?", (closed_rowid,))
        with self.assertRaisesRegex(AuthorityBlocked, "Closed retained history fingerprint changed"):
            auth2.close_pair(key2)
        restore_closed_attempt(assert_still_active=False)

        auth2.close_pair(key2)
        auth2.reserve_index_create(key2, mapping_digest2, INDEX_PREFIX + key2 + ".json", 64)
        auth2.mark_index_committed(key2, mapping_digest2)

    def test_per_request_validation_cost_is_constant_at_50_vs_3700_closed_runs(self) -> None:
        import time as wall_time

        def seed_closed_runs(db_path: Path, count: int, attempts_per_run: int = 53) -> tuple[FakeClock, FakeClock]:
            clk = FakeClock(self.clock())
            mono = FakeClock(self.monotonic())
            TransferAuthority.initialize_retained_observation(
                db_path, writer_id=self.writer_id, account_id=self.account_id, clock=clk
            )
            clk.advance(1.0)
            mono.advance(1.0)
            retained_module.RetainedObservationAuthority.import_legacy_result(
                db_path, result_bytes=self.legacy_raw, writer_id=self.writer_id,
                account_id=self.account_id, clock=clk
            )

            base_m = self.manifest(1)
            payload_obj = base_m["payload"]["object"]
            payload_sha = base_m["payload"]["sha256"]
            receipt_sha = base_m["receipt"]["sha256"]
            payload_bytes = int(base_m["payload"]["bytes"])
            receipt_bytes = int(base_m["receipt"]["bytes"])
            media_bytes = payload_bytes + receipt_bytes
            control_allowance = int(base_m["control_allowance_bytes"])
            budget_bytes = int(base_m["max_charged_bytes"])

            t_cursor = clk() + 10.0
            m_cursor = mono() + 10.0
            conn = sqlite3.connect(db_path)
            try:
                run_rows = []
                att_rows = []
                map_rows = []
                for idx in range(1, count + 1):
                    run_id = f"{idx:032x}"
                    gen = idx + 10
                    m = dict(base_m)
                    m["release_id"] = run_id
                    m["payload"] = dict(base_m["payload"], generation=gen)
                    m["receipt"] = dict(
                        base_m["receipt"],
                        object=payload_obj + f".g{gen}.receipt.json",
                        generation=idx + 100000,
                    )
                    pkey = hashlib.sha256(
                        f"{m['bucket']}:{payload_obj}:{gen}".encode()
                    ).hexdigest()
                    m["folder_name"] = "catbird-overflow-" + pkey[:32]
                    m["index_object"] = INDEX_PREFIX + pkey + ".json"
                    raw_m = json.dumps(m, sort_keys=True, separators=(",", ":")).encode()
                    digest_m = hashlib.sha256(raw_m).hexdigest()

                    started_at = t_cursor
                    deadline_at = started_at + 1200.0
                    ev_json = json.dumps({
                        "run_id": run_id,
                        "writer_id": self.writer_id,
                        "account_id": self.account_id,
                        "reviewed_manifest_sha256": digest_m,
                        "account_identity_verified": True,
                        "all_writers_excluded": True,
                        "exclusion_valid_until": deadline_at,
                        "observed_at": started_at,
                        "evidence_sha256": "e" * 64,
                        "history_basis": "unproven",
                        "quota_rejection_accepted": True,
                    }, sort_keys=True)
                    meta_json = json.dumps(
                        retained_module._expected_metadata(m, digest_m),
                        sort_keys=True,
                    )
                    mapping_sha = f"{idx:064x}"
                    run_rows.append((
                        run_id, digest_m, raw_m, pkey,
                        media_bytes, control_allowance, budget_bytes, meta_json,
                        1, 0, "closed", mapping_sha, ev_json,
                        started_at, deadline_at,
                    ))
                    map_rows.append((
                        run_id, pkey, mapping_sha, m["index_object"],
                        64, f"{(idx + 500000):064x}",
                    ))
                    for a_i in range(attempts_per_run):
                        at_val = started_at + (a_i + 1) * 0.6
                        m_val = m_cursor + (a_i + 1) * 0.6
                        att_rows.append((
                            f"att-{idx}-{a_i}", run_id,
                            at_val, at_val, at_val + 20.0, 20.0,
                            at_val, m_val,
                            1, "media_chunk", "PUT",
                            "media_chunk", payload_sha if a_i == 0 else receipt_sha,
                            "b" * 64, "settled", 200,
                        ))
                    t_cursor = started_at + (attempts_per_run + 2) * 0.6
                    m_cursor = m_cursor + (attempts_per_run + 2) * 0.6

                conn.executemany(
                    "INSERT INTO retained_run VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    run_rows,
                )
                conn.executemany(
                    "INSERT INTO retained_attempt VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    att_rows,
                )
                conn.executemany(
                    "INSERT INTO retained_mapping_reservation VALUES (?,?,?,?,?,?)",
                    map_rows,
                )
                last_wall = att_rows[-1][6]
                last_mono = att_rows[-1][7]
                conn.execute(
                    "UPDATE retained_account SET boot_id=?, last_clock=?, next_slot=?, "
                    "last_dispatch_wall=?, last_dispatch_monotonic=? WHERE id=1",
                    ("boot-bench", t_cursor, last_wall + 0.55, last_wall, last_mono),
                )
                conn.commit()
            finally:
                conn.close()
            clk.value = t_cursor + 1.0
            mono.value = m_cursor + 1.0
            return clk, mono

        def measure_active_run_per_request_calls(n_closed: int, n_requests: int = 20) -> dict[str, float]:
            db_p = Path(self.temp.name) / f"authority_{n_closed}.sqlite"
            clk, mono = seed_closed_runs(db_p, n_closed, attempts_per_run=53)
            open_idx = 2
            m = self.manifest(open_idx, release_id=f"{(n_closed + 9999):032x}")
            m["payload"]["generation"] = n_closed + 9999
            m["receipt"]["object"] = m["payload"]["object"] + f".g{m['payload']['generation']}.receipt.json"
            m["receipt"]["generation"] = n_closed + 200000
            m["folder_name"] = "catbird-overflow-" + pair_key(m)[:32]
            m["index_object"] = INDEX_PREFIX + pair_key(m) + ".json"
            raw, digest = self.raw_and_digest(m)
            ev = self.evidence(
                m, digest, observed_at=clk(),
                exclusion_valid_until=clk() + 1200.0,
            )

            t_open0 = wall_time.perf_counter()
            auth = TransferAuthority.open_retained_run(
                db_p, run_id=m["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw, evidence=ev,
                deadline_at=clk() + 1200.0, clock=clk,
                monotonic_clock=mono, boot_id="boot-bench",
            )
            open_ms = (wall_time.perf_counter() - t_open0) * 1000.0
            try:
                self.reserve(auth, m, digest)
                timings: dict[str, list[float]] = {
                    "assert_active": [],
                    "start_request": [],
                    "seconds_until_dispatch": [],
                    "mark_dispatch": [],
                    "finish_request": [],
                }
                for _i in range(n_requests):
                    clk.advance(0.6)
                    mono.advance(0.6)

                    t0 = wall_time.perf_counter()
                    auth.assert_active()
                    timings["assert_active"].append((wall_time.perf_counter() - t0) * 1000.0)

                    t0 = wall_time.perf_counter()
                    att_id, _ = self.request(auth, 1, "media_chunk")
                    timings["start_request"].append((wall_time.perf_counter() - t0) * 1000.0)

                    t0 = wall_time.perf_counter()
                    auth.seconds_until_dispatch(att_id)
                    timings["seconds_until_dispatch"].append((wall_time.perf_counter() - t0) * 1000.0)

                    t0 = wall_time.perf_counter()
                    auth.mark_dispatch(att_id)
                    timings["mark_dispatch"].append((wall_time.perf_counter() - t0) * 1000.0)

                    t0 = wall_time.perf_counter()
                    auth.finish_request(att_id, 200)
                    timings["finish_request"].append((wall_time.perf_counter() - t0) * 1000.0)

                summary = {k: sum(v) / len(v) for k, v in timings.items()}
                summary["open_run_full_validate_ms"] = open_ms
                summary["per_request_cycle_sum_ms"] = sum(
                    summary[k] for k in ("assert_active", "start_request", "seconds_until_dispatch", "mark_dispatch", "finish_request")
                )
                return summary
            finally:
                auth.close()

        t_50 = measure_active_run_per_request_calls(50)
        t_3700 = measure_active_run_per_request_calls(3700)

        for method in ("assert_active", "start_request", "seconds_until_dispatch", "mark_dispatch", "finish_request"):
            self.assertLess(
                t_3700[method], max(t_50[method] * 2.5, 5.0),
                f"{method} at 3,700 closed runs took {t_3700[method]:.3f} ms (50 runs: {t_50[method]:.3f} ms)",
            )
        # Ensure total 5-call request cycle at 3,700 closed runs (196,100 attempts) stays well under 20 ms
        # and within ~2.5x of 50 closed runs (2,650 attempts), despite 74x more rows.
        self.assertLess(t_3700["per_request_cycle_sum_ms"], 20.0)
        self.assertLess(t_3700["per_request_cycle_sum_ms"], max(t_50["per_request_cycle_sum_ms"] * 2.5, 5.0))

    def test_crash_adoption_adopts_zero_byte_drive_api_get_401_and_5xx(self):
        dead_pid = 999981
        m1 = self.manifest(1)
        raw1, d1 = self.raw_and_digest(m1)
        ev1 = self.evidence(m1, d1)
        with patch.object(retained_module.os, "getpid", return_value=dead_pid):
            run1 = TransferAuthority.open_retained_run(
                self.db_path, run_id=m1["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw1, evidence=ev1,
                deadline_at=self.clock() + 20 * 60, clock=self.clock,
                monotonic_clock=self.monotonic, boot_id="boot-1",
            )
            self.addCleanup(run1.close)
            self.reserve(run1, m1, d1)
            get_401, _ = run1.start_request(
                0, 60, "drive_api", method="GET",
                target_category="drive_api", target_sha256="1" * 64,
                operation_key="2" * 64,
            )
            run1.mark_dispatch(get_401)
            run1.finish_request(get_401, 401)
            self.clock.advance(0.6)
            self.monotonic.advance(0.6)
            get_503, _ = run1.start_request(
                0, 60, "drive_api", method="GET",
                target_category="drive_api", target_sha256="3" * 64,
                operation_key="4" * 64,
            )
            run1.mark_dispatch(get_503)
            run1.finish_request(get_503, 503)
            run1.close()

        with patch.object(retained_module.os, "kill", side_effect=ProcessLookupError):
            open_info = TransferAuthority.inspect_open_retained_run(
                self.db_path, writer_id=self.writer_id,
                account_id=self.account_id, run_id=m1["release_id"],
            )
            self.assertEqual(open_info["status"], "active")
            self.assertTrue(open_info["owner_host_matches"])
            self.assertFalse(open_info["owner_alive"])
            self.assertEqual(open_info["recoverable_unresolved_attempts"], 2)

            self.clock.advance(10.0)
            self.monotonic.advance(10.0)
            ev1_resumed = self.evidence(m1, d1, observed_at=self.clock())
            adopted = TransferAuthority.open_retained_run(
                self.db_path, run_id=m1["release_id"], writer_id=self.writer_id,
                account_id=self.account_id, manifest_bytes=raw1, evidence=ev1_resumed,
                deadline_at=self.clock() + 20 * 60, clock=self.clock,
                monotonic_clock=self.monotonic, boot_id="boot-1",
                resume_open=True,
            )
            self.addCleanup(adopted.close)

        acct_mid = TransferAuthority.inspect_retained_account(
            self.db_path, writer_id=self.writer_id,
            account_id=self.account_id, clock=self.clock,
        )
        self.assertEqual(acct_mid["unresolved_attempts"], 0)
        self.assertEqual(acct_mid["forensic_raw_unresolved_attempts"], 2)

        self.clock.advance(0.6)
        self.monotonic.advance(0.6)
        p1, _ = self.request(adopted, m1["payload"]["bytes"], "media_chunk")
        adopted.mark_dispatch(p1)
        adopted.finish_request(p1, 200)
        self.clock.advance(0.6)
        self.monotonic.advance(0.6)
        r1, _ = self.request(adopted, m1["receipt"]["bytes"], "media_chunk")
        adopted.mark_dispatch(r1)
        adopted.finish_request(r1, 200)
        self.finish_index(adopted, m1, d1)

        acct_done = TransferAuthority.inspect_retained_account(
            self.db_path, writer_id=self.writer_id,
            account_id=self.account_id, clock=self.clock,
        )
        self.assertEqual(acct_done["active_run_count"], 0)
        self.assertEqual(acct_done["unresolved_attempts"], 0)
        self.assertEqual(acct_done["forensic_raw_unresolved_attempts"], 2)

    def test_crash_adoption_refuses_post_and_put_401(self):
        for idx, (kind, method, body_bytes) in enumerate((
            ("overflow_folder_create", "POST", 128),
            ("drive_api", "POST", 0),
            ("media_chunk", "PUT", 1024),
            ("resumable_status", "PUT", 0),
            ("drive_api", "GET", 16),
        ), start=1):
            with self.subTest(kind=kind, method=method, body_bytes=body_bytes):
                sub_db = self.db_path.parent / f"refuse-{idx}.sqlite"
                TransferAuthority.initialize_retained_observation(
                    sub_db, writer_id=self.writer_id, account_id=self.account_id,
                )
                TransferAuthority.import_retained_legacy_diagnostic(
                    sub_db, result_bytes=self.legacy_raw,
                    writer_id=self.writer_id, account_id=self.account_id,
                    clock=self.clock,
                )
                dead_pid = 999900 + idx
                m = self.manifest(idx)
                raw, d = self.raw_and_digest(m)
                ev = self.evidence(m, d)
                with patch.object(retained_module.os, "getpid", return_value=dead_pid):
                    run = TransferAuthority.open_retained_run(
                        sub_db, run_id=m["release_id"], writer_id=self.writer_id,
                        account_id=self.account_id, manifest_bytes=raw, evidence=ev,
                        deadline_at=self.clock() + 20 * 60, clock=self.clock,
                        monotonic_clock=self.monotonic, boot_id="boot-1",
                    )
                    self.reserve(run, m, d)
                    att_id, _ = run.start_request(
                        body_bytes, 60, kind, method=method,
                        target_category=kind, target_sha256="a" * 64,
                        operation_key="b" * 64,
                    )
                    run.mark_dispatch(att_id)
                    run.finish_request(att_id, 401)
                    run.close()

                with patch.object(retained_module.os, "kill", side_effect=ProcessLookupError):
                    open_info = TransferAuthority.inspect_open_retained_run(
                        sub_db, writer_id=self.writer_id,
                        account_id=self.account_id, run_id=m["release_id"],
                    )
                    self.assertEqual(open_info["recoverable_unresolved_attempts"], 0)

                    self.clock.advance(10.0)
                    self.monotonic.advance(10.0)
                    ev_resumed = self.evidence(m, d, observed_at=self.clock())
                    adopted = TransferAuthority.open_retained_run(
                        sub_db, run_id=m["release_id"], writer_id=self.writer_id,
                        account_id=self.account_id, manifest_bytes=raw, evidence=ev_resumed,
                        deadline_at=self.clock() + 20 * 60, clock=self.clock,
                        monotonic_clock=self.monotonic, boot_id="boot-1",
                        resume_open=True,
                    )
                    try:
                        acct = TransferAuthority.inspect_retained_account(
                            sub_db, writer_id=self.writer_id,
                            account_id=self.account_id, clock=self.clock,
                        )
                        self.assertEqual(acct["unresolved_attempts"], 1)
                        with self.assertRaisesRegex(AuthorityBlocked, "Unresolved sends block"):
                            adopted.mark_pair_verified(
                                retained_module._reviewed_manifest(raw, m["release_id"], self.account_id)[2],
                                "c" * 64,
                            )
                        with self.assertRaisesRegex(AuthorityBlocked, "not an unresolved 0-byte Drive API GET"):
                            adopted.reconcile_retried_drive_get(att_id)
                    finally:
                        adopted.close()

                # Forging a crash-adoption reconciliation proof row for POST/PUT/non-0-byte GET must fail validation.
                key = retained_module._reviewed_manifest(raw, m["release_id"], self.account_id)[2]
                forged = {
                    "schema": retained_module.CRASH_ADOPTION_RECONCILIATION_SCHEMA,
                    "attempt_id": att_id,
                    "run_id": m["release_id"],
                    "pair_key": key,
                    "manifest_sha256": d,
                    "account_id": self.account_id,
                    "writer_id": self.writer_id,
                    "original_http_status": 401,
                    "original_charge_bytes": body_bytes,
                    "dispatched": True,
                    "uncommitted_retry_bytes": 0,
                    "applied_at": self.clock(),
                }
                forged_raw = retained_module._canonical(forged)
                forged_sha = hashlib.sha256(forged_raw.encode()).hexdigest()
                with sqlite3.connect(sub_db) as db:
                    db.execute(
                        "INSERT INTO retained_reconciliation VALUES (?,?,?,?,?,?)",
                        (att_id, m["release_id"], key, forged_raw, forged_sha, self.clock()),
                    )
                with self.assertRaisesRegex(
                    AuthorityBlocked,
                    "Retained crash-adoption reconciliation attempt identity differs",
                ):
                    TransferAuthority.inspect_retained_account(
                        sub_db, writer_id=self.writer_id,
                        account_id=self.account_id, clock=self.clock,
                    )

    def test_reconcile_retried_drive_get_on_real_authority_allows_pair_completion(self):
        authority, manifest, digest = self.open_run(1)
        self.reserve(authority, manifest, digest)

        failed_get, _ = authority.start_request(
            0, 60, "drive_api", method="GET",
            target_category="drive_api", target_sha256="1" * 64,
            operation_key="2" * 64,
        )
        authority.mark_dispatch(failed_get)
        authority.finish_request(failed_get, 401)

        acct_before = TransferAuthority.inspect_retained_account(
            self.db_path, writer_id=self.writer_id,
            account_id=self.account_id, clock=self.clock,
        )
        self.assertEqual(acct_before["unresolved_attempts"], 1)

        self.clock.advance(0.6)
        self.monotonic.advance(0.6)
        retried_get, _ = authority.start_request(
            0, 60, "drive_api", method="GET",
            target_category="drive_api", target_sha256="1" * 64,
            operation_key="2" * 64,
        )
        authority.mark_dispatch(retried_get)
        authority.finish_request(retried_get, 200)
        authority.reconcile_retried_drive_get(failed_get)

        acct_after = TransferAuthority.inspect_retained_account(
            self.db_path, writer_id=self.writer_id,
            account_id=self.account_id, clock=self.clock,
        )
        self.assertEqual(acct_after["unresolved_attempts"], 0)
        self.assertEqual(acct_after["forensic_raw_unresolved_attempts"], 1)

        self.clock.advance(0.6)
        self.monotonic.advance(0.6)
        p_att, _ = self.request(authority, manifest["payload"]["bytes"], "media_chunk")
        authority.mark_dispatch(p_att)
        authority.finish_request(p_att, 200)
        self.clock.advance(0.6)
        self.monotonic.advance(0.6)
        r_att, _ = self.request(authority, manifest["receipt"]["bytes"], "media_chunk")
        authority.mark_dispatch(r_att)
        authority.finish_request(r_att, 200)

        self.finish_index(authority, manifest, digest)

        acct_done = TransferAuthority.inspect_retained_account(
            self.db_path, writer_id=self.writer_id,
            account_id=self.account_id, clock=self.clock,
        )
        self.assertEqual(acct_done["active_run_count"], 0)
        self.assertEqual(acct_done["unresolved_attempts"], 0)
        self.assertEqual(acct_done["forensic_raw_unresolved_attempts"], 1)

    def test_validation_stats_counter_tracks_fallbacks_and_reasons(self) -> None:
        """Per-pair validation_stats records 2 full validations (open_run + close_pair) and 0 fallbacks normally, and increments exact fallback reasons."""
        auth, m1, d1 = self.open_run(1)
        try:
            self.reserve(auth, m1, d1)
            for _ in range(3):
                self.clock.advance(0.6)
                self.monotonic.advance(0.6)
                auth.assert_active()
                att_id, _ = self.request(auth, 1, "media_chunk")
                auth.seconds_until_dispatch(att_id)
                auth.mark_dispatch(att_id)
                auth.finish_request(att_id, 200)
            self.finish_index(auth, m1, d1)
            stats = auth.validation_stats()
            self.assertEqual(stats["open_run_full_validates"], 1)
            self.assertEqual(stats["close_pair_full_validates"], 1)
            self.assertEqual(stats["full_validate_calls"], 2)
            self.assertEqual(stats["fallbacks"], 0)
            self.assertEqual(stats["total_changes"], 0)
            self.assertEqual(stats["data_version"], 0)
            self.assertEqual(stats["cheap_fingerprint"], 0)
        finally:
            auth.close()

        self.clock.advance(0.6)
        self.monotonic.advance(0.6)
        auth2, m2, d2 = self.open_run(2)
        try:
            self.reserve(auth2, m2, d2)
            # Trigger total_changes fallback
            auth2._guarded_total_changes -= 1
            auth2.assert_active()
            self.assertEqual(auth2.validation_stats()["total_changes"], 1)
            self.assertEqual(auth2.validation_stats()["fallbacks"], 1)

            # Trigger data_version fallback
            auth2._guarded_data_version -= 1
            auth2.assert_active()
            self.assertEqual(auth2.validation_stats()["data_version"], 1)
            self.assertEqual(auth2.validation_stats()["fallbacks"], 2)

            # Trigger cheap_fingerprint fallback
            auth2._closed_fingerprint = dict(
                auth2._closed_fingerprint,
                max_attempt_rowid=auth2._closed_fingerprint["max_attempt_rowid"] + 1,
            )
            auth2.assert_active()
            self.assertEqual(auth2.validation_stats()["cheap_fingerprint"], 1)
            self.assertEqual(auth2.validation_stats()["fallbacks"], 3)
        finally:
            auth2.close()

    def test_64mib_and_128mib_chunks_and_8mib_reviewed_manifest_compatibility(self) -> None:
        """Verify 64 MiB and 128 MiB media_chunk attempts, status-probe reconciliation, crash adoption, and mixed 8 MiB -> 64 MiB closure on an unchanged reviewed manifest."""
        payload_size = 269 * (1 << 20)
        receipt_size = 1093
        m = self.manifest(1)
        m["payload"] = dict(m["payload"], bytes=payload_size)
        m["receipt"] = dict(m["receipt"], bytes=receipt_size)
        m["control_allowance_bytes"] = 1 << 20
        m["max_charged_bytes"] = payload_size + receipt_size + (1 << 20)
        raw, d = self.raw_and_digest(m)
        ev = self.evidence(m, d)

        auth = TransferAuthority.open_retained_run(
            self.db_path, run_id=m["release_id"], writer_id=self.writer_id,
            account_id=self.account_id, manifest_bytes=raw, evidence=ev,
            deadline_at=self.clock() + 1200.0, clock=self.clock,
            monotonic_clock=self.monotonic, boot_id="boot-1",
        )
        try:
            self.reserve(auth, m, d)
            session_sha = hashlib.sha256(b"https://www.googleapis.com/upload/session-1").hexdigest()

            # 1. Start with two legacy 8 MiB chunks (as if manifest was reviewed and started at 8 MiB)
            for idx in range(2):
                self.clock.advance(0.6)
                self.monotonic.advance(0.6)
                att_id, _ = auth.start_request(
                    8 << 20, 300, "media_chunk", method="PUT",
                    target_category="media_chunk", target_sha256=session_sha,
                    operation_key=f"{(idx + 100):064x}",
                )
                auth.mark_dispatch(att_id)
                auth.finish_request(att_id, 308)

            # 2. Switch to 64 MiB chunk that hits transient HTTP 503 after committing 16 MiB of the 64 MiB chunk
            self.clock.advance(0.6)
            self.monotonic.advance(0.6)
            failed_64m, _ = auth.start_request(
                64 << 20, 300, "media_chunk", method="PUT",
                target_category="media_chunk", target_sha256=session_sha,
                operation_key=f"{200:064x}",
            )
            auth.mark_dispatch(failed_64m)
            auth.finish_request(failed_64m, 503)

            # inspect_open_run must recognize the 64 MiB unresolved media_chunk as recoverable
            open_info = TransferAuthority.inspect_open_retained_run(
                self.db_path, writer_id=self.writer_id, account_id=self.account_id,
                run_id=m["release_id"],
            )
            self.assertEqual(open_info["recoverable_unresolved_attempts"], 1)

            # Status probe returns 308 acknowledging 32 MiB total (16 MiB prior + 16 MiB committed from the 64 MiB chunk)
            self.clock.advance(0.6)
            self.monotonic.advance(0.6)
            probe_64m, _ = auth.start_request(
                0, 60, "resumable_status", method="PUT",
                target_category="resumable_status", target_sha256=session_sha,
                operation_key=f"{201:064x}",
            )
            auth.mark_dispatch(probe_64m)
            auth.finish_request(probe_64m, 308)
            auth.reconcile_attempt_by_status_probe(
                failed_attempt_id=failed_64m,
                probe_attempt_id=probe_64m,
                acknowledged_bytes=32 << 20,
                committed_bytes=16 << 20,
            )

            # 3. Also test a 128 MiB chunk that is interrupted mid-flight (http_status=None),
            # then reconciled via status probe (0 bytes committed) and retried + completed
            self.clock.advance(0.6)
            self.monotonic.advance(0.6)
            interrupted_128m, _ = auth.start_request(
                128 << 20, 300, "media_chunk", method="PUT",
                target_category="media_chunk", target_sha256=session_sha,
                operation_key=f"{300:064x}",
            )
            auth.mark_dispatch(interrupted_128m)
            open_info_128 = TransferAuthority.inspect_open_retained_run(
                self.db_path, writer_id=self.writer_id, account_id=self.account_id,
                run_id=m["release_id"],
            )
            self.assertEqual(open_info_128["recoverable_unresolved_attempts"], 1)

            self.clock.advance(0.6)
            self.monotonic.advance(0.6)
            probe_128m, _ = auth.start_request(
                0, 60, "resumable_status", method="PUT",
                target_category="resumable_status", target_sha256=session_sha,
                operation_key=f"{301:064x}",
            )
            auth.mark_dispatch(probe_128m)
            auth.finish_request(probe_128m, 308)
            auth.reconcile_attempt_by_status_probe(
                failed_attempt_id=interrupted_128m,
                probe_attempt_id=probe_128m,
                acknowledged_bytes=32 << 20,
                committed_bytes=0,
            )

            # Finish remaining 237 MiB (128 MiB + 64 MiB + 45 MiB) + receipt and close the pair under the original manifest
            for rem_idx, chunk_sz in enumerate((128 << 20, 64 << 20, 45 << 20, receipt_size)):
                self.clock.advance(0.6)
                self.monotonic.advance(0.6)
                c_id, _ = auth.start_request(
                    chunk_sz, 300, "media_chunk", method="PUT",
                    target_category="media_chunk", target_sha256=session_sha,
                    operation_key=f"{(400 + rem_idx):064x}",
                )
                auth.mark_dispatch(c_id)
                auth.finish_request(c_id, 200 if rem_idx >= 2 else 308)

            self.finish_index(auth, m, d)

            # Verify _known_usage conservatively counts the wire bytes of both failed/interrupted
            # chunks (48 MiB uncommitted + 128 MiB uncommitted) in addition to the 269 MiB + receipt committed bytes
            acct = TransferAuthority.inspect_retained_account(
                self.db_path, writer_id=self.writer_id, account_id=self.account_id,
                clock=self.clock,
            )
            expected_wire = (
                retained_module.LEGACY_DIAGNOSTIC_CHARGED_BYTES
                + payload_size
                + receipt_size
                + (48 << 20)
                + (128 << 20)
            )
            self.assertEqual(acct["known_local_rolling_bytes"], expected_wire)
            self.assertEqual(acct["unresolved_attempts"], 0)
        finally:
            auth.close()


if __name__ == "__main__":
    unittest.main()





