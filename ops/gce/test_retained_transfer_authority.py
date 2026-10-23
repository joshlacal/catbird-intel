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


if __name__ == "__main__":
    unittest.main()




