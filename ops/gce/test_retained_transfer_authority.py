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
                 recovery_only: bool = False):
        manifest = self.manifest(index) if manifest is None else manifest
        raw, digest = self.raw_and_digest(manifest)
        evidence = self.evidence(manifest, digest) if evidence is None else evidence
        authority = TransferAuthority.open_retained_run(
            self.db_path, run_id=manifest["release_id"], writer_id=self.writer_id,
            account_id=self.account_id, manifest_bytes=raw, evidence=evidence,
            deadline_at=self.clock() + 20 * 60, clock=self.clock,
            monotonic_clock=self.monotonic, boot_id=boot_id,
            recovery_only=recovery_only,
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
        self.finish_index(authority, manifest, digest)
        with self.assertRaises(AuthorityBlocked):
            self.open_run(2, boot_id="different-boot")
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


if __name__ == "__main__":
    unittest.main()
