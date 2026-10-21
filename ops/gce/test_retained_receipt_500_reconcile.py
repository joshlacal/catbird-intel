"""Synthetic-only regression tests for one ambiguous retained receipt PUT.

No test here uses Google credentials, network I/O, or a real Drive/GCS object.
The HTTP 500 attempt is deliberately retained as a raw unresolved charge.
"""
from __future__ import annotations

import hashlib
import io
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
import urllib.error
from unittest import mock

from drive_transfer_authority import TransferAuthority
import gcs_drive_retained_recovery_authority as recovery_module
from gcs_drive_retained_recovery_authority import RecoveryAuthority
from gcs_drive_retained_receipt_recovery import ReceiptRecoveryController
from gcs_drive_drain import Conflict
from gcs_drive_retained_overflow import pair_key
import retained_transfer_authority as retained
from test_gcs_drive_retained_overflow import (
    ACCOUNT, BUCKET, DRIVE_ROOT, NAME, PAYLOAD_BYTES, PAYLOAD_GEN,
    PAYLOAD_OBJECT, RECEIPT_GEN, RECEIPT_OBJECT, RELEASE, SOURCE_EPOCH,
    SOURCE_ID, fixture_manifest,
)

class Clock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, amount: float) -> None:
        self.value += amount


class ReadOnlyVerifier:
    """Full-media synthetic witness; intentionally has no write methods."""

    def __init__(self, receipt_bytes: bytes):
        self.receipt_bytes = receipt_bytes
        self.payload_bytes = PAYLOAD_BYTES
        self.account = ACCOUNT
        self.folder_id = "synthetic-folder-id"
        self.payload_id = "synthetic-payload-id"
        self.receipt_id = "synthetic-receipt-id"
        self.calls: list[tuple] = []
        self.present = True

    def verify_account(self, account_id: str) -> None:
        self.calls.append(("account", account_id))
        if account_id != self.account:
            raise Conflict("Synthetic account differs")

    def verify_folder(self, folder_id: str, name: str, root_id: str) -> None:
        self.calls.append(("folder", folder_id))
        if (folder_id != self.folder_id or
                name != "catbird-overflow-" + pair_key(self.manifest)[:32] or
                root_id != DRIVE_ROOT):
            raise Conflict("Synthetic folder differs")

    def verify_exact_receipt_absent(self, **pins) -> dict:
        self.calls.append(("exact_absence", pins.get("receipt_id")))
        expected = {"account_id": self.account, "root_id": DRIVE_ROOT,
                    "folder_id": self.folder_id,
                    "receipt_id": self.receipt_id}
        if pins != expected or self.present:
            raise Conflict("Synthetic receipt is not proven absent")
        return {**expected, "http_status": 404, "observed_at": self.clock()}

    @staticmethod
    def _digest(data: bytes) -> dict:
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "md5": hashlib.md5(data).hexdigest()}

    def gcs_digest(self, object_name: str, generation: int) -> dict:
        self.calls.append(("gcs", object_name, generation))
        if (object_name, generation) == (PAYLOAD_OBJECT, PAYLOAD_GEN):
            return {**self._digest(self.payload_bytes), "object": object_name,
                    "generation": generation}
        if (object_name, generation) == (RECEIPT_OBJECT, RECEIPT_GEN):
            return {**self._digest(self.receipt_bytes), "object": object_name,
                    "generation": generation}
        raise Conflict("Synthetic GCS source differs")

    def drive_digest(self, file_id: str) -> dict:
        self.calls.append(("drive", file_id))
        if not self.present:
            raise Conflict("Synthetic Drive media is absent")
        data = {self.payload_id: self.payload_bytes,
                self.receipt_id: self.receipt_bytes}.get(file_id)
        if data is None:
            raise Conflict("Synthetic Drive file ID differs")
        return {**self._digest(data), "id": file_id,
                "parent_id": self.folder_id, "account_id": self.account,
                "trashed": False, "media_readback": True}

    def __getattr__(self, name):
        if name in ("upload", "create", "put", "post", "delete", "retry"):
            raise AssertionError("Reconciliation must not send or retry media")
        raise AttributeError(name)


class FakeRecoveryResponse:
    def __init__(self, url: str, status: int, *, body: bytes = b"",
                 headers: dict | None = None):
        self.url, self.status = url, status
        self.body = io.BytesIO(body)
        self.headers = headers or {}

    def geturl(self) -> str:
        return self.url

    def read(self, size: int = -1) -> bytes:
        return self.body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.body.close()


class Receipt500ReconciliationTests(unittest.TestCase):
    """Use a real synthetic retained SQLite ledger and tiny staged source."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "authority.sqlite"
        self.clock = Clock(time.time() + 1)
        self.mono = Clock(time.monotonic() + 1)
        self.writer_id = "synthetic-sole-writer"
        self.manifest, self.receipt_bytes = fixture_manifest()
        self.manifest_raw = json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":")).encode()
        self.manifest_sha = hashlib.sha256(self.manifest_raw).hexdigest()
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_bytes(self.manifest_raw)
        self.pair = pair_key(self.manifest)
        self.work = self.root / ("gcs-drive-retained-" + RELEASE)
        self.stage = self.work / "stage" / self.pair
        self.stage.mkdir(parents=True)
        (self.work / "overflow.lock").write_bytes(b"")
        self.payload_path = self.stage / NAME
        self.receipt_path = self.stage / (NAME + ".receipt.json")
        self.payload_path.write_bytes(PAYLOAD_BYTES)
        self.receipt_path.write_bytes(self.receipt_bytes)
        self.verifier = ReadOnlyVerifier(self.receipt_bytes)
        self.verifier.manifest = self.manifest
        self.verifier.clock = self.clock
        self.payload_session = ("https://www.googleapis.com/upload/drive/v3/files"
                                "?uploadType=resumable&upload_id=synthetic-payload")
        self.receipt_session = ("https://www.googleapis.com/upload/drive/v3/files"
                                "?uploadType=resumable&upload_id=synthetic-receipt")
        self.journal = self.work / "journal.json"
        self._write_journal()
        self._open_synthetic_authority()

    def _write_journal(self) -> None:
        digest = lambda value: {"size": len(value),
                                "sha256": hashlib.sha256(value).hexdigest(),
                                "md5": hashlib.md5(value).hexdigest()}
        entry = {
            "key": self.pair, "status": "file_upload_inflight",
            "release_id": RELEASE, "manifest_sha256": self.manifest_sha,
            "archive_qualified": False,
            "payload_ref": {"name": PAYLOAD_OBJECT, "generation": PAYLOAD_GEN,
                            "size": len(PAYLOAD_BYTES), "metadata": {"sha256": digest(PAYLOAD_BYTES)["sha256"]}},
            "receipt_ref": {"name": RECEIPT_OBJECT, "generation": RECEIPT_GEN,
                            "size": len(self.receipt_bytes), "metadata": {"sha256": digest(self.receipt_bytes)["sha256"]}},
            "payload_name": NAME, "receipt_name": NAME + ".receipt.json",
            "payload_digest": digest(PAYLOAD_BYTES),
            "receipt_digest": digest(self.receipt_bytes),
            "drive": {
                "folder": {"id": self.verifier.folder_id},
                "payload": {"id": self.verifier.payload_id, "session": self.payload_session},
                "receipt": {"id": self.verifier.receipt_id, "session": self.receipt_session},
            },
        }
        document = {"schema": "catbird.gcs-drive-drain.v1", "bucket": BUCKET,
                    "root": DRIVE_ROOT, "source_id": SOURCE_ID,
                    "source_epoch": SOURCE_EPOCH,
                    "segments": {self.pair: entry}, "folders": {}}
        self.journal.write_text(json.dumps(document, sort_keys=True))

    def _legacy_result(self) -> bytes:
        first = self.clock() - 100
        last = first + 0.55
        value = {
            "release": {"id": "synthetic-legacy-release", "status": "closed",
                        "account_id": ACCOUNT,
                        "admission_mode": "diagnostic_quota_observation",
                        "started_at": first - 10, "deadline_at": self.clock() + 100,
                        "next_slot": last + 0.55, "cooldown_until": 0},
            "descriptor": {"release_id": "synthetic-legacy-release",
                           "account_id": ACCOUNT, "writer_id": self.writer_id,
                           "manifest_sha256": "a" * 64},
            "attempts": [
                {"id": "legacy-1", "bytes": 5, "at": first, "slot_at": first,
                 "dispatched_at": first, "charge_through": first + 1,
                 "http_status": 200, "method": "PUT", "status": "settled"},
                {"id": "legacy-2", "bytes": 7, "at": last, "slot_at": last,
                 "dispatched_at": last, "charge_through": last + 1,
                 "http_status": 308, "method": "PUT", "status": "settled"}],
            "charged_bytes": 12, "unresolved_attempts": 0,
            "history_proven": False, "archive_qualified": False,
            "entries": [{"status": "diagnostic_verified_unqualified",
                         "archive_qualified": False,
                         "payload_ref": {"name": retained.DIAGNOSTIC_PAYLOAD_OBJECT,
                                         "generation": retained.LEGACY_DIAGNOSTIC_PAYLOAD_GENERATION,
                                         "size": 268_980_444}}],
        }
        return json.dumps(value, sort_keys=True).encode()

    @staticmethod
    def _operation_key(session: str, kind: str, body: bytes) -> str:
        target_sha = hashlib.sha256(session.encode()).hexdigest()
        content_range = f"bytes 0-{len(body)-1}/{len(body)}"
        body_sha = hashlib.sha256(body).hexdigest()
        return hashlib.sha256("\0".join(
            ("PUT", kind, target_sha, content_range, body_sha)).encode()).hexdigest()

    def _open_synthetic_authority(self) -> None:
        TransferAuthority.initialize_retained_observation(
            self.db, writer_id=self.writer_id, account_id=ACCOUNT)
        raw = self._legacy_result()
        patches = [
            mock.patch.object(retained, "LEGACY_DIAGNOSTIC_SOURCE_SHA256",
                              hashlib.sha256(raw).hexdigest()),
            mock.patch.object(retained, "LEGACY_DIAGNOSTIC_RELEASE_ID", "synthetic-legacy-release"),
            mock.patch.object(retained, "LEGACY_DIAGNOSTIC_ATTEMPTS", 2),
            mock.patch.object(retained, "LEGACY_DIAGNOSTIC_CHARGED_BYTES", 12),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        TransferAuthority.import_retained_legacy_diagnostic(
            self.db, result_bytes=raw, writer_id=self.writer_id,
            account_id=ACCOUNT, clock=self.clock)
        evidence = {"run_id": RELEASE, "writer_id": self.writer_id,
                    "account_id": ACCOUNT,
                    "reviewed_manifest_sha256": self.manifest_sha,
                    "account_identity_verified": True,
                    "all_writers_excluded": True,
                    "quota_rejection_accepted": True,
                    "exclusion_valid_until": self.clock() + 1200,
                    "observed_at": self.clock(), "evidence_sha256": "e" * 64,
                    "history_basis": "unproven"}
        authority = TransferAuthority.open_retained_run(
            self.db, run_id=RELEASE, writer_id=self.writer_id, account_id=ACCOUNT,
            manifest_bytes=self.manifest_raw, evidence=evidence,
            deadline_at=self.clock() + 1200,
            clock=self.clock, monotonic_clock=self.mono,
            boot_id="synthetic-boot")
        self.addCleanup(authority.close)
        metadata = {"manifest_sha256": self.manifest_sha,
                    "folder_name": self.manifest["folder_name"],
                    "drive_root": DRIVE_ROOT,
                    "payload_object": PAYLOAD_OBJECT,
                    "payload_generation": PAYLOAD_GEN,
                    "receipt_object": RECEIPT_OBJECT,
                    "receipt_generation": RECEIPT_GEN}
        authority.reserve_pair(self.pair, len(PAYLOAD_BYTES) + len(self.receipt_bytes),
                               self.manifest["control_allowance_bytes"], metadata)
        self.payload_attempt = self._send(authority, self.payload_session,
                                          PAYLOAD_BYTES, 200)
        self.clock.advance(0.56)
        self.mono.advance(0.56)
        self.receipt_attempt = self._send(authority, self.receipt_session,
                                          self.receipt_bytes, 500)

    def _send(self, authority, session: str, body: bytes, status: int) -> str:
        target_sha = hashlib.sha256(session.encode()).hexdigest()
        attempt, slot = authority.start_request(
            len(body), 20, "media_chunk", method="PUT",
            target_category="media_chunk", target_sha256=target_sha,
            operation_key=self._operation_key(session, "media_chunk", body))
        if self.clock() < slot:
            delta = slot - self.clock() + 0.001
            self.clock.advance(delta)
            self.mono.advance(delta)
        authority.mark_dispatch(attempt)
        authority.finish_request(attempt, status)
        return attempt

    def _raw_attempt(self) -> tuple:
        with closing(sqlite3.connect(self.db)) as db, db:
            return db.execute("SELECT status,http_status,bytes,charge_through "
                              "FROM retained_attempt WHERE id=?",
                              (self.receipt_attempt,)).fetchone()

    def _recovery_evidence(self) -> dict:
        return {
            "schema": "catbird.drive.receipt-recovery-review.v1",
            "account_id": ACCOUNT, "writer_id": self.writer_id,
            "original_run_id": RELEASE,
            "original_attempt_id": self.receipt_attempt,
            "receipt_id": self.verifier.receipt_id,
            "session_sha256": hashlib.sha256(self.receipt_session.encode()).hexdigest(),
            "manifest_sha256": self.manifest_sha, "pair_key": self.pair,
            "permitted_branch": "probe_only",
            "receipt_stage_path": str(self.receipt_path),
            "receipt_sha256": self.manifest["receipt"]["sha256"],
            "folder_id": self.verifier.folder_id,
            "payload_id": self.verifier.payload_id,
            "receipt_name": self.receipt_path.name,
            "all_writers_excluded": True,
            "account_identity_verified": True,
            "exclusion_valid_until": self.clock() + 600,
            "owner_reference": "synthetic-review-only",
            "observed_at": self.clock(),
        }

    def _open_recovery(self, **changes):
        kwargs = {
            "db_path": self.db, "recovery_id": "c" * 32,
            "original_run_id": RELEASE,
            "original_attempt_id": self.receipt_attempt,
            "account_id": ACCOUNT, "writer_id": self.writer_id,
            "receipt_id": self.verifier.receipt_id,
            "session_url": self.receipt_session,
            "evidence": self._recovery_evidence(),
            "deadline_at": self.clock() + 600,
            "clock": self.clock, "monotonic_clock": self.mono,
        }
        kwargs.update(changes)
        # The preceding synthetic writer is in this test process. Model its
        # verified termination at the handoff without changing production DB.
        with mock.patch("gcs_drive_retained_recovery_authority.os.kill",
                        side_effect=ProcessLookupError), mock.patch(
                            "gcs_drive_retained_recovery_authority._boot_id",
                            return_value="synthetic-boot"):
            return RecoveryAuthority.open_new(**kwargs)

    def _prepare_probe(self, recovery: RecoveryAuthority) -> tuple[str, float]:
        receipt_size = len(self.receipt_bytes)
        content_range = f"bytes */{receipt_size}"
        recovery.prepare_request(self.receipt_session, "PUT", "resumable_status",
                                 b"", {"Content-Range": content_range})
        target_sha = hashlib.sha256(self.receipt_session.encode()).hexdigest()
        operation_key = hashlib.sha256("\0".join((
            "PUT", "resumable_status", target_sha, content_range,
            hashlib.sha256(b"").hexdigest(),
        )).encode()).hexdigest()
        return recovery.start_request(
            0, 20, "resumable_status", method="PUT",
            target_category="resumable_status", target_sha256=target_sha,
            operation_key=operation_key,
        )

    def _prepare_receipt_media(self, recovery: RecoveryAuthority) -> tuple[str, float]:
        target_sha = hashlib.sha256(self.receipt_session.encode()).hexdigest()
        content_range = f"bytes 0-{len(self.receipt_bytes)-1}/{len(self.receipt_bytes)}"
        recovery.prepare_request(
            self.receipt_session, "PUT", "media_chunk", self.receipt_bytes,
            {"Content-Range": content_range},
        )
        return recovery.start_request(
            len(self.receipt_bytes), 20, "media_chunk", method="PUT",
            target_category="media_chunk", target_sha256=target_sha,
            operation_key=self._operation_key(
                self.receipt_session, "media_chunk", self.receipt_bytes,
            ),
        )

    def _partial_probe(self, recovery: RecoveryAuthority) -> None:
        attempt_id, slot = self._prepare_probe(recovery)
        self._dispatch_probe(recovery, attempt_id, slot)
        recovery.finish_request(attempt_id, 308)
        recovery.record_probe({"state": "partial", "http_status": 308,
                               "acknowledged": 0})

    def _dispatch_probe(self, recovery: RecoveryAuthority,
                        attempt_id: str, slot: float) -> None:
        wait = recovery.seconds_until_dispatch(attempt_id)
        if wait:
            self.clock.advance(wait + 0.01)
            self.mono.advance(wait + 0.01)
        if self.clock() < slot:
            delta = slot - self.clock() + 0.01
            self.clock.advance(delta)
            self.mono.advance(delta)
        recovery.mark_dispatch(attempt_id)

    def test_recovery_lease_preserves_original_http500_and_charge(self):
        before = self._raw_attempt()
        recovery = self._open_recovery()
        self.addCleanup(recovery.close)
        self.assertEqual(self._raw_attempt(), before)
        with closing(sqlite3.connect(self.db)) as db, db:
            row = db.execute("SELECT original_run_id,original_attempt_id,receipt_id,"
                             "session_sha256,status FROM retained_receipt_recovery").fetchone()
        self.assertEqual(row, (RELEASE, self.receipt_attempt,
                               self.verifier.receipt_id,
                               hashlib.sha256(self.receipt_session.encode()).hexdigest(),
                               "open"))
        with self.assertRaises(Exception):
            self._open_recovery(recovery_id="d" * 32)

    def test_recovery_lease_rejects_wrong_session_id_attempt_or_review(self):
        variants = [
            {"session_url": self.payload_session},
            {"receipt_id": self.verifier.payload_id},
            {"original_attempt_id": self.payload_attempt},
            {"evidence": {**self._recovery_evidence(), "all_writers_excluded": False}},
            {"deadline_at": self.clock() + 1201},
        ]
        for variant in variants:
            with self.subTest(variant=tuple(variant)):
                with self.assertRaises(Exception):
                    self._open_recovery(**variant)
                self.assertEqual(self._raw_attempt()[:3],
                                 ("unresolved", 500, len(self.receipt_bytes)))

    def test_recovery_receipt_name_must_match_real_generation_object(self):
        evidence = self._recovery_evidence()
        self.assertEqual(evidence["receipt_name"], NAME + ".receipt.json")
        self.assertNotEqual(evidence["receipt_name"], Path(RECEIPT_OBJECT).name)
        with self.assertRaises(Exception):
            self._open_recovery(evidence={**evidence,
                                          "receipt_name": Path(RECEIPT_OBJECT).name})
        self.assertEqual(self._raw_attempt()[:2], ("unresolved", 500))

    def test_original_receipt_operation_key_must_bind_full_staged_body(self):
        original = self._raw_attempt()
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("UPDATE retained_attempt SET operation_key=? WHERE id=?",
                       ("0" * 64, self.receipt_attempt))
        with self.assertRaises(Exception):
            self._open_recovery()
        self.assertEqual(self._raw_attempt(), original)

    def test_lease_reserves_entire_receipt_and_control_cap_up_front(self):
        with closing(sqlite3.connect(self.db)) as db, db:
            old_total = db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_attempt"
            ).fetchone()[0]
            imported = db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_import"
            ).fetchone()[0]
        cap = (old_total + imported + len(self.receipt_bytes) +
               recovery_module.MAX_CONTROL_BYTES - 1)
        with mock.patch.object(recovery_module, "GLOBAL_ROLLING_CAP_BYTES", cap):
            with self.assertRaises(Exception):
                self._open_recovery()
        with closing(sqlite3.connect(self.db)) as db, db:
            recovery_count = db.execute(
                "SELECT COUNT(*) FROM retained_receipt_recovery"
            ).fetchone()[0]
        self.assertEqual(recovery_count, 0)
        self.assertEqual(self._raw_attempt()[:2], ("unresolved", 500))

    def test_zero_body_original_session_probe_is_charged_and_paced(self):
        original = self._raw_attempt()
        recovery = self._open_recovery()
        self.addCleanup(recovery.close)
        attempt_id, slot = self._prepare_probe(recovery)
        self.assertGreater(slot, self.clock())
        with self.assertRaises(Exception):
            recovery.mark_dispatch(attempt_id)
        self._dispatch_probe(recovery, attempt_id, slot)
        recovery.finish_request(attempt_id, 404)
        with closing(sqlite3.connect(self.db)) as db, db:
            probe = db.execute(
                "SELECT status,http_status,bytes,kind,method,dispatched_at "
                "FROM retained_receipt_recovery_attempt WHERE id=?", (attempt_id,)
            ).fetchone()
            old = db.execute(
                "SELECT status,http_status,bytes,charge_through "
                "FROM retained_attempt WHERE id=?", (self.receipt_attempt,)
            ).fetchone()
        self.assertEqual(probe[:5],
                         ("settled", 404, 0, "resumable_status", "PUT"))
        self.assertIsNotNone(probe[5])
        self.assertEqual(old, original)

    def test_probe_308_is_terminal_for_query_without_reconciling_http500(self):
        recovery = self._open_recovery()
        self.addCleanup(recovery.close)
        attempt_id, slot = self._prepare_probe(recovery)
        self._dispatch_probe(recovery, attempt_id, slot)
        recovery.finish_request(attempt_id, 308)
        with closing(sqlite3.connect(self.db)) as db, db:
            probe = db.execute(
                "SELECT status,http_status FROM retained_receipt_recovery_attempt "
                "WHERE id=?", (attempt_id,)
            ).fetchone()
        self.assertEqual(probe, ("settled", 308))
        self.assertEqual(self._raw_attempt()[:2], ("unresolved", 500))

    def test_uncertain_status_probe_forbids_any_second_probe(self):
        recovery = self._open_recovery()
        self.addCleanup(recovery.close)
        attempt_id, slot = self._prepare_probe(recovery)
        self._dispatch_probe(recovery, attempt_id, slot)
        recovery.finish_request(attempt_id, 500)
        with closing(sqlite3.connect(self.db)) as db, db:
            probe = db.execute(
                "SELECT status,http_status FROM "
                "retained_receipt_recovery_attempt WHERE id=?",
                (attempt_id,),
            ).fetchone()
        self.assertEqual(probe, ("unresolved", 500))
        with self.assertRaises(Exception):
            self._prepare_probe(recovery)
        self.assertEqual(self._raw_attempt()[:2], ("unresolved", 500))

    def test_missing_probe_response_forbids_second_probe(self):
        recovery = self._open_recovery()
        self.addCleanup(recovery.close)
        attempt_id, slot = self._prepare_probe(recovery)
        self._dispatch_probe(recovery, attempt_id, slot)
        recovery.finish_request(attempt_id, None)
        with self.assertRaises(Exception):
            self._prepare_probe(recovery)
        self.assertEqual(self._raw_attempt()[:2], ("unresolved", 500))

    def test_recovery_preserves_normal_writer_monotonic_pacing(self):
        recovery = self._open_recovery()
        self.addCleanup(recovery.close)
        attempt_id, slot = self._prepare_probe(recovery)
        # Make the wall time appear past the slot while leaving the monotonic
        # clock at the prior normal writer's dispatch instant.
        self.clock.advance(max(0, slot - self.clock()) + 1)
        with self.assertRaises(Exception):
            recovery.mark_dispatch(attempt_id)
        wait = recovery.seconds_until_dispatch(attempt_id)
        self.assertGreater(wait, 0)
        self.mono.advance(wait + 0.01)
        recovery.mark_dispatch(attempt_id)
        recovery.finish_request(attempt_id, 404)
        self.assertEqual(self._raw_attempt()[:2], ("unresolved", 500))

    def test_near_cap_counts_old_unresolved_charge_before_receipt_media(self):
        evidence = {**self._recovery_evidence(),
                    "permitted_branch": "partial_exact_suffix"}
        recovery = self._open_recovery(evidence=evidence)
        self.addCleanup(recovery.close)
        self._partial_probe(recovery)
        with closing(sqlite3.connect(self.db)) as db, db:
            old_total = db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_attempt"
            ).fetchone()[0]
            imported = db.execute(
                "SELECT COALESCE(SUM(bytes),0) FROM retained_import"
            ).fetchone()[0]
        # One byte short of sufficient headroom; the old HTTP 500 is included.
        cap = old_total + imported + len(self.receipt_bytes) - 1
        with mock.patch.object(recovery_module, "GLOBAL_ROLLING_CAP_BYTES", cap):
            with self.assertRaises(Exception):
                self._prepare_receipt_media(recovery)
        with closing(sqlite3.connect(self.db)) as db, db:
            media_attempts = db.execute(
                "SELECT COUNT(*) FROM retained_receipt_recovery_attempt "
                "WHERE kind='media_chunk'"
            ).fetchone()[0]
        self.assertEqual(media_attempts, 0)
        self.assertEqual(self._raw_attempt()[:2], ("unresolved", 500))

    def test_recovery_media_5xx_remains_charged_in_shared_inspection(self):
        evidence = {**self._recovery_evidence(),
                    "permitted_branch": "partial_exact_suffix"}
        recovery = self._open_recovery(evidence=evidence)
        self.addCleanup(recovery.close)
        self._partial_probe(recovery)
        before = TransferAuthority.inspect_retained_account(
            self.db, writer_id=self.writer_id, account_id=ACCOUNT,
            clock=self.clock,
        )
        attempt_id, slot = self._prepare_receipt_media(recovery)
        self._dispatch_probe(recovery, attempt_id, slot)
        recovery.finish_request(attempt_id, 500)
        after = TransferAuthority.inspect_retained_account(
            self.db, writer_id=self.writer_id, account_id=ACCOUNT,
            clock=self.clock,
        )
        # The up-front reservation converts into a charged unresolved media
        # attempt; the shared total must not fall when the 500 arrives.
        self.assertGreaterEqual(after["known_local_rolling_bytes"],
                                before["known_local_rolling_bytes"])
        self.assertEqual(after["unresolved_attempts"],
                         before["unresolved_attempts"] + 1)
        with closing(sqlite3.connect(self.db)) as db, db:
            charged = db.execute(
                "SELECT bytes,status,http_status FROM "
                "retained_receipt_recovery_attempt WHERE id=?", (attempt_id,)
            ).fetchone()
        self.assertEqual(charged, (len(self.receipt_bytes), "unresolved", 500))
        with self.assertRaises(Exception):
            self._prepare_receipt_media(recovery)
        self.assertEqual(self._raw_attempt()[:2], ("unresolved", 500))

    def test_probe_wrong_session_or_range_never_reserves_attempt(self):
        recovery = self._open_recovery()
        self.addCleanup(recovery.close)
        for url, content_range in (
            (self.payload_session, f"bytes */{len(self.receipt_bytes)}"),
            (self.receipt_session, "bytes */1"),
            (self.receipt_session, f"bytes 0-{len(self.receipt_bytes)-1}/{len(self.receipt_bytes)}"),
        ):
            with self.subTest(url=url == self.receipt_session,
                              content_range=content_range):
                with self.assertRaises(Exception):
                    recovery.prepare_request(url, "PUT", "resumable_status", b"",
                                             {"Content-Range": content_range})
        with closing(sqlite3.connect(self.db)) as db, db:
            attempts = db.execute(
                "SELECT COUNT(*) FROM retained_receipt_recovery_attempt"
            ).fetchone()[0]
        self.assertEqual(attempts, 0)

    def test_integrated_expired_same_id_receipt_only_and_forensic_gate(self):
        evidence = {**self._recovery_evidence(),
                    "permitted_branch": "receipt_only_auto"}
        recovery = self._open_recovery(evidence=evidence)
        self.addCleanup(recovery.close)
        self.verifier.present = False
        new_session = ("https://www.googleapis.com/upload/drive/v3/files"
                       "?uploadType=resumable&upload_id=synthetic-new-receipt")
        sends = []

        def safe_request(url, method="GET", data=None, headers=None, *,
                         timeout=60, kind="drive_api"):
            body = data or b""
            content_range = next((value for key, value in (headers or {}).items()
                                  if key.lower() == "content-range"), "")
            target_sha = hashlib.sha256(url.encode()).hexdigest()
            key = recovery_module._operation_key(method, kind, target_sha,
                                                 content_range, body)
            attempt_id, slot = recovery.start_request(
                len(body), timeout, kind, method=method,
                target_category=kind, target_sha256=target_sha,
                operation_key=key,
            )
            self._dispatch_probe(recovery, attempt_id, slot)
            sends.append((kind, method, url, len(body), content_range))
            if kind == "resumable_status":
                recovery.finish_request(attempt_id, 404)
                raise urllib.error.HTTPError(url, 404, "synthetic expired session",
                                             {}, io.BytesIO())
            if kind == "resumable_start":
                recovery.finish_request(attempt_id, 200)
                return FakeRecoveryResponse(
                    url, 200, headers={"Location": new_session},
                )
            if kind == "media_chunk":
                self.assertEqual(url, new_session)
                self.assertEqual(body, self.receipt_bytes)
                recovery.finish_request(attempt_id, 200)
                self.verifier.present = True
                return FakeRecoveryResponse(
                    url, 200,
                    body=json.dumps({"id": self.verifier.receipt_id}).encode(),
                )
            self.fail("Unexpected recovery request kind")

        controller = ReceiptRecoveryController(
            manifest_raw=self.manifest_raw,
            journal_raw=self.journal.read_bytes(),
            original_run_id=RELEASE, original_attempt_id=self.receipt_attempt,
            account_id=ACCOUNT, writer_id=self.writer_id, work_dir=self.work,
            authority=recovery, drive_request=safe_request,
            verifier=self.verifier, clock=self.clock,
        )
        result = controller.run_authorized(operator_reference="synthetic-owner")
        self.assertEqual(result["state"], "forensically_verified")
        self.assertEqual([send[0] for send in sends],
                         ["resumable_status", "resumable_start", "media_chunk"])
        self.assertEqual([send[3] for send in sends],
                         [0, sends[1][3], len(self.receipt_bytes)])
        self.assertEqual(self._raw_attempt()[:3],
                         ("unresolved", 500, len(self.receipt_bytes)))
        state = TransferAuthority.inspect_retained_account(
            self.db, writer_id=self.writer_id, account_id=ACCOUNT,
            clock=self.clock,
        )
        self.assertEqual(state["unresolved_attempts"], 0)
        self.assertEqual(state["nonterminal_status"], "verified")
        with closing(sqlite3.connect(self.db)) as db, db:
            recovery_rows = db.execute(
                "SELECT kind,bytes,status,http_status FROM "
                "retained_receipt_recovery_attempt ORDER BY at,id"
            ).fetchall()
            proof_count = db.execute(
                "SELECT COUNT(*) FROM retained_reconciliation"
            ).fetchone()[0]
            mapping_count = db.execute(
                "SELECT COUNT(*) FROM retained_mapping_reservation"
            ).fetchone()[0]
        self.assertEqual([row[0] for row in recovery_rows],
                         ["resumable_status", "resumable_start", "media_chunk"])
        self.assertTrue(all(row[2] == "settled" for row in recovery_rows))
        self.assertEqual(proof_count, 1)
        self.assertEqual(mapping_count, 0)  # Normal queue remains gated until mapping.

    def test_adopted_open_lease_after_410_status_probe_completes_same_id_recovery(self):
        evidence = {**self._recovery_evidence(),
                    "permitted_branch": "receipt_only_auto"}
        first = self._open_recovery(evidence=evidence)
        attempt_id, slot = self._prepare_probe(first)
        self._dispatch_probe(first, attempt_id, slot)
        # Simulate the exact v5 state where 410 was recorded as unresolved and
        # the process exited before probe_json was written.
        first.db.execute(
            "UPDATE retained_receipt_recovery_attempt SET status='unresolved',"
            "http_status=410,charge_through=MAX(charge_through,?) WHERE id=?",
            (self.clock(), attempt_id),
        )
        first.db.commit()
        first.close()

        self.clock.advance(5)
        self.mono.advance(5)
        adopted = self._open_recovery(
            recovery_id="e" * 32,
            evidence={**evidence, "observed_at": self.clock(),
                      "exclusion_valid_until": self.clock() + 600},
        )
        self.addCleanup(adopted.close)
        self.assertEqual(adopted.release_id, "c" * 32)
        self.assertEqual(adopted.probe_outcome()["state"], "expired")
        self.assertEqual(adopted.probe_outcome()["http_status"], 410)

        self.verifier.present = False
        new_session = ("https://www.googleapis.com/upload/drive/v3/files"
                       "?uploadType=resumable&upload_id=synthetic-new-after-410")
        sends = []

        def safe_request(url, method="GET", data=None, headers=None, *,
                         timeout=60, kind="drive_api"):
            body = data or b""
            content_range = next((value for key, value in (headers or {}).items()
                                  if key.lower() == "content-range"), "")
            target_sha = hashlib.sha256(url.encode()).hexdigest()
            key = recovery_module._operation_key(method, kind, target_sha,
                                                 content_range, body)
            req_id, req_slot = adopted.start_request(
                len(body), timeout, kind, method=method,
                target_category=kind, target_sha256=target_sha,
                operation_key=key,
            )
            self._dispatch_probe(adopted, req_id, req_slot)
            sends.append((kind, method, url, len(body), content_range))
            if kind == "resumable_start":
                adopted.finish_request(req_id, 200)
                return FakeRecoveryResponse(
                    url, 200, headers={"Location": new_session},
                )
            if kind == "media_chunk":
                self.assertEqual(url, new_session)
                self.assertEqual(body, self.receipt_bytes)
                adopted.finish_request(req_id, 200)
                self.verifier.present = True
                return FakeRecoveryResponse(
                    url, 200,
                    body=json.dumps({"id": self.verifier.receipt_id}).encode(),
                )
            self.fail(f"Unexpected recovery request kind: {kind}")

        controller = ReceiptRecoveryController(
            manifest_raw=self.manifest_raw,
            journal_raw=self.journal.read_bytes(),
            original_run_id=RELEASE, original_attempt_id=self.receipt_attempt,
            account_id=ACCOUNT, writer_id=self.writer_id, work_dir=self.work,
            authority=adopted, drive_request=safe_request,
            verifier=self.verifier, clock=self.clock,
        )
        result = controller.run_authorized(operator_reference="synthetic-owner")
        self.assertEqual(result["state"], "forensically_verified")
        self.assertEqual([send[0] for send in sends],
                         ["resumable_start", "media_chunk"])
        with closing(sqlite3.connect(self.db)) as db, db:
            recovery_rows = db.execute(
                "SELECT kind,bytes,status,http_status FROM "
                "retained_receipt_recovery_attempt ORDER BY slot_at,at,id"
            ).fetchall()
        self.assertEqual(
            recovery_rows,
            [("resumable_status", 0, "settled", 410),
             ("resumable_start", sends[0][3], "settled", 200),
             ("media_chunk", len(self.receipt_bytes), "settled", 200)],
        )

    def test_integrated_payload_tail_and_receipt_auto_recovery_and_journal_repair(self):
        import fcntl
        import os
        from gcs_drive_retained_receipt_recovery import finalize_reconciled_journal

        prefix_len = 7
        tail_bytes = PAYLOAD_BYTES[prefix_len:]
        payload_target_sha = hashlib.sha256(self.payload_session.encode()).hexdigest()
        tail_range = f"bytes {prefix_len}-{len(PAYLOAD_BYTES)-1}/{len(PAYLOAD_BYTES)}"
        tail_op_key = hashlib.sha256("\0".join((
            "PUT", "media_chunk", payload_target_sha, tail_range,
            hashlib.sha256(tail_bytes).hexdigest(),
        )).encode()).hexdigest()
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute(
                "UPDATE retained_attempt SET bytes=?,http_status=308,status='settled' "
                "WHERE id=?",
                (prefix_len, self.payload_attempt),
            )
            db.execute(
                "UPDATE retained_attempt SET bytes=?,target_sha256=?,operation_key=?,"
                "http_status=500,status='unresolved' WHERE id=?",
                (len(tail_bytes), payload_target_sha, tail_op_key, self.receipt_attempt),
            )
            db.execute(
                "UPDATE retained_run SET reserved_remaining=budget_bytes-? WHERE id=?",
                (prefix_len + len(tail_bytes), RELEASE),
            )
        doc = json.loads(self.journal.read_text())
        doc["segments"][self.pair]["drive"]["receipt"] = {"id": self.verifier.receipt_id}
        self.journal.write_text(json.dumps(doc, sort_keys=True))
        os.chmod(self.journal, 0o600)
        os.chmod(self.work / "overflow.lock", 0o600)

        evidence = {
            **self._recovery_evidence(),
            "session_sha256": payload_target_sha,
            "permitted_branch": "payload_tail_and_receipt_auto",
            "payload_name": NAME,
            "payload_stage_path": str(self.payload_path),
            "payload_sha256": self.manifest["payload"]["sha256"],
        }
        recovery = self._open_recovery(
            session_url=self.payload_session,
            evidence=evidence,
        )
        self.addCleanup(recovery.close)
        self.verifier.present = False
        new_receipt_session = (
            "https://www.googleapis.com/upload/drive/v3/files"
            "?uploadType=resumable&upload_id=synthetic-receipt-after-payload-tail"
        )
        sends = []

        def safe_request(url, method="GET", data=None, headers=None, *,
                         timeout=60, kind="drive_api"):
            body = data or b""
            content_range = next((value for key, value in (headers or {}).items()
                                  if key.lower() == "content-range"), "")
            target_sha = hashlib.sha256(url.encode()).hexdigest()
            key = recovery_module._operation_key(method, kind, target_sha,
                                                 content_range, body)
            req_id, req_slot = recovery.start_request(
                len(body), timeout, kind, method=method,
                target_category=kind, target_sha256=target_sha,
                operation_key=key,
            )
            self._dispatch_probe(recovery, req_id, req_slot)
            sends.append((kind, method, url, len(body), content_range))
            if kind == "resumable_status":
                self.assertEqual(url, self.payload_session)
                self.assertEqual(content_range, f"bytes */{len(PAYLOAD_BYTES)}")
                recovery.finish_request(req_id, 308)
                raise urllib.error.HTTPError(
                    url, 308, "Resume Incomplete",
                    {"Range": f"bytes=0-{prefix_len - 1}"}, io.BytesIO(),
                )
            if kind == "media_chunk" and url == self.payload_session:
                self.assertEqual(body, tail_bytes)
                self.assertEqual(content_range, tail_range)
                recovery.finish_request(req_id, 200)
                return FakeRecoveryResponse(
                    url, 200,
                    body=json.dumps({"id": self.verifier.payload_id}).encode(),
                )
            if kind == "resumable_start":
                recovery.finish_request(req_id, 200)
                return FakeRecoveryResponse(
                    url, 200, headers={"Location": new_receipt_session},
                )
            if kind == "media_chunk" and url == new_receipt_session:
                self.assertEqual(body, self.receipt_bytes)
                recovery.finish_request(req_id, 200)
                self.verifier.present = True
                return FakeRecoveryResponse(
                    url, 200,
                    body=json.dumps({"id": self.verifier.receipt_id}).encode(),
                )
            self.fail(f"Unexpected recovery request: {kind} {url}")

        controller = ReceiptRecoveryController(
            manifest_raw=self.manifest_raw,
            journal_raw=self.journal.read_bytes(),
            original_run_id=RELEASE, original_attempt_id=self.receipt_attempt,
            account_id=ACCOUNT, writer_id=self.writer_id, work_dir=self.work,
            authority=recovery, drive_request=safe_request,
            verifier=self.verifier, clock=self.clock,
        )
        result = controller.run_authorized(operator_reference="synthetic-owner")
        self.assertEqual(result["state"], "forensically_verified")
        self.assertEqual(
            [(s[0], s[3]) for s in sends],
            [
                ("resumable_status", 0),
                ("media_chunk", len(tail_bytes)),
                ("resumable_start", sends[2][3]),
                ("media_chunk", len(self.receipt_bytes)),
            ],
        )
        lock_fd = os.open(self.work / "overflow.lock", os.O_RDONLY)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            repaired = finalize_reconciled_journal(controller, held_lock_fd=lock_fd)
        finally:
            os.close(lock_fd)
        self.assertEqual(repaired["state"], "drive_verified_no_send")
        self.assertEqual(
            json.loads(self.journal.read_text())["segments"][self.pair]["status"],
            "drive_verified",
        )

    def test_payload_tail_500_expired_session_adopted_and_receipt_uploaded(self) -> None:
        import os

        prefix_len = 7
        tail_bytes = PAYLOAD_BYTES[prefix_len:]
        payload_target_sha = hashlib.sha256(self.payload_session.encode()).hexdigest()
        tail_range = f"bytes {prefix_len}-{len(PAYLOAD_BYTES)-1}/{len(PAYLOAD_BYTES)}"
        tail_op_key = hashlib.sha256("\0".join((
            "PUT", "media_chunk", payload_target_sha, tail_range,
            hashlib.sha256(tail_bytes).hexdigest(),
        )).encode()).hexdigest()
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute(
                "UPDATE retained_attempt SET bytes=?,http_status=308,status='settled' "
                "WHERE id=?",
                (prefix_len, self.payload_attempt),
            )
            db.execute(
                "UPDATE retained_attempt SET bytes=?,target_sha256=?,operation_key=?,"
                "http_status=500,status='unresolved' WHERE id=?",
                (len(tail_bytes), payload_target_sha, tail_op_key, self.receipt_attempt),
            )
            db.execute(
                "UPDATE retained_run SET reserved_remaining=budget_bytes-? WHERE id=?",
                (prefix_len + len(tail_bytes), RELEASE),
            )
        doc = json.loads(self.journal.read_text())
        doc["segments"][self.pair]["drive"]["receipt"] = {"id": self.verifier.receipt_id}
        self.journal.write_text(json.dumps(doc, sort_keys=True))
        os.chmod(self.journal, 0o600)

        evidence = {
            **self._recovery_evidence(),
            "session_sha256": payload_target_sha,
            "permitted_branch": "payload_tail_and_receipt_auto",
            "payload_name": NAME,
            "payload_stage_path": str(self.payload_path),
            "payload_sha256": self.manifest["payload"]["sha256"],
        }
        recovery1 = self._open_recovery(
            session_url=self.payload_session,
            evidence=evidence,
        )
        probe_headers = {"Content-Range": f"bytes */{len(PAYLOAD_BYTES)}"}
        recovery1.prepare_request(
            self.payload_session, "PUT", "resumable_status", b"", probe_headers,
        )
        probe_key = recovery_module._operation_key(
            "PUT", "resumable_status", payload_target_sha,
            probe_headers["Content-Range"], b"",
        )
        req_id, req_slot = recovery1.start_request(
            0, 60.0, "resumable_status", method="PUT",
            target_category="resumable_status",
            target_sha256=payload_target_sha,
            operation_key=probe_key,
        )
        self._dispatch_probe(recovery1, req_id, req_slot)
        recovery1.finish_request(req_id, 410)
        recovery1.close()

        self.clock.advance(5)
        self.mono.advance(5)
        # Simulate adoption of the open recovery row whose status probe returned 410
        recovery2 = self._open_recovery(
            recovery_id="e" * 32,
            session_url=self.payload_session,
            evidence={
                **evidence,
                "observed_at": self.clock(),
                "exclusion_valid_until": self.clock() + 600,
            },
        )
        self.addCleanup(recovery2.close)
        self.assertEqual(recovery2.release_id, "c" * 32)
        self.assertEqual(recovery2.probe_outcome()["state"], "expired")
        self.assertEqual(recovery2.probe_outcome()["http_status"], 410)

        self.verifier.present = False
        new_receipt_session = (
            "https://www.googleapis.com/upload/drive/v3/files"
            "?uploadType=resumable&upload_id=receipt-after-payload-410-session"
        )
        sends: list[tuple[str, int]] = []

        def safe_request(url, method="GET", data=None, headers=None, *,
                         timeout=60, kind="drive_api"):
            body = data or b""
            content_range = next((value for key, value in (headers or {}).items()
                                  if key.lower() == "content-range"), "")
            target_sha = hashlib.sha256(url.encode()).hexdigest()
            key = recovery_module._operation_key(method, kind, target_sha,
                                                 content_range, body)
            r_id, r_slot = recovery2.start_request(
                len(body), timeout, kind, method=method,
                target_category=kind, target_sha256=target_sha,
                operation_key=key,
            )
            self._dispatch_probe(recovery2, r_id, r_slot)
            sends.append((kind, len(body)))
            if kind == "resumable_start":
                recovery2.finish_request(r_id, 200)
                return FakeRecoveryResponse(
                    url, 200, headers={"Location": new_receipt_session},
                )
            if kind == "media_chunk" and url == new_receipt_session:
                self.assertEqual(body, self.receipt_bytes)
                recovery2.finish_request(r_id, 200)
                self.verifier.present = True
                return FakeRecoveryResponse(
                    url, 200,
                    body=json.dumps({"id": self.verifier.receipt_id}).encode(),
                )
            self.fail(f"Unexpected recovery request: {kind} {url}")

        controller = ReceiptRecoveryController(
            manifest_raw=self.manifest_raw,
            journal_raw=self.journal.read_bytes(),
            original_run_id=RELEASE, original_attempt_id=self.receipt_attempt,
            account_id=ACCOUNT, writer_id=self.writer_id, work_dir=self.work,
            authority=recovery2, drive_request=safe_request,
            verifier=self.verifier, clock=self.clock,
        )
        result = controller.run_authorized(operator_reference="synthetic-owner")
        self.assertEqual(result["state"], "forensically_verified")
        self.assertEqual(
            sends,
            [
                ("resumable_start", sends[0][1]),
                ("media_chunk", len(self.receipt_bytes)),
            ],
        )


if __name__ == "__main__":
    unittest.main()


