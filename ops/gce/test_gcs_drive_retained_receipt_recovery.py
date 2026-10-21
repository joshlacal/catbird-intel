"""Synthetic state-machine tests: no Google calls, credentials, or real media."""
from __future__ import annotations

import hashlib
import fcntl
import json
from pathlib import Path
import os
import sqlite3
import tempfile
import time
import unittest
from unittest import mock

from drive_transfer_authority import AuthorityBlocked
from gcs_drive_drain import Conflict
from gcs_drive_retained_overflow import pair_key
from gcs_drive_retained_receipt_recovery import (
    ReceiptRecoveryController, finalize_reconciled_journal,
)
from test_gcs_drive_retained_overflow import (
    ACCOUNT, BUCKET, DRIVE_ROOT, NAME, PAYLOAD_BYTES, PAYLOAD_GEN,
    PAYLOAD_OBJECT, RECEIPT_GEN, RECEIPT_OBJECT, RELEASE, SOURCE_EPOCH,
    SOURCE_ID, fixture_manifest,
)


class FakeAuthority:
    account_id = ACCOUNT
    writer_id = "synthetic-writer"

    def __init__(self):
        self.permitted_branch = "probe_only"
        self.probe = None
        self.saved_session = None
        self.prepared = []
        self.proof = None
        self.db = None
        self.id_absence = None

    def prepare_request(self, *args):
        self.prepared.append(args)

    def record_probe(self, outcome):
        self.probe = {**outcome,
                      "session_sha256": hashlib.sha256(
                          b"https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=old"
                      ).hexdigest(),
                      "observed_at": time.time(), "proof_sha256": "a" * 64}
        return self.probe["proof_sha256"]

    def probe_outcome(self):
        return self.probe

    def record_new_session(self, url):
        self.saved_session = {"url": url,
                              "session_sha256": hashlib.sha256(url.encode()).hexdigest()}
        return self.saved_session["session_sha256"]

    def new_session(self):
        return self.saved_session

    def record_id_absence(self, evidence):
        self.id_absence = evidence

    def apply_verified_proof(self, proof):
        self.proof = proof
        return hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest()


class FakeTransport:
    def __init__(self):
        self.calls = []
        self.probe_result = {"state": "partial", "acknowledged": 4,
                             "http_status": 308}

    def probe(self, *args):
        self.calls.append(("probe", args))
        return self.probe_result

    def resume(self, *args):
        self.calls.append(("resume", args))
        return {"state": "complete", "http_status": 200,
                "metadata": {"id": args[4]}}

    def restart_same_id(self, *args):
        self.calls.append(("restart", args))
        return "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=new"


class FakeVerifier:
    def __init__(self):
        self.absent = True
        self.wrong_receipt = False

    def verify_account(self, account_id):
        if account_id != ACCOUNT:
            raise Conflict("synthetic account differs")

    def verify_folder(self, folder_id, name, root_id):
        if folder_id != "folder12345678" or root_id != DRIVE_ROOT or not name:
            raise Conflict("synthetic folder differs")

    def verify_exact_receipt_absent(self, **pins):
        return {**pins, "http_status": 404 if self.absent else 200,
                "observed_at": time.time()}

    def gcs_digest(self, object_name, generation):
        if (object_name, generation) == (PAYLOAD_OBJECT, PAYLOAD_GEN):
            data = PAYLOAD_BYTES
        elif (object_name, generation) == (RECEIPT_OBJECT, RECEIPT_GEN):
            data = self.receipt
        else:
            raise Conflict("synthetic GCS generation differs")
        return {"object": object_name, "generation": generation,
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "md5": hashlib.md5(data).hexdigest()}

    def drive_digest(self, file_id):
        if file_id == "payload12345678":
            data = PAYLOAD_BYTES
        elif file_id == "receipt12345678":
            data = self.receipt + (b"bad" if self.wrong_receipt else b"")
        else:
            raise Conflict("synthetic Drive ID differs")
        return {"id": file_id, "account_id": ACCOUNT,
                "parent_id": "folder12345678", "trashed": False,
                "media_readback": True, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "md5": hashlib.md5(data).hexdigest()}


class FakeDB:
    def __init__(self, target_sha):
        self.target_sha = target_sha

    def execute(self, *_args):
        return self

    def fetchone(self):
        return {"http_status": 500, "status": "unresolved",
                "method": "PUT", "kind": "media_chunk",
                "bytes": self.receipt_size, "target_sha256": self.target_sha,
                "operation_key": "b" * 64}


class RecoveryControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / "retained"
        self.manifest, self.receipt = fixture_manifest()
        self.raw_manifest = json.dumps(self.manifest, sort_keys=True,
                                       separators=(",", ":")).encode()
        key = pair_key(self.manifest)
        stage = self.work / "stage" / key
        stage.mkdir(parents=True)
        (stage / NAME).write_bytes(PAYLOAD_BYTES)
        (stage / (NAME + ".receipt.json")).write_bytes(self.receipt)
        sha = lambda data: hashlib.sha256(data).hexdigest()
        md5 = lambda data: hashlib.md5(data).hexdigest()
        self.old_session = ("https://www.googleapis.com/upload/drive/v3/files"
                            "?uploadType=resumable&upload_id=old")
        entry = {
            "key": key, "status": "file_upload_inflight",
            "release_id": RELEASE, "manifest_sha256": sha(self.raw_manifest),
            "archive_qualified": False,
            "payload_ref": {"name": PAYLOAD_OBJECT, "generation": PAYLOAD_GEN,
                            "size": len(PAYLOAD_BYTES)},
            "receipt_ref": {"name": RECEIPT_OBJECT, "generation": RECEIPT_GEN,
                            "size": len(self.receipt)},
            "payload_name": NAME, "receipt_name": NAME + ".receipt.json",
            "payload_digest": {"size": len(PAYLOAD_BYTES),
                               "sha256": sha(PAYLOAD_BYTES), "md5": md5(PAYLOAD_BYTES)},
            "receipt_digest": {"size": len(self.receipt),
                               "sha256": sha(self.receipt), "md5": md5(self.receipt)},
            "drive": {"folder": {"id": "folder12345678"},
                      "payload": {"id": "payload12345678"},
                      "receipt": {"id": "receipt12345678",
                                  "session": self.old_session}},
        }
        self.journal = {"schema": "catbird.gcs-drive-drain.v1",
                        "bucket": BUCKET, "root": DRIVE_ROOT,
                        "source_id": SOURCE_ID, "source_epoch": SOURCE_EPOCH,
                        "segments": {key: entry}, "folders": {}}
        self.authority = FakeAuthority()
        self.transport = FakeTransport()
        self.verifier = FakeVerifier()
        self.verifier.receipt = self.receipt
        self.authority.db = FakeDB(hashlib.sha256(self.old_session.encode()).hexdigest())
        self.authority.db.receipt_size = len(self.receipt)
        self.make()

    def make(self):
        self.controller = ReceiptRecoveryController(
            manifest_raw=self.raw_manifest,
            journal_raw=json.dumps(self.journal, sort_keys=True).encode(),
            original_run_id=RELEASE, original_attempt_id="attempt12345678",
            account_id=ACCOUNT, writer_id=self.authority.writer_id,
            work_dir=self.work, authority=self.authority,
            drive_request=lambda *_a, **_k: self.fail("network forbidden"),
            verifier=self.verifier)
        self.controller.transport = self.transport

    def test_probe_never_sends_media(self):
        result = self.controller.probe_only()
        self.assertEqual(result["state"], "partial")
        self.assertEqual([call[0] for call in self.transport.calls], ["probe"])

    def test_partial_requires_separate_review(self):
        self.controller.probe_only()
        with self.assertRaises(AuthorityBlocked):
            self.controller.continue_partial()
        self.assertEqual([call[0] for call in self.transport.calls], ["probe"])
        self.authority.permitted_branch = "partial_exact_suffix"
        self.controller.continue_partial()
        self.assertEqual([call[0] for call in self.transport.calls],
                         ["probe", "resume"])
        self.assertEqual(self.transport.calls[-1][1][2], 4)

    def test_partial_rejects_stale_probe(self):
        self.controller.probe_only()
        self.authority.permitted_branch = "partial_exact_suffix"
        self.authority.probe["observed_at"] -= 61
        with self.assertRaises(AuthorityBlocked):
            self.controller.continue_partial()
        self.assertEqual(len(self.transport.calls), 1)

    def test_expired_requires_exact_id_absence_and_persists_session_first(self):
        self.transport.probe_result = {"state": "expired", "acknowledged": 0,
                                       "http_status": 404}
        self.controller.probe_only()
        self.authority.permitted_branch = "expired_same_id_restart"
        self.verifier.absent = False
        with self.assertRaises(AuthorityBlocked):
            self.controller.start_expired(operator_reference="synthetic-operator")
        self.assertEqual([call[0] for call in self.transport.calls], ["probe"])
        self.verifier.absent = True
        self.controller.start_expired(operator_reference="synthetic-operator")
        self.assertEqual([call[0] for call in self.transport.calls],
                         ["probe", "restart"])
        self.assertEqual(self.authority.id_absence["receipt_id"], "receipt12345678")
        self.assertIsNotNone(self.authority.saved_session)
        self.controller.send_restarted()
        self.assertEqual([call[0] for call in self.transport.calls],
                         ["probe", "restart", "resume"])
        self.assertEqual(self.transport.calls[-1][1][2], 0)

    def test_uncharged_or_unpinned_original_session_rejected(self):
        self.journal["segments"][pair_key(self.manifest)]["drive"]["receipt"]["session"] = None
        with self.assertRaises(Conflict):
            self.make()

    def test_auto_partial_needs_fresh_probe_then_one_suffix(self):
        self.authority.permitted_branch = "receipt_only_auto"
        self.controller.verify_completed = lambda **_kw: {"state": "verified"}
        result = self.controller.run_authorized(operator_reference="synthetic-operator")
        self.assertEqual(result["state"], "verified")
        self.assertEqual([item[0] for item in self.transport.calls],
                         ["probe", "resume"])

    def test_auto_expired_needs_fresh_absence_then_same_id_session(self):
        self.authority.permitted_branch = "receipt_only_auto"
        self.transport.probe_result = {"state": "expired", "acknowledged": 0,
                                       "http_status": 404}
        self.controller.verify_completed = lambda **_kw: {"state": "verified"}
        result = self.controller.run_authorized(operator_reference="synthetic-operator")
        self.assertEqual(result["state"], "verified")
        self.assertEqual([item[0] for item in self.transport.calls],
                         ["probe", "restart", "resume"])
        self.assertIsNotNone(self.authority.id_absence)

    def test_auto_stops_on_malformed_probe_without_media(self):
        self.authority.permitted_branch = "receipt_only_auto"
        self.transport.probe_result = {"state": "partial", "acknowledged": 4,
                                       "http_status": 200}
        with self.assertRaises(AuthorityBlocked):
            self.controller.run_authorized(operator_reference="synthetic-operator")
        self.assertEqual([item[0] for item in self.transport.calls], ["probe"])

    def test_full_both_media_readback_commits_append_only_proof(self):
        result = self.controller.verify_completed(
            operator_reference="synthetic-operator")
        self.assertEqual(result["state"], "forensically_verified")
        self.assertEqual(self.authority.proof["original_http_status"], 500)
        self.assertEqual(self.authority.proof["original_charge_bytes"],
                         len(self.receipt))
        self.assertTrue(self.authority.proof["receipt_drive"]["media_readback"])

    def test_full_readback_rejects_bad_receipt_without_proof(self):
        self.verifier.wrong_receipt = True
        with self.assertRaises(Conflict):
            self.controller.verify_completed(operator_reference="synthetic-operator")
        self.assertIsNone(self.authority.proof)

    def _journal_ledger(self):
        self.db_path = self.work / "authority.sqlite"
        db = sqlite3.connect(self.db_path, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.executescript("""
            CREATE TABLE retained_run (id TEXT, status TEXT, pair_key TEXT,
              manifest_sha256 TEXT, mapping_sha256 TEXT);
            CREATE TABLE retained_account (id INTEGER, account_id TEXT,
              writer_id TEXT, db_path TEXT, process_host TEXT,
              process_pid INTEGER, process_nonce TEXT);
            CREATE TABLE retained_reconciliation (attempt_id TEXT, run_id TEXT,
              pair_key TEXT, proof_json TEXT, proof_sha256 TEXT);
            CREATE TABLE retained_receipt_recovery (id TEXT, status TEXT,
              original_attempt_id TEXT);
            CREATE TABLE retained_attempt (id TEXT, run_id TEXT, status TEXT,
              http_status INTEGER, bytes INTEGER, target_sha256 TEXT);
            CREATE TABLE retained_receipt_recovery_attempt (id TEXT, status TEXT);
        """)
        mapping = self.controller._mapping_sha256()
        proof = {"mapping_sha256": mapping,
                 "journal_sha256": self.controller.journal_sha,
                 "folder_id": self.controller.folder_id,
                 "payload_drive_id": self.controller.payload_id,
                 "receipt_drive_id": self.controller.receipt_id}
        raw = json.dumps(proof, sort_keys=True)
        db.execute("INSERT INTO retained_run VALUES (?,?,?,?,?)",
                   (RELEASE, "verified", self.controller.key,
                    self.controller.manifest_sha, mapping))
        self.authority.process_host = "synthetic-host"
        self.authority.process_pid = os.getpid()
        self.authority.process_nonce = "synthetic-nonce"
        db.execute("INSERT INTO retained_account VALUES (1,?,?,?,?,?,?)",
                   (ACCOUNT, self.authority.writer_id, str(self.db_path),
                    self.authority.process_host, self.authority.process_pid,
                    self.authority.process_nonce))
        db.execute("INSERT INTO retained_reconciliation VALUES (?,?,?,?,?)",
                   ("attempt12345678", RELEASE, self.controller.key,
                    raw, hashlib.sha256(raw.encode()).hexdigest()))
        db.execute("INSERT INTO retained_receipt_recovery VALUES (?,?,?)",
                   ("recovery123", "closed", "attempt12345678"))
        db.execute("INSERT INTO retained_attempt VALUES (?,?,?,?,?,?)",
                   ("attempt12345678", RELEASE, "unresolved", 500,
                    len(self.receipt),
                    hashlib.sha256(self.old_session.encode()).hexdigest()))
        db.execute("INSERT INTO retained_receipt_recovery_attempt VALUES (?,?)",
                   ("probe123", "settled"))
        self.authority.db = db
        self.authority.db_path = str(self.db_path)
        self.authority.release_id = "recovery123"
        self.addCleanup(db.close)
        self.journal_path = self.work / "journal.json"
        self.journal_path.write_text(json.dumps(self.journal, sort_keys=True))
        self.journal_path.chmod(0o600)
        self.lock_path = self.work / "overflow.lock"
        self.lock_path.write_bytes(b"")
        self.lock_path.chmod(0o600)
        self.lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_NOFOLLOW)
        fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.addCleanup(os.close, self.lock_fd)

    @mock.patch("gcs_drive_retained_receipt_recovery.RetainedObservationAuthority.inspect_account",
                return_value={"unresolved_attempts": 0})
    def test_journal_finalizer_is_no_send_and_idempotent(self, _inspect):
        self._journal_ledger()
        first = finalize_reconciled_journal(self.controller,
                                             held_lock_fd=self.lock_fd)
        self.assertEqual(first["state"], "drive_verified_no_send")
        self.assertEqual(json.loads(self.journal_path.read_text())["segments"]
                         [self.controller.key]["status"], "drive_verified")
        evidence = self.work / "retained-recovery-journal-evidence.jsonl"
        original = evidence.read_bytes()
        self.assertEqual((self.work / "journal.before-recovery.json").read_bytes(),
                         self.controller.journal_raw)
        again = finalize_reconciled_journal(self.controller,
                                             held_lock_fd=self.lock_fd)
        self.assertEqual(again, first)
        self.assertEqual(evidence.read_bytes(), original)

    @mock.patch("gcs_drive_retained_receipt_recovery.RetainedObservationAuthority.inspect_account",
                return_value={"unresolved_attempts": 0})
    def test_journal_finalizer_recovers_after_pre_replace_crash(self, _inspect):
        self._journal_ledger()
        with mock.patch("gcs_drive_retained_receipt_recovery.os.replace",
                        side_effect=OSError("synthetic crash")):
            with self.assertRaises(OSError):
                finalize_reconciled_journal(self.controller,
                                             held_lock_fd=self.lock_fd)
        self.assertEqual(json.loads(self.journal_path.read_text())["segments"]
                         [self.controller.key]["status"], "file_upload_inflight")
        self.assertTrue((self.work / "retained-recovery-journal-evidence.jsonl").exists())
        self.assertEqual((self.work / "journal.before-recovery.json").read_bytes(),
                         self.controller.journal_raw)
        result = finalize_reconciled_journal(self.controller,
                                              held_lock_fd=self.lock_fd)
        self.assertEqual(result["state"], "drive_verified_no_send")

    @mock.patch("gcs_drive_retained_receipt_recovery.RetainedObservationAuthority.inspect_account",
                return_value={"unresolved_attempts": 0})
    def test_journal_finalizer_rejects_wrong_lock_inode(self, _inspect):
        self._journal_ledger()
        wrong = os.open(self.journal_path, os.O_RDONLY)
        try:
            with self.assertRaises(Conflict):
                finalize_reconciled_journal(self.controller,
                                             held_lock_fd=wrong)
        finally:
            os.close(wrong)
        self.assertEqual(json.loads(self.journal_path.read_text())["segments"]
                         [self.controller.key]["status"], "file_upload_inflight")


if __name__ == "__main__":
    unittest.main()
