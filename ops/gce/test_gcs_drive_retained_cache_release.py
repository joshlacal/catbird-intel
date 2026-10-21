"""Synthetic-only review of generated cache files; no removal or cloud calls."""
from __future__ import annotations

import hashlib
import json
import os
import fcntl
import io
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout
from types import SimpleNamespace

import gcs_drive_retained_cache_release as cache
import gcs_drive_retained_runner as runner
from gcs_drive_drain import Conflict
from gcs_drive_retained_overflow import (
    BUCKET, DRIVE_ROOT, FINAL_STATUS, INDEX_SCHEMA, _canonical, pair_key,
)
from test_gcs_drive_retained_overflow import (
    ACCOUNT, NAME, PAYLOAD_BYTES, RELEASE, fixture_manifest,
)


class ReadOnlyVerifier:
    def __init__(self, manifest, mapping, payload_bytes, receipt_bytes):
        self.manifest, self.mapping_record = manifest, mapping
        self.raw = {"payload": payload_bytes, "receipt": receipt_bytes}
        self.calls = []
    def verify_account(self, account_id):
        self.calls.append(("account", account_id))
        if account_id != ACCOUNT:
            raise Conflict("wrong account")
    def mapping(self, bucket, name):
        self.calls.append(("mapping", name))
        if bucket != BUCKET or name != self.manifest["index_object"]:
            raise Conflict("wrong mapping")
        return 3001, _canonical(self.mapping_record)
    def gcs_digest(self, object_name, generation):
        self.calls.append(("gcs", object_name, generation))
        kind = next(kind for kind in ("payload", "receipt")
                    if self.manifest[kind]["object"] == object_name)
        if self.manifest[kind]["generation"] != generation:
            raise Conflict("wrong GCS generation")
        return self.digest(self.raw[kind])
    def drive_digest(self, file_id):
        self.calls.append(("drive", file_id))
        kind = "payload" if file_id == "payload-id" else "receipt"
        return {**self.digest(self.raw[kind]), "id": file_id,
                "parent_id": "folder-id", "account_id": ACCOUNT,
                "trashed": False, "media_readback": True}
    @staticmethod
    def digest(raw):
        return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                "md5": hashlib.md5(raw, usedforsecurity=False).hexdigest()}


class SyntheticApprovalVerifier:
    def verify(self, approval):
        return {"authenticated": True, "approved_by": "Josh",
                "approval_reference": approval["approval_reference"],
                "inventory_sha256": approval["inventory_sha256"]}


class CacheReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_root = Path(self.temp.name)
        self.manifest, self.receipt_bytes = fixture_manifest()
        self.key = pair_key(self.manifest)
        self.raw_manifest = _canonical(self.manifest)
        self.manifest_sha = hashlib.sha256(self.raw_manifest).hexdigest()
        self.manifest_path = self.data_root / "manifest.json"
        self.manifest_path.write_bytes(self.raw_manifest)
        self.work = self.data_root / ("gcs-drive-retained-" + RELEASE)
        self.stage = self.work / "stage" / self.key
        self.stage.mkdir(parents=True)
        (self.stage / NAME).write_bytes(PAYLOAD_BYTES)
        (self.stage / (NAME + ".receipt.json")).write_bytes(self.receipt_bytes)
        self.ids = {"folder": "folder-id", "payload": "payload-id", "receipt": "receipt-id"}
        self.digests = {kind: {"size": self.manifest[kind]["bytes"],
                               "sha256": self.manifest[kind]["sha256"],
                               "md5": hashlib.md5(raw, usedforsecurity=False).hexdigest()}
                        for kind, raw in (("payload", PAYLOAD_BYTES),
                                          ("receipt", self.receipt_bytes))}
        self.mapping = {"schema": INDEX_SCHEMA, "pair_key": self.key,
                        "manifest_sha256": self.manifest_sha,
                        "index_bucket": BUCKET,
                        "index_object": self.manifest["index_object"],
                        "payload_ref": dict(self.manifest["payload"]),
                        "receipt_ref": dict(self.manifest["receipt"]),
                        "payload_digest": self.digests["payload"],
                        "receipt_digest": self.digests["receipt"],
                        "drive_account_id": ACCOUNT, "drive_root": DRIVE_ROOT,
                        "folder_id": self.ids["folder"],
                        "payload_id": self.ids["payload"],
                        "receipt_id": self.ids["receipt"],
                        "retained_gcs": True, "retained_local": True,
                        "archive_qualified": False, "status": FINAL_STATUS}
        self.mapping_sha = hashlib.sha256(_canonical(self.mapping)).hexdigest()
        entry = {"key": self.key, "release_id": RELEASE,
                 "manifest_sha256": self.manifest_sha, "status": FINAL_STATUS,
                 "archive_qualified": False, "mapping_sha256": self.mapping_sha,
                 "index_generation": 3001,
                 "drive": {kind: {"id": value} for kind, value in self.ids.items()}}
        for kind in ("payload", "receipt"):
            ref = self.manifest[kind]
            entry[kind + "_ref"] = {"name": ref["object"],
                                    "generation": ref["generation"],
                                    "size": ref["bytes"], "metadata": {}}
            entry[kind + "_digest"] = self.digests[kind]
        self.journal = {"schema": "catbird.gcs-drive-drain.v1", "bucket": BUCKET,
                        "root": DRIVE_ROOT, "segments": {self.key: entry}}
        (self.work / "journal.json").write_bytes(_canonical(self.journal))
        (self.work / "overflow.lock").write_bytes(b"")
        self.db_path = self.data_root / "synthetic.sqlite"
        sqlite3.connect(self.db_path).close()
        self.verifier = ReadOnlyVerifier(self.manifest, self.mapping,
                                         PAYLOAD_BYTES, self.receipt_bytes)
        self.account = {"unresolved_attempts": 0, "nonterminal_run_id": None}
        self.ledger = mock.patch.object(cache.TransferAuthority, "inspect_retained_pair",
                                         return_value={"status": "closed",
                                                       "mapping_sha256": self.mapping_sha})
        self.account_patch = mock.patch.object(cache.TransferAuthority,
                                               "inspect_retained_account",
                                               side_effect=lambda *a, **k: dict(self.account))
        self.ledger.start()
        self.account_patch.start()
        self.addCleanup(self.ledger.stop)
        self.addCleanup(self.account_patch.stop)

    def inspect(self, verifier=None):
        return cache.inspect_cache(manifest_path=self.manifest_path, run_id=RELEASE,
                                   account_id=ACCOUNT, writer_id="synthetic-writer",
                                   authority_db=self.data_root / "synthetic.sqlite",
                                   data_root=self.data_root, verifier=verifier)

    def assert_runner_disk_accepts_evidence(self):
        with mock.patch.object(runner.shutil, "disk_usage",
                               return_value=SimpleNamespace(free=200 * (1 << 30))):
            result = runner.check_disk({"manifest": self.manifest},
                                       data_root=self.data_root, capture_dir=self.data_root)
        self.assertGreater(result["staged_bytes"], 0)
        return result["staged_bytes"]

    def test_local_cli_evidence_stays_outside_runner_scan(self):
        before = self.assert_runner_disk_accepts_evidence()
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(cache.main(["--manifest", str(self.manifest_path),
                                         "--run-id", RELEASE, "--account-id", ACCOUNT,
                                         "--writer-id", "synthetic-writer",
                                         "--authority-db", str(self.data_root / "synthetic.sqlite"),
                                         "--data-root", str(self.data_root)]), 0)
        summary = json.loads(output.getvalue())
        self.assertFalse(summary["eligible_for_approval"])
        self.assertEqual(summary["exclude_reason"], "remote_readback_unavailable")
        self.assertTrue(Path(summary["evidence_log"]).exists())
        self.assertEqual(Path(summary["evidence_log"]).parent.name,
                         "catbird-retained-cache-evidence")
        self.assertEqual(self.assert_runner_disk_accepts_evidence(), before)

    def test_dry_run_local_review_does_not_claim_remote_verification(self):
        row = self.inspect()
        self.assertFalse(row["eligible_for_approval"])
        self.assertFalse(row["gcs_retained"])
        self.assertEqual(row["exclude_reason"], "remote_readback_unavailable")
        self.assertTrue(row["creation_retained_local"])
        self.assertEqual(row["current_cache_state"], "present")
        self.assertEqual(row["stage"]["payload"]["sha256"],
                         self.manifest["payload"]["sha256"])

    def test_exact_readbacks_make_only_a_review_candidate(self):
        row = self.inspect(self.verifier)
        self.assertTrue(row["eligible_for_approval"])
        self.assertTrue(row["gcs_retained"])
        self.assertTrue(row["drive_media_verified"])
        self.assertEqual(row["mapping_generation"], 3001)
        self.assertEqual(sum(row["stage"][k]["bytes"] for k in ("payload", "receipt")),
                         len(PAYLOAD_BYTES) + len(self.receipt_bytes))

    def test_unknown_stage_file_and_hardlink_are_refused(self):
        (self.stage / "unknown").write_bytes(b"x")
        with self.assertRaisesRegex(Conflict, "partial or unknown"):
            self.inspect()
        (self.stage / "unknown").rename(self.data_root / "moved-unknown")
        os.link(self.stage / NAME, self.data_root / "linked-unique-source")
        with self.assertRaisesRegex(Conflict, "private regular"):
            self.inspect()

    def test_symlink_and_changed_digest_are_refused(self):
        (self.stage / NAME).rename(self.data_root / "moved-payload")
        (self.stage / NAME).symlink_to(self.data_root / "moved-payload")
        with self.assertRaises(Conflict):
            self.inspect()

    def test_unresolved_account_and_mapping_mismatch_are_refused(self):
        self.account["unresolved_attempts"] = 1
        with self.assertRaisesRegex(Conflict, "unresolved"):
            self.inspect(self.verifier)
        self.account["unresolved_attempts"] = 0
        self.mapping["payload_id"] = "wrong-id"
        with self.assertRaisesRegex(Conflict, "mapping generation or bytes"):
            self.inspect(self.verifier)

    def test_missing_gcs_or_wrong_drive_media_blocks(self):
        original = self.verifier.gcs_digest
        self.verifier.gcs_digest = lambda *a: {"bytes": 0, "sha256": "0" * 64}
        with self.assertRaisesRegex(Conflict, "Remote media digest"):
            self.inspect(self.verifier)
        self.verifier.gcs_digest = original
        self.verifier.drive_digest = lambda *a: {"bytes": 0, "sha256": "0" * 64}
        with self.assertRaisesRegex(Conflict, "Remote media digest"):
            self.inspect(self.verifier)

    def test_append_only_evidence_never_replaces_prior_line(self):
        path = self.data_root / "cache-review.jsonl"
        first = self.inspect()
        one = cache.append_evidence(path, first)
        two = cache.append_evidence(path, first)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(one, two)
        self.assertEqual(rows[0], rows[1])

    def test_release_is_hard_disabled_even_with_approval_shaped_artifact(self):
        row = self.inspect(self.verifier)
        approval = self.approval(row)
        remove = mock.Mock(side_effect=AssertionError("must never remove"))
        with self.assertRaisesRegex(Conflict, "disabled"):
            cache.release_after_approval(inventory=row, approval=approval,
                                         manifest_path=self.manifest_path,
                                         authority_db=self.data_root / "synthetic.sqlite",
                                         data_root=self.data_root,
                                         verifier=self.verifier, remove_file=remove)
        remove.assert_not_called()
        self.assertTrue((self.stage / NAME).exists())

    def approval(self, row):
        return {"schema": cache.APPROVAL_SCHEMA, "approved_by": "Josh",
                "approval_reference": "synthetic-only",
                "inventory_sha256": hashlib.sha256(_canonical(row)).hexdigest(),
                "run_id": RELEASE, "pair_key": self.key,
                "paths": [row["stage"][k]["path"] for k in ("payload", "receipt")]}

    def test_future_transaction_durable_intent_then_uncertainty_never_retries(self):
        before = self.assert_runner_disk_accepts_evidence()
        row = self.inspect(self.verifier)
        approval = self.approval(row)
        remove = mock.Mock(side_effect=OSError("synthetic unlink failure"))
        with mock.patch.object(cache, "LOCAL_CACHE_RELEASE_ENABLED", True):
            with self.assertRaisesRegex(OSError, "synthetic unlink failure"):
                cache.release_after_approval(inventory=row, approval=approval,
                                             manifest_path=self.manifest_path,
                                             authority_db=self.data_root / "synthetic.sqlite",
                                             data_root=self.data_root, verifier=self.verifier,
                                             approval_verifier=SyntheticApprovalVerifier(),
                                             remove_file=remove)
            log = cache._release_log(self.data_root, RELEASE)
            events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
            self.assertEqual(events, ["cache_release_start", "cache_release_file_intent",
                                      "cache_release_file_uncertain"])
            with self.assertRaisesRegex(Conflict, "prior intent"):
                cache.release_after_approval(inventory=row, approval=approval,
                                             manifest_path=self.manifest_path,
                                             authority_db=self.data_root / "synthetic.sqlite",
                                             data_root=self.data_root, verifier=self.verifier,
                                             approval_verifier=SyntheticApprovalVerifier(),
                                             remove_file=remove)
        remove.assert_called_once()
        self.assertEqual(remove.call_args.args, (NAME,))
        self.assertIn("dir_fd", remove.call_args.kwargs)
        self.assertTrue((self.stage / NAME).exists())
        self.assertTrue((self.stage / (NAME + ".receipt.json")).exists())
        self.assertEqual(self.assert_runner_disk_accepts_evidence(), before)

    def test_remote_callback_finishes_before_account_reservation_then_unlink_is_guarded(self):
        row = self.inspect(self.verifier)
        approval = self.approval(row)
        original = self.verifier.drive_digest
        preflight_writes_possible = []
        def competing_drive_read(file_id):
            contender = sqlite3.connect(self.db_path, isolation_level=None, timeout=0)
            try:
                contender.execute("BEGIN IMMEDIATE")
                contender.execute("ROLLBACK")
                preflight_writes_possible.append(True)
            finally:
                contender.close()
            return original(file_id)
        self.verifier.drive_digest = competing_drive_read
        blocked_at_unlink = []
        def refused_unlink(_name, *, dir_fd):
            contender = sqlite3.connect(self.db_path, isolation_level=None, timeout=0)
            try:
                with self.assertRaises(sqlite3.OperationalError):
                    contender.execute("BEGIN IMMEDIATE")
                blocked_at_unlink.append(True)
            finally:
                contender.close()
            raise OSError("synthetic unlink failure")
        remove = mock.Mock(side_effect=refused_unlink)
        with mock.patch.object(cache, "LOCAL_CACHE_RELEASE_ENABLED", True):
            with self.assertRaisesRegex(OSError, "synthetic unlink failure"):
                cache.release_after_approval(inventory=row, approval=approval,
                                             manifest_path=self.manifest_path,
                                             authority_db=self.db_path,
                                             data_root=self.data_root,
                                             verifier=self.verifier,
                                             approval_verifier=SyntheticApprovalVerifier(),
                                             remove_file=remove)
        self.assertEqual(preflight_writes_possible, [True, True])
        self.assertEqual(blocked_at_unlink, [True])
        self.assertTrue((self.stage / NAME).exists())

    def test_remote_callback_nonterminal_state_blocks_before_intent(self):
        row = self.inspect(self.verifier)
        approval = self.approval(row)
        original = self.verifier.drive_digest
        def observed_drift(file_id):
            self.account["nonterminal_run_id"] = "another-run"
            return original(file_id)
        self.verifier.drive_digest = observed_drift
        remove = mock.Mock(side_effect=AssertionError("must never remove"))
        with mock.patch.object(cache, "LOCAL_CACHE_RELEASE_ENABLED", True):
            with self.assertRaisesRegex(Conflict, "became nonterminal"):
                cache.release_after_approval(inventory=row, approval=approval,
                                             manifest_path=self.manifest_path,
                                             authority_db=self.db_path,
                                             data_root=self.data_root,
                                             verifier=self.verifier,
                                             approval_verifier=SyntheticApprovalVerifier(),
                                             remove_file=remove)
        remove.assert_not_called()
        self.assertFalse(cache._release_log(self.data_root, RELEASE).exists())

    def test_existing_account_write_blocks_release_before_intent(self):
        row = self.inspect(self.verifier)
        approval = self.approval(row)
        contender = sqlite3.connect(self.db_path, isolation_level=None, timeout=0)
        try:
            contender.execute("BEGIN IMMEDIATE")
            with mock.patch.object(cache, "LOCAL_CACHE_RELEASE_ENABLED", True):
                with self.assertRaisesRegex(Conflict, "writer reservation is unavailable"):
                    cache.release_after_approval(inventory=row, approval=approval,
                                                 manifest_path=self.manifest_path,
                                                 authority_db=self.db_path,
                                                 data_root=self.data_root,
                                                 verifier=self.verifier,
                                                 approval_verifier=SyntheticApprovalVerifier(),
                                                 remove_file=mock.Mock())
        finally:
            contender.execute("ROLLBACK")
            contender.close()
        self.assertFalse(cache._release_log(self.data_root, RELEASE).exists())

    def test_future_transaction_respects_existing_retained_lock(self):
        row = self.inspect(self.verifier)
        approval = self.approval(row)
        remove = mock.Mock(side_effect=AssertionError("must never remove"))
        fd = os.open(self.work / "overflow.lock", os.O_WRONLY | os.O_APPEND)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(cache, "LOCAL_CACHE_RELEASE_ENABLED", True):
                with self.assertRaisesRegex(Conflict, "owns the stage"):
                    cache.release_after_approval(inventory=row, approval=approval,
                                                 manifest_path=self.manifest_path,
                                                 authority_db=self.data_root / "synthetic.sqlite",
                                                 data_root=self.data_root, verifier=self.verifier,
                                                 approval_verifier=SyntheticApprovalVerifier(),
                                                 remove_file=remove)
        finally:
            os.close(fd)
        remove.assert_not_called()
        self.assertFalse(cache._release_log(self.data_root, RELEASE).exists())


if __name__ == "__main__":
    unittest.main()
