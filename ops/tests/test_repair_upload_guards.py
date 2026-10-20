"""Offline regression tests for quarantine, collision, and isolation guardrails.

Fixtures contain only file metadata, binary bytes, and author/URI identifiers.
There are no post texts, generated evaluation examples, teacher labels, or model
quality measurements. HTTP, OAuth construction, and upload methods are mocked.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

OPS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS))
import sync_lakehouse_and_training_banks as sync
import upload_repair_v1 as repair


def json_response(value: dict) -> io.BytesIO:
    return io.BytesIO(json.dumps(value).encode())


class OfflineCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="catbird-upload-guard-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        # No test may accidentally contact Drive or another network endpoint.
        self.http_patch = mock.patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("Unexpected network request in offline test"),
        )
        self.http = self.http_patch.start()
        self.addCleanup(self.http_patch.stop)

    def binary(self, name="artifact.bin", data=b"\x00\x01\x02\x03"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def uploader(self):
        # Bypass OAuth construction entirely.
        uploader = sync.DriveUploader.__new__(sync.DriveUploader)
        uploader._lock = threading.Lock()
        uploader._folder_cache = {}
        uploader.token = mock.Mock(return_value="offline-placeholder")
        return uploader

    def metadata(self, path, **overrides):
        return {
            "id": "existing-id",
            "name": path.name,
            "size": str(path.stat().st_size),
            "md5Checksum": hashlib.md5(path.read_bytes()).hexdigest(),
            **overrides,
        }


class DriveCollisionTests(OfflineCase):
    def test_identical_file_returns_existing_id_without_upload(self):
        path = self.binary()
        result = self.uploader().upload_file(
            path, "parent", {path.name: self.metadata(path)}
        )
        self.assertEqual(result["status"], "cached")
        self.assertEqual(result["id"], "existing-id")
        self.http.assert_not_called()

    def test_same_name_with_different_bytes_fails_before_upload(self):
        path = self.binary()
        with self.assertRaisesRegex(RuntimeError, "bytes differ"):
            self.uploader().upload_file(
                path, "parent", {path.name: self.metadata(path, md5Checksum="different")}
            )
        self.http.assert_not_called()

    def test_omitted_inventory_is_looked_up_before_upload(self):
        path = self.binary()
        uploader = self.uploader()
        uploader.list_existing = mock.Mock(return_value={path.name: self.metadata(path)})
        result = uploader.upload_file(path, "parent")
        self.assertEqual(result["id"], "existing-id")
        uploader.list_existing.assert_called_once_with("parent")
        self.http.assert_not_called()

    def test_completed_upload_updates_inventory_for_safe_replay(self):
        path = self.binary()
        uploader = self.uploader()
        inventory = {}
        initial = io.BytesIO()
        initial.headers = {"Location": "https://example.invalid/offline-upload"}
        self.http.side_effect = [initial, json_response(self.metadata(path))]
        first = uploader.upload_file(path, "parent", inventory)
        second = uploader.upload_file(path, "parent", inventory)
        self.assertEqual(first["status"], "uploaded")
        self.assertEqual(second["status"], "cached")
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(self.http.call_count, 2)
        self.assertEqual(inventory[path.name]["id"], "existing-id")

    def test_duplicate_names_across_pages_include_zero_byte_entries(self):
        uploader = self.uploader()
        self.http.side_effect = [
            json_response({"files": [{"id": "first", "name": "same", "size": "0"}],
                           "nextPageToken": "next"}),
            json_response({"files": [{"id": "second", "name": "same", "size": "4"}]}),
        ]
        with self.assertRaisesRegex(RuntimeError, "Duplicate Drive name"):
            uploader.list_existing("parent")
        self.assertEqual(self.http.call_count, 2)
        self.assertTrue(all(call.args[0].get_method() == "GET" for call in self.http.call_args_list))

    def test_duplicate_folder_names_fail_without_creating_folder(self):
        self.http.side_effect = None
        self.http.return_value = json_response(
            {"files": [{"id": "first", "name": "eval"}, {"id": "second", "name": "eval"}]}
        )
        with self.assertRaisesRegex(RuntimeError, "Ambiguous Drive folder"):
            self.uploader().ensure_subfolder("parent", "eval")
        self.assertEqual(self.http.call_count, 1)
        self.assertEqual(self.http.call_args.args[0].get_method(), "GET")


class RepairInputTests(OfflineCase):
    def assert_refused(self, path, message):
        drive = mock.MagicMock()
        with (
            mock.patch.object(repair, "DriveQuarantine", return_value=drive),
            mock.patch.object(sys, "argv", [
                "upload_repair_v1.py", "--destination", "root", "--file", str(path)
            ]),
            self.assertRaisesRegex(ValueError, message),
        ):
            repair.main()
        drive.children.assert_not_called()
        drive.up.upload_file.assert_not_called()
        self.http.assert_not_called()

    def test_old_training_banks_are_refused_outside_quarantine(self):
        for name in sorted(repair.BANK_NAMES):
            with self.subTest(name=name):
                self.assert_refused(
                    self.binary("training-banks/" + name), "invalidated artifact name"
                )

    def test_invalidated_model_and_notebook_names_are_refused(self):
        for name in ("phase5_benchmark_report.json", repair.NOTEBOOK_NAME):
            with self.subTest(name=name):
                self.assert_refused(self.binary(name), "invalidated artifact name")

    def test_quarantine_prefix_is_refused(self):
        self.assert_refused(
            self.binary("_quarantine_invalidated_20261019/artifact.bin"), "quarantined input"
        )

    def test_symlink_alias_cannot_bypass_quarantine(self):
        target = self.binary("_quarantine_invalidated_20261019/artifact.bin")
        alias = self.root / "seemingly-clean.bin"
        alias.symlink_to(target)
        self.assert_refused(alias, "quarantined input")


class BankIsolationTests(OfflineCase):
    def fixture(self, *, buckets=(1, 2), duplicate_uri=False, held_author="held-metadata-id",
                teacher_column=False, omit_baseline=False, receipt_rows=2):
        # Identifier-only Parquet: intentionally no text/post-content column.
        columns = {
            "author_did": ["train-metadata-id-1", "train-metadata-id-2"],
            "uri": ["urn:metadata:1", "urn:metadata:1" if duplicate_uri else "urn:metadata:2"],
            "author_split_bucket": list(buckets),
            **{"regex_baseline_" + field: [0.0, 0.0]
               for field in ("politicalness", "heat", "engagement_bait", "spam")},
        }
        if teacher_column:
            columns["teacher_forbidden_alias"] = [None, None]
        if omit_baseline:
            columns.pop("regex_baseline_heat")
        bank = self.root / "bank.parquet"
        pq.write_table(pa.table(columns), bank)
        digest = hashlib.sha256(bank.read_bytes()).hexdigest()
        bank.with_suffix(".manifest.json").write_text(
            json.dumps({"sha256": digest, "rows": receipt_rows})
        )
        eval_dir = self.root / "eval"
        eval_dir.mkdir()
        holdout = eval_dir / "phase3_holdout_600_unlabeled.jsonl"
        holdout.write_text(json.dumps({"author_did": held_author}) + "\n")
        (eval_dir / "phase3_v1_export_manifest.json").write_text(json.dumps({
            "bank": {"sha256": digest},
            "holdout": {"sha256": hashlib.sha256(holdout.read_bytes()).hexdigest()},
        }))
        return bank, holdout

    def validate(self, bank):
        # Point the validator at temporary metadata receipts, not the real holdout.
        with mock.patch.object(sync, "__file__", str(self.root / "ops" / "sync.py")):
            sync.validate_health_bank(bank)
        self.http.assert_not_called()

    def test_sealed_disjoint_metadata_passes(self):
        bank, _ = self.fixture()
        self.validate(bank)

    def test_changed_binary_fails_receipt_check(self):
        bank, _ = self.fixture()
        with bank.open("ab") as out:
            out.write(b"\x00")
        with self.assertRaisesRegex(RuntimeError, "differs from its export receipt"):
            self.validate(bank)

    def test_reserved_bucket_is_refused(self):
        bank, _ = self.fixture(buckets=(1, 0))
        with self.assertRaisesRegex(RuntimeError, "Reserved holdout author bucket"):
            self.validate(bank)

    def test_duplicate_uri_is_refused(self):
        bank, _ = self.fixture(duplicate_uri=True)
        with self.assertRaisesRegex(RuntimeError, "Duplicate post URI"):
            self.validate(bank)

    def test_teacher_alias_is_refused(self):
        bank, _ = self.fixture(teacher_column=True)
        with self.assertRaisesRegex(RuntimeError, "forbidden teacher aliases"):
            self.validate(bank)

    def test_missing_baseline_column_is_refused(self):
        bank, _ = self.fixture(omit_baseline=True)
        with self.assertRaisesRegex(RuntimeError, "missing explicit regex baseline"):
            self.validate(bank)

    def test_receipt_row_count_mismatch_is_refused(self):
        bank, _ = self.fixture(receipt_rows=3)
        with self.assertRaisesRegex(RuntimeError, "row count differs"):
            self.validate(bank)

    def test_holdout_author_overlap_is_refused(self):
        bank, _ = self.fixture(held_author="train-metadata-id-1")
        with self.assertRaisesRegex(RuntimeError, "Holdout authors overlap"):
            self.validate(bank)

    def test_holdout_binary_change_is_refused(self):
        bank, holdout = self.fixture()
        with holdout.open("ab") as out:
            out.write(b"\n")
        with self.assertRaisesRegex(RuntimeError, "Holdout differs from its sealed receipt"):
            self.validate(bank)

    def test_different_audited_generation_is_refused(self):
        bank, _ = self.fixture()
        path = self.root / "eval/phase3_v1_export_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["bank"]["sha256"] = "different-generation"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "not the holdout's audited training generation"):
            self.validate(bank)


if __name__ == "__main__":
    unittest.main()
