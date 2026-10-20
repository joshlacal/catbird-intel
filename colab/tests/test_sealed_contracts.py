"""Metadata-only regressions using the existing real smoke rows; no inference."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_pipeline.common import (
    create_review_sheet, load_sealed_export, read_rows, sha256,
    validate_input, validate_labeled, validate_pairs, validate_posts,
)

REPO = Path(__file__).resolve().parents[2]
BANK = Path(os.environ.get("CATBIRD_TEST_BANK", str(Path.home() / "catbird-lake-local/drive-sync-staging/training-banks/phase3_health_training_150k_v1_unlabeled.parquet")))
HOLDOUT = Path(os.environ.get("CATBIRD_TEST_HOLDOUT", str(REPO / "eval/phase3_holdout_600_unlabeled.jsonl")))
LABELED = Path(os.environ.get("CATBIRD_TEST_LABELED", str(Path.home() / "catbird-lake-local/real-model-smoke-v1/train-labeled.parquet")))


@unittest.skipUnless(BANK.exists() and HOLDOUT.exists() and LABELED.exists(), "Requires existing sealed real bank/holdout/smoke labels")
class SealedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_rows(BANK, limit=32)
        cls.seal, cls.held = load_sealed_export(HOLDOUT)
        cls.labeled = read_rows(LABELED)

    def test_canonical_inputs_pass(self):
        args = SimpleNamespace(input=BANK, holdout=HOLDOUT, export_manifest=None,
                               command="label", split="train", mode="smoke")
        validate_input(args, self.rows)
        validate_labeled(LABELED, self.labeled, self.seal, mode="smoke")

    def test_missing_float_bool_and_holdout_buckets_rejected(self):
        for value in (None, 0, 1.0, True, "1", 20):
            with self.subTest(value=value):
                row = dict(self.rows[0], author_split_bucket=value)
                with self.assertRaises(ValueError):
                    validate_posts([row])

    def test_uri_author_mismatch_rejected(self):
        row = dict(self.rows[0], author_did=self.rows[1]["author_did"])
        self.assertNotEqual(row["author_did"], self.rows[0]["author_did"])
        with self.assertRaises(ValueError):
            validate_posts([row])

    def test_empty_or_substituted_holdout_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "holdout.jsonl"
            for content in ("", json.dumps(self.held[0]) + "\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    load_sealed_export(path, HOLDOUT.parent / "phase3_v1_export_manifest.json")

    def test_modified_export_seal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "export.json"
            altered = copy.deepcopy(self.seal)
            altered["holdout"]["rows"] = 1
            path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                load_sealed_export(HOLDOUT, path)

    def test_labeled_source_and_teacher_selection_bound(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.parquet"
            shutil.copyfile(LABELED, path)
            original = json.loads(Path(str(LABELED) + ".manifest.json").read_text())
            for location in ("source", "teacher_source", "selection", "model"):
                manifest = copy.deepcopy(original)
                if location == "source":
                    manifest["source_sha256"] = "0" * 64
                elif location == "teacher_source":
                    manifest["teacher_provenance"][0]["identity"]["input_sha256"] = "0" * 64
                elif location == "selection":
                    manifest["teacher_provenance"][0]["identity"]["selected_uri_sha256"] = "0" * 64
                else:
                    manifest["teacher_provenance"][0]["identity"]["model"][1] = "0" * 40
                Path(str(path) + ".manifest.json").write_text(json.dumps(manifest))
                with self.subTest(location=location), self.assertRaises(ValueError):
                    validate_labeled(path, self.labeled, self.seal, mode="smoke")

    def test_review_rerun_preserves_human_fields_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "review.jsonl"
            create_review_sheet(path, self.rows)
            records = read_rows(path)
            records[0].update(human_politicalness=1, human_reviewed=True,
                              reviewer="existing human reviewer", review_notes="retained edit")
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            before = path.read_bytes()
            status = Path(str(path) + ".status.json").read_bytes()
            with self.assertRaises(FileExistsError):
                create_review_sheet(path, self.rows)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(Path(str(path) + ".status.json").read_bytes(), status)

    def test_pair_fields_and_manifest_counts_enforced(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        # Only the first three existing real posts are copied. Community IDs
        # here exercise structure, and are never a graph/training-quality claim.
        triplet = {"community_id": 1, "negative_community_id": 2}
        for prefix, row in zip(("anchor", "positive", "negative"), self.rows):
            for field in ("uri", "author_did", "text", "author_split_bucket"):
                triplet[f"{prefix}_{field}"] = row[field]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pairs.parquet"
            baseline = {"graph_version": "phase1-v1", "supplemental_edges_excluded": True,
                        "known_for_sha256": "1" * 64, "holdout_sha256": sha256(HOLDOUT),
                        "holdout_bucket_excluded": 0, "holdout_author_overlap": 0,
                        "source_author_split_buckets": {str(self.rows[0]["author_split_bucket"]): 3},
                        "source_rows": 3, "source_unique_uris": 3, "rows": 1}
            for defect in (None, "negative_did", "negative_bucket", "uri", "count", "source_histogram"):
                row, manifest = dict(triplet), copy.deepcopy(baseline)
                if defect == "negative_did":
                    row.pop("negative_author_did")
                elif defect == "negative_bucket":
                    row["negative_author_split_bucket"] = None
                elif defect == "uri":
                    row["negative_author_did"] = row["anchor_author_did"]
                elif defect == "count":
                    manifest["rows"] = 2
                elif defect == "source_histogram":
                    manifest["source_author_split_buckets"] = {"0": 3}
                pq.write_table(pa.Table.from_pylist([row]), path)
                manifest["sha256"] = sha256(path)
                path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
                with self.subTest(defect=defect):
                    if defect is None:
                        validate_pairs(path, HOLDOUT)
                    else:
                        with self.assertRaises(ValueError):
                            validate_pairs(path, HOLDOUT)


if __name__ == "__main__":
    unittest.main()
