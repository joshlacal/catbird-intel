"""Structural contracts only; numeric fixtures never represent model inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_pipeline.common import (
    create_review_sheet, read_rows, sha256, teacher_identity, validate_score_row, validate_teacher_identity,
)
from real_pipeline.contracts import HEALTH, MODELS, STUDENT_CONTRACT, model_identity
from real_pipeline.checkpoints import atomic_write
from real_pipeline.language import LANGUAGE_CONTRACT, language_histogram, language_scope

COLAB = Path(__file__).resolve().parents[1]
BANK = Path.home() / "catbird-lake-local/drive-sync-staging/training-banks/phase3_health_training_150k_v1_unlabeled.parquet"


class RevisionThreeContracts(unittest.TestCase):
    def test_atomic_metadata_failure_preserves_existing_and_partial_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "manifest.json"
            output.write_bytes(b"previous manifest")
            with patch("real_pipeline.checkpoints.os.replace", side_effect=OSError("interrupted rename")):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    atomic_write(output, b"new manifest")
            self.assertEqual(output.read_bytes(), b"previous manifest")
            partial = list(Path(tmp).glob(".*.partial"))
            self.assertEqual(len(partial), 1)
            self.assertEqual(partial[0].read_bytes(), b"new manifest")
            with self.assertRaises(FileExistsError):
                atomic_write(output, b"cannot replace shard", overwrite=False)

    def test_fallback_is_pinned_c_variant_and_licenses_are_distinct(self):
        identity = model_identity("nli")
        self.assertEqual(MODELS["nli"], ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c",
                                       "b2730f16019076bb0009481121efbe4705e0e378"))
        self.assertEqual(identity["model_license"], "mit")
        self.assertEqual(identity["commercial_use_clearance"], "not_asserted")
        self.assertEqual(len(identity["training_data_licenses"]["datasets"]), 3)
        self.assertIsNone(model_identity("political")["training_data_licenses"]["datasets"][0]["declared_license"])
        self.assertIn("transformers", identity["runtime_code_licenses"])
        self.assertIsNone(identity["catbird_training_input_license"]["spdx"])

    def test_language_scope_is_explicit_and_conservative(self):
        for langs in (["en"], ["EN"], ["en-US"], ["en", "en-GB"]):
            self.assertTrue(language_scope({"langs": langs})["in_scope"])
        for langs, reason in ((None, "missing_language_metadata"), ([], "missing_language_metadata"),
                              (["fr"], "unsupported_language"), (["en", "fr"], "mixed_languages"),
                              ("en", "invalid_language_metadata"), (["en", None], "invalid_language_metadata"),
                              ([" en"], "invalid_language_metadata")):
            self.assertEqual(language_scope({"langs": langs}), {"in_scope": False, "reason": reason})
        self.assertEqual(LANGUAGE_CONTRACT["unscored_value"], None)

    def test_histogram_distinguishes_missing_multitag_and_multilingual(self):
        # Language metadata fixtures contain no invented post text or labels.
        result = language_histogram([{"langs": ["en", "en-US"]}, {"langs": ["en", "fr"]},
                                     {"langs": []}, {}, {"langs": ["fr"]}, {"langs": "en"}])
        self.assertEqual(result["eligible_declared_english_rows"], 1)
        self.assertEqual(result["multiple_declared_tags_rows"], 2)
        self.assertEqual(result["multiple_primary_languages_rows"], 1)
        self.assertEqual(result["missing_or_null_rows"], 1)
        self.assertEqual(result["empty_list_rows"], 1)
        self.assertEqual(result["invalid_metadata_rows"], 1)
        self.assertEqual(sum(result["unscored_reasons"].values()), 5)

    def test_unscored_values_cannot_be_numeric_zero(self):
        source = {"langs": []}
        record = {"score_status": "unscored", "unscored_reason": "missing_language_metadata",
                  "teacher_politicalness": None, "teacher_politicalness_logits": None}
        validate_score_row(record, source, "political")
        for bad in (dict(record, teacher_politicalness=0), dict(record, score_status="scored"),
                    dict(record, teacher_politicalness_logits=[0, 0])):
            with self.assertRaises(ValueError):
                validate_score_row(bad, source, "political")

    def test_five_component_contract_has_no_implicit_heat_or_hostility(self):
        self.assertEqual(HEALTH, ("politicalness", "outraged", "critical", "engagement_bait", "spam"))
        self.assertEqual(STUDENT_CONTRACT["health_output"], "health_component_logits")
        self.assertNotIn("heat", STUDENT_CONTRACT["health_output_order"])
        self.assertNotIn("hostility", STUDENT_CONTRACT["health_output_order"])

    def test_head_structure_without_pretrained_loading_or_forward(self):
        import torch
        from real_pipeline.student import Student
        # Construct only local Linear layers around an empty encoder fixture.
        # No model download, forward call, optimizer or inference occurs.
        with patch("real_pipeline.student.AutoConfig.from_pretrained",
                   return_value=SimpleNamespace(hidden_size=8)), \
             patch("real_pipeline.student.AutoModel.from_pretrained", return_value=torch.nn.Identity()) as loader:
            student = Student()
        self.assertEqual(tuple(student.health_component_logits.weight.shape), (5, 8))
        self.assertEqual(loader.call_args.kwargs["revision"], MODELS["student"][1])
        legacy = {"health_logits.weight": torch.zeros(4, 8), "health_logits.bias": torch.zeros(4)}
        with self.assertRaises(RuntimeError):
            student.load_state_dict(legacy)

    @unittest.skipUnless(BANK.exists(), "Requires existing sealed real bank")
    def test_model_identity_rejects_legacy_fallback_and_missing_license_scope(self):
        rows = read_rows(BANK, limit=32)
        identity = teacher_identity("nli", "a" * 64, rows, "smoke", 512)
        validate_teacher_identity(identity, "nli", "a" * 64, rows, "smoke")
        for defect in ("legacy_model", "licenses", "scope", "software"):
            changed = copy.deepcopy(identity)
            if defect == "legacy_model":
                changed["model"][0] = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
            elif defect == "licenses":
                changed.pop("model_provenance")
            elif defect == "scope":
                changed["language_contract"] = {}
            else:
                changed["software_identity"]["implementation_source_sha256"] = {}
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate_teacher_identity(changed, "nli", "a" * 64, rows, "smoke")

    @unittest.skipUnless(BANK.exists(), "Requires existing sealed real bank")
    def test_scoped_dispatch_keeps_all_real_rows_and_skips_unscored(self):
        from real_pipeline.teachers import score_scoped_rows
        rows = read_rows(BANK, limit=32)
        expected = [row for row in rows if language_scope(row)["in_scope"]]
        args = SimpleNamespace(teacher="political", batch_size=2, max_length=512, device="cpu")
        # Fixture-only return values verify dispatch, never teacher quality.
        fake = [{"teacher_politicalness": 0.5, "teacher_politicalness_logits": [0.0, 0.0]} for _ in expected]
        with patch("real_pipeline.teachers.nli_scores", return_value=fake) as infer:
            output = score_scoped_rows(args, rows)
        self.assertEqual([r["uri"] for r in infer.call_args.args[0]], [r["uri"] for r in expected])
        self.assertEqual(len(output), 32)
        self.assertEqual(sum(row["score_status"] == "scored" for row in output), len(expected))
        for row, original in zip(output, rows):
            validate_score_row(row, original, "political")
        with patch("real_pipeline.teachers.nli_scores") as infer:
            only_unscored = [row for row in rows if not language_scope(row)["in_scope"]]
            score_scoped_rows(args, only_unscored)
            infer.assert_not_called()

    @unittest.skipUnless(BANK.exists(), "Requires existing sealed real bank")
    def test_orphan_shards_never_acquire_current_teacher_provenance(self):
        from real_pipeline.teachers import label_stage
        rows = read_rows(BANK, limit=32)
        holdout = COLAB.parent / "eval/phase3_holdout_600_unlabeled.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(command="label", input=BANK, holdout=holdout,
                                   export_manifest=None, split="train", mode="smoke",
                                   teacher="political", max_length=512, output=Path(tmp))
            shard = Path(tmp) / "part-0000000.jsonl"
            # Existing bytes must not receive a newly asserted model identity.
            shard.write_text(json.dumps({"uri": rows[0]["uri"], "fixture": "unowned provenance"}) + "\n")
            before = shard.read_bytes()
            with patch("real_pipeline.teachers.score_scoped_rows") as infer:
                with self.assertRaisesRegex(ValueError, "nonempty without provenance"):
                    label_stage(args, rows)
                self.assertFalse((Path(tmp) / "manifest.json").exists())
                identity = teacher_identity("political", sha256(BANK), rows, "smoke", 512)
                (Path(tmp) / "manifest.json").write_text(json.dumps({"identity": identity, "files": {}}))
                with self.assertRaisesRegex(ValueError, "Unmanifested"):
                    label_stage(args, rows)
                infer.assert_not_called()
            self.assertEqual(shard.read_bytes(), before)

    @unittest.skipUnless(BANK.exists(), "Requires existing sealed real bank")
    def test_teacher_partial_write_requires_preservation_before_resume(self):
        from real_pipeline.teachers import label_stage
        rows = read_rows(BANK, limit=32)
        holdout = COLAB.parent / "eval/phase3_holdout_600_unlabeled.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(command="label", input=BANK, holdout=holdout,
                                   export_manifest=None, split="train", mode="smoke",
                                   teacher="political", max_length=512, output=Path(tmp))
            identity = teacher_identity("political", sha256(BANK), rows, "smoke", 512)
            (Path(tmp) / "manifest.json").write_text(json.dumps({"identity": identity, "files": {}}))
            partial = Path(tmp) / ".part-0000000.jsonl.interrupted.partial"
            partial.write_bytes(b"partial inference artifact")
            with patch("real_pipeline.teachers.score_scoped_rows") as infer:
                with self.assertRaisesRegex(ValueError, "Interrupted teacher writes"):
                    label_stage(args, rows)
                infer.assert_not_called()
            self.assertEqual(partial.read_bytes(), b"partial inference artifact")

    @unittest.skipUnless(BANK.exists(), "Requires existing sealed real bank")
    def test_review_first_pass_hides_both_teacher_and_regex_values(self):
        source = dict(read_rows(BANK, limit=1)[0], teacher_politicalness=0.7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blind.jsonl"
            create_review_sheet(path, [source])
            row = json.loads(path.read_text())
            self.assertFalse(any(key.startswith(("teacher_", "regex_baseline_")) for key in row))
            self.assertIsNone(row["human_politicalness"])
            self.assertEqual(row["text"], source["text"])
            self.assertTrue(json.loads(Path(str(path) + ".status.json").read_text())["blind_first_pass"])

    @unittest.skipUnless(BANK.exists(), "Requires existing sealed real bank")
    def test_sealed_language_histogram_matches_actual_metadata(self):
        import pyarrow.parquet as pq
        report = json.loads((COLAB / "language_audit_v3.json").read_text())
        current = language_histogram(pq.read_table(BANK, columns=["langs"]).to_pylist())
        self.assertEqual(report["histogram"], current)
        self.assertEqual(current["rows"], 150000)
        self.assertEqual(current["eligible_declared_english_rows"], 90603)
        self.assertEqual(current["unscored_rows"], 59397)


if __name__ == "__main__":
    unittest.main()
