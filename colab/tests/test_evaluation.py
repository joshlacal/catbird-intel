"""Synthetic numeric structural fixtures only; never model-quality evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_pipeline import evaluation


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.colab = Path(__file__).resolve().parents[1]
        self.contract = json.loads((self.colab / "evaluation_contract_v3.json").read_text())
        # Explicitly reduced resampling count for software tests. The evaluator
        # records this departure from the 10,000-resample qualification contract.
        self.contract["bootstrap"]["replicates"] = 200
        self.protocol = digest("synthetic fixture annotation protocol")
        self.development = digest("synthetic fixture development source")
        self.contract["scope"]["annotation_protocol_sha256"] = self.protocol
        self.contract["freeze"] = {"by": "fixture", "frozen_at": "2025-12-01T00:00:00Z",
                                   "record_sha256": digest("fixture freeze")}
        self.contract["profile"].update(frozen=True, frozen_at="2025-12-01T00:00:00Z",
                                         selection_record_sha256=digest("fixture predeclared politicalness-only profile"))
        self.contract["retrieval"]["required_for_overall_qualification"] = False
        self.contract["retrieval"]["k_values"] = [1, 2]
        for name, spec in self.contract["targets"].items():
            spec["enabled"] = name == "politicalness"
            if name in evaluation.HEADS:
                spec["definition_frozen"] = True
                spec["threshold"].update(value=0.5, frozen=True, frozen_at="2025-12-01T00:00:00Z",
                                         source_dataset_sha256=self.development)
        self.labels, self.reference, self.candidate = [], [], []
        for index in range(8):
            label = 1 - index % 2
            identity = {
                "id": f"fixture-item-{index}", "author_id": f"fixture-author-{index // 2}",
                "event_time": "2026-01-05T00:00:00Z", "role": "final",
                "content_sha256": digest(f"numeric fixture identity {index}"),
                "language": {"status": "supported", "code": "en", "source": "declared"}, "score_status": "scored",
            }
            self.labels.append({
                **identity, "human_labels": {head: label for head in evaluation.HEADS},
                "provenance": {"kind": "human", "reviewer_ids": ["fixture-annotator"], "adjudicated": True,
                               "blinded_to_predictions": True, "blinded_to_teacher_labels": True,
                               "annotation_protocol_sha256": self.protocol},
            })
            prediction = {**copy.deepcopy(identity), "probabilities": {head: 0.9 if label else 0.1 for head in evaluation.HEADS}}
            self.reference.append(prediction)
            self.candidate.append(copy.deepcopy(prediction))
        self.candidate[0]["probabilities"]["politicalness"] = 0.4
        self.candidate[3]["probabilities"]["politicalness"] = 0.6
        hardware = {"architecture": "x86_64", "instruction_set": ["fixture_avx2"], "cpu_model": "SYNTHETIC FIXTURE"}
        runtime = {"name": "SYNTHETIC FIXTURE", "version": "0", "actual_target_execution": False}
        identity = {
            key: digest(f"fixture {key}") for key in
            ("trained_checkpoint_sha256", "training_manifest_sha256", "artifact_sha256", "export_manifest_sha256",
             "tokenizer_sha256", "tokenizer_config_sha256", "preprocessor_sha256", "prediction_code_sha256",
             "input_artifact_sha256")
        }
        identity.update(
            trained=True, social_head_trained=True, output_head_order=list(evaluation.HEADS),
            quantization="fp32", created_at="2026-02-01T00:00:00Z",
            prediction_input_fields=["content", "declared_language"],
            labels_visible_to_predictor=False, teacher_labels_visible_to_predictor=False,
            hardware=hardware, hardware_sha256=evaluation.object_sha256(hardware),
            runtime=runtime, runtime_sha256=evaluation.object_sha256(runtime),
        )
        inventory = [{"id": f"fixture-excluded-{index}", "author_id": f"fixture-excluded-author-{index}",
                      "event_time": "2025-01-01T00:00:00Z", "role": role}
                     for index, role in enumerate(sorted(evaluation.INVENTORY_ROLES))]
        self.manifest = {
            "schema": "catbird-independent-evaluation-input-v3",
            "evidence_kind": "synthetic_structural_fixture",
            "evaluation_started_at": "2026-02-01T00:00:00Z",
            "final_data_first_examined_at": "2026-02-01T00:00:00Z",
            "event_window": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-31T23:59:59Z"},
            "annotation_protocol_sha256": self.protocol, "development_dataset_sha256": self.development,
            "unlabeled_input_sha256": identity["input_artifact_sha256"],
            "exclusion_inventory": inventory, "exclusion_inventory_sha256": evaluation.object_sha256(inventory),
            "inventory_complete_roles": sorted(evaluation.INVENTORY_ROLES),
            "sampling": {"design": "natural_probability", "frame": "SYNTHETIC STRUCTURAL FIXTURE",
                         "design_sha256": digest("fixture sampling"), "dependence_review_sha256": None},
            "population": {"total_rows": 8, "scored_rows": 8, "exclusions": {"unsupported": 0, "mixed": 0, "missing": 0}},
            "reference": identity, "candidate": copy.deepcopy(identity),
        }
        self.manifest["candidate"]["quantization"] = "int8_fixture"
        self.manifest["candidate"]["artifact_sha256"] = digest("fixture int8 artifact")

    def write(self):
        paths = {}
        for name, rows in (("labels", self.labels), ("reference", self.reference), ("candidate", self.candidate)):
            paths[name] = self.directory / f"{name}.jsonl"
            paths[name].write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.manifest["inputs"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
        paths["contract"] = self.directory / "contract.json"
        paths["contract"].write_text(json.dumps(self.contract))
        self.manifest["contract_sha256"] = hashlib.sha256(paths["contract"].read_bytes()).hexdigest()
        paths["manifest"] = self.directory / "manifest.json"
        paths["manifest"].write_text(json.dumps(self.manifest))
        return paths

    def evaluate(self):
        paths = self.write()
        return evaluation.evaluate_files(*(paths[name] for name in ("contract", "manifest", "labels", "reference", "candidate")))

    def add_retrieval(self):
        self.manifest["retrieval"] = {
            "corpus_ids": ["fixture-doc-0", "fixture-doc-1", "fixture-doc-2"],
            "corpus_sha256": digest("fixture corpus"), "relevance_protocol_sha256": digest("fixture relevance"),
            "search_policy_sha256": digest("fixture search"), "candidate_pool_sha256": digest("fixture pool"),
        }
        self.manifest["retrieval"]["corpus_ids_sha256"] = evaluation.object_sha256(self.manifest["retrieval"]["corpus_ids"])
        for name in ("reference", "candidate"):
            self.manifest[name]["retrieval_index_sha256"] = digest(f"fixture {name} index")
        for truth, ref, cand in zip(self.labels, self.reference, self.candidate):
            truth["relevant_ids"] = ["fixture-doc-0"]
            truth["provenance"]["relevance_source"] = "human"
            ref["embedding_256"] = [1.0] + [0.0] * 255
            cand["embedding_256"] = [1.0, 0.001] + [0.0] * 254
            ref["rankings"] = {str(dimension): ["fixture-doc-0", "fixture-doc-1"] for dimension in evaluation.DIMS}
            cand["rankings"] = {str(dimension): ["fixture-doc-1", "fixture-doc-0"] for dimension in evaluation.DIMS}

    def test_paired_metrics_and_diagnostics_are_computed_without_qualification(self):
        report = self.evaluate()
        target = report["health"]["politicalness"]
        self.assertEqual(target["reference"]["f1"], 1.0)
        self.assertEqual(target["candidate"]["f1"], 0.75)
        self.assertEqual(target["delta_candidate_minus_reference"]["f1"], -0.25)
        self.assertEqual(target["decision_flip_rate"], 0.25)
        self.assertEqual(target["diagnostic"]["max_absolute_probability_delta"], 0.5)
        self.assertEqual(target["diagnostic"]["acceptance_use"], "diagnostic_only")
        self.assertEqual(len(target["diagnostic"]["disagreement_ids"]), 2)
        self.assertEqual(report["author_clusters"]["unique"], 4)
        self.assertEqual(report["author_clusters"]["with_repeated_rows"], 4)
        self.assertEqual(report["status"], "not_evaluable")
        self.assertFalse(report["measured_model_evidence"])
        self.assertFalse(report["deployment_authorized"])
        self.assertFalse(report["historical_smoke_modified"])
        self.assertIn("synthetic_fixture_is_not_model_evidence", report["qualification_reasons"])
        self.assertIn("candidate_actual_target_execution_unverified", target["quantization_gate"]["reasons"])

    def test_cluster_bootstrap_retains_whole_authors_and_is_seeded(self):
        boot = {"replicates": 200, "seed": 7, "confidence": 0.95}
        arguments = ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1], ["a", "a", "b", "b"], boot)
        first = evaluation.paired_health_metrics(*arguments)
        self.assertEqual(first, evaluation.paired_health_metrics(*arguments))
        # Every resampled author carries one positive and one negative. Treating
        # rows as independent would permit zero-positive resamples instead.
        self.assertEqual(first["intervals"]["reference"]["recall"]["undefined_replicates"], 0)
        self.assertEqual(first["intervals"]["candidate"]["f1"]["lower"], 0)
        self.assertEqual(first["intervals"]["candidate"]["f1"]["upper"], 1)

    def test_one_sided_bounds_are_distinct_from_central_intervals(self):
        interval = evaluation._interval(list(range(101)), 0.95)
        self.assertAlmostEqual(interval["lower"], 2.5)
        self.assertAlmostEqual(interval["lower_one_sided"], 5)
        self.assertAlmostEqual(interval["upper_one_sided"], 95)
        self.assertAlmostEqual(interval["upper"], 97.5)

    def test_zero_flip_exact_upper_is_positive_at_2000_independent_authors(self):
        values = [0] * 2000
        bound = evaluation.decision_flip_upper_bound(
            values, values, [f"fixture-author-{index}" for index in range(2000)], 0.95,
            {"status": "computed", "upper_one_sided": 0},
        )
        self.assertEqual(bound["method"], "one_sided_exact_binomial_clopper_pearson")
        self.assertAlmostEqual(bound["upper"], 0.001496744895188, places=12)
        self.assertGreater(bound["upper"], 0)

    def test_repeated_author_zero_flip_boundary_cannot_pass_from_zero_bootstrap_upper(self):
        bound = evaluation.decision_flip_upper_bound(
            [0, 1, 0, 1], [0, 1, 0, 1], ["a", "a", "b", "b"], 0.95,
            {"status": "computed", "upper_one_sided": 0},
        )
        self.assertEqual(bound["status"], "not_evaluable")
        self.assertIsNone(bound["upper"])

    def test_exact_nonzero_binomial_bound_solves_upper_tail_equation(self):
        upper = evaluation.exact_binomial_upper(1, 10)
        self.assertAlmostEqual((1 - upper) ** 10 + 10 * upper * (1 - upper) ** 9, 0.05, places=12)

    def test_undefined_bootstrap_denominators_are_not_silently_dropped(self):
        result = evaluation.paired_health_metrics(
            [0, 0], [0, 0], [0, 0], ["a", "b"], {"replicates": 200, "seed": 1, "confidence": 0.95},
        )
        interval = result["intervals"]["candidate"]["f1"]
        self.assertEqual(interval["status"], "not_evaluable")
        self.assertEqual(interval["undefined_replicates"], 200)
        self.assertIsNone(interval["lower_one_sided"])

    def test_manifest_hash_mismatch_fails_before_metrics(self):
        paths = self.write()
        paths["candidate"].write_text(paths["candidate"].read_text() + "\n")
        with self.assertRaisesRegex(evaluation.EvaluationError, "pinned manifest hashes"):
            evaluation.evaluate_files(*(paths[name] for name in ("contract", "manifest", "labels", "reference", "candidate")))

    def test_contract_hash_mismatch_fails_before_metrics(self):
        paths = self.write()
        paths["contract"].write_text(paths["contract"].read_text() + "\n")
        with self.assertRaisesRegex(evaluation.EvaluationError, "contract file identity"):
            evaluation.evaluate_files(*(paths[name] for name in ("contract", "manifest", "labels", "reference", "candidate")))

    def test_row_identity_misalignment_is_rejected(self):
        self.candidate[0]["author_id"] = "different"
        with self.assertRaisesRegex(evaluation.EvaluationError, "misalignment"):
            self.evaluate()

    def test_reordered_predictions_are_rejected(self):
        self.candidate.reverse()
        with self.assertRaisesRegex(evaluation.EvaluationError, "misalignment"):
            self.evaluate()

    def test_teacher_targets_cannot_masquerade_as_human_labels(self):
        self.labels[0]["provenance"]["kind"] = "teacher"
        with self.assertRaisesRegex(evaluation.EvaluationError, "never teacher"):
            self.evaluate()

    def test_teacher_columns_are_rejected_even_with_human_provenance_flag(self):
        self.labels[0]["teacher_politicalness"] = 0.9
        with self.assertRaisesRegex(evaluation.EvaluationError, "teacher-derived"):
            self.evaluate()

    def test_missing_human_labels_are_rejected(self):
        self.labels[0]["human_labels"]["politicalness"] = None
        with self.assertRaisesRegex(evaluation.EvaluationError, "Missing, soft"):
            self.evaluate()

    def test_prediction_label_visibility_is_rejected(self):
        self.manifest["candidate"]["labels_visible_to_predictor"] = True
        with self.assertRaisesRegex(evaluation.EvaluationError, "not independent"):
            self.evaluate()

    def test_training_author_overlap_is_rejected(self):
        self.manifest["exclusion_inventory"][0]["author_id"] = self.labels[0]["author_id"]
        self.manifest["exclusion_inventory_sha256"] = evaluation.object_sha256(self.manifest["exclusion_inventory"])
        with self.assertRaisesRegex(evaluation.EvaluationError, "author leaks"):
            self.evaluate()

    def test_wrong_time_window_is_rejected(self):
        self.manifest["event_window"]["start"] = "2024-01-01T00:00:00Z"
        with self.assertRaisesRegex(evaluation.EvaluationError, "follow all"):
            self.evaluate()

    def test_frozen_threshold_may_not_use_final_data(self):
        self.contract["targets"]["politicalness"]["threshold"]["source_role"] = "final"
        with self.assertRaisesRegex(evaluation.EvaluationError, "selected on final"):
            self.evaluate()

    def test_contract_freeze_must_precede_final_examination(self):
        self.contract["freeze"]["frozen_at"] = "2026-02-02T00:00:00Z"
        with self.assertRaisesRegex(evaluation.EvaluationError, "frozen after"):
            self.evaluate()

    def test_required_target_selection_must_precede_final_examination(self):
        self.contract["profile"]["frozen_at"] = "2026-02-02T00:00:00Z"
        with self.assertRaisesRegex(evaluation.EvaluationError, "profile were selected"):
            self.evaluate()

    def test_politicalness_only_needs_no_other_human_labels_or_trained_social_head(self):
        for row in self.labels:
            row["human_labels"] = {"politicalness": row["human_labels"]["politicalness"]}
        self.contract["targets"]["outraged"]["enabled"] = True
        self.contract["targets"]["heat"]["enabled"] = True
        self.manifest["reference"]["social_head_trained"] = False
        self.manifest["candidate"]["social_head_trained"] = False
        report = self.evaluate()
        self.assertEqual(report["health"]["politicalness"]["status"], "computed")
        self.assertEqual(report["health"]["outraged"]["reasons"], ["optional_target_human_labels_not_supplied"])
        self.assertEqual(report["health"]["heat"]["status"], "not_evaluable")
        self.assertNotIn("social_head_not_trained", report["health"]["politicalness"]["quantization_gate"]["reasons"])
        self.assertEqual(report["profile"]["required_targets"], ["politicalness"])

    def test_joint_profile_requires_retrieval_and_cannot_silently_downgrade(self):
        self.contract["profile"]["name"] = "joint_health_retrieval"
        with self.assertRaisesRegex(evaluation.EvaluationError, "match the frozen"):
            self.evaluate()
        self.contract["retrieval"]["required_for_overall_qualification"] = True
        self.assertEqual(self.evaluate()["retrieval"]["status"], "not_evaluable")

    def test_unicode_line_separator_inside_json_string_is_not_a_record_boundary(self):
        path = self.directory / "unicode.jsonl"
        path.write_text(json.dumps({"id": "fixture\u2028identifier"}, ensure_ascii=False) + "\n")
        self.assertEqual(evaluation._rows(path), [{"id": "fixture\u2028identifier"}])

    def test_unresolved_threshold_returns_not_evaluable(self):
        self.contract["targets"]["politicalness"]["threshold"].update(value=None, frozen=False)
        target = self.evaluate()["health"]["politicalness"]
        self.assertEqual(target["status"], "not_evaluable")
        self.assertIn("threshold_value_unresolved", target["reasons"])
        self.assertEqual(target["diagnostic"]["max_absolute_probability_delta"], 0.5)

    def test_incomplete_contract_fails_closed(self):
        del self.contract["precision_plan"]
        with self.assertRaisesRegex(evaluation.EvaluationError, "Incomplete contract"):
            self.evaluate()

    def test_predeclared_count_gates_cannot_be_weakened(self):
        self.contract["precision_plan"]["minimum_eligible_final_rows"] = 20
        with self.assertRaisesRegex(evaluation.EvaluationError, "cannot be weakened"):
            self.evaluate()

    def test_predeclared_quantization_tolerances_cannot_be_weakened(self):
        original = copy.deepcopy(self.contract)
        changes = [
            ("targets", "politicalness", "maximum_f1_drop", 0.02),
            ("targets", "politicalness", "maximum_decision_flip_rate", 0.02),
            ("retrieval", "embedding_cosine_p1_minimum", "256", 0.98),
            ("retrieval", "maximum_recall_drop", "64", 0.02),
        ]
        for group, item, key, value in changes:
            with self.subTest(item=item, key=key):
                self.contract = copy.deepcopy(original)
                self.contract[group][item][key] = value
                with self.assertRaisesRegex(evaluation.EvaluationError, "cannot"):
                    self.evaluate()

    def test_unsupported_mixed_and_missing_languages_remain_explicit_denominators(self):
        for index, (status, code) in enumerate((("unsupported", "es"), ("mixed", "mixed"), ("missing", None))):
            identity = {
                "id": f"unscored-fixture-{index}", "author_id": f"unscored-author-{index}",
                "event_time": "2026-01-05T00:00:00Z", "role": "final", "content_sha256": digest(f"unscored {index}"),
                "language": {"status": status, "code": code, "source": "declared"}, "score_status": "unscored",
            }
            self.labels.append(identity)
            for rows in (self.reference, self.candidate):
                rows.append({**copy.deepcopy(identity), "probabilities": {head: None for head in evaluation.HEADS}})
        self.manifest["population"] = {"total_rows": 11, "scored_rows": 8,
                                       "exclusions": {"unsupported": 1, "mixed": 1, "missing": 1}}
        report = self.evaluate()
        self.assertEqual(report["population"], self.manifest["population"])
        self.assertEqual(report["health"]["politicalness"]["rows"], 8)

    def test_population_exclusion_counts_cannot_silently_drop_rows(self):
        self.manifest["population"]["total_rows"] = 7
        with self.assertRaisesRegex(evaluation.EvaluationError, "population/exclusion"):
            self.evaluate()

    def test_heat_has_no_implicit_aggregation(self):
        self.contract["targets"]["heat"]["enabled"] = True
        heat = self.evaluate()["health"]["heat"]
        self.assertEqual(heat["status"], "not_evaluable")
        self.assertIn("heat_aggregation_method_unresolved", heat["reasons"])

    def test_frozen_heat_requires_separate_human_label(self):
        self.contract["profile"]["required_targets"].append("heat")
        heat = self.contract["targets"]["heat"]
        heat.update(enabled=True, definition="Independent fixture instrument", definition_frozen=True)
        heat["score"].update(method="max", frozen=True, frozen_at="2025-12-01T00:00:00Z",
                             source_dataset_sha256=self.development)
        with self.assertRaisesRegex(evaluation.EvaluationError, "separate independent"):
            self.evaluate()

    def test_optional_frozen_heat_without_labels_does_not_block_politicalness_metrics(self):
        heat = self.contract["targets"]["heat"]
        heat.update(enabled=True, definition="Independent fixture instrument", definition_frozen=True)
        heat["score"].update(method="max", frozen=True, frozen_at="2025-12-01T00:00:00Z",
                             source_dataset_sha256=self.development)
        report = self.evaluate()
        self.assertEqual(report["health"]["heat"]["reasons"], ["optional_target_human_labels_not_supplied"])
        self.assertEqual(report["health"]["politicalness"]["status"], "computed")

    def test_missing_frozen_precision_review_blocks_qualification(self):
        report = self.evaluate()
        self.assertIn("frozen_precision_design_review_missing", report["qualification_reasons"])

    def test_natural_role_enforces_predeclared_four_week_window(self):
        self.contract["scope"]["evaluation_role"] = "natural"
        self.manifest["event_window"]["end"] = "2026-01-07T23:59:59Z"
        for rows in (self.labels, self.reference, self.candidate):
            for row in rows:
                row["role"] = "natural"
        report = self.evaluate()
        self.assertIn("natural_panel_shorter_than_predeclared_four_weeks", report["qualification_reasons"])

    def test_probability_sampling_without_equal_inclusion_is_descriptive_only(self):
        self.manifest["sampling"]["self_weighting"] = False
        report = self.evaluate()
        self.assertIn("unweighted_metrics_require_documented_equal_inclusion_sampling", report["qualification_reasons"])

    def test_supplied_retrieval_computes_each_dimension_and_recall_k(self):
        self.add_retrieval()
        retrieval = self.evaluate()["retrieval"]
        for dimension in evaluation.DIMS:
            self.assertGreater(retrieval["embeddings"][str(dimension)]["cosine_p1"], 0.99)
            recall = retrieval["recall"][str(dimension)]
            self.assertEqual(recall["1"]["reference"], 1)
            self.assertEqual(recall["1"]["candidate"], 0)
            self.assertEqual(recall["1"]["delta_candidate_minus_reference"], -1)
            self.assertEqual(recall["2"]["delta_candidate_minus_reference"], 0)
            self.assertEqual(recall["1"]["intervals"]["delta"]["lower_one_sided"], -1)

    def test_partial_retrieval_rows_are_rejected(self):
        self.add_retrieval()
        del self.candidate[0]["embedding_256"]
        with self.assertRaisesRegex(evaluation.EvaluationError, "aligned 256"):
            self.evaluate()

    def test_teacher_retrieval_relevance_is_rejected(self):
        self.add_retrieval()
        self.labels[0]["provenance"]["relevance_source"] = "teacher"
        with self.assertRaisesRegex(evaluation.EvaluationError, "independent human"):
            self.evaluate()

    def test_calibration_uses_frozen_bins_and_supplied_probabilities(self):
        metric = self.evaluate()["health"]["politicalness"]["calibration"]
        self.assertAlmostEqual(metric["brier"]["reference"], 0.01)
        self.assertGreater(metric["brier"]["candidate"], metric["brier"]["reference"])
        self.assertEqual(len(metric["candidate"]["bins"]), 10)
        self.assertEqual(sum(bucket["count"] for bucket in metric["candidate"]["bins"]), 8)

    def test_cli_writes_structural_receipt_and_refuses_overwrite(self):
        paths = self.write()
        output = self.directory / "report.json"
        arguments = [sys.executable, "-m", "real_pipeline.evaluation"]
        for name in ("contract", "manifest", "labels", "reference", "candidate"):
            arguments += [f"--{name}", str(paths[name])]
        arguments += ["--output", str(output)]
        result = subprocess.run(arguments, cwd=self.colab, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        receipt = json.loads(output.read_text())
        self.assertEqual(receipt["evidence_kind"], "synthetic_structural_fixture")
        self.assertEqual(receipt["operation"], "metrics_only_no_inference")
        before = output.read_bytes()
        repeated = subprocess.run(arguments, cwd=self.colab, text=True, capture_output=True)
        self.assertEqual(repeated.returncode, 3)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
