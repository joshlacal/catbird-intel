"""Independent metrics-only evaluation; no model loading or inference.

Run from ``colab/``::

    python -m real_pipeline.evaluation --contract evaluation_contract_v3.json \
      --manifest run-identity.json --labels human-labels.jsonl \
      --reference fp32-predictions.jsonl --candidate int8-predictions.jsonl \
      --output evaluation-report.json

The manifest schema is ``catbird-independent-evaluation-input-v3``. It contains
``inputs`` (labels/reference/candidate file SHA256s), ``contract_sha256``, ``evidence_kind`` (either
real_external_predictions or synthetic_structural_fixture), ``evaluation_started_at``,
``final_data_first_examined_at``, ``event_window`` (start/end ISO UTC timestamps), ``annotation_protocol_sha256``,
``development_dataset_sha256``, ``unlabeled_input_sha256``, ``exclusion_inventory``
(id/author_id/event_time/role objects for every training/development/calibration/
model-selection row), ``exclusion_inventory_sha256`` (canonical object_sha256),
``inventory_complete_roles`` (training, development, threshold_calibration,
model_selection), ``sampling`` (design, frame, design_sha256, dependence_review_sha256,
self_weighting and a common inclusion_probability for unweighted population claims),
``population`` (total_rows/scored_rows/exclusions counts), and reference/candidate.

Each model identity contains trained_checkpoint_sha256, training_manifest_sha256,
artifact_sha256, export_manifest_sha256, tokenizer_sha256, tokenizer_config_sha256,
preprocessor_sha256, prediction_code_sha256, input_artifact_sha256, trained=true,
social_head_trained, output_head_order, quantization, created_at,
prediction_input_fields=["content","declared_language"], labels_visible_to_predictor=false,
teacher_labels_visible_to_predictor=false, hardware and runtime objects with their
canonical hardware_sha256/runtime_sha256. Those hashes are recorded attestations;
this executor does not download or inspect model weights or hardware.

All three JSONL files have the same ordered identities:
id/author_id/event_time/role/content_sha256/language/score_status.
Language is {"status":"supported|unsupported|mixed|missing","code":"en|...|null",
"source":"declared"}. Only supported declared English is scored. Every other row
must have score_status="unscored" and null probabilities and embeddings/rankings.
Labels add human_labels (binary integers for the profile's required targets;
optional targets are computed only when labeled; heat requires its own direct
human label) and provenance {kind:"human", reviewer_ids:[...],
adjudicated:true, blinded_to_predictions:true, blinded_to_teacher_labels:true,
annotation_protocol_sha256:...}. Independent human relevance may additionally
use relevant_ids and provenance.relevance_source="human".

Predictions add probabilities keyed in the five-head contract, optional
embedding_256 (256 finite numbers; smaller prefixes are re-normalized here), and
optional rankings {"256":[document IDs], "128":[...], "64":[...]}.
Retrieval inputs require manifest.retrieval with corpus_ids/corpus_ids_sha256,
corpus_sha256, relevance_protocol_sha256, search_policy_sha256 and shared
candidate_pool_sha256. Each identity also records retrieval_index_sha256.

The frozen profile is health_only or joint_health_retrieval, with required_targets
chosen before final-data examination. Politicalness-only evaluation needs no other
target labels or trained social head. Optional diagnostics cannot replace required
gates. A natural-role panel must span at least 28 days. Frozen sampling and precision
review hashes bind the chosen design; unresolved design evidence cannot qualify.
External annotation_agreement receipts and regression_review receipts are checked
and reported, not recomputed from unavailable raw agreement/slice data. The tests
contain explicitly synthetic structural input examples, never benchmark evidence.

Contract freezing never substitutes for input checks. Unresolved definitions,
thresholds, precision design, or tolerances produce not_evaluable. Even complete
synthetic fixtures cannot produce a qualification pass. Percentile intervals are
paired by author cluster, marginal per metric, and do not solve cross-author or
time dependence. Enriched samples receive descriptive sample-pool metrics only.
CLI exits: 0 passes predeclared provisional gates (not deployment authorization),
1 fails provisional gates, 2 not_evaluable, 3 invalid input. Existing output files
are never replaced. Historical smoke receipts and gates are never read or changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Any


HEADS = ("politicalness", "outraged", "critical", "engagement_bait", "spam")
DIMS = (256, 128, 64)
INVENTORY_ROLES = {"training", "development", "threshold_calibration", "model_selection"}
IDENTITY_FIELDS = ("id", "author_id", "event_time", "role", "content_sha256", "language", "score_status")
SHA = re.compile(r"[0-9a-f]{64}")


class EvaluationError(ValueError):
    """Malformed, misaligned or leaking evidence cannot be evaluated."""


def object_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _keys(value: Any, required: set[str], context: str) -> None:
    _require(isinstance(value, dict) and required <= value.keys(), f"Incomplete {context}: requires {sorted(required)}")


def _digest(value: Any, context: str) -> str:
    _require(isinstance(value, str) and SHA.fullmatch(value) is not None, f"Invalid SHA256: {context}")
    return value


def _text(value: Any, context: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"Missing {context}")
    return value


def _time(value: Any, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as error:
        raise EvaluationError(f"Invalid timestamp: {context}") from error
    _require(parsed.tzinfo is not None, f"Timestamp needs timezone: {context}")
    return parsed


def _number(value: Any, context: str, low: float = 0, high: float = 1) -> float:
    _require(type(value) in (float, int) and math.isfinite(value) and low <= value <= high,
             f"Invalid numeric value: {context}")
    return float(value)


def _json(text: str) -> Any:
    def reject_constant(value):
        raise EvaluationError(f"Non-finite JSON constant: {value}")
    try:
        return json.loads(text, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as error:
        raise EvaluationError(f"Invalid JSON: {error}") from error


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for number, line in enumerate(lines, 1):
        _require(bool(line.strip()), f"Blank JSONL row in {path.name}:{number}")
        row = _json(line)
        _require(isinstance(row, dict), f"Expected row object in {path.name}:{number}")
        rows.append(row)
    _require(bool(rows), f"Empty input: {path.name}")
    return rows


def _validate_contract(contract: dict[str, Any]) -> None:
    _keys(contract, {"schema", "status", "comparison_tolerances_status", "data_quality_certified",
                     "freeze", "profile", "scope", "bootstrap", "precision_plan", "annotation_agreement",
                     "calibration", "runtime_qualification", "targets", "retrieval"}, "contract")
    _require(contract["schema"] == "catbird-independent-evaluation-contract-v3", "Wrong evaluation contract schema")
    _require(contract["status"] in {"provisional", "approved"}, "Unknown contract status")
    _require(contract["comparison_tolerances_status"] in {"unresolved", "predeclared_provisional", "approved"}, "Unknown tolerance status")
    _require(type(contract["data_quality_certified"]) is bool, "data_quality_certified must be boolean")
    _keys(contract["freeze"], {"by", "frozen_at", "record_sha256"}, "contract freeze")
    profile = contract["profile"]
    _keys(profile, {"name", "required_targets", "frozen", "frozen_at", "selection_record_sha256"}, "evaluation profile")
    _require(profile["name"] in {"health_only", "joint_health_retrieval"}, "Unknown evaluation profile")
    _require(type(profile["frozen"]) is bool, "Invalid evaluation-profile freeze flag")
    required = profile["required_targets"]
    _require(isinstance(required, list) and bool(required) and all(isinstance(name, str) for name in required)
             and len(set(required)) == len(required) and set(required) <= set(HEADS) | {"heat"},
             "Evaluation profile requires an explicit nonempty target list")
    scope = contract["scope"]
    _keys(scope, {"language_contract", "evaluation_role", "output_head_order", "annotation_protocol_sha256",
                  "sampling_design_sha256", "natural_panel_minimum_days",
                  "independent_authors_required", "future_time_window_required",
                  "natural_probability_sampling_required_for_qualification",
                  "same_hardware_runtime_required_for_quantization"}, "scope")
    _require(scope["language_contract"] == "declared-English-only", "Unsupported language contract")
    _require(scope["output_head_order"] == list(HEADS), "Expected five component outputs; no trained heat output")
    _require(scope["evaluation_role"] in {"final", "natural"}, "Evaluation role must be final or natural")
    _require(scope["independent_authors_required"] is True and scope["future_time_window_required"] is True,
             "Author independence and future-time separation cannot be disabled")
    _require(scope["natural_probability_sampling_required_for_qualification"] is True,
             "This evaluator cannot qualify enriched samples as population evidence")
    _require(type(scope["same_hardware_runtime_required_for_quantization"]) is bool, "Invalid runtime policy")
    if scope["annotation_protocol_sha256"] is not None:
        _digest(scope["annotation_protocol_sha256"], "contract annotation protocol")
    if scope["sampling_design_sha256"] is not None:
        _digest(scope["sampling_design_sha256"], "frozen sampling design")
    _require(type(scope["natural_panel_minimum_days"]) is int and scope["natural_panel_minimum_days"] >= 28,
             "Natural panel requires at least four weeks")
    boot = contract["bootstrap"]
    _keys(boot, {"method", "replicates", "seed", "confidence"}, "bootstrap")
    _require(boot["method"] == "paired_author_cluster_percentile", "Unsupported bootstrap method")
    _require(type(boot["replicates"]) is int and 200 <= boot["replicates"] <= 100000, "Bootstrap replicates must be 200..100000")
    _require(type(boot["seed"]) is int, "Bootstrap seed must be an integer")
    _number(boot["confidence"], "bootstrap confidence", 0.8, 0.999)
    precision = contract["precision_plan"]
    _keys(precision, {"status", "minimum_author_clusters", "minimum_eligible_final_rows",
                      "minimum_positives_per_target", "minimum_positives_per_claimed_slice",
                      "maximum_candidate_f1_interval_width", "power_or_precision_review_sha256"}, "precision plan")
    _require(precision["status"] in {"unresolved", "predeclared_provisional", "approved"}, "Unknown precision-plan status")
    for key in ("minimum_author_clusters", "minimum_positives_per_target"):
        if precision[key] is not None:
            _require(type(precision[key]) is int and precision[key] >= (2 if "author" in key else 1), f"Invalid {key}")
    if precision["maximum_candidate_f1_interval_width"] is not None:
        _number(precision["maximum_candidate_f1_interval_width"], "maximum F1 interval width")
    for key, minimum in (("minimum_eligible_final_rows", 2000), ("minimum_positives_per_target", 400),
                         ("minimum_positives_per_claimed_slice", 150)):
        _require(type(precision[key]) is int and precision[key] >= minimum, f"Predeclared design gate cannot be weakened: {key}")
    agreement = contract["annotation_agreement"]
    _keys(agreement, {"minimum_double_labeled_rows_per_target", "minimum_krippendorff_alpha"}, "annotation agreement")
    _require(type(agreement["minimum_double_labeled_rows_per_target"]) is int
             and agreement["minimum_double_labeled_rows_per_target"] >= 200, "Double-label design needs at least 200 per target")
    _number(agreement["minimum_krippendorff_alpha"], "agreement alpha gate", 0.8, 1)
    _keys(contract["calibration"], {"bins", "maximum_paired_brier_increase"}, "calibration")
    _require(type(contract["calibration"]["bins"]) is int and 2 <= contract["calibration"]["bins"] <= 100, "Invalid calibration bins")
    if contract["calibration"]["maximum_paired_brier_increase"] is not None:
        _number(contract["calibration"]["maximum_paired_brier_increase"], "Brier regression tolerance")
    _keys(contract["runtime_qualification"], {"architecture", "actual_target_execution_required",
          "hardware_instruction_set_required", "latency_and_peak_rss_receipt_required"}, "runtime qualification")
    _require(contract["runtime_qualification"]["architecture"] == "x86_64"
             and all(contract["runtime_qualification"][key] is True for key in
                     ("actual_target_execution_required", "hardware_instruction_set_required",
                      "latency_and_peak_rss_receipt_required")), "Actual x86 runtime evidence cannot be disabled")
    targets = contract["targets"]
    _require(isinstance(targets, dict) and set(targets) == set(HEADS) | {"heat"}, "Contract must specify all five heads and optional heat")
    for name, spec in targets.items():
        _keys(spec, {"enabled", "human_label", "definition", "definition_frozen", "score", "threshold",
                     "minimum_f1_lower_bound", "maximum_f1_drop", "maximum_decision_flip_rate"}, f"target {name}")
        _require(type(spec["enabled"]) is bool and type(spec["definition_frozen"]) is bool, f"Invalid flags for {name}")
        _require(spec["human_label"] == name, f"Target label substitution is not allowed: {name}")
        _keys(spec["threshold"], {"value", "frozen", "frozen_at", "source_role", "source_dataset_sha256"}, f"{name} threshold")
        _require(type(spec["threshold"]["frozen"]) is bool, f"Invalid threshold flag: {name}")
        for key in ("minimum_f1_lower_bound", "maximum_f1_drop", "maximum_decision_flip_rate"):
            if spec[key] is not None:
                _number(spec[key], f"{name}.{key}")
                if key.startswith("maximum_"):
                    _require(spec[key] <= 0.01, f"Predeclared {key} cannot exceed 0.01")
        floor = {"politicalness": 0.90, "heat": 0.75}.get(name)
        if floor is not None:
            _require(spec["minimum_f1_lower_bound"] is not None and spec["minimum_f1_lower_bound"] >= floor,
                     f"Predeclared {name} F1 lower-bound target cannot be weakened")
        if name in HEADS:
            _require(spec["score"] == {"kind": "head", "head": name}, f"Incorrect score identity: {name}")
        else:
            _keys(spec["score"], {"kind", "inputs", "method", "frozen", "frozen_at", "source_role",
                                  "source_dataset_sha256"}, "heat aggregation")
            _require(spec["score"]["kind"] == "aggregation" and spec["score"]["inputs"] == ["outraged", "critical"],
                     "Heat requires explicit aggregation of outraged and critical")
            _require(spec["score"]["method"] in {None, "max", "mean"}, "Supported explicit heat aggregations are max or mean")
            _require(type(spec["score"]["frozen"]) is bool, "Invalid heat frozen flag")
    _require(all(targets[name]["enabled"] for name in required), "A required target cannot be disabled")
    retrieval = contract["retrieval"]
    _keys(retrieval, {"required_for_overall_qualification", "dimensions", "k_values",
                     "embedding_cosine_p1_minimum", "maximum_recall_drop"}, "retrieval")
    _require(type(retrieval["required_for_overall_qualification"]) is bool, "Invalid retrieval requirement")
    _require(retrieval["required_for_overall_qualification"] == (profile["name"] == "joint_health_retrieval"),
             "Retrieval requirement must match the frozen evaluation profile")
    _require(retrieval["dimensions"] == list(DIMS), "All Matryoshka dimensions must be declared")
    ks = retrieval["k_values"]
    _require(isinstance(ks, list) and bool(ks) and len(set(ks)) == len(ks)
             and all(type(k) is int and k > 0 for k in ks), "Invalid Recall@K values")
    for key in ("embedding_cosine_p1_minimum", "maximum_recall_drop"):
        _require(isinstance(retrieval[key], dict) and set(retrieval[key]) == {str(d) for d in DIMS}, f"Incomplete {key}")
        for value in retrieval[key].values():
            if value is not None:
                _number(value, key, -1 if "cosine" in key else 0, 1)
                _require(value >= 0.99 if "cosine" in key else value <= 0.01,
                         f"Predeclared {key} cannot be weakened")


def _validate_model(identity: Any, manifest: dict[str, Any], context: str) -> None:
    hashes = {"trained_checkpoint_sha256", "training_manifest_sha256", "artifact_sha256", "export_manifest_sha256",
              "tokenizer_sha256", "tokenizer_config_sha256", "preprocessor_sha256", "prediction_code_sha256",
              "input_artifact_sha256", "hardware_sha256", "runtime_sha256"}
    _keys(identity, hashes | {"trained", "social_head_trained", "output_head_order", "quantization", "created_at",
                              "prediction_input_fields", "labels_visible_to_predictor",
                              "teacher_labels_visible_to_predictor", "hardware", "runtime"}, context)
    for key in hashes:
        _digest(identity[key], f"{context}.{key}")
    _require(identity["trained"] is True, f"{context} must identify a trained checkpoint")
    _require(type(identity["social_head_trained"]) is bool, f"{context} social-head identity is missing")
    _require(identity["output_head_order"] == list(HEADS), f"{context} head order differs")
    _text(identity["quantization"], f"{context} quantization")
    _require(_time(identity["created_at"], context) >= _time(manifest["evaluation_started_at"], "evaluation start"),
             "Prediction timestamp predates evaluation start")
    _require(identity["prediction_input_fields"] == ["content", "declared_language"], "Predictor input fields may contain target leakage")
    _require(identity["labels_visible_to_predictor"] is False and identity["teacher_labels_visible_to_predictor"] is False,
             "Predictions exposed to labels are not independent evidence")
    _require(identity["input_artifact_sha256"] == manifest["unlabeled_input_sha256"], "Prediction input identity differs")
    for key in ("hardware", "runtime"):
        _require(isinstance(identity[key], dict) and bool(identity[key]), f"Missing {context} {key}")
        _require(object_sha256(identity[key]) == identity[f"{key}_sha256"], f"Incorrect {context} {key} digest")


def _validate_inputs(contract, manifest, labels, reference, candidate):
    _keys(manifest, {"schema", "inputs", "contract_sha256", "evidence_kind", "evaluation_started_at",
                     "final_data_first_examined_at", "event_window",
                     "annotation_protocol_sha256", "development_dataset_sha256", "unlabeled_input_sha256",
                     "exclusion_inventory", "exclusion_inventory_sha256", "inventory_complete_roles",
                     "sampling", "population", "reference", "candidate"}, "input manifest")
    _require(manifest["schema"] == "catbird-independent-evaluation-input-v3", "Wrong input-manifest schema")
    _require(manifest["evidence_kind"] in {"real_external_predictions", "synthetic_structural_fixture"}, "Unknown evidence kind")
    for key in ("annotation_protocol_sha256", "development_dataset_sha256", "unlabeled_input_sha256", "exclusion_inventory_sha256"):
        _digest(manifest[key], key)
    _time(manifest["evaluation_started_at"], "evaluation start")
    _time(manifest["final_data_first_examined_at"], "final data first examined")
    _keys(manifest["event_window"], {"start", "end"}, "event window")
    start, end = (_time(manifest["event_window"][key], key) for key in ("start", "end"))
    _require(start <= end, "Inverted event window")
    inventory = manifest["exclusion_inventory"]
    _require(isinstance(inventory, list) and bool(inventory), "Complete nonempty exclusion inventory is required")
    _require(object_sha256(inventory) == manifest["exclusion_inventory_sha256"], "Exclusion inventory digest differs")
    _require(isinstance(manifest["inventory_complete_roles"], list)
             and set(manifest["inventory_complete_roles"]) == INVENTORY_ROLES, "Exclusion inventory role coverage is incomplete")
    excluded_ids, excluded_authors, times = set(), set(), []
    for row in inventory:
        _keys(row, {"id", "author_id", "event_time", "role"}, "exclusion row")
        excluded_ids.add(_text(row["id"], "excluded ID"))
        excluded_authors.add(_text(row["author_id"], "excluded author"))
        _require(row["role"] in INVENTORY_ROLES, "Unknown exclusion role")
        times.append(_time(row["event_time"], "excluded event"))
    _require(start > max(times), "Evaluation window must follow all training/development/calibration/model-selection events")
    scope = contract["scope"]
    if scope["annotation_protocol_sha256"] is not None:
        _require(scope["annotation_protocol_sha256"] == manifest["annotation_protocol_sha256"], "Human annotation protocol differs from frozen contract")
    for key in ("reference", "candidate"):
        _validate_model(manifest[key], manifest, key)
    for key in ("trained_checkpoint_sha256", "training_manifest_sha256", "tokenizer_sha256",
                "tokenizer_config_sha256", "preprocessor_sha256", "output_head_order", "social_head_trained"):
        _require(manifest["reference"][key] == manifest["candidate"][key], f"Candidate and reference differ in {key}")
    _require(manifest["reference"]["quantization"] == "fp32", "Reference quantization must be fp32")
    _keys(manifest["sampling"], {"design", "frame", "design_sha256", "dependence_review_sha256"}, "sampling design")
    _require(manifest["sampling"]["design"] in {"natural_probability", "enriched", "convenience"}, "Unknown sampling design")
    _text(manifest["sampling"]["frame"], "sampling frame")
    _digest(manifest["sampling"]["design_sha256"], "sampling design")
    if scope["sampling_design_sha256"] is not None:
        _require(scope["sampling_design_sha256"] == manifest["sampling"]["design_sha256"], "Frozen sampling design differs")
    if manifest["sampling"].get("inclusion_probability") is not None:
        _number(manifest["sampling"]["inclusion_probability"], "inclusion probability", 1e-100, 1)
    if manifest["sampling"]["dependence_review_sha256"] is not None:
        _digest(manifest["sampling"]["dependence_review_sha256"], "dependence review")
    _require(len(labels) == len(reference) == len(candidate), "Row counts differ")
    scored, ids, exclusions = [], set(), Counter({"unsupported": 0, "mixed": 0, "missing": 0})
    allowed_provenance = {"kind", "reviewer_ids", "adjudicated", "blinded_to_predictions",
                          "blinded_to_teacher_labels", "annotation_protocol_sha256", "relevance_source"}
    for index, (truth, ref, cand) in enumerate(zip(labels, reference, candidate)):
        for row in (truth, ref, cand):
            _keys(row, set(IDENTITY_FIELDS), "row identity")
        identity = {key: truth[key] for key in IDENTITY_FIELDS}
        _require(all({key: row[key] for key in IDENTITY_FIELDS} == identity for row in (ref, cand)),
                 f"Row identity misalignment at position {index}")
        row_id = _text(truth["id"], "row ID")
        author = _text(truth["author_id"], "row author")
        _require(row_id not in ids, "Duplicate evaluation row ID")
        ids.add(row_id)
        _require(row_id not in excluded_ids and author not in excluded_authors, "Evaluation row or author leaks from development/training")
        _require(start <= _time(truth["event_time"], "evaluation row") <= end, "Row is outside declared evaluation window")
        _require(truth["role"] == scope["evaluation_role"], "Evaluation row has wrong role")
        _digest(truth["content_sha256"], "row content")
        language = truth["language"]
        _keys(language, {"status", "code", "source"}, "language metadata")
        _require(language["source"] == "declared" and language["status"] in {"supported", "unsupported", "mixed", "missing"},
                 "Unsupported language metadata")
        supported = language["status"] == "supported"
        if supported:
            _require(language["code"] == "en" and truth["score_status"] == "scored", "Only declared English can be scored")
        else:
            _require(truth["score_status"] == "unscored" and language["code"] != "en", "Unsupported language must be unscored")
            if language["status"] == "missing":
                _require(language["code"] is None, "Missing language must have null code")
            exclusions[language["status"]] += 1
        _require(set(truth) <= set(IDENTITY_FIELDS) | {"human_labels", "provenance", "relevant_ids"},
                 "Labels contain undeclared fields or teacher-derived targets")
        for prediction in (ref, cand):
            _require(set(prediction) <= set(IDENTITY_FIELDS) | {"probabilities", "embedding_256", "rankings"},
                     "Predictions contain undeclared fields or target leakage")
            probabilities = prediction.get("probabilities")
            _require(isinstance(probabilities, dict) and set(probabilities) == set(HEADS), "Prediction output heads differ")
            for value in probabilities.values():
                if supported:
                    _number(value, "prediction probability")
                else:
                    _require(value is None, "Unscored language rows must retain null probabilities")
            if not supported:
                _require(prediction.get("embedding_256") is None and prediction.get("rankings") is None,
                         "Unscored language rows cannot carry scored embeddings or rankings")
        if not supported:
            continue
        _keys(truth, {"human_labels", "provenance"}, "human labels")
        human = truth["human_labels"]
        required_labels = set(contract["profile"]["required_targets"]) & set(HEADS)
        _require(isinstance(human, dict) and required_labels <= human.keys() and set(human) <= set(HEADS) | {"heat"},
                 "Every scored row needs independent human labels for all required targets")
        _require(all(type(value) is int and value in (0, 1) for value in human.values()),
                 "Missing, soft or nonbinary human labels cannot be used as ground truth")
        provenance = truth["provenance"]
        _keys(provenance, allowed_provenance - {"relevance_source"}, "human provenance")
        _require(set(provenance) <= allowed_provenance, "Undeclared human-label provenance")
        _require(provenance["kind"] == "human" and provenance["adjudicated"] is True
                 and provenance["blinded_to_predictions"] is True and provenance["blinded_to_teacher_labels"] is True,
                 "Ground truth must be independent adjudicated human labels, never teacher targets")
        _require(isinstance(provenance["reviewer_ids"], list) and bool(provenance["reviewer_ids"])
                 and all(isinstance(value, str) and value.strip() for value in provenance["reviewer_ids"]), "Missing human reviewers")
        _require(provenance["annotation_protocol_sha256"] == manifest["annotation_protocol_sha256"], "Row annotation protocol differs")
        scored.append(index)
    expected_population = {"total_rows": len(labels), "scored_rows": len(scored), "exclusions": dict(exclusions)}
    _require(manifest["population"] == expected_population, "Manifest population/exclusion counts differ from supplied rows")
    return scored


def percentile(values: list[float], probability: float) -> float:
    """Linear interpolation between sorted observations at (n - 1) * p."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _interval(values, confidence):
    undefined = sum(value is None for value in values)
    if undefined or not values:
        return {"lower": None, "upper": None, "lower_one_sided": None, "upper_one_sided": None,
                "confidence": confidence, "undefined_replicates": undefined, "status": "not_evaluable"}
    alpha = (1 - confidence) / 2
    return {"lower": percentile(values, alpha), "upper": percentile(values, 1 - alpha),
            "lower_one_sided": percentile(values, 1 - confidence), "upper_one_sided": percentile(values, confidence),
            "confidence": confidence,
            "undefined_replicates": 0, "status": "computed"}


def binary_metrics(counts):
    tp, fp, fn, tn = counts
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def exact_binomial_upper(successes, trials, confidence=0.95):
    """One-sided Clopper-Pearson upper bound via binomial-CDF inversion."""
    _require(type(trials) is int and trials > 0 and type(successes) is int and 0 <= successes <= trials,
             "Invalid binomial counts")
    _number(confidence, "binomial confidence", 0.8, 0.999)
    if successes == trials:
        return 1.0
    alpha = 1 - confidence
    if successes == 0:
        return -math.expm1(math.log(alpha) / trials)

    def cdf(probability):
        term = trials * math.log1p(-probability)
        terms = [term]
        odds = math.log(probability) - math.log1p(-probability)
        for count in range(successes):
            term += math.log(trials - count) - math.log(count + 1) + odds
            terms.append(term)
        maximum = max(terms)
        return math.exp(maximum) * sum(math.exp(value - maximum) for value in terms)

    low, high = 0.0, 1.0
    for _ in range(70):
        middle = (low + high) / 2
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def decision_flip_upper_bound(reference, candidate, authors, confidence, bootstrap_interval):
    flips = sum(left != right for left, right in zip(reference, candidate))
    rows = len(reference)
    independent_rows = len(set(authors)) == rows
    result = {"rows": rows, "observed_flips": flips, "confidence": confidence,
              "one_row_per_author": independent_rows, "cross_author_independence_requires_design_review": True}
    if independent_rows and rows:
        result.update(status="computed", method="one_sided_exact_binomial_clopper_pearson",
                      upper=exact_binomial_upper(flips, rows, confidence))
    elif flips in (0, rows):
        result.update(status="not_evaluable", method="unsupported_cluster_boundary",
                      upper=None, reason="Repeated-author boundary sample has no supported conservative population bound")
    else:
        result.update(status=bootstrap_interval["status"], method="paired_author_cluster_percentile",
                      upper=bootstrap_interval["upper_one_sided"])
    return result


def paired_health_metrics(labels, reference, candidate, authors, bootstrap):
    """Binary pooled metrics with paired author-cluster percentile intervals."""
    groups = defaultdict(lambda: [0] * 10)
    for label, ref, cand, author in zip(labels, reference, candidate, authors):
        group = groups[author]
        for offset, prediction in ((0, ref), (4, cand)):
            position = 0 if label and prediction else 1 if prediction else 2 if label else 3
            group[offset + position] += 1
        group[8] += ref != cand
        group[9] += 1
    values = list(groups.values())

    def measure(total):
        ref, cand = binary_metrics(total[:4]), binary_metrics(total[4:8])
        delta = {key: cand[key] - ref[key] if ref[key] is not None and cand[key] is not None else None
                 for key in ("precision", "recall", "f1")}
        return ref, cand, delta, total[8] / total[9] if total[9] else None

    total = [sum(group[column] for group in values) for column in range(10)]
    ref, cand, delta, flip = measure(total)
    draws = {group: {key: [] for key in ("precision", "recall", "f1")} for group in ("reference", "candidate", "delta")}
    flips = []
    rng = random.Random(bootstrap["seed"])
    for _ in range(bootstrap["replicates"]):
        selected = rng.choices(values, k=len(values))
        aggregate = [sum(group[column] for group in selected) for column in range(10)]
        sampled = measure(aggregate)
        for group, result in zip(draws, sampled[:3]):
            for key in draws[group]:
                draws[group][key].append(result[key])
        flips.append(sampled[3])
    flip_interval = _interval(flips, bootstrap["confidence"])
    return {
        "reference": ref, "candidate": cand, "delta_candidate_minus_reference": delta,
        "decision_flip_rate": flip,
        "intervals": {group: {key: _interval(samples, bootstrap["confidence"]) for key, samples in metrics.items()}
                      for group, metrics in draws.items()},
        "decision_flip_interval": flip_interval,
        "decision_flip_acceptance_bound": decision_flip_upper_bound(reference, candidate, authors,
                                                                    bootstrap["confidence"], flip_interval),
    }


def _paired_mean(reference, candidate, authors, bootstrap):
    groups = defaultdict(lambda: [0.0, 0.0, 0])
    for ref, cand, author in zip(reference, candidate, authors):
        groups[author][0] += ref
        groups[author][1] += cand
        groups[author][2] += 1
    values = list(groups.values())
    draws = {"reference": [], "candidate": [], "delta": []}
    rng = random.Random(bootstrap["seed"])
    for _ in range(bootstrap["replicates"]):
        selected = rng.choices(values, k=len(values))
        size = sum(group[2] for group in selected)
        ref = sum(group[0] for group in selected) / size
        cand = sum(group[1] for group in selected) / size
        for key, value in zip(draws, (ref, cand, cand - ref)):
            draws[key].append(value)
    ref, cand = sum(reference) / len(reference), sum(candidate) / len(candidate)
    return {"reference": ref, "candidate": cand, "delta_candidate_minus_reference": cand - ref,
            "intervals": {key: _interval(value, bootstrap["confidence"]) for key, value in draws.items()}}


def _runtime_reasons(manifest):
    reasons = []
    for name in ("reference", "candidate"):
        identity = manifest[name]
        hardware, runtime = identity["hardware"], identity["runtime"]
        if hardware.get("architecture") != "x86_64":
            reasons.append(f"{name}_actual_x86_target_missing")
        instructions = hardware.get("instruction_set")
        if not isinstance(instructions, list) or not instructions or not all(isinstance(value, str) and value for value in instructions):
            reasons.append(f"{name}_instruction_set_missing")
        if runtime.get("actual_target_execution") is not True:
            reasons.append(f"{name}_actual_target_execution_unverified")
        performance = identity.get("performance")
        if performance is None:
            reasons.append(f"{name}_latency_cpu_rss_receipt_missing")
            continue
        _keys(performance, {"artifact_sha256", "measurement_protocol_sha256", "warm_latency_ms_p50",
                           "warm_latency_ms_p95", "peak_rss_bytes", "cpu_model", "warmup_iterations",
                           "measurement_iterations"}, f"{name} performance receipt")
        for key in ("artifact_sha256", "measurement_protocol_sha256"):
            _digest(performance[key], key)
        for key in ("warm_latency_ms_p50", "warm_latency_ms_p95"):
            _number(performance[key], key, 0, 1e12)
        _require(performance["warm_latency_ms_p50"] <= performance["warm_latency_ms_p95"], "Latency quantiles are inverted")
        for key in ("peak_rss_bytes", "warmup_iterations", "measurement_iterations"):
            _require(type(performance[key]) is int and performance[key] > 0, f"Invalid {key}")
        _text(performance["cpu_model"], "CPU model")
    return reasons


def _agreement_reasons(name, contract, manifest, truth):
    """Validate an externally computed pre-adjudication agreement receipt."""
    receipts = manifest.get("annotation_agreement", {})
    _require(isinstance(receipts, dict), "Invalid annotation-agreement receipts")
    receipt = receipts.get(name)
    if receipt is None:
        return ["independent_pre_adjudication_agreement_missing"]
    _keys(receipt, {"method", "pre_adjudication", "sample_role", "sample_ids", "sample_author_ids",
                    "artifact_sha256", "annotation_protocol_sha256", "double_labeled_rows", "krippendorff_alpha",
                    "raw_agreement", "class_counts", "missing_labels", "insufficient_labels",
                    "confidence_interval", "rubric_independent_annotator"}, f"{name} agreement receipt")
    _require(receipt["method"] == "krippendorff_nominal" and receipt["pre_adjudication"] is True
             and receipt["sample_role"] == "agreement", "Agreement must use independent pre-adjudication ratings")
    _digest(receipt["artifact_sha256"], "agreement receipt")
    _require(receipt["annotation_protocol_sha256"] == manifest["annotation_protocol_sha256"], "Agreement protocol differs")
    size = receipt["double_labeled_rows"]
    _require(type(size) is int and size >= 0, "Invalid agreement sample count")
    sample_ids, sample_authors = receipt["sample_ids"], receipt["sample_author_ids"]
    _require(isinstance(sample_ids, list) and isinstance(sample_authors, list)
             and len(sample_ids) == len(sample_authors) == size
             and all(isinstance(value, str) and value for value in sample_ids + sample_authors)
             and len(set(sample_ids)) == size, "Agreement sample identities are incomplete")
    _require(not (set(sample_ids) & {row["id"] for row in truth})
             and not (set(sample_authors) & {row["author_id"] for row in truth}),
             "Agreement sample overlaps final evaluation")
    alpha = _number(receipt["krippendorff_alpha"], "Krippendorff alpha", -1e6, 1)
    _number(receipt["raw_agreement"], "raw agreement")
    _keys(receipt["class_counts"], {"positive", "negative"}, "agreement class counts")
    for value in list(receipt["class_counts"].values()) + [receipt["missing_labels"], receipt["insufficient_labels"]]:
        _require(type(value) is int and value >= 0, "Invalid agreement/missing-label counts")
    interval = receipt["confidence_interval"]
    _keys(interval, {"lower", "upper", "confidence", "method"}, "agreement uncertainty")
    _number(interval["lower"], "agreement lower", -1e6, 1)
    _number(interval["upper"], "agreement upper", -1e6, 1)
    _number(interval["confidence"], "agreement confidence", 0.8, 0.999)
    _text(interval["method"], "agreement interval method")
    _require(interval["lower"] <= alpha <= interval["upper"], "Agreement interval does not contain the reported estimate")
    reasons = []
    if size < contract["annotation_agreement"]["minimum_double_labeled_rows_per_target"]:
        reasons.append("insufficient_double_labeled_agreement_sample")
    if alpha < contract["annotation_agreement"]["minimum_krippendorff_alpha"]:
        reasons.append("annotation_agreement_below_0_80_requires_revision")
    if receipt["rubric_independent_annotator"] is not True:
        reasons.append("rubric_independent_annotator_missing")
    return reasons


def _regression_reasons(name, contract, manifest):
    """Record external slice/calibration review rather than inventing policy."""
    receipt = manifest.get("regression_review")
    if receipt is None:
        return ["independent_calibration_and_slice_review_missing"], False
    _keys(receipt, {"artifact_sha256", "policy_sha256", "policy_frozen_at", "predeclared_slice_ids",
                   "all_predeclared_slices_covered", "targets"}, "regression review")
    for key in ("artifact_sha256", "policy_sha256"):
        _digest(receipt[key], key)
    _require(_time(receipt["policy_frozen_at"], "regression policy") <= min(
        _time(manifest["evaluation_started_at"], "evaluation start"),
        _time(manifest["final_data_first_examined_at"], "first examination")), "Regression policy was selected on final data")
    slices = receipt["predeclared_slice_ids"]
    _require(isinstance(slices, list) and bool(slices) and all(isinstance(value, str) and value for value in slices)
             and len(set(slices)) == len(slices), "Predeclared type slices are missing")
    _require(isinstance(receipt["targets"], dict), "Regression targets are incomplete")
    target = receipt["targets"].get(name)
    if target is None:
        return ["target_calibration_and_slice_review_missing"], False
    _keys(target, {"calibration_no_disqualifying_regression", "slices"}, "target regression review")
    _require(type(target["calibration_no_disqualifying_regression"]) is bool, "Invalid calibration-review result")
    _require(isinstance(target["slices"], dict) and set(target["slices"]) == set(slices), "Slice-review identity differs")
    reasons, passed = [], target["calibration_no_disqualifying_regression"]
    if receipt["all_predeclared_slices_covered"] is not True:
        reasons.append("predeclared_slice_coverage_incomplete")
    for name, item in target["slices"].items():
        _keys(item, {"positive_rows", "no_disqualifying_regression", "metric_receipt_sha256"}, f"slice {name}")
        _require(type(item["positive_rows"]) is int and item["positive_rows"] >= 0, "Invalid slice positive count")
        _require(type(item["no_disqualifying_regression"]) is bool, "Invalid slice-review result")
        _digest(item["metric_receipt_sha256"], "slice metric receipt")
        if item["positive_rows"] < contract["precision_plan"]["minimum_positives_per_claimed_slice"]:
            reasons.append(f"slice_{name}_fewer_than_150_positives_descriptive_only")
        passed = passed and item["no_disqualifying_regression"]
    return reasons, passed


def _calibration(labels, probabilities, bins):
    buckets = [{"count": 0, "sum_probability": 0.0, "positives": 0} for _ in range(bins)]
    for label, probability in zip(labels, probabilities):
        bucket = buckets[min(int(probability * bins), bins - 1)]
        bucket["count"] += 1
        bucket["sum_probability"] += probability
        bucket["positives"] += label
    ece = 0.0
    for index, bucket in enumerate(buckets):
        size = bucket["count"]
        bucket.update(lower=index / bins, upper=(index + 1) / bins,
                      mean_probability=bucket["sum_probability"] / size if size else None,
                      observed_positive_rate=bucket["positives"] / size if size else None)
        if size:
            ece += size / len(labels) * abs(bucket["mean_probability"] - bucket["observed_positive_rate"])
    return {"expected_calibration_error": ece, "bins": buckets}


def _base_reasons(contract, manifest, authors):
    reasons = []
    freeze = contract["freeze"]
    for key in ("by", "frozen_at", "record_sha256"):
        if freeze[key] is None:
            reasons.append(f"contract_freeze_{key}_unresolved")
        elif key == "record_sha256":
            _digest(freeze[key], key)
        elif key == "frozen_at":
            _require(_time(freeze[key], key) <= min(_time(manifest["evaluation_started_at"], "evaluation start"),
                         _time(manifest["final_data_first_examined_at"], "first examination")),
                     "Contract was frozen after evaluation or final-data examination began")
        else:
            _text(freeze[key], key)
    profile = contract["profile"]
    if not profile["frozen"]:
        reasons.append("evaluation_profile_not_frozen")
    else:
        _digest(profile["selection_record_sha256"], "profile selection")
        _require(_time(profile["frozen_at"], "profile freeze") <= min(
            _time(manifest["evaluation_started_at"], "evaluation start"),
            _time(manifest["final_data_first_examined_at"], "first examination")),
            "Required targets/profile were selected after final-data examination")
    if contract["scope"]["annotation_protocol_sha256"] is None:
        reasons.append("annotation_protocol_not_frozen")
    if contract["scope"]["sampling_design_sha256"] is None:
        reasons.append("sampling_design_not_frozen")
    if manifest["evidence_kind"] == "synthetic_structural_fixture":
        reasons.append("synthetic_fixture_is_not_model_evidence")
    if manifest["sampling"]["design"] != "natural_probability":
        reasons.append("sample_pool_metrics_do_not_estimate_natural_population")
    elif manifest["sampling"].get("self_weighting") is not True or manifest["sampling"].get("inclusion_probability") is None:
        reasons.append("unweighted_metrics_require_documented_equal_inclusion_sampling")
    if contract["scope"]["evaluation_role"] == "natural":
        duration = (_time(manifest["event_window"]["end"], "window end") -
                    _time(manifest["event_window"]["start"], "window start")).total_seconds() / 86400
        if duration < contract["scope"]["natural_panel_minimum_days"]:
            reasons.append("natural_panel_shorter_than_predeclared_four_weeks")
    if manifest["sampling"]["dependence_review_sha256"] is None:
        reasons.append("cross_author_and_time_dependence_unreviewed")
    precision = contract["precision_plan"]
    if precision["status"] == "unresolved":
        reasons.append("precision_design_unresolved")
    if precision["power_or_precision_review_sha256"] is not None:
        _digest(precision["power_or_precision_review_sha256"], "precision design review")
    else:
        reasons.append("frozen_precision_design_review_missing")
    if precision["minimum_author_clusters"] is not None and len(set(authors)) < precision["minimum_author_clusters"]:
        reasons.append("insufficient_independent_author_clusters")
    if len(authors) < precision["minimum_eligible_final_rows"]:
        reasons.append("fewer_than_2000_predeclared_eligible_final_rows")
    if contract["bootstrap"]["replicates"] != 10000 or contract["bootstrap"]["confidence"] != 0.95:
        reasons.append("bootstrap_differs_from_predeclared_10000_resamples_95_percent")
    if len(set(authors)) < 2:
        reasons.append("fewer_than_two_author_clusters")
    return reasons


def _freeze_reasons(spec, manifest, context):
    if not spec["frozen"]:
        return [f"{context}_not_frozen"]
    _require(spec["source_role"] in {"development", "threshold_calibration"}, f"{context} may not be selected on final data")
    _require(spec["source_dataset_sha256"] == manifest["development_dataset_sha256"], f"{context} development source differs")
    _require(_time(spec["frozen_at"], context) <= min(_time(manifest["evaluation_started_at"], "evaluation start"),
                 _time(manifest["final_data_first_examined_at"], "first examination")),
             f"{context} frozen after final evaluation began")
    return []


def _gate(reasons, checks):
    if reasons:
        return {"status": "not_evaluable", "reasons": sorted(set(reasons)), "checks": checks}
    return {"status": "pass" if all(checks.values()) else "fail", "reasons": [], "checks": checks}


def _health(contract, manifest, truth, reference, candidate, authors, base_reasons):
    output = {}
    for name, spec in contract["targets"].items():
        if not spec["enabled"]:
            output[name] = {"status": "disabled"}
            continue
        reasons = []
        if not spec["definition_frozen"] or not isinstance(spec["definition"], str) or not spec["definition"].strip():
            reasons.append("target_definition_not_frozen")
        score = spec["score"]
        if score["kind"] == "aggregation":
            reasons += _freeze_reasons(score, manifest, "heat_aggregation")
            if score["method"] is None:
                reasons.append("heat_aggregation_method_unresolved")
            if reasons:
                output[name] = {"status": "not_evaluable", "reasons": reasons}
                continue
            if not all("heat" in row["human_labels"] for row in truth):
                _require("heat" not in contract["profile"]["required_targets"],
                         "Heat requires a separate independent human heat label")
                output[name] = {"status": "not_evaluable", "reasons": ["optional_target_human_labels_not_supplied"]}
                continue
            combine = max if score["method"] == "max" else lambda values: sum(values) / len(values)
            ref_scores = [combine([row["probabilities"][key] for key in score["inputs"]]) for row in reference]
            cand_scores = [combine([row["probabilities"][key] for key in score["inputs"]]) for row in candidate]
        else:
            ref_scores = [row["probabilities"][score["head"]] for row in reference]
            cand_scores = [row["probabilities"][score["head"]] for row in candidate]
        diagnostic = {"max_absolute_probability_delta": max((abs(a - b) for a, b in zip(ref_scores, cand_scores)), default=None),
                      "acceptance_use": "diagnostic_only"}
        reasons += _freeze_reasons(spec["threshold"], manifest, f"{name}_threshold")
        threshold = spec["threshold"]["value"]
        if threshold is None:
            reasons.append("threshold_value_unresolved")
        else:
            _number(threshold, f"{name} threshold")
        if reasons or not truth:
            output[name] = {"status": "not_evaluable", "reasons": reasons or ["no_scored_rows"], "diagnostic": diagnostic}
            continue
        if not all(name in row["human_labels"] for row in truth):
            output[name] = {"status": "not_evaluable", "reasons": ["optional_target_human_labels_not_supplied"],
                            "diagnostic": diagnostic}
            continue
        labels = [row["human_labels"][name] for row in truth]
        ref_decisions = [int(value >= threshold) for value in ref_scores]
        cand_decisions = [int(value >= threshold) for value in cand_scores]
        metrics = paired_health_metrics(labels, ref_decisions, cand_decisions, authors, contract["bootstrap"])
        metrics.update(status="computed", rows=len(labels), positives=sum(labels), negatives=len(labels) - sum(labels),
                       sample_prevalence=sum(labels) / len(labels), threshold=threshold, diagnostic=diagnostic)
        absolute_deltas = [abs(a - b) for a, b in zip(ref_scores, cand_scores)]
        diagnostic.update(mean_absolute_probability_delta=sum(absolute_deltas) / len(absolute_deltas),
                          absolute_probability_delta_quantiles={str(p): percentile(absolute_deltas, p) for p in (0.5, 0.9, 0.95, 0.99)},
                          disagreement_ids=[row["id"] for row, a, b in zip(truth, ref_decisions, cand_decisions) if a != b])
        brier = _paired_mean([(p - y) ** 2 for p, y in zip(ref_scores, labels)],
                             [(p - y) ** 2 for p, y in zip(cand_scores, labels)], authors, contract["bootstrap"])
        metrics["calibration"] = {"brier": brier, "reference": _calibration(labels, ref_scores, contract["calibration"]["bins"]),
                                  "candidate": _calibration(labels, cand_scores, contract["calibration"]["bins"])}
        agreement_reasons = _agreement_reasons(name, contract, manifest, truth)
        metrics["annotation_agreement"] = {"source": "external_receipt_not_recomputed",
                                           "receipt": manifest.get("annotation_agreement", {}).get(name),
                                           "gate": _gate(agreement_reasons, {"reliable_annotation_contract": not agreement_reasons})}
        quality_reasons = list(base_reasons) + agreement_reasons
        if not 0 < sum(labels) < len(labels):
            quality_reasons.append("both_label_classes_required")
        precision = contract["precision_plan"]
        minimum = precision["minimum_positives_per_target"]
        if minimum is not None and sum(labels) < minimum:
            quality_reasons.append("below_predeclared_positive_count")
        candidate_interval = metrics["intervals"]["candidate"]["f1"]
        if candidate_interval["status"] != "computed":
            quality_reasons.append("candidate_f1_interval_undefined")
        elif precision["maximum_candidate_f1_interval_width"] is not None:
            if candidate_interval["upper"] - candidate_interval["lower"] > precision["maximum_candidate_f1_interval_width"]:
                quality_reasons.append("f1_interval_wider_than_precision_plan")
        target = spec["minimum_f1_lower_bound"]
        if target is None:
            quality_reasons.append("absolute_f1_target_unresolved")
        metrics["quality_gate"] = _gate(quality_reasons, {
            "candidate_f1_lower_bound_meets_target": target is not None and candidate_interval["lower_one_sided"] is not None
                                                   and candidate_interval["lower_one_sided"] >= target,
        })
        parity_reasons = list(base_reasons) + agreement_reasons + _runtime_reasons(manifest)
        if minimum is not None and sum(labels) < minimum:
            parity_reasons.append("below_predeclared_positive_count")
        regression_reasons, regression_passed = _regression_reasons(name, contract, manifest)
        parity_reasons += regression_reasons
        if contract["comparison_tolerances_status"] == "unresolved":
            parity_reasons.append("comparison_tolerances_unresolved")
        for key in ("maximum_f1_drop", "maximum_decision_flip_rate"):
            if spec[key] is None:
                parity_reasons.append(f"{key}_unresolved")
        delta_interval = metrics["intervals"]["delta"]["f1"]
        flip_interval = metrics["decision_flip_interval"]
        flip_bound = metrics["decision_flip_acceptance_bound"]
        if delta_interval["status"] != "computed" or flip_interval["status"] != "computed":
            parity_reasons.append("paired_interval_undefined")
        if flip_bound["status"] != "computed":
            parity_reasons.append("decision_flip_acceptance_bound_not_evaluable")
        if contract["scope"]["same_hardware_runtime_required_for_quantization"]:
            if any(manifest["reference"][key] != manifest["candidate"][key] for key in ("hardware_sha256", "runtime_sha256")):
                parity_reasons.append("hardware_or_runtime_differs")
        metrics["quantization_gate"] = _gate(parity_reasons, {
            "paired_f1_delta_lower_bound_within_tolerance": spec["maximum_f1_drop"] is not None and delta_interval["lower_one_sided"] is not None
                                                           and delta_interval["lower_one_sided"] >= -spec["maximum_f1_drop"],
            "decision_flip_upper_bound_within_tolerance": spec["maximum_decision_flip_rate"] is not None and flip_bound["upper"] is not None
                                                         and flip_bound["upper"] <= spec["maximum_decision_flip_rate"],
            "independent_calibration_and_slice_review_passed": regression_passed,
        })
        brier_tolerance = contract["calibration"]["maximum_paired_brier_increase"]
        if brier_tolerance is not None:
            metrics["quantization_gate"] = _gate(parity_reasons, {
                **metrics["quantization_gate"]["checks"],
                "paired_brier_increase_upper_bound_within_tolerance":
                    brier["intervals"]["delta"]["upper_one_sided"] <= brier_tolerance,
            })
        output[name] = metrics
    return output


def _cosine(a, b):
    norm_a, norm_b = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    _require(norm_a > 0 and norm_b > 0, "Zero embedding prefix cannot define cosine")
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)))


def _retrieval(contract, manifest, truth, reference, candidate, authors, base_reasons):
    spec = contract["retrieval"]
    has_embeddings = any(row.get("embedding_256") is not None for row in reference + candidate)
    has_rankings = any(row.get("rankings") is not None for row in reference + candidate)
    result = {"status": "not_evaluable", "scope": "optional_supplied_retrieval_evidence",
              "embeddings": {}, "recall": {}, "reasons": []}
    if not truth:
        result["reasons"].append("no_scored_rows")
        return result
    gates = []
    reasons = list(base_reasons) + _runtime_reasons(manifest)
    if contract["comparison_tolerances_status"] == "unresolved":
        reasons.append("comparison_tolerances_unresolved")
    if not manifest["candidate"]["social_head_trained"]:
        reasons.append("social_head_not_trained")
    if contract["scope"]["same_hardware_runtime_required_for_quantization"]:
        if any(manifest["reference"][key] != manifest["candidate"][key] for key in ("hardware_sha256", "runtime_sha256")):
            reasons.append("hardware_or_runtime_differs")
    if has_embeddings:
        for row in reference + candidate:
            embedding = row.get("embedding_256")
            _require(isinstance(embedding, list) and len(embedding) == 256, "All scored rows require aligned 256-dimensional embeddings")
            for value in embedding:
                _number(value, "embedding component", -1e100, 1e100)
        for dimension in DIMS:
            cosines = [_cosine(a["embedding_256"][:dimension], b["embedding_256"][:dimension]) for a, b in zip(reference, candidate)]
            p1 = percentile(cosines, 0.01)
            minimum = spec["embedding_cosine_p1_minimum"][str(dimension)]
            gate = _gate(reasons + ([] if minimum is not None else ["cosine_tolerance_unresolved"]),
                         {"empirical_p1_meets_minimum": minimum is not None and p1 >= minimum})
            result["embeddings"][str(dimension)] = {"rows": len(cosines), "cosine_p1": p1, "cosine_minimum": min(cosines),
                                                    "gate": gate, "scope": "empirical_sample_tail"}
            gates.append(gate)
    else:
        result["reasons"].append("embeddings_not_supplied")
    if has_rankings:
        _keys(manifest.get("retrieval"), {"corpus_ids", "corpus_ids_sha256", "corpus_sha256", "relevance_protocol_sha256",
                                         "search_policy_sha256", "candidate_pool_sha256"}, "retrieval provenance")
        provenance = manifest["retrieval"]
        corpus = provenance["corpus_ids"]
        _require(isinstance(corpus, list) and bool(corpus) and all(isinstance(value, str) and value for value in corpus)
                 and len(set(corpus)) == len(corpus), "Invalid retrieval corpus IDs")
        _require(object_sha256(corpus) == provenance["corpus_ids_sha256"], "Retrieval corpus identity differs")
        for key in ("corpus_sha256", "relevance_protocol_sha256", "search_policy_sha256", "candidate_pool_sha256"):
            _digest(provenance[key], key)
        for key in ("reference", "candidate"):
            _digest(manifest[key].get("retrieval_index_sha256"), f"{key} retrieval index")
        pool = set(corpus)
        for row, ref, cand in zip(truth, reference, candidate):
            relevant = row.get("relevant_ids")
            _require(isinstance(relevant, list) and bool(relevant) and len(set(relevant)) == len(relevant)
                     and set(relevant) <= pool and row["id"] not in relevant, "Missing or invalid independent relevance judgments")
            _require(row["provenance"].get("relevance_source") == "human", "Retrieval relevance must be independent human judgments")
            for prediction in (ref, cand):
                rankings = prediction.get("rankings")
                _require(isinstance(rankings, dict) and set(rankings) == {str(d) for d in DIMS}, "All Matryoshka rankings are required")
                for ranking in rankings.values():
                    _require(isinstance(ranking, list) and all(isinstance(item, str) for item in ranking)
                             and len(ranking) >= max(spec["k_values"]) and len(set(ranking)) == len(ranking)
                             and set(ranking) <= pool and row["id"] not in ranking, "Invalid ranking/candidate-pool identity")
        for dimension in DIMS:
            result["recall"][str(dimension)] = {}
            tolerance = spec["maximum_recall_drop"][str(dimension)]
            for k in spec["k_values"]:
                values = []
                for predictions in (reference, candidate):
                    values.append([len(set(row["rankings"][str(dimension)][:k]) & set(label["relevant_ids"])) / len(label["relevant_ids"])
                                   for row, label in zip(predictions, truth)])
                metric = _paired_mean(*values, authors, contract["bootstrap"])
                gate = _gate(reasons + ([] if tolerance is not None else ["recall_tolerance_unresolved"]),
                             {"paired_recall_delta_lower_bound_within_tolerance": tolerance is not None
                              and metric["intervals"]["delta"]["lower_one_sided"] >= -tolerance})
                metric["gate"] = gate
                result["recall"][str(dimension)][str(k)] = metric
                gates.append(gate)
    else:
        result["reasons"].append("rankings_not_supplied")
    if result["reasons"]:
        result["status"] = "not_evaluable"
    elif any(gate["status"] == "not_evaluable" for gate in gates):
        result["status"] = "not_evaluable"
    elif any(gate["status"] == "fail" for gate in gates):
        result["status"] = "fail"
    else:
        result["status"] = "pass"
    return result


def evaluate_files(contract_path, manifest_path, labels_path, reference_path, candidate_path):
    """Validate external files, compute metrics, and return a JSON-safe receipt."""
    paths = {name: Path(path) for name, path in
             (("contract", contract_path), ("manifest", manifest_path), ("labels", labels_path),
              ("reference", reference_path), ("candidate", candidate_path))}
    hashes = {name: _sha_file(path) for name, path in paths.items()}
    contract, manifest = (_json(paths[name].read_text(encoding="utf-8")) for name in ("contract", "manifest"))
    _validate_contract(contract)
    _keys(manifest, {"inputs"}, "manifest input hashes")
    _require(manifest["inputs"] == {name: hashes[name] for name in ("labels", "reference", "candidate")},
             "Supplied prediction/label files differ from pinned manifest hashes")
    _require(manifest.get("contract_sha256") == hashes["contract"], "Frozen contract file identity differs")
    labels, reference, candidate = (_rows(paths[name]) for name in ("labels", "reference", "candidate"))
    selected = _validate_inputs(contract, manifest, labels, reference, candidate)
    truth, reference, candidate = ([rows[index] for index in selected] for rows in (labels, reference, candidate))
    authors = [row["author_id"] for row in truth]
    base_reasons = _base_reasons(contract, manifest, authors)
    health = _health(contract, manifest, truth, reference, candidate, authors, base_reasons)
    retrieval = _retrieval(contract, manifest, truth, reference, candidate, authors, base_reasons)
    statuses = []
    for name in contract["profile"]["required_targets"]:
        result = health[name]
        if result["status"] == "not_evaluable":
            statuses.append("not_evaluable")
        else:
            statuses += [result["quality_gate"]["status"], result["quantization_gate"]["status"]]
    if contract["retrieval"]["required_for_overall_qualification"]:
        statuses.append(retrieval["status"])
    overall = "not_evaluable" if not statuses or "not_evaluable" in statuses else (
        "fails_predeclared_provisional_gates" if "fail" in statuses else "passes_predeclared_provisional_gates")
    author_counts = Counter(authors)
    return {
        "schema": "catbird-independent-evaluation-report-v3", "status": overall,
        "evidence_kind": manifest["evidence_kind"],
        "measured_model_evidence": manifest["evidence_kind"] == "real_external_predictions",
        "operation": "metrics_only_no_inference", "input_sha256": hashes,
        "contract_status": contract["status"], "profile": contract["profile"], "qualification_reasons": base_reasons,
        "data_quality_certified_by_this_executor": False, "deployment_authorized": False,
        "candidate": manifest["candidate"], "reference": manifest["reference"],
        "population": manifest["population"], "sampling": manifest["sampling"],
        "metric_estimand": "unweighted_supplied_sample_pool",
        "event_window": manifest["event_window"], "bootstrap": contract["bootstrap"],
        "author_clusters": {"unique": len(author_counts), "with_repeated_rows": sum(count > 1 for count in author_counts.values()),
                            "maximum_rows_per_author": max(author_counts.values(), default=0)},
        "health": health, "retrieval": retrieval,
        "external_regression_review": manifest.get("regression_review"),
        "historical_smoke_modified": False,
        "limitations": [
            "Human provenance, complete exclusion inventory, model hashes and runtime metadata are caller attestations checked for consistency, not independently observed by this metrics executor.",
            "Intervals are marginal author-cluster percentile intervals; they do not cover cross-author, shared-event, conversation or temporal dependence.",
            "Undefined bootstrap denominators are retained and block the affected interval; they are never silently dropped.",
            "No deployment-population claim follows from an enriched or convenience sample's unweighted precision, recall or F1.",
            "No source or planning counts certify statistical power, label quality, multilingual coverage or production outcomes.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("contract", "manifest", "labels", "reference", "candidate", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        _require(not args.output.exists(), "Output already exists; choose a new versioned report path")
        report = evaluate_files(args.contract, args.manifest, args.labels, args.reference, args.candidate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except (EvaluationError, OSError, TypeError, KeyError) as error:
        print(json.dumps({"status": "invalid_input", "error": str(error)}), file=sys.stderr)
        return 3
    print(json.dumps({"status": report["status"], "report": str(args.output), "operation": "metrics_only_no_inference"}))
    return {"passes_predeclared_provisional_gates": 0, "fails_predeclared_provisional_gates": 1, "not_evaluable": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
