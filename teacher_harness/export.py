"""Versioned trainer export adapter and full annotation bank exporter."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

COLAB_DIR = Path(__file__).resolve().parents[1] / "colab"
if str(COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(COLAB_DIR))

from real_pipeline.checkpoints import atomic_write  # noqa: E402
from real_pipeline.common import sha256  # noqa: E402
from real_pipeline.contracts import HEALTH, STUDENT_CONTRACT  # noqa: E402
from real_pipeline.language import LANGUAGE_CONTRACT, language_scope  # noqa: E402

from .config import HarnessConfig, assert_no_credentials, canonical_sha256  # noqa: E402
from .db import HarnessDB  # noqa: E402
from .routing import evaluate_post_disagreement  # noqa: E402
from .schemas import COMPONENT_ORDER  # noqa: E402

ORDINAL_MAPPING_V1: dict[str, Any] = {
    "schema": "catbird-ordinal-to-unit-target-v1",
    "version": "1.0.0",
    "mapping": {
        0: 0.0,
        1: round(1.0 / 3.0, 8),
        2: round(2.0 / 3.0, 8),
        3: 1.0,
    },
    "insufficient_policy": "exclude_whole_row_by_default; never coerce insufficient or null to 0.0",
    "calibration_status": "uncalibrated_linear_ordinal_anchors; neither ordinal/3 nor model rationale is a calibrated probability",
    "human_calibration_gate_required": True,
}


class TrainerExportValidationError(ValueError):
    """Raised when exported targets violate the v3 student contract or context/language rules."""


def map_ordinal_to_training_score(
    score: int | str | None,
    mapping_spec: dict[str, Any] = ORDINAL_MAPPING_V1,
) -> float:
    """Convert an ordinal score (0, 1, 2, 3) to a finite [0, 1] float; reject 'insufficient' or None."""
    if score is None or score == "insufficient" or isinstance(score, bool):
        raise TrainerExportValidationError(
            f"Cannot map {score!r} to a finite training target; insufficient/null values must never be coerced to zero"
        )
    table = mapping_spec["mapping"]
    if score not in table and str(score) not in table:
        raise TrainerExportValidationError(f"Unsupported ordinal value: {score!r}")
    val = float(table[score] if score in table else table[str(score)])
    if not math.isfinite(val) or not (0.0 <= val <= 1.0):
        raise TrainerExportValidationError(f"Mapped target {val} is not a finite float in [0, 1]")
    return val


def export_annotation_bank_and_trainer_targets(
    db: HarnessDB,
    config: HarnessConfig,
    *,
    output_dir: Path,
    student_observation_mode: str = "text_only",
    allow_simulated_export: bool = True,
    write_parquet: bool = True,
) -> dict[str, Any]:
    """Export both the full multi-teacher annotation bank and the filtered trainer-compatible dataset."""
    if tuple(HEALTH) != COMPONENT_ORDER:
        raise TrainerExportValidationError(
            f"Component order mismatch between colab HEALTH {HEALTH} and harness {COMPONENT_ORDER}"
        )

    db.verify_existing_run_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_posts = db.load_posts()
    conn = db.connect()
    try:
        req_rows = conn.execute(
            """
            SELECT * FROM requests
            ORDER BY uri ASC, observation_mode ASC, model_tier ASC, attempt ASC
            """
        ).fetchall()
    finally:
        conn.close()

    # Group terminal request outcomes per (uri, observation_mode, model_tier) using latest attempt
    obs_by_uri: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    any_simulated = False
    for r in req_rows:
        u = str(r["uri"])
        obs_mode = str(r["observation_mode"])
        tier = str(r["model_tier"])
        if int(r["simulated"] or 0) == 1:
            any_simulated = True
        parsed = json.loads(r["parsed_output_json"]) if r["parsed_output_json"] else None
        obs_by_uri.setdefault(u, {}).setdefault(obs_mode, {})[tier] = {
            "custom_id": r["custom_id"],
            "request_sha256": r["request_sha256"],
            "attempt": int(r["attempt"]),
            "status": r["status"],
            "teacher_model": r["teacher_model"],
            "returned_model": r["returned_model"],
            "stop_reason": r["stop_reason"],
            "actual_input_tokens": r["actual_input_tokens"],
            "actual_output_tokens": r["actual_output_tokens"],
            "actual_billed_microdollars": r["actual_billed_microdollars"],
            "response_sha256": r["response_sha256"],
            "simulated": bool(r["simulated"]),
            "output": parsed,
        }

    annotation_bank_rows: list[dict[str, Any]] = []
    trainer_accepted_rows: list[dict[str, Any]] = []
    trainer_exclusions: list[dict[str, Any]] = []
    exclusion_counts: Counter[str] = Counter()

    for post in all_posts:
        uri = post["uri"]
        split = post["split"]
        scope = language_scope(post)
        uri_obs = obs_by_uri.get(uri, {})
        text_only_teachers = {
            tier: rec["output"]
            for tier, rec in uri_obs.get("text_only", {}).items()
            if rec["status"] == "succeeded" and rec["output"] is not None
        }
        ctx_enriched_teachers = {
            tier: rec["output"]
            for tier, rec in uri_obs.get("context_enriched", {}).items()
            if rec["status"] == "succeeded" and rec["output"] is not None
        }

        disagreement_text_only = (
            evaluate_post_disagreement(text_only_teachers) if text_only_teachers else None
        )
        disagreement_ctx_enriched = (
            evaluate_post_disagreement(ctx_enriched_teachers) if ctx_enriched_teachers else None
        )

        bank_entry = {
            "uri": uri,
            "cid": post.get("cid"),
            "author_did": post["author_did"],
            "author_split_bucket": post["author_split_bucket"],
            "split": split,
            "post_type": post["post_type"],
            "repost_subject_uri": post.get("repost_subject_uri"),
            "langs": post["langs"],
            "language_scope": scope,
            "text_sha256": post["text_sha256"],
            "text_only_context_sha256": post["text_only_context_sha256"],
            "context_enriched_sha256": post["context_enriched_sha256"],
            "sampling_stratum": post.get("sampling_stratum"),
            "sampling_method": post.get("sampling_method"),
            "inclusion_probability": post.get("inclusion_probability"),
            "eligibility_snapshot": post["eligibility_snapshot"],
            "moderation_join": post["moderation_join"],
            "observations": {
                "text_only": uri_obs.get("text_only", {}),
                "context_enriched": uri_obs.get("context_enriched", {}),
            },
            "disagreement": {
                "text_only": disagreement_text_only,
                "context_enriched": disagreement_ctx_enriched,
            },
            "simulated_any": any_simulated,
        }
        assert_no_credentials(bank_entry, f"annotation_bank:{uri}")
        annotation_bank_rows.append(bank_entry)

        # Trainer export filtering: ONLY split == 'train' can enter student training
        if split != "train":
            reason = f"non_train_split:{split}"
            exclusion_counts[reason] += 1
            trainer_exclusions.append({"uri": uri, "split": split, "exclusion_reason": reason})
            continue

        if not scope["in_scope"]:
            reason = f"unsupported_language:{scope['reason']}"
            exclusion_counts[reason] += 1
            trainer_exclusions.append({"uri": uri, "split": split, "exclusion_reason": reason})
            continue

        # Enforce observation mode compatibility: a text-only student must NEVER be trained
        # on context-enriched labels if text_only is missing or disagrees!
        target_obs_map = uri_obs.get(student_observation_mode, {})
        if not target_obs_map:
            if student_observation_mode == "text_only" and uri_obs.get("context_enriched"):
                reason = "context_mismatch_only_context_enriched_available_for_text_only_student"
            else:
                reason = "no_teacher_request_for_observation_mode"
            exclusion_counts[reason] += 1
            trainer_exclusions.append({"uri": uri, "split": split, "exclusion_reason": reason})
            continue

        succeeded_teachers = {
            tier: rec["output"]
            for tier, rec in target_obs_map.items()
            if rec["status"] == "succeeded" and rec["output"] is not None
        }
        if not succeeded_teachers:
            # Classify why no succeeded output exists
            statuses = [rec["status"] for rec in target_obs_map.values()]
            primary_status = statuses[-1] if statuses else "missing"
            reason = f"non_succeeded_request_status:{primary_status}"
            exclusion_counts[reason] += 1
            trainer_exclusions.append({"uri": uri, "split": split, "exclusion_reason": reason})
            continue

        disagreement = evaluate_post_disagreement(succeeded_teachers)
        if disagreement["has_insufficient_component"]:
            reason = "insufficient_evidence_component"
            exclusion_counts[reason] += 1
            trainer_exclusions.append({
                "uri": uri,
                "split": split,
                "exclusion_reason": reason,
                "per_component": disagreement["per_component"],
            })
            continue

        if disagreement["has_unresolved_disagreement"]:
            reason = "unresolved_teacher_disagreement"
            exclusion_counts[reason] += 1
            trainer_exclusions.append({
                "uri": uri,
                "split": split,
                "exclusion_reason": reason,
                "per_component": disagreement["per_component"],
            })
            continue

        # Also check if text_only and context_enriched both exist and disagree on any component
        if student_observation_mode == "text_only" and ctx_enriched_teachers:
            ctx_dis = evaluate_post_disagreement({
                "text_only_haiku": next(iter(succeeded_teachers.values())),
                "ctx_enriched": next(iter(ctx_enriched_teachers.values())),
            })
            if ctx_dis["has_unresolved_disagreement"]:
                reason = "context_dependence_diverges_between_text_only_and_enriched"
                exclusion_counts[reason] += 1
                trainer_exclusions.append({
                    "uri": uri,
                    "split": split,
                    "exclusion_reason": reason,
                })
                continue

        if any_simulated and not allow_simulated_export:
            raise TrainerExportValidationError(
                "Refusing to export simulated fake-transport responses when allow_simulated_export=False"
            )

        # Build finite [0,1] target row in exact COMPONENT_ORDER
        # When multiple agreeing teachers exist (e.g. exact agreement or 1-step adjacent), average their mapped ordinal scores
        trainer_row: dict[str, Any] = {
            "uri": uri,
            "author_did": post["author_did"],
            "author_split_bucket": int(post["author_split_bucket"]),
            "created_at": post["created_at"],
            "text": post["text"],
            "langs": post["langs"],
            "post_type": post["post_type"],
            "reply_parent_uri": post.get("reply_parent_uri", ""),
            "reply_root_uri": post.get("reply_root_uri", ""),
            "quote_uri": post.get("quote_uri", ""),
            "embed_type": post.get("embed_type", ""),
            "image_count": int(post.get("image_count", 0)),
            "has_video": int(post.get("has_video", 0)),
            "external_uri": post.get("external_uri", ""),
            "score_status": "scored",
            "unscored_reason": None,
            "observation_mode": student_observation_mode,
            "simulated_not_for_production_training": any_simulated,
        }

        for comp in COMPONENT_ORDER:
            tier_vals: list[float] = []
            tier_ordinals: dict[str, int] = {}
            for tier, t_out in sorted(succeeded_teachers.items()):
                ord_val = t_out[comp]["score"]
                mapped = map_ordinal_to_training_score(ord_val)
                tier_vals.append(mapped)
                tier_ordinals[tier] = int(ord_val)
            mean_target = round(sum(tier_vals) / len(tier_vals), 8)
            if not math.isfinite(mean_target) or not (0.0 <= mean_target <= 1.0):
                raise TrainerExportValidationError(f"Non-finite or out-of-range target for {comp}: {mean_target}")
            trainer_row[f"teacher_{comp}"] = mean_target
            trainer_row[f"teacher_{comp}_ordinals"] = tier_ordinals

        # Verify no heat or hostility target was introduced
        if "teacher_heat" in trainer_row or "teacher_hostility" in trainer_row:
            raise TrainerExportValidationError("Forbidden legacy heat or undeclared hostility target present")

        trainer_accepted_rows.append(trainer_row)

    # Write immutable artifacts using atomic_write
    bank_path = output_dir / "annotation_bank.jsonl"
    bank_bytes = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for r in annotation_bank_rows).encode("utf-8")
    atomic_write(bank_path, bank_bytes, overwrite=True)

    exclusions_path = output_dir / "trainer_exclusions.jsonl"
    excl_bytes = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for r in trainer_exclusions).encode("utf-8")
    atomic_write(exclusions_path, excl_bytes, overwrite=True)

    trainer_jsonl_path = output_dir / "trainer_targets.jsonl"
    trainer_bytes = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for r in trainer_accepted_rows).encode("utf-8")
    atomic_write(trainer_jsonl_path, trainer_bytes, overwrite=True)

    parquet_sha: str | None = None
    parquet_path = output_dir / "trainer_targets.parquet"
    if write_parquet and trainer_accepted_rows:
        import pyarrow as pa
        import pyarrow.parquet as pq

        tmp_pq = output_dir / ".trainer_targets.parquet.partial"
        pq.write_table(pa.Table.from_pylist(trainer_accepted_rows), tmp_pq, compression="zstd")
        atomic_write(parquet_path, tmp_pq.read_bytes(), overwrite=True)
        tmp_pq.unlink(missing_ok=True)
        parquet_sha = sha256(parquet_path)

    export_manifest = {
        "schema": "catbird-anthropic-trainer-export-v1",
        "student_contract": STUDENT_CONTRACT,
        "language_contract": LANGUAGE_CONTRACT,
        "component_order": list(COMPONENT_ORDER),
        "student_observation_mode": student_observation_mode,
        "ordinal_mapping": ORDINAL_MAPPING_V1,
        "masked_loss_in_current_trainer": False,
        "incomplete_row_policy": "whole_row_excluded_from_trainer_targets_and_preserved_in_annotation_bank",
        "insufficient_coerced_to_zero": False,
        "simulated_responses_present": any_simulated,
        "qualified_for_real_student_training": False,
        "qualification_status_by_component": {
            "politicalness": "unqualified_requires_human_final_set_and_alpha_gate",
            "outraged": "diagnostic_unqualified_no_preregistered_f1_gate",
            "critical": "diagnostic_unqualified_no_preregistered_f1_gate",
            "engagement_bait": "diagnostic_unqualified_no_preregistered_f1_gate",
            "spam": "diagnostic_unqualified_no_preregistered_f1_gate",
        },
        "qualification_note": (
            "A future pass on politicalness never qualifies outraged, critical, engagement_bait, or spam. "
            "Human calibration and independent human final labels remain mandatory."
        ),
        "counts": {
            "total_corpus_rows": len(all_posts),
            "annotation_bank_rows": len(annotation_bank_rows),
            "trainer_accepted_rows": len(trainer_accepted_rows),
            "trainer_excluded_rows": len(trainer_exclusions),
            "exclusion_reasons": dict(sorted(exclusion_counts.items())),
        },
        "artifacts": {
            "annotation_bank_jsonl": {
                "path": bank_path.name,
                "sha256": sha256(bank_path),
            },
            "trainer_exclusions_jsonl": {
                "path": exclusions_path.name,
                "sha256": sha256(exclusions_path),
            },
            "trainer_targets_jsonl": {
                "path": trainer_jsonl_path.name,
                "sha256": sha256(trainer_jsonl_path),
            },
            "trainer_targets_parquet": (
                {"path": parquet_path.name, "sha256": parquet_sha} if parquet_sha else None
            ),
        },
    }
    export_manifest["manifest_sha256"] = canonical_sha256(export_manifest)
    manifest_path = output_dir / "trainer_export_manifest.json"
    atomic_write(
        manifest_path,
        (json.dumps(export_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
        overwrite=True,
    )
    return export_manifest
