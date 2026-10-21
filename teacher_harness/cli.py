"""Command-line interface for the Catbird Anthropic Teacher Harness."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Sequence

from .config import HarnessConfig
from .db import HarnessDB, acquire_writer_lock
from .export import export_annotation_bank_and_trainer_targets
from .pipeline import (
    run_collect,
    run_plan,
    run_prepare,
    run_reconcile,
    run_report,
    run_submit,
)
from .routing import route_review_candidates


def _load_config(args: argparse.Namespace) -> HarnessConfig:
    if getattr(args, "config_dir", None):
        cfg_dir = Path(args.config_dir)
        return HarnessConfig.load(
            rubric_path=cfg_dir / "rubric_v1.json",
            pricing_path=cfg_dir / "pricing_and_budget_v1.json",
            schema_path=cfg_dir / "output_schema_v1.json",
        )
    return HarnessConfig.load()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anthropic_teacher_harness",
        description=(
            "Catbird Timeline-Health Anthropic Teacher Labeling Harness "
            "(offline by default; zero paid API calls without explicit authorization)."
        ),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Optional directory overriding rubric_v1.json, pricing_and_budget_v1.json, and output_schema_v1.json.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_run_dir(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--run-dir",
            type=Path,
            required=True,
            help="Path to the run directory containing state.sqlite3 and immutable artifacts.",
        )

    # prepare
    p_prep = subparsers.add_parser(
        "prepare",
        help="Validate inputs, eligibility, deduplication, author-disjoint splits, and moderation joins.",
    )
    _add_run_dir(p_prep)
    p_prep.add_argument(
        "--input-path",
        type=Path,
        default=Path("/Users/joshlacalamito/catbird-lake-local/drive-sync-staging/training-banks/phase3_health_training_150k_v1_unlabeled.parquet"),
        help="Path to canonical unlabeled training bank parquet.",
    )
    p_prep.add_argument(
        "--holdout-path",
        type=Path,
        default=Path("/Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/phase3_holdout_600_unlabeled.jsonl"),
        help="Path to sealed 600-post holdout JSONL.",
    )
    p_prep.add_argument(
        "--export-manifest-path",
        type=Path,
        default=Path("/Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/phase3_v1_export_manifest.json"),
        help="Path to phase3_v1_export_manifest.json.",
    )
    p_prep.add_argument(
        "--moderation-path",
        type=Path,
        default=None,
        help="Optional moderation label directory or .cbor.gz/.jsonl file.",
    )
    p_prep.add_argument(
        "--moderation-max-chunks",
        type=int,
        default=2,
        help="Cap on number of .cbor.gz chunks read from moderation directory (default: 2).",
    )
    p_prep.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap on input bank rows loaded (for fast offline dry-runs).",
    )
    p_prep.add_argument(
        "--target-train-sample-size",
        type=int,
        default=25000,
        help="Target stratified sample size across eligible train posts (default: 25000).",
    )
    p_prep.add_argument(
        "--moderation-cutoff-utc",
        type=str,
        default="2026-10-20T23:59:59+00:00",
        help="Moderation label snapshot cutoff timestamp (ISO-8601).",
    )
    p_prep.add_argument(
        "--hydration-path",
        type=Path,
        default=None,
        help="Optional parent/quote context hydration JSONL or Parquet file.",
    )
    p_prep.add_argument(
        "--run-id",
        type=str,
        default="catbird-anthropic-teacher-run-v1",
    )
    p_prep.add_argument(
        "--generation-id",
        type=str,
        default="gen-20261020-v1",
    )
    p_prep.add_argument(
        "--allow-synthetic-input-for-tests",
        action="store_true",
        help="Allow synthetic test fixture inputs whose SHA-256 differs from canonical 150k bank.",
    )

    # plan
    p_plan = subparsers.add_parser(
        "plan",
        help="Produce deterministic batch requests, estimated costs, and a reviewable manifest without submitting.",
    )
    _add_run_dir(p_plan)
    p_plan.add_argument(
        "--stage",
        type=str,
        default="pilot",
        choices=["pilot", "haiku_bulk", "sonnet_review", "opus_review"],
    )
    p_plan.add_argument(
        "--model-tier",
        type=str,
        default="haiku",
        choices=["haiku", "sonnet", "opus"],
    )
    p_plan.add_argument(
        "--observation-mode",
        type=str,
        default="text_only",
        choices=["text_only", "context_enriched"],
    )
    p_plan.add_argument(
        "--max-requests",
        type=int,
        default=None,
    )
    p_plan.add_argument(
        "--pilot-paired-design",
        action="store_true",
        help="Plan the paired pilot overlap design across Haiku, Sonnet, and Opus.",
    )
    p_plan.add_argument(
        "--expansion-wave",
        type=str,
        default="pilot",
        choices=["pilot", "wave_25k", "wave_150k", "wave_500k"],
    )

    # submit
    p_sub = subparsers.add_parser(
        "submit",
        help="Atomically reserve budget and submit an explicitly authorized stage in bounded waves.",
    )
    _add_run_dir(p_sub)
    p_sub.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["pilot", "haiku_bulk", "sonnet_review", "opus_review"],
    )
    p_sub.add_argument(
        "--model-tier",
        type=str,
        default="haiku",
        choices=["haiku", "sonnet", "opus"],
    )
    p_sub.add_argument(
        "--transport",
        type=str,
        default="fake",
        choices=["fake", "live"],
        help="Transport backend: 'fake' for offline simulation or 'live' for Anthropic SDK.",
    )
    p_sub.add_argument(
        "--offline",
        dest="offline",
        action="store_true",
        default=True,
        help="Enforce offline mode (default: True).",
    )
    p_sub.add_argument(
        "--no-offline",
        dest="offline",
        action="store_false",
        help="Disable offline lock (requires --authorize-paid-submission and --authorize-corpus-transmission).",
    )
    p_sub.add_argument(
        "--authorize-paid-submission",
        action="store_true",
        help="Explicitly authorize paid Anthropic Batch API requests.",
    )
    p_sub.add_argument(
        "--authorize-corpus-transmission",
        action="store_true",
        help="Explicitly authorize transmitting corpus text to Anthropic.",
    )
    p_sub.add_argument(
        "--expansion-wave",
        type=str,
        default="pilot",
        choices=["pilot", "wave_25k", "wave_150k", "wave_500k"],
    )
    p_sub.add_argument(
        "--max-batch-requests",
        type=int,
        default=250,
    )
    p_sub.add_argument(
        "--max-stage-spend-usd",
        type=float,
        default=8.0,
        help="Explicit CLI stage spend guard in USD (default: 8.00).",
    )
    p_sub.add_argument(
        "--max-outstanding-usd",
        type=float,
        default=2.0,
        help="Explicit CLI outstanding liability guard in USD (default: 2.00).",
    )

    # status
    p_stat = subparsers.add_parser(
        "status",
        help="Inspect run state, request counts, and budget ledger without network calls.",
    )
    _add_run_dir(p_stat)

    # reconcile
    p_rec = subparsers.add_parser(
        "reconcile",
        help="Recover provider state and resolve ambiguous submissions without blind resubmission.",
    )
    _add_run_dir(p_rec)
    p_rec.add_argument("--transport", type=str, default="fake", choices=["fake", "live"])
    p_rec.add_argument("--offline", dest="offline", action="store_true", default=True)
    p_rec.add_argument("--no-offline", dest="offline", action="store_false")
    p_rec.add_argument("--authorize-paid-submission", action="store_true")
    p_rec.add_argument("--authorize-corpus-transmission", action="store_true")

    # collect
    p_col = subparsers.add_parser(
        "collect",
        help="Preserve immutable raw response shards, validate outputs, and settle reservations.",
    )
    _add_run_dir(p_col)
    p_col.add_argument("--transport", type=str, default="fake", choices=["fake", "live"])
    p_col.add_argument("--offline", dest="offline", action="store_true", default=True)
    p_col.add_argument("--no-offline", dest="offline", action="store_false")
    p_col.add_argument("--authorize-paid-submission", action="store_true")
    p_col.add_argument("--authorize-corpus-transmission", action="store_true")

    # route
    p_route = subparsers.add_parser(
        "route",
        help="Deterministically select Sonnet or Opus review candidates and generate blind prompts.",
    )
    _add_run_dir(p_route)
    p_route.add_argument(
        "--target-stage",
        type=str,
        required=True,
        choices=["sonnet_review", "opus_review", "pilot_sonnet", "pilot_opus"],
    )
    p_route.add_argument(
        "--total-quota",
        type=int,
        required=True,
    )
    p_route.add_argument(
        "--observation-mode",
        type=str,
        default="text_only",
        choices=["text_only", "context_enriched"],
    )
    p_route.add_argument(
        "--seed",
        type=str,
        default="catbird-routing-v1",
    )

    # export
    p_exp = subparsers.add_parser(
        "export",
        help="Produce full annotation bank, trainer-compatible targets, and explicit exclusions.",
    )
    _add_run_dir(p_exp)
    p_exp.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory (defaults to <run-dir>/exports).",
    )
    p_exp.add_argument(
        "--student-observation-mode",
        type=str,
        default="text_only",
        choices=["text_only", "context_enriched"],
    )
    p_exp.add_argument(
        "--allow-simulated-export",
        action="store_true",
        default=True,
    )
    p_exp.add_argument(
        "--mask-insufficient-components",
        action="store_true",
        default=False,
        help="Mask insufficient components with -1.0 instead of excluding the whole row.",
    )

    # report
    p_rep = subparsers.add_parser(
        "report",
        help="Summarize costs, coverage, disagreement, and remaining stage gates.",
    )
    _add_run_dir(p_rep)

    # recover-partials
    p_rec_part = subparsers.add_parser(
        "recover-partials",
        help="Quarantine interrupted .partial files into recovery/ before resuming.",
    )
    _add_run_dir(p_rec_part)

    # record-gate
    p_gate = subparsers.add_parser(
        "record-gate",
        help="Record stage-gate evaluation evidence before expanding to larger waves.",
    )
    _add_run_dir(p_gate)
    p_gate.add_argument(
        "--wave-name",
        type=str,
        required=True,
        choices=["pilot", "wave_25k", "wave_150k"],
    )
    p_gate.add_argument("--approved", action="store_true")
    p_gate.add_argument("--schema-valid-rate", type=float, required=True)
    p_gate.add_argument("--human-agreement-alpha", type=float, default=None)

    # retry-transient
    p_retry = subparsers.add_parser(
        "retry-transient",
        help="Create fresh planned attempts for eligible transient errors under bounded attempts.",
    )
    _add_run_dir(p_retry)
    p_retry.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["pilot", "haiku_bulk", "sonnet_review", "opus_review"],
    )
    p_retry.add_argument("--max-attempts", type=int, default=2)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _load_config(args)
    run_dir = Path(args.run_dir).resolve()

    if args.command == "recover-partials":
        with acquire_writer_lock(run_dir, "recover-partials"):
            db = HarnessDB(run_dir, cfg)
            moved = db.recover_partial_artifacts()
        print(json.dumps({"command": "recover-partials", "moved_to_recovery": moved}, indent=2))
        return 0

    if args.command == "prepare":
        res = run_prepare(
            run_dir=run_dir,
            input_path=args.input_path,
            holdout_path=args.holdout_path,
            export_manifest_path=args.export_manifest_path,
            config=cfg,
            sample_limit=args.sample_limit,
            target_train_sample_size=args.target_train_sample_size,
            moderation_path=args.moderation_path,
            moderation_max_chunks=args.moderation_max_chunks,
            moderation_cutoff_utc=args.moderation_cutoff_utc,
            hydration_path=args.hydration_path,
            run_id=args.run_id,
            generation_id=args.generation_id,
            allow_synthetic_input_for_tests=args.allow_synthetic_input_for_tests,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "plan":
        res = run_plan(
            run_dir=run_dir,
            stage=args.stage,
            model_tier=args.model_tier,
            observation_mode=args.observation_mode,
            max_requests=args.max_requests,
            pilot_paired_design=args.pilot_paired_design,
            expansion_wave=args.expansion_wave,
            config=cfg,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "submit":
        res = run_submit(
            run_dir=run_dir,
            stage=args.stage,
            model_tier=args.model_tier,
            transport_kind=args.transport,
            offline=args.offline,
            authorize_paid_submission=args.authorize_paid_submission,
            authorize_corpus_transmission=args.authorize_corpus_transmission,
            expansion_wave=args.expansion_wave,
            max_batch_requests=args.max_batch_requests,
            config=cfg,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "status":
        db = HarnessDB(run_dir, cfg)
        meta = db.verify_existing_run_config()
        conn = db.connect()
        try:
            req_counts = Counter(r["status"] for r in conn.execute("SELECT status FROM requests").fetchall())
            batch_counts = Counter(r["status"] for r in conn.execute("SELECT status FROM batches").fetchall())
        finally:
            conn.close()
        out = {
            "command": "status",
            "run_metadata": meta,
            "request_status_counts": dict(sorted(req_counts.items())),
            "batch_status_counts": dict(sorted(batch_counts.items())),
            "budget_summary": db.get_budget_summary(),
        }
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    if args.command == "reconcile":
        res = run_reconcile(
            run_dir=run_dir,
            transport_kind=args.transport,
            offline=args.offline,
            authorize_paid_submission=args.authorize_paid_submission,
            authorize_corpus_transmission=args.authorize_corpus_transmission,
            config=cfg,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "collect":
        res = run_collect(
            run_dir=run_dir,
            transport_kind=args.transport,
            offline=args.offline,
            authorize_paid_submission=args.authorize_paid_submission,
            authorize_corpus_transmission=args.authorize_corpus_transmission,
            config=cfg,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "route":
        with acquire_writer_lock(run_dir, f"route:{args.target_stage}"):
            db = HarnessDB(run_dir, cfg)
            res = route_review_candidates(
                db,
                cfg,
                target_stage=args.target_stage,
                total_quota=args.total_quota,
                seed=args.seed,
                observation_mode=args.observation_mode,
            )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "export":
        out_dir = Path(args.output_dir) if args.output_dir else (run_dir / "exports")
        with acquire_writer_lock(run_dir, "export"):
            db = HarnessDB(run_dir, cfg)
            res = export_annotation_bank_and_trainer_targets(
                db,
                cfg,
                output_dir=out_dir,
                student_observation_mode=args.student_observation_mode,
                allow_simulated_export=args.allow_simulated_export,
                mask_insufficient_components=args.mask_insufficient_components,
            )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "report":
        res = run_report(run_dir=run_dir, config=cfg)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0

    if args.command == "record-gate":
        with acquire_writer_lock(run_dir, f"record-gate:{args.wave_name}"):
            db = HarnessDB(run_dir, cfg)
            db.record_stage_gate_evidence(
                wave_name=args.wave_name,
                approved=bool(args.approved),
                schema_valid_rate=args.schema_valid_rate,
                human_agreement_alpha=args.human_agreement_alpha,
                evidence_receipt={
                    "wave_name": args.wave_name,
                    "approved": bool(args.approved),
                    "schema_valid_rate": args.schema_valid_rate,
                    "human_agreement_alpha": args.human_agreement_alpha,
                },
            )
        print(json.dumps({"command": "record-gate", "wave_name": args.wave_name, "approved": bool(args.approved)}, indent=2))
        return 0

    if args.command == "retry-transient":
        with acquire_writer_lock(run_dir, f"retry-transient:{args.stage}"):
            db = HarnessDB(run_dir, cfg)
            created = db.create_retry_requests_for_eligible_failures(
                stage=args.stage,
                max_attempts=args.max_attempts,
            )
        print(json.dumps({"command": "retry-transient", "created_custom_ids": created}, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
