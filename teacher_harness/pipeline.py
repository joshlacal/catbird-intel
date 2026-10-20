"""Core pipeline orchestration for prepare, plan, submit, reconcile, collect, route, export, and report."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

COLAB_DIR = Path(__file__).resolve().parents[1] / "colab"
if str(COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(COLAB_DIR))

from real_pipeline.checkpoints import atomic_write  # noqa: E402
from real_pipeline.common import sha256  # noqa: E402

from .config import (  # noqa: E402
    BudgetConstraintError,
    HarnessConfig,
    assert_no_credentials,
    canonical_sha256,
    format_usd,
    scrub_environment_snapshot,
)
from .db import (  # noqa: E402
    HarnessDB,
    acquire_writer_lock,
    utc_now_iso,
)
from .export import export_annotation_bank_and_trainer_targets  # noqa: E402
from .moderation import load_moderation_events  # noqa: E402
from .routing import evaluate_post_disagreement, route_review_candidates  # noqa: E402
from .schemas import (  # noqa: E402
    COMPONENT_ORDER,
    SchemaValidationError,
    build_batch_request_for_post,
    validate_teacher_output_json,
)
from .splits import prepare_and_freeze_corpus  # noqa: E402
from .transport import (  # noqa: E402
    AnthropicTransport,
    FakeAnthropicTransport,
    OfficialAnthropicSDKTransport,
)


def write_immutable_json(path: Path, payload: Any, *, overwrite: bool = True) -> str:
    """Write JSON atomically after scrubbing for credentials, returning file SHA-256."""
    assert_no_credentials(payload, str(path.name))
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, data, overwrite=overwrite)
    return sha256(path)


def run_prepare(
    *,
    run_dir: Path,
    input_path: Path,
    holdout_path: Path,
    export_manifest_path: Path | None = None,
    config: HarnessConfig | None = None,
    sample_limit: int | None = None,
    target_train_sample_size: int | None = None,
    moderation_path: Path | None = None,
    moderation_max_chunks: int | None = 2,
    moderation_cutoff_utc: str = "2026-10-20T23:59:59+00:00",
    run_id: str = "catbird-anthropic-teacher-run-v1",
    generation_id: str = "gen-20261020-v1",
    selection_seed: str = "catbird-teacher-selection-v1",
    allow_synthetic_input_for_tests: bool = False,
) -> dict[str, Any]:
    """Validate inputs, freeze splits, join moderation labels, and initialize transactional DB."""
    cfg = config or HarnessConfig.load()
    run_dir = Path(run_dir)
    with acquire_writer_lock(run_dir, "prepare"):
        mod_events = load_moderation_events(moderation_path, max_chunks=moderation_max_chunks)
        prep = prepare_and_freeze_corpus(
            input_path=input_path,
            holdout_path=holdout_path,
            export_manifest_path=export_manifest_path,
            config=cfg,
            sample_limit=sample_limit,
            target_train_sample_size=target_train_sample_size,
            moderation_events=mod_events,
            moderation_cutoff_utc=moderation_cutoff_utc,
            selection_seed=selection_seed,
            allow_synthetic_input_for_tests=allow_synthetic_input_for_tests,
        )
        db = HarnessDB(run_dir, cfg)
        meta = db.init_or_verify_run(
            run_id=run_id,
            generation_id=generation_id,
            input_manifest_sha256=prep["manifest"]["manifest_sha256"],
        )
        db.store_prepared_posts(prep["posts"])

        manifests_dir = run_dir / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        prep_manifest_path = manifests_dir / "preparation_manifest.json"
        write_immutable_json(prep_manifest_path, prep["manifest"], overwrite=True)

        # Also snapshot frozen rubric, output schema, and pricing into run_dir/manifests/
        write_immutable_json(manifests_dir / "frozen_rubric.json", cfg.rubric, overwrite=True)
        write_immutable_json(manifests_dir / "frozen_output_schema.json", cfg.output_schema, overwrite=True)
        write_immutable_json(manifests_dir / "frozen_pricing_and_budget.json", cfg.pricing, overwrite=True)

        run_manifest = {
            "schema": "catbird-anthropic-teacher-run-manifest-v1",
            "run_metadata": meta,
            "environment": scrub_environment_snapshot(),
            "preparation_manifest_sha256": prep["manifest"]["manifest_sha256"],
        }
        write_immutable_json(run_dir / "run_manifest.json", run_manifest, overwrite=True)
        return prep["manifest"]


def run_plan(
    *,
    run_dir: Path,
    stage: str = "pilot",
    model_tier: str = "haiku",
    observation_mode: str = "text_only",
    max_requests: int | None = None,
    pilot_paired_design: bool = False,
    expansion_wave: str = "pilot",
    config: HarnessConfig | None = None,
) -> dict[str, Any]:
    """Produce deterministic requests, offline cost estimates, and a reviewable plan manifest without submitting."""
    cfg = config or HarnessConfig.load()
    run_dir = Path(run_dir)
    with acquire_writer_lock(run_dir, f"plan:{stage}:{model_tier}"):
        db = HarnessDB(run_dir, cfg)
        meta = db.verify_existing_run_config()

        gate = db.check_expansion_gate(expansion_wave)
        if not gate["allowed"]:
            raise BudgetConstraintError(gate["reason"])

        selected_posts = db.load_posts(split="train", eligible_only=True, selected_only=True)
        if not selected_posts:
            raise ValueError("No eligible selected train posts found; run 'prepare' first")
        all_posts_by_uri = {p["uri"]: p for p in db.load_posts()}

        # Sort deterministically by (selection_priority_key, uri)
        selected_posts.sort(key=lambda p: (str(p.get("selection_priority_key") or ""), p["uri"]))
        if max_requests is not None:
            selected_posts = selected_posts[: int(max_requests)]

        planned_all: list[dict[str, Any]] = []
        tier_plans: dict[str, dict[str, Any]] = {}

        if stage == "pilot" and pilot_paired_design:
            # Pilot design: Haiku on all N items (up to 2,000), Sonnet on 50% (up to 1,000), Opus on 15% (up to 300)
            n_haiku = min(len(selected_posts), 2000)
            n_sonnet = min(n_haiku, max(1, (n_haiku * 1000) // 2000)) if n_haiku >= 2 else n_haiku
            n_opus = min(n_sonnet, max(1, (n_haiku * 300) // 2000)) if n_haiku >= 4 else min(1, n_sonnet)
            plan_slices = [
                ("haiku", selected_posts[:n_haiku]),
                ("sonnet", selected_posts[:n_sonnet]),
                ("opus", selected_posts[:n_opus]),
            ]
        else:
            plan_slices = [(model_tier, selected_posts)]

        for tier_name, slice_posts in plan_slices:
            tier_reqs: list[dict[str, Any]] = []
            for p in slice_posts:
                req = build_batch_request_for_post(
                    p,
                    config=cfg,
                    stage=stage,
                    model_tier=tier_name,  # type: ignore[arg-type]
                    observation_mode=observation_mode,  # type: ignore[arg-type]
                    attempt=1,
                    generation_id=meta["generation_id"],
                    corpus_by_uri=all_posts_by_uri,
                )
                tier_reqs.append(req)
                planned_all.append(req)

            spec = cfg.model_spec(tier_name)
            mean_in = int(spec["planning_mean_input_tokens"])
            mean_out = int(spec["planning_mean_output_tokens"])
            mean_unit_cost = cfg.calculate_cost_microdollars(spec["model_id"], mean_in, mean_out)
            est_mean_total = len(tier_reqs) * mean_unit_cost
            max_res_total = sum(int(r["reservation"]["reserved_microdollars"]) for r in tier_reqs)
            est_input_total = sum(int(r["reservation"]["counted_input_tokens"]) for r in tier_reqs)

            tier_plans[tier_name] = {
                "model_tier": tier_name,
                "model_id": spec["model_id"],
                "request_count": len(tier_reqs),
                "token_count_source": "offline_heuristic_estimate",
                "estimated_counted_input_tokens_total": est_input_total,
                "planning_mean_estimated_cost_microdollars": est_mean_total,
                "planning_mean_estimated_cost_usd": format_usd(est_mean_total),
                "max_reservation_liability_microdollars": max_res_total,
                "max_reservation_liability_usd": format_usd(max_res_total),
                "request_list_sha256": canonical_sha256([r["custom_id"] for r in tier_reqs]),
            }

        store_summary = db.store_planned_requests(planned_all)
        budget_summary = db.get_budget_summary()
        stage_ceiling = int(cfg.pricing["ceilings_microdollars"]["stages"][stage])
        stage_out_limit = int(cfg.pricing["ceilings_microdollars"]["outstanding_liability_limits"][stage])

        total_mean_est = sum(tp["planning_mean_estimated_cost_microdollars"] for tp in tier_plans.values())
        total_max_res = sum(tp["max_reservation_liability_microdollars"] for tp in tier_plans.values())

        manifests_dir = run_dir / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)

        # Save example request artifacts (first 5 requests) for inspection
        examples_path = manifests_dir / f"example_requests_{stage}_{model_tier}.jsonl"
        example_lines = "".join(
            json.dumps(
                {
                    "custom_id": r["custom_id"],
                    "request_sha256": r["request_sha256"],
                    "uri": r["uri"],
                    "stage": r["stage"],
                    "model_id": r["model_id"],
                    "observation_mode": r["observation_mode"],
                    "token_count_source": r["token_count_source"],
                    "reservation": r["reservation"],
                    "wire_request": r["wire_request"],
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
            for r in planned_all[:5]
        ).encode("utf-8")
        atomic_write(examples_path, example_lines, overwrite=True)

        plan_manifest = {
            "schema": "catbird-anthropic-teacher-plan-manifest-v1",
            "created_at": utc_now_iso(),
            "stage": stage,
            "expansion_wave": expansion_wave,
            "observation_mode": observation_mode,
            "pilot_paired_design": pilot_paired_design,
            "offline_mode": True,
            "paid_api_calls_made": 0,
            "corpus_text_transmitted": False,
            "estimate_disclaimer": (
                "All token and cost figures in this offline plan are estimates. "
                "Exact remote token counting (messages.count_tokens) requires explicit transmission authorization."
            ),
            "total_requests_planned": len(planned_all),
            "store_summary": store_summary,
            "tier_plans": tier_plans,
            "total_planning_mean_estimated_cost_microdollars": total_mean_est,
            "total_planning_mean_estimated_cost_usd": format_usd(total_mean_est),
            "total_max_reservation_liability_microdollars": total_max_res,
            "total_max_reservation_liability_usd": format_usd(total_max_res),
            "stage_ceiling_microdollars": stage_ceiling,
            "stage_ceiling_usd": format_usd(stage_ceiling),
            "stage_outstanding_liability_limit_microdollars": stage_out_limit,
            "stage_outstanding_liability_limit_usd": format_usd(stage_out_limit),
            "example_requests_artifact": str(examples_path.relative_to(run_dir)),
            "example_requests_sha256": sha256(examples_path),
            "budget_summary_snapshot": budget_summary,
            "reviewable_future_pilot_launch_command": (
                "PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-teacher-venv/bin/python "
                "catbird-intel/ops/anthropic_teacher_harness.py submit "
                f"--run-dir {run_dir} --stage pilot --model-tier haiku "
                "--transport live --no-offline "
                "--authorize-paid-submission --authorize-corpus-transmission "
                "--max-stage-spend-usd 8.00 --max-outstanding-usd 2.00"
            ),
        }
        plan_manifest["manifest_sha256"] = canonical_sha256(plan_manifest)
        plan_path = manifests_dir / f"plan_{stage}_{'paired' if pilot_paired_design else model_tier}.json"
        write_immutable_json(plan_path, plan_manifest, overwrite=True)
        return plan_manifest


def run_submit(
    *,
    run_dir: Path,
    stage: str,
    model_tier: str,
    transport: AnthropicTransport | None = None,
    transport_kind: str = "fake",
    offline: bool = True,
    authorize_paid_submission: bool = False,
    authorize_corpus_transmission: bool = False,
    expansion_wave: str = "pilot",
    max_batch_requests: int = 250,
    config: HarnessConfig | None = None,
) -> dict[str, Any]:
    """Atomically reserve budget and submit planned requests in bounded waves respecting outstanding limits."""
    cfg = config or HarnessConfig.load()
    run_dir = Path(run_dir)
    with acquire_writer_lock(run_dir, f"submit:{stage}:{model_tier}"):
        db = HarnessDB(run_dir, cfg)
        db.verify_existing_run_config()
        db.verify_ledger_hash_chain()

        gate = db.check_expansion_gate(expansion_wave)
        if not gate["allowed"]:
            raise BudgetConstraintError(gate["reason"])

        if transport is None:
            if transport_kind == "live":
                transport = OfficialAnthropicSDKTransport(
                    offline=offline,
                    authorize_paid_submission=authorize_paid_submission,
                    authorize_corpus_transmission=authorize_corpus_transmission,
                )
            else:
                transport = FakeAnthropicTransport(run_dir / "fake_provider_state.json")

        spec = cfg.model_spec(model_tier)
        model_id = spec["model_id"]
        # Verify model access before reserving or submitting anything
        model_check = transport.verify_model_available(model_id)

        conn = db.connect()
        try:
            ambiguous = conn.execute(
                "SELECT batch_id, status FROM batches WHERE status IN ('submitting', 'submission_ambiguous', 'unresolved')"
            ).fetchall()
            if ambiguous:
                ids = [f"{r['batch_id']}({r['status']})" for r in ambiguous]
                raise BudgetConstraintError(
                    f"Cannot submit while ambiguous/unresolved batches exist: {ids}. Reconcile first."
                )
            planned_rows = conn.execute(
                """
                SELECT * FROM requests
                WHERE stage = ? AND model_tier = ? AND status = 'planned'
                ORDER BY attempt ASC, custom_id ASC
                """,
                (stage, model_tier),
            ).fetchall()
        finally:
            conn.close()

        if not planned_rows:
            return {
                "stage": stage,
                "model_tier": model_tier,
                "submitted_batches": [],
                "note": "No planned requests waiting for submission",
            }

        # Group planned requests into chunks that fit within both max_batch_requests and outstanding liability limit
        budget_snap = db.get_budget_summary()
        stage_info = budget_snap["stages"][stage]
        avail_outstanding = int(stage_info["outstanding_limit_microdollars"]) - int(
            stage_info["outstanding_reserved_microdollars"]
        )
        avail_stage_budget = int(stage_info["remaining_stage_budget_microdollars"])
        avail_total_gen = (
            int(budget_snap["generation_allocation_ceiling_microdollars"])
            - int(budget_snap["total_settled_microdollars"])
            - int(budget_snap["total_outstanding_reserved_microdollars"])
        )
        effective_headroom = min(avail_outstanding, avail_stage_budget, avail_total_gen)
        if effective_headroom <= 0:
            raise BudgetConstraintError(
                f"Insufficient available headroom ({effective_headroom} microdollars) to submit batch for {stage}/{model_tier}"
            )

        selected_rows = []
        running_res = 0
        for r in planned_rows:
            req_res = int(r["reserved_microdollars"])
            if len(selected_rows) >= max_batch_requests:
                break
            if running_res + req_res > effective_headroom:
                if not selected_rows:
                    raise BudgetConstraintError(
                        f"Single request reservation ({req_res} microdollars) exceeds available headroom ({effective_headroom} microdollars)"
                    )
                break
            selected_rows.append(r)
            running_res += req_res

        cids = [str(r["custom_id"]) for r in selected_rows]
        req_list_sha = canonical_sha256(sorted(cids))
        batch_id = f"batch_{stage}_{model_tier}_{req_list_sha[:12]}"

        batches_dir = run_dir / "batches"
        batches_dir.mkdir(parents=True, exist_ok=True)
        req_manifest_rel = f"batches/{batch_id}.requests.jsonl"
        req_manifest_path = run_dir / req_manifest_rel

        wire_lines = []
        sdk_requests = []
        for r in selected_rows:
            wire_obj = json.loads(r["wire_request_json"])
            assert_no_credentials(wire_obj, f"submit_wire:{r['custom_id']}")
            wire_lines.append(json.dumps(wire_obj, sort_keys=True, ensure_ascii=False) + "\n")
            sdk_requests.append(wire_obj)

        atomic_write(req_manifest_path, "".join(wire_lines).encode("utf-8"), overwrite=False)
        req_manifest_sha = sha256(req_manifest_path)

        # Phase A: Atomically reserve liability and persist batch in 'submitting' state BEFORE API call
        reservation_record = db.reserve_batch_for_submission(
            stage=stage,
            model_tier=model_tier,
            custom_ids=cids,
            transport_mode=transport.mode,
            request_manifest_relpath=req_manifest_rel,
            request_manifest_sha256=req_manifest_sha,
        )

        # Phase B: Call provider Batch API; if crash/exception occurs before ID is saved, mark ambiguous!
        try:
            snap = transport.create_batch(
                sdk_requests,  # type: ignore[arg-type]
                client_submission_id=reservation_record["client_submission_id"],
            )
            db.mark_batch_submitted(batch_id, snap.provider_batch_id)
        except Exception as exc:
            db.mark_batch_submission_ambiguous(batch_id, f"{type(exc).__name__}: {exc}")
            raise

        return {
            "batch_id": batch_id,
            "provider_batch_id": snap.provider_batch_id,
            "stage": stage,
            "model_tier": model_tier,
            "model_id": model_id,
            "model_verified": model_check,
            "transport_mode": transport.mode,
            "request_count": len(cids),
            "reserved_microdollars": reservation_record["reserved_microdollars"],
            "reserved_usd": format_usd(reservation_record["reserved_microdollars"]),
            "remaining_planned_requests": len(planned_rows) - len(selected_rows),
        }


def run_reconcile(
    *,
    run_dir: Path,
    transport: AnthropicTransport | None = None,
    transport_kind: str = "fake",
    offline: bool = True,
    authorize_paid_submission: bool = False,
    authorize_corpus_transmission: bool = False,
    config: HarnessConfig | None = None,
) -> dict[str, Any]:
    """Reconcile ambiguous submissions against provider batch list and refresh batch statuses."""
    cfg = config or HarnessConfig.load()
    run_dir = Path(run_dir)
    with acquire_writer_lock(run_dir, "reconcile"):
        db = HarnessDB(run_dir, cfg)
        db.verify_existing_run_config()
        db.verify_ledger_hash_chain()

        if transport is None:
            if transport_kind == "live":
                transport = OfficialAnthropicSDKTransport(
                    offline=offline,
                    authorize_paid_submission=authorize_paid_submission,
                    authorize_corpus_transmission=authorize_corpus_transmission,
                )
            else:
                transport = FakeAnthropicTransport(run_dir / "fake_provider_state.json")

        conn = db.connect()
        try:
            batches = [dict(r) for r in conn.execute("SELECT * FROM batches ORDER BY created_at ASC").fetchall()]
        finally:
            conn.close()

        provider_batches = transport.list_batches(limit=100)
        recovered_ambiguous: list[dict[str, Any]] = []
        unresolved_ambiguous: list[dict[str, Any]] = []
        status_updates: list[dict[str, Any]] = []

        for b in batches:
            bid = str(b["batch_id"])
            status = str(b["status"])

            if status in ("submitting", "submission_ambiguous") and not b["provider_batch_id"]:
                # Inspect expected custom_ids for this batch
                conn = db.connect()
                try:
                    expected_cids = {
                        str(r["custom_id"])
                        for r in conn.execute("SELECT custom_id FROM requests WHERE batch_id = ?", (bid,)).fetchall()
                    }
                finally:
                    conn.close()

                matched_provider_id: str | None = None
                for pb in provider_batches:
                    if pb.custom_ids_sample and set(pb.custom_ids_sample) == expected_cids:
                        matched_provider_id = pb.provider_batch_id
                        break
                    # Fallback: inspect result stream custom_ids if ended
                    if pb.processing_status == "ended":
                        try:
                            stream_cids = {str(item.get("custom_id")) for item in transport.iter_batch_results(pb.provider_batch_id)}
                            if stream_cids and stream_cids.issubset(expected_cids):
                                matched_provider_id = pb.provider_batch_id
                                break
                        except Exception:
                            pass

                if matched_provider_id:
                    db.mark_batch_submitted(bid, matched_provider_id)
                    recovered_ambiguous.append({
                        "batch_id": bid,
                        "recovered_provider_batch_id": matched_provider_id,
                    })
                else:
                    # Provider acceptance cannot be proven; never blindly resubmit and keep reservation locked!
                    db.mark_batch_unresolved(
                        bid,
                        "Ambiguous submission not found in provider batch list; reservation remains locked",
                    )
                    unresolved_ambiguous.append({"batch_id": bid, "status": "unresolved"})
            elif b["provider_batch_id"] and status == "submitted":
                snap = transport.retrieve_batch(str(b["provider_batch_id"]))
                status_updates.append({
                    "batch_id": bid,
                    "provider_batch_id": snap.provider_batch_id,
                    "provider_processing_status": snap.processing_status,
                    "request_counts": snap.request_counts,
                })

        return {
            "recovered_ambiguous_batches": recovered_ambiguous,
            "unresolved_ambiguous_batches": unresolved_ambiguous,
            "provider_status_updates": status_updates,
            "budget_summary": db.get_budget_summary(),
        }


def run_collect(
    *,
    run_dir: Path,
    transport: AnthropicTransport | None = None,
    transport_kind: str = "fake",
    offline: bool = True,
    authorize_paid_submission: bool = False,
    authorize_corpus_transmission: bool = False,
    config: HarnessConfig | None = None,
) -> dict[str, Any]:
    """Download raw results for ended batches, preserve immutable shards, validate outputs, and settle ledger."""
    cfg = config or HarnessConfig.load()
    run_dir = Path(run_dir)
    with acquire_writer_lock(run_dir, "collect"):
        db = HarnessDB(run_dir, cfg)
        db.verify_existing_run_config()
        db.verify_ledger_hash_chain()

        if transport is None:
            if transport_kind == "live":
                transport = OfficialAnthropicSDKTransport(
                    offline=offline,
                    authorize_paid_submission=authorize_paid_submission,
                    authorize_corpus_transmission=authorize_corpus_transmission,
                )
            else:
                transport = FakeAnthropicTransport(run_dir / "fake_provider_state.json")

        posts_by_uri = {p["uri"]: p for p in db.load_posts()}

        conn = db.connect()
        try:
            submitted_batches = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM batches WHERE status = 'submitted' AND provider_batch_id IS NOT NULL ORDER BY created_at ASC"
                ).fetchall()
            ]
        finally:
            conn.close()

        collected_summaries: list[dict[str, Any]] = []
        for b in submitted_batches:
            bid = str(b["batch_id"])
            pbid = str(b["provider_batch_id"])
            snap = transport.retrieve_batch(pbid)
            if snap.processing_status != "ended":
                continue

            raw_items = list(transport.iter_batch_results(pbid))
            results_rel = f"batches/{bid}.results.jsonl"
            results_path = run_dir / results_rel
            raw_bytes = "".join(
                json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n" for item in raw_items
            ).encode("utf-8")
            assert_no_credentials(raw_items, f"batch_results:{bid}")
            atomic_write(results_path, raw_bytes, overwrite=False)
            results_sha = sha256(results_path)

            conn = db.connect()
            try:
                req_map = {
                    str(r["custom_id"]): dict(r)
                    for r in conn.execute("SELECT * FROM requests WHERE batch_id = ?", (bid,)).fetchall()
                }
            finally:
                conn.close()

            reconciled_items: list[dict[str, Any]] = []
            for raw_item in raw_items:
                cid = str(raw_item.get("custom_id") or "")
                if cid not in req_map:
                    raise ValueError(f"Provider returned unknown custom_id {cid!r} in batch {bid}")
                req_row = req_map[cid]
                expected_model = str(req_row["teacher_model"])
                uri = str(req_row["uri"])
                post = posts_by_uri[uri]

                wire_req = json.loads(req_row["wire_request_json"])
                user_env = json.loads(wire_req["params"]["messages"][0]["content"])
                post_text = str(user_env.get("untrusted_post_text") or post["text"])
                ctx_dict = user_env.get("untrusted_context") or {}
                ctx_texts = [str(v) for v in ctx_dict.values() if v]

                item_sha = canonical_sha256(raw_item)
                res_obj = raw_item.get("result") or {}
                res_type = str(res_obj.get("type") or "")
                is_sim = bool(raw_item.get("simulated") or transport.mode == "fake_offline")

                if res_type == "succeeded":
                    msg = res_obj.get("message") or {}
                    returned_model = str(msg.get("model") or "")
                    stop_reason = str(msg.get("stop_reason") or "")
                    stop_details = msg.get("stop_details")
                    usage = msg.get("usage") or {}
                    in_tok = int(usage.get("input_tokens") or 0)
                    out_tok = int(usage.get("output_tokens") or 0)
                    billed_micro = cfg.calculate_cost_microdollars(expected_model, in_tok, out_tok)

                    content_blocks = msg.get("content") or []
                    text_block = next(
                        (str(blk.get("text", "")) for blk in content_blocks if isinstance(blk, dict) and blk.get("type") == "text"),
                        "",
                    )

                    if returned_model != expected_model:
                        reconciled_items.append({
                            "custom_id": cid,
                            "status": "invalid",
                            "returned_model": returned_model,
                            "stop_reason": stop_reason,
                            "stop_details": stop_details,
                            "actual_input_tokens": in_tok,
                            "actual_output_tokens": out_tok,
                            "actual_billed_microdollars": billed_micro,
                            "response_sha256": item_sha,
                            "parsed_output": None,
                            "has_insufficient_component": None,
                            "error_type": "model_identity_mismatch",
                            "error_message": f"Expected model {expected_model!r}, got {returned_model!r}",
                            "simulated": is_sim,
                        })
                    elif stop_reason == "refusal":
                        reconciled_items.append({
                            "custom_id": cid,
                            "status": "refused",
                            "returned_model": returned_model,
                            "stop_reason": stop_reason,
                            "stop_details": stop_details,
                            "actual_input_tokens": in_tok,
                            "actual_output_tokens": out_tok,
                            "actual_billed_microdollars": billed_micro,
                            "response_sha256": item_sha,
                            "parsed_output": None,
                            "has_insufficient_component": None,
                            "error_type": "refusal",
                            "error_message": "Model refused request",
                            "simulated": is_sim,
                        })
                    elif stop_reason == "max_tokens":
                        reconciled_items.append({
                            "custom_id": cid,
                            "status": "truncated",
                            "returned_model": returned_model,
                            "stop_reason": stop_reason,
                            "stop_details": stop_details,
                            "actual_input_tokens": in_tok,
                            "actual_output_tokens": out_tok,
                            "actual_billed_microdollars": billed_micro,
                            "response_sha256": item_sha,
                            "parsed_output": None,
                            "has_insufficient_component": None,
                            "error_type": "truncated_max_tokens",
                            "error_message": "Model hit max_tokens before completing structured JSON",
                            "simulated": is_sim,
                        })
                    else:
                        try:
                            validated = validate_teacher_output_json(
                                text_block,
                                post_text=post_text,
                                context_texts=ctx_texts,
                            )
                            reconciled_items.append({
                                "custom_id": cid,
                                "status": "succeeded",
                                "returned_model": returned_model,
                                "stop_reason": stop_reason,
                                "stop_details": stop_details,
                                "actual_input_tokens": in_tok,
                                "actual_output_tokens": out_tok,
                                "actual_billed_microdollars": billed_micro,
                                "response_sha256": item_sha,
                                "parsed_output": validated.model_dump(mode="json"),
                                "has_insufficient_component": validated.has_insufficient_component(),
                                "error_type": None,
                                "error_message": None,
                                "simulated": is_sim,
                            })
                        except SchemaValidationError as exc:
                            reconciled_items.append({
                                "custom_id": cid,
                                "status": "invalid",
                                "returned_model": returned_model,
                                "stop_reason": stop_reason,
                                "stop_details": stop_details,
                                "actual_input_tokens": in_tok,
                                "actual_output_tokens": out_tok,
                                "actual_billed_microdollars": billed_micro,
                                "response_sha256": item_sha,
                                "parsed_output": None,
                                "has_insufficient_component": None,
                                "error_type": "schema_validation_error",
                                "error_message": str(exc),
                                "simulated": is_sim,
                            })
                elif res_type == "errored":
                    err = res_obj.get("error") or {}
                    err_type = str(err.get("type") or "errored_unknown")
                    err_msg = str(err.get("message") or "Batch request errored")
                    reconciled_items.append({
                        "custom_id": cid,
                        "status": "errored",
                        "returned_model": None,
                        "stop_reason": None,
                        "stop_details": None,
                        "actual_input_tokens": 0,
                        "actual_output_tokens": 0,
                        "actual_billed_microdollars": 0,
                        "response_sha256": item_sha,
                        "parsed_output": None,
                        "has_insufficient_component": None,
                        "error_type": err_type,
                        "error_message": err_msg,
                        "simulated": is_sim,
                    })
                elif res_type == "expired":
                    reconciled_items.append({
                        "custom_id": cid,
                        "status": "expired",
                        "returned_model": None,
                        "stop_reason": None,
                        "stop_details": None,
                        "actual_input_tokens": 0,
                        "actual_output_tokens": 0,
                        "actual_billed_microdollars": 0,
                        "response_sha256": item_sha,
                        "parsed_output": None,
                        "has_insufficient_component": None,
                        "error_type": "expired",
                        "error_message": "Batch reached 24h expiration",
                        "simulated": is_sim,
                    })
                elif res_type == "canceled":
                    reconciled_items.append({
                        "custom_id": cid,
                        "status": "canceled",
                        "returned_model": None,
                        "stop_reason": None,
                        "stop_details": None,
                        "actual_input_tokens": 0,
                        "actual_output_tokens": 0,
                        "actual_billed_microdollars": 0,
                        "response_sha256": item_sha,
                        "parsed_output": None,
                        "has_insufficient_component": None,
                        "error_type": "canceled",
                        "error_message": "Request canceled before execution",
                        "simulated": is_sim,
                    })
                else:
                    reconciled_items.append({
                        "custom_id": cid,
                        "status": "unresolved",
                        "returned_model": None,
                        "stop_reason": None,
                        "stop_details": None,
                        "actual_input_tokens": None,
                        "actual_output_tokens": None,
                        "actual_billed_microdollars": None,
                        "response_sha256": item_sha,
                        "parsed_output": None,
                        "has_insufficient_component": None,
                        "error_type": f"unknown_result_type:{res_type}",
                        "error_message": "Unknown batch result type; reservation held",
                        "simulated": is_sim,
                    })

            settle_summary = db.settle_batch_results(
                batch_id=bid,
                raw_results_relpath=results_rel,
                raw_results_sha256=results_sha,
                reconciled_items=reconciled_items,
            )
            collected_summaries.append(settle_summary)

        return {
            "collected_batches": collected_summaries,
            "budget_summary": db.get_budget_summary(),
        }


def run_report(*, run_dir: Path, config: HarnessConfig | None = None) -> dict[str, Any]:
    """Generate comprehensive JSON and Markdown reports covering costs, coverage, disagreement, and gates."""
    cfg = config or HarnessConfig.load()
    run_dir = Path(run_dir)
    db = HarnessDB(run_dir, cfg)
    meta = db.verify_existing_run_config()
    ledger_valid = db.verify_ledger_hash_chain()
    budget = db.get_budget_summary()

    conn = db.connect()
    try:
        req_rows = [dict(r) for r in conn.execute("SELECT * FROM requests").fetchall()]
        batch_rows = [dict(r) for r in conn.execute("SELECT * FROM batches").fetchall()]
        routing_rows = [dict(r) for r in conn.execute("SELECT * FROM routing_decisions").fetchall()]
        gate_rows = [dict(r) for r in conn.execute("SELECT * FROM stage_gate_evidence").fetchall()]
    finally:
        conn.close()

    status_counts = Counter(r["status"] for r in req_rows)
    stage_model_counts = Counter(f"{r['stage']}/{r['model_tier']}:{r['status']}" for r in req_rows)
    any_simulated = any(int(r.get("simulated") or 0) == 1 for r in req_rows)

    # Component score distributions by tier
    dist_by_tier: dict[str, dict[str, Counter[str]]] = {}
    by_uri_obs: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for r in req_rows:
        if r["status"] == "succeeded" and r["parsed_output_json"]:
            parsed = json.loads(r["parsed_output_json"])
            tier = str(r["model_tier"])
            for comp in COMPONENT_ORDER:
                sc = str(parsed[comp]["score"])
                dist_by_tier.setdefault(tier, {}).setdefault(comp, Counter())[sc] += 1
            by_uri_obs.setdefault((str(r["uri"]), str(r["observation_mode"])), {})[tier] = parsed

    disagreement_counts = Counter()
    comp_disagreement_counts: dict[str, Counter[str]] = {c: Counter() for c in COMPONENT_ORDER}
    multi_teacher_items = 0
    for (_, _), t_map in by_uri_obs.items():
        if len(t_map) >= 2:
            multi_teacher_items += 1
            d_eval = evaluate_post_disagreement(t_map)
            disagreement_counts[d_eval["overall_status"]] += 1
            for comp in COMPONENT_ORDER:
                comp_disagreement_counts[comp][d_eval["per_component"][comp]["status"]] += 1

    # Operational schema-validity gate computation
    succeeded_msg_calls = sum(status_counts[k] for k in ("succeeded", "truncated", "invalid"))
    schema_valid_rate = (
        round(status_counts["succeeded"] / float(succeeded_msg_calls), 6)
        if succeeded_msg_calls > 0
        else 0.0
    )

    prep_manifest_path = run_dir / "manifests" / "preparation_manifest.json"
    prep_manifest = json.loads(prep_manifest_path.read_text(encoding="utf-8")) if prep_manifest_path.exists() else {}

    report_payload = {
        "schema": "catbird-anthropic-teacher-report-v1",
        "generated_at": utc_now_iso(),
        "run_metadata": meta,
        "simulated_dry_run_responses_present": any_simulated,
        "paid_api_calls_made": 0 if any_simulated or not batch_rows else sum(1 for b in batch_rows if b["transport_mode"] == "live_anthropic"),
        "ledger_hash_chain_verified": ledger_valid,
        "budget_summary": {
            **budget,
            "total_cash_ceiling_usd": format_usd(budget["total_cash_ceiling_microdollars"]),
            "total_settled_usd": format_usd(budget["total_settled_microdollars"]),
            "total_outstanding_reserved_usd": format_usd(budget["total_outstanding_reserved_microdollars"]),
            "remaining_total_cash_usd": format_usd(budget["remaining_total_cash_microdollars"]),
        },
        "request_status_counts": dict(sorted(status_counts.items())),
        "stage_model_status_counts": dict(sorted(stage_model_counts.items())),
        "operational_quality": {
            "schema_valid_non_truncated_rate": schema_valid_rate,
            "provisional_gate_threshold": 0.99,
            "operational_gate_passed": schema_valid_rate >= 0.99 and not budget["budget_halted"],
            "insufficient_component_requests": sum(1 for r in req_rows if r.get("has_insufficient_component") == 1),
            "refused_requests": status_counts.get("refused", 0),
        },
        "component_score_distributions": {
            tier: {comp: dict(sorted(cnt.items())) for comp, cnt in comp_map.items()}
            for tier, comp_map in sorted(dist_by_tier.items())
        },
        "disagreement_summary": {
            "multi_teacher_evaluated_items": multi_teacher_items,
            "overall_status_counts": dict(sorted(disagreement_counts.items())),
            "per_component_status_counts": {
                comp: dict(sorted(cnt.items())) for comp, cnt in comp_disagreement_counts.items()
            },
            "opus_or_majority_treated_as_ground_truth": False,
        },
        "routing_summary": {
            "total_routed_items": len(routing_rows),
            "by_stage_and_bucket": dict(
                sorted(Counter(f"{r['target_stage']}/{r['model_tier']}:{r['routing_bucket']}" for r in routing_rows).items())
            ),
        },
        "moderation_join_audit": prep_manifest.get("moderation_join_audit"),
        "recorded_stage_gates": gate_rows,
        "remaining_gates_and_blockers": [
            "Paid Anthropic Batch API execution is not authorized in this task; all current responses in dry-run are simulated (FakeAnthropicTransport).",
            "Exact remote token counting (client.messages.count_tokens) requires explicit transmission authorization before live submission.",
            "Human development (~1,000 items) and human final (~2,000 items, >=400 positives per principal target, double-labeled >=200 with Krippendorff alpha >= 0.80) remain mandatory before student qualification.",
            "Only politicalness has a locked final lower-bound F1 gate (>=0.90); outraged, critical, engagement_bait, and spam remain diagnostic/unqualified until their own gates are preregistered.",
        ],
        "commands": {
            "future_authorized_pilot_launch_capped_at_8_usd": (
                "PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-teacher-venv/bin/python "
                "catbird-intel/ops/anthropic_teacher_harness.py submit "
                f"--run-dir {run_dir} --stage pilot --model-tier haiku "
                "--transport live --no-offline "
                "--authorize-paid-submission --authorize-corpus-transmission"
            ),
            "reconcile_and_recover": (
                "PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-teacher-venv/bin/python "
                f"catbird-intel/ops/anthropic_teacher_harness.py reconcile --run-dir {run_dir}"
            ),
            "collect_results": (
                "PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-teacher-venv/bin/python "
                f"catbird-intel/ops/anthropic_teacher_harness.py collect --run-dir {run_dir}"
            ),
            "recover_interrupted_partials": (
                "PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-teacher-venv/bin/python "
                f"catbird-intel/ops/anthropic_teacher_harness.py recover-partials --run-dir {run_dir}"
            ),
        },
    }

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_immutable_json(reports_dir / "summary_report.json", report_payload, overwrite=True)

    md_lines = [
        "# Catbird Anthropic Teacher Harness — Run Report",
        "",
        f"- **Run ID**: `{meta['run_id']}` (Generation `{meta['generation_id']}`)",
        f"- **Config Bundle SHA-256**: `{meta['config_bundle_sha256']}`",
        f"- **Simulated Dry-Run Responses Present**: `{any_simulated}` (Never usable as real teacher inference)",
        f"- **Paid API Calls Executed**: `{report_payload['paid_api_calls_made']}`",
        f"- **Ledger Hash Chain Verified**: `{ledger_valid}`",
        "",
        "## Budget & Liability Ledger",
        f"- **Total Cash Ceiling**: `{report_payload['budget_summary']['total_cash_ceiling_usd']}`",
        f"- **Total Settled Cost**: `{report_payload['budget_summary']['total_settled_usd']}`",
        f"- **Total Outstanding Reservations**: `{report_payload['budget_summary']['total_outstanding_reserved_usd']}`",
        f"- **Remaining Cash Ceiling**: `{report_payload['budget_summary']['remaining_total_cash_usd']}`",
        f"- **Budget Halted**: `{budget['budget_halted']}`",
        "",
        "## Request Statuses & Multi-Teacher Disagreement",
        f"- **Request Status Counts**: `{json.dumps(report_payload['request_status_counts'])}`",
        f"- **Schema-Valid Non-Truncated Rate**: `{schema_valid_rate:.4f}` (Gate `>= 0.99`)",
        f"- **Multi-Teacher Items Compared**: `{multi_teacher_items}`",
        f"- **Disagreement Breakdown**: `{json.dumps(report_payload['disagreement_summary']['overall_status_counts'])}`",
        "",
        "## Explicit Future Pilot Launch Command (Capped at $8 Pilot / $2 Outstanding)",
        "```sh",
        report_payload["commands"]["future_authorized_pilot_launch_capped_at_8_usd"],
        "```",
        "",
        "## Resume & Crash Recovery Commands",
        "```sh",
        report_payload["commands"]["recover_interrupted_partials"],
        report_payload["commands"]["reconcile_and_recover"],
        report_payload["commands"]["collect_results"],
        "```",
    ]
    atomic_write(reports_dir / "summary_report.md", ("\n".join(md_lines) + "\n").encode("utf-8"), overwrite=True)
    return report_payload
