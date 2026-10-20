"""Deterministic review routing (Sonnet/Opus quotas), blind prompt creation, and disagreement analysis."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
import json
from typing import Any

from .config import HarnessConfig, canonical_sha256
from .db import HarnessDB, utc_now_iso
from .schemas import COMPONENT_ORDER, build_batch_request_for_post


def _det_priority(seed: str, bucket: str, uri: str) -> str:
    return hashlib.sha256(f"{seed}:{bucket}:{uri}".encode("utf-8")).hexdigest()


def _haiku_difficulty_signals(haiku_out: dict[str, Any] | None, post: dict[str, Any]) -> dict[str, Any]:
    if not haiku_out:
        return {"is_difficult": False, "is_rare_or_policy": False, "reasons": []}

    reasons: list[str] = []
    scores = {c: haiku_out[c]["score"] for c in COMPONENT_ORDER}
    any_insuff = any(s == "insufficient" for s in scores.values())
    if any_insuff:
        reasons.append("insufficient_component")
    if haiku_out.get("context_dependent") is True:
        reasons.append("context_dependent")
    if haiku_out.get("stance_ambiguity", "none") != "none":
        reasons.append(f"stance_ambiguity:{haiku_out['stance_ambiguity']}")
    if any(s in (1, 2) for s in scores.values() if isinstance(s, int)):
        reasons.append("borderline_ordinal_1_or_2")

    # Non-political criticism vs political boundary
    if isinstance(scores["critical"], int) and scores["critical"] >= 2 and scores["politicalness"] in (0, 1):
        reasons.append("nonpolitical_or_borderline_criticism")

    mod_join = post.get("moderation_join") or {}
    has_mod = bool(mod_join.get("active_record_labels")) or bool(mod_join.get("source_disagreement"))
    is_rare_positive = any(
        isinstance(scores[c], int) and scores[c] >= 1
        for c in ("spam", "engagement_bait", "outraged")
    )
    if has_mod:
        reasons.append("moderation_source_signal")
    if is_rare_positive:
        reasons.append("rare_positive_component")

    return {
        "is_difficult": bool(
            any_insuff
            or haiku_out.get("context_dependent")
            or haiku_out.get("stance_ambiguity", "none") != "none"
            or any(s in (1, 2) for s in scores.values() if isinstance(s, int))
        ),
        "is_rare_or_policy": bool(has_mod or is_rare_positive),
        "reasons": reasons,
    }


def _pair_disagreement_signals(
    haiku_out: dict[str, Any] | None,
    sonnet_out: dict[str, Any] | None,
    post: dict[str, Any],
) -> dict[str, Any]:
    if not haiku_out or not sonnet_out:
        return {"is_unresolved_high_value": False, "is_rare_boundary": False, "reasons": []}

    reasons: list[str] = []
    max_diff = 0
    insuff_mismatch = False
    for c in COMPONENT_ORDER:
        hs = haiku_out[c]["score"]
        ss = sonnet_out[c]["score"]
        if (hs == "insufficient") != (ss == "insufficient"):
            insuff_mismatch = True
            reasons.append(f"insufficiency_mismatch:{c}({hs}_vs_{ss})")
        elif isinstance(hs, int) and isinstance(ss, int):
            diff = abs(hs - ss)
            if diff > max_diff:
                max_diff = diff
            if diff >= 1:
                reasons.append(f"ordinal_disagreement:{c}({hs}_vs_{ss})")

    is_unresolved = bool(max_diff >= 1 or insuff_mismatch)
    is_boundary = bool(
        haiku_out.get("stance_ambiguity", "none") != "none"
        or sonnet_out.get("stance_ambiguity", "none") != "none"
        or (
            isinstance(haiku_out["critical"]["score"], int)
            and haiku_out["critical"]["score"] >= 2
            and haiku_out["politicalness"]["score"] == 0
        )
        or (
            isinstance(sonnet_out["engagement_bait"]["score"], int)
            and sonnet_out["engagement_bait"]["score"] >= 1
        )
    )
    if is_boundary:
        reasons.append("rubric_boundary_case")
    return {
        "is_unresolved_high_value": is_unresolved,
        "is_rare_boundary": is_boundary,
        "max_ordinal_diff": max_diff,
        "reasons": reasons,
    }


def route_review_candidates(
    db: HarnessDB,
    config: HarnessConfig,
    *,
    target_stage: str,  # "sonnet_review" | "opus_review" | "pilot"
    total_quota: int,
    seed: str = "catbird-routing-v1",
    observation_mode: str = "text_only",
) -> dict[str, Any]:
    """Select review candidates reproducibly using configured quotas and generate blind prompts."""
    if total_quota <= 0:
        raise ValueError("total_quota must be positive")

    meta = db.verify_existing_run_config()
    posts = db.load_posts(split="train", eligible_only=True)
    posts_by_uri = {p["uri"]: p for p in posts}
    all_posts_by_uri = {p["uri"]: p for p in db.load_posts()}

    conn = db.connect()
    try:
        req_rows = conn.execute(
            """
            SELECT uri, model_tier, parsed_output_json
            FROM requests
            WHERE status = 'succeeded' AND observation_mode = ? AND parsed_output_json IS NOT NULL
            ORDER BY attempt DESC
            """,
            (observation_mode,),
        ).fetchall()
    finally:
        conn.close()

    by_uri_tier: dict[str, dict[str, dict[str, Any]]] = {}
    for r in req_rows:
        u = str(r["uri"])
        t = str(r["model_tier"])
        if u in posts_by_uri and t not in by_uri_tier.setdefault(u, {}):
            by_uri_tier[u][t] = json.loads(r["parsed_output_json"])

    eligible_uris = sorted(u for u in by_uri_tier if "haiku" in by_uri_tier[u])
    if not eligible_uris:
        raise ValueError("No succeeded Haiku observations available in train split to route from")

    selected_items: list[dict[str, Any]] = []
    selected_uris: set[str] = set()
    bucket_counts: Counter[str] = Counter()

    if target_stage in ("sonnet_review", "pilot_sonnet"):
        model_tier = "sonnet"
        budget_stage = "pilot" if target_stage == "pilot_sonnet" else "sonnet_review"
        shares = config.pricing["routing_quotas"]["sonnet_review"]["shares"]
        # 1. Compute random_audit FIRST and independently of Haiku confidence/difficulty!
        q_random = int((Decimal(total_quota) * Decimal(str(shares["random_audit"]))).to_integral_value())
        q_rare = int((Decimal(total_quota) * Decimal(str(shares["rare_or_source_policy"]))).to_integral_value())
        q_diff = max(0, total_quota - q_random - q_rare)

        # Independent random audit over ALL eligible Haiku-scored items
        random_ordered = sorted(
            eligible_uris,
            key=lambda u: (_det_priority(seed, "sonnet_random_audit", u), u),
        )
        for u in random_ordered[:q_random]:
            selected_uris.add(u)
            bucket_counts["random_audit"] += 1
            selected_items.append({
                "uri": u,
                "routing_bucket": "random_audit",
                "priority_key": _det_priority(seed, "sonnet_random_audit", u),
                "rationale": {
                    "selected_independently_of_haiku_confidence": True,
                    "quota_share": str(shares["random_audit"]),
                },
            })

        # 2. Difficult / disagreement candidates
        diff_candidates: list[tuple[str, str, list[str]]] = []
        rare_candidates: list[tuple[str, str, list[str]]] = []
        for u in eligible_uris:
            if u in selected_uris:
                continue
            sig = _haiku_difficulty_signals(by_uri_tier[u]["haiku"], posts_by_uri[u])
            if sig["is_difficult"]:
                diff_candidates.append((_det_priority(seed, "sonnet_difficult", u), u, sig["reasons"]))
            if sig["is_rare_or_policy"]:
                rare_candidates.append((_det_priority(seed, "sonnet_rare", u), u, sig["reasons"]))

        diff_candidates.sort()
        for pkey, u, reasons in diff_candidates[:q_diff]:
            selected_uris.add(u)
            bucket_counts["difficult_or_disagreement"] += 1
            selected_items.append({
                "uri": u,
                "routing_bucket": "difficult_or_disagreement",
                "priority_key": pkey,
                "rationale": {"reasons": reasons, "quota_share": str(shares["difficult_or_disagreement"])},
            })

        # 3. Rare / source-policy candidates
        rare_candidates = [(pk, u, r) for pk, u, r in rare_candidates if u not in selected_uris]
        rare_candidates.sort()
        for pkey, u, reasons in rare_candidates[:q_rare]:
            selected_uris.add(u)
            bucket_counts["rare_or_source_policy"] += 1
            selected_items.append({
                "uri": u,
                "routing_bucket": "rare_or_source_policy",
                "priority_key": pkey,
                "rationale": {"reasons": reasons, "quota_share": str(shares["rare_or_source_policy"])},
            })

        # Report underfilled heuristic buckets honestly; only fill if pilot_sonnet overlap explicitly requested
        requested_by_bucket = {
            "difficult_or_disagreement": q_diff,
            "rare_or_source_policy": q_rare,
            "random_audit": q_random,
        }

    elif target_stage in ("opus_review", "pilot_opus"):
        model_tier = "opus"
        budget_stage = "pilot" if target_stage == "pilot_opus" else "opus_review"
        shares = config.pricing["routing_quotas"]["opus_review"]["shares"]
        q_ctrl = int((Decimal(total_quota) * Decimal(str(shares["random_control"]))).to_integral_value())
        q_bound = int((Decimal(total_quota) * Decimal(str(shares["rare_rubric_boundary"]))).to_integral_value())
        q_unres = max(0, total_quota - q_ctrl - q_bound)

        paired_uris = sorted(u for u in eligible_uris if "sonnet" in by_uri_tier[u])
        pool_uris = paired_uris if paired_uris else eligible_uris

        # 1. Independent random controls first
        ctrl_ordered = sorted(
            pool_uris,
            key=lambda u: (_det_priority(seed, "opus_random_control", u), u),
        )
        for u in ctrl_ordered[:q_ctrl]:
            selected_uris.add(u)
            bucket_counts["random_control"] += 1
            selected_items.append({
                "uri": u,
                "routing_bucket": "random_control",
                "priority_key": _det_priority(seed, "opus_random_control", u),
                "rationale": {
                    "selected_independently_of_teacher_disagreement": True,
                    "quota_share": str(shares["random_control"]),
                },
            })

        # 2. Unresolved high-value disagreement cases
        unres_candidates: list[tuple[int, str, str, list[str]]] = []
        bound_candidates: list[tuple[str, str, list[str]]] = []
        for u in pool_uris:
            if u in selected_uris:
                continue
            s_out = by_uri_tier[u].get("sonnet") or by_uri_tier[u]["haiku"]
            sig = _pair_disagreement_signals(by_uri_tier[u]["haiku"], s_out, posts_by_uri[u])
            if sig["is_unresolved_high_value"]:
                unres_candidates.append(
                    (-int(sig.get("max_ordinal_diff", 0)), _det_priority(seed, "opus_unresolved", u), u, sig["reasons"])
                )
            if sig["is_rare_boundary"]:
                bound_candidates.append((_det_priority(seed, "opus_boundary", u), u, sig["reasons"]))

        unres_candidates.sort()
        for _, pkey, u, reasons in unres_candidates[:q_unres]:
            selected_uris.add(u)
            bucket_counts["unresolved_high_value"] += 1
            selected_items.append({
                "uri": u,
                "routing_bucket": "unresolved_high_value",
                "priority_key": pkey,
                "rationale": {"reasons": reasons, "quota_share": str(shares["unresolved_high_value"])},
            })

        bound_candidates = [(pk, u, r) for pk, u, r in bound_candidates if u not in selected_uris]
        bound_candidates.sort()
        for pkey, u, reasons in bound_candidates[:q_bound]:
            selected_uris.add(u)
            bucket_counts["rare_rubric_boundary"] += 1
            selected_items.append({
                "uri": u,
                "routing_bucket": "rare_rubric_boundary",
                "priority_key": pkey,
                "rationale": {"reasons": reasons, "quota_share": str(shares["rare_rubric_boundary"])},
            })

        requested_by_bucket = {
            "unresolved_high_value": q_unres,
            "rare_rubric_boundary": q_bound,
            "random_control": q_ctrl,
        }
    else:
        raise ValueError(f"Unsupported routing target_stage: {target_stage!r}")

    # Build BLIND-FIRST requests for selected items and persist routing decisions + planned requests
    planned_requests: list[dict[str, Any]] = []
    for item in selected_items:
        u = item["uri"]
        post_row = posts_by_uri[u]
        req = build_batch_request_for_post(
            post_row,
            config=config,
            stage=budget_stage,
            model_tier=model_tier,  # type: ignore[arg-type]
            observation_mode=observation_mode,  # type: ignore[arg-type]
            attempt=1,
            generation_id=meta["generation_id"],
            corpus_by_uri=all_posts_by_uri,
        )
        # Verify blind-first invariant: prompt never contains Haiku or Sonnet scores
        wire_msg = req["wire_request"]["params"]["messages"][0]["content"]
        if "haiku_output" in wire_msg or "sonnet_output" in wire_msg or "prior_teacher" in wire_msg:
            raise ValueError(f"Blind-first routing invariant violated on {u}")
        planned_requests.append(req)

    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE;")
        now = utc_now_iso()
        for item in selected_items:
            conn.execute(
                """
                INSERT OR REPLACE INTO routing_decisions (
                    uri, target_stage, model_tier, routing_bucket,
                    priority_key, rationale_json, blind_prompt_verified, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    item["uri"],
                    budget_stage,
                    model_tier,
                    item["routing_bucket"],
                    item["priority_key"],
                    json.dumps(item["rationale"], sort_keys=True),
                    now,
                ),
            )
        conn.execute("COMMIT;")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()

    store_res = db.store_planned_requests(planned_requests)
    return {
        "schema": "catbird-teacher-routing-manifest-v1",
        "target_stage": target_stage,
        "budget_stage": budget_stage,
        "model_tier": model_tier,
        "requested_total_quota": total_quota,
        "selected_total": len(selected_items),
        "shortfall": max(0, total_quota - len(selected_items)),
        "requested_by_bucket": requested_by_bucket,
        "selected_by_bucket": dict(sorted(bucket_counts.items())),
        "blind_first_pass_enforced": True,
        "planned_requests_stored": store_res,
        "routing_sha256": canonical_sha256([item["uri"] for item in selected_items]),
    }


def evaluate_post_disagreement(teacher_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare retained teacher outputs (haiku, sonnet, opus) without declaring Opus or majority ground truth."""
    tiers = sorted(teacher_outputs.keys())
    per_component: dict[str, dict[str, Any]] = {}
    any_unresolved = False
    any_insufficient = False

    for comp in COMPONENT_ORDER:
        comp_scores = {t: teacher_outputs[t][comp]["score"] for t in tiers}
        insuff_tiers = [t for t, s in comp_scores.items() if s == "insufficient"]
        numeric_scores = [int(s) for s in comp_scores.values() if isinstance(s, int)]

        if insuff_tiers and numeric_scores:
            # Mixed insufficient vs numeric: unresolved if any numeric score >= 1, else insufficient
            unresolved = any(n >= 1 for n in numeric_scores)
            status = "unresolved_insufficiency_disagreement" if unresolved else "insufficient_evidence"
            spread = None
        elif insuff_tiers:
            unresolved = False
            status = "insufficient_evidence"
            spread = None
        elif len(numeric_scores) >= 2:
            spread = max(numeric_scores) - min(numeric_scores)
            has_pos = any(n >= 1 for n in numeric_scores)
            has_zero = any(n == 0 for n in numeric_scores)
            # Disagreement of >=2 ordinal steps or binary presence/absence (0 vs >=2) is unresolved
            unresolved = spread >= 2 or (has_zero and has_pos and spread >= 1 and len(set(numeric_scores)) > 1)
            status = "unresolved_disagreement" if unresolved else ("exact_agreement" if spread == 0 else "minor_ordinal_adjacent")
        elif len(numeric_scores) == 1:
            unresolved = False
            status = "single_teacher_observation"
            spread = 0
        else:
            unresolved = False
            status = "no_observation"
            spread = None

        if insuff_tiers:
            any_insufficient = True
        if unresolved:
            any_unresolved = True

        per_component[comp] = {
            "scores_by_tier": comp_scores,
            "status": status,
            "unresolved": unresolved,
            "ordinal_spread": spread,
        }

    overall_status = (
        "unresolved_disagreement"
        if any_unresolved
        else ("insufficient_evidence" if any_insufficient else "resolved_or_single_teacher")
    )
    return {
        "tiers_observed": tiers,
        "overall_status": overall_status,
        "has_unresolved_disagreement": any_unresolved,
        "has_insufficient_component": any_insufficient,
        "ground_truth_claimed": False,
        "opus_or_majority_declared_truth": False,
        "per_component": per_component,
    }
