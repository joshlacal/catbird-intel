#!/usr/bin/env python3
"""Evaluate reported soak evidence offline; never contacts or mutates a service.

Input: JSON object with `gates` mapping names below to evidence objects.
Each requires status=measured, source_receipt, observed_start/end and metrics.
Missing evidence is UNPERFORMED, malformed evidence UNKNOWN. This validates
reported values, not truth of collection. JSON stdout is a review receipt.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

RULES = {
    "continuity": {"coverage": (">=", .999)},
    "final_24h_continuity": {"unexplained_gaps_over_60s": ("<=", 0)},
    "lag": {"p99_ms": ("<=", 5000), "max_ms": ("<=", 60000)},
    "cursor_completeness": {"unresolved_notices": ("<=", 0), "undisposed_skips": ("<=", 0)},
    "delete_invariant": {"visible_deleted_records": ("<=", 0)},
    "visibility_latency": {"sample_count": (">=", 100), "p99_ms": ("<=", 5000), "max_ms": ("<=", 10000)},
    "controlled_create_delete": {"create_observed": (">=", 1), "delete_observed": (">=", 1)},
    "cross_archive": {"sample_count": (">=", 1), "unexplained_conflicts": ("<=", 0), "resurrections": ("<=", 0)},
    "resource_pressure": {"failed_writes": ("<=", 0), "parts_limit_errors": ("<=", 0), "flush_p99_ms": ("<=", 1500), "partition_parts_limit_fraction": ("<=", .5)},
    "merge_trend": {"monotone_growth_windows_6h": ("<=", 0)},
    "disk": {"free_fraction": (">=", .25), "projected_used_with_scratch_fraction": ("<=", .70)},
    "production_interference": {"p99_baseline_ratio": ("<=", 2), "new_service_errors": ("<=", 0)},
}


SOAK_END = datetime(2026, 10, 22, 13, tzinfo=timezone.utc)
WINDOW_HOURS = {"continuity": 72, "final_24h_continuity": 24, "lag": 24,
                "cursor_completeness": 72, "delete_invariant": 24,
                "visibility_latency": 24, "resource_pressure": 24,
                "merge_trend": 6, "disk": 24, "production_interference": 24}
COUNTS = {"unexplained_gaps_over_60s", "unresolved_notices", "undisposed_skips",
          "visible_deleted_records", "sample_count", "create_observed", "delete_observed",
          "unexplained_conflicts", "resurrections", "failed_writes", "parts_limit_errors",
          "monotone_growth_windows_6h", "new_service_errors"}


def parse_time(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include timezone")
    return parsed.astimezone(timezone.utc)


def evaluate(data, *, now=None):
    if not isinstance(data, dict):
        raise ValueError("input must be an object")
    evaluated_at = now if now is not None else datetime.now(timezone.utc)
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise ValueError("evaluation clock must be timezone-aware")
    evaluated_at = evaluated_at.astimezone(timezone.utc)
    results = {}
    gates = data.get("gates", {})
    if not isinstance(gates, dict):
        raise ValueError("gates must be an object")
    for name, rules in RULES.items():
        row = gates.get(name)
        if row is None or (isinstance(row, dict) and row.get("status") in ("unperformed", "not_run")):
            results[name] = {"status": "UNPERFORMED", "required_window_hours": WINDOW_HOURS.get(name)}
            continue
        if not isinstance(row, dict):
            results[name] = {"status": "UNKNOWN", "reasons": ["gate must be an object"]}
            continue
        reasons, failures = [], []
        if row.get("status") != "measured":
            reasons.append("status is not measured")
        if not isinstance(row.get("source_receipt"), str) or not row["source_receipt"].strip():
            reasons.append("missing collection provenance")
        try:
            start, end = parse_time(row.get("observed_start")), parse_time(row.get("observed_end"))
            if start >= end:
                reasons.append("observed_start must precede observed_end")
            if end > evaluated_at:
                reasons.append("observation ends in the future")
            if name in ("controlled_create_delete", "cross_archive"):
                if start < SOAK_END - timedelta(hours=72) or end > SOAK_END:
                    reasons.append("probe/audit window must be contained in target soak")
            if name in WINDOW_HOURS:
                required_start = SOAK_END - timedelta(hours=WINDOW_HOURS[name])
                if start != required_start or end != SOAK_END:
                    reasons.append(f"must summarize exactly the required final {WINDOW_HOURS[name]}h through {SOAK_END.isoformat()}")
        except (ValueError, TypeError, OverflowError):
            reasons.append("invalid timezone-aware ISO observation timestamps")
        if name == "lag" and row.get("clock") != "witnessedAt":
            reasons.append("lag clock is not witnessedAt")
        if name == "visibility_latency" and row.get("clock") != "durable_ingest_commit_to_first_hidden":
            reasons.append("latency is not keyed ingest-commit to first-hidden")
        if name == "controlled_create_delete" and not row.get("write_authorization_receipt"):
            reasons.append("no explicit test-account write authorization receipt")
        if name == "cross_archive":
            if row.get("sampling_rule") != "1000_or_all_if_fewer":
                reasons.append("sampling rule not declared")
            population = row.get("eligible_delete_population")
            if isinstance(population, bool) or not isinstance(population, int) or population < 1 or row.get("sample_count") != min(1000, population):
                reasons.append("sample does not cover 1000 or all eligible deletes")
        for metric, (op, threshold) in rules.items():
            value = row.get(metric)
            invalid = isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            invalid = invalid or (metric in COUNTS and not isinstance(value, int))
            invalid = invalid or ((metric.endswith("fraction") or metric == "coverage") and isinstance(value, (int, float)) and value > 1)
            if invalid:
                reasons.append(f"invalid/missing {metric}")
            elif (op == "<=" and value > threshold) or (op == ">=" and value < threshold):
                failures.append(metric)
        # A known reported violation remains visible even if other evidence is missing.
        results[name] = {"status": "FAIL" if failures else "UNKNOWN" if reasons else "PASS",
                         "reasons": reasons, "failed_metrics": failures,
                         "required_window_hours": WINDOW_HOURS.get(name), "source_receipt": row.get("source_receipt")}
    return {"schema": "catbird_soak_evaluation_v2", "thresholds_provisional": True,
            "required_soak_start": (SOAK_END - timedelta(hours=72)).isoformat(),
            "required_soak_end": SOAK_END.isoformat(),
            "evidence_class": "offline_validation_of_reported_metrics", "gates": results,
            "qualification": "REVIEW_REQUIRED" if all(r["status"] == "PASS" for r in results.values()) else "UNQUALIFIED", "deployment_authorized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = evaluate(json.loads(raw))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
