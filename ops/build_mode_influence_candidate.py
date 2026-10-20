#!/usr/bin/env python3
"""Build a bounded, offline RI/RM candidate from sealed current-state inputs.

RI aggregates the pre-window interests of distinct eligible engagers by producer
and community. RM summarizes scored originals/replies/quotes/repost subjects.
This command never fetches data, runs a model, changes production, assigns follow
modes, or upgrades an upstream validation receipt into proof of source correctness.
Run --print-input-contract for the complete versioned JSONL input contract.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import re
import sys
from typing import Any


VERSION = "mode-influence-candidate-v1"
TABLES = (
    "accounts", "eligible_posts", "communities", "producers", "posts",
    "post_scores", "engagements", "interests", "deletions",
)
KINDS = ("original", "reply", "quote", "repost")
ENGAGEMENT_WEIGHTS = {"like": 1.0, "repost": 2.0}
SHA_RE = re.compile(r"[0-9a-f]{64}")


def input_contract() -> dict[str, Any]:
    return {
        "schema_version": "mode-influence-input-v1",
        "manifest_fields": {
            "generation_id": "nonempty upstream snapshot identity",
            "cutoff_at": "timezone-aware ISO8601; current-state snapshot and exclusive window end",
            "window_start": "timezone-aware ISO8601; inclusive event window start, at most 90 days before cutoff",
            "interests_cutoff_at": "at or before window_start; later interests would leak window outcomes",
            "event_clock": "validated_observed_at",
            "graph_generation": "nonempty graph identity shared by interests, communities, and producers",
            "score_generation": "nonempty sealed scoring-run identity",
            "score_dimensions": {
                "dimension_name": {
                    "model_sha256": "64 lowercase hex characters",
                    "rubric_sha256": "64 lowercase hex characters",
                    "meaning": "explicit target definition; scores are numbers in [0,1] or null",
                },
            },
            "artifacts": "each TABLES key plus validation_receipt: {path: relative path, sha256: hex, rows: integer for JSONL tables}",
        },
        "tables": {
            "accounts": ["did"],
            "eligible_posts": ["uri"],
            "communities": ["community (nonnegative integer)", "sensitivity = non_sensitive",
                            "permitted_for_audience_aggregation = true"],
            "producers": ["did", "primary_community (integer or null)", "primary_score ([0,1] or null)",
                          "followers_est (nonnegative integer or null)"],
            "posts": ["uri", "author_did", "event_at", "content_sha256", "is_reply (boolean)",
                      "is_quote (boolean)", "is_deleted = false"],
            "post_scores": ["uri", "content_sha256", "score_generation",
                            "scores {every dimension: [0,1] or null}",
                            "unscored_reasons {each null dimension: nonempty string}"],
            "engagements": ["uri", "actor_did", "subject_uri", "kind = like|repost", "event_at",
                            "is_deleted = false"],
            "interests": ["did", "community", "score ([0,1])", "as_of (= interests_cutoff_at)",
                          "graph_generation"],
            "deletions": ["kind = account|post|engagement", "key = corresponding DID or record URI"],
        },
        "validation_receipt": {
            "schema_version": "current-state-validation-v1",
            "status": "passed",
            "generation_id": "matches manifest",
            "cutoff_at": "matches manifest",
            "input_sha256": "{every TABLES key: exact corresponding artifact SHA256}",
            "checks": {
                "latest_versions_folded": True, "record_deletions_applied": True,
                "account_status_applied": True, "visibility_policy_applied": True,
            },
            "account_state_generation": "nonempty pinned state identity",
            "visibility_policy_generation": "nonempty pinned policy identity",
        },
        "rules": [
            "All JSONL keys must be unique under their documented identity; this command is not a state-folding engine.",
            "Eligible post URIs must belong to eligible accounts and cannot conflict with the deletion ledger.",
            "Only explicitly permitted non-sensitive communities enter this bounded candidate.",
            "Interest vectors may have missing/truncated mass but must sum to at most one; never renormalize each vector.",
            "Posts can precede window_start for repost-subject resolution; no post or engagement event may reach cutoff_at.",
            "Reply takes precedence over quote, so one authored post contributes to exactly one type.",
            "A repost uses its eligible subject's score; missing/ineligible subjects stay unknown, while deleted entities are suppressed.",
            "Source post content hash must match every joined score row; no post text is required or emitted.",
            "Each producer's distinct eligible engagers contribute once to primary RI; weighted repeated interactions are a separately named comparison.",
            "Producer-normalized observed interest shares are accompanied by complete-audience support and missing-mass bounds.",
            "Popularity retains the legacy follower/engagement formula under an explicit name; absent inputs stay null.",
            "Input validation receipt is an upstream assertion; all outputs remain offline candidates, never qualified follow gates.",
        ],
        "bounds": {"default_max_rows_per_table": 100000, "default_max_bytes_per_artifact": 268435456,
                   "maximum_producer_type_dimension_cells": 100000,
                   "bootstrap_replicates": "0 or 100..2000; default 200, exploratory account-day resampling"},
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def timestamp(value: Any) -> dt.datetime:
    require(isinstance(value, str), "timestamp must be a string")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "timestamp must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def did(value: Any) -> str:
    require(isinstance(value, str) and value.startswith("did:") and "/" not in value
            and not any(c.isspace() for c in value), "invalid DID")
    return value


def record_uri(value: Any, collection: str | None = None) -> tuple[str, str]:
    require(isinstance(value, str), "record URI must be a string")
    match = re.fullmatch(r"at://([^/]+)/([^/]+)/([^/?#\s]+)", value)
    require(match is not None, "invalid record URI")
    owner, actual_collection, _rkey = match.groups()
    did(owner)
    require(collection is None or actual_collection == collection, "record collection mismatch")
    return owner, actual_collection


def integer(value: Any, name: str) -> int:
    require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return value


def score(value: Any, name: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f"{name} must be finite in [0,1]")
    return float(value)


def unique(rows: list[dict[str, Any]], fields: tuple[str, ...], name: str) -> dict[Any, dict[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        key = key[0] if len(key) == 1 else key
        require(key not in result, f"duplicate {name} identity: {key}")
        result[key] = row
    return result


def load_inputs(manifest_path: pathlib.Path, max_rows: int, max_bytes: int) -> tuple[dict, dict, dict]:
    manifest_bytes = manifest_path.read_bytes()
    require(len(manifest_bytes) <= max_bytes, "manifest exceeds byte limit")
    manifest = json.loads(manifest_bytes)
    require(manifest["schema_version"] == "mode-influence-input-v1", "unsupported input contract")
    for name in ("generation_id", "graph_generation", "score_generation"):
        require(isinstance(manifest.get(name), str) and bool(manifest[name]), f"missing {name}")
    require(manifest["event_clock"] == "validated_observed_at", "unsupported event clock")
    cutoff, start, interest_cutoff = map(timestamp, (
        manifest["cutoff_at"], manifest["window_start"], manifest["interests_cutoff_at"]))
    require(dt.timedelta(0) < cutoff - start <= dt.timedelta(days=90), "window must be positive and at most 90 days")
    require(interest_cutoff <= start, "interests must predate the evaluation window")
    dimensions = manifest["score_dimensions"]
    require(isinstance(dimensions, dict) and 0 < len(dimensions) <= 16, "provide 1..16 score dimensions")
    for name, identity in dimensions.items():
        require(isinstance(name, str) and bool(name) and isinstance(identity, dict), "invalid dimension")
        require(bool(identity.get("meaning")), f"{name}: missing target meaning")
        for field in ("model_sha256", "rubric_sha256"):
            require(isinstance(identity.get(field), str) and SHA_RE.fullmatch(identity[field]) is not None,
                    f"{name}: missing {field}")
    expected_keys = set(TABLES) | {"validation_receipt"}
    require(set(manifest["artifacts"]) == expected_keys, "artifact keys must exactly match input contract")
    tables, records = {}, {}
    base = manifest_path.parent.resolve()
    for name in (*TABLES, "validation_receipt"):
        entry = manifest["artifacts"][name]
        relative = pathlib.Path(entry["path"])
        path = (base / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(base), "artifacts must be inside the manifest directory")
        require(path.is_file() and path.stat().st_size <= max_bytes, f"{name}: absent or over byte limit")
        raw = path.read_bytes()
        require(digest(raw) == entry["sha256"], f"{name}: SHA256 mismatch")
        records[name] = {"path": str(path), "sha256": entry["sha256"], "bytes": len(raw)}
        if name == "validation_receipt":
            receipt = json.loads(raw)
        else:
            require(integer(entry["rows"], f"{name}.rows") <= max_rows, f"{name}: over row limit")
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            require(len(rows) == entry["rows"] and all(isinstance(row, dict) for row in rows), f"{name}: invalid row count/type")
            tables[name] = rows
    require(receipt["schema_version"] == "current-state-validation-v1" and receipt["status"] == "passed",
            "current-state validation has not passed")
    require(receipt["generation_id"] == manifest["generation_id"] and timestamp(receipt["cutoff_at"]) == cutoff,
            "current-state receipt generation/cutoff mismatch")
    require(receipt["input_sha256"] == {name: records[name]["sha256"] for name in TABLES},
            "current-state receipt does not bind exact inputs")
    for check in ("latest_versions_folded", "record_deletions_applied", "account_status_applied", "visibility_policy_applied"):
        require(receipt["checks"].get(check) is True, f"upstream check missing: {check}")
    for field in ("account_state_generation", "visibility_policy_generation"):
        require(isinstance(receipt.get(field), str) and bool(receipt[field]), f"missing {field}")
    return manifest, tables, {
        "manifest_sha256": digest(manifest_bytes), "artifacts": records,
        "upstream_current_state_receipt": receipt,
        "qualification": "upstream current-state assertion plus local identity/hash validation; upstream reconstruction not independently proved",
    }


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def describe_dimension(
    observations: list[tuple[str, dict[str, float | None], dict[str, str]]],
    dimension: str, replicates: int, seed: int,
) -> dict[str, Any]:
    values = [scores[dimension] for _day, scores, _reason in observations if scores[dimension] is not None]
    missing = collections.Counter(reason[dimension] for _day, scores, reason in observations if scores[dimension] is None)
    days: dict[str, list[float]] = collections.defaultdict(list)
    for day, scores, _reason in observations:
        if scores[dimension] is not None:
            days[day].append(scores[dimension])
    interval: dict[str, Any] = {
        "status": "not_estimated", "method": "descriptive account-day bootstrap of mean model score",
        "sampling_unit": "observed account UTC day; all scored events on a sampled day stay together",
        "scored_days": len(days), "replicates": replicates,
        "limitations": "exploratory conditional-on-scored-data interval; not calibrated model uncertainty, missing-data correction, or population validity",
        "lower": None, "upper": None,
    }
    if replicates and len(days) >= 2:
        day_stats = [(sum(day_values), len(day_values)) for _day, day_values in sorted(days.items())]
        rng = random.Random(seed)
        means = []
        for _ in range(replicates):
            selected = [day_stats[rng.randrange(len(day_stats))] for _ in day_stats]
            means.append(sum(total for total, _n in selected) / sum(n for _total, n in selected))
        interval.update({"status": "computed", "percentile_level": .95,
                         "lower": quantile(means, .025), "upper": quantile(means, .975)})
    else:
        interval["reason"] = "bootstrap disabled" if not replicates else "fewer than two scored account-days"
    return {
        "events": len(observations), "scored_events": len(values), "unscored_events": len(observations) - len(values),
        "scored_fraction": len(values) / len(observations) if observations else None,
        "missing_reasons": dict(sorted(missing.items())),
        "mean_model_score": sum(values) / len(values) if values else None,
        "quantiles": {str(q): quantile(values, q) for q in (0, .25, .5, .75, 1)},
        "uncertainty": interval,
    }


def build_candidate(manifest: dict, tables: dict, *, bootstrap_replicates: int = 200, seed: int = 0) -> dict[str, Any]:
    require(bootstrap_replicates == 0 or 100 <= bootstrap_replicates <= 2000, "bootstrap replicates must be 0 or 100..2000")
    cutoff, start = timestamp(manifest["cutoff_at"]), timestamp(manifest["window_start"])
    interest_cutoff = timestamp(manifest["interests_cutoff_at"])
    dimensions = sorted(manifest["score_dimensions"])
    deleted = {kind: set() for kind in ("account", "post", "engagement")}
    for row in unique(tables["deletions"], ("kind", "key"), "deletion").values():
        require(row["kind"] in deleted, "unsupported deletion kind")
        if row["kind"] == "account":
            did(row["key"])
        else:
            record_uri(row["key"], "app.bsky.feed.post" if row["kind"] == "post" else None)
        deleted[row["kind"]].add(row["key"])
    accounts = unique(tables["accounts"], ("did",), "account")
    for account in accounts:
        did(account)
        require(account not in deleted["account"], "eligible account conflicts with deletion ledger")
    communities = unique(tables["communities"], ("community",), "community")
    for community, row in communities.items():
        integer(community, "community")
        require(row["sensitivity"] == "non_sensitive" and row["permitted_for_audience_aggregation"] is True,
                "community lacks permitted non-sensitive aggregation scope")
    eligible_posts = unique(tables["eligible_posts"], ("uri",), "eligible post")
    for uri in eligible_posts:
        owner, _collection = record_uri(uri, "app.bsky.feed.post")
        require(owner in accounts and uri not in deleted["post"], "eligible post conflicts with account/deletion universe")
    producers = unique(tables["producers"], ("did",), "producer")
    require(len(producers) * len(KINDS) * len(dimensions) <= 100000, "candidate exceeds bounded profile-cell limit")
    for account, row in producers.items():
        require(account in accounts, "producer outside eligible accounts")
        require((row["primary_community"] is None) == (row["primary_score"] is None), "partial primary membership")
        if row["primary_community"] is not None:
            integer(row["primary_community"], "primary_community")
            require(row["primary_community"] in communities, "primary community outside permitted universe")
            score(row["primary_score"], "primary score")
        if row["followers_est"] is not None:
            integer(row["followers_est"], "followers_est")
    interests: dict[str, dict[int, float]] = collections.defaultdict(dict)
    for (account, community), row in unique(tables["interests"], ("did", "community"), "interest").items():
        integer(community, "interest community")
        require(account in accounts and community in communities, "interest outside eligible universes")
        require(timestamp(row["as_of"]) == interest_cutoff and row["graph_generation"] == manifest["graph_generation"],
                "interest time or graph generation mismatch")
        interests[account][community] = score(row["score"], "interest score")
    for account, vector in interests.items():
        require(len(vector) <= 50 and sum(vector.values()) <= 1 + 1e-9,
                f"{account}: interest vector exceeds 50 communities or unit mass")
    posts = unique(tables["posts"], ("uri",), "post")
    for uri, row in posts.items():
        owner, _collection = record_uri(uri, "app.bsky.feed.post")
        require(owner == did(row["author_did"]), "post URI/author mismatch")
        require(row["is_deleted"] is False, "posts must be pre-folded live rows")
        require(type(row["is_reply"]) is bool and type(row["is_quote"]) is bool, "post type flags must be booleans")
        require(isinstance(row["content_sha256"], str) and SHA_RE.fullmatch(row["content_sha256"]) is not None,
                "missing post content hash")
        require(timestamp(row["event_at"]) < cutoff, "post event reaches or exceeds cutoff")
    scores = unique(tables["post_scores"], ("uri",), "score")
    for uri, row in scores.items():
        require(uri in posts and row["content_sha256"] == posts[uri]["content_sha256"], "score/post content identity mismatch")
        require(row["score_generation"] == manifest["score_generation"], "score generation mismatch")
        require(set(row["scores"]) == set(dimensions), "score dimensions mismatch")
        null_dimensions = {name for name in dimensions if row["scores"][name] is None}
        require(set(row["unscored_reasons"]) == null_dimensions, "null scores require exact missing-reason entries")
        for dimension, value in row["scores"].items():
            if value is None:
                require(isinstance(row["unscored_reasons"][dimension], str) and bool(row["unscored_reasons"][dimension]),
                        "empty unscored reason")
            else:
                score(value, dimension)
    engagements = unique(tables["engagements"], ("uri",), "engagement")
    for uri, row in engagements.items():
        require(row["kind"] in ENGAGEMENT_WEIGHTS, "unsupported engagement kind")
        owner, _collection = record_uri(uri, f"app.bsky.feed.{row['kind']}")
        require(owner == did(row["actor_did"]), "engagement URI/actor mismatch")
        record_uri(row["subject_uri"], "app.bsky.feed.post")
        require(row["is_deleted"] is False, "engagements must be pre-folded live rows")
        require(timestamp(row["event_at"]) < cutoff, "engagement reaches or exceeds cutoff")

    audit = collections.Counter()
    mode_events: dict[str, dict[str, list]] = {
        account: {kind: [] for kind in KINDS} for account in producers
    }
    audience: dict[str, collections.Counter] = {account: collections.Counter() for account in producers}
    audience_events: dict[str, list[tuple[dt.datetime, str, float]]] = {account: [] for account in producers}

    def score_for(uri: str) -> tuple[dict, dict]:
        if uri not in eligible_posts:
            reason = "subject_outside_eligible_post_universe"
        elif uri not in posts:
            reason = "eligible_subject_not_resolved"
        elif uri not in scores:
            reason = "score_row_missing"
        else:
            return scores[uri]["scores"], scores[uri]["unscored_reasons"]
        return {dimension: None for dimension in dimensions}, {dimension: reason for dimension in dimensions}

    for uri, row in posts.items():
        when = timestamp(row["event_at"])
        account = row["author_did"]
        if uri not in eligible_posts or account not in producers:
            audit["authored_post_outside_candidate_universe"] += 1
            continue
        if when < start:
            audit["authored_post_before_window"] += 1
            continue
        kind = "reply" if row["is_reply"] else "quote" if row["is_quote"] else "original"
        values, reasons = score_for(uri)
        mode_events[account][kind].append((when.date().isoformat(), values, reasons))
        audit["authored_events_used"] += 1
    for uri, row in engagements.items():
        actor, subject = row["actor_did"], row["subject_uri"]
        when = timestamp(row["event_at"])
        subject_owner, _collection = record_uri(subject, "app.bsky.feed.post")
        if uri in deleted["engagement"] or actor in deleted["account"] or subject in deleted["post"] or subject_owner in deleted["account"]:
            audit["engagement_suppressed_by_deletion_ledger"] += 1
            continue
        if actor not in accounts or when < start:
            audit["engagement_outside_actor_or_time_universe"] += 1
            continue
        if row["kind"] == "repost" and actor in producers:
            values, reasons = score_for(subject)
            mode_events[actor]["repost"].append((when.date().isoformat(), values, reasons))
            audit["repost_mode_events_used"] += 1
        if subject not in eligible_posts or subject not in posts:
            audit["influence_unresolved_or_ineligible_subject"] += 1
            continue
        target = posts[subject]["author_did"]
        if target not in producers or actor == target:
            audit["influence_outside_producer_or_self_engagement"] += 1
            continue
        weight = ENGAGEMENT_WEIGHTS[row["kind"]]
        audience[target][actor] += weight
        audience_events[target].append((when, actor, weight))
        audit["influence_events_used"] += 1

    mode_profiles, influence, popularity = [], [], []
    for account, producer in sorted(producers.items()):
        modes = mode_events[account]
        total_events = sum(len(observations) for observations in modes.values())
        mode_profiles.append({
            "did": account, "status": "candidate_unqualified", "events": total_events,
            "active_days": len({day for observations in modes.values() for day, _scores, _reasons in observations}),
            "types": {kind: {
                "events": len(observations), "event_share": len(observations) / total_events if total_events else None,
                "dimensions": {dimension: describe_dimension(
                    observations, dimension, bootstrap_replicates,
                    int(hashlib.sha256(f"{seed}|{account}|{kind}|{dimension}".encode()).hexdigest()[:16], 16),
                ) for dimension in dimensions},
            } for kind, observations in modes.items()},
            "assigned_follow_mode": None, "follow_mode_eligibility": "not_evaluated",
        })
        actors = audience[account]
        total_weight = sum(actors.values())
        mass: collections.Counter[int] = collections.Counter()
        weighted_mass: collections.Counter[int] = collections.Counter()
        support: dict[int, int] = collections.Counter()
        for actor, weight in actors.items():
            for community, value in interests.get(actor, {}).items():
                mass[community] += value
                weighted_mass[community] += weight * value
                support[community] += value > 0
        known_mass, known_weighted_mass = sum(mass.values()), sum(weighted_mass.values())
        missing_mass = max(0.0, len(actors) - known_mass)
        missing_weighted_mass = max(0.0, total_weight - known_weighted_mass)
        event_times = [when for when, _actor, _weight in audience_events[account]]
        influence.append({
            "did": account, "status": "candidate_unqualified",
            "method": "sum of distinct eligible engagers' pre-window InterestedIn; normalized across communities per producer",
            "engagement_events": len(event_times), "distinct_engagers": len(actors),
            "engagers_with_interest_rows": sum(actor in interests for actor in actors),
            "engagers_without_interest_rows": sum(actor not in interests for actor in actors),
            "known_interest_mass": known_mass, "missing_interest_mass": missing_mass,
            "interest_mass_coverage": known_mass / len(actors) if actors else None,
            "weighted_engagement_mass": total_weight,
            "weighted_interest_mass_coverage": known_weighted_mass / total_weight if total_weight else None,
            "first_engagement_at": min(event_times).isoformat() if event_times else None,
            "last_engagement_at": max(event_times).isoformat() if event_times else None,
            "active_engagement_days": len({when.date() for when in event_times}),
            "exposure": {"status": "unavailable", "reason": "no impression/exposure log supplied; engagement is not exposure"},
            "communities": [{
                "community": community, "interested_engagers": support[community],
                "raw_interest_mass": mass[community],
                "producer_normalized_observed_share": mass[community] / known_mass if known_mass else None,
                "all_engager_mass_lower": mass[community] / len(actors) if actors else None,
                "all_engager_mass_upper": min(1.0, (mass[community] + missing_mass) / len(actors)) if actors else None,
                "weighted_interaction_raw_mass": weighted_mass[community],
                "weighted_interaction_normalized_observed_share": (
                    weighted_mass[community] / known_weighted_mass if known_weighted_mass else None),
                "weighted_all_event_mass_lower": weighted_mass[community] / total_weight if total_weight else None,
                "weighted_all_event_mass_upper": (
                    min(1.0, (weighted_mass[community] + missing_weighted_mass) / total_weight) if total_weight else None),
            } for community in sorted(mass) if mass[community] > 0],
            "uncertainty": {
                "kind": "missing-interest-mass bounds, not sampling confidence intervals",
                "unlisted_community_upper_mass": min(1.0, missing_mass / len(actors)) if actors else None,
                "sampling_interval": None,
                "qualification": "normalized observed shares describe covered interests only; missing audience and truncated vector mass remain unknown",
            },
        })
        primary, followers = producer["primary_score"], producer["followers_est"]
        popularity.append({
            "did": account, "primary_community": producer["primary_community"],
            "name": "legacy_primary_knownfor_log_engagement_followers_baseline",
            "formula": "primary_score * log1p(primary_score * weighted_engagement_events + 0.1 * followers_est)",
            "primary_score": primary, "followers_est": followers,
            "engagement_events": len(event_times), "distinct_engagers": len(actors), "weighted_engagement_events": total_weight,
            "value": primary * math.log1p(primary * total_weight + .1 * followers) if primary is not None and followers is not None else None,
            "missing_reason": "primary membership or follower estimate unavailable" if primary is None or followers is None else None,
            "qualification": "named popularity heuristic on the corrected eligible input universe; not audience-based Influence",
        })
    return {
        "mode_profiles": mode_profiles, "audience_influence": influence, "popularity_baseline": popularity,
        "audit": dict(sorted(audit.items())),
        "qualification": {
            "status": "candidate_unqualified", "production_publishable": False, "follow_gate_qualified": False,
            "feature_quality_measured": False, "assigned_modes_evaluated": False,
            "source_state": "requires upstream current-state receipt; not rebuilt or independently established here",
            "score_semantics": "descriptive distributions of provided model scores; no binary political/hostility labels or calibrated action thresholds inferred",
            "interest_scope": "explicit permitted non-sensitive communities only; no individual audience profiles emitted",
        },
    }


def build_to_directory(manifest_path: pathlib.Path, output_dir: pathlib.Path, *, max_rows: int = 100000,
                       max_bytes: int = 256 * 1024 * 1024, bootstrap_replicates: int = 200, seed: int = 0) -> dict:
    require(max_rows > 0 and max_bytes > 0, "input bounds must be positive")
    source = pathlib.Path(__file__).resolve()
    source_hash = digest(source.read_bytes())
    manifest, tables, provenance = load_inputs(manifest_path, max_rows, max_bytes)
    result = build_candidate(manifest, tables, bootstrap_replicates=bootstrap_replicates, seed=seed)
    settings = {"bootstrap_replicates": bootstrap_replicates, "seed": seed, "max_rows": max_rows, "max_bytes": max_bytes}
    generation = digest(encoded({"version": VERSION, "input": provenance["manifest_sha256"],
                                 "source": source_hash, "settings": settings}))
    require(not output_dir.exists(), "output directory exists; choose a new candidate directory")
    output_dir.mkdir(parents=True)
    artifacts = {}
    for name in ("mode_profiles", "audience_influence", "popularity_baseline"):
        raw = b"".join(encoded(row) for row in result[name])
        path = output_dir / f"{name}.jsonl"
        with path.open("xb") as fh:
            fh.write(raw)
        artifacts[name] = {"path": path.name, "sha256": digest(raw), "bytes": len(raw), "rows": len(result[name])}
    seal = {
        "schema_version": VERSION, "candidate_generation": generation,
        "source": {"path": str(source), "sha256": source_hash, "python": sys.version},
        "input_provenance": provenance, "upstream_generation": manifest["generation_id"],
        "graph_generation": manifest["graph_generation"], "score_generation": manifest["score_generation"],
        "window_start": manifest["window_start"], "cutoff_at": manifest["cutoff_at"],
        "interests_cutoff_at": manifest["interests_cutoff_at"], "score_dimensions": manifest["score_dimensions"],
        "settings": settings, "artifacts": artifacts, "audit": result["audit"],
        "qualification": result["qualification"],
        "policies": {
            "primary_influence": "distinct engager contribution once per producer/window",
            "comparison_influence": "repeated interactions weighted like=1, repost=2",
            "post_type": "reply takes precedence over quote; authored post counted once",
            "repost": "eligible subject's score; unresolved/ineligible is unknown; deleted entities suppressed",
        },
    }
    require(digest(source.read_bytes()) == source_hash, "builder source changed during execution")
    require(digest(manifest_path.read_bytes()) == provenance["manifest_sha256"], "input manifest changed during execution")
    for record in provenance["artifacts"].values():
        require(digest(pathlib.Path(record["path"]).read_bytes()) == record["sha256"], "input changed during execution")
    with (output_dir / "candidate_manifest.json").open("xb") as fh:
        fh.write(encoded(seal))
    return seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--print-input-contract", action="store_true")
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.print_input_contract:
        print(json.dumps(input_contract(), indent=2))
        return
    if args.manifest is None or args.output_dir is None:
        parser.error("--manifest and --output-dir are required")
    try:
        seal = build_to_directory(args.manifest, args.output_dir, max_rows=args.max_rows, max_bytes=args.max_bytes,
                                  bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise SystemExit(f"Candidate not sealed: {exc}") from exc
    print(json.dumps({"candidate_manifest": str(args.output_dir / "candidate_manifest.json"),
                      "candidate_generation": seal["candidate_generation"], "qualification": seal["qualification"]}, indent=2))


if __name__ == "__main__":
    main()
