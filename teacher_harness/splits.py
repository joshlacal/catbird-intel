"""Input integrity, author/thread/near-duplicate split isolation, and stratified sampling."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

COLAB_DIR = Path(__file__).resolve().parents[1] / "colab"
if str(COLAB_DIR) not in sys.path:
    sys.path.insert(0, str(COLAB_DIR))

from real_pipeline.common import (  # noqa: E402
    SEALED_EXPORT_SHA256,
    load_sealed_export,
    read_rows,
    sha256,
    validate_posts,
)
from real_pipeline.language import LANGUAGE_CONTRACT, language_histogram, language_scope  # noqa: E402

from .config import HarnessConfig, canonical_sha256  # noqa: E402
from .moderation import fold_and_join_moderation_labels  # noqa: E402
from .schemas import resolve_post_observation  # noqa: E402

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@[A-Za-z0-9_.\-]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

EVALUATION_SPLITS = frozenset({
    "holdout",
    "dev",
    "human_final",
    "natural_exposure",
    "later_time_test",
})

BLIND_SEALED_SPLITS = frozenset({
    "holdout",
    "human_final",
    "natural_exposure",
})


class SplitLeakageError(ValueError):
    """Raised when author, near-duplicate text, or thread context leaks between splits."""


def near_duplicate_fingerprint(text: str) -> str:
    """Compute a normalized text fingerprint to cluster duplicate and near-duplicate posts."""
    lowered = (text or "").lower()
    no_urls = _URL_RE.sub(" ", lowered)
    no_mentions = _MENTION_RE.sub(" ", no_urls)
    alnum = _NON_ALNUM_RE.sub(" ", no_mentions).strip()
    tokens = alnum.split()
    if not tokens:
        alnum = _NON_ALNUM_RE.sub("", lowered) or lowered.strip()
    elif len(tokens) >= 6:
        # Canonical token sequence ignoring minor trailing hashtags/punctuation
        alnum = " ".join(tokens[:24])
    else:
        alnum = " ".join(tokens)
    return hashlib.sha256(alnum.encode("utf-8")).hexdigest()[:24]


def _connected_thread_components(rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> dict[str, str]:
    """Find connected conversation components across uri, reply_parent_uri, reply_root_uri, quote_uri, repost_subject_uri."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        curr = x
        while curr != root:
            nxt = parent.get(curr, curr)
            parent[curr] = root
            curr = nxt
        return root

    def union(a: str, b: str) -> None:
        if not a or not b:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for r in (*rows, *holdout_rows):
        uri = str(r.get("uri") or "").strip()
        if not uri:
            continue
        parent.setdefault(uri, uri)
        for key in ("reply_parent_uri", "reply_root_uri", "quote_uri", "repost_subject_uri"):
            ref = str(r.get(key) or "").strip()
            if ref:
                parent.setdefault(ref, ref)
                union(uri, ref)

    return {uri: find(uri) for uri in parent}


def _assign_raw_author_split(
    author_did: str,
    bucket: int,
    *,
    single_bucket_fallback: bool,
    holdout_authors: set[str],
    external_final_authors: set[str],
    external_exposure_authors: set[str],
) -> str:
    if author_did in holdout_authors or bucket == 0:
        return "holdout"
    if author_did in external_final_authors:
        return "human_final"
    if author_did in external_exposure_authors:
        return "natural_exposure"

    effective_bucket = bucket
    if single_bucket_fallback:
        digest_int = int(hashlib.sha256(f"author-split-v1:{author_did}".encode("utf-8")).hexdigest()[:8], 16)
        effective_bucket = 1 + (digest_int % 19)

    if effective_bucket == 19:
        return "human_final"
    if effective_bucket == 18:
        return "natural_exposure"
    if effective_bucket == 17:
        return "later_time_test"
    if effective_bucket == 16:
        return "dev"
    return "train"


def _is_editorial_boundary_candidate(row: dict[str, Any]) -> bool:
    text = str(row.get("text") or "").lower()
    pol = float(row.get("regex_baseline_politicalness") or 0.0)
    heat = float(row.get("regex_baseline_heat") or 0.0)
    bait = float(row.get("regex_baseline_engagement_bait") or 0.0)
    if 0.04 <= pol <= 0.35 or 0.04 <= heat <= 0.35 or 0.03 <= bait <= 0.30:
        return True
    if row.get("post_type") == "quote":
        return True
    boundary_cues = ("camera", "autofocus", "policy", "senate", "election", "satire", "irony", "quote", "critique", "terrible", "wrong answers only")
    return any(cue in text for cue in boundary_cues)


def prepare_and_freeze_corpus(
    *,
    input_path: Path,
    holdout_path: Path,
    export_manifest_path: Path | None = None,
    config: HarnessConfig,
    sample_limit: int | None = None,
    target_train_sample_size: int | None = None,
    moderation_events: list[dict[str, Any]] | None = None,
    moderation_cutoff_utc: str = "2026-10-20T23:59:59+00:00",
    external_final_uris: set[str] | None = None,
    external_final_authors: set[str] | None = None,
    external_exposure_uris: set[str] | None = None,
    external_exposure_authors: set[str] | None = None,
    max_posts_per_author_per_stratum: int = 3,
    max_posts_per_dup_cluster: int = 1,
    selection_seed: str = "catbird-teacher-selection-v1",
    allow_synthetic_input_for_tests: bool = False,
) -> dict[str, Any]:
    """Validate canonical bank/holdout, freeze leak-free splits, join moderation labels, and assign strata."""
    seal, holdout_rows = load_sealed_export(holdout_path, export_manifest_path)
    input_file_sha256 = sha256(input_path)
    expected_bank_sha256 = seal["bank"]["sha256"]
    if not allow_synthetic_input_for_tests and input_file_sha256 != expected_bank_sha256:
        raise ValueError(
            f"Input bank SHA-256 ({input_file_sha256}) does not match canonical v1 bank ({expected_bank_sha256})"
        )

    raw_rows = read_rows(input_path, limit=sample_limit)
    if not raw_rows:
        raise ValueError("Input bank returned zero rows")

    # Separate repost records before validate_posts (which checks app.bsky.feed.post URIs)
    standard_rows = [r for r in raw_rows if r.get("post_type") != "repost"]
    if standard_rows:
        validate_posts(standard_rows, allow_teachers=False, split="train")

    holdout_uris = {str(r["uri"]) for r in holdout_rows}
    holdout_authors = {str(r["author_did"]) for r in holdout_rows}
    holdout_fps = {near_duplicate_fingerprint(str(r.get("text") or "")) for r in holdout_rows}

    ext_final_uris = set(external_final_uris or ())
    ext_final_authors = set(external_final_authors or ())
    ext_exp_uris = set(external_exposure_uris or ())
    ext_exp_authors = set(external_exposure_authors or ())

    # Check if sample_limit caused only a single author_split_bucket to be present
    observed_buckets = {int(r["author_split_bucket"]) for r in raw_rows if "author_split_bucket" in r}
    single_bucket_fallback = len(observed_buckets) <= 2 and sample_limit is not None and sample_limit < 5000

    # First pass: author-level split assignment ( guaranteeing 0 author overlap across splits )
    author_to_split: dict[str, str] = {}
    for r in raw_rows:
        uri = str(r["uri"])
        author = str(r["author_did"])
        bucket = int(r["author_split_bucket"])
        if uri in holdout_uris or author in holdout_authors or bucket == 0:
            raise SplitLeakageError(f"Training bank row {uri} (author {author}) overlaps sealed 600-post holdout")
        if uri in ext_final_uris:
            assigned = "human_final"
        elif uri in ext_exp_uris:
            assigned = "natural_exposure"
        else:
            assigned = _assign_raw_author_split(
                author,
                bucket,
                single_bucket_fallback=single_bucket_fallback,
                holdout_authors=holdout_authors,
                external_final_authors=ext_final_authors,
                external_exposure_authors=ext_exp_authors,
            )
        prev = author_to_split.get(author)
        if prev is not None and prev != assigned:
            # Escalate to the more restrictive evaluation split so author never straddles two splits
            priority = ("holdout", "human_final", "natural_exposure", "later_time_test", "dev", "train")
            assigned = min((prev, assigned), key=lambda s: priority.index(s))
        author_to_split[author] = assigned

    # Thread component and near-duplicate fingerprint computation for leakage prevention
    thread_map = _connected_thread_components(raw_rows, holdout_rows)
    eval_thread_components: set[str] = {thread_map[str(r["uri"])] for r in holdout_rows if str(r["uri"]) in thread_map}
    eval_near_dup_fps: set[str] = set(holdout_fps)

    for r in raw_rows:
        uri = str(r["uri"])
        author = str(r["author_did"])
        sp = author_to_split[author]
        fp = near_duplicate_fingerprint(str(r.get("text") or ""))
        if sp in EVALUATION_SPLITS:
            if uri in thread_map:
                eval_thread_components.add(thread_map[uri])
            eval_near_dup_fps.add(fp)
        # Also if a post directly replies to or quotes a holdout/eval author or URI, taint its component
        for ref_key in ("reply_parent_uri", "reply_root_uri", "quote_uri", "repost_subject_uri"):
            ref_uri = str(r.get(ref_key) or "").strip()
            if ref_uri in holdout_uris or any(ref_uri.startswith(f"at://{ha}/") for ha in holdout_authors):
                if uri in thread_map:
                    eval_thread_components.add(thread_map[uri])

    corpus_by_uri = {str(r["uri"]): r for r in raw_rows}
    seen_uris: set[str] = set()
    prepared_posts: list[dict[str, Any]] = []
    quarantine_counts: Counter[str] = Counter()

    for r in raw_rows:
        uri = str(r["uri"])
        if uri in seen_uris:
            raise SplitLeakageError(f"Duplicate URI in input corpus: {uri}")
        seen_uris.add(uri)

        author = str(r["author_did"])
        raw_split = author_to_split[author]
        lang_info = language_scope(r)
        fp = near_duplicate_fingerprint(str(r.get("text") or ""))
        comp_id = thread_map.get(uri, uri)

        # Check conversation-context and near-duplicate leakage for train candidates
        effective_split = raw_split
        leakage_reason: str | None = None
        if raw_split == "train":
            if comp_id in eval_thread_components:
                effective_split = "quarantined_leakage"
                leakage_reason = "conversation_context_touches_eval_split"
                quarantine_counts[leakage_reason] += 1
            elif fp in eval_near_dup_fps:
                effective_split = "quarantined_leakage"
                leakage_reason = "near_duplicate_text_in_eval_split"
                quarantine_counts[leakage_reason] += 1

        # Resolve text-only and context-enriched observations
        try:
            text_obs = resolve_post_observation(r, corpus_by_uri=corpus_by_uri, observation_mode="text_only")
            ctx_obs = resolve_post_observation(r, corpus_by_uri=corpus_by_uri, observation_mode="context_enriched")
            repost_resolved = True
            unscored_reason = lang_info["reason"]
        except ValueError as exc:
            repost_resolved = False
            text_obs = {
                "text_sha256": hashlib.sha256(str(r.get("text") or "").encode("utf-8")).hexdigest(),
                "context_sha256": canonical_sha256({"unresolvable_repost": True}),
                "repost_subject_uri": r.get("repost_subject_uri"),
                "context_texts": [],
            }
            ctx_obs = text_obs
            unscored_reason = f"unresolved_subject:{exc}"

        eligible_for_labeling = (
            lang_info["in_scope"]
            and repost_resolved
            and effective_split in ("train", "dev")
            and effective_split not in BLIND_SEALED_SPLITS
        )
        eligibility_status = (
            "eligible"
            if eligible_for_labeling
            else (
                f"excluded_split:{effective_split}"
                if effective_split not in ("train", "dev")
                else f"unscored:{unscored_reason}"
            )
        )

        prepared_posts.append({
            "uri": uri,
            "cid": r.get("cid"),
            "author_did": author,
            "author_split_bucket": int(r["author_split_bucket"]),
            "created_at": str(r.get("created_at") or ""),
            "post_type": str(r.get("post_type") or "original"),
            "langs": list(r.get("langs") or []),
            "language_in_scope": bool(lang_info["in_scope"]),
            "unscored_reason": unscored_reason,
            "text": str(r.get("text") or ""),
            "text_sha256": text_obs["text_sha256"],
            "text_only_context_sha256": text_obs["context_sha256"],
            "context_enriched_sha256": ctx_obs["context_sha256"],
            "has_local_parent_or_quote_text": bool(ctx_obs["context_texts"]),
            "near_dup_fingerprint": fp,
            "thread_component_id": comp_id,
            "split": effective_split,
            "leakage_reason": leakage_reason,
            "eligible_for_labeling": eligible_for_labeling,
            "eligibility_snapshot": {
                "status": eligibility_status,
                "source_indexed_at_cutoff_utc": seal["bank"]["source_indexed_at_cutoff_utc"],
                "deletion_visibility_verified_at_export": True,
                "language_contract": LANGUAGE_CONTRACT["schema"],
            },
            "reply_parent_uri": str(r.get("reply_parent_uri") or ""),
            "reply_root_uri": str(r.get("reply_root_uri") or ""),
            "quote_uri": str(r.get("quote_uri") or ""),
            "repost_subject_uri": text_obs["repost_subject_uri"],
            "embed_type": str(r.get("embed_type") or ""),
            "image_count": int(r.get("image_count") or 0),
            "has_video": int(r.get("has_video") or 0),
            "external_uri": str(r.get("external_uri") or ""),
            "regex_baseline_politicalness": r.get("regex_baseline_politicalness"),
            "regex_baseline_heat": r.get("regex_baseline_heat"),
            "regex_baseline_engagement_bait": r.get("regex_baseline_engagement_bait"),
            "regex_baseline_spam": r.get("regex_baseline_spam"),
            "sampling_stratum": None,
            "sampling_method": None,
            "inclusion_probability": None,
            "selection_priority_key": None,
        })

    # Verify zero author overlap across all non-quarantined splits
    split_authors: dict[str, set[str]] = {}
    for p in prepared_posts:
        sp = p["split"]
        if sp == "quarantined_leakage":
            continue
        split_authors.setdefault(sp, set()).add(p["author_did"])
    split_authors["holdout"] = set(holdout_authors)

    split_names = sorted(split_authors.keys())
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i + 1 :]:
            overlap = split_authors[s1] & split_authors[s2]
            if overlap:
                raise SplitLeakageError(f"Author overlap detected between splits {s1} and {s2}: {len(overlap)} authors")

    # Join moderation weak supervision
    eligible_uri_set = {p["uri"] for p in prepared_posts if p["eligible_for_labeling"]}
    mod_result = fold_and_join_moderation_labels(
        moderation_events or [],
        prepared_posts,
        cutoff_utc=moderation_cutoff_utc,
        eligible_uris=eligible_uri_set,
    )
    for p in prepared_posts:
        p["moderation_join"] = mod_result["per_post"][p["uri"]]

    # Assign disjoint sampling strata across eligible train posts
    eligible_train = [p for p in prepared_posts if p["split"] == "train" and p["eligible_for_labeling"]]
    total_budget_items = (
        min(len(eligible_train), int(target_train_sample_size))
        if target_train_sample_size is not None
        else len(eligible_train)
    )

    shares = config.pricing["sampling_strata_shares"]
    stratum_order = (
        "natural_probability",
        "post_type_community_balanced",
        "moderation_enrichment",
        "editorial_boundary",
        "robustness_drift",
    )
    quotas: dict[str, int] = {}
    allocated = 0
    for idx, name in enumerate(stratum_order):
        if idx == len(stratum_order) - 1:
            quotas[name] = max(0, total_budget_items - allocated)
        else:
            q = int(Decimal(total_budget_items) * Decimal(str(shares[name])))
            quotas[name] = q
            allocated += q

    remaining_pool: list[dict[str, Any]] = list(eligible_train)
    stratum_reports: dict[str, dict[str, Any]] = {}

    def _det_key(stratum_name: str, uri: str) -> str:
        return hashlib.sha256(f"{selection_seed}:{stratum_name}:{uri}".encode("utf-8")).hexdigest()

    def _select_with_caps(
        candidates: list[dict[str, Any]],
        *,
        stratum_name: str,
        quota: int,
        sampling_method: str,
        probability_pool_size: int | None,
    ) -> list[dict[str, Any]]:
        ordered = sorted(candidates, key=lambda item: (_det_key(stratum_name, item["uri"]), item["uri"]))
        selected: list[dict[str, Any]] = []
        author_counts: Counter[str] = Counter()
        dup_counts: Counter[str] = Counter()
        for item in ordered:
            if len(selected) >= quota:
                break
            if author_counts[item["author_did"]] >= max_posts_per_author_per_stratum:
                continue
            if dup_counts[item["near_dup_fingerprint"]] >= max_posts_per_dup_cluster:
                continue
            author_counts[item["author_did"]] += 1
            dup_counts[item["near_dup_fingerprint"]] += 1
            selected.append(item)

        inc_prob: float | None = None
        if sampling_method == "probability" and probability_pool_size and probability_pool_size > 0 and selected:
            inc_prob = round(len(selected) / float(probability_pool_size), 8)
        # Never fabricate inclusion probabilities for heuristic sampling!
        for item in selected:
            item["sampling_stratum"] = stratum_name
            item["sampling_method"] = sampling_method
            item["inclusion_probability"] = inc_prob
            item["selection_priority_key"] = _det_key(stratum_name, item["uri"])
        return selected

    selected_uris_all: set[str] = set()

    # 1. natural_probability (40%): pure probability sample from full eligible train pool
    s1_pool = [p for p in remaining_pool if p["uri"] not in selected_uris_all]
    s1_sel = _select_with_caps(
        s1_pool,
        stratum_name="natural_probability",
        quota=quotas["natural_probability"],
        sampling_method="probability",
        probability_pool_size=len(s1_pool),
    )
    selected_uris_all.update(p["uri"] for p in s1_sel)
    stratum_reports["natural_probability"] = {
        "requested": quotas["natural_probability"],
        "candidate_pool": len(s1_pool),
        "selected": len(s1_sel),
        "shortfall": max(0, quotas["natural_probability"] - len(s1_sel)),
        "sampling_method": "probability",
    }

    # 2. post_type_community_balanced (25%): stratified probability sample by post_type
    s2_pool = [p for p in remaining_pool if p["uri"] not in selected_uris_all]
    by_type: dict[str, list[dict[str, Any]]] = {"original": [], "reply": [], "quote": []}
    for p in s2_pool:
        by_type.setdefault(p["post_type"], []).append(p)
    nonempty_types = [t for t, items in by_type.items() if items]
    s2_sel: list[dict[str, Any]] = []
    if nonempty_types and quotas["post_type_community_balanced"] > 0:
        per_type_quota = max(1, quotas["post_type_community_balanced"] // len(nonempty_types))
        for idx, ptype in enumerate(nonempty_types):
            q = (
                quotas["post_type_community_balanced"] - len(s2_sel)
                if idx == len(nonempty_types) - 1
                else min(per_type_quota, quotas["post_type_community_balanced"] - len(s2_sel))
            )
            if q <= 0:
                continue
            picked = _select_with_caps(
                by_type[ptype],
                stratum_name="post_type_community_balanced",
                quota=q,
                sampling_method="probability",
                probability_pool_size=len(by_type[ptype]),
            )
            s2_sel.extend(picked)
            selected_uris_all.update(p["uri"] for p in picked)
    stratum_reports["post_type_community_balanced"] = {
        "requested": quotas["post_type_community_balanced"],
        "candidate_pool": len(s2_pool),
        "selected": len(s2_sel),
        "shortfall": max(0, quotas["post_type_community_balanced"] - len(s2_sel)),
        "sampling_method": "probability",
    }

    # 3. moderation_enrichment (15%): heuristic enrichment (inclusion_probability MUST be None)
    s3_pool = [
        p
        for p in remaining_pool
        if p["uri"] not in selected_uris_all
        and (
            bool(p["moderation_join"]["active_record_labels"])
            or bool(p["moderation_join"]["unresolved_cid_labels"])
            or bool(p["moderation_join"]["author_account_labels_feature_only"])
            or float(p.get("regex_baseline_spam") or 0.0) >= 0.025
        )
    ]
    s3_sel = _select_with_caps(
        s3_pool,
        stratum_name="moderation_enrichment",
        quota=quotas["moderation_enrichment"],
        sampling_method="heuristic_enrichment",
        probability_pool_size=None,
    )
    selected_uris_all.update(p["uri"] for p in s3_sel)
    stratum_reports["moderation_enrichment"] = {
        "requested": quotas["moderation_enrichment"],
        "candidate_pool": len(s3_pool),
        "selected": len(s3_sel),
        "shortfall": max(0, quotas["moderation_enrichment"] - len(s3_sel)),
        "sampling_method": "heuristic_enrichment",
    }

    # 4. editorial_boundary (15%): heuristic enrichment (inclusion_probability MUST be None)
    s4_pool = [
        p
        for p in remaining_pool
        if p["uri"] not in selected_uris_all and _is_editorial_boundary_candidate(p)
    ]
    s4_sel = _select_with_caps(
        s4_pool,
        stratum_name="editorial_boundary",
        quota=quotas["editorial_boundary"],
        sampling_method="heuristic_enrichment",
        probability_pool_size=None,
    )
    selected_uris_all.update(p["uri"] for p in s4_sel)
    stratum_reports["editorial_boundary"] = {
        "requested": quotas["editorial_boundary"],
        "candidate_pool": len(s4_pool),
        "selected": len(s4_sel),
        "shortfall": max(0, quotas["editorial_boundary"] - len(s4_sel)),
        "sampling_method": "heuristic_enrichment",
    }

    # 5. robustness_drift (5%): probability sample from remaining eligible train posts
    s5_pool = [p for p in remaining_pool if p["uri"] not in selected_uris_all]
    s5_sel = _select_with_caps(
        s5_pool,
        stratum_name="robustness_drift",
        quota=quotas["robustness_drift"],
        sampling_method="probability",
        probability_pool_size=len(s5_pool),
    )
    selected_uris_all.update(p["uri"] for p in s5_sel)
    stratum_reports["robustness_drift"] = {
        "requested": quotas["robustness_drift"],
        "candidate_pool": len(s5_pool),
        "selected": len(s5_sel),
        "shortfall": max(0, quotas["robustness_drift"] - len(s5_sel)),
        "sampling_method": "probability",
    }

    # Dev split items that are eligible get assigned to "dev_evaluation" stratum
    for p in prepared_posts:
        if p["split"] == "dev" and p["eligible_for_labeling"]:
            p["sampling_stratum"] = "dev_evaluation"
            p["sampling_method"] = "probability"
            dev_pool_size = sum(1 for x in prepared_posts if x["split"] == "dev" and x["eligible_for_labeling"])
            p["inclusion_probability"] = 1.0 if dev_pool_size > 0 else None
            p["selection_priority_key"] = _det_key("dev_evaluation", p["uri"])

    lang_hist = language_histogram(raw_rows)
    split_counts = Counter(p["split"] for p in prepared_posts)

    preparation_manifest = {
        "schema": "catbird-teacher-preparation-manifest-v1",
        "input_path": str(Path(input_path).resolve()),
        "input_sha256": input_file_sha256,
        "canonical_bank_sha256": expected_bank_sha256,
        "canonical_bank_matched": input_file_sha256 == expected_bank_sha256,
        "export_manifest_sha256": SEALED_EXPORT_SHA256,
        "holdout_path": str(Path(holdout_path).resolve()),
        "holdout_sha256": seal["holdout"]["sha256"],
        "holdout_rows_isolated": len(holdout_rows),
        "holdout_authors_isolated": len(holdout_authors),
        "total_loaded_rows": len(raw_rows),
        "corpus_size_assumption": "actual_validated_local_rows_only; never assumes 500k rows exist",
        "language_audit": {
            "rows": lang_hist["rows"],
            "eligible_declared_english_rows": lang_hist["eligible_declared_english_rows"],
            "unscored_rows": lang_hist["unscored_rows"],
            "unscored_reasons": lang_hist["unscored_reasons"],
        },
        "split_row_counts": dict(sorted(split_counts.items())),
        "split_author_counts": {k: len(v) for k, v in sorted(split_authors.items())},
        "author_overlap_across_splits": 0,
        "leakage_quarantine_counts": dict(sorted(quarantine_counts.items())),
        "sampling_seed": selection_seed,
        "sampling_strata": stratum_reports,
        "total_selected_train_items": len(selected_uris_all),
        "moderation_join_audit": mod_result["audit"],
    }
    preparation_manifest["manifest_sha256"] = canonical_sha256(preparation_manifest)

    return {
        "manifest": preparation_manifest,
        "posts": prepared_posts,
        "moderation": mod_result,
    }
