"""Source-specific moderation weak-supervision folding and join audit."""
from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
import gzip
import io
import json
from pathlib import Path
import struct
from typing import Any

KNOWN_SOURCE_POLICIES: dict[str, dict[str, str]] = {
    "did:plc:ar7c4by46qjdydhdevvrndac": {
        "name": "Bluesky Moderation Service",
        "training_permission": "unknown_public_protocol_only",
        "role": "reference_moderation_service",
    },
    "did:plc:e4elbtctnfqocyfcml6h2lf7": {
        "name": "Skywatch Blue",
        "training_permission": "unknown_public_protocol_only",
        "role": "weak_supervision_spam_engagement_abuse",
    },
    "did:plc:d2mkddsbmnrgr3domzg5qexf": {
        "name": "Blacksky Moderation",
        "training_permission": "unknown_public_protocol_only",
        "role": "complementary_community_moderation",
    },
}


def classify_label_family(val: str, src: str, uri: str) -> str:
    if val.startswith("!"):
        return "system_bang"
    if uri.startswith(f"at://{src}/") or uri == src:
        return "self_label"
    if val in ("spam", "porn", "sexual", "nudity", "graphic-media", "gore", "rude", "intolerant", "threat", "impersonation", "scam"):
        return "moderation_annotation"
    return "community_or_topic_label"


def classify_subject_scope(uri: str) -> str:
    if uri.startswith("did:"):
        return "account"
    if uri.startswith("at://"):
        return "record"
    return "unknown"


def _parse_iso_utc(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    cleaned = ts.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _decode_cbor_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {k: _decode_cbor_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_cbor_bytes(v) for v in value]
    return value


def iter_packed_cbor_labels(directory: Path, *, max_chunks: int | None = None) -> list[dict[str, Any]]:
    """Read raw label events from packed v2 .cbor.gz chunks without touching any writer lock."""
    import cbor2

    chunks_jsonl = directory / "chunks.jsonl"
    events: list[dict[str, Any]] = []
    if not chunks_jsonl.exists():
        return events

    lines = [line for line in chunks_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_chunks is not None:
        lines = lines[:max_chunks]

    for line in lines:
        entry = json.loads(line)
        rel_chunk = entry.get("chunk_path") or f"chunks/{entry['compressed_sha256']}.frames.gz"
        chunk_path = directory / rel_chunk
        raw = gzip.decompress(chunk_path.read_bytes())
        offset = 0
        while offset < len(raw):
            if len(raw) - offset < 8:
                raise ValueError(f"Partial frame length in {chunk_path}")
            (length,) = struct.unpack(">Q", raw[offset : offset + 8])
            offset += 8
            frame = raw[offset : offset + length]
            offset += length
            buf = io.BytesIO(frame)
            header = cbor2.load(buf)
            body = cbor2.load(buf)
            if header.get("op") == 1 and header.get("t") == "#labels":
                seq = body.get("seq")
                for lbl in body.get("labels", []):
                    item = _decode_cbor_bytes(lbl)
                    item["_seq"] = seq
                    events.append(item)
    return events


def load_moderation_events(source_path: Path | None, *, max_chunks: int | None = 2) -> list[dict[str, Any]]:
    """Load moderation label events from a JSONL file or packed archive directory."""
    if source_path is None or not Path(source_path).exists():
        return []
    path = Path(source_path)
    if path.is_dir():
        if (path / "chunks.jsonl").exists():
            return iter_packed_cbor_labels(path, max_chunks=max_chunks)
        jsonl_files = sorted(path.glob("*.jsonl"))
        events: list[dict[str, Any]] = []
        for f in jsonl_files:
            if f.name in ("journal.jsonl", "chunks.jsonl"):
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))
        return events
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fold_and_join_moderation_labels(
    events: list[dict[str, Any]],
    posts: list[dict[str, Any]],
    *,
    cutoff_utc: str = "2026-10-20T23:59:59+00:00",
    eligible_uris: set[str] | None = None,
) -> dict[str, Any]:
    """Fold positive/negated moderation events within (src, uri, cid, val) and audit join against posts."""
    cutoff_dt = _parse_iso_utc(cutoff_utc)
    if cutoff_dt is None:
        raise ValueError(f"Invalid cutoff_utc: {cutoff_utc!r}")

    posts_by_uri: dict[str, dict[str, Any]] = {str(p["uri"]): p for p in posts if "uri" in p}
    posts_by_author: dict[str, list[str]] = {}
    for p in posts:
        did = str(p.get("author_did") or "")
        if did:
            posts_by_author.setdefault(did, []).append(str(p["uri"]))

    if eligible_uris is None:
        eligible_uris = set(posts_by_uri.keys())

    # Fold strictly within (src, uri, cid_or_none, val)
    grouped: dict[tuple[str, str, str | None, str], list[tuple[int, datetime, dict[str, Any]]]] = {}
    raw_source_counts: Counter[str] = Counter()
    raw_value_counts: Counter[str] = Counter()
    unverified_sig_count = 0
    unknown_perm_count = 0
    after_cutoff_ignored = 0

    for idx, ev in enumerate(events):
        src = str(ev.get("src") or "")
        uri = str(ev.get("uri") or "")
        val = str(ev.get("val") or "")
        cts_str = str(ev.get("cts") or "")
        if not src or not uri or not val or not cts_str:
            raise ValueError(f"Moderation event at index {idx} missing required src/uri/val/cts fields")

        cts_dt = _parse_iso_utc(cts_str)
        if cts_dt is None:
            raise ValueError(f"Invalid cts timestamp {cts_str!r}")
        if cts_dt > cutoff_dt:
            after_cutoff_ignored += 1
            continue

        raw_source_counts[src] += 1
        raw_value_counts[val] += 1

        sig_verified = bool(ev.get("signature_verified") is True)
        if not sig_verified:
            unverified_sig_count += 1

        perm = ev.get("training_permission") or KNOWN_SOURCE_POLICIES.get(src, {}).get(
            "training_permission", "unknown_public_protocol_only"
        )
        if str(perm).startswith("unknown"):
            unknown_perm_count += 1

        cid_val = str(ev["cid"]) if ev.get("cid") else None
        key = (src, uri, cid_val, val)
        grouped.setdefault(key, []).append((idx, cts_dt, ev))

    active_states: list[dict[str, Any]] = []
    retracted_states: list[dict[str, Any]] = []
    expired_states: list[dict[str, Any]] = []

    for (src, uri, cid_val, val), history in grouped.items():
        # Order by (cts_dt, original arrival index) within the same source
        history.sort(key=lambda item: (item[1], item[0]))
        _, latest_cts, latest_ev = history[-1]
        is_neg = bool(latest_ev.get("neg") is True)
        exp_dt = _parse_iso_utc(latest_ev.get("exp"))
        sig_verified = bool(latest_ev.get("signature_verified") is True)
        perm = latest_ev.get("training_permission") or KNOWN_SOURCE_POLICIES.get(src, {}).get(
            "training_permission", "unknown_public_protocol_only"
        )
        scope = classify_subject_scope(uri)
        family = classify_label_family(val, src, uri)

        record_state = {
            "src": src,
            "uri": uri,
            "cid": cid_val,
            "val": val,
            "cts": latest_ev.get("cts"),
            "exp": latest_ev.get("exp"),
            "neg": is_neg,
            "has_signature_bytes": "sig" in latest_ev and latest_ev["sig"] is not None,
            "signature_verified": sig_verified,
            "training_permission": perm,
            "subject_scope": scope,
            "label_family": family,
            "events_folded": len(history),
        }

        if is_neg:
            record_state["resolution_status"] = "retracted_by_negation"
            record_state["certified_negative_example"] = False
            retracted_states.append(record_state)
            continue

        if exp_dt is not None and exp_dt <= cutoff_dt:
            record_state["resolution_status"] = "expired_at_cutoff"
            expired_states.append(record_state)
            continue

        record_state["resolution_status"] = "active"
        active_states.append(record_state)

    # Join active states to local corpus with strict account/post separation and CID version matching
    per_post_annotations: dict[str, dict[str, Any]] = {}
    for uri, post in posts_by_uri.items():
        per_post_annotations[uri] = {
            "uri": uri,
            "post_cid": post.get("cid"),
            "observation_state": "unlabeled_observation",
            "absence_is_negative_example": False,
            "active_record_labels": [],
            "unresolved_cid_labels": [],
            "retracted_record_labels": [],
            "expired_record_labels": [],
            "author_account_labels_feature_only": [],
            "source_disagreement": False,
        }

    account_scoped_active = 0
    record_scoped_active = 0
    exact_uri_cid_matches = 0
    uri_only_record_matches = 0
    unresolved_cid_matches = 0
    unmatched_subject_records = 0
    matched_eligible_uris: set[str] = set()
    matched_authors: set[str] = set()

    for state in active_states:
        uri = state["uri"]
        scope = state["subject_scope"]
        if scope == "account":
            account_scoped_active += 1
            # Account label MUST NEVER become a post label for all posts by that author
            for post_uri in posts_by_author.get(uri, []):
                per_post_annotations[post_uri]["author_account_labels_feature_only"].append({
                    **state,
                    "account_feature_only": True,
                    "post_target_eligible": False,
                })
        elif scope == "record":
            record_scoped_active += 1
            if uri not in posts_by_uri:
                unmatched_subject_records += 1
                continue
            post = posts_by_uri[uri]
            post_cid = str(post["cid"]) if post.get("cid") else None
            label_cid = state["cid"]
            if label_cid is not None:
                if post_cid is None or post_cid != label_cid:
                    unresolved_cid_matches += 1
                    per_post_annotations[uri]["unresolved_cid_labels"].append({
                        **state,
                        "join_status": "unresolved_cid_version",
                        "post_cid": post_cid,
                    })
                    continue
                exact_uri_cid_matches += 1
                join_status = "exact_uri_and_cid_match"
            else:
                uri_only_record_matches += 1
                join_status = "exact_uri_no_cid_constraint"

            per_post_annotations[uri]["active_record_labels"].append({
                **state,
                "join_status": join_status,
            })
            per_post_annotations[uri]["observation_state"] = "weakly_labeled_record"
            if uri in eligible_uris:
                matched_eligible_uris.add(uri)
                if post.get("author_did"):
                    matched_authors.add(str(post["author_did"]))

    for state in retracted_states:
        uri = state["uri"]
        if state["subject_scope"] == "record" and uri in per_post_annotations:
            per_post_annotations[uri]["retracted_record_labels"].append(state)

    for state in expired_states:
        uri = state["uri"]
        if state["subject_scope"] == "record" and uri in per_post_annotations:
            per_post_annotations[uri]["expired_record_labels"].append(state)

    # Detect multi-source or retraction disagreement on any post
    for uri, info in per_post_annotations.items():
        active_srcs = {lbl["src"] for lbl in info["active_record_labels"]}
        retracted_srcs = {lbl["src"] for lbl in info["retracted_record_labels"]}
        active_vals = {lbl["val"] for lbl in info["active_record_labels"]}
        if len(active_srcs) > 1 or (active_srcs and retracted_srcs) or len(active_vals) > 1:
            info["source_disagreement"] = True

    audit = {
        "schema": "catbird-moderation-join-audit-v1",
        "cutoff_utc": cutoff_utc,
        "raw_label_events": len(events),
        "events_after_cutoff_ignored": after_cutoff_ignored,
        "distinct_src_uri_cid_val_tuples": len(grouped),
        "active_at_cutoff": len(active_states),
        "retracted_by_negation": len(retracted_states),
        "expired_at_cutoff": len(expired_states),
        "account_scoped_active": account_scoped_active,
        "record_scoped_active": record_scoped_active,
        "exact_uri_and_cid_matches": exact_uri_cid_matches,
        "uri_only_record_matches": uri_only_record_matches,
        "unresolved_cid_version_matches": unresolved_cid_matches,
        "unmatched_subject_records": unmatched_subject_records,
        "eligible_text_examples_matched": len(matched_eligible_uris),
        "distinct_matched_authors": len(matched_authors),
        "unverified_signature_events": unverified_sig_count,
        "unknown_training_permission_events": unknown_perm_count,
        "per_source_raw_counts": dict(sorted(raw_source_counts.items())),
        "per_value_raw_counts": dict(sorted(raw_value_counts.items())),
        "invariants_enforced": {
            "account_labels_never_label_posts": True,
            "absence_of_label_is_never_negative": True,
            "negation_retracts_only_and_is_not_certified_negative": True,
            "cid_mismatch_or_missing_post_cid_kept_unresolved": True,
            "incomplete_moderation_never_blocks_text_only_lane": True,
        },
    }
    return {
        "audit": audit,
        "per_post": per_post_annotations,
        "active_states": active_states,
        "retracted_states": retracted_states,
        "expired_states": expired_states,
    }
