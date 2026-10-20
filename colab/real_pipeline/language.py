"""Declared-language scope and metadata audit; this is not language detection."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


LANGUAGE_CONTRACT = {
    "schema": "catbird-declared-english-scope-v3",
    "eligible": "nonempty valid langs list whose every primary subtag is en",
    "evidence": "post-declared metadata; actual text language is not independently verified",
    "missing_invalid_non_english_or_mixed": "unscored",
    "unscored_value": None,
    "downstream_health_filter": "neutral_no_model_based_penalty; never coerce null to zero or a probability",
    "other_eligibility_rules": "moderation, visibility and deletion rules still apply",
}
_TAG = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")


def language_scope(row: dict, prefix: str = "") -> dict:
    """Conservatively scope by explicit declarations, preserving missing/mixed cases."""
    value = row.get(f"{prefix}_langs" if prefix else "langs")
    if value is None or value == []:
        return {"in_scope": False, "reason": "missing_language_metadata"}
    if not isinstance(value, list) or any(not isinstance(tag, str) or not _TAG.fullmatch(tag) for tag in value):
        return {"in_scope": False, "reason": "invalid_language_metadata"}
    primary = {tag.lower().split("-")[0] for tag in value}
    if primary == {"en"}:
        return {"in_scope": True, "reason": None}
    return {"in_scope": False, "reason": "mixed_languages" if len(primary) > 1 else "unsupported_language"}


def language_histogram(rows: list[dict]) -> dict:
    combinations, tags, primary, reasons = Counter(), Counter(), Counter(), Counter()
    null = empty = invalid = multi_tag = mixed_primary = eligible = 0
    for row in rows:
        value = row.get("langs")
        combinations[json.dumps(value, ensure_ascii=True, separators=(",", ":"))] += 1
        null += value is None
        empty += value == []
        valid = isinstance(value, list) and all(isinstance(tag, str) and _TAG.fullmatch(tag) for tag in value)
        invalid += value is not None and not valid
        if valid:
            tags.update(set(tag.lower() for tag in value))
            roots = {tag.lower().split("-")[0] for tag in value}
            primary.update(roots)
            multi_tag += len(value) > 1
            mixed_primary += len(roots) > 1
        scope = language_scope(row)
        eligible += scope["in_scope"]
        if not scope["in_scope"]:
            reasons[scope["reason"]] += 1
    return {
        "rows": len(rows), "eligible_declared_english_rows": eligible,
        "unscored_rows": len(rows) - eligible, "unscored_reasons": dict(sorted(reasons.items())),
        "missing_or_null_rows": null, "empty_list_rows": empty, "invalid_metadata_rows": invalid,
        "multiple_declared_tags_rows": multi_tag, "multiple_primary_languages_rows": mixed_primary,
        "raw_declared_combinations": dict(sorted(combinations.items(), key=lambda item: (-item[1], item[0]))),
        "normalized_tag_row_counts": dict(sorted(tags.items(), key=lambda item: (-item[1], item[0]))),
        "primary_language_row_counts": dict(sorted(primary.items(), key=lambda item: (-item[1], item[0]))),
        "counting_note": "A multilingual row contributes to multiple tag/primary counts; these are not a partition.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hasher = hashlib.sha256()
    with args.input.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    if digest != args.expected_sha256:
        parser.error("Input does not match the sealed bank digest")
    import pyarrow.parquet as pq
    # Read only the declaration column. No post text, classifier or model is loaded.
    rows = pq.read_table(args.input, columns=["langs"]).to_pylist()
    report = {"schema": "catbird-language-audit-v3", "source_sha256": digest,
              "language_contract": LANGUAGE_CONTRACT, "histogram": language_histogram(rows),
              "model_inference_rows": 0, "human_language_verification": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: report["histogram"][key] for key in
                      ("rows", "eligible_declared_english_rows", "unscored_rows", "unscored_reasons")}))


if __name__ == "__main__":
    main()
