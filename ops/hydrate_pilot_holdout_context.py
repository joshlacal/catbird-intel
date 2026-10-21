#!/usr/bin/env python3
"""Hydrate parent and quote context URIs for the 2,000-post pilot and 600-post holdout.

Queries netcup ClickHouse in read-only mode:
1. Primary: `catbird.v_posts_live` anti-joined against `catbird.deletions`,
   `catbird.v_inactive_accounts`, and `catbird.v_hidden_from_algorithms`.
2. Fallback: `feed.posts` anti-joined against `catbird.deletions`,
   `catbird.v_inactive_accounts`, and `catbird.v_hidden_from_algorithms`.

Outputs:
  catbird-intel/eval/pilot_holdout_parent_quote_hydration_20261021.jsonl
  catbird-intel/eval/pilot_holdout_parent_quote_hydration_20261021.manifest.json
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parents[1]
BANK_PATH = (
    pathlib.Path.home()
    / "catbird-lake-local/drive-sync-staging/training-banks/phase3_health_training_150k_v1_unlabeled.parquet"
)
PILOT_DB_PATH = REPO / "eval/anthropic_teacher_pilot_v1/state.sqlite3"
HOLDOUT_PATH = REPO / "eval/phase3_holdout_600_unlabeled.jsonl"
OUT_JSONL = REPO / "eval/pilot_holdout_parent_quote_hydration_20261021.jsonl"
OUT_MANIFEST = REPO / "eval/pilot_holdout_parent_quote_hydration_20261021.manifest.json"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_at_uri(uri: str) -> Tuple[str, str] | None:
    if not uri.startswith("at://"):
        return None
    parts = uri[5:].split("/")
    if len(parts) != 3 or parts[1] != "app.bsky.feed.post":
        return None
    did, _, rkey = parts
    if not did or not rkey:
        return None
    return did, rkey


def sql_quote(val: str) -> str:
    return "'" + val.replace("\\", "\\\\").replace("'", "\\'") + "'"


def ch_query_json_each_row(query: str) -> List[Dict[str, Any]]:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "netcup", "clickhouse-client", "--format", "JSONEachRow"],
        input=query.encode("utf-8"),
        capture_output=True,
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ClickHouse query failed: {proc.stderr.decode('utf-8', errors='replace')}")
    rows: List[Dict[str, Any]] = []
    for line in proc.stdout.decode("utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    table = pq.read_table(BANK_PATH, columns=["uri"])
    bank_uris: Set[str] = {str(u) for u in table.column("uri").to_pylist() if u}

    conn = sqlite3.connect(PILOT_DB_PATH)
    pilot_rows = conn.execute(
        """
        SELECT p.uri, p.raw_row_json
        FROM posts p
        JOIN requests r ON p.uri = r.uri
        WHERE r.stage = 'pilot' AND r.model_tier = 'haiku'
        ORDER BY p.uri ASC
        """
    ).fetchall()
    conn.close()

    pilot_self_uris: Set[str] = set()
    pilot_referenced_uris: Set[str] = set()
    for uri, raw_json in pilot_rows:
        pilot_self_uris.add(str(uri))
        raw = json.loads(raw_json)
        for key in ("reply_parent_uri", "quote_uri"):
            val = str(raw.get(key) or "").strip()
            if val:
                pilot_referenced_uris.add(val)

    holdout_self_uris: Set[str] = set()
    holdout_referenced_uris: Set[str] = set()
    with HOLDOUT_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("uri"):
                holdout_self_uris.add(str(d["uri"]))
            for key in ("reply_parent_uri", "quote_uri"):
                val = str(d.get(key) or "").strip()
                if val:
                    holdout_referenced_uris.add(val)

    pilot_external_missing = sorted(pilot_referenced_uris - pilot_self_uris)
    holdout_external_missing = sorted(holdout_referenced_uris - holdout_self_uris)
    all_missing = sorted(set(pilot_external_missing) | set(holdout_external_missing))

    parsed_pairs: List[Tuple[str, str, str]] = []
    for uri in all_missing:
        parsed = parse_at_uri(uri)
        if parsed is not None:
            parsed_pairs.append((uri, parsed[0], parsed[1]))

    tuple_literals = ", ".join(f"({sql_quote(did)}, {sql_quote(rkey)})" for _, did, rkey in parsed_pairs)
    q_catbird = f"""
    SELECT
        concat('at://', p.did, '/app.bsky.feed.post/', p.rkey) AS uri,
        p.cid AS cid,
        p.did AS author_did,
        toString(p.created_at) AS created_at,
        p.text AS text,
        p.langs AS langs,
        p.reply_parent AS reply_parent_uri,
        p.reply_root AS reply_root_uri,
        p.quote_uri AS quote_uri,
        p.embed_type AS embed_type,
        p.image_count AS image_count,
        p.has_video AS has_video,
        p.external_uri AS external_uri,
        p.facet_tags AS facet_tags,
        'catbird.v_posts_live' AS hydration_source
    FROM catbird.v_posts_live AS p
    WHERE (p.did, p.rkey) IN ({tuple_literals})
      AND p.did NOT IN (SELECT did FROM catbird.v_inactive_accounts)
      AND p.did NOT IN (SELECT did FROM catbird.v_hidden_from_algorithms)
      AND (p.did, p.rkey) NOT IN (
          SELECT did, rkey FROM catbird.deletions
          WHERE collection = 'app.bsky.feed.post'
      )
    """
    catbird_rows = ch_query_json_each_row(q_catbird)
    hydrated_by_uri: Dict[str, Dict[str, Any]] = {}
    for row in sorted(catbird_rows, key=lambda r: (str(r.get("uri")), str(r.get("created_at")))):
        hydrated_by_uri[str(row["uri"])] = row

    remaining_uris = [uri for uri in all_missing if uri not in hydrated_by_uri]
    feed_rows: List[Dict[str, Any]] = []
    if remaining_uris:
        uri_literals = ", ".join(sql_quote(u) for u in remaining_uris)
        q_feed = f"""
        SELECT
            f.uri AS uri,
            '' AS cid,
            f.author AS author_did,
            toString(f.created) AS created_at,
            f.text AS text,
            CAST([] AS Array(String)) AS langs,
            '' AS reply_parent_uri,
            '' AS reply_root_uri,
            '' AS quote_uri,
            '' AS embed_type,
            toUInt8(0) AS image_count,
            toUInt8(0) AS has_video,
            '' AS external_uri,
            CAST([] AS Array(String)) AS facet_tags,
            'feed.posts' AS hydration_source
        FROM feed.posts AS f
        WHERE f.uri IN ({uri_literals})
          AND f.author NOT IN (SELECT did FROM catbird.v_inactive_accounts)
          AND f.author NOT IN (SELECT did FROM catbird.v_hidden_from_algorithms)
          AND (f.author, splitByChar('/', f.uri)[5]) NOT IN (
              SELECT did, rkey FROM catbird.deletions
              WHERE collection = 'app.bsky.feed.post'
          )
        """
        feed_rows = ch_query_json_each_row(q_feed)
        for row in sorted(feed_rows, key=lambda r: (str(r.get("uri")), str(r.get("created_at")))):
            uri = str(row["uri"])
            if uri not in hydrated_by_uri:
                hydrated_by_uri[uri] = row

    ordered_rows = [hydrated_by_uri[u] for u in sorted(hydrated_by_uri.keys())]
    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for row in ordered_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    jsonl_sha256 = sha256_file(OUT_JSONL)
    resolved_uris = set(hydrated_by_uri.keys()) | bank_uris
    pilot_resolved = len(pilot_referenced_uris & resolved_uris)
    holdout_resolved = len(holdout_referenced_uris & resolved_uris)
    total_referenced = pilot_referenced_uris | holdout_referenced_uris
    total_resolved = len(total_referenced & resolved_uris)

    manifest = {
        "schema": "catbird-parent-quote-hydration-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hydration_jsonl_path": str(OUT_JSONL),
        "hydration_jsonl_sha256": jsonl_sha256,
        "hydration_rows_written": len(ordered_rows),
        "source_bank_path": str(BANK_PATH),
        "source_bank_sha256": sha256_file(BANK_PATH),
        "pilot_db_path": str(PILOT_DB_PATH),
        "holdout_path": str(HOLDOUT_PATH),
        "holdout_sha256": sha256_file(HOLDOUT_PATH),
        "counts": {
            "pilot_referenced_uris": len(pilot_referenced_uris),
            "pilot_already_in_pilot_split": len(pilot_referenced_uris & pilot_self_uris),
            "pilot_already_in_local_bank": len(pilot_referenced_uris & bank_uris),
            "pilot_missing_uris": len(pilot_external_missing),
            "pilot_resolved_uris": pilot_resolved,
            "pilot_resolution_rate": round(pilot_resolved / max(1, len(pilot_referenced_uris)), 4),
            "holdout_referenced_uris": len(holdout_referenced_uris),
            "holdout_already_in_holdout_split": len(holdout_referenced_uris & holdout_self_uris),
            "holdout_already_in_local_bank": len(holdout_referenced_uris & bank_uris),
            "holdout_missing_uris": len(holdout_external_missing),
            "holdout_resolved_uris": holdout_resolved,
            "holdout_resolution_rate": round(holdout_resolved / max(1, len(holdout_referenced_uris)), 4),
            "union_referenced_uris": len(total_referenced),
            "union_missing_uris": len(all_missing),
            "hydrated_from_catbird_v_posts_live": sum(
                1 for r in ordered_rows if r["hydration_source"] == "catbird.v_posts_live"
            ),
            "hydrated_from_feed_posts_fallback": sum(
                1 for r in ordered_rows if r["hydration_source"] == "feed.posts"
            ),
            "union_resolved_uris": total_resolved,
            "union_resolution_rate": round(total_resolved / max(1, len(total_referenced)), 4),
            "unresolvable_deleted_or_pre_window_uris": len(total_referenced) - total_resolved,
        },
        "anti_joins_enforced": [
            "catbird.deletions (collection = 'app.bsky.feed.post')",
            "catbird.v_inactive_accounts",
            "catbird.v_hidden_from_algorithms",
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
