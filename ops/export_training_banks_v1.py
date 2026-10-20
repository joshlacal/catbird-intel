#!/usr/bin/env python3
"""Export clean (unlabeled) Phase 3 training bank + author-disjoint 600-post holdout.

Replaces the 2026-10-19 bank, which aliased the regex ingest scores as
`teacher_*` columns (label laundering). Rules enforced here:

* No column is named `teacher_*`. The regex ingest scores are kept only as
  `regex_baseline_*` so they can be used as a comparison baseline, never as targets.
* Authors are split by `cityHash64(did) % 20`: bucket 0 is holdout-only and never
  appears in the training bank. The holdout is additionally checked against all
  local lakehouse training/pairs banks, including the invalidated Phase 5 bank.
* The holdout is real posts only, stratified by post type and by sampling keyword
  lists. The keywords decide *which posts are sampled*, never what label they get.
* Only live posts (`catbird.v_posts_live`: FINAL, not deleted, author active) from
  authors not in `catbird.v_hidden_from_algorithms`. The deletion ledger is also
  excluded conservatively. No database state is changed.
* Output files are immutable. An interrupted run can reuse a verified bank; an
  existing completed holdout is verified and reported, never resampled.

Outputs (local):
  drive-sync-staging/training-banks/phase3_health_training_150k_v1_unlabeled.parquet
  catbird-intel/eval/phase3_holdout_600_unlabeled.jsonl  (+ .sha256)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from collections import Counter
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parents[1]
LAKE = pathlib.Path.home() / "catbird-lake-local"
STAGING = LAKE / "drive-sync-staging/training-banks"
BANK = STAGING / "phase3_health_training_150k_v1_unlabeled.parquet"
BANK_RECEIPT = BANK.with_suffix(".manifest.json")
EVAL_DIR = REPO / "eval"
HOLDOUT = EVAL_DIR / "phase3_holdout_600_unlabeled.jsonl"
HOLDOUT_SHA = HOLDOUT.with_suffix(".jsonl.sha256")
MANIFEST = EVAL_DIR / "phase3_v1_export_manifest.json"

LIVE = """
catbird.v_posts_live AS p
WHERE length(p.text) >= 20
  AND p.did NOT IN (SELECT did FROM catbird.v_hidden_from_algorithms)
  AND (p.did, p.rkey) NOT IN (
      SELECT did, rkey FROM catbird.deletions
      WHERE collection = 'app.bsky.feed.post'
  )
"""

COLS = """
    concat('at://', p.did, '/app.bsky.feed.post/', p.rkey) AS uri,
    p.did AS author_did,
    cityHash64(p.did) % 20 AS author_split_bucket,
    p.created_at AS created_at,
    p.text AS text,
    p.langs AS langs,
    multiIf(p.quote_uri != '', 'quote', p.reply_parent != '', 'reply', 'original') AS post_type,
    p.reply_parent AS reply_parent_uri,
    p.reply_root AS reply_root_uri,
    p.quote_uri AS quote_uri,
    p.embed_type AS embed_type,
    p.image_count AS image_count,
    p.has_video AS has_video,
    p.external_uri AS external_uri,
    p.facet_tags AS facet_tags
"""

# Sampling-only keyword lists (used to pick posts, never to label them).
POLITICAL_KW = ["trump", "biden", "election", "congress", "senate", "democrat", "republican", "gop",
                "maga", "tariff", "gaza", "israel", "ukraine", "putin", "immigration", "ice ", "supreme court",
                "governor", "vote", "policy", "fascis", "liberal", "conservative", "labour", "tory"]
HEATED_NONPOL_KW = ["refs", "referee", "ref ", "officiating", "trade deadline", "fire the coach", "mvp",
                    "patch notes", "nerf", "dev team", "console war", "gamers", "framerate", "apple tax",
                    "shipped broken", "worst season", "choke", "playoffs", "fans", "overrated"]
TRAP_KW = ["rust", "swift", "macro", "lab ", "homelab", "football", "indie", "espresso", "#humor",
           "wrestling", "state", "party", "house"]


def ch(query: str) -> bytes:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "netcup", "clickhouse-client"],
        input=query.encode(), capture_output=True, timeout=300,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return result.stdout


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_once(path: pathlib.Path, value: dict) -> None:
    with path.open("x") as fh:
        json.dump(value, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")


def bank_inventory() -> list[pathlib.Path]:
    """Enumerate both current and quarantined local lakehouse training banks."""
    return sorted(
        p for p in LAKE.rglob("*.parquet")
        if any(word in str(p.relative_to(LAKE)) for word in ("training", "pairs", "bank"))
        and p != BANK
    )


def training_authors(paths: list[pathlib.Path]) -> tuple[set[str], list[dict]]:
    authors: set[str] = set()
    inventory = []
    for path in paths:
        schema = pq.read_schema(path)
        columns = [c for c in schema.names if c == "author_did" or c.endswith("_author_did")]
        if not columns:
            raise RuntimeError(f"Cannot identify author columns in training bank: {path}")
        table = pq.read_table(path, columns=columns)
        local_authors = {did for c in columns for did in table[c].to_pylist() if did}
        authors.update(local_authors)
        inventory.append({
            "path": str(path), "rows": table.num_rows, "authors": len(local_authors),
            "author_columns": columns, "sha256": sha256(path),
        })
    return authors, inventory


def kw_expr(kws: list[str]) -> str:
    return "multiSearchAnyCaseInsensitive(p.text, [" + ",".join("'" + k.replace("'", "''") + "'" for k in kws) + "])"


def export_bank(cutoff: str) -> dict:
    # The existing view's fixed schema predates the regex score columns. Keep the
    # view authoritative for selection and join only the stored baseline values.
    q = f"""
    SELECT {COLS},
        b.politicalness AS regex_baseline_politicalness,
        b.heat AS regex_baseline_heat,
        b.engagement_bait AS regex_baseline_engagement_bait,
        b.spam AS regex_baseline_spam
    FROM (
        SELECT * FROM {LIVE}
          AND cityHash64(p.did) % 20 != 0
          AND p.indexed_at <= toDateTime64('{cutoff}', 3, 'UTC')
        ORDER BY cityHash64(p.did, p.rkey), p.did, p.rkey
        LIMIT 150000
    ) AS p
    LEFT JOIN (
        SELECT did, rkey, politicalness, heat, engagement_bait, spam
        FROM catbird.posts FINAL
    ) AS b ON p.did = b.did AND p.rkey = b.rkey
    ORDER BY cityHash64(p.did, p.rkey), p.did, p.rkey
    SETTINGS output_format_parquet_compression_method = 'zstd',
        max_threads = 2, priority = 10, max_memory_usage = 2000000000
    FORMAT Parquet
    """
    data = ch(q)
    bank = pq.read_table(pa.BufferReader(data))
    if any(c.startswith("teacher_") for c in bank.column_names):
        raise RuntimeError("Unlabeled bank unexpectedly contains teacher columns")
    if 0 in set(bank["author_split_bucket"].to_pylist()):
        raise RuntimeError("Holdout bucket leaked into training bank")
    if len(set(bank["uri"].to_pylist())) != bank.num_rows:
        raise RuntimeError("Duplicate post URI in training bank")
    with BANK.open("xb") as fh:
        fh.write(data)
    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_indexed_at_cutoff_utc": cutoff,
        "path": str(BANK), "rows": bank.num_rows,
        "authors": len(set(bank["author_did"].to_pylist())),
        "requested_rows": 150000, "shortfall": max(0, 150000 - bank.num_rows),
        "bytes": BANK.stat().st_size, "sha256": sha256(BANK),
        "columns": bank.column_names, "author_split_buckets": sorted(set(bank["author_split_bucket"].to_pylist())),
        "source": "catbird.v_posts_live; stored regex scores joined from catbird.posts FINAL",
        "query_sha256": hashlib.sha256(q.encode()).hexdigest(),
    }
    write_json_once(BANK_RECEIPT, receipt)
    return receipt


def export_holdout(exclude_authors: set[str], cutoff: str) -> tuple[list[dict], list[dict]]:
    strata = [
        ("political_kw", f"{kw_expr(POLITICAL_KW)}", 120),
        ("heated_nonpolitical_kw", f"{kw_expr(HEATED_NONPOL_KW)} AND NOT {kw_expr(POLITICAL_KW)}", 120),
        ("collision_trap_kw", f"{kw_expr(TRAP_KW)} AND NOT {kw_expr(POLITICAL_KW)}", 120),
        ("random_quote", "p.quote_uri != ''", 80),
        ("random_reply", "p.reply_parent != '' AND p.quote_uri = ''", 80),
        ("random_original", "p.reply_parent = '' AND p.quote_uri = ''", 80),
    ]
    rows: list[dict] = []
    seen_uris: set[str] = set()
    seen_authors: set[str] = set()
    sampling = []
    for name, cond, n in strata:
        q = f"""
        SELECT {COLS}
        FROM {LIVE}
          AND cityHash64(p.did) % 20 = 0
          AND has(p.langs, 'en')
          AND p.indexed_at <= toDateTime64('{cutoff}', 3, 'UTC')
          AND {cond}
        ORDER BY cityHash64(p.did, p.rkey, '{name}'), p.did, p.rkey
        LIMIT 1 BY p.did
        SETTINGS max_threads = 2, priority = 10, max_memory_usage = 2000000000
        FORMAT JSONEachRow
        """
        # Fetch all eligible authors in this bounded live-window stratum so that
        # an exclusion-heavy first page cannot create an artificial shortfall.
        # JSON allows U+0085/U+2028/U+2029 in strings; str.splitlines() would
        # incorrectly split genuine post text at those characters.
        candidates = [line for line in ch(q).decode().split("\n") if line]
        got = 0
        for line in candidates:
            r = json.loads(line)
            # at most one post per author across the whole holdout; never a training author
            if r["uri"] in seen_uris or r["author_did"] in seen_authors or r["author_did"] in exclude_authors:
                continue
            r["stratum"] = name
            rows.append(r)
            seen_uris.add(r["uri"])
            seen_authors.add(r["author_did"])
            got += 1
            if got >= n:
                break
        sampling.append({
            "stratum": name, "requested": n, "selected": got, "shortfall": n - got,
            "candidate_authors": len(candidates),
            "query_sha256": hashlib.sha256(q.encode()).hexdigest(),
        })
        print(f"stratum {name}: {got}/{n} from {len(candidates)} candidate authors", flush=True)
    return rows, sampling


def main() -> None:
    STAGING.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if HOLDOUT.exists() or MANIFEST.exists() or HOLDOUT_SHA.exists():
        if not all(p.exists() for p in (HOLDOUT, HOLDOUT_SHA, MANIFEST, BANK, BANK_RECEIPT)):
            raise RuntimeError("Incomplete immutable output set; inspect it before starting a new generation")
        manifest = json.loads(MANIFEST.read_text())
        if sha256(HOLDOUT) != manifest["holdout"]["sha256"] or sha256(BANK) != manifest["bank"]["sha256"]:
            raise RuntimeError("Existing immutable bank or holdout changed")
        if HOLDOUT_SHA.read_text().split()[0] != manifest["holdout"]["sha256"]:
            raise RuntimeError("Existing holdout checksum file does not match")
        current_authors, current_inventory = training_authors(bank_inventory() + [BANK])
        with HOLDOUT.open() as fh:
            holdout_authors = {json.loads(line)["author_did"] for line in fh if line.strip()}
        if holdout_authors & current_authors:
            raise RuntimeError("Existing holdout now overlaps a local training bank")
        print(json.dumps({"status": "existing_immutable_outputs_verified", "manifest": manifest,
                          "current_training_bank_inventory": current_inventory}, indent=2))
        return
    cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if BANK.exists():
        if not BANK_RECEIPT.exists():
            raise RuntimeError("Existing bank lacks an export receipt; inspect before reusing")
        bank_receipt = json.loads(BANK_RECEIPT.read_text())
        if sha256(BANK) != bank_receipt["sha256"]:
            raise RuntimeError("Existing bank checksum mismatch")
        cutoff = bank_receipt["source_indexed_at_cutoff_utc"]
    else:
        bank_receipt = export_bank(cutoff)
    print(json.dumps({"bank": bank_receipt}), flush=True)
    bank = pq.read_table(BANK)
    assert not any(c.startswith("teacher_") for c in bank.column_names), bank.column_names
    assert all(bucket != 0 for bucket in bank["author_split_bucket"].to_pylist())
    train_authors, inventory = training_authors(bank_inventory() + [BANK])
    rows, sampling = export_holdout(train_authors, cutoff)
    overlap = {r["author_did"] for r in rows} & train_authors
    assert not overlap, f"holdout/train author overlap: {len(overlap)}"
    assert len({r["uri"] for r in rows}) == len(rows)
    assert len({r["author_did"] for r in rows}) == len(rows)
    assert all(r["author_split_bucket"] == 0 for r in rows)
    with HOLDOUT.open("x") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    digest = sha256(HOLDOUT)
    with HOLDOUT_SHA.open("x") as fh:
        fh.write(f"{digest}  {HOLDOUT.name}\n")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "unlabeled_data_export_only",
        "bank": bank_receipt,
        "holdout": {
            "path": str(HOLDOUT), "rows": len(rows), "authors": len(rows), "sha256": digest,
            "bytes": HOLDOUT.stat().st_size, "requested_rows": 600, "shortfall": 600 - len(rows),
            "strata": sampling, "post_types": dict(Counter(r["post_type"] for r in rows)),
            "english_only": True, "author_split_bucket": 0,
            "overlap_with_training_authors": len(overlap),
            "source_indexed_at_cutoff_utc": cutoff,
            "labels": "none; sampling keywords are not labels",
        },
        "excluded_training_bank_inventory": inventory,
        "excluded_training_authors": len(train_authors),
        "limitations": [
            "English holdout is keyword-enriched and not a representative population sample.",
            "No political viewpoint labels, human labels, teacher targets, thresholds or quality metrics exist yet.",
            "Live visibility and deletions were enforced at export; recheck the deletion/visibility ledger before use.",
            "Author-isolation audit covers the listed local lakehouse training banks, not unknown external banks.",
        ],
    }
    write_json_once(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
