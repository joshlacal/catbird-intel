#!/usr/bin/env python3
"""Export real post triplets against local Phase 1 v1 KnownFor.

Read-only on netcup: no production table is replaced. Positives have different
authors in the same graph community; negatives have disjoint KnownFor memberships,
preferring the same scene. These are weak graph labels, not observed co-engagement
or validated semantic relevance. No seed/trap identities enter pair construction.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import random
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_triplets(posts: list[dict], memberships: dict[str, list[dict]], limit: int, seed: int) -> tuple[list[dict], dict]:
    eligible = [p for p in posts if p["author_did"] in memberships]
    groups: dict[int, list[dict]] = collections.defaultdict(list)
    scene_groups: dict[int, list[dict]] = collections.defaultdict(list)
    for post in eligible:
        primary = memberships[post["author_did"]][0]
        groups[primary["community"]].append(post)
        scene_groups[primary["scene_id"]].append(post)
    membership_sets = {did: {r["community"] for r in rows} for did, rows in memberships.items()}
    rng = random.Random(seed)
    rng.shuffle(eligible)
    pairs: list[dict] = []
    skipped_no_positive = skipped_no_negative = same_scene = 0
    for anchor in eligible:
        if len(pairs) >= limit:
            break
        author = anchor["author_did"]
        primary = memberships[author][0]
        community = primary["community"]
        positive_pool = groups[community]
        offset = rng.randrange(len(positive_pool))
        positive = next((positive_pool[(offset + i) % len(positive_pool)]
                         for i in range(min(len(positive_pool), 100))
                         if positive_pool[(offset + i) % len(positive_pool)]["author_did"] != author), None)
        if positive is None:
            skipped_no_positive += 1
            continue
        blocked = membership_sets[author] | membership_sets[positive["author_did"]]
        negative = None
        negative_kind = None
        for kind, pool in [("same_scene_disjoint_memberships", scene_groups[primary["scene_id"]]),
                           ("global_disjoint_memberships", eligible)]:
            for _ in range(min(len(pool), 100)):
                candidate = pool[rng.randrange(len(pool))]
                if not (membership_sets[candidate["author_did"]] & blocked):
                    negative, negative_kind = candidate, kind
                    break
            if negative is not None:
                break
        if negative is None:
            skipped_no_negative += 1
            continue
        row = {
            "community_id": community,
            "negative_community_id": memberships[negative["author_did"]][0]["community"],
            "community_weight": primary["score"],
            "scene_id": primary["scene_id"],
            "graph_version": "phase1-v1",
            "positive_kind": "same_community_different_author",
            "negative_kind": negative_kind,
        }
        for prefix, post in [("anchor", anchor), ("positive", positive), ("negative", negative)]:
            for field in ["uri", "text", "author_did", "author_split_bucket"]:
                row[f"{prefix}_{field}"] = post[field]
        pairs.append(row)
        same_scene += negative_kind == "same_scene_disjoint_memberships"
    return pairs, {
        "eligible_posts": len(eligible), "represented_communities": len(groups),
        "skipped_no_positive": skipped_no_positive, "skipped_no_negative": skipped_no_negative,
        "same_scene_negatives": same_scene,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    lake = pathlib.Path.home() / "catbird-lake-local"
    parser.add_argument("--graph-dir", type=pathlib.Path, default=lake / "graph-build/out")
    parser.add_argument("--output", type=pathlib.Path, default=lake / "drive-sync-staging/training-banks/phase5_social_encoder_pairs_v1.parquet")
    parser.add_argument("--holdout", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1] / "eval/phase3_holdout_600_unlabeled.jsonl")
    parser.add_argument("--source-posts", type=pathlib.Path, help="Reuse an existing export with its .manifest.json")
    parser.add_argument("--limit", type=int, default=500000)
    parser.add_argument("--source-limit", type=int, default=3000000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit <= 0 or args.source_limit <= 0:
        raise SystemExit("Limits must be positive")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise SystemExit("Output already exists; archive it explicitly before rerunning")
    provenance_path = args.graph_dir / "input_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("graph_version") != "phase1-v1" or provenance.get("supplemental_edges_excluded") is not True:
        raise SystemExit("Require an archive-only v1 input provenance receipt")
    known_for = args.graph_dir / "known_for.parquet"
    known_for_hash = sha256(known_for)
    eval_path = args.graph_dir / "phase1_eval_report.json"
    evaluation = json.loads(eval_path.read_text())
    if evaluation.get("graph_version") != "phase1-v1" or evaluation.get("known_for_sha256") != known_for_hash:
        raise SystemExit("Phase 1 evaluation does not match this v1 KnownFor")
    kf = pq.read_table(known_for).sort_by([("did", "ascending"), ("rank", "ascending")])
    memberships: dict[str, list[dict]] = collections.defaultdict(list)
    for row in kf.to_pylist():
        memberships[row["did"]].append(row)
    holdout_hash = sha256(args.holdout)
    with args.holdout.open() as fh:
        holdout_authors = {json.loads(line)["author_did"] for line in fh if line.strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = args.source_posts or args.output.with_name("phase5_live_source_posts_v1.parquet")
    source_manifest = source.with_suffix(".manifest.json")
    if args.source_posts:
        source_receipt = json.loads(source_manifest.read_text())
        if (source_receipt.get("holdout_bucket_excluded") != 0
                or source_receipt.get("deletion_ledger_enforced") is not True
                or source_receipt["sha256"] != sha256(source)):
            raise SystemExit("Source receipt missing bucket/deletion exclusion or SHA mismatch")
    else:
        if source.exists():
            if source_manifest.exists():
                raise SystemExit("Source export already exists; use --source-posts to verify/reuse it")
            archive = source.parent / "_quarantine_unsealed"
            archive.mkdir(exist_ok=True)
            suffix = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            source.rename(archive / f"{source.stem}-{suffix}{source.suffix}")
        query = f"""
        SELECT concat('at://', p.did, '/app.bsky.feed.post/', p.rkey) AS uri,
               p.did AS author_did, cityHash64(p.did) % 20 AS author_split_bucket,
               p.created_at AS created_at, p.text AS text, p.langs AS langs
        FROM catbird.v_posts_live AS p
        WHERE length(p.text) BETWEEN 25 AND 2400
          AND cityHash64(p.did) % 20 != 0
          AND p.did NOT IN (SELECT did FROM catbird.v_hidden_from_algorithms)
          AND (p.did, p.rkey) NOT IN (
              SELECT did, rkey FROM catbird.deletions
              WHERE collection = 'app.bsky.feed.post'
          )
        ORDER BY cityHash64(p.did, p.rkey)
        LIMIT {args.source_limit}
        FORMAT Parquet
        SETTINGS output_format_parquet_compression_method = 'zstd',
                 max_threads = 2, priority = 10, max_execution_time = 600,
                 max_memory_usage = 2000000000, max_bytes_before_external_sort = 250000000
        """
        with source.open("xb") as fh:
            subprocess.run(["ssh", "netcup", "clickhouse-client"], input=query.encode(), stdout=fh, check=True)
        source_receipt = {
            "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source": "catbird.v_posts_live", "query": query, "read_only": True,
            "holdout_bucket_excluded": 0, "sha256": sha256(source),
            "deletion_ledger_enforced": True,
            "rows": pq.read_metadata(source).num_rows,
        }
        source_manifest.write_text(json.dumps(source_receipt, indent=2) + "\n")
    posts = pq.read_table(source).to_pylist()
    if len({p["uri"] for p in posts}) != len(posts):
        raise SystemExit("Source URI duplicates found; refusing ambiguous triplets")
    if any(type(p["author_split_bucket"]) is not int or not 1 <= p["author_split_bucket"] <= 19 for p in posts):
        raise SystemExit("Source author split buckets must be integers 1..19")
    for post in posts:
        author = post["author_did"]
        prefix = f"at://{author}/app.bsky.feed.post/"
        if not author.startswith("did:") or not post["uri"].startswith(prefix) or post["uri"] == prefix:
            raise SystemExit("Source post URI does not identify its author DID and record key")
    overlap = {p["author_did"] for p in posts} & holdout_authors
    if overlap:
        raise SystemExit(f"Source contains {len(overlap)} holdout authors; refusing export")
    rows, stats = build_triplets(posts, memberships, args.limit, args.seed)
    if not rows:
        raise SystemExit("No eligible real triplets; no bank written")
    pq.write_table(pa.Table.from_pylist(rows), args.output, compression="zstd")
    output_hash = sha256(args.output)
    receipt = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "graph_version": "phase1-v1", "known_for_sha256": known_for_hash,
        "graph_input_provenance_sha256": sha256(provenance_path),
        "phase1_eval_report_sha256": sha256(eval_path),
        "phase1_traps_separated": evaluation["separation_summary"]["traps_separated"],
        "phase1_traps_total": evaluation["separation_summary"]["traps_total"],
        "a_blue_start_comparison_available": evaluation["pack_comembership"]["a_blue_start_503_partition_available"],
        "supplemental_edges_excluded": True,
        "sha256": output_hash, "holdout_sha256": holdout_hash,
        "holdout_author_overlap": 0, "holdout_bucket_excluded": 0,
        "source_author_split_buckets": dict(sorted(collections.Counter(p["author_split_bucket"] for p in posts).items())),
        "source_rows": len(posts), "source_unique_uris": len(posts), "deletion_ledger_enforced": True,
        "rows": len(rows), "requested_rows": args.limit, "shortfall": max(0, args.limit - len(rows)),
        "source_posts_sha256": source_receipt["sha256"], "source_posts": str(source),
        "seed": args.seed, "supervision": "weak graph membership, not verified co-engagement",
        "seed_manifest_used": False, "human_quality_validated": False,
        "visibility_limitation": "Visibility and deletes checked at export time; recheck before future model use",
        **stats,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
