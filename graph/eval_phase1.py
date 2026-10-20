#!/usr/bin/env python3
"""Evaluate Phase 1 "Done when" criteria, enrich the top 500 communities with
profile/handle/list signals, and optionally ship KnownFor + communities to netcup.

Checks:
  1. 26 seed subcultures and 8 collision traps separate into distinct communities.
  2. Archive curated-list co-membership recovery (real A Blue Start comparison pending).
  3. KnownFor + communities ship to netcup as a versioned table under 50 MB.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import re
import subprocess
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def js_divergence(p: dict[int, float], q: dict[int, float]) -> float:
    """Normalized Jensen-Shannon divergence in [0, 1] (base-2 log)."""
    keys = set(p) | set(q)
    if not keys:
        return 1.0
    sp = sum(p.values()) or 1.0
    sq = sum(q.values()) or 1.0
    kl_pm = 0.0
    kl_qm = 0.0
    for k in keys:
        pk = p.get(k, 0.0) / sp
        qk = q.get(k, 0.0) / sq
        mk = 0.5 * (pk + qk)
        if pk > 1e-12:
            kl_pm += pk * math.log2(pk / mk)
        if qk > 1e-12:
            kl_qm += qk * math.log2(qk / mk)
    return float(min(1.0, max(0.0, 0.5 * (kl_pm + kl_qm))))


def cosine_dict(p: dict[int, float], q: dict[int, float]) -> float:
    if not p or not q:
        return 0.0
    dot = sum(v * q.get(k, 0.0) for k, v in p.items())
    np_ = math.sqrt(sum(v * v for v in p.values()))
    nq_ = math.sqrt(sum(v * v for v in q.values()))
    return dot / max(np_ * nq_, 1e-12)


def load_known_for(graph_dir: pathlib.Path) -> tuple[dict[str, list[tuple[int, int, int, float]]], pa.Table, pa.Table]:
    kf = pq.read_table(graph_dir / "known_for.parquet").sort_by([("did", "ascending"), ("rank", "ascending")])
    comm_tbl = pq.read_table(graph_dir / "communities.parquet")
    by_did: dict[str, list[tuple[int, int, int, float]]] = {}
    dids = kf["did"].to_pylist()
    comms = kf["community"].to_pylist()
    scenes = kf["scene_id"].to_pylist() if "scene_id" in kf.column_names else [0] * len(dids)
    domains = kf["domain_id"].to_pylist() if "domain_id" in kf.column_names else [0] * len(dids)
    scores = kf["score"].to_pylist()
    for d, c, s, dom, sc in zip(dids, comms, scenes, domains, scores):
        by_did.setdefault(d, []).append((int(c), int(s), int(dom), float(sc)))
    return by_did, kf, comm_tbl


def cohort_distribution(dids: list[str], by_did: dict[str, list[tuple[int, int, int, float]]]) -> tuple[dict[int, float], dict[int, float], int]:
    comm_weights: dict[int, float] = collections.Counter()
    scene_weights: dict[int, float] = collections.Counter()
    matched = 0
    for d in dids:
        entries = by_did.get(d)
        if not entries:
            continue
        matched += 1
        for c, s, _dom, w in entries:
            comm_weights[c] += w
            scene_weights[s] += w
    return dict(comm_weights), dict(scene_weights), matched


def eval_subcultures_and_traps(
    manifest: dict[str, Any],
    by_did: dict[str, list[tuple[int, int, int, float]]],
) -> dict[str, Any]:
    subcultures = manifest["subcultures"]
    traps = manifest["collision_traps"]

    sc_results: dict[str, Any] = {}
    sc_comm_dists: dict[str, dict[int, float]] = {}
    sc_scene_dists: dict[str, dict[int, float]] = {}
    primary_comms: set[int] = set()
    primary_scenes: set[int] = set()

    for sc_id, dids in subcultures.items():
        cdist, sdist, matched = cohort_distribution(dids, by_did)
        sc_comm_dists[sc_id] = cdist
        sc_scene_dists[sc_id] = sdist
    # Every cohort gets its actual highest-weight community, independently.
    # Reserving already-used communities for other cohorts inflated separation.
    merged: dict[int, list[str]] = collections.defaultdict(list)
    for sc_id in sorted(subcultures):
        dids = subcultures[sc_id]
        cdist = sc_comm_dists[sc_id]
        sdist = sc_scene_dists[sc_id]
        matched = sum(1 for d in dids if d in by_did)
        top_comms = sorted(
            cdist.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:3]
        top_scenes = sorted(sdist.items(), key=lambda kv: -kv[1])[:3]
        if top_comms:
            primary_comms.add(top_comms[0][0])
            merged[top_comms[0][0]].append(sc_id)
        if top_scenes:
            primary_scenes.add(top_scenes[0][0])
        sc_results[sc_id] = {
            "seed_dids": len(dids),
            "matched_in_known_for": matched,
            "missing_seed_dids": sorted(set(dids) - by_did.keys()),
            "top_communities": [{"community": c, "weight": round(w, 3)} for c, w in top_comms],
            "top_scenes": [{"scene_id": s, "weight": round(w, 3)} for s, w in top_scenes],
        }

    # Pairwise JSD across all 26*(25)/2 = 325 subculture pairs
    sc_ids = sorted(subcultures.keys())
    jsds = []
    for i in range(len(sc_ids)):
        for j in range(i + 1, len(sc_ids)):
            if sc_comm_dists[sc_ids[i]] and sc_comm_dists[sc_ids[j]]:
                jsds.append(js_divergence(sc_comm_dists[sc_ids[i]], sc_comm_dists[sc_ids[j]]))

    trap_results: dict[str, Any] = {}
    traps_separated = 0
    for trap_id, poles in traps.items():
        pole_names = list(poles.keys())
        pole_dists: dict[str, dict[int, float]] = {}
        pole_hier_dists: dict[str, dict[int, float]] = {}
        pole_top_comm: dict[str, int] = {}
        pole_info: dict[str, Any] = {}
        for pname, dids in poles.items():
            cdist, sdist, matched = cohort_distribution(dids, by_did)
            pole_dists[pname] = cdist
            # Hierarchical feature vector: community (0..99999), scene (100000+s), domain (200000+dom)
            hdist: dict[int, float] = collections.Counter()
            for d in dids:
                for c, s, dom, w in by_did.get(d, []):
                    hdist[c] += w
                    hdist[100_000 + s] += 0.35 * w
                    hdist[200_000 + dom] += 0.15 * w
            pole_hier_dists[pname] = dict(hdist)
            top_c = max(cdist.items(), key=lambda kv: kv[1])[0] if cdist else -1
            pole_top_comm[pname] = top_c
            pole_info[pname] = {
                "dids": len(dids),
                "matched": matched,
                "missing_seed_dids": sorted(set(dids) - by_did.keys()),
                "top_community": top_c,
                "top_communities": [
                    {"community": c, "weight": round(w, 3)}
                    for c, w in sorted(cdist.items(), key=lambda kv: -kv[1])[:3]
                ],
            }

        # Check pairwise JSD and disjoint top communities across all poles in the trap
        pair_jsds = []
        pair_cos = []
        for i in range(len(pole_names)):
            for j in range(i + 1, len(pole_names)):
                pa_name, pb_name = pole_names[i], pole_names[j]
                if pole_dists[pa_name] and pole_dists[pb_name]:
                    pair_jsds.append(js_divergence(pole_dists[pa_name], pole_dists[pb_name]))
                    pair_cos.append(cosine_dict(pole_dists[pa_name], pole_dists[pb_name]))

        distinct_top = len(set(pole_top_comm.values())) == len(pole_names) and (-1 not in pole_top_comm.values())
        min_jsd = min(pair_jsds) if pair_jsds else 0.0
        max_cos = max(pair_cos) if pair_cos else 1.0

        # Leave-one-out hierarchical classification accuracy across poles of this trap
        correct = 0
        total_eval = 0
        for pname, dids in poles.items():
            for d in dids:
                entries = by_did.get(d)
                if not entries:
                    continue
                acct_vec: dict[int, float] = collections.Counter()
                for c, s, dom, w in entries:
                    acct_vec[c] += w
                    acct_vec[100_000 + s] += 0.35 * w
                    acct_vec[200_000 + dom] += 0.15 * w
                best_pole = None
                best_sim = -1.0
                for cand_pole, cand_dist in pole_hier_dists.items():
                    loo = dict(cand_dist)
                    if cand_pole == pname:
                        for k, w in acct_vec.items():
                            loo[k] = max(0.0, loo.get(k, 0.0) - w)
                    sim = cosine_dict(acct_vec, loo)
                    if sim > best_sim:
                        best_sim = sim
                        best_pole = cand_pole
                total_eval += 1
                if best_pole == pname:
                    correct += 1

        loo_acc = correct / max(total_eval, 1)
        separated = distinct_top and min_jsd >= 0.60 and loo_acc >= 0.70
        if separated:
            traps_separated += 1

        trap_results[trap_id] = {
            "separated": separated,
            "status": "pass" if separated else "fail",
            "evaluated_accounts": total_eval,
            "correct_loo_accounts": correct,
            "merged_poles": [
                {"community": c, "poles": [p for p, top in pole_top_comm.items() if top == c]}
                for c in sorted(set(pole_top_comm.values()))
                if c >= 0 and sum(top == c for top in pole_top_comm.values()) > 1
            ],
            "distinct_top_communities": distinct_top,
            "min_pairwise_jsd": round(min_jsd, 4),
            "max_pairwise_cosine": round(max_cos, 4),
            "loo_accuracy": round(loo_acc, 4),
            "poles": pole_info,
        }

    return {
        "subcultures_count": len(subcultures),
        "subcultures_with_matches": sum(1 for v in sc_results.values() if v["matched_in_known_for"] > 0),
        "distinct_primary_communities_across_26": len(primary_comms),
        "distinct_primary_scenes_across_26": len(primary_scenes),
        "merged_subcultures": [{"community": c, "subcultures": names} for c, names in sorted(merged.items()) if len(names) > 1],
        "pairwise_jsd_evaluable_pairs": len(jsds),
        "pairwise_jsd_median": round(float(np.median(jsds)), 4) if jsds else 0.0,
        "pairwise_jsd_mean": round(float(np.mean(jsds)), 4) if jsds else 0.0,
        "traps_total": len(traps),
        "traps_separated": traps_separated,
        "collision_traps": trap_results,
        "subcultures": sc_results,
    }


def eval_pack_comembership(
    event_files: list[pathlib.Path],
    by_did: dict[str, list[tuple[int, int, int, float]]],
    n_communities: int,
) -> dict[str, Any]:
    """Measure real archive starter-pack pairs without injecting seed cohorts.

    The former "503 baseline" was a Python hash/rank proxy, not A Blue Start.
    No comparison is reported until the actual published partition is provided.
    Latest revisions are folded for listitems, parent lists and starter packs.
    """
    latest: dict[tuple[str, str, str], tuple[str, bool, str, str]] = {}
    for index, pf in enumerate(event_files):
        tbl = pq.read_table(
            pf, columns=["collection", "did", "rkey", "rev", "operation", "list_uri", "subject"],
            filters=[("collection", "in", ["app.bsky.graph.listitem", "app.bsky.graph.list", "app.bsky.graph.starterpack"])],
        )
        for collection, did, rkey, rev, op, luri, subj in zip(*(tbl[c].to_pylist() for c in tbl.column_names)):
            key = (collection, did, rkey)
            value = (rev or "", op == "delete", luri or "", subj or "")
            if key not in latest or value[:2] > latest[key][:2]:
                latest[key] = value
        if (index + 1) % 100 == 0:
            print(f"pack evaluation: scanned {index + 1}/{len(event_files)} shards", flush=True)
    lists: dict[str, set[str]] = collections.defaultdict(set)
    deleted_lists = {f"at://{did}/{collection}/{rkey}" for (collection, did, rkey), value in latest.items()
                     if collection == "app.bsky.graph.list" and value[1]}
    pack_lists = [value[2] for (collection, _did, _rkey), value in latest.items()
                  if collection == "app.bsky.graph.starterpack" and not value[1] and value[2] not in deleted_lists]
    for (collection, _did, _rkey), (_rev, deleted, luri, subj) in latest.items():
        if collection == "app.bsky.graph.listitem" and not deleted and luri and luri not in deleted_lists and subj.startswith("did:"):
            lists[luri].add(subj)
    del latest
    eval_packs: list[list[str]] = []
    too_small = too_large = 0
    for luri in pack_lists:
        members = lists.get(luri, set())
        present = sorted(members & by_did.keys())
        if 3 <= len(present) <= 200:
            eval_packs.append(present)
        elif len(present) < 3:
            too_small += 1
        else:
            too_large += 1
    total_pairs = top1_hits = top3_hits = scene_hits = 0
    for pack in eval_packs:
        # Bitsets count every pair once, even if it shares multiple memberships.
        # This is exact, unlike summing per-community pair counts for top-3.
        top1_counts = collections.Counter(by_did[d][0][0] for d in pack)
        total_pairs += len(pack) * (len(pack) - 1) // 2
        top1_hits += sum(count * (count - 1) // 2 for count in top1_counts.values())
        community_bits: dict[int, int] = collections.defaultdict(int)
        scene_bits: dict[int, int] = collections.defaultdict(int)
        for i, did in enumerate(pack):
            for community, scene, _domain, _weight in by_did[did]:
                community_bits[community] |= 1 << i
                scene_bits[scene] |= 1 << i
        for i, did in enumerate(pack):
            communities = scenes = 0
            for community, scene, _domain, _weight in by_did[did]:
                communities |= community_bits[community]
                scenes |= scene_bits[scene]
            top3_hits += (communities >> (i + 1)).bit_count()
            scene_hits += (scenes >> (i + 1)).bit_count()
    return {
        "source": "archive starterpack records joined to current listitems; seed cohorts excluded",
        "pair_weighting": "each starter pack contributes all pairs; a pair in multiple packs counts multiple times",
        "starterpacks_evaluated": len(eval_packs), "starterpacks_too_small": too_small, "starterpacks_too_large": too_large,
        "live_starterpack_records": len(pack_lists), "live_curated_lists": len(lists),
        "co_member_pairs": total_pairs, "communities_count": n_communities,
        "known_for_top1_pair_hits": top1_hits, "known_for_top3_pair_hits": top3_hits,
        "known_for_scene_pair_hits": scene_hits,
        "known_for_top1_pair_recall": top1_hits / total_pairs if total_pairs else None,
        "known_for_top3_pair_recall": top3_hits / total_pairs if total_pairs else None,
        "known_for_scene_pair_recall": scene_hits / total_pairs if total_pairs else None,
        "a_blue_start_503_partition_available": False,
        "beats_a_blue_start_503_partition": None,
        "baseline_status": "not measured: previous hash/rank stand-in removed; real partition unavailable",
    }


STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "not", "but", "you",
    "your", "all", "can", "her", "his", "they", "them", "she", "him", "who", "what", "when",
    "where", "why", "how", "about", "into", "over", "after", "more", "other", "some", "such",
    "no", "nor", "too", "very", "just", "than", "then", "now", "here", "there", "out", "our",
    "bsky", "social", "com", "org", "net", "http", "https", "www", "views", "own", "mine",
    "account", "official", "profile", "follow", "posts", "posting", "bluesky", "twitter",
}


def enrich_and_label_communities(
    event_files: list[pathlib.Path],
    graph_dir: pathlib.Path,
    by_did: dict[str, list[tuple[int, int, int, float]]],
    comm_tbl: pa.Table,
) -> pa.Table:
    """Create heuristic profile-term labels and prompts for 500 communities.

    These labels are not LLM- or human-reviewed. Seed identities are never used.
    The archive projection does not provide profile handles or starter-pack titles
    here, so no claim of either enrichment is made.
    """
    profiles: dict[str, tuple[str, str, str]] = {}  # did -> (name, description, handle)
    latest_profile: dict[str, tuple[str, bool]] = {}
    for pf in event_files:
        tbl = pq.read_table(
            pf,
            columns=["collection", "operation", "did", "rev", "name", "description", "extra"],
            filters=[("collection", "==", "app.bsky.actor.profile")],
        )
        if tbl.num_rows == 0:
            continue
        for op, d, rev, name, desc, extra in zip(
            tbl["operation"].to_pylist(),
            tbl["did"].to_pylist(),
            tbl["rev"].to_pylist(),
            tbl["name"].to_pylist(),
            tbl["description"].to_pylist(),
            tbl["extra"].to_pylist(),
        ):
            if d not in by_did:
                continue
            version = (rev or "", op == "delete")
            if d in latest_profile and version <= latest_profile[d]:
                continue
            latest_profile[d] = version
            if op == "delete":
                profiles.pop(d, None)
            else:
                profiles[d] = (name or "", desc or "", extra or "")

    # Group members by primary community
    comm_members: dict[int, list[tuple[str, float]]] = collections.defaultdict(list)
    for d, entries in by_did.items():
        if entries:
            c0, _s0, _dom0, w0 = entries[0]
            comm_members[c0].append((d, w0))

    # Global term frequency across profiles for TF-IDF
    global_df: dict[str, int] = collections.Counter()
    comm_tf: dict[int, collections.Counter] = {}
    comm_tlds: dict[int, collections.Counter] = {}

    for c, mlist in comm_members.items():
        tf: collections.Counter = collections.Counter()
        tlds: collections.Counter = collections.Counter()
        for d, _w in mlist:
            name, desc, handle = profiles.get(d, ("", "", ""))
            if handle and "." in handle and not handle.endswith(".bsky.social"):
                tld = "." + handle.rsplit(".", 1)[-1].lower()
                tlds[tld] += 1
            words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_+#-]{2,20}", (name + " " + desc).lower()))
            for w in words:
                if w not in STOPWORDS and not w.isdigit():
                    tf[w] += 1
                    global_df[w] += 1
        comm_tf[c] = tf
        comm_tlds[c] = tlds

    total_profiles = max(sum(len(v) for v in comm_members.values()), 1)
    c_ids = comm_tbl["community"].to_pylist()
    s_ids = comm_tbl["scene_id"].to_pylist()
    d_ids = comm_tbl["domain_id"].to_pylist()
    sizes = comm_tbl["size"].to_pylist()
    top_dids_col = comm_tbl["top_dids"].to_pylist()
    top_foll_col = comm_tbl["top_followers"].to_pylist()

    labels = []
    keywords_col = []
    seed_match_col = []
    prompt_rows = []

    for c, s_id, d_id, sz, tdids, tfoll in zip(c_ids, s_ids, d_ids, sizes, top_dids_col, top_foll_col):
        tf = comm_tf.get(c, collections.Counter())
        scored_terms = []
        for term, cnt in tf.items():
            if cnt < 2:
                continue
            idf = math.log(1.0 + total_profiles / max(global_df[term], 1))
            scored_terms.append((cnt * idf, term))
        scored_terms.sort(reverse=True)
        top_terms = [t for _, t in scored_terms[:8]]
        if top_terms:
            human_label = " / ".join(t.title() for t in top_terms[:3])
        else:
            human_label = f"Community #{c} (Scene {s_id})"

        labels.append(human_label)
        keywords_col.append(top_terms)
        # Empty legacy schema field; evaluation seeds are not labeling features.
        seed_match_col.append("")

        if len(prompt_rows) < 500:
            sample_Wait = []
            for d in tdids[:8]:
                nm, ds, hdl = profiles.get(d, ("", "", ""))
                if nm or ds or hdl:
                    sample_Wait.append({"did": d, "handle": hdl, "name": nm, "bio": ds[:160]})
            prompt_rows.append(
                {
                    "community": int(c),
                    "scene_id": int(s_id),
                    "domain_id": int(d_id),
                    "size": int(sz),
                    "heuristic_label": human_label,
                    "top_terms": top_terms,
                    "label_source": "unreviewed archive profile term heuristic",
                    "enriched_tlds": [k for k, _ in comm_tlds.get(c, collections.Counter()).most_common(3)],
                    "top_producers": sample_Wait,
                }
            )

    enriched = pa.table(
        {
            "community": pa.array(c_ids, type=pa.uint32()),
            "scene_id": pa.array(s_ids, type=pa.uint16()),
            "domain_id": pa.array(d_ids, type=pa.uint8()),
            "size": pa.array(sizes, type=pa.uint32()),
            "label": pa.array(labels, type=pa.string()),
            "seed_subculture": pa.array(seed_match_col, type=pa.string()),
            "keywords": pa.array(keywords_col, type=pa.list_(pa.string())),
            "top_dids": pa.array(top_dids_col, type=pa.list_(pa.string())),
            "top_followers": pa.array(top_foll_col, type=pa.list_(pa.uint64())),
        }
    )
    pq.write_table(enriched, graph_dir / "communities_labeled.parquet", compression="zstd")
    prompts_path = graph_dir / "colab_community_labeling_prompts.jsonl"
    with prompts_path.open("w") as fh:
        for r in prompt_rows:
            fh.write(json.dumps(r) + "\n")
    return enriched


def ship_to_netcup(graph_dir: pathlib.Path, version: str) -> dict[str, Any]:
    kf_path = graph_dir / "known_for.parquet"
    comm_path = graph_dir / "communities_labeled.parquet"
    subprocess.run(["scp", str(kf_path), str(comm_path), "netcup:/tmp/"], check=True)
    ddl = f"""
    CREATE TABLE IF NOT EXISTS catbird.known_for
    (
        version   LowCardinality(String) DEFAULT '{version}',
        pid       UInt32,
        did       String CODEC(ZSTD(3)),
        community UInt32,
        scene_id  UInt16,
        domain_id UInt8,
        score     Float32,
        rank      UInt8
    )
    ENGINE = ReplacingMergeTree
    ORDER BY (version, did, rank);

    CREATE TABLE IF NOT EXISTS catbird.communities
    (
        version         LowCardinality(String) DEFAULT '{version}',
        community       UInt32,
        scene_id        UInt16,
        domain_id       UInt8,
        size            UInt32,
        label           String,
        seed_subculture LowCardinality(String),
        keywords        Array(String),
        top_dids        Array(String),
        top_followers   Array(UInt64)
    )
    ENGINE = ReplacingMergeTree
    ORDER BY (version, community);

    TRUNCATE TABLE catbird.known_for;
    TRUNCATE TABLE catbird.communities;
    """
    cmd = (
        f"clickhouse-client --multiquery <<'EOSQL'\n{ddl}\nEOSQL\n"
        f"clickhouse-client -q \"INSERT INTO catbird.known_for (pid, did, community, scene_id, domain_id, score, rank) FORMAT Parquet\" < /tmp/known_for.parquet\n"
        f"clickhouse-client -q \"INSERT INTO catbird.communities (community, scene_id, domain_id, size, label, seed_subculture, keywords, top_dids, top_followers) FORMAT Parquet\" < /tmp/communities_labeled.parquet\n"
        f"clickhouse-client -q \"OPTIMIZE TABLE catbird.known_for FINAL\"\n"
        f"clickhouse-client -q \"OPTIMIZE TABLE catbird.communities FINAL\"\n"
    )
    subprocess.run(["ssh", "netcup", cmd], check=True)
    out = subprocess.run(
        [
            "ssh",
            "netcup",
            "clickhouse-client -q \"SELECT table, sum(rows) AS rows, round(sum(bytes_on_disk)/1e6, 3) AS mb FROM system.parts WHERE active AND database='catbird' AND table IN ('known_for', 'communities') GROUP BY table FORMAT JSONEachRow\"",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    tables = [json.loads(line) for line in out.strip().splitlines() if line.strip()]
    total_mb = round(sum(t["mb"] for t in tables), 3)
    return {"version": version, "tables": tables, "total_mb": total_mb, "under_50mb": total_mb < 50.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", action="append", required=True, help="Archive-only directory; repeat for each archive")
    ap.add_argument("--graph-dir", default="/Users/joshlacalamito/catbird-lake-local/graph-build/out")
    ap.add_argument("--manifest", default="/Users/joshlacalamito/catbird-lake-local/graph-build/seed_and_traps_manifest.json")
    ap.add_argument("--ship-netcup", action="store_true")
    ap.add_argument("--version", default="v20261019")
    args = ap.parse_args()

    event_files = sorted({p.resolve() for directory in args.events_dir
                          for p in pathlib.Path(directory).glob("events-*.parquet")})
    if not event_files:
        raise SystemExit("No archive events-*.parquet files found")
    if any("supplemental" in str(p).lower() or "quarantine" in str(p).lower() for p in event_files):
        raise SystemExit("Supplemental/quarantined input is forbidden")
    graph_dir = pathlib.Path(args.graph_dir)
    manifest = json.loads(pathlib.Path(args.manifest).read_text())

    by_did, kf, comm_tbl = load_known_for(graph_dir)
    sep = eval_subcultures_and_traps(manifest, by_did)
    inventory = []
    for index, path in enumerate(event_files):
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise SystemExit(f"Archive file changed while hashing; retry after it seals: {path}")
        inventory.append({"path": str(path), "size": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest})
        if (index + 1) % 100 == 0:
            print(f"evaluation input hashes: {index + 1}/{len(event_files)}", flush=True)
    (graph_dir / "evaluation_input_manifest.json").write_text(json.dumps({
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "exact evaluation file list; does not retroactively freeze graph fold input",
        "files": inventory,
    }, indent=2))
    producer_dids = {line.split("\t")[1] for line in (graph_dir / "producers.tsv").read_text().splitlines()}
    for name, dids in manifest["subcultures"].items():
        sep["subcultures"][name]["matched_as_producers"] = len(set(dids) & producer_dids)
    for trap, poles in manifest["collision_traps"].items():
        for name, dids in poles.items():
            sep["collision_traps"][trap]["poles"][name]["matched_as_producers"] = len(set(dids) & producer_dids)
    pack_eval = eval_pack_comembership(event_files, by_did, comm_tbl.num_rows)
    enrich_and_label_communities(event_files, graph_dir, by_did, comm_tbl)
    for item in inventory:
        current = pathlib.Path(item["path"]).stat()
        if current.st_size != item["size"] or current.st_mtime_ns != item["mtime_ns"]:
            raise SystemExit(f"Archive input changed during evaluation: {item['path']}")

    ship_res = None
    if args.ship_netcup:
        ship_res = ship_to_netcup(graph_dir, args.version)

    report = {
        "graph_version": "phase1-v1",
        "known_for_sha256": sha256_file(graph_dir / "known_for.parquet"),
        "evaluation_input_manifest_sha256": sha256_file(graph_dir / "evaluation_input_manifest.json"),
        "seed_manifest_use": "evaluation only; never clustering or labeling",
        "cluster_stats": json.loads((graph_dir / "cluster_stats.json").read_text()),
        "separation_summary": {
            k: v for k, v in sep.items() if k not in ("subcultures", "collision_traps")
        },
        "collision_traps": sep["collision_traps"],
        "pack_comembership": pack_eval,
        "netcup_ship": ship_res,
    }
    (graph_dir / "phase1_eval_report.json").write_text(json.dumps({**report, "subcultures": sep["subcultures"]}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
