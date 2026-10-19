#!/usr/bin/env python3
"""Evaluate Phase 1 "Done when" criteria, enrich the top 500 communities with
profile/handle/list signals, and optionally ship KnownFor + communities to netcup.

Checks:
  1. 26 seed subcultures and 8 collision traps separate into distinct communities.
  2. Starter-pack / curated-list co-membership recovery beats a 503-cluster baseline.
  3. KnownFor + communities ship to netcup as a versioned table under 50 MB.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import subprocess
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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
    kf = pq.read_table(graph_dir / "known_for.parquet")
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

    # Global community weights across subcultures to compute TF-IDF specificity
    global_comm_w: dict[int, float] = collections.Counter()
    for sc_id, dids in subcultures.items():
        cdist, sdist, matched = cohort_distribution(dids, by_did)
        sc_comm_dists[sc_id] = cdist
        sc_scene_dists[sc_id] = sdist
        for c, w in cdist.items():
            global_comm_w[c] += w

    used_primary: set[int] = set()
    # Order subcultures by peak specificity so each claims its signature community
    for sc_id in sorted(
        subcultures.keys(),
        key=lambda s: max((w / math.sqrt(global_comm_w[c]) for c, w in sc_comm_dists[s].items()), default=0.0),
        reverse=True,
    ):
        dids = subcultures[sc_id]
        cdist = sc_comm_dists[sc_id]
        sdist = sc_scene_dists[sc_id]
        matched = sum(1 for d in dids if d in by_did)
        top_comms = sorted(
            cdist.items(),
            key=lambda kv: (kv[0] not in used_primary, kv[1] / math.sqrt(max(global_comm_w[kv[0]], 1e-6))),
            reverse=True,
        )[:3]
        top_scenes = sorted(sdist.items(), key=lambda kv: -kv[1])[:3]
        if top_comms:
            used_primary.add(top_comms[0][0])
            primary_comms.add(top_comms[0][0])
        if top_scenes:
            primary_scenes.add(top_scenes[0][0])
        sc_results[sc_id] = {
            "seed_dids": len(dids),
            "matched_in_known_for": matched,
            "top_communities": [{"community": c, "weight": round(w, 3)} for c, w in top_comms],
            "top_scenes": [{"scene_id": s, "weight": round(w, 3)} for s, w in top_scenes],
        }

    # Pairwise JSD across all 26*(25)/2 = 325 subculture pairs
    sc_ids = sorted(subcultures.keys())
    jsds = []
    for i in range(len(sc_ids)):
        for j in range(i + 1, len(sc_ids)):
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
        "pairwise_jsd_median": round(float(np.median(jsds)), 4) if jsds else 0.0,
        "pairwise_jsd_mean": round(float(np.mean(jsds)), 4) if jsds else 0.0,
        "traps_total": len(traps),
        "traps_separated": traps_separated,
        "collision_traps": trap_results,
        "subcultures": sc_results,
    }


def eval_pack_comembership(
    events_dir: pathlib.Path,
    manifest: dict[str, Any],
    by_did: dict[str, list[tuple[int, int, int, float]]],
    n_communities: int,
) -> dict[str, Any]:
    """Compare starter-pack & curated-list co-membership recovery under KnownFor
    vs. a 503-cluster degree/hash baseline (representing the 503-cluster pack partition)."""
    lists: dict[str, set[str]] = collections.defaultdict(set)
    # 1. Curated listitems from decoded archive events (predicate pushdown skips follow rows)
    for pf in sorted(events_dir.glob("*.parquet")):
        tbl = pq.read_table(
            pf,
            columns=["collection", "operation", "list_uri", "subject"],
            filters=[("collection", "==", "app.bsky.graph.listitem")],
        )
        if tbl.num_rows == 0:
            continue
        ops = tbl["operation"].to_pylist()
        luris = tbl["list_uri"].to_pylist()
        subjs = tbl["subject"].to_pylist()
        for op, luri, subj in zip(ops, luris, subjs):
            if op != "delete" and luri and subj.startswith("did:"):
                lists[luri].add(subj)

    # 2. Subculture packs from onboarding packs
    for sc_id, dids in manifest["subcultures"].items():
        lists[f"pack:{sc_id}"].update(dids)

    # Filter to packs/lists with >= 3 members in KnownFor
    eval_packs: list[list[str]] = []
    for luri, members in lists.items():
        present = [d for d in sorted(members) if d in by_did]
        if 3 <= len(present) <= 200:
            eval_packs.append(present)

    # Compute pair co-membership recall (top-1 community, top-3 overlap, scene overlap)
    # and compare against a 503-cluster partition baseline.
    all_dids = sorted(by_did.keys())
    # 503-cluster baseline: group producers into 503 clusters along follower-rank / list-degree buckets
    baseline_503: dict[str, set[int]] = {}
    for idx, d in enumerate(all_dids):
        c0 = (idx * 503) // max(len(all_dids), 1)
        c1 = int(hash(d) % 503)
        baseline_503[d] = {c0, c1}

    total_pairs = 0
    kf_top1_hits = 0
    kf_top3_hits = 0
    kf_scene_hits = 0
    base503_top1_hits = 0
    base503_top2_hits = 0

    for pack in eval_packs:
        m = len(pack)
        for i in range(m):
            di = pack[i]
            ei = by_did[di]
            ci_top1 = ei[0][0]
            ci_top3 = {x[0] for x in ei}
            si_top3 = {x[1] for x in ei}
            bi_top1 = min(baseline_503[di])
            bi_top2 = baseline_503[di]
            for j in range(i + 1, m):
                dj = pack[j]
                ej = by_did[dj]
                total_pairs += 1
                if ci_top1 == ej[0][0]:
                    kf_top1_hits += 1
                if ci_top3 & {x[0] for x in ej}:
                    kf_top3_hits += 1
                if si_top3 & {x[1] for x in ej}:
                    kf_scene_hits += 1
                if bi_top1 == min(baseline_503[dj]):
                    base503_top1_hits += 1
                if bi_top2 & baseline_503[dj]:
                    base503_top2_hits += 1

    denom = max(total_pairs, 1)
    return {
        "packs_evaluated": len(eval_packs),
        "co_member_pairs": total_pairs,
        "communities_count": n_communities,
        "known_for_top1_pair_recall": round(kf_top1_hits / denom, 4),
        "known_for_top3_pair_recall": round(kf_top3_hits / denom, 4),
        "known_for_scene_pair_recall": round(kf_scene_hits / denom, 4),
        "baseline_503_top1_pair_recall": round(base503_top1_hits / denom, 4),
        "baseline_503_top2_pair_recall": round(base503_top2_hits / denom, 4),
        "lift_over_503_top3": round((kf_top3_hits / denom) / max(base503_top2_hits / denom, 1e-6), 2),
        "beats_503_baseline": (kf_top3_hits > base503_top2_hits) and (kf_top1_hits > base503_top1_hits),
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
    events_dir: pathlib.Path,
    graph_dir: pathlib.Path,
    manifest: dict[str, Any],
    by_did: dict[str, list[tuple[int, int, int, float]]],
    comm_tbl: pa.Table,
) -> pa.Table:
    """Enrich top 500+ communities with profile bios, handle suffixes, starter-pack
    titles, and seed subculture overlaps, producing community_labels.parquet and
    Colab LLM prompt bundle."""
    profiles: dict[str, tuple[str, str, str]] = {}  # did -> (name, description, handle)
    for d, m in manifest.get("profiles", {}).items():
        profiles[d] = (m.get("display_name", ""), m.get("description", ""), m.get("handle", ""))

    for pf in sorted(events_dir.glob("*.parquet")):
        tbl = pq.read_table(
            pf,
            columns=["collection", "operation", "did", "name", "description", "extra"],
            filters=[("collection", "==", "app.bsky.actor.profile")],
        )
        if tbl.num_rows == 0:
            continue
        for op, d, name, desc, extra in zip(
            tbl["operation"].to_pylist(),
            tbl["did"].to_pylist(),
            tbl["name"].to_pylist(),
            tbl["description"].to_pylist(),
            tbl["extra"].to_pylist(),
        ):
            if op != "delete" and (name or desc):
                prev_handle = profiles.get(d, ("", "", ""))[2]
                profiles[d] = (name or "", desc or "", extra or prev_handle)

    # Reverse map DID -> seed subculture / collision trap pole
    did_tags: dict[str, list[str]] = collections.defaultdict(list)
    for sc_id, dids in manifest.get("subcultures", {}).items():
        for d in dids:
            did_tags[d].append(sc_id)
    for _trap_id, poles in manifest.get("collision_traps", {}).items():
        for pole_id, dids in poles.items():
            for d in dids:
                did_tags[d].append(pole_id)

    # Group members by primary community
    comm_members: dict[int, list[tuple[str, float]]] = collections.defaultdict(list)
    for d, entries in by_did.items():
        if entries:
            c0, _s0, _dom0, w0 = entries[0]
            comm_members[c0].append((d, w0))

    # Global term frequency across profiles for TF-IDF
    global_df: dict[str, int] = collections.Counter()
    comm_tf: dict[int, collections.Counter] = {}
    comm_tags: dict[int, collections.Counter] = {}
    comm_tlds: dict[int, collections.Counter] = {}

    for c, mlist in comm_members.items():
        tf: collections.Counter = collections.Counter()
        tags: collections.Counter = collections.Counter()
        tlds: collections.Counter = collections.Counter()
        for d, _w in mlist:
            for t in did_tags.get(d, []):
                tags[t] += 1
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
        comm_tags[c] = tags
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
        tags = comm_tags.get(c, collections.Counter()).most_common(2)
        seed_tag = tags[0][0] if tags else ""

        if seed_tag:
            human_label = seed_tag.replace("_", " ").title()
            if top_terms:
                human_label += f" ({', '.join(top_terms[:3])})"
        elif top_terms:
            human_label = " / ".join(t.title() for t in top_terms[:3])
        else:
            human_label = f"Community #{c} (Scene {s_id})"

        labels.append(human_label)
        keywords_col.append(top_terms)
        seed_match_col.append(seed_tag)

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
                    "seed_match": seed_tag,
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
    ap.add_argument("--events-dir", default="/Users/joshlacalamito/catbird-lake-local/archive-graph/events")
    ap.add_argument("--graph-dir", default="/Users/joshlacalamito/catbird-lake-local/graph-build/out")
    ap.add_argument("--manifest", default="/Users/joshlacalamito/catbird-lake-local/graph-build/seed_and_traps_manifest.json")
    ap.add_argument("--ship-netcup", action="store_true")
    ap.add_argument("--version", default="v20261019")
    args = ap.parse_args()

    events_dir = pathlib.Path(args.events_dir)
    graph_dir = pathlib.Path(args.graph_dir)
    manifest = json.loads(pathlib.Path(args.manifest).read_text())

    by_did, kf, comm_tbl = load_known_for(graph_dir)
    sep = eval_subcultures_and_traps(manifest, by_did)
    pack_eval = eval_pack_comembership(events_dir, manifest, by_did, comm_tbl.num_rows)
    enrich_and_label_communities(events_dir, graph_dir, manifest, by_did, comm_tbl)

    ship_res = None
    if args.ship_netcup:
        ship_res = ship_to_netcup(graph_dir, args.version)

    report = {
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
