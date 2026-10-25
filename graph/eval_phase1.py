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
import importlib.metadata
import json
import math
import pathlib
import platform
import re
import subprocess
import sys
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


def pair_count(n: int) -> int:
    return n * (n - 1) // 2


def pack_pair_hits(pack: list[str], by_did: dict[str, list[tuple[int, int, int, float]]]) -> dict[str, int]:
    """Count each represented pair once per view, including overlapping views."""
    primary = collections.Counter(by_did[d][0][0] for d in pack)
    hits = {"top1": sum(pair_count(n) for n in primary.values()), "top3": 0, "scene": 0}
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
        hits["top3"] += (communities >> (i + 1)).bit_count()
        hits["scene"] += (scenes >> (i + 1)).bit_count()
    return hits


def primary_partition_null(
    by_did: dict[str, list[tuple[int, int, int, float]]], excluded: set[str] | None = None,
) -> dict[str, Any]:
    """Uniformly permute the actual primary labels; preserve all cluster sizes."""
    excluded = excluded or set()
    counts = collections.Counter(entries[0][0] for did, entries in by_did.items() if entries and did not in excluded)
    n = sum(counts.values())
    same_pairs = sum(pair_count(size) for size in counts.values())
    return {
        "status": "computed" if n >= 2 else "unavailable",
        "model": "uniform permutation of rank-first community labels over the represented DID universe",
        "universe": "all nonempty KnownFor accounts, excluding removed hubs in the dehubbed view",
        "accounts": n, "communities": len(counts),
        "within_community_pairs": same_pairs, "all_account_pairs": pair_count(n),
        "expected_same_primary_probability": same_pairs / pair_count(n) if n >= 2 else None,
        "cluster_size_histogram": dict(sorted(collections.Counter(counts.values()).items())),
        "limitations": [
            "primary partition only; not a valid null for overlapping top-3 memberships or scenes",
            "does not preserve account degree, pack popularity, language, or pack adoption",
            "diagnostic expectation, not a significance test or independent quality gate",
        ],
    }


class PackRecallAccumulator:
    """Streaming summaries keep large pack tables out of evaluator memory."""

    def __init__(self) -> None:
        self.packs = self.raw_members = self.represented_members = 0
        self.raw_pairs = self.represented_pairs = 0
        self.raw_evaluable = self.represented_evaluable = 0
        self.raw_weighted = self.represented_weighted = self.coverage_macro = 0.0
        self.totals = {view: {"hits": 0, "macro_conditional": 0.0, "macro_end_to_end": 0.0, "weighted_hits": 0.0}
                       for view in ("top1", "top3", "scene")}

    def add(self, raw_n: int, represented_n: int, hits: dict[str, int]) -> None:
        raw_pairs, represented_pairs = pair_count(raw_n), pair_count(represented_n)
        weight = 1 / (raw_n - 1) if raw_n >= 2 else 0.0
        self.packs += 1
        self.raw_members += raw_n
        self.represented_members += represented_n
        self.raw_pairs += raw_pairs
        self.represented_pairs += represented_pairs
        self.raw_evaluable += raw_pairs > 0
        self.represented_evaluable += represented_pairs > 0
        self.raw_weighted += weight * raw_pairs
        self.represented_weighted += weight * represented_pairs
        self.coverage_macro += represented_pairs / raw_pairs if raw_pairs else 0.0
        for view, count in hits.items():
            total = self.totals[view]
            total["hits"] += count
            total["weighted_hits"] += weight * count
            total["macro_conditional"] += count / represented_pairs if represented_pairs else 0.0
            total["macro_end_to_end"] += count / raw_pairs if raw_pairs else 0.0

    def report(self, null: dict[str, Any]) -> dict[str, Any]:
        def ratio(numerator: float, denominator: float) -> float | None:
            return numerator / denominator if denominator else None

        metrics = {}
        for view, totals in self.totals.items():
            metrics[view] = {
                "pair_hits": totals["hits"],
                "micro_conditional_recall": ratio(totals["hits"], self.represented_pairs),
                "micro_end_to_end_recall": ratio(totals["hits"], self.raw_pairs),
                "per_pack_macro_conditional_recall": ratio(totals["macro_conditional"], self.represented_evaluable),
                "per_pack_macro_end_to_end_recall": ratio(totals["macro_end_to_end"], self.raw_evaluable),
                "inverse_size_pair_weighted_conditional_recall": ratio(totals["weighted_hits"], self.represented_weighted),
                "inverse_size_pair_weighted_end_to_end_recall": ratio(totals["weighted_hits"], self.raw_weighted),
            }
        probability = null["expected_same_primary_probability"]
        expected = {
            "micro_conditional_recall": 1.0 if self.represented_pairs else None,
            "micro_end_to_end_recall": ratio(self.represented_pairs, self.raw_pairs),
            "per_pack_macro_conditional_recall": 1.0 if self.represented_evaluable else None,
            "per_pack_macro_end_to_end_recall": ratio(self.coverage_macro, self.raw_evaluable),
            "inverse_size_pair_weighted_conditional_recall": 1.0 if self.represented_weighted else None,
            "inverse_size_pair_weighted_end_to_end_recall": ratio(self.represented_weighted, self.raw_weighted),
        }
        null_comparison = {}
        for name, coverage in expected.items():
            baseline = probability * coverage if probability is not None and coverage is not None else None
            observed = metrics["top1"][name]
            null_comparison[name] = {
                "expected": baseline,
                "observed_minus_expected": observed - baseline if observed is not None and baseline is not None else None,
                "observed_over_expected": observed / baseline if observed is not None and baseline else None,
            }
        return {
            "packs_in_base_population": self.packs,
            "packs_with_raw_pairs": self.raw_evaluable, "packs_with_represented_pairs": self.represented_evaluable,
            "raw_member_occurrences": self.raw_members, "represented_member_occurrences": self.represented_members,
            "member_coverage": ratio(self.represented_members, self.raw_members),
            "raw_pair_occurrences": self.raw_pairs, "represented_pair_occurrences": self.represented_pairs,
            "pair_coverage": ratio(self.represented_pairs, self.raw_pairs),
            "inverse_size_weighted_raw_pair_mass": self.raw_weighted,
            "inverse_size_weighted_represented_pair_mass": self.represented_weighted,
            "views": metrics, "primary_partition_same_size_null": null,
            "primary_only_null_comparison": null_comparison,
            "top3_minus_primary_micro_conditional_recall": (
                metrics["top3"]["micro_conditional_recall"] - metrics["top1"]["micro_conditional_recall"]
                if self.represented_pairs else None),
        }


def bulk_follow_diagnostic(projected_columns: set[str]) -> dict[str, Any]:
    required = [
        "source_kind", "source_instance", "segment_generation", "commit_cid", "repo_rev",
        "record_operation", "operation_index", "commit_operation_count", "commit_complete",
        "observed_at", "is_backfill", "repo_sync_state",
    ]
    return {
        "status": "not_performed",
        "reason": "current projection lacks verified live commit grouping and complete source/synchronization provenance",
        "projected_columns_observed": sorted(projected_columns),
        "required_revised_input_contract": required,
        "missing_exact_contract_columns": [name for name in required if name not in projected_columns],
        "classification_contract": [
            "Group creates by repository DID and a verified complete live commit CID/revision; exclude snapshots and backfills.",
            "Compare created subjects with the pack membership valid at the commit time; report overlap count and fraction.",
            "Call this a pack-matching follow burst, not confirmed Follow All intent without client-action evidence.",
            "Report recall separately by burst/adoption stratum with member and pair coverage; freeze thresholds before scoring.",
            "Removing these edges requires a separate organic-graph ablation generation; never alter the current graph silently.",
        ],
        "rev_only_proxy_used": False,
        "pack_follow_intent_inferred": False,
    }


def unperformed_edge_reconstruction() -> dict[str, Any]:
    return {
        "status": "not_performed",
        "reason": "no pinned evaluation edge reference and compatible node universe were supplied",
        "reference": "https://venusatuluri.github.io/KDD_SimClusters_paper.pdf",
        "definition": {
            "predicted_edge": "an unordered account pair sharing at least one community",
            "precision": "observed input edges among predicted edges / all predicted edges",
            "recall": "observed input edges among predicted edges / all input edges",
        },
        "required_inputs": [
            "hashed undirected edge reference, deduplicated without self edges, plus exact node universe",
            "declared input-similarity reconstruction or genuinely held-out follow-edge prediction task",
            "compatible community memberships and explicit overlapping-pair deduplication",
        ],
        "limitations": [
            "pack co-membership recall is not precision and pack nonmembership is not a verified negative",
            "reconstructing fitted input edges is descriptive fit, not held-out generalization",
            "the current directed follow graph cannot be substituted for the paper's undirected reference without defining a new task",
        ],
    }


def eval_follow_edge_capture(graph_dir: pathlib.Path) -> dict[str, Any]:
    """Count the frozen fold output; do not equate producer selection with community recall."""
    required = ["stats.tsv", "producers.tsv", "edges.bin", "input_provenance.json"]
    missing = [name for name in required if not (graph_dir / name).is_file()]
    result: dict[str, Any] = {
        "status": "unavailable", "missing_inputs": missing,
        "metric": "joint consumer-filter and producer-selection edge retention",
        "numerator_definition": "edges from followers with at most 20000 follows to producers with at least 200 eligible followers",
        "denominator_definition": "deduplicated nonself follow edges retained by this fold's account-status filter, before the follower-degree cap",
        "qualification": "counts describe this candidate fold, not authoritative current account state or complete network coverage",
        "pure_producer_selection_capture": None,
        "pure_producer_selection_missing_denominator": "all follow edges from eligible followers before the producer threshold",
        "community_bipartite_edge_capture": {"status": "not_performed",
            "reason": "consumer community memberships were not exported; producer filtering is not community edge capture"},
    }
    if missing:
        return result
    try:
        lines = (graph_dir / "stats.tsv").read_text().strip().splitlines()
        if len(lines) != 2 or len(lines[0].split("\t")) != len(lines[1].split("\t")):
            raise ValueError("stats.tsv must have one header and one equally sized result row")
        stats = dict(zip(lines[0].split("\t"), map(int, lines[1].split("\t"))))
        if any(stats.get(key, -1) < 0 for key in ("follow_edges", "producer_edges", "producers")):
            raise ValueError("required fold counts are missing or negative")
        if stats["producer_edges"] > stats["follow_edges"]:
            raise ValueError("selected edge count exceeds all follow edges")
        producer_count = degree_sum = 0
        with (graph_dir / "producers.tsv").open() as fh:
            for line in fh:
                pid, did, degree = line.rstrip("\n").split("\t")
                if int(pid) != producer_count or not did.startswith("did:") or int(degree) < 0:
                    raise ValueError("invalid dense producer row")
                producer_count += 1
                degree_sum += int(degree)
        if producer_count != stats["producers"] or degree_sum != stats["producer_edges"]:
            raise ValueError("producer rows/degrees disagree with fold counts")
        if (graph_dir / "edges.bin").stat().st_size != stats["producer_edges"] * 8:
            raise ValueError("binary edge bytes disagree with fold counts")
        provenance = json.loads((graph_dir / "input_provenance.json").read_text())
        if provenance["fold_stats"] != stats:
            raise ValueError("provenance fold counts disagree with stats")
        for name in ("stats.tsv", "producers.tsv"):
            if provenance["artifacts"][name]["sha256"] != sha256_file(graph_dir / name):
                raise ValueError(f"{name} differs from the frozen provenance receipt")
        if provenance["artifacts"]["edges.bin"]["bytes"] != (graph_dir / "edges.bin").stat().st_size:
            raise ValueError("binary edge length differs from provenance")
        result.update({
            "status": "computed", "follow_edges": stats["follow_edges"], "producer_edges": stats["producer_edges"],
            "edge_retention_fraction": stats["producer_edges"] / stats["follow_edges"] if stats["follow_edges"] else None,
            "input_provenance_sha256": sha256_file(graph_dir / "input_provenance.json"),
            "verification": "stats and producer SHA256, dense producer rows, degree sum, binary byte count; binary edge SHA256 not re-read",
            "comparison_to_paper_70_percent": "not comparable: the paper reports community capture of bipartite edges",
        })
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update({"status": "invalid_inputs", "reason": str(exc)})
    return result


def eval_pack_comembership(
    event_files: list[pathlib.Path],
    by_did: dict[str, list[tuple[int, int, int, float]]],
    n_communities: int,
    *,
    per_pack_path: pathlib.Path | None = None,
    hub_min_pack_fraction: float = 0.10,
    hub_min_pack_count: int = 2,
) -> dict[str, Any]:
    """Preserve legacy recall and add coverage-aware, weighted pack diagnostics."""
    if not 0 < hub_min_pack_fraction <= 1 or hub_min_pack_count < 2:
        raise ValueError("hub fraction must be in (0,1] and hub count must be at least 2")
    latest: dict[tuple[str, str, str], tuple[str, bool, str, str]] = {}
    projected_columns: set[str] = set()
    for index, pf in enumerate(event_files):
        projected_columns.update(pq.read_schema(pf).names)
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
    pack_lists = [(f"at://{did}/{collection}/{rkey}", value[2]) for (collection, did, rkey), value in latest.items()
                  if collection == "app.bsky.graph.starterpack" and not value[1] and value[2] not in deleted_lists]
    for (collection, _did, _rkey), (_rev, deleted, luri, subj) in latest.items():
        if collection == "app.bsky.graph.listitem" and not deleted and luri and luri not in deleted_lists and subj.startswith("did:"):
            lists[luri].add(subj)
    del latest
    frequency: collections.Counter[str] = collections.Counter()
    for _pack_uri, luri in pack_lists:
        frequency.update(lists.get(luri, set()))
    hubs = {did for did, count in frequency.items()
            if count >= hub_min_pack_count and count / len(pack_lists) >= hub_min_pack_fraction}
    del frequency
    accumulators = {"base": PackRecallAccumulator(), "dehubbed": PackRecallAccumulator()}
    nulls = {"base": primary_partition_null(by_did), "dehubbed": primary_partition_null(by_did, hubs)}
    legacy = PackRecallAccumulator()
    too_small = too_large = raw_too_small = raw_too_large = missing_lists = 0
    removed_occurrences = removed_represented = 0
    writer = None
    output_rows: list[dict[str, Any]] = []
    output_schema = pa.schema([
        ("pack_uri", pa.string()), ("list_uri", pa.string()), ("view", pa.string()),
        ("raw_members", pa.int64()), ("represented_members", pa.int64()),
        ("raw_pairs", pa.int64()), ("represented_pairs", pa.int64()),
        ("top1_pair_hits", pa.int64()), ("top3_pair_hits", pa.int64()), ("scene_pair_hits", pa.int64()),
    ])
    if per_pack_path is not None:
        writer = pq.ParquetWriter(per_pack_path, output_schema, compression="zstd")
    try:
        for pack_uri, luri in sorted(pack_lists):
            members = lists.get(luri, set())
            missing_lists += luri not in lists
            present = sorted(did for did in members if by_did.get(did))
            raw_eligible = 3 <= len(members) <= 200
            legacy_eligible = 3 <= len(present) <= 200
            raw_too_small += len(members) < 3
            raw_too_large += len(members) > 200
            too_small += len(present) < 3
            too_large += len(present) > 200
            if not raw_eligible and not legacy_eligible:
                continue
            hits = pack_pair_hits(present, by_did)
            if legacy_eligible:
                legacy.add(len(present), len(present), hits)
            if not raw_eligible:
                continue
            trimmed = members - hubs
            trimmed_present = sorted(did for did in trimmed if by_did.get(did))
            removed_occurrences += len(members) - len(trimmed)
            removed_represented += len(present) - len(trimmed_present)
            for view, raw, represented, view_hits in [
                ("base", members, present, hits),
                ("dehubbed", trimmed, trimmed_present, pack_pair_hits(trimmed_present, by_did)),
            ]:
                accumulators[view].add(len(raw), len(represented), view_hits)
                if writer is not None:
                    output_rows.append({
                        "pack_uri": pack_uri, "list_uri": luri, "view": view,
                        "raw_members": len(raw), "represented_members": len(represented),
                        "raw_pairs": pair_count(len(raw)), "represented_pairs": pair_count(len(represented)),
                        **{f"{name}_pair_hits": count for name, count in view_hits.items()},
                    })
                    if len(output_rows) >= 4096:
                        writer.write_table(pa.Table.from_pylist(output_rows, schema=output_schema))
                        output_rows.clear()
        if writer is not None and output_rows:
            writer.write_table(pa.Table.from_pylist(output_rows, schema=output_schema))
    finally:
        if writer is not None:
            writer.close()
    reports = {view: accumulators[view].report(nulls[view]) for view in accumulators}
    hub_counts: collections.Counter[str] = collections.Counter()
    for _pack_uri, luri in pack_lists:
        hub_counts.update(lists.get(luri, set()) & hubs)
    hub_rows = [{"did": did, "pack_count": count, "pack_fraction": count / len(pack_lists)}
                for did, count in sorted(hub_counts.items(), key=lambda item: (-item[1], item[0]))]
    per_pack_artifact = None
    if per_pack_path is not None:
        per_pack_artifact = {"path": str(per_pack_path), "sha256": sha256_file(per_pack_path),
                             "rows": accumulators["base"].packs * 2, "schema": str(output_schema)}
    return {
        "source": "archive starterpack records joined to current listitems; seed cohorts excluded",
        "pair_weighting": "legacy keys: all represented pairs per pack; repeated pairs across packs count repeatedly",
        "legacy_population": "3 to 200 represented members, irrespective of raw pack size",
        "starterpacks_evaluated": legacy.packs, "starterpacks_too_small": too_small, "starterpacks_too_large": too_large,
        "live_starterpack_records": len(pack_lists), "live_curated_lists": len(lists),
        "co_member_pairs": legacy.represented_pairs, "communities_count": n_communities,
        **{f"known_for_{name}_pair_hits": values["hits"] for name, values in legacy.totals.items()},
        **{f"known_for_{name}_pair_recall": values["hits"] / legacy.represented_pairs if legacy.represented_pairs else None
           for name, values in legacy.totals.items()},
        "diagnostics_version": "pack-diagnostics-v2",
        "diagnostic_population": "3 to 200 raw observed members before KnownFor filtering; fixed across base and dehubbed views",
        "coverage_qualification": "coverage of observed pack members only; archive completeness and current-state correctness remain unqualified",
        "diagnostic_raw_too_small": raw_too_small, "diagnostic_raw_too_large": raw_too_large,
        "packs_without_observed_live_items": missing_lists,
        "metric_semantics": {
            "conditional": "denominator contains pairs whose two accounts have KnownFor memberships",
            "end_to_end": "denominator contains all raw observed pack pairs, so uncovered accounts cannot create successes",
            "per_pack_macro": "equal weight per pack with a nonzero denominator; counts are reported for each view",
            "inverse_size_pair_weighting": "each pair has weight 1/(raw pack size - 1), computed after hub removal in that view",
            "repeated_pairs": "each occurrence in a distinct starter-pack record remains a separate observation",
            "scene": "coarse scene overlap, not an independent primary community quality score",
        },
        "views": reports,
        "dehub_policy": {
            "scope": "metric-only removal; graph inputs and memberships are unchanged",
            "definition": "accounts occurring in at least the minimum number and fraction of all live starter-pack records",
            "minimum_pack_fraction": hub_min_pack_fraction, "minimum_pack_count": hub_min_pack_count,
            "hub_accounts": len(hubs), "hubs": hub_rows,
            "removed_member_occurrences_in_base_population": removed_occurrences,
            "removed_represented_member_occurrences_in_base_population": removed_represented,
            "threshold_status": "provisional diagnostic; frozen before quality results, not tuned to seed separation",
        },
        "per_pack_artifact": per_pack_artifact,
        "bulk_follow_adoption": bulk_follow_diagnostic(projected_columns),
        "edge_reconstruction": unperformed_edge_reconstruction(),
        "a_blue_start_reference": {
            "publication": "https://www.nature.com/articles/s41597-026-06920-1",
            "full_text": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13066036/",
            "authors_code": "https://github.com/nwlandry/a-blue-start/blob/main/get_starterpack_clustering.py",
            "reported_communities": 503,
            "construction": "Leiden on the unweighted pairwise projection of the largest starter-pack component",
            "replicated_here": False,
            "independent_pack_recall_ground_truth": False,
            "identity_limitation": "published deidentified node IDs cannot be joined to local DIDs; no reidentification attempted",
        },
        "a_blue_start_503_partition_available": False,
        "beats_a_blue_start_503_partition": None,
        "baseline_status": "not measured: published pack-derived partition is not an independent pack-recall ground truth; compatible identity mapping unavailable",
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
    kf: pa.Table | None = None,
) -> pa.Table:
    """Create heuristic profile-term labels and prompts for 500 communities.

    These labels are not LLM- or human-reviewed. Seed identities are never used.
    The archive projection does not provide profile handles or starter-pack titles
    here, so no claim of either enrichment is made.
    """
    hide_dids: set[str] = set()
    hide_path = graph_dir / "hide_from_recs.tsv"
    if hide_path.is_file():
        hide_dids.update(line.strip() for line in hide_path.read_text().splitlines() if line.strip())
    if kf is not None and "hide_from_recs" in kf.column_names:
        for d_val, h_val in zip(kf["did"].to_pylist(), kf["hide_from_recs"].to_pylist()):
            if h_val:
                hide_dids.add(d_val)

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
    raw_top_dids_col = comm_tbl["top_dids"].to_pylist()
    raw_top_foll_col = comm_tbl["top_followers"].to_pylist()
    top_dids_col = []
    top_foll_col = []
    for tdids, tfoll in zip(raw_top_dids_col, raw_top_foll_col):
        filtered_pairs = [(d, f) for d, f in zip(tdids, tfoll) if d not in hide_dids]
        top_dids_col.append([d for d, _ in filtered_pairs])
        top_foll_col.append([f for _, f in filtered_pairs])

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
        rank      UInt8,
        hide_from_recs Bool DEFAULT false
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
        f"clickhouse-client -q \"INSERT INTO catbird.known_for (pid, did, community, scene_id, domain_id, score, rank, hide_from_recs) FORMAT Parquet\" < /tmp/known_for.parquet\n"
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
    observed_source = pathlib.Path(__file__).resolve()
    source_observation = {
        "observed_at_entry": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "path": str(observed_source), "sha256": sha256_file(observed_source),
        "python": platform.python_version(), "argv": sys.argv,
        "packages": {package: importlib.metadata.version(package) for package in ("numpy", "pyarrow")},
        "qualification": "source file observed at evaluator entry and checked at exit; not a retroactive attestation of the waiting driver",
    }
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", action="append", required=True, help="Archive-only directory; repeat for each archive")
    ap.add_argument("--graph-dir", default="/Users/joshlacalamito/catbird-lake-local/graph-build/out")
    ap.add_argument("--manifest", default="/Users/joshlacalamito/catbird-lake-local/graph-build/seed_and_traps_manifest.json")
    ap.add_argument("--ship-netcup", action="store_true")
    ap.add_argument("--version", default="v20261019")
    ap.add_argument("--hub-min-pack-fraction", type=float, default=0.10)
    ap.add_argument("--hub-min-pack-count", type=int, default=2)
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
    pack_eval = eval_pack_comembership(
        event_files, by_did, comm_tbl.num_rows,
        per_pack_path=graph_dir / "pack_recall_diagnostics.parquet",
        hub_min_pack_fraction=args.hub_min_pack_fraction,
        hub_min_pack_count=args.hub_min_pack_count,
    )
    follow_capture = eval_follow_edge_capture(graph_dir)
    enriched_comm = enrich_and_label_communities(event_files, graph_dir, by_did, comm_tbl, kf=kf)
    for item in inventory:
        current = pathlib.Path(item["path"]).stat()
        if current.st_size != item["size"] or current.st_mtime_ns != item["mtime_ns"]:
            raise SystemExit(f"Archive input changed during evaluation: {item['path']}")
    if sha256_file(observed_source) != source_observation["sha256"]:
        raise SystemExit("Evaluator source changed during execution; result is not sealed")

    ship_res = None
    if args.ship_netcup:
        ship_res = ship_to_netcup(graph_dir, args.version)

    hide_col_present = "hide_from_recs" in kf.column_names
    hide_flagged_dids = (
        {d_val for d_val, h_val in zip(kf["did"].to_pylist(), kf["hide_from_recs"].to_pylist()) if h_val}
        if hide_col_present else set()
    )
    hide_flagged_rows = int(sum(1 for h_val in kf["hide_from_recs"].to_pylist() if h_val)) if hide_col_present else 0
    remaining_in_top_dids = sum(
        1 for tdids in enriched_comm["top_dids"].to_pylist() for d_val in tdids if d_val in hide_flagged_dids
    )

    report = {
        "graph_version": "phase1-v1",
        "known_for_sha256": sha256_file(graph_dir / "known_for.parquet"),
        "known_for_bytes": (graph_dir / "known_for.parquet").stat().st_size,
        "known_for_mb": round((graph_dir / "known_for.parquet").stat().st_size / 1e6, 3),
        "known_for_under_50mb": (graph_dir / "known_for.parquet").stat().st_size <= 50_000_000,
        "hide_from_recs": {
            "column_present": hide_col_present,
            "known_for_flagged_rows": hide_flagged_rows,
            "known_for_flagged_accounts": len(hide_flagged_dids),
            "communities_top_dids_flagged_remaining": remaining_in_top_dids,
        },
        "evaluation_input_manifest_sha256": sha256_file(graph_dir / "evaluation_input_manifest.json"),
        "evaluator_source_observation": source_observation,
        "seed_manifest_sha256": sha256_file(pathlib.Path(args.manifest)),
        "seed_manifest_use": "evaluation only; never clustering or labeling",
        "cluster_stats": json.loads((graph_dir / "cluster_stats.json").read_text()),
        "separation_summary": {
            k: v for k, v in sep.items() if k not in ("subcultures", "collision_traps")
        },
        "collision_traps": sep["collision_traps"],
        "pack_comembership": pack_eval,
        "follow_edge_capture": follow_capture,
        "netcup_ship": ship_res,
    }
    (graph_dir / "phase1_eval_report.json").write_text(json.dumps({**report, "subcultures": sep["subcultures"]}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
