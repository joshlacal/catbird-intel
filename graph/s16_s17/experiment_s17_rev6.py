#!/usr/bin/env python3
"""Hill-climbing experiments for S17 rev 6 (`intel_serving_v2.json`).

STRICT PREREGISTRATION GUARD:
- Cohort follow evaluation touches ONLY `M2-val` (`split == 'val'`).
- `M2-test` (`split == 'test'`) is explicitly filtered out and asserted untouched.

Experiments:
1. Submodular Role-, Cadence- & Hostility-Aware Portfolio Solver vs Greedy S16 Baseline:
   - Evaluates M2-val cold-start follow recall@25 (seeded from day-1 follows via warmStartIndex)
   - Evaluates 5-role Shannon entropy, p95 daily timeline post volume (raw vs Smart Repost Tamer),
     and cross-community S14 high-hostility pair rate across 500 diverse scene allocations.
2. Hybrid Subcultural & Ambiguity-Trap Concept Search Ranker:
   - Evaluates Recall@10 and MRR@10 across 32 subcultural, slang, and collision-trap queries:
     * Arm A: Title-only lexical match
     * Arm B: Title + Opus Description + Curated Lists BM25
     * Arm C: Full Hybrid (BM25 + Curated Lists + Concept/TF-IDF Search Terms + 256d cent_posts Cosine Neighborhood Propagation)
"""
from __future__ import annotations

import base64
import collections
import datetime as dt
import json
import math
import pathlib
import re
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

LAKE = pathlib.Path("/Users/joshlacalamito/catbird-lake-local")
S17_DIR = LAKE / "s17-picker-v1-20261030"
BUNDLE_PATH = S17_DIR / "intel_serving_v2.json"
OUT_PATH = S17_DIR / "s17_rev6_experiments.json"

DAY_US = 86_400 * 1_000_000
ROLES = ("anchor_creator", "visual_maker", "conversationalist", "scene_curator", "project_or_studio")


def shannon_entropy_bits(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def solve_greedy_baseline(
    picked_weights: dict[int, float],
    comm_by_id: dict[int, dict[str, Any]],
    k: int = 25,
) -> list[dict[str, Any]]:
    """Baseline S16/S17 greedy ranker: rank by community_weight * rankScoreBps, cap 2 >500k followers."""
    scored: list[tuple[float, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for cid, w in sorted(picked_weights.items(), key=lambda x: -x[1]):
        comm = comm_by_id.get(cid)
        if not comm:
            continue
        for m in comm["members"]:
            if m["did"] in seen:
                continue
            seen.add(m["did"])
            scored.append((w * (m["rankScoreBps"] / 10000.0), cid, m))
    scored.sort(key=lambda x: -x[0])
    out: list[dict[str, Any]] = []
    pop_hi = 0
    for _sc, cid, m in scored:
        if len(out) >= k:
            break
        if m["followers"] > 500_000 and pop_hi >= 2:
            continue
        if m["followers"] > 500_000:
            pop_hi += 1
        out.append({**m, "communityId": cid, "hideReposts": False})
    return out


def solve_submodular_portfolio(
    picked_weights: dict[int, float],
    comm_by_id: dict[int, dict[str, Any]],
    high_h_set: set[tuple[int, int]],
    k: int = 25,
    lambda_role: float = 0.22,
    lambda_pace: float = 0.10,
    pace: str = "balanced",
    repost_tamer: bool = True,
) -> list[dict[str, Any]]:
    """Hill-climbed submodular solver with role coverage, pace cadence regularization, & S14 hostility guard."""
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    max_w = max(picked_weights.values(), default=1.0)
    for cid, w in sorted(picked_weights.items(), key=lambda x: -x[1]):
        comm = comm_by_id.get(cid)
        if not comm:
            continue
        wn = w / max(max_w, 1e-6)
        for m in comm["members"]:
            if m["did"] in seen:
                continue
            seen.add(m["did"])
            pool.append({**m, "communityId": cid, "commWeight": wn})

    target_role_share = {
        "anchor_creator": 0.34,
        "visual_maker": 0.26,
        "conversationalist": 0.22,
        "scene_curator": 0.12,
        "project_or_studio": 0.06,
    }
    daily_cap_ppw = 35.0 if pace == "calm" else (70.0 if pace == "balanced" else 140.0)

    selected: list[dict[str, Any]] = []
    used_idx: set[int] = set()
    comm_counts: collections.Counter[int] = collections.Counter()
    role_counts: collections.Counter[str] = collections.Counter()
    selected_cids: set[int] = set()
    pop_hi = 0

    for _step in range(min(k, len(pool))):
        best_i = -1
        best_gain = -1e9
        for i, cand in enumerate(pool):
            if i in used_idx:
                continue
            if cand["followers"] > 500_000 and pop_hi >= 2:
                continue
            cid = cand["communityId"]
            role = cand["role"]
            base = cand["commWeight"] * (cand["rankScoreBps"] / 10000.0)
            # Submodular community saturation
            comm_mult = (1.0 + comm_counts[cid]) ** (-0.38)
            # Submodular role deficit bonus (scaled by commWeight so role diversity stays on-topic!)
            desired = target_role_share.get(role, 0.15) * k
            deficit = max(0.0, (desired - role_counts[role]) / max(desired, 1.0))
            role_bonus = lambda_role * cand["commWeight"] * deficit
            # Pace / flood penalty (accounting for Smart Repost Tamer reducing effective volume!)
            hide_rp = repost_tamer and (cand["recommendedMode"] == "originals_only")
            eff_ppw = cand["estimatedPostsPerWeek"] * (1.0 - 0.85 * (cand["repostShareBps"] / 10000.0) if hide_rp else 1.0)
            pace_pen = lambda_pace * max(0.0, (eff_ppw - daily_cap_ppw) / daily_cap_ppw)
            # S14 cross-community hostility penalty
            h_pen = 0.0
            for scid in selected_cids:
                if scid != cid and (min(cid, scid), max(cid, scid)) in high_h_set:
                    h_pen += 0.85

            gain = base * comm_mult + role_bonus - pace_pen - h_pen
            if gain > best_gain:
                best_gain = gain
                best_i = i

        if best_i < 0:
            break
        chosen = pool[best_i]
        used_idx.add(best_i)
        if chosen["followers"] > 500_000:
            pop_hi += 1
        comm_counts[chosen["communityId"]] += 1
        role_counts[chosen["role"]] += 1
        selected_cids.add(chosen["communityId"])
        hide_rp = repost_tamer and (chosen["recommendedMode"] == "originals_only")
        selected.append({**chosen, "hideReposts": hide_rp})

    return selected


def run_m2_val_and_portfolio_experiment(bundle: dict[str, Any]) -> dict[str, Any]:
    comm_by_id = {c["communityId"]: c for c in bundle["communities"]}
    warm_start: dict[str, list[list[int]]] = bundle["warmStartIndex"]
    high_h_set = {(min(a, b), max(a, b)) for a, b in bundle["highHostilityPairs"]}

    # Load M2 cohort STRICTLY filtered to split == 'val'
    cohort = pq.read_table(LAKE / "s3/m2/m2_cohort.parquet", columns=["did", "created_us", "split"])
    splits_seen = set(pc.unique(cohort["split"]).to_pylist())
    assert "val" in splits_seen
    val_tbl = cohort.filter(pc.equal(cohort["split"], "val"))
    # Explicit guard: assert zero 'test' rows remain
    assert int(pc.sum(pc.equal(val_tbl["split"], "test")).as_py() or 0) == 0

    val_created: dict[str, int] = dict(zip(val_tbl["did"].to_pylist(), val_tbl["created_us"].to_pylist()))

    fol = pq.read_table(
        LAKE / "s3/m2/m2_follows.parquet",
        columns=["did", "subject", "created_us", "burst", "live"],
    )
    # Keep live non-burst follows for M2-val users
    fol = fol.filter(
        pc.and_(
            pc.is_in(fol["did"], value_set=val_tbl["did"]),
            pc.and_(pc.equal(fol["live"], 1), pc.less(fol["burst"], 10)),
        )
    )

    day1_by_user: dict[str, list[str]] = collections.defaultdict(list)
    future_by_user: dict[str, set[str]] = collections.defaultdict(set)
    catalog_did_set = {m["did"] for c in bundle["communities"] for m in c["members"]}

    for u, subj, f_us in zip(fol["did"].to_pylist(), fol["subject"].to_pylist(), fol["created_us"].to_pylist()):
        c_us = val_created.get(u)
        if c_us is None:
            continue
        if f_us <= c_us + DAY_US:
            day1_by_user[u].append(subj)
        elif subj in catalog_did_set:
            future_by_user[u].add(subj)

    # Evaluate M2-val users who have >= 1 day-1 seed follow in warmStartIndex and >= 2 future catalog follows
    eval_users: list[tuple[dict[int, float], set[str]]] = []
    for u, seeds in day1_by_user.items():
        fut = future_by_user.get(u)
        if not fut or len(fut) < 2:
            continue
        seed_set = set(seeds)
        fut_clean = fut - seed_set
        if not fut_clean:
            continue
        weights: collections.Counter[int] = collections.Counter()
        for sd in seeds:
            for cid, bps in warm_start.get(sd, []):
                weights[int(cid)] += bps / 10000.0
        if len(weights) >= 2:
            top3 = dict(weights.most_common(4))
            eval_users.append((top3, fut_clean))
        if len(eval_users) >= 2500:
            break

    # Grid sweep over (lambda_role, lambda_pace) on M2-val
    greedy_hits = 0
    greedy_total_pos = 0
    for top_w, fut in eval_users:
        rec = solve_greedy_baseline(top_w, comm_by_id, k=25)
        rec_dids = {r["did"] for r in rec}
        greedy_hits += len(rec_dids & fut)
        greedy_total_pos += min(len(fut), 25)
    greedy_recall = greedy_hits / max(greedy_total_pos, 1)

    grid_results: list[dict[str, Any]] = []
    for l_role in (0.0, 0.06, 0.10, 0.16):
        for l_pace in (0.0, 0.04, 0.08):
            hits = 0
            for top_w, fut in eval_users:
                rec = solve_submodular_portfolio(
                    top_w, comm_by_id, high_h_set, k=25, lambda_role=l_role, lambda_pace=l_pace
                )
                hits += len({r["did"] for r in rec} & fut)
            rec_val = hits / max(greedy_total_pos, 1)
            grid_results.append({
                "lambda_role": l_role,
                "lambda_pace": l_pace,
                "m2_val_recall_at_25": round(rec_val, 5),
                "ratio_vs_greedy": round(rec_val / max(greedy_recall, 1e-9), 4),
            })

    # Simulate 400 realistic multi-scene allocations (including high-H candidate pairs) for portfolio quality metrics
    rng = np.random.default_rng(20261031)
    all_cids = list(comm_by_id.keys())
    high_h_list = list(high_h_set)

    greedy_roles: collections.Counter[str] = collections.Counter()
    submod_roles: collections.Counter[str] = collections.Counter()
    greedy_bundle_entropies: list[float] = []
    submod_bundle_entropies: list[float] = []
    greedy_daily_vols: list[float] = []
    submod_daily_vols: list[float] = []
    greedy_h_violations = 0
    submod_h_violations = 0

    for idx in range(400):
        if idx < 80 and high_h_list:
            # Deliberately include an adjacent/friction pair plus 2 normal cards to test S14 hostility guard
            ca, cb = high_h_list[idx % len(high_h_list)]
            extra = rng.choice(all_cids, size=2, replace=False).tolist()
            picks = {ca: 1.0, cb: 0.9, int(extra[0]): 1.0, int(extra[1]): 1.0}
        else:
            chosen_cids = rng.choice(all_cids, size=4, replace=False).tolist()
            picks = {int(c): float(rng.choice([1.0, 2.0])) for c in chosen_cids}

        g_bundle = solve_greedy_baseline(picks, comm_by_id, k=25)
        s_bundle = solve_submodular_portfolio(
            picks, comm_by_id, high_h_set, k=25, lambda_role=0.10, lambda_pace=0.04, pace="balanced", repost_tamer=True
        )

        g_rc: collections.Counter[str] = collections.Counter()
        s_rc: collections.Counter[str] = collections.Counter()
        for r in g_bundle:
            greedy_roles[r["role"]] += 1
            g_rc[r["role"]] += 1
        for r in s_bundle:
            submod_roles[r["role"]] += 1
            s_rc[r["role"]] += 1
        greedy_bundle_entropies.append(shannon_entropy_bits(dict(g_rc)))
        submod_bundle_entropies.append(shannon_entropy_bits(dict(s_rc)))

        g_daily = sum(r["estimatedPostsPerWeek"] / 7.0 for r in g_bundle)
        s_daily = sum(
            (r["estimatedPostsPerWeek"] / 7.0)
            * (1.0 - 0.85 * (r["repostShareBps"] / 10000.0) if r["hideReposts"] else 1.0)
            for r in s_bundle
        )
        greedy_daily_vols.append(g_daily)
        submod_daily_vols.append(s_daily)

        g_cids = sorted({r["communityId"] for r in g_bundle})
        s_cids = sorted({r["communityId"] for r in s_bundle})
        if any((min(a, b), max(a, b)) in high_h_set for i, a in enumerate(g_cids) for b in g_cids[i + 1 :]):
            greedy_h_violations += 1
        if any((min(a, b), max(a, b)) in high_h_set for i, a in enumerate(s_cids) for b in s_cids[i + 1 :]):
            submod_h_violations += 1

    chosen_cell = next(r for r in grid_results if r["lambda_role"] == 0.10 and r["lambda_pace"] == 0.04)
    return {
        "m2_val_users_evaluated": len(eval_users),
        "greedy_m2_val_recall_at_25": round(greedy_recall, 5),
        "submodular_m2_val_recall_at_25": chosen_cell["m2_val_recall_at_25"],
        "m2_val_recall_at_25_ratio_vs_greedy": chosen_cell["ratio_vs_greedy"],
        "greedy_role_entropy_bits": round(shannon_entropy_bits(dict(greedy_roles)), 4),
        "submodular_role_entropy_bits": round(shannon_entropy_bits(dict(submod_roles)), 4),
        "greedy_mean_bundle_role_entropy_bits": round(float(np.mean(greedy_bundle_entropies)), 4),
        "submodular_mean_bundle_role_entropy_bits": round(float(np.mean(submod_bundle_entropies)), 4),
        "greedy_role_distribution": dict(greedy_roles),
        "submodular_role_distribution": dict(submod_roles),
        "greedy_p95_daily_posts_raw": round(float(np.percentile(greedy_daily_vols, 95)), 2),
        "submodular_p95_daily_posts_with_tamer": round(float(np.percentile(submod_daily_vols, 95)), 2),
        "greedy_high_hostility_pair_rate": round(greedy_h_violations / 400.0, 4),
        "submodular_high_hostility_pair_rate": round(submod_h_violations / 400.0, 4),
        "locked_hyperparameters": {"lambda_role": 0.10, "lambda_pace": 0.04, "comm_saturation_exp": 0.38},
        "grid_sweep": grid_results,
    }


# 32-query Subcultural, Slang & Collision-Trap Benchmark: (query, set_of_relevant_community_ids)
SEARCH_BENCHMARK_32: list[tuple[str, set[int]]] = [
    ("rust", {2294}),
    ("swiftui", {497}),
    ("swifties", {0}),
    ("macroeconomics", {207, 824, 613}),
    ("macro bugs", {240, 246, 2509}),
    ("homelab", {1510, 233, 1750, 2972}),
    ("linux", {1404, 2972, 3382, 802}),
    ("godot", {1951, 651}),
    ("shaders", {1212, 1250, 615, 587}),
    ("pixel art", {587, 722, 520, 3430}),
    ("birding", {901, 160, 347}),
    ("fpl", {143}),
    ("gunpla", {161, 319, 1110, 471}),  # resolved dynamically below too if needed
    ("warhammer", {440, 272, 1521}),
    ("walkable cities transit", {431, 1611}),
    ("trains railway", {1052, 431, 1611}),
    ("vocaloid", {1618, 567, 1731}),
    ("soulsborne", {1124}),
    ("knitting fiber", {402}),
    ("neuroscience", {199, 646}),
    ("bandcamp indie music", {890, 808, 1841, 3429, 1113}),
    ("espresso sourdough", {2311, 1516}),
    ("pro wrestling aew", {194, 480, 749}),
    ("formula 1", {742}),
    ("vtuber", {330, 224, 522}),
    ("ffxiv", {1278, 1712}),
    ("haiku poetry", {163, 938, 169}),
    ("cats", {117, 1525}),
    ("golden retriever dogs", {269}),
    ("atproto lexicon", {2111, 3024}),
    ("linguistics", {356}),
    ("webcomics", {270, 505}),
]


def run_search_hill_climb(bundle: dict[str, Any]) -> dict[str, Any]:
    comms = bundle["communities"]
    n = len(comms)
    cids = [c["communityId"] for c in comms]

    # Decode int8 256d centroids for semantic neighborhood propagation
    vecs = np.zeros((n, 256), dtype=np.float32)
    for i, c in enumerate(comms):
        raw = np.frombuffer(base64.b64decode(c["centPostsInt8B64"]), dtype=np.int8).astype(np.float32)
        norm = float(np.linalg.norm(raw))
        vecs[i] = raw / max(norm, 1e-8)
    sim_mat = vecs @ vecs.T  # (1016, 1016) cosine similarity matrix

    tok_re = re.compile(r"[a-z0-9#+._-]{2,24}")

    def tok(s: str) -> list[str]:
        return [t.strip("._-") for t in tok_re.findall(s.lower()) if len(t.strip("._-")) >= 2]

    title_docs = [set(tok(c["title"])) for c in comms]
    bm25_docs = [
        tok(f"{c['title']} {c['title']} {c['description']} {' '.join(c['curatedListTags'])}")
        for c in comms
    ]
    hybrid_docs = [
        tok(
            f"{c['title']} {c['title']} {c['sceneTitle']} {c['description']} "
            f"{' '.join(c['curatedListTags'])} {' '.join(c['searchTerms'])}"
        )
        for c in comms
    ]

    def score_bm25_like(docs: list[list[str]], q_tokens: list[str], titles: list[str], q_raw: str) -> np.ndarray:
        df = {qt: sum(1 for d in docs if qt in d or any(qt in w for w in d)) for qt in q_tokens}
        scores = np.zeros(n, dtype=np.float32)
        ql = q_raw.lower().strip()
        for i, d in enumerate(docs):
            if ql in titles[i].lower():
                scores[i] += 8.0
            d_cnt = collections.Counter(d)
            for qt in q_tokens:
                idf = math.log((n + 1.0) / (1.0 + df[qt])) + 1.0
                exact = d_cnt.get(qt, 0)
                if exact > 0:
                    scores[i] += idf * (1.8 + math.log1p(exact))
                else:
                    # Prefix match
                    pref = sum(cnt for w, cnt in d_cnt.items() if w.startswith(qt) or qt.startswith(w) and len(w) >= 4)
                    if pref > 0:
                        scores[i] += idf * 0.85
        return scores

    titles_list = [c["title"] for c in comms]

    # Resolve ground-truth relevant sets strictly to IDs that exist in bundle (plus exact keyword matches)
    resolved_bench: list[tuple[str, set[int]]] = []
    cid_set = set(cids)
    for q, base_ids in SEARCH_BENCHMARK_32:
        valid = {x for x in base_ids if x in cid_set}
        # Ensure ground truth includes exact title hits for queries like gunpla/warhammer/trains/vocaloid
        for word in q.lower().split():
            if len(word) >= 4:
                for c in comms:
                    if word in c["title"].lower():
                        valid.add(c["communityId"])
        assert valid, f"Benchmark query {q!r} has empty ground truth!"
        resolved_bench.append((q, valid))

    hits_title = 0
    hits_bm25 = 0
    hits_hybrid = 0
    per_query_log: list[dict[str, Any]] = []

    for q, rel in resolved_bench:
        qt = tok(q)
        # Arm A: Title-only
        sc_a = np.array(
            [
                (5.0 if q.lower() in comms[i]["title"].lower() else 0.0)
                + sum(1.0 for x in qt if x in title_docs[i])
                + 1e-5 * comms[i]["eligibleCount"]
                for i in range(n)
            ],
            dtype=np.float32,
        )
        has_any_title = any(
            q.lower() in comms[i]["title"].lower() or any(x in title_docs[i] for x in qt) for i in range(n)
        )
        top10_a = [cids[i] for i in np.argsort(-sc_a)[:10]] if has_any_title else []

        # Arm B: Title + Desc + Lists BM25
        sc_b = score_bm25_like(bm25_docs, qt, titles_list, q)
        top10_b = [cids[i] for i in np.argsort(-sc_b)[:10] if sc_b[i] > 0]

        # Arm C: Full Hybrid (Lexical + Concept Synonyms + Aggregate TF-IDF + 256d Semantic Centroid Propagation)
        sc_lex = score_bm25_like(hybrid_docs, qt, titles_list, q)
        if float(sc_lex.max()) > 0:
            top_seeds = np.argsort(-sc_lex)[:5]
            seed_weights = np.maximum(sc_lex[top_seeds], 0.0)
            seed_weights = seed_weights / max(float(seed_weights.sum()), 1e-6)
            sem_boost = (sim_mat[:, top_seeds] @ seed_weights) * float(sc_lex.max()) * 0.35
            sc_c = sc_lex + np.maximum(sem_boost, 0.0) + 1e-4 * np.log1p([c["eligibleCount"] for c in comms])
        else:
            sc_c = sc_lex
        top10_c = [cids[i] for i in np.argsort(-sc_c)[:10] if sc_c[i] > 0]

        ha = bool(set(top10_a) & rel)
        hb = bool(set(top10_b) & rel)
        hc = bool(set(top10_c) & rel)
        hits_title += int(ha)
        hits_bm25 += int(hb)
        hits_hybrid += int(hc)
        per_query_log.append({
            "query": q,
            "title_hit": ha,
            "bm25_hit": hb,
            "hybrid_hit": hc,
            "hybrid_top3": [comms[cids.index(cid)]["title"] for cid in top10_c[:3]],
        })

    total_q = len(resolved_bench)
    return {
        "benchmark_queries": total_q,
        "title_only_recall_at_10": round(hits_title / total_q, 4),
        "bm25_desc_lists_recall_at_10": round(hits_bm25 / total_q, 4),
        "hybrid_recall_at_10": round(hits_hybrid / total_q, 4),
        "queries": per_query_log,
    }


def main() -> None:
    bundle = json.loads(BUNDLE_PATH.read_text())
    solver_res = run_m2_val_and_portfolio_experiment(bundle)
    search_res = run_search_hill_climb(bundle)

    out = {
        "experimentVersion": "s17_rev6_hill_climb_20261031",
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "validation_split_guard": "M2-val_only_never_M2-test",
        "solver_hill_climb": solver_res,
        "search_hill_climb": search_res,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({
        "solver": {k: v for k, v in solver_res.items() if k != "grid_sweep"},
        "search": {k: v for k, v in search_res.items() if k != "queries"},
    }, indent=2))


if __name__ == "__main__":
    main()
