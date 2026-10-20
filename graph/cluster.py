#!/usr/bin/env python3
"""Cluster the producer similarity graph and emit KnownFor.

Inputs (graph-build/out): producers.tsv, sim.bin (from simgraph)
Outputs:
  communities.parquet   community, size, top_dids (by followers), internal_density
  known_for.parquet     pid, did, community, score, rank   (top 3 memberships)
  cluster_stats.json    resolution, counts, size distribution

Leiden with the Constant Potts Model (no resolution limit) is tuned by
bisection on the resolution so the median size of non-trivial communities is
near --target-size producers (SimClusters found ~100 works best). Membership
scores give overlap: a producer's score for community c is the share of its
similarity weight going to members of c.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import igraph as ig
import leidenalg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def load(dirpath: pathlib.Path):
    pids, dids, followers = [], [], []
    with (dirpath / "producers.tsv").open() as fh:
        for line in fh:
            pid, did, n = line.rstrip("\n").split("\t")
            pids.append(int(pid))
            dids.append(did)
            followers.append(int(n))
    raw = np.fromfile(dirpath / "sim.bin", dtype=np.dtype([("p", "<u4"), ("q", "<u4"), ("w", "<f4")]))
    return dids, np.array(followers), raw


def prune_for_backbone(raw: np.ndarray, top_k: int = 35, min_cos: float = 0.035) -> np.ndarray:
    """Keep up to top_k highest-weight outgoing edges per producer p AND cap
    incoming degree per target q at top_k * 2 so generic mega-hubs cannot act
    as 50,000-degree star centers that glue unrelated communities together."""
    mask = raw["w"] >= min_cos
    sub = raw[mask]
    if len(sub) == 0:
        return sub
    # simgraph groups edges by p but does not promise weight order within p.
    # Always sort before selecting the strongest neighbors.
    order = np.lexsort((sub["q"], -sub["w"], sub["p"]))
    sub = sub[order]
    first_p = np.r_[0, np.flatnonzero(np.diff(sub["p"])) + 1]
    rank_p = np.arange(len(sub)) - np.repeat(first_p, np.diff(np.r_[first_p, len(sub)]))
    sub = sub[rank_p < top_k]
    if len(sub) == 0:
        return sub
    # Cap incoming degree per target q at top_k * 2 (highest cosine first)
    order_q = np.lexsort((-sub["w"], sub["q"]))
    sub_q = sub[order_q]
    first_q = np.r_[0, np.flatnonzero(np.diff(sub_q["q"])) + 1]
    rank_q = np.arange(len(sub_q)) - np.repeat(first_q, np.diff(np.r_[first_q, len(sub_q)]))
    return sub_q[rank_q < (top_k * 2)]


def symmetric_edges(raw: np.ndarray):
    """Undirected edges with weight = max of the two directed estimates."""
    a = np.minimum(raw["p"], raw["q"]).astype(np.int64)
    b = np.maximum(raw["p"], raw["q"]).astype(np.int64)
    key = (a << 32) | b
    order = np.argsort(key, kind="stable")
    key, w = key[order], raw["w"][order]
    uniq, start = np.unique(key, return_index=True)
    wmax = np.maximum.reduceat(w, start)
    return (uniq >> 32).astype(np.int64), (uniq & 0xFFFFFFFF).astype(np.int64), wmax.astype(np.float64)


def run_leiden(g: ig.Graph, resolution: float, seed: int, iterations: int = 2):
    part = leidenalg.find_partition(
        g, leidenalg.CPMVertexPartition, weights="weight", resolution_parameter=resolution, n_iterations=iterations, seed=seed
    )
    return np.array(part.membership)


def median_size(membership: np.ndarray, min_size: int) -> float:
    """Median, over producers in non-trivial communities, of their community
    size. Unlike a per-community median, a single giant blob dominates this."""
    sizes = np.bincount(membership)
    per_node = sizes[membership]
    per_node = per_node[per_node >= min_size]
    return float(np.median(per_node)) if len(per_node) else 0.0


def community_median_size(membership: np.ndarray, min_size: int) -> float:
    sizes = np.bincount(membership)
    return float(np.median(sizes[sizes >= min_size])) if np.any(sizes >= min_size) else 0.0


def build_hierarchy(
    n_comm: int,
    comm: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
    target_scenes: int,
    target_domains: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 3-level tree (domain_id -> scene_id -> community) by clustering
    the cosine-normalized community-to-community similarity graph."""
    if n_comm <= 1:
        return np.zeros(n_comm, dtype=np.int32), np.zeros(n_comm, dtype=np.int32)
    ca = comm[src]
    cb = comm[dst]
    valid = (ca >= 0) & (cb >= 0) & (ca != cb)
    ca, cb, cw = ca[valid], cb[valid], w[valid]
    u = np.minimum(ca, cb).astype(np.int64)
    v = np.maximum(ca, cb).astype(np.int64)
    key = u * n_comm + v
    uk, inv = np.unique(key, return_inverse=True)
    ew = np.bincount(inv, weights=cw)
    cu = (uk // n_comm).astype(np.int64)
    cv = (uk % n_comm).astype(np.int64)
    strength = np.bincount(np.concatenate([cu, cv]), weights=np.concatenate([ew, ew]), minlength=n_comm)
    norm_w = ew / np.maximum(np.sqrt(strength[cu] * strength[cv]), 1e-9)

    # Keep top-20 strongest neighbors per community so meta-graph has <= 20k edges
    keep_mask = norm_w >= 0.005
    cu, cv, norm_w, ew = cu[keep_mask], cv[keep_mask], norm_w[keep_mask], ew[keep_mask]
    if len(norm_w) > 0:
        order_c = np.lexsort((-norm_w, cu))
        cu_s, cv_s, nw_s, ew_s = cu[order_c], cv[order_c], norm_w[order_c], ew[order_c]
        first_c = np.r_[0, np.flatnonzero(np.diff(cu_s)) + 1]
        rank_c = np.arange(len(cu_s)) - np.repeat(first_c, np.diff(np.r_[first_c, len(cu_s)]))
        top_c = rank_c < 20
        cu, cv, norm_w, ew = cu_s[top_c], cv_s[top_c], nw_s[top_c], ew_s[top_c]

    cg = ig.Graph(n=n_comm, edges=list(zip(cu.tolist(), cv.tolist())), directed=False)
    cg.es["weight"] = norm_w.tolist()

    def tune_rb(graph: ig.Graph, target_k: int, lo: float = 0.05, hi: float = 20.0) -> np.ndarray:
        best_m = np.zeros(graph.vcount(), dtype=np.int32)
        best_diff = 10**9
        for _ in range(7):
            mid = (lo * hi) ** 0.5
            part = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=mid,
                n_iterations=2,
                seed=seed,
            )
            m = np.array(part.membership, dtype=np.int32)
            counts = np.bincount(m)
            k = int((counts >= 2).sum()) or int(len(counts))
            if abs(k - target_k) < best_diff:
                best_diff = abs(k - target_k)
                best_m = m
            if k < target_k:
                lo = mid
            else:
                hi = mid
        counts = np.bincount(best_m)
        for node in np.flatnonzero(counts[best_m] == 1):
            inc = graph.incident(int(node))
            if inc:
                best_edge = max(inc, key=lambda eid: graph.es[eid]["weight"])
                e = graph.es[best_edge]
                nbr = e.target if e.source == node else e.source
                best_m[node] = best_m[nbr]
        _, dense = np.unique(best_m, return_inverse=True)
        return dense.astype(np.int32)

    scene_of_comm = tune_rb(cg, target_scenes, lo=0.2, hi=25.0)
    n_scenes = int(scene_of_comm.max()) + 1 if len(scene_of_comm) else 1

    # Build scene-to-scene graph for macro-domains (~15) cleanly nested over scenes
    su = scene_of_comm[cu]
    sv = scene_of_comm[cv]
    s_valid = su != sv
    if np.any(s_valid) and n_scenes > target_domains:
        s_u = np.minimum(su[s_valid], sv[s_valid]).astype(np.int64)
        s_v = np.maximum(su[s_valid], sv[s_valid]).astype(np.int64)
        s_key = s_u * n_scenes + s_v
        s_uk, s_inv = np.unique(s_key, return_inverse=True)
        s_ew = np.bincount(s_inv, weights=ew[s_valid])
        s_cu = (s_uk // n_scenes).astype(np.int64)
        s_cv = (s_uk % n_scenes).astype(np.int64)
        s_str = np.bincount(np.concatenate([s_cu, s_cv]), weights=np.concatenate([s_ew, s_ew]), minlength=n_scenes)
        s_norm = s_ew / np.maximum(np.sqrt(s_str[s_cu] * s_str[s_cv]), 1e-9)
        sg = ig.Graph(n=n_scenes, edges=list(zip(s_cu.tolist(), s_cv.tolist())), directed=False)
        sg.es["weight"] = s_norm.tolist()
        domain_of_scene = tune_rb(sg, target_domains, lo=0.05, hi=10.0)
        domain_of_comm = domain_of_scene[scene_of_comm]
    else:
        domain_of_comm = tune_rb(cg, target_domains, lo=0.05, hi=5.0)

    return scene_of_comm, domain_of_comm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/Users/joshlacalamito/catbird-lake-local/graph-build/out")
    ap.add_argument("--target-size", type=float, default=100)
    ap.add_argument("--size-objective", choices=["community", "producer"], default="community")
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--max-communities", type=int, default=30000)
    ap.add_argument("--cluster-top-k", type=int, default=30)
    ap.add_argument("--cluster-min-cos", type=float, default=0.035)
    ap.add_argument("--target-scenes", type=int, default=150)
    ap.add_argument("--target-domains", type=int, default=15)
    ap.add_argument("--resolution", type=float, default=0, help="CPM resolution (0 to bisect)")
    ap.add_argument("--search-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    d = pathlib.Path(args.dir)

    dids, followers, raw = load(d)
    n = len(dids)
    backbone = prune_for_backbone(raw, top_k=args.cluster_top_k, min_cos=args.cluster_min_cos)
    if not len(backbone):
        raise SystemExit("No similarity edges meet the backbone threshold; clustering failed")
    b_src, b_dst, b_w = symmetric_edges(backbone)
    g = ig.Graph(n=n, edges=list(zip(b_src.tolist(), b_dst.tolist())), directed=False)
    g.es["weight"] = b_w.tolist()

    if args.resolution is not None and args.resolution > 0:
        res = args.resolution
    else:
        lo, hi = 0.0001, 0.08
        res = (lo * hi) ** 0.5
        best_res, best_error = res, float("inf")
        for step in range(args.search_steps):
            res = (lo * hi) ** 0.5
            trial = run_leiden(g, res, args.seed, iterations=1)
            objective = community_median_size if args.size_objective == "community" else median_size
            ms = objective(trial, args.min_size)
            error = abs(ms - args.target_size)
            print(f"resolution trial {step + 1}: resolution={res:.8f}, {args.size_objective}_median={ms}", flush=True)
            if error < best_error:
                best_res, best_error = res, error
            if abs(ms - args.target_size) / args.target_size < 0.10:
                break
            if ms > args.target_size:
                lo = res
            else:
                hi = res
        res = best_res
    memb = run_leiden(g, res, args.seed, iterations=2)

    sizes = np.bincount(memb)
    order = np.argsort(-sizes)
    valid_ids = [cid for cid in order if sizes[cid] >= args.min_size][: args.max_communities]
    remap = -np.ones(len(sizes), dtype=np.int64)
    remap[valid_ids] = np.arange(len(valid_ids))
    comm = remap[memb]
    n_comm = len(valid_ids)

    # Top-100 symmetric edges per producer for soft membership shares and community hierarchy.
    soft_edges = prune_for_backbone(raw, top_k=100, min_cos=0.025)
    src, dst, w = symmetric_edges(soft_edges)
    scene_of_comm, domain_of_comm = build_hierarchy(
        n_comm, comm, src, dst, w, args.target_scenes, args.target_domains, args.seed
    )

    # Membership scores: share of each producer's similarity weight going to
    # members of each community (both edge directions).
    a = np.concatenate([src, dst])
    b = np.concatenate([dst, src])
    ww = np.concatenate([w, w])
    totals = np.bincount(a, weights=ww, minlength=n)
    cb = comm[b]
    keep = cb >= 0
    key = a[keep] * max(n_comm, 1) + cb[keep]
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=ww[keep])
    kp = (uk // max(n_comm, 1)).astype(np.int64)
    kc = (uk % max(n_comm, 1)).astype(np.int64)
    share = sums / np.maximum(totals[kp], 1e-12)
    # Own community gets at least 0.5 so every assigned producer keeps it.
    own = comm[kp] == kc
    share = np.where(own, np.maximum(share, 0.5), share)
    # Producers assigned to a community but with no weighted edges into it.
    have_own = np.zeros(n, dtype=bool)
    have_own[kp[own]] = True
    missing = np.where((comm >= 0) & ~have_own)[0]
    kp = np.concatenate([kp, missing])
    kc = np.concatenate([kc, comm[missing]])
    share = np.concatenate([share, np.full(len(missing), 0.5)])
    own = np.concatenate([own, np.ones(len(missing), dtype=bool)])
    # Top 3 per producer, keeping shares >= 0.1 (or the own community).
    sel = (share >= 0.1) | own
    kp, kc, share = kp[sel], kc[sel], share[sel]
    order = np.lexsort((-share, kp))
    kp, kc, share = kp[order], kc[order], share[order]
    first = np.r_[0, np.flatnonzero(np.diff(kp)) + 1]
    rank = np.arange(len(kp)) - np.repeat(first, np.diff(np.r_[first, len(kp)]))
    top = rank < 3
    top_kc = kc[top].astype(np.uint32)
    kf = pa.table({
        "pid": kp[top].astype(np.uint32),
        "did": pa.array([dids[p] for p in kp[top]]),
        "community": top_kc,
        "scene_id": scene_of_comm[top_kc].astype(np.uint16),
        "domain_id": domain_of_comm[top_kc].astype(np.uint8),
        "score": np.round(share[top], 4).astype(np.float32),
        "rank": rank[top].astype(np.uint8),
    })
    pq.write_table(kf, d / "known_for.parquet", compression="zstd")

    members: dict[int, list[int]] = {}
    for p in np.flatnonzero(comm >= 0):
        members.setdefault(int(comm[p]), []).append(int(p))
    c_rows = {"community": [], "scene_id": [], "domain_id": [], "size": [], "top_dids": [], "top_followers": []}
    for c in sorted(members):
        ps = sorted(members[c], key=lambda p: -followers[p])
        c_rows["community"].append(c)
        c_rows["scene_id"].append(int(scene_of_comm[c]))
        c_rows["domain_id"].append(int(domain_of_comm[c]))
        c_rows["size"].append(len(ps))
        c_rows["top_dids"].append([dids[p] for p in ps[:20]])
        c_rows["top_followers"].append([int(followers[p]) for p in ps[:20]])
    pq.write_table(pa.table(c_rows), d / "communities.parquet", compression="zstd")

    sz = [len(v) for v in members.values()]
    stats = {
        "producers": n,
        "backbone_edges_undirected": int(len(b_w)),
        "sim_edges_undirected": int(len(w)),
        "resolution": res,
        "algorithm": "Leiden CPM with similarity-share overlapping memberships; not twitter/sbf Metropolis-Hastings",
        "seed": args.seed,
        "target_size": args.target_size,
        "size_objective": args.size_objective,
        "max_communities": args.max_communities,
        "communities_before_cap": int((sizes >= args.min_size).sum()),
        "communities_discarded_by_cap": max(0, int((sizes >= args.min_size).sum()) - args.max_communities),
        "size_producer_weighted_median": median_size(memb, args.min_size),
        "communities": len(members),
        "scenes": int(len(np.unique(scene_of_comm))) if len(scene_of_comm) else 0,
        "domains": int(len(np.unique(domain_of_comm))) if len(domain_of_comm) else 0,
        "assigned_producers": int((comm >= 0).sum()),
        "size_median": statistics.median(sz) if sz else 0,
        "size_p90": float(np.quantile(sz, 0.9)) if sz else 0,
        "size_max": max(sz) if sz else 0,
        "known_for_rows": int(kf.num_rows),
        "known_for_mb": round((d / "known_for.parquet").stat().st_size / 1e6, 3),
    }
    (d / "cluster_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
