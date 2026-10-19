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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/Users/joshlacalamito/catbird-lake-local/graph-build/out")
    ap.add_argument("--target-size", type=float, default=100)
    ap.add_argument("--min-size", type=int, default=5)
    ap.add_argument("--resolution", type=float, default=None, help="skip tuning and use this CPM resolution")
    ap.add_argument("--search-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    d = pathlib.Path(args.dir)

    dids, followers, raw = load(d)
    n = len(dids)
    src, dst, w = symmetric_edges(raw)
    g = ig.Graph(n=n, edges=np.column_stack([src, dst]), directed=False)
    g.es["weight"] = w

    if args.resolution is not None:
        res = args.resolution
    else:
        # Bisection on log-resolution with cheap (2-iteration) Leiden runs.
        lo, hi = 1e-4, float(np.quantile(w, 0.99)) if len(w) else 1.0
        res = (lo * hi) ** 0.5
        for _ in range(args.search_steps):
            res = (lo * hi) ** 0.5
            ms = median_size(run_leiden(g, res, args.seed), args.min_size)
            if abs(ms - args.target_size) / args.target_size < 0.15:
                break
            if ms > args.target_size:
                lo = res  # communities too big -> raise resolution
            else:
                hi = res
    memb = run_leiden(g, res, args.seed, iterations=-1)

    sizes = np.bincount(memb)
    valid = sizes >= args.min_size
    # Renumber valid communities densely by size; small ones become -1.
    order = np.argsort(-sizes)
    remap = -np.ones(len(sizes), dtype=np.int64)
    remap[order[valid[order]]] = np.arange(int(valid.sum()))
    comm = remap[memb]
    n_comm = int(valid.sum())

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
    kf = pa.table({
        "pid": kp[top].astype(np.uint32),
        "did": pa.array([dids[p] for p in kp[top]]),
        "community": kc[top].astype(np.uint32),
        "score": np.round(share[top], 4).astype(np.float32),
        "rank": rank[top].astype(np.uint8),
    })
    pq.write_table(kf, d / "known_for.parquet", compression="zstd")

    members: dict[int, list[int]] = {}
    for p in np.flatnonzero(comm >= 0):
        members.setdefault(int(comm[p]), []).append(int(p))
    c_rows = {"community": [], "size": [], "top_dids": [], "top_followers": []}
    for c in sorted(members):
        ps = sorted(members[c], key=lambda p: -followers[p])
        c_rows["community"].append(c)
        c_rows["size"].append(len(ps))
        c_rows["top_dids"].append([dids[p] for p in ps[:20]])
        c_rows["top_followers"].append([int(followers[p]) for p in ps[:20]])
    pq.write_table(pa.table(c_rows), d / "communities.parquet", compression="zstd")

    sz = [len(v) for v in members.values()]
    stats = {
        "producers": n,
        "sim_edges_undirected": int(len(w)),
        "resolution": res,
        "communities": len(members),
        "assigned_producers": int((comm >= 0).sum()),
        "size_median": statistics.median(sz) if sz else 0,
        "size_p90": float(np.quantile(sz, 0.9)) if sz else 0,
        "size_max": max(sz) if sz else 0,
        "known_for_rows": int(kf.num_rows),
    }
    (d / "cluster_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
