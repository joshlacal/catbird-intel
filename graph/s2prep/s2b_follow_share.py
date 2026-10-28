#!/usr/bin/env python3
"""S2b (3): share of sampled likes and reposts whose liker follows the author (expected 0.597 ± 0.01).

Same method as the 2026-10-25 measurement (ground-semantics §3.3, follow_check.py): for every
sampled (liker uid, author pid) with both ends mapped, binary-search the liker's range in
graph-build-v3/out-full-20261024/edges.bin (RowBinary uid UInt32, pid UInt32, grouped by uid but
not sorted by pid within a uid) and scan it for the author.

    s2b_follow_share.py SAMPLE_TSV EDGES_BIN OUT_JSON
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

NA = 4294967295
BUCKETS = [("<1k", 0, 1000), ("1k-10k", 1000, 10000), ("10k-100k", 10000, 100000), (">=100k", 100000, 10**12)]


def lower_bound(f_col: np.ndarray, vals: np.ndarray) -> np.ndarray:
    n = len(f_col)
    lo = np.zeros(len(vals), dtype=np.int64)
    hi = np.full(len(vals), n, dtype=np.int64)
    for _ in range(34):
        act = lo < hi
        mid = (lo + hi) // 2
        less = f_col[np.minimum(mid, n - 1)].astype(np.int64) < vals
        lo = np.where(act & less, mid + 1, lo)
        hi = np.where(act & ~less, mid, hi)
    return lo


def follows(edges: np.ndarray, fid: np.ndarray, pid: np.ndarray) -> np.ndarray:
    start, end = lower_bound(edges["f"], fid), lower_bound(edges["f"], fid + 1)
    found = np.zeros(len(fid), dtype=bool)
    p_col = edges["p"]
    for i in range(len(fid)):
        if end[i] > start[i]:
            found[i] = bool((p_col[start[i]:end[i]] == pid[i]).any())
    return found


def main(sample: str, edges_path: str, out: str) -> dict:
    t0 = time.time()
    rows = [line.rstrip("\n").split("\t") for line in open(sample)]
    kind = np.array([r[0] == "like" for r in rows])
    via = np.array([int(r[1]) for r in rows])
    fid = np.array([int(r[2]) for r in rows], dtype=np.int64)
    pid = np.array([int(r[3]) for r in rows], dtype=np.int64)
    fol = np.array([int(r[4]) for r in rows], dtype=np.int64)
    both = (fid != NA) & (pid != NA)
    edges = np.memmap(edges_path, dtype=np.dtype([("f", "<u4"), ("p", "<u4")]), mode="r")
    found = follows(edges, fid[both], pid[both])
    k, v, fo = kind[both], via[both], fol[both]
    res = {
        "rows": len(rows),
        "liker_mapped": round(float((fid != NA).mean()), 4),
        "author_is_producer": round(float((pid != NA).mean()), 4),
        "both_mapped": int(both.sum()),
        "both_mapped_share": round(float(both.mean()), 4),
        "follows_n": int(found.sum()),
        "follower_share": round(float(found.mean()), 4),
        "follower_share_unrounded": float(found.mean()),
        "by_author_followers": {name: {"n": int(((fo >= lo) & (fo < hi)).sum()),
                                       "share": round(float(found[(fo >= lo) & (fo < hi)].mean()), 4)}
                                for name, lo, hi in BUCKETS},
        "by_via": {str(x): {"n": int((v == x).sum()), "share": round(float(found[v == x].mean()), 4)} for x in (0, 1)},
        "by_kind": {name: {"n": int((k == flag).sum()), "share": round(float(found[k == flag].mean()), 4)}
                    for name, flag in (("like", True), ("repost", False))},
        "two_by_two": {f"via={x} follows={f}": round(float(((v == x) & (found == f)).sum() / len(found)), 4)
                       for x in (0, 1) for f in (True, False)},
        "edges_rows": int(len(edges)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:4]), indent=1))
