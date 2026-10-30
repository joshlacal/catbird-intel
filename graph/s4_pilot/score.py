"""S4 G-pilot gate and bake-off on held-out curated lists (S1 rules file §6, §9.5, B1, R12, R18).

  score.py PILOT_DIR OUT_JSON

Every rule below was written down before any list score was computed.

Lists: `curatelists.tsv` (sha 11099805…), filtered exactly as `audit/list_purity.py` (identical member
sets with >=3 members deduped; <=3 lists per owner with default_rng(7)), then <=500 members (max of the
list's item total and its distinct members), then >=3 members in the arm's universe. Families by
ClickHouse cityHash64(owner) % 8: {0,1,4,5} = A, 6 = B1b. B1b decides; A carries diagnostics only.

kNN form of macro top-1 (R12): per list, the share of its in-universe members whose nearest neighbour
(cosine, itself excluded, neighbours restricted to the universe) is another member of the list,
averaged over lists. Same-size null: each list replaced by a same-size random set from the universe
(default_rng(20261028), lists in file order). Bootstrap: percentile, 2,000 resamples over lists,
default_rng(20261028), 98.75% two-sided for B1b.

Arms (fixed before scoring):
  gate     EG2, producers with >=1 embedded post; vector = normalise(normalise(mean of post vectors)
           + 0.25 * bio vector) (bio term only where a bio was embedded; design-content S1 step 2 prior).
  bake_eg2 / bake_harrier  posts only, from the identical 500k bake-off subsample; universe = producers
           with >=1 bake-off post (identical for both); vector = normalise(mean of post vectors).
G-pilot passes iff mean(gate - null) > 0 and the B1b 98.75% CI lower bound > 0.
Bake-off: harrier replaces EG2 iff mean(harrier - eg2) >= 0.03 and the B1b 98.75% CI lower bound > 0.
Diagnostics on A lists only: gate, posts-only EG2, bios-only EG2 (all producers with a bio), bake-off.
"""
import collections
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

LAKE = pathlib.Path.home() / "catbird-lake-local"
LISTS = LAKE / "communities-cand1-hubdemote/audit-quality/curatelists.tsv"
LISTS_SHA = "11099805e1ec25a78ee2b0cde2450b3074a3d865daf053863fe39d6eee607113"
A_BUCKETS, B1B_BUCKET = {0, 1, 4, 5}, 6
SEED, N_BOOT, BIO_W, BAR_POINTS = 20261028, 2000, 0.25, 0.03
LEVELS = {"B1b": 0.9875, "A": 0.95}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_lists(split_tsv):
    assert sha256(LISTS) == LISTS_SHA
    rng = np.random.default_rng(7)
    rows = []
    for line in open(LISTS):
        uri, owner, name, total, members = line.rstrip("\n").split("\t")
        rows.append((uri, owner, name, sorted(set(m for m in members.split(",") if m)), int(total)))
    rows.sort()
    seen, by_owner = set(), collections.defaultdict(list)
    for r in rows:
        key = tuple(r[3])
        if len(key) >= 3 and key not in seen:
            seen.add(key)
            by_owner[r[1]].append(r)
    lists = []
    for owner, rs in by_owner.items():
        lists += [rs[i] for i in sorted(rng.choice(len(rs), min(3, len(rs)), replace=False))]
    bucket = dict(l.rstrip("\n").split("\t") for l in open(split_tsv))
    fam = {"A": [], "B1b": []}
    for uri, owner, _n, members, total in lists:
        if max(total, len(members)) > 500 or owner not in bucket:
            continue
        b = int(bucket[owner])
        if b == B1B_BUCKET:
            fam["B1b"].append((uri, members))
        elif b in A_BUCKETS:
            fam["A"].append((uri, members))
    return fam, len(rows), len(lists)


def normalise(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pool(pids, vec):
    """Mean of each producer's vectors (rows grouped by pid), normalised. Returns (unique pids, vectors)."""
    o = np.argsort(pids, kind="stable")
    pids, vec = pids[o], vec[o].astype(np.float32)
    u, start = np.unique(pids, return_index=True)
    return u, normalise(np.add.reduceat(vec, start, axis=0))


def nearest(x, chunk=1024):
    """Index of each row's nearest other row by cosine (rows are unit vectors), on MPS in fp32."""
    dev = torch.device("mps")
    X = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
    out = np.empty(len(x), dtype=np.int64)
    for s in range(0, len(x), chunk):
        sims = X[s:s + chunk] @ X.T
        idx = torch.arange(s, min(s + chunk, len(x)), device=dev)
        sims[torch.arange(len(idx), device=dev), idx] = -float("inf")
        out[s:s + chunk] = sims.argmax(dim=1).cpu().numpy()
    return out


def list_hits(groups, nn):
    """Per group (array of universe indices), share of members whose NN is another member."""
    res = np.empty(len(groups))
    for i, g in enumerate(groups):
        s = set(g.tolist())
        res[i] = np.mean([nn[m] in s for m in g])
    return res


def boot_ci(d, level):
    rng = np.random.default_rng(SEED)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def groups_in(fam_lists, index):
    keep, groups = [], []
    for uri, members in fam_lists:
        g = np.array([index[d] for d in members if d in index], dtype=np.int64)
        if len(g) >= 3:
            keep.append(uri)
            groups.append(g)
    return keep, groups


def null_hits(groups, n_universe, nn):
    rng = np.random.default_rng(SEED)
    rand = [rng.choice(n_universe, len(g), replace=False) for g in groups]
    return list_hits(rand, nn)


def vs_null(fam, lists, index, nn, n):
    _uris, groups = groups_in(lists, index)
    obs = list_hits(groups, nn)
    nul = null_hits(groups, n, nn)
    d = obs - nul
    lo, hi = boot_ci(d, LEVELS[fam])
    return dict(lists=len(groups), members=int(sum(len(g) for g in groups)), macro_top1=float(obs.mean()),
                null_macro_top1=float(nul.mean()), diff=float(d.mean()), ci_level=LEVELS[fam], ci=[lo, hi],
                lift=float(obs.mean() / nul.mean()) if nul.mean() > 0 else None)


def paired(fam, lists, index, nn_a, nn_b):
    _uris, groups = groups_in(lists, index)
    a, b = list_hits(groups, nn_a), list_hits(groups, nn_b)
    d = b - a
    lo, hi = boot_ci(d, LEVELS[fam])
    return dict(lists=len(groups), members=int(sum(len(g) for g in groups)), eg2_macro_top1=float(a.mean()),
                harrier_macro_top1=float(b.mean()), diff_harrier_minus_eg2=float(d.mean()), ci_level=LEVELS[fam],
                ci=[lo, hi])


def main(pilot, out_json):
    pilot = pathlib.Path(pilot)
    t0 = time.time()
    fam, n_rows, n_sampled = load_lists(pilot / "inputs/list_owner_split.tsv")
    corpus = pq.read_table(pilot / "inputs/posts_corpus.parquet", columns=["post_idx", "pid", "did", "bakeoff"]).to_pandas()
    did_of = dict(zip(corpus.pid, corpus.did))
    bios = pq.read_table(pilot / "inputs/bios_corpus.parquet", columns=["pid", "did"]).to_pandas()
    did_of.update(zip(bios.pid, bios.did))

    def vecs(name):
        d = pilot / "vectors" / name
        return np.load(d / "ids.npy"), np.load(d / "vec256_fp16.npy")

    pid_post = corpus.set_index("post_idx").pid
    ids, v = vecs("eg2_posts")
    assert np.array_equal(ids, corpus.post_idx.to_numpy())
    post_pids = pid_post.loc[ids].to_numpy()
    u_posts, p_posts = pool(post_pids, v)
    bid, bv = vecs("eg2_bios")
    bio_row = {int(p): i for i, p in enumerate(bid)}
    bv = normalise(bv.astype(np.float32))
    gate = p_posts.copy()
    has_bio = np.array([int(p) in bio_row for p in u_posts])
    gate[has_bio] += BIO_W * bv[[bio_row[int(p)] for p in u_posts[has_bio]]]
    gate = normalise(gate)

    bmask = corpus.bakeoff.to_numpy()
    hids, hv = vecs("harrier_bakeoff")
    assert np.array_equal(hids, corpus.post_idx.to_numpy()[bmask])
    u_bake, p_bake_eg2 = pool(post_pids[bmask], v[bmask])
    u_bake_h, p_bake_h = pool(pid_post.loc[hids].to_numpy(), hv)
    assert np.array_equal(u_bake, u_bake_h)

    res = {"lists": {"rows_read": n_rows, "after_list_purity_dedupe_and_owner_cap": n_sampled,
                     "A_le500": len(fam["A"]), "B1b_le500": len(fam["B1b"])}}
    idx_posts = {did_of[int(p)]: i for i, p in enumerate(u_posts)}
    idx_bake = {did_of[int(p)]: i for i, p in enumerate(u_bake)}
    nn_gate = nearest(gate)
    nn_be, nn_bh = nearest(p_bake_eg2), nearest(p_bake_h)
    res["universe"] = {"gate_producers": len(u_posts), "gate_with_bio": int(has_bio.sum()),
                       "bakeoff_producers": len(u_bake), "bios_only_producers": len(bid)}
    res["G_pilot_B1b"] = vs_null("B1b", fam["B1b"], idx_posts, nn_gate, len(u_posts))
    g = res["G_pilot_B1b"]
    g["pass"] = bool(g["diff"] > 0 and g["ci"][0] > 0)
    res["bakeoff_B1b"] = paired("B1b", fam["B1b"], idx_bake, nn_be, nn_bh)
    b = res["bakeoff_B1b"]
    b["harrier_replaces_eg2"] = bool(b["diff_harrier_minus_eg2"] >= BAR_POINTS and b["ci"][0] > 0)

    diag = {"gate_A": vs_null("A", fam["A"], idx_posts, nn_gate, len(u_posts)),
            "posts_only_A": vs_null("A", fam["A"], idx_posts, nearest(p_posts), len(u_posts)),
            "bakeoff_A": paired("A", fam["A"], idx_bake, nn_be, nn_bh)}
    idx_bios = {did_of[int(p)]: i for i, p in enumerate(bid)}
    diag["bios_only_A"] = vs_null("A", fam["A"], idx_bios, nearest(bv, chunk=256), len(bid))
    res["diagnostics_A_lists_not_gating"] = diag
    res["seconds"] = round(time.time() - t0, 1)
    res["inputs_sha256"] = {str(p): sha256(p) for p in [LISTS, pilot / "inputs/list_owner_split.tsv",
                                                          pilot / "inputs/posts_corpus.parquet", pilot / "inputs/bios_corpus.parquet",
                                                          pilot / "vectors/eg2_posts/vec256_fp16.npy",
                                                          pilot / "vectors/eg2_bios/vec256_fp16.npy",
                                                          pilot / "vectors/harrier_bakeoff/vec256_fp16.npy"]}
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
