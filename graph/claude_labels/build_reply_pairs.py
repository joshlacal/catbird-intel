"""Stratified reply/quote pair sample from the local netcup hot-export days, for hostile-stance labels.

  build_reply_pairs.py EXCLUDE_DIDS_TXT OUT_PARQUET DAY_DIR [DAY_DIR ...]
      [--skip EARLIER_PAIRS_PARQUET ...] [--quota block=10000,reply=20000,quote=20000] [--seed 20261027] [--prefix p]

--skip drops pairs already sampled in an earlier file and counts them toward the per-author caps,
so a top-up sample (e.g. --quota block=15000,reply=0,quote=30000 --seed 20261028 --prefix q) adds
only new pairs.

A pair is (parent post, child post) where the child replies to or quotes the parent, both
posts are live (not deleted by the export's end), both are in the window, the authors
differ, and neither author is in EXCLUDE_DIDS_TXT (hide-from-recs union, user-intents
opt-outs) or set hide_from_algorithmic in the window. `block_after` marks pairs where the
parent's author blocked the child's author within 48 h after the child was created: the
structural hostility signal the labels are meant to measure the recall of (plan S15b).
Sample: all block_after pairs up to 10k, then 20k replies and 20k quotes at random, with at
most 5 pairs per child author and 20 per parent author. Seed 20261027.
"""
import argparse
import random

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("excl")
ap.add_argument("out")
ap.add_argument("days", nargs="+")
ap.add_argument("--skip", action="append", default=[])
ap.add_argument("--quota", default="block=10000,reply=20000,quote=20000")
ap.add_argument("--seed", type=int, default=20261027)
ap.add_argument("--prefix", default="p")
A = ap.parse_args()
EXCL, OUT, DAYS = A.excl, A.out, A.days
rng = random.Random(A.seed)
excl = {l.strip() for l in open(EXCL) if l.strip()}
for d in DAYS:
    v = pq.read_table(f"{d}/visibility.parquet", columns=["did", "hide_from_algorithmic"])
    excl.update(v.filter(pc.equal(v["hide_from_algorithmic"], 1))["did"].to_pylist())

deleted, posts = set(), {}
COLS = ["did", "rkey", "rev", "is_deleted", "created_at", "text", "langs", "reply_parent", "quote_uri"]
for d in DAYS:
    dl = pq.read_table(f"{d}/deletions.parquet", columns=["collection", "did", "rkey"])
    dl = dl.filter(pc.equal(dl["collection"], "app.bsky.feed.post"))
    deleted.update(zip(dl["did"].to_pylist(), dl["rkey"].to_pylist()))
    t = pq.read_table(f"{d}/posts.parquet", columns=COLS)
    for did, rkey, rev, isdel, ca, text, langs, rp, qu in zip(*(t[c].to_pylist() for c in COLS)):
        k = (did, rkey)
        if k not in posts or rev > posts[k][0]:
            posts[k] = (rev, isdel, ca, text, langs, rp, qu)
    print(d, len(posts), flush=True)

live = {f"at://{d}/app.bsky.feed.post/{r}": (d, v) for (d, r), v in posts.items()
        if not v[1] and (d, r) not in deleted and d not in excl}
blocks = {}
for d in DAYS:
    b = pq.read_table(f"{d}/blocks.parquet", columns=["did", "subject", "is_deleted", "indexed_at"])
    for blocker, subj, isdel, at in zip(*(b[c].to_pylist() for c in ["did", "subject", "is_deleted", "indexed_at"])):
        if not isdel:
            blocks.setdefault((blocker, subj), at)

cands = []
for curi, (cdid, (_, _, cca, ctext, clangs, rp, qu)) in live.items():
    for kind, puri in (("reply", rp), ("quote", qu)):
        if not puri or puri not in live or len(ctext.strip()) < 3:
            continue
        pdid, pv = live[puri]
        if pdid == cdid or not pv[3].strip():
            continue
        bat = blocks.get((pdid, cdid))
        after = bat is not None and cca is not None and 0 <= (bat - cca).total_seconds() <= 48 * 3600
        cands.append((kind, after, puri, curi, pdid, cdid, pv[3], ctext, ",".join(clangs), cca))
print("candidates", len(cands), "block_after", sum(c[1] for c in cands), flush=True)

rng.shuffle(cands)
per_child, per_parent, picked = {}, {}, []
seen = set()
for path in A.skip:
    old = pq.read_table(path, columns=["kind", "child_uri", "parent_did", "child_did"]).to_pylist()
    for o in old:
        seen.add((o["kind"], o["child_uri"]))
        per_child[o["child_did"]] = per_child.get(o["child_did"], 0) + 1
        per_parent[o["parent_did"]] = per_parent.get(o["parent_did"], 0) + 1
quota = {k: int(v) for k, v in (kv.split("=") for kv in A.quota.split(","))}
for stratum, keep in (("block", lambda c: c[1]), ("reply", lambda c: c[0] == "reply" and not c[1]),
                      ("quote", lambda c: c[0] == "quote" and not c[1])):
    n = 0
    for c in cands:
        if n >= quota.get(stratum, 0) or not keep(c) or (c[0], c[3]) in seen:
            continue
        if per_child.get(c[5], 0) >= 5 or per_parent.get(c[4], 0) >= 20:
            continue
        per_child[c[5]] = per_child.get(c[5], 0) + 1
        per_parent[c[4]] = per_parent.get(c[4], 0) + 1
        picked.append((stratum,) + c)
        n += 1
    print(stratum, n, flush=True)

names = ["stratum", "kind", "block_after", "parent_uri", "child_uri", "parent_did", "child_did",
         "parent_text", "child_text", "child_langs", "child_created_at"]
tbl = pa.table({k: [p[i] for p in picked] for i, k in enumerate(names)})
tbl = tbl.append_column("pair_id", pa.array([f"{A.prefix}{i:05d}" for i in range(len(picked))]))
pq.write_table(tbl, OUT)
print(f"wrote {len(picked)} pairs -> {OUT}")
