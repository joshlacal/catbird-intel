#!/usr/bin/env python3
"""S5 serving universe for the cand1 partition (whole-network plan §3.1 S5 (i), §3.3; DECISIONS 2026-10-28).

Every DID that could appear on a cand1 card, in a bundle or in an offered starter pack:
  displayable communities = >=25 base-rule candidates (primary KnownFor community, raw Leiden label
  not follow-back/follow-train flagged; hide-from-recs kept, amendment A1);
  per displayable community: card top_dids, top-50 by light_engagers/sqrt(followers), top-25 by
  curatelists; creators and all members of <=2 member-made packs (creator is a base-rule candidate
  of the community, >=5 members, ranked by in-community member share, then size, then URI).
Feed creators are not included: O7 (community-liked feeds) is an S9 output.

usage: serving_universe.py CAND1_DIR OUT_DIR   (writes serving_universe.tsv, card_packs.tsv)
"""
import collections, pickle, sys

import numpy as np
import pyarrow.parquet as pq

src, out = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/"
kf = pq.read_table(src + "known_for.parquet", columns=["pid", "did", "community", "rank"]).to_pandas()
prim = kf[kf["rank"] == 0].drop(columns="rank")
m = np.load(src + "membership.npy")
z = np.load(src + "community_flags.npz")
prim = prim.assign(flagged=(z["follow_back"] | z["follow_train"])[m][prim.pid.values])
q = pq.read_table(src + "audit-quality/producer_quality.parquet",
                  columns=["pid", "followers", "light_engagers", "curatelists"]).to_pandas().set_index("pid")
cand = prim[~prim.flagged].join(q, on="pid")
sz = cand.groupby("community").size()
disp = set(sz[sz >= 25].index)
c2 = cand[cand.community.isin(disp)].copy()
c2["r1"] = c2.light_engagers / np.sqrt(c2.followers.clip(lower=1))
reasons = collections.defaultdict(set)
for d in c2.sort_values(["r1", "followers", "did"], ascending=[False, False, True]).groupby("community").head(50).did:
    reasons[d].add("top50_lightengagers_per_sqrtfollowers")
for d in c2.sort_values(["curatelists", "followers", "did"], ascending=[False, False, True]).groupby("community").head(25).did:
    reasons[d].add("top25_curatelists")
cm = pq.read_table(src + "communities.parquet", columns=["community", "top_dids"]).to_pandas()
for c, lst in zip(cm.community, cm.top_dids):
    if c in disp:
        for d in lst:
            reasons[d].add("card_top_dids")
did2c = dict(zip(c2.did, c2.community))
best = collections.defaultdict(list)
# packs.pkl is our own audit artifact (communities-cand1-hubdemote/audit-mass-followers), not untrusted input.
for uri, mem, _ in pickle.load(open(src + "audit-mass-followers/packs.pkl", "rb")):
    cr = uri[5:].split("/")[0]
    c = did2c.get(cr)
    if c is None or len(mem) < 5:
        continue
    best[c].append((-sum(did2c.get(x) == c for x in mem) / len(mem), -len(mem), uri, mem, cr))
chosen = []
for c, lst in best.items():
    lst.sort(key=lambda t: t[:3])
    for share, n, uri, mem, cr in lst[:2]:
        chosen.append((c, uri, abs(share), -n))
        reasons[cr].add("pack_creator")
        for d in mem:
            reasons[d].add("pack_member")
with open(out + "serving_universe.tsv", "w") as f:
    for d in sorted(reasons):
        f.write(f"{d}\t{','.join(sorted(reasons[d]))}\n")
with open(out + "card_packs.tsv", "w") as f:
    for c, u, s, n in sorted(chosen):
        f.write(f"{c}\t{u}\t{s:.4f}\t{n}\n")
print(len(disp), "displayable communities;", len(reasons), "DIDs;", len(chosen), "packs")
