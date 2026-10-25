"""Evaluate the pre-registered producer eligibility gate (DECISIONS 2026-10-25) on a partition.

  producer_quality.py BUILD AUDIT_DIR QUALITY_DIR MEMBERSHIP_NPY FLAGS_NPZ

AUDIT_DIR: follower_degree.npz, producer_stats.npz, pack_counts.npy (they define FB and CLEAN only).
QUALITY_DIR: producer_features.npz from producer_features.py. Writes gate.json, producer_quality.parquet,
and sample_dids.tsv (40 removed + 40 kept unflagged producers for the live check).
"""
import json, sys
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

BUILD, AUDIT, Q, MEMB, FLAGS = sys.argv[1:6]
fd = np.load(f"{AUDIT}/follower_degree.npz"); st = np.load(f"{AUDIT}/producer_stats.npz")
followers = np.maximum(st["followers"], 1)
mass = fd["mass3000"] / followers
packs = np.load(f"{AUDIT}/pack_counts.npy")
memb = np.load(MEMB).astype(np.int64)
fl = np.load(FLAGS)
fb_c, ft_c, size_c = fl["follow_back"], fl["follow_train"], fl["size"]
fb, ft = fb_c[memb], ft_c[memb]
clean = (packs >= 1) & (mass < 0.4) & ~fb & ~ft
other = ~fb & ~ft
f = np.load(f"{Q}/producer_features.npz")
light = f["light_engagers"]
mutual = st["recip_out"] / followers

rules = {}
for k in (1, 2, 5, 10):
    rules[f"G{k}"] = light >= k
    rules[f"G{k}+R"] = (light >= k) & (mutual < 0.6)

res = {}
for name, ok in rules.items():
    drop = ~ok
    res[name] = {"removed_fb": float(drop[fb].mean()), "removed_clean": float(drop[clean].mean()),
                 "removed_ft": float(drop[ft].mean()), "removed_other": float(drop[other].mean()),
                 "removed_all": int(drop.sum()), "eligible_all": int(ok.sum())}
    res[name]["pass"] = res[name]["removed_fb"] >= 0.70 and res[name]["removed_clean"] <= 0.10
passing = [n for n in rules if res[n]["pass"]]
chosen = min(passing, key=lambda n: res[n]["removed_clean"]) if passing else None
# With no passing rule, the live check still runs on the rule nearest the gate, marked diagnostic.
probe = chosen or min(rules, key=lambda n: max(0.70 - res[n]["removed_fb"], 0) + max(res[n]["removed_clean"] - 0.10, 0))

print(f"FB={fb.sum():,} CLEAN={clean.sum():,} FT={ft.sum():,} producers={len(memb):,}")
print(f"{'rule':<7} {'rm FB':>7} {'rm CLEAN':>9} {'rm FT':>7} {'rm other':>9} {'removed':>10} pass")
for n, r in res.items():
    print(f"{n:<7} {r['removed_fb']:>7.1%} {r['removed_clean']:>9.1%} {r['removed_ft']:>7.1%} {r['removed_other']:>9.1%} "
          f"{r['removed_all']:>10,} {r['pass']}")
print(f"chosen={chosen} probe={probe}")

drop = ~rules[probe]
C = len(size_c)
rm_share = np.bincount(memb, weights=drop, minlength=C) / np.maximum(size_c, 1)
cand = np.flatnonzero(~fb_c & ~ft_c & (rm_share >= 0.5) & (size_c >= 8))
largest = cand[np.argsort(-size_c[cand])][:20]
rng = np.random.default_rng(7)
dids = [l.split("\t")[1] for l in open(f"{BUILD}/producers.tsv")]
pick_rm = rng.choice(np.flatnonzero(drop & other), 40, replace=False)
pick_kp = rng.choice(np.flatnonzero(~drop & other), 40, replace=False)
with open(f"{Q}/sample_dids.tsv", "w") as out:
    for group, picks in (("removed", pick_rm), ("kept", pick_kp)):
        for p in picks:
            out.write(f"{group}\t{p}\t{dids[p]}\n")

json.dump({"preregistration": "docs/DECISIONS.md 2026-10-25 pre-registered producer-quality gate",
           "populations": {"FB": int(fb.sum()), "CLEAN": int(clean.sum()), "FT": int(ft.sum()), "producers": len(memb)},
           "rules": res, "chosen": chosen, "probe_rule_for_checks": probe,
           "unflagged_communities_mostly_removed": int(len(cand)),
           "largest_unflagged_mostly_removed": [{"community": int(c), "size": int(size_c[c]), "removed": float(rm_share[c])}
                                                for c in largest]},
          open(f"{Q}/gate.json", "w"), indent=1)
print("largest unflagged communities the probe rule mostly removes:", ",".join(str(c) for c in largest))

cols = {"pid": np.arange(len(memb)), "did": dids, "community": memb, "followers": st["followers"],
        "mass3k_share": mass, "mutual_share": mutual, "packs": packs,
        **{k: f[k] for k in f.files}, **{f"eligible_{n}": ok for n, ok in rules.items()}}
pq.write_table(pa.table(cols), f"{Q}/producer_quality.parquet", compression="zstd")
