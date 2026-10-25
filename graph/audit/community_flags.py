"""Flag follow-back and follow-train communities for a partition.

  community_flags.py AUDIT_DIR MEMBERSHIP_NPY OUT_NPZ

follow_back:  median member mass3k share >= 0.5 and < 15% of members in any starter pack
follow_train: median member mass3k share >= 0.6 and median member in >= 5 starter packs
"""
import sys
import numpy as np

audit, memb_path, out = sys.argv[1:4]
fd = np.load(f"{audit}/follower_degree.npz"); st = np.load(f"{audit}/producer_stats.npz")
mass = fd["mass3000"] / np.maximum(st["followers"], 1)
packs = np.load(f"{audit}/pack_counts.npy")
memb = np.load(memb_path).astype(np.int64)
ok = memb >= 0
C = memb.max() + 1
order = np.flatnonzero(ok)[np.argsort(memb[ok], kind="stable")]
b = np.searchsorted(memb[order], np.arange(C + 1))
size = np.diff(b)
med_mass = np.zeros(C); in_pack = np.zeros(C); med_packs = np.zeros(C)
for c in range(C):
    m = order[b[c]:b[c + 1]]
    if len(m):
        med_mass[c] = np.median(mass[m]); in_pack[c] = (packs[m] >= 1).mean(); med_packs[c] = np.median(packs[m])
follow_back = (med_mass >= 0.5) & (in_pack < 0.15)
follow_train = (med_mass >= 0.6) & (med_packs >= 5)
np.savez(out, size=size, med_mass3k=med_mass, in_pack=in_pack, median_packs=med_packs,
         follow_back=follow_back, follow_train=follow_train)
n = ok.sum()
print(f"communities={C} assigned={n:,} median_size(>=8)={np.median(size[size>=8]):.0f} kept={(size>=8).sum()}")
for name, f in (("follow_back", follow_back), ("follow_train", follow_train)):
    print(f"{name}: {f.sum()} communities, {size[f].sum():,} producers ({size[f].sum()/n:.1%}); "
          f"in top-50 largest: {f[np.argsort(-size)[:50]].sum()}")
w = np.bincount(memb[ok], weights=mass[ok], minlength=C) / np.maximum(size, 1)
print(f"producers in communities with median mass3k >= 0.6: {size[med_mass>=0.6].sum():,}")
