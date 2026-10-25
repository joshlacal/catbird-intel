import sys, numpy as np
OUT, MEMB = sys.argv[1], sys.argv[2]
fd = np.load(f"{OUT}/follower_degree.npz"); st = np.load(f"{OUT}/producer_stats.npz")
fol = st["followers"].astype(float)
f3, f10 = fd["mass3000"] / np.maximum(fol, 1), fd["mass10000"] / np.maximum(fol, 1)
eff = fd["invdeg"]  # audience size if each follower spread 1 unit over its follows
memb = np.load(MEMB).astype(np.int64); C = memb.max() + 1
order = np.argsort(memb, kind="stable"); b = np.searchsorted(memb[order], np.arange(C + 1))
print(f"global medians: mass3000_share={np.median(f3):.3f} mass10000_share={np.median(f10):.3f}")
rows = []
for c in range(C):
    m = order[b[c]:b[c+1]]
    if len(m): rows.append((c, len(m), np.median(fol[m]), np.median(f3[m]), np.median(f10[m]), np.median(eff[m])))
np.save(f"{OUT}/comm_mass.npy", np.array(rows))
rows.sort(key=lambda r: -r[1])
print("comm\tsize\tmed_fol\tmass3k\tmass10k\tmed_invdeg_audience")
for r in rows[:40]: print(f"{r[0]}\t{r[1]}\t{r[2]:.0f}\t{r[3]:.3f}\t{r[4]:.3f}\t{r[5]:.1f}")
a = np.array(rows)
for thr in (0.5, 0.6, 0.7, 0.8):
    sel = a[:, 3] >= thr
    print(f"communities with median mass3k share >= {thr}: {sel.sum()} holding {a[sel,1].sum():,.0f} producers")
