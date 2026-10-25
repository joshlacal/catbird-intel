"""How much of each producer's audience is mass-followers (users following >= T producers)."""
import sys, time
import numpy as np

BUILD, OUT = sys.argv[1], sys.argv[2]
t0 = time.time()
e = np.memmap(f"{BUILD}/edges.bin", dtype=np.uint32, mode="r").reshape(-1, 2)
n = e.shape[0]; CH = 200_000_000
P = int(open(f"{BUILD}/stats.tsv").read().split("\n")[1].split("\t")[4])
deg = None
for s in range(0, n, CH):
    b = np.bincount(np.asarray(e[s:s + CH, 0]))
    deg = b if deg is None else np.pad(deg, (0, max(0, len(b) - len(deg)))) + np.pad(b, (0, max(0, len(deg) - len(b))))
print(f"users={len(deg):,} with_follows={(deg>0).sum():,} median_deg={np.median(deg[deg>0]):.0f} "
      f"p99={np.percentile(deg[deg>0],99):.0f} {time.time()-t0:.0f}s", flush=True)
TS = [1000, 3000, 10000]
cnt = {t: np.zeros(P) for t in TS}
wsum = np.zeros(P)  # sum of 1/deg: audience weight under inverse-degree damping
for s in range(0, n, CH):
    blk = np.asarray(e[s:s + CH]); d = deg[blk[:, 0]]; p = blk[:, 1]
    for t in TS:
        cnt[t] += np.bincount(p, weights=d >= t, minlength=P)
    wsum += np.bincount(p, weights=1.0 / d, minlength=P)
np.savez(f"{OUT}/follower_degree.npz", deg=deg, **{f"mass{t}": cnt[t] for t in TS}, invdeg=wsum)
users_deg = deg
for t in TS:
    print(f"users following >= {t} producers: {(users_deg>=t).sum():,} "
          f"carrying {users_deg[users_deg>=t].sum()/users_deg.sum():.1%} of all edges")
print(f"done {time.time()-t0:.0f}s")
