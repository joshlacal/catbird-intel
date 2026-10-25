"""Per-community ring signals from producer->producer edges.

Writes OUT/community_ring_stats.tsv and prints largest communities + global baselines.
"""
import sys
import numpy as np

OUT, MEMB = sys.argv[1], sys.argv[2]
st = np.load(f"{OUT}/producer_stats.npz")
fol, out_pp, rec_out, in_pp = st["followers"], st["out_pp"], st["recip_out"], st["in_pp"]
memb = np.load(MEMB).astype(np.int64)
P = len(fol)
rate = np.where(out_pp > 0, rec_out / np.maximum(out_pp, 1), 0.0)
ratio = out_pp / np.maximum(fol, 1)
assigned = memb >= 0
print(f"global producer medians: followers={np.median(fol):.0f} out_pp={np.median(out_pp):.0f} "
      f"followback_rate={np.median(rate[out_pp>0]):.3f} out/in={np.median(ratio):.2f}")

keys = np.load(f"{OUT}/pp_keys.npy", mmap_mode="r")
C = memb.max() + 1
int_e = np.zeros(C); int_r = np.zeros(C); tot_out = np.zeros(C)
CH = 100_000_000
for s in range(0, len(keys), CH):
    k = np.asarray(keys[s:s + CH])
    src = k >> 21; dst = k & ((1 << 21) - 1)
    rev = (dst << 21) | src
    pos = np.searchsorted(keys, rev); pos[pos >= len(keys)] = 0
    r = keys[pos] == rev
    cs, cd = memb[src], memb[dst]
    ok = cs >= 0
    np.add.at(tot_out, cs[ok], 1)
    same = ok & (cs == cd)
    int_e += np.bincount(cs[same], minlength=C)
    int_r += np.bincount(cs[same], weights=r[same], minlength=C)
    print(f"  {s+len(k):,}/{len(keys):,}", flush=True)

rows = []
order = np.argsort(memb, kind="stable")
bounds = np.searchsorted(memb[order], np.arange(C + 1))
for c in range(C):
    m = order[bounds[c]:bounds[c + 1]]
    n = len(m)
    if n == 0:
        continue
    rows.append((c, n, np.median(fol[m]), np.median(out_pp[m]), np.median(rate[m]),
                 np.median(ratio[m]), int_e[c] / max(n * (n - 1), 1),
                 int_e[c] / max(tot_out[c], 1), int_r[c] / max(int_e[c], 1)))
hdr = "community\tsize\tmed_followers\tmed_out_pp\tmed_followback\tmed_out_in\tint_density\tint_share\tint_recip"
with open(f"{OUT}/community_ring_stats.tsv", "w") as f:
    f.write(hdr + "\n")
    for r in rows:
        f.write("\t".join(str(x) if i < 2 else f"{x:.4f}" for i, x in enumerate(r)) + "\n")
rows.sort(key=lambda r: -r[1])
print(hdr)
for r in rows[:40]:
    print("\t".join(str(x) if i < 2 else f"{x:.3f}" for i, x in enumerate(r)))
