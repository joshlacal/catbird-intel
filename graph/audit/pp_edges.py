"""Extract producer->producer follow edges from edges.bin and per-producer reciprocity.

Writes to OUT: pp_keys.npy (sorted uint64 src<<21|dst), producer_stats.npz.
"""
import sys, time
import numpy as np
import pyarrow as pa, pyarrow.csv as pcsv, pyarrow.compute as pc

BUILD, OUT = sys.argv[1], sys.argv[2]
t0 = time.time()
ro = pcsv.ReadOptions(column_names=None, autogenerate_column_names=True)
po = pcsv.ParseOptions(delimiter="\t")
prod = pcsv.read_csv(f"{BUILD}/producers.tsv", read_options=ro, parse_options=po)
pid = prod.column(0).to_numpy().astype(np.int64)
assert (pid == np.arange(len(pid))).all()
followers = prod.column(2).to_numpy().astype(np.int64)
fw = pcsv.read_csv(f"{BUILD}/follower_weights.tsv", read_options=ro, parse_options=po,
                   convert_options=pcsv.ConvertOptions(include_columns=["f0", "f1"]))
uid = fw.column(0).to_numpy().astype(np.int64)
idx = pc.fill_null(pc.index_in(fw.column(1), value_set=prod.column(1)), -1).to_numpy().astype(np.int64)
uid2pid = np.full(uid.max() + 1, -1, np.int64)
uid2pid[uid] = idx
print(f"producers={len(pid)} followers_rows={len(uid)} producers_with_uid={(idx>=0).sum()} {time.time()-t0:.0f}s", flush=True)

e = np.memmap(f"{BUILD}/edges.bin", dtype=np.uint32, mode="r").reshape(-1, 2)
n = e.shape[0]
out_all = np.zeros(len(pid), np.int64)  # producers each producer follows
keys = []
CH = 200_000_000
for s in range(0, n, CH):
    blk = np.asarray(e[s:s + CH])
    src = uid2pid[blk[:, 0]]
    m = src >= 0
    src = src[m]; dst = blk[m, 1].astype(np.int64)
    out_all += np.bincount(src, minlength=len(pid))
    keys.append((src << 21) | dst)
    print(f"  {s+len(blk):,}/{n:,} pp={sum(len(k) for k in keys):,} {time.time()-t0:.0f}s", flush=True)
keys = np.concatenate(keys); keys.sort()
np.save(f"{OUT}/pp_keys.npy", keys)
rev = ((keys & ((1 << 21) - 1)) << 21) | (keys >> 21)
pos = np.searchsorted(keys, rev)
pos[pos >= len(keys)] = 0
recip = keys[pos] == rev
src = keys >> 21; dst = keys & ((1 << 21) - 1)
recip_out = np.bincount(src, weights=recip, minlength=len(pid))
in_pp = np.bincount(dst, minlength=len(pid))
np.savez(f"{OUT}/producer_stats.npz", followers=followers, out_pp=out_all,
         recip_out=recip_out, in_pp=in_pp)
print(f"pp_edges={len(keys):,} reciprocal={recip.mean():.3f} {time.time()-t0:.0f}s")
