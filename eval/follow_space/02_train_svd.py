"""Step 2: collaborative item embedding from train_T follows only (randomized subspace iteration on X^T X).

X[i, j] = idf_j / ||row_i||  for sampled viewer i following producer j (binary, item-IDF weighted, row L2-normalised),
idf_j = log(n_rows / df_j), columns kept iff sample df_j >= MIN_DF. Rows = 01_prep train sample (viewers with 10..3000
train follows, Bernoulli p chosen for ~600M nnz). Output = top-K right singular vectors V and singular values S.

Reads: $SCRATCH/data/train_csr.npz, producers.tsv (for T follower counts in the stats only)
Writes: $SCRATCH/data/svd.npz {V (n x K f32, desc), S (K), item_pid (n i32), df (n i32)}, svd_stats.json

usage: python 02_train_svd.py [K=256] [q_iters=4]
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import common
from common import DATA, N_THREADS, SEED, log

import numpy as np
import scipy.sparse as sp

K = int(sys.argv[1]) if len(sys.argv) > 1 else 256
Q_ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OVERSAMPLE = 32
L = K + OVERSAMPLE
MIN_DF = 30
N_CHUNKS = 40
stats = {"K": K, "q_iters": Q_ITERS, "oversample": OVERSAMPLE, "min_df": MIN_DF}

z = np.load(os.path.join(DATA, "train_csr.npz"))
indptr = z["indptr"]; indices = z["indices"]; n_cols = int(z["n_cols"])
m0 = len(indptr) - 1
log("train csr rows", m0, "nnz", len(indices), "cols", n_cols)

# ---- column filter (sample df >= MIN_DF) ----
df_all = np.bincount(indices, minlength=n_cols)
keep_col = df_all >= MIN_DF
col_map = np.full(n_cols, -1, np.int32)
item_pid = np.flatnonzero(keep_col).astype(np.int32)
col_map[item_pid] = np.arange(len(item_pid), dtype=np.int32)
n = len(item_pid)
mapped = col_map[indices]
del indices
keep = mapped >= 0
row_cnt = np.add.reduceat(keep.astype(np.int64), indptr[:-1]) if m0 else np.zeros(0, np.int64)
row_cnt[np.diff(indptr) == 0] = 0
new_idx = mapped[keep]
del mapped, keep
nonempty = row_cnt > 0
row_cnt = row_cnt[nonempty]
m = len(row_cnt)
new_indptr = np.zeros(m + 1, np.int64); np.cumsum(row_cnt, out=new_indptr[1:])
nnz = int(new_indptr[-1])
stats.update(rows_in=m0, rows=m, items=n, items_dropped_low_df=int((~keep_col & (df_all > 0)).sum()), nnz=nnz)
log("items kept", n, "rows", m, "nnz", nnz)

df = np.bincount(new_idx, minlength=n).astype(np.int64)
idf = np.log(m / df).astype(np.float32)
row_norm = np.sqrt(np.add.reduceat((idf ** 2)[new_idx].astype(np.float64), new_indptr[:-1])).astype(np.float32)
data = idf[new_idx]
data /= np.repeat(row_norm, row_cnt)
X = sp.csr_matrix((data, new_idx, new_indptr), shape=(m, n))
del data, new_idx
log("X built", X.data.dtype, X.indices.dtype, X.indptr.dtype)
XT = X.T.tocsr()
log("XT built")


def row_chunks(M, k):
    bounds = np.linspace(0, M.shape[0], k + 1).astype(np.int64)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        s, e = M.indptr[a], M.indptr[b]
        out.append((a, b, sp.csr_matrix((M.data[s:e], M.indices[s:e], M.indptr[a:b + 1] - s), shape=(b - a, M.shape[1]), copy=False)))
    return out


Xc = row_chunks(X, N_CHUNKS)
XTc = row_chunks(XT, N_CHUNKS)
pool = ThreadPoolExecutor(N_THREADS)


def matmul(chunks, B, out):
    def job(t):
        a, b, M = t
        out[a:b] = M @ B
    list(pool.map(job, chunks))
    return out


rng = np.random.default_rng(SEED)
Q = rng.standard_normal((n, L), dtype=np.float32)
Y = np.empty((m, L), np.float32)
Zb = np.empty((n, L), np.float32)

# GIL check: one chunk serial vs all chunks in the pool
t0 = time.time(); _ = Xc[0][2] @ Q; t_one = time.time() - t0
t0 = time.time(); matmul(Xc, Q, Y); t_all = time.time() - t0
stats["gil_check_one_chunk_s"] = round(t_one, 2); stats["gil_check_all_chunks_s"] = round(t_all, 2)
stats["parallel_speedup_vs_serial_estimate"] = round(t_one * N_CHUNKS / t_all, 2)
log("X@Q one chunk", round(t_one, 2), "s; all chunks in pool", round(t_all, 2), "s; speedup", stats["parallel_speedup_vs_serial_estimate"])

for it in range(Q_ITERS):
    matmul(XTc, Y, Zb)
    Q, _ = np.linalg.qr(Zb)
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    matmul(Xc, Q, Y)
    log("subspace iter", it + 1, "/", Q_ITERS)

# Y = X @ Q now; small eigenproblem on Y^T Y
C = np.zeros((L, L), np.float64)
for a in range(0, m, 500_000):
    Yb = Y[a:a + 500_000].astype(np.float64)
    C += Yb.T @ Yb
lam, W = np.linalg.eigh(C)
order = np.argsort(lam)[::-1]
lam = np.clip(lam[order], 0, None); W = W[:, order]
S = np.sqrt(lam[:K]).astype(np.float32)
V = (Q @ W[:, :K].astype(np.float32)).astype(np.float32)
stats["singular_values_first10"] = [round(float(x), 4) for x in S[:10]]
stats["singular_values_last3"] = [round(float(x), 4) for x in S[-3:]]
stats["energy_captured_frac_of_frobenius"] = round(float((S.astype(np.float64) ** 2).sum() / m), 4)  # ||X||_F^2 = m
stats["peak_rss_gb"] = round(common.peak_gb(), 2)
np.savez(os.path.join(DATA, "svd.npz"), V=V, S=S, item_pid=item_pid, df=df.astype(np.int32))
json.dump(stats, open(os.path.join(DATA, "svd_stats.json"), "w"), indent=1)
log("done", json.dumps(stats))
