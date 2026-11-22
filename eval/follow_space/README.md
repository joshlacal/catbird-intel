# follow_space: collaborative follow embedding vs t1 for onboarding scene ranking

Offline temporal evaluation. Question: does an item embedding learned only from follows up to T = 2026-08-15
rank onboarding scene candidates better than the t1 vectors the live engine uses (intel-serving-v4)?
Read-only on the lake; all outputs under `~/catbird-lake-local/ml-follow-space-20261122/` (SCRATCH).

## Run

```
PY=~/catbird-lake-local/s4-venv/bin/python   # py3.12, numpy 2.5.4, scipy 1.18.1, pyarrow 26
cd catbird-intel/.workspaces/lineup-hygiene/eval/follow_space
/usr/bin/time -l $PY 01_prep.py      > $SCRATCH/01_prep.log 2>&1
/usr/bin/time -l $PY 02_train_svd.py > $SCRATCH/02_train_svd.log 2>&1   # args: [K=256] [q_iters=4]
/usr/bin/time -l $PY 03_eval.py      > $SCRATCH/03_eval.log 2>&1
$PY 04_export.py                                                        # no-op unless 03 passed the gate; --force overrides
```

Threads capped at 10 (common.py sets OMP/VECLIB/OPENBLAS; 02 uses a 10-thread pool over row chunks,
measured 6.9x over serial). 03 env hooks for smoke tests: `SVD_NPZ=<path> SMOKE_SCENES=<n> OUT_TAG=<suffix>`.

| step | what | wall | peak RSS | peak footprint |
|---|---|---|---|---|
| 01_prep | pool, eval events, viewer follows, train CSR sample | 52 s | 16.6 GiB | n/a |
| 02_train_svd | randomized SVD, K=256, 4 subspace iterations | 433 s | 27.0 GiB | 45.4 GiB (48.8 GB) |
| 03_eval | 19 sources x 1016 scenes + quantised re-eval | 209 s | 15.7 GiB | 15.4 GiB (16.5 GB) |
| 04_export | int8 export | not run: gate failed (`--force` writes anyway) | | |

02 footprint is near the 48 GB cap: numpy.linalg.qr upcasts the n x 288 float32 block to float64 internally.
Swap to scipy.linalg.qr (stays float32) if the cap must hold with margin.

## Files

- `common.py` paths, seed 20261122, thread caps, logging with peak RSS
- `01_prep.py` -> data/pool.npz, events.npz, viewer_follows.npz, train_csr.npz, degrees.npy, prep_stats.json
- `02_train_svd.py` -> data/svd.npz (V, S, item_pid, df), svd_stats.json
- `03_eval.py` -> data/eval_results.json, eval_table.md
- `04_export.py` -> follow_vec_i8.bin, follow_vec_dids.txt, manifest.json (only on gate pass)

## Data semantics used

- train edges: s2a train_T `edges.bin` RowBinary (uid u32, pid u32); producers = accounts with >= 100 followers at T
  (s2-prep-20261028 s2a_train.sql:17-19, pid order :94, uid :106-107, edges order :126-127).
- test follows: `test/test_follows.parquet`, window 2026-08-16..2026-10-18; burst flag = >= 10 follows within +-30 s
  (s2a_test.sql:1-11; graph/eval_temporal.py:91,:334 BURST_MIN = 10); del_us = earliest delete.
- t1 vectors: intel-serving-v4 `signals.bin` 24-byte records aligned with dids.txt (vector_store.ts:292); flags u16 @0
  (:397), t1 = flag value 8 (:339-344, :572); vecRow i32 @20 (:410); vector = vec_t1_i8.bin[vecRow] (:428-433).
  Vector rows are NOT aligned with dids.txt lines.
- engine shape (onboarding_live.ts): taste = unit mean of unit vectors, null below 8 (:489-511); affinity cosine (:526);
  z over scene members with known affinity, population std, >= 4 known, clamp +-2, weight 1/(rank+3)^1.5 x exp(0.35 z)
  (:550-575, config :103-127).

## Result

Run of 2026-11-22. Full table, beta sweep, all 18 variants and coverage: `$SCRATCH/eval_table.md`, raw `$SCRATCH/data/eval_results.json`.

150,000 sampled events (non-burst, alive 14 d, positive in pool, viewer has 10..3000 train follows), 1016 scenes,
89.5 candidates per event. Variant chosen on the dev half by viewer hash; the numbers below are the report half (n = 74,699).
Selected variant: d = 256, singular-value power 0, mean-centred.

| scorer | hit@6 | hit@12 | MRR | h12 viewer 10-49 | h12 pos <1500 | h12 pos >=20k |
|---|---|---|---|---|---|---|
| B0 rankScoreBps | .3685 | .5116 | .2384 | .5557 | .2472 | .9409 |
| B1 pool followers | .3336 | .4673 | .2167 | .5062 | .1311 | .9575 |
| B2 t1 cosine | .2529 | .3901 | .1645 | .3899 | .4140 | .3472 |
| B3 t1 engine blend b.35 | .3900 | .5383 | .2522 | .5794 | .2776 | .9481 |
| N1 collab cosine | .2907 | .4230 | .1868 | .4404 | .4166 | .3569 |
| N2 collab engine blend b.35 | .3959 | .5432 | .2572 | .5822 | .2826 | .9448 |
| N3 t1+collab blend b.35 | .4178 | .5694 | .2723 | .6039 | .3164 | .9482 |
| N3q (int8) | .4179 | .5694 | .2723 | .6042 | .3162 | .9482 |
| random (expected) | | .1635 | | .1649 | .1811 | .1550 |

Gate: **FAIL**. Best N (N3 b.35) / best baseline (B3 b.35) = 0.5694 / 0.5383 = 1.058, needs 1.10.
The viewer 10-49 condition holds (.6039 vs .5794). At beta 1.0: N3 .6027 vs B3 .5692 = 1.059, also short.
int8 quantisation costs nothing.

Coverage (share of events): t1 null taste 0.002, positive without vector 0.0004; N null taste 0.0001,
positive without vector 0.0374, mean candidate without vector 0.0343 (items need sample df >= 30).

Caveats: t1 was trained on data that includes the test window (B2/B3 optimistic); pool rankScoreBps and followers
are Nov 2026 values (B0/B1 may leak); live multipliers and Gumbel sampling are not modelled; train edges are
producer-only (>= 100 followers at T); 1,955 pool members have no pid at T; 1,195,989 test events dropped because the viewer
is not in the train sample.
