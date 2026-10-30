# S7: Colab EG2 benchmark + GCS round trip (prepared 2026-10-28, not yet run on Colab)

S7 is the whole-network signals plan's Colab smoke test
([plan](../../../docs/execution-plans/2026-10-25-whole-network-signals-plan.md) row S7, §5.1, Q10; the plan lives in
the Catbird+Petrel workspace repo). It measures how fast EmbeddingGemma 2 (EG2, the post-embedding model S4 kept)
runs on a Colab L4 and an A100, so that S12 (the ~45M-post producer embedding pass) can be asked for with a measured
unit projection. It also proves, once, that a Colab VM can read from and write to our GCS bucket. Nothing here has
run on Colab; it runs only after Josh approves **R-S7**. Decisions: `docs/DECISIONS.md`, 2026-10-28 S7 entry.

| File | Where | What |
|---|---|---|
| `prep.py` | Mac | 100k random posts + 10k slab-format rows from S4's filtered corpus; text + row id only |
| `bench.py` | both | Loads EG2 exactly as S4 did (bf16, text-only, `Clustering` prompt); throughput, tokens/post, NaN, parity |
| `colab_job.py` | VM | Verifies uploads, installs pins, downloads EG2 at revision `f9972ac2…` and checks 13 sha256s, runs bench |
| `roundtrip_job.py` | VM | After `colab auth`: prints granted scopes, reads the synthetic object, writes one small object back |
| `run_colab.sh` | Mac | Print-only driver; `--execute` only after R-S7 |
| `receipt.py` | Mac | Builds `receipt.json`: units per 10M posts per GPU (with arithmetic), S12 projection, round-trip result |
| `s7lib.py` | both | Pure helpers (draws, slab format, redaction, unit guard, parity stats, GCS prefix guard) |

## Inputs (built 2026-10-28; `~/catbird-lake-local/s7-bench-20261028/prep_receipt.json`)

Start: S4's `posts_corpus.parquet` (1,976,028 four-day originals, 195,545 producers; already without hide-from-recs,
visibility-hidden, opt-out and likely-minors-home producers). Re-applied: hide-from-recs union (0 more posts), user-intents
`bulkDataset:false`/`syntheticContentGeneration:false` (0 DIDs: the one declaration on record is all-true), S5
`purge_dids.txt` (4,680 posts / 398 producers dropped), likely-minors = S4 rule ∪ C5 rule (39,206 DIDs; 1,784 more posts
dropped). Eligible: 1,969,564 posts, 194,983 producers.

- **Random:** the 100,000 posts with the smallest `blake2b64('s7-random-100k|did/rkey')` (69,157 producers).
  42.55 tokens/post with the 7-token prompt (35.55 text only), p99 119.
- **Slab-format:** 10,000 producers in `blake2b64('s7-slab-producer|did')` order, one post each
  (`'s7-slab-post'` salt), disjoint from the random set; text + ` [image: <alt>]` per non-empty alt, as
  `graph/vmpass/sql_p2.py` builds O8. S4's corpus has no alt strings, so the 3,616 candidates with media were
  read live from the public AppView (`app.bsky.feed.getPosts`, 145 calls, 0 errors); 14 no longer returned were
  dropped. 3,304 rows carry media, 771 carry alt text (925 alts). 47.09 tokens/post with prompt; rows with alt
  text 100.1. The prompt is not stored in the text: the encoder prepends it, as S12 will.
- **Parity reference:** rows 0–999 (random) and the first 250 slab rows, encoded on the Mac with S4's stack
  (s4-venv: torch 2.14.1, transformers 5.21.0, sentence-transformers 6.2.0, MPS, bf16, batch 64) →
  `upload/parity_mac.npz` (768-d fp32). It matches S4's stored vectors for the same posts at median cosine
  0.99999 (min 0.99993).

Upload = `s7_random.parquet` (11.2 MB), `s7_slab.parquet` (1.2 MB), `parity_mac.npz` (3.9 MB), the code, and a
13-line sha256 list for the model files. Columns are `row_id, text, n_alts` only; the row-id → DID map stays in
`local/s7_rowmap.parquet`. Post text itself can mention accounts (14 random and 4 slab texts contain a `did:` string).

## Run

```bash
./run_colab.sh                 # print-only: preflight, manifest, every command
./run_colab.sh --execute       # after R-S7
```

1. Receipts: `colab usage`, `colab sessions` (account-wide; a C5 session may be listed and is never touched).
2. Mac writes the synthetic object: `s7lib.py randbytes 2000000000 … | gcloud storage cp - gs://catbird-436600-plan-30d/colab-test/synthetic-2e9-random-20261028.bin`
   (streamed from `os.urandom`, sha256 recorded, no local 2 GB file).
3. **L4 session** `s7-l4-<UTC>`: `colab new` → sessions.tsv → watchdog → mkdir + 9 uploads → `colab exec colab_job.py`
   → download results. Then the driver prints the `colab auth` command for Josh and polls the VM's file listing for
   `/content/.adc/adc.json` (≤20 min); then `colab exec roundtrip_job.py` → download. If either fails: the §5.1
   fallback (Josh opens `colab url -s … --open` and runs `from google.colab import auth; auth.authenticate_user()` in
   the browser), polled ≤15 min, then the round trip again. `colab stop`.
4. **A100 session** `s7-a100-<UTC>`: same without the round trip. Skipped if `colab new --gpu A100` fails (no retry
   on another GPU) or if <1.2 units of the cap are left.
5. Receipts after, `receipt.py` → `receipt.json`.

Every colab CLI output passes through `s7lib.redact` (URL query strings and token-shaped strings removed). The
watchdog runs `colab stop` on its own session only, when wall time × rate reaches that session's budget.

## How `colab auth` works in CLI 0.7.4 (read from the installed source, not run)

- It is a **hidden** command (`colab auth -s NAME`; absent from `colab --help`). It runs, on the VM's kernel,
  `os.environ['USE_AUTH_EPHEM']='0'; from google.colab import auth; auth.authenticate_user()`
  (`colab_cli/commands/automation.py`), with stdin relayed from the local terminal and a 600 s timeout.
- With `USE_AUTH_EPHEM=0`, `authenticate_user` runs `gcloud auth login --enable-gdrive-access --no-launch-browser
  --quiet` on the VM, then copies the credential to `/content/.adc/adc.json` and points
  `GOOGLE_APPLICATION_CREDENTIALS` at it (googlecolab/colabtools `google/colab/auth.py`).
- So it is **not headless**: it prints a Google sign-in URL, and a human must sign in, approve, and paste the code
  back at the terminal prompt. The CLI's bundled SKILL.md says the same ("`auth`/`drivemount` genuinely require a
  human at the terminal"). `colab exec` and `colab run` never relay stdin, so **"`colab auth` inside `colab run`" (the
  plan's wording for S7 and S12) cannot work**: the path is `colab new` → Josh runs `colab auth` → `colab exec` → `colab stop`.
- The credential is a gcloud user credential (Google Cloud SDK's OAuth client) with a refresh token, so it keeps
  working for a long S12 job. Expected scopes: gcloud's login defaults (`openid`, `userinfo.email`, `cloud-platform`,
  `appengine.admin`, `sqlservice.login`, `compute`, `accounts.reauth`) **plus `drive`**. The exact list is read from
  Google's tokeninfo endpoint on the VM and printed into the receipt.
- The CLI's local history file (`~/.config/colab-cli/history/<session>.jsonl`) records what is typed at the prompt,
  i.e. the one-time authorization code. That code is single-use and exchanged within seconds; the driver never
  copies the history into a receipt.
- **Do not revoke the token afterwards.** It is a grant to gcloud's own OAuth client, and revoking it can also sign
  out Josh's local gcloud. `colab stop` destroys the VM and the only copy of the credential with it.

## Units (estimate; the run measures and self-stops)

Rates (handoff §1, the table C5 uses): L4 1.71 units/h, A100 5.4 units/h. The driver also reads the live rate from
`colab usage` before and after `colab new`, and uses the larger of the two.

- **Throughput priors** (S7 exists to replace them): EG2's text path is ~130M non-embedding parameters, so ~0.26 GFLOP
  per token and ~11–12 GFLOP per post at 42.5–47 tokens. At 20–35% of bf16 peak: L4 (121 TFLOPS) ~2,200–3,800
  posts/s, point 2,500; A100 (312 TFLOPS) ~2.6× that, point 5,000 (the plan's prior is 3,750).
- **Units per 10M posts:** L4: 1e7 / 2,500 = 4,000 s = 1.111 h × 1.71 = **1.90**. A100: 1e7 / 5,000 = 2,000 s =
  0.556 h × 5.4 = **3.00** (at 3,750 posts/s: 4.00). The A100 is cheaper per post only if it runs >3.16× (5.4/1.71)
  the L4's posts/s.
- **S12 at these priors** (44.9M slab posts × 1.5 inefficiency): L4 4.99 h × 1.71 × 1.5 = **12.8**; A100 2.49 h ×
  5.4 × 1.5 = **20.2** (plan prior 26.9). `receipt.py` recomputes this from the measured slab rate.
- **This run:** L4 session ≈ 4 min allocate + upload, 5 min pinned installs, 1.5 min model download + checks, 2 min
  benchmark (151.5k encodes), 0.5 min results, ~5 min for Josh's `colab auth`, 1.5 min round trip ≈ 19.5 min = 0.33 h
  × 1.71 = **~0.56**. A100 ≈ 12.5 min = 0.21 h × 5.4 = **~1.13**. **Expected ~1.7 units** (plan: ~2.8).
- **Cap 8**, enforced four ways: L4 budget 3.0 units (1.75 h), A100 budget min(4.5, 8 − L4 spent − 0.2); a watchdog
  that stops the session at its budget; `colab exec --timeout`; and the job's own unit guard, which will not start a
  step whose estimate crosses the budget. If Josh never completes the auth, the waits cost at most (20 + 15) min ×
  1.71 = 1.0 unit. The account balance before/after also includes any other session running meanwhile (C5).

## Local verification done (2026-10-28, Mac)

- `test_s7lib.py` (22 tests: draws, O8 slab format, alt parsing, redaction, unit guard and arithmetic, parity
  stats, GCS prefix guard, `roundtrip_job.py` against a local stand-in bucket) and `test_driver.py` (5 tests:
  `run_colab.sh --execute` against a fake colab CLI and fake gcloud: primary round trip, fallback, failed round trip,
  A100 unavailable, watchdog; only `s7-*` sessions are ever stopped, the C5 session never). 27 pass.
- Rehearsal of the exact staged `colab_job.py` (`S7_LOCAL_REHEARSAL=1`, MPS, 3,000 random + 1,000 slab rows, MPS
  allocator ≤7 GB): manifest and 13 model sha256s verified, sweep, both measurements, 0 NaN, parity median 1.0,
  tar written. Cross-device parity (CPU bf16 vs the MPS reference, all 1,250 rows): median 0.99997, min 0.99942,
  0 below 0.99. `roundtrip_job.py` local: 50 MB synthetic read + write-back OK.

## Success checks (plan row S7)

posts/s and tokens/post for slab and random separately, units per 10M posts per GPU; no NaN in bf16; median cosine
≥0.99 vs the Mac on the 1,000 random parity posts; round-trip read and write both succeed, or the §5.1 fallback is
chosen; OAuth scopes in the receipt; balance before and after; `colab sessions` after shows none of this run's
sessions. Next day: Billing → Reports, SKU 22EB-AAE8-FBCD, for the ~1.86 GiB read (Q10).
