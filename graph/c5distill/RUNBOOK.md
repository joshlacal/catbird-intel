# C5 distilled classifiers: runbook (prepared 2026-10-28, not yet run on Colab)

C5 is the whole-network plan's check that the Claude teacher labels can be learned by a small model. It
fine-tunes a multilingual encoder on (a) reply/quote stance with a binary "hostile" read-out, from the Sonnet
pair labels, (b) profile account type, from Sonnet-else-Haiku labels, and (c) community type, from the Opus
labels. The bar is the S1 pre-registration §9.7 (R23); C6 (the ~8B LoRA labeller, cap 400 units) may be asked
only if the hostile classifier clears it. Governing docs: [plan](../../../docs/execution-plans/2026-10-25-whole-network-signals-plan.md)
(workspace repo), [S1 rules](../../../docs/execution-plans/2026-10-28-whole-network-preregistration.md) §9.7 and
amendment A1, [DECISIONS](../../../docs/DECISIONS.md) (C5 entry, 2026-10-28). Paths are relative to the
catbird-intel checkout inside the Catbird+Petrel workspace.

## Files

| File | Runs on | What it does |
|---|---|---|
| `rules_c5.json` | — | Registered splits, exclusions, inputs, hyperparameters and bars. Its sha256 goes in every receipt |
| `prep.py` | Mac | Filters the labelled rows (A1, minors, opt-outs), assigns splits with ClickHouse `cityHash64`, writes the three upload files without DIDs or URIs, plus `prep_receipt.json` |
| `c5lib.py` | both | Kappa, macro-F1, cluster bootstrap, unit guard (tested against sklearn in `test_c5lib.py`) |
| `train.py` | both | One task: fine-tune, select on val, then score the held-out split once and apply the bar |
| `colab_job.py` | Colab VM | Verifies the manifest, pins and checks the model, runs the tasks under the unit guard, tars results |
| `run_colab.sh` | Mac | Print-only driver; `--execute` only after R-C5 is approved |

## Model

`jhu-clsp/mmBERT-base` at revision `c5955035435e2bf121cde7f3c8863ef52ff35d82`, **MIT**, 307M parameters (197M of
them the 256k-token embedding; ~110M in the 22 encoder layers), `pytorch_model.bin` 1,231,188,142 bytes, sha256
`8ea64ec1…a04e`, 1,800+ languages, 8,192-token context. The local smoke runs use `jhu-clsp/mmBERT-small`
(`abc32620…`, MIT, 140M, 563,633,326 bytes, sha256 `be7c9ce2…85b6`). Both are in
`~/catbird-lake-local/models/` with `SHA256SUMS` entries. The Colab job downloads mmBERT-base from the Hub at
the pinned revision and refuses to train if the weights' sha256 differs.

## Data that leaves the Mac (R-C5)

Only the three files `prep.py` writes, about 40 MB in total, via `colab upload` straight onto the session's VM:

| File | Rows | Text it carries | Identifiers |
|---|---|---|---|
| `c5_pairs.parquet` (12.5 MB) | 82,875 | parent and child post text (≤300 chars each) | `pair_id` (p…/q…), a per-run integer author id |
| `c5_profiles.parquet` (19.0 MB) | 162,483 | display name, bio, the Opus label of the profile's community | `pid` (u…) |
| `c5_communities.parquet` (10.2 MB) | 8,809 | the Opus prompt's languages, list names and top-member names and bios; the sample-posts section is cut | community id |

No DID, handle or URI is uploaded. Excluded before upload: 13 pairs with a hide-from-recs author (A1: their
posts are out of the teacher and training), 413 pairs with a likely-minors author, 4,342 likely-minors bios
(Haiku ∪ Sonnet flags, or a likely-minors community), 168 likely-minors communities, and user-intents opt-outs
(0 today). Hide-from-recs bios stay (A1): 1,335 profiles.

**No GCS bucket is used.** The data goes Mac → VM with `colab upload` and results come back with
`colab download`, so there is no `colab auth` OAuth grant and no object to expire. `colab stop` destroys the
VM and the text with it, minutes after the run; that is shorter than any plan bucket's lifecycle. D20 (Colab
data-processing terms, Batch A yes) covers the upload.

*Fallback, only if `colab upload` fails:* the three files go to `gs://catbird-436600-plan-30d/intermediates/c5-<run>/`
(free ingress; the bucket's 7-day `intermediates/` rule removes them, inside the 30-day plan limit), read in Colab
after `colab auth`, which is interactive (Josh runs that cell, ~5 min). This needs the plan-30d bucket to exist
(created by the vmpass `launch_vm.sh buckets` command under its own ask) and a fresh R-C5 line naming the
OAuth grant. Do not use it without that ask.

What comes back to the Mac: `c5_results.tar` (metrics JSON, per-row test predictions keyed by pair_id / pid /
community, the job receipt, a few MB) and `c5_models.tar` (three fp16 students, ~0.6 GB each, ~1.85 GB).

## Units (estimate; the run measures and self-stops)

Rates (handoff §1): L4 1.71 units/h, A100 5.4 units/h. Inputs measured with the mmBERT tokenizer:
pairs 64.4 tokens mean, profiles 54.4, communities 649.

- **Training tokens** with ~2× padding for random batches (1.3× for communities):
  stance 57,969 × 3 epochs × 64.4 × 2 = 22.4M; account type 113,717 × 3 × 54.4 × 2 = 37.1M;
  community type 6,144 × 6 × 649 × 1.3 = 31.1M. Total **90.6M tokens**.
- **Compute:** 6 × 110M encoder parameters × 90.6M tokens × 1.15 (attention, head) = 6.9e16 FLOP.
  L4 bf16 dense 121 TFLOP/s × 25% utilisation = 30 TFLOP/s → **38 min**.
- **Optimizer:** fused AdamW touches all 307M fp32 parameters each step, ~28 B per parameter = 8.6 GB per step
  ÷ 300 GB/s = 29 ms × 15,375 steps (5,436 + 5,331 + 4,608) = **7.4 min**.
- **Evaluation** (val each epoch, test once): ~2 min. **Setup** (VM, 40 MB upload, 1.23 GB model
  download): ~8 min. **Result download** (~1.9 GB): ~5 min.
- **L4 total** ≈ 60 min = 1.0 h × 1.71 = 1.7 units; × 1.5 for inefficiency (the plan's convention) =
  **~2.6 units**. Band 1.3 (40% utilisation, no inefficiency) to 3.7 (15% utilisation, × 1.5).
- **A100** for comparison: compute 15 min at 78 TFLOP/s, optimizer 1.5 min, overhead 15 min ≈ 32 min × 1.5 ×
  5.4 = **~4.3 units**. L4 is cheaper and is the GPU asked for.
- **Self-stop:** each `train.py` checks every 100 steps and before every epoch that the units so far (all wall
  time since VM allocation, × the assigned GPU's rate) plus the rest of the epoch and the final scoring stay
  ≤ the cap; otherwise it stops and scores the best epoch so far. `colab_job.py` does not start a task whose
  estimate (× 1.5) would cross the cap. `colab exec --timeout` is the cap's hours at the GPU rate, at most 6 h.
  An unknown GPU name stops the job unless `C5_RATE` is given. Cap: **20 units** (Batch A), expected ~3.

## Run (after approval)

```bash
cd catbird-intel/graph/c5distill
~/catbird-lake-local/s4-venv/bin/python prep.py --labels ~/catbird-lake-local/claude-labels-20261027 \
  --s5 ~/catbird-lake-local/s5-declarations-20261027 --out ~/catbird-lake-local/c5-distill-20261028/inputs
./run_colab.sh                 # print-only: check the commands and the manifest
./run_colab.sh --execute       # after R-C5
```

Sessions rule (plan §3): the driver creates one named session (`c5-distill-<UTC time>`), writes it to
`~/catbird-lake-local/c5-distill-20261028/sessions.tsv` at creation, stops only that session (also from an
exit trap), and records `colab sessions` and `colab usage` before and after. It never stops a session it did
not create.

## Success check

- `job_receipt.json`: manifest verified, weights sha256 verified, GPU and rate, units estimate; balance from
  `usage_before.txt` / `usage_after.txt`; `sessions_after.txt` shows no session of this run.
- `stance_metrics.json` `test.bar`: `kappa_ge_0_6` and `ci_lower_above_haiku_same_rows`; `learnable` = both.
  `account_type_metrics.json` and `community_type_metrics.json` `test.bar.learnable`.
- The sha256 of each downloaded tar matches the `TAR` line in `colab_exec.log`.
- If the hostile classifier is learnable, C6 may be asked (R-C6, cap 400). If not, C6 is not asked, and the
  errors (`stance_test_predictions.jsonl`) feed the mid-November targeted teacher labels.
