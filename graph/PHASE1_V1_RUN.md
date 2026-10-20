# Phase 1 v1 continuation

Catbird Intel is rebuilding its account communities from archive follow records after the previous evaluation admitted supplemental seed edges and inflated separation. The original archive-only fold is still running; a separate local driver waits for its verified outputs, then builds similarities, clusters accounts, evaluates the result, and exports real post triplets. No v1 graph-quality result or v1 pair bank exists at this checkpoint. Continue the existing processes and publish only artifacts whose completion and checksums have been verified.

Governing documents: [master plan](../../docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md), [correction decision](../../docs/DECISIONS.md#2026-10-20--catbird-intel--invalidate-stub-completion-claims-and-rebuild-from-real-data), [actual clustering/evaluation decision](../../docs/DECISIONS.md#2026-10-20--catbird-intel-phase-1--report-actual-clustering-and-raw-evaluation-failures), and [size-only tuning decision](../../docs/DECISIONS.md#2026-10-20--catbird-intel-phase-1--tune-actual-community-median-using-size-only).

## Checkpoint and process ownership

At **2026-10-20 07:31:35Z**, the original ClickHouse fold, PID **95795**, was materializing latest-revision distinct follows into its temporary MergeTree table. An observation of its data parts found **354,562,820 rows**; this is a progress observation under concurrent merges, not the final follow count. Final `producers.tsv`, `edges.bin`, and `stats.tsv` were absent. The latest machine-readable progress observation, if newer, is `~/catbird-lake-local/graph-build/out/fold_progress_observed.json`.

- Original fold: `task-2745`, PID `95795`. Leave it running.
- Continuation driver: PID `10905`, execution session `22041`.
- Working directory: `/Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel`.
- Driver source: `graph/run_phase1_v1.py`.
- Driver status: `~/catbird-lake-local/graph-build/out/pipeline_status.json`.
- Original fold log: `~/catbird-lake-local/logs/phase1-v1-build.log`. It can remain empty while ClickHouse is doing useful work.
- Stage logs: `~/catbird-lake-local/logs/phase1-v1-{simgraph,cluster,evaluation,pairs}.log`.

Check current state before taking action:

```bash
cd /Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel
ps -p 95795,10905 -o pid,ppid,%cpu,%mem,etime,command
cat ~/catbird-lake-local/graph-build/out/pipeline_status.json
ls -lh ~/catbird-lake-local/graph-build/out
df -h ~/catbird-lake-local
```

The driver uses eight Rayon threads for similarity construction and two Cargo build jobs. The sampler's main work is proportional to producers × at most 1,000 sampled followers × at most 500 followed producers; it is not an all-producer-pairs calculation, but it may still be lengthy. No reliable ETA has been established.

## What the driver does

1. Waits for the original fold PID to exit. Before waiting, it checks that the PID identifies the expected ClickHouse command and `graph-build/fold.sql`.
2. Requires nonempty producer, edge, and stats outputs; validates dense producer IDs, producer count, summed follower degrees, and the binary edge count against `stats.tsv`.
3. Writes `input_provenance.json` with the exact fold SQL hash and hashes of the finished dense graph inputs.
4. Runs release `graph/simgraph` with deterministic per-producer sampling.
5. Runs `graph/cluster.py` with resolution search, a cap of 30,000 communities, and a target ordinary **per-community median of 100 producers**. It records the producer-weighted median, maximum, p90, and cap exclusions separately. Seeds and traps do not select the resolution.
6. Runs `graph/eval_phase1.py` against only the public and GCE archive event directories. It hashes the exact evaluation inputs and checks that they did not change during the evaluation.
7. Runs `ops/export_pairs_v1.py`, reading `catbird.v_posts_live` on netcup with explicit visibility and deletion-ledger exclusions. It joins the local v1 KnownFor map without replacing any production table.

Each successful stage records its command, return code, elapsed time, log path, and output SHA-256 values in `pipeline_status.json`. A stage failure stops subsequent stages. `complete_local_unqualified` means these local steps finished; it does **not** mean the master plan's Phase 1 done criteria passed.

### Executed-source provenance limitation

PID `10905` started at **07:04:35Z**, before the last edit to `run_phase1_v1.py`. That later edit made `--size-objective community` explicit in the cluster command; the running driver has the earlier command in memory, while the current `cluster.py` defaults to the same community objective. No driver-source hash was captured at launch. A hash of the current file must **not** be presented as the hash of the driver code already executing.

`downstream_sources_observed_before_execution.json` records source hashes observed while the driver was still waiting for the fold. It is a dated observation, not an invocation-time attestation. Downstream scripts are loaded only when their stage starts, and the current driver records their command and output hashes but does not capture their source hashes at invocation. Preserve this limit in reports; compare observed and current source hashes when reviewing a stage, and do not silently attribute a later source revision to an earlier execution. Do not restart the original fold or the active driver solely to improve this receipt.

## Resume without duplicating the fold

If the driver is alive, do not start another copy. Read its status and relevant log. A stale status timestamp while `stage=waiting_for_fold` is expected; inspect the original fold process as well.

If the driver has exited, inspect `stage_failed`/`error` and the failing stage log. Also inspect the original fold log if outputs are missing. Do **not** start a second fold simply because the driver stopped.

After resolving the actual failure, resume the continuation only:

```bash
cd /Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  graph/.venv/bin/python graph/run_phase1_v1.py \
  --fold-pid 95795 --rayon-threads 8 --resume
```

The driver refuses a duplicate live driver, verifies the hashes of completed stages, and skips them. Unsealed remnants of failed stages move to `~/catbird-lake-local/_quarantine/phase1-v1-retry/`; they are not deleted. Interrupted source-post exports without a completed receipt move to the bank directory's `_quarantine_unsealed/`. A previously sealed artifact whose hash changed is an error: preserve it and investigate before choosing a new output generation.

If the fold itself failed before producing consistent outputs, the continuation cannot repair that by rerunning downstream commands. Keep its existing SQL/logs/data intact, diagnose the error, and prepare a separate, explicitly frozen archive generation for any replacement fold. Never restore supplemental seed/trap files to an input directory.

## Results to inspect and report

All graph outputs are under `~/catbird-lake-local/graph-build/out/`:

- `input_provenance.json`: finished graph input hashes and the live-glob limitation.
- `cluster_stats.json`: actual algorithm, resolution, counts, and size distribution.
- `phase1_eval_report.json`: all trap pass/fail results, merged poles and subcultures, per-cohort seed matches in both producers and KnownFor, and raw starter-pack pair hits/denominators.
- `evaluation_input_manifest.json`: exact files used for evaluation, with content hashes.
- `known_for.parquet` and `communities.parquet`: v1 memberships and partition metadata.
- `communities_labeled.parquet` and `colab_community_labeling_prompts.jsonl`: **unreviewed profile-term heuristic labels** and future labeling prompts.

The pair bank and its receipt are:

```text
~/catbird-lake-local/drive-sync-staging/training-banks/phase5_social_encoder_pairs_v1.parquet
~/catbird-lake-local/drive-sync-staging/training-banks/phase5_social_encoder_pairs_v1.manifest.json
```

Report actual rows and `shortfall`; the filename deliberately does not promise 500,000 rows. Each anchor/positive/negative includes its original URI, text, author DID and actual author split bucket. Bucket zero is reserved for the holdout. The manifest pins the KnownFor, graph input provenance, raw evaluation, source-post export, bank and holdout hashes.

Before uploading the new pair bank, run the same transported-data validator that Colab uses:

```bash
cd /Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=colab graph/.venv/bin/python - <<'PY'
from pathlib import Path
from real_pipeline.common import validate_pairs
bank = Path.home() / "catbird-lake-local/drive-sync-staging/training-banks/phase5_social_encoder_pairs_v1.parquet"
rows, manifest = validate_pairs(bank, Path("eval/phase3_holdout_600_unlabeled.jsonl"))
print({"validated_rows": len(rows), "bank_sha256": manifest["sha256"],
       "known_for_sha256": manifest["known_for_sha256"]})
PY
```

Then use the explicit, checksum-verifying repair uploader for the bank and receipt:

```bash
PYTHONDONTWRITEBYTECODE=1 graph/.venv/bin/python ops/upload_repair_v1.py \
  --destination training-banks \
  --file ~/catbird-lake-local/drive-sync-staging/training-banks/phase5_social_encoder_pairs_v1.parquet \
  --file ~/catbird-lake-local/drive-sync-staging/training-banks/phase5_social_encoder_pairs_v1.manifest.json
```

For raw graph evaluation/provenance JSON uploads, first copy the verified bytes to versioned filenames in a separate delivery directory, for example `phase1_eval_report_v1_<sha256-prefix>.json`, `phase1_input_provenance_v1_<sha256-prefix>.json`, `phase1_cluster_stats_v1_<sha256-prefix>.json` and `phase1_evaluation_inputs_v1_<sha256-prefix>.json`. Preserve the original files and record the source-to-delivery name/hash mapping. The uploader deliberately rejects the old unversioned `phase1_eval_report.json` name because it was used by invalidated v0 artifacts. It also rejects changing bytes under an existing Drive filename.

Upload those copies to the repair `eval` destination with the same tool and explicit `--file` arguments. Later repair-report updates likewise need a new dated/versioned Drive filename; the 2026-10-20 delivery is immutable. Do not run a broad legacy lake sync to discover and publish everything in a live graph directory. Keep Drive upload receipts and update master-plan §8 with actual observed outcomes. The [real model launch instructions](../colab/README_REAL_MODELS.md) and [completed GPU-pilot addendum](../eval/COLAB_GPU_PILOT_20261020.md) describe the Colab side once the validated pair bank is present.

## Qualification boundaries and protected state

- The already-running fold consumed a **live archive glob**. `archive_inventory_observed.json` was captured after the fold started; it is not proof of the exact consumed shard set. Finished dense graph inputs and all downstream artifacts are hash-pinned, but full frozen-source reproducibility and global follow-network coverage remain unqualified.
- The implemented clustering algorithm is **Leiden CPM with similarity-share overlapping memberships**. It is not Twitter's Metropolis–Hastings implementation.
- No actual A Blue Start 503-cluster partition has been loaded. The old hash/rank stand-in was removed; no claim of beating that published baseline is allowed.
- Seeds/traps are evaluation-only. Shared top communities are reported as merges, missing seeds are reported, and seed IDs never enter construction, clustering, pack recall, or community labeling.
- Pack recall now uses real archive starter-pack records joined to current listitems, folding item, list and pack deletions. It reports counts and denominators, including excluded pack sizes.
- Community labels use archived profile names/descriptions. They have not been labeled by an LLM or reviewed by a human. This projection does not provide profile handles or starter-pack titles to the labeling function.
- Triplet positives mean different authors in the same graph community; negatives have disjoint KnownFor memberships, preferring the same scene. These are **weak graph-membership labels**, not verified co-engagement or proof of post-level relevance.
- Nothing here ships tables to netcup, replaces the live regex scorer, changes production services, or authorizes a production table swap. The ingest soak ends **2026-10-22 13:00Z**.
- Preserve the running archive decoder, GCE/Colab collectors, external edits to `graph/archive_graph_extract.py`, and its original `graph/__pycache__/archive_graph_extract.cpython-314.pyc`. Use jj for any history changes, followed immediately by `jj new`.
