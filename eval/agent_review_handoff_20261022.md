# Catbird Community Intelligence Engine — Agent Review Handoff (`2026-10-22 12:40 UTC` / `8:40 AM EDT`)

## 0. Cold-Reader Summary (What This Is, Why It Exists, What Happened, What Needs Review)

Catbird is building a **Community Intelligence Engine** for Bluesky that discovers organic social neighborhoods from the network follow/engagement graph (Twitter SimClusters-style) and scores timeline health (`politicalness`, `outraged`, `critical`, `engagement_bait`, `spam`) so Day-1 onboarding and "For You" feeds surface high-signal community content without flooding timelines with high-volume political arguments.

Over the past 24 hours (`2026-10-21` → `2026-10-22`), work across three subsystems was implemented, tested, and deployed under [`docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md) and [`docs/execution-plans/2026-10-21-network-capture-resume.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/execution-plans/2026-10-21-network-capture-resume.md):

1. **GCE Network Capture, Google Drive Offload Recovery, Local Stage Release & VM Right-Sizing (`catbird-intel/ops/gce/`):** Recovered a two-fault chain (`HTTP 500` followed by `HTTP 410 Gone` on `seg_00000002vh.jss` / `run_id=be320fc55216e616e563be8b5fd7a76e`), added bounded in-session transient `5xx` resumable status probing (`bytes */total`), implemented cross-boot monotonic request pacing, released verified local stage duplicates after durable GCS + Drive closure, and right-sized `jetstream-selfhost-test` from `n2-highmem-32` (`256 GiB`) to `n2-highmem-8` (`64 GiB`, cutting GCE burn by `75%` to `$7.54/day`). As of `2026-10-22 12:40 UTC`, **`200` pairs (`54.58 GB`)** are durably closed in Google Drive with **`0` failed unresolved attempts**, **`2,088` pairs (`563.87 GB`)** are durable in GCS (`gs://catbird-436600-jetstream-buffer-20261020`, designated as the **archive of record** for the October 27 retention review), and **`148 GiB` free disk (`22%` used)** remains on the VM.
2. **`netcup` Phase 0 Soak & Daily Export Memory Bounding (`catbird-intel/ops/export-day.sh`):** Fixed ClickHouse `Code: 241 MEMORY_LIMIT_EXCEEDED` on the daily `catbird.engagements` `ORDER BY seq` Parquet export by bounding external sort memory (`max_bytes_before_external_sort = 1500000000, max_memory_usage = 4000000000, max_threads = 2`), sealed `/var/lib/catbird-intel/export/2026-10-20/manifest.json` (`14` Parquet files, `2.00 GB`), and verified `day=2026-10-21` exported automatically at `00:30 UTC` while `catbird-intel-ingest.service` reached **`71h 40m` of its 72-hour soak** (ending **`2026-10-22 13:00Z`**) without touching any production service.
3. **Phase 1 Snapshot-Aware Follow-Graph Fold & Phase 3 `Cloudflare/clef-flash` Calibration (`catbird-intel/graph/`, `catbird-intel/teacher_harness/`):**
   - Updated [`catbird-intel/graph/fold_follows.sql`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/graph/fold_follows.sql) so October GCE snapshots (`source_tier = 2`) and `sync`/`deleted` barriers replace August 4 seed (`source_tier = 1`) follow state per account (preventing zombie unfollow resurrection), accounts without an October snapshot fall back to August 4 seed + incremental live commits and are labeled `august_seed_fallback` (`is_stale = 1`) in `account_coverage.tsv` and `follower_weights.tsv`, `g.inactive` orders by `(source_tier, time, seq)` instead of raw `seq`, and [`simgraph`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/graph/simgraph/src/main.rs) supports optional follower coverage weights (`--follower-weights` / `--use-follower-weights`).
   - Hydrated `1,484 / 1,548` (`95.87%`) missing reply-parent and quoted-post texts for the frozen 2,000-post pilot and 600-post holdout from `netcup` ClickHouse, separated media-only NSFW labels from text-conduct moderation labels, fixed the lexical-prior collision in `Cloudflare/clef-flash` (`JointSchemaHead`) `noul` criteria, and added prose-substance structural signals (`_analyze_systemone_text_substance` in [`transport.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/teacher_harness/transport.py)) so bare URLs, mention-only posts, and captionless media posts score `insufficient` — yielding **`81.25%` (`52/64`) all-component scorable retention** (`12/64 = 18.75%` insufficient on `>= 1` component; `93.75% = 60/64` prior to prose-substance gating) and **`92.6%` (`50/54`) binary agreement (`r = 0.8241`)** with `mlburnham/Political_DEBATE_large_v1.0` on M4 Max.

This handoff gives a reviewing agent every governing contract, commit ID, file path, test command, live verification command, and critical audit question needed to perform an independent, adversarial review.

---

## 1. Glossary of Project Jargon

- **`.jss` segment:** Zstandard-compressed Jetstream event segment (`~268.5 MiB` / `~270 MB` each) accompanied by a `.receipt.json` containing SHA-256 and sequence bounds.
- **Stage A (`PDS → GCE → GCS`):** Self-hosted Jetstream (`ghcr.io/bluesky-social/jetstream:v0.4.4`, commit `d4db8c0`) running on GCE `jetstream-selfhost-test`, crawling ATProto PDS (`com.atproto.sync.getRepo`) and live firehose, sealing `.jss` segments on local disk, and uploading verified payload+receipt pairs to GCS (`gs://catbird-436600-jetstream-buffer-20261020`, the archive of record for the October 27 retention review) via `catbird-gcs-buffer.service`, then releasing local capture segments once `local_segments > 3`.
- **Stage B (`GCS → Google Drive`):** The bounded, ledger-enforced continuous transfer controller (`catbird-drive-retained-continuous-20261021.service`) that stages verified pairs from GCS, uploads them to Google Drive (`CatbirdCommunity/archives/gce-selfhost-001/segments/`, folder `16Zd7n4pwTSDMDsnP9chX2Ci0r0ahEy45`) with pinned pre-allocated Drive file IDs, performs full byte-stream SHA-256 and MD5 readback verification on Drive, publishes an immutable GCS mapping object, and then releases the temporary local stage copies in `/data/gcs-drive-retained-*/stage/`.
- **Standing Exclusion Envelope (`catbird.drive.retained-standing-exclusion.v1`):** A root-owned (`0600`) JSON lease file in `/data/gcs-drive-retained-authority/standing-<UTC>.json` valid for 1 hour (`3600s`) that asserts single-writer authority for the Google Drive OAuth account (`06165974855773273129`). Because each batch reserves a `1,290s` (`21.5m`) safety margin before lease expiry, each 1-hour envelope runs for `~38.5 minutes` (`~7–10` pairs) and stops cleanly with `status=75/TEMPFAIL` (`reason: "standing_exclusion_approaching"`) until renewed.
- **`noul` (`JointSchemaHead` in `Cloudflare/clef-flash`):** A 3-way softmax head (`[false, true, insufficient]`) used by `Cloudflare/clef-flash` (`Qwen3.5-9B`). `JointSchemaHead` computes a cosine-similarity lexical prior between question/criteria embeddings and input state embeddings; if `criteria["true"]` shares tokens with state JSON keys (`reply_parent`, `quoted_post`), it creates a false-positive `insufficient` bias.
- **5-Component Decomposed Health Rubric (v3):** Separate `[0..3]` ordinal ratings (normalized `[0.0..1.0]`) for `politicalness`, `outraged`, `critical`, `engagement_bait`, and `spam`, replacing the legacy conflated single `heat` score.

---

## 2. Governing Plans & Decision Log Links

- **Master Plan:** [`docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md)
- **Review Remediation Plan (D28–D35):** [`docs/execution-plans/2026-10-20-catbird-intelligence-review-remediation.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/execution-plans/2026-10-20-catbird-intelligence-review-remediation.md)
- **Moderation Labels & $200 Teacher Plan (D36):** [`docs/execution-plans/2026-10-20-moderation-labels-and-200-dollar-teacher-plan.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/execution-plans/2026-10-20-moderation-labels-and-200-dollar-teacher-plan.md)
- **Network Capture Resume & Drive Offload Plan:** [`docs/execution-plans/2026-10-21-network-capture-resume.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/execution-plans/2026-10-21-network-capture-resume.md)
- **Review Dossier:** [`docs/design-docs/2026-10-20-catbird-community-intelligence-review-dossier.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/design-docs/2026-10-20-catbird-community-intelligence-review-dossier.md)
- **Decision Log (`docs/DECISIONS.md`):** [`docs/DECISIONS.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/DECISIONS.md)

---

## 3. Commits & Changed Files to Review

### 3.1 Workspace Root Repo (`/Users/joshlacalamito/Developer/Catbird+Petrel`)
- `d0959676` (`change-id: mpwykvus`): `docs: Record verified local transfer-stage release authorization in DECISIONS.md`
- `91c2006c` (`change-id: tuztpryz`): `docs: Record GCE payload-tail 410 recovery and n2-highmem-8 resize decision`
- `1654a13d` (`change-id: yzrmsxry`): Decision log entry for parent/quote hydration, Clef-Flash `noul` criteria calibration, `netcup` export sort memory bound, and GCE `5xx` status-probe recovery.
- Updated [`docs/execution-plans/2026-10-21-network-capture-resume.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/execution-plans/2026-10-21-network-capture-resume.md) and [`docs/DECISIONS.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/DECISIONS.md) (designating `gs://catbird-436600-jetstream-buffer-20261020` as archive of record for the October 27 retention review and recording the October 22 reconciliation, snapshot-aware follow fold, and Clef-Flash prose-substance calibration).

### 3.2 `catbird-intel` Repo (`/Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel`)
- *(Note: commit `6b65a629` / `change-id: lvxzypll` containing two `.pyc` files was abandoned via `jj -R catbird-intel abandon lvxzypll`.)*
- **Phase 1 Follow-Graph Fold & Coverage Weighting:**
  - [`catbird-intel/graph/fold_follows.sql`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/graph/fold_follows.sql)
  - [`catbird-intel/graph/simgraph/src/main.rs`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/graph/simgraph/src/main.rs)
  - [`catbird-intel/graph/cluster.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/graph/cluster.py)
  - [`catbird-intel/graph/test_graph_guards.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/graph/test_graph_guards.py)
  - [`catbird-intel/eval/pds_account_count_reconciliation_20261022.json`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/pds_account_count_reconciliation_20261022.json)
  - [`catbird-intel/eval/pds_host_coverage_weights_20261022.tsv`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/pds_host_coverage_weights_20261022.tsv)
- `af51f592` (`change-id: mmnxkqur`): `catbird-intel: Release verified local stage duplicates after durable GCS+Drive closure`
  - [`catbird-intel/ops/gce/gcs_drive_drain.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/gcs_drive_drain.py) (`release_closed_pair_stage`, `cleanup_closed_pair_stages`, `staged_bytes` exclusion of released stage dirs)
  - [`catbird-intel/ops/gce/test_gcs_drive_drain.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/test_gcs_drive_drain.py)
- `938b42b7` (`change-id: murkoxsw`): `catbird-intel: Harden payload-tail 410 recovery validation, lease adoption budget, and cross-boot pacing`
- `fc7bbeb7` (`change-id: qnmyuuyn`): `catbird-intel: Extend payload_tail_and_receipt_auto for expired uncommitted payload sessions and cross-boot monotonic pacing`
- `948757c0` (`change-id: murqppzn`): `catbird-intel: Hydrate parent/quote context, calibrate clef-flash noul criteria, bound export sort, and add GCE 5xx status-probe recovery`
  - **GCE Drive Offload & Recovery:**
    - [`catbird-intel/ops/gce/gcs_drive_retained_recovery.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/gcs_drive_retained_recovery.py)
    - [`catbird-intel/ops/gce/gcs_drive_retained_transport.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/gcs_drive_retained_transport.py)
    - [`catbird-intel/ops/gce/gcs_drive_retained_continuous.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/gcs_drive_retained_continuous.py)
    - [`catbird-intel/ops/gce/recover_retained_receipt500.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/recover_retained_receipt500.py)
    - [`catbird-intel/ops/gce/read_retained_continuous_status.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/read_retained_continuous_status.py)
    - [`catbird-intel/ops/gce/test_gcs_drive_retained_recovery.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/test_gcs_drive_retained_recovery.py)
    - [`catbird-intel/ops/gce/test_gcs_drive_retained_transport.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/test_gcs_drive_retained_transport.py)
    - [`catbird-intel/ops/gce/test_gcs_drive_retained_continuous.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/test_gcs_drive_retained_continuous.py)
  - **`netcup` Daily Export:**
    - [`catbird-intel/ops/export-day.sh`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/export-day.sh)
  - **Phase 3 Teacher Harness & Local Prototype:**
    - [`catbird-intel/teacher_harness/splits.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/teacher_harness/splits.py)
    - [`catbird-intel/teacher_harness/transport.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/teacher_harness/transport.py)
    - [`catbird-intel/teacher_harness/routing.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/teacher_harness/routing.py)
    - [`catbird-intel/teacher_harness/export.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/teacher_harness/export.py)
    - [`catbird-intel/teacher_harness/cli.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/teacher_harness/cli.py)
    - [`catbird-intel/teacher_harness/tests/test_teacher_harness.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/teacher_harness/tests/test_teacher_harness.py)
    - [`catbird-intel/ops/hydrate_pilot_holdout_context.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/hydrate_pilot_holdout_context.py)
    - [`catbird-intel/ops/run_local_clef_prototype.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/run_local_clef_prototype.py)
    - [`catbird-intel/eval/pilot_holdout_parent_quote_hydration_20261021.jsonl`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/pilot_holdout_parent_quote_hydration_20261021.jsonl)
    - [`catbird-intel/eval/pilot_holdout_parent_quote_hydration_20261021_manifest.json`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/pilot_holdout_parent_quote_hydration_20261021_manifest.json)
    - [`catbird-intel/eval/clef_flash_local_prototype_20261021.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/clef_flash_local_prototype_20261021.md)
    - [`catbird-intel/eval/clef_flash_local_prototype_20261021.json`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/clef_flash_local_prototype_20261021.json)

---

## 4. Subsystem-by-Subsystem Technical Summary & Audit Checklist

### 4.1 Subsystem A — GCE Capture, Drive Offload, Fault Recovery, Stage Release & VM Downsize

#### What Was Done
1. **Payload-Tail `HTTP 500` → `HTTP 410 Gone` Recovery (`seg_00000002vh.jss`, index `3725`, `run_id=be320fc55216e616e563be8b5fd7a76e`):**
   - Pre-recovery forensic snapshots saved on VM at `/data/checkpoints/drive-payload500-pre-recovery-20261021T1500Z` and `/data/checkpoints/drive-payload410-pre-recovery-20261021T1618Z`.
   - When the original resumable session URI returned `HTTP 410 Gone` (`expired_session`) and exact-ID Drive lookup (`files.get(fileId=11k3B5M5zQ5O8R7e_s2F5g1L9v3H8w2cT)`) returned `404 File not found`, recovery proved the partial payload session had expired uncommitted on Google's side.
   - Extended `payload_tail_and_receipt_auto` in [`gcs_drive_retained_recovery.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/gcs_drive_retained_recovery.py) to open a fresh resumable upload session with the pinned payload `file_id` (`11k3B5M5zQ5O8R7e_s2F5g1L9v3H8w2cT`), upload all `269,805,079` bytes, verify full GCS and Drive SHA-256 + MD5 readback, upload + verify the `1,093`-byte receipt (`1wwn1QUFRr3BkISm_AwkYqQbfcA4l13dN`), preserve the original failed attempt row in `retained_attempt`, record all recovery charges (`286,204,743` total recovery bytes across 4 historical reconciled attempts) in `retained_receipt_recovery_attempt` / `retained_reconciliation`, repair `journal.json`, and publish the immutable GCS mapping object in lock order.
2. **In-Session Transient `5xx` Status Probe (`bytes */total`) & Cross-Boot Pacing:**
   - Added bounded in-session `500/502/503/504` status probing (`Content-Range: bytes */total`, up to 3 probes with `1s/2s/4s` backoff) in [`gcs_drive_retained_transport.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/gcs_drive_retained_transport.py), recording every probe and retry byte in the SQLite authority ledger before attempting to resume or failing closed.
   - Fixed cross-boot monotonic request pacing so `last_request_wall_epoch` in SQLite is converted into a monotonic floor (`time.monotonic() + (wall_wait_until - time.time())`), preventing `time.monotonic()` vs epoch unit mixing after a VM reboot.
3. **Verified Local Stage Duplicate Release (`release_closed_pair_stage`):**
   - Previously, `/data/gcs-drive-retained-continuous/stage/<pair_key>/` retained `~268.5 MiB` per closed pair on the VM's `197 GiB` root disk, which caused `free_after_reservation_bytes` to hit the `100 GiB` floor (`free_space_below_100gib`) after `135` closed pairs (`~35.6 GiB` of stage copies).
   - Updated [`gcs_drive_drain.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce/gcs_drive_drain.py) so that **only after** a pair is `status == "closed"` in SQLite, its `journal.json` has `mapping_published == true`, and its `mapping_object_generation` is recorded, the local duplicate `.jss` and `.receipt.json` inside `stage/<pair_key>/` are unlinked and `stage_released.json` is written.
   - GCS (`gs://catbird-436600-jetstream-buffer-20261020`) and Google Drive copies remain 100% untouched. Local `staged_bytes` dropped from `35.63 GiB` to **`~11.8 MiB`**, freeing `~36 GiB` and raising VM free disk from `112 GiB` (`41%` used) to **`148 GiB` (`22%` used)**.
4. **VM Right-Sizing (`n2-highmem-32` → `n2-highmem-8`):**
   - Stopped `jetstream-selfhost-test`, resized in-place to `n2-highmem-8` (`8 vCPUs`, `64 GiB RAM`, `$7.54/day` Spot compute), configured Docker container `js` with `--memory=52g` (`55,834,574,848` bytes), `GOMEMLIMIT=46GiB`, `GOGC=100`, `--backfill-global-downloads=64`, `--backfill-host-workers-max=2`, `--backfill-max-active-hosts=32`, `--backfill-async-flush-workers=8`, `--failed-repo-retry-host-workers=1`, `--pending-repo-pass-workers=32`, restarted, and verified zero sequence gaps (`seq = 1,458,283,166` match).
5. **Backfill Throughput Root Cause & Independent Network Reconciliation (`catbird-intel/eval/pds_account_count_reconciliation_20261022.json`):**
   - Verified via `http://172.17.0.2:8080/status?tab=hosts` and `https://bsky.network/xrpc/com.atproto.sync.listHosts` (`6,638` PDS hosts; reconciled network total **`46,231,697` accounts** via `sum_host max(relay_accountCount, collector_enumerated_repos)`) that `89` of `90` `*.host.bsky.network` shards have already exited their `runBootstrap` `listRepos` loop (`74` drained, `14` marked `exhausted` on `us-west` after `HTTP 429`, `1` on `spaces-alpha`). Only **`1` shard (`jellybaby.us-east.host.bsky.network`, `804,839 / 1,143,604` enumerated, `338,763` remaining at `12:05Z`)** remains active in `runBootstrap`, capped by `--backfill-host-workers-max=2` (`~10 repos/sec = ~36,000 repos/hr`).
   - Across all `90` `*.host.bsky.network` shards, **16 shards** have unenumerated shortfall (`14` exhausted `us-west` shards + `jellybaby.us-east` + `spaces-alpha`, totaling `3,725,550` unenumerated accounts on `*.host.bsky.network` and `4,102,596` across all `6,638` hosts).
   - Once `jellybaby.us-east` finishes (`~21:30Z` Oct 22), `runBootstrap` automatically transitions to `runMerge` (`runPendingRepoRetryPass` at `32` workers over `1,395,264` `Pending` repos + replay of `112.7M` buffered live events + `runMergeDiscovery` with 3 retry attempts on exhausted hosts) and then `runSteadyState` (`RunFailedRepoRetry` every 4h over `3,811,972` `Failed` repos).

#### Live State Snapshot (`2026-10-22 12:40 UTC`)
- **Services:** `catbird-drive-retained-continuous-20261021.service = active`, `catbird-gcs-buffer.service = active`, Docker `js = Up`.
- **Google Drive Offload Ledger:** **`200` closed pairs**, **`0` failed unresolved attempts** (`1` active in-flight transfer, `4` historical reconciled), **`54,582,995,284` bytes (`54.58 GB`)** retained charged.
- **GCS Buffer Inventory (Archive of Record):** **`2,088` durable pairs** (`563,874,485,306` bytes / **`563.87 GB`** in `gs://catbird-436600-jetstream-buffer-20261020`), `3` local segments.
- **VM Disk:** **`148 GiB` free (`22%` used)**, `staged_bytes = 11.78 MiB`.

#### Reviewer Verification Commands (Subsystem A)
```bash
# 1. Run local GCE unit test suite (244 tests across recovery, transport, continuous, and drain)
.venv/bin/python -m unittest \
       catbird-intel/ops/gce/test_gcs_drive_retained_recovery.py \
       catbird-intel/ops/gce/test_gcs_drive_retained_transport.py \
       catbird-intel/ops/gce/test_gcs_drive_retained_continuous.py \
       catbird-intel/ops/gce/test_gcs_drive_drain.py

# 2. Inspect live GCE status and SQLite authority ledger (read-only)
gcloud compute ssh jetstream-selfhost-test --project=catbird-436600 --zone=us-west1-b --command='
  sudo /opt/catbird-gcs-buffer/venv/bin/python /opt/catbird-drive-retained-continuous-20261021/read_retained_continuous_status.py
  df -h / /data
'
```

---

### 4.2 Subsystem B — `netcup` Phase 0 Soak & Daily Export Sort Bounding

#### What Was Done
- Inspected why `/var/lib/catbird-intel/export/2026-10-20.partial` failed at `00:31:28 UTC` on `2026-10-21`: ClickHouse threw `Code: 241. DB::Exception: (total) memory limit exceeded: would use 5.59 GiB` during the `SELECT * FROM catbird.engagements ... ORDER BY seq INTO OUTFILE` query.
- Patched [`catbird-intel/ops/export-day.sh`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/export-day.sh) to append `SETTINGS max_bytes_before_external_sort = 1500000000, max_memory_usage = 4000000000, max_threads = 2` to both sorted `catbird.posts` and `catbird.engagements` export queries.
- Deployed `/opt/catbird-intel/ops/export-day.sh` to `netcup`, archived the failed partial directory to `/var/lib/catbird-intel/export/_trash/2026-10-20.partial.failed-oom-20261021T0031Z`, and ran `export-day.sh 2026-10-20`.
- Both `day=2026-10-20` (`14` Parquet files + `manifest.json`, `2.00 GB`) and the automatic `00:30 UTC` `day=2026-10-21` export (`14` Parquet files + `manifest.json`, `2.18 GB`) sealed cleanly without OOM or ingest disruption.
- **`catbird-intel-ingest.service` 72-Hour Soak:** Reaches `72h` continuous uptime at **`2026-10-22 13:00Z`**, with **`158–177 GB` free disk (`64–68%` used)** and all 15 production services untouched.

#### Reviewer Verification Commands (Subsystem B)
```bash
ssh netcup '
  systemctl is-active catbird-intel-ingest.service
  ls -lh /var/lib/catbird-intel/export/2026-10-20/manifest.json /var/lib/catbird-intel/export/2026-10-21/manifest.json
  df -h /
'
```

---

### 4.3 Subsystem C — Phase 1 Snapshot-Aware Follow Fold & Phase 3 Teacher Harness (`Clef-Flash`)

#### What Was Done
1. **Snapshot-Aware Multi-Generation Follow-Graph Fold (`catbird-intel/graph/fold_follows.sql`, `catbird-intel/graph/simgraph/src/main.rs`):**
   - Assigned `source_tier = 1` to the August 4 seed (`archive-graph`) and `source_tier = 2` to the October GCE capture (`gce-archive-graph`).
   - Enforced per-account `sync` and `deleted` reset barriers ordered by `(source_tier, time, seq)` and replaced `source_tier = 1` state whenever an account has an October snapshot (`october_gce_snapshot`) so unfollows between August 4 and October 19 are not resurrected as zombie follows.
   - Retained August 4 seed + incremental live October commits for accounts missing an October snapshot and labeled them `august_seed_fallback` (`is_stale = 1`) in `account_coverage.tsv` and `follower_weights.tsv`.
   - Fixed `g.inactive` to order by `(source_tier, time, seq)` instead of raw `seq` across archives.
   - Added `--follower-weights` / `--use-follower-weights` support in `simgraph` and generated `catbird-intel/eval/pds_host_coverage_weights_20261022.tsv`.
2. **Read-Only Parent/Quote Hydration from `netcup` (`catbird-intel/ops/hydrate_pilot_holdout_context.py`):**
   - Hydrated **`1,484 / 1,548` (`95.87%`)** missing reply-parent and quoted-post texts (`96.27%` pilot, `94.43%` holdout) into [`catbird-intel/eval/pilot_holdout_parent_quote_hydration_20261021.jsonl`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/pilot_holdout_parent_quote_hydration_20261021.jsonl).
3. **Fixing `Cloudflare/clef-flash` (`JointSchemaHead`) `noul` Criteria & Prose-Substance Calibration (`catbird-intel/teacher_harness/transport.py`):**
   - Added `_analyze_systemone_text_substance` (`media_visual_included`, `external_link_body_included`, `substantive_prose_word_count`, `is_link_or_mention_only`, `is_media_without_substantive_prose`) and rubric `boundary_rules` to `/v1/systemone` state envelopes and `SYSTEMONE_INSUFFICIENT_NOUL_CRITERIA`.
   - Verified on M4 Max (`catbird-intel/eval/clef_flash_local_prototype_20261021.md`) that bare URLs (`idx=24, 27, 53`), mention-only posts (`idx=26`), captionless image posts (`idx=30`), and image-dependent quotes (`idx=60`) are flagged `insufficient`, yielding **`52/64` (`81.25%`) all-component scorable retention** (`12/64 = 18.75%` insufficient on `>= 1` component) and **`92.6%` (`50/54`) binary agreement (`r = 0.8241`)** with `mlburnham/Political_DEBATE_large_v1.0` on `politicalness`.

#### Reviewer Verification Commands (Subsystem C)
```bash
# 1. Run all 18 teacher harness unit tests
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=catbird-intel .venv/bin/python -m unittest \
  discover -s catbird-intel/teacher_harness/tests -p "test_*.py" -v

# 2. Run all 26 graph guard, projection, and diagnostics tests (including ClickHouse local fold test)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=catbird-intel/graph .venv/bin/python -m unittest \
  discover -s catbird-intel/graph -p "test_*.py" -v

# 3. Run Rust simgraph unit tests
cargo test --manifest-path catbird-intel/graph/simgraph/Cargo.toml
```

---

## 5. Key Audit Questions for the Reviewing Agent

1. **GCE Stage Release Safety (`catbird-intel/ops/gce/gcs_drive_drain.py`):**
   - Verify that `release_closed_pair_stage()` cannot delete local stage files for any pair unless: (a) SQLite `retained_run.status == 'closed'`, (b) `journal.json` has `mapping_published == True` and `mapping_object_generation` set, and (c) the path resolves strictly inside `work_root / "stage"`.
   - Confirm that GCS (`gs://catbird-436600-jetstream-buffer-20261020`, the archive of record) and Google Drive copies remain untouched and that `staged_bytes` accounting accurately reflects only unreleased stage directories.
2. **Follow-Graph Snapshot Replacement & Staleness Provenance (`catbird-intel/graph/fold_follows.sql`):**
   - Verify that accounts with an October GCE snapshot (`source_tier = 2`) replace their August 4 seed (`source_tier = 1`) follow records so unfollowed edges are not resurrected, while accounts missing an October snapshot fall back to August 4 seed + incremental live commits and are labeled `is_stale = 1` (`snapshot_source = 'august_seed_fallback'`) in `account_coverage.tsv`.
3. **Phase 0 `netcup` 72-Hour Soak Completion (`2026-10-22 13:00Z`):**
   - Review the Phase 0 Done Criteria in [`2026-10-19-catbird-community-intelligence-master-plan.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md#L89-L93) and [`2026-10-20-catbird-intelligence-protocol-and-soak-contract.md`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/docs/design-docs/2026-10-20-catbird-intelligence-protocol-and-soak-contract.md).
4. **Phase 3 Teacher Harness (`catbird-intel/teacher_harness/`):**
   - Verify that `_analyze_systemone_text_substance` and `SYSTEMONE_INSUFFICIENT_NOUL_CRITERIA` (`transport.py`) flag bare-URL, mention-only, and captionless-media posts as `insufficient` while preserving scorable hydrated replies/quotes (`52/64 = 81.25%` scorable).
   - Check that no paid Anthropic API calls can occur when `--offline` is set (`cli.py`, `transport.py`).
