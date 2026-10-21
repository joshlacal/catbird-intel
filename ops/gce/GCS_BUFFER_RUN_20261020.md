# Temporary archive buffer: 20 October deployment

**Successor run:** Josh authorized a new $200 capture/recovery window on October 21. Capture has resumed from the preserved checkpoint; the [October 21 run report](GCS_CAPTURE_RESUME_20261021.md) and [active plan](../../../docs/execution-plans/2026-10-21-network-capture-resume.md) govern current operation. The no-restart/Drive-wait restrictions below describe the closed October 20 run.

**Post-deadline verification — October 21:** at **05:31:19 UTC**, after the original 05:29:29 UTC deadline, a fresh Compute Engine read confirmed **TERMINATED**. A fresh bucket inventory completed at **05:31:36 UTC**: **1,401 payloads, 1,401 generation/identity-matching receipt objects, 379,025,267,581 payload bytes, 379,027,611,753 total live-object bytes**, zero pairing mismatches. Object names, generations, sizes, CRC32C and custom metadata were unchanged from the 04:00 inventory; payload count and byte deltas were zero. Receipt bodies and payload contents were not reread. Both reads succeeded through the existing route; no jobs, settings, credentials, budgets or holds changed. Evidence: `gcs_vm_postdeadline_20261021.json`, `gcs_inventory_postdeadline_20261021.json`, and `gcs_postdeadline_summary_20261021.json` in the local graph-build output directory. Daily retention monitoring continues; the later-read requirement noted below is now satisfied.


**Run stopped early — Spot preemption:** Compute Engine records `compute.instances.preempted` at **2026-10-21 03:57:03 UTC** (October 20, 11:57:03 p.m. EDT). Fresh reads at approximately 03:59 and 04:00 UTC confirmed the VM was actually **TERMINATED**. It was not restarted. This is an early stop, not proof of completion at the originally scheduled 05:29:29 UTC deadline; a later read after that deadline remains scheduled.

A fresh bucket metadata inventory retained **1,401 payloads and 1,401 matching receipt objects**, with **zero generation/identity pairing mismatches**, **379,025,267,581 payload bytes** and **379,027,611,753 total live-object bytes**. There are 1,399 local-release markers. Compared with the last pre-stop supervisor sample at 03:44 UTC (1,391 verified pairs; 376,337,951,895 admitted bytes), this is ten additional pairs and 2,689,659,858 additional total bytes. Matching here means receipt object metadata references the existing payload generation and agrees on index, segment name, source ID and admission epoch; receipt bodies and payload bytes were not reread. Local active-file disposition and complete source coverage remain unverified.

Evidence under `~/catbird-lake-local/graph-build/out/`: `gcs_vm_stopped_20261021T0400Z.json`, `gcs_vm_stop_operation_20261021T0400Z.json`, `gcs_retained_inventory_20261021T0359Z.json`, and `gcs_stop_summary_20261021T0359Z.json`. Monitoring now runs daily for retention, including a fresh actual-state and bucket check after the original deadline. Drive and graph holds remain in place. Retained unique data is preserved; review by October 27 remains necessary because storage charges continue. No restart or additional compute is authorized.


**Collector recovery observed at 19:12 UTC:** the supervisor started the existing collector at **19:00:08 UTC** after verified backlog clearance. At 19:12 it was running with restart policy `no`; 581 verified pairs and 158,044,323,957 cumulative admitted bytes were recorded, with approximately 175 GiB free on the VM. Fresh segment rotations and cloud pair commits establish resumed collection/offload, not complete source coverage. A bounded sample of the latest 2,000 collector log lines (19:11:21–19:12:48 UTC) contained 1,951 repository failures after repeated upstream HTTP 429s, 13 PDS-host exhaustion messages, 14 verification failures and seven stream errors, alongside ten segment rotations. These are log-event counts, not distinct failed-repository counts or an overall failure rate. The buffer remains productive; upstream completeness and recovery need investigation. Drive timer/service remain held inactive; no graph resume or settings change occurred. Evidence: `gcs_buffer_monitor_20261020T1912Z.txt` and `gcs_collector_recovery_20261020T1913Z.txt` under the local graph-build output directory.

**Later owner hold, 20 October:** the GCS-to-Drive timer was stopped before its first 20:00 UTC run. The drain has no shared account-wide 700 GiB/24h reservation ledger or explicit global pacing, so it must not start under the newly surfaced hard quota invariant. Both timer and drain service were verified inactive with no drain journal entries. The GCS buffer and absolute VM stop deadline remain unchanged. See the [reconciliation and release conditions](../../eval/lakehouse_reconciliation_20261020.md#drive-drain-hold-and-quota-evidence); earlier scheduled-start statements below describe deployment history, not current permission.

The historical archive collector was blocked by Google Drive upload throttling. Josh authorized a temporary Google Cloud Storage queue using existing credits, with $15 for storage/operations and $60 for additional compute. The buffer is now running on the existing collector VM and transferring preserved segments before collection resumes. Drive transfer remains separately scheduled and unverified while its quota cools down.

Governing [buffer plan](../../../docs/execution-plans/2026-10-20-gce-archive-buffer.md), [master plan](../../../docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md), and [decision log](../../../docs/DECISIONS.md) live in the parent workspace.

## Deployed configuration

- Project `catbird-436600`, VM `jetstream-selfhost-test`, zone `us-west1-b`.
- Private single-region Standard bucket `catbird-436600-jetstream-buffer-20261020`, in `us-west1`; uniform access, public-access prevention, seven-day soft deletion.
- Dedicated `catbird-archive-buffer` service account, scoped to this bucket. No service-account key files.
- Existing n2-highmem-32 Spot VM and 200 GB balanced persistent disk retained.
- Absolute Compute Engine STOP deadline **2026-10-21 05:29:29 UTC**. Supervisor stops collection three minutes earlier. No automatic reboot or budget extension.
- Cumulative admission ceiling **1,099,511,627,776 bytes**, including observed soft-deleted generations. Stopping admissions does not stop storage charges.
- `catbird-gcs-buffer.service` started at **17:40:46 UTC**. Static unit, no restart policy or boot enablement.
- `catbird-gcs-drive-drain.timer` first fires **20:00 UTC**; ten-minute schedule within this dated run, eight segments per attempt, durable one-hour quota backoff.
- Existing `js` container kept stopped with restart policy `no` during backlog transfer. Its image, configuration and checkpoint are preserved. Old Drive-only offloader remains disabled.

## Verification

All **44 focused offline tests passed**. Deployed units passed `systemd-analyze verify`. Google Cloud Storage SDK **3.17.1** runs in an isolated Python 3.11 environment; exact dependency versions are recorded on the VM at `/opt/catbird-gcs-buffer/requirements-frozen.txt`.

The real one-segment smoke test uploaded index **3671**, SHA-256 `868bce863d80e7e2660dbe835006c05eed6e24393113f57ea22abb58af613c98`, plus its receipt and local-release marker. Total cumulative admitted bytes after that test were **277,184,458**. A fresh process performed a dry-run inventory and recovered exactly one verified pair and one local release with the same cloud byte count.

Continuous supervision then reached sixteen verified pairs and **4,356,981,179 cumulative bytes** by **17:43:03 UTC**, with 502 sealed segments still pending. Filesystem free space had increased to approximately 50 GiB. Collection had **not resumed** at that observation. The supervisor requires a fresh zero-pending scan, no more than three remaining local segments and at least 100 GiB free before starting collection. It stops the writer on uploader failure, capacity exhaustion, low disk, collector exit or deadline.

Deployed source SHA-256 values:

| File | SHA-256 |
|---|---|
| `gcs_archive_buffer.py` | `f9c152b134202200e80043ae3311fb9df29b2cd63b0c725336a0e481bf2ec03b` |
| `gcs_buffer_supervisor.py` | `155cd5fad05fa01b6e2f1107ec9ce021560dc3952905a8597151cc76dd3b6243` |
| `gcs_drive_drain.py` | `37601123c53e6c209944409b9b48e9b3738f0bc1ca71146f741b1c932ed970f1` |

Local evidence is under `~/catbird-lake-local/graph-build/out/`: `gcs_buffer_deployment_20261020.json`, `gcs_buffer_compute_prices_20261020.json`, and `gcs_buffer_runtime_20261020.txt`. These contain settings, the live regional pricing catalog snapshot, runtime progress and dependency versions.

## Cost and remaining checks

The live catalog quoted Spot N2 CPU at $0.01896/core-hour and RAM at $0.002542/GiB-hour: approximately **$1.257472/hour**, or **$15.09 for twelve hours**. Existing disk charges remain separate. One TiB of regional Standard storage for seven days is approximately **$4.71**; seven additional days of soft-deleted retention could roughly double it. These estimates are not a billing cap.

The fifteen-minute recovery monitor now follows this GCS run rather than the superseded Drive-only restart procedure. It must report actual collector resumption, uploader stalls, quota failures or the bounded stop. After stop, retain daily storage checks and review retention by **27 October**. Do not restart the VM automatically.

Drive drain completion, resumed collector output, complete historical coverage and original source-generation provenance remain unverified. The admission epoch is only a queue namespace. Verification covers header/footer XXH3 and whole-file hashes; decompressed block CRC validation remains unperformed. Existing local graph and decoder processes are unchanged.

Final review also corrected expired Drive resumable-session handling (preserve the segment as a conflict and continue others) and interrupted cleanup after a completed cloud transfer (reverify Drive and clean only matching owned temporary files). These fixes have offline regression coverage; live Drive completion remains pending.
