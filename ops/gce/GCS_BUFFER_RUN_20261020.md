# Temporary archive buffer: 20 October deployment

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
