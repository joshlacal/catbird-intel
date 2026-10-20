# Proposed bounded Drive release

The temporary GCS buffer protects archive segments while Drive uploads are held. The new drain is stopped because its local lock and quota-error backoff do not enforce Josh's account-wide 700 GiB rolling-day limit or shared pacing. This plan defines the smallest implementation and evidence needed for one later authorized transfer; it does not authorize uploading, restarting the timer, resuming the graph or extending the VM deadline. Unknown upload history is a blocker, not unused quota.

Governing [buffer plan](../../../docs/execution-plans/2026-10-20-gce-archive-buffer.md), [lakehouse goal](../../../docs/execution-plans/2026-10-19-catbird-25tb-lakehouse-and-embeddings-goal.md), [reconciliation](../../eval/lakehouse_reconciliation_20261020.md), and [decision register](../../../docs/DECISIONS.md). All design choices below are proposals, not deployed safeguards.

## 1. Current evidence and gaps

| Code or evidence | What it establishes | Missing requirement |
|---|---|---|
| `gcs_drive_drain.py:SafeDrive.request` | Authenticated sequential requests | No shared pacing or byte admission |
| `SafeDrive.ensure_file` | Preallocated file IDs, resumable offsets, checksum verification, expired-session conflicts | No durable pre-send reservation; retries/unknown outcomes not charged to a quota authority |
| `Drain.finish` | Independent Drive verification before generation-pinned GCS release | No explicit retain-GCS pilot mode; pilot must not execute releases under unresolved retention policy |
| `gcs_drive_drain.py:main` | Local flock, bounded staging, one-hour 403/429 backoff | No account-wide writer exclusion, rolling ledger or account identity check |
| `scripts/onboarding/warehouse/gdrive_lake_sync.py:RollingQuotaLedger` in parent workspace | Historical per-file accounting implementation | Separate local state; non-atomic check-then-record; parse failure resets to empty; no shared durable reservation authority |
| `ops/export-day.sh` | Per-process rclone pacing | Independent writer; no shared accounting/pacing. Do not execute this script during release planning; its cleanup behavior is outside this task. |
| `ops/upload_repair_v1.py` and underlying uploader | Explicit verified artifact uploads | Another independent path that must participate or remain held |
| Public Colab archive downloader | Single-writer prohibition remains binding; latest read-only server check showed zero sessions | This is only a point observation, not a durable exclusion of a later browser/notebook writer |
| Legacy GCE `/data/offloader/state.json` | 3,671 completed entries / 996,463,022,304 payload bytes in the observed preceding day | Cached reuse, incomplete/uncertain attempts, receipts and other writers prevent treating this as actual account-wide quota usage |

Retain these evidence files under `~/catbird-lake-local/graph-build/out/`:

- `drive_quota_observation_20261020T1839Z.json` — legacy ledger summary and limitations.
- `drive_drain_hold_20261020T1840Z.txt` — inactive drain/timer, no scheduled event or drain journal entries, active GCS buffer.
- `gce_recovery_20261020T1717Z.json` — real historical segment transfer followed by renewed quota errors.
- `gcs_buffer_deployment_20261020.json`, `gcs_buffer_vm_settings_20261020.json` — unchanged storage and absolute stop boundaries.
- GCS buffer journal, immutable object generations and their receipts — candidate identities, not evidence of Drive quota consumption.
- Existing repair-upload receipts `eval/drive_uploads_v1.jsonl`, legacy offloader service logs, any collector upload journals, rclone logs and notebook upload manifests — reconcile individually with writer/account and actual versus cached transfer status.

Drive object size/creation metadata and storage quota alone do not establish rolling uploaded bytes. They can corroborate named transfers, but cannot prove the absence of partial, repeated or unrelated uploads. If an account-wide recent-use source is unavailable, the initial state must conservatively deny admission.

## 2. Smallest enforceable ownership model

Use **one participating writer** for the first release: the existing GCE drain. Do not introduce a new service, VM, credential scope or second archive writer.

Before release, the owner inventories all writing paths using this Google account: old GCE offloader, new drain, public Colab downloader, local/manual Drive uploads, repair uploader, lakehouse/rclone jobs and any other known account integrations. Each must either use the same authority or be explicitly excluded from writing for the whole release window. A mere lack of current processes is insufficient for scheduled jobs.

The Colab downloader remains untouched. If it is active, its ownership is uncertain, or a future restart cannot be excluded through coordination, **do not acquire the drain's writer lease**. Do not stop that notebook to make room. Fresh checks and owner acknowledgment are required immediately before a release. If account-wide exclusion cannot be established without changing security posture, report the blocker; do not rotate or revoke credentials.

Phase-one authority can be a transactionally durable SQLite ledger on the existing GCE disk, because all other writers are held. Identify the Google account using an existing authorized identity endpoint or already verified account record; compare a stable nonsecret identity, not a token string. An unknown or mismatched account fails closed. A later multi-host design must route every writer through one shared authority; independent copies of SQLite or Drive-hosted lock files do not provide that guarantee.

Use a unique release ID, writer ID and process identity. Keep a single process-group/file lock plus a durable lease record. An expired lease does not release reservations or authorize takeover while old in-flight work might exist. A replacement first proves the old writer stopped and reconciles its reservations; otherwise it remains blocked. Before every request, verify ownership and remaining run time. No sending close enough to the absolute stop deadline that the configured request timeout would extend past the safe shutdown window.

## 3. Rolling byte accounting

User ceiling: **700 × 1024³ = 751,619,276,800 bytes**, over every rolling 24-hour window. This is a local upper bound, not a promise that the provider accepts that amount. Treat transfer rejection as authoritative and do not increase this limit to compensate.

Admission computes:

`charged sends in the previous 24h + all active reservations + all unresolved charges + new reservation <= ceiling`

Also require the smaller, separately authorized pilot byte ceiling. All timestamps are UTC; clock regression or inconsistent persisted times fail closed. No startup, corrupt database, stale lease or missing journal resets usage to zero.

For each candidate:

1. Pin payload generation, receipt generation, exact sizes and hashes, destination parent, preallocated file IDs and release identity before any create/send.
2. Reserve the entire payload plus receipt, serialized create/control bodies, and a bounded control allowance **before** opening an upload session. Metadata-only lookup and existing-byte verification must not be confused with a new upload.
3. Persist and sync an attempt record before each body is sent. Convert reservation to conservatively charged bytes only once; retain enough reservation for remaining media/control traffic. Charge repeated payload bytes again unless authoritative evidence proves no bytes were sent. A separate reservation is required before a retry would exceed its budget.
4. An ambiguous response retains the charge and file/session identity. Query the original resumable session under the same pacing authority. Never “refund” because a local timeout occurred; never generate a replacement file ID merely to escape uncertainty.
5. Verified completion closes the remaining unsent reservation after checking Drive size/checksum and receipt identity. Already-existing matching files require no new-media charge, but their lookup traffic remains paced.
6. Only an attempt proved never sent can release its reservation immediately. Ordinary completed charges age out after their recorded sending window. Unresolved/in-flight amounts do **not** age out merely because a local start time is old; prove a conservative last-possible send/finish bound and writer quiescence or retain the charge.

Reconciliation options:

- **Evidence route:** obtain complete actual/uncertain histories for all writers, deduplicate stable attempt identities without erasing retransmissions, and conservatively import them.
- **Quiescence route:** establish an account-wide no-upload interval of at least 24 hours after the latest possible send/completion, with no outstanding sessions capable of sending or finishing later. This is not established merely by waiting since the last known success or since this drain was held.

The current legacy ledger cannot satisfy the evidence route alone. A newly started 24-hour quiescence window would outlast this VM's existing stop deadline; this plan grants no extension or reboot. GCS retention remains monitored while a later compute decision is pending.

## 4. One shared pacing authority

Pace **all Drive API requests**, including file/folder creation, resumable initiation, status queries, chunks, retries and verification. This is more conservative than pacing only creations and makes the pilot audit simple.

Permit at most one request start every **0.55 seconds**, with no burst capacity. Persist the next eligible slot before dispatch; use a monotonic clock within a process and conservative wall-time recovery after restart. Shared serialization must cover concurrent callers. A restart, retry or new file must not reset the schedule. Record timestamp, method, operation kind and request ID without secrets or resumable URL tokens.

Provider 403/429 stops further sends, records a cooldown, preserves ambiguous reservations and exits. No rapid automatic retry, no timer restart and no second writer. HTTP 404/410 for a resumable session remains a preserved conflict requiring reconciliation.

## 5. Exact implementation surface, not yet implemented

- New `ops/gce/drive_transfer_authority.py`: durable ledger schema, account identity, release lease, reservations, request pacing and reconciliation/import report.
- `gcs_drive_drain.py`: inject authority into every `SafeDrive.request`; reserve pair/session identities before mutation; account uncertain/repeated chunks; require explicit release ID and approved budget; add `--retain-gcs` and a strict pilot selection manifest.
- Default behavior must reject live uploads without a validated authority/release record. Do not make the historical unmetered path an implicit fallback.
- Keep the current timer stopped. Initial validation invokes one explicit bounded process rather than enabling a recurring timer.
- Other uploaders remain held during the pilot. Adapting each is separate work; the old ledger must not be imported as if it were complete.
- Preserve all source and failed artifacts. No changes to OAuth, graph processes, public archive notebook, raw-image collection or production.

## 6. Smallest offline validation

Use fake clocks, a temporary SQLite database, tiny numeric payloads and a fake HTTP transport; no credentials, network, model loading or large archive reads. Add focused tests for:

1. Exact cap boundary and overlapping rolling windows, including receipt/control overhead.
2. Two concurrent reservations cannot jointly exceed the ceiling.
3. Crash after reservation/before send, after send/before acknowledgment, and after verified completion/before ledger close.
4. Lost response, resumable offset recovery and repeated chunk accounting.
5. Corrupt/missing state, clock regression and stale ownership fail closed.
6. Lease takeover denied while an older attempt remains unresolved.
7. Request starts from concurrent callers, retries and restarts never violate pacing.
8. Provider quota error stops all sends and retains uncertain charges.
9. Matching preexisting file does not charge new media; conflicting file preserves data.
10. Pilot preserves all GCS generations and local artifacts, even after successful Drive verification.
11. Colab ownership conflict or unknown writer status blocks the pilot.
12. Deadline/run-budget exhaustion prevents new sends and saves a recovery record.

Pass criteria: deterministic ledger balances after every injected interruption; complete attempt provenance; no cap overrun, duplicate create or unpaced request; no retained-object deletion. These tests establish implementation behavior only, not current Drive readiness.

## 7. Evidence for one small authorized transfer

Prepare an approval packet containing:

- Exact deployed source hashes and offline test receipt.
- Current account identity, writer exclusion evidence, reconciled rolling usage and unresolved reservations.
- One existing sealed GCS payload **at most 300 MiB**, its matching receipt and an already verified Drive parent. Exact names, generation IDs, hashes and byte sizes are frozen in the packet; no arbitrary “next segment” selection.
- One-run ceiling **320 MiB total conservatively charged/reserved**, at most **20 minutes**, one segment, no automatic retries, no recurring timer. Abort rather than exceed control allowance.
- `--retain-gcs`, no local cleanup and no source reclassification. This pilot proves copy/verification only.
- Enough time remains before the unchanged VM shutdown deadline; otherwise do not launch.

Only explicit approval of that concrete packet releases the single transfer. Success requires payload **and receipt** uniquely present in Drive with matching size/checksum, persisted IDs, a verified receipt-to-generation link, complete reservation settlement and a recorded pacing trace. A small metadata upload does not count.

One successful pair is a bounded capability result, not account-wide quota recovery or permission for a batch. The graph's existing readiness condition also requires a later successful real transfer without renewed quota errors, adequate Mac disk headroom and explicit graph-resume authorization. Any later transfer or timer re-enable needs its own scope/release decision.

## 8. Blocking decisions remain separate

The immediate blocker is authoritative recent account-wide usage and enforceable writer exclusion, not additional storage capacity. Implementation can be designed/tested offline while this remains unresolved.

The physical-scrub versus preserved `_trash` retention conflict is still Josh's decision. The proposed pilot retains all GCS/local copies and does not execute any scrub. The $15 storage/operations, $60 additional compute, 1 TiB admission ceiling, $250.82 reserve and **2026-10-21 05:29:29 UTC** VM stop remain unchanged.
