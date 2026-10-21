# Retained GCS to Drive overflow: review packet

This packet is an offline review artifact. Josh has directed resumed bounded
offloading; the sole GCE owner coordinates activation after the required gates
and review. This file itself starts no VM, writer, graph processing, or OAuth
change. The collector's GCS path and its budget are owned by the GCE owner.
The old eight-segment drain and legacy offloader remain held.

## One-time ledger preparation (no cloud calls)

Run these on the approved Linux runtime after inspecting the deployed files and
choosing a stable, backed-up SQLite path. `RESULT` must be the sanitized,
hash-pinned successful segment-3677 result. The initializer never overwrites
an existing ledger, and the import leaves the original result/database intact.

```sh
python3 -B retained_transfer_authority.py initialize \
  --db LEDGER --writer-id WRITER_ID --account-id DRIVE_PERMISSION_ID
python3 -B retained_transfer_authority.py import-legacy \
  --db LEDGER --writer-id WRITER_ID --account-id DRIVE_PERMISSION_ID \
  --result RESULT
```

The imported result records 51 settled Drive attempts and 268,982,001 charged
bytes. It is charge evidence, not proof of complete account history. Segment
3677's payload generation is excluded from a second retained copy. Keep the
original diagnostic database and files.

## Offline and no-cloud checks

```sh
python3 -B -m unittest -v test_gcs_drive_retained_overflow.py
python3 -B -m unittest -v test_gcs_drive_retained_queue.py
python3 -B -m unittest -v \
  test_gcs_drive_retained_overflow.RetainedOverflowTests.test_linux_watchdog_kills_worker_and_grandchild
python3 -B gcs_drive_retained_overflow.py \
  --overflow-manifest MANIFEST --run-id RUN_ID \
  --account-id DRIVE_PERMISSION_ID --check-manifest
```

The Linux watchdog smoke starts only local synthetic processes and verifies a
hung worker and grandchild are stopped. The manifest check reads only a local
JSON file. A full independent code review and target Linux smoke are required
before the live path is used.

## Bounded candidate snapshot

The snapshot selector reads at most 5,000 GCS receipt entries, emits at most
eight generation-pinned manifests, and reserves at most 8 × 320 MiB for one
snapshot. It never starts a Drive writer. Choose smaller values when practical.

```sh
python3 -B gcs_drive_retained_queue.py \
  --account-id DRIVE_PERMISSION_ID --writer-id WRITER_ID \
  --authority-db LEDGER --work-dir SNAPSHOT_DIR \
  --max-scan 5000 --max-pairs 8 --max-reserved-bytes 2684354560 \
  --max-run-seconds 1200 --max-snapshot-seconds 300
```

Review each emitted manifest and its raw SHA-256 digest. Exclude a candidate
whose payload generation is already completed or whose source identity is
uncertain. The selector is a bounded snapshot, not an unattended upload loop.
The imported segment-3677 diagnostic marker excludes its payload generation
from this queue without qualifying it as an archive shard. The snapshot lists
current live receipt metadata. An empty result is not proof that historical or
soft-deleted generations are covered; a separate coverage reconciliation is
needed before claiming full backfill completeness.

## One reviewed pair

Before each invocation, the Drive owner must verify the OAuth permission ID,
the app-created root folder, one account-wide Drive writer through the run
deadline, the held legacy offloader/drain, and a fresh evidence file bound to
this exact manifest digest, run ID, writer ID, account ID, and deadline. The
evidence states historical quota use is unproven and that a quota rejection
is accepted. No broader OAuth scope is needed. Keep all GCS and local source
copies. Review collector headroom and the additive fixed 32 MiB GCS mapping
partition; each immutable index object is limited to 16 KiB and every attempted
create is charged durably without a refund.

Use the *same* imported SQLite ledger for every invocation. `WORK_DIR` must be
exactly `/data/gcs-drive-retained-RUN_ID`. The deadline is an absolute Unix
time no more than 20 minutes ahead; the Linux supervisor kills the worker 30
seconds before it. Each invocation copies at most one payload and receipt.

```sh
python3 -B gcs_drive_retained_overflow.py \
  --overflow-manifest MANIFEST --run-id RUN_ID \
  --account-id DRIVE_PERMISSION_ID --writer-id WRITER_ID \
  --authority-db LEDGER --review-evidence FRESH_EVIDENCE_JSON \
  --deadline-epoch DEADLINE_UNIX_SECONDS \
  --work-dir /data/gcs-drive-retained-RUN_ID \
  --oauth-file EXISTING_OAUTH_FILE \
  --once --retain-gcs --quota-observation
```

For a stopped pair already marked `verified` or `index_pending`, verify the
prior process is dead and use the same manifest/run/work directory with fresh
evidence, a fresh bounded deadline, and `--recovery-only`. That mode cannot
send Drive requests; it only completes the immutable GCS mapping. A terminal
`closed` pair can use the same flag to rebuild the optional local index cache
from the shared GCS mapping through the read-only authority inspector.

Exit 75 on Drive HTTP 403/429 means stop the queue for the ledger cooldown.
An unresolved attempt or `file_upload_inflight` needs pinned ID/session and
charge reconciliation before any new pair; do not simply rerun or use another
writer. An unanswered GCS index create is read by its exact object name and
compared byte for byte; if it is absent, stop for manual review without a
second create. The shared mapping is a create-only GCS object under
`indexes/drive-retained-overflow/v1/`; the VM-local JSONL is a cache. This
mapping remains unqualified for the historical Drive archive shard.

Reader integration for Colab and Mac must consume the shared GCS mapping's
exact Drive IDs and source generations. Name-based Drive search is not an
archive qualification under `drive.file` visibility.
