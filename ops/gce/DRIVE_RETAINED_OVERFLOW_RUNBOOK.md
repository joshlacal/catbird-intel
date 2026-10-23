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
python3 -B -m unittest -v test_gcs_drive_retained_runner.py
python3 -B -m unittest -v \
  test_gcs_drive_retained_overflow.RetainedOverflowTests.test_linux_watchdog_kills_worker_and_grandchild
python3 -B -m unittest -v \
  test_gcs_drive_retained_runner.RunnerTests.test_linux_child_timeout_kills_nested_session \
  test_gcs_drive_retained_runner.RunnerTests.test_linux_runner_death_kills_supervisor
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

## Serial snapshot runner

The separate runner accepts one saved, SHA-256 reviewed snapshot packet. Save
the selector's single JSON output as `SNAPSHOT_JSON` in `SNAPSHOT_DIR`, hash
those exact bytes, and review every manifest. Place owner-prepared evidence
for each run at `EVIDENCE_DIR/RUN_ID.json`. Each file must bind that run,
manifest digest, account, writer, and a writer exclusion through its deadline;
`observed_at` expires after five minutes. The runner never asserts these facts
from a command-line flag. If fresh evidence is absent it records
`awaiting_fresh_evidence` and starts no next pair.

```sh
python3 -B gcs_drive_retained_runner.py \
  --snapshot-json SNAPSHOT_JSON --snapshot-sha256 REVIEWED_SHA256 \
  --snapshot-dir SNAPSHOT_DIR --authority-db LEDGER \
  --evidence-dir EVIDENCE_DIR --account-id DRIVE_PERMISSION_ID \
  --writer-id WRITER_ID --oauth-file EXISTING_OAUTH_FILE \
  --status-file /data/gcs-buffer/retained_runner_status.jsonl
```

Read the latest complete valid status line for this run from
`/data/gcs-buffer/retained_runner_status.jsonl` and inspect the retained
per-pair log after any stop. `tail -n 4` is a quick view; a crash can leave
an incomplete last line, so monitoring must parse backward to the last
complete valid line with the current `run_id`. `state`, `stop_reason`,
`pairs_closed`, `last_pair`, `last_mapping_sha256`, `cooldown_until`, and
`updated_epoch` are the key fields. The collector can be healthy and
`uploading` while the retained runner is active; do not require collector
idleness.

One invocation runs at most eight pairs and 8 × 320 MiB of reserved transfer
bytes, serially against the same imported ledger. It starts each pair only
after read-only authority inspection shows no nonterminal run, unresolved
attempt, quota cooldown, duplicate source, or exhausted 32 MiB mapping
partition. A child exit alone never permits the next pair: the runner requires
the exact durable `closed` mapping and another account inspection. A 403/429,
uncertain outcome, or failed child stops the batch without a replay. The
absolute capture STOP is 2026-10-24 10:10 UTC, and the runner does not begin a
pair without its full reviewed duration plus stop margin. Its Linux child
watchdog and parent-death guard terminate the transfer process tree if the
runner times out or dies.

The runner measures the same filesystem as `/data/jetstream/segments`, caps
cumulative retained staging at 32 GiB, and reserves a 100 GiB free floor for
capture before and during each pair. The worker checks the floor again before
each GCS download; the runner checks it every ten seconds while a child runs,
with an allowance for the remaining pair bytes. Concurrent capture writes can
occur between checks, so this is periodic fail-closed monitoring rather than
an atomic free-space guarantee. The runner appends and fsyncs its own
`retained_runner_status.jsonl` heartbeat and writes per-pair logs; it does not
overwrite the collector supervisor status. The monitor reads the last complete
valid JSON line for the current run and computes heartbeat age from
`updated_epoch`; a torn final line remains for inspection. The runner never
deletes GCS objects, local staging,
Drive copies, collector data, or checkpoints. The collector may remain active
while the runner copies. Generated-cache cleanup is a separate reviewed
protocol; a candidate cache file could be removed only after exact original
GCS generation and immutable Drive ID mapping verification, and such removal
would not refund cumulative GCS admission. No cleanup is implemented here.

The selector covers current live receipt metadata only. A later bounded
snapshot can follow after the previous one is closed and reviewed, but an
empty snapshot does not prove complete historic or soft-deleted coverage.
After a pair closes, take a fresh selector snapshot for the remaining work;
for the current seven-candidate batch, use `--max-pairs 7` after segment 3671
is durably closed. An old snapshot containing the closed pair is rejected to
avoid a second copy.
Any `active`, `verified`, or `index_pending` pair, `file_upload_inflight`, or
uncertain index create requires manual reconciliation; the serial runner will
not auto-recover it.

## Collector `meta.pebble` Checkpoint Procedure (Mind the Supervisor Poweroff)

`catbird-gcs-buffer.service` runs `gcs_buffer_supervisor.py` with
`--poweroff-on-exit`. Therefore, running `docker stop js` causes the buffer
supervisor to detect container exit and **immediately power off the GCE VM**
(`shutdown -h now`). Never stop `js` while the Drive writer is mid-pair.

When taking a cold `meta.pebble` checkpoint (such as Checkpoint #2 after
failed-repo retries finish in `steady_state`), follow this exact sequence:

1. **Quiesce the Drive writer on a closed-pair boundary:**
   Wait until the current pair finishes and closes (`status='closed'` in
   `authority.sqlite` and `nonterminal_run_id` is `None`), then stop the writer
   service cleanly before touching `js`:
   ```sh
   sudo systemctl stop catbird-drive-retained-continuous-20261021.service
   ```
2. **Disable Docker auto-restart on `js` and stop the collector container — expect VM poweroff:**
   ```sh
   sudo docker update --restart=no js
   sudo docker stop -t 60 js
   ```
   Wait for `jetstream-selfhost-test` to transition to `TERMINATED` (`gcloud compute instances describe jetstream-selfhost-test --zone=us-west1-b --format='value(status)'`).
3. **Start the VM and copy `meta.pebble` before the collector restarts:**
   Because `catbird-gcs-buffer.service` is `static` (not enabled at boot) and `--restart=no` was set on `js`, `js` remains `exited` on boot and `/data/jetstream/meta.pebble` is completely cold. Copy `/data/jetstream/meta.pebble` into `/data/checkpoints/<checkpoint-name>/`:
   ```sh
   gcloud compute instances start jetstream-selfhost-test --project=catbird-436600 --zone=us-west1-b
   # On VM while js is exited:
   sudo mkdir -p /data/checkpoints/<checkpoint-name>
   sudo cp -a /data/jetstream/meta.pebble /data/checkpoints/<checkpoint-name>/meta.pebble
   ```
4. **Restart the buffer supervisor (which starts `js`), verify cursor/counts, export snapshot, and upload:**
   - **Important:** Do **not** run `docker start js` manually before starting `catbird-gcs-buffer.service`. `gcs_buffer_supervisor.py` asserts that `docker inspect js` is in state `exited no` (`expected stopped collector with restart disabled`), verifies the local segment backlog, and then runs `docker start js` itself (`collector_started_after_verified_backlog`). Starting `js` first or setting `--restart=unless-stopped` causes `gcs_buffer_supervisor.py` to fail preflight and immediately trigger `--poweroff-on-exit`.
   - Start `catbird-gcs-buffer.service` and wait ~5 seconds for it to start `js`, then verify cursor, phase, and repo counts (`Discovered`, `Downloaded`, `Pending`, `Errored`, `Unavailable`) match pre-stop values.
   - Restart `catbird-drive-retained-continuous-20261021.service` with `--max-concurrent-pairs 1`.
   - Run `repo_snapshot_export` **against the checkpoint copy only** (`/data/checkpoints/<checkpoint-name>/meta.pebble`, never `/data/jetstream/meta.pebble`), compress with `zstd -T0`, record row count and SHA-256, and upload to `gs://catbird-436600-jetstream-buffer-20261020/sources/gce-selfhost-001/buffer-admission-20261020/snapshots/`.

