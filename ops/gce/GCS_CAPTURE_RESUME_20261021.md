# Network capture resumed on October 21

The cloud collector is again backfilling Bluesky repositories and recording live events for later analysis in Colab and on the Mac. It resumed from the preserved checkpoint, and new immutable segments are reaching Google Cloud Storage (GCS). A real Google Drive test also uploaded and verified one complete payload/receipt pair, while retaining the originals. This establishes working capture and transfer routes; repository verification failures and incomplete history remain coverage limitations. The next step is a reviewed, paced Drive queue while GCS capture continues.

Authority: Josh's October 21 instruction to continue capture, recover the owned cloud VM when necessary, use Drive or overflow storage, and spend at most a new **$200**. Governing [capture plan](../../../docs/execution-plans/2026-10-21-network-capture-resume.md) and [decision log](../../../docs/DECISIONS.md#2026-10-21--network-capture--resume-through-gcs-overflow-within-200). This supersedes the earlier run's no-restart/Drive-wait restrictions; it does not resume the paused local graph or model jobs.

## Verified recovery

The existing Spot VM `jetstream-selfhost-test` in project `catbird-436600`, zone `us-west1-b`, retained its original 200 GiB disk. The stopped `/data/jetstream` tree was copied to `/data/checkpoints/jetstream-pre-resume-20261021T1014Z` before launch. The original container and image were retained. The image is `sha256:bb779a4831afab96eea1cf399469619772ca8eed69db6f3a5202ae858d393745`, built from Jetstream source `d4db8c08eb887dd148cd7c1d5f4a9fb327e15f06` (v0.4.4).

Read-only inspection of the copied Pebble database found bootstrap phase, saved relay cursor **34,690,804,147**, main next sequence **17,976,801,027**, and bootstrap-live next sequence **60,745,855**. The collector started at **10:23:29 UTC** and its startup logs matched all three values. Main segment **5072** was the preserved partial tail; the previous GCS inventory contained exactly 1,401 contiguous payload indices **3671–5071**. No phase, listing cursor, repository outcome, or sequence reset was performed.

The resumed collector uses 64 global downloads, two workers per host, 32 active hosts, eight flush workers, and one failed-repository retry worker per host. Compaction is disabled so already archived segment identities remain immutable and update/delete events remain available for downstream folding. Bootstrap live files remain in their separate local namespace for the collector's later merge; they are not given to the main-segment pruner.

The retained `.jss` format is already column-oriented and Zstandard compressed; “raw archive” here means original captured records/events, not uncompressed JSON or retained CAR proof trees. The pinned source implements column encoding in [`segment/block.go`](https://github.com/bluesky-social/jetstream/blob/d4db8c08eb887dd148cd7c1d5f4a9fb327e15f06/segment/block.go) and framed Zstandard compression in [`segment/writer.go`](https://github.com/bluesky-social/jetstream/blob/d4db8c08eb887dd148cd7c1d5f4a9fb327e15f06/segment/writer.go). Parquet conversion may help analytics, but further size reduction has not been measured against this representation. No conversion or graph job was resumed.

## Capture evidence

At **10:33:52 UTC**, the buffer reported **1,420 durable pairs**, **384,143,128,107 cumulative admitted bytes**, no pending sealed segment, three local main segments, and **171,966,210,048 free disk bytes** (about 160 GiB). This is 19 new durable pairs since recovery. The collector was running, using about 8.1 GiB RAM in this early sample; the bootstrap live tree occupied 6,335,202,890 bytes. These are observed early resource values, not peak-memory qualification.

The first new pair, index **5072**, has a 269,612,165-byte payload with SHA-256 `a1fedbe8e66e5443c013e5cbc24ea41a58aff450766e801b08a60a293b1a120b`, GCS generation `1792578254302714`. Its 1,093-byte receipt is generation `1792578255676712`, SHA-256 `07871da1b9ca13f91a73cd1995bfc225a24dcc83ee074c7676b4ee8db3da0a44`. Independent download of that small pinned receipt passed 15 checks against the retained payload metadata. The independent check did not download or rehash the payload; decompressed block CRC validation remains unperformed.

At approximately **10:31 UTC**, the live status reported:

| Counter | Value | Change from preserved checkpoint |
| --- | ---: | ---: |
| Relay cursor | 34,692,790,994 | +1,986,847 |
| Completed repositories | 23,742,830 | +141,767 |
| Failed repositories | 3,827,546 | -15,255 |
| Bootstrap live sequence | 63,477,887 | +2,732,032 |
| Main durable sequence | 18,030,200,406 | Progress observed |

No global sequence-gap/overflow warning appeared in the bounded startup or follow-up samples. That absence does not establish gap-free coverage. Chain-break warnings trigger asynchronous repository resync, with one resync per DID (repository identity) per ten minutes. The original commit can be withheld while stream tracking advances. Successful resync supplies current repository state, not every missed historical mutation; failed resync remains a coverage issue. A 39-second follow-up still saw five of the original eleven failing DIDs and two new ones, including deactivated-repository and signature failures. Verification remains enabled. The stored failed-repository count and evidence are preserved for repair; bootstrap and its later repair phase are not complete.

## Real Drive result

The retained-copy diagnostic ran from **10:30:49 to 10:31:21 UTC** and exited successfully. It fetched exact pinned GCS generations for segment **3677**, rehashed the source bytes, uploaded the 268,980,444-byte payload and 1,093-byte receipt, and verified final Drive metadata by pinned IDs. Post-run local hashes matched again. No independent Drive media download was performed.

- [Diagnostic folder](https://drive.google.com/drive/folders/1OKVFX9e-sqzPRwojIeLEVrjBc9f0PI6x)
- [Payload](https://drive.google.com/file/d/1lSYNMCdG4RDQo4YgyE9PMG6g36_pioYk/view)
- [Receipt](https://drive.google.com/file/d/1VMvXsxu3uiV0_ksb6DOGu_DsZfjg6VbT/view)

Release `edef464c6ef8489ba798c22b257da7b4` used manifest SHA-256 `3e5d09f19a2a5310e410bcc397afa60d20a23d74b025df33e6bd4c1c0fa23dee`. Its ledger recorded **51 requests, 268,982,001 charged bytes, zero unresolved attempts**, only HTTP 200/308 responses, and a minimum measured request-start spacing of **0.550615 seconds**. The charge was below the 270,030,113-byte reservation. All GCS and local copies remain.

The fresh app-created folder avoids collisions with unseen historical siblings under the existing `drive.file` OAuth scope. It remains explicitly `archive_qualified: false` and `history_proven: false`; success does not prove a full account-wide daily upload allowance or qualify the old archive shard. The existing unmetered eight-segment drain remains inactive. A sustained retained-copy queue is being implemented separately with shared pacing, cooldown, charges, immutable mappings, and restart handling. It has not been activated by this result.

A read-only `about.storageQuota` check at **10:41:12 UTC** matched the diagnostic account and reported 32,987,496,316,928 total bytes, 1,093,567,733,001 used bytes, and **31,893,928,583,927 available bytes** (about 31.9 TB decimal). This establishes storage headroom, not remaining daily upload allowance; the API does not return that allowance. Receipt: `drive_storage_quota.json`.

## Controls, tests and budget

The GCS supervisor has a per-run ID, fresh-heartbeat checks, shared-filesystem free-space guard, cumulative byte accounting without refunds, and local deadline shutdown. Its run is backed by an externally configured Compute Engine STOP at **2026-10-24 10:10:00 UTC**. Spot recovery never moves that boundary. The monitor runs every fifteen minutes and may recover an actual preemption of this same VM after checking disk, image, journal, budget and locks; it is capped at three recovery attempts per 24 hours. Ordinary crashes, stale output, space exhaustion and journal conflicts require diagnosis.

The existing bucket's software admission ceiling is **2.5 TiB** (2,748,779,069,440 bytes), including observed generations. No new storage purchase or bucket is needed. The plan reserves **$179.33** for up to 72 total running hours, 37 days of storage/disk allowance, IP and operations, leaving **$20.67** contingency. Catalog prices and inventory support the estimate; actual billed spend is not yet verified and this is not a provider-enforced dollar cap. Retained data continues billing after compute stops; no expiry deletion is authorized. Direct bulk GCS-to-Mac egress is outside this envelope.

The continuous queue has a provisional additional **32 MiB cumulative mapping partition**, within the existing contingency. Every mapping create must be reserved before dispatch, including uncertain outcomes. Because the main buffer sees those generations only on its next inventory, the worst-case aggregate bound is 2.5 TiB + 32 MiB; incremental reserved storage cost is below $0.001. This partition does not authorize deleting any GCS original.

The pasted estimate of an entire network in 1.5–3.8 TB is not a measurement of this archive. At the same 72-hour compute and 37-day retention assumptions, **1.5 TB decimal totals about $151.03**, while **3.8 TB decimal totals about $203.15** before any contingency. The current $179.33 figure is a conservative reservation, not money already spent. A smaller transfer-only VM, shorter retention after approved verified-copy cleanup, or measured conversion could reduce costs; none should be counted as savings before qualification. The current VM costs about $1.26/hour, so 72 hours alone costs $90.54. Exact scenarios are in `cost_budget_scenarios_20261021.json` and the adjacent Markdown report.

Validation completed:

- 35 focused buffer/supervisor tests locally and on Linux; independent review of the shutdown/recovery boundaries.
- 95 focused Drive authority/drain/diagnostic tests locally and on the target Linux VM.
- Real Linux watchdog smoke: worker killed at its deadline, descendant started, no late descendant write.
- Real checkpoint resume, sustained new GCS pairs, advancing live/backfill counters, and the completed Drive pair described above.

Deployed GCS source hashes:

| File | SHA-256 |
| --- | --- |
| `gcs_archive_buffer.py` | `c012bc8a59b718a1b5de6040e3693f911d3915d5499c8dfec88eb9da63c0cfed` |
| `gcs_buffer_supervisor.py` | `6424b58bba803d7e06d1faffbf36b1fadbf6b923240ec59066adbd6109bec8f2` |
| `catbird-gcs-buffer.service` | `d6c710254cd14d9e366d2544b100c6a85ded7496cad1223327b234a67e043908` |

The frozen eight-file diagnostic deployment archive has SHA-256 `e65a6cb27ff61887ca3bb0e79261b72da04432dd1a2488e7c2ff04b03daa124a` and lives separately at `/opt/catbird-drive-diagnostic-20261021`; the capture code was not replaced during the Drive test.

## Evidence

Local receipts are under `~/catbird-lake-local/graph-build/out/capture-resume-20261021/`: `checkpoint_before.json`, `runtime_first.txt`, `runtime_1033.txt`, `first_pair_5072_independent_verification.json`, `coverage_receipt.json`, `SOURCE_PROVENANCE.md`, `drive_linux_preflight.txt`, `drive_diagnostic_manifest.json`, `drive_diagnostic_result.json`, and the `cost_*` observations/report. The Drive result is sanitized and contains pinned identities, digests, request charges and status codes, without OAuth credentials or resumable URLs. Authoritative on-VM journals and the original checkpoint are retained.

The frozen `gcs-source-snapshot-20261021T103235Z/` subdirectory contains a JSONL source manifest, full metadata inventory, summary and usage README for later Colab/Mac selection. It pins **1,417 pairs (3671–5087)** and 383,332,736,946 payload bytes, with no observed gaps or pairing conflicts in that selection. Pair 5088 appeared after the listing-start cutoff and is excluded. Manifest SHA-256 is `8e66445ab247c507f402f2b21dcc35800a6117e4b0c117f0b4b09457731d6548`. All rows are explicitly metadata-pairing evidence except the independently hashed receipt for 5072; each consumer must fetch exact generations and validate receipt/payload contents. The collector source pin is a plan-level reference, not an attestation for every historical object.
