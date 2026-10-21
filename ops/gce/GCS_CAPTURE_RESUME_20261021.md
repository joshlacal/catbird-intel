# Network capture resumed on October 21

The cloud collector is backfilling Bluesky repositories and recording live events for later analysis in Colab and on the Mac; at **11:50 UTC**, Google Cloud Storage (GCS) held **1,545 durable pairs** with **417,772,057,928 cumulative admitted bytes**. Automatic continuation between bounded Drive batches is deployed and was observed, bringing the shared ledger to **20 closed retained pairs**, plus the separate diagnostic pair. The Drive loop then stopped at **11:42:38 UTC** after HTTP 500 on a receipt upload, leaving one pair active in the ledger and one unresolved request; GCS capture continued independently. Independent metadata checks and complete payload readbacks cover the earlier eight retained pairs, while the twelve newer closures are supported by worker/ledger evidence and have not received that independent readback qualification. Read-only proof collection and offline reconciliation work are underway for the ambiguous request; repository verification failures and incomplete history still prevent a full-network coverage claim.

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

The fresh app-created folder avoids collisions with unseen historical siblings under the existing `drive.file` OAuth scope. It remains explicitly `archive_qualified: false` and `history_proven: false`; success does not prove a full account-wide daily upload allowance or qualify the old archive shard. The existing unmetered eight-segment drain remains inactive.

### Retained pair and shared mapping

The reviewed retained copier is deployed separately at `/opt/catbird-drive-retained-20261021`. Its shared ledger is `/data/gcs-drive-retained-authority/authority.sqlite`, owned by writer `gce-capture-owner-20261021`; the original diagnostic ledger remains intact. The imported diagnostic charge is 268,982,001 bytes and does not assert complete account history. The bounded selector produced eight source-generation-pinned candidates and excluded already-copied diagnostic segment 3677.

The first retained pair, segment **3671**, ran at **10:58:07–10:58:41 UTC**, run ID `78455b2f9fc9955d71b64653e2e2cc67`. Its 277,182,724-byte payload is GCS generation `1792517970952213`, SHA-256 `868bce863d80e7e2660dbe835006c05eed6e24393113f57ea22abb58af613c98`. It retained GCS and downloaded local copies and closed the ledger with zero unresolved requests. The 54 transfer requests charged 277,184,273 bytes; the immutable mapping charged **1,686 bytes** to the separate cumulative mapping partition.

- Drive folder: `1WsDebm_tARfTURlX6NeOCDMtzRzFLqI3`; payload: `1XQvQ2SDlbPCKhGjHo6mb_qbEXv7s5oXD`; receipt: `1sIbVFjmk3uFVRaGELfoJvqGjUMyFgyXa`.
- Shared mapping: `gs://catbird-436600-jetstream-buffer-20261020/indexes/drive-retained-overflow/v1/c74e98e1208d2f188da5ac7b84553aeed42a81e11e89727928da4bc0f4934684.json`, generation `1792580321635037`, SHA-256 `27da75354922b1bede85a5c94d44fae06ce26089ee01309479bf6fe607c7653d`.
- Independent verification at 10:59:04 UTC passed **31/31 checks** against closed ledger state, immutable mapping bytes, local payload/receipt hashes, pinned GCS generation metadata and exact Drive account/folder/file IDs, sizes and MD5 values. Those checks used Drive metadata, not media readback. The sanitized report is `retained_pair_3671_independent_verification.json` in the local evidence directory.
- A subsequent **full Drive payload readback** passed at **11:05:42 UTC**: all 277,182,724 bytes streamed through SHA-256 and MD5, matching the pinned GCS source and local bytes. No second local media file was saved. The receipt body was not independently fetched from Drive, and decompressed-block validation remains unperformed. This additional result is `retained_pair_3671_drive_media_readback.json`; it qualifies only this payload, not the full archive.

The first target Linux suite passed **142/142 tests**, including the real worker-and-grandchild watchdog. An initial test-only floating-point boundary failure was corrected by testing times strictly below and above the unchanged 0.55-second production minimum; the failed and corrected logs are both preserved. That deployment archive SHA-256 is `7728c86d234ab43a891c3278f8652f094582811a1720a59b1412a36c9fc56b40`, plus test-only patch `4572329320f7d2ae010ddae421b9089554576a48db883a27eb2a514db8a7139d`.

### Seven-pair sequential batch

The sequential runner deployed separately at `/opt/catbird-drive-retained-runner-20261021` completed all seven candidates at **11:17:26 UTC**, runner ID `a57ec719602e456c89df81caf2664fde`. The pinned snapshot SHA-256 is `c11edb795b32c35db92c77a51eba44c3c2719df460937dbe243bb1242c5183b7`; it selected source indices **3672–3676, 3678 and 3679**, excluding the closed first pair and diagnostic. The shared ledger was reused without reset. Final status was `completed/snapshot_exhausted`, no active child, no cooldown and **zero unresolved requests**.

All seven subsequently passed **31/31 independent metadata/local-hash checks each**, followed by complete Drive payload SHA-256 and MD5 readback with all writers stopped. These seven payloads total **1,905,210,567 bytes**; including the earlier retained pair, eight payloads total **2,182,393,291 bytes**, plus 8,744 receipt bytes. Receipt bodies have not been independently read back from Drive, and decompressed-block CRC validation remains unperformed. Every GCS original, local stage copy, mapping and journal remains retained. The exact identities, mapping generations, report hashes and totals are in `retained-seven-verification-20261021/summary.json` and its fourteen per-pair reports.

At completion of this seven-pair batch, the ledger recorded **426 retained transfer request attempts**, **2,182,405,683 retained charged bytes**, plus the **268,982,001-byte** imported diagnostic charge. Mapping reservations then totaled **13,488 bytes**. These historical totals are superseded by the continuous-run observation below; neither set proves the provider's complete rolling upload history. Josh separately confirmed **“No other uploads”** on October 21; the operational exclusion uses that attestation with current process checks, without treating it as provider quota telemetry.

The runner adds a same-attempt conservative pre-dispatch wait, at most eight candidates per snapshot, at most twenty minutes per pair, a **32 GiB cumulative stage ceiling**, append-only status and a **100 GiB free-space floor checked periodically**. The free-space check is a guard, not an atomic disk reservation against concurrent capture. Capture remains independently active if Drive pauses. Deployment archive SHA-256 is `9e9c774f62495b92325ac4904604204380c948d7f97c464e3e7b176e634c44de`. On the target Linux VM, **160/160 tests** and three focused real process-tree smokes passed after correcting a test-only race when `/proc/<pid>/stat` disappears with `ProcessLookupError`. Failed and corrected logs remain in the evidence directory. The conservative pacing did not weaken the 0.55-second minimum. See the [retained-copy runbook](DRIVE_RETAINED_OVERFLOW_RUNBOOK.md).

### Continuous batches and the preserved HTTP 500 stop

The continuous supervisor, which selects another bounded batch after a clean completion, launched at **11:32 UTC** from source commit `c324b8e8`, deployed at `/opt/catbird-drive-retained-continuous-20261021`. The v4 deployment archive SHA-256 is `fc65f8b9b280a899569f6d7707c72e9afd3ce984a6ec2320dac4b2679a2c978e`. Before launch, the target Linux suite passed **178/178 tests**, and three direct supervisor smokes passed: timeout stopped the child and grandchild, monitor failure stopped the child and grandchild, and parent death stopped the direct child. The smoke recorded zero ResourceWarnings and zero cloud calls. The [continuous-copy runbook](DRIVE_RETAINED_CONTINUOUS_DRAFT.md) describes the pinned owner envelope, earliest writer-hold expiry, one-hour maximum assertion, unchanged request pacing and capture deadline. A fresh reviewed envelope is required for renewal.

Its first eight-pair batch closed at **11:37:56 UTC**, and the next batch started approximately **1.3 seconds later** without a manual batch launch. The first batch's worker records pin payload/receipt generations, Drive IDs and immutable mapping identities; its 2,176,704,972 payload bytes and 8,744 receipt bytes remain retained. This is evidence of automatic continuation and durable worker closure, not independent Drive media readback. The first-batch receipt explicitly keeps the earlier eight-pair independent verification separate.

The second batch closed four pairs before its next pair received **HTTP 500 on the final 1,093-byte receipt `media_chunk` PUT**. The supervisor stopped at **11:42:38 UTC** with reason `pair_exit_1`. The preserved attempt is `879f7f74b0234e86b1ed0cf55798aeff`, in run `2a67755ef699cc70ca004e8584e273c0`. This server error leaves the send outcome uncertain; it is not evidence of a daily quota rejection, a missing receipt, or successful pair closure.

The post-stop shared ledger records **20 closed pairs and one active pair**, **one raw unresolved attempt**, **1,111 retained request attempts**, **5,715,321,385 retained charged bytes**, the unchanged **268,982,001-byte imported diagnostic charge**, and **33,720 bytes of cumulative mapping reservations**. “Active” here is the unresolved ledger state: the Drive service is inactive and the runner reports no active child. The failed request, original error and conservative charges remain intact; no automatic retry, ledger reset or cleanup was performed.

The sole cloud owner is gathering read-only proof against the exact Drive file IDs. The separate Drive code owner is implementing an offline, append-only reconciliation path. Neither proof collection nor source implementation alone resolves the stored attempt or authorizes another send. The Drive loop remains stopped pending that reconciliation, while GCS capture continues under its existing controls. This section records the 11:50 UTC status and does not claim the ambiguous pair is recovered.

A read-only `about.storageQuota` check at **10:41:12 UTC** matched the diagnostic account and reported 32,987,496,316,928 total bytes, 1,093,567,733,001 used bytes, and **31,893,928,583,927 available bytes** (about 31.9 TB decimal). This establishes storage headroom, not remaining daily upload allowance; the API does not return that allowance. Receipt: `drive_storage_quota.json`.

## Controls, tests and budget

The GCS supervisor has a per-run ID, fresh-heartbeat checks, shared-filesystem free-space guard, cumulative byte accounting without refunds, and local deadline shutdown. Its run is backed by an externally configured Compute Engine STOP at **2026-10-24 10:10:00 UTC**. Spot recovery never moves that boundary. The monitor runs every fifteen minutes and may recover an actual preemption of this same VM after checking disk, image, journal, budget and locks; it is capped at three recovery attempts per 24 hours. Ordinary crashes, stale output, space exhaustion and journal conflicts require diagnosis.

The existing bucket's software admission ceiling is **2.5 TiB** (2,748,779,069,440 bytes), including observed generations. No new storage purchase or bucket is needed. The plan reserves **$179.33** for up to 72 total running hours, 37 days of storage/disk allowance, IP and operations, leaving **$20.67** contingency. Catalog prices and inventory support the estimate; actual billed spend is not yet verified and this is not a provider-enforced dollar cap. Retained data continues billing after compute stops; no expiry deletion is authorized. Direct bulk GCS-to-Mac egress is outside this envelope.

The continuous queue has a provisional additional **32 MiB cumulative mapping partition**, within the existing contingency. Every mapping create must be reserved before dispatch, including uncertain outcomes. Because the main buffer sees those generations only on its next inventory, the worst-case aggregate bound is 2.5 TiB + 32 MiB; incremental reserved storage cost is below $0.001. This partition does not authorize deleting any GCS original.

The pasted estimate of an entire network in 1.5–3.8 TB is not a measurement of this archive. At the same 72-hour compute and 37-day retention assumptions, **1.5 TB decimal totals about $151.03**, while **3.8 TB decimal totals about $203.15** before any contingency. The current $179.33 figure is a conservative reservation, not money already spent. A smaller transfer-only VM, shorter retention after approved verified-copy cleanup, or measured conversion could reduce costs; none should be counted as savings before qualification. The current VM costs about $1.26/hour, so 72 hours alone costs $90.54. Exact scenarios are in `cost_budget_scenarios_20261021.json` and the adjacent Markdown report.

The [offload transition options report](/Users/joshlacalamito/catbird-lake-local/graph-build/out/capture-resume-20261021/offload_transition_options_20261021.md) identifies a **2-vCPU / 8-GiB `n2-standard-2`** as a candidate for copying sealed files after the large backfill worker is no longer needed. This is planning, not a qualified replacement for live capture or a performed resize. Any writer handoff must first reconcile the ambiguous attempt, fence the old writer, and consistently transfer the same SQLite authority, accounting, file/session identities and reviewed owner exclusion. A separate worker also needs compatible host/layout integration with the existing 100 GiB free-space floor; a cheap machine shape alone does not establish a safe transition. The existing $200 budget and absolute STOP remain unchanged.

Validation completed:

- 35 focused buffer/supervisor tests locally and on Linux; independent review of the shutdown/recovery boundaries.
- 95 focused Drive authority/drain/diagnostic tests locally and on the target Linux VM.
- Real Linux watchdog smoke: worker killed at its deadline, descendant started, no late descendant write.
- Real checkpoint resume, sustained new GCS pairs, advancing live/backfill counters, and the completed Drive pair described above.
- Continuous Drive deployment: 178/178 target Linux tests, three direct supervisor process smokes, a completed eight-pair batch and observed automatic launch of the next batch; the later HTTP 500 stopped Drive with its unresolved send preserved.

Deployed GCS source hashes:

| File | SHA-256 |
| --- | --- |
| `gcs_archive_buffer.py` | `c012bc8a59b718a1b5de6040e3693f911d3915d5499c8dfec88eb9da63c0cfed` |
| `gcs_buffer_supervisor.py` | `6424b58bba803d7e06d1faffbf36b1fadbf6b923240ec59066adbd6109bec8f2` |
| `catbird-gcs-buffer.service` | `d6c710254cd14d9e366d2544b100c6a85ded7496cad1223327b234a67e043908` |

The frozen eight-file diagnostic deployment archive has SHA-256 `e65a6cb27ff61887ca3bb0e79261b72da04432dd1a2488e7c2ff04b03daa124a` and lives separately at `/opt/catbird-drive-diagnostic-20261021`; the capture code was not replaced during the Drive test.

## Evidence

Local receipts are under `~/catbird-lake-local/graph-build/out/capture-resume-20261021/`: `checkpoint_before.json`, `runtime_first.txt`, `runtime_1033.txt`, `first_pair_5072_independent_verification.json`, `coverage_receipt.json`, `SOURCE_PROVENANCE.md`, `drive_linux_preflight.txt`, `drive_diagnostic_manifest.json`, `drive_diagnostic_result.json`, and the `cost_*` observations/report. The Drive result is sanitized and contains pinned identities, digests, request charges and status codes, without OAuth credentials or resumable URLs. Authoritative on-VM journals and the original checkpoint are retained.

Continuous-run evidence in that directory includes `retained_continuous_linux_tests_20261021.log`, `retained_continuous_linux_smoke_20261021.json`, `retained-continuous-first-batch-20261021.json`, and `continuous_runtime_1144.json`. The first-batch receipt is a local collection of closed worker/ledger identities and explicitly reports no independent Drive media readback. The post-stop runtime receipt records both the inactive Drive service and active GCS buffer; it retains the earlier successful batch events alongside the stop.

The frozen `gcs-source-snapshot-20261021T103235Z/` subdirectory contains a JSONL source manifest, full metadata inventory, summary and usage README for later Colab/Mac selection. It pins **1,417 pairs (3671–5087)** and 383,332,736,946 payload bytes, with no observed gaps or pairing conflicts in that selection. Pair 5088 appeared after the listing-start cutoff and is excluded. Manifest SHA-256 is `8e66445ab247c507f402f2b21dcc35800a6117e4b0c117f0b4b09457731d6548`. All rows are explicitly metadata-pairing evidence except the independently hashed receipt for 5072; each consumer must fetch exact generations and validate receipt/payload contents. The collector source pin is a plan-level reference, not an attestation for every historical object.

## Current status at 11:50 UTC

GCS capture remains active with **1,545 durable pairs** and **417,772,057,928 cumulative admitted bytes**, under the unchanged **2.5 TiB main-buffer ceiling**, separate **32 MiB mapping partition**, and **2026-10-24 10:10 UTC absolute STOP**. Drive automatic batch continuation has been demonstrated, but the service is currently stopped after the receipt HTTP 500: **20 retained pairs are closed, one pair remains active in the ledger, and one request remains unresolved**. Preserve the original error, charges, exact object identities, local/GCS copies and prior evidence while read-only proof collection and offline reconciliation proceed. No further send, reset, cleanup, VM resize, complete-network claim or broad archive-readback qualification follows from this status.

### 11:52 UTC exact-ID fault diagnosis

Root's read-only diagnostic confirmed the correct Drive account, root and parent folder; the existing payload is 269,699,527 bytes with the expected MD5. The exact receipt ID `1j3e1LBmebBkes1h8l5UND-LGXXGcN2qs` returned HTTP 404, so a complete receipt cannot be inferred from the earlier HTTP 500. The controller was inactive, its claimed PID absent, the pair lock available, and the original request timeout horizon passed. No ledger mutation, resumable-session request, media retry or cleanup occurred. Evidence: `failed-receipt-readonly-20261021.json` under the capture evidence directory. Recovery remains under review and must preserve the original error, charge and exact existing payload.
