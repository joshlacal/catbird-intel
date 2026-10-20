# Public Bluesky moderation-label backfill — 20 October 2026

Catbird needs moderation labels as source evidence alongside the existing archive. Josh authorized downloading this public label metadata onto the local machine. A recent slice has finished and been independently decoded and hashed, and a separate historical replay is running toward a frozen observed service cursor. Neither result is an eligibility decision, a verified complete history, or a replacement for the politics/heat annotation work.

Governing context: [review remediation plan](../../docs/execution-plans/2026-10-20-catbird-intelligence-review-remediation.md), [full acquisition and $200 teacher plan](../../docs/execution-plans/2026-10-20-moderation-labels-and-200-dollar-teacher-plan.md), and [decision D36](../../docs/DECISIONS.md#2026-10-20--catbird-intel-d36--separate-public-labels-from-the-200-teacher-proposal). The graph fold, its continuation driver, production services, GCE, and GPU resources were not changed by this work. No corpus, post text, or media was transmitted or fetched; requests were public identity/service metadata and the label stream itself.

## Actual source and protocol

`moderation.bsky.app` resolved to `did:plc:ar7c4by46qjdydhdevvrndac`. Its actual PLC DID document advertises `https://mod.bsky.app` as `#atproto_labeler` and contains an `#atproto_label` verification key. Exact DID-document and handle-response bytes are content-hashed in each generation. The actual `app.bsky.labeler.service/self` declaration was fetched from the DID-advertised PDS; its CID is `bafyreidh754j3y7isnikzv2itrkywmhrfqcx646hl6w5r62wqecioychoi`, response SHA-256 `f03b274acf5d23fdfb322fe6845fddaf7999247e992d780ff5bd354ca75b8c03`. This is discovery/provenance, **not signature verification**.

The [official queryLabels lexicon](https://github.com/bluesky-social/atproto/blob/main/lexicons/com/atproto/label/queryLabels.json) permits URI patterns, including prefixes. The actual service rejected `uriPatterns=*` with `InvalidRequest: Support for wildcard label queries is temporarily disabled`; the response is saved in `bluesky-v1/wildcard-query-probe.json`. Therefore this implementation uses the [official subscribeLabels stream](https://github.com/bluesky-social/atproto/blob/main/lexicons/com/atproto/label/subscribeLabels.json) and its service-local cursor, not an invented paginated enumeration endpoint. Raw [event-stream frames](https://atproto.com/specs/event-stream) contain a DAG-CBOR header followed by a body.

A no-cursor observation at 13:13:13Z yielded **sequence 44,494,621**. That observed frame and hash are saved as `bluesky-v1/head-probe.cbor` and `.json`; it is the finite target. Starting from zero returned sequence 1, but subsequent sequence gaps mean this does not establish all-time completeness. A gap may reflect service retention/sequence semantics or missing delivery; this acquisition does not adjudicate the cause. No `OutdatedCursor` notice appeared in the completed recent slice.

## Completed recent slice

Data directory: `/Users/joshlacalamito/catbird-lake-local/moderation-labels/recent-tail-20261020`.

Independent receipt: [moderation_label_backfill_recent_20261020.json](moderation_label_backfill_recent_20261020.json).

- Requested cursor 44,394,621; first observed 44,394,622; last observed/frozen target 44,494,621.
- **95,096 frames / 95,096 labels**, 20 immutable gzip chunks, **13,599,968 compressed bytes** and **30,061,588 raw frame bytes**.
- Label creation timestamps span **18 October 2026 20:28:15.501Z to 20 October 2026 13:13:13.746Z**.
- **4,129 sequence gaps / 4,904 absent sequence positions**. These are explicitly unproven coverage positions, not fabricated labels.
- Subject scopes: **79,215 post AT URIs**, **13,170 account DIDs**, **2,711 other URIs**.
- All 95,096 labels carry signatures, **81,926 are CID-scoped**, and **5,712 carry expiry fields**. Signatures remain unverified; all fields and exact bytes are preserved.
- Acquisition ran for **96.1006 seconds** including close/flush (~989.5 frames/s). This is a measurement of this bounded run, not a network-wide throughput promise.

The full positive/negated breakdown is in the receipt. Examples: `porn` **60,335 positive / 1,235 negated**, `sexual` **16,974 / 225**, `!takedown` **5,283 / 1,450**, `!suspend` **92 / 103**, `spam` **76 / 20**, and `rude` **7 / 1**. The slice is dominated by media-safety labels; it cannot stand in for editorial politics/heat labels. `neg=true` retracts a matching label; it is **not** a benign training example. Absence of a label is **not** evidence that content is safe.

Recent manifest SHA-256: `0a9261356fd1aa25440e8308555d91d0b58a2a3403a4ee0af46a617247d59fcd`.
Recent journal SHA-256: `90df5a15aca041877a01c1553fd3021a926d2772402c7db5f2c1b863cb71bcc4`.

## Historical replay now running

Launch/process snapshot: [moderation_label_backfill_running_20261020.json](moderation_label_backfill_running_20261020.json). Snapshot counts are timestamped, not a live final total.

- PID **83952**, directory `/Users/joshlacalamito/catbird-lake-local/moderation-labels/bluesky-v2`.
- Exact command, process ID, log, and source SHA are in `continuation-launch.json`.
- Frozen target **44,494,621**, pacing ceiling **1,000 frames/s**, generation logical-byte cap **7 GiB**, filesystem-free floor **110 GiB**, and **24-hour receive-loop budget**. Discovery, connection setup, backoff, final flush, and verification are outside a strict whole-process wall-clock bound.
- The initial v1 collector stopped itself after four connection failures at **6,762 labels / sequence 6,816**. Its preserved final receipt reports `connection_retry_exhausted:ConnectionClosedError`; no cause is asserted. V2 independently decoded/hashed that stopped parent and records the parent's exact journal hash and cursor in its immutable manifest. Do not resume v1 while v2 depends on that frozen parent.
- Packed smoke: **30,000 frames in 31.0175 seconds**, **4,297,514 compressed bytes**, **9,843,421 raw frame bytes**, independently decoded/hash-verified. Extrapolating that early sample to the observed sequence span suggests roughly **6.37 decimal GB / 5.94 GiB compressed**, before indexes and uncertainty. This motivated the provisional 7 GiB cap; later label sizes, gaps, rate, service changes, and shared disk growth can invalidate the estimate. The free-space floor stops acquisition independently of that estimate.
- Executed source SHA-256: `bf6e6db10a6a113d6ace91d1715c2906179d17f6d3979b94e6de12f48681f594`. An immutable source copy, Python/platform details and dependency versions are saved in the generation's `execution-*.json` receipt and `executed-source-*.py` file.

The process finishes when the exact target is observed, the target is crossed without observing it, time/frame limits expire, disk limits stop it, a protocol error occurs, or four consecutive connections fail without durable progress. A target crossing is saved separately and is not committed beyond the frozen range. Transient disconnects that made durable cursor progress reset the no-progress retry count; connection notices remain recorded.

## Storage, resume, and verification

Packed v2 keeps exact frames inside immutable gzip chunks, each frame prefixed by an unsigned 64-bit big-endian byte length. The per-chunk journal records compressed and uncompressed SHA-256, counts, cursor boundaries and a hash chain. Persistence order is chunk fsync → journal fsync → atomic checkpoint. Pending data may be replayed after a crash. Orphan chunks remain preserved. A torn journal tail is retained under a new filename before restoring its complete prefix. The checkpoint is a convenience; the journal is authoritative.

Runtime memory retains the current cursor/hash and bounded counters, not a hash for all 44 million sequence positions. Checkpoints retain gap counts and the first 32 examples; every gap can be reconstructed from exact committed frames. No per-frame files or per-frame JSON journal are used in v2. The separately preserved v1 and small smoke generations remain outside v2's logical-byte cap; the filesystem-free floor covers their disk impact too.

**Do not start a second collector.** Inspect PID 83952 and `writer.lock` ownership first. If it has stopped, resume using the exact command in `bluesky-v2/continuation-launch.json`; the existing manifest rejects identity/target changes and a changed parent journal. Public 429/transient failures back off; invalid labels, changed boundary bytes, source changes and persistence errors do not silently advance cursors.

After the writer stops, independently verify the committed archive:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-label-backfill-venv/bin/python \
  catbird-intel/ops/backfill_moderation_labels.py \
  --directory ~/catbird-lake-local/moderation-labels/bluesky-v2 --verify-only
```

That checks compressed/raw hashes, length framing, decoded counters, parent provenance and parent raw frames. It writes a separate immutable verification receipt. It still does not verify label signatures or guarantee service-history completeness.

Tests:

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-label-backfill-venv/bin/python \
  -m unittest discover -s catbird-intel/ops/tests -p test_moderation_labels.py -v
```

**15 distinct tests pass**, covering field preservation, duplicate/cursor conflicts, retained notices/gaps, hash damage, stale checkpoints, orphan chunks, torn journals, injected crashes before journal and after journal/before checkpoint, target crossing, and progress-aware reconnects. The bounded live replay and independently verified recent slice provide runtime evidence beyond these structural fixtures. Historical completion, full current-state reconciliation, label-signature verification and model/serving use remain unperformed.
