# catbird-intel

Catbird community intelligence services on netcup. Governing plan:
[`docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md`](../docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md).

## Correction and active repair (2026-10-20)

The earlier Phase 1–6 completion claims are invalidated; see [master-plan §8](../docs/execution-plans/2026-10-19-catbird-community-intelligence-master-plan.md#8-progress-log) and [the decision log](../docs/DECISIONS.md). Ingest, ClickHouse tables, XRPC and visibility filters are real infrastructure. The live health scorer is a regex baseline, previous student models/vectors were hashed stubs, and previous model benchmarks are not valid measurements. Phase 1 v0 evaluation also leaked seed/trap supplemental follows into construction.

Invalidated implementations and reports are preserved in `archive/invalidated_stubs_20261019/`; they must not be used as model, quality or credit-use evidence. `ops/verify-phase3-and-4.ts` is archived and is not a current verification command. Production ingest remains unchanged through its soak ending **2026-10-22 13:00Z**; a later model deploy or community table swap requires Josh's approval.

- Export real, unlabeled posts and the author-disjoint human-review holdout with `ops/export_training_banks_v1.py`. Keywords select holdout strata; they do not supply labels.
- `ops/sync_lakehouse_and_training_banks.py` uploads only the explicit v1 health bank and does not rediscover invalidated models or v0 training banks.
- `ops/quarantine_invalidated_drive_v1.py` inventories known invalidated Drive artifacts; `--apply` moves and verifies them, preserving checksums and recording recovery metadata.
- See `colab/` for the replacement real-teacher/Ettin pipeline. No quality threshold is locked before Josh reviews the holdout. A 32-row local smoke test validates mechanics only.

## Phase 0: `catbird-intel-ingest`

The service reads the Jetstream v2 live stream (official `@bsky/jetstream` SDK, unauthenticated, `app.bsky.feed.*`, `app.bsky.graph.*`, `app.bsky.actor.*`, plus identity/account events) and writes ClickHouse database `catbird`.

- **Write guarantee:** at-least-once. Rows flush every 2 s or every 20k rows. The cursor (`/var/lib/catbird-intel/ingest.cursor`) is saved only after a flush succeeds.
- **Versioning:** record tables are `ReplacingMergeTree(rev, is_deleted)` keyed `(did, rkey)`. `rev` is the commit's repo-revision TID, so duplicates collapse and an old create can never override a newer delete.
- **Deletes:** a delete becomes a row with `is_deleted = 1` under the same key, plus a row in `catbird.deletions` (30-day ledger). Read with `FINAL ... WHERE is_deleted = 0`, or use the `v_*` views.
- **Retention:** posts and engagements (likes/reposts) keep 14 days. The follow/block/list/starter-pack/feed/profile/visibility tables are permanent current state.
- **Health:** a row every 10 s in `catbird.ingest_health` (lag, events, rows, reconnects, flush time, RSS).
- **Resource caps:** systemd `CPUQuota=200%`, `MemoryMax=3G`, `Nice=10`.

```sh
bun test                    # unit tests
bunx tsc --noEmit           # typecheck
deploy/deploy.sh            # rsync + bun install + apply sql/*.sql + install units (does not start)
ssh netcup sudo systemctl enable --now catbird-intel-ingest
```

Useful checks:

```sql
-- lag and throughput, last hour
SELECT ts, events, lag_ms, flush_ms, rss_mb FROM catbird.ingest_health WHERE ts > now() - INTERVAL 1 HOUR ORDER BY ts;
-- gaps: health intervals with no events or lag over 60 s
SELECT count() FROM catbird.ingest_health WHERE events = 0 OR lag_ms > 60000;
```
