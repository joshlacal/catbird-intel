-- Fold archive graph events into the current follow graph and intern DIDs to
-- dense ids for the similarity tool. Disk-backed so it scales to billions of
-- follow records on the M4 Max (run clickhouse local with a persistent --path).
--
--   clickhouse local --path <work>/ch (use graph/build_graph.sh; it substitutes __EVENTS__ __OUT__ __MIN_FOLLOWERS__ __MAX_FOLLOWS__)
--
-- Multi-generation snapshot & reset semantics (W2):
--   1. Events carry canonical `source_id` (`public_archive`, `gce_main`,
--      `gce_bootstrap_live`, `netcup_live`). `public_archive` (August 4, 2026 seed)
--      maps to `source_tier = 1`; October 2026 sources (`gce_main`,
--      `gce_bootstrap_live`, `netcup_live`) map to `source_tier = 2`.
--   2. Per DID, take the latest snapshot/reset epoch ordered by repository `rev`
--      (never by cross-source `seq`). Keep only records from that snapshot plus
--      later-rev live events (`rev >= snapshot_floor_rev`), and apply deletes
--      ordered by `(source_tier, rev, time)`.
--   3. Apply account status from the latest-rev account event per DID
--      (`tuple(source_tier, rev, time)` when `rev` is present, else
--      `tuple(source_tier, time, rev)` — never cross-source `seq`).
--   4. Apply `netcup`'s `catbird.deletions` (`__OUT__/netcup_deletions*.parquet`)
--      and `account_status` (`__OUT__/netcup_account_status*.parquet`) ledger
--      for the GCE VM stop window (`2026-10-20 00:00:00Z` -> `2026-10-21 10:18:59.999Z`).
--   5. Repos that never download in October fall back to their August snapshot
--      plus live events since `2026-10-19`, flagged `coverage = 'stale_aug'`
--      (`snapshot_source = 'august_seed_fallback'`, `is_stale = 1`).
--   6. Emit `coverage.parquet` (`did, host, snapshot_source, snapshot_rev,
--      coverage ∈ {'oct', 'stale_aug', 'missing'}` + per-host coverage rates
--      and `inverse_coverage_weight`) and `host_coverage.parquet` / `host_coverage.tsv`.
--
-- Outputs (in {out}):
--   producers.tsv         pid \t did \t followers   (pid dense 0..P-1, followers desc)
--   edges.bin             RowBinary (uid UInt32, pid UInt32), sorted by uid
--   stats.tsv             counts for the progress log (6 canonical columns)
--   account_coverage.tsv  did \t snapshot_source \t is_stale \t latest_tier \t latest_time \t has_sync_reset \t is_deleted
--   coverage_stats.tsv    summary of fresh vs. stale fallback accounts and suppressed zombie follows
--   follower_weights.tsv  uid \t did \t is_stale \t weight (dense 0..U-1 matching edges.bin)
--   coverage.parquet      did, host, snapshot_source, snapshot_rev, coverage, host_oct_coverage_rate, host_effective_coverage_rate, inverse_coverage_weight
--   host_coverage.parquet per-PDS-host coverage rates and inverse-coverage weights
--   host_coverage.tsv     TSV mirror of host_coverage.parquet

SET max_threads = 12,
    max_bytes_before_external_group_by = 16000000000,
    max_bytes_before_external_sort = 16000000000,
    optimize_aggregation_in_order = 1,
    input_format_parquet_allow_missing_columns = 1,
    use_structure_from_insertion_table_in_table_functions = 0;

CREATE DATABASE IF NOT EXISTS g;
DROP TABLE IF EXISTS g.follow_events;
DROP TABLE IF EXISTS g.control_events;
DROP TABLE IF EXISTS g.did_hosts_raw;
DROP TABLE IF EXISTS g.did_hosts;
DROP TABLE IF EXISTS g.follow_acct_summary;
DROP TABLE IF EXISTS g.control_acct_summary;
DROP TABLE IF EXISTS g.account_epochs;
DROP TABLE IF EXISTS g.active_follow_events;
DROP TABLE IF EXISTS g.follows;
DROP TABLE IF EXISTS g.zombie_suppressed;
DROP TABLE IF EXISTS g.inactive;
DROP TABLE IF EXISTS g.producers;
DROP TABLE IF EXISTS g.followers_ok;
DROP TABLE IF EXISTS g.follower_uids;
DROP TABLE IF EXISTS g.all_dids;
DROP TABLE IF EXISTS g.account_coverage_base;
DROP TABLE IF EXISTS g.host_coverage;
DROP TABLE IF EXISTS g.coverage;

CREATE TABLE g.did_hosts_raw
(
    did String,
    host String
)
ENGINE = MergeTree ORDER BY did;

INSERT INTO g.did_hosts_raw
SELECT
    did,
    host
FROM file(
    '__OUT__/did_hosts*.parquet',
    Parquet,
    'did String, host String'
)
WHERE did != '' AND host != '';

INSERT INTO g.did_hosts_raw
SELECT
    did,
    ifNull(host, '') AS host
FROM file(
    '__EVENTS__',
    Parquet,
    'did String, host Nullable(String)'
)
WHERE did != '' AND ifNull(host, '') != '';

CREATE TABLE g.did_hosts
(
    did String,
    host String
)
ENGINE = MergeTree ORDER BY did AS
SELECT
    did,
    argMax(host, host != '') AS host
FROM g.did_hosts_raw
GROUP BY did;

CREATE TABLE g.follow_events
(
    did String,
    rkey String,
    rev String,
    operation LowCardinality(String),
    subject String,
    created_at String,
    source_id LowCardinality(String),
    source_tier UInt8,
    time String,
    seq UInt64
)
ENGINE = MergeTree ORDER BY (did, rkey, source_tier, rev, time);

INSERT INTO g.follow_events
SELECT
    did,
    rkey,
    rev,
    operation,
    subject,
    created_at,
    multiIf(
        ifNull(source_id, '') IN ('public_archive', 'gce_main', 'gce_bootstrap_live', 'netcup_live'),
        ifNull(source_id, ''),
        position(_path, 'netcup') > 0,
        'netcup_live',
        position(_path, 'bootstrap-live') > 0 OR position(_path, 'bootstrap_live') > 0,
        'gce_bootstrap_live',
        position(_path, 'gce-archive-graph') > 0 OR position(_path, 'gce_main') > 0 OR position(_path, 'october') > 0 OR time >= '2026-10-01',
        'gce_main',
        'public_archive'
    ) AS resolved_source_id,
    if(
        resolved_source_id = 'public_archive' AND time < '2026-10-01',
        toUInt8(1),
        toUInt8(2)
    ) AS source_tier,
    time,
    seq
FROM file(
    '__EVENTS__',
    Parquet,
    'source_id Nullable(String), seq UInt64, time String, kind String, did String, collection String, rkey String, rev String, operation String, created_at String, subject String'
)
WHERE collection = 'app.bsky.graph.follow' AND (kind = 'commit' OR kind = '');

-- Apply netcup's catbird.deletions ledger for the GCE VM stop window (Oct 20 -> Oct 21 10:18Z).
INSERT INTO g.follow_events
SELECT
    did,
    rkey,
    if(
        rev > 0,
        arrayStringConcat(
            arrayMap(
                i -> substring('234567abcdefghijklmnopqrstuvwxyz', toUInt8(bitAnd(bitShiftRight(rev, (13 - i) * 5), 31)) + 1, 1),
                range(1, 14)
            )
        ),
        ''
    ) AS rev_tid,
    'delete' AS operation,
    '' AS subject,
    '' AS created_at,
    'netcup_live' AS source_id,
    toUInt8(2) AS source_tier,
    formatDateTime(indexed_at, '%Y-%m-%dT%H:%M:%SZ', 'UTC') AS time,
    seq
FROM file(
    '__OUT__/netcup_deletions*.parquet',
    Parquet,
    'collection String, did String, rkey String, rev UInt64, seq UInt64, indexed_at DateTime64(3, \'UTC\')'
)
WHERE collection = 'app.bsky.graph.follow'
  AND indexed_at >= toDateTime64('2026-10-20 00:00:00.000', 3, 'UTC')
  AND indexed_at <= toDateTime64('2026-10-21 10:18:59.999', 3, 'UTC');

CREATE TABLE g.control_events
(
    did String,
    kind LowCardinality(String),
    collection LowCardinality(String),
    operation LowCardinality(String),
    rev String,
    account_active Nullable(Bool),
    account_status LowCardinality(String),
    source_id LowCardinality(String),
    source_tier UInt8,
    time String,
    seq UInt64
)
ENGINE = MergeTree ORDER BY (did, source_tier, rev, time);

INSERT INTO g.control_events
SELECT
    did,
    kind,
    collection,
    operation,
    rev,
    account_active,
    account_status,
    multiIf(
        ifNull(source_id, '') IN ('public_archive', 'gce_main', 'gce_bootstrap_live', 'netcup_live'),
        ifNull(source_id, ''),
        position(_path, 'netcup') > 0,
        'netcup_live',
        position(_path, 'bootstrap-live') > 0 OR position(_path, 'bootstrap_live') > 0,
        'gce_bootstrap_live',
        position(_path, 'gce-archive-graph') > 0 OR position(_path, 'gce_main') > 0 OR position(_path, 'october') > 0 OR time >= '2026-10-01',
        'gce_main',
        'public_archive'
    ) AS resolved_source_id,
    if(
        resolved_source_id = 'public_archive' AND time < '2026-10-01',
        toUInt8(1),
        toUInt8(2)
    ) AS source_tier,
    time,
    seq
FROM file(
    '__EVENTS__',
    Parquet,
    'source_id Nullable(String), seq UInt64, time String, kind String, did String, collection String, operation String, rev String, account_active Nullable(Bool), account_status String'
)
WHERE kind IN ('account', 'sync')
   OR (collection = 'app.bsky.actor.profile' AND operation = 'create');

-- Apply netcup's catbird.account_status ledger for the GCE VM stop window (Oct 20 -> Oct 21 10:18Z).
INSERT INTO g.control_events
SELECT
    did,
    'account' AS kind,
    '' AS collection,
    '' AS operation,
    '' AS rev,
    active != 0 AS account_active,
    status AS account_status,
    'netcup_live' AS source_id,
    toUInt8(2) AS source_tier,
    formatDateTime(indexed_at, '%Y-%m-%dT%H:%M:%SZ', 'UTC') AS time,
    seq
FROM file(
    '__OUT__/netcup_account_status*.parquet',
    Parquet,
    'did String, active UInt8, status LowCardinality(String), seq UInt64, indexed_at DateTime64(3, \'UTC\')'
)
WHERE indexed_at >= toDateTime64('2026-10-20 00:00:00.000', 3, 'UTC')
  AND indexed_at <= toDateTime64('2026-10-21 10:18:59.999', 3, 'UTC');

-- Accounts whose latest account event per DID is inactive are excluded on both sides.
-- Ordered by (source_tier, rev, time) when all account events for that DID carry rev,
-- else (source_tier, time, rev) — never cross-source seq.
CREATE TABLE g.inactive (did String) ENGINE = MergeTree ORDER BY did AS
SELECT did
FROM g.control_events
WHERE kind = 'account'
GROUP BY did
HAVING if(
    min(rev != '') = 1,
    argMax(coalesce(account_active, true), tuple(source_tier, rev, time)) = false
        OR argMax(account_status, tuple(source_tier, rev, time)) IN ('deleted', 'deactivated', 'takendown', 'suspended'),
    argMax(coalesce(account_active, true), tuple(source_tier, time, rev)) = false
        OR argMax(account_status, tuple(source_tier, time, rev)) IN ('deleted', 'deactivated', 'takendown', 'suspended')
);

-- Per-account control-stream summary (sync barriers, deleted barriers, and profile snapshot markers).
CREATE TABLE g.control_acct_summary
(
    did String,
    has_tier1_ctrl UInt8,
    has_tier2_ctrl UInt8,
    has_tier2_sync_or_profile UInt8,
    oct_ctrl_snap_rev String,
    aug_ctrl_snap_rev String,
    has_sync_reset UInt8,
    sync_tier UInt8,
    sync_rev String,
    sync_time String,
    has_deleted_barrier UInt8,
    del_tier UInt8,
    del_rev String,
    del_time String,
    is_currently_deleted UInt8,
    latest_ctrl_tier UInt8,
    latest_ctrl_time String
)
ENGINE = MergeTree ORDER BY did AS
SELECT
    did,
    toUInt8(max(source_tier = 1)) AS has_tier1_ctrl,
    toUInt8(max(source_tier = 2)) AS has_tier2_ctrl,
    toUInt8(max(source_tier = 2 AND (kind = 'sync' OR (collection = 'app.bsky.actor.profile' AND operation = 'create' AND source_id NOT IN ('gce_bootstrap_live', 'netcup_live'))))) AS has_tier2_sync_or_profile,
    argMaxIf(
        rev,
        tuple(toUInt8(kind = 'sync'), time, rev),
        source_tier = 2 AND rev != '' AND (kind = 'sync' OR (collection = 'app.bsky.actor.profile' AND operation = 'create' AND source_id NOT IN ('gce_bootstrap_live', 'netcup_live')))
    ) AS oct_ctrl_snap_rev,
    argMaxIf(
        rev,
        tuple(toUInt8(kind = 'sync'), time, rev),
        source_tier = 1 AND rev != '' AND (kind = 'sync' OR (collection = 'app.bsky.actor.profile' AND operation = 'create'))
    ) AS aug_ctrl_snap_rev,
    toUInt8(max(kind = 'sync')) AS has_sync_reset,
    (maxIf(tuple(source_tier, time, rev), kind = 'sync')).1 AS sync_tier,
    (maxIf(tuple(source_tier, time, rev), kind = 'sync')).3 AS sync_rev,
    (maxIf(tuple(source_tier, time, rev), kind = 'sync')).2 AS sync_time,
    toUInt8(max(kind = 'account' AND account_status = 'deleted')) AS has_deleted_barrier,
    (maxIf(tuple(source_tier, time, rev), kind = 'account' AND account_status = 'deleted')).1 AS del_tier,
    (maxIf(tuple(source_tier, time, rev), kind = 'account' AND account_status = 'deleted')).3 AS del_rev,
    (maxIf(tuple(source_tier, time, rev), kind = 'account' AND account_status = 'deleted')).2 AS del_time,
    toUInt8(
        max(kind = 'account') = 1
        AND if(
            minIf(rev != '', kind = 'account') = 1,
            argMaxIf(account_status, tuple(source_tier, rev, time), kind = 'account'),
            argMaxIf(account_status, tuple(source_tier, time, rev), kind = 'account')
        ) = 'deleted'
    ) AS is_currently_deleted,
    max(source_tier) AS latest_ctrl_tier,
    max(time) AS latest_ctrl_time
FROM g.control_events
GROUP BY did;

-- Per-account follow-stream summary within each account's active post-reset epoch.
CREATE TABLE g.follow_acct_summary
(
    did String,
    has_tier1 UInt8,
    has_tier2 UInt8,
    has_oct_follow_snapshot UInt8,
    oct_snapshot_rev String,
    oct_snapshot_floor_rev String,
    aug_snapshot_rev String,
    aug_snapshot_floor_rev String,
    latest_follow_tier UInt8,
    latest_follow_time String
)
ENGINE = MergeTree ORDER BY did AS
SELECT
    f.did AS did,
    toUInt8(max(f.source_tier = 1)) AS has_tier1,
    toUInt8(max(f.source_tier = 2)) AS has_tier2,
    toUInt8(
        max(f.source_tier = 2 AND f.source_id NOT IN ('gce_bootstrap_live', 'netcup_live') AND post_barrier = 1) = 1 AND (
            maxIf(
                is_hist_create,
                f.source_tier = 2 AND f.operation = 'create' AND f.source_id NOT IN ('gce_bootstrap_live', 'netcup_live') AND post_barrier = 1
            ) = 1
            OR countIf(f.source_tier = 2 AND f.operation = 'create' AND f.source_id NOT IN ('gce_bootstrap_live', 'netcup_live') AND post_barrier = 1)
               > uniqExactIf(f.rev, f.source_tier = 2 AND f.operation = 'create' AND f.source_id NOT IN ('gce_bootstrap_live', 'netcup_live') AND post_barrier = 1)
            OR (
                max(f.source_tier = 1) = 1
                AND countIf(f.source_tier = 2 AND f.operation = 'create' AND f.source_id NOT IN ('gce_bootstrap_live', 'netcup_live') AND post_barrier = 1) > 0
                AND countIf(f.operation = 'create') > uniqExactIf(f.rkey, f.operation = 'create')
            )
        )
    ) AS has_oct_follow_snapshot,
    if(
        maxIf(
            f.rev,
            f.source_tier = 2 AND f.operation = 'create' AND f.source_id NOT IN ('gce_bootstrap_live', 'netcup_live') AND post_barrier = 1 AND is_hist_create = 1
        ) != '',
        maxIf(
            f.rev,
            f.source_tier = 2 AND f.operation = 'create' AND f.source_id NOT IN ('gce_bootstrap_live', 'netcup_live') AND post_barrier = 1 AND is_hist_create = 1
        ),
        if(
            has_oct_follow_snapshot = 1,
            maxIf(f.rev, f.source_tier = 2 AND f.operation = 'create' AND f.source_id NOT IN ('gce_bootstrap_live', 'netcup_live') AND post_barrier = 1),
            ''
        )
    ) AS oct_snapshot_rev,
    oct_snapshot_rev AS oct_snapshot_floor_rev,
    maxIf(f.rev, f.source_tier = 1 AND f.operation = 'create' AND post_barrier = 1) AS aug_snapshot_rev,
    minIf(f.rev, f.source_tier = 1 AND f.operation = 'create' AND f.rev != '' AND post_barrier = 1) AS aug_snapshot_floor_rev,
    max(f.source_tier) AS latest_follow_tier,
    max(f.time) AS latest_follow_time
FROM
(
    SELECT
        fe.did AS did,
        fe.rkey AS rkey,
        fe.rev AS rev,
        fe.operation AS operation,
        fe.source_id AS source_id,
        fe.source_tier AS source_tier,
        fe.time AS time,
        toUInt8(
            fe.created_at = ''
            OR fe.created_at < '2026-10-19T19:50:00Z'
            OR substring(fe.created_at, 1, 10) != substring(fe.time, 1, 10)
            OR abs(
                (toInt64OrZero(substring(fe.time, 12, 2)) * 60 + toInt64OrZero(substring(fe.time, 15, 2)))
                - (toInt64OrZero(substring(fe.created_at, 12, 2)) * 60 + toInt64OrZero(substring(fe.created_at, 15, 2)))
            ) > 10
        ) AS is_hist_create,
        toUInt8(
            (
                c.has_sync_reset = 0
                OR (
                    tuple(fe.source_tier, fe.time) >= tuple(c.sync_tier, c.sync_time)
                    AND (c.sync_rev = '' OR fe.source_tier > c.sync_tier OR fe.rev >= c.sync_rev)
                )
            )
            AND (
                c.has_deleted_barrier = 0
                OR (
                    tuple(fe.source_tier, fe.time) > tuple(c.del_tier, c.del_time)
                    AND (c.del_rev = '' OR fe.source_tier > c.del_tier OR fe.rev > c.del_rev)
                )
            )
        ) AS post_barrier
    FROM g.follow_events AS fe
    LEFT JOIN g.control_acct_summary AS c ON fe.did = c.did
) AS f
GROUP BY f.did;

-- Combined per-account epoch, reset barriers, and snapshot freshness classification.
CREATE TABLE g.account_epochs
(
    did String,
    has_tier1 UInt8,
    has_tier2 UInt8,
    has_october_snapshot UInt8,
    is_stale UInt8,
    snapshot_source LowCardinality(String),
    coverage LowCardinality(String),
    snapshot_rev String,
    snapshot_floor_rev String,
    min_valid_tier UInt8,
    has_sync_reset UInt8,
    sync_tier UInt8,
    sync_rev String,
    sync_time String,
    has_deleted_barrier UInt8,
    del_tier UInt8,
    del_rev String,
    del_time String,
    is_deleted UInt8,
    latest_tier UInt8,
    latest_time String
)
ENGINE = MergeTree ORDER BY did AS
SELECT
    if(f.did != '', f.did, c.did) AS did,
    toUInt8(f.has_tier1 = 1 OR c.has_tier1_ctrl = 1) AS has_tier1,
    toUInt8(f.has_tier2 = 1 OR c.has_tier2_ctrl = 1) AS has_tier2,
    toUInt8(f.has_oct_follow_snapshot = 1 OR c.has_tier2_sync_or_profile = 1 OR (c.has_deleted_barrier = 1 AND c.del_tier = 2)) AS has_october_snapshot,
    toUInt8(
        (f.has_oct_follow_snapshot = 0 AND c.has_tier2_sync_or_profile = 0 AND NOT (c.has_deleted_barrier = 1 AND c.del_tier = 2))
        AND (f.has_tier1 = 1 OR c.has_tier1_ctrl = 1)
    ) AS is_stale,
    multiIf(
        f.has_oct_follow_snapshot = 1 OR c.has_tier2_sync_or_profile = 1 OR (c.has_deleted_barrier = 1 AND c.del_tier = 2),
        'october_gce_snapshot',
        f.has_tier1 = 1 OR c.has_tier1_ctrl = 1,
        'august_seed_fallback',
        'october_live_incremental_only'
    ) AS snapshot_source,
    multiIf(
        f.has_oct_follow_snapshot = 1 OR c.has_tier2_sync_or_profile = 1 OR (c.has_deleted_barrier = 1 AND c.del_tier = 2),
        'oct',
        f.has_tier1 = 1 OR c.has_tier1_ctrl = 1,
        'stale_aug',
        'missing'
    ) AS coverage,
    multiIf(
        f.has_oct_follow_snapshot = 1 OR c.has_tier2_sync_or_profile = 1 OR (c.has_deleted_barrier = 1 AND c.del_tier = 2),
        greatest(f.oct_snapshot_rev, c.oct_ctrl_snap_rev),
        f.has_tier1 = 1 OR c.has_tier1_ctrl = 1,
        greatest(f.aug_snapshot_rev, c.aug_ctrl_snap_rev),
        ''
    ) AS snapshot_rev,
    multiIf(
        f.has_oct_follow_snapshot = 1 OR c.has_tier2_sync_or_profile = 1 OR (c.has_deleted_barrier = 1 AND c.del_tier = 2),
        greatest(f.oct_snapshot_floor_rev, c.oct_ctrl_snap_rev),
        f.has_tier1 = 1 OR c.has_tier1_ctrl = 1,
        if(f.aug_snapshot_floor_rev != '', f.aug_snapshot_floor_rev, c.aug_ctrl_snap_rev),
        ''
    ) AS snapshot_floor_rev,
    if(f.has_oct_follow_snapshot = 1 OR c.has_tier2_sync_or_profile = 1 OR (c.has_deleted_barrier = 1 AND c.del_tier = 2), toUInt8(2), toUInt8(1)) AS min_valid_tier,
    c.has_sync_reset AS has_sync_reset,
    c.sync_tier AS sync_tier,
    c.sync_rev AS sync_rev,
    c.sync_time AS sync_time,
    c.has_deleted_barrier AS has_deleted_barrier,
    c.del_tier AS del_tier,
    c.del_rev AS del_rev,
    c.del_time AS del_time,
    c.is_currently_deleted AS is_deleted,
    greatest(f.latest_follow_tier, c.latest_ctrl_tier) AS latest_tier,
    greatest(f.latest_follow_time, c.latest_ctrl_time) AS latest_time
FROM g.follow_acct_summary AS f
FULL OUTER JOIN g.control_acct_summary AS c ON f.did = c.did;

-- Count August 4 seed follows (or pre-snapshot follows) that would have been resurrected without October snapshot replacement.
CREATE TABLE g.zombie_suppressed (did String, rkey String) ENGINE = MergeTree ORDER BY (did, rkey) AS
SELECT f.did AS did, f.rkey AS rkey
FROM g.follow_events AS f
INNER JOIN g.account_epochs AS a ON f.did = a.did
WHERE a.has_october_snapshot = 1
GROUP BY f.did, f.rkey
HAVING max(toUInt8(f.source_tier = 2 AND (a.snapshot_floor_rev = '' OR f.rev >= a.snapshot_floor_rev))) = 0
   AND argMaxIf(f.operation, tuple(f.rev, f.time), f.source_tier = 1 OR (a.snapshot_floor_rev != '' AND f.rev < a.snapshot_floor_rev)) != 'delete';

-- Filter follow events to each account's active post-reset / post-snapshot epoch.
-- Ordered by repository `rev` (never by cross-source `seq`).
-- For `stale_aug` fallback accounts, keeps the August snapshot (`source_tier = 1`)
-- plus live events since `2026-10-19` (`source_tier = 2 AND time >= '2026-10-19'`).
CREATE TABLE g.active_follow_events
(
    did String,
    rkey String,
    rev String,
    operation LowCardinality(String),
    subject String,
    source_tier UInt8,
    time String
)
ENGINE = MergeTree ORDER BY (did, rkey) AS
SELECT
    f.did AS did,
    f.rkey AS rkey,
    f.rev AS rev,
    f.operation AS operation,
    f.subject AS subject,
    f.source_tier AS source_tier,
    f.time AS time
FROM g.follow_events AS f
INNER JOIN g.account_epochs AS a ON f.did = a.did
WHERE a.is_deleted = 0
  AND f.source_tier >= a.min_valid_tier
  AND (
      a.has_october_snapshot = 1
      OR f.source_tier = 1
      OR (f.source_tier = 2 AND f.time >= '2026-10-19')
  )
  AND (
      a.snapshot_floor_rev = ''
      OR f.source_tier > a.min_valid_tier
      OR f.rev >= a.snapshot_floor_rev
  )
  AND (
      a.has_sync_reset = 0
      OR (
          tuple(f.source_tier, f.time) >= tuple(a.sync_tier, a.sync_time)
          AND (a.sync_rev = '' OR f.source_tier > a.sync_tier OR f.rev >= a.sync_rev)
      )
  )
  AND (
      a.has_deleted_barrier = 0
      OR (
          tuple(f.source_tier, f.time) > tuple(a.del_tier, a.del_time)
          AND (a.del_rev = '' OR f.source_tier > a.del_tier OR f.rev > a.del_rev)
      )
  );

-- Latest revision per record within the active epoch; (source_tier, rev, time) orders revisions (never seq).
CREATE TABLE g.follows (follower String, producer String) ENGINE = MergeTree ORDER BY (follower, producer) AS
SELECT DISTINCT did AS follower, subject AS producer
FROM
(
    SELECT
        did,
        rkey,
        argMax(operation, tuple(source_tier, rev, time)) AS op,
        argMax(subject, tuple(source_tier, rev, time)) AS subject
    FROM g.active_follow_events
    GROUP BY did, rkey
)
WHERE op != 'delete'
  AND startsWith(subject, 'did:')
  AND follower != producer
  AND follower NOT IN (SELECT did FROM g.inactive)
  AND producer NOT IN (SELECT did FROM g.inactive);

-- Follow-spam accounts are dropped from the follower side.
CREATE TABLE g.followers_ok (follower String) ENGINE = MergeTree ORDER BY follower AS
SELECT follower FROM g.follows GROUP BY follower HAVING count() <= __MAX_FOLLOWS__;

CREATE TABLE g.producers (did String, followers UInt64, pid UInt32) ENGINE = MergeTree ORDER BY did AS
SELECT did, followers, toUInt32(row_number() OVER (ORDER BY followers DESC, did) - 1) AS pid
FROM
(
    SELECT producer AS did, count() AS followers
    FROM g.follows WHERE follower IN (SELECT follower FROM g.followers_ok)
    GROUP BY producer HAVING followers >= __MIN_FOLLOWERS__
);

SELECT pid, did, followers FROM g.producers ORDER BY pid
INTO OUTFILE '__OUT__/producers.tsv' TRUNCATE FORMAT TSV;

-- Users are numbered in follower order; edges stream out already grouped by uid.
SELECT toUInt32(dense_rank() OVER (ORDER BY f.follower) - 1) AS uid, p.pid AS pid
FROM g.follows AS f
INNER JOIN g.producers AS p ON f.producer = p.did
WHERE f.follower IN (SELECT follower FROM g.followers_ok)
ORDER BY uid, pid
INTO OUTFILE '__OUT__/edges.bin' TRUNCATE FORMAT RowBinary;

-- Universe of DIDs for coverage.parquet (accounts with events, known PDS host entries, and followed target DIDs).
CREATE TABLE g.all_dids (did String) ENGINE = MergeTree ORDER BY did AS
SELECT DISTINCT did
FROM
(
    SELECT did FROM g.account_epochs WHERE startsWith(did, 'did:')
    UNION ALL
    SELECT did FROM g.did_hosts WHERE startsWith(did, 'did:')
    UNION ALL
    SELECT DISTINCT subject AS did FROM g.follow_events WHERE startsWith(subject, 'did:')
);

CREATE TABLE g.account_coverage_base
(
    did String,
    host String,
    snapshot_source String,
    snapshot_rev String,
    coverage LowCardinality(String)
)
ENGINE = MergeTree ORDER BY (host, did) AS
SELECT
    d.did AS did,
    if(h.host != '', h.host, 'unknown') AS host,
    if(a.did != '', a.snapshot_source, 'missing') AS snapshot_source,
    if(a.did != '', a.snapshot_rev, '') AS snapshot_rev,
    if(a.did != '', a.coverage, 'missing') AS coverage
FROM g.all_dids AS d
LEFT JOIN g.account_epochs AS a ON d.did = a.did
LEFT JOIN g.did_hosts AS h ON d.did = h.did;

CREATE TABLE g.host_coverage
(
    host String,
    total_accounts UInt64,
    oct_accounts UInt64,
    stale_aug_accounts UInt64,
    missing_accounts UInt64,
    oct_coverage_rate Float64,
    stale_aug_coverage_rate Float64,
    missing_rate Float64,
    effective_coverage_rate Float64,
    inverse_coverage_weight Float64
)
ENGINE = MergeTree ORDER BY host AS
SELECT
    host,
    count() AS total_accounts,
    countIf(coverage = 'oct') AS oct_accounts,
    countIf(coverage = 'stale_aug') AS stale_aug_accounts,
    countIf(coverage = 'missing') AS missing_accounts,
    round(countIf(coverage = 'oct') / greatest(count(), 1), 6) AS oct_coverage_rate,
    round(countIf(coverage = 'stale_aug') / greatest(count(), 1), 6) AS stale_aug_coverage_rate,
    round(countIf(coverage = 'missing') / greatest(count(), 1), 6) AS missing_rate,
    round((countIf(coverage = 'oct') + 0.85 * countIf(coverage = 'stale_aug')) / greatest(count(), 1), 6) AS effective_coverage_rate,
    round(1.0 / greatest((countIf(coverage = 'oct') + 0.85 * countIf(coverage = 'stale_aug')) / greatest(count(), 1), 0.05), 6) AS inverse_coverage_weight
FROM g.account_coverage_base
GROUP BY host;

CREATE TABLE g.coverage
(
    did String,
    host String,
    snapshot_source String,
    snapshot_rev String,
    coverage LowCardinality(String),
    host_oct_coverage_rate Float64,
    host_effective_coverage_rate Float64,
    inverse_coverage_weight Float64
)
ENGINE = MergeTree ORDER BY did AS
SELECT
    b.did AS did,
    b.host AS host,
    b.snapshot_source AS snapshot_source,
    b.snapshot_rev AS snapshot_rev,
    b.coverage AS coverage,
    hc.oct_coverage_rate AS host_oct_coverage_rate,
    hc.effective_coverage_rate AS host_effective_coverage_rate,
    multiIf(
        b.coverage = 'oct',
        round(1.0 / greatest(hc.effective_coverage_rate, 0.05), 6),
        b.coverage = 'stale_aug',
        round(0.85 / greatest(hc.effective_coverage_rate, 0.05), 6),
        0.0
    ) AS inverse_coverage_weight
FROM g.account_coverage_base AS b
INNER JOIN g.host_coverage AS hc ON b.host = hc.host;

SELECT
    did,
    host,
    snapshot_source,
    snapshot_rev,
    coverage,
    host_oct_coverage_rate,
    host_effective_coverage_rate,
    inverse_coverage_weight
FROM g.coverage
ORDER BY did
INTO OUTFILE '__OUT__/coverage.parquet' TRUNCATE FORMAT Parquet;

SELECT
    host,
    total_accounts,
    oct_accounts,
    stale_aug_accounts,
    missing_accounts,
    oct_coverage_rate,
    stale_aug_coverage_rate,
    missing_rate,
    effective_coverage_rate,
    inverse_coverage_weight
FROM g.host_coverage
ORDER BY total_accounts DESC, host ASC
INTO OUTFILE '__OUT__/host_coverage.parquet' TRUNCATE FORMAT Parquet;

SELECT
    host,
    total_accounts,
    oct_accounts,
    stale_aug_accounts,
    missing_accounts,
    oct_coverage_rate,
    stale_aug_coverage_rate,
    missing_rate,
    effective_coverage_rate,
    inverse_coverage_weight
FROM g.host_coverage
ORDER BY total_accounts DESC, host ASC
INTO OUTFILE '__OUT__/host_coverage.tsv' TRUNCATE FORMAT TSVWithNames;

-- Dense follower coverage weights aligned 1:1 with `uid` in edges.bin.
CREATE TABLE g.follower_uids (uid UInt32, did String) ENGINE = MergeTree ORDER BY uid AS
SELECT toUInt32(row_number() OVER (ORDER BY follower) - 1) AS uid, follower AS did
FROM
(
    SELECT DISTINCT f.follower AS follower
    FROM g.follows AS f
    INNER JOIN g.producers AS p ON f.producer = p.did
    WHERE f.follower IN (SELECT follower FROM g.followers_ok)
);

SELECT
    u.uid AS uid,
    u.did AS did,
    coalesce(a.is_stale, toUInt8(0)) AS is_stale,
    if(
        c.host != '' AND c.host != 'unknown' AND c.inverse_coverage_weight > 0,
        c.inverse_coverage_weight,
        if(coalesce(a.is_stale, toUInt8(0)) = 1, 0.85, 1.0)
    ) AS weight
FROM g.follower_uids AS u
LEFT JOIN g.account_epochs AS a ON u.did = a.did
LEFT JOIN g.coverage AS c ON u.did = c.did
ORDER BY u.uid
INTO OUTFILE '__OUT__/follower_weights.tsv' TRUNCATE FORMAT TSV;

-- Per-account snapshot provenance and staleness report.
SELECT
    did,
    snapshot_source,
    is_stale,
    latest_tier,
    latest_time,
    has_sync_reset,
    is_deleted
FROM g.account_epochs
ORDER BY did
INTO OUTFILE '__OUT__/account_coverage.tsv' TRUNCATE FORMAT TSVWithNames;

-- Aggregate snapshot coverage & suppressed zombie-follow statistics.
SELECT
    (SELECT count() FROM g.account_epochs) AS total_accounts,
    (SELECT countIf(snapshot_source = 'october_gce_snapshot') FROM g.account_epochs) AS fresh_october_accounts,
    (SELECT countIf(snapshot_source = 'august_seed_fallback') FROM g.account_epochs) AS stale_august_fallback_accounts,
    (SELECT countIf(snapshot_source = 'october_live_incremental_only') FROM g.account_epochs) AS october_live_incremental_only_accounts,
    (SELECT countIf(has_sync_reset = 1) FROM g.account_epochs) AS sync_reset_accounts,
    (SELECT countIf(is_deleted = 1) FROM g.account_epochs) AS deleted_accounts,
    (SELECT count() FROM g.zombie_suppressed) AS zombie_follows_suppressed
INTO OUTFILE '__OUT__/coverage_stats.tsv' TRUNCATE FORMAT TSVWithNames;

SELECT
    (SELECT count() FROM g.follow_events) AS follow_records,
    (SELECT count() FROM g.follows) AS follow_edges,
    (SELECT uniqExact(follower) FROM g.follows) AS followers,
    (SELECT count() FROM g.inactive) AS inactive_accounts,
    (SELECT count() FROM g.producers) AS producers,
    (SELECT sum(followers) FROM g.producers) AS producer_edges
INTO OUTFILE '__OUT__/stats.tsv' TRUNCATE FORMAT TSVWithNames;
