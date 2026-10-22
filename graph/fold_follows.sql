-- Fold archive graph events into the current follow graph and intern DIDs to
-- dense ids for the similarity tool. Disk-backed so it scales to billions of
-- follow records on the M4 Max (run clickhouse local with a persistent --path).
--
--   clickhouse local --path <work>/ch (use graph/build_graph.sh; it substitutes __EVENTS__ __OUT__ __MIN_FOLLOWERS__ __MAX_FOLLOWS__)
--
-- Multi-generation snapshot & reset semantics:
--   1. Events from the August 4, 2026 seed archive (`archive-graph`, time < 2026-10-01)
--      carry `source_tier = 1`; events from the October 2026 GCE capture
--      (`gce-archive-graph`, time >= 2026-10-01) carry `source_tier = 2`.
--   2. Explicit repository reset barriers (`kind = 'sync'` or `kind = 'account' AND account_status = 'deleted'`)
--      clear all prior commit records for that `did` ordered by `(source_tier, time, seq)`.
--   3. When an account (`did`) has a repo snapshot in `source_tier = 2` (rather than only
--      isolated post-Oct-19 incremental live commits), the October snapshot replaces the
--      August 4 seed (`source_tier = 1`) so unfollows between August and October are not
--      resurrected as zombie follows.
--   4. Accounts without an October snapshot fall back to their August 4 seed (`source_tier = 1`)
--      plus any incremental live commits in `source_tier = 2`, and are explicitly labeled
--      `is_stale = 1` (`snapshot_source = 'august_seed_fallback'`) in `account_coverage.tsv`.
--   5. `g.inactive` orders account status events by `(source_tier, time, seq)` rather than raw
--      `seq`, preventing cross-archive sequence number inversions.
--
-- Outputs (in {out}):
--   producers.tsv         pid \t did \t followers   (pid dense 0..P-1, followers desc)
--   edges.bin             RowBinary (uid UInt32, pid UInt32), sorted by uid
--   stats.tsv             counts for the progress log (6 canonical columns)
--   account_coverage.tsv  did \t snapshot_source \t is_stale \t latest_tier \t latest_time \t has_sync_reset \t is_deleted
--   coverage_stats.tsv    summary of fresh vs. stale fallback accounts and suppressed zombie follows
--   follower_weights.tsv  uid \t did \t is_stale \t weight (dense 0..U-1 matching edges.bin)

SET max_threads = 12,
    max_bytes_before_external_group_by = 16000000000,
    max_bytes_before_external_sort = 16000000000,
    optimize_aggregation_in_order = 1,
    use_structure_from_insertion_table_in_table_functions = 0;

CREATE DATABASE IF NOT EXISTS g;
DROP TABLE IF EXISTS g.follow_events;
DROP TABLE IF EXISTS g.control_events;
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

CREATE TABLE g.follow_events
(
    did String,
    rkey String,
    rev String,
    operation LowCardinality(String),
    subject String,
    created_at String,
    source_tier UInt8,
    time String,
    seq UInt64
)
ENGINE = MergeTree ORDER BY (did, rkey, source_tier, rev, seq);

INSERT INTO g.follow_events
SELECT
    did,
    rkey,
    rev,
    operation,
    subject,
    created_at,
    multiIf(
        position(_path, 'gce-archive-graph') > 0 OR position(_path, 'october') > 0 OR time >= '2026-10-01',
        toUInt8(2),
        toUInt8(1)
    ) AS source_tier,
    time,
    seq
FROM file('__EVENTS__', Parquet)
WHERE collection = 'app.bsky.graph.follow' AND (kind = 'commit' OR kind = '');

CREATE TABLE g.control_events
(
    did String,
    kind LowCardinality(String),
    collection LowCardinality(String),
    operation LowCardinality(String),
    account_active Nullable(Bool),
    account_status LowCardinality(String),
    source_tier UInt8,
    time String,
    seq UInt64
)
ENGINE = MergeTree ORDER BY (did, source_tier, time, seq);

INSERT INTO g.control_events
SELECT
    did,
    kind,
    collection,
    operation,
    account_active,
    account_status,
    multiIf(
        position(_path, 'gce-archive-graph') > 0 OR position(_path, 'october') > 0 OR time >= '2026-10-01',
        toUInt8(2),
        toUInt8(1)
    ) AS source_tier,
    time,
    seq
FROM file('__EVENTS__', Parquet)
WHERE kind IN ('account', 'sync')
   OR (collection = 'app.bsky.actor.profile' AND operation = 'create');

-- Accounts whose latest account event across (source_tier, time, seq) is inactive are excluded on both sides.
CREATE TABLE g.inactive (did String) ENGINE = MergeTree ORDER BY did AS
SELECT did
FROM g.control_events
WHERE kind = 'account'
GROUP BY did
HAVING argMax(coalesce(account_active, true), tuple(source_tier, time, seq)) = false
    OR argMax(account_status, tuple(source_tier, time, seq)) IN ('deleted', 'deactivated', 'takendown', 'suspended');

-- Per-account follow-stream summary (streams in `did` order via primary key prefix).
CREATE TABLE g.follow_acct_summary
(
    did String,
    has_tier1 UInt8,
    has_tier2 UInt8,
    has_oct_follow_snapshot UInt8,
    latest_follow_tier UInt8,
    latest_follow_time String
)
ENGINE = MergeTree ORDER BY did AS
SELECT
    did,
    toUInt8(max(source_tier = 1)) AS has_tier1,
    toUInt8(max(source_tier = 2)) AS has_tier2,
    toUInt8(
        max(source_tier = 2) = 1 AND (
            maxIf(
                toUInt8(
                    created_at = ''
                    OR created_at < '2026-10-19T19:50:00Z'
                    OR substring(created_at, 1, 10) != substring(time, 1, 10)
                    OR abs(
                        (toInt64OrZero(substring(time, 12, 2)) * 60 + toInt64OrZero(substring(time, 15, 2)))
                        - (toInt64OrZero(substring(created_at, 12, 2)) * 60 + toInt64OrZero(substring(created_at, 15, 2)))
                    ) > 10
                ),
                source_tier = 2 AND operation = 'create'
            ) = 1
            OR countIf(source_tier = 2 AND operation = 'create') > uniqExactIf(rev, source_tier = 2 AND operation = 'create')
            OR (
                max(source_tier = 1) = 1
                AND countIf(source_tier = 2 AND operation = 'create') > 0
                AND countIf(operation = 'create') > uniqExactIf(rkey, operation = 'create')
            )
        )
    ) AS has_oct_follow_snapshot,
    max(source_tier) AS latest_follow_tier,
    max(time) AS latest_follow_time
FROM g.follow_events
GROUP BY did;

-- Per-account control-stream summary (sync barriers, deleted barriers, and profile snapshot markers).
CREATE TABLE g.control_acct_summary
(
    did String,
    has_tier1_ctrl UInt8,
    has_tier2_ctrl UInt8,
    has_tier2_sync_or_profile UInt8,
    has_sync_reset UInt8,
    sync_tier UInt8,
    sync_time String,
    sync_seq UInt64,
    has_deleted_barrier UInt8,
    del_tier UInt8,
    del_time String,
    del_seq UInt64,
    is_currently_deleted UInt8,
    latest_ctrl_tier UInt8,
    latest_ctrl_time String
)
ENGINE = MergeTree ORDER BY did AS
SELECT
    did,
    toUInt8(max(source_tier = 1)) AS has_tier1_ctrl,
    toUInt8(max(source_tier = 2)) AS has_tier2_ctrl,
    toUInt8(max(source_tier = 2 AND (kind = 'sync' OR (collection = 'app.bsky.actor.profile' AND operation = 'create')))) AS has_tier2_sync_or_profile,
    toUInt8(max(kind = 'sync')) AS has_sync_reset,
    (maxIf(tuple(source_tier, time, seq), kind = 'sync')).1 AS sync_tier,
    (maxIf(tuple(source_tier, time, seq), kind = 'sync')).2 AS sync_time,
    (maxIf(tuple(source_tier, time, seq), kind = 'sync')).3 AS sync_seq,
    toUInt8(max(kind = 'account' AND account_status = 'deleted')) AS has_deleted_barrier,
    (maxIf(tuple(source_tier, time, seq), kind = 'account' AND account_status = 'deleted')).1 AS del_tier,
    (maxIf(tuple(source_tier, time, seq), kind = 'account' AND account_status = 'deleted')).2 AS del_time,
    (maxIf(tuple(source_tier, time, seq), kind = 'account' AND account_status = 'deleted')).3 AS del_seq,
    toUInt8(
        max(kind = 'account') = 1
        AND argMaxIf(account_status, tuple(source_tier, time, seq), kind = 'account') = 'deleted'
    ) AS is_currently_deleted,
    max(source_tier) AS latest_ctrl_tier,
    max(time) AS latest_ctrl_time
FROM g.control_events
GROUP BY did;

-- Combined per-account epoch, reset barriers, and snapshot freshness classification.
CREATE TABLE g.account_epochs
(
    did String,
    has_tier1 UInt8,
    has_tier2 UInt8,
    has_october_snapshot UInt8,
    is_stale UInt8,
    snapshot_source LowCardinality(String),
    min_valid_tier UInt8,
    has_sync_reset UInt8,
    sync_tier UInt8,
    sync_time String,
    sync_seq UInt64,
    has_deleted_barrier UInt8,
    del_tier UInt8,
    del_time String,
    del_seq UInt64,
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
    if(f.has_oct_follow_snapshot = 1 OR c.has_tier2_sync_or_profile = 1 OR (c.has_deleted_barrier = 1 AND c.del_tier = 2), toUInt8(2), toUInt8(1)) AS min_valid_tier,
    c.has_sync_reset AS has_sync_reset,
    c.sync_tier AS sync_tier,
    c.sync_time AS sync_time,
    c.sync_seq AS sync_seq,
    c.has_deleted_barrier AS has_deleted_barrier,
    c.del_tier AS del_tier,
    c.del_time AS del_time,
    c.del_seq AS del_seq,
    c.is_currently_deleted AS is_deleted,
    greatest(f.latest_follow_tier, c.latest_ctrl_tier) AS latest_tier,
    greatest(f.latest_follow_time, c.latest_ctrl_time) AS latest_time
FROM g.follow_acct_summary AS f
FULL OUTER JOIN g.control_acct_summary AS c ON f.did = c.did;

-- Count August 4 seed follows that would have been resurrected without October snapshot replacement.
CREATE TABLE g.zombie_suppressed (did String, rkey String) ENGINE = MergeTree ORDER BY (did, rkey) AS
SELECT f.did AS did, f.rkey AS rkey
FROM g.follow_events AS f
INNER JOIN g.account_epochs AS a ON f.did = a.did
WHERE a.has_october_snapshot = 1
GROUP BY f.did, f.rkey
HAVING max(f.source_tier = 2) = 0
   AND argMaxIf(f.operation, tuple(f.rev, f.time, f.seq), f.source_tier = 1) != 'delete';

-- Filter follow events to each account's active post-reset / post-snapshot epoch.
CREATE TABLE g.active_follow_events
(
    did String,
    rkey String,
    rev String,
    operation LowCardinality(String),
    subject String,
    source_tier UInt8,
    time String,
    seq UInt64
)
ENGINE = MergeTree ORDER BY (did, rkey) AS
SELECT
    f.did AS did,
    f.rkey AS rkey,
    f.rev AS rev,
    f.operation AS operation,
    f.subject AS subject,
    f.source_tier AS source_tier,
    f.time AS time,
    f.seq AS seq
FROM g.follow_events AS f
INNER JOIN g.account_epochs AS a ON f.did = a.did
WHERE f.source_tier >= a.min_valid_tier
  AND (a.has_sync_reset = 0 OR tuple(f.source_tier, f.time, f.seq) >= tuple(a.sync_tier, a.sync_time, a.sync_seq))
  AND (a.has_deleted_barrier = 0 OR tuple(f.source_tier, f.time, f.seq) > tuple(a.del_tier, a.del_time, a.del_seq));

-- Latest revision per record within the active epoch; (source_tier, rev, time, seq) orders revisions.
CREATE TABLE g.follows (follower String, producer String) ENGINE = MergeTree ORDER BY (follower, producer) AS
SELECT DISTINCT did AS follower, subject AS producer
FROM
(
    SELECT
        did,
        rkey,
        argMax(operation, tuple(source_tier, rev, time, seq)) AS op,
        argMax(subject, tuple(source_tier, rev, time, seq)) AS subject
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
    if(coalesce(a.is_stale, toUInt8(0)) = 1, 0.85, 1.0) AS weight
FROM g.follower_uids AS u
LEFT JOIN g.account_epochs AS a ON u.did = a.did
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

