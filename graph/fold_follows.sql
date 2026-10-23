-- Fold archive graph events into the current follow graph and intern DIDs to
-- dense ids for the similarity tool. Disk-backed so it scales to billions of
-- follow records on the M4 Max (run clickhouse local with a persistent --path).
--
--   clickhouse local --path <work>/ch (use graph/build_graph.sh; it substitutes
--   __EVENTS__ __SNAPSHOTS__ __NETCUP__ __OUT__ __MIN_FOLLOWERS__ __MAX_FOLLOWS__)
--
-- Inputs:
--   __EVENTS__     extracted archive events (archive_graph_extract.py Parquet),
--                  any mix of the August public archive and the GCE capture.
--   __SNAPSHOTS__  repo_snapshot_export JSONL from a checkpoint copy of the
--                  collector's meta.pebble: did, status, snapshot_rev,
--                  completed_at, latest_rev, host, active.
--   __NETCUP__     netcup daily export root (<day>/follows.parquet,
--                  <day>/deletions.parquet, <day>/account_status.parquet).
--
-- Ordering. Every record version is ordered by its repo rev decoded to the
-- 64-bit TID integer (netcup already stores that integer). Rows without a rev
-- use their time as a TID (microseconds << 10). Envelope `time` and `seq` are
-- never used to order records: the collector re-stamps bootstrap-live events
-- with the merge time, and seq namespaces differ across sources.
--
-- Snapshot replacement. A repo download writes every record at the commit rev
-- of that download, and the collector records it as backfill.rev. A re-download
-- or resync writes a sync event at its rev. Per DID the floor is the larger of
-- the two; a record version below the floor is superseded, so a follow missing
-- from the latest snapshot disappears even though no delete was ever seen.
-- Without a floor (no completed download), every version counts and the DID is
-- reported as events_only.
--
-- Account status. netcup ingest has run continuously since 2026-10-19 13:00Z,
-- so where it has a row for a DID its latest row (by indexed_at, then seq)
-- decides. Otherwise the latest archive account event by parsed time decides,
-- and a DID with no account event at all is inactive when the collector
-- recorded it unavailable.
--
-- Outputs (in __OUT__):
--   producers.tsv         pid \t did \t followers   (pid dense 0..P-1, followers desc)
--   edges.bin             RowBinary (uid UInt32, pid UInt32), sorted by uid
--   stats.tsv             counts for the progress log
--   follower_weights.tsv  uid \t did \t is_stale \t weight (dense 0..U-1 matching edges.bin)
--   coverage.parquet      per DID: host, snapshot_source, snapshot_rev, coverage, weights
--   host_coverage.parquet / host_coverage.tsv  per-host coverage and capped inverse weights
--   account_coverage.tsv  per DID: snapshot_source, coverage, floor_rev, inactive
--   coverage_stats.tsv    coverage totals and superseded-row count

-- Sized for the 128 GB M4 Max: an explicit 80 GB ceiling, with aggregation and
-- sorts spilling to disk past 20 GB. The follow fold itself streams in key order.
SET max_threads = 14,
    max_memory_usage = 80000000000,
    max_bytes_before_external_group_by = 20000000000,
    max_bytes_before_external_sort = 20000000000,
    join_algorithm = 'auto',
    max_bytes_in_join = 16000000000,
    optimize_aggregation_in_order = 1,
    input_format_parquet_allow_missing_columns = 1,
    use_structure_from_insertion_table_in_table_functions = 0;

CREATE OR REPLACE FUNCTION tid_u64 AS (s) -> if(
    match(s, '^[234567abcdefghij][234567abcdefghijklmnopqrstuvwxyz]{12}$'),
    toUInt64(arraySum(i -> bitShiftLeft(toUInt64(position('234567abcdefghijklmnopqrstuvwxyz', substring(s, i, 1)) - 1), 5 * (13 - i)),
                      range(1, 14))),
    toUInt64(0));
CREATE OR REPLACE FUNCTION time_rev AS (t) -> bitShiftLeft(toUInt64(greatest(toUnixTimestamp64Micro(t), 0)), 10);

CREATE DATABASE IF NOT EXISTS g;
DROP TABLE IF EXISTS g.snapshots;
DROP TABLE IF EXISTS g.floors;
DROP TABLE IF EXISTS g.floor_lookup;
DROP TABLE IF EXISTS g.follow_events;
DROP TABLE IF EXISTS g.follows;
DROP TABLE IF EXISTS g.inactive;
DROP TABLE IF EXISTS g.followers_ok;
DROP TABLE IF EXISTS g.producers;
DROP TABLE IF EXISTS g.accounts;
DROP TABLE IF EXISTS g.host_coverage;
DROP TABLE IF EXISTS g.coverage;
DROP TABLE IF EXISTS g.follower_uids;
DROP TABLE IF EXISTS g.uid_lookup;
DROP TABLE IF EXISTS g.pid_lookup;

-- Collector state, one row per DID. snapshot_rev is set only for completed downloads.
CREATE TABLE g.snapshots
(
    did String, status LowCardinality(String), snap_rev UInt64,
    completed_at DateTime64(6, 'UTC'), host String, active UInt8
)
ENGINE = MergeTree ORDER BY did AS
SELECT
    did,
    argMax(st, r) AS status,
    max(if(st = 'complete', r, toUInt64(0))) AS snap_rev,
    max(done) AS completed_at,
    argMax(h, r) AS host,
    argMax(act, r) AS active
FROM
(
    SELECT did, status AS st, tid_u64(snapshot_rev) AS r, host AS h, active AS act,
           ifNull(parseDateTime64BestEffortOrNull(completed_at, 6, 'UTC'), toDateTime64(0, 6, 'UTC')) AS done
    FROM file('__SNAPSHOTS__', JSONEachRow,
              'did String, status String, snapshot_rev String, completed_at String, latest_rev String, host String, active UInt8')
    WHERE startsWith(did, 'did:')
)
GROUP BY did;

-- Per-DID supersession floor: completed download rev or any later sync/reset rev.
CREATE TABLE g.floors (did String, floor UInt64, snap_rev UInt64, sync_rev UInt64)
ENGINE = MergeTree ORDER BY did AS
SELECT did, greatest(max(s_rev), max(y_rev)) AS floor, max(s_rev) AS snap_rev, max(y_rev) AS sync_rev
FROM
(
    SELECT did, snap_rev AS s_rev, toUInt64(0) AS y_rev FROM g.snapshots WHERE snap_rev > 0
    UNION ALL
    SELECT did, toUInt64(0) AS s_rev, tid_u64(rev) AS y_rev
    FROM file('__EVENTS__', Parquet, 'kind String, did String, rev String')
    WHERE kind = 'sync' AND tid_u64(rev) > 0
)
GROUP BY did;

-- Every follow record version from all sources, in rev order space.
CREATE TABLE g.follow_events (did String, rkey String, rev UInt64, is_delete UInt8, subject String)
ENGINE = MergeTree ORDER BY (did, rkey, rev) AS
SELECT
    did, rkey,
    if(tid_u64(rev) > 0, tid_u64(rev),
       time_rev(ifNull(parseDateTime64BestEffortOrNull(time, 6, 'UTC'), toDateTime64(0, 6, 'UTC')))) AS rev,
    toUInt8(operation = 'delete') AS is_delete,
    subject
FROM file('__EVENTS__', Parquet,
          'kind String, did String, collection String, rkey String, rev String, operation String, subject String, time String')
WHERE collection = 'app.bsky.graph.follow' AND kind IN ('commit', '') AND did != '' AND rkey != ''
UNION ALL
SELECT did, rkey, if(rev > 0, rev, time_rev(indexed_at)) AS rev, is_deleted AS is_delete, subject
FROM file('__NETCUP__/*/follows.parquet', Parquet,
          'did String, rkey String, rev UInt64, is_deleted UInt8, indexed_at DateTime64(3, \'UTC\'), subject String')
UNION ALL
SELECT did, rkey, if(rev > 0, rev, time_rev(indexed_at)) AS rev, toUInt8(1) AS is_delete, '' AS subject
FROM file('__NETCUP__/*/deletions.parquet', Parquet,
          'collection String, did String, rkey String, rev UInt64, indexed_at DateTime64(3, \'UTC\')')
WHERE collection = 'app.bsky.graph.follow';

-- Accounts whose latest status is inactive are excluded on both sides.
CREATE TABLE g.inactive (did String) ENGINE = MergeTree ORDER BY did AS
WITH
    netcup AS (
        SELECT did, argMax(active, (indexed_at, seq)) AS is_active
        FROM file('__NETCUP__/*/account_status.parquet', Parquet,
                  'did String, seq UInt64, indexed_at DateTime64(3, \'UTC\'), active UInt8')
        GROUP BY did
    ),
    archive AS (
        SELECT did, toUInt8(argMax(ifNull(account_active, true), (parsed, seq))) AS is_active
        FROM
        (
            SELECT did, account_active, seq,
                   ifNull(parseDateTime64BestEffortOrNull(time, 6, 'UTC'), toDateTime64(0, 6, 'UTC')) AS parsed
            FROM file('__EVENTS__', Parquet, 'kind String, did String, seq UInt64, time String, account_active Nullable(Bool)')
            WHERE kind = 'account'
        )
        GROUP BY did
    )
SELECT did FROM netcup WHERE is_active = 0
UNION ALL
SELECT did FROM archive WHERE is_active = 0 AND did NOT IN (SELECT did FROM netcup)
UNION ALL
SELECT did FROM g.snapshots
WHERE (status = 'unavailable' OR active = 0)
  AND did NOT IN (SELECT did FROM netcup) AND did NOT IN (SELECT did FROM archive);

-- Floors as an in-memory key lookup, so the fold below reads follow_events in its
-- (did, rkey, rev) order and aggregates as it streams; a missing DID reads 0.
CREATE TABLE g.floor_lookup ENGINE = Join(ANY, LEFT, did) AS SELECT did, floor FROM g.floors;

-- Latest version per record at or above the DID's floor; a delete wins a rev tie.
-- The same account followed twice collapses in the ReplacingMergeTree merge.
CREATE TABLE g.follows (follower String, producer String) ENGINE = ReplacingMergeTree ORDER BY (follower, producer) AS
SELECT did AS follower, latest_subject AS producer
FROM
(
    SELECT did, rkey,
           argMax(is_delete, (rev, is_delete)) AS latest_delete,
           argMax(subject, (rev, is_delete)) AS latest_subject
    FROM g.follow_events
    WHERE rev >= joinGet('g.floor_lookup', 'floor', did)
    GROUP BY did, rkey
)
WHERE latest_delete = 0
  AND startsWith(latest_subject, 'did:')
  AND follower != producer
  AND follower NOT IN (SELECT did FROM g.inactive)
  AND producer NOT IN (SELECT did FROM g.inactive)
SETTINGS optimize_aggregation_in_order = 1, read_in_order_two_level_merge_threshold = 0;
OPTIMIZE TABLE g.follows FINAL;

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

-- Followers with at least one kept producer, numbered in follower order.
CREATE TABLE g.follower_uids (uid UInt32, did String) ENGINE = MergeTree ORDER BY uid AS
SELECT toUInt32(row_number() OVER (ORDER BY follower) - 1) AS uid, follower AS did
FROM
(
    SELECT follower FROM g.follows
    WHERE follower IN (SELECT follower FROM g.followers_ok)
      AND producer IN (SELECT did FROM g.producers)
    GROUP BY follower
);
CREATE TABLE g.uid_lookup ENGINE = Join(ANY, LEFT, did) AS SELECT did, uid FROM g.follower_uids;
CREATE TABLE g.pid_lookup ENGINE = Join(ANY, LEFT, did) AS SELECT did, pid FROM g.producers;

-- g.follows is stored in follower order and uids follow that order, so edges
-- stream out grouped by uid without a global sort.
SELECT joinGet('g.uid_lookup', 'uid', follower) AS uid, joinGet('g.pid_lookup', 'pid', producer) AS pid
FROM g.follows
WHERE follower IN (SELECT did FROM g.follower_uids)
  AND producer IN (SELECT did FROM g.producers)
ORDER BY follower
INTO OUTFILE '__OUT__/edges.bin' TRUNCATE FORMAT RowBinary
SETTINGS optimize_read_in_order = 1;

-- Coverage universe: the collector roster plus any DID with follow records.
-- snapshot: a completed download or sync reset bounds the DID's state.
-- events_only: follow records but no floor (e.g. August archive only).
-- missing: on the roster with no download and no follow records.
CREATE TABLE g.accounts
(
    did String, host String, snapshot_source LowCardinality(String), floor_rev UInt64,
    coverage LowCardinality(String), inactive UInt8
)
ENGINE = MergeTree ORDER BY (host, did) AS
SELECT
    u.did AS did,
    if(s.host != '', s.host, 'unknown') AS host,
    multiIf(f.snap_rev > 0, 'collector_snapshot', f.sync_rev > 0, 'sync_reset', ev.did != '', 'events_only', 'none') AS snapshot_source,
    f.floor AS floor_rev,
    multiIf(f.floor > 0, 'snapshot', ev.did != '', 'events_only', 'missing') AS coverage,
    toUInt8(u.did IN (SELECT did FROM g.inactive)) AS inactive
FROM
(
    SELECT did FROM g.snapshots
    UNION DISTINCT
    SELECT did FROM g.follow_events GROUP BY did
) AS u
LEFT JOIN g.snapshots AS s ON u.did = s.did
LEFT JOIN g.floors AS f ON u.did = f.did
LEFT JOIN (SELECT did FROM g.follow_events GROUP BY did) AS ev ON u.did = ev.did;

-- Per-host coverage over roster accounts with a known host. events_only counts
-- 0.85 of a snapshot; the inverse weight is capped at 3x so one badly covered
-- host cannot dominate producer similarity.
CREATE TABLE g.host_coverage
(
    host String, total_accounts UInt64, snapshot_accounts UInt64, events_only_accounts UInt64,
    missing_accounts UInt64, snapshot_coverage_rate Float64, effective_coverage_rate Float64,
    inverse_coverage_weight Float64
)
ENGINE = MergeTree ORDER BY host AS
SELECT
    host,
    count() AS total_accounts,
    countIf(coverage = 'snapshot') AS snapshot_accounts,
    countIf(coverage = 'events_only') AS events_only_accounts,
    countIf(coverage = 'missing') AS missing_accounts,
    round(snapshot_accounts / count(), 6) AS snapshot_coverage_rate,
    round((snapshot_accounts + 0.85 * events_only_accounts) / count(), 6) AS effective_coverage_rate,
    round(least(3.0, 1.0 / greatest(effective_coverage_rate, 1e-6)), 6) AS inverse_coverage_weight
FROM g.accounts
WHERE host != 'unknown' AND inactive = 0
GROUP BY host;

CREATE TABLE g.coverage
(
    did String, host String, snapshot_source String, snapshot_rev UInt64, coverage LowCardinality(String),
    host_snapshot_coverage_rate Float64, host_effective_coverage_rate Float64, inverse_coverage_weight Float64
)
ENGINE = MergeTree ORDER BY did AS
SELECT
    a.did, a.host, a.snapshot_source, a.floor_rev AS snapshot_rev, a.coverage,
    hc.snapshot_coverage_rate AS host_snapshot_coverage_rate,
    hc.effective_coverage_rate AS host_effective_coverage_rate,
    multiIf(
        a.coverage = 'missing', 0.0,
        hc.host = '', if(a.coverage = 'events_only', 0.85, 1.0),
        round(least(3.0, if(a.coverage = 'events_only', 0.85, 1.0) * hc.inverse_coverage_weight), 6)
    ) AS inverse_coverage_weight
FROM g.accounts AS a
LEFT JOIN g.host_coverage AS hc ON a.host = hc.host;

SELECT did, host, snapshot_source, snapshot_rev, coverage, host_snapshot_coverage_rate,
       host_effective_coverage_rate, inverse_coverage_weight
FROM g.coverage ORDER BY did
INTO OUTFILE '__OUT__/coverage.parquet' TRUNCATE FORMAT Parquet;

SELECT * FROM g.host_coverage ORDER BY total_accounts DESC, host ASC
INTO OUTFILE '__OUT__/host_coverage.parquet' TRUNCATE FORMAT Parquet;

SELECT * FROM g.host_coverage ORDER BY total_accounts DESC, host ASC
INTO OUTFILE '__OUT__/host_coverage.tsv' TRUNCATE FORMAT TSVWithNames;

-- Dense follower coverage weights aligned 1:1 with `uid` in edges.bin.
SELECT u.uid AS uid, u.did AS did,
       toUInt8(c.coverage != 'snapshot') AS is_stale,
       if(c.inverse_coverage_weight > 0, c.inverse_coverage_weight, 1.0) AS weight
FROM g.follower_uids AS u
LEFT JOIN g.coverage AS c ON u.did = c.did
ORDER BY u.uid
INTO OUTFILE '__OUT__/follower_weights.tsv' TRUNCATE FORMAT TSV;

SELECT did, snapshot_source, coverage, floor_rev, inactive FROM g.accounts ORDER BY did
INTO OUTFILE '__OUT__/account_coverage.tsv' TRUNCATE FORMAT TSVWithNames;

SELECT
    (SELECT count() FROM g.accounts) AS total_accounts,
    (SELECT countIf(snapshot_source = 'collector_snapshot') FROM g.accounts) AS collector_snapshot_accounts,
    (SELECT countIf(snapshot_source = 'sync_reset') FROM g.accounts) AS sync_reset_only_accounts,
    (SELECT countIf(coverage = 'events_only') FROM g.accounts) AS events_only_accounts,
    (SELECT countIf(coverage = 'missing') FROM g.accounts) AS missing_accounts,
    (SELECT count() FROM g.inactive) AS inactive_accounts,
    (SELECT count() FROM g.follow_events WHERE rev < joinGet('g.floor_lookup', 'floor', did))
        AS superseded_follow_rows
INTO OUTFILE '__OUT__/coverage_stats.tsv' TRUNCATE FORMAT TSVWithNames;

SELECT
    (SELECT count() FROM g.follow_events) AS follow_records,
    (SELECT count() FROM g.follows) AS follow_edges,
    (SELECT uniqExact(follower) FROM g.follows) AS followers,
    (SELECT count() FROM g.inactive) AS inactive_accounts,
    (SELECT count() FROM g.producers) AS producers,
    (SELECT sum(followers) FROM g.producers) AS producer_edges
INTO OUTFILE '__OUT__/stats.tsv' TRUNCATE FORMAT TSVWithNames;
