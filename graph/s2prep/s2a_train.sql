-- S2a train edges at T (whole-network plan §3.1 S2 (a); S1 rules §1, §9.1, R1).
-- Run with chq.sh --fold (read-only attach of the fold DB from an APFS clone). All tables created
-- here live in clickhouse-local's own temporary database and vanish at exit; outputs are files
-- under __OUT__ (replaced by the caller).
--
-- T = 2026-08-15, day-granular UTC: pre-T means rkey TID < 2026-08-16T00:00:00Z = 1786838400000000 µs.
-- A follow's event time is its rkey TID (P1); a delete's time is its commit rev (P6, ground-semantics P6).
--
-- Per follow record (did, rkey), over all its versions in g.follow_events:
--   snap_live  = fold_follows.sql's rule: the latest version (delete wins a rev tie) is a create and
--                its rev is at or above the DID's snapshot floor.
--   tlive      = live at T by P6: rkey TID < T, no observed delete with rev < T, and either live at
--                the snapshot or deleted by an observed delete with rev >= T. A record that is missing
--                from the snapshot with no observed delete was deleted at an unknown time, so it is not
--                counted as live at T (the conservative reading, logged in DECISIONS 2026-10-28).
-- Both sides exclude inactive accounts, self-follows and non-DID subjects, exactly as the fold does.
-- Producers and the follow-spam cut are recomputed at T from T-live edges (plan §4.8 "producer
-- eligibility leakage: recompute it at T"): followers with <= 20,000 distinct T-live follows count,
-- producers have >= 100 such followers. The same recomputation on snapshot-live edges must give the
-- fold's 2,035,285 producers and 2,036,276,732 producer edges (check (a)).

CREATE TABLE floor_lookup ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS
SELECT did, floor FROM g.floors;

CREATE TABLE rec
(
    did String, subject String, c_us UInt64,
    snap_live UInt8, tlive UInt8, del_after_T UInt8, del_before_T UInt8
)
ENGINE = MergeTree ORDER BY (did, subject)
SETTINGS parts_to_delay_insert = 100000, parts_to_throw_insert = 100000, max_parts_in_total = 1000000;

INSERT INTO rec
SELECT did, subject, c_us, snap_live,
       toUInt8(c_us > 0 AND c_us < 1786838400000000 AND del_before_T = 0 AND (snap_live = 1 OR del_after_T = 1)) AS tlive,
       del_after_T, del_before_T
FROM
(
    SELECT did, rkey,
           bitShiftRight(tid_u64(rkey), 10) AS c_us,
           argMaxIf(subject, rev, is_delete = 0) AS subject,
           toUInt8(argMax(is_delete, (rev, is_delete)) = 0 AND max(rev) >= joinGet('default.floor_lookup', 'floor', did)) AS snap_live,
           toUInt8(countIf(is_delete = 1 AND bitShiftRight(rev, 10) >= 1786838400000000) > 0) AS del_after_T,
           toUInt8(countIf(is_delete = 1 AND bitShiftRight(rev, 10) < 1786838400000000) > 0) AS del_before_T
    FROM g.follow_events
    WHERE did NOT IN (SELECT did FROM g.inactive) AND (__SAMPLE__)
    GROUP BY did, rkey
)
WHERE startsWith(subject, 'did:') AND subject != did AND subject NOT IN (SELECT did FROM g.inactive)
  AND (snap_live = 1 OR tlive = 1)
SETTINGS optimize_aggregation_in_order = 1, read_in_order_two_level_merge_threshold = 0,
         max_insert_threads = 4, max_insert_block_size = 8000000, min_insert_block_size_rows = 8000000, min_insert_block_size_bytes = 0;

SELECT 'rec' AS step, count() AS records, countIf(snap_live = 1) AS snap_live_records, countIf(tlive = 1) AS tlive_records,
       countIf(tlive = 1 AND snap_live = 0) AS tlive_deleted_after_T, countIf(snap_live = 1 AND c_us = 0) AS snap_live_non_tid_rkey,
       countIf(snap_live = 1 AND c_us >= 1786838400000000) AS snap_live_created_on_or_after_T,
       countIf(del_before_T = 1) AS records_with_delete_before_T, max(c_us * tlive) AS max_tlive_c_us
FROM rec FORMAT TSVWithNames;

-- Per-follower distinct follow counts at the snapshot and at T (in order on rec's sort key).
CREATE TABLE deg ENGINE = MergeTree ORDER BY did AS
SELECT did, uniqExactIf(subject, snap_live = 1) AS n_snap, uniqExactIf(subject, tlive = 1) AS n_t
FROM rec GROUP BY did
SETTINGS optimize_aggregation_in_order = 1, read_in_order_two_level_merge_threshold = 0;

CREATE TABLE ok_lookup ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS
SELECT did, toUInt8(n_snap BETWEEN 1 AND 20000) AS ok_s, toUInt8(n_t BETWEEN 1 AND 20000) AS ok_t FROM deg;

SELECT 'deg' AS step, countIf(n_snap > 0) AS followers_snap, countIf(n_snap BETWEEN 1 AND 20000) AS followers_ok_snap,
       countIf(n_t > 0) AS followers_t, countIf(n_t BETWEEN 1 AND 20000) AS followers_ok_t,
       sum(n_snap) AS distinct_pairs_snap, sum(n_t) AS distinct_pairs_t
FROM deg FORMAT TSVWithNames;

-- Follower counts per subject from distinct pairs (pairs deduped in order, then hashed by subject).
CREATE TABLE prod_cnt ENGINE = MergeTree ORDER BY subject AS
SELECT subject,
       countIf(s = 1 AND joinGet('default.ok_lookup', 'ok_s', did) = 1) AS f_snap,
       countIf(t = 1 AND joinGet('default.ok_lookup', 'ok_t', did) = 1) AS f_t
FROM (SELECT did, subject, max(snap_live) AS s, max(tlive) AS t FROM rec GROUP BY did, subject)
GROUP BY subject
SETTINGS optimize_aggregation_in_order = 1, read_in_order_two_level_merge_threshold = 0;

-- Check (a): the snapshot recomputation must equal the fold (and g.producers) exactly.
SELECT 'check_a' AS step,
       countIf(f_snap >= 100) AS producers_snap, sumIf(f_snap, f_snap >= 100) AS producer_edges_snap,
       countIf(f_t >= 100) AS producers_t, sumIf(f_t, f_t >= 100) AS producer_edges_t,
       (SELECT count() FROM g.producers) AS fold_producers, (SELECT sum(followers) FROM g.producers) AS fold_producer_edges,
       (SELECT count() FROM g.producers AS gp INNER JOIN (SELECT subject, f_snap FROM prod_cnt WHERE f_snap >= 100) AS pc
          ON gp.did = pc.subject AND gp.followers = pc.f_snap) AS producers_matching_fold_exactly
FROM prod_cnt FORMAT TSVWithNames;

-- Producers at T: pid dense by followers at T desc, then did (fold_follows.sql's order).
CREATE TABLE prod_t ENGINE = MergeTree ORDER BY did AS
SELECT subject AS did, f_t AS followers, toUInt32(row_number() OVER (ORDER BY f_t DESC, subject) - 1) AS pid
FROM prod_cnt WHERE f_t >= 100;
CREATE TABLE pid_lookup ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS SELECT did, pid FROM prod_t;

-- T-live producer edges, one row per distinct (follower, producer), followers numbered in did order.
CREATE TABLE tedges (did String, pid UInt32, c_us UInt64) ENGINE = MergeTree ORDER BY did
SETTINGS parts_to_delay_insert = 100000, parts_to_throw_insert = 100000, max_parts_in_total = 1000000 AS
SELECT did, joinGet('default.pid_lookup', 'pid', subject) AS pid, c_us
FROM (SELECT did, subject, max(tlive) AS t, minIf(c_us, tlive = 1) AS c_us FROM rec GROUP BY did, subject)
WHERE t = 1 AND joinGet('default.ok_lookup', 'ok_t', did) = 1 AND subject IN (SELECT did FROM prod_t)
SETTINGS optimize_aggregation_in_order = 1, read_in_order_two_level_merge_threshold = 0, max_insert_threads = 4, max_insert_block_size = 8000000, min_insert_block_size_rows = 8000000, min_insert_block_size_bytes = 0;

CREATE TABLE uid_t ENGINE = MergeTree ORDER BY uid AS
SELECT toUInt32(row_number() OVER (ORDER BY did) - 1) AS uid, did FROM (SELECT DISTINCT did FROM tedges);
CREATE TABLE uid_lookup ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS SELECT did, uid FROM uid_t;

SELECT 'train' AS step, count() AS train_producer_edges, uniqExact(did) AS train_followers, uniqExact(pid) AS train_producers,
       max(c_us) AS max_train_c_us, fromUnixTimestamp64Micro(toInt64(max(c_us)), 'UTC') AS max_train_tid_time,
       toUInt8(max(c_us) < 1786838400000000) AS max_train_tid_before_T, min(c_us) AS min_train_c_us
FROM tedges FORMAT TSVWithNames;

-- Outputs, in the graph-build layout simgraph reads, plus Parquet copies.
SELECT pid, did, followers FROM prod_t ORDER BY pid
INTO OUTFILE '__OUT__/producers.tsv' TRUNCATE FORMAT TSV;

SELECT u.uid AS uid, u.did AS did, toUInt8(ifNull(c.coverage, '') != 'snapshot') AS is_stale,
       if(ifNull(c.inverse_coverage_weight, 0) > 0, c.inverse_coverage_weight, 1.0) AS weight
FROM uid_t AS u LEFT JOIN g.coverage AS c ON u.did = c.did
ORDER BY u.uid
INTO OUTFILE '__OUT__/follower_weights.tsv' TRUNCATE FORMAT TSV
SETTINGS join_use_nulls = 1;

SELECT joinGet('default.uid_lookup', 'uid', did) AS uid, pid FROM tedges ORDER BY did
INTO OUTFILE '__OUT__/edges.bin' TRUNCATE FORMAT RowBinary
SETTINGS optimize_read_in_order = 1;

SELECT joinGet('default.uid_lookup', 'uid', did) AS uid, pid, c_us FROM tedges ORDER BY did
INTO OUTFILE '__OUT__/edges.parquet' TRUNCATE FORMAT Parquet
SETTINGS optimize_read_in_order = 1, output_format_parquet_compression_method = 'zstd';

SELECT 'snapshot_vs_t_followers' AS step,
       countIf(ok_s = 1 AND ok_t = 0) AS ok_at_snapshot_not_t, countIf(ok_s = 0 AND ok_t = 1) AS ok_at_t_not_snapshot
FROM (SELECT did, toUInt8(n_snap BETWEEN 1 AND 20000) AS ok_s, toUInt8(n_t BETWEEN 1 AND 20000) AS ok_t FROM deg)
FORMAT TSVWithNames;

SELECT 'producers_t_vs_snapshot' AS step,
       (SELECT count() FROM prod_t) AS producers_t,
       (SELECT count() FROM prod_t WHERE did IN (SELECT did FROM g.producers)) AS producers_t_also_snapshot,
       (SELECT count() FROM g.producers WHERE did NOT IN (SELECT did FROM prod_t)) AS snapshot_producers_not_t
FORMAT TSVWithNames;
