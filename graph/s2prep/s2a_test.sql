-- S2a test follows (whole-network plan §3.1 S2 (a); S1 rules §1, §5 R8, §7.1). Run with chq.sh --fold.
-- __OUT__ and __PRODUCERS_T__ (the producers.tsv written by s2a_train.sql) are replaced by the caller.
--
-- Test follows = follow records whose rkey TID falls in (T, 2026-10-18] by UTC date, i.e.
-- 1786838400000000 <= TID µs < 1792368000000000. Each row carries:
--   burst_b      follows by the same account with |TID - t| <= 30 s, itself included (S1 R8); every
--                follow record with a TID rkey counts, whatever its subject or later fate;
--   live_snap    fold_follows.sql's live-at-snapshot rule (latest version a create, rev >= floor);
--   del_us       the earliest observed delete's rev time in µs (0 if none observed);
--   subject_pid_t / subject_pid_snap   the subject's pid in the T producer set / the fold's set
--                (4294967295 when not a producer).
-- Filters as in the fold: follower and subject active, subject a DID, no self-follows.

CREATE TABLE floor_lookup ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS
SELECT did, floor FROM g.floors;
CREATE TABLE pid_t_lookup ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS
SELECT did, pid FROM file('__PRODUCERS_T__', TSV, 'pid UInt32, did String, followers UInt64');
CREATE TABLE pid_s_lookup ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS
SELECT did, pid FROM g.producers;

CREATE TABLE tf ENGINE = MergeTree ORDER BY (did, c_us) AS
SELECT did, rkey, subject, c_us, live_snap, del_us,
       count() OVER (PARTITION BY did ORDER BY c_us RANGE BETWEEN 30000000 PRECEDING AND 30000000 FOLLOWING) AS burst_b
FROM
(
    SELECT did, rkey,
           bitShiftRight(tid_u64(rkey), 10) AS c_us,
           argMaxIf(subject, rev, is_delete = 0) AS subject,
           toUInt8(argMax(is_delete, (rev, is_delete)) = 0 AND max(rev) >= joinGet('default.floor_lookup', 'floor', did)) AS live_snap,
           bitShiftRight(minIf(rev, is_delete = 1), 10) AS del_us
    FROM g.follow_events
    WHERE bitShiftRight(tid_u64(rkey), 10) >= 1786838400000000 - 30000000
      AND bitShiftRight(tid_u64(rkey), 10) < 1792368000000000 + 30000000
    GROUP BY did, rkey
)
SETTINGS optimize_aggregation_in_order = 1, read_in_order_two_level_merge_threshold = 0;

CREATE TABLE inactive_set ENGINE = Set SETTINGS persistent = 0 AS SELECT did FROM g.inactive;

SELECT did, subject, rkey, c_us, toUInt16(least(burst_b, 65535)) AS burst_b, live_snap, del_us,
       ifNull(joinGetOrNull('default.pid_t_lookup', 'pid', subject), 4294967295) AS subject_pid_t,
       ifNull(joinGetOrNull('default.pid_s_lookup', 'pid', subject), 4294967295) AS subject_pid_snap
FROM tf
WHERE c_us >= 1786838400000000 AND c_us < 1792368000000000
  AND startsWith(subject, 'did:') AND subject != did
  AND did NOT IN default.inactive_set AND subject NOT IN default.inactive_set
ORDER BY did, c_us
INTO OUTFILE '__OUT__/test_follows.parquet' TRUNCATE FORMAT Parquet
SETTINGS optimize_read_in_order = 1, output_format_parquet_compression_method = 'zstd';

-- Summary: counts, bulk share, burst-size distribution (all test follows and the M1 window).
SELECT 'test_follows' AS step, count() AS follows, countIf(live_snap = 1) AS live_at_snapshot,
       countIf(del_us > 0) AS delete_observed, countIf(burst_b >= 10) AS bulk_b_ge_10,
       round(countIf(burst_b >= 10) / count(), 4) AS bulk_share,
       round(countIf(burst_b >= 10 AND live_snap = 1) / countIf(live_snap = 1), 4) AS bulk_share_live,
       countIf(subject_pid_t != 4294967295) AS to_producer_t,
       round(countIf(burst_b >= 10 AND subject_pid_t != 4294967295) / countIf(subject_pid_t != 4294967295), 4) AS bulk_share_to_producer_t,
       countIf(c_us < 1789257600000000) AS in_m1_window,
       countIf(c_us < 1789257600000000 AND live_snap = 1 AND burst_b < 10 AND subject_pid_t != 4294967295) AS m1_primary_candidates_all_users,
       uniqExact(did) AS followers,
       min(c_us) AS min_c_us, max(c_us) AS max_c_us
FROM file('__OUT__/test_follows.parquet', Parquet) FORMAT TSVWithNames;

SELECT 'burst_hist' AS step, multiIf(burst_b = 1, '1', burst_b < 5, '2-4', burst_b < 10, '5-9', burst_b < 25, '10-24',
       burst_b < 100, '25-99', burst_b < 1000, '100-999', '1000+') AS b_bucket,
       count() AS follows, round(count() / (SELECT count() FROM file('__OUT__/test_follows.parquet', Parquet)), 4) AS share
FROM file('__OUT__/test_follows.parquet', Parquet) GROUP BY b_bucket ORDER BY min(burst_b) FORMAT TSVWithNames;

SELECT 'by_day' AS step, toDate(fromUnixTimestamp64Micro(toInt64(c_us), 'UTC')) AS day, count() AS follows,
       round(countIf(burst_b >= 10) / count(), 4) AS bulk_share, round(countIf(live_snap = 1) / count(), 4) AS live_share
FROM file('__OUT__/test_follows.parquet', Parquet) GROUP BY day ORDER BY day FORMAT TSVWithNames;
