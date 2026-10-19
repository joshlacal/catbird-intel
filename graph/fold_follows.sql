-- Fold archive graph events into the current follow graph and intern DIDs to
-- dense ids for the similarity tool. Disk-backed so it scales to billions of
-- follow records on the M4 Max (run clickhouse local with a persistent --path).
--
--   clickhouse local --path <work>/ch (use graph/build_graph.sh; it substitutes __EVENTS__ __OUT__ __MIN_FOLLOWERS__ __MAX_FOLLOWS__)
--
--
-- Outputs (in {out}):
--   producers.tsv   pid \t did \t followers   (pid dense 0..P-1, followers desc)
--   edges.bin       RowBinary (uid UInt32, pid UInt32), sorted by uid
--   stats.tsv       counts for the progress log

SET max_threads = 12,
    max_bytes_before_external_group_by = 16000000000,
    max_bytes_before_external_sort = 16000000000,
    optimize_aggregation_in_order = 1,
    use_structure_from_insertion_table_in_table_functions = 0;

CREATE DATABASE IF NOT EXISTS g;
DROP TABLE IF EXISTS g.follow_events;
DROP TABLE IF EXISTS g.follows;
DROP TABLE IF EXISTS g.inactive;
DROP TABLE IF EXISTS g.producers;
DROP TABLE IF EXISTS g.followers_ok;

CREATE TABLE g.follow_events
(
    did String, rkey String, rev String, operation LowCardinality(String), subject String
)
ENGINE = MergeTree ORDER BY (did, rkey);

INSERT INTO g.follow_events
SELECT did, rkey, rev, operation, subject
FROM file('__EVENTS__', Parquet)
WHERE collection = 'app.bsky.graph.follow';

-- Accounts whose latest account event is inactive are excluded on both sides.
CREATE TABLE g.inactive (did String) ENGINE = MergeTree ORDER BY did AS
SELECT did FROM file('__EVENTS__', Parquet)
WHERE kind = 'account'
GROUP BY did HAVING argMax(account_active, seq) = false;

-- Latest revision per record; TIDs sort lexicographically in time order.
CREATE TABLE g.follows (follower String, producer String) ENGINE = MergeTree ORDER BY (follower, producer) AS
SELECT DISTINCT did AS follower, subject AS producer
FROM
(
    SELECT did, rkey, argMax(operation, rev) AS op, argMax(subject, rev) AS subject
    FROM g.follow_events
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

SELECT
    (SELECT count() FROM g.follow_events) AS follow_records,
    (SELECT count() FROM g.follows) AS follow_edges,
    (SELECT uniqExact(follower) FROM g.follows) AS followers,
    (SELECT count() FROM g.inactive) AS inactive_accounts,
    (SELECT count() FROM g.producers) AS producers,
    (SELECT sum(followers) FROM g.producers) AS producer_edges
INTO OUTFILE '__OUT__/stats.tsv' TRUNCATE FORMAT TSVWithNames;
