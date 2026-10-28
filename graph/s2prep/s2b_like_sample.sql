-- S2b (3) sample for the 59.7% follower share: the original 1% sample of 2026-10-21 non-self likes
-- and reposts of posts (cityHash64(rkey) % 100 = 7), mapped to the same follow source the original
-- measurement used: graph-build-v3/out-full-20261024 follower_weights.tsv (uid) and producers.tsv
-- (pid), then checked against that build's edges.bin by s2b_follow_share.py. Query copied from the
-- 2026-10-25 run (with --join_use_nulls=1, as in the run whose output the report quotes); only the
-- output path changes. __OUT__ is replaced by the caller.
SET join_use_nulls = 1;
WITH s AS (SELECT kind, toUInt8(via_uri != '') via, did AS liker, subject_did AS author
  FROM file('/Users/joshlacalamito/catbird-lake-local/netcup-export/2026-10-21/engagements.parquet', Parquet)
  WHERE is_deleted=0 AND subject_collection='app.bsky.feed.post' AND did != subject_did AND cityHash64(rkey) % 100 = 7),
 fw AS (SELECT id AS fid, did FROM file('/Users/joshlacalamito/catbird-lake-local/graph-build-v3/out-full-20261024/follower_weights.tsv', TSV, 'id UInt32, did String, a UInt32, w Float32') WHERE did IN (SELECT liker FROM s)),
 pr AS (SELECT id AS pid, did, followers FROM file('/Users/joshlacalamito/catbird-lake-local/graph-build-v3/out-full-20261024/producers.tsv', TSV, 'id UInt32, did String, followers UInt32') WHERE did IN (SELECT author FROM s))
SELECT s.kind, s.via, ifNull(fw.fid, 4294967295) fid, ifNull(pr.pid, 4294967295) pid, ifNull(pr.followers, 0) followers
FROM s LEFT JOIN fw ON s.liker = fw.did LEFT JOIN pr ON s.author = pr.did
INTO OUTFILE '__OUT__' TRUNCATE FORMAT TSV;
