-- S2b reproduction gate (plan §3.1 S2 check (b); S1 rules §9.1): the three ground-semantics base
-- rates, recomputed with the exact queries of the 2026-10-25 measurement
-- (docs/research/2026-10-25-whole-network-plan-agent-reports/ground-semantics.md §3.3-3.4; the
-- original commands were recovered from that session's transcript, workflow wf_e83f4af9-575;
-- only the output format and a metric-name column are added).
-- Run from ~/catbird-lake-local/netcup-export (chq.sh does the cd). Read-only: file() inputs only,
-- except the INTO OUTFILE sample that feeds s2b_follow_share.py.
-- Expected: parent-author like 0.4920, reply-back 0.2800 (each ±0.01). The 0.597 follower share
-- is computed by s2b_follow_share.py from the sample written by s2b_like_sample.sql.

-- (1) 49.2%: replies on 10-20 to parents created on or after 10-19, replier != parent author,
--     liked by the parent author on any of the four days.
WITH lk AS (SELECT cityHash64(did, subject_did, subject_rkey) k FROM file('2026-10-{19,20,21,22}/engagements.parquet', Parquet) WHERE is_deleted=0 AND kind='like' AND subject_collection='app.bsky.feed.post'),
 r AS (SELECT did AS replier, rkey,
        splitByChar('/', reply_parent)[3] AS par_did, splitByChar('/', reply_parent)[5] AS par_rkey,
        bitShiftRight(arrayFold((acc, c) -> acc*32 + toUInt64(position('234567abcdefghijklmnopqrstuvwxyz', c) - 1), splitByString('', splitByChar('/', reply_parent)[5]), toUInt64(0)), 10) AS par_us
       FROM file('2026-10-20/posts.parquet', Parquet) WHERE is_deleted=0 AND reply_parent != '' AND length(splitByChar('/', reply_parent)[5]) = 13),
 w AS (SELECT * FROM r WHERE replier != par_did AND par_us >= toUnixTimestamp(toDateTime('2026-10-19 00:00:00','UTC'))*1000000)
SELECT 'parent_author_like' AS metric, count() replies,
 round(countIf(cityHash64(par_did, replier, rkey) IN (SELECT k FROM lk))/count(),4) parent_author_liked_reply,
 round(countIf(cityHash64(replier, par_did, par_rkey) IN (SELECT k FROM lk))/count(),4) replier_liked_parent,
 countIf(cityHash64(par_did, replier, rkey) IN (SELECT k FROM lk)) parent_author_liked_n
FROM w FORMAT TSVWithNames;

-- (2) 28.0%: the same replies, answered by the parent author with a post whose reply_parent is
--     the reply, on any of the four days.
WITH rr AS (SELECT cityHash64(did, reply_parent) k FROM file('2026-10-{19,20,21,22}/posts.parquet', Parquet) WHERE is_deleted=0 AND reply_parent != ''),
 r AS (SELECT did, rkey, splitByChar('/', reply_parent)[3] par_did, bitShiftRight(arrayFold((acc, c) -> acc*32 + toUInt64(position('234567abcdefghijklmnopqrstuvwxyz', c) - 1), splitByString('', splitByChar('/', reply_parent)[5]), toUInt64(0)), 10) par_us
       FROM file('2026-10-20/posts.parquet', Parquet) WHERE is_deleted=0 AND reply_parent != '' AND length(splitByChar('/', reply_parent)[5]) = 13),
 w AS (SELECT * FROM r WHERE did != par_did AND par_us >= toUnixTimestamp(toDateTime('2026-10-19 00:00:00','UTC'))*1000000)
SELECT 'reply_back' AS metric, count() replies,
 round(countIf(cityHash64(par_did, concat('at://', did, '/app.bsky.feed.post/', rkey)) IN (SELECT k FROM rr))/count(),4) parent_author_replied_back,
 countIf(cityHash64(par_did, concat('at://', did, '/app.bsky.feed.post/', rkey)) IN (SELECT k FROM rr)) replied_back_n
FROM w FORMAT TSVWithNames;
