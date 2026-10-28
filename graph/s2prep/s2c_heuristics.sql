-- S2c heuristic outcome tests H1, H2, H5 on the four local netcup days 2026-10-19..22
-- (whole-network plan §3.1 S2 (c), §4.4, §4.8 V3/V4/V5; S1 rules §9.1 (c): reported only).
-- Directional only: four days are far shorter than the plan's 14-day outcome windows, so every
-- outcome below uses a fixed 48 h horizon. Nothing here gates anything. Run with chq.sh --fold
-- (follow state comes from g.follow_events). __CAND1__ is the cand1 directory, __OUT__ the output dir.
--
-- Shared definitions:
--   like time = rkey TID (P1); one like per (liker, post), earliest wins (P2); self-likes dropped.
--   follows(u, p, t) = a follow record u -> p with TID <= t and no observed delete at or before t (P6).
--   A follow outcome must be a new record with TID in (t, t + 48 h] that is live at the snapshot.
--   Author size bucket = followers in the fold's producer set (<100 = not a producer).

CREATE TABLE lk ENGINE = MergeTree ORDER BY (liker, author) AS
SELECT did AS liker, subject_did AS author, subject_rkey AS prkey,
       min(bitShiftRight(tid_u64(rkey), 10)) AS t_us,
       argMin(toUInt8(via_uri != ''), tid_u64(rkey)) AS via
FROM file('2026-10-{19,20,21,22}/engagements.parquet', Parquet)
WHERE is_deleted = 0 AND kind = 'like' AND subject_collection = 'app.bsky.feed.post'
  AND did != subject_did AND tid_u64(rkey) > 0
GROUP BY liker, author, prkey;

CREATE TABLE px ENGINE = MergeTree ORDER BY (author, rkey) AS
SELECT did AS author, rkey, toUInt8(reply_parent != '') AS is_reply,
       splitByChar('/', reply_parent)[3] AS par_author, splitByChar('/', reply_root)[3] AS root_author,
       bitShiftRight(tid_u64(rkey), 10) AS t_post, reply_parent, reply_root
FROM file('2026-10-{19,20,21,22}/posts.parquet', Parquet)
WHERE is_deleted = 0;

CREATE TABLE lkp ENGINE = MergeTree ORDER BY (liker, author) AS
SELECT l.liker, l.author, l.prkey, l.t_us, l.via, p.is_reply, p.par_author, p.root_author
FROM lk AS l INNER JOIN px AS p ON l.author = p.author AND l.prkey = p.rkey;

CREATE TABLE bucket_lookup ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS
SELECT did, multiIf(followers < 1000, 1, followers < 10000, 2, followers < 100000, 3, 4) AS b FROM g.producers;

SELECT 'inputs' AS step, (SELECT count() FROM lk) AS likes_dedup, (SELECT count() FROM px) AS posts,
       (SELECT count() FROM lkp) AS likes_joined_to_posts FORMAT TSVWithNames;

-- ---------------------------------------------------------------------------------------------
-- H1 courtesy likes (V3 direction): after a parent-author like on a's reply ("ack"), does the
-- liker like a top-level post of a within 48 h less often than after an ordinary like of a's reply
-- by someone who is neither the parent nor the root author ("interest")? Exposures: first such
-- like per (liker, a) with TID on 10-19 or 10-20 (so 48 h fits in the data). Pairs with any ack
-- exposure are removed from the interest group.
CREATE TABLE h1_exp ENGINE = MergeTree ORDER BY (liker, author) AS
WITH exp AS (
    SELECT liker, author, if(liker = par_author, 'ack', 'interest') AS grp, min(t_us) AS t0
    FROM lkp
    WHERE is_reply = 1 AND t_us >= 1792368000000000 AND t_us < 1792540800000000
      AND (liker = par_author OR (liker != par_author AND liker != root_author))
    GROUP BY liker, author, grp
)
SELECT * FROM exp
WHERE grp = 'ack' OR cityHash64(liker, author) NOT IN (SELECT cityHash64(liker, author) FROM exp WHERE grp = 'ack');

SELECT 'H1' AS step, if(grouping(grp) = 1, 'ALL', grp) AS grp_label,
       if(grouping(author_bucket) = 1, 'ALL', ['not_producer', '<1k', '1k-10k', '10k-100k', '>=100k'][author_bucket + 1]) AS bucket_label,
       count() AS pairs, countIf(n > 0) AS liked_top_level_within_48h, round(countIf(n > 0) / count(), 4) AS rate
FROM (
    SELECT e.grp AS grp, joinGet('default.bucket_lookup', 'b', e.author) AS author_bucket, o.n AS n
    FROM h1_exp AS e
    LEFT JOIN (
        SELECT e2.liker, e2.author, e2.grp, countIf(t.t_us > e2.t0 AND t.t_us <= e2.t0 + 172800000000) AS n
        FROM h1_exp AS e2 INNER JOIN (SELECT liker, author, t_us FROM lkp WHERE is_reply = 0) AS t
          ON t.liker = e2.liker AND t.author = e2.author
        GROUP BY e2.liker, e2.author, e2.grp
    ) AS o ON o.liker = e.liker AND o.author = e.author AND o.grp = e.grp
)
GROUP BY ROLLUP(grp, author_bucket) ORDER BY grp_label, bucket_label
SETTINGS join_use_nulls = 0
FORMAT TSVWithNames;

-- ---------------------------------------------------------------------------------------------
-- Follow records for every pair any test needs (liker -> author, both directions for H2, and the
-- H5 controls), with TID, snapshot liveness and the earliest observed delete.
CREATE TABLE orig_cnt ENGINE = Join(ANY, LEFT, author) SETTINGS persistent = 0 AS
SELECT author, toUInt32(count()) AS n_orig FROM px WHERE is_reply = 0 GROUP BY author;

-- H5 inputs: per (liker, author) likes without via, their distinct days and the qualifying time
-- (first like on the second distinct UTC day).
CREATE TABLE h5_pairs ENGINE = MergeTree ORDER BY (liker, author) AS
SELECT liker, author, count() AS n_likes, countIf(via = 0) AS n_novia,
       uniqExactIf(toDate(fromUnixTimestamp64Micro(toInt64(t_us), 'UTC')), via = 0) AS days_novia,
       arraySort(groupArrayIf(t_us, via = 0)) AS ts_novia
FROM lk GROUP BY liker, author;

CREATE TABLE prod_bucketed ENGINE = MergeTree ORDER BY (b, r) AS
SELECT did, multiIf(followers < 1000, 1, followers < 10000, 2, followers < 100000, 3, 4) AS b,
       toUInt32(row_number() OVER (PARTITION BY multiIf(followers < 1000, 1, followers < 10000, 2, followers < 100000, 3, 4) ORDER BY cityHash64(did, 's2c')) - 1) AS r
FROM g.producers;
CREATE TABLE bucket_size ENGINE = Join(ANY, LEFT, b) SETTINGS persistent = 0 AS SELECT b, toUInt32(count()) AS n FROM prod_bucketed GROUP BY b;
CREATE TABLE bucket_rank ENGINE = Join(ANY, LEFT, b, r) SETTINGS persistent = 0 AS SELECT b, r, did FROM prod_bucketed;

-- Treated: >= 2 no-via likes on >= 2 distinct days; single: exactly one like in all, no via.
-- Qualifying time tq: first no-via like on the second distinct day (treated) or the single like.
CREATE TABLE h5_q ENGINE = MergeTree ORDER BY (liker, author) AS
SELECT liker, author, grp, tq, joinGet('default.bucket_lookup', 'b', author) AS b
FROM (
    SELECT liker, author,
           if(days_novia >= 2 AND n_novia >= 2, 'repeat_2days', 'single') AS grp,
           if(days_novia >= 2,
              arrayFirst(x -> toDate(fromUnixTimestamp64Micro(toInt64(x), 'UTC')) != toDate(fromUnixTimestamp64Micro(toInt64(ts_novia[1]), 'UTC')), ts_novia),
              ts_novia[1]) AS tq
    FROM h5_pairs
    WHERE (days_novia >= 2 AND n_novia >= 2) OR (n_likes = 1 AND n_novia = 1)
)
WHERE joinGet('default.bucket_lookup', 'b', author) > 0;

-- Random controls: same liker, a producer from the author's size bucket chosen by hash, no like
-- between them in the four days; qualifying time copied from the treated pair.
CREATE TABLE h5_ctrl ENGINE = MergeTree ORDER BY (liker, author) AS
SELECT liker, ctrl AS author, 'control_no_like' AS grp, tq, b
FROM (
    SELECT liker, tq, b,
           joinGet('default.bucket_rank', 'did', b, toUInt32(cityHash64(liker, author, 'ctrl') % joinGet('default.bucket_size', 'n', b))) AS ctrl
    FROM h5_q WHERE grp = 'repeat_2days'
)
WHERE ctrl != liker AND cityHash64(liker, ctrl) NOT IN (SELECT cityHash64(liker, author) FROM lk);

CREATE TABLE pairset ENGINE = Set SETTINGS persistent = 0 AS
SELECT cityHash64(liker, author) FROM lk
UNION ALL SELECT cityHash64(author, liker) FROM lk
UNION ALL SELECT cityHash64(liker, author) FROM h5_ctrl;

CREATE TABLE floor_lookup ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS SELECT did, floor FROM g.floors;

CREATE TABLE fr ENGINE = MergeTree ORDER BY h AS
SELECT cityHash64(did, subject) AS h, c_us, live_snap, del_us
FROM (
    SELECT did, rkey, bitShiftRight(tid_u64(rkey), 10) AS c_us,
           argMaxIf(subject, rev, is_delete = 0) AS subject,
           toUInt8(argMax(is_delete, (rev, is_delete)) = 0 AND max(rev) >= joinGet('default.floor_lookup', 'floor', did)) AS live_snap,
           bitShiftRight(minIf(rev, is_delete = 1), 10) AS del_us
    FROM g.follow_events GROUP BY did, rkey
)
WHERE c_us > 0 AND cityHash64(did, subject) IN default.pairset
SETTINGS optimize_aggregation_in_order = 1, read_in_order_two_level_merge_threshold = 0;

SELECT 'follow_records' AS step, count() AS records FROM fr FORMAT TSVWithNames;
SELECT 'follow_records_by_day' AS step, toDate(fromUnixTimestamp64Micro(toInt64(c_us), 'UTC')) AS day, count() AS records,
       countIf(live_snap = 1) AS live_at_snapshot
FROM fr WHERE c_us >= 1792195200000000 GROUP BY day ORDER BY day FORMAT TSVWithNames;

-- ---------------------------------------------------------------------------------------------
-- H5 discovery likes (V5 direction): pairs with >= 2 discovery likes (no via, liker not following
-- the author at like time) on >= 2 days, vs single discovery likes and vs random no-like pairs
-- (same liker, same author size bucket). Outcome: a new follow live at the snapshot within 48 h of
-- tq. Only qualifying times with tq + 48 h <= 2026-10-23T00:00Z (the end of the four complete
-- netcup days) are kept, so treated pairs are those whose second day is 10-20.
CREATE TABLE h5_all ENGINE = MergeTree ORDER BY (liker, author) AS
SELECT * FROM h5_q UNION ALL SELECT * FROM h5_ctrl;

SELECT 'H5' AS step, if(grouping(grp) = 1, 'ALL', grp) AS grp_label,
       if(grouping(b) = 1, 'ALL', ['not_producer', '<1k', '1k-10k', '10k-100k', '>=100k'][b + 1]) AS bucket_label, count() AS pairs,
       countIf(conv = 1) AS followed_within_48h_live, round(countIf(conv = 1) / count(), 5) AS rate
FROM (
    SELECT a.liker, a.author, a.grp, a.b, a.tq,
           max(toUInt8(f.c_us > 0 AND f.c_us <= a.tq AND (f.del_us = 0 OR f.del_us > a.tq))) AS followed_at_tq,
           max(toUInt8(f.c_us > a.tq AND f.c_us <= a.tq + 172800000000 AND f.live_snap = 1)) AS conv
    FROM h5_all AS a LEFT JOIN fr AS f ON f.h = cityHash64(a.liker, a.author)
    WHERE a.tq + 172800000000 <= 1792713600000000
    GROUP BY a.liker, a.author, a.grp, a.b, a.tq
)
WHERE followed_at_tq = 0
GROUP BY ROLLUP(grp, b) ORDER BY grp_label, bucket_label
SETTINGS join_use_nulls = 0
FORMAT TSVWithNames;

-- The 'discovery' condition for the treated pairs is checked at tq only (the second-day like);
-- the share of treated pairs already following at their first like is reported for context.

-- ---------------------------------------------------------------------------------------------
-- H2 maintenance likes (V4 direction): over the four days, ordered pair (a, b) is flagged when
-- n_ab >= 5 and n_ba >= 5 distinct originals liked, min/max >= 0.5, n_ab / originals_b >= 0.5,
-- originals_b >= 5, and a and b follow each other at 2026-10-19T00:00Z. Unordered pairs flagged in
-- either direction are compared with mutual pairs that exchanged >= 1 original like each way but
-- are not flagged, on the share in the same cand1 community (raw Leiden label from membership.npy),
-- by the pair's author-size buckets. Both ends must be cand1 producers.
CREATE TABLE memb ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS
SELECT p.did AS did, m.label AS label
FROM file('__CAND1__/producers.tsv', TSV, 'pid UInt64, did String, followers UInt64') AS p
INNER JOIN (SELECT rowNumberInAllBlocks() AS pid, array AS label FROM file('__CAND1__/membership.npy', Npy) SETTINGS max_threads = 1) AS m
ON p.pid = m.pid;

CREATE TABLE lo ENGINE = MergeTree ORDER BY (a, b) AS
SELECT liker AS a, author AS b, toUInt32(count()) AS n FROM lkp WHERE is_reply = 0 GROUP BY a, b;

CREATE TABLE mutual_lookup ENGINE = Set SETTINGS persistent = 0 AS
SELECT h FROM fr WHERE c_us <= 1792368000000000 AND (del_us = 0 OR del_us > 1792368000000000) GROUP BY h;

SELECT 'H2' AS step, if(grouping(grp) = 1, 'ALL', grp) AS grp_label,
       if(grouping(bucket_lo) = 1, 'ALL', toString(bucket_lo)) AS bucket_lo_label,
       if(grouping(bucket_hi) = 1, 'ALL', toString(bucket_hi)) AS bucket_hi_label, count() AS pairs,
       countIf(same = 1) AS same_community, round(countIf(same = 1) / count(), 4) AS same_share
FROM (
    SELECT x.a AS a, x.b AS b,
           if((x.n >= 5 AND y.n >= 5 AND least(x.n, y.n) / greatest(x.n, y.n) >= 0.5
               AND joinGet('default.orig_cnt', 'n_orig', x.b) >= 5 AND x.n / joinGet('default.orig_cnt', 'n_orig', x.b) >= 0.5)
              OR (x.n >= 5 AND y.n >= 5 AND least(x.n, y.n) / greatest(x.n, y.n) >= 0.5
               AND joinGet('default.orig_cnt', 'n_orig', x.a) >= 5 AND y.n / joinGet('default.orig_cnt', 'n_orig', x.a) >= 0.5),
              'flagged', 'unflagged_mutual') AS grp,
           toUInt8(joinGet('default.memb', 'label', x.a) = joinGet('default.memb', 'label', x.b)) AS same,
           least(joinGet('default.bucket_lookup', 'b', x.a), joinGet('default.bucket_lookup', 'b', x.b)) AS bucket_lo,
           greatest(joinGet('default.bucket_lookup', 'b', x.a), joinGet('default.bucket_lookup', 'b', x.b)) AS bucket_hi
    FROM lo AS x INNER JOIN lo AS y ON x.a = y.b AND x.b = y.a
    WHERE x.a < x.b
      AND cityHash64(x.a, x.b) IN default.mutual_lookup AND cityHash64(x.b, x.a) IN default.mutual_lookup
      AND isNotNull(joinGetOrNull('default.memb', 'label', x.a)) AND isNotNull(joinGetOrNull('default.memb', 'label', x.b))
)
GROUP BY ROLLUP(grp, bucket_lo, bucket_hi) ORDER BY grp_label, bucket_lo_label, bucket_hi_label
FORMAT TSVWithNames;

-- ---------------------------------------------------------------------------------------------
-- Blind label sheets (plan S2 (c); V3 and V7). URIs only, no handles or text; Josh hydrates them
-- in the labelling tool. Drawn by hash rule, not chosen.
-- V3: 50 random ack-likes (parent author liked a reply to their post on 10-19..20).
SELECT concat('at://', l.author, '/app.bsky.feed.post/', l.prkey) AS reply_uri,
       p.reply_parent AS parent_uri, '' AS label_ack_only_or_interest_or_unclear
FROM lkp AS l INNER JOIN px AS p ON l.author = p.author AND l.prkey = p.rkey
WHERE l.is_reply = 1 AND l.liker = l.par_author AND l.t_us >= 1792368000000000 AND l.t_us < 1792540800000000
ORDER BY cityHash64(l.liker, l.author, l.prkey, 'v3') LIMIT 50
INTO OUTFILE '__OUT__/label_sheet_v3_ack_likes.tsv' TRUNCATE FORMAT TSVWithNames;

-- V7: 50 block-after items, 25 replies and 25 quotes posted on 10-20 where the target and the actor
-- blocked each other (either direction) within 48 h, by block-record TID.
CREATE TABLE bl ENGINE = Join(ANY, LEFT, k) SETTINGS persistent = 0 AS
SELECT cityHash64(did, subject) AS k, min(bitShiftRight(tid_u64(rkey), 10)) AS b_us
FROM file('2026-10-{19,20,21,22}/blocks.parquet', Parquet) WHERE is_deleted = 0 AND tid_u64(rkey) > 0 GROUP BY k;

SELECT typ, uri, target_uri, '' AS label_hostile_or_mocking_or_not_or_unclear
FROM (
    SELECT typ, uri, target_uri, actor, target, t_us,
           row_number() OVER (PARTITION BY typ ORDER BY cityHash64(uri, 'v7')) AS rn
    FROM (
        SELECT if(reply_parent != '', 'reply', 'quote') AS typ,
               concat('at://', did, '/app.bsky.feed.post/', rkey) AS uri,
               if(reply_parent != '', reply_parent, quote_uri) AS target_uri,
               did AS actor, splitByChar('/', if(reply_parent != '', reply_parent, quote_uri))[3] AS target,
               bitShiftRight(tid_u64(rkey), 10) AS t_us
        FROM file('2026-10-20/posts.parquet', Parquet)
        WHERE is_deleted = 0 AND (reply_parent != '' OR quote_uri != '')
    )
    WHERE actor != target AND (
        (ifNull(joinGetOrNull('default.bl', 'b_us', cityHash64(target, actor)), 0) > t_us
         AND joinGet('default.bl', 'b_us', cityHash64(target, actor)) <= t_us + 172800000000)
     OR (ifNull(joinGetOrNull('default.bl', 'b_us', cityHash64(actor, target)), 0) > t_us
         AND joinGet('default.bl', 'b_us', cityHash64(actor, target)) <= t_us + 172800000000))
)
WHERE rn <= 25 ORDER BY typ, rn
INTO OUTFILE '__OUT__/label_sheet_v7_block_after.tsv' TRUNCATE FORMAT TSVWithNames;

SELECT 'memb_check' AS step, (SELECT count() FROM file('__CAND1__/membership.npy', Npy)) AS membership_rows,
       (SELECT count() FROM file('__CAND1__/producers.tsv', TSV, 'pid UInt64, did String, followers UInt64')) AS producers_rows
FORMAT TSVWithNames;
