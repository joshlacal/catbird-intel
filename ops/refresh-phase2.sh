#!/usr/bin/env bash
# Refresh Phase 2 & Phase 3 tables in ClickHouse `catbird` DB on netcup:
#   1. catbird.interested_in       (active users' top-50 communities, social_proof >= 2)
#   2. catbird.producer_influence  (producer Influence + mode mix + role)
#   3. catbird.author_mode_mix     (Phase 3 per-author mode mix by post type + Follow originals only eligibility)
#   4. catbird.item_communities    (streaming post community vectors, t=0 prior + 8h half-life engagers)
#   5. catbird.community_items     (top-200 posts per community by community-local engagement residual)
#   6. catbird.community_directory (top accounts by role, feeds, packs with hub/floor/ceiling filters)
set -euo pipefail

VERSION="${1:-$(clickhouse-client -q "SELECT toString(version) FROM catbird.known_for ORDER BY version DESC LIMIT 1")}"
if [[ -z "$VERSION" ]]; then
  echo "No version found in catbird.known_for" >&2
  exit 1
fi

POL_REGEX='\\b(trump|biden|kamala|harris|vance|walz|desantis|obama|pelosi|schumer|mcconnell|aoc|bernie|sanders|maga|democrat|democrats|republican|republicans|gop|dnc|rnc|liberal|liberals|conservative|conservatives|leftist|right-wing|far-right|fascist|fascism|authoritarian|antifa|woke|dei|scotus|supreme court|congress|senate|filibuster|impeach|ballot|midterms|electoral|polling|pollster|gerrymander|white house|doj|ice raids|deportation|tariffs|zionist|intifada|netanyahu|hamas|idf|gaza|kremlin|putin|zelensky|nato|starmer|tories|labour party)\\b'
HEAT_REGEX='\\b(idiot|idiots|moron|morons|scumbag|disgusting|pathetic|coward|cowards|evil|vile|despicable|trash|clown|clowns|liar|liars|corrupt|grifter|grifters|traitor|traitors|braindead|delusional|unhinged|deranged|hypocrite|bootlicker|shill|hack|nazis|bigot|embarrassing|shameful|ridiculous|absurd|laughable|dishonest|bad faith|gaslighting|propaganda|disgrace|outrageous|incompetent|malpractice|complicit|cowardice|ratioed|cope harder|utterly clueless|worst take)\\b'

clickhouse-client --multiquery <<SQL
-- 1. Refresh InterestedIn (follows weight 2.0 + likes 1.0 + reposts 2.0 -> KnownFor, social_proof >= 2, top 50)
TRUNCATE TABLE catbird.interested_in;

INSERT INTO catbird.interested_in (did, community, scene_id, score, social_proof, rank, updated_at)
WITH
  kf AS (
    SELECT did, community, scene_id, score
    FROM catbird.known_for FINAL
    WHERE version = '$VERSION' AND rank < 3
  ),
  user_edges AS (
    SELECT did AS user_did, subject AS producer_did, 2.0 AS w
    FROM catbird.follows
    WHERE is_deleted = 0
    UNION ALL
    SELECT did AS user_did, subject_did AS producer_did, multiIf(kind = 'repost', 2.0, 1.0) AS w
    FROM catbird.engagements
    WHERE is_deleted = 0 AND indexed_at >= now() - INTERVAL 7 DAY
  ),
  agg AS (
    SELECT
      u.user_did AS did,
      k.community AS community,
      any(k.scene_id) AS scene_id,
      sum(u.w * k.score) AS raw_score,
      toUInt16(least(uniqExact(u.producer_did), 65535)) AS social_proof
    FROM user_edges u
    INNER JOIN kf k ON u.producer_did = k.did
    WHERE u.user_did != u.producer_did
    GROUP BY did, community
    HAVING social_proof >= 2
  ),
  ranked AS (
    SELECT
      did,
      community,
      scene_id,
      raw_score,
      social_proof,
      row_number() OVER (PARTITION BY did ORDER BY raw_score DESC, community ASC) AS rnk,
      sum(raw_score) OVER (PARTITION BY did) AS total_score
    FROM agg
  )
SELECT
  did,
  community,
  scene_id,
  toFloat32(raw_score / greatest(total_score, 1e-6)) AS score,
  social_proof,
  toUInt8(rnk - 1) AS rank,
  now() AS updated_at
FROM ranked
WHERE rnk <= 50;

-- 2. Refresh Producer Influence + Posting-Mode Mix + Account Role
TRUNCATE TABLE catbird.producer_influence;

INSERT INTO catbird.producer_influence
  (did, community, influence, engagers, followers_est, role, original_share, reply_share, repost_share, quote_share, political_share, updated_at)
WITH
  kf AS (
    SELECT did, community, score
    FROM catbird.known_for FINAL
    WHERE version = '$VERSION' AND rank = 0
  ),
  foll AS (
    SELECT community, top_dids, top_followers
    FROM catbird.communities FINAL
    WHERE version = '$VERSION'
  ),
  foll_map AS (
    SELECT
      arrayJoin(arrayZip(top_dids, top_followers)) AS pair,
      pair.1 AS did,
      toUInt32(least(pair.2, 100000000)) AS followers_est
    FROM foll
  ),
  post_modes AS (
    SELECT
      did,
      count() AS n_posts,
      countIf(reply_parent = '' AND quote_uri = '') AS n_orig,
      countIf(reply_parent != '') AS n_reply,
      countIf(quote_uri != '') AS n_quote,
      countIf(
        politicalness >= 0.45 OR match(lowerUTF8(text), '$POL_REGEX')
      ) AS n_pol
    FROM catbird.posts
    WHERE is_deleted = 0 AND indexed_at >= now() - INTERVAL 14 DAY
    GROUP BY did
  ),
  repost_counts AS (
    SELECT did, countIf(kind = 'repost') AS n_reposts
    FROM catbird.engagements
    WHERE is_deleted = 0 AND indexed_at >= now() - INTERVAL 14 DAY
    GROUP BY did
  ),
  eng_inf AS (
    SELECT
      e.subject_did AS did,
      k.community AS community,
      uniqExact(e.did) AS engagers,
      sum(multiIf(e.kind = 'repost', 2.0, 1.0) * k.score) AS raw_eng
    FROM catbird.engagements e
    INNER JOIN kf k ON e.subject_did = k.did
    WHERE e.is_deleted = 0 AND e.indexed_at >= now() - INTERVAL 14 DAY
    GROUP BY did, community
  )
SELECT
  k.did AS did,
  k.community AS community,
  toFloat32(k.score * log1p(coalesce(ei.raw_eng, 0) + coalesce(fm.followers_est, 50) * 0.1)) AS influence,
  toUInt32(coalesce(ei.engagers, 0)) AS engagers,
  toUInt32(coalesce(fm.followers_est, 50)) AS followers_est,
  multiIf(
    (coalesce(rc.n_reposts, 0) + coalesce(pm.n_quote, 0)) / greatest(coalesce(pm.n_posts, 0) + coalesce(rc.n_reposts, 0), 1) >= 0.50, 'curator',
    coalesce(pm.n_reply, 0) / greatest(coalesce(pm.n_posts, 0), 1) >= 0.45, 'conversational',
    'creator'
  ) AS role,
  toFloat32(coalesce(pm.n_orig, 1) / greatest(coalesce(pm.n_posts, 0) + coalesce(rc.n_reposts, 0), 1)) AS original_share,
  toFloat32(coalesce(pm.n_reply, 0) / greatest(coalesce(pm.n_posts, 0) + coalesce(rc.n_reposts, 0), 1)) AS reply_share,
  toFloat32(coalesce(rc.n_reposts, 0) / greatest(coalesce(pm.n_posts, 0) + coalesce(rc.n_reposts, 0), 1)) AS repost_share,
  toFloat32(coalesce(pm.n_quote, 0) / greatest(coalesce(pm.n_posts, 0) + coalesce(rc.n_reposts, 0), 1)) AS quote_share,
  toFloat32(coalesce(pm.n_pol, 0) / greatest(coalesce(pm.n_posts, 1), 1)) AS political_share,
  now() AS updated_at
FROM kf k
LEFT JOIN eng_inf ei ON k.did = ei.did AND k.community = ei.community
LEFT JOIN (SELECT did, max(followers_est) AS followers_est FROM foll_map GROUP BY did) fm ON k.did = fm.did
LEFT JOIN post_modes pm ON k.did = pm.did
LEFT JOIN repost_counts rc ON k.did = rc.did;

-- 3. Refresh Phase 3 Author Mode Mix & "Follow originals only" eligibility
TRUNCATE TABLE catbird.author_mode_mix;

INSERT INTO catbird.author_mode_mix
  (did, handle, community, followers_est, total_events_14d, orig_posts, reply_posts, quote_posts, reposts,
   orig_political_share, reply_political_share, quote_political_share, repost_political_share, overall_political_share,
   orig_heat_mean, overall_heat_mean, follow_originals_only_eligible, recommend_mode, role, active_days_14d,
   has_no_unauth_label, hide_from_algorithmic, updated_at)
WITH
  kf AS (
    SELECT did, community, score
    FROM catbird.known_for FINAL
    WHERE version = '$VERSION' AND rank = 0
  ),
  pi AS (
    SELECT did, max(followers_est) AS followers_est, any(role) AS role
    FROM catbird.producer_influence
    GROUP BY did
  ),
  ident AS (
    SELECT did, argMax(handle, seq) AS handle
    FROM catbird.identities
    GROUP BY did
  ),
  prof_labels AS (
    SELECT did, max(has(self_labels, '!no-unauthenticated')) AS no_unauth
    FROM catbird.profiles
    WHERE is_deleted = 0
    GROUP BY did
  ),
  vis AS (
    SELECT did, max(hide_algo) AS hide_algo
    FROM (
      SELECT did, toUInt8(hide_from_algorithmic) AS hide_algo
      FROM catbird.visibility
      WHERE is_deleted = 0
      UNION ALL
      SELECT did, toUInt8(hide_from_recs) AS hide_algo
      FROM catbird.known_for FINAL
      WHERE version = '$VERSION' AND hide_from_recs = 1
    )
    GROUP BY did
  ),
  post_breakdown AS (
    SELECT
      did,
      countIf(reply_parent = '' AND quote_uri = '') AS orig_posts,
      countIf(reply_parent != '') AS reply_posts,
      countIf(quote_uri != '') AS quote_posts,
      countIf(reply_parent = '' AND quote_uri = '' AND (politicalness >= 0.45 OR match(lowerUTF8(text), '$POL_REGEX'))) AS orig_pol,
      countIf(reply_parent != '' AND (politicalness >= 0.45 OR match(lowerUTF8(text), '$POL_REGEX'))) AS reply_pol,
      countIf(quote_uri != '' AND (politicalness >= 0.45 OR match(lowerUTF8(text), '$POL_REGEX'))) AS quote_pol,
      avgIf(greatest(heat, multiIf(match(lowerUTF8(text), '$HEAT_REGEX'), 0.75, 0.05)), reply_parent = '' AND quote_uri = '') AS orig_heat,
      avg(greatest(heat, multiIf(match(lowerUTF8(text), '$HEAT_REGEX'), 0.75, 0.05))) AS all_heat,
      uniqExact(toDate(indexed_at)) AS active_days
    FROM catbird.posts
    WHERE is_deleted = 0 AND indexed_at >= now() - INTERVAL 14 DAY
    GROUP BY did
  ),
  repost_breakdown AS (
    SELECT
      e.did AS did,
      count() AS reposts,
      countIf( coalesce(sp.is_pol, 0) = 1 ) AS repost_pol
    FROM catbird.engagements e
    LEFT JOIN (
      SELECT
        did,
        rkey,
        max(politicalness >= 0.45 OR match(lowerUTF8(text), '$POL_REGEX')) AS is_pol
      FROM catbird.posts
      WHERE is_deleted = 0
      GROUP BY did, rkey
    ) sp ON e.subject_did = sp.did AND e.subject_rkey = sp.rkey
    WHERE e.is_deleted = 0 AND e.kind = 'repost' AND e.indexed_at >= now() - INTERVAL 14 DAY
    GROUP BY e.did
  )
SELECT
  k.did AS did,
  coalesce(nullIf(id.handle, ''), k.did) AS handle,
  k.community AS community,
  toUInt32(coalesce(p.followers_est, 50)) AS followers_est,
  toUInt32(coalesce(pb.orig_posts, 0) + coalesce(pb.reply_posts, 0) + coalesce(pb.quote_posts, 0) + coalesce(rb.reposts, 0)) AS total_events_14d,
  toUInt32(coalesce(pb.orig_posts, 0)) AS orig_posts,
  toUInt32(coalesce(pb.reply_posts, 0)) AS reply_posts,
  toUInt32(coalesce(pb.quote_posts, 0)) AS quote_posts,
  toUInt32(coalesce(rb.reposts, 0)) AS reposts,
  toFloat32(coalesce(pb.orig_pol, 0) / greatest(coalesce(pb.orig_posts, 0), 1)) AS orig_political_share,
  toFloat32(coalesce(pb.reply_pol, 0) / greatest(coalesce(pb.reply_posts, 0), 1)) AS reply_political_share,
  toFloat32(coalesce(pb.quote_pol, 0) / greatest(coalesce(pb.quote_posts, 0), 1)) AS quote_political_share,
  toFloat32(coalesce(rb.repost_pol, 0) / greatest(coalesce(rb.reposts, 0), 1)) AS repost_political_share,
  toFloat32(
    (coalesce(pb.orig_pol, 0) + coalesce(pb.reply_pol, 0) + coalesce(pb.quote_pol, 0) + coalesce(rb.repost_pol, 0))
    / greatest(coalesce(pb.orig_posts, 0) + coalesce(pb.reply_posts, 0) + coalesce(pb.quote_posts, 0) + coalesce(rb.reposts, 0), 1)
  ) AS overall_political_share,
  toFloat32(if(isNaN(pb.orig_heat) OR pb.orig_heat = 0, 0.05, pb.orig_heat)) AS orig_heat_mean,
  toFloat32(if(isNaN(pb.all_heat) OR pb.all_heat = 0, 0.06, pb.all_heat)) AS overall_heat_mean,
  toUInt8(
    coalesce(pb.orig_posts, 0) >= 1
    AND (coalesce(pb.orig_pol, 0) / greatest(coalesce(pb.orig_posts, 0), 1)) <= 0.18
    AND (
      (coalesce(rb.repost_pol, 0) / greatest(coalesce(rb.reposts, 0), 1)) >= 0.30
      OR (coalesce(pb.quote_pol, 0) / greatest(coalesce(pb.quote_posts, 0), 1)) >= 0.30
      OR (coalesce(pb.reply_pol, 0) / greatest(coalesce(pb.reply_posts, 0), 1)) >= 0.35
      OR (coalesce(rb.reposts, 0) + coalesce(pb.quote_posts, 0)) >= greatest(coalesce(pb.orig_posts, 0) * 2, 4)
    )
  ) AS follow_originals_only_eligible,
  multiIf(
    (coalesce(pb.orig_pol, 0) / greatest(coalesce(pb.orig_posts, 0), 1)) > 0.45,
    'Topic feed only',
    coalesce(pb.orig_posts, 0) >= 1
      AND (coalesce(pb.orig_pol, 0) / greatest(coalesce(pb.orig_posts, 0), 1)) <= 0.18
      AND (
        (coalesce(rb.repost_pol, 0) / greatest(coalesce(rb.reposts, 0), 1)) >= 0.30
        OR (coalesce(pb.quote_pol, 0) / greatest(coalesce(pb.quote_posts, 0), 1)) >= 0.30
        OR (coalesce(rb.reposts, 0) + coalesce(pb.quote_posts, 0)) >= greatest(coalesce(pb.orig_posts, 0) * 2, 4)
      ),
    'Follow originals only',
    'Follow all'
  ) AS recommend_mode,
  coalesce(p.role, 'creator') AS role,
  toUInt8(least(coalesce(pb.active_days, 1), 14)) AS active_days_14d,
  toUInt8(coalesce(pl.no_unauth, 0)) AS has_no_unauth_label,
  toUInt8(coalesce(v.hide_algo, 0)) AS hide_from_algorithmic,
  now() AS updated_at
FROM kf k
LEFT JOIN pi p ON k.did = p.did
LEFT JOIN ident id ON k.did = id.did
LEFT JOIN prof_labels pl ON k.did = pl.did
LEFT JOIN vis v ON k.did = v.did
LEFT JOIN post_breakdown pb ON k.did = pb.did
LEFT JOIN repost_breakdown rb ON k.did = rb.did;

-- 4. Refresh Item -> Communities (t=0 KnownFor prior + 8h half-life time-decayed engager InterestedIn)
TRUNCATE TABLE catbird.item_communities;

INSERT INTO catbird.item_communities (item_uri, did, rkey, community, score, created_at, updated_at)
WITH
  active_posts AS (
    SELECT did, rkey, max(rev) AS rev, any(created_at) AS post_created_at
    FROM catbird.posts
    WHERE is_deleted = 0
      AND created_at >= now() - INTERVAL 72 HOUR
      AND reply_parent = ''
      AND did NOT IN (SELECT did FROM catbird.visibility WHERE is_deleted = 0 AND hide_from_algorithmic = 1)
      AND did NOT IN (SELECT did FROM catbird.known_for FINAL WHERE version = '$VERSION' AND hide_from_recs = 1)
      AND did NOT IN (SELECT did FROM catbird.account_status WHERE active = 0)
    GROUP BY did, rkey
    HAVING (did, rkey) NOT IN (
      SELECT did, rkey FROM catbird.deletions WHERE collection = 'app.bsky.feed.post'
    )
  ),
  author_prior AS (
    SELECT
      p.did AS did,
      p.rkey AS rkey,
      p.post_created_at AS created_at,
      k.community AS community,
      3.0 * k.score AS w
    FROM active_posts p
    INNER JOIN (
      SELECT did, community, score FROM catbird.known_for FINAL WHERE version = '$VERSION'
    ) k ON p.did = k.did
  ),
  engager_votes AS (
    SELECT
      e.subject_did AS did,
      e.subject_rkey AS rkey,
      any(p.post_created_at) AS created_at,
      ii.community AS community,
      sum(
        multiIf(e.kind = 'repost', 2.0, 1.0)
        * ii.score
        * exp(-greatest(dateDiff('minute', p.post_created_at, e.indexed_at), 0) / 60.0 * 0.086643)
      ) AS w
    FROM catbird.engagements e
    INNER JOIN active_posts p ON e.subject_did = p.did AND e.subject_rkey = p.rkey
    INNER JOIN (
      SELECT did, community, score FROM catbird.interested_in WHERE rank < 10
    ) ii ON e.did = ii.did
    WHERE e.is_deleted = 0
    GROUP BY did, rkey, community
  ),
  combined AS (
    SELECT did, rkey, any(created_at) AS post_created_at, community, sum(w) AS total_w
    FROM (
      SELECT * FROM author_prior
      UNION ALL
      SELECT * FROM engager_votes
    )
    GROUP BY did, rkey, community
  ),
  ranked AS (
    SELECT
      did,
      rkey,
      post_created_at AS created_at,
      community,
      total_w,
      sum(total_w) OVER (PARTITION BY did, rkey) AS norm_w,
      row_number() OVER (PARTITION BY did, rkey ORDER BY total_w DESC) AS rnk
    FROM combined
  )
SELECT
  concat('at://', did, '/app.bsky.feed.post/', rkey) AS item_uri,
  did,
  rkey,
  community,
  toFloat32(total_w / greatest(norm_w, 1e-6)) AS score,
  created_at,
  now() AS updated_at
FROM ranked
WHERE rnk <= 3;

-- 5. Refresh Community -> Items (top 200 per community by community-local engagement residual)
TRUNCATE TABLE catbird.community_items;

INSERT INTO catbird.community_items
  (community, rank, item_uri, did, rkey, created_at, engagers, expected_engagements, residual_score, community_score, politicalness, heat, has_no_unauth_label, text_preview, updated_at)
WITH
  eng_per_post AS (
    SELECT
      subject_did AS did,
      subject_rkey AS rkey,
      toUInt32(uniqExact(did)) AS engagers,
      sum(multiIf(kind = 'repost', 2.0, 1.0)) AS weighted_eng
    FROM catbird.engagements
    WHERE is_deleted = 0 AND subject_collection = 'app.bsky.feed.post'
    GROUP BY did, rkey
  ),
  prof_no_unauth AS (
    SELECT did, max(has(self_labels, '!no-unauthenticated')) AS p_no_unauth
    FROM catbird.profiles
    WHERE is_deleted = 0
    GROUP BY did
  ),
  post_meta AS (
    SELECT
      did,
      rkey,
      any(text) AS post_text,
      max(politicalness) AS post_pol,
      max(heat) AS post_heat,
      max(has(self_labels, '!no-unauthenticated')) AS post_no_unauth
    FROM catbird.posts
    WHERE is_deleted = 0 AND created_at >= now() - INTERVAL 72 HOUR AND reply_parent = ''
    GROUP BY did, rkey
  ),
  author_size AS (
    SELECT did, max(followers_est) AS followers_est
    FROM catbird.producer_influence
    GROUP BY did
  ),
  candidates AS (
    SELECT
      ic.community AS community,
      ic.item_uri AS item_uri,
      ic.did AS did,
      ic.rkey AS rkey,
      ic.created_at AS created_at,
      ic.score AS community_score,
      coalesce(ep.engagers, 0) AS engagers,
      coalesce(ep.weighted_eng, 0.0) AS weighted_eng,
      log1p(coalesce(ep.weighted_eng, 0.0)) AS obs_log_eng,
      toUInt8(least(floor(log10(greatest(coalesce(asz.followers_est, 50), 10))), 6)) AS size_bucket,
      toUInt8(least(floor(greatest(dateDiff('minute', ic.created_at, now64(3)), 0) / 120.0), 36)) AS age_bucket,
      toHour(ic.created_at) AS hour_bucket,
      substringUTF8(coalesce(pm.post_text, ''), 1, 240) AS text_preview,
      toFloat32(greatest(
        coalesce(pm.post_pol, 0.0),
        multiIf(match(lowerUTF8(coalesce(pm.post_text, '')), '$POL_REGEX'), 0.78, 0.05)
      )) AS politicalness,
      toFloat32(greatest(
        coalesce(pm.post_heat, 0.0),
        multiIf(match(lowerUTF8(coalesce(pm.post_text, '')), '$HEAT_REGEX'), 0.75, 0.06)
      )) AS heat,
      toUInt8(greatest(coalesce(pm.post_no_unauth, 0), coalesce(pnu.p_no_unauth, 0))) AS has_no_unauth_label
    FROM catbird.item_communities ic
    INNER JOIN post_meta pm ON ic.did = pm.did AND ic.rkey = pm.rkey
    LEFT JOIN eng_per_post ep ON ic.did = ep.did AND ic.rkey = ep.rkey
    LEFT JOIN author_size asz ON ic.did = asz.did
    LEFT JOIN prof_no_unauth pnu ON ic.did = pnu.did
  ),
  baselines AS (
    SELECT
      community,
      size_bucket,
      age_bucket,
      avg(obs_log_eng) AS expected_log_eng
    FROM candidates
    GROUP BY community, size_bucket, age_bucket
  ),
  scored AS (
    SELECT
      c.community AS community,
      c.item_uri AS item_uri,
      c.did AS did,
      c.rkey AS rkey,
      c.created_at AS created_at,
      c.engagers AS engagers,
      toFloat32(exp(coalesce(b.expected_log_eng, 0.5)) - 1.0) AS expected_engagements,
      toFloat32(
        (c.obs_log_eng - coalesce(b.expected_log_eng, 0.5) + 0.5 * log1p(c.engagers))
        * c.community_score
        * exp(-greatest(dateDiff('minute', c.created_at, now64(3)), 0) / 60.0 / 18.0)
      ) AS residual_score,
      c.community_score AS community_score,
      c.politicalness AS politicalness,
      c.heat AS heat,
      c.has_no_unauth_label AS has_no_unauth_label,
      c.text_preview AS text_preview,
      row_number() OVER (PARTITION BY c.community, c.did ORDER BY c.obs_log_eng DESC) AS author_rnk
    FROM candidates c
    LEFT JOIN baselines b
      ON c.community = b.community AND c.size_bucket = b.size_bucket AND c.age_bucket = b.age_bucket
  ),
  dedup_author AS (
    SELECT
      *,
      row_number() OVER (PARTITION BY community ORDER BY residual_score DESC, created_at DESC) AS comm_rnk
    FROM scored
    WHERE author_rnk <= 2
  )
SELECT
  community,
  toUInt16(comm_rnk) AS rank,
  item_uri,
  did,
  rkey,
  created_at,
  engagers,
  expected_engagements,
  residual_score,
  community_score,
  politicalness,
  heat,
  has_no_unauth_label,
  text_preview,
  now() AS updated_at
FROM dedup_author
WHERE comm_rnk <= 200;

-- 6. Refresh Community Browse Directory (accounts by role + feeds + starter packs, with hub/floor/ceiling filters)
TRUNCATE TABLE catbird.community_directory;

INSERT INTO catbird.community_directory
  (community, kind, role, rank, uri_or_did, handle_or_name, description, influence, adoption_count, hub_concentration, original_share, political_share, has_no_unauth_label, updated_at)
WITH
  prof AS (
    SELECT
      did,
      argMax(display_name, rev) AS display_name,
      argMax(description, rev) AS description,
      max(has(self_labels, '!no-unauthenticated')) AS no_unauth
    FROM catbird.profiles
    WHERE is_deleted = 0
    GROUP BY did
  ),
  ident AS (
    SELECT did, argMax(handle, seq) AS handle
    FROM catbird.identities
    GROUP BY did
  ),
  kf_conc AS (
    SELECT did, max(score) AS max_comm_score, count() AS comm_count
    FROM catbird.known_for FINAL
    WHERE version = '$VERSION'
    GROUP BY did
  ),
  acct_candidates AS (
    SELECT
      pi.community AS community,
      pi.role AS role,
      pi.did AS uri_or_did,
      coalesce(nullIf(id.handle, ''), nullIf(pr.display_name, ''), pi.did) AS handle_or_name,
      substringUTF8(coalesce(pr.description, ''), 1, 240) AS description,
      pi.influence AS influence,
      pi.followers_est + pi.engagers AS adoption_count,
      coalesce(kc.max_comm_score, 0.5) AS hub_concentration,
      pi.original_share AS original_share,
      pi.political_share AS political_share,
      toUInt8(coalesce(pr.no_unauth, 0)) AS has_no_unauth_label,
      row_number() OVER (PARTITION BY pi.community, pi.role ORDER BY pi.influence DESC) AS rnk
    FROM catbird.producer_influence pi
    LEFT JOIN prof pr ON pi.did = pr.did
    LEFT JOIN ident id ON pi.did = id.did
    LEFT JOIN kf_conc kc ON pi.did = kc.did
    WHERE pi.followers_est >= 20
      AND pi.followers_est <= 250000
      AND coalesce(kc.max_comm_score, 0.5) >= 0.25
      AND pi.did NOT IN (SELECT did FROM catbird.visibility WHERE is_deleted = 0 AND hide_from_algorithmic = 1)
      AND pi.did NOT IN (SELECT did FROM catbird.known_for FINAL WHERE version = '$VERSION' AND hide_from_recs = 1)
      AND pi.did NOT IN (SELECT did FROM catbird.account_status WHERE active = 0)
  )
SELECT
  community,
  'account' AS kind,
  role,
  toUInt16(rnk) AS rank,
  uri_or_did,
  handle_or_name,
  description,
  influence,
  adoption_count,
  hub_concentration,
  original_share,
  political_share,
  has_no_unauth_label,
  now() AS updated_at
FROM acct_candidates
WHERE rnk <= 25;

-- 6b. Add custom feeds & starter packs per community (created by KnownFor producers or matching community keywords)
INSERT INTO catbird.community_directory
  (community, kind, role, rank, uri_or_did, handle_or_name, description, influence, adoption_count, hub_concentration, original_share, political_share, has_no_unauth_label, updated_at)
WITH
  kf AS (
    SELECT did, community, score
    FROM catbird.known_for FINAL
    WHERE version = '$VERSION'
  ),
  feed_cands AS (
    SELECT
      k.community AS community,
      concat('at://', f.did, '/app.bsky.feed.generator/', f.rkey) AS uri_or_did,
      any(f.display_name) AS handle_or_name,
      substringUTF8(any(f.description), 1, 240) AS description,
      max(k.score) * 5.0 AS influence,
      row_number() OVER (PARTITION BY k.community ORDER BY max(k.score) DESC, uri_or_did ASC) AS rnk
    FROM catbird.feedgens f
    INNER JOIN kf k ON f.did = k.did
    WHERE f.is_deleted = 0 AND f.display_name != ''
    GROUP BY k.community, uri_or_did
  )
SELECT
  community,
  'feed' AS kind,
  'feed' AS role,
  toUInt16(rnk) AS rank,
  uri_or_did,
  handle_or_name,
  description,
  toFloat32(influence) AS influence,
  toUInt32(10) AS adoption_count,
  toFloat32(0.8) AS hub_concentration,
  toFloat32(1.0) AS original_share,
  toFloat32(0.05) AS political_share,
  toUInt8(0) AS has_no_unauth_label,
  now() AS updated_at
FROM feed_cands
WHERE rnk <= 10;

INSERT INTO catbird.community_directory
  (community, kind, role, rank, uri_or_did, handle_or_name, description, influence, adoption_count, hub_concentration, original_share, political_share, has_no_unauth_label, updated_at)
WITH
  kf AS (
    SELECT did, community, score
    FROM catbird.known_for FINAL
    WHERE version = '$VERSION'
  ),
  pack_cands AS (
    SELECT
      k.community AS community,
      concat('at://', s.did, '/app.bsky.graph.starterpack/', s.rkey) AS uri_or_did,
      any(s.name) AS handle_or_name,
      substringUTF8(any(s.description), 1, 240) AS description,
      sum(k.score) AS influence,
      count() AS member_overlap,
      row_number() OVER (PARTITION BY k.community ORDER BY sum(k.score) DESC, uri_or_did ASC) AS rnk
    FROM catbird.starterpacks s
    LEFT JOIN catbird.listitems li ON s.list_uri = li.list_uri AND li.is_deleted = 0
    INNER JOIN kf k ON (li.subject = k.did OR s.did = k.did)
    WHERE s.is_deleted = 0 AND s.name != ''
    GROUP BY k.community, uri_or_did
  )
SELECT
  community,
  'pack' AS kind,
  'pack' AS role,
  toUInt16(rnk) AS rank,
  uri_or_did,
  handle_or_name,
  description,
  toFloat32(influence) AS influence,
  toUInt32(greatest(member_overlap, 3)) AS adoption_count,
  toFloat32(0.85) AS hub_concentration,
  toFloat32(1.0) AS original_share,
  toFloat32(0.05) AS political_share,
  toUInt8(0) AS has_no_unauth_label,
  now() AS updated_at
FROM pack_cands
WHERE rnk <= 10;
SQL
