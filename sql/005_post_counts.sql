-- Per-post reply and quote counts, kept current by a materialized view on
-- catbird.posts. Read with sum(): SELECT uri, sum(replies), sum(quotes)
-- FROM catbird.post_counts WHERE uri IN (...) GROUP BY uri.
-- Consumer: feed-proto `discover` lens (controversy discount).
--
-- ponytail: deleted replies/quotes are not subtracted and at-least-once
-- ingest can double-count a retried flush; fine for a ratio heuristic.
--
-- One-off backfill after first apply (not idempotent, so not in this file):
--   INSERT INTO catbird.post_counts
--   SELECT u, rkey_date, countIf(k = 1), countIf(k = 2)
--   FROM catbird.posts ARRAY JOIN [reply_parent, quote_uri] AS u, [1, 2] AS k
--   WHERE is_deleted = 0 AND u != '' AND indexed_at < '<MV creation time UTC>'
--   GROUP BY u, rkey_date;

CREATE TABLE IF NOT EXISTS catbird.post_counts
(
    uri     String CODEC(ZSTD(3)),
    day     Date,
    replies UInt64,
    quotes  UInt64
)
ENGINE = SummingMergeTree((replies, quotes))
ORDER BY (uri, day)
TTL day + INTERVAL 14 DAY;

CREATE MATERIALIZED VIEW IF NOT EXISTS catbird.post_counts_mv TO catbird.post_counts AS
SELECT u AS uri, rkey_date AS day, countIf(k = 1) AS replies, countIf(k = 2) AS quotes
FROM catbird.posts
ARRAY JOIN [reply_parent, quote_uri] AS u, [1, 2] AS k
WHERE is_deleted = 0 AND u != ''
GROUP BY uri, day;
