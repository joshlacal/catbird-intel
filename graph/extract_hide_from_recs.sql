-- Extract active accounts with hideFromAlgorithmicRecommendations == true
-- (`app.bsky.actor.contentVisibilityDeclaration`), honoring snapshot/sync floors
-- and record updates/deletes by folding the latest record per DID ordered by
-- (rev, is_delete) across all archive event shards + netcup live visibility
-- and deletions parquets, excluding inactive accounts in account_coverage.tsv.
--
-- Reproduces `/Users/joshlacalamito/catbird-lake-local/graph-build-v3/hide_from_recs-20261025/hide_from_recs.tsv`
-- (55,031 DIDs, SHA-256 b37d3a4ed9a90cb49c6498641d0dfb80564e250f165027bd9737b9b90e548dbe).
--
-- Usage:
--   clickhouse local --queries-file graph/extract_hide_from_recs.sql \
--     > /Users/joshlacalamito/catbird-lake-local/graph-build-v3/hide_from_recs-20261025/hide_from_recs.tsv

CREATE OR REPLACE FUNCTION tid_u64 AS (s) -> if(
    match(s, '^[234567abcdefghij][234567abcdefghijklmnopqrstuvwxyz]{12}$'),
    toUInt64(arraySum(i -> bitShiftLeft(toUInt64(position('234567abcdefghijklmnopqrstuvwxyz', substring(s, i, 1)) - 1), 5 * (13 - i)),
                      range(1, 14))),
    toUInt64(0));
CREATE OR REPLACE FUNCTION time_rev AS (t) -> bitShiftLeft(toUInt64(greatest(toUnixTimestamp64Micro(t), 0)), 10);

WITH
  acct AS (
    SELECT did, floor_rev, inactive
    FROM file(
      '/Users/joshlacalamito/catbird-lake-local/graph-build-v3/out-full-20261024/account_coverage.tsv',
      TSVWithNames,
      'did String, snapshot_source LowCardinality(String), coverage LowCardinality(String), floor_rev UInt64, inactive UInt8'
    )
  ),
  vis_events AS (
    SELECT
      did,
      rkey,
      if(tid_u64(rev) > 0, tid_u64(rev),
         time_rev(ifNull(parseDateTime64BestEffortOrNull(time, 6, 'UTC'), toDateTime64(0, 6, 'UTC')))) AS rev,
      toUInt8(operation = 'delete') AS is_delete,
      toUInt8(extra = '1' AND operation != 'delete') AS hide
    FROM file(
      '/Users/joshlacalamito/catbird-lake-local/{archive-graph/events,gce-archive-graph/events,gce-archive-graph-gcs/a/events,gce-archive-graph-gcs/b/events,gce-archive-graph-gcs/c/events,gce-archive-graph-gcs-6414/d/events,gce-archive-graph-gcs-6414/e/events,gce-archive-graph-gcs-6414/f/events,supplement-events/run1,supplement-events/retry}/events-*.parquet',
      Parquet,
      'kind String, did String, collection String, rkey String, rev String, operation String, extra String, time String'
    )
    WHERE collection = 'app.bsky.actor.contentVisibilityDeclaration'
      AND kind IN ('commit', '')
      AND did != ''
      AND rkey != ''
    UNION ALL
    SELECT
      did,
      'self' AS rkey,
      if(rev > 0, rev, time_rev(indexed_at)) AS rev,
      is_deleted AS is_delete,
      toUInt8(hide_from_algorithmic = 1 AND is_deleted = 0) AS hide
    FROM file(
      '/Users/joshlacalamito/catbird-lake-local/netcup-export/*/visibility.parquet',
      Parquet,
      'did String, rev UInt64, is_deleted UInt8, hide_from_algorithmic UInt8, indexed_at DateTime64(3, \'UTC\')'
    )
    UNION ALL
    SELECT
      did,
      rkey,
      if(rev > 0, rev, time_rev(indexed_at)) AS rev,
      toUInt8(1) AS is_delete,
      toUInt8(0) AS hide
    FROM file(
      '/Users/joshlacalamito/catbird-lake-local/netcup-export/*/deletions.parquet',
      Parquet,
      'collection String, did String, rkey String, rev UInt64, indexed_at DateTime64(3, \'UTC\')'
    )
    WHERE collection = 'app.bsky.actor.contentVisibilityDeclaration'
  ),
  folded AS (
    SELECT
      v.did AS did,
      argMax(v.is_delete, (v.rev, v.is_delete)) AS latest_delete,
      argMax(v.hide, (v.rev, v.is_delete)) AS latest_hide,
      max(coalesce(a.inactive, 0)) AS is_inactive
    FROM vis_events v
    LEFT JOIN acct a ON v.did = a.did
    WHERE v.rev >= coalesce(a.floor_rev, 0)
    GROUP BY v.did
  )
SELECT did
FROM folded
WHERE latest_delete = 0 AND latest_hide = 1 AND is_inactive = 0
ORDER BY did
FORMAT TSVRaw;
