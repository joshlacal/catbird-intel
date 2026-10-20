#!/usr/bin/env bash
# Phase 0 "Done when" checks for the catbird-intel ingest. Read-only.
#   1. Continuous ingest: health intervals with no events or lag > 60 s.
#   2. Deletes: no record visible under FINAL whose delete is in the ledger;
#      ledger freshness sampled for 60 s (NOT delete visibility latency).
#   3. Disk: 14-day steady-state projection of the catbird database.
# Usage (on netcup): verify-phase0.sh [SINCE_UTC, e.g. '2026-10-19 13:20:00']
set -euo pipefail
SINCE="${1:-2026-10-19 13:20:00}"
q() { clickhouse-client -q "$1"; }

echo "== 1. continuity since $SINCE UTC"
q "SELECT count() AS health_rows,
          min(ts) AS first, max(ts) AS last,
          round(dateDiff('second', min(ts), max(ts)) / 3600, 2) AS hours,
          countIf(events = 0) AS zero_event_intervals,
          countIf(lag_ms > 60000) AS lag_over_60s_intervals,
          max(lag_ms) AS max_lag_ms,
          quantile(0.99)(lag_ms) AS p99_lag_ms,
          sum(reconnects) AS reconnects,
          max(rss_mb) AS max_rss_mb
   FROM catbird.ingest_health WHERE ts >= toDateTime64('$SINCE', 3, 'UTC') FORMAT Vertical"
# Gaps between consecutive health rows (service down) longer than 60 s.
q "SELECT count() AS health_gaps_over_60s, max(gap) AS longest_gap_s FROM (
     SELECT dateDiff('second', lagInFrame(ts, 1, ts) OVER (ORDER BY ts ROWS BETWEEN 1 PRECEDING AND CURRENT ROW), ts) AS gap
     FROM catbird.ingest_health WHERE ts >= toDateTime64('$SINCE', 3, 'UTC')) WHERE gap > 60 FORMAT Vertical"

echo "== 2a. delete invariant (should all be 0)"
for spec in "engagements:app.bsky.feed.like','app.bsky.feed.repost" "posts:app.bsky.feed.post" "follows:app.bsky.graph.follow" "blocks:app.bsky.graph.block" "listitems:app.bsky.graph.listitem"; do
  t=${spec%%:*}; c=${spec#*:}
  n=$(q "SELECT count() FROM (SELECT did, rkey, rev FROM catbird.$t FINAL WHERE is_deleted = 0) p INNER JOIN (SELECT did, rkey, max(rev) AS del_rev FROM catbird.deletions WHERE collection IN ('$c') GROUP BY did, rkey) d USING (did, rkey) WHERE p.rev <= d.del_rev")
  echo "$t visible_but_deleted=$n"
done
q "SELECT count() AS ledger_deletes FROM catbird.deletions FORMAT TSV"

echo "== 2b. ledger freshness only (age of newest ledger row, sampled 60x at 1 s)"
for _ in $(seq 60); do
  q "SELECT dateDiff('millisecond', max(indexed_at), now64(3)) FROM catbird.deletions"
  sleep 1
done | sort -n | awk '{a[NR]=$1} END {printf "ledger_head_age_p50=%dms p95=%dms max=%dms (freshness only; idle delete traffic increases age)\n", a[int(NR*0.5)], a[int(NR*0.95)], a[NR]}'

echo "delete_visibility_latency_gate=UNPERFORMED: requires keyed durable-ingest commit and first-hidden serving receipts; ledger age is not latency"
echo "controlled_create_delete_probe=UNPERFORMED: requires explicit test-account write authorization; this script is read-only"

echo "== 3. disk"
q "SELECT table, sum(rows) AS row_count, formatReadableSize(sum(bytes_on_disk)) AS size,
          round(sum(bytes_on_disk) / greatest(sum(rows), 1), 1) AS b_per_row
   FROM system.parts WHERE active AND database = 'catbird' GROUP BY table ORDER BY sum(bytes_on_disk) DESC FORMAT PrettyCompactMonoBlock"
q "WITH (SELECT sum(bytes_on_disk) FROM system.parts WHERE active AND database = 'catbird' AND table IN ('posts','engagements','other_records')) AS hot,
        (SELECT sum(bytes_on_disk) FROM system.parts WHERE active AND database = 'catbird' AND table NOT IN ('posts','engagements','other_records')) AS state,
        (SELECT dateDiff('second', min(ts), max(ts)) FROM catbird.ingest_health) AS secs
   SELECT formatReadableSize(hot / secs * 86400 * 14) AS hot_14d_projection,
          formatReadableSize(state / secs * 86400 * 14) AS state_growth_14d,
          formatReadableSize((hot + state) / secs * 86400 * 14) AS total_14d_projection FORMAT Vertical"
