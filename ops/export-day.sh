#!/usr/bin/env bash
# Daily sealed export of the catbird hot store to Parquet (zstd) + manifest.
#
# Exports every row *indexed* on one UTC day (raw rows, tombstones included,
# no FINAL), so the cold lake is an event log that can be replayed. The
# deletions ledger for the same day is exported alongside; downstream readers
# anti-join it. Upload is done by `upload_cmd` when a Drive remote is set.
#
# Usage: export-day.sh [YYYY-MM-DD]   (default: yesterday UTC)
set -euo pipefail

DAY="${1:-$(date -u -d 'yesterday' +%F)}"
NEXT="$(date -u -d "$DAY + 1 day" +%F)"
ROOT="${EXPORT_ROOT:-/var/lib/catbird-intel/export}"
OUT="$ROOT/$DAY"
DB="${CH_DATABASE:-catbird}"
RCLONE_REMOTE="${RCLONE_REMOTE:-}"          # e.g. catbird-drive:CatbirdCommunity/catbird-lake/hot-export
KEEP_DAYS="${KEEP_DAYS:-7}"
SCHEMA_VERSION=1

TABLES=(posts engagements follows blocks lists listitems starterpacks feedgens profiles visibility account_status identities other_records deletions)

mkdir -p "$OUT.partial"
rm -f "$OUT.partial"/*

for t in "${TABLES[@]}"; do
  f="$OUT.partial/${t}.parquet"
  clickhouse-client --query "
    SELECT * FROM ${DB}.${t}
    WHERE indexed_at >= toDateTime64('${DAY} 00:00:00', 3, 'UTC')
      AND indexed_at <  toDateTime64('${NEXT} 00:00:00', 3, 'UTC')
    ORDER BY seq
    INTO OUTFILE '${f}'
    FORMAT Parquet
    SETTINGS output_format_parquet_compression_method = 'zstd',
             output_format_parquet_row_group_size = 1000000,
             max_threads = 2, priority = 10"
done

# Manifest: rows, bytes, sha256 and seq range per file.
{
  echo "{"
  echo "  \"schema\": \"catbird.hot-export.v${SCHEMA_VERSION}\","
  echo "  \"day\": \"${DAY}\","
  echo "  \"exported_at\": \"$(date -u +%FT%TZ)\","
  echo "  \"files\": ["
  first=1
  for t in "${TABLES[@]}"; do
    f="$OUT.partial/${t}.parquet"
    stats="$(clickhouse-local --query "SELECT count(), min(seq), max(seq) FROM file('${f}', Parquet) FORMAT TSV")"
    read -r rows minseq maxseq <<<"$stats"
    bytes="$(stat -c %s "$f")"
    sha="$(sha256sum "$f" | cut -d' ' -f1)"
    [ $first -eq 1 ] || echo ","
    first=0
    printf '    {"table": "%s", "file": "%s.parquet", "rows": %s, "bytes": %s, "sha256": "%s", "min_seq": %s, "max_seq": %s}' \
      "$t" "$t" "$rows" "$bytes" "$sha" "${minseq:-0}" "${maxseq:-0}"
  done
  echo
  echo "  ]"
  echo "}"
} > "$OUT.partial/manifest.json"

rm -rf "$OUT"
mv "$OUT.partial" "$OUT"
echo "sealed $OUT ($(du -sh "$OUT" | cut -f1))"

if [ -n "$RCLONE_REMOTE" ]; then
  rclone copy "$OUT" "$RCLONE_REMOTE/day=$DAY" \
    --drive-chunk-size 256M --transfers 2 --checkers 2 --tpslimit 2 --drive-stop-on-upload-limit
  rclone check "$OUT" "$RCLONE_REMOTE/day=$DAY" --one-way --size-only
  touch "$OUT/.uploaded"
  echo "uploaded to $RCLONE_REMOTE/day=$DAY"
fi

# Retain local staging: uploaded days for KEEP_DAYS; without a remote, keep
# 30 days (about 60 GB) so nothing is lost before Drive upload is configured.
RETAIN="$KEEP_DAYS"
[ -n "$RCLONE_REMOTE" ] || RETAIN="${KEEP_DAYS_NO_REMOTE:-30}"
find "$ROOT" -mindepth 1 -maxdepth 1 -type d -name '20*' -mtime +"$RETAIN" | while read -r d; do
  if [ -z "$RCLONE_REMOTE" ] || [ -e "$d/.uploaded" ]; then rm -rf -- "$d"; fi
done
