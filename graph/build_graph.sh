#!/usr/bin/env bash
# Build the dense-id follow graph from extracted archive events.
# Usage: build_graph.sh [MIN_FOLLOWERS] [MAX_FOLLOWS]
#
# SNAPSHOTS is repo_snapshot_export JSONL taken from a checkpoint copy of the
# collector's meta.pebble; NETCUP is the root of netcup daily exports
# (<day>/{follows,deletions,account_status}.parquet). Both are required: without
# snapshot revs the fold cannot tell which old follows a repo download replaced.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LAKE="${LAKE:-/Users/joshlacalamito/catbird-lake-local}"
EVENTS="${EVENTS:-$LAKE/archive-graph/events/*.parquet}"
SNAPSHOTS="${SNAPSHOTS:-$LAKE/repo-snapshots/repo_snapshots*.jsonl*}"
NETCUP="${NETCUP:-$LAKE/netcup-export}"
WORK="${WORK:-$LAKE/graph-build}"
OUT="${OUT:-$WORK/out}"
MIN_FOLLOWERS="${1:-100}"
MAX_FOLLOWS="${2:-20000}"
mkdir -p "$OUT" "$WORK/ch"
# Comment lines keep their placeholders: ClickHouse 26.9's client chokes on paths
# like ".../*.parquet" inside "--" comments ("Could not parse the number of retries").
sed -e "/^[[:space:]]*--/!{
s#__EVENTS__#${EVENTS}#g
s#__SNAPSHOTS__#${SNAPSHOTS}#g
s#__NETCUP__#${NETCUP}#g
s#__OUT__#${OUT}#g
s#__MIN_FOLLOWERS__#${MIN_FOLLOWERS}#g
s#__MAX_FOLLOWS__#${MAX_FOLLOWS}#g
}" "$HERE/fold_follows.sql" > "$WORK/fold.sql"
clickhouse local --path "$WORK/ch" --queries-file "$WORK/fold.sql" -- --user_files_path="$LAKE"
cat "$OUT/stats.tsv" "$OUT/coverage_stats.tsv"
