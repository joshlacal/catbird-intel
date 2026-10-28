#!/bin/bash
# Read-only ClickHouse-local runner for whole-network plan step S2 (local prep).
#
#   chq.sh NAME SQL_FILE [--fold] [--memory BYTES] [--threads N]
#
# Writes, under $S2_OUT/logs/: NAME.sql (the exact statements run), NAME.out (stdout),
# NAME.err (stderr), NAME.time (/usr/bin/time -l, incl. peak RSS), NAME.disk (df samples).
#
# --fold attaches the fold DB tables from an APFS clone ($S2_OUT/chclone, made with
# `cp -c -R` from graph-build-v3/ch/store), never from graph-build-v3/ch itself. Merges
# are stopped before and after every ATTACH, so nothing is rewritten even in the clone.
# fold_follows.sql is never run. The process is capped at --memory (default 40 GB) via
# max_server_memory_usage; spill goes to ~/catbird-lake-local/_ch-spill/s2 with a 30 GB
# per-query cap (plan §7.3). A watchdog kills the query if /System/Volumes/Data drops
# below 100 GiB free (plan S0d gate).
set -euo pipefail

name=$1; sql=$2; shift 2
fold=0; mem=40000000000; threads=10
while [ $# -gt 0 ]; do
  case $1 in
    --fold) fold=1 ;;
    --memory) mem=$2; shift ;;
    --threads) threads=$2; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

OUT=${S2_OUT:-$HOME/catbird-lake-local/s2-prep-20261028}
CLONE=$OUT/chclone/
SPILL=$HOME/catbird-lake-local/_ch-spill/s2/
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/logs" "$SPILL" "$OUT/chtmp"
# clickhouse-local keeps its temporary database here and removes it on exit.
export TMPDIR=$OUT/chtmp/
log=$OUT/logs/$name

free_gib() { df -g /System/Volumes/Data | awk 'NR==2{print $4}'; }
if [ "$(free_gib)" -lt 100 ]; then echo "disk gate: $(free_gib) GiB free < 100 GiB" >&2; exit 3; fi

{
  if [ $fold = 1 ]; then sed "s|__CLONE__|$CLONE|g" "$HERE/attach_fold_ro.sql"; fi
  cat "$sql"
} > "$log.sql"

session_mem=$((mem * 9 / 10))
cmd=(clickhouse local
  --max_threads="$threads"
  --max_memory_usage="$session_mem"
  --max_bytes_before_external_group_by=$((mem / 4))
  --max_bytes_before_external_sort=$((mem / 4))
  --max_temporary_data_on_disk_size_for_query=30000000000
  --queries-file "$log.sql"
  --
  --max_server_memory_usage="$mem"
  --tmp_path="$SPILL"
  --custom_local_disks_base_directory="$CLONE")

printf '%q ' "${cmd[@]}" > "$log.cmd"; echo >> "$log.cmd"
echo "start $(date -u +%FT%TZ) free_gib=$(free_gib)" > "$log.disk"
start=$(date +%s)
( cd "$HOME/catbird-lake-local/netcup-export" && exec /usr/bin/time -l "${cmd[@]}" ) < /dev/null > "$log.out" 2> "$log.err" &
pid=$!
status=0
while kill -0 "$pid" 2>/dev/null; do
  f=$(free_gib)
  echo "$(date -u +%FT%TZ) free_gib=$f" >> "$log.disk"
  if [ "$f" -lt 100 ]; then
    echo "disk gate tripped: $f GiB free; killing $pid" | tee -a "$log.disk" >&2
    pkill -P "$pid" || true; kill "$pid" || true; status=4
  fi
  sleep 20
done
wait "$pid" || status=$?
end=$(date +%s)
echo "end $(date -u +%FT%TZ) free_gib=$(free_gib) wall_s=$((end - start)) status=$status" >> "$log.disk"
# /usr/bin/time writes its report to stderr after the child's own stderr.
grep -E 'real|maximum resident set size|peak memory footprint' "$log.err" > "$log.time" || true
exit $status
