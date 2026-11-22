#!/usr/bin/env bash
# Nightly rebuild of the onboarding account-signals file from the daily export.
#
# The export keeps about a week of days on this machine, and the signals builder needs 14 days of
# posts plus a few weeks of follows. So this job keeps its own input folder: for every sealed export
# day it hard-links the three tables the builder reads (no extra disk while the export still holds
# the day), and prunes them on its own, longer schedule.
#
# The running API re-reads the signals file when its modification time changes, so replacing the
# file is the whole deploy. The previous file is kept beside it as *.prev.json.
#
# Usage: build-onboarding-signals-nightly.sh [YYYY-MM-DD]   (default: yesterday UTC)
set -euo pipefail

DAY="${1:-$(date -u -d 'yesterday' +%F)}"
EXPORT_ROOT="${EXPORT_ROOT:-/var/lib/catbird-intel/export}"
INPUT_ROOT="${SIGNALS_INPUT_ROOT:-/var/lib/catbird-intel/signals-input}"
SERVING_DIR="${SIGNALS_SERVING_DIR:-/var/lib/catbird-intel/onboarding-20261122}"
POOL="${SIGNALS_POOL:-$SERVING_DIR/intel_onboarding_pool_v1.json}"
OUT="${SIGNALS_OUT:-$SERVING_DIR/intel_onboarding_signals_v1.json}"
PYTHON="${SIGNALS_PYTHON:-/home/josh/.venvs/intel-signals/bin/python}"
ACTIVITY_DAYS="${ACTIVITY_DAYS:-14}"
REMOVAL_DAYS="${REMOVAL_DAYS:-14}"
FOLLOW_WINDOW_DAYS="${FOLLOW_WINDOW_DAYS:-20}"
THREADS="${SIGNALS_THREADS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILDER="${SIGNALS_BUILDER:-$SCRIPT_DIR/../graph/s16_s17/build_onboarding_signals.py}"
TABLES=(posts follows account_status)

mkdir -p "$INPUT_ROOT"

# 1. Take in every sealed export day (a sealed day has a manifest).
for src in "$EXPORT_ROOT"/20*; do
  [ -d "$src" ] && [ -f "$src/manifest.json" ] || continue
  day="$(basename "$src")"
  mkdir -p "$INPUT_ROOT/$day"
  for t in "${TABLES[@]}"; do
    dst="$INPUT_ROOT/$day/$t.parquet"
    [ -e "$dst" ] && continue
    [ -f "$src/$t.parquet" ] || continue
    ln "$src/$t.parquet" "$dst" 2>/dev/null || cp --reflink=auto "$src/$t.parquet" "$dst.partial" && { [ -e "$dst" ] || mv "$dst.partial" "$dst"; }
  done
done

if [ ! -f "$INPUT_ROOT/$DAY/posts.parquet" ]; then
  echo "no sealed export for $DAY yet; signals file left as it is" >&2
  exit 1
fi

# 2. Build beside the live file, then swap it in.
FOLLOW_TO="$(date -u -d "$DAY - $REMOVAL_DAYS days" +%F)"
FOLLOW_FROM="$(date -u -d "$FOLLOW_TO - $((FOLLOW_WINDOW_DAYS - 1)) days" +%F)"
NEXT="$SERVING_DIR/.intel_onboarding_signals_v1.next.json"
REPORT="$SERVING_DIR/intel_onboarding_signals_v1.report.json"
rm -f "$NEXT"

"$PYTHON" "$BUILDER" \
  --export-root "$INPUT_ROOT" --pool "$POOL" --as-of "$DAY" \
  --activity-days "$ACTIVITY_DAYS" --removal-days "$REMOVAL_DAYS" \
  --follow-from "$FOLLOW_FROM" --follow-to "$FOLLOW_TO" \
  --threads "$THREADS" --out "$NEXT" --report "$REPORT" 2>"$SERVING_DIR/intel_onboarding_signals_v1.build.log"

# A file less than half the size of the one in use means something upstream is wrong: keep the old one.
if [ -f "$OUT" ]; then
  old="$(stat -c %s "$OUT")"
  new="$(stat -c %s "$NEXT")"
  if [ "$((new * 2))" -lt "$old" ]; then
    echo "new signals file is $new bytes against $old in use; not replaced" >&2
    exit 3
  fi
  cp -p "$OUT" "${OUT%.json}.prev.json"
fi
chmod 644 "$NEXT"
mv "$NEXT" "$OUT"
echo "signals for $DAY in place: $(stat -c %s "$OUT") bytes, follows $FOLLOW_FROM..$FOLLOW_TO"

# 3. Prune the input folder: posts are the bulk and are only needed for the activity window (plus
# one earlier day the builder uses as a sanity check); follows and status are small and kept for the
# follow window.
POSTS_KEEP=$((ACTIVITY_DAYS + 2))
DAYS_KEEP=$((REMOVAL_DAYS + FOLLOW_WINDOW_DAYS + 3))
posts_cut="$(date -u -d "$DAY - $POSTS_KEEP days" +%F)"
days_cut="$(date -u -d "$DAY - $DAYS_KEEP days" +%F)"
for d in "$INPUT_ROOT"/20*; do
  [ -d "$d" ] || continue
  day="$(basename "$d")"
  if [[ "$day" < "$days_cut" ]]; then
    rm -f "$d"/posts.parquet "$d"/follows.parquet "$d"/account_status.parquet
    rmdir "$d" 2>/dev/null || true
  elif [[ "$day" < "$posts_cut" ]]; then
    rm -f "$d"/posts.parquet
  fi
done
