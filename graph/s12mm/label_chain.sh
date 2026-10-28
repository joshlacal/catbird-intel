#!/usr/bin/env bash
# S7g/S12m label gate chain (Mac, free, read-only to Bluesky). The Bluesky labeler replay runs as several
# parallel streams (bluesky-v2 plus bluesky-v3-s* seq ranges, each with its own writer.lock), so this:
#   1. waits until EVERY writer lock under moderation-labels/ is free: bluesky-v2, every bluesky-v3-s*, every
#      recent-tail-* (no collector may still be writing, and only one subscribeLabels connection is ours);
#   2. re-verifies every recent-tail-* store with the collector's --verify-only (a tail whose parent script was
#      stopped never ran its own verify step);
#   3. tails from the highest seq the union already reaches (>= 44,494,621, the replay target) to the live
#      head, into a new store recent-tail-<UTC minute>, then verifies it;
#   4. resolves labels over v1 + v2 + every v3-s* + every tail with graph/c4images/labels.py --label-root, which
#      dedupes frames by seq, checks each store's hash chain, refuses seq holes between stores and requires the
#      union to reach the probed head (--coverage-to);
#   5. builds the D9 gate bundle (gate_bundle.py). It never uploads; the upload command goes into status.txt.
#
#   label_chain.sh RUN_ID RECORD_DIDS_FILE        # usually started with nohup
#
# STATUS ($OUT/status.txt): WAITING, LOCKS_FREE, TAIL_STARTED, TAIL_DONE, RESOLVED, BUNDLE_DONE, or FAILED <stage>.
# The bundle watermark is the time the head was probed (labels issued later are not covered; the VM refuses a
# bundle older than 36 h).
set -uo pipefail
RUN_ID="${1:?run id}"; RECORD_DIDS="${2:?record dids file}"
LAKE="$HOME/catbird-lake-local"
LABELS="${LABELS_ROOT:-$LAKE/moderation-labels}"
LPY="$LAKE/label-backfill-venv/bin/python"
SPY="$LAKE/s4-venv/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
COLLECTOR="${COLLECTOR:-$HERE/../../ops/backfill_moderation_labels.py}"
RESOLVER="$HERE/../c4images/labels.py"
TARGET=44494621
POLL="${POLL_SECONDS:-60}"
OUT="${OUT_DIR:-$LAKE/s12mm-label-chain-$RUN_ID}"
mkdir -p "$OUT"
st() { echo "$1 $(date -u +%FT%TZ)" | tee -a "$OUT/status.txt"; }
fail() { st "FAILED $1"; exit 1; }

lock_held() {  # exit 0 if $1 is locked by another process
  "$LPY" - "$1" <<'EOF'
import fcntl, sys
with open(sys.argv[1], "a") as f:
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(0)
sys.exit(1)
EOF
}
stores_with_locks() { ls -d "$LABELS"/bluesky-v2 "$LABELS"/bluesky-v3-s* "$LABELS"/recent-tail-* 2>/dev/null; }
any_held() {
  local s held=""
  for s in $(stores_with_locks); do
    [[ -f "$s/writer.lock" ]] && lock_held "$s/writer.lock" && held="$held $(basename "$s")"
  done
  [[ -n "$held" ]] && { echo "$held"; return 0; }
  return 1
}
checkpoint_cursor() { "$LPY" -c "import json,sys; print(json.load(open(sys.argv[1]))['state']['cursor'])" "$1/checkpoint.json"; }

[[ -f "$COLLECTOR" && -f "$RESOLVER" ]] || fail "missing_collector_or_resolver"
[[ $(ls -d "$LABELS"/bluesky-v3-s* 2>/dev/null | wc -l) -ge 1 ]] || fail "no_v3_streams"
st "WAITING"
last=""
while held="$(any_held)"; do
  [[ "$held" != "$last" ]] && { echo "  locks held:$held $(date -u +%FT%TZ)" >> "$OUT/status.txt"; last="$held"; }
  sleep "$POLL"
done
st "LOCKS_FREE"

# 2. verify the existing tails
for t in "$LABELS"/recent-tail-*; do
  [[ -f "$t/manifest.json" ]] || continue
  "$LPY" -u "$COLLECTOR" --directory "$t" --verify-only >> "$OUT/verify.log" 2>&1 || fail "verify_$(basename "$t")"
done

# 3. tail from the union's reach to the live head
START=$TARGET
for t in "$LABELS"/recent-tail-*; do
  [[ -f "$t/checkpoint.json" ]] || continue
  c="$(checkpoint_cursor "$t")"; [[ "$c" -gt "$START" ]] && START="$c"
done
WM="$(date -u +%FT%TZ)"
HEAD="$("$LPY" - "$LABELS/bluesky-v2/identity.json" <<'EOF'
import io, json, sys
import cbor2
from websockets.sync.client import connect
endpoint = json.load(open(sys.argv[1]))["endpoint"].replace("https://", "wss://", 1)
with connect(endpoint + "/xrpc/com.atproto.label.subscribeLabels", open_timeout=15, max_size=8 << 20) as ws:
    for _ in range(1000):
        raw = ws.recv(timeout=300)
        s = io.BytesIO(raw)
        header, body = cbor2.load(s), cbor2.load(s)
        if header.get("t") == "#labels":
            print(body["seq"])
            break
EOF
)" || fail "head_probe"
[[ "$HEAD" =~ ^[0-9]+$ ]] || fail "head_probe_value"
st "TAIL_STARTED start=$START head=$HEAD watermark=$WM"
if [[ "$HEAD" -gt "$START" ]]; then
  DIR="$LABELS/recent-tail-$(date -u +%Y%m%dT%H%MZ)"
  "$LPY" -u "$COLLECTOR" --directory "$DIR" --start-cursor "$START" --target-seq "$HEAD" \
    --max-frames 20000000 --max-seconds 14400 --frames-per-second 1000 --batch-frames 5000 \
    --max-total-bytes $((3 * 1024 * 1024 * 1024)) --min-free-bytes $((110 * 1024 * 1024 * 1024)) \
    > "$OUT/tail.log" 2>&1 || fail "tail"
  "$LPY" -u "$COLLECTOR" --directory "$DIR" --verify-only >> "$OUT/verify.log" 2>&1 || fail "verify_new_tail"
fi
st "TAIL_DONE"

# 4. resolve over every store, with the union coverage check
LABELS_TSV="$OUT/labels_active.tsv"
"$LPY" -I "$RESOLVER" "$LABELS_TSV" --label-root "$LABELS" --coverage-to "$HEAD" --record-dids "$RECORD_DIDS" \
  > "$OUT/resolve.log" 2>&1 || fail "resolve"
st "RESOLVED $(tail -1 "$OUT/resolve.log" | cut -c1-400)"

# 5. gate bundle (print-only upload)
"$SPY" "$HERE/gate_bundle.py" "$OUT/gate" --labels "$LABELS_TSV" --watermark "$WM" --upload "$RUN_ID" \
  > "$OUT/bundle.log" 2>&1 || fail "bundle"
st "BUNDLE_DONE watermark=$WM labels=$LABELS_TSV"
echo "# upload (coordinator): $(grep '^+ ' "$OUT/bundle.log" | sed 's/^+ //')" | tee -a "$OUT/status.txt"
