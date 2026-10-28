#!/usr/bin/env bash
# S7g/S12m label gate chain (Mac, free, read-only to Bluesky): wait for the bluesky-v2 replay to release its
# writer lock, run the C4 catch-up tail to the live head (label_tail.sh), then build the D9 gate bundle with
# gate_bundle.py. Never uploads: the upload command is printed into the log for the coordinator.
#
#   label_chain.sh RUN_ID RECORD_DIDS_FILE        # usually started with nohup; see STATUS below
#
# STATUS ($OUT/status.txt) gets one line per stage: WAITING, TAIL_STARTED <UTC>, TAIL_DONE, BUNDLE_DONE, or
# FAILED <stage>. The bundle's watermark is the time the tail probed the live head (taken just before it runs),
# so labels issued after that are not covered; the VM refuses a bundle older than 36 h.
set -uo pipefail
RUN_ID="${1:?run id}"; RECORD_DIDS="${2:?record dids file}"
LAKE="$HOME/catbird-lake-local"
V2="$LAKE/moderation-labels/bluesky-v2"
LPY="$LAKE/label-backfill-venv/bin/python"
SPY="$LAKE/s4-venv/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
TAIL="${TAIL_SCRIPT:-$HERE/../../../c4-images/graph/c4images/label_tail.sh}"
OUT="$LAKE/s12mm-label-chain-$RUN_ID"
mkdir -p "$OUT"
st() { echo "$1 $(date -u +%FT%TZ)" | tee -a "$OUT/status.txt"; }
locked() {
  "$LPY" - "$V2/writer.lock" <<'EOF'
import fcntl, sys
with open(sys.argv[1], "a") as f:
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(0)      # held: the replay is still running
sys.exit(1)              # free (released immediately when this process exits)
EOF
}
[[ -x "$TAIL" ]] || [[ -f "$TAIL" ]] || { st "FAILED no_tail_script"; exit 1; }
st "WAITING"
while locked; do sleep 60; done
WM="$(date -u +%FT%TZ)"
st "TAIL_STARTED"
if ! RECORD_DIDS="$RECORD_DIDS" bash "$TAIL" > "$OUT/tail.log" 2>&1; then st "FAILED tail"; exit 1; fi
st "TAIL_DONE"
LABELS_TSV="$(grep '^resolved labels: ' "$OUT/tail.log" | tail -1 | sed 's/^resolved labels: //')"
[[ -f "$LABELS_TSV" ]] || { st "FAILED no_labels_tsv"; exit 1; }
if ! "$SPY" "$HERE/gate_bundle.py" "$OUT/gate" --labels "$LABELS_TSV" --watermark "$WM" --upload "$RUN_ID" \
     > "$OUT/bundle.log" 2>&1; then st "FAILED bundle"; exit 1; fi
st "BUNDLE_DONE watermark=$WM labels=$LABELS_TSV"
echo "# upload (coordinator): $(grep '^+ ' "$OUT/bundle.log" | sed 's/^+ //')" | tee -a "$OUT/status.txt"
