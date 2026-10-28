#!/usr/bin/env bash
# Catch-up tail of Bluesky moderation labels (seq 44,494,621 = 2026-10-20 13:13Z -> live head) for the
# C4 image gate, then the resolved label file the gate reads. Run it just before any image fetch and
# only after the bluesky-v2 replay has stopped: both would hold a subscribeLabels connection to the same
# labeler. The script refuses to start while bluesky-v2's writer lock is held. Not run by the agent.
#
#   graph/c4images/label_tail.sh            # probe head, tail, verify, resolve
#
# Output: ~/catbird-lake-local/moderation-labels/recent-tail-<UTC date>/ (packed store, same collector)
#         ~/catbird-lake-local/c4-prep-20261028/labels/labels_active_<UTC date>.tsv (labels.py)
set -euo pipefail

LAKE="$HOME/catbird-lake-local"
PY="$LAKE/label-backfill-venv/bin/python"        # stdlib + cbor2 + websockets, as the running replay
HERE="$(cd "$(dirname "$0")" && pwd)"
COLLECTOR="$HERE/../../ops/backfill_moderation_labels.py"
LABELS="$LAKE/moderation-labels"
V2="$LABELS/bluesky-v2"
START=44494621                                     # recent-tail-20261020 reached this target
DAY="$(date -u +%Y%m%d)"
DIR="$LABELS/recent-tail-$DAY"
RECORD_DIDS="${RECORD_DIDS:-}"                     # optional: candidate authors for record-level labels

# 1. Refuse while the bluesky-v2 replay (or anything else) holds its writer lock.
"$PY" - "$V2/writer.lock" <<'EOF' || { echo "bluesky-v2 writer lock is held: the replay is still running" >&2; exit 1; }
import fcntl, sys
with open(sys.argv[1], "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
EOF

# 2. Live head: connect without a cursor and take the first #labels frame's seq (one connection).
HEAD="$("$PY" - "$V2/identity.json" <<'EOF'
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
)"
echo "label head seq $HEAD (start $START)"
[ "$HEAD" -gt "$START" ]

# 3. The tail itself: a new packed generation from START to HEAD, same bounds style as the v2 replay.
"$PY" -u "$COLLECTOR" --directory "$DIR" --start-cursor "$START" --target-seq "$HEAD" \
  --max-frames 20000000 --max-seconds 14400 --frames-per-second 1000 --batch-frames 5000 \
  --max-total-bytes $((3 * 1024 * 1024 * 1024)) --min-free-bytes $((110 * 1024 * 1024 * 1024))

# 4. Independent verification of every committed chunk (collector's own verify path).
"$PY" -u "$COLLECTOR" --directory "$DIR" --verify-only

# 5. Resolve labels in force now over every store (account-level for all DIDs; record-level for
#    RECORD_DIDS if given, else all, which needs several GB of RAM).
OUT="$LAKE/c4-prep-20261028/labels/labels_active_$DAY.tsv"
"$PY" -I "$HERE/labels.py" "$OUT" "$LABELS/bluesky-v1" "$V2" "$LABELS/recent-tail-20261020" "$DIR" \
  ${RECORD_DIDS:+--record-dids "$RECORD_DIDS"}
echo "resolved labels: $OUT"
