#!/usr/bin/env bash
# GCE startup script for the image-CID sidecar pass (attached by launch_cid.sh; runs
# as root on a fresh Debian 12 Spot VM). Steps:
#   1. RAID-0 the local NVMe SSDs at /mnt/ssd.
#   2. Install Node (pinned tarball, sha256-checked), Python + pyarrow + orjson and,
#      for the fold, ClickHouse (the version vmpass pins); unpack the staged code.
#   3. cid-segments=1: decode the GCS half (cidpass.py segments); every finished
#      segment is already in plan-30d with a ledger marker, so a preempted run
#      resumes on a fresh VM.
#   4. Fold, if every segment of the full range is in the ledger; else log and stop.
#   5. Write the VM receipt, copy the log, and delete this VM.
# The log goes to plan-30d/intermediates/<run>/_logs/ every 5 minutes. The VM also
# deletes itself at --max-run-duration (termination action DELETE).
set -uo pipefail
export HOME="${HOME:-/root}"   # GCE runs startup scripts as root without HOME; gcloud needs it
exec > >(tee -a /var/log/cidpass.log) 2>&1

md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
attr() { md "attributes/$1"; }
RUN_ID="$(attr cid-run-id)"; DO_SEGS="$(attr cid-segments)"; SRC="$(attr cid-src)"
RANGE="$(attr cid-range)"; FULL="$(attr cid-full-range)"; WORKERS="$(attr cid-workers)"
PLAN30="$(attr cid-plan30)"; PLAN60="$(attr cid-plan60)"; CUSTOM_TIME="$(attr cid-custom-time)"
CH_VERSION="$(attr cid-ch-version)"; NODE_VERSION="$(attr cid-node-version)"; NODE_SHA="$(attr cid-node-sha256)"
NAME="$(md name)"; ZONE="$(md zone | awk -F/ '{print $NF}')"
T0=$(date +%s); SEG_RC=-1; FOLD_RC=-1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
upload_log() { gcloud storage cp --quiet /var/log/cidpass.log "$PLAN30/intermediates/$RUN_ID/_logs/$NAME.log" >/dev/null 2>&1 || true; }
finish() {
  local rc=$1
  FINISHED=1
  log "cidpass exit code $rc (segments $SEG_RC, fold $FOLD_RC)"
  mkdir -p /mnt/ssd
  cat > /mnt/ssd/vm_receipt.json <<EOF
{"run_id": "$RUN_ID", "vm": "$NAME", "zone": "$ZONE", "provisioning": "SPOT", "exit_code": $rc,
 "segments_rc": $SEG_RC, "fold_rc": $FOLD_RC, "segments_seconds": ${SEG_S:-0}, "fold_seconds": ${FOLD_S:-0},
 "wall_seconds": $(( $(date +%s) - T0 )), "node": "$(node --version 2>/dev/null)", "decoder_sha256": "$(head -c 400 /mnt/ssd/code/decoder.sha256 2>/dev/null | tr '\n' ' ')"}
EOF
  gcloud storage cp --quiet /mnt/ssd/vm_receipt.json "$PLAN30/outputs/$RUN_ID/receipt/vm-$NAME.json" || true
  upload_log
  log "deleting this VM ($NAME); Active instances for this run should now be 0"
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || poweroff
}
# Any exit that skips finish (an unset variable, a failed command) still uploads the
# log and deletes this VM, so a broken bootstrap never sits idle until max-run.
trap 'rc=$?; [[ -n "${FINISHED:-}" ]] || finish "$(( rc == 0 ? 99 : rc ))"' EXIT
log "cidpass $RUN_ID on $NAME ($ZONE): segments=$DO_SEGS range=$RANGE workers=$WORKERS"

# 1. local SSD RAID-0
apt-get update -qq && apt-get install -y -qq mdadm python3-venv python3-pip curl xz-utils >/dev/null
DEVS=$(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | sort)
# shellcheck disable=SC2086  # $DEVS is a list of device paths
mdadm --create /dev/md0 --level=0 --raid-devices="$(echo "$DEVS" | wc -l)" $DEVS --force --run
mkfs.ext4 -F -q -E lazy_itable_init=1 /dev/md0 && mkdir -p /mnt/ssd && mount -o discard,defaults,noatime /dev/md0 /mnt/ssd
df -h /mnt/ssd
upload_log
( while sleep 300; do upload_log; done ) &

# 2. tools and code
cd /mnt/ssd || exit 1
curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o node.tar.xz
echo "$NODE_SHA  node.tar.xz" | sha256sum -c - || { log "node tarball sha256 mismatch"; finish 3; exit 0; }
tar -xJf node.tar.xz && export PATH="/mnt/ssd/node-v${NODE_VERSION}-linux-x64/bin:$PATH"
node --version
gcloud storage cp --quiet "$PLAN30/inputs/$RUN_ID/cidpass-code.tar.gz" code.tar.gz && mkdir -p code && tar -xzf code.tar.gz -C code
python3 -m venv venv && venv/bin/pip install -q pyarrow orjson
if ! curl -fsSL "https://packages.clickhouse.com/tgz/stable/clickhouse-common-static-${CH_VERSION}-amd64.tgz" -o ch.tgz; then
  log "pinned ClickHouse $CH_VERSION not found; using the official installer"
  curl -fsSL https://clickhouse.com/ | sh && mv clickhouse /usr/local/bin/clickhouse
else
  tar -xzf ch.tgz && install -m 0755 clickhouse-common-static-*/usr/bin/clickhouse /usr/local/bin/clickhouse
fi
clickhouse local --version
PASS=(/mnt/ssd/venv/bin/python /mnt/ssd/code/graph/cidpass/cidpass.py)
# the decoder runs on a tiny synthetic segment first, so a broken toolchain stops here, not 3,722 segments in
"${PASS[@]}" status --store30 "$PLAN30" --run-id "$RUN_ID" --expect-range "$FULL"
mkdir -p /mnt/ssd/smoke && node /mnt/ssd/code/graph/cidpass/make_segments.mjs --decoder-dir /mnt/ssd/code/decoder \
  --bench 2000 --index 1 --out /mnt/ssd/smoke/archive >/dev/null \
  && "${PASS[@]}" segments --source local --src /mnt/ssd/smoke/archive --store30 /mnt/ssd/smoke/store --run-id smoke \
     --work /mnt/ssd/smoke/work --decoder /mnt/ssd/code/decoder/decode_segments.mjs --workers 1 \
  || { log "decoder smoke run failed"; finish 3; exit 0; }
upload_log

# 3. the GCS half
if [[ "$DO_SEGS" == "1" ]]; then
  S0=$(date +%s)
  "${PASS[@]}" segments --source gcs --src "$SRC" --min-index "${RANGE%-*}" --max-index "${RANGE#*-}" \
    --listing-cache /mnt/ssd/listing.json --store30 "$PLAN30" --run-id "$RUN_ID" --work /mnt/ssd/work \
    --decoder /mnt/ssd/code/decoder/decode_segments.mjs --workers "$WORKERS" --progress-seconds 180
  SEG_RC=$?
  SEG_S=$(( $(date +%s) - S0 ))
  upload_log
  [[ $SEG_RC -eq 0 ]] || { finish "$SEG_RC"; exit 0; }
fi

# 4. the fold (refuses with exit 4 while any segment of the full range is missing)
F0=$(date +%s)
"${PASS[@]}" fold --store30 "$PLAN30" --store60 "$PLAN60" --custom-time "$CUSTOM_TIME" --run-id "$RUN_ID" \
  --work /mnt/ssd/fold --expect-range "$FULL" --clickhouse /usr/local/bin/clickhouse
FOLD_RC=$?
FOLD_S=$(( $(date +%s) - F0 ))
[[ $FOLD_RC -eq 4 ]] && log "fold deferred: the Drive half is not finished; run launch_cid.sh fold $RUN_ID later"
finish "$(( FOLD_RC == 4 ? 0 : FOLD_RC ))"
