#!/usr/bin/env bash
# GCE startup script for the us-west1 pass (attached by launch_vm.sh; runs as root
# on a fresh Debian 12 Spot VM). Steps:
#   1. RAID-0 the 8 local NVMe SSDs at /mnt/ssd (3,000 GiB).
#   2. Install ClickHouse (pinned version; the S8 cross-check compares cityHash64
#      with the Mac), Python + pyarrow, and Rust; build simgraph from source
#      (the Mac binary is arm64 macOS) and run it once on a tiny synthetic input.
#   3. Stage all 119,640 extract objects to local SSD once (one GET per object).
#   4. Run passrun.py (S8 smoke or S9 main pass); every finished step is already
#      in the plan-30d/plan-60d buckets, so a preempted run resumes on a fresh VM.
#   5. Write the VM receipt, copy the log, and delete this VM.
# The VM also deletes itself at --max-run-duration (termination action DELETE).
set -uo pipefail
# GCE runs startup scripts as root without HOME; rustup, cargo and gcloud need it.
export HOME="${HOME:-/root}"
exec > >(tee -a /var/log/vmpass.log) 2>&1

md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
attr() { md "attributes/$1"; }
RUN_ID="$(attr vmpass-run-id)"; MODE="$(attr vmpass-mode)"
PLAN30="$(attr vmpass-plan30)"; PLAN60="$(attr vmpass-plan60)"; EXTRACT="$(attr vmpass-extract)"
CH_VERSION="$(attr vmpass-ch-version)"; RULES_SHA="$(attr vmpass-rules-sha256 || true)"
NAME="$(md name)"; ZONE="$(md zone | awk -F/ '{print $NF}')"
T0=$(date +%s)
log() { echo "[$(date -u +%FT%TZ)] $*"; }
upload_log() { gcloud storage cp --quiet /var/log/vmpass.log "$PLAN30/intermediates/$RUN_ID/_logs/$NAME.log" >/dev/null 2>&1 || true; }
finish() {
  local rc=$1
  FINISHED=1
  log "pass exit code $rc"
  cat > /mnt/ssd/vm_receipt.json <<EOF
{"run_id": "$RUN_ID", "mode": "$MODE", "vm": "$NAME", "zone": "$ZONE", "provisioning": "SPOT",
 "exit_code": $rc, "stage_seconds": ${STAGE_S:-0}, "build_seconds": ${BUILD_S:-0}, "wall_seconds": $(( $(date +%s) - T0 )),
 "clickhouse": "$(clickhouse local --version 2>/dev/null)", "rustc": "$(~/.cargo/bin/rustc --version 2>/dev/null)",
 "simgraph_tiny_run": $(cat /mnt/ssd/simgraph_tiny.json 2>/dev/null || echo null), "staged_objects": ${STAGED:-0}}
EOF
  gcloud storage cp --quiet /mnt/ssd/vm_receipt.json "$PLAN30/outputs/$RUN_ID/receipt/vm-$NAME.json" || true
  upload_log
  log "deleting this VM ($NAME); Active instances for this run should now be 0"
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || poweroff
}

# Any exit that skips finish (an unset variable, a failed command) still uploads the
# log and deletes this VM, so a broken bootstrap never sits idle until max-run.
trap 'rc=$?; [[ -n "${FINISHED:-}" ]] || finish "$(( rc == 0 ? 99 : rc ))"' EXIT
log "vmpass $MODE $RUN_ID on $NAME ($ZONE)"
# 1. local SSD RAID-0
apt-get update -qq && apt-get install -y -qq mdadm python3-venv python3-pip build-essential pkg-config curl >/dev/null
DEVS=$(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | sort)
# shellcheck disable=SC2086  # $DEVS is a list of device paths
mdadm --create /dev/md0 --level=0 --raid-devices="$(echo "$DEVS" | wc -l)" $DEVS --force --run
mkfs.ext4 -F -q -E lazy_itable_init=1 /dev/md0 && mkdir -p /mnt/ssd && mount -o discard,defaults,noatime /dev/md0 /mnt/ssd
df -h /mnt/ssd
upload_log

# 2. tools
B0=$(date +%s)
cd /mnt/ssd || exit 1
if ! curl -fsSL "https://packages.clickhouse.com/tgz/stable/clickhouse-common-static-${CH_VERSION}-amd64.tgz" -o ch.tgz; then
  log "pinned ClickHouse $CH_VERSION not found; using the official installer (the cityHash64 cross-check guards parity)"
  curl -fsSL https://clickhouse.com/ | sh && mv clickhouse /usr/local/bin/clickhouse
else
  tar -xzf ch.tgz && install -m 0755 clickhouse-common-static-*/usr/bin/clickhouse /usr/local/bin/clickhouse
fi
clickhouse local --version
gcloud storage cp --quiet "$PLAN30/inputs/$RUN_ID/code.tar.gz" /mnt/ssd/code.tar.gz && mkdir -p /mnt/ssd/code && tar -xzf /mnt/ssd/code.tar.gz -C /mnt/ssd/code
python3 -m venv /mnt/ssd/venv && /mnt/ssd/venv/bin/pip install -q pyarrow
curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
# shellcheck source=/dev/null
source "$HOME/.cargo/env"
(cd /mnt/ssd/code/simgraph && cargo build --release) || { log "simgraph build failed"; finish 3; exit 0; }
SIMGRAPH=/mnt/ssd/code/simgraph/target/release/simgraph
# tiny-input run: 3 producers, 4 users
mkdir -p /mnt/ssd/tiny && printf '0\t-\t3\n1\t-\t3\n2\t-\t2\n' > /mnt/ssd/tiny/producers.tsv
python3 - <<'EOF'
import struct
edges = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2), (3, 1)]
open('/mnt/ssd/tiny/edges.bin', 'wb').write(b''.join(struct.pack('<II', u, p) for u, p in edges))
EOF
"$SIMGRAPH" --dir /mnt/ssd/tiny --min-support 1 > /mnt/ssd/simgraph_tiny.json || { log "simgraph tiny run failed"; finish 3; exit 0; }
log "simgraph tiny run: $(cat /mnt/ssd/simgraph_tiny.json)"
BUILD_S=$(( $(date +%s) - B0 ))
upload_log

# 3. stage the extract once (one GET per object) and the manifest
S0=$(date +%s)
mkdir -p /mnt/ssd/stage
gcloud storage cp --quiet "$EXTRACT/manifest.tsv" /mnt/ssd/stage/manifest.tsv
BUCKET_SHARDS="$(attr vmpass-bucket-shards || true)"; BUCKET_SHARDS="${BUCKET_SHARDS:-0}"
if [[ "$BUCKET_SHARDS" -gt 0 ]]; then
  # S8 on a bucketing subset: stage only the K evenly spaced directories passrun will bucket
  mapfile -t SHARD_DIRS < <(/mnt/ssd/venv/bin/python /mnt/ssd/code/vmpass/passrun.py shards --manifest /mnt/ssd/stage/manifest.tsv --k "$BUCKET_SHARDS")
  log "staging a $BUCKET_SHARDS-directory subset: ${SHARD_DIRS[*]}"
  for d in "${SHARD_DIRS[@]}"; do
    gcloud storage cp --recursive --quiet "$EXTRACT/$d" /mnt/ssd/stage/ || { log "staging $d failed"; finish 4; exit 0; }
  done
else
  gcloud storage cp --recursive --quiet "$EXTRACT/v2-*" /mnt/ssd/stage/ || { log "staging failed"; finish 4; exit 0; }
fi
STAGED=$(find /mnt/ssd/stage -name '*.parquet' | wc -l)
STAGE_S=$(( $(date +%s) - S0 ))
log "staged $STAGED objects in ${STAGE_S}s"
upload_log

# 4. the pass (resumes from the ledger if this RUN_ID already has finished steps)
# rules: the uploaded file (sha256 checked against the launcher's metadata), else the
# committed S1 translation; S9 also takes the O3b scales S8 measured (S1 R4)
RULES_ARG=(--rules /mnt/ssd/code/vmpass/rules_s1.json)
if gcloud storage cp --quiet "$PLAN30/inputs/$RUN_ID/rules.json" /mnt/ssd/rules.json 2>/dev/null; then
  RULES_ARG=(--rules /mnt/ssd/rules.json --rules-sha256 "$RULES_SHA")
fi
if gcloud storage cp --quiet "$PLAN30/inputs/$RUN_ID/o3b_scales.json" /mnt/ssd/o3b_scales.json 2>/dev/null; then
  RULES_ARG+=(--o3b-scales /mnt/ssd/o3b_scales.json)
elif [[ "$MODE" == "s9" ]]; then
  log "S9 without S8's o3b_scales.json: O3b uses the rules' default scales (S1 R4 expects S8's)"
fi
( while sleep 180; do upload_log; done ) &   # passrun also streams work/progress.log to _logs/progress.log
SHARD_ARG=()
[[ "$BUCKET_SHARDS" -gt 0 ]] && SHARD_ARG=(--bucket-shards "$BUCKET_SHARDS")
/mnt/ssd/venv/bin/python /mnt/ssd/code/vmpass/passrun.py run --mode "$MODE" --stage /mnt/ssd/stage --work /mnt/ssd/work \
  --store30 "$PLAN30" --store60 "$PLAN60" --run-id "$RUN_ID" --inputs "$PLAN30/inputs/$RUN_ID" \
  --manifest /mnt/ssd/stage/manifest.tsv --simgraph "$SIMGRAPH" --stage-seconds "$STAGE_S" "${RULES_ARG[@]}" "${SHARD_ARG[@]}"
finish $?
