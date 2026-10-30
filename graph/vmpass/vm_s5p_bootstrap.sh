#!/usr/bin/env bash
# Self-deleting GCE Spot startup script for R-S5p (cap $0.25, max-run 30m).
set -uo pipefail
export HOME="${HOME:-/root}"
exec > >(tee -a /var/log/s5p.log) 2>&1

md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
attr() { md "attributes/$1"; }
RUN_ID="$(attr vmpass-run-id)"
S9_RUN_ID="$(attr vmpass-s9-run-id)"
PLAN30="$(attr vmpass-plan30)"
PLAN60="$(attr vmpass-plan60)"
CH_VERSION="$(attr vmpass-ch-version)"
NAME="$(md name)"
ZONE="$(md zone | awk -F/ '{print $NF}')"
T0=$(date +%s)
log() { echo "[$(date -u +%FT%TZ)] $*"; }
upload_log() { gcloud storage cp --quiet /var/log/s5p.log "$PLAN30/intermediates/$RUN_ID/_logs/$NAME.log" >/dev/null 2>&1 || true; }

finish() {
  local rc=$1
  FINISHED=1
  log "s5p exit code $rc"
  mkdir -p /mnt/work
  cat > /mnt/work/vm_receipt.json <<EOF
{"run_id": "$RUN_ID", "step": "s5p", "vm": "$NAME", "zone": "$ZONE", "provisioning": "SPOT",
 "exit_code": $rc, "wall_seconds": $(( $(date +%s) - T0 )),
 "clickhouse": "$(clickhouse local --version 2>/dev/null)"}
EOF
  gcloud storage cp --quiet /mnt/work/vm_receipt.json "$PLAN30/outputs/$RUN_ID/vm-$NAME.json" || true
  upload_log
  log "deleting this VM ($NAME)"
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || poweroff
}
trap 'rc=$?; [[ -n "${FINISHED:-}" ]] || finish "$(( rc == 0 ? 99 : rc ))"' EXIT

log "starting R-S5p $RUN_ID on $NAME ($ZONE)"
mkdir -p /mnt/work && cd /mnt/work || exit 1
apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null
if ! curl -fsSL "https://packages.clickhouse.com/tgz/stable/clickhouse-common-static-${CH_VERSION}-amd64.tgz" -o ch.tgz; then
  curl -fsSL https://clickhouse.com/ | sh && mv clickhouse /usr/local/bin/clickhouse
else
  tar -xzf ch.tgz && install -m 0755 clickhouse-common-static-*/usr/bin/clickhouse /usr/local/bin/clickhouse
fi
clickhouse local --version
gcloud storage cp --quiet "$PLAN30/inputs/$RUN_ID/s5p_job.py" /mnt/work/s5p_job.py
gcloud storage cp --quiet "$PLAN30/inputs/$RUN_ID/purge_dids.txt" /mnt/work/purge_dids.txt
python3 /mnt/work/s5p_job.py --work /mnt/work --run-id "$RUN_ID" --s9-run-id "$S9_RUN_ID" --plan30 "$PLAN30" --plan60 "$PLAN60"
finish $?
