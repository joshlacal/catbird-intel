#!/usr/bin/env bash
# GCE startup script for S12m (attached by launch_s12mm.sh; runs as root on a fresh Spot g2-standard-8 with
# the Deep Learning VM image common-cu129-ubuntu-2404-nvidia-580). Steps:
#   1. check the L4 and driver (nvidia-smi), turn swap off and require none (image bytes must never be paged out);
#   2. a venv with the pinned S4/S7 stack; if CUDA is unusable with the pinned torch, retry torch from the
#      PyTorch cu129 index; record what was installed;
#   3. EG2 weights from plan-30d/inputs/models/, every file checked against the Mac's SHA256SUMS;
#   4. code + (bench inputs | gate bundle) from plan-30d/inputs/<RUN_ID>/;
#   5. job.py bench | run (resumes from the ledger); 6. receipt + log upload, then delete this VM.
# --max-run-duration (termination action DELETE) is the hard stop; the EXIT trap deletes the VM on any early exit.
set -uo pipefail
export HOME="${HOME:-/root}"
exec > >(tee -a /var/log/s12mm.log) 2>&1

md() { curl -sf -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
attr() { md "attributes/$1"; }
RUN_ID="$(attr s12mm-run-id)"; MODE="$(attr s12mm-mode)"
PLAN30="$(attr s12mm-plan30)"; PLAN60="$(attr s12mm-plan60)"
CAP="$(attr s12mm-cap-usd)"; RATE="$(attr s12mm-usd-per-hour)"; RPS="$(attr s12mm-rps)"
ALLOW_ABOVE="$(attr s12mm-allow-above || echo 0)"; SOFT="$(attr s12mm-soft || echo 70)"
POSTS="$(attr s12mm-posts || true)"; SIDECAR="$(attr s12mm-sidecar || true)"
NAME="$(md name)"; ZONE="$(md zone | awk -F/ '{print $NF}')"
T0=$(date +%s)
W=/opt/s12mm
OUT="$PLAN60/vectors/s12mm/$RUN_ID"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
upload_log() { gcloud storage cp --quiet /var/log/s12mm.log "$PLAN30/intermediates/s12mm-$RUN_ID/_logs/$NAME.log" >/dev/null 2>&1 || true; }
finish() {
  local rc=$1
  FINISHED=1
  log "job exit code $rc"
  cat > /tmp/vm_receipt.json <<EOF
{"run_id": "$RUN_ID", "mode": "$MODE", "vm": "$NAME", "zone": "$ZONE", "provisioning": "SPOT", "exit_code": $rc,
 "wall_seconds": $(( $(date +%s) - T0 )), "usd_per_hour": $RATE, "cap_usd": $CAP, "rps": $RPS,
 "gpu": "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null)",
 "torch_install": "$(cat $W/torch_install.txt 2>/dev/null)"}
EOF
  gcloud storage cp --quiet /tmp/vm_receipt.json "$OUT/receipt/vmboot-$NAME.json" || true
  upload_log
  log "deleting this VM ($NAME)"
  gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet || poweroff
}
trap 'rc=$?; [[ -n "${FINISHED:-}" ]] || finish "$(( rc == 0 ? 99 : rc ))"' EXIT
log "s12mm $MODE $RUN_ID on $NAME ($ZONE), rps $RPS, soft $SOFT, cap \$$CAP at \$$RATE/h"

# 1. GPU, driver, swap
nvidia-smi || { log "no GPU/driver"; finish 2; exit 0; }
swapoff -a || true
if [[ $(wc -l < /proc/swaps) -gt 1 ]]; then log "swap still active; refusing (image bytes must not reach disk)"; finish 2; exit 0; fi
mkdir -p $W && cd $W || exit 1
upload_log

# 2. Python stack (pinned as in S4/S7; recorded)
python3 -m venv $W/venv
PIP=$W/venv/bin/pip; PY=$W/venv/bin/python
$PIP install -q --upgrade pip
$PIP install -q "torch==2.14.1" "torchvision==0.29.1" "transformers==5.21.0" "sentence-transformers==6.2.0" \
  "pillow" "pyarrow" "numpy" "pandas" || { log "pip install failed"; finish 3; exit 0; }
if ! $PY -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  log "pinned torch cannot use CUDA here; retrying torch 2.14.1 from the PyTorch cu129 index"
  $PIP install -q --force-reinstall "torch==2.14.1" "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu129 \
    || { log "cu129 torch install failed"; finish 3; exit 0; }
fi
$PY -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))' | tee $W/torch_install.txt
$PY -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || { log "CUDA unusable"; finish 3; exit 0; }

# 3. weights (checked)
mkdir -p $W/models
gcloud storage cp --recursive --quiet "$PLAN30/inputs/models/embeddinggemma-2" $W/models/ || { finish 4; exit 0; }
gcloud storage cp --quiet "$PLAN30/inputs/models/embeddinggemma-2.SHA256SUMS" $W/models/SHA256SUMS || { finish 4; exit 0; }
(cd $W/models && sha256sum -c --quiet SHA256SUMS) || { log "model sha256 mismatch"; finish 4; exit 0; }

# 4. code and inputs
gcloud storage cp --quiet "$PLAN30/inputs/$RUN_ID/code.tar.gz" $W/code.tar.gz && mkdir -p $W/code && tar -xzf $W/code.tar.gz -C $W/code \
  || { finish 4; exit 0; }
ABOVE=(); [[ "$ALLOW_ABOVE" == "1" ]] && ABOVE=(--allow-above-etiquette)
COMMON=(--model-dir "$W/models/embeddinggemma-2" --work "$W/work" --run-id "$RUN_ID" --out "$OUT" --rps "$RPS" "${ABOVE[@]}"
        --cap-usd "$CAP" --usd-per-hour "$RATE" --vm-start-epoch "$T0")
( while sleep 600; do upload_log; done ) &

# 5. the job
if [[ "$MODE" == "bench" ]]; then
  mkdir -p $W/bench
  gcloud storage cp --recursive --quiet "$PLAN30/inputs/$RUN_ID/bench/*" $W/bench/ || { finish 4; exit 0; }
  GATE=(); if gcloud storage cp --recursive --quiet "$PLAN30/inputs/$RUN_ID/gate" $W/ ; then GATE=(--gate-bundle "$W/gate"); fi
  PROD=(); [[ -f $W/bench/bench_producers.txt ]] && PROD=(--bench-producers "$W/bench/bench_producers.txt")
  SIDE=(); [[ -n "$SIDECAR" ]] && SIDE=(--sidecar "$SIDECAR")
  $PY $W/code/s12mm/job.py bench "${COMMON[@]}" "${GATE[@]}" "${PROD[@]}" "${SIDE[@]}" --image-posts "$POSTS" \
    --bench-text $W/bench/s7_slab.parquet --bench-random $W/bench/s7_random.parquet --parity-ref $W/bench/parity_mac.npz \
    --mm-parity-ref $W/bench/mm_parity_mac.npz --soft-tokens-list 70,280
  finish $?
else
  gcloud storage cp --recursive --quiet "$PLAN30/inputs/$RUN_ID/gate" $W/ || { log "no gate bundle uploaded"; finish 4; exit 0; }
  SIDE=(); [[ -n "$SIDECAR" ]] && SIDE=(--sidecar "$SIDECAR")
  $PY $W/code/s12mm/job.py run "${COMMON[@]}" --gate-bundle $W/gate --posts "$POSTS" "${SIDE[@]}" --soft-tokens "$SOFT"
  finish $?
fi
