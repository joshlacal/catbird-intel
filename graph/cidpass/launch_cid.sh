#!/usr/bin/env bash
# Launcher for the image-CID sidecar pass (R-CID).
#
# PRINT-ONLY BY DEFAULT. Nothing touches GCP or Drive unless --execute is given,
# and --execute is for Josh (or an agent he approved) after the R-CID ask:
#
#   launch_cid.sh drive-mac RUN_ID     # print the Mac command for segments 0-3670 (Drive; no GCP compute)
#   launch_cid.sh gcs RUN_ID           # print the Spot VM for segments 3671-7392 (Coldline) + the fold
#   launch_cid.sh fold RUN_ID          # print a small Spot VM that only folds (both halves already in the ledger)
#   launch_cid.sh status RUN_ID        # ledger coverage for RUN_ID (one listing call)
#   launch_cid.sh vms                  # list VMs carrying this plan's label
#   launch_cid.sh delete RUN_ID        # delete only the VM this launcher recorded for RUN_ID
#   ... --execute                      # actually run the printed commands
#
# Why two halves: segments 3671-7392 (3,722 objects, 1,012 GB, Coldline) are in the
# GCE capture's buffer bucket; segments 0-3670 are not in GCS at all and hold about
# half of all posts (the Oct 24 extract's v2-drive groups: 120 GB of 241.6 GB of
# post Parquet). Drive (listed 2026-10-28) holds all of 0-7392 (2,009 GB). Drive
# downloads are free and the Mac already has Drive access, so the Drive half runs on
# the Mac and no Drive credential goes onto a VM. MAC_MAX_INDEX=7392 makes the Mac
# do everything ($0, ~2x the wall time) and the gcs VM is then not needed.
#
# VM rules (as graph/vmpass/launch_vm.sh): Spot only (no capacity -> reschedule,
# never on-demand), us-west1, --max-run-duration with --instance-termination-action
# =DELETE, labels plan=whole-network-signals so the label-scoped budget sees it, a
# sessions file so only VMs this launcher made are ever deleted, and a bootstrap
# that uploads its log every 5 minutes and deletes the VM on any exit.
set -euo pipefail

MODE="${1:-}"; shift || true
RUN_ID="${1:-}"; [[ "${RUN_ID:-}" == --* ]] && RUN_ID="" || shift || true
EXECUTE=0
for a in "$@"; do [[ "$a" == "--execute" ]] && EXECUTE=1; done

PROJECT="${PROJECT:-catbird-436600}"
ZONES="${ZONES:-us-west1-b us-west1-a us-west1-c}"
MACHINE="${MACHINE:-n2-highcpu-64}"; N_SSD="${N_SSD:-4}"; WORKERS="${WORKERS:-60}"   # 4 SSDs: ~1.2 GB/s of segment + JSONL writes at full rate
FOLD_MACHINE="${FOLD_MACHINE:-n2-highmem-8}"
MAXRUN="${MAXRUN:-3h}"; FOLD_MAXRUN="${FOLD_MAXRUN:-2h}"
PLAN30="${PLAN30:-gs://catbird-436600-plan-30d}"
PLAN60="${PLAN60:-gs://catbird-436600-plan-60d}"
GCS_SRC="gs://catbird-436600-jetstream-buffer-20261020/sources/gce-selfhost-001/buffer-admission-20261020"
DRIVE_FOLDER="${DRIVE_FOLDER:-16Zd7n4pwTSDMDsnP9chX2Ci0r0ahEy45}"   # CatbirdCommunity/archives/gce-selfhost-001/segments
GCS_RANGE="3671-7392"; FULL_RANGE="0-7392"
LABEL_KEY="plan"; LABEL_VAL="whole-network-signals"
CH_VERSION="${CH_VERSION:-26.9.24.8}"
NODE_VERSION="22.23.1"
NODE_SHA256="9749e988f437343b7fa832c69ded82a312e41a03116d766797ac14f6f9eee578"   # node-v22.23.1-linux-x64.tar.xz (nodejs.org SHASUMS256)
SESSIONS="${SESSIONS:-$HOME/catbird-lake-local/vmpass-sessions.tsv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH="$(cd "$HERE/.." && pwd)"
PY="${PY:-$GRAPH/.venv/bin/python}"   # needs pyarrow (orjson optional); a jj workspace has no .venv, so pass PY=
MAC_WORK="${MAC_WORK:-$HOME/catbird-lake-local/cidpass-$RUN_ID}"
CUSTOM_TIME="${CUSTOM_TIME:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

run() {  # print, and run only with --execute
  printf '+ %s\n' "$*"
  if [[ $EXECUTE -eq 1 ]]; then eval "$*"; fi
}
need_run_id() { [[ -n "$RUN_ID" ]] || { echo "usage: $0 $MODE RUN_ID [--execute]" >&2; exit 2; }; }

stage_code() {  # code + the decoder with node_modules go up first (free ingress); the VM pulls them
  local dir="/tmp/cidpass-$RUN_ID" tarball="/tmp/cidpass-$RUN_ID.tar.gz"
  run "rm -rf '$dir' && mkdir -p '$dir/graph/vmpass' && bash '$HERE/setup_decoder.sh' '$dir/decoder' | tee '$dir/decoder.sha256'"
  run "cp -R '$HERE' '$dir/graph/cidpass' && cp '$GRAPH/post_extract.py' '$GRAPH/gcs_segments.py' '$GRAPH/archive_graph_extract.py' '$dir/graph/' && cp '$GRAPH/vmpass/cityhash.py' '$GRAPH/vmpass/store.py' '$dir/graph/vmpass/'"
  run "tar -C '$dir' --exclude='__pycache__' -czf '$tarball' graph decoder decoder.sha256"
  run "gcloud storage cp '$tarball' '$PLAN30/inputs/$RUN_ID/cidpass-code.tar.gz'"
}

create_vm() {  # $1 step, $2 machine, $3 n_ssd, $4 max-run, $5 do_segments (0/1)
  local step="$1" machine="$2" nssd="$3" maxrun="$4" segs="$5"
  local name; name="cidpass-${step}-$(echo "$RUN_ID" | tr 'A-Z_.' 'a-z--')"
  local ssd=""; for _ in $(seq "$nssd"); do ssd+="--local-ssd=interface=NVME "; done
  echo "# Spot capacity: try each zone; if none has capacity, stop and reschedule. Never on-demand."
  for z in $ZONES; do
    local cmd
    cmd="gcloud compute instances create '$name' --project='$PROJECT' --zone='$z' \
 --machine-type='$machine' $ssd\
 --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration='$maxrun' \
 --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=50GB --boot-disk-type=pd-balanced \
 --scopes=storage-rw,compute-rw --labels='$LABEL_KEY=$LABEL_VAL,step=cid-$step,run=$(echo "$RUN_ID" | tr 'A-Z_.' 'a-z--')' \
 --metadata-from-file=startup-script='$HERE/cid_bootstrap.sh' \
 --metadata='cid-run-id=$RUN_ID,cid-segments=$segs,cid-src=$GCS_SRC,cid-range=$GCS_RANGE,cid-full-range=$FULL_RANGE,cid-workers=$WORKERS,cid-plan30=$PLAN30,cid-plan60=$PLAN60,cid-custom-time=$CUSTOM_TIME,cid-ch-version=$CH_VERSION,cid-node-version=$NODE_VERSION,cid-node-sha256=$NODE_SHA256'"
    printf '+ %s\n' "$cmd"
    if [[ $EXECUTE -eq 1 ]]; then
      if eval "$cmd"; then
        mkdir -p "$(dirname "$SESSIONS")"
        printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$RUN_ID" "$name" "$z" >> "$SESSIONS"
        echo "# recorded $name in $SESSIONS (sessions rule: only recorded VMs are ever deleted)"
        return 0
      fi
      echo "# no Spot capacity in $z (or create failed); trying the next zone"
    else
      echo "# (dry run: the next zone is only tried if this create fails)"
      return 0
    fi
  done
  echo "# no zone had Spot capacity: reschedule; do NOT fall back to on-demand" >&2
  return 1
}

case "$MODE" in
  drive-mac)
    need_run_id
    echo "# Drive half on the Mac: segments 0-${MAC_MAX_INDEX:-3670} (~1 TB download from Drive, no fee; parts upload to plan-30d is free ingress)."
    echo "# Bound by Drive download speed (~35-100 MB/s): ~3-8 h. Resumable: rerun the same command after any stop."
    run "mkdir -p '$MAC_WORK' && bash '$HERE/setup_decoder.sh' '$MAC_WORK/decoder' | tee '$MAC_WORK/decoder.sha256'"
    run "caffeinate -i '$PY' '$HERE/cidpass.py' segments --source drive --src '$DRIVE_FOLDER' --min-index 0 --max-index ${MAC_MAX_INDEX:-3670} \
 --listing-cache '$MAC_WORK/drive-listing.json' --store30 '$PLAN30' --run-id '$RUN_ID' --work '$MAC_WORK/work' \
 --decoder '$MAC_WORK/decoder/decode_segments.mjs' --workers ${MAC_WORKERS:-8} --progress-seconds 180 2>&1 | tee -a '$MAC_WORK/drive-half.log'"
    ;;
  gcs)
    need_run_id
    echo "# GCS half: segments $GCS_RANGE (3,722 objects, 1,012.4 GB Coldline) on $MACHINE Spot, then the fold if the Drive half is already in the ledger."
    echo "# Cost cap: VM \$1.48/h (64 x \$0.01896 + 64 GiB x \$0.00254 + 1,500 GiB SSD + boot) x $MAXRUN = \$4.43"
    echo "#   + Coldline retrieval 1,012.4 GB x \$0.02 = \$20.25 (+<=\$0.33 re-reads of in-flight segments after a preemption) + ops ~\$0.75 = R-CID cap \$26"
    stage_code
    create_vm gcs "$MACHINE" "$N_SSD" "$MAXRUN" 1
    ;;
  fold)
    need_run_id
    echo "# Fold only: needs every segment $FULL_RANGE in the ledger (check: $0 status $RUN_ID). Reads parts from plan-30d (Standard, same region: no retrieval fee)."
    echo "# Cost cap: n2-highmem-8 Spot (8 x \$0.01896 + 64 GiB x \$0.00254 + 750 GiB SSD) = \$0.36/h x $FOLD_MAXRUN = \$0.72"
    stage_code
    create_vm fold "$FOLD_MACHINE" 2 "$FOLD_MAXRUN" 0
    ;;
  status)
    need_run_id
    run "'$PY' '$HERE/cidpass.py' status --store30 '$PLAN30' --run-id '$RUN_ID' --expect-range '$FULL_RANGE'"
    ;;
  vms)
    run "gcloud compute instances list --project='$PROJECT' --filter='labels.$LABEL_KEY=$LABEL_VAL' --format='table(name,zone.basename(),status,labels.step,labels.run)'"
    ;;
  delete)
    need_run_id
    line="$(awk -F'\t' -v r="$RUN_ID" '$2 == r && $3 ~ /^cidpass-/' "$SESSIONS" 2>/dev/null | tail -1 || true)"
    [[ -n "$line" ]] || { echo "no cidpass VM recorded for $RUN_ID in $SESSIONS; refusing (sessions rule)" >&2; exit 1; }
    name="$(echo "$line" | cut -f3)"; zone="$(echo "$line" | cut -f4)"
    run "gcloud compute instances delete '$name' --project='$PROJECT' --zone='$zone' --quiet"
    ;;
  *)
    sed -n '2,24p' "$0"; exit 2 ;;
esac
[[ $EXECUTE -eq 1 ]] || echo "# dry run: nothing was executed (add --execute after R-CID is approved)"
