#!/usr/bin/env bash
# Launcher for S12m (natively multimodal S12, S1 amendment A2) on ONE GCE Spot L4 in us-west1.
#
# PRINT-ONLY BY DEFAULT. Nothing touches GCP unless --execute is given; launching is the coordinator's
# (Josh's standing "spend the credits" approval, each run under its own cap and receipt):
#
#   launch_s12mm.sh zones                 # read-only: L4 per us-west1 zone, L4/GPU quotas, the DLVM image
#   launch_s12mm.sh stage-model           # one-time: Mac EG2 weights + SHA256SUMS -> plan-30d/inputs/models/
#   launch_s12mm.sh bench  RUN_ID         # R-S7g benchmark/smoke: max-run 90 min, cap $1.50
#   launch_s12mm.sh s12    RUN_ID         # R-S12m main run: max-run 24 h per VM, cap CAP_USD (default $45)
#   launch_s12mm.sh resume RUN_ID         # a fresh VM continuing RUN_ID from its shard ledger (same caps)
#   launch_s12mm.sh status | delete RUN_ID
#   ... --execute
#
# Rules: Spot only (if no zone has capacity, stop and reschedule; never on-demand); g2-standard-8 (8 vCPU,
# 32 GB, 1 x NVIDIA L4 attached by the machine type); --max-run-duration with termination action DELETE; the
# VM also deletes itself on any exit (EXIT trap); label plan=whole-network-signals for the D1b budget; one VM
# at a time per run (one IP: no parallel fetchers). Image: Deep Learning VM `common-cu129-ubuntu-2404-nvidia-580`
# (NVIDIA driver 580 baked in, CUDA 12.9 toolkit; driver 580 also runs PyTorch's CUDA 13.0 wheels), pinned to
# the image name below; the job's Python stack is pip-pinned in a venv (S4/S7's torch 2.14.1 /
# transformers 5.21.0 / sentence-transformers 6.2.0) and recorded in the receipt.
set -euo pipefail

MODE="${1:-}"; shift || true
RUN_ID="${1:-}"; [[ "${RUN_ID:-}" == --* ]] && RUN_ID="" || shift || true
EXECUTE=0
for a in "$@"; do [[ "$a" == "--execute" ]] && EXECUTE=1; done

PROJECT="${PROJECT:-catbird-436600}"
REGION="us-west1"
ZONES="${ZONES:-us-west1-b us-west1-a us-west1-c}"     # all three list nvidia-l4 (checked 2026-10-28)
MACHINE="g2-standard-8"
IMAGE_PROJECT="deeplearning-platform-release"
IMAGE="${IMAGE:-common-cu129-ubuntu-2404-nvidia-580-v20261019}"
PLAN30="${PLAN30:-gs://catbird-436600-plan-30d}"
PLAN60="${PLAN60:-gs://catbird-436600-plan-60d}"
LABEL_KEY="plan"; LABEL_VAL="whole-network-signals"
USD_PER_HOUR="${USD_PER_HOUR:-0.5338}"   # Spot g2-standard-8 $0.5108 + 150 GB pd-balanced $0.0205 + Spot IP $0.0025
RPS="${RPS:-50}"                          # plan §5.3 etiquette cap; above 50 needs ALLOW_ABOVE=1 and Bluesky's OK
ALLOW_ABOVE="${ALLOW_ABOVE:-0}"
SOFT="${SOFT:-70}"
POSTS="${POSTS:-}"                        # S12m input prefix (O8 shards), gs://catbird-436600-plan-60d/...
SIDECAR="${SIDECAR:-}"                    # image-CID sidecar prefix (did, rkey, image_cids), optional if joined
MODEL_LOCAL="${MODEL_LOCAL:-$HOME/catbird-lake-local/models}"
S7IN="${S7IN:-$HOME/catbird-lake-local/s7-bench-20261028/upload}"
MM_PARITY="${MM_PARITY:-$HOME/catbird-lake-local/s12mm-rehearsal-20261028/mm_parity_mac.npz}"
BENCH_PRODUCERS="${BENCH_PRODUCERS:-}"
SESSIONS="${SESSIONS:-$HOME/catbird-lake-local/s12mm-sessions.tsv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH="$(cd "$HERE/.." && pwd)"

run() { printf '+ %s\n' "$*"; if [[ $EXECUTE -eq 1 ]]; then eval "$*"; fi; }
need_run_id() { [[ -n "$RUN_ID" ]] || { echo "usage: $0 $MODE RUN_ID [--execute]" >&2; exit 2; }; }

stage_code() {
  local tb="/tmp/s12mm-$RUN_ID.tar.gz"
  run "tar -C '$GRAPH' --exclude='__pycache__' -czf '$tb' s12mm s7bench/s7lib.py s7bench/bench.py c4images/gate.py c4images/cdn.py vmpass/store.py"
  run "gcloud storage cp '$tb' '$PLAN30/inputs/$RUN_ID/code.tar.gz'"
  echo "# gate bundle (D9): run gate_bundle.py right after the label tail, then upload it for this run:"
  echo "#   python3 $HERE/gate_bundle.py ~/catbird-lake-local/s12mm-gate-$RUN_ID --labels <labels.tsv> --watermark <ISO> --upload $RUN_ID --execute"
}

create_vm() {  # $1 step, $2 max-run, $3 cap USD
  local step="$1" maxrun="$2" cap="$3" name cmd
  name="s12mm-${step}-$(echo "$RUN_ID" | tr 'A-Z_.' 'a-z--')"
  echo "# Spot capacity: try each zone; if none has capacity, stop and reschedule. Never on-demand."
  for z in $ZONES; do
    cmd="gcloud compute instances create '$name' --project='$PROJECT' --zone='$z' \
 --machine-type='$MACHINE' --provisioning-model=SPOT --instance-termination-action=DELETE \
 --max-run-duration='$maxrun' --maintenance-policy=TERMINATE \
 --image='$IMAGE' --image-project='$IMAGE_PROJECT' --boot-disk-size=150GB --boot-disk-type=pd-balanced \
 --scopes=storage-rw,compute-rw --labels='$LABEL_KEY=$LABEL_VAL,step=s12mm-$step,run=$(echo "$RUN_ID" | tr 'A-Z_.' 'a-z--')' \
 --metadata-from-file=startup-script='$HERE/vm_bootstrap_s12mm.sh' \
 --metadata='s12mm-run-id=$RUN_ID,s12mm-mode=${step%%-*},s12mm-plan30=$PLAN30,s12mm-plan60=$PLAN60,s12mm-cap-usd=$cap,s12mm-usd-per-hour=$USD_PER_HOUR,s12mm-rps=$RPS,s12mm-allow-above=$ALLOW_ABOVE,s12mm-soft=$SOFT,s12mm-posts=$POSTS,s12mm-sidecar=$SIDECAR'"
    printf '+ %s\n' "$cmd"
    if [[ $EXECUTE -eq 1 ]]; then
      if eval "$cmd"; then
        mkdir -p "$(dirname "$SESSIONS")"
        printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$RUN_ID" "$name" "$z" >> "$SESSIONS"
        echo "# recorded $name in $SESSIONS (only recorded VMs are ever deleted by this launcher)"
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
  zones)
    EXECUTE=1   # read-only listings
    run "gcloud compute accelerator-types list --project='$PROJECT' --filter='zone~us-west1 AND name=nvidia-l4' --format='table(name,zone.basename())'"
    run "gcloud compute regions describe $REGION --project='$PROJECT' --format=json | python3 -c \"import json,sys; [print(q) for q in json.load(sys.stdin)['quotas'] if 'L4' in q['metric']]\""
    run "gcloud compute project-info describe --project='$PROJECT' --format=json | python3 -c \"import json,sys; [print(q) for q in json.load(sys.stdin)['quotas'] if q['metric']=='GPUS_ALL_REGIONS']\""
    run "gcloud compute images describe '$IMAGE' --project='$IMAGE_PROJECT' --format='value(name,family,status,creationTimestamp)'"
    echo "# GPUS_ALL_REGIONS must be >= 1 or every GPU VM create fails (it was 0 on 2026-10-28)"
    exit 0 ;;
  stage-model)
    echo "# one-time upload of the Mac's EG2 weights (1.5 GB, free ingress, ~\$0.03/month in plan-30d for 30 days)"
    run "grep ' ./embeddinggemma-2/' '$MODEL_LOCAL/SHA256SUMS' > /tmp/eg2.SHA256SUMS"
    run "gcloud storage cp --recursive '$MODEL_LOCAL/embeddinggemma-2' '$PLAN30/inputs/models/'"
    run "gcloud storage cp /tmp/eg2.SHA256SUMS '$PLAN30/inputs/models/embeddinggemma-2.SHA256SUMS'" ;;
  bench)
    need_run_id
    echo "# R-S7g: ~10k text + ~10k image posts; fetch at ${RPS} req/s; 70 and 280 soft tokens; cap \$1.50 = \$${USD_PER_HOUR}/h x 1.5 h + GETs"
    [[ -n "$POSTS" ]] || echo "# WARNING: POSTS (S12m input prefix with image CIDs) is required for the real image sample" >&2
    stage_code
    for f in s7_slab.parquet s7_random.parquet parity_mac.npz; do run "gcloud storage cp '$S7IN/$f' '$PLAN30/inputs/$RUN_ID/bench/$f'"; done
    run "gcloud storage cp '$MM_PARITY' '$PLAN30/inputs/$RUN_ID/bench/mm_parity_mac.npz'"
    [[ -n "$BENCH_PRODUCERS" ]] && run "gcloud storage cp '$BENCH_PRODUCERS' '$PLAN30/inputs/$RUN_ID/bench/bench_producers.txt'"
    create_vm bench 90m 1.50 ;;
  s12)
    need_run_id
    [[ -n "$POSTS" ]] || { echo "POSTS=gs://...O8 prefix is required" >&2; exit 2; }
    CAP="${CAP_USD:-45}"
    echo "# R-S12m: cap \$$CAP across all VMs of this run (the job's guard sums earlier VMs' ledger spend); max-run 24 h per VM"
    stage_code
    create_vm s12 24h "$CAP" ;;
  resume)
    need_run_id
    CAP="${CAP_USD:-45}"
    echo "# resume $RUN_ID on a fresh VM: finished units are skipped (ledger), the cap is shared"
    create_vm s12-resume 24h "$CAP" ;;
  status)
    run "gcloud compute instances list --project='$PROJECT' --filter='labels.$LABEL_KEY=$LABEL_VAL AND labels.step~^s12mm' --format='table(name,zone.basename(),status,labels.step,labels.run)'" ;;
  delete)
    need_run_id
    line="$(awk -F'\t' -v r="$RUN_ID" '$2 == r' "$SESSIONS" 2>/dev/null | tail -1 || true)"
    [[ -n "$line" ]] || { echo "no VM recorded for $RUN_ID in $SESSIONS; refusing" >&2; exit 1; }
    run "gcloud compute instances delete '$(echo "$line" | cut -f3)' --project='$PROJECT' --zone='$(echo "$line" | cut -f4)' --quiet" ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
[[ $EXECUTE -eq 1 ]] || echo "# dry run: nothing was executed (add --execute to run the printed commands)"
