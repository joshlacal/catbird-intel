#!/usr/bin/env bash
# Launcher for the us-west1 pass (plan S8 smoke test, S9 main pass).
#
# PRINT-ONLY BY DEFAULT. Nothing touches GCP unless --execute is given, and
# --execute is for Josh (or an agent he approved) after the run ask (R-S8 / R-S9):
#
#   launch_vm.sh s8  RUN_ID            # print the S8 commands (max-run 1.5 h, cap $4.50; R-S8 2026-10-28)
#   launch_vm.sh s9  RUN_ID            # print the S9 commands (max-run 9 h, cap $32)
#   launch_vm.sh resume RUN_ID         # S9's one resumed restart (fresh VM, max-run 2 h 45 m)
#   launch_vm.sh buckets               # print the D1b bucket + lifecycle + label commands
#   launch_vm.sh status                # list VMs carrying this plan's label
#   launch_vm.sh delete RUN_ID         # delete only the VM this launcher recorded for RUN_ID
#   ... --execute                      # actually run the printed commands
#
# Rules from the plan (§3.1 S8/S9, §7.1): Spot only (if no zone has capacity,
# reschedule; never on-demand), n2-highmem-64 + 8 x 375 GiB local NVMe SSD in
# us-west1, --max-run-duration with --instance-termination-action=DELETE, a fresh
# VM for S9, labels so the label-scoped budget (D1b) sees the spend. The Rust
# tools (simgraph) are built from source on the VM; the Mac binary is arm64.
set -euo pipefail

MODE="${1:-}"; shift || true
RUN_ID="${1:-}"; [[ "${RUN_ID:-}" == --* ]] && RUN_ID="" || shift || true
EXECUTE=0
for a in "$@"; do [[ "$a" == "--execute" ]] && EXECUTE=1; done

PROJECT="${PROJECT:-catbird-436600}"
REGION="us-west1"
ZONES="${ZONES:-us-west1-b us-west1-a us-west1-c}"
MACHINE="${MACHINE:-n2-highmem-64}"
N_SSD="${N_SSD:-8}"
PLAN30="${PLAN30:-gs://catbird-436600-plan-30d}"
PLAN60="${PLAN60:-gs://catbird-436600-plan-60d}"
EXTRACT="gs://catbird-436600-jetstream-buffer-20261020/derived/all-extract-20261024"
LABEL_KEY="plan"; LABEL_VAL="whole-network-signals"   # the label the D1b "core cap" budget filters on
CH_VERSION="${CH_VERSION:-26.9.24.8}"
HERE0="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES="${RULES:-$HERE0/rules_s1.json}"   # the S1 translation (S9 refuses a DRAFT rules file)
O3B_SCALES="${O3B_SCALES:-}"             # S9: S8's outputs/<s8 run>/S8/o3b_scales.json (S1 R4)
SESSIONS="${SESSIONS:-$HOME/catbird-lake-local/vmpass-sessions.tsv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPH="$(cd "$HERE/.." && pwd)"

run() {  # print, and run only with --execute
  printf '+ %s\n' "$*"
  if [[ $EXECUTE -eq 1 ]]; then eval "$*"; fi
}

need_run_id() { [[ -n "$RUN_ID" ]] || { echo "usage: $0 $MODE RUN_ID [--execute]" >&2; exit 2; }; }

stage_code() {  # code + inputs go up first (free ingress); the VM pulls them
  local tarball="/tmp/vmpass-$RUN_ID.tar.gz"
  run "tar -C '$GRAPH' --exclude='simgraph/target' --exclude='__pycache__' --exclude='.venv' -czf '$tarball' vmpass simgraph archive_graph_extract.py post_extract.py gcs_segments.py"
  run "gcloud storage cp '$tarball' '$PLAN30/inputs/$RUN_ID/code.tar.gz'"
  if [[ -n "$RULES" ]]; then run "gcloud storage cp '$RULES' '$PLAN30/inputs/$RUN_ID/rules.json'"; fi
  if [[ -n "$O3B_SCALES" ]]; then run "gcloud storage cp '$O3B_SCALES' '$PLAN30/inputs/$RUN_ID/o3b_scales.json'"; fi
  # Side inputs (free ingress), so producer-keyed outputs use the edges.bin pids and every exclusion is applied.
  local lake="$HOME/catbird-lake-local"
  local -a inputs=(
    "producers.tsv=${PRODUCERS:-$lake/graph-build-v3/out-full-20261024/producers.tsv}"
    "hide.tsv=${HIDE:-$lake/s5-declarations-20261027/hide_from_recs_union.tsv}"
    "intents.tsv=${INTENTS:-$lake/s5-declarations-20261027/user_intents_records.tsv}"
    "status.tsv=${STATUS:-}" "plc.tsv=${PLC:-}" "comm_cand1.tsv=${COMM1:-}" "comm_cand5.tsv=${COMM5:-}"
    "m2_cohort.parquet=${M2_COHORT:-}")   # S2a's m2_cohort.parquet (did, created_us): O13's M2 users
  for kv in "${inputs[@]}"; do
    local name="${kv%%=*}" path="${kv#*=}"
    if [[ -n "$path" ]]; then run "gcloud storage cp '$path' '$PLAN30/inputs/$RUN_ID/$name'"
    else echo "# optional input $name not given"; fi
  done
}

create_vm() {  # $1 step label, $2 max-run duration
  local step="$1" maxrun="$2"
  local name="vmpass-${step}-${RUN_ID}" rules_sha="" cmd
  [[ -n "$RULES" && -f "$RULES" ]] && rules_sha="$(shasum -a 256 "$RULES" | cut -d' ' -f1)"
  echo "# Spot capacity: try each zone; if none has capacity, stop and reschedule. Never on-demand."
  for z in $ZONES; do
    cmd="gcloud compute instances create '$name' --project='$PROJECT' --zone='$z' \
 --machine-type='$MACHINE' $(for _ in $(seq $N_SSD); do printf -- '--local-ssd=interface=NVME '; done)\
 --provisioning-model=SPOT --instance-termination-action=DELETE --max-run-duration='$maxrun' \
 --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=100GB --boot-disk-type=pd-balanced \
 --scopes=storage-rw,compute-rw --labels='$LABEL_KEY=$LABEL_VAL,step=$step,run=$(echo "$RUN_ID" | tr 'A-Z_.' 'a-z--')' \
 --metadata-from-file=startup-script='$HERE/vm_bootstrap.sh' \
 --metadata='vmpass-run-id=$RUN_ID,vmpass-mode=${step%%-*},vmpass-plan30=$PLAN30,vmpass-plan60=$PLAN60,vmpass-extract=$EXTRACT,vmpass-ch-version=$CH_VERSION,vmpass-rules-sha256=$rules_sha,vmpass-bucket-shards=${BUCKET_SHARDS:-0},vmpass-python=${VM_PYTHON:-3.14}'"
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
  s8)
    need_run_id
    echo "# S8 smoke test: cap \$4.50 = \$2.715/h x <=1.5 h max-run (+ ~120k Class B GETs ~ \$0.05)"
    stage_code
    create_vm s8 90m
    ;;
  s9)
    need_run_id
    [[ -n "$O3B_SCALES" ]] || echo "# WARNING: S9 should get O3B_SCALES=<S8 run>/S8/o3b_scales.json (S1 R4); its sha256 goes in R-S9" >&2
    echo "# S9 main pass: cap \$32 = (9 h + one 2.75 h restart) x \$2.715/h; a fresh VM re-stages (10-30 min)"
    stage_code
    create_vm s9 9h
    ;;
  s8-resume)
    need_run_id
    echo "# S8 resumed on a fresh VM, same RUN_ID: inputs, inventory, read_rate, hash check and the DID map are skipped (ledger);"
    echo "# staging and bucketing rerun. The code tarball is re-uploaded so the VM runs this checkout. BUCKET_SHARDS=K stages and"
    echo "# buckets only K evenly spaced directories (projection scaled by their row fraction). Max-run \${S8_MAXRUN:-90m}."
    stage_code
    create_vm s8-resume "${S8_MAXRUN:-90m}"
    ;;
  resume)
    need_run_id
    echo "# S9's one resumed restart on a fresh VM (same RUN_ID: finished partitions are skipped). A second preemption stops the run."
    create_vm s9-resume 165m
    ;;
  buckets)
    echo "# D1b (approved 2026-10-28): two labelled us-west1 STANDARD buckets with lifecycle rules"
    cat > /tmp/vmpass-lifecycle-30d.json <<'EOF'
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 30}},
          {"action": {"type": "Delete"}, "condition": {"age": 7, "matchesPrefix": ["intermediates/"]}}]}
EOF
    cat > /tmp/vmpass-lifecycle-60d.json <<'EOF'
{"rule": [{"action": {"type": "Delete"}, "condition": {"daysSinceCustomTime": 60}}]}
EOF
    for b in "$PLAN30:30d" "$PLAN60:60d"; do
      url="${b%:*}"; life="${b##*:}"
      run "gcloud storage buckets create '$url' --project='$PROJECT' --location='$REGION' --default-storage-class=STANDARD --uniform-bucket-level-access --public-access-prevention"
      run "gcloud storage buckets update '$url' --update-labels='$LABEL_KEY=$LABEL_VAL' --lifecycle-file=/tmp/vmpass-lifecycle-$life.json"
    done
    echo "# plan-60d: the pass sets customTime on every first write (gcloud storage cp --custom-time); S5r must copy it"
    ;;
  status)
    run "gcloud compute instances list --project='$PROJECT' --filter='labels.$LABEL_KEY=$LABEL_VAL' --format='table(name,zone.basename(),status,labels.step,labels.run)'"
    echo "# 'Active instances: 0' means the list above is empty"
    ;;
  delete)
    need_run_id
    line="$(awk -F'\t' -v r="$RUN_ID" '$2 == r' "$SESSIONS" 2>/dev/null | tail -1 || true)"
    [[ -n "$line" ]] || { echo "no VM recorded for $RUN_ID in $SESSIONS; refusing (sessions rule)" >&2; exit 1; }
    name="$(echo "$line" | cut -f3)"; zone="$(echo "$line" | cut -f4)"
    run "gcloud compute instances delete '$name' --project='$PROJECT' --zone='$zone' --quiet"
    ;;
  *)
    sed -n '2,20p' "$0"; exit 2 ;;
esac
[[ $EXECUTE -eq 1 ]] || echo "# dry run: nothing was executed (add --execute after the run ask is approved)"
