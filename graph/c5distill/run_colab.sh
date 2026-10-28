#!/usr/bin/env bash
# C5 Colab driver. PRINT-ONLY unless --execute; run with --execute only after Josh approves R-C5.
#
#   run_colab.sh [--execute] [--gpu L4] [--cap 20]
#
# One named session (sessions rule, plan §3): created, recorded in sessions.tsv, used, stopped. It never
# stops or touches a session it did not create. Data goes Mac -> VM by `colab upload` (no GCS, no Drive,
# no `colab auth` OAuth grant); results come back by `colab download`; `colab stop` destroys the VM and the
# uploaded text with it.
set -euo pipefail
EXEC=0; GPU=L4; CAP=20
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXEC=1;; --gpu) GPU="$2"; shift;; --cap) CAP="$2"; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac; shift
done
case "$GPU" in L4) RATE=1.71;; A100) RATE=5.4;; *) echo "GPU must be L4 or A100 (R-C5 names one)" >&2; exit 2;; esac
if awk "BEGIN{exit !($CAP > 20)}"; then echo "cap above the Batch A cap of 20" >&2; exit 2; fi

HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="${C5_RUN_DIR:-$HOME/catbird-lake-local/c5-distill-20261028}"
IN="$RUN/inputs"; STAGE="$RUN/colab-stage"; RES="$RUN/colab-results"
SESSION="c5-distill-$(date -u +%Y%m%dT%H%M)"
COLAB=(~/.local/bin/colab --config "$RUN/colab-cli-state.json")
# Wall bound for the exec call: the cap's hours at this GPU's rate, at most 6 h (expected ~1.2 h on L4).
TIMEOUT=$(awk "BEGIN{t=$CAP/$RATE*3600; if (t>21600) t=21600; printf \"%d\", t}")

run() { echo "+ $*"; if [ $EXEC = 1 ]; then "$@"; fi; }

echo "== C5 Colab run: session=$SESSION gpu=$GPU cap=$CAP units rate=$RATE/h exec-timeout=${TIMEOUT}s execute=$EXEC"
mkdir -p "$STAGE" "$RES"
for f in c5_pairs.parquet c5_profiles.parquet c5_communities.parquet; do cp "$IN/$f" "$STAGE/$f"; done
for f in rules_c5.json c5lib.py train.py colab_job.py; do cp "$HERE/$f" "$STAGE/$f"; done
python3 - "$STAGE" <<'EOF'
import hashlib, json, os, sys
root = sys.argv[1]
files = {}
for rel in ["rules_c5.json", "c5lib.py", "train.py", "c5_pairs.parquet", "c5_profiles.parquet", "c5_communities.parquet"]:
    files[rel] = hashlib.sha256(open(os.path.join(root, rel), "rb").read()).hexdigest()
json.dump({"files": files}, open(os.path.join(root, "c5_manifest.json"), "w"), indent=1)
print(json.dumps(files, indent=1))
EOF
du -sh "$STAGE"

run "${COLAB[@]}" usage | tee "$RES/usage_before.txt"
run "${COLAB[@]}" sessions | tee "$RES/sessions_before.txt"
run "${COLAB[@]}" new -s "$SESSION" --gpu "$GPU"
[ $EXEC = 1 ] && printf '%s\t%s\t%s\tcreated\n' "$(date -u +%FT%TZ)" "$SESSION" "$GPU" >> "$RUN/sessions.tsv"
trap '[ $EXEC = 1 ] && { "${COLAB[@]}" stop -s "$SESSION" || true; printf "%s\t%s\t-\tstopped(trap)\n" "$(date -u +%FT%TZ)" "$SESSION" >> "$RUN/sessions.tsv"; }' EXIT
ALLOC_T0=$(date +%s)
# The contents API will not create directories, so make /content/c5 first (the 2026-10-28 first try got a 500).
printf 'import os\nos.makedirs("/content/c5", exist_ok=True)\nprint("mkdir ok")\n' > "$STAGE/_mkdir.py"
run "${COLAB[@]}" exec -s "$SESSION" -f "$STAGE/_mkdir.py" --timeout 60
for rel in c5_manifest.json rules_c5.json c5lib.py train.py colab_job.py \
           c5_pairs.parquet c5_profiles.parquet c5_communities.parquet; do
  run "${COLAB[@]}" upload -s "$SESSION" "$STAGE/$rel" "/content/c5/$rel"
done
ALLOC_MIN=$(( ( $(date +%s) - ALLOC_T0 ) / 60 + 3 ))
run "${COLAB[@]}" exec -s "$SESSION" -f "$STAGE/colab_job.py" --timeout "$TIMEOUT" \
    --env C5_UNIT_CAP="$CAP" --env C5_ALLOC_MIN="$ALLOC_MIN" 2>&1 | tee "$RES/colab_exec.log" || true
run "${COLAB[@]}" download -s "$SESSION" /content/c5_results.tar "$RES/c5_results.tar"
run "${COLAB[@]}" download -s "$SESSION" /content/c5_models.tar "$RES/c5_models.tar"
run "${COLAB[@]}" stop -s "$SESSION"
[ $EXEC = 1 ] && printf '%s\t%s\t-\tstopped\n' "$(date -u +%FT%TZ)" "$SESSION" >> "$RUN/sessions.tsv"
trap - EXIT
run "${COLAB[@]}" sessions | tee "$RES/sessions_after.txt"
run "${COLAB[@]}" usage | tee "$RES/usage_after.txt"
echo "== then: tar -xf $RES/c5_results.tar -C $RES; compare the TAR sha256 lines in colab_exec.log; check 'Active assignments: 0'"
