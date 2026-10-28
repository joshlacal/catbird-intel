#!/usr/bin/env bash
# S7 Colab driver. PRINT-ONLY unless --execute; run with --execute only after Josh approves R-S7.
#
#   run_colab.sh [--execute] [--cap 8] [--gpus L4,A100] [--no-roundtrip] [--auth-wait-min 20] [--fallback-wait-min 15]
#
# Whole-network plan §3.1 row S7, §5.1. Two named sessions, one per GPU (the CLI binds one accelerator to a session
# at `colab new`; there is no switching), L4 first. Sessions rule (plan §3): each session is recorded in sessions.tsv
# when created and only those are ever stopped; a session this script did not create is never touched. Each session
# has (a) a budget = what is left of the cap, at most 3 units for L4; (b) a watchdog that runs `colab stop` on THAT
# session when its wall time x rate reaches the budget; (c) `colab exec --timeout`; (d) the job's own unit guard.
# Data Mac -> VM by `colab upload` (text + row id only); results back by `colab download`; `colab stop` destroys the VM.
# The GCS round trip runs in the L4 session after Josh runs `colab auth` himself (it needs a human: see RUNBOOK.md).
set -euo pipefail
EXEC=0; CAP=8; GPUS="L4,A100"; RT=1; AUTH_WAIT_MIN=20; FB_WAIT_MIN=15
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXEC=1;; --cap) CAP="$2"; shift;; --gpus) GPUS="$2"; shift;; --no-roundtrip) RT=0;;
    --auth-wait-min) AUTH_WAIT_MIN="$2"; shift;; --fallback-wait-min) FB_WAIT_MIN="$2"; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac; shift
done
if awk "BEGIN{exit !($CAP > 8)}"; then echo "cap above R-S7's 8 units" >&2; exit 2; fi

HERE="$(cd "$(dirname "$0")" && pwd)"
PY=(/usr/bin/python3 -I)
RUN="${S7_RUN_DIR:-$HOME/catbird-lake-local/s7-bench-20261028}"
UP="$RUN/upload"; STAGE="$RUN/colab-stage"; RES="$RUN/colab-results"
COLAB_BIN="${S7_COLAB_BIN:-$HOME/.local/bin/colab}"; GCLOUD="${S7_GCLOUD_BIN:-gcloud}"   # overridable only for the driver test
COLAB=("$COLAB_BIN" --config "$RUN/colab-cli-state.json")
POLL_S="${S7_POLL_S:-20}"
BUCKET=catbird-436600-plan-30d
RT_OBJECT="colab-test/synthetic-2e9-random-20261028.bin"; RT_BYTES="${S7_RT_BYTES:-2000000000}"
STAMP=$(date -u +%Y%m%dT%H%M)
red() { "${PY[@]}" "$HERE/s7lib.py" redact; }          # strips URL query strings and token-shaped strings
run() { echo "+ $*"; if [ $EXEC = 1 ]; then "$@"; fi; }
now() { date +%s; }

# ---- preflight (always) ----------------------------------------------------------------------------------------
FREE_GIB=$(df -g "$HOME" | awk 'NR==2{print $4}')
echo "== S7 Colab run: cap=$CAP gpus=$GPUS roundtrip=$RT execute=$EXEC run=$RUN df_free=${FREE_GIB}GiB"
[ "$FREE_GIB" -ge 100 ] || { echo "df gate: <100 GiB free" >&2; exit 2; }
for f in s7_random.parquet s7_slab.parquet parity_mac.npz; do [ -s "$UP/$f" ] || { echo "missing $UP/$f (run prep.py, then bench.py --emit-parity)" >&2; exit 2; }; done
mkdir -p "$STAGE" "$RES"
for f in s7_random.parquet s7_slab.parquet parity_mac.npz; do cp "$UP/$f" "$STAGE/$f"; done
for f in s7lib.py bench.py colab_job.py roundtrip_job.py; do cp "$HERE/$f" "$STAGE/$f"; done
"${PY[@]}" - "$STAGE" "$HOME/catbird-lake-local/models" <<'EOF'
import hashlib, json, os, sys
stage, models = sys.argv[1], sys.argv[2]
want = {}
for line in open(os.path.join(models, "SHA256SUMS")):
    h, p = line.split(None, 1); p = p.strip()
    if p.startswith("./embeddinggemma-2/"):
        want[p[len("./embeddinggemma-2/"):]] = h
want["chat_template.jinja"] = hashlib.sha256(open(os.path.join(models, "embeddinggemma-2/chat_template.jinja"), "rb").read()).hexdigest()
json.dump(want, open(os.path.join(stage, "eg2_sha256.json"), "w"), indent=1, sort_keys=True)
files = {}
for rel in ["s7lib.py", "bench.py", "roundtrip_job.py", "s7_random.parquet", "s7_slab.parquet", "parity_mac.npz", "eg2_sha256.json"]:
    files[rel] = hashlib.sha256(open(os.path.join(stage, rel), "rb").read()).hexdigest()
json.dump({"files": files}, open(os.path.join(stage, "s7_manifest.json"), "w"), indent=1)
print("upload manifest:", json.dumps({k: (v[:12], os.path.getsize(os.path.join(stage, k))) for k, v in files.items()}))
print("model files checked on the VM:", len(want))
EOF
du -sh "$STAGE"
UPLOADS=(s7_manifest.json s7lib.py bench.py colab_job.py roundtrip_job.py eg2_sha256.json s7_random.parquet s7_slab.parquet parity_mac.npz)

echo "== the synthetic GCS object (Mac -> gs://$BUCKET/$RT_OBJECT, $RT_BYTES random bytes, free ingress):"
GEN="$RUN/rt_object.sha256"
GEN_CMD="${PY[*]} $HERE/s7lib.py randbytes $RT_BYTES $GEN | $GCLOUD storage cp - gs://$BUCKET/$RT_OBJECT"
echo "  $GEN_CMD"
echo "  gcloud storage objects describe gs://$BUCKET/$RT_OBJECT --format='value(size,crc32c_hash,time_created)'"

# ---- receipts before --------------------------------------------------------------------------------------------
run "${COLAB[@]}" usage 2>&1 | red | tee "$RES/usage_before.txt" || true
run "${COLAB[@]}" sessions 2>&1 | red | tee "$RES/sessions_before.txt" || true

if [ $RT = 1 ] && [ $EXEC = 1 ]; then
  if "$GCLOUD" storage objects describe "gs://$BUCKET/$RT_OBJECT" --format='value(size)' >/dev/null 2>&1; then
    echo "object exists; reusing it (sha256 from $GEN)"
  else
    eval "$GEN_CMD"
  fi
  [ -s "$GEN" ] || { echo "no sha256 recorded for the test object; refusing the round trip" >&2; RT=0; }
  "$GCLOUD" storage objects describe "gs://$BUCKET/$RT_OBJECT" --format=json > "$RES/rt_object_describe.json" || RT=0
fi
RT_SHA=$( [ -s "$GEN" ] && cut -c1-64 "$GEN" || echo "<sha256 recorded when the object is written>")

SPENT=0
record() { printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" >> "$RUN/sessions.tsv"; }

wait_for_adc() {  # $1 session, $2 minutes; polls the VM's file listing (contents API, never the kernel)
  local s=$1 deadline=$(( $(now) + $(awk "BEGIN{printf \"%d\", $2 * 60}") ))
  while :; do                                     # always polls at least once
    if "${COLAB[@]}" ls -s "$s" /content/.adc 2>/dev/null | grep -q adc.json; then return 0; fi
    [ "$(now)" -lt "$deadline" ] || return 1
    sleep "$POLL_S"
  done
}

roundtrip() {  # $1 session, $2 log suffix
  local s=$1
  "${COLAB[@]}" exec -s "$s" -f "$STAGE/roundtrip_job.py" --timeout 900 \
      --env S7_RT_OBJECT="$RT_OBJECT" --env S7_RT_SHA256="$RT_SHA" --env S7_RT_BYTES="$RT_BYTES" --env S7_SESSION="$s" \
      2>&1 | red | tee "$RES/roundtrip_$2.log" || true
  "${COLAB[@]}" download -s "$s" /content/s7_rt.tar "$RES/s7_rt_$2.tar" 2>&1 | red || true
  grep -q ROUNDTRIP_OK "$RES/roundtrip_$2.log"
}

for GPU in ${GPUS//,/ }; do
  case "$GPU" in L4) RATE=1.71; SCAP=3.0; MIN=0.5;; A100) RATE=5.4; SCAP=4.5; MIN=1.2;; *) echo "GPU must be L4 or A100" >&2; exit 2;; esac
  LEFT=$(awk "BEGIN{printf \"%.3f\", $CAP - $SPENT - 0.2}")           # 0.2-unit margin for stop latency
  SCAP=$(awk "BEGIN{c=$SCAP; if ($LEFT < c) c=$LEFT; printf \"%.3f\", c}")
  if awk "BEGIN{exit !($SCAP < $MIN)}"; then echo "== skip $GPU: $SCAP units left, below the $MIN-unit minimum"; continue; fi
  SESSION="s7-$(echo "$GPU" | tr 'A-Z' 'a-z')-$STAMP"
  echo "== $GPU session $SESSION: budget $SCAP units at >= $RATE units/h"
  U0=$( [ $EXEC = 1 ] && "${COLAB[@]}" usage 2>&1 | red || echo "")
  if ! run "${COLAB[@]}" new -s "$SESSION" --gpu "$GPU" 2>&1 | red; then
    echo "colab new failed for $GPU; not retrying another GPU"; continue
  fi
  T0=$(now)
  if [ $EXEC = 1 ]; then
    record "$SESSION" "$GPU" created
    U1=$("${COLAB[@]}" usage 2>&1 | red); echo "$U1" > "$RES/usage_after_new_$GPU.txt"
    LIVE=$("${PY[@]}" -c "
import re,sys
def r(t):
    m=re.search(r'Usage rate:\s*([0-9.]+)',t); return float(m.group(1)) if m else 0.0
print(max(0.0, r(sys.argv[2]) - r(sys.argv[1])))" "$U0" "$U1")
    RATE=$(awk "BEGIN{r=$RATE; if ($LIVE > r) r=$LIVE; print r}")
    echo "measured rate delta $LIVE/h -> guard rate $RATE/h"
  fi
  BUDGET_S=$(awk "BEGIN{printf \"%d\", $SCAP / $RATE * 3600}")
  [ -n "${S7_TEST_BUDGET_S:-}" ] && BUDGET_S=$S7_TEST_BUDGET_S       # driver test only (test_driver.py)
  # Watchdog: stops THIS session (only) when its projected spend reaches its budget.
  if [ $EXEC = 1 ]; then
    # stdio detached so an outstanding sleep never holds the caller's pipe; TERM (normal finish) also kills the sleep.
    ( trap 'kill $SP 2>/dev/null; exit 0' TERM; sleep "$BUDGET_S" & SP=$!; wait $SP
      "${COLAB[@]}" stop -s "$SESSION"; record "$SESSION" - "stopped(watchdog ${BUDGET_S}s)" ) </dev/null >/dev/null 2>&1 &
    WD=$!
    trap '[ $EXEC = 1 ] && { kill $WD 2>/dev/null || true; "${COLAB[@]}" stop -s "$SESSION" || true; record "$SESSION" - "stopped(trap)"; }' EXIT
  else
    echo "+ (watchdog) sleep $BUDGET_S; colab stop -s $SESSION"
  fi
  printf 'import os\nos.makedirs("/content/s7", exist_ok=True)\nprint("mkdir ok")\n' > "$STAGE/_mkdir.py"
  run "${COLAB[@]}" exec -s "$SESSION" -f "$STAGE/_mkdir.py" --timeout 60 2>&1 | red
  for rel in "${UPLOADS[@]}"; do run "${COLAB[@]}" upload -s "$SESSION" "$STAGE/$rel" "/content/s7/$rel" 2>&1 | red; done
  ALLOC_MIN=$(( ( $(now) - T0 ) / 60 + 3 ))
  LEFT_S=$(( BUDGET_S - ( $(now) - T0 ) ))
  run "${COLAB[@]}" exec -s "$SESSION" -f "$STAGE/colab_job.py" --timeout "$LEFT_S" \
      --env S7_UNIT_CAP="$SCAP" --env S7_ALLOC_MIN="$ALLOC_MIN" --env S7_RATE="$RATE" 2>&1 | red | tee "$RES/colab_exec_$GPU.log" || true
  run "${COLAB[@]}" download -s "$SESSION" /content/s7_results.tar "$RES/s7_results_$GPU.tar" 2>&1 | red || true

  if [ "$GPU" = L4 ] && [ $RT = 1 ]; then
    cat <<MSG

==== JOSH, NOW (in your own terminal; this needs a human and cannot run in the agent's shell) ====
  ~/.local/bin/colab --config "$RUN/colab-cli-state.json" auth -s $SESSION
It prints a Google sign-in URL. Open it, sign in as the account that owns catbird-436600, approve the
requested access (Google Cloud + Google Drive, see RUNBOOK.md), copy the code it shows, paste it back at the
prompt. This script polls the VM for /content/.adc/adc.json for up to $AUTH_WAIT_MIN min, then runs the round trip.
==================================================================================================
MSG
    RT_RESULT=failed
    if [ $EXEC = 1 ]; then
      if wait_for_adc "$SESSION" "$AUTH_WAIT_MIN" && roundtrip "$SESSION" primary; then RT_RESULT=ok-primary; fi
      if [ $RT_RESULT = failed ]; then
        cat <<MSG

==== FALLBACK (plan §5.1): Josh runs the auth cell interactively ====
  Open the Colab UI attached to this same VM:  ~/.local/bin/colab --config "$RUN/colab-cli-state.json" url -s $SESSION --open
  In a new cell run:   from google.colab import auth; auth.authenticate_user()
  and complete the sign-in in the browser. Do NOT stop or restart the runtime. Polling up to $FB_WAIT_MIN min.
=====================================================================
MSG
        if wait_for_adc "$SESSION" "$FB_WAIT_MIN" && roundtrip "$SESSION" fallback; then RT_RESULT=ok-fallback; fi
      fi
    else
      echo "+ (poll) colab ls -s $SESSION /content/.adc  every 20 s, up to $AUTH_WAIT_MIN min"
      echo "+ ${COLAB[*]} exec -s $SESSION -f $STAGE/roundtrip_job.py --timeout 900 --env S7_RT_OBJECT=$RT_OBJECT --env S7_RT_SHA256=$RT_SHA --env S7_RT_BYTES=$RT_BYTES --env S7_SESSION=$SESSION"
      echo "+ ${COLAB[*]} download -s $SESSION /content/s7_rt.tar $RES/s7_rt_primary.tar"
      echo "+ (on failure) print the §5.1 fallback instructions (colab url --open + auth cell), poll $FB_WAIT_MIN min, retry"
    fi
    echo "$RT_RESULT" > "$RES/roundtrip_result.txt"
    echo "== round trip: $RT_RESULT"
  fi

  run "${COLAB[@]}" stop -s "$SESSION" 2>&1 | red || true    # may already be stopped by the watchdog
  if [ $EXEC = 1 ]; then
    kill $WD 2>/dev/null || true; trap - EXIT
    record "$SESSION" - stopped
    SPENT=$(awk "BEGIN{printf \"%.3f\", $SPENT + ($(now) - $T0) / 3600 * $RATE}")
    echo "== $GPU done; projected units so far $SPENT of $CAP"
  fi
done

# ---- receipts after ---------------------------------------------------------------------------------------------
run "${COLAB[@]}" sessions 2>&1 | red | tee "$RES/sessions_after.txt" || true
run "${COLAB[@]}" usage 2>&1 | red | tee "$RES/usage_after.txt" || true
echo "$SPENT" > "$RES/units_projected_by_driver.txt"
run "${PY[@]}" "$HERE/receipt.py" "$RUN"
echo "== then: check 'Active assignments' in usage_after.txt is what it was before (C5 or none), receipt.json, and"
echo "   tomorrow Billing -> Reports, SKU 22EB-AAE8-FBCD, for the ~2 GB read (Q10)."
