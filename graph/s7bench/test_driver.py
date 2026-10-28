"""Control-flow test of run_colab.sh --execute against a FAKE colab CLI and a fake gcloud (nothing leaves the Mac).

Checks the sessions rule (only the run's own sessions are created/stopped; the pre-existing C5 session is never
touched), sessions.tsv, the round trip's primary path, the §5.1 fallback path, the skip when `colab new` fails, and
that the receipt is assembled. Needs ~/catbird-lake-local/models/SHA256SUMS (the driver reads it), so Mac-only.
"""
import os
import pathlib
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SUMS = pathlib.Path.home() / "catbird-lake-local/models/SHA256SUMS"

FAKE_COLAB = r"""#!/bin/bash
# fake colab: state in $FAKE_DIR; every call appended to $FAKE_DIR/calls.log
[ "$1" = --config ] && shift 2
echo "$*" >> "$FAKE_DIR/calls.log"
cmd=$1; shift
sess=""; [ "$1" = -s ] && sess=$2
n_active() { ls "$FAKE_DIR"/active.* 2>/dev/null | wc -l | tr -d ' '; }
case "$cmd" in
  usage) r=$(awk "BEGIN{print 1.71 + $(n_active) * ${FAKE_RATE:-1.71}}")
         echo "Current balance: 4990.00 compute units"; echo "Usage rate: $r/hr"; echo "Active assignments: $(( $(n_active) + 1 ))";;
  sessions) echo "c5-distill-20261028T1249  L4  BUSY"; for f in "$FAKE_DIR"/active.*; do [ -e "$f" ] && echo "${f##*/active.}  GPU  IDLE"; done; true;;
  new) gpu=$4; [ "$gpu" = "${FAKE_FAIL_NEW:-none}" ] && { echo "400 no quota" >&2; exit 1; }; touch "$FAKE_DIR/active.$sess"; echo "created $sess";;
  exec) f=$4
        case "$f" in
          *colab_job.py) sleep "${FAKE_JOB_S:-0}"; echo "[s7job] RESULT random 2500.0 posts/s 42.5 tok/post url https://x.colab.dev/p?token=SECRETSECRET123";;
          *roundtrip_job.py) if [ "${FAKE_RT_FAIL:-0}" = 1 ]; then echo "[s7rt] ROUNDTRIP_FAILED x"; else echo "[s7rt] ROUNDTRIP_OK"; fi;;
          *) echo "mkdir ok";;
        esac;;
  upload) echo "uploaded";;
  download) : > "$4";;
  ls) [ "${FAKE_ADC:-1}" = 1 ] && echo "adc.json";;
  url) echo "https://colab.research.google.com/x?token=SECRETSECRET123";;
  stop) rm -f "$FAKE_DIR/active.$sess"; echo "stopped $sess";;
  *) echo "fake: unknown $cmd" >&2; exit 1;;
esac
"""

FAKE_GCLOUD = r"""#!/bin/bash
echo "$*" >> "$FAKE_DIR/gcloud.log"
if [ "$1 $2" = "storage cp" ]; then cat > /dev/null; touch "$FAKE_DIR/object"; exit 0; fi
if [ "$1 $2 $3" = "storage objects describe" ]; then [ -e "$FAKE_DIR/object" ] && { echo '{"size": "1000"}'; exit 0; }; exit 1; fi
exit 1
"""


def setup(tmp_path):
    fake = tmp_path / "fake"
    fake.mkdir()
    for name, body in [("colab", FAKE_COLAB), ("gcloud", FAKE_GCLOUD)]:
        p = fake / name
        p.write_text(body)
        p.chmod(0o755)
    run = tmp_path / "run"
    (run / "upload").mkdir(parents=True)
    for f in ["s7_random.parquet", "s7_slab.parquet", "parity_mac.npz"]:
        (run / "upload" / f).write_bytes(b"x" * 10)
    return fake, run


def drive(tmp_path, *args, **env):
    fake, run = setup(tmp_path)
    e = dict(os.environ, S7_RUN_DIR=str(run), S7_COLAB_BIN=str(fake / "colab"), S7_GCLOUD_BIN=str(fake / "gcloud"),
             S7_RT_BYTES="1000", S7_POLL_S="0.2", FAKE_DIR=str(fake), **env)
    p = subprocess.run([str(HERE / "run_colab.sh"), "--execute", "--auth-wait-min", "0.02", "--fallback-wait-min", "0.02", *args],
                       env=e, capture_output=True, text=True, timeout=300)
    calls = (fake / "calls.log").read_text().splitlines()
    return p, calls, run


pytestmark = pytest.mark.skipif(not SUMS.exists(), reason="needs the Mac's model SHA256SUMS")


def test_execute_primary_path(tmp_path):
    p, calls, run = drive(tmp_path)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    news = [c for c in calls if c.startswith("new ")]
    stops = [c for c in calls if c.startswith("stop ")]
    assert [c.split()[2] for c in news] == [c.split()[2] for c in stops]  # every created session stopped, in order
    assert all(c.split()[2].startswith("s7-") for c in stops) and not any("c5-distill" in c for c in stops)
    assert any("--gpu L4" in c for c in news) and any("--gpu A100" in c for c in news)
    assert (run / "colab-results/roundtrip_result.txt").read_text().strip() == "ok-primary"
    tsv = (run / "sessions.tsv").read_text()
    assert tsv.count("\tcreated") == 2 and tsv.count("\tstopped\n") == 2
    assert "SECRETSECRET123" not in p.stdout and "SECRETSECRET123" not in (run / "colab-results/colab_exec_L4.log").read_text()
    assert (run / "receipt.json").exists()
    assert (run / "rt_object.sha256").read_text().strip()  # the Mac recorded the test object's sha256


def test_execute_fallback_then_failed(tmp_path):
    p, calls, run = drive(tmp_path, FAKE_ADC="0")
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "FALLBACK (plan §5.1)" in p.stdout
    assert (run / "colab-results/roundtrip_result.txt").read_text().strip() == "failed"
    assert not any(c.startswith("exec") and "roundtrip_job.py" in c for c in calls)  # never ran without a credential


def test_execute_roundtrip_fails_then_fallback(tmp_path):
    p, calls, run = drive(tmp_path, FAKE_RT_FAIL="1")
    assert p.returncode == 0
    assert sum(1 for c in calls if c.startswith("exec") and "roundtrip_job.py" in c) == 2  # primary + fallback
    assert (run / "colab-results/roundtrip_result.txt").read_text().strip() == "failed"


def test_a100_unavailable_is_skipped(tmp_path):
    p, calls, run = drive(tmp_path, FAKE_FAIL_NEW="A100")
    assert p.returncode == 0, p.stderr[-2000:]
    assert "colab new failed for A100" in p.stdout
    assert [c.split()[2] for c in calls if c.startswith("stop ")] == [c.split()[2] for c in calls if c.startswith("new ") and "L4" in c]


def test_watchdog_stops_own_session(tmp_path):
    p, calls, run = drive(tmp_path, "--gpus", "L4", "--no-roundtrip", S7_TEST_BUDGET_S="1", FAKE_JOB_S="3")
    assert p.returncode == 0, p.stderr[-2000:]
    tsv = (run / "sessions.tsv").read_text()
    assert "stopped(watchdog 1s)" in tsv
    assert not any("c5-distill" in c for c in calls if c.startswith("stop "))
