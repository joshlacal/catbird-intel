"""S7g benchmark and S12m main run on Colab L4 sessions (GCE had no L4 capacity). PRINT-ONLY unless --execute.

  colab_driver.py bench  --run-id ID [--execute] [--cap 3] [--ramp] [--rps 50]
  colab_driver.py run    --run-id ID --units DIR [--execute] [--cap 250] [--sessions 1..3] [--batch-units 24]
                         [--rps 50 per session] [--allow-above-etiquette] [--soft-tokens 70] [--text-bs 64] [--mm-bs 8]
                         [--session-max-h 11.5] [--max-drops 3] [--max-sessions 0 per lane] [--max-batches 0]
  colab_driver.py status --run-id ID [--units DIR]
  common: [--gate DIR] [--run-dir DIR] [--gpu L4] [--max-label-lag-h 36]

Pattern (C5's run_colab.sh): one named session at a time, recorded in RUN_DIR/sessions.tsv when created; a
remote mkdir by `colab exec` before any `colab upload` (the contents API will not create directories);
`colab exec -f colab_job.py --timeout T --env K=V ...`; results back by `colab download`; on ANY exit (normal,
exception, SIGINT/SIGTERM/SIGHUP) the driver stops its own live session and only that one; `colab usage` and
`colab sessions` are captured before and after. No GCS, no Drive, no `colab auth`.

Uploads (from the Mac, AppleDouble-free: COPYFILE_DISABLE=1, bsdtar --no-mac-metadata, --exclude '._*', and the
tar listing is checked): code.tar.gz (graph/s12mm without tests + the s7bench/c4images/vmpass modules it
imports), gate.tar.gz (gate_manifest.json, excluded_dids.txt, labelled_uris.txt; gate_reasons.json stays on the
Mac), eg2_sha256.json (from ~/catbird-lake-local/models/SHA256SUMS), s12c_manifest.json (sha256 of each upload);
bench: bench.tar (S7's s7_slab/s7_random parquet + parity_mac.npz, mm_parity_mac.npz, the S7g image posts and
bench_producers.txt); run: one batch-<k>.tar per batch of input units (split_units.py) with its batch.json.

bench: one L4 session, setup exec then bench exec under a unit cap (default 3; the job's guard and the exec
timeout both derive from it); downloads bench_results.tar into RUN_DIR/results/.
run: --sessions N (1..3) parallel lanes, each with its own L4 session (own CLI state file, own sessions.tsv
lines, stopped by the exit trap), taking disjoint units from the shared Mac ledger by lease (colab_ledger.claim;
a dropped session's leases are released at once, a killed driver's expire). `colab new` runs one lane at a time;
if it fails for a lane after another lane has a session, that lane retires and the run continues with fewer.
The unit guard sums every session; each batch may spend an equal share of what is left under the cap. Each lane
loops until every unit of the manifest is in the Mac ledger (colab_ledger.py) or the run's unit cap would be
crossed: session -> setup -> batches (upload batch k+1 while batch k runs; download + verify batch k while k+1
runs) -> stop at --session-max-h or a drop -> next session resumes from the ledger. A batch that a drop
interrupts is simply redone. Stops and asks (exit 6) when the gate bundle is older than --max-label-lag-h, exits
5 on the job's CDN circuit breaker and 7 on a failed no-image-bytes audit, 4 when the cap is reached.

Test hooks (test_colab.py only): S12C_COLAB_BIN (a fake CLI), --rehearsal (passes S12C_REHEARSAL=1 and
S12C_REHEARSAL_ENV's KEY=VAL;... pairs to every exec; refused with the real CLI), S12C_NOW (fixed clock).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
GRAPH = HERE.parent
sys.path.insert(0, str(HERE))
import colab_ledger as CL  # noqa: E402

LAKE = pathlib.Path.home() / "catbird-lake-local"
REAL_COLAB = pathlib.Path.home() / ".local/bin/colab"
RATES = {"L4": 1.71, "A100": 5.4}
L4_TEXT_PPS, L4_MM70_PPS = 432.0, 31.3      # S7g s7g-colab-20261028a on a Colab L4
MAX_SESSIONS = 3
REMOTE = "/content/s12mm"
UPLOAD_CHUNK = int(os.environ.get("S12C_UPLOAD_CHUNK", 16 * 1024 * 1024))  # larger single uploads hit a contents-API 500 (2026-10-28)
REMOTE_ROOT_EXPR = ("os.path.join(os.environ.get('S12C_CONTENT', '/content') if os.environ.get('S12C_REHEARSAL') == '1'"
                    " else '/content', 's12mm')")
EXTRA_CODE = ["s7bench/s7lib.py", "s7bench/bench.py", "c4images/gate.py", "c4images/cdn.py", "vmpass/store.py"]
GATE_FILES = ("gate_manifest.json", "excluded_dids.txt", "labelled_uris.txt")
BENCH_INPUTS = {"s7_slab.parquet": "s7-bench-20261028/upload/s7_slab.parquet",
                "s7_random.parquet": "s7-bench-20261028/upload/s7_random.parquet",
                "parity_mac.npz": "s7-bench-20261028/upload/parity_mac.npz",
                "mm_parity_mac.npz": "s12mm-rehearsal-20261028/mm_parity_mac.npz",
                "image_posts/part-0.parquet": "s7g-input-20261028/out/image_posts/part-0.parquet",
                "bench_producers.txt": "s7g-input-20261028/out/bench_producers.txt"}
DEFAULT_GATE = LAKE / "s12mm-label-chain-s7g-20261028a/gate"
_URL = re.compile(r"(https?://[^\s?#'\"<>]+)\?[^\s'\"<>]*")
_TOKENS = [   # s7lib's patterns: OAuth tokens/codes and key=value secrets; sha256 hex and session names survive
    (re.compile(r"ya29\.[0-9A-Za-z._\-]+"), "ya29.<redacted>"),
    (re.compile(r"1//[0-9A-Za-z_\-]{20,}"), "1//<redacted>"),
    (re.compile(r"\b4/[0-9A-Za-z_\-]{20,}"), "4/<redacted>"),
    (re.compile(r"(?i)\b(access_token|refresh_token|id_token|token|code|key|sig|signature|authorization)"
                r"(\"?\s*[:=]\s*\"?)(bearer\s+)?[^\s\"'&,}]{8,}"), r"\1\2<redacted>"),
]
EXIT_CAP, EXIT_CIRCUIT, EXIT_STALE, EXIT_AUDIT, EXIT_SESSION, EXIT_JOB = 4, 5, 6, 7, 8, 9
FIXED_CLOCK = {"on": False}     # S12C_NOW is honoured only under --rehearsal


def redact(s):
    s = _URL.sub(r"\1?<redacted>", s or "")
    for rx, rep in _TOKENS:
        s = rx.sub(rep, s)
    return s


def utcnow():
    v = os.environ.get("S12C_NOW") if FIXED_CLOCK["on"] else None
    return datetime.fromisoformat(v.replace("Z", "+00:00")) if v else datetime.now(timezone.utc)


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Stop(Exception):
    def __init__(self, code, why):
        super().__init__(why)
        self.code, self.why = code, why


# --- Colab CLI ------------------------------------------------------------------------------------------------

class Colab:
    """The colab CLI with one --config state file. The CLI's own docs say to isolate parallel runs by state file,
    so every session gets its own (RUN_DIR/colab-state/<session>.json, derivable from the name for a later
    reconcile); the driver's account-level calls (usage, sessions receipts) use RUN_DIR/colab-cli-state.json."""

    def __init__(self, a, led, cfg=None, log=None, lock=None):
        self.a = a
        self.bin = os.environ.get("S12C_COLAB_BIN", str(REAL_COLAB))
        self.cfg = str(cfg or pathlib.Path(a.run_dir) / "colab-cli-state.json")
        self.execute, self.led = a.execute, led
        self.log = log or open(led.dir / "logs" / f"driver-{time.strftime('%Y%m%dT%H%M%S')}.log", "a")
        self.lock = lock or threading.Lock()

    def for_session(self, name):
        d = pathlib.Path(self.a.run_dir) / "colab-state"
        d.mkdir(exist_ok=True)
        return Colab(self.a, self.led, d / f"{name}.json", self.log, self.lock)

    def __call__(self, *args, timeout=None, tee=None):
        cmd = [self.bin, "--config", self.cfg, *map(str, args)]
        line = "+ " + shlex.join(cmd)
        with self.lock:
            print(redact(line), flush=True)
            self.log.write(redact(line) + "\n")
        if not self.execute:
            return 0, ""
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            rc, out = r.returncode, redact(r.stdout + r.stderr)
        except subprocess.TimeoutExpired as e:
            rc, out = 124, redact(((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""))
                                  + f"\n[driver] local timeout after {timeout}s")
        with self.lock:
            self.log.write(out + "\n")
            self.log.flush()
        if tee:
            pathlib.Path(tee).write_text(out)
        return rc, out


_SEQ = itertools.count(1)       # session names stay unique when a lane replaces a session within the same second


class Session:
    """One named Colab session. Stopped on any exit by the driver's handler; only sessions this object created."""

    def __init__(self, drv, mode, lane=0):
        self.drv, self.led, self.lane = drv, drv.led, lane
        self.gpu = drv.a.gpu
        self.name = (f"s12c-{mode}-{re.sub(r'[^a-z0-9]+', '-', drv.a.run_id.lower())}"
                     f"{f'-l{lane}' if mode == 'run' else ''}-{time.strftime('%Y%m%dT%H%M%S')}-{next(_SEQ)}")
        self.c = drv.colab.for_session(self.name)
        self.live, self.t0, self.rate = False, None, RATES[self.gpu]
        self._lk = threading.Lock()

    def create(self):
        _rc, u0 = self.c("usage")
        rc, out = self.c("new", "-s", self.name, "--gpu", self.gpu, timeout=900)
        if rc != 0:
            raise Stop(EXIT_SESSION, f"colab new failed for {self.gpu}: {out[-400:]}")
        self.t0, self.live = time.time(), True
        if self.c.execute:
            self.led.record(self.name, "created", self.gpu)
        _rc, u1 = self.c("usage")
        b0, r0, r1 = CL.parse_usage(u0)["balance"], CL.parse_usage(u0)["rate_per_hour"], CL.parse_usage(u1)["rate_per_hour"]
        live = max(0.0, (r1 or 0) - (r0 or 0))
        self.rate = max(RATES[self.gpu], live)   # never below the table rate
        if self.c.execute:
            self.led.spend_update(self.name, gpu=self.gpu, rate=self.rate, t_created=self.t0, t_last_seen=self.t0,
                                  balance_before=b0, measured_rate_delta=live)
        self.drv.say(f"session {self.name}: {self.gpu} at {self.rate} units/h (measured delta {live})")
        return self

    def heartbeat(self):
        if self.live and self.c.execute:
            self.led.spend_update(self.name, t_last_seen=time.time())

    def alive(self):
        if not self.c.execute:
            return True
        rc, out = self.c("sessions", timeout=120)
        return rc == 0 and self.name in out

    def stop(self, why="stopped"):
        with self._lk:                     # a lane and the exit trap may both try
            if not self.live:
                return
            self.live = False
        self.c("stop", "-s", self.name, timeout=300)
        if self.c.execute:
            _rc, u = self.c("usage")
            self.led.record(self.name, why, "-")
            self.led.spend_update(self.name, t_stopped=time.time(), ended=why, balance_after=CL.parse_usage(u)["balance"])

    def dropped(self):
        with self._lk:
            self.live = False
        if self.c.execute:
            self.led.record(self.name, "dropped", "-")
            self.led.spend_update(self.name, t_stopped=time.time(), ended="dropped")

    def age_h(self):
        return (time.time() - self.t0) / 3600 if self.t0 else 0.0

    def exec_job(self, env, timeout, tee=None):
        envs = []
        for k, v in {**env, **self.drv.rehearsal_env()}.items():
            envs += ["--env", f"{k}={v}"]
        hb = threading.Event()

        def beat():
            while not hb.wait(60):
                self.heartbeat()
        threading.Thread(target=beat, daemon=True).start()
        try:
            return self.c("exec", "-s", self.name, "-f", HERE / "colab_job.py", "--timeout", int(timeout), *envs,
                          timeout=timeout + 300, tee=tee)
        finally:
            hb.set()
            self.heartbeat()


# --- staging --------------------------------------------------------------------------------------------------

def build_tar(dst, root, members, gz=True):
    """bsdtar with no macOS metadata; refuses if any AppleDouble member slips in."""
    env = {**os.environ, "COPYFILE_DISABLE": "1"}
    flags = ["--no-mac-metadata"] if sys.platform == "darwin" else []
    cmd = ["tar", *flags, "--exclude", "._*", "--exclude", "__pycache__", "--exclude", "*.pyc",
           "-C", str(root), "-czf" if gz else "-cf", str(dst), *members]
    subprocess.run(cmd, check=True, env=env)
    with tarfile.open(dst) as tf:
        bad = [m.name for m in tf.getmembers() if os.path.basename(m.name).startswith("._")]
    if bad:
        raise Stop(2, f"AppleDouble members in {dst}: {bad[:5]}")
    return dst


def code_members():
    mods = sorted(p.name for p in HERE.glob("*.py") if not p.name.startswith(("test_", "._")))
    return [f"s12mm/{m}" for m in mods] + EXTRA_CODE


def compile_check(stage):
    """Every shipped module must compile under Python 3.11 and 3.12 (Colab ships one of them)."""
    out = {}
    for v in ("3.11", "3.12"):
        exe = next((p for p in (pathlib.Path.home() / f".local/bin/python{v}", pathlib.Path(f"/opt/homebrew/bin/python{v}"))
                    if p.exists()), None)
        if exe is None:
            out[v] = "interpreter not found"
            continue
        files = [str(GRAPH / m) for m in code_members()] + [str(HERE / "colab_job.py")]
        r = subprocess.run([str(exe), "-I", "-c", "import sys\nfor f in sys.argv[1:]: compile(open(f).read(), f, 'exec')",
                            *files], capture_output=True, text=True)
        out[v] = "ok" if r.returncode == 0 else r.stderr[-800:]
    if any(v != "ok" and v != "interpreter not found" for v in out.values()):
        raise Stop(2, f"compile check failed: {out}")
    return out


def eg2_sha(models):
    want = {}
    for line in open(models / "SHA256SUMS"):
        h, p = line.split(None, 1)
        p = p.strip()
        if p.startswith("./embeddinggemma-2/"):
            want[p[len("./embeddinggemma-2/"):]] = h
    tmpl = models / "embeddinggemma-2/chat_template.jinja"
    if tmpl.exists() and "chat_template.jinja" not in want:
        want["chat_template.jinja"] = CL.sha256_file(tmpl)
    if "model.safetensors" not in want:
        raise Stop(2, "SHA256SUMS has no ./embeddinggemma-2/model.safetensors")
    return want


def gate_info(gate_dir):
    man = json.loads((pathlib.Path(gate_dir) / "gate_manifest.json").read_text())
    for n in ("excluded_dids.txt", "labelled_uris.txt"):
        if CL.sha256_file(pathlib.Path(gate_dir) / n) != man["sha256"][n]:
            raise Stop(2, f"gate bundle {gate_dir}: {n} does not match its manifest")
    wm = datetime.fromisoformat(man["label_watermark_utc"].replace("Z", "+00:00"))
    return wm, man


# --- the driver -----------------------------------------------------------------------------------------------

class Driver:
    def __init__(self, a):
        self.a = a
        self.led = CL.RunLedger(a.run_dir)
        self.colab = Colab(a, self.led)
        self.session = None                      # bench's one session
        self.live_sessions = set()               # run: every live session across lanes (the exit trap stops these)
        self.lock, self.create_lock = threading.Lock(), threading.Lock()
        self.halt = threading.Event()
        self.fatal = None
        self.lane_end = {}
        self.stage = self.led.dir / "stage"
        self.stage.mkdir(exist_ok=True)

    def say(self, *x):
        msg = " ".join(str(v) for v in x)
        print(f"== {msg}", flush=True)
        self.colab.log.write(f"== {msg}\n")

    def rehearsal_env(self):
        if not self.a.rehearsal:
            return {}
        env = {"S12C_REHEARSAL": "1"}
        for kv in os.environ.get("S12C_REHEARSAL_ENV", "").split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k] = v
        return env

    # receipts ------------------------------------------------------------------------------------------------
    def receipts(self, when):
        d = self.led.dir / "receipts"
        d.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        self.colab("usage", tee=d / f"usage_{when}_{stamp}.txt" if self.a.execute else None)
        self.colab("sessions", tee=d / f"sessions_{when}_{stamp}.txt" if self.a.execute else None)

    def balance_now(self):
        if not self.a.execute:
            return None
        _rc, out = self.colab("usage")
        return CL.parse_usage(out)["balance"]

    def check_gate(self, extra_s=0.0):
        wm, _man = gate_info(self.a.gate)
        age_h = (utcnow() - wm).total_seconds() / 3600
        if age_h < 0 or age_h > self.a.max_label_lag_h:
            raise Stop(EXIT_STALE, f"gate bundle watermark {wm:%Y-%m-%dT%H:%M:%SZ} is {age_h:.1f} h old (limit "
                                   f"{self.a.max_label_lag_h} h): refresh it (label_chain.sh -> gate_bundle.py) and rerun "
                                   f"with --gate <new dir>; finished units are kept in the ledger")
        return age_h

    # common staging + session setup --------------------------------------------------------------------------
    def stage_common(self, extra):
        """Builds code.tar.gz, gate.tar.gz, eg2_sha256.json (+ extra {name: path}) and the sha256 manifest."""
        self.say("compile check (3.11, 3.12):", compile_check(self.stage))
        build_tar(self.stage / "code.tar.gz", GRAPH, code_members())
        build_tar(self.stage / "gate.tar.gz", self.a.gate, list(GATE_FILES))
        (self.stage / "eg2_sha256.json").write_text(json.dumps(eg2_sha(pathlib.Path(self.a.models)), indent=1, sort_keys=True))
        files = {"code.tar.gz": self.stage / "code.tar.gz", "gate.tar.gz": self.stage / "gate.tar.gz",
                 "eg2_sha256.json": self.stage / "eg2_sha256.json", **extra}
        man = {"files": {k: CL.sha256_file(v) for k, v in files.items()}, "run_id": self.a.run_id, "created": iso(time.time())}
        (self.stage / "s12c_manifest.json").write_text(json.dumps(man, indent=1))
        files["s12c_manifest.json"] = self.stage / "s12c_manifest.json"
        self.say("uploads:", {k: f"{pathlib.Path(v).stat().st_size / 1e6:.1f} MB" for k, v in files.items()})
        return files

    def mkdirs(self, s):
        p = self.stage / f"_mkdir-{s.name}.py"
        p.write_text("import os\nfor d in ['in', 'batches', 'dl', 'out', 'work', 'in/bench']:\n"
                     "    os.makedirs(os.path.join(os.environ.get('S12C_CONTENT', '/content') if os.environ.get('S12C_REHEARSAL') == '1'"
                     " else '/content', 's12mm', d), exist_ok=True)\nprint('mkdir ok')\n")
        envs = []
        for k, v in self.rehearsal_env().items():
            envs += ["--env", f"{k}={v}"]
        rc, out = s.c("exec", "-s", s.name, "-f", p, "--timeout", 120, *envs, timeout=400)
        if self.a.execute and (rc != 0 or "mkdir ok" not in out):
            raise Stop(EXIT_SESSION, f"remote mkdir failed: {out[-300:]}")

    def upload(self, s, local, remote):
        # The contents API answers 500 to large single uploads (an 81.5 MB gate.tar.gz failed on 2026-10-28;
        # 19 MB files went through), so anything over UPLOAD_CHUNK goes up in parts; the job joins them.
        local = pathlib.Path(local)
        if local.stat().st_size <= UPLOAD_CHUNK:
            self._upload_one(s, local, remote)
            return
        parts_dir = self.stage / "_parts" / s.name / remote.replace("/", "__")
        parts_dir.mkdir(parents=True, exist_ok=True)
        names = []
        with open(local, "rb") as f:
            for i in itertools.count():
                buf = f.read(UPLOAD_CHUNK)
                if not buf:
                    break
                p = parts_dir / f"part.{i:04d}"
                p.write_bytes(buf)
                names.append(p)
        for p in names:
            self._upload_one(s, p, f"{remote}.{p.name}")
        # colab_job.py joins the parts (and checks sha256) when its mode first reads the file: a separate
        # `colab exec` here would queue behind a running batch on the session's single kernel.
        for p in names:
            p.unlink()

    def _upload_one(self, s, local, remote):
        rc, out = s.c("upload", "-s", s.name, local, f"{REMOTE}/{remote}", timeout=3600)
        if self.a.execute and rc != 0:
            raise Stop(EXIT_SESSION, f"upload {remote} failed: {out[-300:]}")

    def setup_session(self, s, files):
        self.mkdirs(s)
        for name, p in files.items():
            self.upload(s, p, name)
        rc, out = s.exec_job({"S12C_MODE": "setup"}, 2400, tee=self.led.dir / "logs" / f"{s.name}-setup.log")
        if self.a.execute and "SETUP_OK" not in out:
            why = "\n".join(ln for ln in out.splitlines() if "[s12c]" in ln)[-1500:]
            # a reported failure (sha256, CUDA, swap, compile) is deterministic: stop; no report = the session went away
            raise Stop(EXIT_JOB if "SETUP_FAILED" in out or "[s12c] ERROR" in out else EXIT_SESSION, "setup failed: " + why)

    def check_rehearsal(self):
        if self.a.rehearsal and os.environ.get("S12C_COLAB_BIN", str(REAL_COLAB)) == str(REAL_COLAB):
            raise Stop(2, "--rehearsal is only for the fake colab CLI (S12C_COLAB_BIN)")

    # bench ---------------------------------------------------------------------------------------------------
    def bench(self):
        a = self.a
        self.check_rehearsal()
        if a.cap > 8:
            raise Stop(2, "bench cap above 8 units")
        age = self.check_gate()
        self.say(f"S7g bench {a.run_id}: gpu {a.gpu}, cap {a.cap} units, rps {a.rps}, ramp {a.ramp or 'off'}, gate age {age:.1f} h")
        bdir = self.stage / "bench"
        missing = [k for k, rel in BENCH_INPUTS.items() if not (pathlib.Path(a.lake) / rel).exists()]
        if missing:
            raise Stop(2, f"missing bench inputs: {missing}")
        for k, rel in BENCH_INPUTS.items():
            dst = bdir / k
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((pathlib.Path(a.lake) / rel).read_bytes())
        build_tar(self.stage / "bench.tar", bdir, list(BENCH_INPUTS), gz=False)
        files = self.stage_common({"bench.tar": self.stage / "bench.tar"})
        self.receipts("before")
        s = self.session = Session(self, "bench").create()
        t_budget = a.cap / s.rate * 3600
        self.setup_session(s, files)
        left = t_budget - (time.time() - s.t0) if a.execute else t_budget
        cap_left = a.cap - (time.time() - s.t0) / 3600 * s.rate if a.execute else a.cap
        env = {"S12C_MODE": "bench", "S12C_RUN_ID": a.run_id, "S12C_CAP": f"{max(0.0, cap_left - 0.1):.3f}",
               "S12C_RATE": s.rate, "S12C_RPS": a.rps, "S12C_TIMEOUT": int(max(600, left)),
               "S12C_SESSION_T0": f"{s.t0 or time.time():.0f}", "S12C_N_TEXT": a.n_text, "S12C_N_IMAGE_POSTS": a.n_image_posts}
        if a.ramp:
            env["S12C_RAMP"] = a.ramp
        rc, out = s.exec_job(env, max(600, left), tee=self.led.dir / "logs" / f"{s.name}-bench.log")
        res = self.led.dir / "results"
        res.mkdir(exist_ok=True)
        m = re.search(r"\[s12c\] BENCH RC (-?\d+) TAR (\S+) (\d+) ([0-9a-f]{64})", out)
        s.c("download", "-s", s.name, f"{REMOTE}/dl/bench_results.tar", res / "bench_results.tar", timeout=3600)
        verdict = {"exec_rc": rc, "job_rc": int(m.group(1)) if m else None}
        if a.execute and m:
            got = CL.sha256_file(res / "bench_results.tar") if (res / "bench_results.tar").exists() else None
            verdict["tar_sha256_ok"] = got == m.group(4)
            if got == m.group(4):
                with tarfile.open(res / "bench_results.tar") as tf:
                    tf.extractall(res, filter="data")
        s.stop()
        self.session = None
        self.receipts("after")
        (res / "driver_receipt.json").write_text(json.dumps({**verdict, "session": s.name, "rate": s.rate,
                                                             "units_est": round(self.led.spent_units()[0], 3)}, indent=1))
        self.say("bench done:", verdict, "results in", res)
        return 0 if verdict.get("job_rc") == 0 or not a.execute else 1

    # run -----------------------------------------------------------------------------------------------------
    def load_manifest(self):
        man = json.loads((pathlib.Path(self.a.units) / "manifest.json").read_text())
        if man.get("format") != "s12mm-colab-units/1":
            raise Stop(2, f"unknown units format {man.get('format')}")
        if man["unit_rows"] > 20_000:
            raise Stop(2, "units larger than 20k rows")
        return man

    def unit_secs_estimate(self, man):
        """Seconds per unit in one session: measured wall from the ledger once there is some, else the S7g L4
        measurements (text 432 posts/s at batch 64, mm70 31.3 posts/s at batch 8; the fetch of unit k+1 overlaps
        the GPU work of unit k, so a unit costs the larger of the two)."""
        walls = [r["vm"].get("wall_s") for r in self.led.done().values() if isinstance(r.get("vm"), dict) and r["vm"].get("wall_s")]
        if len(walls) >= 3:
            return sum(walls[-20:]) / len(walls[-20:]) * 1.15
        t = man["totals"]
        per_posts = t["posts"] / max(1, t["units"])
        imgs = t["images"] / max(1, t["units"])
        return max(imgs / float(self.a.rps), per_posts / L4_TEXT_PPS + t["image_posts"] / max(1, t["units"]) / L4_MM70_PPS) * 1.15 + 10

    def make_batch(self, s, k, units):
        name = f"l{s.lane}-{s.name[-15:]}-b{k:03d}"
        d = self.stage / "batches" / name
        (d / "units").mkdir(parents=True, exist_ok=True)
        for u in units:
            src = pathlib.Path(self.a.units) / u["file"]
            if CL.sha256_file(src) != u["sha256"]:
                raise Stop(2, f"input unit {u['unit']} does not match the manifest")
            dst = d / "units" / f"{u['unit']}.parquet"
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    import shutil
                    shutil.copyfile(src, dst)
        (d / "batch.json").write_text(json.dumps({"batch": name, "unit_rows": self.man["unit_rows"], "units": units}, indent=1))
        tar = build_tar(self.stage / "batches" / f"{name}.tar", d, ["batch.json", "units"], gz=False)
        return name, tar, CL.sha256_file(tar), units

    def fetch_batch(self, s, name, units, line):
        """Download + verify one finished batch; returns (verified units, job rc, receipt dicts)."""
        m = re.search(rf"\[s12c\] BATCH {re.escape(name)} RC (-?\d+) UNITS (\d+) TAR (\S+) (\d+) (\S+)", line or "")
        if not m:
            return [], None, []
        rc = int(m.group(1))
        if m.group(3) == "-":
            return [], rc, []
        inc = self.led.dir / "incoming" / name
        tar = self.led.dir / "incoming" / f"{name}.tar"
        r, _out = s.c("download", "-s", s.name, f"{REMOTE}/dl/{name}.tar", tar, timeout=3600)
        if not self.a.execute:
            return [], rc, []
        if r != 0 or not tar.exists() or CL.sha256_file(tar) != m.group(5):
            self.say(f"batch {name}: download failed or sha256 mismatch; its units stay to do")
            return [], rc, []
        with tarfile.open(tar) as tf:
            tf.extractall(inc, filter="data")
        import mmlib as L
        ok, recs = [], []
        by_unit = {u["unit"]: u for u in units}
        for f in sorted((inc / "_ledger").glob("u-*.json")) if (inc / "_ledger").exists() else []:
            vm_rec = json.loads(f.read_text())
            unit = vm_rec["unit"].rsplit("-", 1)[0]          # job unit "<file stem>-00000"
            if unit not in by_unit:
                self.say(f"batch {name}: unexpected unit {vm_rec['unit']}; ignored")
                continue
            good, why = self.led.verify_unit(by_unit[unit], vm_rec, inc / "units" / f"{f.stem}.parquet",
                                             audit=L.audit_output_table)
            if good:
                ok.append(unit)
            else:
                self.say(f"unit {unit}: not marked ({why})")
        for f in sorted((inc / "receipt").glob("*.json")) if (inc / "receipt").exists() else []:
            rr = json.loads(f.read_text())
            recs.append(rr)
            (self.led.dir / "vectors" / "receipt" / f"{name}-{f.name}").write_text(f.read_text())
        tar.unlink()
        return ok, rc, recs

    def run(self):
        a = self.a
        self.check_rehearsal()
        self.man = man = self.load_manifest()
        self.reconcile()
        age = self.check_gate()
        todo = self.led.todo(man)
        per_unit = self.unit_secs_estimate(man)
        spent, det = self.led.spent_units(balance_now=self.balance_now())
        rate = RATES[a.gpu]
        need = len(todo) * per_unit / 3600 * rate
        self.say(f"S12m {a.run_id}: {len(todo)} of {man['totals']['units']} units to do, ~{per_unit:.0f} s/unit/session "
                 f"at {a.rps} req/s per session, {a.sessions} parallel session(s) -> ~{need:.1f} units more; spent {det}; "
                 f"cap {a.cap}; gate age {age:.1f} h")
        if spent + need > a.cap:
            self.say(f"WARNING: projected {spent + need:.1f} units exceeds the cap {a.cap}; the guard will stop at the cap")
        if not todo:
            self.say("nothing to do: every unit is in the ledger")
            return 0
        self.files = self.stage_common({})
        self.receipts("before")
        self.per_unit = per_unit
        lanes = [threading.Thread(target=self.lane, args=(i,), name=f"lane{i}", daemon=True) for i in range(a.sessions)]
        for i, t in enumerate(lanes):
            t.start()
            if i + 1 < len(lanes):
                self.halt.wait(0.2 if a.rehearsal else 30)    # stagger session creation and setup downloads
        try:
            while any(t.is_alive() for t in lanes):
                for t in lanes:
                    t.join(0.5)
        finally:
            self.halt.set()
            for t in lanes:
                t.join(5)
            for s in list(self.live_sessions):
                s.stop("stopped(trap)")
            self.receipts("after")
        done = self.led.done()
        spent, det = self.led.spent_units(balance_now=self.balance_now())
        self.say(f"ledger: {len(done)} of {man['totals']['units']} units done; spend {det}; lanes {self.lane_end}")
        if self.fatal:
            self.say(f"STOP ({self.fatal.code}): {self.fatal.why}")
            return self.fatal.code
        if self.led.todo(man) and a.execute and not a.max_sessions:
            return EXIT_SESSION      # every lane ended (no session could be had) with units left
        return 0

    def fail(self, e):
        with self.lock:
            if self.fatal is None:
                self.fatal = e
        self.halt.set()

    def lane(self, i):
        """One parallel lane: sessions one after another, each taking disjoint leased units from the shared ledger."""
        a, man = self.a, self.man
        drops, sessions = 0, 0
        try:
            while self.led.unclaimed_todo(man) and not self.halt.is_set():
                if a.max_sessions and sessions >= a.max_sessions:
                    self.lane_end[i] = "max-sessions"
                    return
                spent, det = self.led.spent_units(balance_now=self.balance_now())
                if spent + self.per_unit / 3600 * RATES[a.gpu] * self.n_live() + 0.5 > a.cap:
                    raise Stop(EXIT_CAP, f"unit cap: spent {det} of {a.cap}; another session would cross it")
                self.check_gate()
                with self.create_lock:              # one `colab new` at a time: the rate delta stays attributable
                    if self.halt.is_set():
                        return
                    s = Session(self, "run", lane=i)
                    try:
                        s.create()
                    except Stop as e:
                        if i == 0 and sessions == 0 and not self.live_sessions:
                            raise
                        self.lane_end[i] = f"degraded: {e.why[:200]}"
                        self.say(f"lane {i}: no further session ({e.why[:200]}); continuing with fewer sessions")
                        return
                    self.live_sessions.add(s)
                sessions += 1
                end = "done"
                try:
                    self.setup_session(s, self.files)
                    end = self.session_batches(s)
                except Stop as e:
                    if e.code != EXIT_SESSION:
                        raise
                    end = "drop:" + e.why
                finally:
                    self.led.release(s.name)             # anything still leased goes back to the pool now
                if end.startswith("drop"):
                    drops += 1
                    if s.alive():
                        s.stop("stopped(after-failure)")
                    else:
                        s.dropped()
                    self.say(f"lane {i}: session {s.name} lost ({end}); drops {drops}/{a.max_drops}")
                    if drops >= a.max_drops:
                        raise Stop(EXIT_SESSION, f"lane {i}: {drops} session drops in a row")
                else:
                    drops = 0
                    s.stop()
                self.live_sessions.discard(s)
                if end.startswith("cap"):
                    raise Stop(EXIT_CAP, end)
                if not a.execute:
                    break
            self.lane_end.setdefault(i, "done" if not self.led.todo(man) else "halted" if self.halt.is_set() else "pool empty")
        except Stop as e:
            self.lane_end[i] = f"stop {e.code}"
            self.fail(e)
        except Exception as e:  # a bug in one lane stops every lane cleanly
            self.lane_end[i] = f"error {type(e).__name__}"
            self.fail(Stop(EXIT_JOB, f"lane {i}: {type(e).__name__}: {e}"))
        finally:
            for s in [x for x in list(self.live_sessions) if x.lane == i]:
                s.stop("stopped(trap)")
                self.live_sessions.discard(s)

    def n_live(self):
        return max(1, len(self.live_sessions))

    def batch_share(self, s):
        """Units this session's next batch may spend: an equal share of what is left under the cap."""
        spent, _ = self.led.spent_units()
        return (self.a.cap - spent - 0.3) / self.n_live()

    def session_batches(self, s):
        """Batches until the session ends. Returns 'done', 'session-max', 'batch-limit', 'halt', 'cap: ...' or
        'drop: ...'. Units are leased per batch; a lease lasts the batch's exec timeout + 30 min and is renewed
        when the batch finishes, until its download is verified."""
        a, man, per_unit = self.a, self.man, self.per_unit
        pool = ThreadPoolExecutor(2, thread_name_prefix=f"xfer-{s.lane}")
        st = {"k": 0, "confirmed": [], "empty": 0}
        pending_dl = None
        result = "done"
        unit_cost = per_unit / 3600 * s.rate

        def lease_s(n):
            return int(n * per_unit * 3 + 900 + 1800)

        def plan_batch():
            if a.max_batches and st["k"] >= a.max_batches:
                return None
            n = min(a.batch_units, int(self.batch_share(s) / unit_cost))
            if n <= 0:
                return None
            got = self.led.claim(s.name, man, n, lease_s(n))
            if not got:
                return None
            st["k"] += 1
            return self.make_batch(s, st["k"], got)

        def upload(b):
            self.upload(s, b[1], f"batches/{b[0]}.tar")
            return b

        try:
            b = plan_batch()
            if b is None:
                return "cap: no unit fits under the cap" if (self.led.unclaimed_todo(man) and
                                                             not a.max_batches) else "done"
            cur = pool.submit(upload, b)
            while cur is not None:
                try:
                    name, _tar, sha, units = cur.result()
                except Stop as e:
                    if pending_dl is not None:       # the batch that just finished comes home before the drop
                        self.finish_download(s, st, *pending_dl)
                        pending_dl = None
                    return "drop: " + e.why
                if self.halt.is_set():
                    result = "halt"
                    break
                if s.age_h() + len(units) * per_unit / 3600 > a.session_max_h:
                    result = "session-max"
                    break
                self.check_gate()
                share = self.batch_share(s)
                timeout = int(min(len(units) * per_unit * 3 + 900, max(600, share / s.rate * 3600)))
                nxt_b = plan_batch()
                nxt = pool.submit(upload, nxt_b) if nxt_b else None
                env = {"S12C_MODE": "batch", "S12C_RUN_ID": a.run_id, "S12C_BATCH": name, "S12C_BATCH_SHA256": sha,
                       "S12C_CAP": f"{max(0.0, share):.3f}", "S12C_RATE": s.rate, "S12C_RPS": a.rps,
                       "S12C_SOFT": a.soft_tokens, "S12C_TIMEOUT": timeout, "S12C_CLEAN": ",".join(st["confirmed"][-4:]),
                       "S12C_MAX_LAG_H": a.max_label_lag_h, "S12C_TEXT_BS_RUN": a.text_bs, "S12C_MM_BS_RUN": a.mm_bs}
                if a.allow_above_etiquette:
                    env["S12C_ALLOW_ABOVE"] = "1"
                self.say(f"lane {s.lane} batch {name}: {len(units)} units, share {share:.2f} units, exec timeout {timeout}s")
                rc, out = s.exec_job(env, timeout, tee=self.led.dir / "logs" / f"{name}.log")
                if pending_dl is not None:
                    self.finish_download(s, st, *pending_dl)
                    pending_dl = None
                line = next((ln for ln in out.splitlines() if f"BATCH {name} RC" in ln), None)
                if not a.execute:
                    break
                if line is None and "[s12c] ERROR" in out and s.alive():
                    raise Stop(EXIT_JOB, "colab_job error in batch " + name + ": " +
                               "\n".join(ln for ln in out.splitlines() if "[s12c]" in ln)[-1200:])
                if line is None:
                    if nxt is not None:
                        _wait_quiet(nxt)
                    return f"drop: exec rc {rc} without a batch result"
                self.led.renew(s.name, [u["unit"] for u in units], 3600)   # held until verified on the Mac
                job_rc = int(re.search(r"RC (-?\d+)", line).group(1))
                pending_dl = (pool.submit(self.fetch_batch, s, name, units, line), name, units)
                if job_rc != 0:
                    self.finish_download(s, st, *pending_dl)
                    pending_dl = None
                    if nxt is not None:
                        _wait_quiet(nxt)
                    codes = {EXIT_CIRCUIT: (EXIT_CIRCUIT, "the job's CDN circuit breaker tripped (429s/errors): stop and look"),
                             EXIT_STALE: (EXIT_STALE, "the job found the gate bundle stale: refresh it and rerun"),
                             EXIT_AUDIT: (EXIT_AUDIT, "the no-image-bytes audit failed: stop and look")}
                    raise Stop(*codes.get(job_rc, (EXIT_SESSION, f"job exit {job_rc} in batch {name}")))
                if nxt is None:
                    if not self.led.unclaimed_todo(man):
                        result = "done"
                    elif a.max_batches and st["k"] >= a.max_batches:
                        result = "batch-limit"
                    else:
                        result = "cap: next batch would cross the cap"
                    break
                cur = nxt
            if pending_dl is not None:
                self.finish_download(s, st, *pending_dl)
            return result
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def finish_download(self, s, st, fut, name, units):
        try:
            ok, _rc, _recs = fut.result()
        except Exception as e:  # a dropped session mid-download: the units go back to the pool
            self.say(f"batch {name}: download failed ({type(e).__name__}: {e})")
            ok = []
        self.led.release(s.name, [u["unit"] for u in units])
        if ok:
            st["confirmed"].append(name)
            st["empty"] = 0
        else:
            st["empty"] += 1
        self.say(f"batch {name}: {len(ok)} of {len(units)} units verified and in the ledger")
        if st["empty"] >= 3:
            raise Stop(EXIT_JOB, "3 batches in a row brought back no verified unit: stop and look")

    def reconcile(self):
        """Sessions a crashed earlier driver left open: stop them if Colab still lists them (they are ours, recorded
        in sessions.tsv), else close their spend at last-seen + grace. Their leases are released either way."""
        opened = self.led.open_sessions()
        for name in opened:
            c = self.colab.for_session(name)
            listing = ""
            if self.a.execute:
                _rc, listing = c("sessions", timeout=120)
            if name in listing:
                c("stop", "-s", name, timeout=300)
                if self.a.execute:
                    self.led.record(name, "stopped(reconcile)", "-")
                    self.led.spend_update(name, t_stopped=time.time(), ended="stopped(reconcile)")
            elif self.a.execute:
                self.led.record(name, "ended(unobserved)", "-")
                self.led.spend_update(name, closed_unobserved=True, ended="unobserved")
            self.led.release(name)
            self.say(f"reconciled open session {name}")

    def status(self):
        d = self.led.done()
        out = {"run_dir": str(self.led.dir), "done_units": len(d), "spend": self.led.spent_units()[1],
               "sessions": self.led.sessions(), "open_sessions": self.led.open_sessions()}
        if self.a.units:
            man = self.load_manifest()
            out.update(total_units=man["totals"]["units"], todo_units=len(self.led.todo(man)),
                       done_posts=sum(r["rows"] for r in d.values()), total_posts=man["totals"]["posts"])
        try:
            wm, _m = gate_info(self.a.gate)
            out["gate_age_h"] = round((utcnow() - wm).total_seconds() / 3600, 1)
        except (OSError, KeyError, Stop):
            out["gate_age_h"] = None
        print(json.dumps(out, indent=1))
        return 0


def _wait_quiet(fut):
    try:
        fut.result()
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["bench", "run", "status"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--run-dir")
    ap.add_argument("--gate", default=str(DEFAULT_GATE))
    ap.add_argument("--lake", default=str(LAKE))
    ap.add_argument("--models", default=str(LAKE / "models"))
    ap.add_argument("--gpu", default="L4", choices=sorted(RATES))
    ap.add_argument("--cap", type=float)
    ap.add_argument("--rps", type=float, default=50.0)
    ap.add_argument("--allow-above-etiquette", action="store_true")
    ap.add_argument("--max-label-lag-h", type=float, default=36.0)
    ap.add_argument("--ramp", nargs="?", const="25,50,100,200", default="")
    ap.add_argument("--n-text", type=int, default=10_000)
    ap.add_argument("--n-image-posts", type=int, default=10_000)
    ap.add_argument("--units")
    ap.add_argument("--batch-units", type=int, default=24)
    ap.add_argument("--soft-tokens", type=int, default=70)
    ap.add_argument("--session-max-h", type=float, default=11.5)
    ap.add_argument("--max-drops", type=int, default=3)
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=0, help="per session (0: until --session-max-h)")
    ap.add_argument("--sessions", type=int, default=1, help=f"parallel L4 sessions, 1..{MAX_SESSIONS}")
    ap.add_argument("--text-bs", type=int, default=64)
    ap.add_argument("--mm-bs", type=int, default=8)
    ap.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args(argv)
    a.run_dir = a.run_dir or str(LAKE / f"s12mm-colab-{a.run_id}")
    FIXED_CLOCK["on"] = a.rehearsal
    if a.cap is None:
        a.cap = 3.0 if a.mode == "bench" else 250.0
    if a.mode == "run" and a.cap > 250:  # Josh raised the approval from 220 on 2026-10-29
        ap.error("S12m's approved Colab cap is 250 units")
    if not 1 <= a.sessions <= MAX_SESSIONS:
        ap.error(f"--sessions must be 1..{MAX_SESSIONS}")
    if a.mode == "run" and not a.units:
        ap.error("run needs --units (split_units.py output)")
    if a.rps > 50 and not a.allow_above_etiquette:
        ap.error("--rps above 50 needs --allow-above-etiquette (Bluesky's OK and Josh's say-so)")
    drv = Driver(a)

    def on_signal(signum, _frame):
        raise Stop(130, f"signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)
    try:
        if a.mode == "status":
            return drv.status()
        print(f"== execute={a.execute} run_dir={a.run_dir} colab={drv.colab.bin}", flush=True)
        rc = drv.bench() if a.mode == "bench" else drv.run()
        if not a.execute:
            print("== dry run: nothing was executed (add --execute to run the printed commands)")
        return rc
    except Stop as e:
        print(f"== STOP ({e.code}): {e.why}", flush=True)
        return e.code
    finally:
        drv.halt.set()
        if drv.session is not None and drv.session.live:   # the exit trap: only our own live sessions
            drv.session.stop("stopped(trap)")
        for s in list(drv.live_sessions):
            s.stop("stopped(trap)")


if __name__ == "__main__":
    sys.exit(main())
