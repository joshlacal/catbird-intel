"""S12m / S7g Colab job: run in the session's kernel by `colab exec -f colab_job.py` (colab_driver.py; never by hand).

Configuration comes from --env (colab exec has no argv). S12C_MODE picks the step:
  setup   verify every uploaded file against s12c_manifest.json (sha256); unpack code.tar.gz and gate.tar.gz;
          record the runtime (Python, nvidia-smi, preinstalled packages); refuse if swap is active (after trying
          `swapoff -a`); pip-install the S4/S7 pins (transformers 5.21.0, sentence-transformers 6.2.0, torch
          2.14.1 / torchvision 0.29.1, reverted to the runtime's torch if CUDA is then unusable); download
          google/embeddinggemma-2 at the pinned revision from Hugging Face and check every file's sha256 against
          eg2_sha256.json (built by the driver from the Mac's ~/catbird-lake-local/models/SHA256SUMS); compile the
          code under this Python. Prints `[s12c] SETUP_OK` or `[s12c] SETUP_FAILED <why>`.
  bench   (S7g) unpack bench.tar; job.py bench with the uploaded inputs, a local-directory --out, the unit guard
          (S12C_CAP units at S12C_RATE units/h) and, with S12C_RAMP, the CDN rate ramp; tar the results to
          /content/s12mm/dl/bench_results.tar. Prints `[s12c] BENCH RC <rc> TAR <path> <bytes> <sha256>`.
  batch   (S12m) check the uploaded batch tar's sha256 (S12C_BATCH_SHA256) and every unit's sha256 against its
          batch.json; delete the VM copies of earlier batches the driver confirmed (S12C_CLEAN); job.py run over
          the batch (local stores; S12C_CAP / S12C_RATE; S12C_RPS; S12C_TIMEOUT bounds the job so it ends before the
          exec does); tar units/, _ledger/u-*.json and receipt/ to /content/s12mm/dl/<batch>.tar.
          Prints `[s12c] BATCH <batch> RC <rc> UNITS <n> TAR <path> <bytes> <sha256>`.
No GCS, no Drive, no `colab auth`: data arrives by `colab upload` and leaves by `colab download`.

S12C_REHEARSAL=1 (Mac only, through the fake colab CLI of test_colab.py): S12C_CONTENT replaces /content, no pip,
no download (S12C_MODEL_DIR is checked against the same sha256 list unless S12C_FAKE_ENCODER=1), the job gets
--rehearsal --now S12C_NOW, and run mode serves synthetic thumbnails from a loopback server (synth.ImageServer).
"""
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import tarfile
import threading
import time

T0 = time.time()
REHEARSAL = os.environ.get("S12C_REHEARSAL") == "1"
CONTENT = os.environ.get("S12C_CONTENT", "/content") if REHEARSAL else "/content"
ROOT = os.path.join(CONTENT, "s12mm")
CODE = os.path.join(ROOT, "code")
GATE = os.path.join(ROOT, "gate")
DL = os.path.join(ROOT, "dl")
MODEL_DIR = os.path.join(CONTENT, "models", "embeddinggemma-2")
REPO, REV = "google/embeddinggemma-2", "f9972ac22e696a105c482202646979e33e4b5a1e"
PINS = ["transformers==5.21.0", "sentence-transformers==6.2.0"]
TORCH_PINS = ["torch==2.14.1", "torchvision==0.29.1"]
TAG = "[s12c]"


def say(*a):
    print(TAG, *a, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sh(cmd, timeout=1800, **kw):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def safe_extract(tar_path, dst):
    """Extract regular files and directories only, never outside dst, never AppleDouble `._*` members."""
    os.makedirs(dst, exist_ok=True)
    base = os.path.realpath(dst)
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if name.startswith("._") or not (m.isfile() or m.isdir()):
                continue
            target = os.path.realpath(os.path.join(dst, m.name))
            if not target.startswith(base + os.sep) and target != base:
                raise RuntimeError(f"tar member escapes {dst}: {m.name}")
            if m.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with tf.extractfile(m) as src, open(target, "wb") as out:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    out.write(b)


def tar_dir(dst, members):
    """members: [(path, arcname)]; returns (path, bytes, sha256)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with tarfile.open(tmp, "w") as tf:
        for p, arc in members:
            tf.add(p, arcname=arc)
    os.replace(tmp, dst)
    return dst, os.path.getsize(dst), sha256(dst)


def swap_state():
    try:
        with open("/proc/swaps") as f:
            return len(f.read().strip().splitlines()) > 1
    except OSError:
        return None


def torch_probe():
    rc, out = sh([sys.executable, "-c", "import torch, json; print(json.dumps([torch.__version__, torch.cuda.is_available(), "
                  "torch.version.cuda]))"], timeout=300)
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return [None, False, out[-300:]]


def pip_install(pkgs):
    rc, out = sh(["uv", "pip", "install", "--system", "-q"] + pkgs)
    if rc != 0:
        rc, out = sh([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
    return rc, out[-1500:]


def fail(step, why, rec=None):
    if rec is not None:
        rec["fatal"] = why
        write_json(os.path.join(ROOT, f"{step}.json"), rec)
    say(f"{step.upper()}_FAILED", why)
    return 1


def write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".part", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(p + ".part", p)


# --- setup ----------------------------------------------------------------------------------------------------

def setup():
    rec = {"started_unix": T0, "rehearsal": REHEARSAL, "python": sys.version.split()[0],
           "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])[1],
           "preinstalled": {p: ver(p) for p in ["torch", "torchvision", "torchaudio", "transformers",
                                                "sentence-transformers", "huggingface-hub", "pyarrow", "numpy", "pillow"]},
           "euid": os.geteuid() if hasattr(os, "geteuid") else None}
    say("runtime", json.dumps(rec))
    if sys.version_info < (3, 10):
        return fail("setup", f"python {sys.version.split()[0]} < 3.10", rec)
    man = json.load(open(os.path.join(ROOT, "s12c_manifest.json")))
    for rel, want in man["files"].items():
        got = sha256(os.path.join(ROOT, rel))
        if got != want:
            return fail("setup", f"sha256 mismatch for {rel}", rec)
    rec["manifest_verified"] = len(man["files"])
    safe_extract(os.path.join(ROOT, "code.tar.gz"), CODE)
    safe_extract(os.path.join(ROOT, "gate.tar.gz"), GATE)
    rc, out = sh([sys.executable, "-m", "compileall", "-q", CODE])
    if rc != 0:
        return fail("setup", f"code does not compile under {sys.version.split()[0]}: {out[-500:]}", rec)
    sys.path.insert(0, os.path.join(CODE, "s12mm"))
    import mmlib as L
    b = L.load_bundle(GATE)          # checks the bundle's own sha256s
    rec["gate"] = {"watermark": L.iso(b.watermark), "excluded_dids": len(b.excluded_dids),
                   "labelled_uris": len(b.labelled_uris), "sha256": b.sha256}
    if swap_state():
        rec["swapoff"] = sh(["swapoff", "-a"])
        if swap_state():
            return fail("setup", "swap is active and swapoff failed: image bytes could be paged to disk", rec)
    rec["swap_active"] = swap_state()

    if not REHEARSAL:
        rc, out = pip_install(PINS + TORCH_PINS)
        rec["install"] = {"pins": PINS + TORCH_PINS, "rc": rc, "tail": out}
        probe = torch_probe()
        rec["install"]["torch_probe"] = probe
        if not probe[1] and rec["preinstalled"]["torch"]:
            back = [f"torch=={rec['preinstalled']['torch']}"] + (
                [f"torchvision=={rec['preinstalled']['torchvision']}"] if rec["preinstalled"]["torchvision"] else [])
            rec["install"]["torch_reverted"] = {"to": back, "rc": pip_install(back)[0], "probe": torch_probe()}
        if ver("torchaudio") and ver("torch") != rec["preinstalled"]["torch"]:
            rec["install"]["torchaudio_uninstalled"] = sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"])[0]
        rec["installed"] = {p: ver(p) for p in ["torch", "torchvision", "transformers", "sentence-transformers"]}
        if not torch_probe()[1]:
            return fail("setup", "CUDA not usable after install", rec)

    want = json.load(open(os.path.join(ROOT, "eg2_sha256.json")))
    if REHEARSAL and os.environ.get("S12C_FAKE_ENCODER") == "1":
        mdir, bad = "fake", {}
    else:
        if REHEARSAL:
            mdir = os.environ["S12C_MODEL_DIR"]
        else:
            from huggingface_hub import snapshot_download
            mdir = snapshot_download(REPO, revision=REV, local_dir=MODEL_DIR, allow_patterns=sorted(want))
        bad = {}
        for rel, h in want.items():
            p = os.path.join(mdir, rel)
            got = sha256(p) if os.path.exists(p) else "missing"
            if got != h:
                bad[rel] = got
    rec["model"] = {"repo": REPO, "revision": REV, "dir": mdir, "files_checked": len(want), "sha256_mismatch": bad}
    if bad:
        return fail("setup", f"model sha256 mismatch: {sorted(bad)}", rec)
    rec["setup_seconds"] = round(time.time() - T0, 1)
    write_json(os.path.join(ROOT, "setup.json"), rec)
    say("SETUP_OK", json.dumps({"python": rec["python"], "gate_watermark": rec["gate"]["watermark"],
                                "model_files": len(want), "seconds": rec["setup_seconds"],
                                "installed": rec.get("installed")}))
    return 0


# --- job runs -------------------------------------------------------------------------------------------------

def model_dir():
    return json.load(open(os.path.join(ROOT, "setup.json")))["model"]["dir"]


def common_args(work, out):
    a = ["--model-dir", model_dir(), "--work", work, "--run-id", os.environ["S12C_RUN_ID"], "--out", out,
         "--gate-bundle", GATE, "--rps", os.environ.get("S12C_RPS", "50"),
         "--cap-usd", os.environ["S12C_CAP"], "--usd-per-hour", os.environ["S12C_RATE"], "--rate-unit", "colab_units",
         "--max-label-lag-h", os.environ.get("S12C_MAX_LAG_H", "36")]
    if os.environ.get("S12C_ALLOW_ABOVE") == "1":
        a.append("--allow-above-etiquette")
    if REHEARSAL:
        a += ["--rehearsal", "--now", os.environ["S12C_NOW"], "--device", os.environ.get("S12C_DEVICE", "cpu"),
              "--mps-gb", os.environ.get("S12C_MPS_GB", "4")]
        if os.environ.get("S12C_FAKE_ENCODER") == "1":
            a.append("--fake-encoder")
    return a


def run_job(args, timeout, log_path):
    """job.py in a child process (fresh CUDA context per batch; our own deadline before the exec's)."""
    cmd = [sys.executable, os.path.join(CODE, "s12mm", "job.py")] + args
    say("job", " ".join(cmd[2:4]), "timeout", int(timeout))
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=CODE,
                             env={**os.environ, "TOKENIZERS_PARALLELISM": "false"})
        killer = threading.Timer(max(30.0, timeout), p.kill)
        killer.start()
        try:
            for line in p.stdout:
                log.write(line)
                if line.startswith("["):          # the job's progress lines; everything else stays in the log
                    print(line.rstrip()[:2000], flush=True)
            rc = p.wait()
        finally:
            killer.cancel()
    return rc


def bench():
    work, out = os.path.join(ROOT, "work", "bench"), os.path.join(ROOT, "out", "bench")
    inb = os.path.join(ROOT, "in", "bench")
    if not os.path.exists(os.path.join(inb, "s7_slab.parquet")):
        safe_extract(os.path.join(ROOT, "bench.tar"), inb)
    args = ["bench"] + common_args(work, out) + [
        "--bench-text", os.path.join(inb, "s7_slab.parquet"), "--bench-random", os.path.join(inb, "s7_random.parquet"),
        "--parity-ref", os.path.join(inb, "parity_mac.npz"), "--mm-parity-ref", os.path.join(inb, "mm_parity_mac.npz"),
        "--soft-tokens-list", os.environ.get("S12C_SOFTS", "70,280"),
        "--n-text", os.environ.get("S12C_N_TEXT", "10000"), "--n-image-posts", os.environ.get("S12C_N_IMAGE_POSTS", "10000"),
        "--vm-start-epoch", os.environ.get("S12C_SESSION_T0", str(T0))]
    if os.environ.get("S12C_SYNTHETIC_IMAGES") == "1" and REHEARSAL:
        args += ["--synthetic-images"]
        if os.environ.get("S12C_SYNTHETIC_LIMIT_RPS"):
            args += ["--synthetic-limit-rps", os.environ["S12C_SYNTHETIC_LIMIT_RPS"]]
    else:
        args += ["--image-posts", os.path.join(inb, "image_posts")]
        if os.path.exists(os.path.join(inb, "bench_producers.txt")):
            args += ["--bench-producers", os.path.join(inb, "bench_producers.txt")]
    for k, flag in (("S12C_RAMP", "--cdn-ramp"), ("S12C_RAMP_STEP_S", "--ramp-step-s"),
                    ("S12C_RAMP_MIN_STEP_S", "--ramp-min-step-s"), ("S12C_TEXT_BS", "--text-batch-sizes"),
                    ("S12C_MM_BS", "--mm-batch-sizes"), ("S12C_SWEEP_N_MM", "--sweep-n-mm")):
        if os.environ.get(k):
            args += [flag, os.environ[k]]
    os.makedirs(work, exist_ok=True)
    log_path = os.path.join(ROOT, "bench_job.log")
    rc = run_job(args, float(os.environ.get("S12C_TIMEOUT", "5400")) - 120, log_path)
    members = [(log_path, "bench_job.log"), (os.path.join(ROOT, "setup.json"), "setup.json")]
    for rel in ("bench/bench.json", "bench/vectors.parquet"):
        p = os.path.join(out, rel)
        if os.path.exists(p):
            members.append((p, os.path.basename(rel)))
    write_json(os.path.join(ROOT, "bench_receipt.json"), {"rc": rc, "seconds": round(time.time() - T0, 1)})
    members.append((os.path.join(ROOT, "bench_receipt.json"), "bench_receipt.json"))
    path, n, h = tar_dir(os.path.join(DL, "bench_results.tar"), members)
    say("BENCH", "RC", rc, "TAR", path, n, h)
    return rc


def batch():
    import shutil
    name = os.environ["S12C_BATCH"]
    tar = os.path.join(ROOT, "batches", f"{name}.tar")
    if sha256(tar) != os.environ["S12C_BATCH_SHA256"]:
        say("BATCH", name, "RC", 90, "UNITS", 0, "TAR", "-", 0, "-", "(input tar sha256 mismatch)")
        return 90
    for old in [x for x in os.environ.get("S12C_CLEAN", "").split(",") if x]:
        for d in (os.path.join(ROOT, "out", old), os.path.join(ROOT, "work", old), os.path.join(ROOT, "in", old)):
            shutil.rmtree(d, ignore_errors=True)
        for f in (os.path.join(DL, f"{old}.tar"), os.path.join(ROOT, "batches", f"{old}.tar")):
            if os.path.exists(f):
                os.remove(f)
    inb = os.path.join(ROOT, "in", name)
    shutil.rmtree(inb, ignore_errors=True)
    safe_extract(tar, inb)
    meta = json.load(open(os.path.join(inb, "batch.json")))
    for u in meta["units"]:
        if sha256(os.path.join(inb, "units", f"{u['unit']}.parquet")) != u["sha256"]:
            say("BATCH", name, "RC", 91, "UNITS", 0, "TAR", "-", 0, "-", f"(unit {u['unit']} sha256 mismatch)")
            return 91
    work, out = os.path.join(ROOT, "work", name), os.path.join(ROOT, "out", name)
    srv = None
    args = ["run"] + common_args(work, out) + ["--posts", os.path.join(inb, "units"), "--soft-tokens",
                                               os.environ.get("S12C_SOFT", "70"), "--unit-rows", str(meta["unit_rows"]),
                                               "--text-bs", os.environ.get("S12C_TEXT_BS_RUN", "64"),
                                               "--mm-bs", os.environ.get("S12C_MM_BS_RUN", "8")]
    if REHEARSAL:
        sys.path.insert(0, os.path.join(CODE, "s12mm"))
        from synth import ImageServer
        srv = ImageServer(latency_ms=int(os.environ.get("S12C_FAKE_CDN_MS", "2"))).__enter__()
        args += ["--cdn-base", srv.base]
    try:
        rc = run_job(args, float(os.environ.get("S12C_TIMEOUT", "3600")) - 120, os.path.join(ROOT, f"{name}.log"))
    finally:
        if srv is not None:
            srv.__exit__()
    members, units = [(os.path.join(ROOT, f"{name}.log"), "job.log")], []
    led = os.path.join(out, "_ledger")
    for f in sorted(os.listdir(led)) if os.path.isdir(led) else []:
        if f.startswith("u-") and f.endswith(".json"):
            uf = os.path.join(out, "units", f[:-5] + ".parquet")
            if os.path.exists(uf):
                members += [(os.path.join(led, f), f"_ledger/{f}"), (uf, f"units/{f[:-5]}.parquet")]
                units.append(f[:-5])
    rdir = os.path.join(out, "receipt")
    for f in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
        members.append((os.path.join(rdir, f), f"receipt/{f}"))
    path, n, h = tar_dir(os.path.join(DL, f"{name}.tar"), members)
    shutil.rmtree(inb, ignore_errors=True)          # the input text is not kept on the VM
    say("BATCH", name, "RC", rc, "UNITS", len(units), "TAR", path, n, h)
    return rc


def main():
    os.makedirs(DL, exist_ok=True)
    mode = os.environ.get("S12C_MODE", "")
    try:
        if mode == "setup":
            return setup()
        if mode == "bench":
            return bench()
        if mode == "batch":
            return batch()
        say("UNKNOWN_MODE", mode)
        return 2
    except Exception as e:  # report, never hang the exec
        import traceback
        say("ERROR", type(e).__name__, str(e)[:500])
        traceback.print_exc()
        return 99


S12C_RC = main()
say("RC", S12C_RC)
