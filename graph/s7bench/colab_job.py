"""S7 Colab job: run on the session VM by `colab exec -f colab_job.py` (see run_colab.sh; never by hand on the Mac).

Expects /content/s7/ to hold what run_colab.sh uploaded (flat): s7_manifest.json, s7lib.py, bench.py,
s7_random.parquet, s7_slab.parquet, parity_mac.npz, eg2_sha256.json. Configuration via --env (colab exec has no argv):
  S7_UNIT_CAP       units this session may spend (the driver passes what is left of the 8-unit cap; never raised here)
  S7_ALLOC_MIN      minutes billed before this script started (allocation, uploads)
  S7_RATE           units/hour measured by the driver from `colab usage` (default: the GPU table in s7lib)
  S7_TORCH_PIN      1 (default) installs torch==2.14.1/torchvision==0.29.1 (the Mac's S4 stack); reverted to the
                    runtime's own torch if CUDA is then unusable; 0 keeps the runtime's torch
  S7_LOCAL_REHEARSAL=1 with S7_ROOT, S7_OUT, S7_MODEL_DIR, S7_DEVICE, S7_LIMIT_RANDOM, S7_LIMIT_SLAB: Mac rehearsal of
                    this exact file (no installs, no download; local model checked against the same sha256 list)

Steps: verify every uploaded file's sha256; record the runtime; install the pins; download google/embeddinggemma-2 at
the pinned revision and check every file's sha256 against the Mac's ~/catbird-lake-local/models/SHA256SUMS; run
bench.py under the unit guard; write job_receipt.json; tar the results to /content/s7_results.tar. No GCS here (the
round trip is roundtrip_job.py, a separate exec after `colab auth`).
"""
import importlib.metadata as md
import json
import os
import subprocess
import sys
import tarfile
import time

T0 = time.time()
REHEARSAL = os.environ.get("S7_LOCAL_REHEARSAL") == "1"
ROOT = os.environ.get("S7_ROOT", "/content/s7") if REHEARSAL else "/content/s7"
OUT = os.environ.get("S7_OUT", "/content/s7_out") if REHEARSAL else "/content/s7_out"
TAR = os.path.join(os.path.dirname(OUT), "s7_results.tar")
CAP = float(os.environ.get("S7_UNIT_CAP", "3"))
ALLOC_MIN = float(os.environ.get("S7_ALLOC_MIN", "5"))
REPO, REV = "google/embeddinggemma-2", "f9972ac22e696a105c482202646979e33e4b5a1e"
PINS = ["transformers==5.21.0", "sentence-transformers==6.2.0"]
TORCH_PINS = ["torch==2.14.1", "torchvision==0.29.1"]
# Prior seconds per step (only to decide whether to START it): install, model download+check, the bench itself
# (parity 1,250 + warm-up + 4 x 10k sweep + 100k + 10k rows at a pessimistic 1,000 posts/s, x 1.5).
EST_S = {"install": 420, "download": 180, "bench": (1250 + 256 + 40_000 + 110_000) / 1000 * 1.5 + 120}


def sh(cmd, timeout=1800):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def pip_install(pkgs):
    rc, out = sh(["uv", "pip", "install", "--system", "-q"] + pkgs)
    if rc != 0:
        rc, out = sh([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
    return rc, out[-2000:]


def torch_probe():
    rc, out = sh([sys.executable, "-c", "import torch, json; print(json.dumps([torch.__version__, torch.cuda.is_available(), "
                   "torch.version.cuda]))"], timeout=300)
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception:
        return [None, False, out[-500:]]


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, ROOT)
    import s7lib

    def say(*a):
        print("[s7job]", s7lib.redact(" ".join(str(x) for x in a)), flush=True)

    rec = {"started_unix": T0, "cap": CAP, "alloc_min": ALLOC_MIN, "rehearsal": REHEARSAL, "python": sys.version.split()[0],
           "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])[1],
           "preinstalled": {p: ver(p) for p in ["torch", "torchvision", "torchaudio", "transformers",
                                                "sentence-transformers", "huggingface-hub", "google-cloud-storage"]}}
    say(json.dumps(rec))
    man = json.load(open(os.path.join(ROOT, "s7_manifest.json")))
    for rel, want in man["files"].items():
        got = s7lib.sha256_file(os.path.join(ROOT, rel))
        if got != want:
            sys.exit(f"[s7job] sha256 mismatch for {rel}: {got} != {want}")
    rec["manifest_verified"] = len(man["files"])

    gpu_name = rec["nvidia_smi"].split(",")[0].strip() if "," in rec["nvidia_smi"] else ""
    if REHEARSAL:
        gpu, rate = "local", float(os.environ.get("S7_RATE", "1.71"))
    else:
        gpu, rate = s7lib.gpu_rate(gpu_name)
        if os.environ.get("S7_RATE"):
            rate = max(float(os.environ["S7_RATE"]), rate or 0.0)  # never below the table rate
        if rate is None:
            sys.exit(f"[s7job] unknown GPU {gpu_name!r} and no S7_RATE: stopping")
    guard = s7lib.UnitGuard(CAP, rate, ALLOC_MIN + (time.time() - T0) / 60)
    rec.update(gpu=gpu, gpu_name=gpu_name, rate=rate)

    def step_ok(name):
        if guard.allows(EST_S[name]):
            return True
        rec.setdefault("skipped", []).append({"step": name, **guard.state()})
        say("SKIP", name, json.dumps(guard.state()))
        return False

    if not REHEARSAL:
        if not step_ok("install"):
            return finish(rec, guard, say)
        pins = PINS + (TORCH_PINS if os.environ.get("S7_TORCH_PIN", "1") == "1" else [])
        rc, out = pip_install(pins)
        rec["install"] = {"pins": pins, "rc": rc, "tail": s7lib.redact(out)}
        probe = torch_probe()
        rec["install"]["torch_probe"] = probe
        if not probe[1] and rec["preinstalled"]["torch"]:
            # The pinned wheel cannot use this runtime's driver: go back to the runtime's own torch.
            back = [f"torch=={rec['preinstalled']['torch']}"] + (
                [f"torchvision=={rec['preinstalled']['torchvision']}"] if rec["preinstalled"]["torchvision"] else [])
            rc2, out2 = pip_install(back)
            rec["install"]["torch_reverted"] = {"to": back, "rc": rc2, "probe": torch_probe()}
        if ver("torchaudio") and ver("torch") != rec["preinstalled"]["torch"]:
            rec["install"]["torchaudio_uninstalled"] = sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"])[0]
        rec["installed"] = {p: ver(p) for p in ["torch", "torchvision", "torchaudio", "transformers", "sentence-transformers"]}
        say("installed", json.dumps(rec["installed"]), "probe", json.dumps(torch_probe()))
        if not torch_probe()[1]:
            rec["fatal"] = "CUDA not usable after install"
            return finish(rec, guard, say)

    want = json.load(open(os.path.join(ROOT, "eg2_sha256.json")))
    if REHEARSAL:
        mdir = os.environ["S7_MODEL_DIR"]
    else:
        if not step_ok("download"):
            return finish(rec, guard, say)
        from huggingface_hub import snapshot_download
        mdir = snapshot_download(REPO, revision=REV, local_dir="/content/models/embeddinggemma-2",
                                 allow_patterns=sorted(want))
    bad = {rel: s7lib.sha256_file(os.path.join(mdir, rel)) for rel in want
           if s7lib.sha256_file(os.path.join(mdir, rel)) != want[rel]}
    rec["model"] = {"repo": REPO, "revision": REV, "dir": mdir, "files_checked": len(want), "sha256_mismatch": bad}
    if bad:
        rec["fatal"] = "model sha256 mismatch"
        return finish(rec, guard, say)
    say("model ok", len(want), "files")

    if not step_ok("bench"):
        return finish(rec, guard, say)
    cmd = [sys.executable, os.path.join(ROOT, "bench.py"), "--model-dir", mdir, "--data", ROOT, "--out", OUT,
           "--parity-ref", os.path.join(ROOT, "parity_mac.npz"), "--unit-cap", str(CAP), "--rate", str(rate),
           "--preamble-min", f"{(time.time() - guard.t0) / 60:.2f}"]
    if REHEARSAL:
        cmd += ["--device", os.environ.get("S7_DEVICE", "mps"), "--limit-random", os.environ.get("S7_LIMIT_RANDOM", "3000"),
                "--limit-slab", os.environ.get("S7_LIMIT_SLAB", "1000"), "--batch-sizes", "32,64", "--sweep-n", "1000"]
    t = time.time()
    rc = subprocess.call(cmd)
    rec["bench"] = {"rc": rc, "secs": round(time.time() - t, 1)}
    say("bench rc", rc, json.dumps(guard.state()))
    return finish(rec, guard, say)


def finish(rec, guard, say):
    import s7lib
    rec["finished_unix"] = time.time()
    rec["units_est_total"] = round(guard.spent(), 3)
    rec["pip_freeze_tail"] = sh([sys.executable, "-m", "pip", "freeze"])[1].splitlines()[:400]
    json.dump(rec, open(os.path.join(OUT, "job_receipt.json"), "w"), indent=1)
    with tarfile.open(TAR, "w") as tf:
        for f in sorted(os.listdir(OUT)):
            tf.add(os.path.join(OUT, f), arcname=f)
    say("TAR", TAR, os.path.getsize(TAR), s7lib.sha256_file(TAR))
    b = os.path.join(OUT, "bench.json")
    if os.path.exists(b):
        j = json.load(open(b))
        for k in ["random", "slab"]:
            if isinstance(j.get(k), dict) and "posts_per_s" in j[k]:
                say("RESULT", k, j[k]["posts_per_s"], "posts/s", j["tokens"][k]["tokens_per_post_with_prompt"], "tok/post",
                    j[k].get("units_per_10m_arith", ""))
        if "parity" in j:
            say("PARITY median random_full", j["parity"]["random_full"]["median"], "bar", j["parity"]["bar_median_ge_0_99_on_random"])
        say("NAN_OR_INF_ROWS", j.get("nan_or_inf_rows_total"))
    say("UNITS_EST", rec["units_est_total"])
    return rec


main()
