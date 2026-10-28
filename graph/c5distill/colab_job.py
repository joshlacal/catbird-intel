"""C5 Colab job: run on the Colab VM by `colab exec -f colab_job.py` (see run_colab.sh; never by hand on the Mac).

Expects /content/c5/ to hold what run_colab.sh uploaded (flat): c5_manifest.json, rules_c5.json, c5lib.py, train.py
and c5_{pairs,profiles,communities}.parquet. Configuration comes from --env (colab exec has no argv):
  C5_UNIT_CAP      units this run may spend (default 20, the Batch A cap; never raised by the job)
  C5_ALLOC_MIN     minutes already billed before this script started (VM allocation, uploads; default 5)
  C5_TASKS         comma list, default "stance,account_type,community_type" (stance first: it gates C6)
  C5_SAVE_MODELS   1 (default) saves fp16 students for download; 0 keeps only metrics and predictions
  C5_RATE          units/hour override when the GPU is not in c5lib.RATES
  C5_LOCAL_REHEARSAL=1 with C5_ROOT, C5_OUT, C5_MODEL_DIR: Mac rehearsal of this exact file (no CUDA needed,
                   mmBERT-small, train.py --sanity, so the held-out split is never read); never used on Colab

Steps: print the runtime; verify every uploaded file against the manifest's sha256; install transformers only if
the runtime's is too old for ModernBERT; download mmBERT-base at the pinned revision and check its weights'
sha256; train each task with the unit guard (the process's guard counts all time since this job began plus
C5_ALLOC_MIN); skip a task whose estimate would cross the cap; write job_receipt.json; tar the outputs.
No data is read from or written to GCS or Drive. The VM, and the uploaded text with it, is destroyed by
`colab stop`, which run_colab.sh issues for this session only.
"""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time

T0 = time.time()
REHEARSAL = os.environ.get("C5_LOCAL_REHEARSAL") == "1"
ROOT = os.environ.get("C5_ROOT", "/content/c5") if REHEARSAL else "/content/c5"
OUT = os.environ.get("C5_OUT", "/content/c5_out") if REHEARSAL else "/content/c5_out"
TARS = os.path.dirname(OUT)
CAP = float(os.environ.get("C5_UNIT_CAP", "20"))
ALLOC_MIN = float(os.environ.get("C5_ALLOC_MIN", "5"))
TASKS = [t for t in os.environ.get("C5_TASKS", "stance,account_type,community_type").split(",") if t]
SAVE = os.environ.get("C5_SAVE_MODELS", "1") == "1"
REPO, REV = "jhu-clsp/mmBERT-base", "c5955035435e2bf121cde7f3c8863ef52ff35d82"
WEIGHTS_SHA = "8ea64ec1ea4eb8fca0fc14b69a2ae571de6bfbc25fd214bb932dd4aba6a3a04e"
SANITY_WEIGHTS_SHA = "be7c9ce2af947fb5bb17a3bc919552f898c5ef4164bf0a7190cf25a0f98f85b6"  # mmBERT-small
# Prior L4 minutes per task (FLOP arithmetic at ~25% MFU plus the AdamW pass over 307M params; RUNBOOK.md).
# Scaled by the GPU's speed relative to L4 and used only to decide whether to START a task.
EST_L4_MIN = {"stance": 12, "account_type": 18, "community_type": 15}
SPEED_VS_L4 = {"L4": 1.0, "T4": 0.35, "A100": 2.8, "G4": 3.0}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sh(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        return str(e)
    return (r.stdout + r.stderr).strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, ROOT)
    import c5lib

    rec = {"started_unix": T0, "cap": CAP, "alloc_min": ALLOC_MIN, "tasks": TASKS,
           "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
           "python": sys.version.split()[0]}
    print("[c5job]", json.dumps(rec), flush=True)

    man = json.load(open(os.path.join(ROOT, "c5_manifest.json")))
    for rel, want in man["files"].items():
        got = sha256(os.path.join(ROOT, rel))
        if got != want:
            sys.exit(f"[c5job] sha256 mismatch for {rel}: {got} != {want}")
    rec["manifest_verified"] = len(man["files"])

    import torch
    if REHEARSAL:
        dname, gpu, rate = "local-rehearsal", "local", 0.0
    elif not torch.cuda.is_available():
        sys.exit("[c5job] no CUDA device; stopping before any training")
    else:
        dname = torch.cuda.get_device_name(0)
        gpu, rate = c5lib.gpu_rate(dname)
    if os.environ.get("C5_RATE"):
        rate = float(os.environ["C5_RATE"])
    if rate is None:
        sys.exit(f"[c5job] unknown GPU {dname!r} and no C5_RATE: stopping")
    guard = c5lib.UnitGuard(CAP, rate, ALLOC_MIN + (time.time() - T0) / 60)  # counts from VM allocation
    rec.update(device=dname, gpu=gpu, rate=rate,
               vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None)

    import transformers
    from packaging.version import Version
    if Version(transformers.__version__) < Version("4.48"):
        print("[c5job]", sh([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.3"]), flush=True)
        rec["transformers_installed"] = "4.57.3"
    if REHEARSAL:
        mdir, want = os.environ["C5_MODEL_DIR"], SANITY_WEIGHTS_SHA
    else:
        from huggingface_hub import snapshot_download
        mdir, want = snapshot_download(REPO, revision=REV, local_dir="/content/models/mmbert-base",
                                       allow_patterns=["*.json", "pytorch_model.bin"]), WEIGHTS_SHA
    if sha256(os.path.join(mdir, "pytorch_model.bin")) != want:
        sys.exit("[c5job] student weights sha256 mismatch: stopping")
    rec["model"] = {"dir": mdir, "weights_sha256": want, "repo": REPO if not REHEARSAL else "jhu-clsp/mmBERT-small",
                    "revision": REV if not REHEARSAL else "abc32620dd4f6ab06f5fbe905dc25f310618e09f"}

    rec["runs"] = []
    for task in TASKS:
        est_s = EST_L4_MIN[task] * 60 / SPEED_VS_L4.get(gpu, 1.0) * 1.5
        if not guard.allows(est_s):
            rec["runs"].append({"task": task, "skipped": f"estimate {est_s / 60:.0f} min would cross the cap", **guard.state()})
            print("[c5job] SKIP", task, guard.state(), flush=True)
            continue
        cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--task", task, "--data", ROOT,
               "--model", mdir, "--out", OUT, "--unit-cap", str(CAP), "--rate", str(rate),
               "--preamble-min", f"{(time.time() - guard.t0) / 60:.2f}"] + (["--save-model"] if SAVE else [])
        if REHEARSAL:
            cmd += ["--sanity", "--sanity-train", "800", "--sanity-eval", "300", "--epochs", "1",
                    "--batch", "4" if task == "community_type" else "8",
                    "--max-len", "256" if task == "community_type" else "128"]
        t = time.time()
        rc = subprocess.call(cmd)
        rec["runs"].append({"task": task, "rc": rc, "secs": round(time.time() - t, 1), **guard.state()})
        print("[c5job] DONE", json.dumps(rec["runs"][-1]), flush=True)

    rec["finished_unix"] = time.time()
    rec["units_est_total"] = round(guard.spent(), 3)
    json.dump(rec, open(os.path.join(OUT, "job_receipt.json"), "w"), indent=1)
    small = [f for f in os.listdir(OUT) if not f.endswith("_model")]
    with tarfile.open(os.path.join(TARS, "c5_results.tar"), "w") as tf:
        for f in small:
            tf.add(os.path.join(OUT, f), arcname=f)
    if SAVE:
        with tarfile.open(os.path.join(TARS, "c5_models.tar"), "w") as tf:
            for f in os.listdir(OUT):
                if f.endswith("_model"):
                    tf.add(os.path.join(OUT, f), arcname=f)
    for f in [os.path.join(TARS, "c5_results.tar")] + ([os.path.join(TARS, "c5_models.tar")] if SAVE else []):
        print("[c5job] TAR", f, os.path.getsize(f), sha256(f), flush=True)
    for r in rec["runs"]:
        m = os.path.join(OUT, f"{r['task']}_metrics.json")
        if os.path.exists(m):
            t = json.load(open(m))["test"]
            print("[c5job] BAR", r["task"], json.dumps(t["bar"]), flush=True)
    print("[c5job] UNITS_EST", rec["units_est_total"], flush=True)


main()
