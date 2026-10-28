"""Assemble the S4 receipt.json from the step receipts, run summaries and score output.

  build_receipt.py PILOT_DIR

Re-hashes every model file listed in models/SHA256SUMS and every S4 input and output, and copies
throughput (posts/s and tokens/post on MPS, per model and input) from each run_summary.json.
"""
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

LAKE = pathlib.Path.home() / "catbird-lake-local"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(pilot):
    pilot = pathlib.Path(pilot)
    models = {}
    for line in open(LAKE / "models/SHA256SUMS"):
        want, rel = line.split()
        got = sha256(LAKE / "models" / rel)
        models[rel] = {"sha256": got, "matches_SHA256SUMS": got == want}
    runs = {}
    for name in ["eg2_posts", "harrier_bakeoff", "eg2_bios"]:
        s = json.load(open(pilot / "vectors" / name / "run_summary.json"))
        chunks = [json.loads(l) for l in open(pilot / "vectors" / name / "chunks.jsonl")]
        s["chunks"] = len(chunks)
        s["chunk_nan_rows_total"] = sum(c["nan_rows"] for c in chunks)
        s["full_norm_range"] = [min(c["full_norm_min"] for c in chunks), max(c["full_norm_max"] for c in chunks)]
        s["embed_seconds_all_chunks"] = round(sum(c["seconds"] for c in chunks), 1)
        s["posts_per_s_all_chunks"] = round(s["n"] / s["embed_seconds_all_chunks"], 1)
        s["outputs_sha256"] = {f: sha256(pilot / "vectors" / name / f) for f in ["ids.npy", "vec256_fp16.npy"]}
        runs[name] = s
    pip = subprocess.run([str(LAKE / "s4-venv/bin/python"), "-m", "pip", "list", "--format=freeze"],
                         capture_output=True, text=True).stdout
    if not pip.strip():
        pip = subprocess.run(["uv", "pip", "list", "--format=freeze", "--python", str(LAKE / "s4-venv/bin/python")],
                             capture_output=True, text=True).stdout
    receipt = {
        "step": "S4 EmbeddingGemma 2 Mac pilot and bake-off (whole-network plan)",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md (S1, amendment A1)",
        "machine": {"chip": "Apple M4 Max", "ram_gib": 128, "device": "mps", "macos": platform.mac_ver()[0]},
        "commands": [l.rstrip("\n") for l in open(pilot / "commands.txt")],
        "models": models,
        "prep_corpus": json.load(open(pilot / "inputs/prep_corpus_receipt.json")),
        "prep_bios": json.load(open(pilot / "inputs/bios_corpus.parquet.receipt.json")),
        "runs": runs,
        "diagnostics_list_free": json.load(open(pilot / "diag_dims.json")),
        "score": json.load(open(pilot / "score.json")),
        "chain_log": [l.rstrip("\n") for l in open(pilot / "logs/chain.log")],
        "library_versions_pip_freeze": pip.split(),
        "inputs_extra_sha256": {str(p): sha256(p) for p in [pilot / "inputs/list_owner_split.tsv",
                                                             pilot / "inputs/bios_raw_all_producers.parquet",
                                                             pilot / "inputs/list_owners.txt"]},
    }
    json.dump(receipt, open(pilot / "receipt.json", "w"), indent=1)
    print(pilot / "receipt.json", sha256(pilot / "receipt.json"))


if __name__ == "__main__":
    main(sys.argv[1])
