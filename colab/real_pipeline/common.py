"""Real-data/provenance contracts shared by the offline Colab stages."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

MODELS = {
    "political": ("mlburnham/Political_DEBATE_large_v1.0", "1a3aff1ecb97ad93a6de598f7414b1707de7e7c3"),
    "qwen": ("Qwen/Qwen2.5-32B-Instruct-AWQ", "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"),
    "nli": ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2"),
    "student": ("jhu-clsp/ettin-encoder-68m", "ac19ae4bc51093b31c475665ac872a936d056cc2"),
}
HEALTH = ("politicalness", "heat", "engagement_bait", "spam")
POLITICAL_HYPOTHESIS = "This text is about politics, government, elections, or public policy."
RUBRICS = {
    "outraged": "The author expresses anger, indignation, or outrage.",
    "critical": "The author attacks, condemns, or disparages a person, group, or institution.",
    "engagement_bait": "The author manipulates readers into replying, liking, or sharing mainly to increase engagement.",
    "spam": "This text is unsolicited or repetitive promotional spam, a scam, or deceptive advertising.",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def source_inventory():
    root = Path(__file__).resolve().parents[1]
    sources = sorted(root.glob("*.py")) + sorted((root / "real_pipeline").glob("*.py")) + sorted(root.glob("requirements-*.txt"))
    return {str(path.relative_to(root)): sha256(path) for path in sources}


def environment():
    versions = {}
    for name in ("torch", "transformers", "vllm", "onnx", "onnxruntime", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"platform": platform.platform(), "packages": versions, "implementation_source_sha256": source_inventory()}


def read_rows(path, limit=None):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        if limit is not None:
            batches = pq.ParquetFile(path).iter_batches(batch_size=limit, use_threads=False)
            batch = next(batches, None)
            return batch.to_pylist() if batch is not None else []
        return pq.read_table(path).to_pylist()
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def validate_posts(rows, allow_teachers=False):
    if not rows:
        raise ValueError("Empty real-post bank")
    uris = set()
    for row in rows:
        if not allow_teachers and any(key.startswith("teacher_") for key in row):
            raise ValueError("Input bank must be unlabeled; teacher columns require pipeline provenance")
        if not str(row.get("uri", "")).startswith("at://did:"):
            raise ValueError("Missing original AT URI")
        if not row.get("author_did", "").startswith("did:") or not row.get("text", "").strip():
            raise ValueError("Missing original author/text")
        if row["uri"] in uris:
            raise ValueError("Duplicate post URI")
        uris.add(row["uri"])


def require_disjoint(rows, holdout):
    authors = {row["author_did"] for row in rows}
    overlap = authors & {row["author_did"] for row in holdout}
    if overlap:
        raise ValueError(f"Training/holdout author overlap: {len(overlap)}")


def local_guard(mode, rows, device):
    # Bulk CPU/MPS training is intentionally prohibited, including on macOS CUDA shims.
    if mode != "smoke" and (platform.system() == "Darwin" or not device.startswith("cuda")):
        raise ValueError("Full labeling/training requires a separate CUDA Colab runtime")
    if mode == "smoke" and len(rows) != 32:
        raise ValueError("Local smoke must use exactly 32 real rows")


def validate_pairs(path, holdout_path):
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("graph_version") != "phase1-v1" or manifest.get("supplemental_edges_excluded") is not True:
        raise ValueError("Pairs require archive-only phase1-v1 provenance; v0 pairs are invalid")
    if manifest.get("sha256") != sha256(path):
        raise ValueError("Pair digest mismatch")
    if len(manifest.get("known_for_sha256", "")) != 64:
        raise ValueError("Missing exact KnownFor digest")
    if manifest.get("holdout_sha256") != sha256(holdout_path):
        raise ValueError("Pair holdout provenance mismatch")
    holdout_authors = {r["author_did"] for r in read_rows(holdout_path)}
    rows = read_rows(path)
    for row in rows:
        for prefix in ("anchor", "positive", "negative"):
            if not str(row.get(f"{prefix}_uri", "")).startswith("at://did:") or not row.get(f"{prefix}_text"):
                raise ValueError("Pairs must contain real original post URIs and texts")
            if row.get(f"{prefix}_author_did") in holdout_authors:
                raise ValueError("Pair bank leaks a holdout author")
        if len({row[f"{p}_uri"] for p in ("anchor", "positive", "negative")}) != 3:
            raise ValueError("Pair bank contains a repeated post within a triplet")
        if row["anchor_author_did"] == row["positive_author_did"]:
            raise ValueError("Positive pairs must have distinct authors")
        if row["community_id"] == row["negative_community_id"]:
            raise ValueError("Negative belongs to anchor community")
    if not rows:
        raise ValueError("Pair bank empty")
    return rows, manifest
