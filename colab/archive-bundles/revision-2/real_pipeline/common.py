"""Real-data/provenance contracts shared by the offline Colab stages."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

MODELS = {
    "political": ("mlburnham/Political_DEBATE_large_v1.0", "1a3aff1ecb97ad93a6de598f7414b1707de7e7c3"),
    "qwen": ("Qwen/Qwen2.5-32B-Instruct-AWQ", "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"),
    "nli": ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2"),
    "student": ("jhu-clsp/ettin-encoder-68m", "ac19ae4bc51093b31c475665ac872a936d056cc2"),
}
HEALTH = ("politicalness", "heat", "engagement_bait", "spam")
SEALED_EXPORT_SHA256 = "18230128db1ba0347e9a3050a25a25bf14faf09944a458247d7955a1e71a5117"
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


def validate_original_post(row, prefix=""):
    key = lambda name: f"{prefix}_{name}" if prefix else name
    author, uri, text = (row.get(key(name)) for name in ("author_did", "uri", "text"))
    if not isinstance(author, str) or not author.startswith("did:") or "/" in author:
        raise ValueError("Missing original author DID")
    expected = f"at://{author}/app.bsky.feed.post/"
    if not isinstance(uri, str) or not uri.startswith(expected) or not uri[len(expected):] or "/" in uri[len(expected):]:
        raise ValueError("Original post URI does not match its author DID")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing original post text")


def validate_posts(rows, allow_teachers=False, split="train"):
    if not rows:
        raise ValueError("Empty real-post bank")
    uris = set()
    for row in rows:
        if not allow_teachers and any(key.startswith("teacher_") for key in row):
            raise ValueError("Input bank must be unlabeled; teacher columns require pipeline provenance")
        validate_original_post(row)
        bucket = row.get("author_split_bucket")
        if type(bucket) is not int or bucket not in (range(1, 20) if split == "train" else (0,)):
            raise ValueError(f"Invalid or missing {split} author split bucket")
        if row["uri"] in uris:
            raise ValueError("Duplicate post URI")
        uris.add(row["uri"])


def require_disjoint(rows, holdout):
    authors = {row["author_did"] for row in rows}
    overlap = authors & {row["author_did"] for row in holdout}
    if overlap:
        raise ValueError(f"Training/holdout author overlap: {len(overlap)}")


def load_sealed_export(holdout_path, export_manifest=None):
    if not holdout_path:
        raise ValueError("Every operation requires the canonical --holdout")
    holdout_path = Path(holdout_path)
    manifest_path = Path(export_manifest) if export_manifest else holdout_path.parent / "phase3_v1_export_manifest.json"
    if sha256(manifest_path) != SEALED_EXPORT_SHA256:
        raise ValueError("Export manifest is not the audited v1 seal; a new export needs an audited source revision")
    manifest = json.loads(manifest_path.read_text())
    if sha256(holdout_path) != manifest["holdout"]["sha256"]:
        raise ValueError("Canonical holdout digest mismatch")
    holdout = read_rows(holdout_path)
    validate_posts(holdout, split="holdout")
    if len(holdout) != manifest["holdout"]["rows"] or len({r["author_did"] for r in holdout}) != manifest["holdout"]["authors"]:
        raise ValueError("Canonical holdout row/author count mismatch")
    return manifest, holdout


def selected_uri_sha256(rows):
    return hashlib.sha256("\n".join(row["uri"] for row in rows).encode()).hexdigest()


def validate_teacher_identity(identity, teacher, source_sha256, rows, mode):
    expected_rubrics = {"politicalness": POLITICAL_HYPOTHESIS} if teacher == "political" else RUBRICS
    if (identity.get("schema") != "catbird-real-teacher-v1"
            or identity.get("teacher") != teacher
            or identity.get("model") != list(MODELS[teacher])
            or identity.get("input_sha256") != source_sha256
            or identity.get("rows") != len(rows)
            or identity.get("selected_uri_sha256") != selected_uri_sha256(rows)
            or identity.get("mode") != mode
            or identity.get("rubrics") != expected_rubrics):
        raise ValueError("Teacher identity is not bound to the sealed source, selection, model and rubric")


def validate_labeled(path, rows, seal, split="train", mode="colab"):
    manifest = json.loads(Path(str(path) + ".manifest.json").read_text())
    expected_source = seal["bank" if split == "train" else "holdout"]["sha256"]
    if (manifest.get("schema") != "catbird-real-labeled-v1"
            or manifest.get("sha256") != sha256(path)
            or manifest.get("source_sha256") != expected_source
            or manifest.get("rows") != len(rows)):
        raise ValueError("Labeled bank lacks sealed source/selection provenance or changed")
    provenance = manifest.get("teacher_provenance", [])
    teachers = [p.get("identity", {}).get("teacher") for p in provenance]
    if len(teachers) != 2 or set(teachers) not in ({"political", "qwen"}, {"political", "nli"}):
        raise ValueError("Unexpected teachers")
    for stage in provenance:
        if stage.get("status") != "complete" or not stage.get("files"):
            raise ValueError("Teacher stage was not completed")
        validate_teacher_identity(stage["identity"], stage["identity"]["teacher"], expected_source, rows, mode)
    for row in rows:
        for metric in (*HEALTH, "outraged", "critical"):
            value = row.get(f"teacher_{metric}")
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid real teacher target")
        if row["teacher_heat"] != max(row["teacher_outraged"], row["teacher_critical"]):
            raise ValueError("Heat must preserve the declared teacher aggregation")
    return manifest


def validate_input(args, rows):
    seal, holdout = load_sealed_export(args.holdout, args.export_manifest)
    labeled = args.command in ("train", "review-sheet")
    validate_posts(rows, allow_teachers=labeled, split=args.split)
    if args.command == "train" and args.split != "train":
        raise ValueError("Held-out posts can never train the student")
    if args.command == "review-sheet" and args.split != "holdout":
        raise ValueError("Human review sheets use the sealed holdout")
    if args.split == "train":
        require_disjoint(rows, holdout)
    if labeled:
        validate_labeled(args.input, rows, seal, args.split, args.mode)
    else:
        source = seal["bank" if args.split == "train" else "holdout"]
        if sha256(args.input) != source["sha256"]:
            raise ValueError("Raw input is not the canonical sealed bank/holdout")
        expected_rows = 32 if args.mode == "smoke" else source["rows"]
        if len(rows) != expected_rows:
            raise ValueError("Unexpected selected row count")
    if args.split == "holdout":
        originals = {r["uri"]: r for r in holdout}
        if set(originals) != {r["uri"] for r in rows}:
            raise ValueError("Holdout selection differs from the sealed set")
        for row in rows:
            if any(row[key] != originals[row["uri"]][key] for key in ("author_did", "text", "author_split_bucket")):
                raise ValueError("Holdout original content changed")
    return seal


def create_review_sheet(output, rows):
    output = Path(output)
    status = Path(str(output) + ".status.json")
    if output.exists() or status.exists():
        raise FileExistsError("Review sheet already exists; preserve human edits and use a new versioned output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("politicalness", "outraged", "critical", "engagement_bait", "spam")
    with output.open("x") as handle:
        for row in rows:
            reviewed = dict(row)
            reviewed.update({f"human_{name}": None for name in fields})
            reviewed.update(human_reviewed=False, reviewer=None, review_notes="")
            handle.write(json.dumps(reviewed, default=str, ensure_ascii=False) + "\n")
    with status.open("x") as handle:
        json.dump({"rows": len(rows), "human_reviewed": False, "thresholds_locked": False, "F1": None}, handle, indent=2)


def local_guard(mode, rows, device):
    # Bulk CPU/MPS training is intentionally prohibited, including on macOS CUDA shims.
    if mode != "smoke" and (platform.system() == "Darwin" or not device.startswith("cuda")):
        raise ValueError("Full labeling/training requires a separate CUDA Colab runtime")
    if mode == "smoke" and len(rows) != 32:
        raise ValueError("Local smoke must use exactly 32 real rows")


def validate_pairs(path, holdout_path, export_manifest=None):
    _, holdout = load_sealed_export(holdout_path, export_manifest)
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
    if manifest.get("holdout_bucket_excluded") != 0 or manifest.get("holdout_author_overlap") != 0:
        raise ValueError("Pair manifest lacks explicit holdout bucket/author exclusion")
    histogram = manifest.get("source_author_split_buckets", {})
    if (not histogram or any(str(key) not in {str(i) for i in range(1, 20)}
                             or type(value) is not int or value <= 0 for key, value in histogram.items())
            or sum(histogram.values()) != manifest.get("source_rows")
            or manifest.get("source_rows") != manifest.get("source_unique_uris")):
        raise ValueError("Pair source author bucket/count audit is incomplete")
    holdout_authors = {r["author_did"] for r in holdout}
    rows = read_rows(path)
    if manifest.get("rows") != len(rows):
        raise ValueError("Pair manifest row count mismatch")
    for row in rows:
        for prefix in ("anchor", "positive", "negative"):
            validate_original_post(row, prefix)
            bucket = row.get(f"{prefix}_author_split_bucket")
            if type(bucket) is not int or bucket not in range(1, 20):
                raise ValueError("Pair author bucket must be an explicit non-holdout integer")
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
