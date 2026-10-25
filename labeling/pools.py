"""Post pools and author-agnostic labeling cards with parent/quote context joined inline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import INTEL_DIR, POOLS_DIR

HOLDOUT_PATH = INTEL_DIR / "eval" / "phase3_holdout_600_unlabeled.jsonl"
HOLDOUT_SHA256 = "fd57d61c490cf11860c74341f4c5d16d884d65ae0a0b9dd200b867c52b6b9792"
HYDRATION_PATH = INTEL_DIR / "eval" / "pilot_holdout_parent_quote_hydration_20261021.jsonl"
EXTRA_POOL_PATH = POOLS_DIR / "human_final_extra_v1.jsonl"
TRAP_PAIRS_PATH = POOLS_DIR / "trap_pairs_v1.jsonl"
EXTRA_CONTEXT_PATH = POOLS_DIR / "context_extra_v1.jsonl"

CONTEXT_TEXT_LIMIT = 1200


@dataclass(frozen=True)
class PoolSpec:
    name: str
    path: Path
    expected_sha256: str | None = None


POOLS: dict[str, PoolSpec] = {
    "holdout600": PoolSpec("holdout600", HOLDOUT_PATH, HOLDOUT_SHA256),
    "extra_v1": PoolSpec("extra_v1", EXTRA_POOL_PATH),
    "trap_pairs_v1": PoolSpec("trap_pairs_v1", TRAP_PAIRS_PATH),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_pool(name: str, *, subset: str | None = None) -> tuple[list[dict[str, Any]], str]:
    """Rows of a registered pool (optionally one subset) and the pool file's SHA-256."""
    spec = POOLS.get(name)
    if spec is None:
        raise ValueError(f"Unknown pool {name!r}; known: {sorted(POOLS)}")
    if not spec.path.exists():
        raise FileNotFoundError(f"Pool {name} not built yet: {spec.path} (run `python3 -m labeling build-pools`)")
    digest = sha256_file(spec.path)
    if spec.expected_sha256 and digest != spec.expected_sha256:
        raise ValueError(f"Pool {name} hash {digest} != frozen {spec.expected_sha256}")
    rows = read_jsonl(spec.path)
    if subset is not None:
        rows = [r for r in rows if r.get("subset") == subset]
        if not rows:
            raise ValueError(f"Pool {name} has no rows in subset {subset!r}")
    return rows, digest


def context_paths() -> list[Path]:
    """Files whose rows can supply parent/quote text, in lookup priority order."""
    candidates = [HOLDOUT_PATH, HYDRATION_PATH, EXTRA_POOL_PATH, EXTRA_CONTEXT_PATH]
    return [p for p in candidates if p.exists()]


def load_context_index(paths: list[Path] | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """uri -> text across context files, plus each file's SHA-256 for the session receipt."""
    index: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for path in paths if paths is not None else context_paths():
        hashes[str(path.relative_to(INTEL_DIR)) if path.is_relative_to(INTEL_DIR) else str(path)] = sha256_file(path)
        for row in read_jsonl(path):
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                index.setdefault(str(row["uri"]), text)
    return index, hashes


def _context_block(ref_uri: str, index: dict[str, str]) -> dict[str, Any] | None:
    if not ref_uri:
        return None
    text = index.get(ref_uri)
    if text is None:
        return {"available": False, "text": None}
    truncated = len(text) > CONTEXT_TEXT_LIMIT
    return {"available": True, "text": text[:CONTEXT_TEXT_LIMIT], "truncated": truncated}


def _external_host(uri: str) -> str:
    if not uri:
        return ""
    try:
        return urlparse(uri).hostname or uri[:60]
    except ValueError:
        return uri[:60]


def make_card(row: dict[str, Any], index: dict[str, str]) -> dict[str, Any]:
    """Author-agnostic card: no DID, handle or avatar; mentions inside the text stay as written."""
    image_count = int(row.get("image_count") or 0)
    card: dict[str, Any] = {
        "uri": row["uri"],
        "text": row.get("text") or "",
        "post_type": row.get("post_type") or "original",
        "reply_parent": _context_block(str(row.get("reply_parent_uri") or ""), index),
        "quoted": _context_block(str(row.get("quote_uri") or ""), index),
        "media": {
            "embed_type": row.get("embed_type") or "",
            "image_count": image_count,
            "has_video": bool(row.get("has_video")),
            "external_host": _external_host(str(row.get("external_uri") or "")),
            # The v1 export carries no alt text; say so rather than implying there is none.
            "alt_text": None,
            "alt_text_exported": False,
        },
        "stratum": row.get("stratum") or row.get("subset") or "",
    }
    if "pair_id" in row:
        card["pair_id"] = row["pair_id"]
        card["options"] = [row["option_1"], row["option_2"]]
    return card
