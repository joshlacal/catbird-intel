"""Build the extra gold pool and the collision-trap pairs from real exported posts.

Run with pyarrow available, e.g.:
    uv run --with pyarrow python3 -m labeling build-pools

Extra posts come from author bucket 19 (`human_final` in teacher_harness/splits.py),
which never enters teacher training, one post per author. Trap pairs reuse the
holdout's collision-trap stratum and top up from bucket 19 with the same keyword
rule the holdout export used (ops/export_training_banks_v1.py). Keywords pick
posts; they never supply labels.

Parent/quote context resolves from local files only (hydration file, holdout,
150k bank). Unresolved references show on the card as unavailable; they are
counted in the manifest.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import INTEL_DIR, POOLS_DIR
from .pools import (
    EXTRA_CONTEXT_PATH,
    EXTRA_POOL_PATH,
    HOLDOUT_PATH,
    HYDRATION_PATH,
    TRAP_PAIRS_PATH,
    load_pool,
    read_jsonl,
    sha256_file,
)

BANK_PATH = Path.home() / "catbird-lake-local/drive-sync-staging/training-banks/phase3_health_training_150k_v1_unlabeled.parquet"
BANK_SHA256 = "4bea16ece379a1c2f0e8d559bca97357b9eb653cc3f8083681d984cd07e0b538"
HUMAN_FINAL_BUCKET = 19
# Bucket 19 holds too few trap-keyword posts for 300 pairs; top up from bucket 16 (`dev`),
# which is also excluded from teacher training.
TRAP_TOP_UP_BUCKETS = (19, 16)
MANIFEST_PATH = POOLS_DIR / "pools_v1.manifest.json"

# Copied from ops/export_training_banks_v1.py so selection matches the holdout stratum.
POLITICAL_KW = ["trump", "biden", "election", "congress", "senate", "democrat", "republican", "gop",
                "maga", "tariff", "gaza", "israel", "ukraine", "putin", "immigration", "ice ", "supreme court",
                "governor", "vote", "policy", "fascis", "liberal", "conservative", "labour", "tory"]
TRAP_KW = ["rust", "swift", "macro", "lab ", "homelab", "football", "indie", "espresso", "#humor",
           "wrestling", "state", "party", "house"]

# Two candidate communities per trap keyword (provisional; DECISIONS.md 2026-10-23).
# "0 = neither" covers every other reading.
TRAP_CANDIDATES: dict[str, tuple[str, str]] = {
    "rust": ("Rust programming language (developers, crates, compilers)", "Rust, the multiplayer survival video game"),
    "swift": ("Swift / iOS and Apple-platform developers", "Taylor Swift fans (Swifties)"),
    "macro": ("Macroeconomics (inflation, rates, markets)", "Macro photography (extreme close-ups of insects, flowers)"),
    "lab ": ("Laboratory science (wet lab, research, academia)", "Homelab self-hosting (servers, NAS, networking)"),
    "homelab": ("Homelab self-hosting (servers, NAS, networking)", "Laboratory science (wet lab, research, academia)"),
    "football": ("American football (NFL, college football)", "Association football / soccer"),
    "indie": ("Indie video games and developers", "Indie music and bands"),
    "espresso": ("Coffee and espresso enthusiasts", "Sabrina Carpenter fans ('Espresso')"),
    "#humor": ("Comedians and comedy writing", "Meme and joke repost accounts"),
    "wrestling": ("Professional wrestling (WWE, AEW)", "Amateur, collegiate or Olympic wrestling"),
    "state": ("US state government and state politics", "Software state (state management, state machines)"),
    "party": ("Political parties and partisan politics", "Parties, nightlife and celebrations"),
    "house": ("US House of Representatives / Congress", "House music, DJs and dance music"),
}


def first_keyword(text: str, keywords: list[str]) -> str | None:
    lowered = text.lower()
    return next((kw for kw in keywords if kw in lowered), None)


def is_trap_candidate(text: str) -> bool:
    return first_keyword(text, TRAP_KW) is not None and first_keyword(text, POLITICAL_KW) is None


def _bank_rows(bank_path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq  # only the builder needs pyarrow

    digest = sha256_file(bank_path)
    if digest != BANK_SHA256:
        raise ValueError(f"Bank hash {digest} != frozen {BANK_SHA256}")
    return pq.read_table(bank_path).to_pylist()


def _eligible(row: dict[str, Any], buckets: tuple[int, ...] = (HUMAN_FINAL_BUCKET,)) -> bool:
    return (
        int(row["author_split_bucket"]) in buckets
        and "en" in list(row.get("langs") or [])
        and bool(str(row.get("text") or "").strip())
    )


def _strip(row: dict[str, Any]) -> dict[str, Any]:
    """Pool rows keep the holdout's columns; regex baseline scores are deliberately dropped."""
    keep = ("uri", "author_did", "author_split_bucket", "created_at", "text", "langs", "post_type",
            "reply_parent_uri", "reply_root_uri", "quote_uri", "embed_type", "image_count", "has_video",
            "external_uri", "facet_tags")
    out = {k: row.get(k) for k in keep}
    out["created_at"] = str(out["created_at"])
    out["langs"] = list(out["langs"] or [])
    out["facet_tags"] = list(out["facet_tags"] or [])
    return out


def one_per_author(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    by_author: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: r["uri"]):
        by_author.setdefault(row["author_did"], []).append(row)
    picked = [rng.choice(group) for _, group in sorted(by_author.items())]
    rng.shuffle(picked)
    return picked


def select_extra(rows: list[dict[str, Any]], *, seed: int, n_pilot: int, n_extra: int) -> list[dict[str, Any]]:
    candidates = one_per_author([r for r in rows if _eligible(r) and not is_trap_candidate(r["text"])], random.Random(seed))
    if len(candidates) < n_pilot + n_extra:
        raise ValueError(f"Only {len(candidates)} eligible bucket-{HUMAN_FINAL_BUCKET} authors")
    out = []
    for i, row in enumerate(candidates[: n_pilot + n_extra]):
        out.append(_strip(row) | {"subset": "pilot50" if i < n_pilot else "extra", "stratum": "human_final_random"})
    return out


def make_trap_pairs(
    holdout: list[dict[str, Any]],
    bank: list[dict[str, Any]],
    *,
    exclude_authors: set[str],
    seed: int,
    n_total: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sources = [dict(r, trap_source="holdout_collision_trap_kw") for r in holdout if r["stratum"] == "collision_trap_kw"]
    used_authors = {r["author_did"] for r in sources} | exclude_authors
    top_up: list[dict[str, Any]] = []
    for bucket in TRAP_TOP_UP_BUCKETS:
        top_up += one_per_author(
            [r for r in bank if _eligible(r, (bucket,)) and is_trap_candidate(r["text"]) and r["author_did"] not in used_authors],
            rng,
        )
    need = n_total - len(sources)
    if len(top_up) < need:
        raise ValueError(f"Only {len(top_up)} bucket {TRAP_TOP_UP_BUCKETS} trap candidates for {need} needed")
    sources += [
        _strip(r) | {"stratum": "bank_trap_kw", "trap_source": f"bank_bucket{r['author_split_bucket']}_trap_kw"}
        for r in top_up[:need]
    ]
    pairs = []
    for row in sorted(sources, key=lambda r: r["uri"]):
        keyword = first_keyword(row["text"], TRAP_KW)
        if keyword is None:
            raise ValueError(f"{row['uri']} has no trap keyword")
        first, second = TRAP_CANDIDATES[keyword]
        swapped = rng.random() < 0.5  # randomize which description is "1" so position can't bias
        pair = dict(row)
        pair.update(
            pair_id=f"trap-{len(pairs) + 1:04d}",
            trap_keyword=keyword,
            option_1=second if swapped else first,
            option_2=first if swapped else second,
            options_swapped=swapped,
        )
        pairs.append(pair)
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


def build_pools(*, seed: int = 20261023, n_pilot: int = 50, n_extra: int = 400, n_traps: int = 300,
                bank_path: Path = BANK_PATH) -> dict[str, Any]:
    for path in (EXTRA_POOL_PATH, TRAP_PAIRS_PATH):
        if path.exists():
            raise FileExistsError(f"{path} exists; pools are frozen once built (move it aside to rebuild)")
    holdout, holdout_sha = load_pool("holdout600")
    bank = _bank_rows(bank_path)
    extra = select_extra(bank, seed=seed, n_pilot=n_pilot, n_extra=n_extra)
    traps = make_trap_pairs(holdout, bank, exclude_authors={r["author_did"] for r in extra}, seed=seed + 1, n_total=n_traps)

    known = {r["uri"] for r in read_jsonl(HYDRATION_PATH)} | {r["uri"] for r in holdout}
    bank_by_uri = {r["uri"]: r for r in bank}
    refs = sorted({u for r in extra + traps for u in (r.get("reply_parent_uri"), r.get("quote_uri")) if u} - known)
    context = [{"uri": u, "text": bank_by_uri[u]["text"], "hydration_source": "bank_v1"} for u in refs if u in bank_by_uri]

    extra_sha = write_jsonl(EXTRA_POOL_PATH, extra)
    traps_sha = write_jsonl(TRAP_PAIRS_PATH, traps)
    context_sha = write_jsonl(EXTRA_CONTEXT_PATH, context)
    manifest = {
        "schema": "catbird-gold-label-pools-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "uv run --with pyarrow python3 -m labeling build-pools",
        "seed": seed,
        "inputs": {
            "bank": {"path": str(bank_path), "sha256": BANK_SHA256},
            "holdout": {"path": str(HOLDOUT_PATH.relative_to(INTEL_DIR)), "sha256": holdout_sha},
            "hydration": {"path": str(HYDRATION_PATH.relative_to(INTEL_DIR)), "sha256": sha256_file(HYDRATION_PATH)},
        },
        "outputs": {
            "extra_pool": {"path": str(EXTRA_POOL_PATH.relative_to(INTEL_DIR)), "sha256": extra_sha, "rows": len(extra),
                           "pilot50": n_pilot, "extra": n_extra},
            "trap_pairs": {"path": str(TRAP_PAIRS_PATH.relative_to(INTEL_DIR)), "sha256": traps_sha, "rows": len(traps),
                           "by_source": {src: sum(t["trap_source"] == src for t in traps) for src in sorted({t["trap_source"] for t in traps})}},
            "context": {"path": str(EXTRA_CONTEXT_PATH.relative_to(INTEL_DIR)), "sha256": context_sha,
                        "referenced_outside_hydration": len(refs), "resolved_from_bank": len(context),
                        "unresolved": len(refs) - len(context)},
        },
        "selection": {
            "extra": f"bucket {HUMAN_FINAL_BUCKET} (human_final), langs has 'en', non-empty text, not a trap-keyword post, one post per author, seeded shuffle",
            "traps": f"all holdout collision_trap_kw posts + top-up from buckets {TRAP_TOP_UP_BUCKETS} in that order by TRAP_KW and NOT POLITICAL_KW, one post per author, authors disjoint from extra pool",
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
