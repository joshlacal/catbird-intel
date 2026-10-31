#!/usr/bin/env python3
"""Build S17 Harness v2 & v0 Onboarding Catalog (`cards_v1.json <= 2 MB`, 0 safety violations, rev 3).

Reads:
- Frozen partition `cand5-full-matched` (`communities.parquet`, `known_for.parquet`, `membership.npy`)
- Locked S13 production eligibility & scores (`s13_prod_eligibility_and_scores.parquet`, flags at theta=0.5)
- Merged Oct 30 S5 live safety & PDS status sweep (`s5-sweep-20261030/{profiles.parquet, status.tsv, purge_dids.txt, card_packs_filter.tsv, receipt.json}`)
- Validated S14 compatibility matrix `H_prod_cand5_full.parquet` (`is_high_top1pct`)
- Clean O7 community-liked feeds (`s10-outputs-20261030/O7/O7.parquet`) + `cand5_card_packs.tsv`
- Clean S12 medoid post URIs (`s12-closeout-20261030/medoids.parquet`)
- Claude Opus `cand5` community labels (`claude-labels-cand5-20261028/community_labels.parquet`) + profile labels
- S17 rev 3 full-data signals (`candidate_signals.parquet`, `card_centroids.npz`, `author_feeds.sqlite`, `manual_overrides.json`):
  * S12m multimodal post-embedding & S16 Tier-1 social-head interloper + Y3-bleed eviction
  * S12m 64-bucket per-post template/duplicate slop bot eviction
  * S5 follow-churn (`followsCount`/`followersCount`) & automated firehose cadence (`>=120 posts/day`) eviction
  * Live `getAuthorFeed` repost-wall, dormant, live-template-spam, political-drift, and live-label eviction
  * Bounded coherence quality multiplier inside candidate ranking + `inspector_index.json` sidecar for the v2 UI

Reversible via `--strict-registered-only`.
"""
from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import pickle
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import s3_prep  # noqa: E402
from s6picker import inputs as s6_in, keywords as s6_kw  # noqa: E402

MAX_JSON_BYTES = 2_097_152
BUNDLE_DEFAULT = 25
POP_LIMIT = 500_000
POP_MAX = 2
SUP_MAX = 2
MIN_LIVE_POSTS = 3
PACKS_PER_CARD = 2
FEEDS_PER_CARD = 3
MEDOIDS_PER_CARD = 3
Y3_CLASSES = {
    "news",
    "politics",
    "religion",
    "sexuality_gender",
    "health",
    "ethnicity_nationality",
    "adult",
    "likely_minors",
}
OPUS_LABEL_Y3_CLASSES = {
    "politics",
    "religion",
    "sexuality_gender",
    "health",
    "adult",
    "likely_minors",
}
DROP_PROFILE_CLASSES = {"likely_minors", "adult"}

# Provisional S17 rev-2/rev-3 safety + organic-human curation rules (reversible via --strict-registered-only):
EXPLICIT_BLOCK_CIDS = {1707, 409, 1849, 1056, 1856, 2774, 2245, 2826}
RE_DESC_ADULT_MINORS = re.compile(
    r"(?:(?<!\w)18\+|(?<!\w)21\+|(?<!\w)r-?18\b|\bmdni\b|\bnsfw\b|\bonlyfans\b|\bfansly\b|"
    r"\blewd\b|\bfetish\b|\bporn\b|\bunderage\b|\bminors?\b|\badult work\b|(?<!young )\badult fan\b)",
    re.I,
)
RE_RP_UNSAFE = re.compile(
    r"\b(high[- ]school|school[- ]life|school roleplay|school oc|yokai[- ]school|magic[- ]school|"
    r"academy|university|hogwarts|idol|k-?pop|celebrity|face[- ]?claim)\b",
    re.I,
)
RE_NON_ORGANIC_CARD = re.compile(
    r"\b(image bots?|paper bots?|horoscope bots?|promo bots?|gimmick (?:bots?|accounts?)|tech bots?|"
    r"automated (?:bot|nep|repec|feeds?)|ai[- ]art (?:generators?|creators?)|"
    r"book[- ]deal(?:/promo)?|amazon affiliates?|prints and photography sellers|promoting wall art|"
    r"arxiv feeds?|news feeds, bots)\b",
    re.I,
)

RE_STRIP_NEG = re.compile(
    r"(?:\b(?:no|not|non|anti|never|without|zero|ban|fuck|don'?t|doesn'?t|won'?t)\b|🚫|❌|🚷)[^.\n;|•!]{0,65}",
    re.I,
)
RE_RAW_ADULT = re.compile(
    r"(?:🔞|(?<!\w)18\+|(?<!\w)21\+|(?<!\w)r-?18\b|\bmdni\b|\bminors?\s+dni\b|"
    r"\bminors?\s+do\s+not\s+interact\b|\badults?\s+only\b)",
    re.I,
)
RE_CLEAN_ADULT = re.compile(
    r"(?:\bnsfw\b|\bonlyfans\b|\bfansly\b|\byiff\b|\bporn\b|\bhentai\b|\beroge\b|\badult\s+content\b)",
    re.I,
)
RE_HDL_BOT = re.compile(
    r"(?:\.bot$|\.ap\.brid\.gy$|(?:^|[.-])(?:bot|rss|feed|mirror|alerts?)(?:$|[.-]))",
    re.I,
)
RE_NAME_BOT = re.compile(
    r"(?:\[bot\]|\(bot\)|\b(?:unofficial|automated|mirror|rss|repost)\s+bot\b|(?:^|\s)bot(?:$|\s)|"
    r"\[twitter/x\s*mirror\]|\(mirror\)|#(?:f4f|bluecrew|ifb|fbr)\b|@f4f\b)",
    re.I,
)
RE_BIO_BOT = re.compile(
    r"(?:\bautomated\s+(?:account|feed|posts?|bot|updates?|mirror|weather|alerts?)\b|"
    r"\bunoff?icial\s+(?:bot|mirror|feed)\b|\bmirror\s+(?:of|from|bot|account)\b|"
    r"\bbridgy\s+fed\b|\brss\s+(?:feed|bot)\b|"
    r"\bposts?\s+(?:every|once\s+an?)\s+(?:hour|day|\d+\s*(?:min|hour)))",
    re.I,
)
RE_BIO_F4F = re.compile(
    r"(?:#(?:f4f|ifb|fbr|followback|teamfollowback|bluecrew)\b|\bi\s+follow\s+back\b|"
    r"\bfollow\s+4\s+follow\b|\bfollow\s+for\s+follow\b)",
    re.I,
)
RE_BIO_AI = re.compile(
    r"(?:\bai\s+artist\b|\bai[- ]generated\s+(?:art|images|music)\b|\bmidjourney\s+(?:artist|creator)\b|"
    r"\bstable\s+diffusion\s+artist\b|\bprompt\s+artist\b)",
    re.I,
)


def screen_account_organic_and_safe(handle: str, name: str, bio: str, *, allow_feed_handle: bool = False) -> str | None:
    """Return exclusion reason if account is unlabeled 18+/NSFW/MDNI, bot/mirror, F4F ring, or AI-slop."""
    h = (handle or "").strip().lower()
    nm = (name or "").strip()
    b = (bio or "").strip()
    raw = f"{nm}\n{b}"
    cleaned = RE_STRIP_NEG.sub(" ", raw)
    if RE_RAW_ADULT.search(raw) or RE_CLEAN_ADULT.search(cleaned):
        return "bio_adult_or_mdni"
    if not allow_feed_handle:
        if RE_HDL_BOT.search(h):
            return "bot_or_mirror_handle"
        if RE_NAME_BOT.search(RE_STRIP_NEG.sub(" ", nm)) or RE_BIO_BOT.search(cleaned):
            return "automated_bot_or_mirror"
    if RE_BIO_F4F.search(cleaned):
        return "follow_back_ring_marker"
    if RE_BIO_AI.search(cleaned):
        return "ai_slop_account"
    return None


def _finite(v: float | None) -> float | None:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def screen_rev3_data_signals(
    sig: dict | None,
    live_row: dict | None,
    y3_comm_set: set[int],
) -> str | None:
    """Return exact rev-3 data/embedding/live-feed exclusion reason, or None if kept."""
    if live_row is not None:
        lv = live_row.get("live_verdict") or ""
        if lv.startswith("drop_"):
            return f"live_{lv[5:]}"

    if sig is None:
        return None

    # 1. Automated firehose cadence (>= 120 posts/day over >= 15k posts)
    ppd = float(sig.get("posts_per_day") or 0.0)
    pcnt = int(sig.get("posts_count") or 0)
    if ppd >= 120.0 and pcnt >= 15_000:
        return "automated_firehose_cadence"

    # 2. Stealth mass-follow churn / follow-farmer
    fwing = int(sig.get("follows_count") or 0)
    frat = float(sig.get("follow_ratio") or 0.0)
    if (fwing >= 5_000 and frat >= 3.0) or (fwing >= 15_000 and frat >= 1.8):
        return "stealth_follow_churn"

    # 3. Slab per-post near-duplicate template/promo bot (>= 15 embedded posts, >= 1.5 posts/day, >= 85% near-dup)
    sn = int(sig.get("slab_n_posts") or 0)
    sdup = float(sig.get("slab_near_dup_frac") or 0.0)
    snn = float(sig.get("slab_nn_mean_cos") or 0.0)
    if sn >= 15 and ppd >= 1.5 and sdup >= 0.85 and snn >= 0.93:
        return "slab_template_dup_bot"

    # 4. S12m + S16 Tier-1 off-scene / Y3-bleed interloper screen
    if sig.get("has_vec") and int(sig.get("mm_post_count") or 0) >= 5:
        sp = _finite(sig.get("sem_cos_posts"))
        t1 = _finite(sig.get("t1_cos"))
        bo_cid = int(sig.get("best_other_cid") or -1)
        bo_sem = _finite(sig.get("best_other_sem_cos"))
        if sp is not None and bo_sem is not None:
            margin = bo_sem - sp
            # Y3-sensitive bleed (account's posts strongly match a Y3 adult/politics/news/etc. cluster, not its hobby home)
            if sp < 0.08 and bo_cid in y3_comm_set and bo_sem >= 0.45 and margin >= 0.40:
                return "embedding_y3_bleed_interloper"
            # General graph-bridge wrong-scene interloper
            if (sp < -0.05 and (t1 is None or t1 < 0.12)) or (
                sp < 0.05 and (t1 is not None and t1 < 0.05) and margin >= 0.40
            ):
                return "embedding_off_scene_interloper"

    return None


def effective_candidate_score(base_s13: float, sig: dict | None, live_row: dict | None, strict: bool) -> float:
    """In rev 3, gently weight S13 score by bounded S12m+Tier-1 scene coherence and penalize borderline warnings."""
    if strict or sig is None:
        return float(base_s13)
    mult = 1.0
    if sig.get("has_vec") and int(sig.get("mm_post_count") or 0) >= 3:
        sp = _finite(sig.get("sem_cos_posts"))
        t1 = _finite(sig.get("t1_cos"))
        if sp is not None and t1 is not None:
            mult = max(0.60, min(1.32, 0.78 + 0.32 * sp + 0.22 * t1))
        elif sp is not None:
            mult = max(0.65, min(1.25, 0.82 + 0.38 * sp))
    else:
        mult *= 0.88  # prefer candidates with slab-verified content over unembedded accounts

    if int(sig.get("slab_n_posts") or 0) >= 10 and float(sig.get("slab_near_dup_frac") or 0.0) >= 0.65:
        mult *= 0.85
    if float(sig.get("follow_ratio") or 0.0) >= 2.2 and int(sig.get("follows_count") or 0) >= 3_000:
        mult *= 0.85
    if live_row is not None:
        lv = live_row.get("live_verdict") or ""
        if lv.startswith("warn_"):
            mult *= 0.84
    return float(base_s13) * mult


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


_B32_SET = set("abcdefghijklmnopqrstuvwxyz234567")


def compact_did(did: str) -> str:
    if did.startswith("did:plc:") and len(did) == 32 and set(did[8:]) <= _B32_SET:
        raw15 = base64.b32decode(did[8:].upper())
        out = base64.urlsafe_b64encode(raw15).decode("ascii").rstrip("=")
        return out
    return did


def expand_did(s: str) -> str:
    if s.startswith("did:"):
        return s
    pad = "=" * ((4 - len(s) % 4) % 4)
    raw15 = base64.urlsafe_b64decode(s + pad)
    return "did:plc:" + base64.b32encode(raw15).decode("ascii").lower()


def compact_pack_uri(u: str) -> str:
    parts = u.split("/")
    if len(parts) == 5 and parts[0] == "at:" and parts[1] == "" and parts[3] == "app.bsky.graph.starterpack" and not parts[2].startswith("did:web:"):
        return f"{compact_did(parts[2])}/{parts[4]}"
    return u


def expand_pack_uri(s: str) -> str:
    if s.startswith("at://"):
        return s
    cdid, rkey = s.split("/", 1)
    return f"at://{expand_did(cdid)}/app.bsky.graph.starterpack/{rkey}"


def compact_feed_uri(u: str) -> str:
    parts = u.split("/")
    if len(parts) == 5 and parts[0] == "at:" and parts[1] == "" and parts[3] == "app.bsky.feed.generator" and not parts[2].startswith("did:web:"):
        return f"{compact_did(parts[2])}/{parts[4]}"
    return u


def expand_feed_uri(s: str) -> str:
    if s.startswith("at://"):
        return s
    cdid, rkey = s.split("/", 1)
    return f"at://{expand_did(cdid)}/app.bsky.feed.generator/{rkey}"


def compact_handle(h: str) -> str:
    if h.endswith(".bsky.social"):
        return h[:-12] + "~"
    return h


def expand_handle(s: str) -> str:
    if s.endswith("~"):
        return s[:-1] + ".bsky.social"
    return s


def load_comparator_cached(out_dir: pathlib.Path, live_ok_map: dict[str, dict], safety_excl: set[str]) -> dict:
    comp_path = out_dir / "comparator.json"
    if comp_path.exists():
        try:
            return json.loads(comp_path.read_text())
        except Exception:
            pass
    url = "https://public.api.bsky.app/xrpc/app.bsky.unspecced.getSuggestedOnboardingUsers?limit=25"
    req = urllib.request.Request(url, headers={"User-Agent": "catbird-s17-harness/2.0", "Accept": "application/json"})
    checked_at = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            status_code = resp.status
            body_sha = hashlib.sha256(raw).hexdigest()
            payload = json.loads(raw.decode("utf-8", errors="replace"))
            actors = payload.get("actors") or []
            kept = []
            for a in actors:
                if not isinstance(a, dict):
                    continue
                d = a.get("did", "")
                if not d.startswith("did:") or d in safety_excl:
                    continue
                labels = [l.get("val", "") for l in (a.get("labels") or []) if isinstance(l, dict) and not l.get("neg")]
                if labels or a.get("handle") == "handle.invalid":
                    continue
                kept.append(
                    {
                        "did": d,
                        "handle": a.get("handle", ""),
                        "name": (a.get("displayName") or "")[:50],
                        "followers": int(a.get("followersCount") or (live_ok_map.get(d, {}).get("followersCount") or 0)),
                        "bio": " ".join((a.get("description") or "").split())[:100],
                    }
                )
            return {
                "available": len(kept) > 0,
                "http_status": status_code,
                "checked_utc": checked_at,
                "raw_sha256": body_sha,
                "raw_actors_returned": len(actors),
                "clean_actors_kept": len(kept),
                "actors": kept[:25],
                "note": "Public unauthenticated getSuggestedOnboardingUsers succeeded." if kept else "Returned 0 clean actors.",
            }
    except urllib.error.HTTPError as e:
        err_body = e.read() if hasattr(e, "read") else b""
        return {
            "available": False,
            "http_status": int(e.code),
            "checked_utc": checked_at,
            "raw_sha256": hashlib.sha256(err_body).hexdigest(),
            "raw_actors_returned": 0,
            "clean_actors_kept": 0,
            "actors": [],
            "note": f"Unauthenticated getSuggestedOnboardingUsers returned HTTP {e.code} (requires user auth); deferred to S18 authenticated dev build per plan row S17.",
        }
    except Exception as e:
        return {
            "available": False,
            "http_status": 0,
            "checked_utc": checked_at,
            "raw_sha256": "",
            "raw_actors_returned": 0,
            "clean_actors_kept": 0,
            "actors": [],
            "note": f"Comparator probe error ({type(e).__name__}: {e}); deferred to S18.",
        }


def collect_raw_response_hashes(lake: pathlib.Path, s5_dir: pathlib.Path, out_dir: pathlib.Path) -> tuple[list[str], str]:
    existing = out_dir / "raw_response_hashes.txt"
    hashes: list[str] = []
    if existing.exists():
        with open(existing) as fh:
            hashes = [ln.strip() for ln in fh if ln.strip()]
    if not hashes:
        s6_hashes = lake / "s6-picker-20261028/raw_response_hashes.txt"
        if s6_hashes.exists():
            with open(s6_hashes) as fh:
                hashes = [ln.strip() for ln in fh if ln.strip()]
        work_dir = s5_dir / "work"
        if work_dir.exists():
            for dp, _dn, fn in os.walk(work_dir):
                if "_trash" in dp:
                    continue
                for n in sorted(fn):
                    if n.endswith(".jsonl"):
                        p = pathlib.Path(dp) / n
                        with open(p, "rb") as fh:
                            for line in fh:
                                if line.strip():
                                    hashes.append(hashlib.sha256(line.rstrip(b"\n")).hexdigest())
    if not hashes:
        hashes.append(sha256_file(s5_dir / "profiles.parquet"))
    agg_sha = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    return hashes, agg_sha


def main() -> None:
    t0 = time.time()
    lake = pathlib.Path.home() / "catbird-lake-local"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(lake / "s17-picker-v1-20261030"))
    ap.add_argument(
        "--strict-registered-only",
        action="store_true",
        help="Disable provisional S17 rev-2/rev-3 safety, embedding, and live-feed curation rules.",
    )
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cand5_dir = lake / "s3/matched/cand5-full-matched"
    s5_dir = lake / "s5-sweep-20261030"
    s10_dir = lake / "s10-outputs-20261030"
    s12_dir = lake / "s12-closeout-20261030"
    s13_dir = lake / "s13-elig-20261030"
    s14_dir = lake / "s14-hostility-20261030"
    s16_dir = lake / "s16-s17-20261030"
    claude_comm_path = lake / "claude-labels-cand5-20261028/community_labels.parquet"

    s6_in.set_partition("cand5", str(claude_comm_path))

    # 1. Verify S5 positive controls
    s5_receipt = json.loads((s5_dir / "receipt.json").read_text())
    ctrl = s5_receipt.get("controls", {})
    controls_map = {
        "labelled": bool(ctrl.get("labelled", {}).get("caught")),
        "taken_down": bool(ctrl.get("taken_down", {}).get("caught")),
        "hide_from_recs_live_getRecord": bool(ctrl.get("hide_from_recs_live_getRecord", {}).get("caught")),
        "no_unauthenticated": bool(ctrl.get("no_unauthenticated", {}).get("caught")),
        "scan_trusted": bool(ctrl.get("scan_trusted")),
    }
    if not all(controls_map.values()):
        sys.exit(f"ABORT: S5 controls not all True: {controls_map}")

    # Load optional manual overrides from Inspector UI
    overrides_path = out_dir / "manual_overrides.json"
    manual_excl_dids: dict[str, str] = {}
    manual_excl_cids: dict[int, str] = {}
    if not args.strict_registered_only and overrides_path.exists():
        try:
            ov = json.loads(overrides_path.read_text())
            manual_excl_dids = {str(k): str(v or "manual_inspector_exclude") for k, v in (ov.get("excluded_dids") or {}).items()}
            manual_excl_cids = {int(k): str(v or "manual_inspector_exclude") for k, v in (ov.get("excluded_cards") or {}).items()}
        except Exception:
            pass

    # 2. Load live profiles.parquet + safety exclusion sets
    purge_dids = s6_in.did_set(s5_dir / "purge_dids.txt")
    base_excl = s6_in.did_set(s13_dir / "base_safety_exclude_dids.txt")
    rule2_excl = s6_in.did_set(s5_dir / "eligibility_rule2_exclude_dids.txt")
    optout_dids = s6_in.did_set(lake / "s5-declarations-20261027/user_intents_optout_dids.tsv")
    hfr_dids = s6_in.did_set(lake / "s5-declarations-20261027/hide_from_recs_union.tsv")

    inactive_status: set[str] = set()
    with open(s5_dir / "status.tsv") as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1] == "false":
                inactive_status.add(parts[0])

    hard_safety_excl = purge_dids | base_excl | rule2_excl | optout_dids | inactive_status

    t_prof = pq.read_table(
        s5_dir / "profiles.parquet",
        columns=[
            "did",
            "found",
            "handle",
            "postsCount",
            "followersCount",
            "followsCount",
            "no_unauthenticated",
            "any_mod_label",
            "hard_label",
            "adult_graphic",
            "self_labels",
            "hide_from_recs",
            "repo_status",
        ],
    )
    live_status_by_did: dict[str, str] = {}
    live_prof_by_did: dict[str, dict] = {}
    for r in t_prof.to_pylist():
        d = r["did"]
        if r.get("hide_from_recs"):
            hfr_dids.add(d)
        if not r["found"] or r["repo_status"] != "active" or d in inactive_status or d in purge_dids:
            live_status_by_did[d] = "missing_or_inactive"
            continue
        if r["no_unauthenticated"]:
            live_status_by_did[d] = "no_unauth"
            continue
        if r["any_mod_label"] or r["hard_label"] or r["adult_graphic"] or len(r["self_labels"] or []) > 0:
            live_status_by_did[d] = "labelled"
            continue
        if not r["handle"] or r["handle"] == "handle.invalid":
            live_status_by_did[d] = "invalid_handle"
            continue
        live_status_by_did[d] = "ok"
        live_prof_by_did[d] = {
            "handle": r["handle"],
            "postsCount": int(r["postsCount"] or 0),
            "followersCount": int(r["followersCount"] or 0),
            "followsCount": int(r["followsCount"] or 0),
        }
    del t_prof

    snap = s6_in.profile_snapshot()  # did -> (display_name, description)
    claude_c, claude_p = s6_in.claude_labels()
    claude_comm_extra: dict[int, dict] = {}
    t_cl_extra = pq.read_table(
        claude_comm_path,
        columns=["community", "label", "label_native", "description", "community_type", "onboarding_fit", "confidence"],
    )
    for rc in t_cl_extra.to_pylist():
        claude_comm_extra[int(rc["community"])] = {
            "label": (rc.get("label") or "").strip(),
            "label_native": (rc.get("label_native") or "").strip(),
            "description": " ".join((rc.get("description") or "").split()),
            "community_type": (rc.get("community_type") or "").strip(),
            "onboarding_fit": (rc.get("onboarding_fit") or "").strip(),
            "confidence": (rc.get("confidence") or "").strip(),
        }
    del t_cl_extra

    # 3. Load cand5 partition, locked S13 scores, and S17 rev-3 signals + live feeds
    comm_rows = pq.read_table(cand5_dir / "communities.parquet", columns=["community", "scene_id", "size", "top_dids"]).to_pylist()
    full_dids = [ln.rstrip("\n").split("\t")[1] for ln in open(cand5_dir / "producers.tsv")]
    n_prod = len(full_dids)
    did2pid = {d: i for i, d in enumerate(full_dids)}

    t_kf = pq.read_table(cand5_dir / "known_for.parquet", columns=["pid", "did", "community", "rank"])
    pid_kf = t_kf["pid"].to_numpy().astype(np.int64)
    com_kf = t_kf["community"].to_numpy().astype(np.int64)
    rk_kf = t_kf["rank"].to_numpy()

    pid_home = np.full(n_prod, -1, dtype=np.int64)
    mask_r0 = rk_kf == 0
    pid_home[pid_kf[mask_r0]] = com_kf[mask_r0]
    home_by_did = {full_dids[p]: int(pid_home[p]) for p in np.flatnonzero(pid_home >= 0).tolist()}

    by_pid_comms: dict[int, list[int]] = collections.defaultdict(list)
    for p_i, c_i in zip(pid_kf.tolist(), com_kf.tolist()):
        by_pid_comms[p_i].append(c_i)
    co_counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for cs in by_pid_comms.values():
        if len(cs) > 1:
            for a in cs:
                for b in cs:
                    if a != b:
                        co_counts[a][b] += 1
    del by_pid_comms

    memb_full = np.load(cand5_dir / "membership.npy").astype(np.int64)
    fl05_full = np.load(s13_dir / "flags_full_theta05.npz")
    raw_bad_full = (
        fl05_full["follow_back"].astype(bool)
        | fl05_full["follow_train"].astype(bool)
        | fl05_full["low_post_follow_heavy"].astype(bool)
    )
    comm_bad_full: dict[int, bool] = {}
    for r in comm_rows:
        raws = [int(memb_full[did2pid[d]]) for d in r["top_dids"] if d in did2pid]
        maj = collections.Counter(raws).most_common(1)[0][0] if raws else -1
        comm_bad_full[int(r["community"])] = bool(maj >= 0 and raw_bad_full[maj])

    t_s13 = pq.read_table(
        s13_dir / "s13_prod_eligibility_and_scores.parquet",
        columns=["pid", "eligible", "supplement", "score"],
    )
    s13_elig = t_s13["eligible"].to_numpy().astype(bool)
    s13_sup = t_s13["supplement"].to_numpy().astype(bool)
    s13_score = t_s13["score"].to_numpy().astype(np.float64)
    _, light_eng = s6_in.load_quality()

    t_h = pq.read_table(s14_dir / "H_prod_cand5_full.parquet", columns=["c_actor", "c_target", "is_high_top1pct"])
    high_h_pairs: set[tuple[int, int]] = set()
    for ca, ct, hi in zip(t_h["c_actor"].to_pylist(), t_h["c_target"].to_pylist(), t_h["is_high_top1pct"].to_pylist()):
        if hi:
            high_h_pairs.add((int(ca), int(ct)))
            high_h_pairs.add((int(ct), int(ca)))

    _a_owners, a_names = s6_in.a_lists(cache_dir, home_by_did)
    all_names = s6_in.all_list_names()

    # Determine Y3-sensitive community set across ALL cand5 communities (used both for card filtering and Y3-bleed interloper check)
    y3_comm_set: set[int] = set()
    comm_sens_classes: dict[int, list[str]] = {}
    for r in comm_rows:
        c = int(r["community"])
        cl = claude_c.get(c) or {}
        cx = claude_comm_extra.get(c) or {}
        a_top = [n for n, _k in a_names.get(c, [])]
        opus_label = (cl.get("label") or cx.get("label") or "").strip()
        label_native = (cx.get("label_native") or "").strip()
        names_screened = a_top[:6] + all_names.get(c, [])[:6]
        kw_classes = {k.replace("/", "_").replace("-", "_") for nm in names_screened for k in s6_kw.classes(nm)}
        opus_classes = {s.replace("/", "_").replace("-", "_") for s in (cl.get("sensitivity") or []) if s and s != "none"}
        opus_lbl_kw = (
            {
                k.replace("/", "_").replace("-", "_")
                for nm in (opus_label, label_native)
                if nm
                for k in s6_kw.classes(nm)
            }
            & OPUS_LABEL_Y3_CLASSES
            if not args.strict_registered_only
            else set()
        )
        su = sorted((kw_classes | opus_classes | opus_lbl_kw) & Y3_CLASSES)
        comm_sens_classes[c] = su
        if su:
            y3_comm_set.add(c)

    # Load S17 rev 3 candidate_signals.parquet + card_centroids.npz + author_feeds.sqlite
    sig_by_pid: dict[int, dict] = {}
    sig_pq_path = out_dir / "candidate_signals.parquet"
    if sig_pq_path.exists():
        ts = pq.read_table(sig_pq_path)
        for row_s in ts.to_pylist():
            sig_by_pid[int(row_s["pid"])] = row_s

    card_coh_map: dict[int, tuple[float, float]] = {}
    cent_npz_path = out_dir / "card_centroids.npz"
    if cent_npz_path.exists():
        c_npz = np.load(cent_npz_path)
        for ci, cp, ct in zip(c_npz["cids"].tolist(), c_npz["coh_posts"].tolist(), c_npz["coh_t1"].tolist()):
            card_coh_map[int(ci)] = (round(float(cp), 4), round(float(ct), 4))

    live_feed_by_did: dict[str, dict] = {}
    af_db_path = out_dir / "author_feeds.sqlite"
    if af_db_path.exists():
        conn = sqlite3.connect(str(af_db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        for r_af in conn.execute(
            "SELECT did, avatar_url, n_items, n_orig, n_quote, n_reply, n_repost, repost_frac, reply_frac, "
            "authored_frac, days_since_last_any, days_since_last_authored, top_ext_domain, same_domain_frac, "
            "live_dup_text_frac, live_pol_frac, median_likes, live_sem_cos, live_verdict FROM author_feeds"
        ):
            live_feed_by_did[r_af["did"]] = dict(r_af)
        conn.close()

    # Screen accounts (rev 2 bio/handle + rev 3 embedding/slab/churn/cadence/live-feed + manual overrides)
    acct_excl_reason_by_did: dict[str, str] = {}
    feed_creator_excl_dids: set[str] = set()
    evicted_examples: list[dict] = []

    if not args.strict_registered_only:
        for d, lp in live_prof_by_did.items():
            nm, bio = snap.get(d, ("", ""))
            rsn = manual_excl_dids.get(d) or screen_account_organic_and_safe(lp["handle"], nm, bio, allow_feed_handle=False)
            if not rsn:
                p_i = did2pid.get(d)
                sig_r = sig_by_pid.get(p_i) if p_i is not None else None
                rsn = screen_rev3_data_signals(sig_r, live_feed_by_did.get(d), y3_comm_set)
            if rsn:
                acct_excl_reason_by_did[d] = rsn
            rsn_fd = manual_excl_dids.get(d) or screen_account_organic_and_safe(lp["handle"], nm, bio, allow_feed_handle=True)
            if rsn_fd:
                feed_creator_excl_dids.add(d)

    (out_dir / "non_organic_exclude_dids.txt").write_text(
        "".join(f"{d}\t{acct_excl_reason_by_did[d]}\n" for d in sorted(acct_excl_reason_by_did))
    )

    # 4. Load starter packs, O7 feeds, and clean S12 medoids
    pack_keep_uris: set[str] = set()
    with open(s5_dir / "card_packs_filter.tsv") as fh:
        for ln in fh:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[-1] == "keep":
                pack_keep_uris.add(parts[1])

    cand5_offers: dict[int, list[dict]] = collections.defaultdict(list)
    with open(s5_dir / "cand5_card_packs.tsv") as fh:
        for ln in fh:
            c_s, uri_s, sh_s, n_s = ln.rstrip("\n").split("\t")[:4]
            if uri_s in pack_keep_uris:
                cand5_offers[int(c_s)].append({"uri": uri_s, "share": float(sh_s), "n": int(n_s)})

    wanted_pack_uris = {p["uri"] for ps in cand5_offers.values() for p in ps}
    pack_members_by_uri: dict[str, list[str]] = {}
    with open(cand5_dir / "audit-mass-followers/packs.pkl", "rb") as fh:
        for uri_s, mems, _pids in pickle.load(fh):
            if uri_s in wanted_pack_uris:
                pack_members_by_uri[uri_s] = sorted(set(mems))
    pnames = s6_in.pack_names(cache_dir)

    fg_meta = {
        f"at://{r['did']}/app.bsky.feed.generator/{r['rkey']}": r
        for r in s6_in.feedgens()
    }
    t_o7 = pq.read_table(s10_dir / "O7/O7.parquet")
    o7_by_comm: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    for f_uri, p_map, c_i, n_lk in zip(
        t_o7["feed_uri"].to_pylist(),
        t_o7["partition_map"].to_pylist(),
        t_o7["community"].to_pylist(),
        t_o7["n_likers"].to_pylist(),
    ):
        if p_map == "cand5" and int(n_lk) >= 10:
            o7_by_comm[int(c_i)].append((int(n_lk), f_uri))

    medoid_by_did: dict[str, list[str]] = {}
    t_med = pq.read_table(s12_dir / "medoids.parquet", columns=["did", "uris"])
    for d_s, uris_l in zip(t_med["did"].to_pylist(), t_med["uris"].to_pylist()):
        if (
            d_s not in hfr_dids
            and d_s not in hard_safety_excl
            and d_s not in acct_excl_reason_by_did
            and live_status_by_did.get(d_s) == "ok"
        ):
            clean_u = [u for u in (uris_l or []) if isinstance(u, str) and u.startswith(f"at://{d_s}/app.bsky.feed.post/")]
            if clean_u:
                medoid_by_did[d_s] = clean_u

    # 5. Evaluate every cand5 community
    pids_by_comm: dict[int, list[int]] = collections.defaultdict(list)
    for p_i in np.flatnonzero(pid_home >= 0).tolist():
        pids_by_comm[int(pid_home[p_i])].append(p_i)

    excluded_cards: dict[str, list[int]] = collections.defaultdict(list)
    drop_counts: collections.Counter = collections.Counter()
    pack_feed_stats: collections.Counter = collections.Counter()

    kept_cards_raw: list[dict] = []
    packs_detail_out: dict[str, dict] = {}
    evicted_by_reason: dict[str, list[tuple[float, dict]]] = collections.defaultdict(list)

    for r in comm_rows:
        c = int(r["community"])
        if comm_bad_full.get(c, False):
            excluded_cards["flagged_junk_or_ring"].append(c)
            continue

        cl = claude_c.get(c) or {}
        cx = claude_comm_extra.get(c) or {}
        a_top = [n for n, _k in a_names.get(c, [])]
        opus_label = (cl.get("label") or cx.get("label") or "").strip()
        full_desc = cx.get("description") or " ".join((cl.get("description") or "").split())
        comm_type = cx.get("community_type") or ""
        onb_fit = cx.get("onboarding_fit") or ""
        conf_lvl = cx.get("confidence") or ""
        if not a_top and not opus_label:
            excluded_cards["unlabelled"].append(c)
            continue

        sens_union = comm_sens_classes.get(c, [])

        # Candidate filtering
        elig_pids: list[int] = []
        card_would_keep = (not sens_union) and (c not in EXPLICIT_BLOCK_CIDS) and (c not in manual_excl_cids)
        for p_i in pids_by_comm.get(c, []):
            if not s13_elig[p_i]:
                drop_counts["s13_ineligible"] += 1
                continue
            if raw_bad_full[memb_full[p_i]]:
                drop_counts["raw_community_flagged"] += 1
                continue
            d = full_dids[p_i]
            if d in hard_safety_excl:
                drop_counts["hard_safety_or_purge"] += 1
                continue
            if DROP_PROFILE_CLASSES & claude_p.get(d, set()):
                drop_counts["claude_profile_minors_or_adult"] += 1
                continue
            st = live_status_by_did.get(d, "unchecked")
            if st != "ok":
                drop_counts[f"live_{st}"] += 1
                continue
            lp = live_prof_by_did[d]
            if lp["postsCount"] < MIN_LIVE_POSTS:
                drop_counts["live_postsCount_lt_3"] += 1
                continue
            acct_rsn = acct_excl_reason_by_did.get(d)
            if acct_rsn:
                drop_counts[f"acct_{acct_rsn}"] += 1
                if card_would_keep and acct_rsn not in ("bio_adult_or_mdni",):
                    sig_r = sig_by_pid.get(p_i) or {}
                    nm_e, bio_e = snap.get(d, ("", ""))
                    bo_cid = int(sig_r.get("best_other_cid") or -1)
                    bo_lbl = (claude_comm_extra.get(bo_cid) or {}).get("label", "") if bo_cid >= 0 else ""
                    lf_r = live_feed_by_did.get(d) or {}
                    evicted_by_reason[acct_rsn].append(
                        (
                            -float(s13_score[p_i]),
                            {
                                "did": d,
                                "handle": lp["handle"],
                                "name": nm_e[:50],
                                "bio": " ".join(bio_e.split())[:220],
                                "followers": lp["followersCount"],
                                "follows": lp["followsCount"],
                                "posts": lp["postsCount"],
                                "reason": acct_rsn,
                                "card_cid": c,
                                "card_title": opus_label or (" · ".join(a_top[:2])),
                                "s13_score": round(float(s13_score[p_i]), 3),
                                "sem_cos_posts": round(float(sig_r["sem_cos_posts"]), 3) if _finite(sig_r.get("sem_cos_posts")) is not None else None,
                                "t1_cos": round(float(sig_r["t1_cos"]), 3) if _finite(sig_r.get("t1_cos")) is not None else None,
                                "best_other_cid": bo_cid if bo_cid >= 0 else None,
                                "best_other_label": bo_lbl,
                                "best_other_sem_cos": round(float(sig_r["best_other_sem_cos"]), 3) if _finite(sig_r.get("best_other_sem_cos")) is not None else None,
                                "slab_near_dup_frac": round(float(sig_r.get("slab_near_dup_frac") or 0.0), 3),
                                "posts_per_day": round(float(sig_r.get("posts_per_day") or 0.0), 1),
                                "follow_ratio": round(float(sig_r.get("follow_ratio") or 0.0), 2),
                                "avatar_url": lf_r.get("avatar_url") or "",
                                "live_verdict": lf_r.get("live_verdict") or "",
                                "live_sem_cos": lf_r.get("live_sem_cos"),
                                "repost_frac": lf_r.get("repost_frac"),
                                "live_pol_frac": lf_r.get("live_pol_frac"),
                            },
                        )
                    )
                continue
            elig_pids.append(p_i)

        if len(elig_pids) < BUNDLE_DEFAULT:
            excluded_cards["lt25_eligible"].append(c)
            continue

        if sens_union:
            excluded_cards["y3_sensitive"].append(c)
            for sc in sens_union:
                excluded_cards[f"y3_class_{sc}"].append(c)
            continue

        if not args.strict_registered_only:
            blob_txt = f"{opus_label} :: {full_desc}"
            if c in manual_excl_cids:
                excluded_cards["manual_inspector_exclude"].append(c)
                continue
            if c in EXPLICIT_BLOCK_CIDS:
                excluded_cards["y3_explicit_review_block"].append(c)
                continue
            if RE_DESC_ADULT_MINORS.search(blob_txt):
                excluded_cards["y3_desc_adult_or_minors"].append(c)
                continue
            if comm_type == "roleplay" and RE_RP_UNSAFE.search(blob_txt):
                excluded_cards["y3_roleplay_school_or_idol"].append(c)
                continue
            if comm_type == "spam_or_commercial":
                excluded_cards["organic_spam_or_commercial"].append(c)
                continue
            if onb_fit == "poor":
                excluded_cards["organic_onboarding_fit_poor"].append(c)
                continue
            if comm_type == "mixed_or_unclear":
                excluded_cards["organic_mixed_or_unclear"].append(c)
                continue
            if conf_lvl == "low":
                excluded_cards["organic_low_confidence"].append(c)
                continue
            if RE_NON_ORGANIC_CARD.search(blob_txt):
                excluded_cards["organic_bot_promo_or_ai_slop"].append(c)
                continue

        # Rank eligible candidates by rev-3 coherence-weighted S13 score (or raw S13 score under --strict-registered-only)
        eff_score_by_pid = {
            p_i: effective_candidate_score(
                float(s13_score[p_i]),
                sig_by_pid.get(p_i),
                live_feed_by_did.get(full_dids[p_i]),
                args.strict_registered_only,
            )
            for p_i in elig_pids
        }
        elig_pids.sort(
            key=lambda p_i: (
                -eff_score_by_pid[p_i],
                -int(live_prof_by_did[full_dids[p_i]]["followersCount"]),
                full_dids[p_i],
            )
        )

        card_cands: list[int] = []
        big = 0
        sup_cnt = 0
        for p_i in elig_pids:
            fc = live_prof_by_did[full_dids[p_i]]["followersCount"]
            is_sup = bool(s13_sup[p_i])
            if len(card_cands) < BUNDLE_DEFAULT:
                if fc > POP_LIMIT and big >= POP_MAX:
                    continue
                if (not args.strict_registered_only) and is_sup and sup_cnt >= SUP_MAX:
                    continue
                if fc > POP_LIMIT:
                    big += 1
                if is_sup:
                    sup_cnt += 1
            card_cands.append(p_i)
            if len(card_cands) >= 27:
                break
        if len(card_cands) < BUNDLE_DEFAULT:
            excluded_cards["lt25_after_pop_or_sup_cap"].append(c)
            continue

        # Clean starter packs
        kept_packs: list[dict] = []
        for off_p in sorted(cand5_offers.get(c, []), key=lambda x: (-x["share"], x["uri"])):
            u_pk = off_p["uri"]
            mems = pack_members_by_uri.get(u_pk)
            if not mems:
                continue
            bad_m = [
                m
                for m in mems
                if (m in hard_safety_excl)
                or (m in acct_excl_reason_by_did)
                or (live_status_by_did.get(m, "unchecked") != "ok")
            ]
            if bad_m:
                pack_feed_stats["packs_dropped_member_check"] += 1
                continue
            creator_did = u_pk.split("/")[2]
            if (
                creator_did in hard_safety_excl
                or creator_did in acct_excl_reason_by_did
                or live_status_by_did.get(creator_did, "unchecked") != "ok"
            ):
                pack_feed_stats["packs_dropped_creator_check"] += 1
                continue
            pack_feed_stats["packs_kept"] += 1
            in_card_cnt = round(off_p["share"] * len(mems))
            pk_obj = {
                "u": u_pk,
                "n": (pnames.get(u_pk) or "Community starter pack")[:55],
                "ch": compact_handle(live_prof_by_did[creator_did]["handle"]),
                "mc": len(mems),
                "ic": in_card_cnt,
            }
            kept_packs.append(pk_obj)
            packs_detail_out[u_pk] = {
                "uri": u_pk,
                "name": pk_obj["n"],
                "creator_did": creator_did,
                "creator_handle": live_prof_by_did[creator_did]["handle"],
                "in_card": in_card_cnt,
                "members": [
                    {
                        "did": m,
                        "handle": live_prof_by_did[m]["handle"],
                        "name": (snap.get(m, ("", ""))[0])[:40],
                        "followers": live_prof_by_did[m]["followersCount"],
                        "hfr": m in hfr_dids,
                    }
                    for m in mems
                ],
            }
            if len(kept_packs) >= PACKS_PER_CARD:
                break

        # Clean O7 feeds
        kept_feeds: list[dict] = []
        for n_lk, f_uri in sorted(o7_by_comm.get(c, []), key=lambda x: (-x[0], x[1])):
            parts_u = f_uri.split("/")
            if len(parts_u) < 5:
                continue
            c_did = parts_u[2]
            if c_did in hfr_dids:
                pack_feed_stats["feeds_dropped_creator_hfr"] += 1
                continue
            if (
                c_did in hard_safety_excl
                or c_did in feed_creator_excl_dids
                or live_status_by_did.get(c_did, "unchecked") != "ok"
            ):
                pack_feed_stats["feeds_dropped_creator_status"] += 1
                continue
            fg_m = fg_meta.get(f_uri)
            disp_nm = ((fg_m["display_name"] if fg_m else "") or parts_u[-1])[:48]
            pack_feed_stats["feeds_kept"] += 1
            kept_feeds.append(
                {
                    "u": f_uri,
                    "n": disp_nm,
                    "ch": compact_handle(live_prof_by_did[c_did]["handle"]),
                    "lk": int(n_lk),
                }
            )
            if len(kept_feeds) >= FEEDS_PER_CARD:
                break

        card_medoids: list[str] = []
        for p_i in card_cands:
            d = full_dids[p_i]
            if d in hfr_dids:
                continue
            ms = medoid_by_did.get(d)
            if ms:
                card_medoids.append(ms[0])
                if len(card_medoids) >= MEDOIDS_PER_CARD:
                    break

        list_fallback_title = " · ".join(a_top[:3]) if a_top else opus_label
        eng_sum = int(sum(light_eng[p_i] for p_i in pids_by_comm.get(c, []) if p_i < len(light_eng)))
        kept_cards_raw.append(
            {
                "community": c,
                "scene": int(r["scene_id"]),
                "size": int(r["size"]),
                "eligible": len(elig_pids),
                "engagers": eng_sum,
                "lang": (cl.get("primary_language") or "en").strip() or "en",
                "opus_label": opus_label or list_fallback_title,
                "list_title": list_fallback_title or opus_label,
                "desc": full_desc,
                "community_type": comm_type,
                "onboarding_fit": onb_fit,
                "confidence": conf_lvl,
                "list_names": a_names.get(c, [])[:5],
                "cand_pids": card_cands,
                "eff_scores": [round(eff_score_by_pid[p_i], 4) for p_i in card_cands],
                "packs": kept_packs,
                "feeds": kept_feeds,
                "medoids": card_medoids,
            }
        )

    kept_cards_raw.sort(key=lambda cd: (-cd["engagers"], cd["community"]))
    kept_comm_set = {cd["community"] for cd in kept_cards_raw}

    for cd in kept_cards_raw:
        c = cd["community"]
        nbs = [
            nb
            for nb, _cnt in co_counts[c].most_common(40)
            if (nb in kept_comm_set) and ((c, nb) not in high_h_pairs)
        ][:6]
        cd["h_nbs"] = nbs

    # 6. Deterministic 30-card S17 review draw by cityHash64(str(community))
    c_strs = [str(cd["community"]) for cd in kept_cards_raw]
    ch_in = "\n".join(c_strs).encode()
    ch_out = subprocess.check_output(
        [
            "clickhouse",
            "local",
            "--structure",
            "c String",
            "--query",
            "SELECT c, cityHash64(c) AS h FROM table ORDER BY h ASC, c ASC FORMAT TSV",
        ],
        input=ch_in,
    ).decode()
    ordered_by_hash = [int(ln.split("\t")[0]) for ln in ch_out.splitlines() if ln.strip()]
    s17_review_30 = ordered_by_hash[:30]

    rng_ab = np.random.default_rng(20261030)
    card_by_id = {cd["community"]: cd for cd in kept_cards_raw}
    blind_public: list[dict] = []
    blind_secret: dict[str, dict] = {}
    for idx_r, cid in enumerate(s17_review_30):
        cd = card_by_id[cid]
        opus_first = bool(rng_ab.integers(0, 2) == 0)
        opt_a = cd["opus_label"] if opus_first else cd["list_title"]
        opt_b = cd["list_title"] if opus_first else cd["opus_label"]
        blind_public.append(
            {
                "slot": idx_r + 1,
                "c": cid,
                "option_a": opt_a,
                "option_b": opt_b,
                "same_text": opt_a.strip().lower() == opt_b.strip().lower(),
            }
        )
        blind_secret[str(cid)] = {
            "slot": idx_r + 1,
            "community": cid,
            "option_a_source": "opus_llm" if opus_first else "curated_list_name",
            "option_b_source": "curated_list_name" if opus_first else "opus_llm",
            "opus_label": cd["opus_label"],
            "curated_list_title": cd["list_title"],
        }

    # 7. Write sidecar descriptions, starter packs, blind key, and rich `inspector_index.json`
    comparator = load_comparator_cached(out_dir, live_prof_by_did, hard_safety_excl | set(acct_excl_reason_by_did))
    (out_dir / "comparator.json").write_text(json.dumps(comparator, indent=2, ensure_ascii=False))
    (out_dir / "packs_members.json").write_text(json.dumps(packs_detail_out, ensure_ascii=False))
    (out_dir / "blind_review_key.json").write_text(json.dumps(blind_secret, indent=2, ensure_ascii=False))

    descs_out = {
        str(cd["community"]): {
            "d": cd["desc"],
            "ct": cd["community_type"],
            "fit": cd["onboarding_fit"],
            "cf": cd["confidence"],
            "lt": cd["list_title"],
        }
        for cd in kept_cards_raw
    }
    descs_bytes = json.dumps(descs_out, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    (out_dir / "cards_descriptions.json").write_bytes(descs_bytes)

    # 8. Assemble compact `cards_v1.json` guaranteed <= 2,097,152 bytes
    def serialize_catalog(cands_per_card: int, desc_limit: int, name_limit: int, name_top_k: int = 10) -> tuple[bytes, dict]:
        top_named_pids: set[int] = set()
        for cd in kept_cards_raw:
            top_named_pids.update(cd["cand_pids"][:name_top_k])

        acct_idx: dict[int, int] = {}
        accts_table: list[list] = []

        def get_acct_id(p_i: int) -> int:
            aid = acct_idx.get(p_i)
            if aid is not None:
                return aid
            d = full_dids[p_i]
            cd_str = compact_did(d)
            assert expand_did(cd_str) == d, f"DID roundtrip mismatch: {d} -> {cd_str}"
            lp = live_prof_by_did[d]
            nm = ((snap.get(d, ("", ""))[0] or "")[:name_limit].strip()) if (name_limit > 0 and p_i in top_named_pids) else ""
            flags = (1 if d in hfr_dids else 0) | (2 if bool(s13_sup[p_i]) else 0)
            aid = len(accts_table)
            acct_idx[p_i] = aid
            row: list = [cd_str, compact_handle(lp["handle"]), int(lp["followersCount"])]
            if nm or flags:
                row.append(nm)
            if flags:
                row.append(flags)
            accts_table.append(row)
            return aid

        scenes_map: dict[int, list[dict]] = collections.defaultdict(list)
        cards_compact: dict[str, dict] = {}
        for cd in kept_cards_raw:
            scenes_map[cd["scene"]].append(cd)
            c_pids = cd["cand_pids"][:cands_per_card]
            m_ids = [get_acct_id(p_i) for p_i in c_pids]
            did_to_aid = {full_dids[p_i]: m_ids[idx_p] for idx_p, p_i in enumerate(c_pids)}

            md_compact = []
            for mu in cd["medoids"]:
                parts_m = mu.split("/")
                if len(parts_m) == 5 and parts_m[2] in did_to_aid:
                    md_compact.append([did_to_aid[parts_m[2]], parts_m[4]])

            pk_compact = [
                [compact_pack_uri(p["u"]), p["n"][:36], p["ch"], int(p["mc"])]
                for p in cd["packs"]
            ]
            fd_compact = [
                [compact_feed_uri(f["u"]), f["n"][:36], f["ch"], int(f["lk"])]
                for f in cd["feeds"]
            ]

            c_entry: dict = {
                "s": cd["scene"],
                "sz": cd["size"],
                "el": cd["eligible"],
                "t": cd["opus_label"][:58],
                "ln": [[n[:28], int(k)] for n, k in cd["list_names"][:2]],
                "m": m_ids,
            }
            if cd["lang"] != "en":
                c_entry["lg"] = cd["lang"]
            desc_s = cd["desc"][:desc_limit].strip() if desc_limit > 0 else ""
            if desc_s:
                c_entry["d"] = desc_s
            if pk_compact:
                c_entry["pk"] = pk_compact
            if fd_compact:
                c_entry["fd"] = fd_compact
            if md_compact:
                c_entry["md"] = md_compact
            if cd["h_nbs"]:
                c_entry["nb"] = cd["h_nbs"]
            cards_compact[str(cd["community"])] = c_entry

        scene_list = []
        for s_id, cs in scenes_map.items():
            cnt = collections.Counter()
            for cd in cs:
                for n, k in cd["list_names"]:
                    cnt[n] += k
            sname = " / ".join(n for n, _ in cnt.most_common(2)) if cnt else cs[0]["opus_label"]
            scene_list.append(
                {
                    "s": s_id,
                    "n": sname[:56],
                    "cs": [cd["community"] for cd in cs],
                    "eg": sum(cd["engagers"] for cd in cs),
                }
            )
        scene_list.sort(key=lambda x: (-x["eg"], x["s"]))

        high_h_compact = sorted(
            [ca, ct]
            for ca, ct in high_h_pairs
            if ca < ct and (ca in kept_comm_set) and (ct in kept_comm_set)
        )

        payload = {
            "version": "onboarding_v0_20261031_rev3" if not args.strict_registered_only else "onboarding_v0_20261030_strict",
            "partition": "cand5-full-matched",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "locked_s13": {"k": 3, "m": 3, "theta": 0.5, "ranker": "w_lin_plus_curators"},
            "rev2_organic_safety_curation": not args.strict_registered_only,
            "rev3_embedding_and_live_feed_curation": not args.strict_registered_only,
            "h_enabled": True,
            "counts": {
                "v0_cards": len(cards_compact),
                "unique_candidate_accounts": len(accts_table),
                "scenes": len(scene_list),
                "excluded": {k: len(v) for k, v in excluded_cards.items() if not k.startswith("y3_class_")},
                "excluded_by_y3_class": {
                    sc: len(excluded_cards.get(f"y3_class_{sc}", [])) for sc in sorted(Y3_CLASSES)
                },
            },
            "comparator": {
                "available": comparator["available"],
                "http_status": comparator["http_status"],
                "note": comparator["note"],
                "actors": comparator["actors"],
            },
            "blind_review_30": blind_public,
            "high_h_pairs": high_h_compact,
            "scenes": scene_list,
            "accts": accts_table,
            "cards": cards_compact,
        }
        raw_b = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return raw_b, payload

    chosen_bytes = b""
    chosen_payload: dict = {}
    for c_lim, d_lim, n_lim, n_topk in ((25, 0, 26, 10), (25, 0, 22, 10), (25, 0, 18, 8), (25, 0, 14, 8), (25, 0, 0, 0)):
        b_try, p_try = serialize_catalog(c_lim, d_lim, n_lim, n_topk)
        print(f"[s17-build] trial (cands={c_lim}, desc={d_lim}, name={n_lim}, topk={n_topk}) -> {len(b_try):,} bytes", flush=True)
        if len(b_try) <= MAX_JSON_BYTES - 8_192:
            chosen_bytes, chosen_payload = b_try, p_try
            break
    if not chosen_bytes:
        sys.exit(f"ABORT: cards_v1.json exceeded {MAX_JSON_BYTES} bytes even at minimal trim!")

    cards_json_path = out_dir / "cards_v1.json"
    cards_json_path.write_bytes(chosen_bytes)
    cards_sha = hashlib.sha256(chosen_bytes).hexdigest()
    (out_dir / f"onboarding_v0_{cards_sha[:8]}.json").write_bytes(chosen_bytes)

    # 8b. Build rich `inspector_index.json` sidecar for the new S17 v2 Inspector UI
    accts_tbl = chosen_payload["accts"]
    inspector_accts: list[dict] = []
    borderline_suspects: list[tuple[float, dict]] = []

    aid_to_card_slot: dict[int, tuple[int, int]] = {}
    for c_key, cd_c in chosen_payload["cards"].items():
        for slot_i, aid in enumerate(cd_c["m"]):
            if aid not in aid_to_card_slot:
                aid_to_card_slot[aid] = (int(c_key), slot_i + 1)

    for aid, row in enumerate(accts_tbl):
        d = expand_did(row[0])
        p_i = did2pid[d]
        lp = live_prof_by_did[d]
        nm_f, bio_f = snap.get(d, ("", ""))
        sig_r = sig_by_pid.get(p_i) or {}
        lf_r = live_feed_by_did.get(d)
        bo_cid = int(sig_r.get("best_other_cid") or -1)
        bo_lbl = (claude_comm_extra.get(bo_cid) or {}).get("label", "") if bo_cid >= 0 else ""

        sp = _finite(sig_r.get("sem_cos_posts"))
        t1 = _finite(sig_r.get("t1_cos"))
        bo_sem = _finite(sig_r.get("best_other_sem_cos"))
        dup = float(sig_r.get("slab_near_dup_frac") or 0.0)
        frat = float(sig_r.get("follow_ratio") or 0.0)
        ppd = float(sig_r.get("posts_per_day") or 0.0)
        v9g = float(sig_r.get("v9_demotion_gap") or 0.0)

        # Compute residual discrepancy risk score among KEPT candidates so Josh can spot-check edge cases first
        flags_list: list[str] = []
        risk = 0.0
        if not sig_r.get("has_vec") or int(sig_r.get("mm_post_count") or 0) < 3:
            flags_list.append("few_slab_posts")
            risk += 1.4
        if sp is not None and sp < 0.16:
            flags_list.append(f"low_sem_cos({sp:+.2f})")
            risk += (0.20 - sp) * 8.0
        if t1 is not None and t1 < 0.14:
            flags_list.append(f"low_social_cos({t1:+.2f})")
            risk += (0.18 - t1) * 6.0
        if sp is not None and bo_sem is not None and (bo_sem - sp) >= 0.25 and sp < 0.25:
            flags_list.append(f"closer_to_c{bo_cid}({bo_sem:+.2f})")
            risk += (bo_sem - sp) * 5.0
        if int(sig_r.get("slab_n_posts") or 0) >= 8 and dup >= 0.55:
            flags_list.append(f"repetitive_slab({int(dup*100)}%)")
            risk += dup * 3.5
        if frat >= 1.8 and lp["followsCount"] >= 2_500:
            flags_list.append(f"follow_heavy({frat:.1f}x)")
            risk += 2.2
        if ppd >= 70.0:
            flags_list.append(f"high_cadence({ppd:.0f}/d)")
            risk += 1.8
        if v9g >= 0.40:
            flags_list.append(f"high_block_or_combative({int(v9g*100)}%)")
            risk += v9g * 2.5
        if lf_r is not None:
            lv = lf_r.get("live_verdict") or ""
            if lv.startswith("warn_"):
                flags_list.append(lv)
                risk += 2.6

        meds_acct = [u.rsplit("/", 1)[-1] for u in (medoid_by_did.get(d) or [])[:3]]
        acct_meta = {
            "nm": nm_f[:55],
            "bio": " ".join(bio_f.split())[:260],
            "pc": lp["postsCount"],
            "fw": lp["followsCount"],
            "sp": round(sp, 3) if sp is not None else None,
            "t1": round(t1, 3) if t1 is not None else None,
            "boc": bo_cid if (bo_cid >= 0 and bo_sem is not None and sp is not None and bo_sem > sp + 0.12) else None,
            "bol": bo_lbl[:48] if (bo_cid >= 0 and bo_sem is not None and sp is not None and bo_sem > sp + 0.12) else "",
            "bos": round(bo_sem, 3) if (bo_cid >= 0 and bo_sem is not None and sp is not None and bo_sem > sp + 0.12) else None,
            "dup": round(dup, 3),
            "img": round(float(sig_r.get("slab_img_frac") or 0.0), 2),
            "ppd": round(ppd, 1),
            "fr": round(frat, 2),
            "v9": round(v9g, 2),
            "s13": round(float(s13_score[p_i]), 2),
            "md": meds_acct,
            "fl": flags_list,
        }
        if lf_r is not None:
            acct_meta["lf"] = {
                "av": lf_r.get("avatar_url") or "",
                "vd": lf_r.get("live_verdict") or "",
                "lsc": lf_r.get("live_sem_cos"),
                "rp": lf_r.get("repost_frac"),
                "ry": lf_r.get("reply_frac"),
                "no": lf_r.get("n_orig"),
                "da": lf_r.get("days_since_last_any"),
                "dom": lf_r.get("top_ext_domain") or "",
            }
        inspector_accts.append(acct_meta)

        if risk >= 2.2 and aid in aid_to_card_slot:
            cid_slot, rank_pos = aid_to_card_slot[aid]
            borderline_suspects.append(
                (
                    -risk,
                    {
                        "aid": aid,
                        "did": d,
                        "handle": lp["handle"],
                        "card_cid": cid_slot,
                        "rank_pos": rank_pos,
                        "risk": round(risk, 2),
                        "flags": flags_list,
                    },
                )
            )

    borderline_suspects.sort(key=lambda x: x[0])

    # Top evicted examples per rev-3 category so Inspector shows concrete before/after proof
    evicted_showcase: dict[str, list[dict]] = {}
    for rsn, items in sorted(evicted_by_reason.items()):
        items.sort(key=lambda x: x[0])
        evicted_showcase[rsn] = [it[1] for it in items[:15]]

    card_health: dict[str, dict] = {}
    for cd in kept_cards_raw:
        cid = cd["community"]
        c_pids = cd["cand_pids"][:25]
        sps = [_finite((sig_by_pid.get(p) or {}).get("sem_cos_posts")) for p in c_pids]
        sps_clean = [x for x in sps if x is not None]
        t1s = [_finite((sig_by_pid.get(p) or {}).get("t1_cos")) for p in c_pids]
        t1s_clean = [x for x in t1s if x is not None]
        cp_cent, ct_cent = card_coh_map.get(cid, (0.0, 0.0))
        hydrated_n = sum(1 for p in c_pids if full_dids[p] in live_feed_by_did)
        sup_n = sum(1 for p in c_pids if bool(s13_sup[p]))
        card_health[str(cid)] = {
            "coh_posts": round(float(np.mean(sps_clean)), 3) if sps_clean else cp_cent,
            "min_sem": round(float(np.min(sps_clean)), 3) if sps_clean else None,
            "coh_t1": round(float(np.mean(t1s_clean)), 3) if t1s_clean else ct_cent,
            "hydrated_25": hydrated_n,
            "sup_25": sup_n,
            "all_lists": cd["list_names"],
        }

    inspector_payload = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cards_v1_sha256": cards_sha,
        "hydrated_feeds_count": len(live_feed_by_did),
        "evicted_counts_in_kept_cards": {k: len(v) for k, v in sorted(evicted_by_reason.items(), key=lambda kv: -len(kv[1]))},
        "evicted_showcase": evicted_showcase,
        "borderline_suspects": [x[1] for x in borderline_suspects[:200]],
        "card_health": card_health,
        "acct_meta": inspector_accts,
    }
    (out_dir / "inspector_index.json").write_bytes(
        json.dumps(inspector_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )

    # 9. Independent verification (`verify_s17`) across EVERY card, candidate, pack member, feed creator, and medoid
    violations: list[str] = []
    if len(chosen_bytes) > MAX_JSON_BYTES:
        violations.append(f"cards_v1.json size {len(chosen_bytes)} > {MAX_JSON_BYTES}")

    descs_loaded = json.loads((out_dir / "cards_descriptions.json").read_text())
    if set(descs_loaded.keys()) != set(chosen_payload["cards"].keys()):
        violations.append("cards_descriptions.json key mismatch against cards_v1.json")

    for c_key, cd in chosen_payload["cards"].items():
        cid = int(c_key)
        if comm_bad_full.get(cid, False):
            violations.append(f"card {cid}: flagged community present")
        cl = claude_c.get(cid) or {}
        cx = claude_comm_extra.get(cid) or {}
        if any(s and s != "none" for s in (cl.get("sensitivity") or [])):
            violations.append(f"card {cid}: Opus sensitive card present ({cl.get('sensitivity')})")
        a_top = [n for n, _k in a_names.get(cid, [])]
        names_scr = a_top[:6] + all_names.get(cid, [])[:6]
        kw_hit = {k for nm in names_scr for k in s6_kw.classes(nm)}
        if kw_hit:
            violations.append(f"card {cid}: keyword sensitive card present ({sorted(kw_hit)})")
        if len(cd["m"]) < BUNDLE_DEFAULT:
            violations.append(f"card {cid}: <25 candidates ({len(cd['m'])})")

        if not args.strict_registered_only:
            if cid in EXPLICIT_BLOCK_CIDS or cid in manual_excl_cids:
                violations.append(f"card {cid}: in block/manual exclude list")
            opus_lbl = cx.get("label") or ""
            nat_lbl = cx.get("label_native") or ""
            full_d = cx.get("description") or ""
            ct_v = cx.get("community_type") or ""
            fit_v = cx.get("onboarding_fit") or ""
            cf_v = cx.get("confidence") or ""
            lbl_kw_hit = {
                k.replace("/", "_").replace("-", "_")
                for nm in (opus_lbl, nat_lbl)
                if nm
                for k in s6_kw.classes(nm)
            } & OPUS_LABEL_Y3_CLASSES
            if lbl_kw_hit:
                violations.append(f"card {cid}: Opus label Y3 keyword hit ({sorted(lbl_kw_hit)})")
            blob_v = f"{opus_lbl} :: {full_d}"
            if RE_DESC_ADULT_MINORS.search(blob_v):
                violations.append(f"card {cid}: description adult/minors hit")
            if ct_v == "roleplay" and RE_RP_UNSAFE.search(blob_v):
                violations.append(f"card {cid}: unsafe school/idol roleplay hit")
            if ct_v in ("spam_or_commercial", "mixed_or_unclear") or fit_v == "poor" or cf_v == "low":
                violations.append(f"card {cid}: non-organic/low-fit card ({ct_v}/{fit_v}/{cf_v})")
            if RE_NON_ORGANIC_CARD.search(blob_v):
                violations.append(f"card {cid}: automated bot/promo/AI-slop card hit")

        big_25 = 0
        sup_25 = 0
        for pos, aid in enumerate(cd["m"]):
            row = accts_tbl[aid]
            d = expand_did(row[0])
            fc = int(row[2])
            fl_bits = int(row[4]) if len(row) > 4 else 0
            if pos < BUNDLE_DEFAULT and fc > POP_LIMIT:
                big_25 += 1
            if pos < BUNDLE_DEFAULT and (fl_bits & 2):
                sup_25 += 1
            if live_status_by_did.get(d) != "ok":
                violations.append(f"card {cid}: candidate {d} live status != ok")
            if d in hard_safety_excl:
                violations.append(f"card {cid}: candidate {d} in hard_safety_excl")
            if DROP_PROFILE_CLASSES & claude_p.get(d, set()):
                violations.append(f"card {cid}: candidate {d} Claude profile minors/adult")
            if not args.strict_registered_only:
                if d in acct_excl_reason_by_did:
                    violations.append(f"card {cid}: candidate {d} failed rev3 organic/embedding/live screen ({acct_excl_reason_by_did[d]})")
        if big_25 > POP_MAX:
            violations.append(f"card {cid}: >2 accounts >500k in top 25 ({big_25})")
        if (not args.strict_registered_only) and sup_25 > SUP_MAX:
            violations.append(f"card {cid}: >2 supplement accounts in top 25 ({sup_25})")

        for pk in cd.get("pk", []):
            u_pk = expand_pack_uri(pk[0])
            if u_pk not in pack_keep_uris:
                violations.append(f"card {cid}: pack {u_pk} not in card_packs_filter keep")
            det = packs_detail_out.get(u_pk, {})
            for mem in det.get("members", []):
                md = mem["did"]
                if live_status_by_did.get(md) != "ok" or md in hard_safety_excl or md in acct_excl_reason_by_did:
                    violations.append(f"card {cid}: pack {u_pk} member {md} not clean/ok")

        for fd in cd.get("fd", []):
            u_fd = expand_feed_uri(fd[0])
            c_did = u_fd.split("/")[2]
            if (
                c_did in hfr_dids
                or c_did in hard_safety_excl
                or c_did in feed_creator_excl_dids
                or live_status_by_did.get(c_did) != "ok"
            ):
                violations.append(f"card {cid}: feed {u_fd} creator {c_did} not allowed")

        for m_pair in cd.get("md", []):
            m_did = expand_did(accts_tbl[int(m_pair[0])][0])
            if (
                m_did in hfr_dids
                or m_did in hard_safety_excl
                or m_did in acct_excl_reason_by_did
                or live_status_by_did.get(m_did) != "ok"
            ):
                violations.append(f"card {cid}: medoid {m_did}/{m_pair[1]} author not allowed")

        for nb in cd.get("nb", []):
            if (cid, int(nb)) in high_h_pairs:
                violations.append(f"card {cid}: high-H neighbor {nb} in suggested neighbors")

    raw_hashes, raw_hashes_sha = collect_raw_response_hashes(lake, s5_dir, out_dir)
    (out_dir / "raw_response_hashes.txt").write_text("\n".join(raw_hashes) + "\n")

    ver_report = {
        "zero_safety_violations": len(violations) == 0,
        "violations_count": len(violations),
        "violations": violations[:100],
        "rev2_organic_safety_curation": not args.strict_registered_only,
        "rev3_embedding_and_live_feed_curation": not args.strict_registered_only,
        "cards_v1_bytes": len(chosen_bytes),
        "cards_v1_le_2mb": len(chosen_bytes) <= MAX_JSON_BYTES,
        "cards_v1_sha256": cards_sha,
        "cards_descriptions_bytes": len(descs_bytes),
        "cards_descriptions_sha256": hashlib.sha256(descs_bytes).hexdigest(),
        "v0_cards_count": len(chosen_payload["cards"]),
        "unique_candidate_accounts": len(accts_tbl),
        "non_organic_or_bio_unsafe_dids_excluded": len(acct_excl_reason_by_did),
        "live_hydrated_feeds_used": len(live_feed_by_did),
        "blind_review_cards_count": len(s17_review_30),
        "controls_ok": all(controls_map.values()),
        "raw_response_records_hashed": len(raw_hashes),
        "raw_response_hashes_sha256": raw_hashes_sha,
    }
    (out_dir / "verify.json").write_text(json.dumps(ver_report, indent=2))
    if violations:
        sys.exit(f"ABORT: {len(violations)} safety violations detected in S17 verify: {violations[:5]}")

    receipt = {
        "step": "S17 Harness v2 & Onboarding v0 Catalog Build (rev 3 embedding+spam+live-feed hardened, 127.0.0.1 only)",
        "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md rows S16 & S17, §3.3",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md §9.6, §10, Amendments A1 & A2 + DECISIONS S17 rev 2 & rev 3",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t0, 2),
        "strict_registered_only": bool(args.strict_registered_only),
        "inputs_sha256": {
            "communities_parquet": sha256_file(cand5_dir / "communities.parquet"),
            "known_for_parquet": sha256_file(cand5_dir / "known_for.parquet"),
            "s13_prod_scores": sha256_file(s13_dir / "s13_prod_eligibility_and_scores.parquet"),
            "h_prod_cand5_full": sha256_file(s14_dir / "H_prod_cand5_full.parquet"),
            "s5_receipt": sha256_file(s5_dir / "receipt.json"),
            "s5_profiles": sha256_file(s5_dir / "profiles.parquet"),
            "s5_purge_dids": sha256_file(s5_dir / "purge_dids.txt"),
            "o7_feeds": sha256_file(s10_dir / "O7/O7.parquet"),
            "s12_medoids": sha256_file(s12_dir / "medoids.parquet"),
            "claude_cand5_labels": sha256_file(claude_comm_path),
            "s16_report": sha256_file(s16_dir / "s16_report.json"),
            "candidate_signals": sha256_file(sig_pq_path) if sig_pq_path.exists() else "",
        },
        "s5_controls": controls_map,
        "raw_appview_responses": {
            "records_hashed": len(raw_hashes),
            "hashes_sha256": raw_hashes_sha,
            "hashes_file": "raw_response_hashes.txt",
        },
        "counts": chosen_payload["counts"],
        "member_drops": dict(drop_counts),
        "account_organic_safety_exclusions": dict(collections.Counter(acct_excl_reason_by_did.values())),
        "evicted_in_kept_v0_cards": inspector_payload["evicted_counts_in_kept_cards"],
        "packs_and_feeds": dict(pack_feed_stats),
        "comparator_probe": {k: v for k, v in comparator.items() if k != "actors"},
        "verify": ver_report,
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(
        f"[s17-build] PASS: v0_cards={len(chosen_payload['cards']):,}, bytes={len(chosen_bytes):,}/{MAX_JSON_BYTES:,}, "
        f"live_feeds={len(live_feed_by_did):,}, violations=0 ({time.time()-t0:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
