#!/usr/bin/env python3
"""Polite live `app.bsky.feed.getAuthorFeed` hydrator & live spam/coherence scorer (`<= 9 req/s` under D8, rev 4).

Invariants:
- Strictly `<= 9.0 req/s` global token-bucket against `public.api.bsky.app` with descriptive `User-Agent`
  and `429` exponential backoff.
- ZERO raw image bytes ever fetched or written to disk: stores compact JSON metadata and CDN URL strings only.
- Enforces A1 (`hideFromAlgorithmicRecommendations` never stores/renders post content) and D9/S5 label checks
  (any author or post label on adult/graphic/moderation drops media URLs and flags the account if adult/unsafe).
- Encodes live authored post texts on local Mac MPS with `EmbeddingGemma 2` (`bf16`) to score
  `live_sem_cos` against each card's `cent_text_posts` centroid (`card_centroids.npz`).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pyarrow.parquet as pq

USER_AGENT = "catbird-s17-inspector/2.1 (+https://catbird.blue; local-onboarding-audit)"

POL_RE = re.compile(
    r"\b(trump|biden|kamala|harris|vance|walz|desantis|obama|pelosi|schumer|mcconnell|aoc|"
    r"bernie|sanders|maga|democrat|democrats|republican|republicans|gop|dnc|rnc|liberal|"
    r"liberals|conservative|conservatives|leftist|right-wing|far-right|fascist|fascism|"
    r"authoritarian|antifa|woke|dei|scotus|supreme court|congress|senate|filibuster|impeach|"
    r"ballot|midterms|electoral|polling|pollster|gerrymander|white house|doj|ice raids|"
    r"deportation|tariffs|zionist|intifada|netanyahu|hamas|idf|gaza|kremlin|putin|zelensky|"
    r"nato|starmer|tories|labour party)\b",
    re.I,
)

ADULT_POST_LABEL_VALS = frozenset(
    {
        "porn",
        "sexual",
        "nudity",
        "graphic-media",
        "gore",
        "nsfl",
        "corpse",
        "!hide",
        "!takedown",
        "!suspend",
        "csam",
        "non-consensual",
    }
)

SYNDICATOR_OR_AFFILIATE_DOMAINS = frozenset(
    {
        "dlvr.it",
        "ift.tt",
        "amzn.to",
        "books2read.com",
        "mybook.to",
        "geni.us",
        "shrsl.com",
        "rstyle.me",
        "linktr.ee",
    }
)

GENERIC_SHORTENER_DOMAINS = frozenset(
    {
        "bit.ly",
        "buff.ly",
        "ow.ly",
        "tinyurl.com",
        "rb.gy",
        "cutt.ly",
        "t.co",
    }
)

CREATOR_PLATFORM_ALLOWLIST = frozenset(
    {
        "",
        "bsky.app",
        "youtube.com",
        "youtu.be",
        "twitch.tv",
        "github.com",
        "gitlab.com",
        "codeberg.org",
        "crates.io",
        "pypi.org",
        "itch.io",
        "bandcamp.com",
        "soundcloud.com",
        "archiveofourown.org",
        "wikipedia.org",
        "arxiv.org",
        "doi.org",
        "biorxiv.org",
        "medrxiv.org",
        "ko-fi.com",
        "patreon.com",
        "substack.com",
        "letterboxd.com",
        "flickr.com",
        "pixiv.net",
        "artstation.com",
        "cara.app",
        "vgen.co",
    }
)

URL_RE = re.compile(r"https?://([^/\s?#]+)", re.I)
URL_STRIP_RE = re.compile(r"https?://\S+", re.I)
HASHTAG_RE = re.compile(r"(?:^|\s)#[a-z0-9_]{2,}", re.I)
WS_RE = re.compile(r"\s+")


def _is_academic_or_civic_domain(dom: str) -> bool:
    d = (dom or "").lower()
    return d.endswith(".edu") or ".ac." in d or d.endswith(".gov") or ".gov." in d


class PoliteRateLimiter:
    """Thread-safe strict rate pacer (`max_rps <= 9.0` req/s)."""

    def __init__(self, max_rps: float = 8.5):
        self.min_interval = 1.0 / max(min(max_rps, 9.5), 0.5)
        self._lock = threading.Lock()
        self._next_ok = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_ok:
                sleep_s = self._next_ok - now
                self._next_ok += self.min_interval
            else:
                sleep_s = 0.0
                self._next_ok = now + self.min_interval
        if sleep_s > 0:
            time.sleep(sleep_s)

    def penalize(self, backoff_s: float) -> None:
        with self._lock:
            self._next_ok = max(self._next_ok, time.monotonic() + backoff_s)


def init_db(db_path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS author_feeds (
            did TEXT PRIMARY KEY,
            fetched_utc TEXT NOT NULL,
            http_status INTEGER NOT NULL,
            raw_sha256 TEXT NOT NULL,
            is_hfr INTEGER NOT NULL DEFAULT 0,
            avatar_url TEXT NOT NULL DEFAULT '',
            live_handle TEXT NOT NULL DEFAULT '',
            live_name TEXT NOT NULL DEFAULT '',
            live_safety_flag TEXT NOT NULL DEFAULT '',
            n_items INTEGER NOT NULL DEFAULT 0,
            n_orig INTEGER NOT NULL DEFAULT 0,
            n_quote INTEGER NOT NULL DEFAULT 0,
            n_reply INTEGER NOT NULL DEFAULT 0,
            n_repost INTEGER NOT NULL DEFAULT 0,
            repost_frac REAL NOT NULL DEFAULT 0.0,
            reply_frac REAL NOT NULL DEFAULT 0.0,
            authored_frac REAL NOT NULL DEFAULT 0.0,
            days_since_last_any REAL NOT NULL DEFAULT 9999.0,
            days_since_last_authored REAL NOT NULL DEFAULT 9999.0,
            ext_link_frac REAL NOT NULL DEFAULT 0.0,
            top_ext_domain TEXT NOT NULL DEFAULT '',
            same_domain_frac REAL NOT NULL DEFAULT 0.0,
            live_dup_text_frac REAL NOT NULL DEFAULT 0.0,
            live_pol_frac REAL NOT NULL DEFAULT 0.0,
            median_likes REAL NOT NULL DEFAULT 0.0,
            avg_hashtags REAL NOT NULL DEFAULT 0.0,
            bare_link_frac REAL NOT NULL DEFAULT 0.0,
            live_sem_cos REAL,
            live_verdict TEXT NOT NULL DEFAULT '',
            posts_json TEXT NOT NULL DEFAULT '[]'
        );
        """
    )
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(author_feeds)").fetchall()}
    if "avg_hashtags" not in existing_cols:
        conn.execute("ALTER TABLE author_feeds ADD COLUMN avg_hashtags REAL NOT NULL DEFAULT 0.0;")
    if "bare_link_frac" not in existing_cols:
        conn.execute("ALTER TABLE author_feeds ADD COLUMN bare_link_frac REAL NOT NULL DEFAULT 0.0;")
    conn.commit()
    return conn


def _parse_iso_days(ts: str, now_dt: dt.datetime) -> float | None:
    if not ts:
        return None
    try:
        d = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return max((now_dt - d).total_seconds() / 86400.0, 0.0)
    except Exception:
        return None


def _extract_embed_info(embed: dict | None, allow_media_urls: bool) -> tuple[str, list[str], list[str], str, str]:
    """Return `(media_type, thumb_urls, alt_texts, ext_uri, ext_title)` without ever fetching bytes."""
    if not isinstance(embed, dict):
        return "", [], [], "", ""
    etype = embed.get("$type") or ""
    thumbs: list[str] = []
    alts: list[str] = []
    ext_uri = ""
    ext_title = ""
    media_type = ""

    inner_media = embed.get("media") if "recordWithMedia" in etype else embed
    if isinstance(inner_media, dict):
        mtype = inner_media.get("$type") or ""
        if "images" in mtype:
            media_type = "image"
            for im in (inner_media.get("images") or [])[:4]:
                if isinstance(im, dict):
                    t_u = (im.get("thumb") or "").strip()
                    if allow_media_urls and t_u.startswith("https://cdn.bsky.app/"):
                        thumbs.append(t_u)
                    alt = WS_RE.sub(" ", (im.get("alt") or "").strip())
                    if alt:
                        alts.append(alt[:240])
        elif "video" in mtype:
            media_type = "video"
            t_u = (inner_media.get("thumbnail") or "").strip()
            if allow_media_urls and t_u.startswith("https://cdn.bsky.app/"):
                thumbs.append(t_u)
            alt = WS_RE.sub(" ", (inner_media.get("alt") or "").strip())
            if alt:
                alts.append(alt[:240])
        elif "external" in mtype:
            ext = inner_media.get("external") or {}
            if isinstance(ext, dict):
                ext_uri = (ext.get("uri") or "").strip()[:400]
                ext_title = WS_RE.sub(" ", (ext.get("title") or "").strip())[:140]
                if not media_type:
                    media_type = "link"

    if "record" in etype and not media_type:
        media_type = "quote"
    return media_type, thumbs, alts, ext_uri, ext_title


def _extract_quote_snippet(embed: dict | None) -> dict | None:
    if not isinstance(embed, dict):
        return None
    etype = embed.get("$type") or ""
    rec_wrap = None
    if "recordWithMedia" in etype:
        rec_wrap = (embed.get("record") or {}).get("record")
    elif "embed.record" in etype:
        rec_wrap = embed.get("record")
    if not isinstance(rec_wrap, dict):
        return None
    q_au = rec_wrap.get("author") or {}
    q_val = rec_wrap.get("value") or {}
    if not isinstance(q_au, dict) or not isinstance(q_val, dict):
        return None
    q_lbls = [l.get("val", "") for l in (q_au.get("labels") or []) if isinstance(l, dict) and not l.get("neg")]
    p_lbls = [l.get("val", "") for l in (rec_wrap.get("labels") or []) if isinstance(l, dict) and not l.get("neg")]
    if q_lbls or p_lbls:
        return None
    q_txt = WS_RE.sub(" ", (q_val.get("text") or "").strip())[:220]
    q_hdl = (q_au.get("handle") or "").strip()
    if not q_hdl and not q_txt:
        return None
    return {"handle": q_hdl, "text": q_txt}


def classify_live_verdict(row: dict) -> str:
    """Assign automated rev-4 live-feed verdict (`clean`, `drop_*`, or `warn_*`)."""
    if row.get("live_safety_flag"):
        return f"drop_{row['live_safety_flag']}"
    http_st = int(row.get("http_status") or 0)
    n_items = int(row.get("n_items") or 0)
    if http_st in (400, 404) and n_items == 0:
        return "drop_account_gone"
    if n_items == 0:
        return "drop_empty_feed"

    days_any = float(row.get("days_since_last_any") or 9999.0)
    if days_any > 120.0:
        return "drop_dormant_120d"

    n_orig = int(row.get("n_orig") or 0)
    n_quote = int(row.get("n_quote") or 0)
    n_reply = int(row.get("n_reply") or 0)
    n_oq = n_orig + n_quote
    repost_frac = float(row.get("repost_frac") or 0.0)
    reply_frac = float(row.get("reply_frac") or 0.0)
    dup_frac = float(row.get("live_dup_text_frac") or 0.0)
    pol_frac = float(row.get("live_pol_frac") or 0.0)
    dom = (row.get("top_ext_domain") or "").lower()
    sdom = float(row.get("same_domain_frac") or 0.0)
    ext_frac = float(row.get("ext_link_frac") or 0.0)
    med_likes = float(row.get("median_likes") or 0.0)
    avg_tags = float(row.get("avg_hashtags") or 0.0)
    bare_frac = float(row.get("bare_link_frac") or 0.0)

    # 1. Repost wall & reply-only timeline screens
    if n_items >= 12 and ((repost_frac >= 0.85 and n_oq <= 1) or repost_frac >= 0.90):
        return "drop_repost_wall"
    if n_items >= 15 and reply_frac >= 0.90 and n_oq <= 1:
        return "drop_reply_wall"

    # 2. Live near-duplicate template spam (spare viral human catchphrases with >=100 median likes unless >=80% dup)
    if (n_oq + n_reply) >= 6 and (dup_frac >= 0.80 or (dup_frac >= 0.60 and med_likes <= 35.0)):
        return "drop_live_template_spam"

    # 3. RSS syndicators, affiliate shorteners, bare link dumps, and commercial/scraper link farms
    if n_oq >= 4 and dom in SYNDICATOR_OR_AFFILIATE_DOMAINS and sdom >= 0.40:
        return "drop_link_or_promo_spam"
    if n_oq >= 4 and dom in GENERIC_SHORTENER_DOMAINS and sdom >= 0.50 and (
        med_likes <= 6.0 or bare_frac >= 0.60 or avg_tags >= 6.0
    ):
        return "drop_link_or_promo_spam"
    if (
        n_oq >= 6
        and sdom >= 0.80
        and dom not in CREATOR_PLATFORM_ALLOWLIST
        and not _is_academic_or_civic_domain(dom)
        and (med_likes <= 4.0 or bare_frac >= 0.65 or dup_frac >= 0.30)
    ):
        return "drop_link_or_promo_spam"
    if n_oq >= 6 and bare_frac >= 0.80 and med_likes <= 1.5:
        return "drop_link_or_promo_spam"

    # 4. Hashtag-stuffing spam
    if n_oq >= 5 and (
        (avg_tags >= 12.0 and med_likes <= 8.0)
        or (avg_tags >= 8.0 and (ext_frac >= 0.35 or med_likes <= 3.0))
    ):
        return "drop_hashtag_spam"

    # 5. Live partisan political drift
    if n_items >= 10 and pol_frac >= 0.60:
        return "drop_political_drift"

    # Soft warnings (penalized in ranking & surfaced in Inspector)
    if n_oq >= 6 and sdom >= 0.80 and dom not in CREATOR_PLATFORM_ALLOWLIST and not _is_academic_or_civic_domain(dom):
        return "warn_single_domain_promo"
    if n_oq >= 5 and avg_tags >= 7.5:
        return "warn_hashtag_heavy"
    if n_items >= 12 and reply_frac >= 0.82 and n_oq <= 2:
        return "warn_reply_heavy"
    if n_items >= 10 and repost_frac >= 0.72:
        return "warn_heavy_reposter"
    if float(row.get("days_since_last_authored") or 9999.0) > 90.0:
        return "warn_no_recent_originals_90d"
    if n_items >= 8 and pol_frac >= 0.42:
        return "warn_political_drift"
    return "clean"


def _compute_hashtag_and_bare_link_stats(posts: list[dict]) -> tuple[float, float]:
    auth_posts = [p for p in posts if isinstance(p, dict) and p.get("kind") in ("post", "quote")]
    if not auth_posts:
        return 0.0, 0.0
    tag_cnts: list[int] = []
    bare_cnt = 0
    for p in auth_posts:
        txt = p.get("text") or ""
        tag_cnts.append(len(HASHTAG_RE.findall(txt)))
        has_link = bool(p.get("ext_uri") or p.get("ext_domain"))
        stripped = URL_STRIP_RE.sub("", txt).strip()
        if has_link and len(stripped) < 20:
            bare_cnt += 1
    return (
        round(float(sum(tag_cnts)) / len(auth_posts), 3),
        round(float(bare_cnt) / len(auth_posts), 4),
    )


def rescore_cached_verdicts(conn: sqlite3.Connection) -> int:
    """Upgrade all existing SQLite rows with `avg_hashtags`, `bare_link_frac`, and rev-4 `live_verdict`."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM author_feeds").fetchall()
    updates: list[tuple[float, float, str, str]] = []
    for r in rows:
        d = dict(r)
        try:
            posts = json.loads(d.get("posts_json") or "[]")
        except Exception:
            posts = []
        avg_tags, bare_frac = _compute_hashtag_and_bare_link_stats(posts)
        # Preserve higher precision if already computed on full feed during fetch
        if float(d.get("avg_hashtags") or 0.0) > avg_tags:
            avg_tags = float(d["avg_hashtags"])
        if float(d.get("bare_link_frac") or 0.0) > bare_frac:
            bare_frac = float(d["bare_link_frac"])
        d["avg_hashtags"] = avg_tags
        d["bare_link_frac"] = bare_frac
        vd = classify_live_verdict(d)
        lsc = d.get("live_sem_cos")
        auth_n = sum(1 for p in posts if isinstance(p, dict) and p.get("kind") in ("post", "quote", "reply"))
        if vd == "clean" and lsc is not None and auth_n >= 3 and float(lsc) < 0.04:
            vd = "warn_low_live_coherence"
        if (
            abs(float(r["avg_hashtags"] or 0.0) - avg_tags) > 1e-4
            or abs(float(r["bare_link_frac"] or 0.0) - bare_frac) > 1e-4
            or r["live_verdict"] != vd
        ):
            updates.append((avg_tags, bare_frac, vd, d["did"]))
    if updates:
        conn.executemany(
            "UPDATE author_feeds SET avg_hashtags = ?, bare_link_frac = ?, live_verdict = ? WHERE did = ?",
            updates,
        )
        conn.commit()
    return len(updates)


def fetch_and_parse_one(
    did: str,
    limiter: PoliteRateLimiter,
    hfr_dids: set[str],
    hard_excl: set[str],
    limit: int = 30,
) -> dict:
    """Fetch `getAuthorFeed` for `did` via `limiter` and return parsed row dict ready for SQLite."""
    now_dt = dt.datetime.now(dt.timezone.utc)
    fetched_utc = now_dt.isoformat()
    is_hfr = 1 if did in hfr_dids else 0
    if did in hard_excl:
        return {
            "did": did,
            "fetched_utc": fetched_utc,
            "http_status": 451,
            "raw_sha256": "",
            "is_hfr": is_hfr,
            "avatar_url": "",
            "live_handle": "",
            "live_name": "",
            "live_safety_flag": "hard_excluded",
            "n_items": 0,
            "n_orig": 0,
            "n_quote": 0,
            "n_reply": 0,
            "n_repost": 0,
            "repost_frac": 0.0,
            "reply_frac": 0.0,
            "authored_frac": 0.0,
            "days_since_last_any": 9999.0,
            "days_since_last_authored": 9999.0,
            "ext_link_frac": 0.0,
            "top_ext_domain": "",
            "same_domain_frac": 0.0,
            "live_dup_text_frac": 0.0,
            "live_pol_frac": 0.0,
            "median_likes": 0.0,
            "avg_hashtags": 0.0,
            "bare_link_frac": 0.0,
            "live_sem_cos": None,
            "live_verdict": "drop_hard_excluded",
            "posts_json": "[]",
        }

    url = (
        "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?"
        + urllib.parse.urlencode({"actor": did, "limit": str(limit)})
    )
    raw = b""
    status_code = 0
    for attempt in range(3):
        limiter.wait()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = resp.read()
                status_code = int(resp.status)
                break
        except urllib.error.HTTPError as e:
            status_code = int(e.code)
            raw = e.read() if hasattr(e, "read") else b""
            if status_code == 429:
                ra = float(e.headers.get("Retry-After") or (2.5 * (attempt + 1)))
                limiter.penalize(min(max(ra, 1.5), 30.0))
                continue
            if status_code >= 500 and attempt < 2:
                time.sleep(0.8 * (attempt + 1))
                continue
            break
        except Exception:
            if attempt < 2:
                time.sleep(0.6 * (attempt + 1))
                continue
            break

    raw_sha = hashlib.sha256(raw).hexdigest() if raw else ""
    if status_code != 200 or not raw:
        safety_fl = "account_gone_or_blocked" if status_code in (400, 404) else ""
        base_r = {
            "did": did,
            "fetched_utc": fetched_utc,
            "http_status": status_code,
            "raw_sha256": raw_sha,
            "is_hfr": is_hfr,
            "avatar_url": "",
            "live_handle": "",
            "live_name": "",
            "live_safety_flag": safety_fl,
            "n_items": 0,
            "n_orig": 0,
            "n_quote": 0,
            "n_reply": 0,
            "n_repost": 0,
            "repost_frac": 0.0,
            "reply_frac": 0.0,
            "authored_frac": 0.0,
            "days_since_last_any": 9999.0,
            "days_since_last_authored": 9999.0,
            "ext_link_frac": 0.0,
            "top_ext_domain": "",
            "same_domain_frac": 0.0,
            "live_dup_text_frac": 0.0,
            "live_pol_frac": 0.0,
            "median_likes": 0.0,
            "avg_hashtags": 0.0,
            "bare_link_frac": 0.0,
            "live_sem_cos": None,
            "live_verdict": "",
            "posts_json": "[]",
        }
        base_r["live_verdict"] = classify_live_verdict(base_r)
        return base_r

    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        payload = {}
    feed = payload.get("feed") or []

    avatar_url = ""
    live_handle = ""
    live_name = ""
    safety_flag = ""

    n_items = 0
    n_orig = 0
    n_quote = 0
    n_reply = 0
    n_repost = 0
    days_any: list[float] = []
    days_auth: list[float] = []
    domains: list[str] = []
    authored_with_ext = 0
    authored_total = 0
    authored_oq_total = 0
    oq_tag_counts: list[int] = []
    oq_bare_links = 0
    authored_texts_norm: list[str] = []
    pol_hits = 0
    orig_likes: list[int] = []

    kept_originals: list[dict] = []
    kept_others: list[dict] = []

    for item in feed:
        if not isinstance(item, dict):
            continue
        pv = item.get("post") or {}
        if not isinstance(pv, dict):
            continue
        reason = item.get("reason") or {}
        is_repost = isinstance(reason, dict) and "reasonRepost" in (reason.get("$type") or "")
        au = pv.get("author") or {}
        adid = (au.get("did") or "").strip()
        ahdl = (au.get("handle") or "").strip()
        anm = (au.get("displayName") or "").strip()

        if adid == did and not live_handle:
            live_handle = ahdl
            live_name = anm[:60]
            av = (au.get("avatar") or "").strip()
            albls = [l.get("val", "") for l in (au.get("labels") or []) if isinstance(l, dict) and not l.get("neg")]
            if albls:
                safety_flag = f"author_label:{albls[0]}"
            elif av.startswith("https://cdn.bsky.app/") and not is_hfr:
                avatar_url = av

        rec = pv.get("record") or {}
        if not isinstance(rec, dict):
            continue

        plbls = [l.get("val", "").lower() for l in (pv.get("labels") or []) if isinstance(l, dict) and not l.get("neg")]
        slbls = []
        if isinstance(rec.get("labels"), dict):
            slbls = [
                (v.get("val") or "").lower()
                for v in (rec["labels"].get("values") or [])
                if isinstance(v, dict) and v.get("val")
            ]
        all_post_lbls = set(plbls) | set(slbls)
        if adid == did and (all_post_lbls & ADULT_POST_LABEL_VALS):
            safety_flag = safety_flag or f"post_adult_label:{sorted(all_post_lbls & ADULT_POST_LABEL_VALS)[0]}"

        n_items += 1
        created_at = (rec.get("createdAt") or pv.get("indexedAt") or "").strip()
        d_age = _parse_iso_days(created_at, now_dt)
        if d_age is not None:
            days_any.append(d_age)

        txt = (rec.get("text") or "").strip()
        embed = pv.get("embed")
        is_reply = bool(rec.get("reply")) and not is_repost
        has_quote = isinstance(embed, dict) and (
            "embed.record" in (embed.get("$type") or "") or "recordWithMedia" in (embed.get("$type") or "")
        )

        if is_repost:
            n_repost += 1
            kind = "repost"
        elif is_reply:
            n_reply += 1
            kind = "reply"
        elif has_quote:
            n_quote += 1
            kind = "quote"
        else:
            n_orig += 1
            kind = "post"

        allow_media = (not is_hfr) and (not safety_flag) and (not all_post_lbls) and (adid not in hard_excl)
        mtype, thumbs, alts, ext_uri, ext_title = _extract_embed_info(embed, allow_media)
        quote_snip = _extract_quote_snippet(embed) if has_quote and allow_media else None

        blob_for_pol = f"{txt} {ext_title} {(quote_snip or {}).get('text', '')}"
        item_pol = bool(POL_RE.search(blob_for_pol))
        if item_pol:
            pol_hits += 1

        ext_dom = ""
        if ext_uri:
            m_u = URL_RE.match(ext_uri)
            if m_u:
                ext_dom = m_u.group(1).lower().removeprefix("www.")
        elif txt:
            m_u = URL_RE.search(txt)
            if m_u:
                ext_dom = m_u.group(1).lower().removeprefix("www.")

        if not is_repost and adid == did:
            authored_total += 1
            if d_age is not None:
                days_auth.append(d_age)
            if ext_dom:
                authored_with_ext += 1
                domains.append(ext_dom)
            stripped_no_url = URL_STRIP_RE.sub("", txt)
            norm_t = WS_RE.sub(" ", stripped_no_url.lower()).strip()
            if len(norm_t) >= 18:
                authored_texts_norm.append(norm_t)
            if kind in ("post", "quote"):
                authored_oq_total += 1
                orig_likes.append(int(pv.get("likeCount") or 0))
                oq_tag_counts.append(len(HASHTAG_RE.findall(txt)))
                if (ext_uri or ext_dom) and len(stripped_no_url.strip()) < 20:
                    oq_bare_links += 1

        # Never store post text/media for hide_from_recs accounts (Amendment A1) or labelled posts
        if is_hfr or all_post_lbls or (adid in hard_excl) or (adid in hfr_dids):
            continue

        uri = (pv.get("uri") or "").strip()
        rkey = uri.rsplit("/", 1)[-1] if "/app.bsky.feed.post/" in uri else ""
        entry = {
            "uri": uri,
            "rkey": rkey,
            "kind": kind,
            "author_handle": ahdl,
            "author_name": anm[:50],
            "text": txt[:600],
            "created_at": created_at,
            "likes": int(pv.get("likeCount") or 0),
            "reposts": int(pv.get("repostCount") or 0),
            "replies": int(pv.get("replyCount") or 0),
            "media_type": mtype,
            "thumbs": thumbs[:3],
            "alts": alts[:2],
            "ext_uri": ext_uri,
            "ext_title": ext_title,
            "ext_domain": ext_dom,
            "quote": quote_snip,
            "is_political": item_pol,
        }
        if kind in ("post", "quote"):
            kept_originals.append(entry)
        else:
            kept_others.append(entry)

    dup_hits = 0
    if len(authored_texts_norm) >= 4:
        for i, ti in enumerate(authored_texts_norm):
            p_i = ti[:42]
            toks_i = set(ti.split())
            is_d = False
            for j, tj in enumerate(authored_texts_norm):
                if i == j:
                    continue
                if len(p_i) >= 28 and ti[:42] == tj[:42]:
                    is_d = True
                    break
                toks_j = set(tj.split())
                if len(toks_i) >= 5 and len(toks_j) >= 5:
                    jac = len(toks_i & toks_j) / max(len(toks_i | toks_j), 1)
                    if jac >= 0.82:
                        is_d = True
                        break
            if is_d:
                dup_hits += 1
    dup_frac = float(dup_hits) / max(len(authored_texts_norm), 1) if len(authored_texts_norm) >= 4 else 0.0

    top_dom = ""
    same_dom_frac = 0.0
    if domains:
        dom_cnt = collections.Counter(domains)
        top_dom, top_c = dom_cnt.most_common(1)[0]
        same_dom_frac = float(top_c) / max(authored_total, 1)

    avg_tags = round(float(sum(oq_tag_counts)) / max(len(oq_tag_counts), 1), 3) if oq_tag_counts else 0.0
    bare_frac = round(float(oq_bare_links) / max(authored_oq_total, 1), 4) if authored_oq_total else 0.0

    showcase = (kept_originals[:8] + kept_others[:4])[:10]

    row = {
        "did": did,
        "fetched_utc": fetched_utc,
        "http_status": status_code,
        "raw_sha256": raw_sha,
        "is_hfr": is_hfr,
        "avatar_url": avatar_url,
        "live_handle": live_handle,
        "live_name": live_name,
        "live_safety_flag": safety_flag,
        "n_items": n_items,
        "n_orig": n_orig,
        "n_quote": n_quote,
        "n_reply": n_reply,
        "n_repost": n_repost,
        "repost_frac": round(float(n_repost) / max(n_items, 1), 4),
        "reply_frac": round(float(n_reply) / max(n_items, 1), 4),
        "authored_frac": round(float(n_orig + n_quote) / max(n_items, 1), 4),
        "days_since_last_any": round(min(days_any) if days_any else 9999.0, 2),
        "days_since_last_authored": round(min(days_auth) if days_auth else 9999.0, 2),
        "ext_link_frac": round(float(authored_with_ext) / max(authored_total, 1), 4),
        "top_ext_domain": top_dom,
        "same_domain_frac": round(same_dom_frac, 4),
        "live_dup_text_frac": round(dup_frac, 4),
        "live_pol_frac": round(float(pol_hits) / max(n_items, 1), 4),
        "median_likes": float(np.median(orig_likes)) if orig_likes else 0.0,
        "avg_hashtags": avg_tags,
        "bare_link_frac": bare_frac,
        "live_sem_cos": None,
        "live_verdict": "",
        "posts_json": json.dumps(showcase, ensure_ascii=False, separators=(",", ":")) if not safety_flag else "[]",
    }
    row["live_verdict"] = classify_live_verdict(row)
    return row


UPSERT_SQL = """
INSERT INTO author_feeds (
    did, fetched_utc, http_status, raw_sha256, is_hfr, avatar_url, live_handle, live_name,
    live_safety_flag, n_items, n_orig, n_quote, n_reply, n_repost, repost_frac, reply_frac,
    authored_frac, days_since_last_any, days_since_last_authored, ext_link_frac, top_ext_domain,
    same_domain_frac, live_dup_text_frac, live_pol_frac, median_likes, avg_hashtags, bare_link_frac,
    live_sem_cos, live_verdict, posts_json
) VALUES (
    :did, :fetched_utc, :http_status, :raw_sha256, :is_hfr, :avatar_url, :live_handle, :live_name,
    :live_safety_flag, :n_items, :n_orig, :n_quote, :n_reply, :n_repost, :repost_frac, :reply_frac,
    :authored_frac, :days_since_last_any, :days_since_last_authored, :ext_link_frac, :top_ext_domain,
    :same_domain_frac, :live_dup_text_frac, :live_pol_frac, :median_likes, :avg_hashtags, :bare_link_frac,
    :live_sem_cos, :live_verdict, :posts_json
)
ON CONFLICT(did) DO UPDATE SET
    fetched_utc=excluded.fetched_utc,
    http_status=excluded.http_status,
    raw_sha256=excluded.raw_sha256,
    is_hfr=excluded.is_hfr,
    avatar_url=excluded.avatar_url,
    live_handle=excluded.live_handle,
    live_name=excluded.live_name,
    live_safety_flag=excluded.live_safety_flag,
    n_items=excluded.n_items,
    n_orig=excluded.n_orig,
    n_quote=excluded.n_quote,
    n_reply=excluded.n_reply,
    n_repost=excluded.n_repost,
    repost_frac=excluded.repost_frac,
    reply_frac=excluded.reply_frac,
    authored_frac=excluded.authored_frac,
    days_since_last_any=excluded.days_since_last_any,
    days_since_last_authored=excluded.days_since_last_authored,
    ext_link_frac=excluded.ext_link_frac,
    top_ext_domain=excluded.top_ext_domain,
    same_domain_frac=excluded.same_domain_frac,
    live_dup_text_frac=excluded.live_dup_text_frac,
    live_pol_frac=excluded.live_pol_frac,
    median_likes=excluded.median_likes,
    avg_hashtags=excluded.avg_hashtags,
    bare_link_frac=excluded.bare_link_frac,
    live_sem_cos=COALESCE(excluded.live_sem_cos, author_feeds.live_sem_cos),
    live_verdict=excluded.live_verdict,
    posts_json=excluded.posts_json;
"""


def score_live_posts_eg2(
    conn: sqlite3.Connection,
    did_to_home_cid: dict[str, int],
    centroids_npz_path: pathlib.Path,
    model_dir: pathlib.Path,
) -> int:
    """Encode hydrated live authored posts on Mac MPS with EG2 (`bf16`) and update `live_sem_cos`."""
    if not centroids_npz_path.exists() or not model_dir.exists():
        return 0
    c_npz = np.load(centroids_npz_path)
    cids = c_npz["cids"].astype(np.int32)
    cid_to_idx = {int(c): i for i, c in enumerate(cids)}
    mu_tx = c_npz["mu_text_posts"].astype(np.float32)
    cent_tx = c_npz["cent_text_posts"].astype(np.float32)

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM author_feeds WHERE live_sem_cos IS NULL AND posts_json != '[]'"
    ).fetchall()
    if not rows:
        return 0

    import torch

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "s12mm"))
    import encoder as eg2_enc  # noqa: E402
    import mmlib as L  # noqa: E402

    all_texts: list[str] = []
    spans: list[tuple[str, int, int, int, dict]] = []
    for r in rows:
        rd = dict(r)
        did = rd["did"]
        hc = did_to_home_cid.get(did)
        if hc is None or hc not in cid_to_idx:
            continue
        try:
            posts = json.loads(rd["posts_json"])
        except Exception:
            continue
        txs = []
        for p in posts:
            if p.get("kind") in ("post", "quote", "reply"):
                body = (p.get("text") or "").strip()
                alts = p.get("alts") or []
                if alts:
                    body = (body + " " + " ".join(f"[image: {a}]" for a in alts)).strip()
                if p.get("ext_title") and len(body) < 40:
                    body = f"{body} {p['ext_title']}".strip()
                if len(body) >= 4:
                    txs.append(body[:1000])
        if not txs:
            continue
        s0 = len(all_texts)
        all_texts.extend(txs[:8])
        spans.append((did, cid_to_idx[hc], s0, len(all_texts), rd))

    if not all_texts:
        return 0

    print(f"[hydrate-feeds] Encoding {len(all_texts):,} live posts across {len(spans):,} accounts on MPS with EG2...", flush=True)
    enc = eg2_enc.EG2(model_dir)
    v256_chunks = []
    for i0 in range(0, len(all_texts), 1024):
        fv = enc.encode_text(all_texts[i0 : i0 + 1024], bs=24)
        v256_chunks.append(L.to_stored(fv).astype(np.float32))
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    v256 = np.concatenate(v256_chunks, axis=0)

    updates = []
    for did, cidx, s0, s1, meta_r in spans:
        pv = v256[s0:s1].mean(axis=0)
        pv = pv / max(float(np.linalg.norm(pv)), 1e-12)
        pvc = pv - mu_tx
        pvc = pvc / max(float(np.linalg.norm(pvc)), 1e-12)
        sim = float(np.dot(pvc, cent_tx[cidx]))
        verdict = classify_live_verdict(meta_r)
        if verdict == "clean" and (s1 - s0) >= 3 and sim < 0.04:
            verdict = "warn_low_live_coherence"
        updates.append((round(sim, 4), verdict, did))

    conn.executemany("UPDATE author_feeds SET live_sem_cos = ?, live_verdict = ? WHERE did = ?", updates)
    conn.commit()
    return len(updates)


def build_priority_queue(lake: pathlib.Path, out_dir: pathlib.Path, max_dids: int) -> tuple[list[str], dict[str, int]]:
    """Prioritize:
    1. All 25 candidates in each of the 30 rule-drawn S17 review cards,
    2. All borderline/suspect candidates in top-25 of any card (embedding margin, slab dup, mass3k ring, modlists, C5 bot/brand),
    3. Top-4 showcase accounts of major scenes and top-2 of every card.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    cards_path = out_dir / "cards_v1.json"
    sig_path = out_dir / "candidate_signals.parquet"

    did_to_home: dict[str, int] = {}
    ordered: list[str] = []
    seen: set[str] = set()

    def push(d: str) -> None:
        if d and d not in seen and len(ordered) < max_dids:
            seen.add(d)
            ordered.append(d)

    if sig_path.exists():
        ts_home = pq.read_table(sig_path, columns=["did", "community"])
        for d_s, c_i in zip(ts_home["did"].to_pylist(), ts_home["community"].to_pylist()):
            did_to_home[d_s] = int(c_i)
        del ts_home

    if cards_path.exists():
        import base64

        c_obj = json.loads(cards_path.read_text())
        accts = c_obj.get("accts") or []

        def exp_did(s: str) -> str:
            if s.startswith("did:"):
                return s
            pad = "=" * ((4 - len(s) % 4) % 4)
            return "did:plc:" + base64.b32encode(base64.urlsafe_b64decode(s + pad)).decode("ascii").lower()

        acct_dids = [exp_did(r[0]) for r in accts]
        for cid_s, cd_m in (c_obj.get("cards") or {}).items():
            for aid_m in cd_m.get("m") or []:
                did_to_home[acct_dids[aid_m]] = int(cid_s)

        # 1. All 30 S17 review cards (every candidate!)
        for item in c_obj.get("blind_review_30") or []:
            cd = c_obj["cards"].get(str(item["c"])) or {}
            for aid in cd.get("m") or []:
                push(acct_dids[aid])

        # 2. All borderline / suspect candidates across all v0 cards
        if sig_path.exists():
            ts = pq.read_table(sig_path)
            ts = ts.filter(pc.is_in(ts["did"], value_set=pa.array(acct_dids, type=pa.string())))
            sig_by_did: dict[str, dict] = {r["did"]: r for r in ts.to_pylist()}
            suspects: list[tuple[float, str]] = []
            for d in acct_dids:
                sr = sig_by_did.get(d)
                if not sr:
                    suspects.append((-10.0, d))
                    continue
                sp = sr["sem_cos_posts"]
                t1 = sr["t1_cos"]
                dup = sr["slab_near_dup_frac"]
                fr = sr["follow_ratio"]
                ppd = sr["posts_per_day"]
                v9 = sr["v9_demotion_gap"]
                m3k = float(sr.get("pq_mass3k_share") or 0.0)
                modl = int(sr.get("pq_modlists") or 0)
                c5_bot = sr.get("c5_p_bot")
                c5_brand = sr.get("c5_p_brand")
                margin = (
                    (sr["best_other_sem_cos"] - sp)
                    if (
                        sp is not None
                        and not np.isnan(sp)
                        and sr["best_other_sem_cos"] is not None
                        and not np.isnan(sr["best_other_sem_cos"])
                    )
                    else 0.0
                )
                risk = 0.0
                if not sr["has_vec"] or sr["mm_post_count"] < 3:
                    risk += 2.4
                if sp is not None and not np.isnan(sp) and sp < 0.18:
                    risk += (0.25 - sp) * 10.0
                if t1 is not None and not np.isnan(t1) and t1 < 0.18:
                    risk += (0.25 - t1) * 8.0
                if margin > 0.15:
                    risk += margin * 6.0
                if dup >= 0.42 and sr["slab_n_posts"] >= 6:
                    risk += dup * 5.0
                if fr >= 1.4 and sr["follows_count"] >= 1800:
                    risk += 3.0
                if m3k >= 0.24:
                    risk += (m3k - 0.18) * 10.0
                if modl >= 8:
                    risk += min(4.0, modl * 0.2)
                if c5_bot is not None and not np.isnan(c5_bot) and c5_bot >= 0.25:
                    risk += float(c5_bot) * 6.0
                if c5_brand is not None and not np.isnan(c5_brand) and c5_brand >= 0.65:
                    risk += 2.2
                if ppd >= 30.0:
                    risk += 2.5
                if v9 >= 0.35:
                    risk += v9 * 3.0
                if risk >= 1.5:
                    suspects.append((-risk, d))
            suspects.sort()
            for _, d in suspects:
                push(d)

        # 3. Top showcase accounts across scenes & cards
        for sc in c_obj.get("scenes") or []:
            for cid in (sc.get("cs") or [])[:4]:
                cd = c_obj["cards"].get(str(cid)) or {}
                for aid in (cd.get("m") or [])[:4]:
                    push(acct_dids[aid])
        for _cid_s, cd in (c_obj.get("cards") or {}).items():
            for aid in (cd.get("m") or [])[:2]:
                push(acct_dids[aid])

    return ordered, did_to_home


def main() -> None:
    t0 = time.time()
    lake = pathlib.Path.home() / "catbird-lake-local"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(lake / "s17-picker-v1-20261030"))
    ap.add_argument("--max-dids", type=int, default=4600)
    ap.add_argument("--rps", type=float, default=8.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-eg2", action="store_true")
    ap.add_argument("--rescore-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out).expanduser()
    db_path = out_dir / "author_feeds.sqlite"
    conn = init_db(db_path)

    n_rescored = rescore_cached_verdicts(conn)
    print(f"[hydrate-feeds] Rescored {n_rescored:,} existing cached feeds under rev-4 live spam rules.", flush=True)
    if args.rescore_only:
        verdicts = dict(
            conn.execute("SELECT live_verdict, COUNT(*) FROM author_feeds GROUP BY live_verdict ORDER BY COUNT(*) DESC").fetchall()
        )
        print(f"[hydrate-feeds] Rescore-only complete. Verdict summary: {json.dumps(verdicts)}", flush=True)
        return

    s5_dir = lake / "s5-sweep-20261030"
    s13_dir = lake / "s13-elig-20261030"
    hfr_dids = {
        ln.split("\t")[0].strip()
        for ln in open(lake / "s5-declarations-20261027/hide_from_recs_union.tsv")
        if ln.startswith("did:")
    }
    hard_excl = (
        {ln.strip() for ln in open(s5_dir / "purge_dids.txt") if ln.startswith("did:")}
        | {ln.strip() for ln in open(s13_dir / "base_safety_exclude_dids.txt") if ln.startswith("did:")}
        | {ln.strip() for ln in open(s5_dir / "eligibility_rule2_exclude_dids.txt") if ln.startswith("did:")}
    )

    queue_dids, did_to_home = build_priority_queue(lake, out_dir, args.max_dids)
    already = {
        r[0]
        for r in conn.execute(
            "SELECT did FROM author_feeds WHERE http_status = 200 OR http_status IN (400, 404, 451)"
        ).fetchall()
    }
    todo = [d for d in queue_dids if d not in already]
    print(
        f"[hydrate-feeds] Priority queue={len(queue_dids):,}, already cached={len(queue_dids) - len(todo):,}, fetching live={len(todo):,} at <={args.rps} req/s...",
        flush=True,
    )

    limiter = PoliteRateLimiter(max_rps=args.rps)
    done = 0
    batch_rows: list[dict] = []
    if todo:
        with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
            futs = [ex.submit(fetch_and_parse_one, d, limiter, hfr_dids, hard_excl, 30) for d in todo]
            for fut in cf.as_completed(futs):
                row = fut.result()
                batch_rows.append(row)
                done += 1
                if len(batch_rows) >= 50:
                    conn.executemany(UPSERT_SQL, batch_rows)
                    conn.commit()
                    batch_rows.clear()
                if done % 200 == 0 or done == len(todo):
                    el = max(time.time() - t0, 0.001)
                    print(f"[hydrate-feeds] {done:,}/{len(todo):,} ({done/el:.2f} req/s)", flush=True)
        if batch_rows:
            conn.executemany(UPSERT_SQL, batch_rows)
            conn.commit()

    if not args.skip_eg2:
        model_dir = lake / "models/embeddinggemma-2"
        n_scored = score_live_posts_eg2(conn, did_to_home, out_dir / "card_centroids.npz", model_dir)
        print(f"[hydrate-feeds] Scored {n_scored:,} hydrated feeds with EG2 live-post cosine.", flush=True)

    verdicts = dict(
        conn.execute("SELECT live_verdict, COUNT(*) FROM author_feeds GROUP BY live_verdict ORDER BY COUNT(*) DESC").fetchall()
    )
    print(f"[hydrate-feeds] Complete in {time.time() - t0:.1f}s. Verdict summary: {json.dumps(verdicts)}", flush=True)


if __name__ == "__main__":
    main()
