#!/usr/bin/env python3
"""Localhost-only server for S17 Inspector v2 (`127.0.0.1:8765`, never 0.0.0.0, never published).

Endpoints:
- GET  /                             -> page_s17.html (Inspector v2)
- GET  /cards_v1.json                -> verified <=2 MB v0 catalog
- GET  /cards_descriptions.json      -> full uncut scene descriptions & organic quality metadata
- GET  /inspector_index.json         -> per-account embedding/spam/live-feed audit index & evicted showcase
- GET  /blind_review_key.json        -> unblinded provenance key for the 30-card sign-off draw
- GET  /api/author_feed?did=...      -> cached or polite live `getAuthorFeed` (0 image bytes on disk; CDN URLs only)
- GET  /api/author_feeds_batch?dids= -> batch read from SQLite + polite live hydration for up to 8 missing DIDs/call
- GET  /api/overrides                -> current manual overrides (`excluded_dids`, `excluded_cards`, `approved_dids`)
- POST /api/overrides                -> CSRF-protected atomic update of `manual_overrides.json`
- POST /api/rebuild                  -> CSRF-protected re-execution of `build_s17.py` + hot state reload
- GET  /api/pack?uri=...             -> inspect-only starter pack members from packs_members.json
- GET  /api/preview?cards=           -> live-hydrated "Preview from Oct 22 data" (<=2/author, <=5% high-H, 0 deleted/labelled/hfr/bot)
- GET  /api/medoids?uris=            -> live-hydrated medoid sample posts (text only, 0 image bytes)
- GET  /api/ratings                  -> saved S17 sign-off ratings (+ ?reveal=1 scored against blind_review_key.json)
- POST /api/ratings                  -> atomically save S17 sign-off ratings to ratings.json (CSRF-protected same-origin JSON)
"""
from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures as cf
import datetime as dt
import http.server
import json
import pathlib
import re
import sqlite3
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request

import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hydrate_author_feeds as haf  # noqa: E402

POL_RE = haf.POL_RE

LIVE_RAW_ADULT_RE = re.compile(
    r"(?:🔞|(?<!\w)18\+|(?<!\w)21\+|(?<!\w)r-?18\b|\bmdni\b|\bminors?\s+dni\b|\badults?\s+only\b)",
    re.I,
)
LIVE_HDL_BOT_RE = re.compile(
    r"(?:\.bot$|\.ap\.brid\.gy$|(?:^|[.-])(?:bot|rss|feed|mirror|alerts?)(?:$|[.-]))",
    re.I,
)
LIVE_NAME_BOT_RE = re.compile(
    r"(?:\[bot\]|\(bot\)|\b(?:unofficial|automated|mirror|rss|repost)\s+bot\b|\[twitter/x\s*mirror\]|\(mirror\)|#(?:f4f|bluecrew|ifb|fbr)\b|@f4f\b)",
    re.I,
)


def hydrate_posts_live(
    uris: list[str],
    hfr_dids: set[str],
    hard_excl: set[str],
    limiter: haf.PoliteRateLimiter,
) -> list[dict]:
    """Hydrate up to 100 AT-URIs in batches of 25 via public.api.bsky.app getPosts; drop deleted/labelled/hfr/bots."""
    clean_out: list[dict] = []
    for s in range(0, min(len(uris), 100), 25):
        batch = [u for u in uris[s : s + 25] if u.startswith("at://")]
        if not batch:
            continue
        limiter.wait()
        qs = "&".join("uris=" + urllib.parse.quote(u, safe="") for u in batch)
        url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": haf.USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            continue
        by_uri = {p.get("uri"): p for p in (data.get("posts") or []) if isinstance(p, dict)}
        for u in batch:
            pv = by_uri.get(u)
            if not pv:
                continue
            au = pv.get("author") or {}
            adid = au.get("did") or ""
            ahandle = au.get("handle") or ""
            aname = au.get("displayName") or ""
            if not adid or adid in hfr_dids or adid in hard_excl or ahandle == "handle.invalid":
                continue
            if LIVE_RAW_ADULT_RE.search(aname) or LIVE_HDL_BOT_RE.search(ahandle) or LIVE_NAME_BOT_RE.search(aname):
                continue
            alabels = [l.get("val", "") for l in (au.get("labels") or []) if isinstance(l, dict) and not l.get("neg")]
            plabels = [l.get("val", "") for l in (pv.get("labels") or []) if isinstance(l, dict) and not l.get("neg")]
            rec = pv.get("record") or {}
            slabels = ((rec.get("labels") or {}).get("values") or []) if isinstance(rec.get("labels"), dict) else []
            if alabels or plabels or slabels:
                continue
            txt = (rec.get("text") or "").strip()
            embed = pv.get("embed") or {}
            mtype, thumbs, alts, ext_u, ext_t = haf._extract_embed_info(embed, allow_media_urls=True)
            av = (au.get("avatar") or "").strip()
            clean_out.append(
                {
                    "uri": u,
                    "rkey": u.rsplit("/", 1)[-1],
                    "did": adid,
                    "handle": ahandle,
                    "name": (au.get("displayName") or "")[:50],
                    "avatar": av if av.startswith("https://cdn.bsky.app/") else "",
                    "text": txt[:500],
                    "created_at": rec.get("createdAt") or pv.get("indexedAt") or "",
                    "likes": int(pv.get("likeCount") or 0),
                    "reposts": int(pv.get("repostCount") or 0),
                    "replies": int(pv.get("replyCount") or 0),
                    "media_badge": mtype,
                    "thumbs": thumbs[:2],
                    "alts": alts[:2],
                    "ext_uri": ext_u,
                    "ext_title": ext_t,
                }
            )
    return clean_out


def format_feed_row(row_dict: dict) -> dict:
    """Convert an `author_feeds` SQLite row dict into JSON-serializable response format."""
    try:
        posts = json.loads(row_dict.get("posts_json") or "[]")
    except Exception:
        posts = []
    if row_dict.get("is_hfr"):
        posts = []
    return {
        "did": row_dict["did"],
        "fetched_utc": row_dict.get("fetched_utc") or "",
        "http_status": int(row_dict.get("http_status") or 0),
        "is_hfr": bool(row_dict.get("is_hfr")),
        "avatar_url": row_dict.get("avatar_url") or "",
        "live_handle": row_dict.get("live_handle") or "",
        "live_name": row_dict.get("live_name") or "",
        "live_safety_flag": row_dict.get("live_safety_flag") or "",
        "n_items": int(row_dict.get("n_items") or 0),
        "n_orig": int(row_dict.get("n_orig") or 0),
        "n_quote": int(row_dict.get("n_quote") or 0),
        "n_reply": int(row_dict.get("n_reply") or 0),
        "n_repost": int(row_dict.get("n_repost") or 0),
        "repost_frac": round(float(row_dict.get("repost_frac") or 0.0), 3),
        "reply_frac": round(float(row_dict.get("reply_frac") or 0.0), 3),
        "authored_frac": round(float(row_dict.get("authored_frac") or 0.0), 3),
        "days_since_last_any": round(float(row_dict.get("days_since_last_any") or 9999.0), 1),
        "days_since_last_authored": round(float(row_dict.get("days_since_last_authored") or 9999.0), 1),
        "top_ext_domain": row_dict.get("top_ext_domain") or "",
        "same_domain_frac": round(float(row_dict.get("same_domain_frac") or 0.0), 3),
        "live_dup_text_frac": round(float(row_dict.get("live_dup_text_frac") or 0.0), 3),
        "live_pol_frac": round(float(row_dict.get("live_pol_frac") or 0.0), 3),
        "median_likes": round(float(row_dict.get("median_likes") or 0.0), 1),
        "live_sem_cos": row_dict.get("live_sem_cos"),
        "live_verdict": row_dict.get("live_verdict") or "",
        "posts": posts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    lake = pathlib.Path.home() / "catbird-lake-local"
    ap.add_argument("--out", default=str(lake / "s17-picker-v1-20261030"))
    ap.add_argument("--s16-dir", default=str(lake / "s16-s17-20261030"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out).expanduser()
    s16_dir = pathlib.Path(args.s16_dir).expanduser()
    here_dir = pathlib.Path(__file__).resolve().parent
    page_path = here_dir / "page_s17.html"
    build_script = here_dir / "build_s17.py"

    cards_path = out_dir / "cards_v1.json"
    descs_path = out_dir / "cards_descriptions.json"
    inspector_path = out_dir / "inspector_index.json"
    packs_path = out_dir / "packs_members.json"
    blind_key_path = out_dir / "blind_review_key.json"
    ratings_path = out_dir / "ratings.json"
    overrides_path = out_dir / "manual_overrides.json"
    non_org_path = out_dir / "non_organic_exclude_dids.txt"
    af_db_path = out_dir / "author_feeds.sqlite"

    allowed_origins = {f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"}

    # Polite shared rate limiter (`<= 8.5 req/s`) and SQLite WAL handle
    limiter = haf.PoliteRateLimiter(max_rps=8.5)
    db_conn = haf.init_db(af_db_path)
    db_conn.row_factory = sqlite3.Row
    db_lock = threading.Lock()
    state_lock = threading.Lock()

    s5_dir = lake / "s5-sweep-20261030"
    s13_dir = lake / "s13-elig-20261030"
    hfr_dids = {
        ln.split("\t")[0].strip()
        for ln in open(lake / "s5-declarations-20261027/hide_from_recs_union.tsv")
        if ln.startswith("did:")
    }
    base_hard_excl = (
        {ln.strip() for ln in open(s5_dir / "purge_dids.txt") if ln.startswith("did:")}
        | {ln.strip() for ln in open(s13_dir / "base_safety_exclude_dids.txt") if ln.startswith("did:")}
        | {ln.strip() for ln in open(s5_dir / "eligibility_rule2_exclude_dids.txt") if ln.startswith("did:")}
    )

    state: dict = {}
    preview_cache: dict[str, dict] = {}

    def read_overrides_unlocked() -> dict:
        if overrides_path.exists():
            try:
                ov = json.loads(overrides_path.read_text())
                return {
                    "excluded_dids": dict(ov.get("excluded_dids") or {}),
                    "excluded_cards": dict(ov.get("excluded_cards") or {}),
                    "approved_dids": sorted(set(ov.get("approved_dids") or [])),
                    "updated_utc": ov.get("updated_utc") or "",
                }
            except Exception:
                pass
        return {"excluded_dids": {}, "excluded_cards": {}, "approved_dids": [], "updated_utc": ""}

    def reload_catalog_unlocked() -> None:
        cb = cards_path.read_bytes()
        db = descs_path.read_bytes() if descs_path.exists() else b"{}"
        ib = inspector_path.read_bytes() if inspector_path.exists() else b"{}"
        co = json.loads(cb)
        pm = json.loads(packs_path.read_text()) if packs_path.exists() else {}
        bk = json.loads(blind_key_path.read_text()) if blind_key_path.exists() else {}
        ov = read_overrides_unlocked()
        he = set(base_hard_excl)
        if non_org_path.exists():
            he |= {ln.split("\t")[0].strip() for ln in open(non_org_path) if ln.startswith("did:")}
        he |= set(ov["excluded_dids"].keys())
        state["cards_bytes"] = cb
        state["descs_bytes"] = db
        state["inspector_bytes"] = ib
        state["cards_obj"] = co
        state["packs_map"] = pm
        state["blind_key"] = bk
        state["overrides"] = ov
        state["hard_excl"] = he
        preview_cache.clear()

    reload_catalog_unlocked()

    # Load high-H pairs
    high_h_by_actor: dict[int, set[int]] = collections.defaultdict(set)
    t_h = pq.read_table(
        lake / "s14-hostility-20261030/H_prod_cand5_full.parquet",
        columns=["c_actor", "c_target", "is_high_top1pct"],
    )
    for ca, ct, hi in zip(t_h["c_actor"].to_pylist(), t_h["c_target"].to_pylist(), t_h["is_high_top1pct"].to_pylist()):
        if hi:
            high_h_by_actor[int(ca)].add(int(ct))

    # Load Oct 22 preview post bank
    preview_by_comm: dict[int, list[dict]] = collections.defaultdict(list)
    oct22_pq = s16_dir / "oct22_preview_posts.parquet"
    if oct22_pq.exists():
        tp = pq.read_table(oct22_pq)
        for r in tp.to_pylist():
            if r["did"] not in hfr_dids and r["did"] not in base_hard_excl:
                preview_by_comm[int(r["community"])].append(r)

    def db_get_many(dids: list[str]) -> dict[str, dict]:
        if not dids:
            return {}
        placeholders = ",".join("?" for _ in dids)
        out: dict[str, dict] = {}
        with db_lock:
            for r in db_conn.execute(f"SELECT * FROM author_feeds WHERE did IN ({placeholders})", dids):
                out[r["did"]] = format_feed_row(dict(r))
        return out

    def hydrate_and_store_many(dids: list[str]) -> dict[str, dict]:
        if not dids:
            return {}
        results: list[dict] = []
        if len(dids) == 1:
            results.append(haf.fetch_and_parse_one(dids[0], limiter, hfr_dids, base_hard_excl, limit=25))
        else:
            with cf.ThreadPoolExecutor(max_workers=min(3, len(dids))) as pool:
                futs = [pool.submit(haf.fetch_and_parse_one, d, limiter, hfr_dids, base_hard_excl, 25) for d in dids]
                for f in cf.as_completed(futs):
                    results.append(f.result())
        with db_lock:
            db_conn.executemany(haf.UPSERT_SQL, results)
            db_conn.commit()
        return {r["did"]: format_feed_row(r) for r in results}

    def score_reveal(saved: dict, blind_key: dict) -> dict:
        card_ratings = saved.get("cards") or {}
        acct_ratings = saved.get("accounts") or {}
        comp_ratings = saved.get("comparator") or {}

        n_scene = 0
        rec_yes = 0
        n_acc_lbl = 0
        acc_yes = 0
        ab_non_tie = 0
        opus_pref = 0
        missed_sens = 0

        per_card_reveal = []
        for cid_s, meta in sorted(blind_key.items(), key=lambda kv: kv[1]["slot"]):
            cr = card_ratings.get(cid_s) or {}
            if cr.get("recognise") in ("yes", "no"):
                n_scene += 1
                if cr["recognise"] == "yes":
                    rec_yes += 1
            if cr.get("label_ok") in ("yes", "no"):
                n_acc_lbl += 1
                if cr["label_ok"] == "yes":
                    acc_yes += 1
            pref = cr.get("prefer")
            if pref in ("A", "B") and not (meta["opus_label"].strip().lower() == meta["curated_list_title"].strip().lower()):
                ab_non_tie += 1
                chosen_src = meta["option_a_source"] if pref == "A" else meta["option_b_source"]
                if chosen_src == "opus_llm":
                    opus_pref += 1
            if cr.get("sensitive_missed") == "yes":
                missed_sens += 1
            per_card_reveal.append({**meta, "rating": cr})

        n_accts = len(acct_ratings)
        on_scene = sum(1 for v in acct_ratings.values() if v == "on_scene_active")
        junk_dorm = sum(1 for v in acct_ratings.values() if v == "junk_or_dormant")

        comp_tot = sum(1 for v in comp_ratings.values() if v in ("ours", "bsky"))
        comp_ours = sum(1 for v in comp_ratings.values() if v == "ours")

        return {
            "bars": {
                "scene_recognised_pct": round(100.0 * rec_yes / max(n_scene, 1), 1) if n_scene else None,
                "scene_rated_n": n_scene,
                "scene_bar_ge_70": (rec_yes / n_scene >= 0.70) if n_scene else None,
                "label_accepted_pct": round(100.0 * acc_yes / max(n_acc_lbl, 1), 1) if n_acc_lbl else None,
                "label_rated_n": n_acc_lbl,
                "label_bar_ge_85": (acc_yes / n_acc_lbl >= 0.85) if n_acc_lbl else None,
                "opus_preferred_pct": round(100.0 * opus_pref / max(ab_non_tie, 1), 1) if ab_non_tie else None,
                "ab_decided_n": ab_non_tie,
                "opus_bar_ge_60": (opus_pref / ab_non_tie >= 0.60) if ab_non_tie else None,
                "sensitive_missed_count": missed_sens,
                "sensitive_bar_eq_0": (missed_sens == 0) if n_scene else None,
                "bundle_on_scene_pct": round(100.0 * on_scene / max(n_accts, 1), 1) if n_accts else None,
                "bundle_junk_pct": round(100.0 * junk_dorm / max(n_accts, 1), 1) if n_accts else None,
                "bundle_rated_n": n_accts,
                "comparator_ours_wins": comp_ours,
                "comparator_total": comp_tot,
            },
            "revealed_cards": per_card_reveal,
        }

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, obj: dict | bytes, status: int = 200) -> None:
            body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _check_csrf_and_read_json(self) -> dict | None:
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            origin = (self.headers.get("Origin") or "").strip()
            sfs = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
            xs17 = (self.headers.get("X-S17-Client") or "").strip()
            if (
                ctype != "application/json"
                or xs17 != "same-origin"
                or origin not in allowed_origins
                or (sfs and sfs != "same-origin")
            ):
                self._send_json({"error": "csrf_rejected"}, status=403)
                return None
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 2_000_000:
                self._send_json({"error": "invalid_payload_size"}, status=400)
                return None
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except ValueError:
                self._send_json({"error": "invalid_json"}, status=400)
                return None

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                body = page_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src https://cdn.bsky.app data:; base-uri 'none'; form-action 'none'",
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/cards_v1.json":
                self._send_json(state["cards_bytes"])
                return

            if path == "/cards_descriptions.json":
                self._send_json(state["descs_bytes"])
                return

            if path == "/inspector_index.json":
                self._send_json(state["inspector_bytes"])
                return

            if path == "/blind_review_key.json":
                self._send_json(state["blind_key"])
                return

            if path == "/api/overrides":
                with state_lock:
                    self._send_json(state["overrides"])
                return

            if path == "/api/author_feed":
                did = (qs.get("did") or [""])[0].strip()
                refresh = (qs.get("refresh") or ["0"])[0] == "1"
                if not did.startswith("did:"):
                    self._send_json({"error": "invalid_did"}, status=400)
                    return
                if did in hfr_dids:
                    self._send_json(
                        {
                            "did": did,
                            "is_hfr": True,
                            "live_verdict": "hfr_posts_withheld",
                            "posts": [],
                        }
                    )
                    return
                if not refresh:
                    cached = db_get_many([did])
                    if did in cached:
                        self._send_json(cached[did])
                        return
                fresh = hydrate_and_store_many([did])
                self._send_json(fresh[did])
                return

            if path == "/api/author_feeds_batch":
                raw_d = (qs.get("dids") or [""])[0]
                fetch_missing = (qs.get("fetch_missing") or ["1"])[0] != "0"
                req_dids: list[str] = []
                for tok in raw_d.split(","):
                    d = tok.strip()
                    if d.startswith("did:") and d not in req_dids:
                        req_dids.append(d)
                    if len(req_dids) >= 35:
                        break
                if not req_dids:
                    self._send_json({"feeds": {}, "remaining_uncached": []})
                    return
                found = db_get_many(req_dids)
                for d in req_dids:
                    if d in hfr_dids and d not in found:
                        found[d] = {"did": d, "is_hfr": True, "live_verdict": "hfr_posts_withheld", "posts": []}
                missing = [d for d in req_dids if d not in found]
                if fetch_missing and missing:
                    batch_now = missing[:8]
                    fresh = hydrate_and_store_many(batch_now)
                    found.update(fresh)
                    missing = [d for d in req_dids if d not in found]
                self._send_json({"feeds": found, "remaining_uncached": missing})
                return

            if path == "/api/pack":
                uri = (qs.get("uri") or [""])[0]
                pk = state["packs_map"].get(uri)
                if not pk:
                    self._send_json({"error": "pack_not_found"}, status=404)
                    return
                self._send_json(pk)
                return

            if path == "/api/medoids":
                raw_u = (qs.get("uris") or [""])[0]
                uris = [u.strip() for u in raw_u.split(",") if u.strip().startswith("at://")][:12]
                posts = hydrate_posts_live(uris, hfr_dids, state["hard_excl"], limiter)
                self._send_json({"posts": posts})
                return

            if path == "/api/preview":
                cards_obj = state["cards_obj"]
                hard_excl_now = state["hard_excl"]
                raw_c = (qs.get("cards") or [""])[0]
                picked = []
                for tok in raw_c.split(","):
                    tok = tok.strip()
                    if tok.isdigit() and tok in cards_obj["cards"]:
                        cid = int(tok)
                        if cid not in picked:
                            picked.append(cid)
                picked = picked[:5]
                if not picked:
                    self._send_json({"label": "Preview from Oct 22 data", "posts": [], "political_share": 0.0})
                    return

                ckey = ",".join(str(c) for c in picked)
                if ckey in preview_cache:
                    self._send_json(preview_cache[ckey])
                    return

                hh_targets: set[int] = set()
                for c in picked:
                    hh_targets |= high_h_by_actor.get(c, set())

                weight_by_c = {c: 1.0 for c in picked}
                for c in picked:
                    for nb in cards_obj["cards"][str(c)].get("nb", [])[:3]:
                        if int(nb) not in weight_by_c and int(nb) not in hh_targets:
                            weight_by_c[int(nb)] = 0.35

                cand_by_uri: dict[str, dict] = {}
                for c, cw in weight_by_c.items():
                    for r in preview_by_comm.get(c, []):
                        if r["did"] in hard_excl_now:
                            continue
                        u = r["uri"]
                        sc = cw * float(r["weighted_residual"])
                        prev = cand_by_uri.get(u)
                        if prev is None or sc > prev["score"]:
                            cand_by_uri[u] = {**r, "score": sc, "src_comm": c, "from_picked": c in picked}

                ordered = sorted(
                    cand_by_uri.values(),
                    key=lambda x: (0 if x["from_picked"] else 1, -x["score"], x["uri"]),
                )
                pre_uris: list[str] = []
                meta_by_uri: dict[str, dict] = {}
                acnt: dict[str, int] = {}
                hh_cnt = 0
                for item in ordered:
                    au = item["did"]
                    if acnt.get(au, 0) >= 2:
                        continue
                    if int(item["src_comm"]) in hh_targets:
                        if hh_cnt >= 2:
                            continue
                        hh_cnt += 1
                    acnt[au] = acnt.get(au, 0) + 1
                    pre_uris.append(item["uri"])
                    meta_by_uri[item["uri"]] = item
                    if len(pre_uris) >= 75:
                        break

                if len(pre_uris) < 20:
                    accts_tbl = cards_obj.get("accts") or []
                    for c in picked:
                        for mp in cards_obj["cards"][str(c)].get("md", []):
                            if isinstance(mp, list) and len(mp) == 2:
                                s_did = accts_tbl[int(mp[0])][0]
                                if not s_did.startswith("did:"):
                                    pad = "=" * ((4 - len(s_did) % 4) % 4)
                                    s_did = "did:plc:" + base64.b32encode(base64.urlsafe_b64decode(s_did + pad)).decode("ascii").lower()
                                mu = f"at://{s_did}/app.bsky.feed.post/{mp[1]}"
                            else:
                                mu = str(mp)
                            if mu not in meta_by_uri:
                                pre_uris.append(mu)
                                meta_by_uri[mu] = {
                                    "src_comm": c,
                                    "residual": 0.0,
                                    "n_comm": 0,
                                    "n_all": 0,
                                    "politicalness": 0.0,
                                    "pol_keyword": False,
                                }

                live_posts = hydrate_posts_live(pre_uris, hfr_dids, hard_excl_now, limiter)[:50]
                pol_hits = 0
                for lp in live_posts:
                    m = meta_by_uri.get(lp["uri"], {})
                    c_id = int(m.get("src_comm", picked[0]))
                    lp["community"] = c_id
                    lp["community_title"] = cards_obj["cards"].get(str(c_id), {}).get("t", f"c{c_id}")
                    lp["residual"] = round(float(m.get("residual", 0.0)), 2)
                    lp["n_comm"] = int(m.get("n_comm", 0))
                    lp["n_all"] = int(m.get("n_all", 0))
                    is_pol = (
                        bool(m.get("pol_keyword"))
                        or float(m.get("politicalness", 0.0)) >= 0.45
                        or bool(POL_RE.search(lp["text"]))
                    )
                    lp["is_political"] = is_pol
                    if is_pol:
                        pol_hits += 1

                resp_obj = {
                    "label": "Preview from Oct 22 data",
                    "picked_cards": picked,
                    "candidate_uris_checked": len(pre_uris),
                    "live_posts_returned": len(live_posts),
                    "political_posts_count": pol_hits,
                    "political_share": round(pol_hits / max(len(live_posts), 1), 4),
                    "posts": live_posts,
                }
                preview_cache[ckey] = resp_obj
                self._send_json(resp_obj)
                return

            if path == "/api/ratings":
                with state_lock:
                    saved = (
                        json.loads(ratings_path.read_text())
                        if ratings_path.exists()
                        else {"cards": {}, "accounts": {}, "comparator": {}}
                    )
                    bkey = state["blind_key"]
                if (qs.get("reveal") or ["0"])[0] == "1":
                    saved = {**saved, "reveal": score_reveal(saved, bkey)}
                self._send_json(saved)
                return

            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path == "/api/ratings":
                incoming = self._check_csrf_and_read_json()
                if incoming is None:
                    return
                with state_lock:
                    cur = (
                        json.loads(ratings_path.read_text())
                        if ratings_path.exists()
                        else {"cards": {}, "accounts": {}, "comparator": {}}
                    )
                    for section in ("cards", "accounts", "comparator"):
                        if isinstance(incoming.get(section), dict):
                            cur.setdefault(section, {}).update(incoming[section])
                    if "signoff_note" in incoming:
                        cur["signoff_note"] = str(incoming["signoff_note"])[:1000]
                    cur["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                    tmp = ratings_path.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(cur, indent=2, ensure_ascii=False))
                    tmp.replace(ratings_path)
                    bkey = state["blind_key"]
                self._send_json({"ok": True, "updated_utc": cur["updated_utc"], "reveal": score_reveal(cur, bkey)})
                return

            if path == "/api/overrides":
                incoming = self._check_csrf_and_read_json()
                if incoming is None:
                    return
                action = str(incoming.get("action") or "").strip()
                with state_lock:
                    ov = read_overrides_unlocked()
                    approved_set = set(ov["approved_dids"])
                    if action == "exclude_did":
                        did = str(incoming.get("did") or "").strip()
                        reason = str(incoming.get("reason") or "manual_inspector_exclude").strip()[:80]
                        if did.startswith("did:"):
                            ov["excluded_dids"][did] = reason
                            approved_set.discard(did)
                            state["hard_excl"].add(did)
                    elif action == "restore_did":
                        did = str(incoming.get("did") or "").strip()
                        ov["excluded_dids"].pop(did, None)
                    elif action == "approve_did":
                        did = str(incoming.get("did") or "").strip()
                        if did.startswith("did:"):
                            approved_set.add(did)
                            ov["excluded_dids"].pop(did, None)
                    elif action == "exclude_card":
                        cid = str(incoming.get("cid") or "").strip()
                        reason = str(incoming.get("reason") or "manual_inspector_exclude").strip()[:80]
                        if cid.isdigit():
                            ov["excluded_cards"][cid] = reason
                    elif action == "restore_card":
                        cid = str(incoming.get("cid") or "").strip()
                        ov["excluded_cards"].pop(cid, None)
                    else:
                        self._send_json({"error": "unknown_action"}, status=400)
                        return
                    ov["approved_dids"] = sorted(approved_set)
                    ov["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                    tmp = overrides_path.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(ov, indent=2, ensure_ascii=False))
                    tmp.replace(overrides_path)
                    state["overrides"] = ov
                    preview_cache.clear()
                self._send_json({"ok": True, "overrides": ov})
                return

            if path == "/api/rebuild":
                incoming = self._check_csrf_and_read_json()
                if incoming is None:
                    return
                with state_lock:
                    proc = subprocess.run(
                        [sys.executable, str(build_script), "--out", str(out_dir)],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    if proc.returncode != 0:
                        self._send_json(
                            {"ok": False, "error": "build_failed", "stderr": proc.stderr[-1200:], "stdout": proc.stdout[-1200:]},
                            status=500,
                        )
                        return
                    reload_catalog_unlocked()
                    ver = json.loads((out_dir / "verify.json").read_text())
                self._send_json({"ok": True, "verify": ver})
                return

            self.send_error(404)

        def log_message(self, fmt: str, *args) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[s17-serve] S17 Inspector v2 listening on http://127.0.0.1:{args.port}/ (127.0.0.1 only)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
