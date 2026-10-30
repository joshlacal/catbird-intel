#!/usr/bin/env python3
"""Localhost-only server for S17 Harness v1 (`127.0.0.1:8765`, never 0.0.0.0, never published).

Endpoints:
- GET  /                        -> page_s17.html
- GET  /cards_v1.json           -> verified <=2 MB v0 catalog
- GET  /cards_descriptions.json -> full uncut scene descriptions & organic quality metadata
- GET  /api/pack?uri=...        -> inspect-only starter pack members from packs_members.json
- GET  /api/preview?cards=      -> live-hydrated "Preview from Oct 22 data" (<=2/author, <=5% high-H, 0 deleted/labelled/hfr/bot)
- GET  /api/medoids?uris=       -> live-hydrated medoid sample posts (text only, 0 image bytes)
- GET  /api/ratings             -> saved blind S17 review ratings (+ ?reveal=1 scored against blind_review_key.json)
- POST /api/ratings             -> atomically save blind S17 review ratings to ratings.json (CSRF-protected same-origin JSON only)
"""
from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import http.server
import json
import pathlib
import re
import threading
import urllib.error
import urllib.parse
import urllib.request

import pyarrow.parquet as pq

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


def hydrate_posts_live(uris: list[str], hfr_dids: set[str], hard_excl: set[str]) -> list[dict]:
    """Hydrate up to 100 AT-URIs in batches of 25 via public.api.bsky.app getPosts; drop deleted/labelled/hfr/bots."""
    clean_out: list[dict] = []
    for s in range(0, min(len(uris), 100), 25):
        batch = [u for u in uris[s : s + 25] if u.startswith("at://")]
        if not batch:
            continue
        qs = "&".join("uris=" + urllib.parse.quote(u, safe="") for u in batch)
        url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "catbird-s17-harness/1.0", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception:
            continue
        by_uri = {p.get("uri"): p for p in (data.get("posts") or []) if isinstance(p, dict)}
        for u in batch:
            pv = by_uri.get(u)
            if not pv:
                continue  # deleted or taken down -> vanishes live
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
            etype = embed.get("$type", "")
            has_img = "images" in etype
            has_vid = "video" in etype
            clean_out.append(
                {
                    "uri": u,
                    "did": adid,
                    "handle": ahandle,
                    "name": (au.get("displayName") or "")[:50],
                    "text": txt[:500],
                    "created_at": rec.get("createdAt") or pv.get("indexedAt") or "",
                    "likes": int(pv.get("likeCount") or 0),
                    "reposts": int(pv.get("repostCount") or 0),
                    "replies": int(pv.get("replyCount") or 0),
                    "media_badge": "image" if has_img else ("video" if has_vid else ""),
                }
            )
    return clean_out


def main() -> None:
    ap = argparse.ArgumentParser()
    lake = pathlib.Path.home() / "catbird-lake-local"
    ap.add_argument("--out", default=str(lake / "s17-picker-v1-20261030"))
    ap.add_argument("--s16-dir", default=str(lake / "s16-s17-20261030"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out).expanduser()
    s16_dir = pathlib.Path(args.s16_dir).expanduser()
    page_path = pathlib.Path(__file__).resolve().parent / "page_s17.html"
    cards_path = out_dir / "cards_v1.json"
    descs_path = out_dir / "cards_descriptions.json"
    packs_path = out_dir / "packs_members.json"
    blind_key_path = out_dir / "blind_review_key.json"
    ratings_path = out_dir / "ratings.json"
    non_org_path = out_dir / "non_organic_exclude_dids.txt"

    cards_bytes = cards_path.read_bytes()
    descs_bytes = descs_path.read_bytes() if descs_path.exists() else b"{}"
    cards_obj = json.loads(cards_bytes)
    packs_map = json.loads(packs_path.read_text()) if packs_path.exists() else {}
    blind_key = json.loads(blind_key_path.read_text()) if blind_key_path.exists() else {}
    allowed_origins = {f"http://127.0.0.1:{args.port}", f"http://localhost:{args.port}"}

    # Load safety + organic exclusion sets for live hydration defense-in-depth
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
    if non_org_path.exists():
        hard_excl |= {ln.split("\t")[0].strip() for ln in open(non_org_path) if ln.startswith("did:")}

    # Load high-H pairs
    high_h_by_actor: dict[int, set[int]] = collections.defaultdict(set)
    t_h = pq.read_table(lake / "s14-hostility-20261030/H_prod_cand5_full.parquet", columns=["c_actor", "c_target", "is_high_top1pct"])
    for ca, ct, hi in zip(t_h["c_actor"].to_pylist(), t_h["c_target"].to_pylist(), t_h["is_high_top1pct"].to_pylist()):
        if hi:
            high_h_by_actor[int(ca)].add(int(ct))

    # Load Oct 22 preview post bank
    preview_by_comm: dict[int, list[dict]] = collections.defaultdict(list)
    oct22_pq = s16_dir / "oct22_preview_posts.parquet"
    if oct22_pq.exists():
        tp = pq.read_table(oct22_pq)
        for r in tp.to_pylist():
            if r["did"] not in hfr_dids and r["did"] not in hard_excl:
                preview_by_comm[int(r["community"])].append(r)

    lock = threading.Lock()
    preview_cache: dict[str, dict] = {}

    def score_reveal(saved: dict) -> dict:
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
                    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'none'; base-uri 'none'; form-action 'none'",
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/cards_v1.json":
                self._send_json(cards_bytes)
                return

            if path == "/cards_descriptions.json":
                self._send_json(descs_bytes)
                return

            if path == "/api/pack":
                uri = (qs.get("uri") or [""])[0]
                pk = packs_map.get(uri)
                if not pk:
                    self._send_json({"error": "pack_not_found"}, status=404)
                    return
                self._send_json(pk)
                return

            if path == "/api/medoids":
                raw_u = (qs.get("uris") or [""])[0]
                uris = [u.strip() for u in raw_u.split(",") if u.strip().startswith("at://")][:12]
                posts = hydrate_posts_live(uris, hfr_dids, hard_excl)
                self._send_json({"posts": posts})
                return

            if path == "/api/preview":
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

                # Gather Oct 22 candidate posts from picked cards (plus H-compatible neighbors if needed)
                weight_by_c = {c: 1.0 for c in picked}
                for c in picked:
                    for nb in cards_obj["cards"][str(c)].get("nb", [])[:3]:
                        if int(nb) not in weight_by_c and int(nb) not in hh_targets:
                            weight_by_c[int(nb)] = 0.35

                cand_by_uri: dict[str, dict] = {}
                for c, cw in weight_by_c.items():
                    for r in preview_by_comm.get(c, []):
                        u = r["uri"]
                        sc = cw * float(r["weighted_residual"])
                        prev = cand_by_uri.get(u)
                        if prev is None or sc > prev["score"]:
                            cand_by_uri[u] = {**r, "score": sc, "src_comm": c, "from_picked": c in picked}

                # Sort by (from_picked desc, score desc, uri asc) and enforce <=2/author, <=5% high-H before live hydration
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

                # Fallback to card medoids if Oct 22 export had <20 posts for tiny niche scene
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
                                meta_by_uri[mu] = {"src_comm": c, "residual": 0.0, "n_comm": 0, "n_all": 0, "politicalness": 0.0, "pol_keyword": False}

                live_posts = hydrate_posts_live(pre_uris, hfr_dids, hard_excl)[:50]
                pol_hits = 0
                for lp in live_posts:
                    m = meta_by_uri.get(lp["uri"], {})
                    c_id = int(m.get("src_comm", picked[0]))
                    lp["community"] = c_id
                    lp["community_title"] = cards_obj["cards"].get(str(c_id), {}).get("t", f"c{c_id}")
                    lp["residual"] = round(float(m.get("residual", 0.0)), 2)
                    lp["n_comm"] = int(m.get("n_comm", 0))
                    lp["n_all"] = int(m.get("n_all", 0))
                    is_pol = bool(m.get("pol_keyword")) or float(m.get("politicalness", 0.0)) >= 0.45 or bool(POL_RE.search(lp["text"]))
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
                with lock:
                    saved = json.loads(ratings_path.read_text()) if ratings_path.exists() else {"cards": {}, "accounts": {}, "comparator": {}}
                if (qs.get("reveal") or ["0"])[0] == "1":
                    saved = {**saved, "reveal": score_reveal(saved)}
                self._send_json(saved)
                return

            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/api/ratings":
                self.send_error(404)
                return
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
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 2_000_000:
                self._send_json({"error": "invalid_payload_size"}, status=400)
                return
            raw = self.rfile.read(length)
            try:
                incoming = json.loads(raw.decode("utf-8"))
            except ValueError:
                self._send_json({"error": "invalid_json"}, status=400)
                return

            with lock:
                cur = json.loads(ratings_path.read_text()) if ratings_path.exists() else {"cards": {}, "accounts": {}, "comparator": {}}
                for section in ("cards", "accounts", "comparator"):
                    if isinstance(incoming.get(section), dict):
                        cur.setdefault(section, {}).update(incoming[section])
                cur["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
                tmp = ratings_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(cur, indent=2, ensure_ascii=False))
                tmp.replace(ratings_path)

            self._send_json({"ok": True, "updated_utc": cur["updated_utc"]})

        def log_message(self, fmt: str, *args) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[s17-serve] S17 Harness v1 listening on http://127.0.0.1:{args.port}/ (127.0.0.1 only)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
