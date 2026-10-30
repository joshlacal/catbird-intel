#!/usr/bin/env python3
"""Hydrate & evaluate S13(b) (40+40 removed/kept) and blind S13(c) (100+100 combative/control).

- Uses public AppView (`getProfiles` + `getAuthorFeed`, `RATE=8` req/s, zero image bytes).
- Hashes every raw AppView response line (SHA-256) into `s13bc_raw_hashes.txt`.
- Evaluates S13(b) (`removed_junk_rate >= 2x kept_junk_rate` and `<= 4/40` good follows among removed),
  writing full per-account table `s13b_audit.json`.
- Evaluates S13(c) strictly BLIND: shuffles the 200 accounts into `blind_000..blind_199` before judging
  argumentativeness from bio + recent posts/replies/quotes alone, then unblinds against `s13c_key.json`
  to test `top_combative >= 70%` AND `controls <= 20%`.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import pathlib
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

UA = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"
HOST = "https://public.api.bsky.app"
RATE = float(os.environ.get("RATE", "8"))


class Bucket:
    def __init__(self, rate: float):
        self.rate = rate
        self.lock = threading.Lock()
        self.next = time.monotonic()
        self.pause_until = 0.0

    def take(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                wait = max(self.next, self.pause_until) - now
                if wait <= 0:
                    self.next = max(self.next, now) + 1.0 / self.rate
                    return
            time.sleep(min(wait, 1.0))

    def pause(self, secs: float) -> None:
        with self.lock:
            self.pause_until = max(self.pause_until, time.monotonic() + secs)


BUCKET = Bucket(RATE)

ARG_RE = re.compile(
    r"\b("
    r"idiot|idiots|moron|morons|stupid|dumbass|clown|clowns|shut up|stfu|fuck off|fuck you|"
    r"pathetic|braindead|delusional|grifter|grifters|scumbag|trash|garbage|liar|liars|lying|"
    r"coward|cowards|fascist|fascists|nazi|nazis|bigot|bigots|racist|racists|terf|terfs|"
    r"bootlicker|shill|shills|troll|trolls|blocked|blocking|instant block|ratio|cope|seethe|"
    r"hypocrite|hypocrites|disgusting|vile|evil|propaganda|cultist|cult|genocide|zionist|"
    r"maga|libtard|rightoid|leftoid|tankie|tankies|shitlib|wtf is wrong with you|you people"
    r")\b",
    re.IGNORECASE,
)


def xrpc_get(path: str, params: dict) -> tuple[int, dict, str]:
    url = f"{HOST}/xrpc/{path}?" + urllib.parse.urlencode(params, doseq=True)
    last = (-1, {"error": "unknown"}, b"{}")
    for attempt in range(6):
        BUCKET.take()
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": UA, "Accept": "application/json", "Accept-Encoding": "gzip"},
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return 200, json.loads(raw), hashlib.sha256(raw).hexdigest()
        except urllib.error.HTTPError as e:
            raw = e.read() or b"{}"
            if e.headers.get("Content-Encoding") == "gzip":
                try:
                    raw = gzip.decompress(raw)
                except Exception:
                    pass
            try:
                err = json.loads(raw)
            except Exception:
                err = {"http": e.code}
            if e.code == 429:
                BUCKET.pause(15.0)
                continue
            if e.code in (400, 404):
                return e.code, err, hashlib.sha256(raw).hexdigest()
            last = (e.code, err, raw)
        except Exception as e:
            last = (-1, {"error": type(e).__name__}, repr(e).encode())
        time.sleep(min(15, 2**attempt))
    return last[0], last[1], hashlib.sha256(last[2] if isinstance(last[2], bytes) else b"").hexdigest()


def parse_dt(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def slim_feed_item(it: dict, actor_did: str) -> dict | None:
    post = it.get("post") or {}
    author = (post.get("author") or {}).get("did")
    if author != actor_did:
        return {"kind": "repost", "created": ((it.get("reason") or {}).get("indexedAt") or post.get("indexedAt") or "")}
    rec = post.get("record") or {}
    is_reply = bool(rec.get("reply"))
    embed = rec.get("embed") or {}
    etype = embed.get("$type", "")
    is_quote = "record" in etype
    text = " ".join((rec.get("text") or "").split())[:280]
    return {
        "kind": "reply" if is_reply else ("quote" if is_quote else "original"),
        "created": rec.get("createdAt") or post.get("indexedAt") or "",
        "text": text,
    }


def judge_s13b_account(prof: dict | None, feed_code: int, items: list[dict], now_utc: dt.datetime) -> dict:
    """Classify an S13(b) profile + recent feed into junk/dormant/follow-ring vs active good follow."""
    if prof is None or not prof.get("found", False) or feed_code != 200:
        return {"junk_or_dormant": True, "good_follow": False, "reason": "missing_or_takendown"}
    labs = [l.get("val", "") for l in (prof.get("labels") or []) if not l.get("neg")]
    if labs or prof.get("handle") == "handle.invalid":
        return {"junk_or_dormant": True, "good_follow": False, "reason": f"labelled_or_invalid:{','.join(labs)}"}
    posts_cnt = int(prof.get("postsCount") or 0)
    fol_cnt = int(prof.get("followersCount") or 0)
    follows_cnt = int(prof.get("followsCount") or 0)
    if posts_cnt < 3:
        return {"junk_or_dormant": True, "good_follow": False, "reason": "posts_lt_3"}
    if posts_cnt <= 10 and follows_cnt >= 0.8 * max(fol_cnt, 1):
        return {"junk_or_dormant": True, "good_follow": False, "reason": "low_post_follow_heavy"}
    if follows_cnt >= 2500 and follows_cnt >= 1.5 * max(fol_cnt, 1):
        return {"junk_or_dormant": True, "good_follow": False, "reason": "mass_follow_ring"}

    d90 = now_utc - dt.timedelta(days=90)
    d30 = now_utc - dt.timedelta(days=30)
    d365 = now_utc - dt.timedelta(days=365)
    orig_dates = []
    any_dates = []
    orig_texts = []
    for it in items:
        ts = parse_dt(it.get("created"))
        if ts:
            any_dates.append(ts)
        if it.get("kind") in ("original", "quote"):
            if ts:
                orig_dates.append(ts)
            if it.get("text"):
                orig_texts.append(it["text"])

    orig_90d = sum(t >= d90 for t in orig_dates)
    orig_30d = sum(t >= d30 for t in orig_dates)
    orig_12m = sum(t >= d365 for t in orig_dates)
    any_90d = sum(t >= d90 for t in any_dates)

    if any_90d == 0:
        return {"junk_or_dormant": True, "good_follow": False, "reason": "dormant_90d"}
    if orig_90d == 0:
        return {"junk_or_dormant": True, "good_follow": False, "reason": "no_originals_90d_repost_or_reply_only"}

    has_substance = sum(len(x) >= 25 for x in orig_texts) >= 2
    good = bool(orig_12m >= 3 and orig_30d >= 1 and has_substance and follows_cnt < 2.0 * max(fol_cnt, 20))
    return {
        "junk_or_dormant": False,
        "good_follow": good,
        "reason": "active_good_follow" if good else "active_low_cadence_or_sparse",
        "orig_30d": orig_30d,
        "orig_90d": orig_90d,
        "orig_12m_in_sample": orig_12m,
    }


def judge_blind_s13c(blind_records: list[dict]) -> dict[str, dict]:
    """Strictly blind evaluation of argumentativeness from (blind_id, bio, recent feed items).
    Sees NO group label, NO combativeness score, NO DID, NO follower count."""
    verdicts: dict[str, dict] = {}
    for rec in blind_records:
        bid = rec["blind_id"]
        bio = rec.get("bio") or ""
        items = [it for it in rec.get("items", []) if it.get("kind") != "repost"]
        n_items = len(items)
        if n_items == 0:
            verdicts[bid] = {"argumentative": False, "arg_hits": 0, "reply_quote_share": 0.0, "n_authored": 0}
            continue
        n_rq = sum(it.get("kind") in ("reply", "quote") for it in items)
        rq_share = n_rq / n_items
        arg_posts = 0
        for it in items:
            txt = it.get("text") or ""
            # Direct insult/combativeness lexicon or aggressive 2nd-person confrontational reply/quote
            has_lex = bool(ARG_RE.search(txt))
            confrontational_rq = (
                it.get("kind") in ("reply", "quote")
                and bool(re.search(r"\b(you're|you are|your take|wrong|bullshit|lie|lying|wtf|lmao no|stfu)\b", txt, re.I))
                and ("?" in txt or "!" in txt or len(txt) > 60)
            )
            if has_lex or confrontational_rq:
                arg_posts += 1
        arg_share = arg_posts / n_items
        bio_combat = bool(re.search(r"\b(block freely|instant block|nodms|argue|anti-|resist)\b", bio, re.I))
        # Predominantly argumentative account: >=20% of authored recent items carry hostile/confrontational markers
        # (or >=3 hostile items AND >=65% reply/quote ratio, or bio_combat + >=15% hostile items)
        is_arg = bool(
            (n_items >= 5 and arg_share >= 0.20)
            or (arg_posts >= 3 and rq_share >= 0.65)
            or (bio_combat and arg_posts >= 2 and arg_share >= 0.12)
        )
        verdicts[bid] = {
            "argumentative": is_arg,
            "arg_posts": arg_posts,
            "arg_share": round(arg_share, 4),
            "reply_quote_share": round(rq_share, 4),
            "n_authored": n_items,
        }
    return verdicts


def main() -> None:
    lake = pathlib.Path.home() / "catbird-lake-local"
    out_dir = lake / "s13-elig-20261030"
    raw_dir = out_dir / "raw_live"
    raw_dir.mkdir(parents=True, exist_ok=True)

    s13b_rows = []
    for ln in open(out_dir / "sample_s13b.tsv"):
        grp, pid_s, did, le_s, mut_s = ln.rstrip("\n").split("\t")
        s13b_rows.append({"group": grp, "pid": int(pid_s), "did": did, "le_disc": float(le_s), "mutual": float(mut_s)})

    s13c_rows = []
    for ln in open(out_dir / "sample_s13c.tsv"):
        grp, pid_s, did, comb_s, fol_s = ln.rstrip("\n").split("\t")
        s13c_rows.append({
            "group": grp,
            "pid": int(pid_s),
            "did": did,
            "combativeness": None if comb_s == "null" else float(comb_s),
            "followers": int(fol_s),
        })

    all_dids = sorted({r["did"] for r in s13b_rows} | {r["did"] for r in s13c_rows})
    print(f"[s13bc] hydrating {len(all_dids)} unique DIDs (getProfiles + getAuthorFeed)...", flush=True)

    raw_hashes: list[str] = []
    profiles_by_did: dict[str, dict] = {}
    prof_jsonl_path = raw_dir / "s13bc_profiles.jsonl"
    with open(prof_jsonl_path, "w") as pfh:
        for i in range(0, len(all_dids), 25):
            batch = all_dids[i : i + 25]
            code, body, rhash = xrpc_get("app.bsky.actor.getProfiles", {"actors": batch})
            raw_hashes.append(rhash)
            pfh.write(json.dumps({"batch": i // 25, "code": code, "sha256": rhash, "dids": batch, "body": body}) + "\n")
            got = {p["did"]: p for p in (body.get("profiles") or [])} if code == 200 else {}
            for d in batch:
                p = got.get(d)
                if p is None:
                    profiles_by_did[d] = {"did": d, "found": False}
                else:
                    profiles_by_did[d] = {
                        "did": d,
                        "found": True,
                        "handle": p.get("handle"),
                        "displayName": (p.get("displayName") or "")[:80],
                        "description": " ".join((p.get("description") or "").split())[:240],
                        "postsCount": p.get("postsCount"),
                        "followersCount": p.get("followersCount"),
                        "followsCount": p.get("followsCount"),
                        "labels": p.get("labels") or [],
                    }

    feeds_by_did: dict[str, tuple[int, list[dict], str]] = {}

    def fetch_one_feed(d: str) -> tuple[str, int, list[dict], str]:
        code, body, rhash = xrpc_get("app.bsky.feed.getAuthorFeed", {"actor": d, "filter": "posts_with_replies", "limit": 40})
        items = []
        if code == 200:
            for it in body.get("feed") or []:
                slim = slim_feed_item(it, d)
                if slim is not None:
                    items.append(slim)
        return d, code, items, rhash

    feed_jsonl_path = raw_dir / "s13bc_feeds.jsonl"
    with ThreadPoolExecutor(max_workers=10) as ex, open(feed_jsonl_path, "w") as ffh:
        futs = [ex.submit(fetch_one_feed, d) for d in all_dids]
        for f in as_completed(futs):
            d, code, items, rhash = f.result()
            feeds_by_did[d] = (code, items, rhash)
            raw_hashes.append(rhash)
            ffh.write(json.dumps({"did": d, "code": code, "sha256": rhash, "items": items}) + "\n")

    raw_hashes_sorted = sorted(raw_hashes)
    (out_dir / "s13bc_raw_hashes.txt").write_text("".join(h + "\n" for h in raw_hashes_sorted))
    raw_hashes_digest = hashlib.sha256("\n".join(raw_hashes_sorted).encode()).hexdigest()

    # Evaluate S13(b)
    now_utc = dt.datetime.now(dt.timezone.utc)
    s13b_evaluated = []
    for r in s13b_rows:
        d = r["did"]
        prof = profiles_by_did.get(d)
        fcode, fitems, _ = feeds_by_did.get(d, (-1, [], ""))
        j = judge_s13b_account(prof, fcode, fitems, now_utc)
        s13b_evaluated.append({
            **r,
            "handle": (prof or {}).get("handle"),
            "postsCount": (prof or {}).get("postsCount"),
            "followersCount": (prof or {}).get("followersCount"),
            "followsCount": (prof or {}).get("followsCount"),
            **j,
        })

    rm_rows = [x for x in s13b_evaluated if x["group"] == "removed"]
    kp_rows = [x for x in s13b_evaluated if x["group"] == "kept"]
    rm_junk = sum(x["junk_or_dormant"] for x in rm_rows) / len(rm_rows)
    kp_junk = sum(x["junk_or_dormant"] for x in kp_rows) / len(kp_rows)
    rm_good = sum(x["good_follow"] for x in rm_rows)
    kp_good = sum(x["good_follow"] for x in kp_rows)
    s13b_pass = bool(rm_junk >= 2.0 * max(kp_junk, 1e-9) and rm_good <= 4)

    s13b_summary = {
        "n_removed": len(rm_rows),
        "n_kept": len(kp_rows),
        "removed_junk_or_dormant_count": int(sum(x["junk_or_dormant"] for x in rm_rows)),
        "removed_junk_or_dormant_rate": float(rm_junk),
        "kept_junk_or_dormant_count": int(sum(x["junk_or_dormant"] for x in kp_rows)),
        "kept_junk_or_dormant_rate": float(kp_junk),
        "junk_rate_ratio": float(rm_junk / kp_junk) if kp_junk > 0 else float("inf"),
        "removed_good_follows_count": int(rm_good),
        "kept_good_follows_count": int(kp_good),
        "bar": "removed junk rate >= 2x kept AND <= 4 of 40 removed are good follows (S1 §9.8 S13b)",
        "pass": s13b_pass,
        "accounts": s13b_evaluated,
    }
    (out_dir / "s13b_audit.json").write_text(json.dumps(s13b_summary, indent=2))

    # Evaluate S13(c) strictly BLIND
    rng = np.random.default_rng(20261030)
    perm = rng.permutation(len(s13c_rows))
    blind_inputs = []
    key_map = {}
    for idx_b, orig_i in enumerate(perm):
        row = s13c_rows[int(orig_i)]
        bid = f"blind_{idx_b:03d}"
        d = row["did"]
        prof = profiles_by_did.get(d) or {}
        _, fitems, _ = feeds_by_did.get(d, (-1, [], ""))
        blind_inputs.append({
            "blind_id": bid,
            "bio": prof.get("description") or "",
            "items": fitems,
        })
        key_map[bid] = row
    (out_dir / "s13c_key.json").write_text(json.dumps(key_map, indent=2))

    blind_verdicts = judge_blind_s13c(blind_inputs)

    # Unblind and score against S1 §9.8 S13(c) bar
    comb_arg = []
    ctrl_arg = []
    unblinded_rows = []
    for bid, meta in sorted(key_map.items()):
        v = blind_verdicts[bid]
        entry = {**meta, "blind_id": bid, **v}
        unblinded_rows.append(entry)
        if meta["group"] == "top_combative":
            comb_arg.append(v["argumentative"])
        else:
            ctrl_arg.append(v["argumentative"])

    comb_rate = float(np.mean(comb_arg))
    ctrl_rate = float(np.mean(ctrl_arg))
    s13c_pass = bool(comb_rate >= 0.70 and ctrl_rate <= 0.20)
    s13c_summary = {
        "n_top_combative": len(comb_arg),
        "n_controls": len(ctrl_arg),
        "top_combative_argumentative_count": int(sum(comb_arg)),
        "top_combative_argumentative_rate": comb_rate,
        "controls_argumentative_count": int(sum(ctrl_arg)),
        "controls_argumentative_rate": ctrl_rate,
        "bar": "Blind: >=70% of top-100 combative vs <=20% of 100 controls judged argumentative, else combativeness demotion is dropped (S1 §9.8 S13c)",
        "pass": s13c_pass,
        "combativeness_demotion_enabled": s13c_pass,
        "raw_response_hashes_sha256": raw_hashes_digest,
        "raw_response_count": len(raw_hashes_sorted),
        "accounts": unblinded_rows,
    }
    (out_dir / "s13c_audit.json").write_text(json.dumps(s13c_summary, indent=2))
    print(json.dumps({
        "s13b": {k: v for k, v in s13b_summary.items() if k != "accounts"},
        "s13c": {k: v for k, v in s13c_summary.items() if k != "accounts"},
    }, indent=2))


if __name__ == "__main__":
    main()
