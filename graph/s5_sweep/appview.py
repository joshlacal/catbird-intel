#!/usr/bin/env python3
"""S5 (i)/(vi): public AppView sweeps with checkpoint/resume and one global rate limit.

  appview.py profiles DIDS_FILE OUT.jsonl   getProfiles, 25 per call; one line per requested DID
  appview.py authorfeed DIDS_FILE OUT.jsonl getAuthorFeed (posts_no_replies, limit 100); originals only

Resume: lines already in OUT.jsonl (keyed by batch id) are skipped. Each batch is written in one
append + flush, so a crash loses at most in-flight batches. Per-batch exceptions are captured
(never left inside a future), so one bad response cannot wedge the pool.
"""
import json, os, sys, threading, time, urllib.parse, urllib.request, urllib.error, gzip
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"
HOST = "https://public.api.bsky.app"
RATE = float(os.environ.get("RATE", "8"))


class Bucket:
    def __init__(self, rate):
        self.rate, self.lock, self.next, self.pause_until = rate, threading.Lock(), time.monotonic(), 0.0

    def take(self):
        while True:
            with self.lock:
                now = time.monotonic()
                wait = max(self.next, self.pause_until) - now
                if wait <= 0:
                    self.next = max(self.next, now) + 1.0 / self.rate
                    return
            time.sleep(min(wait, 1.0))

    def pause(self, secs):
        with self.lock:
            self.pause_until = max(self.pause_until, time.monotonic() + secs)


B = Bucket(RATE)
SL = threading.Lock()
STATS = {"calls": 0, "http": {}, "r429": 0}


def get(path, params):
    url = f"{HOST}/xrpc/{path}?" + urllib.parse.urlencode(params, doseq=True)
    last = None
    for attempt in range(8):
        B.take()
        with SL:
            STATS["calls"] += 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json", "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                with SL:
                    STATS["http"]["200"] = STATS["http"].get("200", 0) + 1
                return 200, json.loads(body)
        except urllib.error.HTTPError as e:
            code = e.code
            with SL:
                STATS["http"][str(code)] = STATS["http"].get(str(code), 0) + 1
            try:
                err = json.loads(e.read() or b"{}")
            except Exception:
                err = {}
            if code == 429:
                with SL:
                    STATS["r429"] += 1
                reset = e.headers.get("ratelimit-reset")
                secs = (float(reset) - time.time()) if reset else 30
                B.pause(max(5, min(secs, 300)))
                continue
            if code in (400, 404):
                return code, err   # definitive answer (e.g. actor not found / taken down)
            last = (code, err)
        except Exception as e:
            with SL:
                k = "neterr:" + type(e).__name__
                STATS["http"][k] = STATS["http"].get(k, 0) + 1
            last = (-1, {"error": type(e).__name__})
        time.sleep(min(30, 2 ** attempt))
    return last


def slim_labels(ls):
    return [{"src": l.get("src"), "val": l.get("val"), "neg": l.get("neg", False), "uri": l.get("uri")} for l in (ls or [])]


def do_profiles(bid, dids):
    code, d = get("app.bsky.actor.getProfiles", {"actors": dids})
    if code != 200:
        return {"batch": bid, "error": code, "detail": d, "dids": dids}
    got = {p["did"]: p for p in d.get("profiles", [])}
    rows = []
    for x in dids:
        p = got.get(x)
        if p is None:
            rows.append({"did": x, "found": False})
        else:
            rows.append({"did": x, "found": True, "handle": p.get("handle"), "postsCount": p.get("postsCount"),
                         "followersCount": p.get("followersCount"), "followsCount": p.get("followsCount"),
                         "createdAt": p.get("createdAt"), "indexedAt": p.get("indexedAt"),
                         "labels": slim_labels(p.get("labels")), "verification": (p.get("verification") or {}).get("verifiedStatus")})
    return {"batch": bid, "rows": rows}


def do_authorfeed(bid, dids):
    (did,) = dids
    code, d = get("app.bsky.feed.getAuthorFeed", {"actor": did, "filter": "posts_no_replies", "limit": 100})
    if code != 200:
        return {"batch": bid, "did": did, "error": code, "detail": d}
    times = []
    for it in d.get("feed", []):
        post = it.get("post", {})
        rec = post.get("record", {})
        if it.get("reason") or rec.get("reply") or (post.get("author") or {}).get("did") != did:
            continue
        times.append(rec.get("createdAt"))
    return {"batch": bid, "did": did, "items": len(d.get("feed", [])), "originals": len(times),
            "original_created": times, "cursor": d.get("cursor")}


def main():
    mode, src, out = sys.argv[1:4]
    dids = [l.split("\t")[0].strip() for l in open(src) if l.startswith("did:")]
    size = 25 if mode == "profiles" else 1
    batches = [(i, dids[i * size:(i + 1) * size]) for i in range((len(dids) + size - 1) // size)]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add(r["batch"])
            except Exception:
                pass   # a torn last line from a crash is ignored and redone
    todo = [b for b in batches if b[0] not in done]
    print(f"{mode}: {len(dids)} DIDs, {len(batches)} batches, {len(todo)} to do", flush=True)
    fn = do_profiles if mode == "profiles" else do_authorfeed
    t0, n = time.time(), 0
    wl = threading.Lock()

    def safe(b):
        try:
            return fn(*b)
        except Exception as e:  # per-batch capture: never leave an exception inside a future
            return {"batch": b[0], "error": "exception", "detail": repr(e)[:300], "dids": b[1]}

    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(safe, b) for b in todo]
        for f in as_completed(futs):
            r = f.result()
            with wl:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
            n += 1
            if n % 1000 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(todo)} {STATS}", flush=True)
    print(json.dumps({"mode": mode, "wall_s": time.time() - t0, "stats": STATS}), flush=True)


if __name__ == "__main__":
    main()
