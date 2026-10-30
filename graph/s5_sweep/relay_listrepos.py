#!/usr/bin/env python3
"""S5 (vii): relay com.atproto.sync.listRepos sweep, sharded by the relay's numeric cursor.

Writes per-shard zstd-framed TSV (did, active, status) of EVERY listed repo, with a checkpoint
per shard (cursor + byte offset) so a crash resumes exactly. One global token bucket keeps the
whole sweep <= RATE req/s on the relay host; 429s pause every worker.

usage: listrepos.py OUTDIR [--shards 24] [--max-id 25000000] [--rate 9]
"""
import argparse, json, os, sys, threading, time, urllib.request, urllib.error, gzip
from compression import zstd

UA = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"
BASE = "https://bsky.network/xrpc/com.atproto.sync.listRepos"


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


stats_lock = threading.Lock()
STATS = {"calls": 0, "http": {}, "errors": 0, "retries": 0, "r429": 0}


def bump(k, sub=None):
    with stats_lock:
        if sub is None:
            STATS[k] += 1
        else:
            STATS[k][sub] = STATS[k].get(sub, 0) + 1


def fetch(bucket, cursor):
    url = f"{BASE}?limit=1000" + (f"&cursor={cursor}" if cursor is not None else "")
    for attempt in range(12):
        bucket.take()
        bump("calls")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json",
                                                       "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                bump("http", "200")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            bump("http", str(e.code))
            if e.code == 429:
                bump("r429")
                ra = e.headers.get("Retry-After") or e.headers.get("ratelimit-reset")
                try:
                    secs = float(ra)
                    if secs > 1e9:
                        secs = secs - time.time()
                except (TypeError, ValueError):
                    secs = 30
                bucket.pause(max(5, min(secs, 300)))
            else:
                time.sleep(min(60, 2 ** attempt))
        except Exception as e:  # network errors: back off and retry
            bump("http", "neterr:" + type(e).__name__)
            time.sleep(min(60, 2 ** attempt))
        bump("retries")
    raise RuntimeError(f"giving up on cursor {cursor}")


def shard_worker(out, idx, start, end, bucket, done):
    ck_path = os.path.join(out, f"shard{idx:02d}.ck.json")
    data_path = os.path.join(out, f"shard{idx:02d}.tsv.zst")
    ck = {"cursor": start, "offset": 0, "pages": 0, "rows": 0, "counts": {}, "done": False}
    if os.path.exists(ck_path):
        ck = json.load(open(ck_path))
    if ck["done"]:
        done[idx] = ck
        return
    with open(data_path, "ab") as fh:
        fh.truncate(ck["offset"])
        fh.seek(ck["offset"])
        cursor = ck["cursor"]
        while True:
            try:
                d = fetch(bucket, cursor)
            except Exception as e:
                bump("errors")
                print(f"shard {idx} FAILED at {cursor}: {e}", flush=True)
                done[idx] = dict(ck, failed=str(e))
                return
            repos = d.get("repos") or []
            lines = []
            for r in repos:
                a = "true" if r.get("active", True) else "false"
                s = r.get("status") or ""
                lines.append(f"{r['did']}\t{a}\t{s}\n")
                key = f"{a}/{s}"
                ck["counts"][key] = ck["counts"].get(key, 0) + 1
            if lines:
                fh.write(zstd.compress("".join(lines).encode()))
                fh.flush()
                os.fsync(fh.fileno())
            nxt = d.get("cursor")
            ck.update(offset=fh.tell(), pages=ck["pages"] + 1, rows=ck["rows"] + len(lines))
            fin = (not repos) or (nxt is None) or (end is not None and int(nxt) >= end)
            ck["cursor"] = nxt if nxt is not None else cursor
            ck["done"] = fin
            tmp = ck_path + ".tmp"
            with open(tmp, "w") as cf:
                json.dump(ck, cf)
            os.replace(tmp, ck_path)
            if fin:
                done[idx] = ck
                return
            cursor = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--shards", type=int, default=24)
    ap.add_argument("--max-id", type=int, default=25_000_000)
    ap.add_argument("--rate", type=float, default=9.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    step = a.max_id // a.shards
    bounds = [(i * step, (i + 1) * step if i < a.shards - 1 else None) for i in range(a.shards)]
    json.dump({"bounds": bounds, "rate": a.rate, "endpoint": BASE, "ua": UA}, open(os.path.join(a.out, "plan.json"), "w"))
    bucket = Bucket(a.rate)
    done = {}
    t0 = time.time()
    th = [threading.Thread(target=shard_worker, args=(a.out, i, s, e, bucket, done), daemon=True)
          for i, (s, e) in enumerate(bounds)]
    for t in th:
        t.start()
    while any(t.is_alive() for t in th):
        time.sleep(30)
        rows = 0
        for i in range(a.shards):
            p = os.path.join(a.out, f"shard{i:02d}.ck.json")
            if os.path.exists(p):
                try:
                    rows += json.load(open(p))["rows"]
                except Exception:
                    pass
        with stats_lock:
            print(f"[{time.strftime('%H:%M:%S')}] calls={STATS['calls']} rows={rows} done={len(done)}/{a.shards} "
                  f"429={STATS['r429']} http={STATS['http']}", flush=True)
    summary = {"wall_s": time.time() - t0, "stats": STATS, "shards": {str(k): v for k, v in sorted(done.items())}}
    with open(os.path.join(a.out, f"run-{int(t0)}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    failed = [k for k, v in done.items() if v.get("failed")]
    print("FINISHED", "failed shards:", failed, flush=True)
    sys.exit(1 if failed or len(done) != a.shards else 0)


if __name__ == "__main__":
    main()
