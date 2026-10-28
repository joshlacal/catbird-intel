#!/usr/bin/env python3
"""S5 (viii): PLC directory /export for operations created in [START, END), sharded by UTC day.

Pages are streamed and filtered in memory; only genesis operations (prev == null) are kept, as
`did <TAB> createdAt <TAB> nullified <TAB> type` lines in a per-day file. Page cursor = the last
op's createdAt minus 1 ms (export `after` is exclusive, so ops sharing the boundary millisecond
are re-read, never skipped); duplicates are dropped by CID. Per-day checkpoint: cursor, seen CIDs
of the boundary millisecond, byte offset. Global rate limit RATE req/s on plc.directory.

usage: plcexport.py OUTDIR [START=2026-08-14] [END=2026-09-10]
"""
import datetime as dt, json, os, sys, threading, time, urllib.request, urllib.error, gzip, collections

UA = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"
URL = "https://plc.directory/export"
RATE = float(os.environ.get("RATE", "8"))
lock = threading.Lock()
STATS = collections.Counter()


class Bucket:
    def __init__(self, rate):
        self.rate, self.l, self.next, self.pu = rate, threading.Lock(), time.monotonic(), 0.0

    def take(self):
        while True:
            with self.l:
                now = time.monotonic()
                w = max(self.next, self.pu) - now
                if w <= 0:
                    self.next = max(self.next, now) + 1.0 / self.rate
                    return
            time.sleep(min(w, 1.0))

    def pause(self, s):
        with self.l:
            self.pu = max(self.pu, time.monotonic() + s)


B = Bucket(RATE)


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def parse(s):
    return dt.datetime.strptime(s.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z")


def fetch(after):
    url = f"{URL}?count=1000&after={after}"
    for attempt in range(12):
        B.take()
        with lock:
            STATS["calls"] += 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=90) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                with lock:
                    STATS["http_200"] += 1
                return [json.loads(x) for x in body.splitlines() if x.strip()]
        except urllib.error.HTTPError as e:
            with lock:
                STATS[f"http_{e.code}"] += 1
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                B.pause(max(10, min(float(ra) if ra else 60, 300)))
                continue
        except Exception as e:
            with lock:
                STATS["neterr_" + type(e).__name__] += 1
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"giving up at {after}")


def day_worker(out, day, results):
    start, end = day, day + dt.timedelta(days=1)
    ckp, dp = os.path.join(out, f"{day:%Y%m%d}.ck.json"), os.path.join(out, f"{day:%Y%m%d}.tsv")
    ck = {"after": iso(start - dt.timedelta(milliseconds=1)), "boundary": [], "offset": 0, "pages": 0,
          "ops": 0, "genesis": 0, "done": False, "types": {}}
    if os.path.exists(ckp):
        ck = json.load(open(ckp))
    if ck["done"]:
        results[day] = ck
        return
    try:
        with open(dp, "ab") as fh:
            fh.truncate(ck["offset"])
            fh.seek(ck["offset"])
            while True:
                ops = fetch(ck["after"])
                seen = set(ck["boundary"])
                lines, fin = [], not ops
                for o in ops:
                    t = parse(o["createdAt"])
                    if t >= end:
                        fin = True
                        break
                    if o["cid"] in seen or t < start:
                        continue
                    ck["ops"] += 1
                    op = o.get("operation") or {}
                    typ = op.get("type", "?")
                    ck["types"][typ] = ck["types"].get(typ, 0) + 1
                    if op.get("prev") is None and typ in ("plc_operation", "create"):
                        ck["genesis"] += 1
                        lines.append(f"{o['did']}\t{o['createdAt']}\t{int(bool(o.get('nullified')))}\t{typ}\n")
                if lines:
                    fh.write("".join(lines).encode())
                    fh.flush()
                    os.fsync(fh.fileno())
                if ops and not fin:
                    last = parse(ops[-1]["createdAt"])
                    lastms = ops[-1]["createdAt"]
                    ck["boundary"] = [o["cid"] for o in ops if o["createdAt"] == lastms]
                    ck["after"] = iso(last - dt.timedelta(milliseconds=1))
                    if len(ck["boundary"]) >= 1000:
                        raise RuntimeError(f"1000 ops in one millisecond at {lastms}")
                    if len(ops) < 1000:
                        fin = True
                ck.update(offset=fh.tell(), pages=ck["pages"] + 1, done=fin)
                tmp = ckp + ".tmp"
                json.dump(ck, open(tmp, "w"))
                os.replace(tmp, ckp)
                if fin:
                    results[day] = ck
                    return
    except Exception as e:
        results[day] = dict(ck, failed=repr(e)[:300])


def main():
    out = sys.argv[1]
    s = dt.datetime.fromisoformat((sys.argv[2] if len(sys.argv) > 2 else "2026-08-14") + "T00:00:00+00:00")
    e = dt.datetime.fromisoformat((sys.argv[3] if len(sys.argv) > 3 else "2026-09-10") + "T00:00:00+00:00")
    os.makedirs(out, exist_ok=True)
    days = [s + dt.timedelta(days=i) for i in range((e - s).days)]
    results, t0 = {}, time.time()
    th = [threading.Thread(target=day_worker, args=(out, d, results), daemon=True) for d in days]
    for t in th:
        t.start()
    while any(t.is_alive() for t in th):
        time.sleep(30)
        print(f"[{time.strftime('%H:%M:%S')}] done {len(results)}/{len(days)} {dict(STATS)}", flush=True)
    summ = {"wall_s": time.time() - t0, "stats": dict(STATS), "days": {k.strftime("%Y-%m-%d"): v for k, v in sorted(results.items())}}
    for v in summ["days"].values():
        v.pop("boundary", None)
    json.dump(summ, open(os.path.join(out, f"run-{int(t0)}.json"), "w"), indent=1)
    failed = [k for k, v in summ["days"].items() if v.get("failed")]
    print("FINISHED failed:", failed, flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
