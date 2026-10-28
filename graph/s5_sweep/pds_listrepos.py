#!/usr/bin/env python3
"""S5 (vii-b): com.atproto.sync.listRepos on every PDS host (the relay lists only recently seen repos).

Hosts: every hostname in the relay's com.atproto.sync.listHosts (all statuses), plus any extra
hosts file given. Each host is paged with its own cursor at <= RATE req/s (PDS policy 3000/5 min);
HOSTS_PAR hosts run at once. Per host: zstd-framed TSV (did, active, status) + checkpoint
(cursor, byte offset), so a crash resumes exactly. A host that fails is recorded, never retried
forever (its repos stay unlisted, which is the conservative direction for purging).
usage: pds_listrepos.py OUTDIR [EXTRA_HOSTS_FILE]
"""
import json, os, sys, time, threading, urllib.request, urllib.error, urllib.parse, gzip, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from compression import zstd

UA = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"
RATE = float(os.environ.get("RATE", "8"))
HOSTS_PAR = int(os.environ.get("HOSTS_PAR", "24"))
LOCK = threading.Lock()
STATS = collections.Counter()


def http(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json", "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return json.loads(body)


def list_hosts():
    hosts, cur = [], None
    while True:
        d = http("https://bsky.network/xrpc/com.atproto.sync.listHosts?limit=1000" + (f"&cursor={cur}" if cur else ""))
        hosts += d.get("hosts", [])
        cur = d.get("cursor")
        if not cur or not d.get("hosts"):
            return hosts
        time.sleep(0.2)


def host_worker(out, host):
    safe = host.replace("/", "_")
    ckp, dp = os.path.join(out, f"{safe}.ck.json"), os.path.join(out, f"{safe}.tsv.zst")
    ck = {"host": host, "cursor": None, "offset": 0, "pages": 0, "rows": 0, "counts": {}, "done": False}
    if os.path.exists(ckp):
        ck = json.load(open(ckp))
    if ck["done"] or ck.get("failed"):
        return ck
    nxt_t = time.monotonic()
    with open(dp, "ab") as fh:
        fh.truncate(ck["offset"])
        fh.seek(ck["offset"])
        while True:
            url = f"https://{host}/xrpc/com.atproto.sync.listRepos?limit=1000" + (
                f"&cursor={urllib.parse.quote(ck['cursor'], safe='')}" if ck["cursor"] else "")
            d, err = None, None
            for attempt in range(8):
                w = nxt_t - time.monotonic()
                if w > 0:
                    time.sleep(w)
                nxt_t = time.monotonic() + 1.0 / RATE
                with LOCK:
                    STATS["calls"] += 1
                try:
                    d = http(url, timeout=12 if ck["pages"] == 0 else 90)
                    with LOCK:
                        STATS["http_200"] += 1
                    break
                except urllib.error.HTTPError as e:
                    with LOCK:
                        STATS[f"http_{e.code}"] += 1
                    err = f"http_{e.code}"
                    if e.code == 429:
                        reset = e.headers.get("ratelimit-reset")
                        time.sleep(max(5, min((float(reset) - time.time()) if reset else 60, 300)))
                        continue
                    if e.code in (400, 401, 403, 404, 501) or (ck["pages"] == 0 and attempt >= 1):
                        break
                except Exception as e:
                    with LOCK:
                        STATS["neterr_" + type(e).__name__] += 1
                    err = "neterr_" + type(e).__name__
                    if ck["pages"] == 0:
                        break   # unreachable host: one try, do not spend minutes on it
                time.sleep(min(30, 2 ** attempt))
            if d is None:
                ck["failed"] = err
                break
            repos = d.get("repos") or []
            lines = []
            for r in repos:
                if not isinstance(r, dict) or not str(r.get("did", "")).startswith("did:"):
                    continue
                a = "true" if r.get("active", True) else "false"
                s = r.get("status") or ""
                lines.append(f"{r['did']}\t{a}\t{s}\n")
                k = f"{a}/{s}"
                ck["counts"][k] = ck["counts"].get(k, 0) + 1
            if lines:
                fh.write(zstd.compress("".join(lines).encode()))
                fh.flush()
                os.fsync(fh.fileno())
            cur = d.get("cursor")
            ck.update(offset=fh.tell(), pages=ck["pages"] + 1, rows=ck["rows"] + len(lines))
            ck["done"] = (not repos) or (not cur) or cur == ck["cursor"]
            ck["cursor"] = cur or ck["cursor"]
            if ck["pages"] > 5000:
                ck["failed"] = "page_cap"
            tmp = ckp + ".tmp"
            json.dump(ck, open(tmp, "w"))
            os.replace(tmp, ckp)
            if ck["done"] or ck.get("failed"):
                break
    json.dump(ck, open(ckp + ".tmp", "w"))
    os.replace(ckp + ".tmp", ckp)
    return ck


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    hp = os.path.join(out, "hosts.json")
    if os.path.exists(hp):
        hosts = json.load(open(hp))
    else:
        hosts = list_hosts()
        json.dump(hosts, open(hp, "w"))
    names = {h["hostname"] for h in hosts}
    if len(sys.argv) > 2:
        names |= {l.strip() for l in open(sys.argv[2]) if l.strip()}
    # big hosts first so the long poles start early
    cnt = {h["hostname"]: h.get("accountCount", 0) for h in hosts}
    live = {h["hostname"] for h in hosts if h.get("status") in ("active", "idle")}
    order = sorted(names, key=lambda n: (n not in live and n in cnt, -cnt.get(n, 0), n))
    print(f"{len(order)} hosts", flush=True)
    t0, res = time.time(), {}
    with ThreadPoolExecutor(HOSTS_PAR) as ex:
        futs = {ex.submit(host_worker, out, h): h for h in order}
        for i, f in enumerate(as_completed(futs), 1):
            h = futs[f]
            try:
                res[h] = f.result()
            except Exception as e:
                res[h] = {"host": h, "failed": "exception " + repr(e)[:200]}
            if i % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] hosts {i}/{len(order)} rows={sum(r.get('rows', 0) for r in res.values())} {dict(STATS)}", flush=True)
    summ = {"wall_s": time.time() - t0, "stats": dict(STATS), "hosts": res}
    json.dump(summ, open(os.path.join(out, f"run-{int(t0)}.json"), "w"), indent=1)
    print("FINISHED", json.dumps({"hosts": len(res), "ok": sum(1 for r in res.values() if r.get("done")),
                                  "failed": sum(1 for r in res.values() if r.get("failed")),
                                  "rows": sum(r.get("rows", 0) for r in res.values()), "stats": dict(STATS)}), flush=True)


if __name__ == "__main__":
    main()
