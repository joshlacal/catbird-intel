#!/usr/bin/env python3
"""First-record TID for did:web repos (M2 cohort fallback, S1 §2).

For each did:web: DID document -> PDS; describeRepo -> collections; per collection
listRecords(limit=1, reverse=true) -> oldest rkey; min TID-shaped rkey time across collections.
Writes did, first_us (0 when unknown), n_collections; and a jsonl log with errors.
usage: didweb_first.py DIDS_FILE OUT.tsv LOG.jsonl
"""
import json, sys, time, threading, urllib.request, urllib.error, urllib.parse, collections, re
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"
locks, last = collections.defaultdict(threading.Lock), collections.defaultdict(float)
B32 = "234567abcdefghijklmnopqrstuvwxyz"
TID = re.compile(r"^[234567abcdefghij][234567abcdefghijklmnopqrstuvwxyz]{12}$")
CALLS = collections.Counter()


def tid_us(rkey):
    if not TID.match(rkey):
        return 0
    v = 0
    for ch in rkey:
        v = v * 32 + B32.index(ch)
    return v >> 10


def get(url, rate=3.0):
    host = urllib.parse.urlparse(url).netloc
    with locks[host]:
        w = last[host] + 1.0 / rate - time.monotonic()
        if w > 0:
            time.sleep(w)
        last[host] = time.monotonic()
    CALLS[host] += 1
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        return -1, {"error": type(e).__name__}


def one(did):
    host = urllib.parse.unquote(did[len("did:web:"):]).replace(":", "/")
    c, doc = get(f"https://{host}/.well-known/did.json")
    if c != 200:
        return did, 0, 0, {"doc": c}
    pds = next((s.get("serviceEndpoint") for s in doc.get("service", []) or []
                if s.get("id") in ("#atproto_pds", f"{did}#atproto_pds")), None)
    if not pds:
        return did, 0, 0, {"doc": "no_pds"}
    pds = pds.rstrip("/")
    c, d = get(f"{pds}/xrpc/com.atproto.repo.describeRepo?repo={did}")
    if c != 200:
        return did, 0, 0, {"describe": c}
    cols = d.get("collections", []) or []
    best = 0
    for col in cols[:40]:
        c, r = get(f"{pds}/xrpc/com.atproto.repo.listRecords?repo={did}&collection={col}&limit=1&reverse=true")
        for rec in (r.get("records") or []) if c == 200 else []:
            t = tid_us(rec["uri"].rsplit("/", 1)[-1])
            if t and (best == 0 or t < best):
                best = t
    return did, best, len(cols), {}


def main():
    dids = [l.split("\t")[0].strip() for l in open(sys.argv[1]) if l.startswith("did:web:")]
    with open(sys.argv[2], "w") as out, open(sys.argv[3], "w") as log, ThreadPoolExecutor(16) as ex:
        futs = {ex.submit(one, d): d for d in dids}
        for f in as_completed(futs):
            try:
                did, t, n, err = f.result()
            except Exception as e:
                did, t, n, err = futs[f], 0, 0, {"exception": repr(e)[:200]}
            out.write(f"{did}\t{t}\t{n}\n")
            log.write(json.dumps({"did": did, "first_us": t, "n": n, **err}) + "\n")
    print(json.dumps({"dids": len(dids), "calls_by_host_top": CALLS.most_common(10), "calls": sum(CALLS.values())}))


if __name__ == "__main__":
    main()
