#!/usr/bin/env python3
"""S5 (vii) completeness check: getRepoStatus for a sample of our-data DIDs absent from listRepos.

For each DID: relay com.atproto.sync.getRepoStatus, then DID document -> PDS getRepoStatus.
usage: absent_check.py SAMPLE_DIDS OUT.jsonl
"""
import json, sys, time, threading, urllib.request, urllib.error, urllib.parse, collections
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"
locks = collections.defaultdict(threading.Lock)
last = collections.defaultdict(float)


def get(url, rate=8.0):
    host = urllib.parse.urlparse(url).netloc
    with locks[host]:
        w = last[host] + 1.0 / rate - time.monotonic()
        if w > 0:
            time.sleep(w)
        last[host] = time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}
    except Exception as e:
        return -1, {"error": type(e).__name__}


def check(did):
    out = {"did": did}
    out["relay"] = get(f"https://bsky.network/xrpc/com.atproto.sync.getRepoStatus?did={did}")
    if did.startswith("did:plc:"):
        c, doc = get(f"https://plc.directory/{did}", rate=3.0)  # shares the host with the PLC export (6/s)
    else:
        c, doc = get(f"https://{did[len('did:web:'):]}/.well-known/did.json")
    out["doc_status"] = c
    pds = None
    if c == 200:
        for s in doc.get("service", []) or []:
            if s.get("id") in ("#atproto_pds", f"{did}#atproto_pds"):
                pds = s.get("serviceEndpoint")
    out["pds"] = pds
    if pds:
        out["pds_status"] = get(f"{pds.rstrip('/')}/xrpc/com.atproto.sync.getRepoStatus?did={did}", rate=4.0)
    return out


def main():
    dids = [l.strip() for l in open(sys.argv[1]) if l.startswith("did:")]
    with open(sys.argv[2], "w") as fh, ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(check, d): d for d in dids}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:
                r = {"did": futs[f], "exception": repr(e)[:200]}
            fh.write(json.dumps(r) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
