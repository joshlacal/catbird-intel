#!/usr/bin/env python3
"""Relay com.atproto.sync.getRepoStatus for a DID list (relay-level takedowns of third-party-PDS
accounts are invisible to the PDS listing). <= 8 req/s. usage: relay_status.py DIDS OUT.tsv"""
import json, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import threading

UA = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"
lk, nxt = threading.Lock(), [time.monotonic()]


def one(did):
    with lk:
        w = nxt[0] - time.monotonic()
        nxt[0] = max(nxt[0], time.monotonic()) + 1 / 8.0
    if w > 0:
        time.sleep(w)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    f"https://bsky.network/xrpc/com.atproto.sync.getRepoStatus?did={did}", headers={"User-Agent": UA}), timeout=30) as r:
                d = json.loads(r.read())
                return did, 200, d.get("active"), d.get("status") or ""
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(30)
                continue
            try:
                err = json.loads(e.read() or b"{}").get("error", "")
            except Exception:
                err = ""
            return did, e.code, None, err
        except Exception as e:
            time.sleep(2 ** attempt)
            last = type(e).__name__
    return did, -1, None, last


dids = [l.strip() for l in open(sys.argv[1]) if l.startswith("did:")]
with ThreadPoolExecutor(8) as ex, open(sys.argv[2], "w") as fh:
    for did, code, active, status in ex.map(one, dids):
        fh.write(f"{did}\t{code}\t{'' if active is None else str(active).lower()}\t{status}\n")
