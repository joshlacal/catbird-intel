"""Latest profile (display name, bio) for a fixed DID set, from the archive event Parquet.

  extract_profiles.py DIDS_TXT EVAL_INPUT_MANIFEST_JSON OUT_PARQUET

Reads only `app.bsky.actor.profile` rows of the files listed in an eval input manifest
(the same files eval_phase1.py read), keeps the highest rev per DID, and drops DIDs whose
latest profile op is a delete. Read-only on the inputs.
"""
import concurrent.futures as cf
import json
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DIDS, MANIFEST, OUT = sys.argv[1:4]
want = pa.array(sorted({l.strip() for l in open(DIDS) if l.strip()}))
files = [f["path"] for f in json.load(open(MANIFEST))["files"] if "/events-" in f["path"]]
COLS = ["did", "rev", "operation", "name", "description", "extra"]


def scan(path):
    t = pq.read_table(path, columns=COLS + ["collection"],
                      filters=[("collection", "==", "app.bsky.actor.profile")])
    return t.filter(pc.is_in(t["did"], value_set=want)).select(COLS)


best = {}
with cf.ThreadPoolExecutor(12) as ex:
    for i, t in enumerate(ex.map(scan, files)):
        for d, rev, op, name, desc, extra in zip(*(t[c].to_pylist() for c in COLS)):
            key = (rev or "", op == "delete")
            if d not in best or key > best[d][0]:
                best[d] = (key, op, name or "", desc or "", extra or "")
        if i % 500 == 0:
            print(f"{i}/{len(files)} files, {len(best)} dids", flush=True)

rows = [(d, v[2], v[3], v[4]) for d, v in sorted(best.items()) if v[1] != "delete"]
pq.write_table(pa.table({k: [r[i] for r in rows] for i, k in enumerate(["did", "name", "description", "extra"])}), OUT)
print(f"wanted {len(want)}, found {len(best)}, written {len(rows)} -> {OUT}")
