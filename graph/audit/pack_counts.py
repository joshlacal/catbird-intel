"""Starter-pack listing count per producer, using eval_phase1's latest-rev-wins rules."""
import sys, json, collections, multiprocessing as mp
import numpy as np, pyarrow.parquet as pq

COLS = ["collection", "did", "rkey", "rev", "operation", "list_uri", "subject"]
COLL = ["app.bsky.graph.listitem", "app.bsky.graph.list", "app.bsky.graph.starterpack"]


def scan(path):
    tbl = pq.read_table(path, columns=COLS, filters=[("collection", "in", COLL)])
    latest = {}
    for collection, did, rkey, rev, op, luri, subj in zip(*(tbl[c].to_pylist() for c in COLS)):
        key = (collection, did, rkey)
        value = (rev or "", op == "delete", luri or "", subj or "")
        if key not in latest or value[:2] > latest[key][:2]:
            latest[key] = value
    return latest


if __name__ == "__main__":
    manifest, producers, out = sys.argv[1:4]
    files = [f["path"] for f in json.load(open(manifest))["files"]]
    latest = {}
    with mp.Pool(14) as pool:
        for i, part in enumerate(pool.imap_unordered(scan, files, chunksize=8)):
            for k, v in part.items():
                if k not in latest or v[:2] > latest[k][:2]:
                    latest[k] = v
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(files)} keys={len(latest):,}", flush=True)
    deleted = {f"at://{d}/{c}/{r}" for (c, d, r), v in latest.items() if c == "app.bsky.graph.list" and v[1]}
    packs = [v[2] for (c, d, r), v in latest.items() if c == "app.bsky.graph.starterpack" and not v[1] and v[2] not in deleted]
    lists = collections.defaultdict(set)
    for (c, _d, _r), (_rev, dl, luri, subj) in latest.items():
        if c == "app.bsky.graph.listitem" and not dl and luri and luri not in deleted and subj.startswith("did:"):
            lists[luri].add(subj)
    freq = collections.Counter()
    for luri in packs:
        freq.update(lists.get(luri, ()))
    dids = [l.split("\t")[1] for l in open(producers)]
    counts = np.array([freq.get(d, 0) for d in dids], np.int32)
    np.save(out, counts)
    print(f"packs={len(packs):,} producers_in_any_pack={(counts>0).sum():,}")
