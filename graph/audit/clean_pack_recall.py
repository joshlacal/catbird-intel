"""Starter-pack recall split by how mass-followed each pack's members are.

Follow-train packs reward partitions that keep follow-trains together, so the
headline recall is reported next to a view restricted to packs whose median
member mass3k share (fraction of followers who follow >= 3,000 producers) is
below 0.4. Pack quality depends only on the follow graph, never on a partition.

  dump:   clean_pack_recall.py dump MANIFEST PRODUCERS OUT_PICKLE
  score:  clean_pack_recall.py score PACKS_PICKLE FOLLOWER_DEGREE_NPZ PRODUCER_STATS_NPZ GRAPH_DIR [GRAPH_DIR ...]

The pickle is written and read only by this script on the local machine.
"""
import json, pickle, sys, collections, multiprocessing as mp, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import eval_phase1 as ev  # noqa: E402
from pack_counts import scan  # noqa: E402

CLEAN = 0.4
BUCKETS = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def dump(manifest, producers, out):
    files = [f["path"] for f in json.load(open(manifest))["files"]]
    latest = {}
    with mp.Pool(14) as pool:
        for part in pool.imap_unordered(scan, files, chunksize=8):
            for k, v in part.items():
                if k not in latest or v[:2] > latest[k][:2]:
                    latest[k] = v
    deleted = {f"at://{d}/{c}/{r}" for (c, d, r), v in latest.items() if c == "app.bsky.graph.list" and v[1]}
    packs = [(f"at://{d}/{c}/{r}", v[2]) for (c, d, r), v in latest.items()
             if c == "app.bsky.graph.starterpack" and not v[1] and v[2] not in deleted]
    lists = collections.defaultdict(set)
    for (c, _d, _r), (_rev, dl, luri, subj) in latest.items():
        if c == "app.bsky.graph.listitem" and not dl and luri and luri not in deleted and subj.startswith("did:"):
            lists[luri].add(subj)
    pid = {line.split("\t")[1]: int(line.split("\t")[0]) for line in open(producers)}
    out_packs = [(uri, sorted(lists.get(luri, ())), [pid[d] for d in lists.get(luri, ()) if d in pid]) for uri, luri in packs]
    pickle.dump(out_packs, open(out, "wb"))
    print(f"packs={len(out_packs):,}")


def score(packs_path, fd_path, st_path, graph_dirs):
    packs = pickle.load(open(packs_path, "rb"))
    mass = np.load(fd_path)["mass3000"] / np.maximum(np.load(st_path)["followers"], 1)
    quality = [float(np.median(mass[p])) if p else np.nan for _u, _m, p in packs]
    results = {}
    for gd in graph_dirs:
        by_did, _kf, _comm = ev.load_known_for(pathlib.Path(gd))
        acc = collections.defaultdict(lambda: np.zeros(4))  # pairs, top1, top3, scene
        for (_uri, members, _p), q in zip(packs, quality):
            present = [d for d in members if d in by_did]
            if not 3 <= len(present) <= 200 or np.isnan(q):
                continue
            h = ev.pack_pair_hits(present, by_did)
            row = np.array([ev.pair_count(len(present)), h["top1"], h["top3"], h["scene"]], float)
            acc["all"] += row
            acc["clean" if q < CLEAN else "massy"] += row
            for lo, hi in BUCKETS:
                if lo <= q < hi:
                    acc[f"[{lo},{hi})"] += row
        results[gd] = {k: {"pairs": int(v[0]), "top1": v[1] / v[0], "top3": v[2] / v[0], "scene": v[3] / v[0]}
                       for k, v in acc.items()}
        print(f"\n== {gd}")
        for k in ["all", "clean", "massy"] + [f"[{lo},{hi})" for lo, hi in BUCKETS]:
            r = results[gd].get(k)
            if r:
                print(f"  {k:<12} pairs={r['pairs']:>13,} top1={r['top1']:.4f} top3={r['top3']:.4f} scene={r['scene']:.4f}")
    return results


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(*sys.argv[2:5])
    else:
        res = score(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
        print(json.dumps(res))
