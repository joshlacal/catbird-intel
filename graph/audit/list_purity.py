"""Curated-list co-membership as an independent check on partitions (DECISIONS 2026-10-25).

  list_purity.py CURATELISTS_TSV PACKS_PICKLE FOLLOWER_DEGREE_NPZ PRODUCER_STATS_NPZ MEMBERSHIP_NPY FLAGS_NPZ OUT_DIR GRAPH_DIR [GRAPH_DIR ...]

Curate lists are people's own topical groupings and, unlike starter packs, they did not
create follows in bulk. Lists and packs are scored the same way: pair-weighted and per-list
macro top-1 / top-3 / scene, top-1 lift over the uniform same-size null, split by the median
member mass3k share. Writes OUT_DIR/list_purity.json and, for the first GRAPH_DIR,
community_list_names.tsv: each KnownFor community's most common list names, its follow-back-name share,
and the share of its members in follow_back / follow_train flagged raw Leiden communities (MEMBERSHIP_NPY ids).
The packs pickle is written by clean_pack_recall.py dump on this machine; nothing else is unpickled.
"""
import collections, json, pathlib, pickle, re, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import eval_phase1 as ev  # noqa: E402

LISTS, PACKS, FD, ST, MEMB, FLAGS, OUT = sys.argv[1:8]
GRAPHS = sys.argv[8:]
CLEAN = 0.4
# List names that announce a reciprocity circle rather than a topic.
CIRCLE = re.compile(r"follow\s*-?\s*back|f4f|\bfb\b|mutual|takip|フォロバ|相互|follow\s*train|#?resist|blue\s*wave|\brp\b|roleplay", re.I)

rng = np.random.default_rng(7)
rows = []
for line in open(LISTS):
    uri, owner, name, _total, members = line.rstrip("\n").split("\t")
    rows.append((uri, owner, name, sorted(set(m for m in members.split(",") if m))))
rows.sort()
seen, by_owner = set(), collections.defaultdict(list)
for r in rows:
    key = tuple(r[3])
    if len(key) >= 3 and key not in seen:
        seen.add(key); by_owner[r[1]].append(r)
lists = []
for owner, rs in by_owner.items():
    lists += [rs[i] for i in sorted(rng.choice(len(rs), min(3, len(rs)), replace=False))]
print(f"curatelists read={len(rows):,} with >=3 producers, deduped, <=3/owner: {len(lists):,} owners={len(by_owner):,}")

pid = None
mass = np.load(FD)["mass3000"] / np.maximum(np.load(ST)["followers"], 1)
memb = np.load(MEMB).astype(np.int64); fl = np.load(FLAGS)
fb_p, ft_p = fl["follow_back"][memb], fl["follow_train"][memb]


def score(groups, by_did, quality, null_p):
    acc = collections.defaultdict(lambda: np.zeros(4)); macro = collections.defaultdict(list)
    for members, q in zip(groups, quality):
        present = [d for d in members if d in by_did]
        if not 3 <= len(present) <= 200 or np.isnan(q):
            continue
        h = ev.pack_pair_hits(present, by_did); n = ev.pair_count(len(present))
        row = np.array([n, h["top1"], h["top3"], h["scene"]], float)
        for view in ("all", "clean" if q < CLEAN else "massy"):
            acc[view] += row; macro[view].append(h["top1"] / n)
    return {v: {"groups": len(macro[v]), "pairs": int(a[0]), "top1": a[1] / a[0], "top3": a[2] / a[0], "scene": a[3] / a[0],
                "macro_top1": float(np.mean(macro[v])), "lift_top1": a[1] / a[0] / null_p, "macro_lift_top1": float(np.mean(macro[v])) / null_p}
            for v, a in acc.items()}


packs = pickle.load(open(PACKS, "rb"))
results = {}
for gd in GRAPHS:
    by_did, _kf, comm_tbl = ev.load_known_for(pathlib.Path(gd))
    if pid is None:
        dids = [l.split("\t")[1] for l in open(pathlib.Path(gd) / "producers.tsv")]
        pid = {d: i for i, d in enumerate(dids)}
    null_p = ev.primary_partition_null(by_did)["expected_same_primary_probability"]

    def quality(groups):
        return [float(np.median(mass[[pid[d] for d in g if d in pid]])) if any(d in pid for d in g) else np.nan for g in groups]

    results[gd] = {"null_same_primary": null_p,
                   "lists": score([l[3] for l in lists], by_did, quality([l[3] for l in lists]), null_p),
                   "packs": score([m for _u, m, _p in packs], by_did, quality([m for _u, m, _p in packs]), null_p)}
    print(f"\n== {gd}  null={null_p:.2e}")
    for src in ("lists", "packs"):
        for v in ("all", "clean", "massy"):
            r = results[gd][src].get(v)
            if r:
                print(f"  {src:<5} {v:<5} n={r['groups']:>7,} top1={r['top1']:.4f} top3={r['top3']:.4f} scene={r['scene']:.4f} "
                      f"macro_top1={r['macro_top1']:.4f} lift={r['lift_top1']:.0f}x macro_lift={r['macro_lift_top1']:.0f}x")

    if gd == GRAPHS[0]:
        names = collections.defaultdict(collections.Counter); circle = collections.Counter(); hits = collections.Counter()
        for _uri, _owner, name, members in lists:
            norm = re.sub(r"\s+", " ", name.strip().lower())[:60]
            per_c = collections.Counter(by_did[d][0][0] for d in members if d in by_did)
            for c, n in per_c.items():
                if n >= 3:
                    names[c][norm] += 1
                hits[c] += n
                if CIRCLE.search(name):
                    circle[c] += n
        size = dict(zip(comm_tbl["community"].to_pylist(), comm_tbl["size"].to_pylist())) if "size" in comm_tbl.column_names else {}
        flagged = collections.defaultdict(lambda: np.zeros(3))
        for d, entries in by_did.items():
            if d in pid:
                flagged[entries[0][0]] += (1, fb_p[pid[d]], ft_p[pid[d]])
        with open(f"{OUT}/community_list_names.tsv", "w") as f:
            f.write("community\tsize\tfb_share\tft_share\tlist_member_hits\tcircle_name_share\ttop_list_names\n")
            for c in sorted(flagged, key=lambda c: -size.get(c, 0)):
                top = "; ".join(f"{n} ({k})" for n, k in names[c].most_common(6))
                n, b, t = flagged[c]
                f.write(f"{c}\t{size.get(c, '')}\t{b / n:.2f}\t{t / n:.2f}\t{hits[c]}\t{circle[c] / max(hits[c], 1):.3f}\t{top}\n")

json.dump(results, open(f"{OUT}/list_purity.json", "w"), indent=1)
