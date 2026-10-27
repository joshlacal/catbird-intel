"""One labelling prompt per cand1 community: curated-list names, top producers' bios, sample posts.

  build_community_prompts.py GRAPH_DIR LIST_NAMES_TSV PROFILES_PARQUET POST_EXCLUDE_TXT PROFILE_EXCLUDE_TXT OUT_JSONL DAY_DIR [DAY_DIR ...]

GRAPH_DIR holds communities.parquet and known_for.parquet. hideFromAlgorithmicRecommendations
is about posts in discovery feeds, not profiles (DECISIONS 2026-10-25), so producers in
POST_EXCLUDE_TXT (hide-from-recs union) or flagged hide_from_recs contribute no posts but
keep their bios; PROFILE_EXCLUDE_TXT (user-intents opt-outs) removes both. Sample posts are
live original posts (not replies) from the netcup days by the community's rank-0 KnownFor
members, at most one per author, preferring >=40 characters. Seed 20261027.
"""
import collections
import json
import random
import sys

import pyarrow.compute as pc
import pyarrow.parquet as pq

GRAPH, LISTS, PROFILES, POST_EXCL, PROFILE_EXCL, OUT, *DAYS = sys.argv[1:]
rng = random.Random(20261027)
excl = {l.strip() for l in open(PROFILE_EXCL) if l.strip()}
post_excl = excl | {l.strip() for l in open(POST_EXCL) if l.strip()}

kf = pq.read_table(f"{GRAPH}/known_for.parquet", columns=["did", "community", "rank", "hide_from_recs"])
post_excl.update(kf.filter(kf["hide_from_recs"])["did"].to_pylist())
kf0 = kf.filter(pc.equal(kf["rank"], 0))
home = {d: c for d, c in zip(kf0["did"].to_pylist(), kf0["community"].to_pylist()) if d not in post_excl}

prof = pq.read_table(PROFILES)
bios = {d: (n, b) for d, n, b in zip(*(prof[c].to_pylist() for c in ["did", "name", "description"]))}
lists = {}
for line in open(LISTS):
    c, _size, _fb, _ft, hits, _circle, names = line.rstrip("\n").split("\t")
    if c != "community":
        lists[int(c)] = (int(hits), names)

posts = collections.defaultdict(dict)  # community -> author -> (long, text)
langs = collections.defaultdict(collections.Counter)
for d in DAYS:
    t = pq.read_table(f"{d}/posts.parquet", columns=["did", "is_deleted", "text", "langs", "reply_parent"])
    t = t.filter(pc.and_(pc.equal(t["is_deleted"], 0), pc.equal(t["reply_parent"], "")))
    for did, text, lg in zip(*(t[c].to_pylist() for c in ["did", "text", "langs"])):
        c = home.get(did)
        if c is None or not text.strip():
            continue
        langs[c][lg[0] if lg else "?"] += 1
        long = len(text) >= 40
        old = posts[c].get(did)
        if old is None or (long and not old[0]) or (long == old[0] and rng.random() < 0.3):
            posts[c][did] = (long, text)

comms = pq.read_table(f"{GRAPH}/communities.parquet")
n = 0
with open(OUT, "w") as out:
    for c, scene, size, tops, folls in zip(*(comms[k].to_pylist() for k in
                                              ["community", "scene_id", "size", "top_dids", "top_followers"])):
        prods = [(d, f) for d, f in zip(tops, folls) if d not in excl][:12]
        lines = [f"Community {c} (scene {scene}), {size} accounts.",
                 f"Post languages in a 4-day sample: " + (", ".join(f"{k} {v}" for k, v in langs[c].most_common(5)) or "no posts"),
                 f"Most common names of curated lists containing members ({lists.get(c, (0, ''))[0]} memberships): "
                 + (lists.get(c, (0, ""))[1] or "none"),
                 "Most-followed members (display name | bio):"]
        for d, f in prods:
            name, bio = bios.get(d, ("", ""))
            lines.append(f"- {f} followers | {name.strip()[:60]} | {' '.join(bio.split())[:200]}")
        sample = list(posts[c].values())
        rng.shuffle(sample)
        sample.sort(key=lambda p: not p[0])
        lines.append("Sample recent original posts by members:")
        lines += [f"- {' '.join(t.split())[:280]}" for _, t in sample[:10]] or ["- none"]
        out.write(json.dumps({"custom_id": f"c{c}", "community": c, "size": size,
                              "prompt": "\n".join(lines)}, ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {n} prompts -> {OUT}")
