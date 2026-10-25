"""Live AppView profiles for producer_quality.py's sample_dids.tsv: removed vs kept producers.

  profile_check.py SAMPLE_TSV
"""
import collections, json, sys, urllib.parse, urllib.request
import numpy as np

rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
groups = collections.defaultdict(list)
for i in range(0, len(rows), 25):
    chunk = rows[i:i + 25]
    q = "&".join("actors=" + urllib.parse.quote(d) for _g, _p, d in chunk)
    r = json.load(urllib.request.urlopen(f"https://public.api.bsky.app/xrpc/app.bsky.actor.getProfiles?{q}", timeout=30))
    byd = {p["did"]: p for p in r.get("profiles", [])}
    for g, _p, d in chunk:
        groups[g].append(byd.get(d))
for g, profs in groups.items():
    live = [p for p in profs if p]
    posts = np.array([p.get("postsCount", 0) for p in live]); fol = np.array([p.get("followersCount", 0) for p in live])
    fws = np.array([p.get("followsCount", 0) for p in live])
    print(f"\n== {g}: {len(live)}/{len(profs)} resolvable; median posts={np.median(posts):.0f} followers={np.median(fol):.0f} "
          f"follows={np.median(fws):.0f}; posts<10: {(posts<10).mean():.0%}; follows>=0.8*followers: {(fws>=0.8*fol).mean():.0%}")
    for p in live:
        desc = (p.get("description") or "").replace("\n", " ")[:60]
        print(f"  @{p.get('handle',''):<34} fol={p.get('followersCount',0):>6} follows={p.get('followsCount',0):>6} "
              f"posts={p.get('postsCount',0):>6} | {desc}")
