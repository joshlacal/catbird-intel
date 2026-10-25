import sys, json, urllib.request, urllib.parse, numpy as np
BUILD, OUT, MEMB = sys.argv[1], sys.argv[2], sys.argv[3]
comms = [int(x) for x in sys.argv[4].split(",")]
dids = [l.split("\t")[1] for l in open(f"{BUILD}/producers.tsv")]
memb = np.load(MEMB); st = np.load(f"{OUT}/producer_stats.npz"); fd = np.load(f"{OUT}/follower_degree.npz")
rng = np.random.default_rng(7)
for c in comms:
    m = np.flatnonzero(memb == c); pick = rng.choice(m, size=min(12, len(m)), replace=False)
    q = "&".join("actors=" + urllib.parse.quote(dids[p]) for p in pick)
    try:
        r = json.load(urllib.request.urlopen(f"https://public.api.bsky.app/xrpc/app.bsky.actor.getProfiles?{q}", timeout=20))
    except Exception as e:
        print(c, "ERR", e); continue
    byd = {p["did"]: p for p in r.get("profiles", [])}
    print(f"\n== community {c} (n={len(m)})")
    for p in pick:
        pr = byd.get(dids[p], {})
        desc = (pr.get("description") or "").replace("\n", " ")[:70]
        print(f"  @{pr.get('handle','<gone>'):<34} fol={pr.get('followersCount','?'):>6} follows={pr.get('followsCount','?'):>6} "
              f"posts={pr.get('postsCount','?'):>6} mass3k={fd['mass3000'][p]/max(st['followers'][p],1):.2f} | {desc}")
