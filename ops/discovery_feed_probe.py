#!/usr/bin/env python3
"""Discovery feed probe: public-signal candidate pool for one viewer.

Server half of the split-brain For You idea, read-only, nothing deployed.
Candidates = recent top-level posts by strangers that >= 2 of the viewer's
follows liked/reposted. Score = shrunk local/global hit rate
  social / (global likes+reposts + 20) x ln(1 + social)   (underdog: 4 follows of
  12 total beats 60 follows of 40k; ln(e+global)^alpha from the brief let viral win)
  x cand5 bridge bonus (engagers from distinct communities)
  x controversy discount ((replies+quotes) >> (likes+reposts) = dunk tank)
  x recency decay.
No private signals: only the public follow list and firehose engagement.

  uv run --with duckdb ops/discovery_feed_probe.py josh.uno --hours 48
"""
import argparse, json, math, os, subprocess, sys, time, urllib.parse, urllib.request
from collections import Counter

import duckdb

API = "https://public.api.bsky.app/xrpc/"
LAKE = os.path.expanduser("~/catbird-lake-local")
KNOWN_FOR = f"{LAKE}/s3/matched/cand5-full-matched/known_for.parquet"
LABELS = f"{LAKE}/claude-labels-cand5-20261028/community_labels.parquet"
ADULT = ("porn", "sexual", "nudity", "graphic-media", "gore")


def xrpc(method, **params):
    with urllib.request.urlopen(f"{API}{method}?{urllib.parse.urlencode(params)}", timeout=30) as r:
        return json.load(r)


def follows_of(did):
    out, cursor = [], None
    while True:
        page = xrpc("app.bsky.graph.getFollows", actor=did, limit=100, **({"cursor": cursor} if cursor else {}))
        out += [f["did"] for f in page["follows"]]
        cursor = page.get("cursor")
        if not cursor or not page["follows"]:
            return out


def ch(sql):
    # ponytail: ssh + clickhouse-client per call; a real service talks HTTP to 127.0.0.1:8123 (src/clickhouse.ts).
    cmd = ["ssh", "netcup", "clickhouse-client", "--readonly=1", "--max_threads=4",
           "--max_memory_usage=8000000000", "--max_execution_time=180"]
    r = subprocess.run(cmd, input=sql + "\nFORMAT JSONEachRow", capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stderr)
    return [json.loads(line) for line in r.stdout.splitlines()]


def lit(items):
    return "[" + ",".join("'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'" for s in items) + "]"


def communities(dids):
    con = duckdb.connect()
    con.execute("CREATE TEMP TABLE q(did VARCHAR)")
    con.executemany("INSERT INTO q VALUES (?)", [(d,) for d in dids])
    rows = con.execute(f"""
        SELECT k.did, k.community, coalesce(l.label, '#' || k.community::VARCHAR), k.hide_from_recs
        FROM '{KNOWN_FOR}' k JOIN q USING (did)
        LEFT JOIN '{LABELS}' l ON l.community = k.community
        WHERE k.rank = 0""").fetchall()
    return {d: (c, label, hide) for d, c, label, hide in rows}


def candidates(viewer, follows, hours):
    F = lit(follows)
    return ch(f"""
WITH
fe AS (
    SELECT did AS engager, kind, any(subject_did) AS sd, any(subject_rkey) AS sr
    FROM catbird.engagements
    WHERE did IN {F} AND indexed_at > now() - INTERVAL {hours} HOUR
    GROUP BY did, kind, rkey
    HAVING max(is_deleted) = 0 AND any(subject_collection) = 'app.bsky.feed.post'
),
activity AS (SELECT engager, count() AS n FROM fe GROUP BY engager),
cand AS (
    SELECT sd, sr, groupArray((engager, toString(kind))) AS hits
    FROM fe
    WHERE sd NOT IN {F} AND sd != '{viewer}'
    GROUP BY sd, sr
    HAVING uniqExact(engager) >= 2
),
post AS (
    SELECT did, rkey, any(created_at) AS created_at, any(text) AS text, any(langs) AS langs,
           any(embed_type) AS embed_type, any(external_uri) AS external_uri
    FROM catbird.posts
    WHERE (did, rkey) IN (SELECT sd, sr FROM cand)
    GROUP BY did, rkey
    HAVING max(is_deleted) = 0 AND any(reply_parent) = ''
       AND created_at > now() - INTERVAL {hours} HOUR
       AND NOT hasAny(any(self_labels), {lit(ADULT)})
),
uris AS (SELECT concat('at://', did, '/app.bsky.feed.post/', rkey) AS uri FROM post),
glob AS (
    SELECT subject_did AS d, subject_rkey AS r, count() AS lr
    FROM catbird.engagements
    WHERE (subject_did, subject_rkey) IN (SELECT did, rkey FROM post) AND is_deleted = 0
    GROUP BY d, r
),
talk AS (
    SELECT u, count() AS rq FROM (
        SELECT reply_parent AS u FROM catbird.posts WHERE reply_parent IN (SELECT uri FROM uris) AND is_deleted = 0
        UNION ALL
        SELECT quote_uri AS u FROM catbird.posts WHERE quote_uri IN (SELECT uri FROM uris) AND is_deleted = 0
    ) GROUP BY u
)
SELECT p.did AS author, p.rkey AS rkey, toUnixTimestamp(p.created_at) AS ts, p.text AS text, p.langs AS langs,
       p.embed_type AS embed, p.external_uri AS link, c.hits AS hits,
       ifNull(g.lr, 0) AS lr, ifNull(t.rq, 0) AS rq,
       (SELECT groupArray((engager, n)) FROM activity) AS activity
FROM post p
JOIN cand c ON c.sd = p.did AND c.sr = p.rkey
LEFT JOIN glob g ON g.d = p.did AND g.r = p.rkey
LEFT JOIN talk t ON t.u = concat('at://', p.did, '/app.bsky.feed.post/', p.rkey)
WHERE p.did NOT IN (SELECT did FROM catbird.v_hidden_from_algorithms)
  AND p.did NOT IN (SELECT did FROM catbird.v_inactive_accounts)
  AND p.did NOT IN (SELECT subject FROM catbird.blocks WHERE did = '{viewer}' AND is_deleted = 0)
  AND p.did NOT IN (SELECT did FROM catbird.blocks WHERE subject = '{viewer}' AND is_deleted = 0)
""")


def controversy_discount(lr, rq, free_ratio=0.5, power=1.5):
    """(replies+quotes)/(likes+reposts) above free_ratio is a dogpile signature."""
    ratio = (rq + 1) / (lr + 1)
    return 1.0 if ratio <= free_ratio else (free_ratio / ratio) ** power


def score(c, activity, comm, now, prior=20, tau_h=24.0):
    engagers = {e for e, _ in c["hits"]}
    # follows who like everything say less per like
    social = sum((1.5 if k == "repost" else 1.0) / math.log2(2 + activity.get(e, 1)) for e, k in c["hits"])
    groups = {comm[e][0] for e in engagers if e in comm}
    bridge = 1 + 0.5 * max(0, len(groups) - 1)
    lr = max(c["lr"], len(c["hits"]))  # engagements before ingest start are invisible
    age_h = max(0.0, (now - c["ts"]) / 3600)
    return (social / (lr + prior) * math.log1p(social) * bridge
            * controversy_discount(lr, c["rq"]) * math.exp(-age_h / tau_h)), groups, lr


def diversify(ranked, comm, n, per_author=1, per_community=2):
    """Cheap MMR stand-in on cluster ids; real MMR on embeddings belongs on-device."""
    out, authors, comms = [], Counter(), Counter()
    for c in ranked:
        home = comm.get(c["author"], (None,))[0]
        if authors[c["author"]] >= per_author or (home is not None and comms[home] >= per_community):
            continue
        out.append(c); authors[c["author"]] += 1; comms[home] += 1
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor", help="handle or DID")
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--pool", type=int, default=300, help="over-fetched pool handed to the client")
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--json", help="write the pool as a getFeedSkeleton-shaped file")
    a = ap.parse_args()

    viewer = a.actor if a.actor.startswith("did:") else xrpc("com.atproto.identity.resolveHandle", handle=a.actor)["did"]
    t0 = time.time()
    follows = follows_of(viewer)
    rows = candidates(viewer, follows, a.hours)
    print(f"viewer {viewer}: {len(follows)} follows, {len(rows)} candidates ({time.time() - t0:.1f}s)", file=sys.stderr)
    if not rows:
        return
    activity = dict(rows[0]["activity"])
    comm = communities(set(follows) | {r["author"] for r in rows})
    rows = [r for r in rows if not comm.get(r["author"], (0, "", False))[2]]  # cand5 hide_from_recs
    now = time.time()
    for r in rows:
        r["score"], r["groups"], r["lr"] = score(r, activity, comm, now)
    ranked = sorted((r for r in rows if r["score"] > 0), key=lambda r: -r["score"])
    pool = ranked[: a.pool]

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"feed": [{"post": f"at://{r['author']}/app.bsky.feed.post/{r['rkey']}"} for r in pool]}, f)

    labels = {c: lab for c, lab, _ in comm.values()}
    for i, r in enumerate(diversify(pool, comm, a.show), 1):
        engagers = {e for e, _ in r["hits"]}
        home = comm.get(r["author"], (None, "?"))[1]
        bridges = " + ".join(sorted(labels[g] for g in r["groups"]))[:110]
        text = " ".join(r["text"].split())[:160] or f"[{r['embed']}] {r['link']}"
        print(f"{i:2}. {r['score']:.3f}  {len(engagers)} follows / {r['lr']} global ({100 * len(engagers) / r['lr']:.0f}%)"
              f"  rq={r['rq']}  [{home}]\n    {text}\n    via: {bridges or '-'}\n"
              f"    https://bsky.app/profile/{r['author']}/post/{r['rkey']}")


if __name__ == "__main__":
    main()
