"""Synthetic extract in the exact all-extract-20261024 schemas (no personal data).

Every DID, post and text here is generated. The world is small but has one
deliberate instance of each case the pass must handle: duplicate rows across
shards, a deleted like, a deleted follow, a follow made after a like, a block
burst, a confirmed reply, a courtesy (ack) like, a like on a reply, a
supplement author (LU), a purged (deactivated) account, a bulkDataset:false
account, a syntheticContentGeneration:false producer, a hide-from-recs
producer, a curate list, a mod list with a list-block, a threadgate with a
hidden reply, a postgate with a detached quote, profiles, and Oct 20 likes for
the positive control.
"""
from __future__ import annotations

import datetime as dt
import pathlib
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq

GRAPH = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GRAPH))
from archive_graph_extract import SCHEMA as EV_SCHEMA, tid_from_uint64  # noqa: E402
from post_extract import POST_SCHEMA  # noqa: E402

B32 = "234567abcdefghijklmnopqrstuvwxyz"


def did(n: int) -> str:
    rnd = random.Random(f"did{n}")
    return "did:plc:" + "".join(rnd.choice(B32) for _ in range(24))


def ts(day: str, sec: int = 0) -> float:
    return dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() + sec


_clock = [0]


def tid(t: float) -> str:
    _clock[0] = (_clock[0] + 1) % 1024
    return tid_from_uint64((int(t * 1_000_000) << 10) | _clock[0])


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class World:
    """Names for the special accounts, so tests can assert on them."""

    N_PROD = 40
    N_USERS = 260

    def __init__(self) -> None:
        self.producers = [did(i) for i in range(self.N_PROD)]
        self.users = [did(1000 + i) for i in range(self.N_USERS)]
        self.everyone = self.producers + self.users
        self.hide_prod = self.producers[3]
        self.synth_prod = self.producers[4]
        self.sup_prod = self.producers[5]          # supplement account: events but no posts
        self.purged = self.users[7]                # deactivated (account event active=false)
        self.bulk_no = self.users[8]               # bulkDataset:false
        self.burst_blocker = self.users[9]
        self.replier = self.users[10]
        self.p_main = self.producers[0]
        self.late_follower = self.users[11]
        self.o13_user = self.users[254]           # cityHash64(did) % 100 == 13: an M1 user
        self.o13_prod = self.producers[12]
        self.m2_users = [self.users[50], self.users[51], self.users[254]]


def build(stage: pathlib.Path, side: pathlib.Path, seed: int = 7) -> World:
    rnd = random.Random(seed)
    w = World()
    ev: list[dict] = []
    po: list[dict] = []
    ingest = ts("2026-10-21", 3600)
    seq = [0]

    def base(d: str, src: str = "gce_main", t_ing: float | None = None) -> dict:
        seq[0] += 1
        return {"source_id": src, "seq": seq[0], "time": iso(t_ing if t_ing else ingest + rnd.random() * 3600),
                "did": d, "account_active": None, "account_status": ""}

    def commit(d, coll, rkey, op="create", rev=None, src="gce_main", t_ing=None, **kw) -> dict:
        row = {**base(d, src, t_ing), "kind": "commit", "collection": coll, "rkey": rkey,
               "rev": rev or tid(ingest), "operation": op, "created_at": kw.pop("created_at", ""),
               "subject": "", "list_uri": "", "name": "", "description": "", "extra": ""}
        row.update(kw)
        ev.append(row)
        return row

    def post(d, t, text="", reply=None, root=None, quote=None, op="create", rkey=None, src="gce_main"):
        rk = rkey or tid(t)
        po.append({**{k: v for k, v in base(d, src).items() if k in ("source_id", "seq", "time", "did")},
                   "rkey": rk, "rev": tid(ingest), "operation": op, "created_at": iso(t),
                   "text": text if op != "delete" else "", "langs": ["en"] if op != "delete" else [],
                   "reply_root": root or reply or "", "reply_parent": reply or "", "embed_type": "app.bsky.embed.record" if quote else "",
                   "quote_uri": quote or "", "external_uri": "", "image_count": 0, "image_alts": [],
                   "facet_links": [], "facet_tags": [], "facet_mentions": [], "self_labels": []})
        return f"at://{d}/app.bsky.feed.post/{rk}"

    # profiles (updated twice: the later revision wins)
    for i, p in enumerate(w.producers):
        commit(p, "app.bsky.actor.profile", "self", rev=tid(ingest - 10), name=f"Old name {i}", description="old bio")
        commit(p, "app.bsky.actor.profile", "self", op="update", rev=tid(ingest), name=f"Producer {i}", description=f"synthetic bio {i}")

    # producer posts across the whole year
    days = ["2025-11-20", "2026-03-10", "2026-05-20", "2026-06-15", "2026-07-10", "2026-07-25",
            "2026-08-05", "2026-08-10", "2026-08-20", "2026-09-01", "2026-09-25", "2026-10-01", "2026-10-10", "2026-10-15", "2026-10-20"]
    posts_by: dict[str, list[tuple[str, float]]] = {}
    for p in w.producers:
        if p == w.sup_prod:
            continue
        for d in days:
            for k in range(2):
                t = ts(d, 3600 * (k + 1) + rnd.randint(0, 600))
                uri = post(p, t, text=f"synthetic post by {p[-4:]} on {d} #{k}")
                posts_by.setdefault(p, []).append((uri, t))
    # supplement producer: records in events only (no post rows), engagement on made-up post URIs
    for d in days[-6:]:
        commit(w.sup_prod, "app.bsky.graph.follow", tid(ts(d)), src="repo_supplement", subject=w.p_main, created_at=iso(ts(d)))
        posts_by.setdefault(w.sup_prod, []).append((f"at://{w.sup_prod}/app.bsky.feed.post/{tid(ts(d, 50))}", ts(d, 50)))

    # follows: users follow 3-12 producers; one delete in the live tail
    follows: dict[str, list[str]] = {}
    for u in w.users:
        ps = rnd.sample(w.producers, rnd.randint(3, 12))
        follows[u] = ps
        for p in ps:
            t = ts("2026-0" + str(rnd.randint(1, 7)) + "-1" + str(rnd.randint(0, 9)), rnd.randint(0, 80000))
            commit(u, "app.bsky.graph.follow", tid(t), subject=p, created_at=iso(t + rnd.choice([0, 0, 2, 90000])))
    # producers follow each other (mutuals) and some users back
    for i, p in enumerate(w.producers[:20]):
        for q in w.producers[i + 1:i + 6]:
            t = ts("2026-02-01", i * 100)
            commit(p, "app.bsky.graph.follow", tid(t), subject=q, created_at=iso(t))
            commit(q, "app.bsky.graph.follow", tid(t + 5), subject=p, created_at=iso(t + 5))
    gone = follows[w.users[20]][0]
    fr = tid(ts("2026-05-02"))
    commit(w.users[20], "app.bsky.graph.follow", fr, subject=gone, created_at=iso(ts("2026-05-02")))
    commit(w.users[20], "app.bsky.graph.follow", fr, op="delete", rev=tid(ts("2026-10-22")), t_ing=ts("2026-10-22", 60))
    # a follow made after the user's first like of that producer
    commit(w.late_follower, "app.bsky.graph.follow", tid(ts("2026-08-25")), subject=w.p_main, created_at=iso(ts("2026-08-25")))

    # likes and reposts
    like_keys: list[tuple[str, str]] = []
    for u in w.users:
        targets = follows[u] + rnd.sample(w.producers, 3)
        for p in targets:
            for (uri, t) in rnd.sample(posts_by.get(p, []), min(4, len(posts_by.get(p, [])))):
                tl = t + rnd.randint(60, 86400)
                rk = tid(tl)
                commit(u, "app.bsky.feed.like", rk, subject=uri, created_at=iso(tl))
                like_keys.append((u, rk))
                if rnd.random() < 0.25:
                    commit(u, "app.bsky.feed.repost", tid(tl + 30), subject=uri, created_at=iso(tl + 30))
    lf_uri, lf_t = posts_by[w.p_main][7]  # 2026-08-05: liked before the 08-25 follow
    commit(w.late_follower, "app.bsky.feed.like", tid(lf_t + 100), subject=lf_uri, created_at=iso(lf_t + 100))
    # duplicate rows across shards: the same like record (same rev) lands in two files
    dup = next(r for r in ev if r["collection"] == "app.bsky.feed.like")
    dup2 = dict(dup)
    dup2["seq"] = 10_000_000
    # a deleted like (unlike in the live tail)
    u_del, rk_del = like_keys[5]
    commit(u_del, "app.bsky.feed.like", rk_del, op="delete", rev=tid(ts("2026-10-22", 100)), t_ing=ts("2026-10-22", 200))

    # replies: replier replies to p_main's posts; p_main likes one reply (CR) and courtesy-likes nothing;
    # another user likes the reply (LR); the parent author... ack like = p_main liking replies to its own post
    conv = []
    for k, (uri, t) in enumerate(posts_by[w.p_main][8:12]):
        tr = t + 600
        r_uri = post(w.replier, tr, text=f"synthetic reply {k}", reply=uri, root=uri)
        conv.append((r_uri, tr))
    commit(w.p_main, "app.bsky.feed.like", tid(conv[0][1] + 3600), subject=conv[0][0], created_at=iso(conv[0][1] + 3600))  # CR + ACK
    commit(w.users[30], "app.bsky.feed.like", tid(conv[1][1] + 60), subject=conv[1][0], created_at=iso(conv[1][1] + 60))  # LR
    # hostile reply: blocked within 48 h
    h_uri = post(w.users[31], posts_by[w.p_main][9][1] + 900, text="synthetic hostile reply", reply=posts_by[w.p_main][9][0])
    commit(w.p_main, "app.bsky.graph.block", tid(posts_by[w.p_main][9][1] + 4000), subject=w.users[31], created_at=iso(posts_by[w.p_main][9][1] + 4000))
    # quotes of p_main by followers and non-followers
    q_uris = []
    for k, u in enumerate(w.users[40:60]):
        uri, t = posts_by[w.p_main][10 + (k % 4)]
        q_uris.append(post(u, t + 1200 + k, text=f"synthetic quote {k}", quote=uri))
    # threadgate hiding the hostile reply; postgate detaching a quote; one gate with allow=[]
    gate_post = posts_by[w.p_main][9][0].rsplit("/", 1)[1]
    commit(w.p_main, "app.bsky.feed.threadgate", gate_post,
           extra='{"post":"%s","$type":"app.bsky.feed.threadgate","allow":[],"createdAt":"x","hiddenReplies":["%s"]}' % (posts_by[w.p_main][9][0], h_uri))
    qpost = posts_by[w.p_main][10][0].rsplit("/", 1)[1]
    commit(w.p_main, "app.bsky.feed.postgate", qpost,
           extra='{"post":"%s","$type":"app.bsky.feed.postgate","createdAt":"x","embeddingRules":[],"detachedEmbeddingUris":["%s"]}' % (posts_by[w.p_main][10][0], q_uris[0]))
    # a deleted post
    post(w.producers[6], ts("2026-08-11"), text="to be deleted", rkey="3mzzzzzzzzzz2")
    post(w.producers[6], ts("2026-08-11"), op="delete", rkey="3mzzzzzzzzzz2")

    # block burst: 12 blocks within 20 s, plus ordinary blocks
    for k in range(12):
        t = ts("2026-07-01", k)
        commit(w.burst_blocker, "app.bsky.graph.block", tid(t), subject=w.producers[k % 20 + 10], created_at=iso(t))
    for u in w.users[100:160]:
        t = ts("2026-06-01", rnd.randint(0, 9_000_000))
        commit(u, "app.bsky.graph.block", tid(t), subject=w.producers[1], created_at=iso(t))

    # lists: curate list (owner in any half), mod list + list-block
    for owner in w.users[60:90]:
        lrk = tid(ts("2026-04-01"))
        luri = f"at://{owner}/app.bsky.graph.list/{lrk}"
        commit(owner, "app.bsky.graph.list", lrk, name="synthetic list", extra="app.bsky.graph.defs#curatelist")
        for p in rnd.sample(w.producers, 6):
            commit(owner, "app.bsky.graph.listitem", tid(ts("2026-04-02", rnd.randint(0, 9999))), subject=p, list_uri=luri)
    mod_owner = w.users[95]
    mrk = tid(ts("2026-03-01"))
    muri = f"at://{mod_owner}/app.bsky.graph.list/{mrk}"
    commit(mod_owner, "app.bsky.graph.list", mrk, name="synthetic modlist", extra="app.bsky.graph.defs#modlist")
    commit(mod_owner, "app.bsky.graph.listitem", tid(ts("2026-03-02")), subject=w.producers[2], list_uri=muri)
    for u in w.users[160:215]:
        commit(u, "app.bsky.graph.listblock", tid(ts("2026-05-05", rnd.randint(0, 99999))), subject=muri)

    # feed generator likes
    feed = f"at://{w.producers[0]}/app.bsky.feed.generator/synthfeed"
    for u in w.users[:30]:
        commit(u, "app.bsky.feed.like", tid(ts("2026-09-01", rnd.randint(0, 9999))), subject=feed)

    # accounts: the purged user is deactivated; the bulk_no user and purged user also act
    for x in (w.purged, w.bulk_no):
        commit(x, "app.bsky.feed.like", tid(ts("2026-08-01")), subject=posts_by[w.p_main][0][0])
        commit(x, "app.bsky.graph.follow", tid(ts("2026-08-01", 5)), subject=w.p_main)
        post(x, ts("2026-08-02"), text="synthetic post from an excluded account")
    ev.append({**base(w.purged), "kind": "account", "collection": "", "rkey": "", "rev": "", "operation": "",
               "created_at": "", "subject": "", "list_uri": "", "name": "", "description": "", "extra": "",
               "account_active": False, "account_status": "deactivated"})

    # Oct 20 likes for the positive control; snapshot of some likers on Oct 20 (early)
    for k, u in enumerate(w.users[200:240]):
        uri, _ = posts_by[w.producers[k % 10]][-1]
        commit(u, "app.bsky.feed.like", tid(ts("2026-10-20", 3600 + k)), subject=uri,
               t_ing=ts("2026-10-20", 7200) if k % 3 == 0 else None)

    # O13 edge cases: users[254] is in the M1 hash sample (cityHash64(did) % 100 == 13)
    u13, p13 = w.o13_user, w.o13_prod
    pb = {d: [x for x in posts_by[p13] if dt.datetime.fromtimestamp(x[1], dt.timezone.utc).strftime("%Y-%m-%d") == d]
          for d in ("2026-03-10", "2026-05-20", "2026-06-15", "2026-09-01", "2026-10-10")}
    for (uri, _), when in ((pb["2026-03-10"][0], ts("2026-05-17", 86399)), (pb["2026-03-10"][1], ts("2026-05-18", 1)),
                           (pb["2026-05-20"][0], ts("2026-07-01", 36000)), (pb["2026-05-20"][1], ts("2026-07-01", 54000)),
                           (pb["2026-10-10"][0], ts("2026-10-12", 86399)), (pb["2026-10-10"][1], ts("2026-10-13", 1))):
        commit(u13, "app.bsky.feed.like", tid(when), subject=uri, created_at=iso(when))
    commit(u13, "app.bsky.feed.repost", tid(ts("2026-07-01", 43200)), subject=pb["2026-06-15"][0][0], created_at=iso(ts("2026-07-01", 43200)))
    r13 = post(u13, pb["2026-09-01"][0][1] + 600, text="synthetic o13 reply", reply=pb["2026-09-01"][0][0])
    commit(p13, "app.bsky.feed.like", tid(pb["2026-09-01"][0][1] + 4200), subject=r13, created_at=iso(pb["2026-09-01"][0][1] + 4200))

    rnd.shuffle(ev)
    rnd.shuffle(po)
    shards = 3
    parts = [ev[s::shards] for s in range(shards)]
    home = next(s for s in range(shards) if any(r is dup for r in parts[s]))
    parts[(home + 1) % shards].append(dup2)  # cross-shard duplicate of one like record
    for s in range(shards):
        d = stage / f"v2-test-{s}" / "events"
        d.mkdir(parents=True, exist_ok=True)
        part = parts[s]
        pq.write_table(pa.Table.from_pylist(part, schema=EV_SCHEMA), d / f"events-0000{s}-0000{s}-00-{s:064x}.parquet")
        pq.write_table(pa.Table.from_pylist(po[s::shards], schema=POST_SCHEMA), d / f"posts-0000{s}-0000{s}-00-{s:064x}.parquet")
    # manifest in the GCS layout
    lines = ["path\tbytes\trows"]
    for f in sorted(stage.rglob("*.parquet")):
        top = "posts" if f.name.startswith("posts-") else "all"
        lines.append(f"/{top}/{f.relative_to(stage)}\t{f.stat().st_size}\t{pq.ParquetFile(f).metadata.num_rows}")
    (stage / "manifest.tsv").write_text("\n".join(lines) + "\n")

    side.mkdir(parents=True, exist_ok=True)
    fol_count = {p: 0 for p in w.producers}
    for u, ps in follows.items():
        for p in ps:
            fol_count[p] += 1
    order = sorted(w.producers, key=lambda p: -fol_count[p])
    order.remove(w.p_main)
    order.insert(0, w.p_main)
    (side / "producers.tsv").write_text("".join(f"{i}\t{p}\t{fol_count[p]}\n" for i, p in enumerate(order)))
    (side / "hide.tsv").write_text(w.hide_prod + "\n")
    (side / "intents.tsv").write_text("did\tbulkDataset\tprotocolBridging\tpublicAccessArchive\tsyntheticContentGeneration\n"
                                      f"{w.bulk_no}\tfalse\ttrue\ttrue\ttrue\n{w.synth_prod}\ttrue\ttrue\ttrue\tfalse\n")
    (side / "m2_cohort.tsv").write_text("".join(d + "\n" for d in w.m2_users))
    (side / "comm_cand1.tsv").write_text("".join(f"{i}\t{i % 4}\n" for i in range(len(order))))
    (side / "comm_cand5.tsv").write_text("".join(f"{i}\t{i % 3}\n" for i in range(len(order))))
    w.pid = {p: i for i, p in enumerate(order)}
    return w
