"""Per-producer quality features that do not come from the follow graph.

  producer_features.py BUILD AUDIT_DIR NETCUP_ROOT EVENTS_GLOB OUT_DIR

BUILD holds producers.tsv and follower_weights.tsv; AUDIT_DIR holds follower_degree.npz.
NETCUP_ROOT days 2026-10-19..22 give likes, reposts and posts (the only local engagement).
EVENTS_GLOB is the archive graph events (same files as the eval manifest) for blocks and lists.
A record's latest rev wins and deletes are honoured, as in pack_counts.py.

Writes OUT_DIR/producer_features.npz (aligned to producer id) and curatelists.tsv for list_purity.py.
"""
import subprocess, sys, time
import numpy as np
import pyarrow.csv as pcsv

BUILD, AUDIT, NETCUP, EVENTS, OUT = sys.argv[1:6]
DAYS = "{2026-10-19,2026-10-20,2026-10-21,2026-10-22}"
MASS = 3000
t0 = time.time()
SETTINGS = ("SET max_memory_usage = 40000000000; SET max_bytes_before_external_group_by = 12000000000; "
            "SET max_bytes_before_external_sort = 12000000000; SET max_threads = 8;")


def ch(sql, out):
    subprocess.run(["clickhouse", "local", "--multiquery", "-q", f"{SETTINGS} {sql} INTO OUTFILE '{out}' TRUNCATE FORMAT TSV"],
                   check=True)
    print(f"  wrote {out} {time.time()-t0:.0f}s", flush=True)


ro = pcsv.ReadOptions(column_names=None, autogenerate_column_names=True)
po = pcsv.ParseOptions(delimiter="\t")
fw = pcsv.read_csv(f"{BUILD}/follower_weights.tsv", read_options=ro, parse_options=po,
                   convert_options=pcsv.ConvertOptions(include_columns=["f0", "f1"]))
deg = np.load(f"{AUDIT}/follower_degree.npz")["deg"]
uid = fw.column(0).to_numpy()
heavy = np.flatnonzero(deg[uid] >= MASS)
with open(f"{OUT}/mass_followers.tsv", "w") as f:
    for d in fw.column(1).take(heavy).to_pylist():
        f.write(d + "\n")
print(f"mass followers (>= {MASS} producers) = {len(heavy):,} {time.time()-t0:.0f}s", flush=True)

PROD = f"file('{BUILD}/producers.tsv', TSV, 'pid UInt32, did String, followers UInt64')"
HEAVY = f"file('{OUT}/mass_followers.tsv', TSV, 'did String')"
ENG = f"""(SELECT did, t.2 AS author FROM (
            SELECT did, kind, rkey, argMax(tuple(is_deleted, subject_did), rev) AS t
            FROM file('{NETCUP}/{DAYS}/engagements.parquet', Parquet,
                      'did String, kind String, rkey String, rev UInt64, is_deleted UInt8, subject_did String')
            WHERE kind IN ('like', 'repost') GROUP BY did, kind, rkey)
          WHERE t.1 = 0 AND t.2 != '' AND did != t.2)"""

# Distinct engagers, light engagers, and like-back: the author also engaged with that engager.
ch(f"""WITH prod AS (SELECT did FROM {PROD}),
       a AS (SELECT DISTINCT did AS engager, author FROM {ENG} WHERE author IN (SELECT did FROM prod)),
       b AS (SELECT DISTINCT did AS b_author, author AS b_engager FROM {ENG} WHERE did IN (SELECT did FROM prod))
   SELECT a.author, count(), countIf(a.engager NOT IN (SELECT did FROM {HEAVY})), countIf(b.b_author != '')
   FROM a LEFT JOIN b ON a.author = b.b_author AND a.engager = b.b_engager
   GROUP BY a.author""", f"{OUT}/engagers.tsv")

ch(f"""SELECT did, count(), countIf(t.2 = '') FROM (
         SELECT did, rkey, argMax(tuple(is_deleted, reply_parent), rev) AS t
         FROM file('{NETCUP}/{DAYS}/posts.parquet', Parquet, 'did String, rkey String, rev UInt64, is_deleted UInt8, reply_parent String')
         WHERE did IN (SELECT did FROM {PROD}) GROUP BY did, rkey)
       WHERE t.1 = 0 GROUP BY did""", f"{OUT}/posts.tsv")

EV = f"file('{EVENTS}', Parquet, 'collection String, did String, rkey String, rev String, operation String, subject String, list_uri String, name String, extra String')"
ch(f"""SELECT t.2, uniqExact(did) FROM (
         SELECT did, rkey, argMax(tuple(operation = 'delete', subject), tuple(rev, operation = 'delete')) AS t
         FROM {EV} WHERE collection = 'app.bsky.graph.block' GROUP BY did, rkey)
       WHERE NOT t.1 AND t.2 IN (SELECT did FROM {PROD}) GROUP BY t.2""", f"{OUT}/blockers.tsv")

# Lists: latest list record gives purpose and name; latest listitem gives membership.
LISTS = f"""(SELECT concat('at://', did, '/app.bsky.graph.list/', rkey) AS uri, did AS owner, t.2 AS purpose, t.3 AS name FROM (
               SELECT did, rkey, argMax(tuple(operation = 'delete', extra, name), tuple(rev, operation = 'delete')) AS t
               FROM {EV} WHERE collection = 'app.bsky.graph.list' GROUP BY did, rkey) WHERE NOT t.1)"""
ITEMS = f"""(SELECT DISTINCT t.2 AS list_uri, t.3 AS subject FROM (
               SELECT did, rkey, argMax(tuple(operation = 'delete', list_uri, subject), tuple(rev, operation = 'delete')) AS t
               FROM {EV} WHERE collection = 'app.bsky.graph.listitem' GROUP BY did, rkey)
             WHERE NOT t.1 AND startsWith(t.3, 'did:'))"""
ch(f"""CREATE TEMPORARY TABLE l AS SELECT * FROM {LISTS};
       CREATE TEMPORARY TABLE i AS SELECT * FROM {ITEMS} WHERE list_uri IN (SELECT uri FROM l);
       SELECT i.subject, countIf(l.purpose = 'app.bsky.graph.defs#modlist'), countIf(l.purpose = 'app.bsky.graph.defs#curatelist')
       FROM i JOIN l ON i.list_uri = l.uri
       WHERE i.subject IN (SELECT did FROM {PROD}) AND i.subject != l.owner GROUP BY i.subject""", f"{OUT}/lists_per_producer.tsv")
ch(f"""CREATE TEMPORARY TABLE l AS SELECT * FROM {LISTS} WHERE purpose = 'app.bsky.graph.defs#curatelist';
       CREATE TEMPORARY TABLE i AS SELECT * FROM {ITEMS} WHERE list_uri IN (SELECT uri FROM l);
       SELECT l.uri, l.owner, replaceRegexpAll(l.name, '[\\t\\n\\r]', ' '), count(),
              arrayStringConcat(groupArrayIf(i.subject, i.subject IN (SELECT did FROM {PROD})), ',')
       FROM i JOIN l ON i.list_uri = l.uri GROUP BY l.uri, l.owner, l.name
       HAVING count() BETWEEN 5 AND 500""", f"{OUT}/curatelists.tsv")

dids = [l.split("\t")[1] for l in open(f"{BUILD}/producers.tsv")]
pid = {d: i for i, d in enumerate(dids)}
P = len(dids)


def load(name, ncols):
    cols = [np.zeros(P, np.int64) for _ in range(ncols)]
    for line in open(f"{OUT}/{name}"):
        parts = line.rstrip("\n").split("\t")
        i = pid.get(parts[0])
        if i is not None:
            for c in range(ncols):
                cols[c][i] = int(parts[c + 1])
    return cols


engagers, light_engagers, likeback = load("engagers.tsv", 3)
posts, originals = load("posts.tsv", 2)
(blockers,) = load("blockers.tsv", 1)
modlists, curatelists = load("lists_per_producer.tsv", 2)
np.savez(f"{OUT}/producer_features.npz", engagers=engagers, light_engagers=light_engagers, likeback=likeback,
         posts=posts, originals=originals, blockers=blockers, modlists=modlists, curatelists=curatelists)
for k, v in (("engagers", engagers), ("light_engagers", light_engagers), ("posts", posts), ("originals", originals),
             ("blockers", blockers), ("modlists", modlists), ("curatelists", curatelists)):
    print(f"{k:<15} nonzero={(v>0).mean():.1%} median={np.median(v):.0f} p90={np.percentile(v,90):.0f}")
print(f"done {time.time()-t0:.0f}s")
