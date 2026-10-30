#!/usr/bin/env python3
"""S5 deliverables from the raw sweeps (run after status.tsv exists).

  profiles.parquet                    one row per getProfiles DID: found, counts, labels by source,
                                      adult_graphic / hard_label / no_unauthenticated / hide_from_recs,
                                      repo_status (from status.tsv; 'active' if not in it)
  eligibility_rule2_exclude_dids.txt  universe DIDs failing eligibility rule (2) other than
                                      hide-from-recs (A1): missing from getProfiles, adult/graphic or
                                      hard-labelled, !no-unauthenticated, or inactive/absent repo
  card_packs_filter.tsv               community, pack URI, share, size, #bad members, keep|drop
  supplement_activity.parquet         (vi) live postsCount + getAuthorFeed originals vs cutoff T

usage: summarize.py OUT_DIR   (OUT_DIR/work holds the raw sweep files)
"""
import collections, datetime as dt, json, os, pickle, sys

import pyarrow as pa
import pyarrow.parquet as pq

O = sys.argv[1].rstrip("/") + "/"
W = O + "work/"
LAKE = os.path.expanduser("~/catbird-lake-local/")
MOD = "did:plc:ar7c4by46qjdydhdevvrndac"
ADULT = {"porn", "sexual", "nudity", "sexual-figurative", "graphic-media", "gore", "corpse", "self-harm"}
HARD = {"!hide", "!takedown", "!unspecced-takedown", "!warn"}

reasons = dict(l.rstrip("\n").split("\t") for l in open(O + "serving_universe.tsv"))
hide = {l.strip() for l in open(LAKE + "s5-declarations-20261027/hide_from_recs_union.tsv") if l.startswith("did:")}
status = {}
for l in open(O + "status.tsv"):
    d, _a, s = l.rstrip("\n").split("\t")
    status[d] = s or "inactive"
rows = {}
for l in open(W + "profiles/profiles.jsonl"):
    for x in json.loads(l).get("rows", []):
        rows[x["did"]] = x
cols = collections.defaultdict(list)
for d in sorted(rows):
    x = rows[d]
    labs = [l for l in x.get("labels", []) if not l["neg"]]
    modv = sorted({l["val"] for l in labs if l["src"] == MOD})
    selfv = sorted({l["val"] for l in labs if l["src"] == d})
    allv = set(modv) | set(selfv)
    cols["did"].append(d); cols["found"].append(bool(x["found"])); cols["handle"].append(x.get("handle"))
    for k in ("postsCount", "followersCount", "followsCount"):
        cols[k].append(x.get(k))
    cols["createdAt"].append(x.get("createdAt")); cols["mod_labels"].append(modv); cols["self_labels"].append(selfv)
    cols["other_labels"].append(sorted({l["val"] for l in labs if l["src"] not in (MOD, d)}))
    cols["adult_graphic"].append(bool(allv & ADULT)); cols["hard_label"].append(bool(allv & HARD))
    cols["any_mod_label"].append(bool(modv)); cols["no_unauthenticated"].append("!no-unauthenticated" in allv)
    cols["hide_from_recs"].append(d in hide); cols["repo_status"].append(status.get(d, "active"))
    cols["in_universe"].append(d in reasons); cols["universe_reasons"].append(reasons.get(d, "control"))
t = pa.table(cols)
pq.write_table(t, O + "profiles.parquet.tmp", compression="zstd"); os.replace(O + "profiles.parquet.tmp", O + "profiles.parquet")
u = t.to_pandas(); u = u[u.in_universe]
excl = (~u.found) | u.adult_graphic | u.hard_label | u.no_unauthenticated | (u.repo_status != "active")
with open(O + "eligibility_rule2_exclude_dids.txt.tmp", "w") as f:
    for d in sorted(u.did[excl]):
        f.write(d + "\n")
os.replace(O + "eligibility_rule2_exclude_dids.txt.tmp", O + "eligibility_rule2_exclude_dids.txt")
# packs.pkl is our own audit artifact (communities-cand1-hubdemote), not untrusted input.
packs = {a: m for a, m, _ in pickle.load(open(LAKE + "communities-cand1-hubdemote/audit-mass-followers/packs.pkl", "rb"))}
bad = set(u.did[excl])
with open(O + "card_packs_filter.tsv", "w") as f:
    for c, uri, s, n in (l.rstrip("\n").split("\t") for l in open(O + "card_packs.tsv")):
        nb = sum(m in bad for m in packs[uri])
        f.write(f"{c}\t{uri}\t{s}\t{n}\t{nb}\t{'drop' if nb else 'keep'}\n")

# (vi) supplement producers: originals in the getAuthorFeed sample relative to T = 2026-08-15
prof = {r["did"]: r for r in pq.read_table(O + "profiles.parquet", columns=["did", "found", "postsCount"]).to_pylist()}
inprod = {l.split("\t")[0]: l.rstrip("\n").split("\t")[2] == "1" for l in open(W + "authorfeed/supplement_in_universe.tsv")}


def p(s):
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


T = dt.datetime(2026, 8, 16, tzinfo=dt.timezone.utc)   # pre-T = before Aug 16 00:00Z (S1 R1)
Y, D30 = T - dt.timedelta(days=365), T - dt.timedelta(days=30)
cols = collections.defaultdict(list)
for l in open(W + "authorfeed/authorfeed.jsonl"):
    r = json.loads(l); d = r["did"]; ok = "error" not in r
    ts = [x for x in (p(s) for s in r.get("original_created", [])) if x]
    oldest = min(ts) if ts else None
    cols["did"].append(d); cols["is_producer"].append(inprod.get(d)); cols["feed_ok"].append(ok)
    cols["postsCount"].append((prof.get(d) or {}).get("postsCount")); cols["profile_found"].append((prof.get(d) or {}).get("found"))
    cols["feed_items"].append(r.get("items")); cols["originals_sampled"].append(r.get("originals"))
    cols["oldest_original"].append(oldest.isoformat() if oldest else None)
    cols["newest_original"].append(max(ts).isoformat() if ts else None)
    cols["originals_12m_before_T"].append(sum(Y < x < T for x in ts))
    cols["originals_30d_before_T"].append(sum(D30 <= x < T for x in ts))
    # the sample covers (T-12m, T) when the feed ended, or its oldest original predates T-12m
    cols["sample_reaches_T_minus_12m"].append(bool(ok and (r.get("cursor") is None or (oldest is not None and oldest <= Y))))
    cols["feed_exhausted"].append(ok and r.get("cursor") is None)
pq.write_table(pa.table(cols), O + "supplement_activity.parquet.tmp", compression="zstd")
os.replace(O + "supplement_activity.parquet.tmp", O + "supplement_activity.parquet")
print("ok")
