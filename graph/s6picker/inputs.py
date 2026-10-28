"""Local (offline) inputs for the S6 baseline picker, with the two slow derivations cached.

All paths are frozen partition / S1 artifacts under ~/catbird-lake-local. `set_partition` picks the
partition (cand1, the old working one, or cand5, frozen 2026-10-28; default cand1). Nothing here touches
the network. The cache (OUT/cache) is keyed by the input files' sha256, so a changed input rebuilds it.
"""
import collections
import hashlib
import json
import pathlib
import pickle
import re

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

LAKE = pathlib.Path.home() / "catbird-lake-local"
CAND1 = LAKE / "communities-cand1-hubdemote"
CAND5 = LAKE / "s3/matched/cand5-full-matched"
# Per partition: graph dir, its Claude community labels (None until labelled; community ids don't transfer),
# whether S5's card_packs.tsv (keyed by that partition's community ids) applies, and the profile snapshot.
PARTITIONS = {
    "cand1": {"dir": CAND1, "claude_community": LAKE / "claude-labels-20261027/community_labels_v2.parquet",
              "s5_card_packs": True, "profile_snapshot": LAKE / "claude-labels-20261027/top_profiles_all.parquet"},
    "cand5": {"dir": CAND5, "claude_community": None, "s5_card_packs": False,
              "profile_snapshot": LAKE / "claude-labels-cand5-20261028/top_profiles_all.parquet"},
}
PARTITION = {"name": "cand1", "s5_card_packs": True}
PATHS = {
    "communities": CAND1 / "communities.parquet",
    "known_for": CAND1 / "known_for.parquet",
    "membership": CAND1 / "membership.npy",
    "flags": CAND1 / "community_flags.npz",
    "producer_quality": CAND1 / "audit-quality/producer_quality.parquet",
    "list_names_all": CAND1 / "audit-quality/community_list_names.tsv",
    "packs": CAND1 / "audit-mass-followers/packs.pkl",
    "lists_A": LAKE / "s3/lists/lists_A.tsv",
    "lists_split_receipt": LAKE / "s3/lists/split_receipt.json",
    "claude_community": LAKE / "claude-labels-20261027/community_labels_v2.parquet",
    "claude_profile": LAKE / "claude-labels-20261027/profile_labels.parquet",
    "profile_snapshot": LAKE / "claude-labels-20261027/top_profiles_all.parquet",
    "hide_from_recs_union": LAKE / "s5-declarations-20261027/hide_from_recs_union.tsv",
    "user_intents_optout": LAKE / "s5-declarations-20261027/user_intents_optout_dids.tsv",
    "preregistration": pathlib.Path.home() / "Developer/Catbird+Petrel/docs/execution-plans/2026-10-28-whole-network-preregistration.md",
}
FEEDGEN_DAYS = sorted((LAKE / "netcup-export").glob("2026-10-*/feedgens.parquet"))
PACK_DAYS = sorted((LAKE / "netcup-export").glob("2026-10-*/starterpacks.parquet"))


def set_partition(name, claude_community=None):
    """Point PATHS at one partition's artifacts. claude_community overrides that partition's label file."""
    cfg = PARTITIONS[name]
    d = cfg["dir"]
    PATHS.update({
        "communities": d / "communities.parquet",
        "known_for": d / "known_for.parquet",
        "membership": d / "membership.npy",
        "flags": d / "community_flags.npz",
        "producer_quality": d / "audit-quality/producer_quality.parquet",
        "list_names_all": d / "audit-quality/community_list_names.tsv",
        "packs": d / "audit-mass-followers/packs.pkl",
        "profile_snapshot": cfg["profile_snapshot"],
    })
    label = pathlib.Path(claude_community).expanduser() if claude_community else cfg["claude_community"]
    if label is None:
        PATHS.pop("claude_community", None)
    else:
        PATHS["claude_community"] = label
    PARTITION.update(name=name, s5_card_packs=cfg["s5_card_packs"])
    return PATHS


def sha256(path, _memo={}):
    path = pathlib.Path(path).resolve()
    if path not in _memo:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        _memo[path] = h.hexdigest()
    return _memo[path]


def load_graph():
    """The partition's communities, rank-0 homes, and follow-back/follow-train flags mapped to community ids.

    community_flags.npz is keyed by raw Leiden label (membership.npy). Each community maps to
    the raw label held by the majority of its top_dids (DECISIONS: the same mapping eval_temporal uses).
    """
    comm = pq.read_table(PATHS["communities"], columns=["community", "scene_id", "size", "top_dids"]).to_pylist()
    kf = pq.read_table(PATHS["known_for"], columns=["pid", "did", "community", "rank"])
    kf0 = kf.filter(pc.equal(kf["rank"], 0))
    pid0 = np.asarray(kf0["pid"]).astype(np.int64)
    com0 = np.asarray(kf0["community"]).astype(np.int64)
    did0 = kf0["did"].to_pylist()
    memb = np.load(PATHS["membership"]).astype(np.int64)
    fl = np.load(PATHS["flags"])
    raw_flag = fl["follow_back"] | fl["follow_train"]
    did2pid = dict(zip(did0, pid0.tolist()))
    flagged_c = {}
    for r in comm:
        raws = [memb[did2pid[d]] for d in r["top_dids"] if d in did2pid]
        raw = collections.Counter(raws).most_common(1)[0][0] if raws else -1
        flagged_c[r["community"]] = {
            "raw": int(raw),
            "follow_back": bool(raw >= 0 and fl["follow_back"][raw]),
            "follow_train": bool(raw >= 0 and fl["follow_train"][raw]),
        }
    producer_flagged = raw_flag[memb]  # by pid
    return comm, pid0, com0, did0, producer_flagged, flagged_c


def load_quality():
    t = pq.read_table(PATHS["producer_quality"], columns=["pid", "followers", "light_engagers"])
    n = int(np.asarray(t["pid"]).max()) + 1
    followers = np.zeros(n, np.int64); light = np.zeros(n, np.int64)
    pid = np.asarray(t["pid"])
    followers[pid] = np.asarray(t["followers"]); light[pid] = np.asarray(t["light_engagers"])
    return followers, light


def norm_name(name):
    return re.sub(r"\s+", " ", name.strip().lower())[:60]


def a_lists(cache, home):
    """A-half curated lists (S1 §6 split): per-DID distinct A owners, and per-community top list names.

    Names follow list_purity.py: >=3 producer members, identical member sets deduped, <=3 lists per
    owner (default_rng(7)), a list counts for community c when >=3 of its members have c as home.
    """
    key = sha256(PATHS["lists_A"])[:16] + "_" + sha256(PATHS["known_for"])[:12]
    f = pathlib.Path(cache) / f"a_lists_{key}.json"
    if f.exists():
        d = json.loads(f.read_text())
        return d["owners"], {int(k): v for k, v in d["names"].items()}
    owners = collections.defaultdict(set)
    rows = []
    with open(PATHS["lists_A"]) as fp:
        for line in fp:
            uri, owner, name, _total, members = line.rstrip("\n").split("\t")
            ms = sorted(set(m for m in members.split(",") if m))
            for m in ms:
                owners[m].add(owner)
            per_c = collections.Counter(home[m] for m in ms if m in home)
            rows.append((uri, owner, name, hashlib.blake2b("\n".join(ms).encode(), digest_size=12).digest(), len(ms),
                         {c: n for c, n in per_c.items() if n >= 3}))
    rows.sort(key=lambda r: r[0])
    seen, by_owner = set(), collections.defaultdict(list)
    for r in rows:
        if r[4] >= 3 and r[3] not in seen:
            seen.add(r[3]); by_owner[r[1]].append(r)
    rng = np.random.default_rng(7)
    names = collections.defaultdict(collections.Counter)
    for owner, rs in by_owner.items():
        for i in sorted(rng.choice(len(rs), min(3, len(rs)), replace=False)):
            for c in rs[i][5]:
                names[c][norm_name(rs[i][2])] += 1
    own = {d: len(s) for d, s in owners.items()}
    top = {c: cnt.most_common(8) for c, cnt in names.items()}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"owners": own, "names": {str(c): v for c, v in top.items()}}))
    return own, top


def all_list_names():
    """community_list_names.tsv (all list halves): the names S1's keyword screen also reads."""
    out = {}
    for line in open(PATHS["list_names_all"]):
        parts = line.rstrip("\n").split("\t")
        if parts[0] == "community":
            continue
        out[int(parts[0])] = [re.sub(r" \(\d+\)$", "", n) for n in parts[6].split("; ") if n]
    return out


def s5_card_packs(s5_dir):
    """S5's offer set: universe/card_packs.tsv (community, pack uri, share of members in the card, members).

    It is keyed by the community ids of the partition S5 swept (cand1), so it is skipped for another one."""
    if not PARTITION["s5_card_packs"]:
        return None, None
    root = pathlib.Path(s5_dir).expanduser()
    f = root / "card_packs.tsv" if (root / "card_packs.tsv").exists() else root / "work/universe/card_packs.tsv"
    if not f.exists():
        return None, None
    out = collections.defaultdict(list)
    for line in open(f):
        c, uri, share, n = line.rstrip("\n").split("\t")[:4]
        out[int(c)].append({"uri": uri, "share": float(share), "n": int(n)})
    return out, sha256(f)


def packs_by_community(cache, pid_home, offers=None):
    """Candidate starter packs per card, with full member lists from packs.pkl.

    With S5's offer set (s5_card_packs), a card offers exactly the packs S5 swept, ordered by share.
    Without it (dry run), a pack is offered by the community holding >=5 and >=50% of its producer members.
    """
    want = sorted({p["uri"] for v in offers.values() for p in v}) if offers else None
    key = sha256(PATHS["packs"])[:16] + "_" + sha256(PATHS["known_for"])[:12] + ("_" + hashlib.sha256("\n".join(want).encode()).hexdigest()[:12] if want else "")
    f = pathlib.Path(cache) / f"packs_{key}.json"
    if f.exists():
        return {int(k): v for k, v in json.loads(f.read_text()).items()}
    wanted = set(want or [])
    by_uri, out = {}, collections.defaultdict(list)
    for uri, members, pids in pickle.load(open(PATHS["packs"], "rb")):  # written by clean_pack_recall.py dump here
        if offers:
            if uri in wanted:
                by_uri[uri] = sorted(set(members))
            continue
        homes = [int(pid_home[p]) for p in pids if 0 <= p < len(pid_home) and pid_home[p] >= 0]
        if len(homes) < 5:
            continue
        c, n = collections.Counter(homes).most_common(1)[0]
        if n >= 5 and n / len(homes) >= 0.5:
            out[c].append({"uri": uri, "members": sorted(set(members)), "in_card": n, "share": n / len(homes)})
    if offers:
        for c, v in offers.items():
            for p in sorted(v, key=lambda p: (-p["share"], p["uri"])):
                if p["uri"] in by_uri:
                    out[c].append({"uri": p["uri"], "members": by_uri[p["uri"]], "share": p["share"],
                                   "in_card": round(p["share"] * p["n"])})
    top = {c: sorted(v, key=lambda p: (-p["share"], p["uri"]))[:5] for c, v in out.items()}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({str(c): v for c, v in top.items()}))
    return top


PACK_ARCHIVE_DIRS = [LAKE / "archive-graph/events", LAKE / "gce-archive-graph/events",
                     *sorted((LAKE / "gce-archive-graph-gcs").glob("*/events"))]
PACK_SUPPLEMENT_GLOB = (LAKE / "supplement-events", "**/*.parquet")  # repo-supplement events, same schema


_TID = "234567abcdefghijklmnopqrstuvwxyz"


def _tid_int(rev):
    """A repo rev TID (13 base32-sortable chars) as the integer the netcup exports store."""
    return sum(_TID.index(c) << (5 * (12 - i)) for i, c in enumerate(rev)) if isinstance(rev, str) and len(rev) == 13 else 0


def _pack_names_in(path):
    """Latest (rev as int, seq) starterpack record per (did, rkey) in one archive events file: (rev, seq, deleted, name)."""
    t = pq.read_table(path, columns=["did", "rkey", "rev", "seq", "operation", "name"],
                      filters=[("collection", "=", "app.bsky.graph.starterpack")])
    out = {}
    for did, rkey, rev, seq, op, name in zip(*(t[c].to_pylist() for c in ("did", "rkey", "rev", "seq", "operation", "name"))):
        v = (_tid_int(rev), seq or 0, op == "delete", name or "")
        if (did, rkey) not in out or v[:2] > out[(did, rkey)][:2]:
            out[(did, rkey)] = v
    return out


def pack_names(cache=None):
    """Starter-pack display names (public record metadata, not a safety input), latest-rev-wins.

    The archive events extract and the repo-supplement events (the record sets packs.pkl was dumped from;
    their `name` column is the record's name) supply the history; the netcup daily exports are layered on top. A deleted pack has no name.
    The archive scan is cached in CACHE, keyed by the event files' paths and sizes."""
    import multiprocessing as mp
    files = sorted([str(f) for d in PACK_ARCHIVE_DIRS for f in d.glob("*.parquet")] +
                   [str(f) for f in PACK_SUPPLEMENT_GLOB[0].glob(PACK_SUPPLEMENT_GLOB[1])])
    key = hashlib.sha256("\n".join(f"{f}\t{pathlib.Path(f).stat().st_size}" for f in files).encode()).hexdigest()[:16]
    cf = pathlib.Path(cache) / f"pack_names_archive_v2_{key}.json" if cache else None
    if cf and cf.exists():
        latest = {tuple(k.split(" ", 1)): tuple(v) for k, v in json.loads(cf.read_text()).items()}
    else:
        latest = {}
        with mp.Pool(14) as pool:
            for part in pool.imap_unordered(_pack_names_in, files, chunksize=16):
                for k, v in part.items():
                    if k not in latest or v[:2] > latest[k][:2]:
                        latest[k] = v
        if cf:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps({f"{d} {r}": v for (d, r), v in latest.items()}, ensure_ascii=False))
    for p in PACK_DAYS:
        for r in pq.read_table(p, columns=["did", "rkey", "rev", "is_deleted", "name", "seq"]).to_pylist():
            v = (int(r["rev"] or 0), r["seq"] or 0, bool(r["is_deleted"]), r["name"] or "")
            k = (r["did"], r["rkey"])
            if k not in latest or v[:2] > tuple(latest[k][:2]):
                latest[k] = v
    return {f"at://{d}/app.bsky.graph.starterpack/{r}": " ".join(v[3].split())[:80]
            for (d, r), v in latest.items() if not v[2] and v[3]}


def feedgens():
    latest = {}
    for p in FEEDGEN_DAYS:
        for r in pq.read_table(p, columns=["did", "rkey", "is_deleted", "seq", "display_name", "description"]).to_pylist():
            k = (r["did"], r["rkey"])
            if k not in latest or r["seq"] > latest[k]["seq"]:
                latest[k] = r
    return [r for r in latest.values() if not r["is_deleted"]]


def claude_labels():
    """Community labels for this partition (none if it has no label file yet) and per-DID profile classes."""
    comm = ({r["community"]: r for r in pq.read_table(PATHS["claude_community"]).to_pylist()}
            if "claude_community" in PATHS else {})
    prof = {}
    for r in pq.read_table(PATHS["claude_profile"], columns=["did", "haiku_sensitivity", "sonnet_sensitivity"]).to_pylist():
        s = set(r["haiku_sensitivity"] or []) | set(r["sonnet_sensitivity"] or [])
        if s:
            prof[r["did"]] = s
    return comm, prof


def did_set(path):
    return {l.split("\t")[0].strip() for l in open(path) if l.strip() and l.startswith("did:")}


def profile_snapshot():
    """Display names and bios from the Oct 27 local profile-record snapshot (S5 (i) keeps neither)."""
    t = pq.read_table(PATHS["profile_snapshot"], columns=["did", "name", "description"]).to_pylist()
    return {r["did"]: (r["name"] or "", r["description"] or "") for r in t}


def s5_extra(s5_dir):
    """S5's own derived safety outputs, unioned in as extra exclusions (defence in depth) when present:
    eligibility_rule2_exclude_dids.txt (S5's rule (2) list), status.tsv (relay listRepos: DIDs with
    active=false, i.e. purge-eligible), and card_packs_filter.tsv (S5's keep/drop per offered pack)."""
    root = pathlib.Path(s5_dir).expanduser()
    out = {"rule2": set(), "inactive": set(), "pack_drop": set(), "files": {}}
    f = root / "eligibility_rule2_exclude_dids.txt"
    if f.exists():
        out["rule2"] = did_set(f); out["files"][f.name] = sha256(f)
    f = root / "status.tsv"
    if f.exists():
        with open(f) as fp:
            for line in fp:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[1] == "false":
                    out["inactive"].add(parts[0])
        out["files"][f.name] = sha256(f)
    f = root / "card_packs_filter.tsv"
    if f.exists():
        for line in open(f):
            parts = line.rstrip("\n").split("\t")
            if parts[-1] != "keep":
                out["pack_drop"].add(parts[1])
        out["files"][f.name] = sha256(f)
    return out
