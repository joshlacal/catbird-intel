"""S4 pilot corpus: four-day producer originals, with every S1 exclusion applied before embedding.

  prep_corpus.py OUT_DIR

Whole-network plan S4 (docs/execution-plans/2026-10-25-whole-network-signals-plan.md §3.1) and the
S1 rules file (2026-10-28-whole-network-preregistration.md, amendment A1). Read-only on every input.

Producer set (plan S4 row, design-content §3): producers in producers.tsv with >=3 originals
(non-replies, quotes included; latest revision, not deleted) in the netcup days 2026-10-19..22.
Exclusions, recorded with counts:
  posts  - hide-from-recs union (S5 live, 59,053 DIDs), any visibility.parquet row with
           hide_from_algorithmic=1, user-intents opt-outs, likely-minors home community;
  bios   - user-intents opt-outs and likely-minors home community only (A1: hide-from-recs bios allowed).
Home community = known_for.parquet rank 0 (never membership.npy ids), flagged when the Opus
community label's `sensitivity` contains `likely_minors`.
Embedded posts: non-empty text, <=32 per producer chosen by a fixed hash order. The bake-off
subsample is the 500,000 embedded posts with the smallest salted hash (identical for both models).
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LAKE = pathlib.Path.home() / "catbird-lake-local"
DAYS = ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]
PRODUCERS = LAKE / "graph-build-v3/out-full-20261024/producers.tsv"
HIDE = LAKE / "s5-declarations-20261027/hide_from_recs_union.tsv"
INTENTS = LAKE / "s5-declarations-20261027/user_intents_optout_dids.tsv"
LABELS = LAKE / "claude-labels-20261027/community_labels.parquet"
KNOWN_FOR = LAKE / "claude-labels-20261027/graph-cand1-topfix/known_for.parquet"
MIN_ORIGINALS, CAP, BAKEOFF_N = 3, 32, 500_000
EXPECTED_SHA = {HIDE: "7d369fa5ced8ca5f22d8cf21b4b2fd0de549a8dcf887f4c92c0217eb0d51ca52",
                LABELS: "702e4b1ef48a812e19a81377cf29348436c8848b59f17818a2d1c8fdd45cef2f"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def hash64(salt, keys):
    """Deterministic 64-bit order key: first 8 bytes of blake2b(salt|key)."""
    return np.array([int.from_bytes(hashlib.blake2b(f"{salt}|{k}".encode(), digest_size=8).digest(), "big")
                     for k in keys], dtype=np.uint64)


def read_dids(path):
    return {l.strip().split("\t")[0] for l in open(path) if l.strip()}


def minors_dids():
    lab = pq.read_table(LABELS, columns=["community", "sensitivity"]).to_pandas()
    flagged = {int(c) for c, s in zip(lab.community, lab.sensitivity) if s is not None and "likely_minors" in list(s)}
    kf = pq.read_table(KNOWN_FOR, columns=["did", "community", "rank"]).to_pandas()
    home = kf[kf["rank"] == 0]
    return set(home.did[home.community.isin(flagged)]), len(flagged), int(home.did.nunique())


def main(out):
    out = pathlib.Path(out)
    receipt = {"inputs": {}, "counts": {}}
    for p, want in EXPECTED_SHA.items():
        got = sha256(p)
        assert got == want, f"{p}: sha256 {got} != registered {want}"
    for p in [PRODUCERS, HIDE, INTENTS, LABELS, KNOWN_FOR]:
        receipt["inputs"][str(p)] = sha256(p)

    producers = pd.read_csv(PRODUCERS, sep="\t", header=None, names=["pid", "did", "followers"],
                            dtype={"pid": np.int64, "did": str, "followers": np.int64})
    prod_set = pa.array(producers.did.tolist())

    posts, deletes, vis_hide = [], [], set()
    for day in DAYS:
        d = LAKE / "netcup-export" / day
        man = json.load(open(d / "manifest.json"))
        for f in man["files"]:
            if f["table"] in ("posts", "deletions", "visibility"):
                got = sha256(d / f["file"])
                assert got == f["sha256"], f"{d / f['file']}: sha256 mismatch with manifest"
                receipt["inputs"][str(d / f["file"])] = got
        t = pq.read_table(d / "posts.parquet", columns=["did", "rkey", "rev", "is_deleted", "text", "reply_parent"])
        t = t.filter(pc.is_in(t["did"], value_set=prod_set))
        posts.append(t)
        dl = pq.read_table(d / "deletions.parquet", columns=["collection", "did", "rkey"])
        dl = dl.filter(pc.equal(dl["collection"], "app.bsky.feed.post")).select(["did", "rkey"])
        deletes.append(dl)
        v = pq.read_table(d / "visibility.parquet", columns=["did", "hide_from_algorithmic"]).to_pandas()
        vis_hide |= set(v.did[v.hide_from_algorithmic == 1])
    df = pa.concat_tables(posts).to_pandas()
    rows_read = len(df)
    df = df.sort_values("rev").drop_duplicates(["did", "rkey"], keep="last")
    latest = len(df)
    df = df[df.is_deleted == 0]
    dels = pa.concat_tables(deletes).to_pandas().drop_duplicates()
    dkey = set(zip(dels.did, dels.rkey))
    gone = np.fromiter(((a, b) in dkey for a, b in zip(df.did, df.rkey)), bool, len(df))
    df = df[~gone]
    orig = df[df.reply_parent == ""].drop(columns=["reply_parent", "is_deleted"])
    per = orig.groupby("did").size()
    active = set(per.index[per >= MIN_ORIGINALS])
    c = receipt["counts"]
    c.update(producer_post_rows_read=rows_read, producer_posts_latest_rev=latest,
             producer_posts_live=int(len(df)), deleted_by_deletions_parquet=int(gone.sum()),
             producer_originals_live=int(len(orig)), producers_ge1_original=int(len(per)),
             producers_ge3_originals=len(active))

    hide = read_dids(HIDE)
    intents = read_dids(INTENTS)
    minors, n_minor_comms, n_home = minors_dids()
    receipt["minors_rule"] = {"likely_minors_communities": n_minor_comms, "producers_with_rank0_home": n_home,
                              "producers_in_likely_minors_home": len(minors)}
    post_excl = {"hide_from_recs_union": hide, "visibility_hide_from_algorithmic": vis_hide,
                 "user_intents_optout": intents, "likely_minors_home": minors}
    excl_counts, excluded = {}, set()
    for name, s in post_excl.items():
        hit = active & s
        excl_counts[name] = {"producers": len(hit), "originals": int(per[list(hit)].sum()) if hit else 0,
                             "producers_not_already_excluded": len(hit - excluded)}
        excluded |= hit
    kept = active - excluded
    excl_counts["any_post_exclusion"] = {"producers": len(excluded), "originals": int(per[list(excluded)].sum())}
    excl_counts["visibility_dids_total"] = len(vis_hide)
    excl_counts["visibility_dids_not_in_union"] = len(vis_hide - hide)
    receipt["post_exclusions_among_ge3_producers"] = excl_counts

    o = orig[orig.did.isin(kept)].copy()
    o["key"] = o.did + "/" + o.rkey
    o["order"] = hash64("s4-cap32", o.key)
    # Design-content definition (cap 32 over all originals, empty text included), for reproduction only.
    o = o.sort_values(["did", "order"])
    o["rank_all"] = o.groupby("did").cumcount()
    c["design_def_posts_cap32_incl_empty"] = int((o.rank_all < CAP).sum())
    c["producers_after_exclusions"] = len(kept)
    o["text"] = o.text.str.strip()
    nonempty = o[o.text != ""].copy()
    c["originals_empty_text_dropped"] = int((o.text == "").sum())
    nonempty["rank_ne"] = nonempty.groupby("did").cumcount()
    corpus = nonempty[nonempty.rank_ne < CAP][["did", "rkey", "text", "key"]].reset_index(drop=True)
    c["embedded_posts"] = len(corpus)
    c["embedded_producers"] = int(corpus.did.nunique())
    bake = hash64("s4-bakeoff-500k", corpus.key)
    order = np.argsort(bake, kind="stable")
    corpus["bakeoff"] = False
    corpus.loc[order[:BAKEOFF_N], "bakeoff"] = True
    c["bakeoff_posts"] = int(corpus.bakeoff.sum())
    c["bakeoff_producers"] = int(corpus[corpus.bakeoff].did.nunique())
    pid = dict(zip(producers.did, producers.pid))
    corpus["pid"] = corpus.did.map(pid).astype(np.int64)
    corpus = corpus.sort_values(["pid", "rkey"]).reset_index(drop=True)
    corpus.insert(0, "post_idx", np.arange(len(corpus), dtype=np.int64))
    pq.write_table(pa.Table.from_pandas(corpus.drop(columns=["key"]), preserve_index=False), out / "posts_corpus.parquet")

    # Bio exclusion list (posts-side lists are not applied to bios: amendment A1).
    with open(out / "bio_exclude_dids.txt", "w") as f:
        for d in sorted(intents | minors):
            f.write(d + "\n")
    receipt["outputs"] = {str(out / n): sha256(out / n) for n in ["posts_corpus.parquet", "bio_exclude_dids.txt"]}
    json.dump(receipt, open(out / "prep_corpus_receipt.json", "w"), indent=1)
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
