"""Mac side of S7g: the benchmark's image-post input and its BENCH_PRODUCERS list (nothing is uploaded here).

  bench_input.py PARTS_DIR OUT_DIR [--min-posts 10000]

PARTS_DIR: CID sidecar part files (graph/cidpass PART_SCHEMA) of completed segments of run cid-20261028a,
copied from plan-30d/intermediates/cid-20261028a/cid-parts/. Per (did, rkey) the latest revision by (rev, seq)
is kept (a latest delete, or a latest revision without images, drops the post); only `kind = image` rows from
`app.bsky.embed.images` (directly or inside recordWithMedia) count, the post_extract.py `image_cids` rule, so
link-card thumbnails, gallery and video embeds are never fetched.

Text and self-labels come from the netcup export days Oct 19-22 (S4's window): latest revision, not deleted, no
reply parent, author never visibility-hidden on any day; formatted as O8 slab text (text + " [image: <alt>]" per
non-empty alt). The parts are mostly older posts (the capture carries backfill), so only Oct 19-22 posts can get
text locally. Rows: uri, did, rkey, p (S4 pid or null), text, image_cids, self_labels, tier (0 = S7 slab
producer, 1 = in S4's filtered corpus, 2 = other); the remaining D9 exclusions are the VM gate's job.

BENCH_PRODUCERS = known image-heavy members (>= 5 originals, >= 50% image originals: Q20's K = 5 reading) of the
4,827 qualifying B1b lists, so the report-only G-img comparison has its lists. The VM's gate still applies D9.
Writes OUT_DIR/image_posts/part-0.parquet, OUT_DIR/bench_producers.txt, OUT_DIR/bench_input_receipt.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mmlib as L  # noqa: E402

LAKE = pathlib.Path.home() / "catbird-lake-local"
CORPUS = LAKE / "s4-pilot-20261028/inputs/posts_corpus.parquet"
ROWMAP = LAKE / "s7-bench-20261028/local/s7_rowmap.parquet"
NETCUP_DAYS = [LAKE / f"netcup-export/2026-10-{d}/posts.parquet" for d in (19, 20, 21, 22)]
Q20 = LAKE / "c4-prep-20261028/q20"
LISTS = LAKE / "communities-cand1-hubdemote/audit-quality/curatelists.tsv"
IMAGE_SOURCES = {"images", "recordWithMedia"}


def latest_images(parts_dir, candidates):
    """{(did, rkey): [(image_index, cid, alt)]} for candidate posts whose latest revision has embed.images
    images. `candidates` is a pyarrow array of "did/rkey" keys; other rows are filtered out in Arrow first."""
    best = {}   # (did, rkey) -> ((rev, seq), op, [(idx, cid, alt)])
    files = sorted(pathlib.Path(parts_dir).rglob("*.parquet"))
    cols = ["did", "rkey", "rev", "seq", "operation", "row_type", "kind", "source", "image_index", "cid", "alt"]
    cand = pa.table({"k": candidates})
    for s0 in range(0, len(files), 400):   # one hash join per 400 files (is_in per file rebuilt the set 12k times)
        t = pa.concat_tables([pq.read_table(f, columns=cols) for f in files[s0:s0 + 400]])
        t = t.append_column("k", pc.binary_join_element_wise(t["did"], t["rkey"], "/"))
        d = t.join(cand, "k", join_type="inner").drop(["k"]).to_pydict()
        for i in range(len(d["did"])):
            k = (d["did"][i], d["rkey"][i])
            rank = (d["rev"][i] or "", d["seq"][i] or 0)
            cur = best.get(k)
            if cur is None or rank > cur[0]:
                cur = best[k] = [rank, d["operation"][i], []]
            elif rank < cur[0]:
                continue
            if (d["row_type"][i] == "media" and d["kind"][i] == "image" and d["source"][i] in IMAGE_SOURCES
                    and d["cid"][i]):
                cur[2].append((d["image_index"][i], d["cid"][i], d["alt"][i] or ""))
    out = {}
    for k, (_r, op, imgs) in best.items():
        if op != "delete" and imgs:
            out[k] = sorted(set(imgs))
    return out, len(files)


def netcup_originals():
    """Latest revision per (did, rkey) over the netcup export days Oct 19-22 (S4's window): non-deleted
    originals (no reply parent; quotes included) whose author was never visibility-hidden on any day (S4's
    stricter reading), with image_count >= 1. Returns {(did, rkey): (text, self_labels)}."""
    hidden = set()
    for p in NETCUP_DAYS:
        v = pq.read_table(p.parent / "visibility.parquet", columns=["did", "hide_from_algorithmic"])
        hidden |= set(v.filter(pc.equal(v["hide_from_algorithmic"], 1))["did"].to_pylist())
    best = {}
    cols = ["did", "rkey", "rev", "seq", "is_deleted", "reply_parent", "image_count", "text", "self_labels"]
    for p in NETCUP_DAYS:
        t = pq.read_table(p, columns=cols)
        t = t.filter(pc.greater_equal(pc.fill_null(t["image_count"], 0), 1))
        d = t.to_pydict()
        for i in range(len(d["did"])):
            k = (d["did"][i], d["rkey"][i])
            rank = (d["rev"][i] or "", d["seq"][i] or 0)
            if k not in best or rank >= best[k][0]:
                best[k] = (rank, bool(d["is_deleted"][i]), d["reply_parent"][i] or "", d["text"][i] or "",
                           list(d["self_labels"][i] or []))
    return {k: (v[3], v[4]) for k, v in best.items() if not v[1] and not v[2] and k[0] not in hidden}, len(hidden)


def image_heavy_b1b_members():
    heavy_lists = {ln.split("\t", 1)[0] for ln in open(Q20 / "q20_lists_k5_at50.tsv") if ln.strip()}
    counts = {}
    for ln in open(Q20 / "member_image_counts.tsv"):
        d, n, im = ln.rstrip("\n").split("\t")
        counts[d] = (int(n), int(im))
    out = set()
    for ln in open(LISTS):
        uri, _o, _n, _t, members = ln.rstrip("\n").split("\t")
        if uri in heavy_lists:
            for m in members.split(","):
                n, im = counts.get(m, (0, 0))
                if n >= 5 and im / n >= 0.5:
                    out.add(m)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--min-posts", type=int, default=10_000)
    a = ap.parse_args(argv)
    posts, n_hidden = netcup_originals()
    cand = pa.array([f"{d}/{r}" for d, r in posts])
    imgs, n_files = latest_images(a.parts_dir, cand)
    s4 = pq.read_table(CORPUS, columns=["did", "rkey", "pid"]).to_pydict()
    s4pid = {(d, r): p for d, r, p in zip(s4["did"], s4["rkey"], s4["pid"])}
    rm = pq.read_table(ROWMAP, columns=["row_id", "did"]).to_pydict()
    slab = {d for i, d in zip(rm["row_id"], rm["did"]) if i >= 1_000_000}
    rows, c = [], collections.Counter()
    for (d, r), im in imgs.items():
        text, labels = posts[(d, r)]
        tier = 0 if d in slab else 1 if (d, r) in s4pid else 2
        rows.append({"uri": L.post_uri(d, r), "did": d, "rkey": r, "p": s4pid.get((d, r)),
                     "text": text + "".join(f" [image: {x}]" for _i, _c, x in im if x),
                     "image_cids": [cid for _i, cid, _a in im], "self_labels": labels, "tier": tier})
        c[("tier0_s7_slab_producer", "tier1_s4_corpus", "tier2_netcup_other")[tier]] += 1
        c["self_labelled"] += bool(labels)
    rows.sort(key=lambda x: (x["tier"], x["uri"]))
    heavy = image_heavy_b1b_members()
    out = pathlib.Path(a.out_dir)
    (out / "image_posts").mkdir(parents=True, exist_ok=True)
    sch = pa.schema([("uri", pa.string()), ("did", pa.string()), ("rkey", pa.string()), ("p", pa.int64()),
                     ("text", pa.string()), ("image_cids", pa.list_(pa.string())),
                     ("self_labels", pa.list_(pa.string())), ("tier", pa.int8())])
    pq.write_table(pa.Table.from_pylist(rows, schema=sch), out / "image_posts" / "part-0.parquet", compression="zstd")
    (out / "bench_producers.txt").write_text("".join(f"{d}\n" for d in sorted(heavy)))
    rec = {"parts_files": n_files, "netcup_image_originals_oct19_22": len(posts), "visibility_hidden_dids": n_hidden,
           "joined_posts": len(rows),
           "images": sum(len(r["image_cids"]) for r in rows), "counts": dict(c),
           "bench_producers_image_heavy_b1b": len(heavy),
           "posts_by_bench_producers": sum(r["did"] in heavy for r in rows),
           "meets_min_before_gate": len(rows) >= a.min_posts,
           "inputs_sha256": {str(p): L.sha256_file(p) for p in [CORPUS, ROWMAP, Q20 / "q20_lists_k5_at50.tsv",
                                                                Q20 / "member_image_counts.tsv"]},
           "sha256": {"image_posts/part-0.parquet": L.sha256_file(out / "image_posts" / "part-0.parquet"),
                      "bench_producers.txt": L.sha256_file(out / "bench_producers.txt")}}
    (out / "bench_input_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "inputs_sha256"}, indent=1))


if __name__ == "__main__":
    main()
