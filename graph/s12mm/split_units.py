"""Mac side of S12m on Colab: cut O8 + the CID sidecar into self-contained work units of <= 20k posts.

  split_units.py OUT_DIR --o8 DIR [--o8 DIR ...] --sidecar DIR [--sidecar DIR ...]
                 [--unit-rows 20000] [--buckets 64] [--batch-rows 500000] [--hash-inputs]

Inputs (local copies):
  --o8       S9's O8 slab shards, `j*.parquet` (vmpass sql_p2: p, uri, created, text, langs, image_count,
             self_labels); any *.parquet under the directory is read.
  --sidecar  the folded CID sidecar `image_cids-pPP.parquet` (cidpass SIDECAR_SCHEMA: did, rkey, post_created,
             rev, kind, image_index, cid, mime, alt, aspect_w, aspect_h, source). Only `kind = image` rows from
             `images` or `recordWithMedia` count (the post_extract.py `image_cids` rule, as in bench_input.py),
             so link-card thumbs and `gallery` embeds are never fetched; CIDs are ordered by image_index.

Upload format (format id `s12mm-colab-units/1`), written to OUT_DIR:
  units/<unit>.parquet   one work unit, <= --unit-rows rows, zstd; columns
                         uri string, p int64, text string, image_cids list<string>, self_labels list<string>
                         (exactly mmlib's input schema, the sidecar already joined, so the VM needs no sidecar).
                         <unit> = b<bucket:04d>-<n:04d>; job.py names its output u-<unit>-00000.parquet.
  manifest.json          format, unit_rows, buckets, the selection rule, inputs (path, bytes[, sha256]),
                         and per unit: file, rows, image_posts, images, bytes, sha256. The Colab driver uploads
                         units in manifest order and checks every sha256 on the VM before running a unit.
  split_receipt.json     totals, peak RSS, timings.
  record_dids.txt        the distinct DIDs of posts with image CIDs: the RECORD_DIDS_FILE for label_chain.sh, whose
                         gate bundle must carry those accounts' record-level labels (stays on the Mac).

Memory stays bounded (target <= 8 GB): pass 1 streams the sidecar and pass 2 streams O8 in --batch-rows record
batches into --buckets on-disk bucket files keyed by blake2b(did) (all of a producer's posts land in one bucket);
pass 3 joins one bucket at a time (left join on did/rkey, vectorised with index_in) and cuts it into units, sorted
by (p, uri). Duplicate O8 URIs within a bucket are dropped (first kept) and counted. Temp buckets go under
OUT_DIR/_tmp and are removed at the end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import resource
import shutil
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
FORMAT = "s12mm-colab-units/1"
IMAGE_SOURCES = ("images", "recordWithMedia")
SIDE_COLS = ["did", "rkey", "kind", "source", "image_index", "cid"]


def unit_schema():
    import pyarrow as pa
    return pa.schema([("uri", pa.string()), ("p", pa.int64()), ("text", pa.string()),
                      ("image_cids", pa.list_(pa.string())), ("self_labels", pa.list_(pa.string()))])


def sha256_file(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def bucket_of(did: str, n: int) -> int:
    return int.from_bytes(hashlib.blake2b(did.encode(), digest_size=4).digest(), "big") % n


def bucket_ids(did_col, n):
    """Bucket per row, computed once per distinct DID of the batch."""
    import pyarrow as pa
    import pyarrow.compute as pc
    enc = pc.dictionary_encode(did_col).combine_chunks()
    lut = pa.array([bucket_of(d, n) for d in enc.dictionary.to_pylist()], pa.int32())
    return lut.take(enc.indices)


def files_under(dirs):
    out = []
    for d in dirs:
        d = pathlib.Path(d)
        out += sorted(p for p in (d.rglob("*.parquet") if d.is_dir() else [d]) if not p.name.startswith("._"))
    return out


class Buckets:
    """One ParquetWriter per bucket, opened lazily."""

    def __init__(self, d, name, schema, n):
        self.d, self.name, self.schema, self.n, self.w = pathlib.Path(d), name, schema, n, {}
        self.d.mkdir(parents=True, exist_ok=True)

    def path(self, b):
        return self.d / f"{self.name}-b{b:04d}.parquet"

    def write(self, table, bids):
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        for b in pc.unique(bids).to_pylist():
            part = table.filter(pc.equal(bids, b))
            if b not in self.w:
                self.w[b] = pq.ParquetWriter(self.path(b), self.schema, compression="zstd")
            self.w[b].write_table(part)

    def close(self):
        for w in self.w.values():
            w.close()


def pass_sidecar(files, tmp, n, batch_rows, rec):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sch = pa.schema([("did", pa.string()), ("rkey", pa.string()), ("image_index", pa.int16()), ("cid", pa.string())])
    out = Buckets(tmp, "side", sch, n)
    for f in files:
        pf = pq.ParquetFile(f)
        for rb in pf.iter_batches(batch_size=batch_rows, columns=SIDE_COLS):
            t = pa.Table.from_batches([rb])
            rec["sidecar_rows_read"] += t.num_rows
            keep = pc.and_(pc.equal(t["kind"], "image"), pc.is_in(t["source"], value_set=pa.array(IMAGE_SOURCES)))
            keep = pc.and_(keep, pc.invert(pc.is_null(t["cid"])))
            t = t.filter(keep).select(["did", "rkey", "image_index", "cid"])
            t = t.cast(sch)
            rec["sidecar_image_rows"] += t.num_rows
            if t.num_rows:
                out.write(t, bucket_ids(t["did"], n))
    out.close()
    return out


def pass_o8(files, tmp, n, batch_rows, rec):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sch = pa.schema([("uri", pa.string()), ("p", pa.int64()), ("text", pa.string()),
                     ("self_labels", pa.list_(pa.string())), ("did", pa.string()), ("rkey", pa.string()),
                     ("image_count", pa.int64())])
    out = Buckets(tmp, "o8", sch, n)
    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        cols = [c for c in ("uri", "p", "text", "self_labels", "image_count") if c in names]
        for rb in pf.iter_batches(batch_size=batch_rows, columns=cols):
            t = pa.Table.from_batches([rb])
            rec["o8_rows_read"] += t.num_rows
            parts = pc.split_pattern(t["uri"], "/")          # at: '' did app.bsky.feed.post rkey
            n0 = t.num_rows
            ok = pc.fill_null(pc.and_(pc.starts_with(t["uri"], "at://"), pc.equal(pc.list_value_length(parts), 5)), False)
            t, parts = t.filter(ok), parts.filter(ok)
            ok = pc.fill_null(pc.equal(pc.list_element(parts, 3), "app.bsky.feed.post"), False)
            t, parts = t.filter(ok), parts.filter(ok)
            rec["o8_bad_uri_dropped"] += n0 - t.num_rows
            cols_out = {"uri": t["uri"], "p": t["p"].cast(pa.int64()) if "p" in t.column_names else pa.nulls(t.num_rows, pa.int64()),
                        "text": t["text"] if "text" in t.column_names else pa.nulls(t.num_rows, pa.string()),
                        "self_labels": (t["self_labels"].cast(pa.list_(pa.string())) if "self_labels" in t.column_names
                                        else pa.nulls(t.num_rows, pa.list_(pa.string()))),
                        "did": pc.list_element(parts, 2), "rkey": pc.list_element(parts, 4),
                        "image_count": (t["image_count"].cast(pa.int64()) if "image_count" in t.column_names
                                        else pa.nulls(t.num_rows, pa.int64()))}
            tt = pa.table(cols_out, schema=sch)
            if tt.num_rows:
                out.write(tt, bucket_ids(tt["did"], n))
    out.close()
    return out


def join_bucket(o8_path, side_path, rec):
    """The bucket's O8 rows with image_cids (ordered by image_index) left-joined on did/rkey, sorted by (p, uri)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(o8_path)
    key = pc.binary_join_element_wise(t["did"], t["rkey"], "/")
    uniq = pc.unique(t["uri"])
    if len(uniq) != t.num_rows:   # keep the first row of each uri
        first = pc.index_in(uniq, value_set=t["uri"])
        rec["o8_duplicate_uris_dropped"] += t.num_rows - len(uniq)
        t, key = t.take(first), key.take(first)
    cids = pa.nulls(t.num_rows, pa.list_(pa.string()))
    if side_path.exists():
        s = pq.read_table(side_path)
        s = s.append_column("k", pc.binary_join_element_wise(s["did"], s["rkey"], "/"))
        s = s.filter(pc.is_in(s["k"], value_set=key))             # only this bucket's slab posts
        if s.num_rows:
            s = s.sort_by([("k", "ascending"), ("image_index", "ascending")])
            g = s.group_by("k", use_threads=False).aggregate([("cid", "list")])
            idx = pc.index_in(key, value_set=g["k"])
            got = g["cid_list"].take(pc.fill_null(idx, 0))
            cids = pc.if_else(pc.is_null(idx), pa.nulls(t.num_rows, got.type), got).cast(pa.list_(pa.string()))
    has = pc.fill_null(pc.greater(pc.list_value_length(cids), 0), False)
    claimed = pc.fill_null(pc.greater(t["image_count"], 0), False)
    rec["o8_image_count_without_cids"] += pc.sum(pc.cast(pc.and_(claimed, pc.invert(has)), pa.int64())).as_py() or 0
    out = pa.table({"uri": t["uri"], "p": t["p"], "text": t["text"], "image_cids": cids, "self_labels": t["self_labels"]},
                   schema=unit_schema())
    return out.sort_by([("p", "ascending"), ("uri", "ascending")])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--o8", action="append", required=True)
    ap.add_argument("--sidecar", action="append", required=True)
    ap.add_argument("--unit-rows", type=int, default=20_000)
    ap.add_argument("--buckets", type=int, default=64)
    ap.add_argument("--batch-rows", type=int, default=500_000)
    ap.add_argument("--hash-inputs", action="store_true")
    a = ap.parse_args(argv)
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    out = pathlib.Path(a.out_dir)
    if (out / "manifest.json").exists():
        sys.exit(f"{out}/manifest.json exists: units are immutable once a run uses them; pick a new OUT_DIR")
    t0 = time.time()
    tmp = out / "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (out / "units").mkdir(parents=True, exist_ok=True)
    o8_files, side_files = files_under(a.o8), files_under(a.sidecar)
    if not o8_files or not side_files:
        sys.exit("no parquet under --o8 or --sidecar")
    import collections
    rec = collections.Counter()
    pass_sidecar(side_files, tmp, a.buckets, a.batch_rows, rec)
    t1 = time.time()
    o8b = pass_o8(o8_files, tmp, a.buckets, a.batch_rows, rec)
    t2 = time.time()
    units, img_dids = [], set()
    for b in range(a.buckets):
        op = o8b.path(b)
        if not op.exists():
            continue
        t = join_bucket(op, tmp / f"side-b{b:04d}.parquet", rec)
        for n, s in enumerate(range(0, t.num_rows, a.unit_rows)):
            u = t.slice(s, a.unit_rows)
            name = f"b{b:04d}-{n:04d}"
            f = out / "units" / f"{name}.parquet"
            pq.write_table(u, f, compression="zstd")
            lens = pc.fill_null(pc.list_value_length(u["image_cids"]), 0)
            img_dids.update(x.split("/")[2] for x in u.filter(pc.greater(lens, 0))["uri"].to_pylist())
            units.append({"unit": name, "file": f"units/{name}.parquet", "rows": u.num_rows,
                          "image_posts": int(pc.sum(pc.cast(pc.greater(lens, 0), "int64")).as_py() or 0),
                          "images": int(pc.sum(lens).as_py() or 0), "bytes": f.stat().st_size, "sha256": sha256_file(f)})
        op.unlink()
        (tmp / f"side-b{b:04d}.parquet").unlink(missing_ok=True)
    shutil.rmtree(tmp, ignore_errors=True)
    inputs = [{"path": str(p), "bytes": p.stat().st_size, **({"sha256": sha256_file(p)} if a.hash_inputs else {})}
              for p in o8_files + side_files]
    man = {"format": FORMAT, "unit_rows": a.unit_rows, "buckets": a.buckets, "bucket_key": "blake2b-32(did) % buckets",
           "rule": "image_cids = sidecar kind=image, source in (images, recordWithMedia), ordered by image_index; "
                   "left join on (did, rkey) parsed from the O8 uri",
           "schema": [f"{f.name}: {f.type}" for f in unit_schema()], "inputs": inputs, "units": units,
           "totals": {"units": len(units), "posts": sum(u["rows"] for u in units),
                      "image_posts": sum(u["image_posts"] for u in units), "images": sum(u["images"] for u in units),
                      "bytes": sum(u["bytes"] for u in units)}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    (out / "record_dids.txt").write_text("".join(f"{d}\n" for d in sorted(img_dids)))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt = {"counts": dict(rec), "totals": man["totals"], "seconds": {"sidecar": round(t1 - t0, 1),
               "o8": round(t2 - t1, 1), "join_and_cut": round(time.time() - t2, 1)},
               "record_dids": len(img_dids), "peak_rss_gb": round(peak / (2**30 if sys.platform == "darwin" else 2**20), 2),
               "manifest_sha256": sha256_file(out / "manifest.json")}
    (out / "split_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
