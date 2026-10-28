#!/usr/bin/env python3
"""Image-CID sidecar pass: read raw Jetstream segments once, write post image CIDs.

Two steps, both resumable:

  segments  For each sealed .jss segment (GCS buffer, Drive folder, or a local
            archive for tests): fetch + verify it, decode only app.bsky.feed.post
            with the node decoder, turn every post commit into PART_SCHEMA rows
            (cid_rows.py), split them by cityHash64(did) % 16, upload the 16 part
            files and then a per-segment ledger marker. A segment with a marker is
            never fetched again, so a preempted run restarts where it stopped.
  fold      Once every segment in --expect-range has a marker: per partition, keep
            each post's latest revision by (rev, seq) and drop posts whose latest
            revision is a delete (or an update with no media), then write the
            sidecar Parquet to the 60-day store with customTime set.

  status    Print ledger coverage for a run (one listing call).

Store layout (bucket-relative keys):
  <store30>/intermediates/<run>/cid-parts/seg_NNNNNN/seg_NNNNNN-pPP.parquet
  <store30>/intermediates/<run>/_ledger/cid-seg-NNNNNN.json
  <store60>/sidecar/image-cids/<run>/image_cids-pPP.parquet + manifest.json

Progress: one JSON line per finished segment and a `progress` summary line
every --progress-seconds (done/total, segments/min, GB read, ETA).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
GRAPH = HERE.parent
sys.path.insert(0, str(GRAPH))
sys.path.insert(0, str(GRAPH / "vmpass"))
sys.path.insert(0, str(HERE))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import cid_rows  # noqa: E402
from cityhash import city_hash64  # noqa: E402
from store import open_store, sha256_file  # noqa: E402

try:
    from orjson import loads as _loads
except ImportError:  # the VM installs orjson; json works, ~3x slower
    _loads = json.loads

N_PARTS = 16
STAGE_GROUP = "00000000"
POST = "app.bsky.feed.post"


def ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(**kw) -> None:
    print(json.dumps({"ts": ts(), **kw}, separators=(",", ":")), flush=True)


def part_of(did: str, cache: dict) -> int:
    p = cache.get(did)
    if p is None:
        p = cache[did] = city_hash64(did) % N_PARTS
    return p


# ---------------------------------------------------------------- sources

def list_source(kind: str, src: str, listing_cache: str | None = None) -> dict[int, dict]:
    """index -> segment dict in the shape gcs_segments / archive_graph_extract use."""
    if listing_cache and pathlib.Path(listing_cache).exists():
        return {int(k): v for k, v in json.loads(pathlib.Path(listing_cache).read_text()).items()}
    if kind == "gcs":
        import gcs_segments
        out = gcs_segments.completed_segments(src)
    elif kind == "drive":
        import archive_graph_extract as age
        out = age.completed_segments(src)
    elif kind == "local":
        out = {}
        for rec in sorted(pathlib.Path(src, "segments").glob("*/seg_*.jss.receipt.json")):
            name = rec.name[: -len(".receipt.json")]
            jss = rec.parent / name
            if jss.exists():
                out[int(name[4:-4], 36)] = {"group": rec.parent.name, "jss": {"name": name, "size": jss.stat().st_size,
                                                                            "path": str(jss)}, "receipt": {"path": str(rec)}}
    else:
        raise ValueError(kind)
    if listing_cache:
        pathlib.Path(listing_cache).write_text(json.dumps({str(k): v for k, v in out.items()}))
    return out


def fetch(kind: str, seg: dict, dest: pathlib.Path) -> pathlib.Path:
    """Fetch a segment and its receipt into dest/ (the decoder's layout), verified."""
    if kind == "gcs":
        import gcs_segments
        return gcs_segments.download_segment(seg, dest)
    if kind == "drive":
        import archive_graph_extract as age
        return age.download_segment(seg, dest)
    dest.mkdir(parents=True, exist_ok=True)
    name = seg["jss"]["name"]
    shutil.copyfile(seg["jss"]["path"], dest / name)
    shutil.copyfile(seg["receipt"]["path"], dest / (name + ".receipt.json"))
    return dest / name


# ---------------------------------------------------------------- one segment

def decode(decoder: pathlib.Path, archive: pathlib.Path, out: pathlib.Path) -> dict:
    cmd = ["node", str(decoder), "--archive", str(archive), "--out", str(out), "--collections", POST,
           "--max-events", "1000000000", "--max-bytes", "1000GiB", "--max-segments", "1", "--shard-mb", "1024"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(decoder.parent))
    if r.returncode != 0:
        raise RuntimeError(f"decoder failed: {r.stderr[-1500:]}")
    return json.loads((out / "checkpoint.json").read_text())


def rows_from_decoded(out: pathlib.Path, state: dict) -> tuple[list[list[dict]], dict]:
    parts: list[list[dict]] = [[] for _ in range(N_PARTS)]
    cache: dict = {}
    st = {"post_events": 0, "parsed": 0, "media_rows": 0, "marker_rows": 0, "images": 0, "external_thumbs": 0,
          "posts_with_media": 0, "deletes": 0}
    for shard in state.get("shards", []):
        with open(out / shard["path"], "rb") as fh:
            for line in fh:
                st["post_events"] += 1
                if not cid_rows.needs_parse(line):
                    continue
                st["parsed"] += 1
                rows = cid_rows.rows_for_event(_loads(line))
                if not rows:
                    continue
                p = part_of(rows[0]["did"], cache)
                parts[p].extend(rows)
                if rows[0]["row_type"] == "marker":
                    st["marker_rows"] += 1
                    st["deletes"] += rows[0]["operation"] == "delete"
                else:
                    st["posts_with_media"] += 1
                    st["media_rows"] += len(rows)
                    for r in rows:
                        st["images" if r["kind"] == "image" else "external_thumbs"] += 1
    return parts, st


def write_parts(parts: list[list[dict]], index: int, dest: pathlib.Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for p, rows in enumerate(parts):
        path = dest / f"seg_{index:06d}-p{p:02d}.parquet"
        table = pa.Table.from_pylist(rows, schema=cid_rows.PART_SCHEMA)
        pq.write_table(table, path, compression="zstd", compression_level=3)
        files.append({"part": p, "file": path.name, "rows": len(rows), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path)})
    return files


def ledger_key(run_id: str, index: int) -> str:
    return f"intermediates/{run_id}/_ledger/cid-seg-{index:06d}.json"


def parts_prefix(run_id: str, index: int) -> str:
    return f"intermediates/{run_id}/cid-parts/seg_{index:06d}"


_CTX: dict = {}


def _init_worker(ctx: dict) -> None:
    _CTX.update(ctx)


def process_segment(job: tuple[int, dict]) -> dict:
    index, seg = job
    c = _CTX
    store = open_store(c["store30"])
    wdir = pathlib.Path(c["work"]) / f"seg_{index:06d}"
    shutil.rmtree(wdir, ignore_errors=True)
    t0 = time.time()
    last_err = last_trace = ""
    for attempt in range(3):
        try:
            # The decoder only reads group dirs named like hex ("0000000e"); Drive's
            # overflow folders ("catbird-overflow-<hash>") are not, so stage under a fixed one.
            seg_dir = wdir / "archive" / "segments" / STAGE_GROUP
            fetch(c["source"], seg, seg_dir)
            t1 = time.time()
            receipt = json.loads((seg_dir / (seg["jss"]["name"] + ".receipt.json")).read_text())
            state = decode(pathlib.Path(c["decoder"]), wdir / "archive", wdir / "dec")
            decoded = state.get("decoded_segments", [])
            if len(decoded) != 1 or decoded[0]["index"] != index:
                raise RuntimeError(f"decoder covered {[d.get('index') for d in decoded]}, expected [{index}]")
            (seg_dir / seg["jss"]["name"]).unlink(missing_ok=True)
            t2 = time.time()
            parts, st = rows_from_decoded(wdir / "dec", state)
            files = write_parts(parts, index, wdir / "parts")
            shutil.rmtree(wdir / "dec", ignore_errors=True)
            t3 = time.time()
            pfx = parts_prefix(c["run_id"], index)
            store.put_files([(wdir / "parts" / f["file"], f"{pfx}/{f['file']}") for f in files])
            record = {
                "index": index, "name": seg["jss"]["name"], "source": c["source"], "sha256": receipt.get("sha256"),
                "bytes": int(seg["jss"]["size"]), "decoder_events": decoded[0].get("events_emitted"),
                "skipped_invalid_records": state.get("last_run", {}).get("skipped_invalid_records"),
                **st, "parts_prefix": pfx, "files": files, "attempts": attempt + 1,
                "fetch_s": round(t1 - t0, 1), "decode_s": round(t2 - t1, 1), "rows_s": round(t3 - t2, 1),
                "upload_s": round(time.time() - t3, 1), "host": os.uname().nodename, "finished": ts(),
            }
            store.write_text(ledger_key(c["run_id"], index), json.dumps(record, sort_keys=True))
            shutil.rmtree(wdir, ignore_errors=True)
            return {"ok": True, **{k: v for k, v in record.items() if k != "files"}}
        except Exception as exc:  # retried; the ledger marker is written only on success
            last_err = f"{type(exc).__name__}: {exc}"[-800:]
            last_trace = traceback.format_exc()[-800:]
            shutil.rmtree(wdir, ignore_errors=True)
            time.sleep(c.get("retry_sleep", 5) * (attempt + 1))
    return {"ok": False, "index": index, "error": last_err, "trace": last_trace}


# ---------------------------------------------------------------- ledger

def done_indices(store, run_id: str) -> set[int]:
    prefix = f"intermediates/{run_id}/_ledger/"
    if store.url.startswith("gs://"):
        r = subprocess.run(["gcloud", "storage", "ls", f"{store.url}/{prefix}cid-seg-*.json"],
                           capture_output=True, text=True)
        if r.returncode != 0 and "matched no objects" not in (r.stderr or "") and "One or more URLs" not in (r.stderr or ""):
            raise RuntimeError(f"ledger listing failed: {r.stderr[-400:]}")
        names = [ln.rsplit("/", 1)[-1] for ln in r.stdout.split()]
    else:
        names = [p.name for p in (pathlib.Path(store.url) / prefix).glob("cid-seg-*.json")]
    return {int(n[len("cid-seg-"):-len(".json")]) for n in names if n.startswith("cid-seg-")}


def parse_range(text: str) -> tuple[int, int]:
    a, b = text.split("-")
    return int(a), int(b)


# ---------------------------------------------------------------- commands

def cmd_segments(a) -> int:
    store = open_store(a.store30)
    available = list_source(a.source, a.src, a.listing_cache)
    lo, hi = a.min_index, a.max_index if a.max_index is not None else max(available, default=-1)
    wanted = {i: s for i, s in available.items() if lo <= i <= hi}
    done = done_indices(store, a.run_id)
    pending = sorted(i for i in wanted if i not in done)
    if a.max_segments:
        pending = pending[: a.max_segments]
    total_bytes = sum(int(wanted[i]["jss"]["size"]) for i in pending)
    log(msg="start", source=a.source, run_id=a.run_id, range=[lo, hi], available=len(wanted),
        already_done=len([i for i in wanted if i in done]), pending=len(pending), pending_gb=round(total_bytes / 1e9, 1), workers=a.workers,
        missing_in_range=[i for i in range(lo, hi + 1) if i not in available][:50])
    if not pending:
        return 0
    pathlib.Path(a.work).mkdir(parents=True, exist_ok=True)
    ctx = {"store30": a.store30, "work": a.work, "source": a.source, "decoder": a.decoder, "run_id": a.run_id,
           "retry_sleep": a.retry_sleep}
    t0 = last = time.time()
    n_ok = n_fail = 0
    gb = 0.0
    agg = {"post_events": 0, "media_rows": 0, "marker_rows": 0, "images": 0, "external_thumbs": 0}
    failed = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(ctx,), maxtasksperchild=50) as pool:
        for res in pool.imap_unordered(process_segment, [(i, wanted[i]) for i in pending]):
            if res["ok"]:
                n_ok += 1
                gb += res["bytes"] / 1e9
                for k in agg:
                    agg[k] += res.get(k, 0)
                log(msg="segment", **{k: res.get(k) for k in ("index", "post_events", "media_rows", "marker_rows",
                                                              "fetch_s", "decode_s", "rows_s", "upload_s", "attempts")})
            else:
                n_fail += 1
                failed.append(res["index"])
                log(msg="segment_failed", index=res["index"], error=res["error"])
            now = time.time()
            if now - last >= a.progress_seconds or n_ok + n_fail == len(pending):
                last = now
                rate = n_ok / max(now - t0, 1e-6) * 60
                left = len(pending) - n_ok - n_fail
                log(msg="progress", done=n_ok, failed=n_fail, pending=left, of=len(pending),
                    seg_per_min=round(rate, 2), gb_read=round(gb, 1), elapsed_min=round((now - t0) / 60, 1),
                    eta_min=round(left / rate, 1) if rate else None, **agg)
    log(msg="segments_done", ok=n_ok, failed=failed, elapsed_min=round((time.time() - t0) / 60, 1), **agg)
    return 0 if not failed else 3


FOLD_SQL = """
SET max_bytes_before_external_group_by = 12000000000, max_bytes_before_external_sort = 12000000000,
    join_algorithm = 'full_sorting_merge';
CREATE TABLE parts ENGINE = MergeTree ORDER BY (did, rkey) AS
SELECT * FROM file('{src}', Parquet, '{structure}');
CREATE TABLE tops ENGINE = MergeTree ORDER BY (did, rkey) AS
SELECT did, rkey, max((rev, seq)) AS top, argMax(row_type, (rev, seq)) AS top_type
FROM parts GROUP BY did, rkey HAVING top_type = 'media';
SELECT p.did AS did, p.rkey AS rkey,
  fromUnixTimestamp64Micro(p.post_created_us, 'UTC') AS post_created,
  p.rev AS rev, p.kind AS kind, p.image_index AS image_index, p.cid AS cid, p.mime AS mime, p.alt AS alt,
  p.aspect_w AS aspect_w, p.aspect_h AS aspect_h, p.source AS source
FROM parts AS p INNER JOIN tops AS t ON p.did = t.did AND p.rkey = t.rkey
WHERE p.row_type = 'media' AND (p.rev, p.seq) = t.top
ORDER BY did, rkey, kind, image_index
LIMIT 1 BY did, rkey, kind, image_index
INTO OUTFILE '{out}' FORMAT Parquet
SETTINGS output_format_parquet_compression_method = 'zstd', output_format_parquet_row_group_size = 1000000;
"""


# Explicit structure: Parquet strings would otherwise infer as Nullable(String), which
# a MergeTree sorting key refuses.
PART_STRUCTURE = ("did String, rkey String, post_created_us Nullable(Int64), rev String, kind String, "
                  "image_index Nullable(Int16), cid String, mime String, alt String, aspect_w Nullable(Int32), "
                  "aspect_h Nullable(Int32), source String, row_type String, operation String, seq UInt64")
assert [c.split()[0] for c in PART_STRUCTURE.split(", ")] == cid_rows.PART_SCHEMA.names


def fold_partition(parts_dir: pathlib.Path, out: pathlib.Path, clickhouse: str, tmp: pathlib.Path) -> None:
    out.unlink(missing_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    sql = FOLD_SQL.format(structure=PART_STRUCTURE, src=str(parts_dir / "*.parquet").replace("'", "\\'"), out=str(out).replace("'", "\\'"))
    r = subprocess.run([clickhouse, "local", "--path", str(tmp), "--multiquery", "--query", sql],
                       capture_output=True, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode != 0:
        raise RuntimeError(f"fold failed: {r.stderr[-1500:]}")


def fetch_partition(store, run_id: str, p: int, segs: list[int], dest: pathlib.Path) -> None:
    """All segments' part files for partition p, in one wildcard copy (not one call per file)."""
    dest.mkdir(parents=True, exist_ok=True)
    pattern = f"intermediates/{run_id}/cid-parts/seg_*/seg_*-p{p:02d}.parquet"
    if store.url.startswith("gs://"):
        r = subprocess.run(["gcloud", "storage", "cp", "--quiet", f"{store.url}/{pattern}", f"{dest}/"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"part download failed: {r.stderr[-800:]}")
    else:
        for src in pathlib.Path(store.url).glob(pattern):
            shutil.copyfile(src, dest / src.name)
    keep = {f"seg_{i:06d}-p{p:02d}.parquet" for i in segs}
    for f in dest.iterdir():  # parts of segments outside --expect-range are not folded
        if f.name not in keep:
            f.unlink()
    got = {f.name for f in dest.iterdir()}
    if got != keep:
        raise RuntimeError(f"partition {p}: {len(keep - got)} part files missing, e.g. {sorted(keep - got)[:5]}")


def fetch_ledgers(store, run_id: str, dest: pathlib.Path) -> dict[int, dict]:
    dest.mkdir(parents=True, exist_ok=True)
    pattern = f"intermediates/{run_id}/_ledger/cid-seg-*.json"
    if store.url.startswith("gs://"):
        r = subprocess.run(["gcloud", "storage", "cp", "--quiet", f"{store.url}/{pattern}", f"{dest}/"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ledger download failed: {r.stderr[-800:]}")
    else:
        for src in pathlib.Path(store.url).glob(pattern):
            shutil.copyfile(src, dest / src.name)
    return {int(f.name[8:-5]): json.loads(f.read_text()) for f in dest.glob("cid-seg-*.json")}


def cmd_fold(a) -> int:
    s30 = open_store(a.store30)
    s60 = open_store(a.store60, custom_time=a.custom_time) if a.store60.startswith("gs://") else open_store(a.store60)
    lo, hi = parse_range(a.expect_range)
    done = done_indices(s30, a.run_id)
    missing = [i for i in range(lo, hi + 1) if i not in done]
    log(msg="fold_start", run_id=a.run_id, expect=[lo, hi], done=len(done), missing=len(missing), first_missing=missing[:20])
    if missing and not a.allow_partial:
        log(msg="fold_refused", reason="segments missing from the ledger; finish them or pass --allow-partial")
        return 4
    segs = sorted(i for i in done if lo <= i <= hi)
    work = pathlib.Path(a.work)
    ledgers = fetch_ledgers(s30, a.run_id, work / "ledger")
    out_prefix = f"sidecar/image-cids/{a.run_id}"
    totals = {k: sum(int(ledgers[i].get(k) or 0) for i in segs) for k in
              ("bytes", "post_events", "media_rows", "marker_rows", "images", "external_thumbs", "posts_with_media",
               "deletes", "skipped_invalid_records")}
    manifest = {"run_id": a.run_id, "expect_range": [lo, hi], "segments": len(segs), "missing": missing,
                "sources": sorted({ledgers[i]["source"] for i in segs}), "segment_totals": totals,
                "partitioning": "cityHash64(did) % 16", "schema": [f.name for f in cid_rows.SIDECAR_SCHEMA],
                "parts": {}, "created": ts()}
    parts = [int(p) for p in a.partitions.split(",")] if a.partitions else list(range(N_PARTS))
    for p in parts:
        t0 = time.time()
        pdir = work / f"fold-p{p:02d}"
        shutil.rmtree(pdir, ignore_errors=True)
        pdir.mkdir(parents=True)
        fetch_partition(s30, a.run_id, p, segs, pdir)
        for i in segs:  # every part file must match the sha256 its segment's ledger recorded
            want = next(f for f in ledgers[i]["files"] if f["part"] == p)
            if sha256_file(pdir / want["file"]) != want["sha256"]:
                raise RuntimeError(f"sha256 mismatch for {want['file']}")
        out = work / f"image_cids-p{p:02d}.parquet"
        fold_partition(pdir, out, a.clickhouse, work / f"ch-p{p:02d}")
        meta = pq.ParquetFile(out).metadata
        rec = {"rows": meta.num_rows, "bytes": out.stat().st_size, "sha256": sha256_file(out)}
        s60.put(out, f"{out_prefix}/{out.name}")
        manifest["parts"][f"p{p:02d}"] = rec
        shutil.rmtree(pdir, ignore_errors=True)
        out.unlink(missing_ok=True)
        log(msg="fold_partition", part=p, seconds=round(time.time() - t0, 1), **rec)
    if not a.partitions:
        mpath = work / "manifest.json"
        mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        s60.put(mpath, f"{out_prefix}/manifest.json")
    log(msg="fold_done", rows=sum(v["rows"] for v in manifest["parts"].values()),
        gb=round(sum(v["bytes"] for v in manifest["parts"].values()) / 1e9, 2))
    return 0


def cmd_status(a) -> int:
    done = done_indices(open_store(a.store30), a.run_id)
    lo, hi = parse_range(a.expect_range)
    missing = [i for i in range(lo, hi + 1) if i not in done]
    print(json.dumps({"run_id": a.run_id, "done": len(done), "expect": [lo, hi], "missing": len(missing),
                      "first_missing": missing[:20]}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("segments")
    s.add_argument("--source", choices=("gcs", "drive", "local"), required=True)
    s.add_argument("--src", required=True, help="gs:// buffer prefix, Drive segments folder id, or local archive dir")
    s.add_argument("--listing-cache", default=None, help="JSON file: reuse (or save) the source listing")
    s.add_argument("--store30", required=True)
    s.add_argument("--run-id", required=True)
    s.add_argument("--work", required=True)
    s.add_argument("--decoder", required=True, help="path to decode_segments.mjs (with node_modules beside it)")
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    s.add_argument("--min-index", type=int, default=0)
    s.add_argument("--max-index", type=int, default=None)
    s.add_argument("--max-segments", type=int, default=0, help="stop after this many (0: all)")
    s.add_argument("--progress-seconds", type=int, default=180)
    s.add_argument("--retry-sleep", type=float, default=5.0, help="seconds x attempt between a segment's 3 tries")
    f = sub.add_parser("fold")
    f.add_argument("--store30", required=True)
    f.add_argument("--store60", required=True)
    f.add_argument("--custom-time", default=None, help="customTime for plan-60d objects (RFC 3339)")
    f.add_argument("--run-id", required=True)
    f.add_argument("--work", required=True)
    f.add_argument("--expect-range", default="0-7392")
    f.add_argument("--allow-partial", action="store_true")
    f.add_argument("--partitions", default="", help="comma list (default: all 16, plus the manifest)")
    f.add_argument("--clickhouse", default="clickhouse")
    t = sub.add_parser("status")
    t.add_argument("--store30", required=True)
    t.add_argument("--run-id", required=True)
    t.add_argument("--expect-range", default="0-7392")
    a = ap.parse_args(argv)
    return {"segments": cmd_segments, "fold": cmd_fold, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
