#!/usr/bin/env python3
"""Stream completed Jetstream archive segments from Google Drive, decode only
graph collections, and write compact Parquet for Phase 1 community discovery.

Read-only on Drive (never touches .archive.lock or the downloader's files).
Segments are processed in strict index order because the decoder enforces a
monotonic sequence. Each raw .jss is deleted locally after successful decode;
only the extracted graph events are kept.

Usage:
  archive_graph_extract.py [--work DIR] [--batch N] [--parallel N] [--once]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

GWS = "/usr/local/bin/gws"
ARCHIVE_SEGMENTS_FOLDER = "1Co7E4BnsWVVF_9Z4qkB0pn1Jzq05F8ae"  # public-archive-001/segments
DECODER = pathlib.Path(
    "/Users/joshlacalamito/Developer/catbird-community-engine-20261018/tools/community-engine/backfill/decode_segments.mjs"
)
COLLECTIONS = [
    "app.bsky.graph.follow",
    "app.bsky.graph.block",
    "app.bsky.graph.list",
    "app.bsky.graph.listitem",
    "app.bsky.graph.starterpack",
    "app.bsky.feed.generator",
    "app.bsky.actor.profile",
    "app.bsky.actor.contentVisibilityDeclaration",
]

SCHEMA = pa.schema(
    [
        ("seq", pa.uint64()),
        ("time", pa.string()),
        ("kind", pa.string()),
        ("did", pa.string()),
        ("collection", pa.string()),
        ("rkey", pa.string()),
        ("rev", pa.string()),
        ("operation", pa.string()),
        ("created_at", pa.string()),
        ("subject", pa.string()),
        ("list_uri", pa.string()),
        ("name", pa.string()),
        ("description", pa.string()),
        ("extra", pa.string()),
        ("account_active", pa.bool_()),
        ("account_status", pa.string()),
    ]
)


def gws_json(*args: str, cwd: str | None = None) -> dict:
    last_err = ""
    for attempt in range(5):
        r = subprocess.run([GWS, *args], capture_output=True, text=True, cwd=cwd)
        if r.returncode == 0:
            start = r.stdout.find("{")
            if start >= 0:
                try:
                    return json.loads(r.stdout[start:])
                except json.JSONDecodeError as exc:
                    last_err = str(exc)
            else:
                return {}
        else:
            last_err = (r.stderr or r.stdout)[-300:]
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"gws_json failed after 5 attempts ({args[:3]}): {last_err}")


def list_children(folder_id: str) -> list[dict]:
    items: list[dict] = []
    token = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken,files(id,name,size,mimeType)",
            "pageSize": 1000,
        }
        if token:
            params["pageToken"] = token
        resp = gws_json("drive", "files", "list", "--params", json.dumps(params))
        items += resp.get("files", [])
        token = resp.get("nextPageToken")
        if not token:
            return items


def find_preserved_pair(seg_base: str) -> tuple[dict, dict] | None:
    """When an active segment is still flushing or truncated on Drive FUSE, check
    if a full-size (>=50MB) .jss + .receipt.json pair exists in the same parent
    folder (including preserved _trash generations)."""
    params = {
        "q": f"name contains '{seg_base}'",
        "fields": "files(id,name,size,parents,modifiedTime)",
        "pageSize": 100,
    }
    files = gws_json("drive", "files", "list", "--params", json.dumps(params)).get("files", [])
    by_parent: dict[str, dict[str, dict]] = {}
    for f in files:
        for p in f.get("parents") or []:
            by_parent.setdefault(p, {})[f["name"]] = f
    jss_name = f"{seg_base}.jss"
    rec_name = f"{seg_base}.jss.receipt.json"
    for p, fmap in by_parent.items():
        j = fmap.get(jss_name)
        r = fmap.get(rec_name)
        if j and r and int(j.get("size") or 0) >= 50_000_000:
            return j, r
    return None


def completed_segments(segments_folder: str = ARCHIVE_SEGMENTS_FOLDER) -> dict[int, dict]:
    """Map segment index -> {group, jss, receipt} for verified full-size segments."""
    out: dict[int, dict] = {}
    for group in list_children(segments_folder):
        if group.get("mimeType") != "application/vnd.google-apps.folder":
            continue
        files = {f["name"]: f for f in list_children(group["id"])}
        seen_bases = {
            n[:-17] if n.endswith(".jss.receipt.json") else n[:-4]
            for n in files
            if n.startswith("seg_") and (n.endswith(".jss") or n.endswith(".jss.receipt.json"))
        }
        for base in sorted(seen_bases):
            name = f"{base}.jss"
            f = files.get(name)
            receipt = files.get(name + ".receipt.json")
            if not receipt:
                continue
            if not f or int(f.get("size") or 0) < 200_000_000:
                pair = find_preserved_pair(base)
                if not pair:
                    continue
                f, receipt = pair
            index = int(base[4:], 36)
            out[index] = {"group": group["name"], "jss": f, "receipt": receipt}
    return out


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_segment(seg: dict, dest_dir: pathlib.Path) -> pathlib.Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = seg["jss"]["name"]
    target = dest_dir / name
    # gws returns JSON media on stdout (it does not honor -o for JSON), so the
    # receipt is captured and written here; the binary segment uses -o.
    receipt = gws_json("drive", "files", "get", "--params", json.dumps({"fileId": seg["receipt"]["id"], "alt": "media"}))
    if receipt.get("name") != name or "sha256" not in receipt:
        raise RuntimeError(f"unexpected receipt for {name}: {str(receipt)[:200]}")
    (dest_dir / (name + ".receipt.json")).write_text(json.dumps(receipt, indent=2))
    expected_bytes = int(receipt.get("bytes") or seg["jss"].get("size") or 0)
    expected_sha = receipt["sha256"]
    if target.exists() and target.stat().st_size == expected_bytes and sha256_file(target) == expected_sha:
        return target
    last_err = ""
    for attempt in range(6):
        target.unlink(missing_ok=True)
        r = subprocess.run(
            [GWS, "drive", "files", "get", "--params", json.dumps({"fileId": seg["jss"]["id"], "alt": "media"}), "-o", name],
            cwd=str(dest_dir),
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and target.exists():
            actual_bytes = target.stat().st_size
            if expected_bytes and actual_bytes != expected_bytes:
                last_err = f"truncated download {actual_bytes} != {expected_bytes}"
            else:
                got = sha256_file(target)
                if got == expected_sha:
                    return target
                last_err = f"sha256 mismatch {got} != {expected_sha}"
        else:
            last_err = (r.stderr or r.stdout)[-300:]
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"download failed for {name} after 6 attempts: {last_err}")


def row_from_event(ev: dict) -> dict:
    rec = ev.get("record") if isinstance(ev.get("record"), dict) else {}
    kind = (ev.get("$type") or "").rsplit("#", 1)[-1] or ev.get("kind", "")
    collection = ev.get("collection", "")
    subject = rec.get("subject") if isinstance(rec.get("subject"), str) else ""
    extra = ""
    if collection == "app.bsky.graph.starterpack":
        feeds = rec.get("feeds")
        # Malformed records exist upstream (e.g. `feeds` as an int); treat non-lists as empty.
        extra = json.dumps([f.get("uri") for f in feeds if isinstance(f, dict)] if isinstance(feeds, list) else [])
    elif collection == "app.bsky.feed.generator":
        extra = rec.get("did", "") if isinstance(rec.get("did"), str) else ""
    elif collection == "app.bsky.actor.contentVisibilityDeclaration":
        extra = "1" if rec.get("hideFromAlgorithmicRecommendations") is True else "0"
    elif collection == "app.bsky.graph.list":
        extra = rec.get("purpose", "") if isinstance(rec.get("purpose"), str) else ""
    account = ev.get("account") if isinstance(ev.get("account"), dict) else {}
    return {
        "seq": int(ev.get("seq") or 0),
        "time": ev.get("time", ""),
        "kind": kind,
        "did": ev.get("did", ""),
        "collection": collection,
        "rkey": ev.get("rkey", ""),
        "rev": ev.get("rev", ""),
        "operation": ev.get("operation", ""),
        "created_at": rec.get("createdAt", "") if isinstance(rec.get("createdAt"), str) else "",
        "subject": subject,
        "list_uri": rec.get("list", "") if isinstance(rec.get("list"), str) else "",
        "name": (rec.get("name") or rec.get("displayName") or "")[:512] if isinstance(rec.get("name") or rec.get("displayName") or "", str) else "",
        "description": (rec.get("description") or "")[:2000] if isinstance(rec.get("description") or "", str) else "",
        "extra": extra,
        "account_active": bool(account.get("active")) if account else None,
        "account_status": account.get("status", "") if account else "",
    }


def convert_shards(decoded_dir: pathlib.Path, out_dir: pathlib.Path, done: set[str], batch_tag: str) -> int:
    meta = json.loads((decoded_dir / "checkpoint.json").read_text())
    rows_total = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, shard in enumerate(meta.get("shards", [])):
        path = shard["path"]
        sha = shard.get("sha256", pathlib.Path(path).stem)
        if sha in done or path in done:
            continue
        src = decoded_dir / path
        if not src.exists():
            done.add(sha)
            continue
        rows = []
        with src.open() as fh:
            for line in fh:
                if line.strip():
                    rows.append(row_from_event(json.loads(line)))
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        dst = out_dir / f"events-{batch_tag}-{idx:02d}-{sha}.parquet"
        pq.write_table(table, dst, compression="zstd", compression_level=6)
        rows_total += len(rows)
        src.unlink()
        done.add(sha)
    return rows_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/Users/joshlacalamito/catbird-lake-local/archive-graph")
    ap.add_argument("--segments-folder", default=ARCHIVE_SEGMENTS_FOLDER, help="Google Drive segments folder ID")
    ap.add_argument("--batch", type=int, default=12, help="segments per decode round")
    ap.add_argument("--parallel", type=int, default=4, help="parallel downloads")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll", type=int, default=600, help="seconds to wait when no new segments")
    args = ap.parse_args()

    work = pathlib.Path(args.work)
    staging = work / "staging" / "segments"
    decoded = work / "decoded_batch"
    events = work / "events"
    ledger_path = work / "ledger.json"
    work.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"next_index": 0, "segments": {}, "shards_done": []}
    shards_done = set(ledger["shards_done"])

    while True:
        available = completed_segments(args.segments_folder)
        pending_indices = [idx for idx in sorted(available) if available[idx]["jss"]["name"] not in ledger["segments"]]
        batch = [(idx, available[idx]) for idx in pending_indices[: args.batch]]
        if not batch:
            print(json.dumps({"ts": time.strftime("%FT%TZ", time.gmtime()), "msg": "waiting", "done_segments": len(ledger["segments"]), "available": len(available)}), flush=True)
            if args.once:
                return 0
            time.sleep(args.poll)
            continue

        # Ensure staging only holds segments belonging to the current batch.
        batch_names = {seg["jss"]["name"] for _, seg in batch}
        if staging.exists():
            for p in staging.rglob("*"):
                if p.is_file():
                    base_jss = p.name[:-13] if p.name.endswith(".receipt.json") else p.name
                    if base_jss not in batch_names:
                        p.unlink(missing_ok=True)

        t0 = time.time()
        ready_batch = []
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(download_segment, seg, staging / seg["group"]): (i, seg) for i, seg in batch}
            for fut in cf.as_completed(futures):
                i, seg = futures[fut]
                try:
                    fut.result()
                    ready_batch.append((i, seg))
                except Exception as exc:
                    name = seg["jss"]["name"]
                    for suffix in ("", ".receipt.json"):
                        (staging / seg["group"] / (name + suffix)).unlink(missing_ok=True)
                    print(json.dumps({"ts": time.strftime("%FT%TZ", time.gmtime()), "msg": "deferring_segment_until_drivefs_flush", "segment": name, "error": str(exc)[:200]}), flush=True)
        ready_batch.sort(key=lambda x: x[0])
        batch = ready_batch
        if not batch:
            time.sleep(30)
            continue
        dl_s = time.time() - t0
        dl_bytes = sum(int(seg["jss"]["size"]) for _, seg in batch)

        if decoded.exists():
            shutil.rmtree(decoded)
        decoded.mkdir(parents=True, exist_ok=True)

        t1 = time.time()
        cmd = [
            "node", str(DECODER),
            "--archive", str(work / "staging"),
            "--out", str(decoded),
            "--collections", ",".join(COLLECTIONS),
            "--max-events", "1000000000",
            "--max-bytes", "1000GiB",
            "--max-segments", str(len(batch)),
            "--shard-mb", "256",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(DECODER.parent))
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
            return 2
        summary = json.loads(r.stdout)
        batch_tag = f"{batch[0][0]:06d}-{batch[-1][0]:06d}"
        rows = convert_shards(decoded, events, shards_done, batch_tag)
        dec_s = time.time() - t1

        for i, seg in batch:
            name = seg["jss"]["name"]
            ledger["segments"][name] = {"index": i, "sha256": json.loads((staging / seg["group"] / (name + ".receipt.json")).read_text())["sha256"]}
            (staging / seg["group"] / name).unlink(missing_ok=True)
            (staging / seg["group"] / (name + ".receipt.json")).unlink(missing_ok=True)
        ledger["next_index"] = max(ledger.get("next_index", 0), batch[-1][0] + 1)
        ledger["shards_done"] = sorted(shards_done)
        tmp = ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger))
        os.replace(tmp, ledger_path)

        print(json.dumps({
            "ts": time.strftime("%FT%TZ", time.gmtime()),
            "segments": [b[0] for b in batch],
            "done_total": len(ledger["segments"]),
            "download_mb_s": round(dl_bytes / 1e6 / max(dl_s, 0.001), 1),
            "decode_s": round(dec_s, 1),
            "graph_rows": rows,
            "decoder_last_run": summary.get("last_run"),
        }), flush=True)
        if args.once:
            return 0


if __name__ == "__main__":
    sys.exit(main())
