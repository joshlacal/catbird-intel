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
    out = subprocess.run([GWS, *args], capture_output=True, text=True, cwd=cwd, check=True).stdout
    start = out.find("{")
    return json.loads(out[start:]) if start >= 0 else {}


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


def completed_segments() -> dict[int, dict]:
    """Map segment index -> {group, jss, receipt} for segments with receipts."""
    out: dict[int, dict] = {}
    for group in list_children(ARCHIVE_SEGMENTS_FOLDER):
        if group.get("mimeType") != "application/vnd.google-apps.folder":
            continue
        files = {f["name"]: f for f in list_children(group["id"])}
        for name, f in files.items():
            if not (name.startswith("seg_") and name.endswith(".jss")):
                continue
            receipt = files.get(name + ".receipt.json")
            if not receipt:
                continue
            index = int(name[4:-4], 36)
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
    # gws returns JSON media on stdout (it does not honor -o for JSON), so the
    # receipt is captured and written here; the binary segment uses -o.
    receipt = gws_json("drive", "files", "get", "--params", json.dumps({"fileId": seg["receipt"]["id"], "alt": "media"}))
    if receipt.get("name") != name or "sha256" not in receipt:
        raise RuntimeError(f"unexpected receipt for {name}: {str(receipt)[:200]}")
    (dest_dir / (name + ".receipt.json")).write_text(json.dumps(receipt, indent=2))
    for attempt in range(5):
        r = subprocess.run(
            [GWS, "drive", "files", "get", "--params", json.dumps({"fileId": seg["jss"]["id"], "alt": "media"}), "-o", name],
            cwd=str(dest_dir),
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and (dest_dir / name).exists():
            break
        time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"download failed for {name}: {r.stderr[-300:]}")
    got = sha256_file(dest_dir / name)
    if got != receipt.get("sha256"):
        raise RuntimeError(f"sha256 mismatch for {name}: {got} != {receipt.get('sha256')}")
    return dest_dir / name


def row_from_event(ev: dict) -> dict:
    rec = ev.get("record") if isinstance(ev.get("record"), dict) else {}
    kind = (ev.get("$type") or "").rsplit("#", 1)[-1] or ev.get("kind", "")
    collection = ev.get("collection", "")
    subject = rec.get("subject") if isinstance(rec.get("subject"), str) else ""
    extra = ""
    if collection == "app.bsky.graph.starterpack":
        extra = json.dumps([f.get("uri") for f in rec.get("feeds", []) if isinstance(f, dict)])
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


def convert_shards(decoded_dir: pathlib.Path, out_dir: pathlib.Path, done: set[str]) -> int:
    meta = json.loads((decoded_dir / "checkpoint.json").read_text())
    rows_total = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for shard in meta.get("shards", []):
        path = shard["path"]
        if path in done:
            continue
        src = decoded_dir / path
        if not src.exists():
            done.add(path)
            continue
        rows = []
        with src.open() as fh:
            for line in fh:
                if line.strip():
                    rows.append(row_from_event(json.loads(line)))
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        dst = out_dir / (pathlib.Path(path).stem + ".parquet")
        pq.write_table(table, dst, compression="zstd", compression_level=6)
        rows_total += len(rows)
        src.unlink()
        done.add(path)
    return rows_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/Users/joshlacalamito/catbird-lake-local/archive-graph")
    ap.add_argument("--batch", type=int, default=12, help="segments per decode round")
    ap.add_argument("--parallel", type=int, default=4, help="parallel downloads")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll", type=int, default=600, help="seconds to wait when no new contiguous segments")
    args = ap.parse_args()

    work = pathlib.Path(args.work)
    staging = work / "staging" / "segments"
    decoded = work / "decoded"
    events = work / "events"
    ledger_path = work / "ledger.json"
    work.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"next_index": 0, "segments": {}, "shards_done": []}
    shards_done = set(ledger["shards_done"])

    while True:
        available = completed_segments()
        batch = []
        idx = ledger["next_index"]
        while idx in available and len(batch) < args.batch:
            batch.append((idx, available[idx]))
            idx += 1
        if not batch:
            print(json.dumps({"ts": time.strftime("%FT%TZ", time.gmtime()), "msg": "waiting", "next_index": ledger["next_index"], "available": len(available)}), flush=True)
            if args.once:
                return 0
            time.sleep(args.poll)
            continue

        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(download_segment, seg, staging / seg["group"]): i for i, seg in batch}
            for fut in cf.as_completed(futures):
                fut.result()
        dl_s = time.time() - t0
        dl_bytes = sum(int(seg["jss"]["size"]) for _, seg in batch)

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
        rows = convert_shards(decoded, events, shards_done)
        dec_s = time.time() - t1

        for i, seg in batch:
            name = seg["jss"]["name"]
            ledger["segments"][name] = {"index": i, "sha256": json.loads((staging / seg["group"] / (name + ".receipt.json")).read_text())["sha256"]}
            (staging / seg["group"] / name).unlink(missing_ok=True)
            (staging / seg["group"] / (name + ".receipt.json")).unlink(missing_ok=True)
        ledger["next_index"] = batch[-1][0] + 1
        ledger["shards_done"] = sorted(shards_done)
        tmp = ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger))
        os.replace(tmp, ledger_path)

        print(json.dumps({
            "ts": time.strftime("%FT%TZ", time.gmtime()),
            "segments": [b[0] for b in batch],
            "download_mb_s": round(dl_bytes / 1e6 / max(dl_s, 0.001), 1),
            "decode_s": round(dec_s, 1),
            "graph_rows": rows,
            "decoder_last_run": summary.get("last_run"),
        }), flush=True)
        if args.once:
            return 0


if __name__ == "__main__":
    sys.exit(main())
