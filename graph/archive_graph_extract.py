#!/usr/bin/env python3
"""Stream completed Jetstream archive segments from Google Drive (or, with
--gcs-prefix, the GCE capture's Cloud Storage buffer), decode only the selected
collections, and write compact Parquet for Phase 1 community discovery.

Read-only on Drive (never touches .archive.lock or the downloader's files).
Segments are processed in strict index order because the decoder enforces a
monotonic sequence. Each raw .jss is deleted locally after successful decode;
only the extracted graph events are kept.

Usage:
  archive_graph_extract.py [--work DIR] [--batch N] [--parallel N] [--once]
  archive_graph_extract.py --gcs-prefix gs://BUCKET/sources/SRC/EPOCH --min-index N \
      --work DIR --decoder PATH/decode_segments.mjs
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

import gcs_segments
from post_extract import POST_COLLECTION, POST_SCHEMA, post_row_from_event

GWS = os.environ.get("CATBIRD_GWS", "/usr/local/bin/gws")
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

_ACTIVE_COLLECTIONS = set(COLLECTIONS)

SOURCE_IDS = (
    "public_archive",
    "gce_main",
    "gce_bootstrap_live",
    "netcup_live",
    "repo_supplement",
)

_SOURCE_ALIASES = {
    "public_archive": "public_archive",
    "public-archive": "public_archive",
    "archive_graph": "public_archive",
    "archive-graph": "public_archive",
    "gce_main": "gce_main",
    "gce-main": "gce_main",
    "gce_archive_graph": "gce_main",
    "gce-archive-graph": "gce_main",
    "october": "gce_main",
    "gce_bootstrap_live": "gce_bootstrap_live",
    "gce-bootstrap-live": "gce_bootstrap_live",
    "bootstrap_live": "gce_bootstrap_live",
    "bootstrap-live": "gce_bootstrap_live",
    "netcup_live": "netcup_live",
    "netcup-live": "netcup_live",
    "netcup": "netcup_live",
    "repo_supplement": "repo_supplement",
}

_TID_ALPHABET = "234567abcdefghijklmnopqrstuvwxyz"
_TID_LOOKUP = {ch: idx for idx, ch in enumerate(_TID_ALPHABET)}

SCHEMA = pa.schema(
    [
        ("source_id", pa.string()),
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


def normalize_source_id(value: str | None, default: str = "public_archive") -> str:
    if value is None or not str(value).strip():
        return default
    key = str(value).strip().lower()
    if key in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[key]
    raise ValueError(f"Unsupported source_id {value!r}; expected one of {SOURCE_IDS}")


def infer_source_id(
    work: pathlib.Path | str = "",
    segments_folder: str = ARCHIVE_SEGMENTS_FOLDER,
    explicit: str | None = None,
) -> str:
    if explicit:
        return normalize_source_id(explicit)
    w = str(work).lower()
    if "bootstrap_live" in w or "bootstrap-live" in w:
        return "gce_bootstrap_live"
    if "netcup" in w:
        return "netcup_live"
    if "gce" in w or (segments_folder and segments_folder != ARCHIVE_SEGMENTS_FOLDER):
        return "gce_main"
    return "public_archive"


def tid_from_uint64(value: int) -> str:
    if value < 0 or value >= (1 << 64):
        raise ValueError(f"TID uint64 out of range: {value}")
    chars = [_TID_ALPHABET[(value >> ((12 - i) * 5)) & 0x1F] for i in range(13)]
    return "".join(chars)


def uint64_from_tid(tid: str) -> int:
    if len(tid) != 13 or any(ch not in _TID_LOOKUP for ch in tid):
        raise ValueError(f"Invalid 13-char base32-sortable TID: {tid!r}")
    val = 0
    for ch in tid:
        val = (val << 5) | _TID_LOOKUP[ch]
    return val




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


def should_keep_event(ev: dict) -> bool:
    kind = (ev.get("$type") or "").rsplit("#", 1)[-1] or ev.get("kind", "")
    if not kind:
        if isinstance(ev.get("sync"), dict):
            kind = "sync"
        elif isinstance(ev.get("account"), dict):
            kind = "account"
        elif isinstance(ev.get("identity"), dict):
            kind = "identity"
        elif isinstance(ev.get("commit"), dict):
            kind = "commit"
    if kind in ("sync", "account", "identity"):
        return True
    commit_obj = ev.get("commit") if isinstance(ev.get("commit"), dict) else {}
    collection = ev.get("collection") or commit_obj.get("collection") or ""
    return kind == "commit" and collection in _ACTIVE_COLLECTIONS


def row_from_event(ev: dict, source_id: str | None = None) -> dict:
    resolved_source = normalize_source_id(
        ev.get("source_id") if isinstance(ev.get("source_id"), str) and ev.get("source_id") else source_id,
        default="public_archive",
    )
    commit_obj = ev.get("commit") if isinstance(ev.get("commit"), dict) else {}
    sync_obj = ev.get("sync") if isinstance(ev.get("sync"), dict) else {}
    account = ev.get("account") if isinstance(ev.get("account"), dict) else {}
    identity = ev.get("identity") if isinstance(ev.get("identity"), dict) else {}

    rec = ev.get("record") if isinstance(ev.get("record"), dict) else (
        commit_obj.get("record") if isinstance(commit_obj.get("record"), dict) else {}
    )
    kind = (ev.get("$type") or "").rsplit("#", 1)[-1] or ev.get("kind", "")
    if not kind:
        if sync_obj:
            kind = "sync"
        elif account:
            kind = "account"
        elif identity:
            kind = "identity"
        elif commit_obj:
            kind = "commit"

    collection = ev.get("collection") or commit_obj.get("collection") or ""
    rkey = ev.get("rkey") or commit_obj.get("rkey") or ""
    operation = ev.get("operation") or commit_obj.get("operation") or ""
    raw_rev = (
        ev.get("rev")
        or commit_obj.get("rev")
        or sync_obj.get("rev")
        or account.get("rev")
        or ""
    )
    if isinstance(raw_rev, int):
        rev = tid_from_uint64(raw_rev) if raw_rev > 0 else ""
    elif isinstance(raw_rev, str):
        rev = raw_rev
    else:
        rev = ""

    did = (
        ev.get("did")
        or sync_obj.get("did")
        or account.get("did")
        or identity.get("did")
        or ""
    )
    time_val = (
        ev.get("time")
        or sync_obj.get("time")
        or account.get("time")
        or identity.get("time")
        or ""
    )

    raw_subject = rec.get("subject")
    if isinstance(raw_subject, dict):  # likes and reposts reference a post by strongRef
        raw_subject = raw_subject.get("uri")
    subject = raw_subject if isinstance(raw_subject, str) else ""
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

    account_active = None
    if account and "active" in account and account.get("active") is not None:
        account_active = bool(account.get("active"))
    account_status = account.get("status", "") if account and isinstance(account.get("status"), str) else ""

    return {
        "source_id": resolved_source,
        "seq": int(ev.get("seq") or 0),
        "time": time_val,
        "kind": kind,
        "did": did,
        "collection": collection,
        "rkey": rkey,
        "rev": rev,
        "operation": operation,
        "created_at": rec.get("createdAt", "") if isinstance(rec.get("createdAt"), str) else "",
        "subject": subject,
        "list_uri": rec.get("list", "") if isinstance(rec.get("list"), str) else "",
        "name": (rec.get("name") or rec.get("displayName") or "")[:512] if isinstance(rec.get("name") or rec.get("displayName") or "", str) else "",
        "description": (rec.get("description") or "")[:2000] if isinstance(rec.get("description") or "", str) else "",
        "extra": extra,
        "account_active": account_active,
        "account_status": account_status,
    }


def _iso_utc(val: object) -> str:
    if val is None:
        return ""
    if hasattr(val, "isoformat"):
        s = val.isoformat()
        if s.endswith("+00:00"):
            return s[:-6] + "Z"
        return s if s.endswith("Z") else s + "Z"
    return str(val)


def rows_from_netcup_export(
    export_dir: pathlib.Path,
    source_id: str = "netcup_live",
    collections: set[str] | None = None,
) -> list[dict]:
    resolved_source = normalize_source_id(source_id, default="netcup_live")
    allowed = collections if collections is not None else set(COLLECTIONS)
    rows: list[dict] = []

    deletions_path = export_dir / "deletions.parquet"
    if deletions_path.exists():
        del_table = pq.read_table(deletions_path)
        for item in del_table.to_pylist():
            coll = item.get("collection") or ""
            if allowed and coll not in allowed:
                continue
            raw_rev = int(item.get("rev") or 0)
            rows.append(
                row_from_event(
                    {
                        "source_id": resolved_source,
                        "seq": int(item.get("seq") or 0),
                        "time": _iso_utc(item.get("indexed_at")),
                        "kind": "commit",
                        "did": item.get("did") or "",
                        "collection": coll,
                        "rkey": item.get("rkey") or "",
                        "rev": tid_from_uint64(raw_rev) if raw_rev > 0 else "",
                        "operation": "delete",
                    },
                    source_id=resolved_source,
                )
            )

    status_path = export_dir / "account_status.parquet"
    if status_path.exists():
        st_table = pq.read_table(status_path)
        for item in st_table.to_pylist():
            rows.append(
                row_from_event(
                    {
                        "source_id": resolved_source,
                        "seq": int(item.get("seq") or 0),
                        "time": _iso_utc(item.get("indexed_at")),
                        "kind": "account",
                        "did": item.get("did") or "",
                        "account": {
                            "active": bool(int(item.get("active") or 0)),
                            "status": item.get("status") or "",
                        },
                    },
                    source_id=resolved_source,
                )
            )
    return rows


def post_row(ev: dict, source_id: str) -> dict | None:
    """Post row for a post commit; None for anything else (sync/account/identity
    events stay in the graph extraction)."""
    base = row_from_event(ev, source_id=source_id)
    if base["kind"] != "commit" or base["collection"] != POST_COLLECTION:
        return None
    commit_obj = ev.get("commit") if isinstance(ev.get("commit"), dict) else {}
    rec = ev.get("record") if isinstance(ev.get("record"), dict) else commit_obj.get("record")
    return post_row_from_event({**base, "record": rec})


def convert_shards(
    decoded_dir: pathlib.Path,
    out_dir: pathlib.Path,
    done: set[str],
    batch_tag: str,
    source_id: str = "public_archive",
    posts: bool = False,
) -> int:
    resolved_source = normalize_source_id(source_id, default="public_archive")
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
                    ev = json.loads(line)
                    if posts:
                        row = post_row(ev, resolved_source)
                        if row is not None:
                            rows.append(row)
                    elif should_keep_event(ev):
                        rows.append(row_from_event(ev, source_id=resolved_source))
        table = pa.Table.from_pylist(rows, schema=POST_SCHEMA if posts else SCHEMA)
        dst = out_dir / f"{'posts' if posts else 'events'}-{batch_tag}-{idx:02d}-{sha}.parquet"
        pq.write_table(table, dst, compression="zstd", compression_level=6)
        rows_total += len(rows)
        src.unlink()
        done.add(sha)
    return rows_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/Users/joshlacalamito/catbird-lake-local/archive-graph")
    ap.add_argument("--segments-folder", default=ARCHIVE_SEGMENTS_FOLDER, help="Google Drive segments folder ID")
    ap.add_argument(
        "--source-id",
        default=None,
        help="Canonical source_id (public_archive, gce_main, gce_bootstrap_live, netcup_live)",
    )
    ap.add_argument("--gcs-prefix", default=None,
                    help="read segments from this Cloud Storage buffer prefix instead of Drive "
                         "(gs://bucket/sources/<source>/<epoch>); run it inside us-west1")
    ap.add_argument("--min-index", type=int, default=0, help="skip segments below this index")
    ap.add_argument("--max-index", type=int, default=None, help="skip segments above this index")
    ap.add_argument("--collections", default=",".join(COLLECTIONS),
                    help="comma-separated NSIDs to keep (add app.bsky.feed.like,app.bsky.feed.repost for engagement)")
    ap.add_argument("--content", choices=("graph", "posts"), default="graph",
                    help="graph: graph/actor collections in SCHEMA; posts: app.bsky.feed.post rows in POST_SCHEMA")
    ap.add_argument("--decoder", default=str(DECODER), help="path to decode_segments.mjs")
    ap.add_argument("--batch", type=int, default=12, help="segments per decode round")
    ap.add_argument("--parallel", type=int, default=4, help="parallel downloads")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--static-listing", action="store_true",
                    help="list segments once and reuse it (a closed range; avoids re-listing Drive every batch)")
    ap.add_argument("--poll", type=int, default=600, help="seconds to wait when no new segments")
    args = ap.parse_args()

    global _ACTIVE_COLLECTIONS
    posts = args.content == "posts"
    collections = [POST_COLLECTION] if posts else [c for c in args.collections.split(",") if c]
    _ACTIVE_COLLECTIONS = set(collections)
    decoder = pathlib.Path(args.decoder)
    work = pathlib.Path(args.work)
    if args.gcs_prefix:
        source_id = normalize_source_id(args.source_id or "gce_main")
        list_segments = lambda: gcs_segments.completed_segments(args.gcs_prefix)
        fetch_segment = gcs_segments.download_segment
    else:
        source_id = infer_source_id(work, args.segments_folder, args.source_id)
        list_segments = lambda: completed_segments(args.segments_folder)
        fetch_segment = download_segment
    if args.static_listing:
        listed = list_segments()
        list_segments = lambda: listed
    staging = work / "staging" / "segments"
    decoded = work / "decoded_batch"
    events = work / "events"
    ledger_path = work / "ledger.json"
    work.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"next_index": 0, "segments": {}, "shards_done": []}
    shards_done = set(ledger["shards_done"])

    while True:
        available = {idx: seg for idx, seg in list_segments().items()
                     if idx >= args.min_index and (args.max_index is None or idx <= args.max_index)}
        pending_indices = [idx for idx in sorted(available) if available[idx]["jss"]["name"] not in ledger["segments"]]
        batch = [(idx, available[idx]) for idx in pending_indices[: args.batch]]
        if not batch:
            print(json.dumps({"ts": time.strftime("%FT%TZ", time.gmtime()), "msg": "waiting", "source_id": source_id, "done_segments": len(ledger["segments"]), "available": len(available)}), flush=True)
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
            futures = {pool.submit(fetch_segment, seg, staging / seg["group"]): (i, seg) for i, seg in batch}
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
            "node", str(decoder),
            "--archive", str(work / "staging"),
            "--out", str(decoded),
            "--collections", ",".join(collections),
            "--max-events", "1000000000",
            "--max-bytes", "1000GiB",
            "--max-segments", str(len(batch)),
            "--shard-mb", "256",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(decoder.parent))
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
            return 2
        summary = json.loads(r.stdout)
        batch_tag = f"{batch[0][0]:06d}-{batch[-1][0]:06d}"
        rows = convert_shards(decoded, events, shards_done, batch_tag, source_id=source_id, posts=posts)
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
            "source_id": source_id,
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

