#!/usr/bin/env python3
"""Upload purged S12m vector buckets, ledger, manifests, and purge receipt to Google Drive via `gdrive`.

Safety & integrity rules:
  - Creates a dedicated new folder (`catbird-s12m-vectors-20261029a-clean`) on Google Drive; NEVER touches
    `.archive.lock` or any `catbird-overflow-*` folder, and never changes sharing permissions.
  - Packs each of the 64 DID buckets (`u-b0000-*.parquet` .. `u-b0063-*.parquet`, already zstd-compressed)
    into a deterministic tar in `~/catbird-lake-local/_scratch-slot/` (reusing worker slot files under D14c),
    uploads via `gdrive mutate upload --parent <FOLDER_ID> --json`, verifies Drive's reported byte size matches
    the local byte size, and checkpoints every uploaded file in `drive_upload_ledger.json` for exact resume.
  - Also uploads `metadata_and_ledger.tar.gz` (`ledger/`, `vectors/receipt/`, `purge_receipt_20261030.json`,
    `s12m-units-s9/manifest.json`, `s12m-units-s9/split_receipt.json`, `gate_manifest.json`) and
    `purge_receipt_20261030.json` directly for quick inspection on Drive.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import threading
import time

GDRIVE_BIN = "gdrive"
BUCKET_RE = re.compile(r"^u-(b\d{4})-\d{4}-\d+\.parquet$")
DRIVE_ID_RE = re.compile(r"\bID:\s*([A-Za-z0-9_-]{10,})(?=\s|$|[),.;])")


def hashes_file(path: pathlib.Path) -> tuple[str, str]:
    """Return (sha256_hex, md5_hex) in a single streaming pass."""
    h_sha = hashlib.sha256()
    h_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h_sha.update(b)
            h_md5.update(b)
    return h_sha.hexdigest(), h_md5.hexdigest()


def gdrive_readonly(args: list[str], retries: int = 4) -> dict | list:
    cmd = [GDRIVE_BIN, "readonly", *args, "--json"]
    last_err = ""
    for attempt in range(retries):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            return json.loads(p.stdout)
        last_err = (p.stderr or p.stdout).strip()
        time.sleep(min(30, 2 ** (attempt + 1)))
    raise RuntimeError(f"gdrive readonly {' '.join(args)} failed after {retries} tries: {last_err}")


def _extract_drive_id(stdout: str) -> str:
    s = stdout.strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and obj.get("id"):
                return str(obj["id"])
        except json.JSONDecodeError:
            pass
    m = DRIVE_ID_RE.search(s)
    if m:
        return m.group(1)
    raise RuntimeError(f"Could not extract Drive ID from gdrive mutate output: {s!r}")


def gdrive_mutate_verified(args: list[str], retries: int = 4) -> dict:
    """Run `gdrive mutate ...`, extract the created/uploaded Drive ID, and return `gdrive readonly info <id> --json`."""
    cmd = [GDRIVE_BIN, "mutate", *args, "--json"]
    last_err = ""
    for attempt in range(retries):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            fid = _extract_drive_id(p.stdout or p.stderr)
            info = gdrive_readonly(["info", fid])
            assert isinstance(info, dict) and info.get("id") == fid, f"Unexpected info response: {info}"
            return info
        last_err = (p.stderr or p.stdout).strip()
        time.sleep(min(30, 2 ** (attempt + 1)))
    raise RuntimeError(f"gdrive mutate {' '.join(args)} failed after {retries} tries: {last_err}")


def ensure_folder(name: str, parent_id: str | None = None) -> str:
    ls_args = ["ls", *([parent_id] if parent_id else []), "--max", "200"]
    res = gdrive_readonly(ls_args)
    files = res.get("files") if isinstance(res, dict) else res
    if isinstance(files, list):
        for f in files:
            if f.get("name") == name and f.get("mimeType") == "application/vnd.google-apps.folder":
                return f["id"]
    mk_args = ["mkdir", name]
    if parent_id:
        mk_args += ["--parent", parent_id]
    created = gdrive_mutate_verified(mk_args)
    return created["id"]


def find_existing_verified(name: str, parent_id: str, local_bytes: int, l_md5: str) -> dict | None:
    res = gdrive_readonly(["ls", parent_id, "--max", "200"])
    files = res.get("files") if isinstance(res, dict) else res
    if not isinstance(files, list):
        return None
    for f in files:
        if f.get("name") == name and int(f.get("size", -1)) == local_bytes:
            info = gdrive_readonly(["info", f["id"]])
            if isinstance(info, dict) and int(info.get("size", -1)) == local_bytes:
                d_md5 = info.get("md5Checksum")
                if not d_md5 or d_md5.lower() == l_md5.lower():
                    return info
    return None


def verify_upload(local_path: pathlib.Path, parent_id: str) -> dict:
    local_bytes = local_path.stat().st_size
    l_sha256, l_md5 = hashes_file(local_path)
    up = find_existing_verified(local_path.name, parent_id, local_bytes, l_md5)
    if up is None:
        up = gdrive_mutate_verified(["upload", str(local_path), "--parent", parent_id])
    d_size = int(up.get("size", -1))
    d_md5 = up.get("md5Checksum")
    if d_size != local_bytes:
        raise RuntimeError(f"Drive size mismatch on {local_path.name}: got {d_size}, expected {local_bytes}")
    if d_md5 and d_md5.lower() != l_md5.lower():
        raise RuntimeError(f"Drive md5 mismatch on {local_path.name}: got {d_md5}, expected {l_md5}")
    return {
        "drive_id": up["id"],
        "bytes": local_bytes,
        "drive_size": d_size,
        "sha256": l_sha256,
        "md5": l_md5,
        "drive_md5": d_md5,
        "web_view_link": up.get("webViewLink"),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="~/catbird-lake-local/s12mm-colab-s12m-20261029a")
    ap.add_argument("--units-dir", default="~/catbird-lake-local/s12m-units-s9")
    ap.add_argument("--gate-dir", default="~/catbird-lake-local/s12mm-label-chain-s12m-20261029a/gate")
    ap.add_argument("--scratch-dir", default="~/catbird-lake-local/_scratch-slot/gdrive_upload_s12m")
    ap.add_argument("--folder-name", default="catbird-s12m-vectors-20261029a-clean")
    ap.add_argument("--folder-id", default="1e3M9gzFo0zwYIQQxOlJhMnP4WCkXQ2jm")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)

    if not args.folder_name.startswith("catbird-s12m-"):
        raise SystemExit("Refusing unexpected Drive folder name")

    t0 = time.time()
    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    units_dir = pathlib.Path(args.units_dir).expanduser().resolve()
    gate_dir = pathlib.Path(args.gate_dir).expanduser().resolve()
    scratch_dir = pathlib.Path(args.scratch_dir).expanduser().resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)

    purge_receipt = run_dir / "purge_receipt_20261030.json"
    if not purge_receipt.exists():
        raise SystemExit(f"Missing {purge_receipt}: run purge_s12m.py before uploading to Google Drive!")

    ledger_path = run_dir / "drive_upload_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"uploaded": {}}
    lock = threading.Lock()

    if ledger.get("folder_id"):
        root_id = ledger["folder_id"]
    elif args.folder_id:
        info = gdrive_readonly(["info", args.folder_id])
        assert isinstance(info, dict) and info.get("name") == args.folder_name, f"Folder ID mismatch: {info}"
        root_id = args.folder_id
    else:
        root_id = ensure_folder(args.folder_name)

    buckets_id = ledger.get("buckets_folder_id") or ensure_folder("vectors_buckets", parent_id=root_id)
    ledger["folder_name"] = args.folder_name
    ledger["folder_id"] = root_id
    ledger["buckets_folder_id"] = buckets_id

    def save_ledger() -> None:
        tmp = ledger_path.with_name("drive_upload_ledger.json.tmp")
        tmp.write_text(json.dumps(ledger, indent=1) + "\n")
        os.replace(tmp, ledger_path)

    save_ledger()

    # 1. Upload purge_receipt_20261030.json + metadata_and_ledger.tar.gz to root_id
    meta_tar = scratch_dir / "metadata_and_ledger.tar.gz"
    if "metadata_and_ledger.tar.gz" not in ledger["uploaded"]:
        with tarfile.open(meta_tar, "w:gz") as tf:
            tf.add(purge_receipt, arcname="purge_receipt_20261030.json")
            if (run_dir / "ledger").exists():
                tf.add(run_dir / "ledger", arcname="ledger")
            if (run_dir / "vectors/receipt").exists():
                tf.add(run_dir / "vectors/receipt", arcname="vectors_receipt")
            if (units_dir / "manifest.json").exists():
                tf.add(units_dir / "manifest.json", arcname="units_manifest.json")
            if (units_dir / "split_receipt.json").exists():
                tf.add(units_dir / "split_receipt.json", arcname="split_receipt.json")
            if (gate_dir / "gate_manifest.json").exists():
                tf.add(gate_dir / "gate_manifest.json", arcname="gate_manifest.json")
        ledger["uploaded"]["metadata_and_ledger.tar.gz"] = verify_upload(meta_tar, root_id)
        save_ledger()

    if "purge_receipt_20261030.json" not in ledger["uploaded"]:
        ledger["uploaded"]["purge_receipt_20261030.json"] = verify_upload(purge_receipt, root_id)
        save_ledger()

    # 2. Group the 1,853 clean vector parquets into 64 bucket tars (`b0000.tar` .. `b0063.tar`)
    vfiles = sorted((run_dir / "vectors/units").glob("u-*.parquet"))
    by_bucket: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
    for f in vfiles:
        m = BUCKET_RE.match(f.name)
        if not m:
            raise SystemExit(f"unexpected vector filename: {f.name}")
        by_bucket[m.group(1)].append(f)

    todo_buckets = [b for b in sorted(by_bucket) if f"{b}.tar" not in ledger["uploaded"]]
    print(f"Drive folder={root_id} buckets_folder={buckets_id} total_buckets={len(by_bucket)} todo={len(todo_buckets)}", flush=True)

    slot_q = collections.deque(range(args.workers))
    slot_lk = threading.Lock()

    def upload_bucket(b: str) -> dict:
        with slot_lk:
            slot = slot_q.popleft()
        try:
            slot_dir = scratch_dir / f"slot_{slot}"
            slot_dir.mkdir(parents=True, exist_ok=True)
            tar_path = slot_dir / f"{b}.tar"
            # Write uncompressed tar over existing slot file (truncate/overwrite under D14c scratch-slot rule)
            with tarfile.open(tar_path, "w") as tf:
                for vf in by_bucket[b]:
                    tf.add(vf, arcname=vf.name)
            entry = verify_upload(tar_path, buckets_id)
            entry["units"] = len(by_bucket[b])
            # Truncate scratch slot file to 0 bytes to keep disk free without rm
            with open(tar_path, "wb"):
                pass
            with lock:
                ledger["uploaded"][f"{b}.tar"] = entry
                save_ledger()
                done_n = sum(1 for k in ledger["uploaded"] if k.startswith("b") and k.endswith(".tar"))
                print(
                    f"[{done_n}/{len(by_bucket)}] uploaded {b}.tar "
                    f"({entry['bytes'] / (1<<20):.1f} MiB) id={entry['drive_id']}",
                    flush=True,
                )
            return entry
        finally:
            with slot_lk:
                slot_q.append(slot)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(upload_bucket, b): b for b in todo_buckets}
        for fut in concurrent.futures.as_completed(futs):
            fut.result()

    # Truncate meta_tar scratch copy too
    if meta_tar.exists():
        with open(meta_tar, "wb"):
            pass

    # 3. Verify remote listing in buckets_id has all 64 bucket tars and upload final manifest
    remote_list = gdrive_readonly(["ls", buckets_id, "--max", "200"])
    r_files = remote_list.get("files", []) if isinstance(remote_list, dict) else remote_list
    bucket_entries = {k: v for k, v in ledger["uploaded"].items() if k.startswith("b") and k.endswith(".tar")}
    total_bucket_bytes = sum(v["bytes"] for v in bucket_entries.values())
    total_units = sum(v["units"] for v in bucket_entries.values())
    assert len(bucket_entries) == len(by_bucket) and total_units == len(vfiles), "Incomplete bucket upload!"

    manifest = {
        "step": "S12m clean vectors Google Drive backup (Josh storage override 2026-10-30)",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t0, 1),
        "drive_folder_name": args.folder_name,
        "drive_folder_id": root_id,
        "drive_folder_url": f"https://drive.google.com/drive/folders/{root_id}",
        "drive_buckets_folder_id": buckets_id,
        "totals": {
            "bucket_tars": len(bucket_entries),
            "vector_unit_files": total_units,
            "bucket_tar_bytes": total_bucket_bytes,
            "all_uploaded_bytes": sum(v["bytes"] for v in ledger["uploaded"].values()),
            "remote_buckets_listed": len(r_files) if isinstance(r_files, list) else None,
        },
        "files": ledger["uploaded"],
    }
    man_local = run_dir / "drive_upload_receipt_20261030.json"
    man_local.write_text(json.dumps(manifest, indent=1) + "\n")
    up_man = verify_upload(man_local, root_id)
    manifest["drive_receipt_file_id"] = up_man["drive_id"]
    man_local.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({
        "drive_folder_id": root_id,
        "drive_folder_url": manifest["drive_folder_url"],
        "bucket_tars": len(bucket_entries),
        "vector_unit_files": total_units,
        "total_gib": round(manifest["totals"]["all_uploaded_bytes"] / (1 << 30), 3),
        "wall_seconds": manifest["wall_seconds"],
    }, indent=1))


if __name__ == "__main__":
    main()
