#!/usr/bin/env python3
"""High-throughput GCE Self-Hosted Jetstream Segment Offloader.

Watches /data/jetstream/segments for sealed .jss segments, verifies each sealed
segment's header/footer/block-CRCs + SHA256 via make_receipt.mjs, uploads
seg_XXXXXXXXXX.jss + seg_XXXXXXXXXX.jss.receipt.json in parallel (3 workers)
directly from GCE (us-west1) to Google Drive over Google's internal network,
verifies Drive size + MD5, and immediately prunes local segments older than
(active_index - 2) so the GCE disk stays under ~1.5 GB throughout the 41.9M-repo
backfill while keeping 100% of CPU available for Jetstream.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request

SEGMENTS_DIR = pathlib.Path("/data/jetstream/segments")
WORK_DIR = pathlib.Path("/data/offloader")
BACKFILL_DIR = pathlib.Path("/opt/catbird-offloader/backfill")
OAUTH_FILE = pathlib.Path("/root/.config/catbird-drive/oauth.json")

# Google Drive destination folder ID (CatbirdCommunity/archives/gce-selfhost-001/segments)
DRIVE_GCE_SEGMENTS_ROOT = "16Zd7n4pwTSDMDsnP9chX2Ci0r0ahEy45"


class DriveClient:
    def __init__(self, oauth_path: pathlib.Path) -> None:
        cfg = json.loads(oauth_path.read_text())
        self.client_id = cfg["client_id"]
        self.client_secret = cfg["client_secret"]
        self.refresh_token = cfg["refresh_token"]
        self._token = ""
        self._exp = 0.0
        self._lock = threading.Lock()
        self._group_cache: dict[str, str] = {}

    def token(self) -> str:
        with self._lock:
            now = time.time()
            if self._token and now < self._exp - 60:
                return self._token
            body = urllib.parse.urlencode(
                {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                }
            ).encode()
            req = urllib.request.Request(
                "https://oauth2.googleapis.com/token",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            self._token = data["access_token"]
            self._exp = now + float(data.get("expires_in", 3600))
            return self._token

    def ensure_subfolder(self, parent_id: str, name: str) -> str:
        cache_key = f"{parent_id}/{name}"
        with self._lock:
            if cache_key in self._group_cache:
                return self._group_cache[cache_key]
        q = (
            f"'{parent_id}' in parents and name = '{name}' "
            f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        params = urllib.parse.urlencode({"q": q, "fields": "files(id,name)"})
        req = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files?{params}",
            headers={"Authorization": f"Bearer {self.token()}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            files = json.loads(resp.read().decode()).get("files", [])
        if files:
            fid = files[0]["id"]
            with self._lock:
                self._group_cache[cache_key] = fid
            return fid
        meta = json.dumps(
            {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_id],
            }
        ).encode()
        req = urllib.request.Request(
            "https://www.googleapis.com/drive/v3/files?fields=id,name",
            data=meta,
            headers={
                "Authorization": f"Bearer {self.token()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            fid = json.loads(resp.read().decode())["id"]
        with self._lock:
            self._group_cache[cache_key] = fid
        return fid

    def upload_file_verified(
        self,
        local_path: pathlib.Path,
        parent_id: str,
        remote_name: str | None = None,
        mime_type: str = "application/octet-stream",
    ) -> dict:
        name = remote_name or local_path.name
        size = local_path.stat().st_size
        md5 = hashlib.md5()
        with local_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                md5.update(chunk)
        expected_md5 = md5.hexdigest()

        q = f"'{parent_id}' in parents and name = '{name}' and trashed = false"
        params = urllib.parse.urlencode({"q": q, "fields": "files(id,name,size,md5Checksum)"})
        req = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files?{params}",
            headers={"Authorization": f"Bearer {self.token()}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            existing = json.loads(resp.read().decode()).get("files", [])
        for f in existing:
            if int(f.get("size") or -1) == size and f.get("md5Checksum") == expected_md5:
                return f

        meta = json.dumps({"name": name, "parents": [parent_id]}).encode()
        init_req = urllib.request.Request(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,name,size,md5Checksum",
            data=meta,
            headers={
                "Authorization": f"Bearer {self.token()}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Type": mime_type,
                "X-Upload-Content-Length": str(size),
            },
            method="POST",
        )
        with urllib.request.urlopen(init_req, timeout=30) as resp:
            upload_url = resp.headers["Location"]

        with local_path.open("rb") as fh:
            data_bytes = fh.read()
        put_req = urllib.request.Request(
            upload_url,
            data=data_bytes,
            headers={
                "Content-Length": str(size),
                "Content-Type": mime_type,
            },
            method="PUT",
        )
        with urllib.request.urlopen(put_req, timeout=300) as resp:
            uploaded = json.loads(resp.read().decode())

        if int(uploaded.get("size") or -1) != size or uploaded.get("md5Checksum") != expected_md5:
            raise RuntimeError(
                f"Drive upload verification failed for {name}: got {uploaded}, expected size={size} md5={expected_md5}"
            )
        return uploaded


def offload_one(seg_path: pathlib.Path, index: int, drive: DriveClient) -> dict:
    t0 = time.time()
    stage_dir = WORK_DIR / f"stage_{index:06d}"
    shutil.rmtree(stage_dir, ignore_errors=True)
    group_name = f"{index // 256:08x}"
    stage_group = stage_dir / "segments" / group_name
    stage_group.mkdir(parents=True, exist_ok=True)
    staged_jss = stage_group / seg_path.name
    os.link(seg_path, staged_jss)

    r_rec = subprocess.run(
        ["node", str(BACKFILL_DIR / "make_receipt.mjs"), str(staged_jss)],
        cwd=str(BACKFILL_DIR),
        capture_output=True,
        text=True,
        check=True,
    )
    receipt = json.loads(r_rec.stdout.strip().splitlines()[-1])
    receipt_path = stage_group / f"{seg_path.name}.receipt.json"
    t_verify = time.time() - t0

    t_up0 = time.time()
    drive_group_id = drive.ensure_subfolder(DRIVE_GCE_SEGMENTS_ROOT, group_name)
    up_jss = drive.upload_file_verified(staged_jss, drive_group_id, seg_path.name)
    up_rec = drive.upload_file_verified(
        receipt_path, drive_group_id, receipt_path.name, mime_type="application/json"
    )
    t_upload = time.time() - t_up0
    shutil.rmtree(stage_dir, ignore_errors=True)

    return {
        "index": index,
        "name": seg_path.name,
        "bytes": receipt["bytes"],
        "sha256": receipt["sha256"],
        "first_seq": receipt.get("first_seq"),
        "last_seq": receipt.get("last_seq"),
        "verify_s": round(t_verify, 2),
        "upload_s": round(t_upload, 2),
        "upload_mb_s": round((receipt["bytes"] / (1024 * 1024)) / max(t_upload, 0.01), 2),
        "drive_jss_id": up_jss["id"],
        "drive_receipt_id": up_rec["id"],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def prune_local(segs: list[tuple[int, pathlib.Path]], max_idx: int, state: dict) -> None:
    for idx, p in segs:
        if idx <= max_idx - 3 and str(idx) in state["uploaded"] and p.exists():
            p.unlink(missing_ok=True)
            print(
                json.dumps(
                    {
                        "event": "local_segment_pruned",
                        "index": idx,
                        "name": p.name,
                        "kept_from_index": max_idx - 2,
                    }
                ),
                flush=True,
            )


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    state_file = WORK_DIR / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"uploaded": {}}
    state_lock = threading.Lock()
    drive = DriveClient(OAUTH_FILE)

    while True:
        try:
            segs: list[tuple[int, pathlib.Path]] = []
            for p in SEGMENTS_DIR.glob("seg_*.jss"):
                if p.name.endswith(".receipt.json"):
                    continue
                idx = int(p.name[4:-4], 36)
                segs.append((idx, p))
            segs.sort(key=lambda x: x[0])

            if len(segs) >= 2:
                max_idx = segs[-1][0]
                prune_local(segs, max_idx, state)

                pending = [(idx, p) for idx, p in segs[:-1] if str(idx) not in state["uploaded"]]
                if pending:
                    batch = pending[:6]
                    with cf.ThreadPoolExecutor(max_workers=3) as pool:
                        futs = {pool.submit(offload_one, p, idx, drive): (idx, p) for idx, p in batch}
                        for fut in cf.as_completed(futs):
                            idx, p = futs[fut]
                            info = fut.result()
                            with state_lock:
                                state["uploaded"][str(idx)] = info
                                state_file.write_text(json.dumps(state, indent=2))
                            print(json.dumps({"event": "segment_offloaded", **info}), flush=True)
                            # Re-check max_idx and prune immediately
                            cur_segs = [
                                (int(q.name[4:-4], 36), q)
                                for q in SEGMENTS_DIR.glob("seg_*.jss")
                                if not q.name.endswith(".receipt.json")
                            ]
                            if cur_segs:
                                cur_max = max(x[0] for x in cur_segs)
                                prune_local(cur_segs, cur_max, state)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "offloader_error",
                        "error": str(exc),
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                ),
                flush=True,
            )
        time.sleep(3)


if __name__ == "__main__":
    main()
