#!/usr/bin/env python3
"""Upload sealed daily ClickHouse Parquet exports from netcup to Google Drive.

Target Drive folder: My Drive/catbird-intel-hot-export, created by this app on first run.
  The token is minted by mint_drive_token.py with only the drive.file scope, so the app sees
  and writes only files it created itself, never the rest of the Drive. Set
  HOT_EXPORT_FOLDER_ID to target an existing folder instead (needs a full-Drive token).
Subfolder per day:   day=YYYY-MM-DD

Reads OAuth credentials from DRIVE_OAUTH_PATH (default ~/.config/catbird-intel/drive-oauth.json).
Verifies every file in the local day directory exists on Drive with matching size before writing
the `.uploaded` marker.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HOT_EXPORT_FOLDER_ID = os.environ.get("HOT_EXPORT_FOLDER_ID", "")
HOT_EXPORT_ROOT_NAME = os.environ.get("HOT_EXPORT_ROOT_NAME", "catbird-intel-hot-export")
DEFAULT_OAUTH_PATH = os.path.expanduser("~/.config/catbird-intel/drive-oauth.json")
CHUNK_SIZE = 32 * 1024 * 1024  # 32 MiB


class DriveClient:
    def __init__(self, oauth_path: str) -> None:
        cfg = json.loads(pathlib.Path(oauth_path).read_text())
        self.client_id = cfg["client_id"]
        self.client_secret = cfg["client_secret"]
        self.refresh_token = cfg["refresh_token"]
        self._token = ""
        self._exp = 0.0
        self._lock = threading.Lock()

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
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            self._token = data["access_token"]
            self._exp = now + float(data.get("expires_in", 3600))
            return self._token

    def _api(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        for attempt in range(5):
            hdrs = {"Authorization": f"Bearer {self.token()}"}
            if headers:
                hdrs.update(headers)
            req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read()
                    return json.loads(raw.decode()) if raw else {}
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(2**attempt)
                    continue
                raise

    def list_children(self, parent_id: str) -> dict[str, dict]:
        q = f"'{parent_id}' in parents and trashed = false"
        params = urllib.parse.urlencode(
            {"q": q, "fields": "files(id,name,size,mimeType)", "pageSize": 200}
        )
        res = self._api("GET", f"https://www.googleapis.com/drive/v3/files?{params}")
        out: dict[str, dict] = {}
        for f in res.get("files", []):
            out[f["name"]] = f
        return out

    def ensure_folder(self, parent_id: str, name: str) -> str:
        existing = self.list_children(parent_id)
        if name in existing:
            return existing[name]["id"]
        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        res = self._api(
            "POST",
            "https://www.googleapis.com/drive/v3/files?fields=id",
            body=json.dumps(meta).encode(),
            headers={"Content-Type": "application/json"},
        )
        return res["id"]

    def delete_file(self, file_id: str) -> None:
        self._api("DELETE", f"https://www.googleapis.com/drive/v3/files/{file_id}")

    def upload_file(self, parent_id: str, path: pathlib.Path, existing: dict | None) -> int:
        size = path.stat().st_size
        if existing and int(existing.get("size", -1)) == size:
            return 0
        if existing:
            self.delete_file(existing["id"])

        meta = {"name": path.name, "parents": [parent_id]}
        init_req = urllib.request.Request(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,size",
            data=json.dumps(meta).encode(),
            headers={
                "Authorization": f"Bearer {self.token()}",
                "Content-Type": "application/json; charset=UTF-8",
                "X-Upload-Content-Length": str(size),
            },
            method="POST",
        )
        with urllib.request.urlopen(init_req, timeout=60) as resp:
            session_uri = resp.headers["Location"]

        if size == 0:
            put_req = urllib.request.Request(
                session_uri,
                data=b"",
                headers={"Content-Length": "0"},
                method="PUT",
            )
            with urllib.request.urlopen(put_req, timeout=60):
                pass
            return 0

        with path.open("rb") as f:
            offset = 0
            while offset < size:
                chunk = f.read(CHUNK_SIZE)
                end = offset + len(chunk) - 1
                for attempt in range(5):
                    put_req = urllib.request.Request(
                        session_uri,
                        data=chunk,
                        headers={
                            "Content-Length": str(len(chunk)),
                            "Content-Range": f"bytes {offset}-{end}/{size}",
                        },
                        method="PUT",
                    )
                    try:
                        with urllib.request.urlopen(put_req, timeout=300) as resp:
                            _ = resp.read()
                        break
                    except urllib.error.HTTPError as e:
                        if e.code == 308:
                            break
                        if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                            time.sleep(2**attempt)
                            continue
                        raise
                offset = end + 1
        return size


def sync_day_dir(client: DriveClient, day_dir: pathlib.Path, day_name: str) -> None:
    files = sorted(
        p for p in day_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    if not files:
        raise RuntimeError(f"No export files found in {day_dir}")

    folder_name = f"day={day_name}"
    root_id = HOT_EXPORT_FOLDER_ID or client.ensure_folder("root", HOT_EXPORT_ROOT_NAME)
    folder_id = client.ensure_folder(root_id, folder_name)
    remote = client.list_children(folder_id)

    uploaded_bytes = 0
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        futs = {
            pool.submit(client.upload_file, folder_id, p, remote.get(p.name)): p
            for p in files
        }
        for fut in cf.as_completed(futs):
            uploaded_bytes += fut.result()

    # Verify all files present with exact byte size
    verified = client.list_children(folder_id)
    for p in files:
        rem = verified.get(p.name)
        if not rem or int(rem.get("size", -1)) != p.stat().st_size:
            raise RuntimeError(f"Verification failed for {folder_name}/{p.name}")

    (day_dir / ".uploaded").write_text(
        json.dumps(
            {
                "folder_id": folder_id,
                "day": day_name,
                "files": len(files),
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        + "\n"
    )
    print(
        f"verified {folder_name} ({len(files)} files, {uploaded_bytes / (1024**2):.1f} MiB uploaded) -> {folder_id}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_root_or_day", help="/var/lib/catbird-intel/export or specific day dir")
    ap.add_argument("--day", help="Specific YYYY-MM-DD when first arg is a day dir")
    ap.add_argument("--oauth", default=os.environ.get("DRIVE_OAUTH_PATH", DEFAULT_OAUTH_PATH))
    args = ap.parse_args()

    if not os.path.exists(args.oauth):
        sys.exit(f"OAuth credentials file not found: {args.oauth}")

    client = DriveClient(args.oauth)
    target = pathlib.Path(args.export_root_or_day)
    if args.day:
        sync_day_dir(client, target, args.day)
        return

    if (target / "manifest.json").exists():
        sync_day_dir(client, target, target.name)
        return

    day_dirs = sorted(
        d
        for d in target.iterdir()
        if d.is_dir() and d.name.startswith("20") and (d / "manifest.json").exists()
    )
    for d in day_dirs:
        if (d / ".uploaded").exists():
            print(f"skip already-uploaded {d.name}", flush=True)
            continue
        sync_day_dir(client, d, d.name)


if __name__ == "__main__":
    main()
