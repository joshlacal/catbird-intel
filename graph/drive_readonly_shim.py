#!/usr/bin/env python3
"""Minimal stand-in for the `gws` CLI on Linux VMs, read-only Drive only.

archive_graph_extract.py calls exactly two gws forms:
  gws drive files list --params JSON           -> JSON on stdout
  gws drive files get  --params JSON [-o NAME] -> media to stdout, or to NAME in cwd

Point CATBIRD_GWS at this file and CATBIRD_DRIVE_CREDENTIALS at an
authorized_user JSON (client_id, client_secret, refresh_token) minted with the
drive.readonly scope. Anything else exits 2, so the shim cannot write to Drive.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://www.googleapis.com/drive/v3/files"
TOKEN_URL = "https://oauth2.googleapis.com/token"
_CACHE = os.path.expanduser("~/.cache/catbird-drive-readonly-token.json")


def access_token() -> str:
    try:
        with open(_CACHE) as fh:
            cached = json.load(fh)
        if cached["expires_at"] > time.time() + 120:
            return cached["access_token"]
    except (OSError, ValueError, KeyError):
        pass
    with open(os.environ["CATBIRD_DRIVE_CREDENTIALS"]) as fh:
        creds = json.load(fh)
    body = urllib.parse.urlencode({
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token",
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=body), timeout=60) as resp:
        tok = json.load(resp)
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    fd = os.open(_CACHE + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump({"access_token": tok["access_token"], "expires_at": time.time() + int(tok.get("expires_in", 3600))}, fh)
    os.replace(_CACHE + ".tmp", _CACHE)
    return tok["access_token"]


def request(url: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token()}"})
    return urllib.request.urlopen(req, timeout=600)


def main(argv: list[str]) -> int:
    if len(argv) < 5 or argv[:2] != ["drive", "files"] or argv[2] not in ("list", "get") or argv[3] != "--params":
        print(f"drive_readonly_shim: unsupported command {argv[:4]}", file=sys.stderr)
        return 2
    params = json.loads(argv[4])
    out_name = argv[6] if len(argv) >= 7 and argv[5] == "-o" else None
    if argv[2] == "list":
        query = urllib.parse.urlencode({k: v for k, v in params.items()})
        with request(f"{API}?{query}") as resp:
            sys.stdout.write(resp.read().decode())
        return 0
    file_id = params.pop("fileId")
    query = urllib.parse.urlencode(params)
    with request(f"{API}/{urllib.parse.quote(file_id)}?{query}") as resp:
        if out_name:
            with open(out_name, "wb") as fh:
                shutil.copyfileobj(resp, fh, 8 << 20)
        else:
            shutil.copyfileobj(resp, sys.stdout.buffer)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except urllib.error.HTTPError as exc:
        print(f"drive_readonly_shim: HTTP {exc.code} {exc.read()[:300]!r}", file=sys.stderr)
        sys.exit(1)
