#!/usr/bin/env python3
"""Mint the narrow Google Drive token that netcup's daily export uploader uses.

Run on the Mac (it opens a browser for Google consent):

    python3 ops/mint_drive_token.py ~/Downloads/client_secret_XXXX.json \
        --out ~/.config/catbird-intel/drive-oauth.json

The client JSON is the "Desktop app" OAuth client downloaded from
Google Auth Platform > Clients in project catbird-436600. The token asks for
only https://www.googleapis.com/auth/drive.file, which lets the uploader see
and change only the files it created itself. Copy the output to netcup with:

    scp ~/.config/catbird-intel/drive-oauth.json netcup:.config/catbird-intel/drive-oauth.json

Nothing secret is printed. The output file is written with mode 0600.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.server
import json
import os
import pathlib
import secrets
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser

SCOPE = "https://www.googleapis.com/auth/drive.file"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"


def load_client(path: pathlib.Path) -> tuple[str, str]:
    data = json.loads(path.read_text())
    section = data.get("installed") or data.get("web")
    if not section:
        raise SystemExit("Client file has no 'installed' section; create a Desktop app client.")
    return section["client_id"], section["client_secret"]


def wait_for_code(state: str) -> tuple[http.server.HTTPServer, int, threading.Event, dict]:
    result: dict = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if qs.get("state", [""])[0] != state:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"State mismatch.")
                return
            result.update({k: v[0] for k, v in qs.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Catbird export token received. You can close this tab.")
            done.set()

        def log_message(self, *_args) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], done, result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("client_file", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("~/.config/catbird-intel/drive-oauth.json").expanduser())
    args = ap.parse_args()

    client_id, client_secret = load_client(args.client_file.expanduser())
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)

    server, port, done, result = wait_for_code(state)
    redirect_uri = f"http://127.0.0.1:{port}"
    url = AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })
    print("Opening Google consent in your browser. If it does not open, visit:\n" + url)
    webbrowser.open(url)
    if not done.wait(timeout=600):
        server.shutdown()
        raise SystemExit("Timed out waiting for consent.")
    server.shutdown()
    if "code" not in result:
        raise SystemExit(f"Consent failed: {result.get('error', 'no code returned')}")

    body = urllib.parse.urlencode({
        "code": result["code"],
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "code_verifier": verifier,
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=body)) as resp:
        tok = json.load(resp)
    if "refresh_token" not in tok:
        raise SystemExit("Google returned no refresh token; revoke the app at myaccount.google.com/permissions and retry.")
    granted = tok.get("scope", "")
    if granted.split() != [SCOPE]:
        raise SystemExit(f"Unexpected scopes granted: {granted}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump({"client_id": client_id, "client_secret": client_secret,
                   "refresh_token": tok["refresh_token"], "scope": granted}, fh)
    print(f"Wrote {args.out} (scope: drive.file only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
