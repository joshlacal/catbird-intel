#!/usr/bin/env python3
"""One-request-at-a-time transport for a pinned retained receipt recovery.

The injected request function MUST be SafeDrive.request bound to the shared
retained authority. It reserves, paces, dispatches, and records every request.
This module has no credential loader, retry loop, or command-line entry point.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import stat
import urllib.error
import urllib.parse

from gcs_drive_drain import Conflict


MAX_RECEIPT_BYTES = 1 << 20
MAX_METADATA_BYTES = 16 << 10
MAX_SESSION_URL_BYTES = 4096
REQUEST_TIMEOUT_SECONDS = 60
DRIVE_UPLOAD_URL = (
    "https://www.googleapis.com/upload/drive/v3/files"
    "?uploadType=resumable&fields=id"
)
_ID = re.compile(r"[A-Za-z0-9_-]{8,128}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MD5 = re.compile(r"[0-9a-f]{32}\Z")
_RANGE = re.compile(r"bytes=0-(\d+)\Z")


def _check_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise Conflict("Pinned receipt " + label + " is invalid")
    return value


def _check_size(size: int) -> int:
    if type(size) is not int or not 0 < size <= MAX_RECEIPT_BYTES:
        raise Conflict("Pinned receipt size exceeds bound")
    return size


def _session_url(url: str) -> str:
    if not isinstance(url, str) or len(url.encode("utf-8")) > MAX_SESSION_URL_BYTES:
        raise Conflict("Pinned resumable session URL is invalid")
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "www.googleapis.com"
            or parsed.path != "/upload/drive/v3/files" or parsed.fragment
            or not parsed.query):
        raise Conflict("Pinned resumable session host or path differs")
    try:
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True,
                                      strict_parsing=True)
    except ValueError as error:
        raise Conflict("Pinned resumable session query is malformed") from error
    if (query.get("uploadType") != ["resumable"] or
            len(query.get("upload_id", [])) != 1 or
            not query["upload_id"][0]):
        raise Conflict("Pinned resumable session identity differs")
    return url


def _pinned_session(url: str, expected_sha256: str) -> str:
    url = _session_url(url)
    if (not isinstance(expected_sha256, str) or
            not _SHA256.fullmatch(expected_sha256) or
            hashlib.sha256(url.encode("utf-8")).hexdigest() != expected_sha256):
        raise Conflict("Private resumable session fingerprint differs")
    return url


def _strict_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise Conflict("Drive completion metadata has duplicate keys")
        value[key] = item
    return value


def _reject_constant(_value):
    raise Conflict("Drive completion metadata is nonfinite")


def _metadata(response, expected_id: str, requested_url: str) -> dict:
    observed_url = response.geturl() if hasattr(response, "geturl") else requested_url
    if observed_url != requested_url:
        raise Conflict("Drive recovery response redirected")
    raw = response.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise Conflict("Drive completion metadata exceeds bound")
    try:
        value = json.loads(raw, object_pairs_hook=_strict_pairs,
                           parse_constant=_reject_constant)
    except (UnicodeError, ValueError, TypeError) as error:
        raise Conflict("Drive completion metadata is malformed") from error
    if not isinstance(value, dict) or value.get("id") != expected_id:
        raise Conflict("Drive completion ID differs from pinned receipt")
    # The caller still independently verifies exact-ID metadata and full media.
    return {"id": expected_id}


def _acknowledged(headers, size: int, *, allow_absent: bool) -> int:
    raw = headers.get("Range") if headers is not None else None
    if not raw:
        if allow_absent:
            return 0
        raise Conflict("Drive receipt continuation lacks an acknowledged range")
    match = _RANGE.fullmatch(raw)
    if not match:
        raise Conflict("Drive receipt acknowledged range is malformed")
    acknowledged = int(match.group(1)) + 1
    if not 0 < acknowledged < size:
        raise Conflict("Drive receipt acknowledged range exceeds bound")
    return acknowledged


def _receipt_bytes(path: Path, size: int, sha256: str, md5: str) -> bytes:
    if (not isinstance(path, Path) or path.is_symlink() or
            not isinstance(sha256, str) or not _SHA256.fullmatch(sha256) or
            not isinstance(md5, str) or not _MD5.fullmatch(md5)):
        raise Conflict("Pinned local receipt identity is invalid")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != size:
        raise Conflict("Pinned local receipt file differs")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        current = os.fstat(fd)
        if ((current.st_dev, current.st_ino, current.st_size,
             current.st_mtime_ns, current.st_ctime_ns) !=
            (before.st_dev, before.st_ino, before.st_size,
             before.st_mtime_ns, before.st_ctime_ns)):
            raise Conflict("Pinned local receipt changed before read")
        data = b""
        while len(data) <= size:
            chunk = os.read(fd, min(1 << 16, size + 1 - len(data)))
            if not chunk:
                break
            data += chunk
        after = os.fstat(fd)
    finally:
        os.close(fd)
    path_after = path.lstat()
    if (len(data) != size or
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
             after.st_ctime_ns) !=
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_ctime_ns) or
            (path_after.st_dev, path_after.st_ino, path_after.st_size,
             path_after.st_mtime_ns, path_after.st_ctime_ns) !=
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_ctime_ns) or
            hashlib.sha256(data).hexdigest() != sha256 or
            hashlib.md5(data).hexdigest() != md5):
        raise Conflict("Pinned local receipt bytes changed")
    return data


class ReceiptRecoveryTransport:
    """Network syntax only; caller owns forensic authority and decisions."""

    def __init__(self, request_fn):
        if not callable(request_fn):
            raise Conflict("Recovery request authority is unavailable")
        self.request = request_fn

    def probe(self, session: str, size: int, pinned_id: str,
              expected_session_sha256: str) -> dict:
        """Make exactly one empty status PUT; never send receipt media."""
        session = _pinned_session(session, expected_session_sha256)
        size = _check_size(size)
        pinned_id = _check_id(pinned_id, "ID")
        headers = {"Content-Length": "0", "Content-Range": f"bytes */{size}"}
        try:
            with self.request(session, "PUT", b"", headers,
                              timeout=REQUEST_TIMEOUT_SECONDS,
                              kind="resumable_status") as response:
                if response.status not in (200, 201):
                    raise Conflict("Drive receipt status response is unexpected")
                metadata = _metadata(response, pinned_id, session)
                return {"state": "complete", "acknowledged": size,
                        "metadata": metadata, "http_status": response.status}
        except urllib.error.HTTPError as error:
            try:
                if error.code == 308:
                    offset = _acknowledged(error.headers, size, allow_absent=True)
                    return {"state": "partial", "acknowledged": offset,
                            "http_status": 308}
                if error.code in (404, 410):
                    return {"state": "expired", "acknowledged": 0,
                            "http_status": error.code}
                raise Conflict("Drive receipt status probe is uncertain") from None
            finally:
                error.close()
        except (OSError, http.client.HTTPException):
            raise Conflict("Drive receipt status probe transport is uncertain") from None

    def resume(self, session: str, path: Path, offset: int, size: int,
               pinned_id: str, expected_sha256: str, expected_md5: str,
               expected_session_sha256: str) -> dict:
        """Send exactly the unsent suffix once; no automatic retry."""
        session = _pinned_session(session, expected_session_sha256)
        size = _check_size(size)
        pinned_id = _check_id(pinned_id, "ID")
        if type(offset) is not int or not 0 <= offset < size:
            raise Conflict("Reviewed receipt continuation offset differs")
        data = _receipt_bytes(path, size, expected_sha256, expected_md5)[offset:]
        headers = {"Content-Length": str(len(data)),
                   "Content-Range": f"bytes {offset}-{size-1}/{size}"}
        try:
            with self.request(session, "PUT", data, headers,
                              timeout=REQUEST_TIMEOUT_SECONDS,
                              kind="media_chunk") as response:
                if response.status not in (200, 201):
                    raise Conflict("Drive receipt continuation response is unexpected")
                metadata = _metadata(response, pinned_id, session)
                return {"state": "complete", "metadata": metadata,
                        "http_status": response.status}
        except urllib.error.HTTPError as error:
            try:
                if error.code == 308:
                    acknowledged = _acknowledged(error.headers, size,
                                                allow_absent=False)
                    if acknowledged < offset:
                        raise Conflict("Drive receipt continuation regressed")
                    return {"state": "partial", "acknowledged": acknowledged,
                            "http_status": 308}
                raise Conflict("Drive receipt continuation is uncertain") from None
            finally:
                error.close()
        except (OSError, http.client.HTTPException):
            raise Conflict("Drive receipt continuation transport is uncertain") from None

    def restart_same_id(self, pinned_id: str, parent_id: str,
                        name: str, size: int) -> str:
        """Start one new session with the original pre-generated receipt ID."""
        pinned_id = _check_id(pinned_id, "ID")
        parent_id = _check_id(parent_id, "parent")
        size = _check_size(size)
        if (not isinstance(name, str) or len(name.encode("utf-8")) > 255 or
                not name.endswith(".receipt.json") or "/" in name or "\\" in name or
                any(ord(char) < 32 for char in name)):
            raise Conflict("Pinned receipt name is invalid")
        body = json.dumps({"id": pinned_id, "name": name,
                           "parents": [parent_id]}, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Content-Length": str(len(body)),
                   "X-Upload-Content-Type": "application/octet-stream",
                   "X-Upload-Content-Length": str(size)}
        try:
            with self.request(DRIVE_UPLOAD_URL, "POST", body, headers,
                              timeout=REQUEST_TIMEOUT_SECONDS,
                              kind="resumable_start") as response:
                if response.status not in (200, 201):
                    raise Conflict("Drive receipt session creation is unexpected")
                if hasattr(response, "geturl") and response.geturl() != DRIVE_UPLOAD_URL:
                    raise Conflict("Drive receipt session creation redirected")
                session = _session_url(response.headers.get("Location"))
                # Caller MUST persist this private session before any media send.
                return session
        except urllib.error.HTTPError as error:
            try:
                raise Conflict("Drive receipt session creation is uncertain") from None
            finally:
                error.close()
        except (OSError, http.client.HTTPException):
            raise Conflict("Drive receipt session transport is uncertain") from None
