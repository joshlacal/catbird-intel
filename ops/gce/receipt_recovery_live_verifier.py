#!/usr/bin/env python3
"""Bounded read-only proof for an ACTIVE retained receipt recovery.

This adapter deliberately has no closed-cache/ledger prerequisite. Its caller
must validate the exact recovery manifest and journal, exclude other writers,
and install an outer process watchdog. No method uploads, deletes, mutates a
ledger, or probes a resumable upload. OAuth refresh is the sole HTTP POST.

Use as a context manager. Pass the controller's validated manifest, the two
journal digests, original Drive IDs/names, existing OAuth path, and the same
absolute deadline as the recovery lease. Inject all three transports in tests.
Production dependencies: google-auth 2.62.0 and requests 2.34.2. GCS reads use
the JSON REST API directly; google-cloud-storage is not imported.
"""
from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import urllib.parse

from gcs_drive_drain import Conflict


PACE_SECONDS = 0.60
MAX_SESSION_SECONDS = 600
MAX_STREAM_SECONDS = 90
REQUEST_TIMEOUT_SECONDS = 30
MAX_PAYLOAD_BYTES = 320 << 20
MAX_RECEIPT_BYTES = 1 << 20
MAX_JSON_BYTES = 32 << 10
MAX_OAUTH_BYTES = 64 << 10
READ_CHUNK = 1 << 20
# Full expired-session wrapper path: root preflight 3, controller preflight 3,
# fresh exact-absence proof 4, full media verification/final identity 8 = 18.
# Two spare reads are bounded headroom, not a retry policy.
MAX_DRIVE_GETS = 20
MAX_GCS_GETS = 4
MAX_AUTH_REQUESTS = 8
FOLDER_MIME = "application/vnd.google-apps.folder"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
DRIVE_ID = re.compile(r"[A-Za-z0-9_-]{10,160}\Z")
BUCKET_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]\Z")


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Conflict("Recovery response contains duplicate JSON keys")
        result[key] = value
    return result


def _bad_constant(_value):
    raise Conflict("Recovery response contains a nonfinite JSON number")


def _json(raw: bytes) -> dict:
    try:
        value = json.loads(raw, object_pairs_hook=_strict_pairs,
                           parse_constant=_bad_constant)
    except (ValueError, UnicodeError, TypeError) as error:
        raise Conflict("Recovery response JSON is malformed") from error
    if not isinstance(value, dict):
        raise Conflict("Recovery response must be a JSON object")
    return value


def _finite(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise Conflict("Invalid recovery " + label)
    return float(value)


def _plain_session():
    import requests
    from requests.adapters import HTTPAdapter
    session = requests.Session()
    session.trust_env = False
    session.mount("https://", HTTPAdapter(max_retries=0))
    session.mount("http://", HTTPAdapter(max_retries=0))
    return session


class _AuthResponse:
    def __init__(self, status, headers, data):
        self.status, self.headers, self.data = status, headers, data


class _BoundedADCRequest:
    """Bound discovery/refresh too; a repeated auth URL never dispatches twice.

    GCE identity discovery uses the two fixed metadata hosts. A pre-existing
    service-account ADC grant may use the fixed OAuth token host. No requested
    scope is added. Retry loops inside google-auth cannot make a repeated send.
    """
    def __init__(self, owner):
        self.owner = owner
        self.session = _plain_session()
        self.seen = set()

    def __call__(self, url, method="GET", body=None, headers=None,
                 timeout=REQUEST_TIMEOUT_SECONDS, **_kwargs):
        parsed = urllib.parse.urlsplit(url)
        metadata = (method == "GET" and body is None and
                    parsed.scheme == "http" and
                    parsed.netloc in ("169.254.169.254", "metadata.google.internal"))
        token = (method == "POST" and url == "https://oauth2.googleapis.com/token")
        key = (method, url)
        if (not (metadata or token) or parsed.fragment or key in self.seen or
                len(self.seen) >= MAX_AUTH_REQUESTS):
            raise Conflict("ADC identity request is outside its bounded allowlist")
        self.owner._remaining()
        self.seen.add(key)
        response = None
        try:
            response = self.session.request(
                method, url, data=body, headers=headers or {},
                timeout=self.owner._timeout(), allow_redirects=False, stream=True)
            self.owner._response_headers(response, expected_status=None)
            data = self.owner._body(response, MAX_JSON_BYTES, self.owner.mono_deadline)
            return _AuthResponse(response.status_code, response.headers, data)
        except Conflict:
            raise
        except Exception:
            raise Conflict("ADC identity read failed") from None
        finally:
            if response is not None:
                response.close()

    def close(self):
        self.session.close()


class ReceiptRecoveryLiveVerifier:
    """Controller-compatible exact-ID proof, independent of cache eligibility."""

    def __init__(self, *, manifest: dict, entry_digests: dict, drive_ids: dict,
                 payload_name: str, receipt_name: str, oauth_file: Path,
                 deadline_epoch: float, gcs_http=None, drive_http=None,
                 drive_auth=None, clock=time.time, mono=time.monotonic,
                 sleep=time.sleep):
        self.clock, self.mono, self.sleep = clock, mono, sleep
        now = _finite(clock(), "wall clock")
        tick = _finite(mono(), "monotonic clock")
        deadline = _finite(deadline_epoch, "deadline")
        if deadline <= now:
            raise Conflict("Recovery verifier deadline already expired")
        self.wall_deadline = min(deadline, now + MAX_SESSION_SECONDS)
        self.mono_deadline = tick + (self.wall_deadline - now)
        if not isinstance(manifest, dict) or not isinstance(entry_digests, dict):
            raise Conflict("Recovery manifest or journal digests are missing")
        self.manifest = copy.deepcopy(manifest)
        self.digests = copy.deepcopy(entry_digests)
        self.ids = copy.deepcopy(drive_ids)
        if (not isinstance(self.ids, dict) or set(self.ids) != {"folder", "payload", "receipt"}
                or any(not isinstance(value, str) or not DRIVE_ID.fullmatch(value)
                       for value in self.ids.values()) or len(set(self.ids.values())) != 3):
            raise Conflict("Recovery Drive IDs are missing, duplicate, or malformed")
        self.account = self.manifest.get("account_id")
        self.root = self.manifest.get("drive_root")
        self.bucket = self.manifest.get("bucket")
        self.folder_name = self.manifest.get("folder_name")
        if (not isinstance(self.account, str) or not 0 < len(self.account) <= 256
                or not isinstance(self.root, str) or not DRIVE_ID.fullmatch(self.root)
                or not isinstance(self.bucket, str) or not BUCKET_NAME.fullmatch(self.bucket)
                or not isinstance(self.folder_name, str) or not 0 < len(self.folder_name) <= 512
                or self.root in self.ids.values()):
            raise Conflict("Recovery manifest account, bucket, root, or folder differs")
        if (not isinstance(payload_name, str) or not payload_name
                or payload_name in (".", "..") or "/" in payload_name
                or not isinstance(receipt_name, str) or receipt_name != payload_name + ".receipt.json"):
            raise Conflict("Recovery journal staged names differ")
        self.names = {"payload": payload_name, "receipt": receipt_name}
        for kind, limit in (("payload", MAX_PAYLOAD_BYTES), ("receipt", MAX_RECEIPT_BYTES)):
            source, digest = self.manifest.get(kind), self.digests.get(kind)
            if (not isinstance(source, dict) or not isinstance(digest, dict)
                    or type(source.get("bytes")) is not int or not 0 < source["bytes"] <= limit
                    or type(source.get("generation")) is not int or source["generation"] <= 0
                    or not isinstance(source.get("object"), str)
                    or not 0 < len(source["object"].encode()) <= 1024
                    or not HEX64.fullmatch(str(source.get("sha256", "")))
                    or digest.get("size") != source["bytes"]
                    or digest.get("sha256") != source["sha256"]
                    or not HEX32.fullmatch(str(digest.get("md5", "")))):
                raise Conflict("Recovery source or journal digest is not exact and bounded")
        if self.manifest["payload"]["object"] == self.manifest["receipt"]["object"]:
            raise Conflict("Recovery source objects overlap")
        self.oauth_file = Path(oauth_file)
        self.gcs_http, self.drive_http, self.drive_auth = gcs_http, drive_http, drive_auth
        self._owned_sessions = []
        self._adc_request = None
        self._token_value = None
        self._closed = False
        self._account_ok = self._folder_ok = False
        self.last_drive_completion = float("-inf")
        self.drive_gets = self.gcs_gets = self.media_bytes = 0
        self.max_media_bytes = 2 * sum(self.manifest[k]["bytes"] for k in ("payload", "receipt"))

    def __enter__(self):
        self._remaining()
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            for session in self._owned_sessions:
                session.close()
            if self._adc_request is not None:
                self._adc_request.close()

    def _remaining(self) -> float:
        if self._closed:
            raise Conflict("Recovery verifier is closed")
        remaining = min(self.wall_deadline - _finite(self.clock(), "wall clock"),
                        self.mono_deadline - _finite(self.mono(), "monotonic clock"))
        if remaining <= 0:
            raise Conflict("Recovery verifier shared deadline expired")
        return remaining

    def _timeout(self):
        seconds = min(REQUEST_TIMEOUT_SECONDS, self._remaining())
        return (seconds, seconds)

    def _session(self, kind):
        if kind == "drive":
            if self.drive_http is None:
                self.drive_http = _plain_session()
                self._owned_sessions.append(self.drive_http)
            return self.drive_http
        if self.gcs_http is None:
            import google.auth
            from google.auth.transport.requests import AuthorizedSession
            from requests.adapters import HTTPAdapter
            self._adc_request = _BoundedADCRequest(self)
            credentials, _project = google.auth.default(request=self._adc_request)
            self._remaining()
            self.gcs_http = AuthorizedSession(
                credentials, max_refresh_attempts=0,
                refresh_timeout=min(REQUEST_TIMEOUT_SECONDS, self._remaining()),
                auth_request=self._adc_request)
            self.gcs_http.trust_env = False
            self.gcs_http.mount("https://", HTTPAdapter(max_retries=0))
            self._owned_sessions.append(self.gcs_http)
        return self.gcs_http

    def _response_headers(self, response, expected_status=200):
        self._remaining()
        if response.history or 300 <= response.status_code < 400:
            raise Conflict("Recovery read redirected")
        if expected_status is not None and response.status_code != expected_status:
            raise Conflict("Recovery read HTTP status " + str(response.status_code))
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise Conflict("Recovery response content encoding differs")
        if response.headers.get("Content-Range") is not None:
            raise Conflict("Recovery response is a partial range")

    def _body(self, response, limit: int, mono_deadline: float) -> bytes:
        parts, total = [], 0
        while True:
            self._remaining()
            if self.mono() >= mono_deadline:
                raise Conflict("Recovery response stream deadline expired")
            chunk = response.raw.read(min(READ_CHUNK, limit - total + 1), decode_content=False)
            if not isinstance(chunk, bytes):
                raise Conflict("Recovery response returned nonbytes")
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise Conflict("Recovery response exceeds byte bound")
            parts.append(chunk)
        self._remaining()
        if self.mono() >= mono_deadline:
            raise Conflict("Recovery response stream deadline expired")
        return b"".join(parts)

    def _token(self):
        self._remaining()
        if self._token_value is None:
            if self.drive_auth is not None:
                token = self.drive_auth.token()
            else:
                fd = os.open(self.oauth_file, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    info = os.fstat(fd)
                    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                            or info.st_uid != os.geteuid() or info.st_mode & 0o077
                            or not 0 < info.st_size <= MAX_OAUTH_BYTES):
                        raise Conflict("Existing OAuth file is not private and bounded")
                    raw = os.read(fd, MAX_OAUTH_BYTES + 1)
                    after = os.fstat(fd)
                    if (len(raw) != info.st_size or
                            (after.st_size, after.st_mtime_ns, after.st_ctime_ns) !=
                            (info.st_size, info.st_mtime_ns, info.st_ctime_ns)):
                        raise Conflict("Existing OAuth file changed during read")
                finally:
                    os.close(fd)
                grant = _json(raw)
                fields = {key: grant.get(key) for key in ("client_id", "client_secret", "refresh_token")}
                if any(not isinstance(value, str) or not 0 < len(value) <= 8192
                       for value in fields.values()):
                    raise Conflict("Existing OAuth grant is malformed")
                fields["grant_type"] = "refresh_token"
                response = None
                try:
                    response = self._session("drive").request(
                        "POST", "https://oauth2.googleapis.com/token",
                        data=urllib.parse.urlencode(fields).encode(),
                        headers={"Content-Type": "application/x-www-form-urlencoded",
                                 "Accept-Encoding": "identity"},
                        timeout=self._timeout(), allow_redirects=False, stream=True)
                    self._response_headers(response)
                    value = _json(self._body(response, MAX_JSON_BYTES, self.mono_deadline))
                    token = value.get("access_token")
                except Conflict:
                    raise
                except Exception:
                    raise Conflict("Existing OAuth refresh failed") from None
                finally:
                    if response is not None:
                        response.close()
            self._remaining()
            if not isinstance(token, str) or not 0 < len(token) <= 8192:
                raise Conflict("Existing OAuth refresh returned no bounded token")
            self._token_value = token
        return self._token_value

    @contextlib.contextmanager
    def _get(self, kind, url, *, allow_absent=False):
        parsed = urllib.parse.urlsplit(url)
        host = "www.googleapis.com" if kind == "drive" else "storage.googleapis.com"
        prefixes = ("/drive/v3/",) if kind == "drive" else ("/storage/v1/", "/download/storage/v1/")
        if (parsed.scheme != "https" or parsed.netloc != host or parsed.fragment
                or not parsed.path.startswith(prefixes)):
            raise Conflict("Recovery GET target is outside fixed API host")
        self._remaining()
        headers = {"Accept-Encoding": "identity"}
        if kind == "drive":
            headers["Authorization"] = "Bearer " + self._token()
            self.drive_gets += 1
            if self.drive_gets > MAX_DRIVE_GETS:
                raise Conflict("Recovery Drive read count exceeded")
            wait = PACE_SECONDS - (self.mono() - self.last_drive_completion)
            if wait > 0:
                if wait >= self._remaining():
                    raise Conflict("Recovery pacing cannot fit before deadline")
                self.sleep(wait)
        else:
            self.gcs_gets += 1
            if self.gcs_gets > MAX_GCS_GETS:
                raise Conflict("Recovery GCS read count exceeded")
        response = None
        try:
            options = {"headers": headers, "timeout": self._timeout(),
                       "allow_redirects": False, "stream": True}
            session = self._session(kind)
            if kind == "gcs":
                options["max_allowed_time"] = self._remaining()
            response = session.request("GET", url, **options)
            self._response_headers(response, expected_status=None)
            if response.status_code != 200 and not (allow_absent and response.status_code == 404):
                raise Conflict("Recovery read HTTP status " + str(response.status_code))
            yield response
            self._remaining()
        except Conflict:
            raise
        except Exception:
            raise Conflict("Recovery " + kind + " read failed") from None
        finally:
            if response is not None:
                response.close()
            if kind == "drive":
                self.last_drive_completion = self.mono()

    def _metadata(self, kind, url):
        with self._get(kind, url) as response:
            return _json(self._body(response, MAX_JSON_BYTES, self.mono_deadline))

    @staticmethod
    def _drive_url(file_id, media=False):
        if not isinstance(file_id, str) or not DRIVE_ID.fullmatch(file_id):
            raise Conflict("Recovery Drive file ID is malformed")
        query = {"alt": "media"} if media else {
            "fields": "id,name,size,md5Checksum,mimeType,parents,trashed"}
        return "https://www.googleapis.com/drive/v3/files/" + file_id + "?" + urllib.parse.urlencode(query)

    def verify_account(self, account_id):
        if account_id != self.account:
            raise Conflict("Recovery account input differs")
        value = self._metadata("drive", "https://www.googleapis.com/drive/v3/about?fields=user(permissionId)")
        if not isinstance(value.get("user"), dict) or value["user"].get("permissionId") != self.account:
            raise Conflict("Recovery Drive permission ID differs")
        self._account_ok = True

    def verify_folder(self, folder_id, name, root):
        if (not self._account_ok or folder_id != self.ids["folder"]
                or name != self.folder_name or root != self.root):
            raise Conflict("Recovery folder input differs")
        root_meta = self._metadata("drive", self._drive_url(self.root))
        if (root_meta.get("id") != self.root or root_meta.get("mimeType") != FOLDER_MIME
                or root_meta.get("trashed") is not False):
            raise Conflict("Recovery Drive root metadata differs")
        value = self._metadata("drive", self._drive_url(folder_id))
        if (value.get("id") != folder_id or value.get("name") != name
                or value.get("parents") != [root] or value.get("mimeType") != FOLDER_MIME
                or value.get("trashed") is not False):
            raise Conflict("Recovery Drive folder metadata differs")
        self._folder_ok = True

    def verify_exact_payload_absent(self, account_id, root_id, folder_id, payload_id):
        if (account_id != self.account or root_id != self.root
                or folder_id != self.ids["folder"] or payload_id != self.ids["payload"]):
            raise Conflict("Recovery absence proof IDs differ")
        if not self._account_ok:
            self.verify_account(account_id)
        if not self._folder_ok:
            self.verify_folder(folder_id, self.folder_name, root_id)
        with self._get("drive", self._drive_url(payload_id), allow_absent=True) as response:
            if response.status_code == 404:
                return {"account_id": account_id, "root_id": root_id, "folder_id": folder_id,
                        "payload_id": payload_id, "http_status": 404, "observed_at": self.clock()}
            meta = _json(self._body(response, MAX_JSON_BYTES, self.mono_deadline))
        wanted = self.manifest["payload"]
        if (meta.get("id") != payload_id or meta.get("name") != self.names["payload"]
                or meta.get("parents") != [self.ids["folder"]]
                or meta.get("trashed") is not False
                or meta.get("mimeType") != "application/octet-stream"
                or str(meta.get("size")) != str(wanted["bytes"])
                or meta.get("md5Checksum") != self.digests["payload"]["md5"]):
            raise Conflict("Recovery exact Drive payload metadata differs")
        return None

    def verify_exact_receipt_absent(self, account_id, root_id, folder_id, receipt_id):
        if (account_id != self.account or root_id != self.root
                or folder_id != self.ids["folder"] or receipt_id != self.ids["receipt"]):
            raise Conflict("Recovery absence proof IDs differ")
        # Fresh account/root/folder reads precede this exact-ID 404 observation.
        self.verify_account(account_id)
        self.verify_folder(folder_id, self.folder_name, root_id)
        with self._get("drive", self._drive_url(receipt_id), allow_absent=True) as response:
            if response.status_code != 404:
                raise Conflict("Exact recovery receipt ID is present; absence is unproven")
        return {"account_id": account_id, "root_id": root_id, "folder_id": folder_id,
                "receipt_id": receipt_id, "http_status": 404, "observed_at": self.clock()}

    def _media(self, kind, url, role):
        wanted = self.manifest[role]
        self.media_bytes += wanted["bytes"]
        if self.media_bytes > self.max_media_bytes:
            raise Conflict("Recovery cumulative media read bound exceeded")
        stream_deadline = min(self.mono_deadline, self.mono() + MAX_STREAM_SECONDS)
        with self._get(kind, url) as response:
            length = response.headers.get("Content-Length")
            if length is not None and str(length) != str(wanted["bytes"]):
                raise Conflict("Recovery media Content-Length differs")
            generation = response.headers.get("x-goog-generation")
            if kind == "gcs" and generation is not None and str(generation) != str(wanted["generation"]):
                raise Conflict("Recovery media response generation differs")
            sha, md5, total = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
            while True:
                self._remaining()
                if self.mono() >= stream_deadline:
                    raise Conflict("Recovery media stream deadline expired")
                chunk = response.raw.read(min(READ_CHUNK, wanted["bytes"] - total + 1),
                                          decode_content=False)
                if not isinstance(chunk, bytes):
                    raise Conflict("Recovery media reader returned nonbytes")
                if not chunk:
                    break
                total += len(chunk)
                if total > wanted["bytes"]:
                    raise Conflict("Recovery media exceeds pinned byte count")
                sha.update(chunk)
                md5.update(chunk)
            self._remaining()
            if self.mono() >= stream_deadline:
                raise Conflict("Recovery media stream deadline expired")
        if (total != wanted["bytes"] or sha.hexdigest() != wanted["sha256"]
                or md5.hexdigest() != self.digests[role]["md5"]):
            raise Conflict("Recovery full media digest differs from source and local journal")
        return {"bytes": total, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}

    def gcs_digest(self, object_name, generation):
        role = next((k for k in ("payload", "receipt")
                     if self.manifest[k]["object"] == object_name), None)
        if (not self._account_ok or not self._folder_ok or role is None
                or type(generation) is not int or generation != self.manifest[role]["generation"]):
            raise Conflict("Recovery GCS generation input differs")
        wanted = self.manifest[role]
        path = "/storage/v1/b/" + urllib.parse.quote(self.bucket, safe="")
        path += "/o/" + urllib.parse.quote(object_name, safe="")
        query = {"generation": str(generation), "ifGenerationMatch": str(generation)}
        metadata_query = {**query, "fields": "bucket,name,generation,size,md5Hash,metadata,contentEncoding"}
        meta = self._metadata("gcs", "https://storage.googleapis.com" + path + "?" + urllib.parse.urlencode(metadata_query))
        try:
            metadata_md5 = base64.b64decode(meta.get("md5Hash", ""), validate=True).hex()
        except (ValueError, TypeError):
            raise Conflict("Recovery GCS MD5 metadata is malformed") from None
        if (meta.get("bucket") != self.bucket or meta.get("name") != object_name
                or str(meta.get("generation")) != str(generation)
                or str(meta.get("size")) != str(wanted["bytes"])
                or metadata_md5 != self.digests[role]["md5"]
                or not isinstance(meta.get("metadata"), dict)
                or meta["metadata"].get("sha256") != wanted["sha256"]
                or meta.get("contentEncoding", "identity") != "identity"):
            raise Conflict("Recovery exact GCS generation metadata differs")
        url = "https://storage.googleapis.com/download" + path + "?" + urllib.parse.urlencode({**query, "alt": "media"})
        return {**self._media("gcs", url, role), "object": object_name, "generation": generation}

    def drive_digest(self, file_id):
        role = next((k for k in ("payload", "receipt") if self.ids[k] == file_id), None)
        if not self._account_ok or not self._folder_ok or role is None:
            raise Conflict("Recovery Drive media input differs")
        wanted = self.manifest[role]
        meta = self._metadata("drive", self._drive_url(file_id))
        expected_mime = "application/octet-stream" if role == "payload" else "application/json"
        if (meta.get("id") != file_id or meta.get("name") != self.names[role]
                or meta.get("parents") != [self.ids["folder"]]
                or meta.get("trashed") is not False or meta.get("mimeType") != expected_mime
                or str(meta.get("size")) != str(wanted["bytes"])
                or meta.get("md5Checksum") != self.digests[role]["md5"]):
            raise Conflict("Recovery exact Drive file metadata differs")
        return {**self._media("drive", self._drive_url(file_id, media=True), role),
                "id": file_id, "account_id": self.account,
                "parent_id": self.ids["folder"], "trashed": False, "media_readback": True}
