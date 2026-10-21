#!/usr/bin/env python3
"""Read-only, generation-pinned remote verifier for retained stage review.

This is a library adapter, not a transfer or release CLI. Construct it only
after the sole Drive owner pauses the account writer at a maintenance boundary.
It never uploads, deletes, lists Drive siblings, or changes OAuth scopes.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import urllib.parse
import urllib.error
import urllib.request

from gcs_drive_drain import Conflict, FOLDER
from gcs_drive_retained_overflow import (
    BUCKET, DRIVE_ROOT, INDEX_OBJECT_LIMIT, load_overflow_manifest,
)


PACE_SECONDS = 0.60
MAX_DRIVE_CALLS = 10
MAX_GCS_CALLS = 8
MAX_MEDIA_BYTES = 300 * (1 << 20)
MAX_TOTAL_MEDIA_BYTES = 4 * MAX_MEDIA_BYTES
MAX_MEDIA_SECONDS = 90
MAX_SESSION_SECONDS = 240
READ_CHUNK = 1 << 20
GCS_RANGE_CHUNK = 8 << 20
MAX_JSON_BYTES = 16 << 10
MAX_OAUTH_BYTES = 16 << 10
ID = re.compile(r"[A-Za-z0-9_-]{10,160}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise Conflict("Read-only media request redirected")


class SafeDriveHTTP:
    """One exact GET per call, no retries or cross-host Authorization redirect."""

    def __init__(self):
        self.opener = urllib.request.build_opener(NoRedirect())

    def open(self, path: str, token: str):
        parsed = urllib.parse.urlsplit(path)
        if (parsed.scheme != "https" or parsed.netloc != "www.googleapis.com"
                or not parsed.path.startswith("/drive/v3/") or parsed.fragment):
            raise Conflict("Drive read target differs from fixed API host")
        request = urllib.request.Request(
            path, method="GET", headers={"Authorization": "Bearer " + token,
                                         "Accept-Encoding": "identity"})
        try:
            return self.opener.open(request, timeout=30)
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            raise Conflict("Drive read failed or was redirected") from None


class SafeOAuthRefresh:
    """Refresh only the existing grant; never follow a redirect with secrets."""

    def __init__(self, path: Path):
        self.path = path
        self.opener = urllib.request.build_opener(NoRedirect())

    def token(self) -> str:
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or info.st_uid != os.geteuid() or info.st_mode & 0o077
                        or not 0 < info.st_size <= MAX_OAUTH_BYTES):
                    raise Conflict("OAuth file is not private and bounded")
                raw = os.read(fd, MAX_OAUTH_BYTES + 1)
            finally:
                os.close(fd)
            if len(raw) > MAX_OAUTH_BYTES:
                raise Conflict("OAuth file exceeds byte bound")
            cfg = json.loads(raw)
            fields = (cfg[key] for key in ("client_id", "client_secret", "refresh_token"))
            client_id, client_secret, refresh_token = fields
            if any(not isinstance(value, str) or not value or len(value) > 4096
                   for value in (client_id, client_secret, refresh_token)):
                raise Conflict("Existing OAuth grant is malformed")
            body = urllib.parse.urlencode({
                "client_id": client_id, "client_secret": client_secret,
                "refresh_token": refresh_token, "grant_type": "refresh_token",
            }).encode()
            request = urllib.request.Request(
                "https://oauth2.googleapis.com/token", data=body, method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded",
                         "Accept-Encoding": "identity"})
            with self.opener.open(request, timeout=30) as response:
                value = _read_json(response)
            token = value.get("access_token")
            if not isinstance(token, str) or not token or len(token) > 8192:
                raise Conflict("OAuth refresh returned no bounded token")
            return token
        except Conflict:
            raise
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            raise Conflict("Existing OAuth refresh failed") from None


def _read_json(response, limit: int = MAX_JSON_BYTES) -> dict:
    if response.status != 200:
        raise Conflict("Read-only metadata HTTP status differs")
    if response.headers.get("Content-Encoding", "identity") != "identity":
        raise Conflict("Read-only response has unexpected content encoding")
    raw = response.read(limit + 1)
    if len(raw) > limit:
        raise Conflict("Read-only metadata body exceeds bound")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError, TypeError) as error:
        raise Conflict("Read-only metadata JSON is malformed") from error
    if not isinstance(value, dict):
        raise Conflict("Read-only metadata must be an object")
    return value


def _digest_stream(stream, *, max_bytes: int, deadline: float, mono) -> dict:
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    count = 0
    while True:
        if mono() > deadline:
            raise Conflict("Remote media read exceeded elapsed-time bound")
        chunk = stream.read(min(READ_CHUNK, max_bytes - count + 1))
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise Conflict("Remote media reader returned nonbytes")
        count += len(chunk)
        if count > max_bytes:
            raise Conflict("Remote media read exceeded byte bound")
        sha.update(chunk)
        md5.update(chunk)
    if mono() > deadline:
        raise Conflict("Remote media read exceeded elapsed-time bound")
    return {"bytes": count, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


class LiveReadOnlyVerifier:
    """Implements cache-release RemoteVerifier using existing credentials.

    `local` comes from inspect_cache(..., verifier=None), and `manifest` and
    `index_generation` come from that same pinned run's manifest/journal. The
    parent inspector independently compares every returned identity and hash.
    Inject transports in synthetic tests only; production defaults use the
    existing GCE service identity and existing Drive OAuth refresh grant.
    """

    def __init__(self, *, local: dict, manifest: dict, index_generation: int,
                 oauth_file: Path, bucket=None, drive_auth=None, drive_http=None,
                 mono=time.monotonic, sleep=time.sleep):
        if (local.get("eligible_for_approval") is not False
                or local.get("exclude_reason") != "remote_readback_unavailable"
                or local.get("account_id") != manifest.get("account_id")
                or local.get("mapping_bucket") != BUCKET
                or local.get("mapping_object") != manifest.get("index_object")
                or local.get("drive_root") != DRIVE_ROOT
                or type(index_generation) is not int or index_generation <= 0
                or not HEX64.fullmatch(str(local.get("mapping_sha256") or ""))):
            raise Conflict("Remote verifier local inventory is not pinned")
        for kind in ("payload", "receipt"):
            wanted = manifest.get(kind)
            stage = local.get("stage", {}).get(kind)
            if (not isinstance(wanted, dict) or local.get("source", {}).get(kind) != wanted
                    or type(wanted.get("bytes")) is not int
                    or not 0 < wanted["bytes"] <= MAX_MEDIA_BYTES
                    or stage is None or stage.get("bytes") != wanted["bytes"]
                    or not HEX64.fullmatch(str(wanted.get("sha256") or ""))
                    or not re.fullmatch(r"[0-9a-f]{32}", str(stage.get("md5") or ""))):
                raise Conflict("Remote verifier source media is not bounded and pinned")
        self.local, self.manifest = local, manifest
        self.index_generation = index_generation
        self.mono, self.sleep = mono, sleep
        self.started = mono()
        self.deadline = self.started + MAX_SESSION_SECONDS
        self.last_drive_completion = float("-inf")
        self.drive_calls = 0
        self.gcs_calls = 0
        self.media_bytes = 0
        self.account_verified = False
        self.drive_token = None
        if bucket is None:
            from google.cloud import storage
            bucket = storage.Client().bucket(BUCKET)
        self.bucket = bucket
        self.drive_auth = drive_auth if drive_auth is not None else SafeOAuthRefresh(oauth_file)
        self.drive_http = drive_http if drive_http is not None else SafeDriveHTTP()

    def _check_time(self):
        if self.mono() > self.deadline:
            raise Conflict("Remote verifier session exceeded time bound")

    def _gcs_call(self, fn):
        self._check_time()
        self.gcs_calls += 1
        if self.gcs_calls > MAX_GCS_CALLS:
            raise Conflict("Remote verifier GCS request bound reached")
        try:
            result = fn()
        except Conflict:
            raise
        except Exception:
            raise Conflict("Remote GCS read failed") from None
        self._check_time()
        return result

    def _drive_call(self, fn):
        self._check_time()
        self.drive_calls += 1
        if self.drive_calls > MAX_DRIVE_CALLS:
            raise Conflict("Remote verifier Drive request bound reached")
        # Space the next start from the previous completion. This remains safe
        # even if the process is descheduled just before the HTTP open call.
        wait = PACE_SECONDS - (self.mono() - self.last_drive_completion)
        if wait > 0:
            self.sleep(wait)
        self._check_time()
        try:
            result = fn()
        except Conflict:
            raise
        except Exception:
            raise Conflict("Remote Drive read failed") from None
        finally:
            self.last_drive_completion = self.mono()
        self._check_time()
        return result

    def _token(self):
        # One existing-grant refresh before the first GET; no scope request.
        self._check_time()
        if self.drive_token is None:
            try:
                token = self.drive_auth.token()
            except Conflict:
                raise
            except Exception as error:
                raise Conflict("Existing OAuth refresh failed") from error
            self._check_time()
            if not isinstance(token, str) or not token or len(token) > 8192:
                raise Conflict("OAuth refresh returned no bounded token")
            self.drive_token = token
        return self.drive_token

    def _drive_json(self, path: str) -> dict:
        token = self._token()
        def request():
            with self.drive_http.open(path, token) as response:
                return _read_json(response)
        return self._drive_call(request)

    @staticmethod
    def _file_url(file_id: str, *, media: bool = False) -> str:
        if not isinstance(file_id, str) or not ID.fullmatch(file_id):
            raise Conflict("Drive file ID is malformed")
        root = "https://www.googleapis.com/drive/v3/files/" + urllib.parse.quote(file_id, safe="")
        if media:
            return root + "?alt=media"
        fields = "id,name,size,md5Checksum,mimeType,parents,trashed"
        return root + "?fields=" + urllib.parse.quote(fields, safe="")

    def verify_account(self, account_id: str) -> None:
        if account_id != self.local["account_id"]:
            raise Conflict("Remote verifier account differs")
        about = self._drive_json("https://www.googleapis.com/drive/v3/about?fields=user(permissionId)")
        if about.get("user", {}).get("permissionId") != account_id:
            raise Conflict("Drive permission ID differs from retained mapping")
        root = self._drive_json(self._file_url(DRIVE_ROOT))
        if (root.get("id") != DRIVE_ROOT or root.get("mimeType") != FOLDER
                or root.get("trashed") is not False):
            raise Conflict("Drive root identity differs")
        folder_id = self.local["drive_ids"]["folder"]
        folder = self._drive_json(self._file_url(folder_id))
        if (folder.get("id") != folder_id or folder.get("mimeType") != FOLDER
                or folder.get("trashed") is not False
                or folder.get("name") != self.manifest["folder_name"]
                or folder.get("parents") != [DRIVE_ROOT]):
            raise Conflict("Drive retained folder identity differs")
        self.account_verified = True

    def mapping(self, bucket: str, name: str) -> tuple[int, bytes]:
        if (bucket != BUCKET or name != self.local["mapping_object"]
                or not self.account_verified):
            raise Conflict("Remote mapping target or account differs")
        generation = self.index_generation
        observed = self._gcs_call(lambda: self.bucket.get_blob(
            name, generation=generation, retry=None, timeout=30))
        if (observed is None or int(observed.generation) != generation
                or type(observed.size) is not int
                or not 0 < observed.size <= INDEX_OBJECT_LIMIT
                or (observed.metadata or {}).get("kind") != "drive_retained_overflow_index"
                or (observed.metadata or {}).get("sha256") != self.local["mapping_sha256"]):
            raise Conflict("Remote mapping metadata differs")
        raw = self._gcs_call(lambda: observed.download_as_bytes(
            if_generation_match=generation, checksum="auto", retry=None, timeout=30))
        if (not isinstance(raw, bytes) or len(raw) != observed.size
                or hashlib.sha256(raw).hexdigest() != self.local["mapping_sha256"]):
            raise Conflict("Remote mapping body differs")
        return generation, raw

    def gcs_digest(self, object_name: str, generation: int) -> dict:
        kind = next((kind for kind in ("payload", "receipt")
                     if self.manifest[kind]["object"] == object_name), None)
        if (kind is None or generation != self.manifest[kind]["generation"]
                or not self.account_verified):
            raise Conflict("Remote source generation differs")
        wanted = self.manifest[kind]
        observed = self._gcs_call(lambda: self.bucket.get_blob(
            object_name, generation=generation, retry=None, timeout=30))
        if observed is None:
            raise Conflict("Remote GCS generation is unavailable")
        try:
            md5_metadata = base64.b64decode(observed.md5_hash, validate=True).hex()
        except (AttributeError, TypeError, ValueError, binascii.Error) as error:
            raise Conflict("Remote GCS MD5 metadata is unavailable") from error
        if (int(observed.generation) != generation
                or observed.size != wanted["bytes"]
                or (observed.metadata or {}).get("sha256") != wanted["sha256"]
                or md5_metadata != self.local["stage"][kind]["md5"]):
            raise Conflict("Remote GCS source metadata differs")
        self.media_bytes += wanted["bytes"]
        if self.media_bytes > MAX_TOTAL_MEDIA_BYTES:
            raise Conflict("Remote verifier cumulative media bound reached")
        def read():
            with observed.open("rb", chunk_size=GCS_RANGE_CHUNK,
                               if_generation_match=generation, retry=None, timeout=30) as stream:
                return _digest_stream(stream, max_bytes=wanted["bytes"],
                                      deadline=min(self.deadline, self.mono()+MAX_MEDIA_SECONDS),
                                      mono=self.mono)
        actual = self._gcs_call(read)
        if (actual["bytes"] != wanted["bytes"]
                or actual["sha256"] != wanted["sha256"]
                or actual["md5"] != md5_metadata):
            raise Conflict("Remote GCS full media differs from pinned generation")
        return actual

    def drive_digest(self, file_id: str) -> dict:
        kind = next((kind for kind in ("payload", "receipt")
                     if self.local["drive_ids"][kind] == file_id), None)
        if kind is None or not self.account_verified:
            raise Conflict("Remote Drive media ID is not pinned")
        wanted = self.manifest[kind]
        meta = self._drive_json(self._file_url(file_id))
        folder_id = self.local["drive_ids"]["folder"]
        stage_name = Path(self.local["stage"][kind]["path"]).name
        if (meta.get("id") != file_id or meta.get("name") != stage_name
                or meta.get("parents") != [folder_id]
                or meta.get("trashed") is not False
                # Verified retained Drive metadata uses role-specific MIME.
                or meta.get("mimeType") != ("application/octet-stream" if kind == "payload"
                                                else "application/json")
                or str(meta.get("size")) != str(wanted["bytes"])
                or meta.get("md5Checksum") != self.local["stage"][kind]["md5"]):
            raise Conflict("Remote Drive file metadata differs")
        self.media_bytes += wanted["bytes"]
        if self.media_bytes > MAX_TOTAL_MEDIA_BYTES:
            raise Conflict("Remote verifier cumulative media bound reached")
        token = self._token()
        def read():
            with self.drive_http.open(self._file_url(file_id, media=True), token) as response:
                if response.status != 200:
                    raise Conflict("Remote Drive media HTTP status differs")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise Conflict("Remote Drive media content encoding differs")
                if response.headers.get("Content-Range") is not None:
                    raise Conflict("Remote Drive media returned a partial range")
                length = response.headers.get("Content-Length")
                if length is not None and str(length) != str(wanted["bytes"]):
                    raise Conflict("Remote Drive media length differs")
                return _digest_stream(response, max_bytes=wanted["bytes"],
                                      deadline=min(self.deadline, self.mono()+MAX_MEDIA_SECONDS),
                                      mono=self.mono)
        actual = self._drive_call(read)
        if (actual["bytes"] != wanted["bytes"]
                or actual["sha256"] != wanted["sha256"]
                or actual["md5"] != self.local["stage"][kind]["md5"]):
            raise Conflict("Remote Drive full media differs from pinned source")
        return {**actual, "id": file_id, "parent_id": folder_id,
                "account_id": self.local["account_id"], "trashed": False,
                "media_readback": True}


def review_exact_cache(*, manifest_path: Path, run_id: str, account_id: str,
                       writer_id: str, authority_db: Path, oauth_file: Path,
                       data_root: Path = Path("/data"), evidence_path: Path | None = None,
                       bucket=None, drive_auth=None, drive_http=None,
                       mono=time.monotonic, sleep=time.sleep) -> dict:
    """Produce one full read-only v2 inventory; optionally append local evidence.

    The first v2 inspection pins local stage, journal, and closed authority.
    The second independently repeats those checks after full remote readback.
    There is no release call or remote mutation in this entry point.
    """
    from gcs_drive_retained_cache_release import (
        MAX_JOURNAL, _json_file, append_evidence, inspect_cache,
    )
    from gcs_drive_retained_overflow import _canonical

    args = {"manifest_path": manifest_path, "run_id": run_id,
            "account_id": account_id, "writer_id": writer_id,
            "authority_db": authority_db, "data_root": data_root}
    local = inspect_cache(**args)
    journal = _json_file(data_root / ("gcs-drive-retained-" + run_id) / "journal.json",
                         MAX_JOURNAL)
    if hashlib.sha256(_canonical(journal)).hexdigest() != local["journal_sha256"]:
        raise Conflict("Remote review journal changed after local inventory")
    try:
        generation = journal["segments"][local["pair_key"]]["index_generation"]
    except (KeyError, TypeError) as error:
        raise Conflict("Remote review mapping generation is unavailable") from error
    manifest, manifest_sha = load_overflow_manifest(
        manifest_path, release_id=run_id, account_id=account_id)
    if manifest_sha != local["manifest_sha256"]:
        raise Conflict("Remote review manifest changed after local inventory")
    verifier = LiveReadOnlyVerifier(
        local=local, manifest=manifest,
        index_generation=generation, oauth_file=oauth_file,
        bucket=bucket, drive_auth=drive_auth, drive_http=drive_http,
        mono=mono, sleep=sleep)
    verified = inspect_cache(**args, verifier=verifier)
    if verified.get("eligible_for_approval") is not True:
        raise Conflict("Remote review did not establish eligibility")
    if evidence_path is not None:
        append_evidence(evidence_path, verified)
    return verified
