#!/usr/bin/env python3
"""Bounded same-region GCE GCS -> Drive drain; no cloud imports at module load.

One sealed payload/receipt at a time, strict duplicate refusal, durable journal,
generation-pinned reads/deletes. GCS release additionally needs the buffer's
local-release marker. HTTP 403 exits 75 with a retry deadline; no retry loop.
The CLI accepts only a pinned, metered, retain-GCS pilot; the historical timer
arguments fail before any cloud or OAuth access.
See docs/execution-plans/2026-10-20-gce-archive-buffer.md.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass

from drive_transfer_authority import TransferAuthority

GIB = 1 << 30
RECEIPT_LIMIT = 1 << 20
PILOT_PAYLOAD_LIMIT = 300 << 20
PILOT_CHARGE_LIMIT = 320 << 20
PILOT_DURATION_LIMIT = 20 * 60
FIELDS = "id,name,size,md5Checksum,mimeType,parents,trashed"
FOLDER = "application/vnd.google-apps.folder"
_DRIVE_REQUEST_LOCK = threading.Lock()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is a separate Drive request and needs a separate ledger slot."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect())


class Conflict(RuntimeError):
    pass


@dataclass(frozen=True)
class ObjectRef:
    name: str
    generation: int
    size: int
    metadata: dict


def digest(path: Path) -> dict:
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf8") as fh:
        os.fchmod(fh.fileno(), 0o600)
        json.dump(value, fh, sort_keys=True, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def matches_drive(info: dict, name: str, expected: dict, parent: str) -> bool:
    return (info.get("name") == name and str(info.get("size")) == str(expected["size"])
            and info.get("md5Checksum") == expected["md5"] and not info.get("trashed", False)
            and parent in info.get("parents", []))


def load_pilot_manifest(path: Path, *, bucket: str, source_id: str,
                        source_epoch: str, drive_root: str, release_id: str) -> tuple[dict, str]:
    """Freeze one exact source pair and an already existing Drive shard parent."""
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or manifest.get("schema") != "catbird.drive.pilot.v1":
        raise Conflict("Unsupported pilot manifest")
    for key, value in (("bucket", bucket), ("source_id", source_id),
                       ("source_epoch", source_epoch), ("drive_root", drive_root),
                       ("release_id", release_id)):
        if manifest.get(key) != value:
            raise Conflict("Pilot manifest mismatch: " + key)
    if manifest.get("retain_gcs") is not True:
        raise Conflict("Pilot must retain GCS and local artifacts")
    if (type(manifest.get("max_duration_seconds")) is not int
            or not 0 < manifest["max_duration_seconds"] <= PILOT_DURATION_LIMIT):
        raise Conflict("Pilot duration exceeds 20 minutes")
    if (type(manifest.get("max_charged_bytes")) is not int
            or not 0 < manifest["max_charged_bytes"] <= PILOT_CHARGE_LIMIT):
        raise Conflict("Pilot charged-byte cap exceeds 320 MiB")
    control = manifest.get("control_allowance_bytes")
    if type(control) is not int or not 0 < control <= RECEIPT_LIMIT:
        raise Conflict("Pilot control allowance must be at most 1 MiB")
    objects = []
    for kind, limit in (("payload", PILOT_PAYLOAD_LIMIT), ("receipt", RECEIPT_LIMIT)):
        item = manifest.get(kind)
        if not isinstance(item, dict) or not isinstance(item.get("object"), str):
            raise Conflict("Pilot " + kind + " identity is missing")
        if (type(item.get("generation")) is not int or item["generation"] <= 0
                or type(item.get("bytes")) is not int or not 0 < item["bytes"] <= limit
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise Conflict("Pilot " + kind + " identity is invalid")
        objects.append(item)
    payload, receipt = objects
    if receipt["object"] != payload["object"] + f".g{payload['generation']}.receipt.json":
        raise Conflict("Pilot receipt does not name pinned payload generation")
    if payload["bytes"] + receipt["bytes"] + control > manifest["max_charged_bytes"]:
        raise Conflict("Pilot media and control allowance exceed charged-byte cap")
    expected_prefix = f"sources/{source_id}/{source_epoch}/segments/"
    if not payload["object"].startswith(expected_prefix):
        raise Conflict("Pilot payload outside source namespace")
    name = payload["object"].split("/")[-2]
    if not re.fullmatch(r"seg_[0-9a-z]{10}\.jss", name):
        raise Conflict("Pilot payload segment name is invalid")
    index = int(name[4:-4], 36)
    shard = f"{index//256:08x}"
    if payload["object"] != expected_prefix + f"{shard}/{name}/{payload['sha256']}.jss":
        raise Conflict("Pilot payload path is not content-addressed as expected")
    parent = manifest.get("drive_parent")
    if (not isinstance(parent, dict) or parent.get("shard") != shard
            or not isinstance(parent.get("id"), str) or not parent["id"]):
        raise Conflict("Pilot Drive shard parent is not pinned")
    return manifest, hashlib.sha256(raw).hexdigest()


class SafeDrive:
    """Reuse only legacy OAuth refresh; never legacy upload/duplicate behavior."""
    @staticmethod
    def assert_complete_visibility():
        # drive.file can omit preexisting files the app has not created/opened.
        # No independently verifiable global listing proof is available yet.
        raise Conflict("Drive file-list uniqueness is unproven under drive.file; independent visibility proof required")

    def __init__(self, oauth: Path, authority: TransferAuthority):
        if authority is None:
            raise Conflict("Drive requests require a transfer authority")
        self.authority = authority
        from gce_segment_offloader import DriveClient
        self.auth = DriveClient(oauth)

    @contextlib.contextmanager
    def _cross_process_dispatch_lock(self):
        db_raw = getattr(self.authority, "db_path", None)
        if not isinstance(db_raw, (str, Path)):
            yield
            return
        db_path = Path(db_raw)
        if not db_path.is_file():
            yield
            return
        lock_path = db_path.with_name(db_path.name + ".dispatch.lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def request(self, url, method="GET", data=None, headers=None, timeout=60, kind="drive_api"):
        # The SQLite slot alone is insufficient if one thread or worker process is
        # delayed after booking it. Serialize slot booking + dispatch across workers,
        # then release the cross-process lock so HTTP body streaming overlaps.
        with _DRIVE_REQUEST_LOCK:
            target = urllib.parse.urlsplit(url)
            if target.scheme != "https" or target.netloc != "www.googleapis.com":
                raise Conflict("Drive request target is not the approved Google API host")
            target_sha256 = hashlib.sha256(url.encode()).hexdigest()
            body_sha256 = hashlib.sha256(data or b"").hexdigest()
            content_range = next((value for key, value in (headers or {}).items()
                                  if key.lower() == "content-range"), "")
            operation_key = hashlib.sha256("\0".join(
                (method.upper(), kind, target_sha256, content_range, body_sha256)).encode()).hexdigest()
            with self._cross_process_dispatch_lock():
                # No credential refresh or Drive request is allowed before validation.
                self.authority.assert_active(timeout)
                token = self.auth.token()
                attempt_id, slot_epoch = self.authority.start_request(
                    len(data or b""), timeout, kind, method=method.upper(),
                    target_category=kind, target_sha256=target_sha256,
                    operation_key=operation_key)
                self.last_attempt_id = attempt_id
                delay = slot_epoch - self.authority.clock()
                if delay > 0:
                    time.sleep(delay)
                # The retained authority also enforces elapsed monotonic time.
                # A wall clock jump or a wake one float ULP before the boundary
                # must wait on this same precharged attempt, never make a second
                # reservation or dispatch early.
                until_dispatch = getattr(self.authority, "seconds_until_dispatch", None)
                if until_dispatch is not None:
                    for _ in range(8):
                        extra = until_dispatch(attempt_id)
                        if type(extra) not in (int, float) or not math.isfinite(extra) or extra < 0:
                            raise Conflict("Drive dispatch wait is invalid")
                        if extra == 0:
                            break
                        time.sleep(extra + 0.001)
                    else:
                        raise Conflict("Drive dispatch floor could not be reached")
                self.authority.assert_active(timeout, slot_at=slot_epoch)
                req = urllib.request.Request(url, data=data, method=method, headers={
                    "Authorization": "Bearer " + token, **(headers or {})})
                # Persist the real start time while holding the dispatch lock.
                # A delayed first request must not let the next one start in a burst.
                self.authority.mark_dispatch(attempt_id)
            try:
                response = _NO_REDIRECT_OPENER.open(req, timeout=timeout)
            except urllib.error.HTTPError as error:
                error.attempt_id = attempt_id
                self.authority.finish_request(attempt_id, error.code)
                raise
            # A timeout/transport failure leaves the precharged attempt unresolved.
            self.authority.finish_request(attempt_id, response.status)
            return response

    def json_request(self, url, method="GET", body=None):
        with self.request(url, method, None if body is None else json.dumps(body).encode(),
                          {"Content-Type": "application/json"}) as response:
            return json.load(response)

    def verify_account(self, account_id: str):
        # about.get is read-only and supports drive.file; failure is a hard gate.
        about = self.json_request("https://www.googleapis.com/drive/v3/about?fields=user(permissionId)")
        observed = about.get("user", {}).get("permissionId")
        if not isinstance(observed, str) or not observed or observed != account_id:
            raise Conflict("Drive OAuth account identity is unavailable or mismatched")

    def named(self, parent: str, name: str):
        escape = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
        query = f"'{escape(parent)}' in parents and name = '{escape(name)}' and trashed = false"
        files, page = [], None
        while True:
            params = {"q": query, "fields": f"nextPageToken,files({FIELDS})", "pageSize": 1000}
            if page:
                params["pageToken"] = page
            result = self.json_request("https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(params))
            files.extend(result.get("files", []))
            if len(files) > 1:
                raise Conflict(f"Duplicate Drive name: {name}")
            page = result.get("nextPageToken")
            if not page:
                return files

    def reserve_id(self):
        return self.json_request("https://www.googleapis.com/drive/v3/files/generateIds?count=1&space=drive")["ids"][0]

    def metadata(self, file_id):
        return self.json_request("https://www.googleapis.com/drive/v3/files/" + urllib.parse.quote(file_id, safe="") +
                                 "?fields=" + urllib.parse.quote(FIELDS))

    def folder(self, parent, name, state, save):
        existing = self.named(parent, name)
        if existing:
            if existing[0].get("mimeType") != FOLDER:
                raise Conflict("Drive shard name is not a folder")
            return existing[0]["id"]
        if "id" not in state:
            state["id"] = self.reserve_id()
            save()
        self.json_request("https://www.googleapis.com/drive/v3/files?fields=id", "POST",
                          {"id": state["id"], "name": name, "parents": [parent], "mimeType": FOLDER})
        existing = self.named(parent, name)
        if len(existing) != 1 or existing[0]["id"] != state["id"]:
            raise Conflict("Drive shard creation not uniquely verified")
        return state["id"]

    def preflight(self, parent, name, expected):
        existing = self.named(parent, name)
        if existing and not matches_drive(existing[0], name, expected, parent):
            raise Conflict(f"Different existing Drive bytes: {name}")
        return existing[0] if existing else None

    def ensure_file(self, path, parent, expected, state, save):
        authority = getattr(self, "authority", None)
        unresolved_fn = getattr(authority, "unresolved_session_media_attempts", None)
        reconcile = getattr(authority, "reconcile_attempt_by_status_probe", None)
        pending_media = (
            unresolved_fn(state["session"])
            if "session" in state and callable(unresolved_fn)
            else []
        )
        if not pending_media:
            existing = self.preflight(parent, path.name, expected)
            if existing:
                return self.metadata(existing["id"])
        # Preallocated, durably saved IDs prevent duplicate creates after uncertain outcomes.
        if "id" not in state:
            state["id"] = self.reserve_id()
            save()
        if "session" not in state:
            metadata = {"id": state["id"], "name": path.name, "parents": [parent]}
            url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=" + urllib.parse.quote(FIELDS)
            with self.request(url, "POST", json.dumps(metadata).encode(), {
                "Content-Type": "application/json", "X-Upload-Content-Type": "application/octet-stream",
                "X-Upload-Content-Length": str(expected["size"]),
            }, kind="resumable_start") as response:
                state["session"] = response.headers["Location"]
            save()
        session = state["session"]
        if not session.startswith("https://www.googleapis.com/"):
            raise Conflict("Untrusted resumable-upload URL")
        # Query acknowledged offset on every invocation, including recovery after a lost response.
        offset = None
        failed_status_ids = []
        last_status_err = None
        for delay in (0.0, 1.0, 2.0, 4.0):
            if delay > 0:
                time.sleep(delay)
            try:
                with self.request(session, "PUT", b"", {"Content-Length": "0", "Content-Range": f"bytes */{expected['size']}"}, kind="resumable_status") as response:
                    completed = json.load(response)
                    probe_attempt_id = getattr(self, "last_attempt_id", None)
                    if callable(reconcile):
                        for pm in pending_media:
                            reconcile(
                                failed_attempt_id=pm["attempt_id"],
                                probe_attempt_id=probe_attempt_id,
                                acknowledged_bytes=expected["size"],
                                committed_bytes=pm["bytes"],
                            )
                        for fid in failed_status_ids:
                            reconcile(
                                failed_attempt_id=fid,
                                probe_attempt_id=probe_attempt_id,
                                acknowledged_bytes=expected["size"],
                                committed_bytes=0,
                            )
                    return self.metadata(completed["id"])
            except urllib.error.HTTPError as error:
                if error.code in (404, 410):
                    raise Conflict(f"Expired Drive resumable session (HTTP {error.code}); preserved ID/session require reconciliation") from error
                if error.code in (500, 502, 503, 504):
                    last_status_err = error
                    fid = getattr(error, "attempt_id", getattr(self, "last_attempt_id", None))
                    if fid and not pending_media:
                        failed_status_ids.append(fid)
                    continue
                if error.code != 308:
                    raise
                acknowledged = error.headers.get("Range", "")
                if acknowledged:
                    match = re.fullmatch(r"bytes=0-(\d+)", acknowledged)
                    if not match:
                        raise Conflict("Invalid Drive upload range")
                    offset = int(match[1]) + 1
                else:
                    offset = 0
                probe_attempt_id = getattr(error, "attempt_id", getattr(self, "last_attempt_id", None))
                if callable(reconcile):
                    for pm in pending_media:
                        committed = max(0, min(pm["bytes"], offset - pm["settled_prior_bytes"]))
                        reconcile(
                            failed_attempt_id=pm["attempt_id"],
                            probe_attempt_id=probe_attempt_id,
                            acknowledged_bytes=offset,
                            committed_bytes=committed,
                        )
                    for fid in failed_status_ids:
                        reconcile(
                            failed_attempt_id=fid,
                            probe_attempt_id=probe_attempt_id,
                            acknowledged_bytes=offset,
                            committed_bytes=0,
                        )
                break
        if offset is None:
            raise last_status_err
        if not 0 <= offset < expected["size"]:
            raise Conflict("Drive upload offset outside payload")
        recoveries_used = 0
        with path.open("rb") as fh:
            fh.seek(offset)
            while offset < expected["size"]:
                chunk = fh.read(min(8 << 20, expected["size"] - offset))
                if not chunk:
                    raise Conflict("Local upload file shortened")
                end = offset + len(chunk)
                try:
                    with self.request(session, "PUT", chunk, {"Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {offset}-{end-1}/{expected['size']}"}, timeout=300,
                        kind="media_chunk") as response:
                        uploaded = json.load(response)
                        if end != expected["size"]:
                            raise Conflict("Drive completed before all bytes")
                        return self.metadata(uploaded["id"])
                except urllib.error.HTTPError as error:
                    if error.code in (404, 410):
                        raise Conflict(f"Expired Drive resumable session (HTTP {error.code}); preserved ID/session require reconciliation") from error
                    if error.code in (500, 502, 503, 504):
                        if recoveries_used >= 2:
                            raise
                        recoveries_used += 1
                        failed_attempt_id = getattr(error, "attempt_id", getattr(self, "last_attempt_id", None))
                        reconcile = getattr(self.authority, "reconcile_attempt_by_status_probe", None)
                        last_probe_err = None
                        probed_offset = None
                        for delay in (1.0, 2.0, 4.0):
                            time.sleep(delay)
                            try:
                                with self.request(session, "PUT", b"", {
                                    "Content-Length": "0",
                                    "Content-Range": f"bytes */{expected['size']}",
                                }, kind="resumable_status") as probe_resp:
                                    completed = json.load(probe_resp)
                                    if end != expected["size"]:
                                        raise Conflict("Drive completed before all bytes")
                                    if callable(reconcile):
                                        reconcile(
                                            failed_attempt_id=failed_attempt_id,
                                            probe_attempt_id=getattr(self, "last_attempt_id", None),
                                            acknowledged_bytes=expected["size"],
                                            committed_bytes=len(chunk),
                                        )
                                    return self.metadata(completed["id"])
                            except urllib.error.HTTPError as probe_err:
                                if probe_err.code in (404, 410):
                                    raise Conflict(f"Expired Drive resumable session (HTTP {probe_err.code}); preserved ID/session require reconciliation") from probe_err
                                if probe_err.code in (500, 502, 503, 504):
                                    last_probe_err = probe_err
                                    continue
                                if probe_err.code != 308:
                                    raise
                                range_hdr = probe_err.headers.get("Range", "")
                                if range_hdr:
                                    match = re.fullmatch(r"bytes=0-(\d+)", range_hdr)
                                    if not match:
                                        raise Conflict("Invalid Drive upload range") from probe_err
                                    acknowledged = int(match[1]) + 1
                                else:
                                    acknowledged = 0
                                if not (offset <= acknowledged <= end and 0 <= acknowledged < expected["size"]):
                                    raise Conflict("Drive status probe range outside chunk window") from probe_err
                                if callable(reconcile):
                                    reconcile(
                                        failed_attempt_id=failed_attempt_id,
                                        probe_attempt_id=getattr(probe_err, "attempt_id", getattr(self, "last_attempt_id", None)),
                                        acknowledged_bytes=acknowledged,
                                        committed_bytes=acknowledged - offset,
                                    )
                                probed_offset = acknowledged
                                break
                        if probed_offset is None:
                            raise last_probe_err
                        offset = probed_offset
                        fh.seek(offset)
                        continue
                    if error.code != 308:
                        raise
                    if error.headers.get("Range") != f"bytes=0-{end-1}":
                        raise Conflict("Drive did not acknowledge exact chunk; restart to query status")
                offset = end
        raise Conflict("Drive did not finalize upload")


class GCS:
    def __init__(self, bucket_name):
        from google.cloud import storage  # Lazy: unit tests require no cloud package or credentials.
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)
        self.bucket_name = bucket_name

    @staticmethod
    def ref(blob):
        return ObjectRef(blob.name, int(blob.generation), int(blob.size), blob.metadata or {})

    def receipts(self, prefix):
        for blob in self.client.list_blobs(self.bucket, prefix=prefix, page_size=100):
            if blob.name.endswith(".receipt.json") and (blob.metadata or {}).get("kind") == "receipt":
                yield self.ref(blob)

    def lookup(self, name, generation=None):
        blob = self.bucket.get_blob(name, generation=generation, retry=None, timeout=60)
        return self.ref(blob) if blob else None

    def download(self, ref, path):
        self.bucket.blob(ref.name, generation=ref.generation).download_to_filename(
            str(path), if_generation_match=ref.generation, checksum="auto", retry=None, timeout=300)

    def delete(self, ref):
        try:
            self.bucket.blob(ref.name, generation=ref.generation).delete(
                if_generation_match=ref.generation, retry=None, timeout=60)
        except Exception as error:
            if getattr(error, "code", None) != 404:
                raise


class Drain:
    def __init__(self, cloud, drive, work: Path, root, source_id, source_epoch,
                 max_segment_bytes=GIB, max_temp_bytes=2*GIB, min_free_bytes=GIB,
                 pilot_manifest=None, manifest_sha256=None, authority=None):
        self.cloud, self.drive, self.work, self.root = cloud, drive, work, root
        self.source_id, self.source_epoch = source_id, source_epoch
        self.pilot_manifest, self.manifest_sha256, self.authority = pilot_manifest, manifest_sha256, authority
        if pilot_manifest is not None and (authority is None or not manifest_sha256):
            raise Conflict("Pilot requires a pinned manifest and transfer authority")
        for component in (source_id, source_epoch):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", component):
                raise Conflict("Invalid source namespace")
        self.prefix = f"sources/{source_id}/{source_epoch}/segments/"
        self.max_segment_bytes, self.max_temp_bytes, self.min_free_bytes = max_segment_bytes, max_temp_bytes, min_free_bytes
        work.mkdir(parents=True, exist_ok=True)
        self.journal_path = work / "journal.json"
        self.state = json.loads(self.journal_path.read_text()) if self.journal_path.exists() else {
            "schema": "catbird.gcs-drive-drain.v1", "bucket": cloud.bucket_name, "root": root,
            "source_id": source_id, "source_epoch": source_epoch, "segments": {}, "folders": {}}
        for key, expected in (("bucket", cloud.bucket_name), ("root", root), ("source_id", source_id), ("source_epoch", source_epoch)):
            if self.state.get(key) != expected:
                raise Conflict("Journal namespace mismatch: " + key)

    def save(self):
        atomic_json(self.journal_path, self.state)

    def space(self, required):
        used = sum(path.stat().st_size for path in (self.work / "stage").rglob("*") if path.is_file())
        if used + required > self.max_temp_bytes or shutil.disk_usage(self.work).free < required + self.min_free_bytes:
            raise Conflict("Temporary-disk bound reached; retained artifacts require review")

    def download_checked(self, ref, path, expected_sha, limit):
        if not 0 < ref.size <= limit:
            raise Conflict("Object exceeds configured size bound")
        if path.exists():
            result = digest(path)
        else:
            temporary = path.with_name(path.name + ".partial")
            if temporary.exists():
                if getattr(self.authority, "resume_open", False) and not temporary.is_symlink():
                    temporary.unlink()
                else:
                    raise Conflict("Preserved partial download requires review: " + str(temporary))
            self.space(ref.size)
            if self.pilot_manifest is not None:
                self.authority.assert_active(300)
            self.cloud.download(ref, temporary)
            result = digest(temporary)
            if result["size"] != ref.size or result["sha256"] != expected_sha:
                raise Conflict("Downloaded bytes do not match immutable identity")
            os.replace(temporary, path)
        if result["size"] != ref.size or result["sha256"] != expected_sha:
            raise Conflict("Local artifact identity changed")
        return result

    def parse_receipt(self, ref, path):
        receipt = json.loads(path.read_text())
        name, index, sha = receipt.get("name", ""), receipt.get("index"), receipt.get("sha256", "")
        if receipt.get("schema") != "catbird.jetstream.archive.v1" or not re.fullmatch(r"seg_[0-9a-z]{10}\.jss", name):
            raise Conflict("Unsupported receipt or segment name")
        if type(index) is not int or int(name[4:-4], 36) != index or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise Conflict("Invalid receipt index/hash")
        if (receipt.get("source_id"), receipt.get("source_epoch")) != (self.source_id, self.source_epoch):
            raise Conflict("Receipt source namespace mismatch")
        pointer = receipt["gcs"]
        generation = int(pointer["generation"])
        expected_object = self.prefix + f"{index//256:08x}/{name}/{sha}.jss"
        if (generation <= 0 or pointer.get("bucket") != self.cloud.bucket_name or pointer.get("object") != expected_object
                or pointer.get("sha256") != sha or pointer.get("bytes") != receipt.get("bytes")
                or ref.name != expected_object + f".g{generation}.receipt.json"):
            raise Conflict("Receipt GCS identity mismatch")
        if (ref.metadata.get("payload_object") != expected_object or str(ref.metadata.get("payload_generation")) != str(generation)):
            raise Conflict("Receipt metadata pointer mismatch")
        if self.pilot_manifest is not None:
            self.authority.assert_active(60)
        payload = self.cloud.lookup(expected_object, generation)
        if payload is None or payload.size != receipt["bytes"] or payload.metadata.get("sha256") != sha:
            raise Conflict("Pinned payload missing or identity differs")
        return receipt, payload

    def verify_remote(self, entry):
        for kind in ("payload", "receipt"):
            expected = entry[kind + "_digest"]
            info = self.drive.metadata(entry["drive"][kind]["id"])
            existing = self.drive.preflight(entry["parent"], entry[kind + "_name"], expected)
            if not matches_drive(info, entry[kind + "_name"], expected, entry["parent"]) or not existing or existing["id"] != info["id"]:
                raise Conflict("Drive pair no longer uniquely verified")

    def owned_stage(self, entry):
        if not re.fullmatch(r"[0-9a-f]{64}", entry["key"]):
            raise Conflict("Invalid owned staging identity")
        stage = self.work / "stage" / entry["key"]
        if stage.is_symlink() or stage.parent.is_symlink():
            raise Conflict("Refusing symlinked staging directory")
        return stage

    def clean_verified_temp(self, entry):
        stage = self.owned_stage(entry)
        if not re.fullmatch(r"seg_[0-9a-z]{10}\.jss", entry["payload_name"]) or entry["receipt_name"] != entry["payload_name"] + ".receipt.json":
            raise Conflict("Invalid owned staging filenames")
        verified = []
        for kind in ("payload", "receipt"):
            path = stage / entry[kind + "_name"]
            if path.is_symlink():
                raise Conflict("Refusing symlinked temporary artifact")
            if path.exists():
                if not path.is_file() or digest(path) != entry[kind + "_digest"]:
                    raise Conflict("Refusing to clean modified temporary artifact")
                verified.append(path)
        for path in verified:
            path.unlink()
        if stage.exists() and not any(stage.iterdir()):
            stage.rmdir()

    def recover_completed_cleanup(self, entry):
        """A completed cloud transfer may still own temp files after a process stop."""
        stage = self.owned_stage(entry)
        if stage.exists():
            self.verify_remote(entry)
            marker = stage / "local-release.json"
            if marker.is_symlink():
                raise Conflict("Refusing symlinked release marker")
            if marker.exists() and (not marker.is_file() or digest(marker) != entry.get("release_digest")):
                raise Conflict("Refusing to clean modified release marker")
            self.clean_verified_temp(entry)
            if marker.exists():
                marker.unlink()
            if stage.exists():
                if any(stage.iterdir()):
                    raise Conflict("Unknown files remain in owned staging directory")
                stage.rmdir()
        entry["cleanup_complete"] = True
        self.save()

    def finish(self, entry):
        self.verify_remote(entry)
        receipt_ref = ObjectRef(**entry["receipt_ref"])
        release = self.cloud.lookup(receipt_ref.name + ".local-release.json")
        if release is None:
            entry["status"] = "awaiting_local_release"
            self.save()
            self.clean_verified_temp(entry)
            return entry["status"]
        # Release marker is small; preserve it on every failure for inspection.
        stage = self.work / "stage" / entry["key"]
        stage.mkdir(parents=True, exist_ok=True)
        release_path = stage / "local-release.json"
        marker_digest = self.download_checked(release, release_path, release.metadata.get("sha256"), RECEIPT_LIMIT)
        marker = json.loads(release_path.read_text())
        expected = {"payload_object": entry["payload_ref"]["name"], "payload_generation": entry["payload_ref"]["generation"],
                    "receipt_object": receipt_ref.name, "receipt_generation": receipt_ref.generation,
                    "sha256": entry["payload_digest"]["sha256"]}
        if marker.get("schema") != "catbird.gcs.local-release.v1" or any(str(marker.get(k)) != str(v) for k, v in expected.items()):
            raise Conflict("Local-release marker does not match pinned pair")
        entry["release_ref"] = asdict(release)
        entry["release_digest"] = marker_digest
        entry["status"] = "drive_verified_releasing"
        self.save()  # Durable authority for resuming after any individual delete.
        for kind in ("payload", "receipt", "release"):
            self.cloud.delete(ObjectRef(**entry[kind + "_ref"]))
            entry[kind + "_removed"] = True
            self.save()
        entry["status"] = "complete"
        self.save()
        self.recover_completed_cleanup(entry)
        return "complete"

    def one(self, ref):
        if self.pilot_manifest is not None:
            return self.pilot_one(ref)
        key = hashlib.sha256(f"{ref.name}:{ref.generation}".encode()).hexdigest()
        entry = self.state["segments"].get(key)
        if entry and entry["status"] == "complete":
            self.recover_completed_cleanup(entry)
            return "complete"
        if entry and entry["status"] in ("drive_verified", "awaiting_local_release", "drive_verified_releasing"):
            # If marker deletion already succeeded but the final state write did not, resume using durable refs.
            if entry["status"] == "drive_verified_releasing" and "release_ref" in entry:
                self.verify_remote(entry)
                for kind in ("payload", "receipt", "release"):
                    self.cloud.delete(ObjectRef(**entry[kind + "_ref"]))
                entry["status"] = "complete"
                self.save()
                self.recover_completed_cleanup(entry)
                return "complete"
            return self.finish(entry)
        stage = self.work / "stage" / key
        stage.mkdir(parents=True, exist_ok=True)
        # Name comes from metadata first, then is checked against signed-by-hash receipt bytes.
        name = ref.metadata.get("segment_name", "")
        if not re.fullmatch(r"seg_[0-9a-z]{10}\.jss", name):
            raise Conflict("Missing safe receipt metadata segment name")
        receipt_path = stage / (name + ".receipt.json")
        receipt_digest = self.download_checked(ref, receipt_path, ref.metadata.get("sha256"), RECEIPT_LIMIT)
        receipt, payload_ref = self.parse_receipt(ref, receipt_path)
        if receipt["name"] != name:
            raise Conflict("Segment metadata/body mismatch")
        payload_path = stage / name
        if not 0 < payload_ref.size <= self.max_segment_bytes:
            raise Conflict("Payload exceeds configured size bound")
        try:
            expected_md5 = base64.b64decode(receipt["gcs"]["md5_hash"], validate=True)
        except (KeyError, ValueError) as error:
            raise Conflict("Missing or invalid payload MD5 in receipt") from error
        if len(expected_md5) != 16:
            raise Conflict("Invalid payload MD5 length")
        payload_digest = {"size": payload_ref.size, "sha256": receipt["sha256"], "md5": expected_md5.hex()}
        if entry is None or "payload_ref" not in entry:
            entry = {"key": key, "status": "receipt_verified", "payload_ref": asdict(payload_ref), "receipt_ref": asdict(ref),
                     "payload_name": name, "receipt_name": receipt_path.name, "payload_digest": payload_digest,
                     "receipt_digest": receipt_digest, "drive": {"payload": {}, "receipt": {}}}
            self.state["segments"][key] = entry
            self.save()
        group = f"{receipt['index']//256:08x}"
        folder_state = self.state["folders"].setdefault(group, {})
        entry["parent"] = self.drive.folder(self.root, group, folder_state, self.save)
        self.save()
        # Refuse either pre-existing conflict before starting either upload.
        for kind in ("payload", "receipt"):
            self.drive.preflight(entry["parent"], entry[kind + "_name"], entry[kind + "_digest"])
        # Conflicting old Drive receipts are detected before downloading the bulk payload.
        actual_payload = self.download_checked(payload_ref, payload_path, receipt["sha256"], self.max_segment_bytes)
        if actual_payload != payload_digest:
            raise Conflict("Payload MD5 differs from receipt")
        entry["status"] = "downloaded"
        self.save()
        for kind, path in (("payload", payload_path), ("receipt", receipt_path)):
            uploaded = self.drive.ensure_file(path, entry["parent"], entry[kind + "_digest"], entry["drive"][kind], self.save)
            if not matches_drive(uploaded, path.name, entry[kind + "_digest"], entry["parent"]):
                raise Conflict("Drive upload metadata mismatch")
            entry["drive"][kind]["id"] = uploaded["id"]
            self.save()
        self.verify_remote(entry)
        entry["status"] = "drive_verified"
        self.save()
        return self.finish(entry)

    def pilot_one(self, ref):
        """Copy exactly the approved pair and retain every GCS/local artifact."""
        manifest = self.pilot_manifest
        if manifest is None or self.authority is None:
            raise Conflict("Pilot authority is absent")
        self.authority.assert_active(300)
        expected_receipt, expected_payload = manifest["receipt"], manifest["payload"]
        if (ref.name != expected_receipt["object"] or ref.generation != expected_receipt["generation"]
                or ref.size != expected_receipt["bytes"]
                or ref.metadata.get("sha256") != expected_receipt["sha256"]):
            raise Conflict("Pinned pilot receipt differs from GCS metadata")
        key = hashlib.sha256(f"{ref.name}:{ref.generation}".encode()).hexdigest()
        entry = self.state["segments"].get(key)
        if entry and (entry.get("pilot_release_id") != manifest["release_id"]
                      or entry.get("pilot_manifest_sha256") != self.manifest_sha256):
            raise Conflict("Pilot journal belongs to a different release")
        if entry and entry.get("status") == "drive_verified_retained":
            raise Conflict("Pilot already verified; a new release is required for further requests")
        if entry and entry.get("status") not in ("receipt_verified", "downloaded"):
            raise Conflict("Pilot journal has an unreconciled state")
        stage = self.work / "stage" / key
        stage.mkdir(parents=True, exist_ok=True)
        name = ref.metadata.get("segment_name", "")
        if not re.fullmatch(r"seg_[0-9a-z]{10}\.jss", name):
            raise Conflict("Missing safe receipt metadata segment name")
        receipt_path = stage / (name + ".receipt.json")
        receipt_digest = self.download_checked(ref, receipt_path, expected_receipt["sha256"], RECEIPT_LIMIT)
        receipt, payload_ref = self.parse_receipt(ref, receipt_path)
        if (payload_ref.name != expected_payload["object"]
                or payload_ref.generation != expected_payload["generation"]
                or payload_ref.size != expected_payload["bytes"]
                or payload_ref.metadata.get("sha256") != expected_payload["sha256"]
                or receipt["sha256"] != expected_payload["sha256"]
                or receipt["name"] != name):
            raise Conflict("Pinned pilot payload differs from receipt or GCS metadata")
        try:
            payload_md5 = base64.b64decode(receipt["gcs"]["md5_hash"], validate=True)
        except (KeyError, ValueError) as error:
            raise Conflict("Missing or invalid payload MD5 in pilot receipt") from error
        if len(payload_md5) != 16:
            raise Conflict("Invalid payload MD5 in pilot receipt")
        payload_digest = {"size": payload_ref.size, "sha256": receipt["sha256"], "md5": payload_md5.hex()}
        self.drive.assert_complete_visibility()
        self.authority.reserve_pair(key, payload_ref.size + ref.size,
                                    manifest["control_allowance_bytes"], {
            "manifest_sha256": self.manifest_sha256,
            "payload_object": payload_ref.name,
            "payload_generation": payload_ref.generation,
            "receipt_object": ref.name,
            "receipt_generation": ref.generation,
        })
        parent_id = manifest["drive_parent"]["id"]
        if entry is None:
            entry = {"key": key, "status": "receipt_verified", "pilot_release_id": manifest["release_id"],
                     "pilot_manifest_sha256": self.manifest_sha256,
                     "payload_ref": asdict(payload_ref), "receipt_ref": asdict(ref),
                     "payload_name": name, "receipt_name": receipt_path.name,
                     "payload_digest": payload_digest, "receipt_digest": receipt_digest,
                     "parent": parent_id,
                     "drive": {"payload": {}, "receipt": {}}}
            self.state["segments"][key] = entry
            self.save()
        elif (entry["parent"] != parent_id or entry["payload_digest"] != payload_digest
              or entry["receipt_digest"] != receipt_digest):
            raise Conflict("Pilot journal pair identity changed")
        self.drive.verify_account(self.authority.account_id)
        # The shard must already exist under the approved root. Never create one in a pilot.
        parent = self.drive.metadata(parent_id)
        named = self.drive.named(self.root, manifest["drive_parent"]["shard"])
        if (parent.get("id") != parent_id or parent.get("name") != manifest["drive_parent"]["shard"]
                or parent.get("mimeType") != FOLDER or parent.get("trashed", False)
                or self.root not in parent.get("parents", []) or len(named) != 1
                or named[0].get("id") != parent_id):
            raise Conflict("Approved Drive shard parent is not uniquely visible")
        # Pin both Drive IDs durably before either file can be created or sent.
        # generateIds is a metered Drive request through SafeDrive.json_request.
        for kind in ("payload", "receipt"):
            existing = self.drive.preflight(parent_id, entry[kind + "_name"], entry[kind + "_digest"])
            pinned_id = entry["drive"][kind].get("id")
            if existing:
                if pinned_id is not None and existing["id"] != pinned_id:
                    raise Conflict("Pilot Drive file ID changed before upload")
                entry["drive"][kind]["id"] = existing["id"]
            elif pinned_id is None:
                entry["drive"][kind]["id"] = self.drive.reserve_id()
            self.save()
        payload_path = stage / name
        if self.download_checked(payload_ref, payload_path, expected_payload["sha256"], PILOT_PAYLOAD_LIMIT) != payload_digest:
            raise Conflict("Pilot payload MD5 differs from receipt")
        entry["status"] = "downloaded"
        self.save()
        for kind, path in (("payload", payload_path), ("receipt", receipt_path)):
            uploaded = self.drive.ensure_file(path, parent_id, entry[kind + "_digest"], entry["drive"][kind], self.save)
            pinned_id = entry["drive"][kind].get("id")
            if (pinned_id is not None and uploaded.get("id") != pinned_id) or not matches_drive(
                    uploaded, path.name, entry[kind + "_digest"], parent_id):
                raise Conflict("Pilot Drive file ID or metadata differs from pinned journal")
            entry["drive"][kind]["id"] = uploaded["id"]
            self.save()
        self.verify_remote(entry)
        entry["status"] = "drive_verified_retained"
        self.save()
        self.authority.close_pair(key)
        return "drive_verified_retained"

    def run_pilot(self):
        if self.pilot_manifest is None:
            raise Conflict("Pinned pilot manifest required")
        receipt = self.pilot_manifest["receipt"]
        self.authority.assert_active(60)
        ref = self.cloud.lookup(receipt["object"], receipt["generation"])
        if ref is None:
            raise Conflict("Pinned pilot receipt generation is missing")
        result = self.pilot_one(ref)
        print(json.dumps({"event": "drive_pilot", "receipt": ref.name,
                          "generation": ref.generation, "status": result}), flush=True)
        return result

    def run(self, maximum, retry_conflicts=False):
        if self.pilot_manifest is not None:
            raise Conflict("Pilot must use exact pinned one-pair selection")
        if not 1 <= maximum <= 1024:
            raise ValueError("max-segments must be between 1 and 1024")
        processed, seen = 0, set()
        self.last_counts = {"complete": 0, "awaiting_local_release": 0, "conflict": 0}
        pending = [ObjectRef(**e["receipt_ref"]) for e in self.state["segments"].values()
                   if (e["status"] != "complete" or self.owned_stage(e).exists())
                   and (retry_conflicts or e["status"] != "conflict")]
        def candidates():
            yield from pending
            yield from self.cloud.receipts(self.prefix)
        for ref in candidates():
            identity = (ref.name, ref.generation)
            if identity in seen:
                continue
            seen.add(identity)
            key = hashlib.sha256(f"{ref.name}:{ref.generation}".encode()).hexdigest()
            existing = self.state["segments"].get(key)
            if existing and existing["status"] == "complete" and not self.owned_stage(existing).exists():
                continue
            if existing and existing["status"] == "conflict":
                if not retry_conflicts:
                    continue
                existing["status"] = existing.pop("resume_status", "downloaded")
                self.save()
            try:
                result = self.one(ref)
            except Conflict as error:
                entry = self.state["segments"].setdefault(key, {"key": key, "receipt_ref": asdict(ref)})
                entry["resume_status"] = entry.get("status", "downloaded")
                entry["status"], entry["conflict"] = "conflict", str(error)
                self.save()
                result = "conflict"
                print(json.dumps({"event": "drain_conflict_preserved", "receipt": ref.name,
                                  "generation": ref.generation, "reason": str(error)}), flush=True)
            self.last_counts[result] = self.last_counts.get(result, 0) + 1
            print(json.dumps({"event": "drain_segment", "receipt": ref.name, "generation": ref.generation, "status": result}), flush=True)
            processed += 1
            if processed >= maximum:
                break
        print(json.dumps({"event": "drain_batch", "processed": processed, **self.last_counts,
                          "retained_conflicts": sum(e["status"] == "conflict" for e in self.state["segments"].values())}), flush=True)
        return processed


def gce_region():
    if sys.platform != "linux":
        raise Conflict("Raw drain is allowed only on same-region GCE, never on the Mac")
    request = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/instance/zone", headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(request, timeout=3) as response:
        if response.headers.get("Metadata-Flavor") != "Google":
            raise Conflict("GCE metadata identity not verified")
        zone = response.read(512).decode().rsplit("/", 1)[-1]
    return zone.rsplit("-", 1)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-epoch", required=True)
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--oauth-file", type=Path, default=Path("/root/.config/catbird-drive/oauth.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("/data/gcs-drive-drain"))
    parser.add_argument("--max-segments", type=int, default=1)
    parser.add_argument("--max-segment-bytes", type=int, default=GIB)
    parser.add_argument("--max-temp-bytes", type=int, default=2*GIB)
    parser.add_argument("--once", action="store_true", required=True)
    parser.add_argument("--retry-conflicts", action="store_true")
    parser.add_argument("--authority-db", required=True, type=Path)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--writer-id", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--retain-gcs", action="store_true")
    args = parser.parse_args(argv)
    if (args.max_segments != 1 or args.retry_conflicts or not args.retain_gcs
            or not 0 < args.max_segment_bytes <= args.max_temp_bytes <= 16*GIB):
        parser.error("Invalid pinned retain-GCS pilot limits")
    if not args.work_dir.resolve().is_relative_to(Path("/data")):
        parser.error("work-dir must remain under /data")
    try:
        manifest, manifest_sha256 = load_pilot_manifest(
            args.pilot_manifest, bucket=args.bucket, source_id=args.source_id,
            source_epoch=args.source_epoch, drive_root=args.drive_root,
            release_id=args.release_id)
        SafeDrive.assert_complete_visibility()
        authority = TransferAuthority.open_existing(
            args.authority_db, release_id=args.release_id, writer_id=args.writer_id,
            account_id=args.account_id, manifest_sha256=manifest_sha256)
        authority.validate_manifest_limits(manifest["max_charged_bytes"],
                                           manifest["max_duration_seconds"])
        authority.assert_active(300)
        region = gce_region()
        args.work_dir.mkdir(parents=True, exist_ok=True)
        with (args.work_dir / "drain.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            retry_path = args.work_dir / "retry-after.json"
            if retry_path.exists() and json.loads(retry_path.read_text())["retry_after_epoch"] > time.time():
                print(json.dumps({"event": "drain_backoff_active", "state": str(retry_path)}))
                return 75
            try:
                cloud = GCS(args.bucket)
                if region != "us-west1" or cloud.bucket.location.lower() != region:
                    raise Conflict("GCE and bucket must both be single-region us-west1")
                for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
                    with contextlib.suppress(FileNotFoundError, PermissionError):
                        if b"gce_segment_offloader.py" in cmdline.read_bytes():
                            raise Conflict("Stop legacy Drive offloader before drain")
                drain = Drain(cloud, SafeDrive(args.oauth_file, authority), args.work_dir, args.drive_root,
                              args.source_id, args.source_epoch, min(args.max_segment_bytes, PILOT_PAYLOAD_LIMIT),
                              args.max_temp_bytes, pilot_manifest=manifest,
                              manifest_sha256=manifest_sha256, authority=authority)
                drain.run_pilot()
            except Exception as error:
                code = getattr(error, "code", None)
                if code in (403, 429):
                    deadline = int(time.time()) + 3600
                    atomic_json(retry_path, {"retry_after_epoch": deadline, "http_status": code})
                    print(json.dumps({"event": "drain_quota_backoff", "http_status": code, "retry_after_epoch": deadline}), flush=True)
                    return 75
                raise
        return 0
    except Exception as error:
        # HTTP errors can contain resumable URL tokens; never print their text.
        print(json.dumps({"event": "drain_failed_preserved", "error_type": type(error).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
