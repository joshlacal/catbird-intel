#!/usr/bin/env python3
"""One-pair retained GCS -> Drive overflow copier; no destructive drain path."""
from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import sys
import time
import urllib.error
import urllib.parse

from drive_transfer_authority import TransferAuthority
from gcs_drive_diagnostic import (DiagnosticSafeDrive, DriveHttpStop,
                                  PAYLOAD as COMPLETED_DIAGNOSTIC_PAYLOAD,
                                  _arm_parent_death, _stop_worker_group)
from gcs_drive_drain import (Conflict, Drain, GCS, FOLDER, PILOT_PAYLOAD_LIMIT,
                             PILOT_CHARGE_LIMIT, PILOT_DURATION_LIMIT,
                             RECEIPT_LIMIT, gce_region, matches_drive)


SCHEMA = "catbird.drive.retained-overflow.v1"
FINAL_STATUS = "offloaded_retained_unqualified"
INDEX_SCHEMA = "catbird.drive.retained-overflow-index.v1"
SOURCE_ID = "gce-selfhost-001"
SOURCE_EPOCH = "buffer-admission-20261020"
BUCKET = "catbird-436600-jetstream-buffer-20261020"
DRIVE_ROOT = "16Zd7n4pwTSDMDsnP9chX2Ci0r0ahEy45"
CONTROL_ALLOWANCE = 1 << 20
INDEX_PREFIX = "indexes/drive-retained-overflow/v1/"
INDEX_OBJECT_LIMIT = 16 << 10
WATCHDOG_MARGIN_SECONDS = 30
CAPTURE_SEGMENTS_DIR = Path("/data/jetstream/segments")
CAPTURE_RESTART_FLOOR_BYTES = 100 * (1 << 30)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Conflict("Overflow JSON contains duplicate keys")
        result[key] = value
    return result


def _parse_manifest(raw: bytes) -> dict:
    def bad_constant(_value):
        raise Conflict("Overflow manifest contains nonfinite JSON number")
    try:
        value = json.loads(raw, object_pairs_hook=_no_duplicate_keys,
                           parse_constant=bad_constant)
    except (UnicodeError, ValueError) as error:
        raise Conflict("Overflow manifest JSON is invalid") from error
    if not isinstance(value, dict):
        raise Conflict("Overflow manifest must be a JSON object")
    return value


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def pair_key(manifest: dict) -> str:
    # The payload generation is the unique retained copy target. A newly
    # generated receipt for that same payload must not admit a second copy.
    payload = manifest["payload"]
    source = f"{manifest['bucket']}:{payload['object']}:{payload['generation']}"
    return hashlib.sha256(source.encode()).hexdigest()


def load_overflow_manifest(path: Path, *, release_id: str, account_id: str) -> tuple[dict, str]:
    """Validate one generation-pinned pair, then bind its raw bytes by hash.

    Review of the packet and account-wide writer ownership is external. A
    manifest hash is an identity binding, not a signature or authorization.
    """
    raw = path.read_bytes()
    if not raw or len(raw) > (1 << 20):
        raise Conflict("Overflow manifest bytes are missing or too large")
    manifest = _parse_manifest(raw)
    validate_overflow_manifest(manifest, release_id=release_id, account_id=account_id)
    return manifest, hashlib.sha256(raw).hexdigest()


def validate_overflow_manifest(manifest: dict, *, release_id: str, account_id: str) -> None:
    """Recheck the exact manifest at object construction, not only file load."""
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise Conflict("Unsupported retained-overflow manifest")
    if not re.fullmatch(r"[0-9a-f]{32}", release_id):
        raise Conflict("Overflow release ID must be UUID hex")
    expected = {"release_id": release_id, "account_id": account_id,
                "bucket": BUCKET, "source_id": SOURCE_ID,
                "source_epoch": SOURCE_EPOCH, "drive_root": DRIVE_ROOT,
                "retain_gcs": True, "retain_local": True,
                "archive_qualified": False, "control_allowance_bytes": CONTROL_ALLOWANCE}
    for field, value in expected.items():
        if manifest.get(field) != value or type(manifest.get(field)) is not type(value):
            raise Conflict("Overflow manifest mismatch: " + field)
    duration = manifest.get("max_duration_seconds")
    if type(duration) is not int or not 0 < duration <= PILOT_DURATION_LIMIT:
        raise Conflict("Overflow duration exceeds one 20-minute run")
    refs = []
    for kind, limit in (("payload", PILOT_PAYLOAD_LIMIT), ("receipt", RECEIPT_LIMIT)):
        item = manifest.get(kind)
        if not isinstance(item, dict):
            raise Conflict("Overflow " + kind + " identity is missing")
        if (not isinstance(item.get("object"), str)
                or type(item.get("generation")) is not int or item["generation"] <= 0
                or type(item.get("bytes")) is not int or not 0 < item["bytes"] <= limit
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise Conflict("Overflow " + kind + " identity is invalid")
        refs.append(item)
    payload, receipt = refs
    if (payload["object"] == COMPLETED_DIAGNOSTIC_PAYLOAD["object"]
            and payload["generation"] == COMPLETED_DIAGNOSTIC_PAYLOAD["generation"]):
        raise Conflict("Completed diagnostic payload is excluded from retained recopy")
    prefix = f"sources/{SOURCE_ID}/{SOURCE_EPOCH}/segments/"
    if not payload["object"].startswith(prefix):
        raise Conflict("Overflow payload is outside the retained GCS namespace")
    components = payload["object"][len(prefix):].split("/")
    if len(components) != 3 or not re.fullmatch(r"seg_[0-9a-z]{10}\.jss", components[1]):
        raise Conflict("Overflow payload segment path is malformed")
    segment_index = int(components[1][4:-4], 36)
    if components != [f"{segment_index//256:08x}", components[1], payload["sha256"] + ".jss"]:
        raise Conflict("Overflow payload path is not content-addressed")
    if receipt["object"] != payload["object"] + f".g{payload['generation']}.receipt.json":
        raise Conflict("Overflow receipt does not pin the payload generation")
    if manifest.get("max_charged_bytes") != payload["bytes"] + receipt["bytes"] + CONTROL_ALLOWANCE:
        raise Conflict("Overflow byte reservation must equal both media objects plus control")
    if manifest["max_charged_bytes"] > PILOT_CHARGE_LIMIT:
        raise Conflict("Overflow one-pair budget exceeds 320 MiB")
    if "drive_parent" in manifest or "delete_gcs" in manifest:
        raise Conflict("Overflow manifest cannot select a historical shard or delete source")
    name = "catbird-overflow-" + pair_key(manifest)[:32]
    if manifest.get("folder_name") != name:
        raise Conflict("Overflow folder name differs from exact pair identity")
    if (manifest.get("index_bucket") != BUCKET
            or manifest.get("index_object") != INDEX_PREFIX + pair_key(manifest) + ".json"):
        raise Conflict("Overflow shared index destination is not pinned to this pair")


def create_pinned_folder(drive, *, folder_id: str, name: str, root: str) -> dict:
    body = json.dumps({"id": folder_id, "name": name, "parents": [root],
                       "mimeType": FOLDER}, sort_keys=True, separators=(",", ":")).encode()
    url = "https://www.googleapis.com/drive/v3/files?fields=" + urllib.parse.quote(
        "id,name,mimeType,parents,trashed")
    with drive.request(url, "POST", body, {"Content-Type": "application/json"},
                       kind="overflow_folder_create") as response:
        return json.load(response)


class OverflowGCS(GCS):
    """Version-pinned shared index I/O; no source-object deletion method used."""

    def publish_index(self, name: str, data: bytes, digest: str) -> None:
        if (not name.startswith(INDEX_PREFIX) or not name.endswith(".json")
                or not 0 < len(data) <= INDEX_OBJECT_LIMIT
                or hashlib.sha256(data).hexdigest() != digest):
            raise Conflict("Overflow shared index write identity differs")
        # Current-only reads miss soft-deleted and noncurrent generations.
        # Any prior exact-name generation disqualifies a first create.
        for options in ({"versions": True}, {"soft_deleted": True}):
            for prior in self.client.list_blobs(self.bucket, prefix=name, **options):
                if prior.name == name:
                    raise Conflict("Overflow shared mapping name has a prior generation")
        # The generation precondition fences a concurrent create after listing.
        blob = self.bucket.blob(name)
        blob.metadata = {"sha256": digest, "kind": "drive_retained_overflow_index"}
        blob.upload_from_string(data, content_type="application/json",
                                if_generation_match=0, checksum="auto",
                                retry=None, timeout=60)

    def read_index(self, name: str) -> tuple[int, bytes] | None:
        if not name.startswith(INDEX_PREFIX) or not name.endswith(".json"):
            raise Conflict("Overflow shared index read target differs")
        blob = self.bucket.get_blob(name, retry=None, timeout=60)
        if blob is None:
            return None
        if (type(blob.size) is not int or not 0 < blob.size <= INDEX_OBJECT_LIMIT
                or (blob.metadata or {}).get("kind") != "drive_retained_overflow_index"
                or not re.fullmatch(r"[0-9a-f]{64}", (blob.metadata or {}).get("sha256", ""))):
            raise Conflict("Overflow shared index metadata or size is invalid")
        generation = int(blob.generation)
        data = self.bucket.blob(name, generation=generation).download_as_bytes(
            if_generation_match=generation, checksum="auto", retry=None, timeout=60)
        if (not isinstance(data, bytes)
                or (blob.metadata or {}).get("sha256") != hashlib.sha256(data).hexdigest()
                or (blob.metadata or {}).get("kind") != "drive_retained_overflow_index"):
            raise Conflict("Overflow shared index bytes or metadata differ")
        return generation, data


class RetainedOverflow(Drain):
    """One exact GCS pair per call, copied to a new app-created Drive folder."""

    def __init__(self, cloud, drive, authority, work: Path, manifest: dict,
                 manifest_sha256: str, manifest_bytes: bytes, *,
                 capture_segments_dir: Path = CAPTURE_SEGMENTS_DIR,
                 min_capture_free_bytes: int = CAPTURE_RESTART_FLOOR_BYTES):
        if (authority is None or authority.manifest_sha256 != manifest_sha256
                or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)):
            raise Conflict("Overflow authority does not bind the reviewed manifest")
        if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
            raise Conflict("Overflow raw manifest differs from reviewed digest")
        verified = _parse_manifest(manifest_bytes)
        if _canonical(verified) != _canonical(manifest):
            raise Conflict("Overflow in-memory manifest differs from reviewed bytes")
        validate_overflow_manifest(verified, release_id=authority.release_id,
                                   account_id=authority.account_id)
        if getattr(cloud, "bucket_name", None) != BUCKET:
            raise Conflict("Overflow GCS client bucket differs from reviewed source")
        if work.is_symlink():
            raise Conflict("Overflow work directory is symlinked")
        work.mkdir(parents=True, exist_ok=True)
        if (work / "journal.json").is_symlink() or (work / "stage").is_symlink():
            raise Conflict("Overflow journal or stage is symlinked")
        lock_fd = os.open(work / "overflow.lock", os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        self._lock = os.fdopen(lock_fd, "a")
        os.fchmod(self._lock.fileno(), 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._lock.close()
            raise Conflict("Another retained-overflow writer owns this work directory") from error
        try:
            super().__init__(cloud, drive, work, DRIVE_ROOT, SOURCE_ID, SOURCE_EPOCH,
                             max_segment_bytes=PILOT_PAYLOAD_LIMIT,
                             pilot_manifest=copy.deepcopy(verified), manifest_sha256=manifest_sha256,
                             authority=authority)
        except BaseException:
            self.close()
            raise
        self.index_path = work / "retained-index.jsonl"
        self._verified_manifest = _canonical(verified)
        self.capture_segments_dir = capture_segments_dir
        if (type(min_capture_free_bytes) is not int or
                not 0 < min_capture_free_bytes <= CAPTURE_RESTART_FLOOR_BYTES):
            raise Conflict("Overflow capture floor is invalid")
        self.min_capture_free_bytes = min_capture_free_bytes

    def _assert_manifest_unchanged(self) -> None:
        if _canonical(self.pilot_manifest) != self._verified_manifest:
            raise Conflict("Overflow manifest changed after construction")

    def space(self, required):
        """Protect capture restart headroom before each GCS stage download."""
        if type(required) is not int or required <= 0:
            raise Conflict("Overflow stage reservation is invalid")
        try:
            if os.stat(self.capture_segments_dir).st_dev != os.stat(self.work).st_dev:
                raise Conflict("Overflow stage and capture use different filesystems")
            free = shutil.disk_usage(self.capture_segments_dir).free
        except OSError as error:
            raise Conflict("Capture segment filesystem is unavailable") from error
        if free - required < self.min_capture_free_bytes:
            raise Conflict("Overflow stage would breach capture restart headroom")
        super().space(required)

    def close(self) -> None:
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def one(self, _ref):
        raise Conflict("Historical Drain.one may release GCS; retained overflow forbids it")

    def finish(self, _entry):
        raise Conflict("Historical Drain.finish may release GCS; retained overflow forbids it")

    def clean_verified_temp(self, _entry):
        raise Conflict("Retained overflow never cleans local stage")

    def recover_completed_cleanup(self, _entry):
        raise Conflict("Retained overflow forbids inherited local cleanup")

    def pilot_one(self, _ref):
        raise Conflict("Retained overflow forbids inherited pilot upload")

    def run_pilot(self):
        raise Conflict("Retained overflow requires exact reviewed run_one")

    def run(self, _maximum, retry_conflicts=False):
        raise Conflict("Retained overflow forbids historical multi-segment drain")

    def parse_receipt(self, ref, path):
        raw = path.read_bytes()
        if len(raw) != ref.size or not 0 < len(raw) <= RECEIPT_LIMIT:
            raise Conflict("Overflow receipt bytes differ or exceed bound")
        receipt = _parse_manifest(raw)  # Parse exactly once, with no duplicate keys.
        name, index, sha = receipt.get("name", ""), receipt.get("index"), receipt.get("sha256", "")
        if (receipt.get("schema") != "catbird.jetstream.archive.v1"
                or not isinstance(name, str)
                or not re.fullmatch(r"seg_[0-9a-z]{10}\.jss", name)
                or type(index) is not int or int(name[4:-4], 36) != index
                or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
            raise Conflict("Overflow receipt segment identity is malformed")
        if (receipt.get("source_id"), receipt.get("source_epoch")) != (self.source_id, self.source_epoch):
            raise Conflict("Overflow receipt source namespace differs")
        pointer = receipt.get("gcs")
        if not isinstance(pointer, dict):
            raise Conflict("Overflow receipt lacks GCS pointer")
        generation = pointer.get("generation")
        if type(generation) is str and re.fullmatch(r"[1-9][0-9]*", generation):
            generation = int(generation)
        if type(generation) is not int or generation <= 0:
            raise Conflict("Overflow receipt payload generation is invalid")
        expected_object = self.prefix + f"{index//256:08x}/{name}/{sha}.jss"
        if (pointer.get("bucket") != self.cloud.bucket_name
                or pointer.get("object") != expected_object
                or pointer.get("sha256") != sha
                or type(receipt.get("bytes")) is not int
                or pointer.get("bytes") != receipt["bytes"]
                or ref.name != expected_object + f".g{generation}.receipt.json"
                or ref.metadata.get("payload_object") != expected_object
                or str(ref.metadata.get("payload_generation")) != str(generation)):
            raise Conflict("Overflow receipt GCS identity differs")
        self.authority.assert_active(60)
        payload = self.cloud.lookup(expected_object, generation)
        if (payload is None or payload.size != receipt["bytes"]
                or payload.metadata.get("sha256") != sha):
            raise Conflict("Overflow receipt payload generation is unavailable")
        return receipt, payload

    def _stage(self, key: str, name: str) -> tuple[Path, Path]:
        stage = self.work / "stage" / key
        if stage.is_symlink() or stage.parent.is_symlink():
            raise Conflict("Overflow stage is symlinked")
        stage.mkdir(parents=True, exist_ok=True)
        receipt, payload = stage / (name + ".receipt.json"), stage / name
        for path in (receipt, payload):
            if path.is_symlink() or path.with_name(path.name + ".partial").is_symlink():
                raise Conflict("Overflow staged file is symlinked")
        return receipt, payload

    def _index_records(self) -> list[dict]:
        if self.index_path.is_symlink():
            raise Conflict("Overflow ID index is symlinked")
        if not self.index_path.exists():
            return []
        try:
            raw = self.index_path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise Conflict("Overflow ID index has a torn final append")
            lines = raw.decode("utf-8").splitlines()
            records = [json.loads(line) for line in lines]
        except (OSError, UnicodeError, ValueError) as error:
            raise Conflict("Overflow ID index is incomplete or malformed") from error
        if any(not isinstance(row, dict) or row.get("schema") != INDEX_SCHEMA for row in records):
            raise Conflict("Overflow ID index schema differs")
        keys = [row.get("pair_key") for row in records]
        if len(keys) != len(set(keys)):
            raise Conflict("Overflow ID index contains a duplicate pair")
        return records

    def _append_index(self, record: dict) -> None:
        for row in self._index_records():
            if row.get("pair_key") == record["pair_key"]:
                if row != record:
                    raise Conflict("Existing overflow ID mapping differs")
                return
        data = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        fd = os.open(self.index_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        try:
            if os.write(fd, data) != len(data):
                raise Conflict("Overflow ID index append was short")
            os.fsync(fd)
        finally:
            os.close(fd)
        directory = os.open(self.work, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _pin_ids(self, entry: dict) -> None:
        for kind in ("folder", "payload", "receipt"):
            if not entry["drive"][kind].get("id"):
                entry["drive"][kind]["id"] = self.drive.reserve_id()
                self.save()
        ids = [entry["drive"][kind]["id"] for kind in ("folder", "payload", "receipt")]
        if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 3:
            raise Conflict("Overflow Drive IDs are missing or duplicated")
        entry["status"] = "ids_pinned"
        self.save()

    def _verify_folder(self, entry: dict) -> None:
        info = self.drive.metadata(entry["drive"]["folder"]["id"])
        if (info.get("id") != entry["drive"]["folder"]["id"]
                or info.get("name") != self.pilot_manifest["folder_name"]
                or info.get("mimeType") != FOLDER or info.get("trashed") is not False
                or info.get("parents") != [self.root]):
            raise Conflict("Pinned overflow folder identity differs")

    def _ensure_folder(self, entry: dict) -> None:
        if entry["status"] == "folder_create_inflight":
            try:
                self._verify_folder(entry)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    raise Conflict("Uncertain overflow folder create is not visible; no retry") from error
                raise
            entry["status"] = "folder_reconciled_unsettled"
            self.save()
            raise Conflict("Uncertain overflow folder create needs charge review")
        if entry["status"] == "ids_pinned":
            entry["status"] = "folder_create_inflight"
            self.save()
            info = create_pinned_folder(self.drive, folder_id=entry["drive"]["folder"]["id"],
                                        name=self.pilot_manifest["folder_name"], root=self.root)
            if info.get("id") != entry["drive"]["folder"]["id"]:
                raise Conflict("Overflow folder create returned different ID")
            entry["status"] = "folder_create_ack"
            self.save()
        if entry["status"] in ("folder_create_ack", "folder_verified"):
            self._verify_folder(entry)
            entry["status"] = "folder_verified"
            self.save()
            return
        raise Conflict("Overflow folder state is not resumable")

    def _index_record(self, entry: dict) -> dict:
        def source(kind):
            reviewed = self.pilot_manifest[kind]
            return {field: reviewed[field] for field in ("object", "generation", "bytes", "sha256")}
        return {"schema": INDEX_SCHEMA, "pair_key": entry["key"],
                "manifest_sha256": entry["manifest_sha256"],
                "index_bucket": self.pilot_manifest["index_bucket"],
                "index_object": self.pilot_manifest["index_object"],
                "payload_ref": source("payload"), "receipt_ref": source("receipt"),
                "payload_digest": entry["payload_digest"],
                "receipt_digest": entry["receipt_digest"],
                "drive_account_id": self.authority.account_id,
                "drive_root": self.root, "folder_id": entry["drive"]["folder"]["id"],
                "payload_id": entry["drive"]["payload"]["id"],
                "receipt_id": entry["drive"]["receipt"]["id"],
                "retained_gcs": True, "retained_local": True,
                "archive_qualified": False, "status": FINAL_STATUS}

    def _publish_shared_index(self, entry: dict, record: dict) -> int:
        """Create one immutable GCS mapping. Lost replies are read, never reposted."""
        data = _canonical(record)
        digest = hashlib.sha256(data).hexdigest()
        if not 0 < len(data) <= INDEX_OBJECT_LIMIT:
            raise Conflict("Overflow shared mapping body exceeds 16 KiB")
        if entry.get("mapping_sha256") not in (None, digest):
            raise Conflict("Overflow mapping digest differs from journal")
        entry["mapping_sha256"] = digest
        name = self.pilot_manifest["index_object"]
        reservation_id = self.authority.reserve_index_create(entry["key"], digest,
                                                              name, len(data))
        if (entry.get("index_reservation_id") is not None
                and entry["index_reservation_id"] != reservation_id):
            raise Conflict("Overflow shared index reservation changed")
        entry["index_reservation_id"] = reservation_id
        self.save()
        if entry["status"] == "index_pending":
            entry["status"] = "index_put_inflight"
            self.save()
            try:
                self.cloud.publish_index(name, data, digest)
            except Conflict:
                raise
            except Exception as error:
                # A lost response may have created the exact object. Read it
                # once; never issue a second create or choose another name.
                observed_after_error = self.cloud.read_index(name)
                if observed_after_error is None:
                    raise Conflict("Uncertain shared mapping is not visible; no repost") from error
        elif entry["status"] != "index_put_inflight":
            raise Conflict("Overflow shared index state is not recoverable")
        observed = self.cloud.read_index(name)
        if observed is None:
            raise Conflict("Uncertain shared mapping is not visible; no repost")
        generation, raw = observed
        if (type(generation) is not int or generation <= 0 or raw != data
                or hashlib.sha256(raw).hexdigest() != digest):
            raise Conflict("Shared overflow mapping differs from reviewed bytes")
        entry["index_generation"] = generation
        entry["status"] = "index_verified"
        self.save()
        return generation

    def _verify_drive_mapping(self, entry: dict) -> None:
        self.drive.verify_account(self.authority.account_id)
        self._verify_folder(entry)
        folder_id = entry["drive"]["folder"]["id"]
        for kind in ("payload", "receipt"):
            info = self.drive.metadata(entry["drive"][kind]["id"])
            if (info.get("id") != entry["drive"][kind]["id"]
                    or not matches_drive(info, entry[kind + "_name"], entry[kind + "_digest"], folder_id)):
                raise Conflict("Overflow pair cannot be verified by pinned IDs")

    def _recover_mapping(self, entry: dict, status: str) -> dict:
        """Recovery after Drive sends: only verify fixed IDs and finish index."""
        if status not in ("verified", "index_pending", "closed"):
            raise Conflict("Overflow authority is not in no-send recovery state")
        # mark_pair_verified is a durable attestation of the exact pinned IDs.
        # Afterwards SafeDrive forbids all requests. Replay may only publish
        # the mapping already bound to that ledger attestation.
        record = self._index_record(entry)
        digest = hashlib.sha256(_canonical(record)).hexdigest()
        if entry.get("mapping_sha256") != digest:
            raise Conflict("Overflow verified mapping digest differs")
        if status == "verified":
            self.authority.close_pair(entry["key"])
            status = "index_pending"
        if status == "index_pending":
            if entry["status"] in ("drive_verified", "mapping_verified"):
                entry["status"] = "index_pending"
                self.save()
            if entry["status"] in ("index_pending", "index_put_inflight"):
                self._publish_shared_index(entry, record)
            if entry["status"] != "index_verified":
                raise Conflict("Overflow mapping publication is incomplete")
            self.authority.mark_index_committed(entry["key"], digest)
        elif entry["status"] not in ("index_verified", FINAL_STATUS):
            raise Conflict("Closed overflow authority has no verified shared index")
        observed = self.cloud.read_index(self.pilot_manifest["index_object"])
        if (observed is None or observed[1] != _canonical(record)
                or (entry.get("index_generation") is not None
                    and observed[0] != entry["index_generation"])):
            raise Conflict("Committed overflow mapping is not readable")
        self._append_index(record)  # VM-local cache only; shared GCS is authoritative.
        entry["status"] = FINAL_STATUS
        self.save()
        return {"status": FINAL_STATUS, "pair_key": entry["key"],
                "folder_id": record["folder_id"], "payload_id": record["payload_id"],
                "receipt_id": record["receipt_id"], "index_generation": observed[0],
                "archive_qualified": False}

    def recover_closed_cache(self, mapping_sha256: str) -> dict:
        """Rebuild optional VM cache from an already closed immutable GCS index."""
        self._assert_manifest_unchanged()
        key = pair_key(self.pilot_manifest)
        entry = self.state["segments"].get(key)
        if not entry or entry.get("mapping_sha256") != mapping_sha256:
            raise Conflict("Closed overflow pair lacks matching verified local record")
        if entry.get("status") not in ("drive_verified", "index_pending",
                                       "index_put_inflight", "index_verified", FINAL_STATUS):
            raise Conflict("Closed overflow journal state predates verified IDs")
        record = self._index_record(entry)
        if hashlib.sha256(_canonical(record)).hexdigest() != mapping_sha256:
            raise Conflict("Closed overflow mapping digest differs")
        observed = self.cloud.read_index(self.pilot_manifest["index_object"])
        if (observed is None or observed[1] != _canonical(record)
                or (entry.get("index_generation") is not None
                    and observed[0] != entry["index_generation"])):
            raise Conflict("Closed overflow shared index differs")
        self._append_index(record)
        entry["index_generation"] = observed[0]
        entry["status"] = FINAL_STATUS
        self.save()
        return {"status": FINAL_STATUS, "pair_key": key, "index_generation": observed[0],
                "archive_qualified": False}

    def run_one(self) -> dict:
        """Run one reviewed pair; uncertain writes stop rather than replay media."""
        self._assert_manifest_unchanged()
        manifest = self.pilot_manifest
        key = pair_key(manifest)
        if set(self.state["segments"]) - {key}:
            raise Conflict("Overflow work directory contains another source pair")
        entry = self.state["segments"].get(key)
        indexed = next((row for row in self._index_records() if row["pair_key"] == key), None)
        if entry:
            if (entry.get("manifest_sha256") != self.manifest_sha256
                    or entry.get("release_id") != manifest["release_id"]
                    or entry.get("archive_qualified") is not False):
                raise Conflict("Overflow journal identity differs")
            if entry.get("status") in ("file_upload_inflight", "folder_reconciled_unsettled"):
                raise Conflict("Overflow replay requires review; no second media send")
            if entry.get("status") not in ("source_verified", "ids_pinned", "folder_create_inflight",
                                           "folder_create_ack", "folder_verified", "drive_verified",
                                           "index_pending", "index_put_inflight", "index_verified",
                                           FINAL_STATUS):
                raise Conflict("Overflow journal state is not resumable")
            pair_state = self.authority.pair_status(key, self.manifest_sha256)
            if pair_state in ("verified", "index_pending", "closed"):
                return self._recover_mapping(entry, pair_state)
            if entry["status"] == "drive_verified" and pair_state == "active":
                # Crash between durable ID verification and ledger attestation.
                # This branch does no media send or ID allocation.
                self._verify_drive_mapping(entry)
                record = self._index_record(entry)
                digest = hashlib.sha256(_canonical(record)).hexdigest()
                if entry.get("mapping_sha256") != digest:
                    raise Conflict("Overflow pending mapping identity differs")
                self.authority.mark_pair_verified(key, digest)
                return self._recover_mapping(entry, "verified")
            if entry["status"] in ("drive_verified", "index_pending", "index_put_inflight",
                                   "index_verified", FINAL_STATUS):
                raise Conflict("Overflow journal is ahead of authority; no second send")
        if indexed is not None:
            raise Conflict("Pair already has an overflow ID mapping; inspect it instead of recopying")
        self.authority.assert_active(60)
        wanted = manifest["receipt"]
        ref = self.cloud.lookup(wanted["object"], wanted["generation"])
        if (ref is None or ref.name != wanted["object"] or ref.generation != wanted["generation"]
                or ref.size != wanted["bytes"] or ref.metadata.get("sha256") != wanted["sha256"]):
            raise Conflict("Overflow GCS receipt generation or digest differs")
        name = ref.metadata.get("segment_name", "")
        if not re.fullmatch(r"seg_[0-9a-z]{10}\.jss", name):
            raise Conflict("Overflow receipt segment metadata is invalid")
        receipt_path, payload_path = self._stage(key, name)
        receipt_digest = self.download_checked(ref, receipt_path, wanted["sha256"], RECEIPT_LIMIT)
        receipt, payload_ref = self.parse_receipt(ref, receipt_path)
        expected = manifest["payload"]
        if (payload_ref.name != expected["object"] or payload_ref.generation != expected["generation"]
                or payload_ref.size != expected["bytes"] or payload_ref.metadata.get("sha256") != expected["sha256"]
                or receipt.get("name") != name or receipt.get("sha256") != expected["sha256"]):
            raise Conflict("Overflow payload differs from reviewed GCS generation")
        try:
            md5 = base64.b64decode(receipt["gcs"]["md5_hash"], validate=True)
        except (KeyError, ValueError) as error:
            raise Conflict("Overflow receipt MD5 is invalid") from error
        if len(md5) != 16:
            raise Conflict("Overflow receipt MD5 length differs")
        payload_digest = {"size": payload_ref.size, "sha256": expected["sha256"], "md5": md5.hex()}
        if self.download_checked(payload_ref, payload_path, expected["sha256"], PILOT_PAYLOAD_LIMIT) != payload_digest:
            raise Conflict("Overflow payload bytes differ from receipt")
        self.authority.reserve_pair(key, payload_ref.size + ref.size,
                                    manifest["control_allowance_bytes"], {
            "manifest_sha256": self.manifest_sha256,
            "folder_name": manifest["folder_name"], "drive_root": self.root,
            "payload_object": payload_ref.name, "payload_generation": payload_ref.generation,
            "receipt_object": ref.name, "receipt_generation": ref.generation})
        if entry is None:
            entry = {"key": key, "status": "source_verified", "release_id": manifest["release_id"],
                     "manifest_sha256": self.manifest_sha256, "archive_qualified": False,
                     "payload_ref": vars(payload_ref), "receipt_ref": vars(ref),
                     "payload_name": name, "receipt_name": receipt_path.name,
                     "payload_digest": payload_digest, "receipt_digest": receipt_digest,
                     "drive": {"folder": {}, "payload": {}, "receipt": {}}}
            self.state["segments"][key] = entry
            self.save()
        elif (entry.get("payload_ref") != vars(payload_ref) or entry.get("receipt_ref") != vars(ref)
              or entry.get("payload_digest") != payload_digest or entry.get("receipt_digest") != receipt_digest):
            raise Conflict("Overflow source journal identity differs")
        self.drive.verify_account(self.authority.account_id)
        root_info = self.drive.metadata(self.root)
        if (root_info.get("id") != self.root or root_info.get("mimeType") != FOLDER
                or root_info.get("trashed") is not False):
            raise Conflict("Overflow Drive root is not visible as a folder")
        if entry["status"] == "source_verified":
            self._pin_ids(entry)
        elif entry["status"] != "drive_verified":
            ids = [entry["drive"][kind].get("id") for kind in ("folder", "payload", "receipt")]
            if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 3:
                raise Conflict("Overflow journal lacks three pinned Drive IDs")
        if entry["status"] != "drive_verified":
            self._ensure_folder(entry)
            folder_id = entry["drive"]["folder"]["id"]
            entry["status"] = "file_upload_inflight"
            self.save()
            for kind, path in (("payload", payload_path), ("receipt", receipt_path)):
                uploaded = self.drive.ensure_file(path, folder_id, entry[kind + "_digest"],
                                                  entry["drive"][kind], self.save)
                if (uploaded.get("id") != entry["drive"][kind]["id"]
                        or not matches_drive(uploaded, path.name, entry[kind + "_digest"], folder_id)):
                    raise Conflict("Overflow upload differs from pinned ID or bytes")
                self.save()
        self._verify_drive_mapping(entry)
        self.authority.assert_active()
        record = self._index_record(entry)
        digest = hashlib.sha256(_canonical(record)).hexdigest()
        entry["mapping_sha256"] = digest
        entry["status"] = "drive_verified"
        self.save()
        self.authority.mark_pair_verified(key, digest)
        return self._recover_mapping(entry, "verified")


def main(argv=None) -> int:
    """Run exactly one reviewed pair or check its manifest without cloud calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overflow-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--check-manifest", action="store_true")
    parser.add_argument("--authority-db", type=Path)
    parser.add_argument("--review-evidence", type=Path)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--writer-id")
    parser.add_argument("--oauth-file", type=Path,
                        default=Path("/root/.config/catbird-drive/oauth.json"))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--retain-gcs", action="store_true")
    parser.add_argument("--quota-observation", action="store_true")
    parser.add_argument("--recovery-only", action="store_true")
    parser.add_argument("--max-concurrent-runs", type=int, default=1, choices=(1, 2, 3, 4))
    args = parser.parse_args(argv)
    try:
        manifest, manifest_sha256 = load_overflow_manifest(
            args.overflow_manifest, release_id=args.run_id, account_id=args.account_id)
    except Exception as error:
        print(json.dumps({"event": "overflow_manifest_rejected", "error_type": type(error).__name__}),
              file=sys.stderr, flush=True)
        return 1
    if args.check_manifest:
        print(json.dumps({"event": "overflow_manifest_checked", "pair_key": pair_key(manifest),
                          "manifest_sha256": manifest_sha256,
                          "max_charged_bytes": manifest["max_charged_bytes"],
                          "max_duration_seconds": manifest["max_duration_seconds"],
                          "archive_qualified": False, "retained_gcs": True}), flush=True)
        return 0
    if not (args.once and args.retain_gcs and args.quota_observation):
        parser.error("Live overflow requires --once --retain-gcs --quota-observation")
    if any(value is None for value in (args.authority_db, args.review_evidence,
                                       args.deadline_epoch, args.writer_id, args.work_dir)):
        parser.error("Live overflow requires authority DB, evidence, deadline, writer, work directory")
    if (args.work_dir.is_symlink() or
            not args.work_dir.resolve().is_relative_to(Path("/data")) or
            args.work_dir.name != "gcs-drive-retained-" + args.run_id):
        parser.error("Overflow work directory must be the exact run child of /data")
    now, monotonic = time.time(), time.monotonic()
    remaining = args.deadline_epoch - now
    if (not math.isfinite(remaining) or not WATCHDOG_MARGIN_SECONDS < remaining
            <= manifest["max_duration_seconds"]):
        parser.error("Overflow deadline exceeds reviewed duration or watchdog margin")
    try:
        evidence = _parse_manifest(args.review_evidence.read_bytes())
    except Exception:
        print(json.dumps({"event": "overflow_preserved", "error_type": "ReviewEvidenceInvalid"}),
              file=sys.stderr, flush=True)
        return 1
    cutoff = monotonic + remaining - WATCHDOG_MARGIN_SECONDS
    def worker() -> int:
        return _run_worker_guarded(args, manifest, manifest_sha256, evidence)
    return _supervise_worker(worker, cutoff)


def _run_worker_guarded(args, manifest: dict, digest: str, evidence: dict) -> int:
    authority = None
    try:
        raw = args.overflow_manifest.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise Conflict("Overflow manifest bytes changed after preflight")
        prior = next((row for row in TransferAuthority.list_retained_pairs(
            args.authority_db, writer_id=args.writer_id, account_id=args.account_id)
            if row["run_id"] == args.run_id), None)
        if prior is not None and prior["status"] == "closed":
            if not args.recovery_only:
                raise Conflict("Retained pair already closed; no second copy")
            observed = TransferAuthority.inspect_retained_pair(
                args.authority_db, writer_id=args.writer_id, account_id=args.account_id,
                run_id=args.run_id, pair_key=pair_key(manifest), manifest_sha256=digest)
            if observed["status"] != "closed" or not re.fullmatch(
                    r"[0-9a-f]{64}", observed.get("mapping_sha256") or ""):
                raise Conflict("Closed retained mapping inspection differs")
            cloud = OverflowGCS(BUCKET)
            if gce_region() != "us-west1" or cloud.bucket.location.lower() != "us-west1":
                raise Conflict("Closed mapping GCS location differs")
            class ClosedAuthority:
                release_id = args.run_id
                account_id = args.account_id
                manifest_sha256 = digest
            with RetainedOverflow(cloud, None, ClosedAuthority(), args.work_dir,
                                  manifest, digest, raw) as worker:
                result = worker.recover_closed_cache(observed["mapping_sha256"])
            print(json.dumps({"event": "overflow_closed_cache_recovered", **result,
                              "history_proven": False}), flush=True)
            return 0
        authority = TransferAuthority.open_retained_run(
            args.authority_db, run_id=args.run_id, writer_id=args.writer_id,
            account_id=args.account_id, manifest_bytes=raw, evidence=evidence,
            deadline_at=args.deadline_epoch, recovery_only=args.recovery_only,
            max_concurrent_runs=getattr(args, "max_concurrent_runs", 1))
        for required in ("reserve_index_create", "mark_pair_verified", "close_pair",
                         "mark_index_committed", "pair_status"):
            if not callable(getattr(authority, required, None)):
                raise Conflict("Retained authority lacks shared index protocol")
        if not args.recovery_only:
            authority.validate_manifest_limits(manifest["max_charged_bytes"],
                                               manifest["max_duration_seconds"])
            authority.assert_active(300)
        cloud = OverflowGCS(BUCKET)
        if gce_region() != "us-west1" or cloud.bucket.location.lower() != "us-west1":
            raise Conflict("GCE and retained GCS bucket must be single-region us-west1")
        for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
            with contextlib.suppress(FileNotFoundError, PermissionError):
                if b"gce_segment_offloader.py" in cmdline.read_bytes():
                    raise Conflict("Legacy Drive offloader is active")
        drive = None if args.recovery_only else DiagnosticSafeDrive(args.oauth_file, authority)
        with RetainedOverflow(cloud, drive,
                              authority, args.work_dir, manifest, digest, raw) as worker:
            result = worker.run_one()
        print(json.dumps({"event": "overflow_retained", **result,
                          "history_proven": False, "admission_mode": "retained_pair_observation"}), flush=True)
        return 0
    except DriveHttpStop as error:
        output = {"event": "overflow_drive_http_stop", "http_status": error.status,
                  "archive_qualified": False, "history_proven": False}
        if error.reason:
            output["drive_reason"] = error.reason
        print(json.dumps(output), flush=True)
        return 75
    except Exception as error:
        if isinstance(error, urllib.error.HTTPError):
            error.close()
        print(json.dumps({"event": "overflow_preserved", "error_type": type(error).__name__,
                          "archive_qualified": False, "history_proven": False}),
              file=sys.stderr, flush=True)
        return 1
    finally:
        if authority is not None:
            authority.close()


def _supervise_worker(worker, cutoff_monotonic: float) -> int:
    """Independent Linux hard stop; child receives a parent-death signal too."""
    if not sys.platform.startswith("linux") or not math.isfinite(cutoff_monotonic):
        raise Conflict("Overflow watchdog needs a finite Linux cutoff")
    if cutoff_monotonic <= time.monotonic():
        raise Conflict("Overflow watchdog cutoff expired")
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt("Overflow supervisor interrupted")
    previous_term = signal.signal(signal.SIGTERM, interrupt)
    previous_hup = signal.signal(signal.SIGHUP, interrupt)
    child = None
    reaped = False
    try:
        parent = os.getpid()
        child = os.fork()
        if child == 0:
            try:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.signal(signal.SIGHUP, signal.SIG_DFL)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
                os.setsid()
                _arm_parent_death(parent)
                signal.setitimer(signal.ITIMER_REAL,
                                 max(0.001, cutoff_monotonic - time.monotonic()))
                code = worker()
            except BaseException:
                os._exit(1)
            os._exit(code if isinstance(code, int) and 0 <= code <= 255 else 1)
        while True:
            if time.monotonic() >= cutoff_monotonic:
                _stop_worker_group(child, child_reaped=False)
                reaped = True
                print(json.dumps({"event": "overflow_watchdog_stop", "archive_qualified": False}),
                      file=sys.stderr, flush=True)
                return 124
            pid, status = os.waitpid(child, os.WNOHANG)
            if pid == child:
                _stop_worker_group(child, child_reaped=True)
                reaped = True
                return os.waitstatus_to_exitcode(status)
            time.sleep(max(0, min(0.05, cutoff_monotonic - time.monotonic())))
    except BaseException:
        if child is not None and child > 0 and not reaped:
            _stop_worker_group(child, child_reaped=False)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGHUP, previous_hup)


if __name__ == "__main__":
    raise SystemExit(main())
