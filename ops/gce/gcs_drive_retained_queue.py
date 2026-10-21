#!/usr/bin/env python3
"""Build a bounded, generation-pinned snapshot for the retained Drive copier.

This selector performs GCS reads only. It does not open Drive, admit a run, or
start a writer. Each emitted manifest is rechecked by the retained authority
and copier before any upload. Incomplete snapshots and downloads are retained
for inspection; no source object or local file is deleted here.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from dataclasses import dataclass

from gcs_drive_drain import Conflict, GCS, RECEIPT_LIMIT, PILOT_PAYLOAD_LIMIT, PILOT_CHARGE_LIMIT
from gcs_drive_retained_overflow import (
    BUCKET, CONTROL_ALLOWANCE, DRIVE_ROOT, INDEX_PREFIX, SCHEMA, SOURCE_EPOCH, SOURCE_ID,
    pair_key, validate_overflow_manifest,
)


PREFIX = f"sources/{SOURCE_ID}/{SOURCE_EPOCH}/segments/"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MAX_SCAN = 5000
MAX_PAIRS = 8
MAX_RESERVED_BYTES = MAX_PAIRS * PILOT_CHARGE_LIMIT
MAX_SNAPSHOT_SECONDS = 300


@dataclass(frozen=True)
class Candidate:
    pair_key: str
    release_id: str
    manifest_path: Path
    manifest_sha256: str
    reserved_bytes: int


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise Conflict(f"Invalid {label}")
    return value


def _generation(value: object) -> int:
    if type(value) is int:
        return _positive_int(value, "GCS generation")
    if isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        return int(value)
    raise Conflict("Invalid GCS generation")


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise Conflict("Snapshot receipt JSON contains duplicate keys")
        result[key] = value
    return result


def _bad_json_constant(_value: str):
    raise Conflict("Snapshot receipt JSON contains nonfinite number")


def _receipt_bytes(cloud, ref, work: Path, key: str) -> bytes:
    if not 0 < ref.size <= RECEIPT_LIMIT:
        raise Conflict("Receipt exceeds snapshot size limit")
    expected_sha = ref.metadata.get("sha256")
    if not isinstance(expected_sha, str) or not SHA256.fullmatch(expected_sha):
        raise Conflict("Receipt SHA-256 metadata is missing")
    stage = work / "receipt-snapshot"
    if stage.is_symlink():
        raise Conflict("Snapshot staging path is symlinked")
    stage.mkdir(parents=True, exist_ok=True)
    path = stage / (key + ".g" + str(ref.generation) + ".json")
    if path.is_symlink():
        raise Conflict("Snapshot receipt path is symlinked")
    if not path.exists():
        # A failed download remains in place and blocks replay until examined.
        cloud.download(ref, path)
    raw = path.read_bytes()
    if len(raw) != ref.size or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise Conflict("Snapshot receipt bytes differ from pinned GCS metadata")
    return raw


def _manifest_for_receipt(cloud, ref, receipt_raw: bytes, account_id: str,
                          max_run_seconds: int) -> tuple[dict, bytes]:
    try:
        receipt = json.loads(receipt_raw, object_pairs_hook=_unique_json_pairs,
                             parse_constant=_bad_json_constant)
    except (ValueError, UnicodeError) as error:
        raise Conflict("Snapshot receipt JSON is malformed") from error
    if not isinstance(receipt, dict) or receipt.get("schema") != "catbird.jetstream.archive.v1":
        raise Conflict("Snapshot receipt schema differs")
    name, index, sha = receipt.get("name"), receipt.get("index"), receipt.get("sha256")
    if (not isinstance(name, str) or not re.fullmatch(r"seg_[0-9a-z]{10}\.jss", name)
            or type(index) is not int or int(name[4:-4], 36) != index
            or not isinstance(sha, str) or not SHA256.fullmatch(sha)):
        raise Conflict("Snapshot receipt segment identity differs")
    if ref.metadata.get("segment_name") != name:
        raise Conflict("Snapshot receipt segment metadata differs")
    if (receipt.get("source_id"), receipt.get("source_epoch")) != (SOURCE_ID, SOURCE_EPOCH):
        raise Conflict("Snapshot receipt source namespace differs")
    size = _positive_int(receipt.get("bytes"), "payload size")
    if size > PILOT_PAYLOAD_LIMIT:
        raise Conflict("Snapshot payload exceeds one-pair size limit")
    pointer = receipt.get("gcs")
    if not isinstance(pointer, dict):
        raise Conflict("Snapshot receipt lacks GCS pointer")
    try:
        md5 = base64.b64decode(pointer["md5_hash"], validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise Conflict("Snapshot receipt MD5 is malformed") from error
    if len(md5) != 16:
        raise Conflict("Snapshot receipt MD5 is malformed")
    payload_name = PREFIX + f"{index // 256:08x}/{name}/{sha}.jss"
    generation = _generation(pointer.get("generation"))
    if (pointer.get("bucket") != BUCKET or pointer.get("object") != payload_name
            or pointer.get("sha256") != sha or pointer.get("bytes") != size
            or ref.name != payload_name + f".g{generation}.receipt.json"
            or ref.metadata.get("payload_object") != payload_name
            or str(ref.metadata.get("payload_generation")) != str(generation)):
        raise Conflict("Snapshot receipt GCS pointer differs")
    payload = cloud.lookup(payload_name, generation)
    if (payload is None or payload.generation != generation or payload.size != size
            or payload.metadata.get("sha256") != sha):
        raise Conflict("Snapshot payload generation is missing or changed")
    source = f"{BUCKET}:{ref.name}:{ref.generation}"
    release_id = hashlib.sha256(("catbird-retained-run-v1:" + account_id + ":" + source).encode()).hexdigest()[:32]
    manifest = {
        "schema": SCHEMA, "release_id": release_id, "account_id": account_id,
        "bucket": BUCKET, "source_id": SOURCE_ID, "source_epoch": SOURCE_EPOCH,
        "drive_root": DRIVE_ROOT,
        "payload": {"object": payload_name, "generation": generation,
                    "bytes": size, "sha256": sha},
        "receipt": {"object": ref.name, "generation": ref.generation,
                    "bytes": ref.size, "sha256": hashlib.sha256(receipt_raw).hexdigest()},
        "control_allowance_bytes": CONTROL_ALLOWANCE,
        "max_charged_bytes": size + ref.size + CONTROL_ALLOWANCE,
        "max_duration_seconds": max_run_seconds,
        "retain_gcs": True, "retain_local": True, "archive_qualified": False,
    }
    manifest["folder_name"] = "catbird-overflow-" + pair_key(manifest)[:32]
    manifest["index_bucket"] = BUCKET
    manifest["index_object"] = INDEX_PREFIX + pair_key(manifest) + ".json"
    validate_overflow_manifest(manifest, release_id=release_id, account_id=account_id)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest, raw


def build_snapshot(cloud, *, account_id: str, work: Path, ledger_pairs: list[dict],
                   max_scan: int, max_pairs: int, max_reserved_bytes: int,
                   max_run_seconds: int = 1200) -> list[Candidate]:
    """Persist at most `max_pairs` reviewed candidates from one bounded GCS scan.

    `ledger_pairs` must come from the validated read-only retained authority API.
    Any unfinished pair stops selection so no later pair can leapfrog recovery.
    """
    for label, value in (("max scan", max_scan), ("max pairs", max_pairs),
                         ("max reserved bytes", max_reserved_bytes),
                         ("max run seconds", max_run_seconds)):
        _positive_int(value, label)
    if (max_pairs > max_scan or max_scan > MAX_SCAN or max_pairs > MAX_PAIRS
            or max_reserved_bytes > MAX_RESERVED_BYTES
            or not 0 < max_run_seconds <= 1200):
        raise Conflict("Snapshot controls exceed reviewed bounds")
    if not isinstance(account_id, str) or not account_id:
        raise Conflict("Snapshot account is missing")
    if cloud.bucket_name != BUCKET:
        raise Conflict("Snapshot GCS bucket differs from reviewed namespace")
    if work.is_symlink():
        raise Conflict("Snapshot work path is symlinked")
    work.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(work / "snapshot.lock", os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(row.get("status") not in ("closed", "diagnostic_verified_unqualified")
               for row in ledger_pairs):
            raise Conflict("An earlier retained pair needs recovery before selection")
        for row in ledger_pairs:
            if (row.get("bucket") != BUCKET or not isinstance(row.get("payload_object"), str)
                    or type(row.get("payload_generation")) is not int
                    or row["payload_generation"] <= 0):
                raise Conflict("Retained ledger source identity is missing")
        completed = {row["pair_key"] for row in ledger_pairs
                     if row["status"] == "closed"}
        completed_sources = {(row["payload_object"], row["payload_generation"])
                             for row in ledger_pairs}
        seen: set[str] = set()
        seen_sources: set[tuple[str, int]] = set()
        selected: list[Candidate] = []
        charged = 0
        last_name = ""
        for scanned, ref in enumerate(cloud.receipts(PREFIX), 1):
            if scanned > max_scan:
                raise Conflict("Snapshot GCS scan bound exceeded")
            if (not isinstance(ref.name, str) or ref.name <= last_name
                    or type(ref.generation) is not int or ref.generation <= 0
                    or ref.metadata.get("kind") != "receipt"):
                raise Conflict("Snapshot receipt listing differs or is unordered")
            last_name = ref.name
            suffix = re.fullmatch(r"(.+)\.g([1-9][0-9]*)\.receipt\.json", ref.name)
            if suffix is None:
                raise Conflict("Snapshot receipt name cannot pin a payload generation")
            source = (suffix.group(1), int(suffix.group(2)))
            key = hashlib.sha256(
                f"{BUCKET}:{suffix.group(1)}:{int(suffix.group(2))}".encode()).hexdigest()
            if key in seen:
                raise Conflict("Snapshot receipt listing repeats a generation")
            seen.add(key)
            if key in completed or source in completed_sources:
                continue
            receipt_raw = _receipt_bytes(cloud, ref, work, key)
            manifest, raw = _manifest_for_receipt(cloud, ref, receipt_raw,
                                                   account_id, max_run_seconds)
            if pair_key(manifest) != key:
                raise Conflict("Snapshot pair key differs")
            source = (manifest["payload"]["object"], manifest["payload"]["generation"])
            if source in seen_sources:
                raise Conflict("Snapshot repeats a payload generation in another receipt")
            seen_sources.add(source)
            reserve = manifest["max_charged_bytes"]
            if charged + reserve > max_reserved_bytes:
                break
            path = work / ("manifest-" + key + ".json")
            if path.is_symlink():
                raise Conflict("Snapshot manifest path is symlinked")
            if path.exists():
                if path.read_bytes() != raw:
                    raise Conflict("Existing snapshot manifest changed")
            else:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                try:
                    if os.write(fd, raw) != len(raw):
                        raise Conflict("Snapshot manifest write was short")
                    os.fsync(fd)
                finally:
                    os.close(fd)
            selected.append(Candidate(key, manifest["release_id"], path,
                                      hashlib.sha256(raw).hexdigest(), reserve))
            charged += reserve
            if len(selected) >= max_pairs:
                break
        directory = os.open(work, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return selected
    finally:
        os.close(lock_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--writer-id", required=True)
    parser.add_argument("--authority-db", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-scan", type=int, required=True)
    parser.add_argument("--max-pairs", type=int, required=True)
    parser.add_argument("--max-reserved-bytes", type=int, required=True)
    parser.add_argument("--max-run-seconds", type=int, default=1200)
    parser.add_argument("--max-snapshot-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if not 0 < args.max_snapshot_seconds <= MAX_SNAPSHOT_SECONDS:
        raise Conflict("Snapshot watchdog duration exceeds five minutes")
    if not sys.platform.startswith("linux"):
        raise Conflict("Live GCS snapshot requires the reviewed Linux watchdog")
    from drive_transfer_authority import TransferAuthority
    from gcs_drive_diagnostic import supervise_worker
    def worker() -> int:
        rows = TransferAuthority.list_retained_pairs(args.authority_db,
                                                       account_id=args.account_id,
                                                       writer_id=args.writer_id)
        selected = build_snapshot(GCS(BUCKET), account_id=args.account_id,
                                  work=args.work_dir, ledger_pairs=rows,
                                  max_scan=args.max_scan, max_pairs=args.max_pairs,
                                  max_reserved_bytes=args.max_reserved_bytes,
                                  max_run_seconds=args.max_run_seconds)
        print(json.dumps({"schema": "catbird.drive.retained-snapshot.v1",
                          "candidates": [{"pair_key": c.pair_key, "release_id": c.release_id,
                                          "manifest_path": str(c.manifest_path),
                                          "manifest_sha256": c.manifest_sha256,
                                          "reserved_bytes": c.reserved_bytes} for c in selected]},
                         sort_keys=True), flush=True)
        return 0
    return supervise_worker(worker, time.monotonic() + args.max_snapshot_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
