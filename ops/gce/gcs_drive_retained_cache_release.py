#!/usr/bin/env python3
"""Review generated retained-copy stage files; cache release is disabled.

This module never writes to Drive or GCS. A future release must supply a
separately reviewed read-only verifier and an exact human approval artifact.
The deployed default cannot remove a file, even when an approval-shaped JSON
object is supplied. Historical retained_local=True remains a creation receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import fcntl
from pathlib import Path
import re
import sqlite3
import stat
import time
from typing import Protocol

from drive_transfer_authority import TransferAuthority
from gcs_drive_drain import Conflict
from gcs_drive_retained_overflow import (
    BUCKET, DRIVE_ROOT, FINAL_STATUS, INDEX_SCHEMA, _canonical,
    load_overflow_manifest, pair_key,
)


SCHEMA = "catbird.drive.retained-cache-review.v1"
RELEASE_SCHEMA = "catbird.drive.retained-cache-release.v1"
APPROVAL_SCHEMA = "catbird.drive.retained-cache-approval.v1"
# A later explicit Josh decision and code review must change this constant.
LOCAL_CACHE_RELEASE_ENABLED = False
MAX_JOURNAL = 1 << 20
MAX_MAPPING = 16 << 10
MAX_EVIDENCE_LOG = 8 << 20
MAX_REMOTE_PREFLIGHT_SECONDS = 300
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
RUN = re.compile(r"[0-9a-f]{32}\Z")


class RemoteVerifier(Protocol):
    """Read-only, independently paced implementation supplied at review time.

    Each digest method must stream the *exact pinned generation or Drive ID* to
    EOF, and return size, SHA-256 and MD5 of observed bytes. A metadata-only
    lookup is insufficient. No implementation is provided in this packet.
    """
    def verify_account(self, account_id: str) -> None: ...
    def mapping(self, bucket: str, name: str) -> tuple[int, bytes]: ...
    def gcs_digest(self, object_name: str, generation: int) -> dict: ...
    def drive_digest(self, file_id: str) -> dict: ...


class ApprovalVerifier(Protocol):
    """Independently authenticate the referenced Josh approval record."""
    def verify(self, approval: dict) -> dict: ...


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Conflict("Cache evidence has duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(_value):
    raise Conflict("Cache evidence has nonfinite JSON number")


def _json_file(path: Path, limit: int) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise Conflict("Cache evidence file is missing, symlinked or too large")
    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object,
                           parse_constant=_bad_constant)
    except (ValueError, UnicodeError, TypeError) as error:
        raise Conflict("Cache evidence JSON is malformed") from error
    if not isinstance(value, dict):
        raise Conflict("Cache evidence must be an object")
    return value


def _dir(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise Conflict("Cache path contains a non-directory or symlink")
    return info


def _hash_file(path: Path, parent_device: int) -> dict:
    """Hash one regular single-link file through O_NOFOLLOW and recheck inode."""
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_dev != parent_device or path.is_symlink()):
        raise Conflict("Cache candidate is not a private regular stage file")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if ((opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns,
             opened.st_ctime_ns) !=
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_ctime_ns)):
            raise Conflict("Cache candidate changed before hash")
        sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
        count = 0
        while block := os.read(fd, 1 << 20):
            sha.update(block)
            md5.update(block)
            count += len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    path_after = path.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if count != before.st_size or identity(after) != identity(before) or identity(path_after) != identity(before):
        raise Conflict("Cache candidate changed during hash")
    return {"path": str(path), "device": before.st_dev, "inode": before.st_ino,
            "bytes": count, "sha256": sha.hexdigest(), "md5": md5.hexdigest(),
            "mtime_ns": before.st_mtime_ns, "ctime_ns": before.st_ctime_ns,
            "link_count": before.st_nlink}


def _identity_digest(observed: dict, *, size: int, sha: str, md5: str | None = None) -> None:
    if (not isinstance(observed, dict) or observed.get("bytes") != size
            or observed.get("sha256") != sha
            or md5 is not None and observed.get("md5") != md5):
        raise Conflict("Remote media digest differs from pinned source")


def _recheck_local_closed(record: dict, authority_db: Path, data_root: Path) -> None:
    """Revalidate journal and whole-account stop state after long readbacks."""
    work = data_root / ("gcs-drive-retained-" + record["run_id"])
    journal = _json_file(work / "journal.json", MAX_JOURNAL)
    if hashlib.sha256(_canonical(journal)).hexdigest() != record["journal_sha256"]:
        raise Conflict("Cache journal changed since inventory")
    pair = TransferAuthority.inspect_retained_pair(
        authority_db, writer_id=record["writer_id"], account_id=record["account_id"],
        run_id=record["run_id"], pair_key=record["pair_key"],
        manifest_sha256=record["manifest_sha256"])
    account = TransferAuthority.inspect_retained_account(
        authority_db, writer_id=record["writer_id"], account_id=record["account_id"])
    if (pair.get("status") != "closed"
            or pair.get("mapping_sha256") != record["mapping_sha256"]
            or account.get("unresolved_attempts") != 0
            or account.get("nonterminal_run_id") is not None):
        raise Conflict("Cache account or pair became nonterminal")


def inspect_cache(*, manifest_path: Path, run_id: str, account_id: str,
                  writer_id: str, authority_db: Path, data_root: Path = Path("/data"),
                  verifier: RemoteVerifier | None = None) -> dict:
    """Return one exact local candidate and conservative exclusion reasons.

    No local files or cloud objects are changed. Remote proof is unavailable
    unless a separately reviewed read-only verifier is injected.
    """
    if not RUN.fullmatch(run_id):
        raise Conflict("Cache run ID is malformed")
    manifest, manifest_sha = load_overflow_manifest(
        manifest_path, release_id=run_id, account_id=account_id)
    key = pair_key(manifest)
    root = data_root / ("gcs-drive-retained-" + run_id)
    stage = root / "stage" / key
    root_dev = _dir(data_root).st_dev
    if _dir(root).st_dev != root_dev or _dir(root / "stage").st_dev != root_dev:
        raise Conflict("Cache work directory crosses filesystem")
    stage_dev = _dir(stage).st_dev
    if stage_dev != root_dev:
        raise Conflict("Cache stage crosses filesystem")
    # The retained worker stages the segment's original `seg_*.jss` name,
    # while the GCS object basename is its content hash.
    name = Path(manifest["payload"]["object"]).parent.name
    expected_names = {name, name + ".receipt.json"}
    if {entry.name for entry in stage.iterdir()} != expected_names:
        raise Conflict("Cache stage contains a partial or unknown file")
    files = {"payload": _hash_file(stage / name, stage_dev),
             "receipt": _hash_file(stage / (name + ".receipt.json"), stage_dev)}
    for kind in ("payload", "receipt"):
        wanted = manifest[kind]
        if files[kind]["bytes"] != wanted["bytes"] or files[kind]["sha256"] != wanted["sha256"]:
            raise Conflict("Cache stage bytes differ from pinned GCS generation")

    journal = _json_file(root / "journal.json", MAX_JOURNAL)
    if (journal.get("bucket") != BUCKET or journal.get("root") != DRIVE_ROOT
            or not isinstance(journal.get("segments"), dict)
            or set(journal["segments"]) != {key}):
        raise Conflict("Cache journal namespace or pair differs")
    entry = journal["segments"][key]
    if (not isinstance(entry, dict) or entry.get("key") != key
            or entry.get("release_id") != run_id
            or entry.get("manifest_sha256") != manifest_sha
            or entry.get("status") != FINAL_STATUS
            or entry.get("archive_qualified") is not False):
        raise Conflict("Cache journal is not terminal retained pair")
    for kind in ("payload", "receipt"):
        ref, wanted = entry.get(kind + "_ref"), manifest[kind]
        digest = entry.get(kind + "_digest")
        if (not isinstance(ref, dict) or ref.get("name") != wanted["object"]
                or ref.get("generation") != wanted["generation"]
                or ref.get("size") != wanted["bytes"]
                or not isinstance(digest, dict)
                or digest.get("size") != wanted["bytes"]
                or digest.get("sha256") != wanted["sha256"]
                or digest.get("md5") != files[kind]["md5"]):
            raise Conflict("Cache journal source or digest differs")
    ids = {kind: entry.get("drive", {}).get(kind, {}).get("id")
           for kind in ("folder", "payload", "receipt")}
    if any(not isinstance(value, str) or not value for value in ids.values()) or len(set(ids.values())) != 3:
        raise Conflict("Cache journal lacks unique pinned Drive IDs")
    pair = TransferAuthority.inspect_retained_pair(
        authority_db, writer_id=writer_id, account_id=account_id,
        run_id=run_id, pair_key=key, manifest_sha256=manifest_sha)
    account = TransferAuthority.inspect_retained_account(
        authority_db, writer_id=writer_id, account_id=account_id)
    mapping_sha = pair.get("mapping_sha256")
    if (pair.get("status") != "closed" or not isinstance(mapping_sha, str)
            or not HEX64.fullmatch(mapping_sha)
            or account.get("unresolved_attempts") != 0
            or account.get("nonterminal_run_id") is not None):
        raise Conflict("Cache ledger has unresolved or nonterminal authority")
    if entry.get("mapping_sha256") != mapping_sha:
        raise Conflict("Cache journal and ledger mapping digest differ")

    record = {"schema": SCHEMA, "observed_at": time.time(),
              "run_id": run_id, "pair_key": key, "manifest_sha256": manifest_sha,
              "journal_sha256": hashlib.sha256(_canonical(journal)).hexdigest(),
              "account_id": account_id, "writer_id": writer_id,
              "mapping_bucket": manifest["index_bucket"],
              "mapping_object": manifest["index_object"],
              "mapping_sha256": mapping_sha,
              "drive_root": DRIVE_ROOT, "drive_ids": ids,
              "source": {kind: dict(manifest[kind]) for kind in ("payload", "receipt")},
              "stage": files, "stage_directory": str(stage),
              "creation_retained_local": True,
              "current_cache_state": "present",
              "archive_qualified": False,
              "gcs_retained": False, "drive_media_verified": False,
              "mapping_generation": None, "eligible_for_approval": False,
              "exclude_reason": "remote_readback_unavailable"}
    if verifier is None:
        return record
    verifier.verify_account(account_id)
    mapping_generation, raw = verifier.mapping(manifest["index_bucket"],
                                               manifest["index_object"])
    if (type(mapping_generation) is not int or mapping_generation <= 0
            or not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_MAPPING
            or hashlib.sha256(raw).hexdigest() != mapping_sha
            or entry.get("index_generation") != mapping_generation):
        raise Conflict("Cache shared mapping generation or bytes differ")
    try:
        mapping = json.loads(raw, object_pairs_hook=_strict_object, parse_constant=_bad_constant)
    except (ValueError, UnicodeError, TypeError) as error:
        raise Conflict("Cache shared mapping JSON is malformed") from error
    if (not isinstance(mapping, dict) or raw != _canonical(mapping)
            or mapping.get("schema") != INDEX_SCHEMA
            or mapping.get("pair_key") != key
            or mapping.get("manifest_sha256") != manifest_sha
            or mapping.get("index_bucket") != manifest["index_bucket"]
            or mapping.get("index_object") != manifest["index_object"]
            or mapping.get("drive_account_id") != account_id
            or mapping.get("drive_root") != DRIVE_ROOT
            or mapping.get("folder_id") != ids["folder"]
            or mapping.get("payload_id") != ids["payload"]
            or mapping.get("receipt_id") != ids["receipt"]
            or mapping.get("retained_gcs") is not True
            or mapping.get("retained_local") is not True
            or mapping.get("archive_qualified") is not False
            or mapping.get("status") != FINAL_STATUS):
        raise Conflict("Cache shared mapping identity differs")
    for kind in ("payload", "receipt"):
        ref = mapping.get(kind + "_ref")
        if ref != {field: manifest[kind][field] for field in ("object", "generation", "bytes", "sha256")}:
            raise Conflict("Cache shared mapping source generation differs")
        if mapping.get(kind + "_digest") != entry[kind + "_digest"]:
            raise Conflict("Cache shared mapping media digest differs")
        wanted = manifest[kind]
        observed_gcs = verifier.gcs_digest(wanted["object"], wanted["generation"])
        _identity_digest(observed_gcs, size=wanted["bytes"], sha=wanted["sha256"],
                         md5=files[kind]["md5"])
        observed_drive = verifier.drive_digest(ids[kind])
        _identity_digest(observed_drive, size=wanted["bytes"], sha=wanted["sha256"],
                         md5=files[kind]["md5"])
        if (observed_drive.get("id") != ids[kind]
                or observed_drive.get("parent_id") != ids["folder"]
                or observed_drive.get("trashed") is not False
                or observed_drive.get("account_id") != account_id
                or observed_drive.get("media_readback") is not True):
            raise Conflict("Cache Drive identity or full media readback differs")
    record.update(mapping_generation=mapping_generation, gcs_retained=True,
                  drive_media_verified=True, eligible_for_approval=True,
                  exclude_reason=None)
    _recheck_local_closed(record, authority_db, data_root)
    return record


def append_evidence(path: Path, record: dict) -> str:
    """Fsync one append-only review/release line; never replace old evidence."""
    if path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
        raise Conflict("Cache evidence destination is unavailable")
    raw = _canonical(record)
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_dev != path.parent.stat().st_dev
                or info.st_uid != os.geteuid() or info.st_mode & 0o077
                or info.st_size > MAX_EVIDENCE_LOG):
            raise Conflict("Cache evidence log is not a private bounded file")
        if os.write(fd, raw + b"\n") != len(raw) + 1:
            raise Conflict("Cache evidence append was short")
        os.fsync(fd)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def _release_log(data_root: Path, run_id: str) -> Path:
    if not RUN.fullmatch(run_id):
        raise Conflict("Cache release run ID is malformed")
    return _evidence_root(data_root) / ("release-" + run_id + ".jsonl")


def _review_log(data_root: Path, run_id: str) -> Path:
    if not RUN.fullmatch(run_id):
        raise Conflict("Cache review run ID is malformed")
    return _evidence_root(data_root) / ("review-" + run_id + ".jsonl")


def _evidence_root(data_root: Path) -> Path:
    """Evidence lives outside the runner's gcs-drive-retained-* scan."""
    root = data_root / "catbird-retained-cache-evidence"
    if root.is_symlink():
        raise Conflict("Cache evidence directory is symlinked")
    existed = root.exists()
    root.mkdir(mode=0o700, exist_ok=True)
    info = _dir(root)
    if (info.st_dev != _dir(data_root).st_dev or info.st_uid != os.geteuid()
            or info.st_mode & 0o077):
        raise Conflict("Cache evidence directory is not private to the owner")
    if not existed:
        data_fd = os.open(data_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(data_fd)
        finally:
            os.close(data_fd)
    return root


def _no_prior_release(path: Path) -> None:
    """A prior intent, even without an outcome, makes replay a manual review."""
    if not path.exists():
        return
    if path.is_symlink() or path.stat().st_size > MAX_EVIDENCE_LOG:
        raise Conflict("Cache release log is unreadable or oversized")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.geteuid() or info.st_mode & 0o077):
            raise Conflict("Cache release log is not a private regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for raw in stream:
                if not raw.endswith(b"\n") or len(raw) > MAX_JOURNAL:
                    raise Conflict("Cache release log has incomplete evidence")
                try:
                    item = json.loads(raw, object_pairs_hook=_strict_object,
                                      parse_constant=_bad_constant)
                except (ValueError, UnicodeError, TypeError) as error:
                    raise Conflict("Cache release log has malformed evidence") from error
                if (isinstance(item, dict)
                        and str(item.get("event", "")).startswith("cache_release_")):
                    raise Conflict("Cache release has a prior intent or outcome; review manually")
    finally:
        os.close(fd)


def _hash_stage_again(stage_fd: int, name: str, wanted: dict) -> None:
    """Re-open a generated file relative to a pinned parent descriptor."""
    if name != Path(wanted["path"]).name or name in (".", "..") or "/" in name:
        raise Conflict("Cache release name is not an exact stage basename")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=stage_fd)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_dev != wanted["device"]
                or before.st_ino != wanted["inode"]
                or before.st_size != wanted["bytes"]
                or before.st_mtime_ns != wanted["mtime_ns"]
                or before.st_ctime_ns != wanted["ctime_ns"]):
            raise Conflict("Cache release file identity changed")
        sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
        count = 0
        while block := os.read(fd, 1 << 20):
            sha.update(block)
            md5.update(block)
            count += len(block)
        after = os.fstat(fd)
        path_now = os.stat(name, dir_fd=stage_fd, follow_symlinks=False)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if (identity(after) != identity(before) or identity(path_now) != identity(before)
                or count != wanted["bytes"] or sha.hexdigest() != wanted["sha256"]
                or md5.hexdigest() != wanted["md5"]):
            raise Conflict("Cache release bytes changed immediately before removal")
    finally:
        os.close(fd)


def _reserve_account_writes(authority_db: Path) -> sqlite3.Connection:
    """Hold SQLite's writer reservation; every retained admission uses this DB.

    The direct worker does not honor the runner's .runner.lock. Its admission,
    request and closure writes all use BEGIN IMMEDIATE on this same ledger.
    This reservation therefore excludes it as well as bounded/supervised runs.
    It is deliberately short-fail on contention and never writes a ledger row.
    """
    if authority_db.is_symlink() or not authority_db.is_file():
        raise Conflict("Cache account ledger is unavailable or symlinked")
    canonical = authority_db.resolve(strict=True)
    info = canonical.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise Conflict("Cache account ledger is not a private regular file")
    uri = canonical.as_uri().replace("file://", "file:") + "?mode=rw"
    db = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=0)
    try:
        db.execute("PRAGMA busy_timeout=0")
        db.execute("BEGIN IMMEDIATE")
        return db
    except sqlite3.Error as error:
        db.close()
        raise Conflict("Cache account admission writer reservation is unavailable") from error


def release_after_approval(*, inventory: dict, approval: dict, manifest_path: Path,
                           authority_db: Path, data_root: Path, verifier: RemoteVerifier,
                           approval_verifier: ApprovalVerifier | None = None,
                           remove_file=os.unlink) -> dict:
    """Future-only transaction; shipped default refuses before any file action.

    The caller must authenticate Josh's approval and enforce exclusive stage
    ownership outside this module. Merely writing `approved_by` in JSON is
    not authorization. A future reviewed release may enable this path.
    """
    if not LOCAL_CACHE_RELEASE_ENABLED:
        raise Conflict("Local cache release is disabled pending explicit Josh decision")
    if approval_verifier is None:
        raise Conflict("Cache release requires authenticated Josh approval provenance")
    run_id = inventory.get("run_id")
    if not isinstance(run_id, str) or not RUN.fullmatch(run_id):
        raise Conflict("Cache release run ID is malformed")
    work = data_root / ("gcs-drive-retained-" + run_id)
    if _dir(work).st_dev != _dir(data_root).st_dev:
        raise Conflict("Cache release work crosses filesystem")
    lock_path = work / "overflow.lock"
    if lock_path.is_symlink():
        raise Conflict("Cache release lock is symlinked")
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    try:
        lock_info = os.fstat(lock_fd)
        if (not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1
                or lock_info.st_dev != _dir(work).st_dev):
            raise Conflict("Cache release lock identity differs")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise Conflict("Retained worker or recovery owns the stage") from error
        proof = approval_verifier.verify(approval)
        if (not isinstance(proof, dict) or proof.get("authenticated") is not True
                or proof.get("approved_by") != "Josh"
                or proof.get("approval_reference") != approval.get("approval_reference")
                or proof.get("inventory_sha256") != approval.get("inventory_sha256")):
            raise Conflict("Cache release approval provenance is unverified")
        # Potentially slow full media readbacks happen *before* taking the
        # account-wide SQLite writer reservation. They cannot stall unrelated
        # Drive admissions. Any run admitted during these reads is caught by
        # the final account/journal check under the reservation below.
        preflight_started = time.monotonic()
        current = inspect_cache(manifest_path=manifest_path, run_id=inventory["run_id"],
                                account_id=inventory["account_id"],
                                writer_id=inventory["writer_id"],
                                authority_db=authority_db, data_root=data_root,
                                verifier=verifier)
        if time.monotonic() - preflight_started > MAX_REMOTE_PREFLIGHT_SECONDS:
            raise Conflict("Cache remote preflight exceeded five minutes")
        account_guard = _reserve_account_writes(authority_db)
        try:
            return _release_locked(inventory=inventory, approval=approval,
                                   current=current, authority_db=authority_db,
                                   data_root=data_root,
                                   approval_proof=proof,
                                   lock_fd=lock_fd, account_guard=account_guard,
                                   remove_file=remove_file)
        finally:
            account_guard.execute("ROLLBACK")
            account_guard.close()
    finally:
        os.close(lock_fd)


def _release_locked(*, inventory: dict, approval: dict, current: dict,
                    authority_db: Path, data_root: Path,
                    approval_proof: dict, lock_fd: int,
                    account_guard: sqlite3.Connection,
                    remove_file=os.unlink) -> dict:
    # Private entry also retains the hard gate. The caller holds overflow.lock.
    if not LOCAL_CACHE_RELEASE_ENABLED:
        raise Conflict("Local cache release is disabled pending explicit Josh decision")
    if not account_guard.in_transaction:
        raise Conflict("Cache release lacks an account write reservation")
    attached = account_guard.execute("PRAGMA database_list").fetchall()
    if (len(attached) != 1 or attached[0][2] != str(authority_db.resolve(strict=True))):
        raise Conflict("Cache release reserved the wrong account ledger")
    work = data_root / ("gcs-drive-retained-" + inventory["run_id"])
    expected_lock = (work / "overflow.lock").lstat()
    actual_lock = os.fstat(lock_fd)
    if ((actual_lock.st_dev, actual_lock.st_ino) !=
            (expected_lock.st_dev, expected_lock.st_ino)
            or not stat.S_ISREG(actual_lock.st_mode)):
        raise Conflict("Cache release is missing the retained worker lock")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        raise Conflict("Retained worker or recovery owns the stage") from error
    if (approval_proof.get("authenticated") is not True
            or approval_proof.get("approved_by") != "Josh"
            or approval_proof.get("approval_reference") != approval.get("approval_reference")
            or approval_proof.get("inventory_sha256") != approval.get("inventory_sha256")):
        raise Conflict("Cache release approval provenance is unverified")
    if (not inventory.get("eligible_for_approval")
            or inventory.get("schema") != SCHEMA
            or inventory.get("creation_retained_local") is not True
            or inventory.get("current_cache_state") != "present"
            or inventory.get("gcs_retained") is not True
            or inventory.get("drive_media_verified") is not True
            or approval.get("schema") != APPROVAL_SCHEMA
            or approval.get("approved_by") != "Josh"
            or not isinstance(approval.get("approval_reference"), str)
            or not approval["approval_reference"]
            or approval.get("inventory_sha256") != hashlib.sha256(_canonical(inventory)).hexdigest()
            or approval.get("run_id") != inventory.get("run_id")
            or approval.get("pair_key") != inventory.get("pair_key")
            or approval.get("paths") != [inventory["stage"][kind]["path"]
                                          for kind in ("payload", "receipt")]):
        raise Conflict("Cache release approval does not bind exact inventory")
    inventory_sha = hashlib.sha256(_canonical(inventory)).hexdigest()
    approval_sha = hashlib.sha256(_canonical(approval)).hexdigest()
    release_id = hashlib.sha256((inventory_sha + approval_sha).encode("ascii")).hexdigest()
    log_path = _release_log(data_root, inventory["run_id"])
    _no_prior_release(log_path)
    if (not current["eligible_for_approval"]
            or current["pair_key"] != inventory["pair_key"]
            or current["manifest_sha256"] != inventory["manifest_sha256"]
            or current["source"] != inventory["source"]
            or current["stage_directory"] != inventory["stage_directory"]
            or any(current["stage"][kind] != inventory["stage"][kind]
                   for kind in ("payload", "receipt"))
            or current["mapping_sha256"] != inventory["mapping_sha256"]
            or current["mapping_generation"] != inventory["mapping_generation"]
            or current["drive_ids"] != inventory["drive_ids"]):
        raise Conflict("Cache release evidence changed since approval")
    _recheck_local_closed(current, authority_db, data_root)
    stage = data_root / ("gcs-drive-retained-" + inventory["run_id"]) / "stage" / inventory["pair_key"]
    if str(stage) != inventory["stage_directory"] or stage.is_symlink():
        raise Conflict("Cache release stage path changed")
    fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    released = []
    try:
        parent = os.fstat(fd)
        path_parent = stage.lstat()
        if (not stat.S_ISDIR(parent.st_mode)
                or (parent.st_dev, parent.st_ino) != (path_parent.st_dev, path_parent.st_ino)):
            raise Conflict("Cache release stage directory identity changed")
        append_evidence(log_path, {"schema": RELEASE_SCHEMA, "event": "cache_release_start",
                                   "release_id": release_id, "inventory_sha256": inventory_sha,
                                   "approval_sha256": approval_sha,
                                   "approval_reference": approval["approval_reference"],
                                   "run_id": inventory["run_id"], "pair_key": inventory["pair_key"],
                                   "at": time.time()})
        expected = {Path(inventory["stage"][kind]["path"]).name
                    for kind in ("payload", "receipt")}
        for kind in ("payload", "receipt"):
            wanted = inventory["stage"][kind]
            name = Path(wanted["path"]).name
            if name not in expected or {entry.name for entry in stage.iterdir()} != expected:
                raise Conflict("Cache release stage listing changed")
            # One durable intent per file, before re-opening, hashing or unlinking.
            append_evidence(log_path, {"schema": RELEASE_SCHEMA, "event": "cache_release_file_intent",
                                       "release_id": release_id, "kind": kind,
                                       "path": wanted["path"], "device": wanted["device"],
                                       "inode": wanted["inode"], "bytes": wanted["bytes"],
                                       "sha256": wanted["sha256"], "md5": wanted["md5"],
                                       "at": time.time()})
            _recheck_local_closed(current, authority_db, data_root)
            _hash_stage_again(fd, name, wanted)
            if (stage.lstat().st_dev, stage.lstat().st_ino) != (parent.st_dev, parent.st_ino):
                raise Conflict("Cache release stage directory changed before removal")
            try:
                remove_file(name, dir_fd=fd)
                os.fsync(fd)
                try:
                    os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise Conflict("Cache release file remains after removal")
            except BaseException as error:
                # Unlink may have happened before an exception. Never auto-retry.
                append_evidence(log_path, {"schema": RELEASE_SCHEMA,
                                           "event": "cache_release_file_uncertain",
                                           "release_id": release_id, "kind": kind,
                                           "path": wanted["path"],
                                           "error_type": type(error).__name__, "at": time.time()})
                raise
            append_evidence(log_path, {"schema": RELEASE_SCHEMA, "event": "cache_release_file_removed",
                                       "release_id": release_id, "kind": kind,
                                       "path": wanted["path"], "bytes": wanted["bytes"],
                                       "at": time.time()})
            released.append(kind)
            expected.remove(name)
        append_evidence(log_path, {"schema": RELEASE_SCHEMA, "event": "cache_release_complete",
                                   "release_id": release_id, "released": released,
                                   "creation_retained_local": True,
                                   "current_cache_state": "released", "at": time.time()})
    finally:
        os.close(fd)
    return {"schema": RELEASE_SCHEMA, "release_id": release_id,
            "released": released, "evidence_log": str(log_path)}


def main(argv: list[str] | None = None) -> int:
    """Local-only inventory CLI; there is no cloud verifier or release switch."""
    parser = argparse.ArgumentParser(description="Inventory generated retained stage cache (local only)")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--writer-id", required=True)
    parser.add_argument("--authority-db", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    args = parser.parse_args(argv)
    record = inspect_cache(manifest_path=args.manifest, run_id=args.run_id,
                           account_id=args.account_id, writer_id=args.writer_id,
                           authority_db=args.authority_db, data_root=args.data_root)
    path = _review_log(args.data_root, args.run_id)
    evidence_sha = append_evidence(path, record)
    print(json.dumps({"schema": SCHEMA, "run_id": args.run_id,
                      "pair_key": record["pair_key"], "eligible_for_approval": False,
                      "exclude_reason": record["exclude_reason"],
                      "local_stage_bytes": sum(record["stage"][kind]["bytes"]
                                               for kind in ("payload", "receipt")),
                      "evidence_log": str(path), "evidence_sha256": evidence_sha},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
