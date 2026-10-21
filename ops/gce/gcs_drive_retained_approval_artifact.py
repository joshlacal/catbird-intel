#!/usr/bin/env python3
"""Verify a root-authored, hash-pinned local cache release approval record.

This verifies the operator's exact artifact, not the human conversation. The
sole owner must first establish Josh's actual authorization and pin its file
hash. The owner-only creator writes evidence; this module cannot enable cache
release or delete.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid

from gcs_drive_drain import Conflict
from gcs_drive_retained_cache_release import APPROVAL_SCHEMA
from gcs_drive_retained_overflow import _canonical
from gcs_drive_retained_runner import STOP_AT


HASH = re.compile(r"[0-9a-f]{64}\Z")
RUN = re.compile(r"[0-9a-f]{32}\Z")
MAX_ARTIFACT_BYTES = 16 * 1024
SCOPE = "generated_local_transfer_cache_only"
EVIDENCE_ROOT = Path("/data/catbird-retained-cache-evidence")
FIELDS = frozenset({
    "schema", "approved_by", "approval_reference", "authorization_scope",
    "policy_sha256", "inventory_sha256", "run_id", "pair_key", "paths",
    "retain_gcs", "retain_drive", "issued_at", "valid_until",
})


def _create_owned_file(*, root: Path, record: dict, expected_uid: int) -> tuple[Path, str]:
    """Private fixture seam. Never overwrite or remove an approval-shaped file."""
    path = root / ("approval-" + record["run_id"] + "-" + uuid.uuid4().hex + ".json")
    raw = _canonical(record)
    if len(raw) > MAX_ARTIFACT_BYTES or not root.is_absolute() or root.is_symlink():
        raise Conflict("Approval artifact destination is invalid")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parent = os.fstat(directory)
        if (not stat.S_ISDIR(parent.st_mode) or parent.st_uid != expected_uid
                or stat.S_IMODE(parent.st_mode) != 0o700):
            raise Conflict("Approval directory is not owner-private")
        fd = os.open(path.name, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        try:
            os.fchmod(fd, 0o600)
            created = os.fstat(fd)
            if (not stat.S_ISREG(created.st_mode) or created.st_uid != expected_uid
                    or created.st_nlink != 1 or created.st_dev != parent.st_dev):
                raise Conflict("Created approval file identity differs")
            if os.write(fd, raw) != len(raw):
                raise Conflict("Approval artifact write was short")
            os.fsync(fd)
            after = os.fstat(fd)
            named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            if ((after.st_dev, after.st_ino, after.st_size) !=
                    (named.st_dev, named.st_ino, named.st_size)
                    or after.st_size != len(raw)):
                raise Conflict("Created approval file path changed after write")
        finally:
            os.close(fd)
        os.fsync(directory)
        root_after = root.lstat()
        if (root_after.st_dev, root_after.st_ino) != (parent.st_dev, parent.st_ino):
            raise Conflict("Approval evidence directory changed after write")
    finally:
        os.close(directory)
    return path, hashlib.sha256(raw).hexdigest()


def create_owner_artifact_after_human_yes(*, inventory: dict,
                                          authorization_reference: str,
                                          authorization_epoch: float,
                                          policy_sha256: str,
                                          valid_until: float,
                                          root: Path = EVIDENCE_ROOT) -> tuple[Path, str]:
    """Owner-only file creation after independently witnessing Josh's yes.

    This function cannot authenticate a chat message. Calling it is the sole
    owner's attestation that the referenced human permission was actually
    received for the reviewed policy. It never enables or invokes removal.
    """
    if os.geteuid() != 0 or not HASH.fullmatch(policy_sha256):
        raise Conflict("Root operator or pinned retention policy is missing")
    now = time.time()
    authorized_at = _finite_epoch(authorization_epoch, "human authorization time")
    expiry = _finite_epoch(valid_until, "expiry")
    if not authorized_at <= now < expiry <= STOP_AT:
        raise Conflict("Approval creation is outside authorized window")
    if (not isinstance(authorization_reference, str)
            or not 0 < len(authorization_reference) <= 512
            or authorization_reference != authorization_reference.strip()
            or any(ord(char) < 0x20 for char in authorization_reference)):
        raise Conflict("Human authorization reference is invalid")
    if (not isinstance(inventory, dict)
            or not RUN.fullmatch(str(inventory.get("run_id", "")))
            or inventory.get("eligible_for_approval") is not True
            or inventory.get("gcs_retained") is not True
            or inventory.get("drive_media_verified") is not True
            or inventory.get("current_cache_state") != "present"):
        raise Conflict("Approved inventory run is invalid")
    try:
        paths = [inventory["stage"][kind]["path"] for kind in ("payload", "receipt")]
        key = inventory["pair_key"]
    except (KeyError, TypeError) as error:
        raise Conflict("Approved inventory paths are missing") from error
    stage = Path("/data") / ("gcs-drive-retained-" + inventory["run_id"]) / "stage" / str(key)
    if (not isinstance(key, str) or not HASH.fullmatch(key)
            or any(not isinstance(item, str) or Path(item).parent != stage
                   or item != str(stage / Path(item).name) for item in paths)
            or len(set(paths)) != 2):
        raise Conflict("Approved inventory stage paths differ")
    record = {
        "schema": APPROVAL_SCHEMA, "approved_by": "Josh",
        "approval_reference": authorization_reference,
        "authorization_scope": SCOPE, "policy_sha256": policy_sha256,
        "inventory_sha256": hashlib.sha256(_canonical(inventory)).hexdigest(),
        "run_id": inventory["run_id"], "pair_key": key, "paths": paths,
        "retain_gcs": True, "retain_drive": True,
        "issued_at": now, "valid_until": expiry,
    }
    return _create_owned_file(root=root, record=record, expected_uid=0)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Conflict("Approval artifact has duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(_value):
    raise Conflict("Approval artifact has nonfinite JSON value")


def _finite_epoch(value, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise Conflict("Approval artifact has invalid " + label)
    return float(value)


class OwnerPinnedApproval:
    """ApprovalVerifier for the future, currently disabled cache release API.

    `expected_sha256` and `authorization_reference` must come from the sole
    owner's reviewed task record. `expected_policy_sha256` pins the proposal
    Josh approved. None of these values independently authenticate a person.
    """

    def __init__(self, *, path: Path, expected_sha256: str,
                 inventory: dict, authorization_reference: str,
                 authorization_epoch: float, expected_policy_sha256: str,
                 root: Path = EVIDENCE_ROOT):
        if (not HASH.fullmatch(expected_sha256)
                or not HASH.fullmatch(expected_policy_sha256)
                or not isinstance(authorization_reference, str)
                or not 0 < len(authorization_reference) <= 512
                or authorization_reference != authorization_reference.strip()
                or any(ord(char) < 0x20 for char in authorization_reference)):
            raise Conflict("Pinned approval inputs are invalid")
        if (not isinstance(inventory, dict)
                or not RUN.fullmatch(str(inventory.get("run_id", "")))
                or inventory.get("eligible_for_approval") is not True
                or inventory.get("gcs_retained") is not True
                or inventory.get("drive_media_verified") is not True
                or inventory.get("current_cache_state") != "present"):
            raise Conflict("Pinned approval inventory is invalid")
        self.authorization_epoch = _finite_epoch(authorization_epoch, "human authorization time")
        if self.authorization_epoch >= STOP_AT:
            raise Conflict("Human authorization time is after capture STOP")
        try:
            paths = [inventory["stage"][kind]["path"] for kind in ("payload", "receipt")]
            key = inventory["pair_key"]
            raw = _canonical(inventory)
        except (KeyError, TypeError, ValueError) as error:
            raise Conflict("Pinned approval inventory is incomplete") from error
        stage = Path("/data") / ("gcs-drive-retained-" + inventory["run_id"]) / "stage" / str(key)
        if (not isinstance(key, str) or not HASH.fullmatch(key)
                or any(not isinstance(item, str) or Path(item).parent != stage
                       or Path(item).name in ("", ".", "..")
                       or item != str(stage / Path(item).name) for item in paths)
                or len(set(paths)) != 2):
            raise Conflict("Pinned approval paths differ")
        if (not root.is_absolute() or not path.is_absolute() or path.parent != root
                or not re.fullmatch("approval-" + inventory["run_id"] +
                                    r"-[0-9a-f]{32}\.json", path.name)):
            raise Conflict("Approval artifact path differs from private evidence root")
        self.path = path
        self.root = root
        self.expected_sha256 = expected_sha256
        self.inventory_sha256 = hashlib.sha256(raw).hexdigest()
        self.run_id = inventory["run_id"]
        self.pair_key = key
        self.paths = paths
        self.authorization_reference = authorization_reference
        self.expected_policy_sha256 = expected_policy_sha256
        self._clock_anchor: tuple[float, float] | None = None

    def _read_root_file(self) -> dict:
        return self._read_owned_file(0)

    def _read_owned_file(self, expected_uid: int) -> dict:
        """Internal fixture seam; the production entry always requires UID 0."""
        # O_NOFOLLOW on both directory and entry avoids a swapped final link.
        parent = self.root.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode) or self.root.parent.is_symlink():
            raise Conflict("Approval parent is not a directory")
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            root_info = os.fstat(root_fd)
            path_info = self.root.lstat()
            if (not stat.S_ISDIR(root_info.st_mode)
                    or (root_info.st_dev, root_info.st_ino) != (path_info.st_dev, path_info.st_ino)
                    or root_info.st_uid != expected_uid
                    or stat.S_IMODE(root_info.st_mode) != 0o700):
                raise Conflict("Approval directory is not root-owned private storage")
            fd = os.open(self.path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
            try:
                before = os.fstat(fd)
                if (not stat.S_ISREG(before.st_mode) or before.st_uid != expected_uid
                        or stat.S_IMODE(before.st_mode) != 0o600
                        or before.st_nlink != 1 or before.st_dev != root_info.st_dev
                        or not 0 < before.st_size <= MAX_ARTIFACT_BYTES):
                    raise Conflict("Approval file is not root-owned private evidence")
                raw = os.read(fd, MAX_ARTIFACT_BYTES + 1)
                after = os.fstat(fd)
                path_after = os.stat(self.path.name, dir_fd=root_fd, follow_symlinks=False)
                identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                if (len(raw) != before.st_size or identity(before) != identity(after)
                        or identity(before) != identity(path_after)):
                    raise Conflict("Approval file changed during review")
            finally:
                os.close(fd)
        finally:
            os.close(root_fd)
        if hashlib.sha256(raw).hexdigest() != self.expected_sha256:
            raise Conflict("Approval bytes differ from pinned hash")
        try:
            value = json.loads(raw, object_pairs_hook=_unique_pairs,
                               parse_constant=_bad_constant)
        except (ValueError, UnicodeError, TypeError) as error:
            raise Conflict("Approval JSON is malformed") from error
        if not isinstance(value, dict) or set(value) != FIELDS or raw != _canonical(value):
            raise Conflict("Approval JSON is not the exact canonical schema")
        return value

    def read_approval(self) -> dict:
        """Read exact pinned root bytes for the future release caller."""
        return self._read_root_file()

    def verify(self, approval: dict) -> dict:
        """Return the narrow proof shape expected by cache release v2."""
        record = self._read_root_file()
        if not isinstance(approval, dict) or approval != record:
            raise Conflict("Passed approval differs from pinned root artifact")
        issued = _finite_epoch(record["issued_at"], "issue time")
        valid_until = _finite_epoch(record["valid_until"], "expiry")
        now = time.time()
        tick = time.monotonic()
        if self._clock_anchor is None:
            self._clock_anchor = (now, tick)
        anchor_wall, anchor_tick = self._clock_anchor
        elapsed = tick - anchor_tick
        monotonic_wall = anchor_wall + elapsed
        if (elapsed < 0 or now + 2 < monotonic_wall
                or not self.authorization_epoch <= issued <= now < valid_until <= STOP_AT
                or monotonic_wall >= valid_until):
            raise Conflict("Approval artifact is outside its bounded policy window")
        if (record["schema"] != APPROVAL_SCHEMA
                or record["approved_by"] != "Josh"
                or record["approval_reference"] != self.authorization_reference
                or record["authorization_scope"] != SCOPE
                or record["policy_sha256"] != self.expected_policy_sha256
                or record["inventory_sha256"] != self.inventory_sha256
                or record["run_id"] != self.run_id
                or record["pair_key"] != self.pair_key
                or record["paths"] != self.paths
                or record["retain_gcs"] is not True
                or record["retain_drive"] is not True):
            raise Conflict("Approval artifact does not bind the reviewed local-only scope")
        return {"authenticated": True, "approved_by": "Josh",
                "approval_reference": self.authorization_reference,
                "inventory_sha256": self.inventory_sha256}
