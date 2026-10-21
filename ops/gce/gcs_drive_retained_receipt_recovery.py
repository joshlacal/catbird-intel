#!/usr/bin/env python3
"""Exact-ID, receipt-only recovery after one ambiguous retained media PUT.

This is an opt-in controller, not a queue worker. The sole operator supplies
the reviewed manifest, existing journal, canonical recovery authority, and
read-only verifier. Individual phase methods perform at most one Drive write.
The separately reviewed conditional run can follow one fresh status probe with
one exact receipt-only operation inside the same short exclusive lease.
"""
from __future__ import annotations

import hashlib
import json
import os
import copy
from pathlib import Path
import re
import stat
import time
import uuid

from drive_transfer_authority import AuthorityBlocked
from gcs_drive_drain import Conflict
from gcs_drive_retained_overflow import (
    FINAL_STATUS, INDEX_SCHEMA, _canonical as _canonical_bytes,
    _parse_manifest, pair_key,
    validate_overflow_manifest,
)
from gcs_drive_retained_receipt_transport import ReceiptRecoveryTransport
from retained_transfer_authority import RetainedObservationAuthority


MAX_PROBE_AGE_SECONDS = 60
MAX_LOCAL_PAYLOAD_BYTES = 320 << 20
MAX_LOCAL_RECEIPT_BYTES = 1 << 20
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX32 = re.compile(r"[0-9a-f]{32}\Z")


def _local_digest(path: Path, expected: dict, md5: str, limit: int) -> dict:
    """Hash an exact retained stage file with an inode stability check."""
    if (not isinstance(path, Path) or path.is_symlink() or
            not isinstance(md5, str) or not HEX32.fullmatch(md5)):
        raise Conflict("Pinned local stage path or MD5 differs")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
            before.st_size != expected["bytes"] or not 0 < before.st_size <= limit):
        raise Conflict("Pinned local stage identity differs")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                                row.st_mtime_ns, row.st_ctime_ns)
        if identity(opened) != identity(before):
            raise Conflict("Pinned local stage changed before read")
        sha = hashlib.sha256()
        md = hashlib.md5(usedforsecurity=False)
        total = 0
        while True:
            chunk = os.read(fd, min(1 << 20, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise Conflict("Pinned local stage exceeded byte bound")
            sha.update(chunk)
            md.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (identity(after) != identity(before) or
            identity(path.lstat()) != identity(before) or
            total != expected["bytes"] or sha.hexdigest() != expected["sha256"] or
            md.hexdigest() != md5):
        raise Conflict("Pinned local stage digest or inode changed")
    return {"bytes": total, "sha256": sha.hexdigest(), "md5": md.hexdigest()}


class ReceiptRecoveryController:
    """Single receipt, original folder/payload/receipt IDs, no generic replay."""

    def __init__(self, *, manifest_raw: bytes, journal_raw: bytes,
                 original_run_id: str, original_attempt_id: str,
                 account_id: str, writer_id: str, work_dir: Path,
                 authority, drive_request, verifier, clock=time.time):
        if not isinstance(manifest_raw, bytes) or not 0 < len(manifest_raw) <= 1 << 20:
            raise Conflict("Reviewed retained manifest bytes are unavailable")
        manifest = _parse_manifest(manifest_raw)
        validate_overflow_manifest(manifest, release_id=original_run_id,
                                   account_id=account_id)
        manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        key = pair_key(manifest)
        if not isinstance(journal_raw, bytes) or not 0 < len(journal_raw) <= 1 << 20:
            raise Conflict("Retained journal bytes are unavailable")
        journal = _parse_manifest(journal_raw)
        if (journal.get("schema") != "catbird.gcs-drive-drain.v1" or
                journal.get("bucket") != manifest["bucket"] or
                journal.get("root") != manifest["drive_root"] or
                journal.get("source_id") != manifest["source_id"] or
                journal.get("source_epoch") != manifest["source_epoch"] or
                set(journal.get("segments", {})) != {key}):
            raise Conflict("Retained journal source namespace differs")
        entry = journal["segments"][key]
        if (not isinstance(entry, dict) or
                entry.get("status") != "file_upload_inflight" or
                entry.get("key") != key or
                entry.get("release_id") != original_run_id or
                entry.get("manifest_sha256") != manifest_sha or
                entry.get("archive_qualified") is not False):
            raise Conflict("Retained journal is not the exact ambiguous pair")
        drive = entry.get("drive")
        if not isinstance(drive, dict) or set(drive) != {"folder", "payload", "receipt"}:
            raise Conflict("Retained journal lacks pinned Drive IDs")
        ids = [drive[kind].get("id") for kind in ("folder", "payload", "receipt")]
        if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != 3:
            raise Conflict("Retained journal Drive IDs differ")
        for kind in ("payload", "receipt"):
            source = manifest[kind]
            ref = entry.get(kind + "_ref")
            digest = entry.get(kind + "_digest")
            if (not isinstance(ref, dict) or not isinstance(digest, dict) or
                    ref.get("name") != source["object"] or
                    ref.get("generation") != source["generation"] or
                    ref.get("size") != source["bytes"] or
                    digest.get("size") != source["bytes"] or
                    digest.get("sha256") != source["sha256"] or
                    not isinstance(digest.get("md5"), str) or
                    not HEX32.fullmatch(digest["md5"])):
                raise Conflict("Retained journal source digest differs")
        receipt_session = drive["receipt"].get("session")
        payload_session = drive["payload"].get("session")
        if isinstance(receipt_session, str):
            self.is_payload_tail = False
            original_session = receipt_session
        elif (receipt_session is None and isinstance(payload_session, str) and
                getattr(authority, "permitted_branch", None) ==
                "payload_tail_and_receipt_auto"):
            self.is_payload_tail = True
            original_session = payload_session
        else:
            raise Conflict("Original receipt session is unavailable")
        session_sha = hashlib.sha256(original_session.encode()).hexdigest()
        if (not isinstance(work_dir, Path) or work_dir.is_symlink() or
                work_dir.joinpath("stage").is_symlink()):
            raise Conflict("Retained work directory is not trusted")
        stage = work_dir / "stage" / key
        if stage.is_symlink():
            raise Conflict("Retained stage is symlinked")
        receipt_name = entry.get("receipt_name")
        payload_name = entry.get("payload_name")
        if (not isinstance(receipt_name, str) or not receipt_name.endswith(".receipt.json")
                or not isinstance(payload_name, str) or
                receipt_name != payload_name + ".receipt.json" or
                "/" in receipt_name or "/" in payload_name):
            raise Conflict("Retained staged filenames differ")
        if (getattr(authority, "account_id", None) != account_id or
                getattr(authority, "writer_id", None) != writer_id or
                not callable(drive_request) or verifier is None):
            raise Conflict("Receipt recovery authority or verifier differs")
        old_attempt = authority.db.execute(
            "SELECT * FROM retained_attempt WHERE id=? AND run_id=?",
            (original_attempt_id, original_run_id)).fetchone()
        if (old_attempt is None or old_attempt["status"] != "unresolved" or
                old_attempt["http_status"] != 500 or
                old_attempt["method"] != "PUT" or
                old_attempt["kind"] != "media_chunk" or
                old_attempt["target_sha256"] != session_sha):
            raise AuthorityBlocked("Original charged HTTP 500 receipt attempt differs")
        if self.is_payload_tail:
            if (type(old_attempt["bytes"]) is not int or
                    not 0 < old_attempt["bytes"] <=
                        min(8 << 20, manifest["payload"]["bytes"])):
                raise AuthorityBlocked("Original charged HTTP 500 receipt attempt differs")
            self.original_charge_bytes = old_attempt["bytes"]
        else:
            if old_attempt["bytes"] != manifest["receipt"]["bytes"]:
                raise AuthorityBlocked("Original charged HTTP 500 receipt attempt differs")
            self.original_charge_bytes = manifest["receipt"]["bytes"]
        self.manifest, self.manifest_sha, self.key = manifest, manifest_sha, key
        self.entry, self.journal_sha = entry, hashlib.sha256(journal_raw).hexdigest()
        self.journal_doc, self.journal_raw, self.work_dir = journal, journal_raw, work_dir
        self.run_id, self.attempt_id = original_run_id, original_attempt_id
        self.account_id, self.writer_id = account_id, writer_id
        self.folder_id, self.payload_id, self.receipt_id = ids
        self.session, self.session_sha = original_session, session_sha
        self.payload_path, self.receipt_path = stage / payload_name, stage / receipt_name
        self.payload_name = payload_name
        self.receipt_name = receipt_name
        self.authority, self.verifier, self.clock = authority, verifier, clock

        def charged_request(url, method="GET", data=None, headers=None,
                            timeout=60, kind="drive_api"):
            # The durable allowlist is checked immediately before SafeDrive
            # reserves its actual globally paced request attempt.
            authority.prepare_request(url, method, kind, data, headers or {})
            return drive_request(url, method, data, headers, timeout=timeout, kind=kind)

        self.transport = ReceiptRecoveryTransport(charged_request)

    def _branch(self, expected: str) -> None:
        if self.authority.permitted_branch not in (expected, "receipt_only_auto"):
            raise AuthorityBlocked("Receipt recovery branch is not reviewed")

    def _recorded_probe(self, state: str) -> dict:
        outcome = self.authority.probe_outcome()
        if not isinstance(outcome, dict) or outcome.get("state") != state or \
                outcome.get("session_sha256") != self.session_sha or \
                not HEX64.fullmatch(str(outcome.get("proof_sha256", ""))):
            raise AuthorityBlocked("Pinned receipt status probe differs")
        when = outcome.get("observed_at")
        now = self.clock()
        if (type(when) not in (int, float) or type(now) not in (int, float) or
                not when <= now <= when + MAX_PROBE_AGE_SECONDS):
            raise AuthorityBlocked("Pinned receipt status probe is stale")
        if ((state == "partial" and
             (outcome.get("http_status") != 308 or
              type(outcome.get("acknowledged")) is not int or
              not 0 <= outcome["acknowledged"] < self.manifest["receipt"]["bytes"])) or
            (state == "expired" and
             (outcome.get("http_status") not in (404, 410) or outcome.get("acknowledged") != 0)) or
            (state == "complete" and
             (outcome.get("http_status") not in (200, 201) or
              outcome.get("acknowledged") != self.manifest["receipt"]["bytes"]))):
            raise AuthorityBlocked("Pinned receipt status probe outcome is malformed")
        return outcome

    def probe_only(self) -> dict:
        """One empty status PUT.  This method never sends media."""
        if self.authority.permitted_branch not in (
                "probe_only", "partial_exact_suffix", "expired_same_id_restart",
                "receipt_only_auto"):
            raise AuthorityBlocked("Receipt status probe is not reviewed")
        result = self.transport.probe(self.session, self.manifest["receipt"]["bytes"],
                                      self.receipt_id, self.session_sha)
        digest = self.authority.record_probe(result)
        return {"state": result["state"], "acknowledged": result["acknowledged"],
                "http_status": result["http_status"], "proof_sha256": digest}

    def continue_partial(self) -> dict:
        """One exact suffix PUT after a separately reviewed fresh 308 probe."""
        self._branch("partial_exact_suffix")
        outcome = self._recorded_probe("partial")
        digest = self.entry["receipt_digest"]
        return self.transport.resume(
            self.session, self.receipt_path, outcome["acknowledged"],
            self.manifest["receipt"]["bytes"], self.receipt_id,
            self.manifest["receipt"]["sha256"], digest["md5"], self.session_sha)

    def start_expired(self, *, operator_reference: str) -> dict:
        """One same-ID session POST after 404 plus fresh exact-ID absence proof.

        This does not send any media.  The new private session is durably stored
        by the authority before send_restarted() may be called.
        """
        self._branch("expired_same_id_restart")
        self._recorded_probe("expired")
        if not isinstance(operator_reference, str) or not operator_reference.strip():
            raise AuthorityBlocked("Receipt recovery operator reference is missing")
        self.verifier.verify_account(self.account_id)
        self.verifier.verify_folder(self.folder_id,
                                    self.manifest["folder_name"],
                                    self.manifest["drive_root"])
        absent = self.verifier.verify_exact_receipt_absent(
            account_id=self.account_id, root_id=self.manifest["drive_root"],
            folder_id=self.folder_id, receipt_id=self.receipt_id)
        if (not isinstance(absent, dict) or absent.get("account_id") != self.account_id or
                absent.get("root_id") != self.manifest["drive_root"] or
                absent.get("folder_id") != self.folder_id or
                absent.get("receipt_id") != self.receipt_id or
                absent.get("http_status") != 404 or
                type(absent.get("observed_at")) not in (int, float) or
                not absent["observed_at"] <= self.clock() <=
                    absent["observed_at"] + MAX_PROBE_AGE_SECONDS):
            raise AuthorityBlocked("Fresh exact-ID receipt absence is unproven")
        self.authority.record_id_absence({
            "receipt_id": self.receipt_id, "folder_id": self.folder_id,
            "account_id": self.account_id, "http_status": 404,
            "observed_at": absent["observed_at"],
            "operator_reference": operator_reference,
        })
        session = self.transport.restart_same_id(
            self.receipt_id, self.folder_id, self.receipt_name,
            self.manifest["receipt"]["bytes"])
        digest = self.authority.record_new_session(session)
        return {"state": "new_session_persisted", "session_sha256": digest}

    def send_restarted(self) -> dict:
        """One full RECEIPT PUT to a separately created and saved same-ID session."""
        self._branch("expired_same_id_restart")
        self._recorded_probe("expired")
        private = self.authority.new_session()
        if (not isinstance(private, dict) or
                not isinstance(private.get("url"), str) or
                hashlib.sha256(private["url"].encode()).hexdigest() !=
                    private.get("session_sha256") or
                not HEX64.fullmatch(str(private.get("session_sha256", "")))):
            raise AuthorityBlocked("New receipt session was not durably pinned")
        return self.transport.resume(
            private["url"], self.receipt_path, 0,
            self.manifest["receipt"]["bytes"], self.receipt_id,
            self.manifest["receipt"]["sha256"],
            self.entry["receipt_digest"]["md5"], private["session_sha256"])

    def _mapping_sha256(self) -> str:
        def source(kind):
            item = self.manifest[kind]
            return {name: item[name] for name in
                    ("object", "generation", "bytes", "sha256")}
        record = {
            "schema": INDEX_SCHEMA, "pair_key": self.key,
            "manifest_sha256": self.manifest_sha,
            "index_bucket": self.manifest["index_bucket"],
            "index_object": self.manifest["index_object"],
            "payload_ref": source("payload"), "receipt_ref": source("receipt"),
            "payload_digest": self.entry["payload_digest"],
            "receipt_digest": self.entry["receipt_digest"],
            "drive_account_id": self.account_id,
            "drive_root": self.manifest["drive_root"],
            "folder_id": self.folder_id, "payload_id": self.payload_id,
            "receipt_id": self.receipt_id, "retained_gcs": True,
            "retained_local": True, "archive_qualified": False,
            "status": FINAL_STATUS,
        }
        return hashlib.sha256(_canonical_bytes(record)).hexdigest()

    def verify_completed(self, *, operator_reference: str) -> dict:
        """Read full exact-generation sources and exact-ID Drive media; no send.

        The verifier is a read-only adapter.  Every returned SHA-256, MD5,
        length, source generation, account, parent and Drive ID is checked here
        before the authority appends a forensic reconciliation record.  The
        original HTTP 500 attempt and its byte charge are never rewritten.
        """
        if not isinstance(operator_reference, str) or not operator_reference.strip():
            raise Conflict("Recovery operator reference is missing")
        self.verifier.verify_account(self.account_id)
        self.verifier.verify_folder(self.folder_id,
                                    self.manifest["folder_name"],
                                    self.manifest["drive_root"])
        proof = {
            "schema": "catbird.drive.retained-receipt-500-reconciliation.v1",
            "attempt_id": self.attempt_id, "run_id": self.run_id,
            "pair_key": self.key, "manifest_sha256": self.manifest_sha,
            "account_id": self.account_id, "writer_id": self.writer_id,
            "original_http_status": 500,
            "original_charge_bytes": self.original_charge_bytes,
            "target_sha256": self.session_sha,
            "receipt_ref": self.manifest["receipt"],
            "mapping_sha256": self._mapping_sha256(),
            "journal_sha256": self.journal_sha,
            "folder_id": self.folder_id,
            "payload_drive_id": self.payload_id,
            "receipt_drive_id": self.receipt_id,
            "operator_reference": operator_reference,
            "applied_at": self.clock(),
        }
        if self.is_payload_tail:
            proof["permitted_branch"] = "payload_tail_and_receipt_auto"
            proof["payload_session_sha256"] = self.session_sha
        for kind, file_id, path, bound in (
            ("payload", self.payload_id, self.payload_path, MAX_LOCAL_PAYLOAD_BYTES),
            ("receipt", self.receipt_id, self.receipt_path, MAX_LOCAL_RECEIPT_BYTES),
        ):
            expected = self.manifest[kind]
            local = _local_digest(path, expected,
                                  self.entry[kind + "_digest"]["md5"], bound)
            observed_gcs = self.verifier.gcs_digest(
                expected["object"], expected["generation"])
            observed_drive = self.verifier.drive_digest(file_id)
            if not isinstance(observed_gcs, dict) or not isinstance(observed_drive, dict):
                raise Conflict("Recovery source or Drive full-media proof is missing")
            wanted = {"bytes": expected["bytes"], "sha256": expected["sha256"],
                      "md5": local["md5"]}
            if any(observed_gcs.get(field) != value for field, value in wanted.items()) or \
                    observed_gcs.get("object") != expected["object"] or \
                    observed_gcs.get("generation") != expected["generation"]:
                raise Conflict("Exact GCS generation full-media proof differs")
            if any(observed_drive.get(field) != value for field, value in wanted.items()) or \
                    observed_drive.get("id") != file_id or \
                    observed_drive.get("account_id") != self.account_id or \
                    observed_drive.get("parent_id") != self.folder_id or \
                    observed_drive.get("trashed") is not False or \
                    observed_drive.get("media_readback") is not True:
                raise Conflict("Exact Drive ID full-media proof differs")
            proof[kind + "_local"] = local
            proof[kind + "_gcs"] = observed_gcs
            proof[kind + "_drive"] = observed_drive
        attempt = self.authority.db.execute(
            "SELECT * FROM retained_attempt WHERE id=? AND run_id=?",
            (self.attempt_id, self.run_id)).fetchone()
        if (attempt is None or attempt["http_status"] != 500 or
                attempt["status"] != "unresolved" or
                attempt["bytes"] != self.original_charge_bytes or
                attempt["target_sha256"] != self.session_sha):
            raise AuthorityBlocked("Original HTTP 500 attempt changed")
        proof["operation_key"] = attempt["operation_key"]
        self.verifier.verify_account(self.account_id)
        digest = self.authority.apply_verified_proof(proof)
        return {"state": "forensically_verified", "proof_sha256": digest,
                "mapping_sha256": proof["mapping_sha256"]}

    def run_authorized(self, *, operator_reference: str) -> dict:
        """One reviewed conditional receipt recovery in a single short lease.

        A fresh probe selects only its documented branch.  Each transport call
        is a separate charged request and a 308 continuation result stops here,
        requiring another reviewed lease rather than an automatic loop.
        """
        if self.authority.permitted_branch == "payload_tail_and_receipt_auto":
            if not isinstance(operator_reference, str) or not operator_reference.strip():
                raise AuthorityBlocked("Receipt recovery operator reference is missing")
            existing_probe = self.authority.probe_outcome()
            if existing_probe is None:
                probe_res = self.transport.probe(
                    self.session, self.manifest["payload"]["bytes"],
                    self.payload_id, self.session_sha,
                    max_bytes=MAX_LOCAL_PAYLOAD_BYTES)
                self.authority.record_probe(probe_res)
                result = self.authority.probe_outcome()
            else:
                result = existing_probe
            state = result["state"]
            if state == "partial":
                payload_res = self.transport.resume(
                    self.session, self.payload_path, result["acknowledged"],
                    self.manifest["payload"]["bytes"], self.payload_id,
                    self.manifest["payload"]["sha256"],
                    self.entry["payload_digest"]["md5"], self.session_sha,
                    max_bytes=MAX_LOCAL_PAYLOAD_BYTES)
                if payload_res["state"] != "complete":
                    return {"state": "payload_recovery_partial_stopped",
                            "acknowledged": payload_res.get("acknowledged")}
            elif state == "expired":
                payload_absent = self.verifier.verify_exact_payload_absent(
                    account_id=self.account_id,
                    root_id=self.manifest["drive_root"],
                    folder_id=self.folder_id,
                    payload_id=self.payload_id,
                )
                if payload_absent is not None:
                    if (not isinstance(payload_absent, dict) or
                            payload_absent.get("account_id") != self.account_id or
                            payload_absent.get("root_id") != self.manifest["drive_root"] or
                            payload_absent.get("folder_id") != self.folder_id or
                            payload_absent.get("payload_id") != self.payload_id or
                            payload_absent.get("http_status") != 404 or
                            type(payload_absent.get("observed_at")) not in (int, float) or
                            not payload_absent["observed_at"] <= self.clock() <=
                                payload_absent["observed_at"] + MAX_PROBE_AGE_SECONDS):
                        raise AuthorityBlocked("Fresh exact-ID payload absence is unproven")
                    self.authority.record_id_absence({
                        "payload_id": self.payload_id,
                        "folder_id": self.folder_id,
                        "account_id": self.account_id,
                        "http_status": 404,
                        "observed_at": payload_absent["observed_at"],
                        "operator_reference": operator_reference,
                    })
                    payload_session = self.transport.restart_same_id(
                        self.payload_id, self.folder_id, self.payload_name,
                        self.manifest["payload"]["bytes"],
                        max_bytes=MAX_LOCAL_PAYLOAD_BYTES)
                    payload_session_sha = self.authority.record_new_session(
                        payload_session)
                    payload_res = self.transport.resume(
                        payload_session, self.payload_path, 0,
                        self.manifest["payload"]["bytes"], self.payload_id,
                        self.manifest["payload"]["sha256"],
                        self.entry["payload_digest"]["md5"],
                        payload_session_sha,
                        max_bytes=MAX_LOCAL_PAYLOAD_BYTES)
                    if payload_res["state"] != "complete":
                        return {"state": "payload_recovery_partial_stopped",
                                "acknowledged": payload_res.get("acknowledged")}
            elif state != "complete":
                raise AuthorityBlocked("Payload probe result is not a reviewed branch")
            self.verifier.verify_account(self.account_id)
            self.verifier.verify_folder(self.folder_id,
                                        self.manifest["folder_name"],
                                        self.manifest["drive_root"])
            absent = self.verifier.verify_exact_receipt_absent(
                account_id=self.account_id, root_id=self.manifest["drive_root"],
                folder_id=self.folder_id, receipt_id=self.receipt_id)
            if (not isinstance(absent, dict) or absent.get("account_id") != self.account_id or
                    absent.get("root_id") != self.manifest["drive_root"] or
                    absent.get("folder_id") != self.folder_id or
                    absent.get("receipt_id") != self.receipt_id or
                    absent.get("http_status") != 404 or
                    type(absent.get("observed_at")) not in (int, float) or
                    not absent["observed_at"] <= self.clock() <=
                        absent["observed_at"] + MAX_PROBE_AGE_SECONDS):
                raise AuthorityBlocked("Fresh exact-ID receipt absence is unproven")
            self.authority.record_id_absence({
                "receipt_id": self.receipt_id, "folder_id": self.folder_id,
                "account_id": self.account_id, "http_status": 404,
                "observed_at": absent["observed_at"],
                "operator_reference": operator_reference,
            })
            receipt_session = self.transport.restart_same_id(
                self.receipt_id, self.folder_id, self.receipt_name,
                self.manifest["receipt"]["bytes"])
            receipt_session_sha = self.authority.record_new_session(receipt_session)
            receipt_res = self.transport.resume(
                receipt_session, self.receipt_path, 0,
                self.manifest["receipt"]["bytes"], self.receipt_id,
                self.manifest["receipt"]["sha256"],
                self.entry["receipt_digest"]["md5"], receipt_session_sha)
            if receipt_res["state"] != "complete":
                return {"state": "receipt_recovery_partial_stopped",
                        "acknowledged": receipt_res.get("acknowledged")}
            return self.verify_completed(operator_reference=operator_reference)
        if self.authority.permitted_branch != "receipt_only_auto":
            raise AuthorityBlocked("Conditional receipt recovery is not reviewed")
        existing_probe = self.authority.probe_outcome()
        result = existing_probe if existing_probe is not None else self.probe_only()
        state = result["state"]
        if state == "partial":
            result = self.continue_partial()
        elif state == "expired":
            self.start_expired(operator_reference=operator_reference)
            result = self.send_restarted()
        elif state != "complete":
            raise AuthorityBlocked("Receipt probe result is not a reviewed branch")
        if result["state"] != "complete":
            return {"state": "receipt_recovery_partial_stopped",
                    "acknowledged": result.get("acknowledged")}
        return self.verify_completed(operator_reference=operator_reference)


def _read_private_regular(path: Path, limit: int) -> bytes:
    if path.is_symlink():
        raise Conflict("Retained recovery evidence path is symlinked")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
            before.st_size > limit or before.st_mode & 0o077):
        raise Conflict("Retained recovery evidence file differs")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        current = os.fstat(fd)
        if (current.st_dev, current.st_ino, current.st_size,
            current.st_mtime_ns) != (before.st_dev, before.st_ino,
                                     before.st_size, before.st_mtime_ns):
            raise Conflict("Retained recovery evidence changed before read")
        raw = os.read(fd, limit + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (len(raw) != before.st_size or len(raw) > limit or
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) !=
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)):
        raise Conflict("Retained recovery evidence changed during read")
    return raw


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def finalize_reconciled_journal(controller: ReceiptRecoveryController, *,
                                held_lock_fd: int) -> dict:
    """No-send repair of one journal after an immutable full-media proof.

    The caller MUST hold `flock(LOCK_EX|LOCK_NB)` on `work/overflow.lock`
    throughout this call. It may then exit cleanly; the separate recovery-only
    mapping worker acquires the same lock afresh after this unit is inactive.
    This function verifies that the supplied live fd names the exact regular
    inode. It never locks, unlocks, calls Drive/GCS, or deletes staged files.
    A durable sidecar is appended and synced before the journal is atomically
    replaced and synced.
    """
    if not isinstance(controller, ReceiptRecoveryController):
        raise Conflict("Exact receipt recovery controller is required")
    work = controller.work_dir
    if work.is_symlink() or not work.is_dir():
        raise Conflict("Retained recovery work directory differs")
    lock_path = work / "overflow.lock"
    if lock_path.is_symlink() or type(held_lock_fd) is not int:
        raise Conflict("Retained recovery lock path or fd differs")
    lock_info, fd_info = lock_path.lstat(), os.fstat(held_lock_fd)
    if (not stat.S_ISREG(lock_info.st_mode) or lock_info.st_nlink != 1 or
            lock_info.st_mode & 0o077 or
            (lock_info.st_dev, lock_info.st_ino) != (fd_info.st_dev, fd_info.st_ino)):
        raise Conflict("Caller does not hold the exact overflow lock inode")
    path = work / "journal.json"
    journal_before = path.lstat()
    raw = _read_private_regular(path, 1 << 20)
    doc = _parse_manifest(raw)
    expected = copy.deepcopy(controller.journal_doc)
    expected_entry = expected["segments"][controller.key]
    expected_entry["status"] = "drive_verified"
    mapping_sha = controller._mapping_sha256()
    expected_entry["mapping_sha256"] = mapping_sha
    updated_raw = _canonical_bytes(expected)
    updated_sha = hashlib.sha256(updated_raw).hexdigest()
    old_sha = controller.journal_sha
    state = doc.get("segments", {}).get(controller.key, {}).get("status")
    if state == "file_upload_inflight":
        if hashlib.sha256(raw).hexdigest() != old_sha or doc != controller.journal_doc:
            raise Conflict("Original retained journal changed before repair")
    elif state == "drive_verified":
        if doc != expected or hashlib.sha256(raw).hexdigest() != updated_sha:
            raise Conflict("Recovered retained journal differs")
    else:
        raise Conflict("Retained journal is not in a repairable state")
    # Read-only semantic validation precedes and follows the account write
    # transaction. The transaction pins the ledger while file evidence is
    # synced, so another admitted writer cannot interleave this transition.
    account = RetainedObservationAuthority.inspect_account(
        Path(controller.authority.db_path), writer_id=controller.writer_id,
        account_id=controller.account_id, clock=controller.clock)
    if account["unresolved_attempts"] != 0:
        raise AuthorityBlocked("Unresolved send still blocks journal repair")
    db = controller.authority.db
    if db.in_transaction:
        raise AuthorityBlocked("Journal repair requires a clean ledger transaction")
    db.execute("BEGIN IMMEDIATE")
    try:
        account_row = db.execute(
            "SELECT * FROM retained_account WHERE id=1").fetchone()
        run = db.execute("SELECT * FROM retained_run WHERE id=?",
                         (controller.run_id,)).fetchone()
        reconcile = db.execute(
            "SELECT * FROM retained_reconciliation WHERE attempt_id=? AND run_id=?",
            (controller.attempt_id, controller.run_id)).fetchone()
        original_attempt = db.execute(
            "SELECT * FROM retained_attempt WHERE id=? AND run_id=?",
            (controller.attempt_id, controller.run_id)).fetchone()
        recovery = db.execute(
            "SELECT * FROM retained_receipt_recovery WHERE id=?",
            (controller.authority.release_id,)).fetchone()
        unresolved_raw = db.execute(
            "SELECT COUNT(*) FROM retained_attempt a LEFT JOIN "
            "retained_reconciliation r ON r.attempt_id=a.id "
            "WHERE a.status='unresolved' AND r.attempt_id IS NULL"
        ).fetchone()[0]
        recovery_unresolved = db.execute(
            "SELECT COUNT(*) FROM retained_receipt_recovery_attempt "
            "WHERE status='unresolved'"
        ).fetchone()[0]
        if (account_row is None or
                account_row["account_id"] != controller.account_id or
                account_row["writer_id"] != controller.writer_id or
                account_row["db_path"] != controller.authority.db_path or
                (account_row["process_host"], account_row["process_pid"],
                 account_row["process_nonce"]) !=
                (controller.authority.process_host,
                 controller.authority.process_pid,
                 controller.authority.process_nonce) or
                run is None or run["status"] != "verified" or
                run["pair_key"] != controller.key or
                run["manifest_sha256"] != controller.manifest_sha or
                run["mapping_sha256"] != mapping_sha or
                original_attempt is None or
                original_attempt["status"] != "unresolved" or
                original_attempt["http_status"] != 500 or
                original_attempt["bytes"] !=
                    controller.original_charge_bytes or
                original_attempt["target_sha256"] != controller.session_sha or
                reconcile is None or reconcile["pair_key"] != controller.key or
                not HEX64.fullmatch(str(reconcile["proof_sha256"])) or
                recovery is None or recovery["status"] != "closed" or
                recovery["original_attempt_id"] != controller.attempt_id or
                unresolved_raw != 0 or recovery_unresolved != 0):
            raise AuthorityBlocked("Retained forensic proof is not closed and exact")
        try:
            proof = json.loads(reconcile["proof_json"])
        except (TypeError, ValueError) as error:
            raise AuthorityBlocked("Retained forensic proof cannot be decoded") from error
        if (hashlib.sha256(reconcile["proof_json"].encode()).hexdigest() !=
                reconcile["proof_sha256"] or
                proof.get("mapping_sha256") != mapping_sha or
                proof.get("journal_sha256") != old_sha or
                proof.get("folder_id") != controller.folder_id or
                proof.get("payload_drive_id") != controller.payload_id or
                proof.get("receipt_drive_id") != controller.receipt_id):
            raise AuthorityBlocked("Retained forensic proof differs from journal")
        original_sidecar = work / "journal.before-recovery.json"
        if original_sidecar.exists() or original_sidecar.is_symlink():
            prior_original = _read_private_regular(original_sidecar, 1 << 20)
            if prior_original != controller.journal_raw:
                raise Conflict("Original retained journal backup differs")
        else:
            if state != "file_upload_inflight":
                raise Conflict("Already repaired journal lacks original byte backup")
            fd = os.open(original_sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_NOFOLLOW, 0o600)
            try:
                if os.write(fd, controller.journal_raw) != len(controller.journal_raw):
                    raise Conflict("Original retained journal backup was short")
                os.fsync(fd)
            finally:
                os.close(fd)
            _sync_directory(work)
        record = {
            "schema": "catbird.drive.retained-recovery-journal-transition.v1",
            "run_id": controller.run_id, "attempt_id": controller.attempt_id,
            "pair_key": controller.key, "manifest_sha256": controller.manifest_sha,
            "reconciliation_sha256": reconcile["proof_sha256"],
            "old_journal_sha256": old_sha,
            "new_journal_sha256": updated_sha,
            "mapping_sha256": mapping_sha,
            "account_id": controller.account_id,
            "folder_id": controller.folder_id,
            "payload_id": controller.payload_id,
            "receipt_id": controller.receipt_id,
        }
        sidecar = work / "retained-recovery-journal-evidence.jsonl"
        line = _canonical_bytes(record) + b"\n"
        if sidecar.exists() or sidecar.is_symlink():
            prior = _read_private_regular(sidecar, 1 << 20)
            if prior != line:
                raise Conflict("Existing retained recovery transition evidence differs")
        else:
            fd = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_NOFOLLOW, 0o600)
            try:
                if os.write(fd, line) != len(line):
                    raise Conflict("Retained recovery evidence append was short")
                os.fsync(fd)
            finally:
                os.close(fd)
            _sync_directory(work)
        if state == "file_upload_inflight":
            journal_current = path.lstat()
            if (journal_current.st_dev, journal_current.st_ino,
                journal_current.st_size, journal_current.st_mtime_ns) != (
                journal_before.st_dev, journal_before.st_ino,
                journal_before.st_size, journal_before.st_mtime_ns):
                raise Conflict("Retained journal inode changed during repair")
            temp = work / ("journal.recovery-" + uuid.uuid4().hex + ".tmp")
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                         os.O_NOFOLLOW, 0o600)
            try:
                if os.write(fd, updated_raw) != len(updated_raw):
                    raise Conflict("Retained recovery journal write was short")
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temp, path)
            _sync_directory(work)
        db.execute("COMMIT")
    except BaseException:
        if db.in_transaction:
            db.execute("ROLLBACK")
        raise
    return {"state": "drive_verified_no_send", "mapping_sha256": mapping_sha,
            "journal_sha256": updated_sha}
