"""Synthetic-only tests for the offline retained-overflow first slice."""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from gcs_drive_drain import Conflict, FOLDER, ObjectRef
from gcs_drive_diagnostic import PAYLOAD as COMPLETED_DIAGNOSTIC_PAYLOAD
from gcs_drive_retained_overflow import (
    BUCKET, CONTROL_ALLOWANCE, DRIVE_ROOT, FINAL_STATUS, RetainedOverflow,
    INDEX_PREFIX, OverflowGCS, SOURCE_EPOCH, SOURCE_ID, _supervise_worker,
    load_overflow_manifest, pair_key,
)


RELEASE = "2" * 32
ACCOUNT = "test@example.invalid"
NAME = "seg_0000000001.jss"
PAYLOAD_BYTES = b"synthetic-payload"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD_BYTES).hexdigest()
PAYLOAD_MD5 = hashlib.md5(PAYLOAD_BYTES).digest()
PAYLOAD_OBJECT = (f"sources/{SOURCE_ID}/{SOURCE_EPOCH}/segments/00000000/"
                  f"{NAME}/{PAYLOAD_SHA}.jss")
PAYLOAD_GEN = 1001
RECEIPT_OBJECT = PAYLOAD_OBJECT + f".g{PAYLOAD_GEN}.receipt.json"
RECEIPT_GEN = 1002


def fixture_manifest():
    receipt = {
        "schema": "catbird.jetstream.archive.v1", "name": NAME, "index": 1,
        "source_id": SOURCE_ID, "source_epoch": SOURCE_EPOCH,
        "sha256": PAYLOAD_SHA, "bytes": len(PAYLOAD_BYTES),
        "gcs": {"bucket": BUCKET, "object": PAYLOAD_OBJECT,
                "generation": PAYLOAD_GEN, "sha256": PAYLOAD_SHA,
                "bytes": len(PAYLOAD_BYTES),
                "md5_hash": base64.b64encode(PAYLOAD_MD5).decode()},
    }
    receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
    payload = {"object": PAYLOAD_OBJECT, "generation": PAYLOAD_GEN,
               "bytes": len(PAYLOAD_BYTES), "sha256": PAYLOAD_SHA}
    receipt_ref = {"object": RECEIPT_OBJECT, "generation": RECEIPT_GEN,
                   "bytes": len(receipt_bytes),
                   "sha256": hashlib.sha256(receipt_bytes).hexdigest()}
    manifest = {"schema": "catbird.drive.retained-overflow.v1",
                "release_id": RELEASE, "account_id": ACCOUNT, "bucket": BUCKET,
                "source_id": SOURCE_ID, "source_epoch": SOURCE_EPOCH,
                "drive_root": DRIVE_ROOT, "payload": payload, "receipt": receipt_ref,
                "control_allowance_bytes": CONTROL_ALLOWANCE,
                "max_charged_bytes": len(PAYLOAD_BYTES) + len(receipt_bytes) + CONTROL_ALLOWANCE,
                "max_duration_seconds": 1200, "retain_gcs": True,
                "retain_local": True, "archive_qualified": False}
    manifest["folder_name"] = "catbird-overflow-" + pair_key(manifest)[:32]
    manifest["index_bucket"] = BUCKET
    manifest["index_object"] = INDEX_PREFIX + pair_key(manifest) + ".json"
    return manifest, receipt_bytes


class FakeCloud:
    bucket_name = BUCKET

    def __init__(self, manifest, receipt_bytes):
        self.deleted = []
        self.index = {}
        self.index_creates = 0
        self.lose_index_response = False
        self.fail_index_before_create = False
        self._data = {PAYLOAD_OBJECT: PAYLOAD_BYTES, RECEIPT_OBJECT: receipt_bytes}
        self._refs = {
            PAYLOAD_OBJECT: ObjectRef(PAYLOAD_OBJECT, PAYLOAD_GEN, len(PAYLOAD_BYTES),
                                      {"sha256": PAYLOAD_SHA}),
            RECEIPT_OBJECT: ObjectRef(RECEIPT_OBJECT, RECEIPT_GEN, len(receipt_bytes),
                                      {"sha256": manifest["receipt"]["sha256"],
                                       "segment_name": NAME, "payload_object": PAYLOAD_OBJECT,
                                       "payload_generation": str(PAYLOAD_GEN)}),
        }

    def lookup(self, name, generation=None):
        ref = self._refs.get(name)
        return ref if ref and (generation is None or generation == ref.generation) else None

    def download(self, ref, path):
        Path(path).write_bytes(self._data[ref.name])

    def delete(self, ref):
        self.deleted.append(ref)
        raise AssertionError("GCS deletion is forbidden")

    def publish_index(self, name, data, digest):
        self.index_creates += 1
        if self.fail_index_before_create:
            raise TimeoutError("synthetic index request never dispatched")
        if name in self.index:
            raise Conflict("index exists")
        assert hashlib.sha256(data).hexdigest() == digest
        self.index[name] = (3001, data)
        if self.lose_index_response:
            raise TimeoutError("synthetic lost index response")

    def read_index(self, name):
        return self.index.get(name)


class FakeAuthority:
    def __init__(self, digest):
        self.manifest_sha256 = digest
        self.release_id = RELEASE
        self.account_id = ACCOUNT
        self.reserved = []
        self.closed = []
        self.verified = []
        self.committed = []
        self.status = "active"
        self.fail_mark_after_commit = False
        self.fail_mark_before_commit = False
        self.fail_close_after_commit = False
        self.fail_index_commit_after_commit = False
        self.index_reservations = {}

    def assert_active(self, *_args):
        pass

    def reserve_pair(self, key, media, control, metadata):
        self.reserved.append((key, media, control, metadata))

    def close_pair(self, key):
        self.closed.append(key)
        self.status = "index_pending"
        if self.fail_close_after_commit:
            raise TimeoutError("synthetic crash after close")

    def pair_status(self, _key, _digest):
        return self.status

    def mark_pair_verified(self, key, digest):
        if self.fail_mark_before_commit:
            raise TimeoutError("synthetic crash before verification mark")
        self.verified.append((key, digest))
        self.status = "verified"
        if self.fail_mark_after_commit:
            raise TimeoutError("synthetic crash after verification mark")

    def mark_index_committed(self, key, digest):
        self.committed.append((key, digest))
        self.status = "closed"
        if self.fail_index_commit_after_commit:
            raise TimeoutError("synthetic crash after index commit")

    def reserve_index_create(self, key, digest, name, body_bytes):
        binding = (key, digest, name, body_bytes)
        old = self.index_reservations.get(key)
        if old is not None and old != binding:
            raise Conflict("index reservation changed")
        self.index_reservations[key] = binding
        return "reservation-1"


class FakeDrive:
    def __init__(self):
        self.ids = iter(("folder-id", "payload-id", "receipt-id"))
        self.objects = {DRIVE_ROOT: {"id": DRIVE_ROOT, "mimeType": FOLDER,
                                     "trashed": False, "parents": []}}
        self.folder_posts = 0
        self.media_creates = 0
        self.lose_folder_response = False
        self.lose_media_response = False

    def verify_account(self, account):
        if account != ACCOUNT:
            raise Conflict("Wrong synthetic account")

    def reserve_id(self):
        return next(self.ids)

    def metadata(self, file_id):
        return self.objects[file_id]

    @contextlib.contextmanager
    def request(self, _url, method, body, _headers, *, kind):
        assert method == "POST" and kind == "overflow_folder_create"
        obj = json.loads(body)
        self.folder_posts += 1
        self.objects[obj["id"]] = {**obj, "trashed": False}
        if self.lose_folder_response:
            raise TimeoutError("synthetic lost folder response")
        yield io.BytesIO(json.dumps(self.objects[obj["id"]]).encode())

    def ensure_file(self, path, parent, expected, state, _save):
        self.media_creates += 1
        got = {"id": state["id"], "name": path.name, "size": str(expected["size"]),
               "md5Checksum": expected["md5"], "parents": [parent], "trashed": False}
        assert len(path.read_bytes()) == expected["size"]
        self.objects[state["id"]] = got
        if self.lose_media_response:
            raise TimeoutError("synthetic lost media response")
        return got


class RetainedOverflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name) / "work"
        self.manifest, self.receipt_bytes = fixture_manifest()
        self.path = Path(self.temp.name) / "manifest.json"
        self.path.write_text(json.dumps(self.manifest, sort_keys=True))
        self.manifest, self.digest = load_overflow_manifest(
            self.path, release_id=RELEASE, account_id=ACCOUNT)
        self.cloud = FakeCloud(self.manifest, self.receipt_bytes)
        self.drive = FakeDrive()
        self.authority = FakeAuthority(self.digest)

    def copy(self):
        return RetainedOverflow(self.cloud, self.drive, self.authority,
                                self.work, self.manifest, self.digest, self.path.read_bytes(),
                                capture_segments_dir=Path(self.temp.name),
                                min_capture_free_bytes=1)

    def test_retained_copy_has_immutable_mapping_and_never_deletes(self):
        with self.copy() as copier:
            result = copier.run_one()
        self.assertEqual(result["status"], FINAL_STATUS)
        self.assertFalse(result["archive_qualified"])
        self.assertEqual(self.drive.folder_posts, 1)
        self.assertEqual(self.drive.media_creates, 2)
        self.assertEqual(self.cloud.deleted, [])
        self.assertTrue((self.work / "stage" / pair_key(self.manifest) / NAME).is_file())
        rows = [json.loads(line) for line in (self.work / "retained-index.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["payload_ref"]["generation"], PAYLOAD_GEN)
        self.assertEqual(rows[0]["receipt_ref"]["generation"], RECEIPT_GEN)
        self.assertEqual(rows[0]["payload_id"], "payload-id")
        self.assertEqual(self.authority.closed, [pair_key(self.manifest)])
        self.assertEqual(self.cloud.index_creates, 1)
        self.assertEqual(self.cloud.index[self.manifest["index_object"]][1],
                         json.dumps(rows[0], sort_keys=True, separators=(",", ":")).encode())
        with self.copy() as again:
            self.assertEqual(again.run_one()["status"], FINAL_STATUS)
        self.assertEqual(self.drive.media_creates, 2)

    def test_capture_restart_floor_checked_before_each_download(self):
        from gcs_drive_retained_overflow import CAPTURE_RESTART_FLOOR_BYTES
        with self.copy() as copier:
            copier.min_capture_free_bytes = CAPTURE_RESTART_FLOOR_BYTES
            with mock.patch("gcs_drive_retained_overflow.shutil.disk_usage") as disk:
                disk.return_value.free = CAPTURE_RESTART_FLOOR_BYTES + 9
                with self.assertRaisesRegex(Conflict, "capture restart headroom"):
                    copier.space(10)
                disk.return_value.free = CAPTURE_RESTART_FLOOR_BYTES + 10
                with mock.patch.object(type(copier).__mro__[1], "space"):
                    copier.space(10)
        self.assertEqual(self.drive.media_creates, 0)

    def test_lost_folder_reply_keeps_pinned_id_and_forbids_repost(self):
        self.drive.lose_folder_response = True
        with self.assertRaises(TimeoutError):
            with self.copy() as copier:
                copier.run_one()
        state = json.loads((self.work / "journal.json").read_text())["segments"][pair_key(self.manifest)]
        self.assertEqual(state["status"], "folder_create_inflight")
        self.assertEqual(state["drive"]["folder"]["id"], "folder-id")
        self.drive.lose_folder_response = False
        with self.assertRaisesRegex(Conflict, "needs charge review"):
            with self.copy() as copier:
                copier.run_one()
        self.assertEqual(self.drive.folder_posts, 1)
        self.assertEqual(self.drive.media_creates, 0)

    def test_manifest_rejects_old_shard_and_mutated_identity(self):
        for mutation in (lambda m: m.update(drive_parent={"id": "old-shard"}),
                         lambda m: m["payload"].update(generation=0),
                         lambda m: m.update(folder_name="old-archive-shard"),
                         lambda m: m.update(retain_gcs=False)):
            changed = json.loads(json.dumps(self.manifest))
            mutation(changed)
            self.path.write_text(json.dumps(changed))
            with self.assertRaises(Conflict):
                load_overflow_manifest(self.path, release_id=RELEASE, account_id=ACCOUNT)

    def test_lost_media_reply_blocks_second_send_and_keeps_copies(self):
        self.drive.lose_media_response = True
        with self.assertRaises(TimeoutError):
            with self.copy() as copier:
                copier.run_one()
        self.assertEqual(self.drive.media_creates, 1)
        state = json.loads((self.work / "journal.json").read_text())["segments"][pair_key(self.manifest)]
        self.assertEqual(state["status"], "file_upload_inflight")
        self.drive.lose_media_response = False
        with self.assertRaisesRegex(Conflict, "no second media send"):
            with self.copy() as copier:
                copier.run_one()
        self.assertEqual(self.drive.media_creates, 1)
        self.assertEqual(self.cloud.deleted, [])
        self.assertTrue((self.work / "stage" / pair_key(self.manifest) / NAME).exists())

    def test_historical_delete_paths_are_closed(self):
        with self.copy() as copier:
            with self.assertRaises(Conflict):
                copier.one(None)
            with self.assertRaises(Conflict):
                copier.finish({})
            with self.assertRaises(Conflict):
                copier.clean_verified_temp({})
            with self.assertRaises(Conflict):
                copier.recover_completed_cleanup({})
            with self.assertRaises(Conflict):
                copier.pilot_one(None)
            with self.assertRaises(Conflict):
                copier.run_pilot()
            with self.assertRaises(Conflict):
                copier.run(8)
        self.assertEqual(self.cloud.deleted, [])

    def test_duplicate_and_nonfinite_receipt_json_are_rejected(self):
        variants = (
            b'{"schema":"catbird.jetstream.archive.v1",' + self.receipt_bytes[1:],
            self.receipt_bytes.replace(b'"index": 1', b'"index": NaN'),
        )
        for raw in variants:
            with self.subTest(raw=raw[:60]):
                self.manifest["receipt"]["bytes"] = len(raw)
                self.manifest["receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
                self.manifest["max_charged_bytes"] = (
                    len(PAYLOAD_BYTES) + len(raw) + CONTROL_ALLOWANCE)
                self.path.write_text(json.dumps(self.manifest, sort_keys=True))
                loaded, digest = load_overflow_manifest(
                    self.path, release_id=RELEASE, account_id=ACCOUNT)
                cloud = FakeCloud(loaded, raw)
                authority = FakeAuthority(digest)
                with tempfile.TemporaryDirectory() as directory:
                    with RetainedOverflow(cloud, self.drive, authority,
                                          Path(directory), loaded, digest,
                                          self.path.read_bytes(),
                                          capture_segments_dir=Path(directory),
                                          min_capture_free_bytes=1) as copier:
                        with self.assertRaises(Conflict):
                            copier.run_one()
                self.assertEqual(self.drive.media_creates, 0)

    def test_reviewed_raw_bytes_and_later_manifest_mutation_are_rejected(self):
        changed = json.loads(json.dumps(self.manifest))
        changed["payload"]["bytes"] += 1
        with self.assertRaisesRegex(Conflict, "in-memory manifest differs"):
            RetainedOverflow(self.cloud, self.drive, self.authority, self.work,
                             changed, self.digest, self.path.read_bytes())
        with self.copy() as copier:
            self.manifest["payload"]["bytes"] += 1
            self.assertEqual(copier.pilot_manifest["payload"]["bytes"], len(PAYLOAD_BYTES))
            copier.pilot_manifest["payload"]["bytes"] += 1
            with self.assertRaisesRegex(Conflict, "changed after construction"):
                copier.run_one()
        self.assertEqual(self.drive.media_creates, 0)

    def test_wrong_gcs_bucket_rejected_before_copy(self):
        self.cloud.bucket_name = "other-synthetic-bucket"
        with self.assertRaisesRegex(Conflict, "client bucket differs"):
            self.copy()
        self.assertEqual(self.drive.media_creates, 0)

    def test_crash_after_verification_mark_replays_without_drive_send(self):
        self.authority.fail_mark_after_commit = True
        with self.assertRaises(TimeoutError):
            with self.copy() as copier:
                copier.run_one()
        self.assertEqual(self.authority.status, "verified")
        self.authority.fail_mark_after_commit = False
        with self.copy() as copier:
            self.assertEqual(copier.run_one()["status"], FINAL_STATUS)
        self.assertEqual(self.drive.media_creates, 2)
        self.assertEqual(self.cloud.index_creates, 1)

    def test_crash_before_verification_mark_rechecks_pinned_ids_no_send(self):
        self.authority.fail_mark_before_commit = True
        with self.assertRaises(TimeoutError):
            with self.copy() as copier:
                copier.run_one()
        self.assertEqual(self.authority.status, "active")
        self.authority.fail_mark_before_commit = False
        with self.copy() as copier:
            self.assertEqual(copier.run_one()["status"], FINAL_STATUS)
        self.assertEqual(self.drive.media_creates, 2)

    def test_crash_after_close_replays_shared_mapping_without_drive_send(self):
        self.authority.fail_close_after_commit = True
        with self.assertRaises(TimeoutError):
            with self.copy() as copier:
                copier.run_one()
        self.assertEqual(self.authority.status, "index_pending")
        self.authority.fail_close_after_commit = False
        with self.copy() as copier:
            self.assertEqual(copier.run_one()["status"], FINAL_STATUS)
        self.assertEqual(self.drive.media_creates, 2)
        self.assertEqual(self.cloud.index_creates, 1)

    def test_lost_shared_index_reply_reconciles_exact_bytes_no_repost(self):
        self.cloud.lose_index_response = True
        with self.copy() as copier:
            self.assertEqual(copier.run_one()["status"], FINAL_STATUS)
        self.assertEqual(self.cloud.index_creates, 1)
        self.cloud.lose_index_response = False
        with self.copy() as copier:
            self.assertEqual(copier.run_one()["status"], FINAL_STATUS)
        self.assertEqual(self.cloud.index_creates, 1)
        self.assertEqual(self.drive.media_creates, 2)

    def test_index_request_not_dispatched_blocks_repost(self):
        self.cloud.fail_index_before_create = True
        with self.assertRaisesRegex(Conflict, "no repost"):
            with self.copy() as copier:
                copier.run_one()
        self.cloud.fail_index_before_create = False
        with self.assertRaisesRegex(Conflict, "no repost"):
            with self.copy() as copier:
                copier.run_one()
        self.assertEqual(self.cloud.index_creates, 1)
        self.assertEqual(self.drive.media_creates, 2)

    def test_crash_after_index_commit_finalizes_local_cache_without_send(self):
        self.authority.fail_index_commit_after_commit = True
        with self.assertRaises(TimeoutError):
            with self.copy() as copier:
                copier.run_one()
        self.assertEqual(self.authority.status, "closed")
        self.authority.fail_index_commit_after_commit = False
        with self.copy() as copier:
            self.assertEqual(copier.recover_closed_cache(
                self.authority.committed[0][1])["status"], FINAL_STATUS)
        self.assertEqual(self.cloud.index_creates, 1)
        self.assertEqual(self.drive.media_creates, 2)

    def test_shared_index_generation_drift_is_rejected(self):
        with self.copy() as copier:
            copier.run_one()
        name = self.manifest["index_object"]
        self.cloud.index[name] = (3002, self.cloud.index[name][1])
        with self.assertRaisesRegex(Conflict, "not readable"):
            with self.copy() as copier:
                copier.run_one()
        self.assertEqual(self.drive.media_creates, 2)

    def test_preexisting_different_shared_mapping_is_not_reused(self):
        self.cloud.index[self.manifest["index_object"]] = (99, b"other bytes")
        with self.assertRaisesRegex(Conflict, "index exists"):
            with self.copy() as copier:
                copier.run_one()
        self.assertEqual(self.drive.media_creates, 2)
        self.assertEqual(self.authority.status, "index_pending")

    def test_torn_local_cache_append_blocks_replay(self):
        with self.copy() as copier:
            copier.run_one()
        (self.work / "retained-index.jsonl").write_bytes(b'{"schema":"' + b"x" * 10)
        with self.assertRaisesRegex(Conflict, "torn final append"):
            with self.copy() as copier:
                copier.run_one()

    def test_cli_manifest_check_is_offline(self):
        result = subprocess.run([sys.executable, "-m", "gcs_drive_retained_overflow",
                                 "--overflow-manifest", str(self.path), "--run-id", RELEASE,
                                 "--account-id", ACCOUNT, "--check-manifest"],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["event"], "overflow_manifest_checked")
        self.assertEqual(self.drive.media_creates, 0)

    def test_new_receipt_generation_same_payload_has_same_pair_key(self):
        changed = json.loads(json.dumps(self.manifest))
        changed["receipt"]["generation"] += 1
        self.assertEqual(pair_key(changed), pair_key(self.manifest))
        self.assertEqual(changed["index_object"], self.manifest["index_object"])

    def test_completed_diagnostic_payload_cannot_be_recopied(self):
        changed = json.loads(json.dumps(self.manifest))
        changed["payload"] = dict(COMPLETED_DIAGNOSTIC_PAYLOAD)
        with self.assertRaisesRegex(Conflict, "excluded from retained recopy"):
            from gcs_drive_retained_overflow import validate_overflow_manifest
            validate_overflow_manifest(changed, release_id=RELEASE, account_id=ACCOUNT)

    def test_hidden_prior_shared_index_generation_blocks_create(self):
        class Prior:
            name = self.manifest["index_object"]
        class Client:
            def list_blobs(self, _bucket, prefix, **options):
                assert prefix == Prior.name
                return [Prior()] if options.get("soft_deleted") else []
        class Bucket:
            def blob(self, _name):
                raise AssertionError("create must not run")
        cloud = object.__new__(OverflowGCS)
        cloud.client, cloud.bucket = Client(), Bucket()
        data = b"synthetic-map"
        with self.assertRaisesRegex(Conflict, "prior generation"):
            cloud.publish_index(Prior.name, data, hashlib.sha256(data).hexdigest())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux worker guard smoke")
    def test_linux_watchdog_kills_worker_and_grandchild(self):
        marker = Path(self.temp.name) / "grandchild.pid"
        def hung_worker():
            child = os.fork()
            if child == 0:
                marker.write_text(str(os.getpid()))
                while True:
                    time.sleep(60)
            while not marker.exists():
                time.sleep(0.01)
            while True:
                time.sleep(60)
        code = _supervise_worker(hung_worker, time.monotonic() + 0.5)
        self.assertEqual(code, 124)
        self.assertTrue(marker.exists())
        pid = int(marker.read_text())
        for _ in range(50):
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            self.fail("overflow watchdog left grandchild running")


if __name__ == "__main__":
    unittest.main()
