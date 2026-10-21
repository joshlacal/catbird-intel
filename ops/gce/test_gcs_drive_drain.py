"""Structural offline tests: no Google package, credentials, or network required."""
import hashlib
import base64
import email.message
import json
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import urllib.response

from gcs_drive_drain import (Conflict, Drain, FOLDER, ObjectRef, SafeDrive,
                             _NO_REDIRECT_OPENER, _NoRedirect, atomic_json,
                             digest, load_pilot_manifest, main, matches_drive)


class Cloud:
    bucket_name = "fixture-bucket"
    def __init__(self):
        self.objects = {}
        self.deleted = []
        self.downloaded = []
        self.fail_delete = None
    def add(self, name, data, generation, metadata):
        ref = ObjectRef(name, generation, len(data), metadata)
        self.objects[(name, generation)] = (ref, data)
        return ref
    def lookup(self, name, generation=None):
        candidates = [pair[0] for (key, gen), pair in self.objects.items() if key == name and (generation is None or gen == generation)]
        return max(candidates, key=lambda r: r.generation) if candidates else None
    def receipts(self, prefix):
        return [ref for ref, _ in self.objects.values() if ref.name.startswith(prefix) and ref.metadata.get("kind") == "receipt"]
    def download(self, ref, path):
        self.downloaded.append((ref.name, ref.generation))
        path.write_bytes(self.objects[(ref.name, ref.generation)][1])
    def delete(self, ref):
        if self.fail_delete == ref.name:
            self.fail_delete = None
            raise OSError("injected deletion interruption")
        self.deleted.append((ref.name, ref.generation))
        self.objects.pop((ref.name, ref.generation), None)


class Drive:
    def __init__(self):
        self.files = {}
        self.uploads = []
        self.failure = None
    def folder(self, root, name, state, save):
        state["id"] = "folder-" + name
        save()
        return state["id"]
    def preflight(self, parent, name, expected):
        found = [f for f in self.files.values() if f["name"] == name and parent in f["parents"]]
        if len(found) > 1 or (found and not matches_drive(found[0], name, expected, parent)):
            raise Conflict("Drive conflicting identity")
        return found[0] if found else None
    def ensure_file(self, path, parent, expected, state, save):
        existing = self.preflight(parent, path.name, expected)
        if existing:
            return existing
        if self.failure:
            raise self.failure
        file_id = str(len(self.files)+1)
        info = {"id": file_id, "name": path.name, "size": str(expected["size"]),
                "md5Checksum": expected["md5"], "parents": [parent], "trashed": False}
        self.files[file_id] = info
        self.uploads.append(path.name)
        return info
    def metadata(self, file_id):
        return dict(self.files[file_id])


class PilotDrive(Drive):
    def __init__(self):
        super().__init__()
        self.account_id = "synthetic-account"
        self.preallocated = []
        self.parent = {"id": "folder-00000001", "name": "00000001", "mimeType": FOLDER,
                       "parents": ["drive-root"], "trashed": False}
    def verify_account(self, account_id):
        if account_id != self.account_id:
            raise Conflict("Drive OAuth account identity is unavailable or mismatched")
    def assert_complete_visibility(self):
        # Explicit synthetic fixture assumption; the real drive.file client lacks proof.
        return None
    def folder(self, *args):
        raise AssertionError("pilot attempted to create a folder")
    def named(self, root, name):
        return [dict(self.parent)] if root == "drive-root" and name == "00000001" else []
    def metadata(self, file_id):
        return dict(self.parent) if file_id == self.parent["id"] else super().metadata(file_id)
    def reserve_id(self):
        file_id = "synthetic-generated-" + str(len(self.preallocated) + 1)
        self.preallocated.append(file_id)
        return file_id
    def ensure_file(self, path, parent, expected, state, save):
        entry = next(iter(self.drain.state["segments"].values()))
        if not all(entry["drive"][kind].get("id") for kind in ("payload", "receipt")):
            raise AssertionError("both Drive IDs were not pinned before upload")
        existing = self.preflight(parent, path.name, expected)
        if existing:
            return existing
        if self.failure:
            raise self.failure
        if "id" not in state:
            raise AssertionError("pilot file ID was not journaled before upload")
        file_id = state["id"]
        info = {"id": file_id, "name": path.name, "size": str(expected["size"]),
                "md5Checksum": expected["md5"], "parents": [parent], "trashed": False}
        self.files[file_id] = info
        self.uploads.append(path.name)
        return info


class PilotAuthority:
    def __init__(self):
        self.account_id = "synthetic-account"
        self.reservations = []
        self.closed = []
    def assert_active(self, timeout_s):
        if timeout_s > 300:
            raise AssertionError("unexpected request timeout")
    def reserve_pair(self, *args):
        self.reservations.append(args)
    def close_pair(self, key):
        self.closed.append(key)


class DrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.cloud, self.drive = Cloud(), Drive()
        self.data = b"\x00\x01SEALED_STRUCTURAL_BYTES\xff"
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.name = "seg_0000000074.jss"  # base36 256 -> Drive shard 00000001
        self.payload_name = f"sources/source/epoch/segments/00000001/{self.name}/{self.sha}.jss"
        self.payload = self.cloud.add(self.payload_name, self.data, 11, {"kind": "payload", "sha256": self.sha})
        self.body = {"schema": "catbird.jetstream.archive.v1", "name": self.name, "index": 256,
            "sha256": self.sha, "bytes": len(self.data), "source_id": "source", "source_epoch": "epoch",
            "gcs": {"bucket": self.cloud.bucket_name, "object": self.payload_name, "generation": 11,
                    "bytes": len(self.data), "sha256": self.sha,
                    "md5_hash": base64.b64encode(hashlib.md5(self.data).digest()).decode()}}
        self.make_receipt()
        self.drain = self.new_drain()
    def make_receipt(self):
        raw = json.dumps(self.body).encode()
        self.receipt = self.cloud.add(self.payload_name + ".g11.receipt.json", raw, 12,
            {"kind": "receipt", "sha256": hashlib.sha256(raw).hexdigest(), "segment_name": self.name,
             "payload_object": self.payload_name, "payload_generation": "11"})
    def new_drain(self, **options):
        return Drain(self.cloud, self.drive, self.work, "drive-root", "source", "epoch", min_free_bytes=0, **options)
    def release(self):
        body = {"schema": "catbird.gcs.local-release.v1", "payload_object": self.payload_name,
                "payload_generation": 11, "receipt_object": self.receipt.name, "receipt_generation": 12, "sha256": self.sha}
        data = json.dumps(body).encode()
        return self.cloud.add(self.receipt.name + ".local-release.json", data, 13,
                              {"kind": "local_release", "sha256": hashlib.sha256(data).hexdigest()})
    def pilot_manifest(self):
        return {"schema": "catbird.drive.pilot.v1", "release_id": "synthetic-release",
                "bucket": self.cloud.bucket_name, "source_id": "source", "source_epoch": "epoch",
                "drive_root": "drive-root", "drive_parent": {"id": "folder-00000001", "shard": "00000001"},
                "payload": {"object": self.payload_name, "generation": 11,
                            "bytes": len(self.data), "sha256": self.sha},
                "receipt": {"object": self.receipt.name, "generation": 12,
                            "bytes": self.receipt.size, "sha256": self.receipt.metadata["sha256"]},
                "control_allowance_bytes": 1024, "max_charged_bytes": 320 << 20,
                "max_duration_seconds": 1200, "retain_gcs": True}
    def load_pilot(self, manifest=None):
        path = self.work/"pilot-manifest.json"
        path.write_text(json.dumps(manifest or self.pilot_manifest()))
        return load_pilot_manifest(path, bucket=self.cloud.bucket_name, source_id="source",
                                   source_epoch="epoch", drive_root="drive-root",
                                   release_id="synthetic-release")
    def new_pilot_drain(self, manifest=None):
        self.drive = PilotDrive()
        self.authority = PilotAuthority()
        manifest, manifest_sha = self.load_pilot(manifest)
        drain = Drain(self.cloud, self.drive, self.work, "drive-root", "source", "epoch",
                      min_free_bytes=0, pilot_manifest=manifest,
                      manifest_sha256=manifest_sha, authority=self.authority)
        self.drive.drain = drain
        return drain
    def test_pilot_copies_exact_pair_without_any_release_or_cleanup(self):
        marker = self.release()
        drain = self.new_pilot_drain()
        with patch.object(self.cloud, "receipts", side_effect=AssertionError("broad listing")):
            self.assertEqual(drain.run_pilot(), "drive_verified_retained")
        self.assertEqual(self.drive.uploads, [self.name, self.name + ".receipt.json"])
        self.assertEqual(self.cloud.deleted, [])
        self.assertIsNotNone(self.cloud.lookup(self.payload_name, 11))
        self.assertIsNotNone(self.cloud.lookup(self.receipt.name, 12))
        self.assertIsNotNone(self.cloud.lookup(marker.name, 13))
        self.assertEqual(len(list((self.work/"stage").rglob("*.jss"))), 1)
        self.assertEqual(len(list((self.work/"stage").rglob("*.receipt.json"))), 1)
        self.assertEqual(len(self.authority.reservations), 1)
        self.assertEqual(len(self.authority.closed), 1)
        self.assertEqual(len(self.drive.preallocated), 2)
        entry = next(iter(drain.state["segments"].values()))
        self.assertEqual(entry["status"], "drive_verified_retained")
        self.assertEqual(entry["pilot_manifest_sha256"], self.load_pilot()[1])
        with self.assertRaisesRegex(Conflict, "already verified"):
            drain.run_pilot()
    def test_pilot_rejects_unpinned_pair_without_any_drive_upload(self):
        manifest = self.pilot_manifest()
        manifest["receipt"]["generation"] = 99
        drain = self.new_pilot_drain(manifest)
        with self.assertRaisesRegex(Conflict, "generation is missing"):
            drain.run_pilot()
        self.assertEqual(self.drive.uploads, [])
        self.assertEqual(self.cloud.deleted, [])
    def test_pilot_rejects_missing_or_changed_parent(self):
        drain = self.new_pilot_drain()
        self.drive.parent["name"] = "wrong-shard"
        with self.assertRaisesRegex(Conflict, "uniquely visible"):
            drain.run_pilot()
        self.assertEqual(self.drive.uploads, [])
        self.assertEqual(self.cloud.deleted, [])
    def test_pilot_rejects_oauth_account_mismatch(self):
        drain = self.new_pilot_drain()
        self.drive.account_id = "other-account"
        with self.assertRaisesRegex(Conflict, "account identity"):
            drain.run_pilot()
        self.assertEqual(self.drive.uploads, [])
        self.assertEqual(self.cloud.deleted, [])
    def test_pilot_rejects_partial_drive_file_visibility_before_reservation(self):
        drain = self.new_pilot_drain()
        self.drive.hidden_file = {"id": "unseen-existing-file", "name": self.name,
                                  "md5Checksum": "different", "parents": [self.drive.parent["id"]]}
        self.assertIsNone(self.drive.preflight(self.drive.parent["id"], self.name,
                                               {"size": len(self.data), "md5": hashlib.md5(self.data).hexdigest()}))
        self.drive.assert_complete_visibility = SafeDrive.assert_complete_visibility
        with self.assertRaisesRegex(Conflict, "uniqueness is unproven"):
            drain.run_pilot()
        self.assertEqual(self.drive.uploads, [])
        self.assertEqual(self.authority.reservations, [])
        self.assertEqual(self.cloud.deleted, [])
    def test_pilot_http_failure_preserves_pair_and_staging(self):
        drain = self.new_pilot_drain()
        error = urllib.error.HTTPError("synthetic", 403, "quota", {}, None)
        self.addCleanup(error.close)
        self.drive.failure = error
        with self.assertRaises(urllib.error.HTTPError):
            drain.run_pilot()
        self.assertEqual(self.cloud.deleted, [])
        self.assertTrue(list((self.work/"stage").rglob("*.jss")))
        self.assertEqual(self.authority.closed, [])
    def test_pilot_rechecks_deadline_before_payload_download(self):
        drain = self.new_pilot_drain()
        original_assert = self.authority.assert_active
        checks = []
        def expire_before_payload(timeout_s):
            checks.append(timeout_s)
            if checks.count(300) == 3:
                raise Conflict("synthetic deadline expired")
            original_assert(timeout_s)
        self.authority.assert_active = expire_before_payload
        with self.assertRaisesRegex(Conflict, "deadline expired"):
            drain.run_pilot()
        self.assertEqual(self.cloud.downloaded, [(self.receipt.name, self.receipt.generation)])
        self.assertEqual(self.cloud.deleted, [])
        self.assertEqual(self.drive.uploads, [])
    def test_pilot_manifest_enforces_caps_and_retention(self):
        for change in ({"retain_gcs": False}, {"max_charged_bytes": (320 << 20)+1},
                       {"max_duration_seconds": 1201}, {"control_allowance_bytes": 0},
                       {"control_allowance_bytes": (1 << 20) + 1}):
            manifest = self.pilot_manifest()
            manifest.update(change)
            with self.subTest(change=change), self.assertRaises(Conflict):
                self.load_pilot(manifest)
    def test_pilot_refuses_broad_run_and_missing_authority(self):
        manifest, sha = self.load_pilot()
        with self.assertRaisesRegex(Conflict, "requires a pinned manifest"):
            Drain(self.cloud, self.drive, self.work, "drive-root", "source", "epoch",
                  pilot_manifest=manifest, manifest_sha256=sha)
        drain = self.new_pilot_drain()
        with self.assertRaisesRegex(Conflict, "exact pinned one-pair"):
            drain.run(1)
    def test_upload_waits_for_local_release_without_removing_cloud(self):
        self.assertEqual(self.drain.one(self.receipt), "awaiting_local_release")
        self.assertEqual(self.cloud.deleted, [])
        self.assertEqual(len(self.drive.uploads), 2)
        self.assertEqual(list((self.work/"stage").rglob("*.jss")), [])
    def test_release_uploads_verifies_deletes_exact_generations(self):
        marker = self.release()
        self.assertEqual(self.drain.one(self.receipt), "complete")
        self.assertEqual(self.cloud.deleted, [(self.payload_name, 11), (self.receipt.name, 12), (marker.name, 13)])
        self.assertEqual(len(self.drive.uploads), 2)
        self.assertEqual(json.loads((self.work/"journal.json").read_text())["segments"].popitem()[1]["status"], "complete")
    def test_restart_after_waiting_uses_drive_without_payload_redownload(self):
        self.drain.one(self.receipt)
        self.cloud.downloaded.clear()
        self.release()
        self.assertEqual(self.new_drain().run(1), 1)
        self.assertEqual(len(self.drive.uploads), 2)
        self.assertTrue(all(name.endswith(".local-release.json") for name, _ in self.cloud.downloaded))
    def test_failure_between_deletes_recovers_without_losing_pair(self):
        self.release()
        self.cloud.fail_delete = self.receipt.name
        with self.assertRaises(OSError):
            self.drain.one(self.receipt)
        self.assertIsNone(self.cloud.lookup(self.payload_name, 11))
        self.assertIsNotNone(self.cloud.lookup(self.receipt.name, 12))
        self.new_drain().run(1)
        self.assertIsNone(self.cloud.lookup(self.receipt.name, 12))
    def test_newer_payload_generation_is_never_deleted(self):
        self.release()
        self.cloud.add(self.payload_name, b"new bytes", 99, {"sha256": "different"})
        self.drain.one(self.receipt)
        self.assertIsNotNone(self.cloud.lookup(self.payload_name, 99))
    def test_receipt_pointer_or_namespace_mismatch_refused(self):
        self.body["source_epoch"] = "other"
        self.make_receipt()
        with self.assertRaises(Conflict):
            self.drain.one(self.receipt)
        self.assertEqual(self.drive.uploads, [])
        self.assertEqual(self.cloud.deleted, [])
    def test_bad_payload_bytes_preserved_and_no_upload(self):
        self.cloud.objects[(self.payload_name, 11)] = (self.payload, b"wrong")
        with self.assertRaises(Conflict):
            self.drain.one(self.receipt)
        self.assertTrue(list((self.work/"stage").rglob("*.partial")))
        self.assertEqual(self.drive.uploads, [])
        self.assertEqual(self.cloud.deleted, [])
    def test_bad_receipt_metadata_digest_refused(self):
        bad = ObjectRef(self.receipt.name, 12, self.receipt.size, {**self.receipt.metadata, "sha256": "0"*64})
        with self.assertRaises(Conflict):
            self.drain.one(bad)
        self.assertEqual(self.cloud.deleted, [])
    def test_receipt_conflict_refuses_payload_upload_too(self):
        self.drive.files["bad"] = {"id": "bad", "name": self.name+".receipt.json", "size": "9",
                                  "md5Checksum": "wrong", "parents": ["folder-00000001"]}
        with self.assertRaises(Conflict):
            self.drain.one(self.receipt)
        self.assertEqual(self.drive.uploads, [])
        self.assertEqual(self.cloud.deleted, [])
        self.assertNotIn((self.payload_name, 11), self.cloud.downloaded)
    def test_duplicate_even_identical_refused(self):
        self.drain.one(self.receipt)
        self.drive.files["duplicate"] = {**next(iter(self.drive.files.values())), "id": "duplicate"}
        self.release()
        resumed = self.new_drain()
        resumed.run(1)
        self.assertEqual(resumed.last_counts["conflict"], 1)
        self.assertEqual(self.cloud.deleted, [])
    def test_403_no_loop_preserves_local_cloud_and_journal(self):
        self.drive.failure = urllib.error.HTTPError("fixture", 403, "quota", {}, None)
        self.addCleanup(self.drive.failure.close)
        with self.assertRaises(urllib.error.HTTPError):
            self.drain.run(8)
        self.assertEqual(self.cloud.deleted, [])
        self.assertTrue(list((self.work/"stage").rglob("*.jss")))
        self.assertTrue((self.work/"journal.json").exists())
    def test_capacity_bound_counts_retained_artifacts(self):
        drain = self.new_drain(max_temp_bytes=1)
        with self.assertRaises(Conflict):
            drain.one(self.receipt)
        self.assertEqual(self.cloud.downloaded, [])
    def test_segment_size_bound_preserves_receipt(self):
        with self.assertRaises(Conflict):
            self.new_drain(max_segment_bytes=1).one(self.receipt)
        self.assertEqual(self.drive.uploads, [])
    def test_journal_namespace_cannot_be_reused(self):
        self.drain.save()
        with self.assertRaises(Conflict):
            Drain(self.cloud, self.drive, self.work, "wrong-root", "source", "epoch")
    def test_maximum_validation(self):
        for maximum in (0, 1025):
            with self.assertRaises(ValueError):
                self.drain.run(maximum)
    def test_drive_named_follows_pagination_and_refuses_duplicates(self):
        drive = object.__new__(SafeDrive)
        calls = []
        def page(url):
            calls.append(url)
            return {"files": [{"id": str(len(calls))}], **({"nextPageToken": "p2"} if len(calls)==1 else {})}
        drive.json_request = page
        with self.assertRaises(Conflict):
            drive.named("root", "name")
        self.assertIn("pageToken=p2", calls[1])
    def test_drive_account_readback_requires_exact_permission_id(self):
        drive = object.__new__(SafeDrive)
        drive.json_request = lambda url: {"user": {"permissionId": "synthetic-account"}}
        drive.verify_account("synthetic-account")
        with self.assertRaisesRegex(Conflict, "account identity"):
            drive.verify_account("other-account")
        drive.json_request = lambda url: {"user": {}}
        with self.assertRaisesRegex(Conflict, "account identity"):
            drive.verify_account("synthetic-account")
    def test_drive_request_charges_before_send_and_keeps_unknown_attempt(self):
        drive = object.__new__(SafeDrive)
        events = []
        class Authority:
            def clock(self):
                return time.time()
            def assert_active(self, timeout, *, slot_at=None):
                events.append(("active", slot_at))
            def start_request(self, body_bytes, timeout, kind, **identity):
                events.append(("start", body_bytes, kind, identity))
                return "synthetic-attempt", time.time()
            def mark_dispatch(self, attempt_id):
                events.append(("dispatch", attempt_id))
            def finish_request(self, attempt_id, status):
                events.append(("finish", status))
        drive.authority = Authority()
        class Auth:
            def token(self):
                events.append(("token",))
                return "synthetic-token"
        drive.auth = Auth()
        def lost_after_send(request, timeout=60):
            events.append(("urlopen",))
            raise urllib.error.URLError("lost")
        with patch.object(_NO_REDIRECT_OPENER, "open", side_effect=lost_after_send):
            with self.assertRaises(urllib.error.URLError):
                drive.request("https://www.googleapis.com/drive/v3/files", "POST", b"abc", kind="control")
        self.assertEqual(events[0], ("active", None))
        self.assertEqual(events[2][:3], ("start", 3, "control"))
        self.assertEqual(events[2][3]["method"], "POST")
        self.assertEqual(events[2][3]["target_category"], "control")
        self.assertRegex(events[2][3]["target_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(events[2][3]["operation_key"], r"^[0-9a-f]{64}$")
        self.assertLess(events.index(("dispatch", "synthetic-attempt")), events.index(("urlopen",)))
        self.assertNotIn(("finish", 200), events)
        self.assertFalse(any(event[0] == "finish" for event in events))
    def test_retained_dispatch_wait_uses_one_precharged_attempt(self):
        drive = object.__new__(SafeDrive)
        events = []
        class Authority:
            def clock(self):
                return time.time()
            def assert_active(self, timeout, *, slot_at=None):
                events.append(("active", slot_at))
            def start_request(self, body_bytes, timeout, kind, **identity):
                events.append(("start", body_bytes))
                return "one-attempt", time.time()
            def seconds_until_dispatch(self, attempt_id):
                events.append(("wait", attempt_id))
                return 0.45 if events.count(("wait", attempt_id)) == 1 else 0
            def mark_dispatch(self, attempt_id):
                events.append(("dispatch", attempt_id))
            def finish_request(self, attempt_id, status):
                events.append(("finish", status))
        drive.authority = Authority()
        drive.auth = type("Auth", (), {"token": lambda self: "synthetic-token"})()
        class Response(io.BytesIO):
            status = 200
        def fake_open(request, timeout=60):
            events.append(("urlopen",))
            return Response(b"{}")
        with patch("gcs_drive_drain.time.sleep", side_effect=lambda seconds: events.append(("sleep", seconds))), \
             patch.object(_NO_REDIRECT_OPENER, "open", side_effect=fake_open):
            with drive.request("https://www.googleapis.com/drive/v3/files"):
                pass
        self.assertEqual([event[0] for event in events].count("start"), 1)
        self.assertEqual([event[0] for event in events].count("dispatch"), 1)
        self.assertEqual([event[0] for event in events].count("urlopen"), 1)
        self.assertLess(events.index(("wait", "one-attempt")),
                        events.index(("dispatch", "one-attempt")))
        self.assertLess(events.index(("dispatch", "one-attempt")),
                        events.index(("urlopen",)))
        self.assertIn(("sleep", 0.451), events)
    def test_retained_dispatch_deadline_refusal_never_sends(self):
        drive = object.__new__(SafeDrive)
        events = []
        class Authority:
            def clock(self):
                return time.time()
            def assert_active(self, timeout, *, slot_at=None):
                return None
            def start_request(self, body_bytes, timeout, kind, **identity):
                events.append("start")
                return "one-attempt", time.time()
            def seconds_until_dispatch(self, attempt_id):
                events.append("wait")
                raise RuntimeError("deadline before pacing floor")
            def mark_dispatch(self, attempt_id):
                events.append("dispatch")
        drive.authority = Authority()
        drive.auth = type("Auth", (), {"token": lambda self: "synthetic-token"})()
        with patch.object(_NO_REDIRECT_OPENER, "open", side_effect=AssertionError("unexpected send")):
            with self.assertRaisesRegex(RuntimeError, "deadline before pacing floor"):
                drive.request("https://www.googleapis.com/drive/v3/files")
        self.assertEqual(events, ["start", "wait"])
    def test_same_endpoint_upload_starts_have_distinct_hashed_body_identity(self):
        drive = object.__new__(SafeDrive)
        identities = []
        class Authority:
            def clock(self):
                return time.time()
            def assert_active(self, timeout, *, slot_at=None):
                return None
            def start_request(self, body_bytes, timeout, kind, **identity):
                identities.append(identity)
                return str(len(identities)), time.time()
            def mark_dispatch(self, attempt_id):
                return None
            def finish_request(self, attempt_id, status):
                return None
        drive.authority = Authority()
        drive.auth = type("Auth", (), {"token": lambda self: "synthetic-token"})()
        class Response(io.BytesIO):
            status = 200
        endpoint = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
        with patch.object(_NO_REDIRECT_OPENER, "open", side_effect=lambda *args, **kwargs: Response(b"{}")):
            for body in (b'{"id":"first"}', b'{"id":"second"}'):
                with drive.request(endpoint, "POST", body, kind="resumable_start"):
                    pass
        self.assertEqual(len(identities), 2)
        self.assertEqual(identities[0]["target_sha256"], identities[1]["target_sha256"])
        self.assertNotEqual(identities[0]["operation_key"], identities[1]["operation_key"])
        self.assertNotIn("first", str(identities))
        self.assertNotIn("second", str(identities))
    def test_redirect_never_dispatches_a_second_unmetered_request(self):
        drive = object.__new__(SafeDrive)
        events = []
        class Authority:
            def clock(self):
                return time.time()
            def assert_active(self, timeout, *, slot_at=None):
                return None
            def start_request(self, body_bytes, timeout, kind, **identity):
                events.append(("reserved", identity["target_sha256"]))
                return "one-attempt", time.time()
            def mark_dispatch(self, attempt_id):
                events.append(("dispatched", attempt_id))
            def finish_request(self, attempt_id, status):
                events.append(("finished", status))
        drive.authority = Authority()
        drive.auth = type("Auth", (), {"token": lambda self: "synthetic-token"})()
        class RedirectingHTTPS(urllib.request.HTTPSHandler):
            def __init__(self):
                super().__init__()
                self.urls = []
            def https_open(self, request):
                self.urls.append(request.full_url)
                headers = email.message.Message()
                headers["Location"] = "https://www.googleapis.com/drive/v3/other"
                response = urllib.response.addinfourl(io.BytesIO(b""), headers,
                                                        request.full_url, code=302)
                response.msg = "Found"
                return response
        handler = RedirectingHTTPS()
        opener = urllib.request.build_opener(_NoRedirect(), handler)
        with patch("gcs_drive_drain._NO_REDIRECT_OPENER", opener):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                drive.request("https://www.googleapis.com/drive/v3/files")
        self.addCleanup(raised.exception.close)
        self.assertEqual(raised.exception.code, 302)
        self.assertEqual(handler.urls, ["https://www.googleapis.com/drive/v3/files"])
        self.assertEqual([event[0] for event in events],
                         ["reserved", "dispatched", "finished"])
        self.assertEqual(events[-1], ("finished", 302))
    def test_drive_request_rejects_non_google_host_before_token_or_reservation(self):
        drive = object.__new__(SafeDrive)
        drive.auth = type("Auth", (), {"token": lambda self: self.fail("token accessed")})()
        drive.authority = type("Authority", (), {"assert_active": lambda self, *args: self.fail("authority accessed")})()
        with self.assertRaisesRegex(Conflict, "approved Google API host"):
            drive.request("https://example.invalid/redirect")
    def test_drive_request_serializes_actual_dispatch(self):
        drive = object.__new__(SafeDrive)
        entered = threading.Event()
        release = threading.Event()
        started = []
        class Authority:
            def clock(self):
                return time.time()
            def assert_active(self, timeout, *, slot_at=None):
                return None
            def start_request(self, body_bytes, timeout, kind, **identity):
                started.append(time.time())
                return str(len(started)), time.time()
            def mark_dispatch(self, attempt_id):
                return None
            def finish_request(self, attempt_id, status):
                return None
        drive.authority = Authority()
        drive.auth = type("Auth", (), {"token": lambda self: "synthetic-token"})()
        class Response(io.BytesIO):
            status = 200
        def fake_urlopen(request, timeout=60):
            if len(started) == 1:
                entered.set()
                if not release.wait(1):
                    raise AssertionError("test dispatch did not release")
            return Response(b"{}")
        errors = []
        def call():
            try:
                drive.request("https://www.googleapis.com/drive/v3/files")
            except Exception as error:
                errors.append(error)
        with patch.object(_NO_REDIRECT_OPENER, "open", side_effect=fake_urlopen):
            first = threading.Thread(target=call)
            second = threading.Thread(target=call)
            first.start()
            self.assertTrue(entered.wait(1))
            second.start()
            self.assertEqual(len(started), 1)
            release.set()
            first.join(1)
            second.join(1)
        self.assertFalse(errors)
        self.assertEqual(len(started), 2)
    def test_atomic_journal_roundtrip(self):
        path = self.work/"atomic.json"
        atomic_json(path, {"step": 1})
        atomic_json(path, {"step": 2})
        self.assertEqual(json.loads(path.read_text()), {"step": 2})
        self.assertFalse(path.with_name("atomic.json.tmp").exists())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
    def test_historical_timer_arguments_fail_before_cloud_or_oauth_access(self):
        args = ["--bucket", "fixture", "--source-id", "source", "--source-epoch", "epoch",
                "--drive-root", "root", "--work-dir", str(self.work), "--once"]
        with patch("gcs_drive_drain.gce_region") as region, \
             patch("gcs_drive_drain.GCS") as cloud, \
             patch("gcs_drive_drain.SafeDrive") as drive:
            with self.assertRaises(SystemExit):
                main(args)
            region.assert_not_called()
            cloud.assert_not_called()
            drive.assert_not_called()
    def test_cli_blocks_unproven_drive_visibility_before_cloud_access(self):
        self.load_pilot()
        args = ["--bucket", self.cloud.bucket_name, "--source-id", "source",
                "--source-epoch", "epoch", "--drive-root", "drive-root",
                "--work-dir", "/data/synthetic-pilot", "--once", "--retain-gcs",
                "--authority-db", str(self.work/"synthetic-ledger.sqlite"),
                "--release-id", "synthetic-release", "--writer-id", "synthetic-writer",
                "--account-id", "synthetic-account", "--pilot-manifest",
                str(self.work/"pilot-manifest.json")]
        with patch("gcs_drive_drain.TransferAuthority.open_existing") as authority_open, \
             patch("gcs_drive_drain.gce_region") as region, \
             patch("gcs_drive_drain.GCS") as cloud, \
             patch("gcs_drive_drain.SafeDrive.__init__") as oauth, \
             patch("gcs_drive_drain.sys.stderr", io.StringIO()):
            self.assertEqual(main(args), 1)
            authority_open.assert_not_called()
            region.assert_not_called()
            cloud.assert_not_called()
            oauth.assert_not_called()
    def test_drive_mutation_after_upload_prevents_release(self):
        self.drain.one(self.receipt)
        next(iter(self.drive.files.values()))["md5Checksum"] = "different"
        self.release()
        resumed = self.new_drain()
        resumed.run(1)
        self.assertEqual(resumed.last_counts["conflict"], 1)
        self.assertEqual(self.cloud.deleted, [])
    def test_conflicting_legacy_receipt_does_not_block_other_segments(self):
        original = self.receipt
        self.drive.files["old-receipt"] = {"id": "old-receipt", "name": self.name+".receipt.json",
            "size": "1", "md5Checksum": "different", "parents": ["folder-00000001"]}
        self.name = "seg_0000000075.jss"
        self.payload_name = f"sources/source/epoch/segments/00000001/{self.name}/{self.sha}.jss"
        self.cloud.add(self.payload_name, self.data, 11, {"kind": "payload", "sha256": self.sha})
        self.body = {**self.body, "name": self.name, "index": 257,
                     "gcs": {**self.body["gcs"], "object": self.payload_name}}
        self.make_receipt()
        self.drain.run(8)
        self.assertEqual(self.drain.last_counts, {"complete": 0, "awaiting_local_release": 1, "conflict": 1})
        self.assertEqual(len(self.drive.uploads), 2)
        self.assertIsNotNone(self.cloud.lookup(original.name, original.generation))
        resumed = self.new_drain()
        resumed.run(8)
        self.assertEqual(resumed.last_counts["conflict"], 0)
        self.assertEqual(len(self.drive.uploads), 2)
    def test_resumable_upload_uses_saved_id_and_bounded_chunks(self):
        path = self.work/"large.bin"
        with path.open("wb") as fh:
            fh.write(b"\x00" * (9 << 20))
        expected = digest(path)
        drive = object.__new__(SafeDrive)
        drive.preflight = lambda *args: None
        drive.reserve_id = lambda: "reserved"
        final = {"id": "reserved", "name": path.name, "size": str(expected["size"]),
                 "md5Checksum": expected["md5"], "parents": ["parent"]}
        drive.metadata = lambda file_id: final
        state, saved, chunks = {}, [], []
        class Response(io.BytesIO):
            def __init__(self, body=b"", headers=None):
                super().__init__(body)
                self.headers = headers or {}
        def request(url, method="GET", data=None, headers=None, timeout=60, kind="drive_api"):
            if method == "POST":
                self.assertEqual(saved[0]["id"], "reserved")
                return Response(headers={"Location": "https://www.googleapis.com/upload/session"})
            if headers["Content-Range"].startswith("bytes */"):
                self.assertIn("session", saved[-1])
                error = urllib.error.HTTPError(url, 308, "resume", {}, None)
                self.addCleanup(error.close)
                raise error
            chunks.append(len(data))
            if len(chunks) == 1:
                error = urllib.error.HTTPError(url, 308, "resume", {"Range": "bytes=0-8388607"}, None)
                self.addCleanup(error.close)
                raise error
            return Response(json.dumps(final).encode())
        drive.request = request
        result = drive.ensure_file(path, "parent", expected, state, lambda: saved.append(dict(state)))
        self.assertEqual(result, final)
        self.assertEqual(chunks, [8 << 20, 1 << 20])
    def test_expired_sessions_preserve_id_and_session_as_conflicts(self):
        path = self.work/"upload.bin"
        path.write_bytes(b"\x00\x01\x02")
        expected = digest(path)
        for code in (404, 410):
            drive = object.__new__(SafeDrive)
            drive.preflight = lambda *args: None
            error = urllib.error.HTTPError("fixture", code, "expired", {}, None)
            self.addCleanup(error.close)
            state = {"id": "reserved", "session": "https://www.googleapis.com/upload/expired"}
            before = dict(state)
            with patch.object(drive, "request", side_effect=error), \
                 patch.object(drive, "reserve_id") as reserve:
                with self.assertRaisesRegex(Conflict, "preserved ID/session require reconciliation"):
                    drive.ensure_file(path, "parent", expected, state, lambda: None)
                reserve.assert_not_called()
            self.assertEqual(state, before)
            self.assertEqual(digest(path), expected)
    def test_completed_journal_recovers_interrupted_temporary_cleanup(self):
        self.release()
        with patch.object(self.drain, "clean_verified_temp", side_effect=OSError("injected interruption")):
            with self.assertRaises(OSError):
                self.drain.one(self.receipt)
        self.assertTrue(list((self.work/"stage").rglob("*.jss")))
        deleted = list(self.cloud.deleted)
        resumed = self.new_drain()
        self.assertEqual(resumed.run(8), 1)
        self.assertEqual(self.cloud.deleted, deleted)  # Cleanup recovery never repeats cloud mutations.
        self.assertEqual(list((self.work/"stage").rglob("*")), [])
        entry = next(iter(resumed.state["segments"].values()))
        self.assertEqual(entry["status"], "complete")
        self.assertTrue(entry["cleanup_complete"])
    def test_completed_cleanup_preserves_modified_artifacts(self):
        self.release()
        with patch.object(self.drain, "clean_verified_temp", side_effect=OSError("injected interruption")):
            with self.assertRaises(OSError):
                self.drain.one(self.receipt)
        payload = next((self.work/"stage").rglob("*.jss"))
        payload.write_bytes(b"CHANGED_LOCAL_ARTIFACT")
        deleted = list(self.cloud.deleted)
        resumed = self.new_drain()
        resumed.run(8)
        self.assertEqual(resumed.last_counts["conflict"], 1)
        self.assertEqual(payload.read_bytes(), b"CHANGED_LOCAL_ARTIFACT")
        self.assertEqual(len([p for p in payload.parent.iterdir() if p.is_file()]), 3)
        self.assertEqual(self.cloud.deleted, deleted)
    def test_completed_cleanup_refuses_symlinked_artifact(self):
        self.release()
        with patch.object(self.drain, "clean_verified_temp", side_effect=OSError("injected interruption")):
            with self.assertRaises(OSError):
                self.drain.one(self.receipt)
        payload = next((self.work/"stage").rglob("*.jss"))
        preserved = self.work/"preserved-original.jss"
        payload.rename(preserved)
        payload.symlink_to(preserved)
        resumed = self.new_drain()
        resumed.run(8)
        self.assertEqual(resumed.last_counts["conflict"], 1)
        self.assertTrue(payload.is_symlink())
        self.assertEqual(preserved.read_bytes(), self.data)


if __name__ == "__main__":
    unittest.main()
