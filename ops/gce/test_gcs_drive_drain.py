"""Structural offline tests: no Google package, credentials, or network required."""
import hashlib
import base64
import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from gcs_drive_drain import Conflict, Drain, ObjectRef, SafeDrive, atomic_json, digest, main, matches_drive


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
    def test_atomic_journal_roundtrip(self):
        path = self.work/"atomic.json"
        atomic_json(path, {"step": 1})
        atomic_json(path, {"step": 2})
        self.assertEqual(json.loads(path.read_text()), {"step": 2})
        self.assertFalse(path.with_name("atomic.json.tmp").exists())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
    def test_cli_403_persists_backoff_before_next_cloud_call(self):
        error = urllib.error.HTTPError("fixture", 403, "quota", {}, None)
        self.addCleanup(error.close)
        args = ["--bucket", "fixture", "--source-id", "source", "--source-epoch", "epoch",
                "--drive-root", "root", "--work-dir", str(self.work), "--once"]
        with patch("gcs_drive_drain.gce_region", return_value="us-west1"), \
             patch.object(Path, "is_relative_to", return_value=True), \
             patch("gcs_drive_drain.GCS", side_effect=error) as cloud:
            self.assertEqual(main(args), 75)
            self.assertEqual(main(args), 75)
            self.assertEqual(cloud.call_count, 1)
        self.assertEqual(json.loads((self.work/"retry-after.json").read_text())["http_status"], 403)
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
        def request(url, method="GET", data=None, headers=None, timeout=60):
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
