"""Tiny synthetic diagnostic fixtures: no credentials, network, or cloud calls."""
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error

import gcs_drive_diagnostic as diagnostic
import make_drive_diagnostic_manifest as manifest_maker
from drive_transfer_authority import DIAGNOSTIC_QUOTA_OBSERVATION, TransferAuthority
from gcs_drive_drain import Conflict, FOLDER, ObjectRef, SafeDrive, _NO_REDIRECT_OPENER


class Cloud:
    bucket_name = diagnostic.BUCKET

    def __init__(self):
        self.objects = {}
        self.downloaded = []
        self.deleted = []

    def add(self, name, body, generation, metadata):
        ref = ObjectRef(name, generation, len(body), metadata)
        self.objects[(name, generation)] = (ref, body)
        return ref

    def lookup(self, name, generation=None):
        return self.objects.get((name, generation), (None, None))[0]

    def download(self, ref, path):
        self.downloaded.append((ref.name, ref.generation))
        path.write_bytes(self.objects[(ref.name, ref.generation)][1])

    def delete(self, ref):
        self.deleted.append(ref)
        raise AssertionError("diagnostic attempted GCS deletion")


class Authority:
    account_id = "synthetic-account"

    def __init__(self, media, control):
        self.media = media
        self.control = control
        self.reservations = []
        self.closed = []

    def assert_active(self, timeout):
        if timeout > 300:
            raise AssertionError("diagnostic exceeded request deadline")

    def reserve_pair(self, key, media, control, metadata):
        if media != self.media or control != self.control:
            raise AssertionError("diagnostic did not reserve exact pair and control bytes")
        self.reservations.append((key, media, control, metadata))

    def close_pair(self, key):
        self.closed.append(key)


class Drive:
    def __init__(self):
        self.ids = []
        self.files = {}
        self.folder_posts = []
        self.uploads = []
        self.ambiguous_create = False
        self.create_visible = False
        self.fail_upload = False
        self.drain = None

    def verify_account(self, account_id):
        if account_id != "synthetic-account":
            raise Conflict("wrong synthetic Drive account")

    def metadata(self, file_id):
        if file_id == diagnostic.DRIVE_ROOT:
            return {"id": file_id, "mimeType": FOLDER, "trashed": False}
        if file_id not in self.files:
            error = urllib.error.HTTPError("synthetic", 404, "missing", {}, None)
            error.close()
            raise error
        return dict(self.files[file_id])

    def reserve_id(self):
        file_id = f"synthetic-generated-{len(self.ids) + 1}"
        self.ids.append(file_id)
        return file_id

    def request(self, url, method="GET", data=None, headers=None, timeout=60, kind="drive_api"):
        if method != "POST" or kind != "diagnostic_folder_create":
            raise AssertionError("unexpected synthetic Drive request")
        entry = next(iter(self.drain.state["segments"].values()))
        saved = next(iter(json.loads(self.drain.journal_path.read_text())["segments"].values()))
        for state in (entry, saved):
            if not all(state["drive"][name].get("id") for name in ("folder", "payload", "receipt")):
                raise AssertionError("all three IDs must be durably pinned before folder POST")
            if state["status"] != "folder_create_inflight":
                raise AssertionError("folder intent not journaled before POST")
        body = json.loads(data)
        self.folder_posts.append(body)
        if self.create_visible or not self.ambiguous_create:
            self.files[body["id"]] = {**body, "trashed": False}
        if self.ambiguous_create:
            raise urllib.error.URLError("synthetic lost folder-create response")
        return io.BytesIO(json.dumps({"id": body["id"]}).encode())

    def ensure_file(self, path, parent, expected, state, save):
        entry = next(iter(self.drain.state["segments"].values()))
        saved = next(iter(json.loads(self.drain.journal_path.read_text())["segments"].values()))
        if not all(saved["drive"][kind].get("id") for kind in ("folder", "payload", "receipt")):
            raise AssertionError("file IDs not journaled before media send")
        if entry["status"] != "file_upload_inflight":
            raise AssertionError("file-upload intent not journaled")
        if self.fail_upload:
            raise urllib.error.URLError("synthetic uncertain upload")
        file_id = state["id"]
        info = {"id": file_id, "name": path.name, "size": str(expected["size"]),
                "md5Checksum": expected["md5"], "parents": [parent], "trashed": False}
        self.files[file_id] = info
        self.uploads.append((file_id, path.name))
        return info


class ManifestTests(unittest.TestCase):
    def test_exact_approved_cap_and_pair_are_required(self):
        self.assertEqual(diagnostic.TOTAL_CAP, 270030113)
        release_id = "a" * 32
        manifest = {"schema": "catbird.drive.diagnostic.v1", "release_id": release_id,
                    "account_id": "synthetic-account", "bucket": diagnostic.BUCKET,
                    "source_id": diagnostic.SOURCE_ID, "source_epoch": diagnostic.SOURCE_EPOCH,
                    "drive_root": diagnostic.DRIVE_ROOT, "payload": diagnostic.PAYLOAD,
                    "receipt": diagnostic.RECEIPT,
                    "folder_name": "catbird-drive-diagnostic-" + release_id,
                    "control_allowance_bytes": 1048576, "max_charged_bytes": 270030113,
                    "max_duration_seconds": 1200, "retain_gcs": True, "retain_local": True,
                    "archive_qualified": False}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest))
            loaded, sha = diagnostic.load_diagnostic_manifest(
                path, release_id=release_id, account_id="synthetic-account")
            self.assertEqual(loaded, manifest)
            self.assertEqual(sha, hashlib.sha256(path.read_bytes()).hexdigest())
            for changed in ({"max_charged_bytes": 270030114},
                            {"control_allowance_bytes": 1048577},
                            {"max_duration_seconds": 1201},
                            {"retain_gcs": False}, {"archive_qualified": True},
                            {"retain_gcs": 1}, {"archive_qualified": 0},
                            {"folder_name": "0000000e"},
                            {"payload": {**diagnostic.PAYLOAD, "generation": 1}},
                            {"drive_parent": {"id": "archive-shard"}}):
                with self.subTest(changed=changed):
                    path.write_text(json.dumps({**manifest, **changed}))
                    with self.assertRaises(Conflict):
                        diagnostic.load_diagnostic_manifest(
                            path, release_id=release_id, account_id="synthetic-account")

    def test_offline_manifest_generator_writes_exact_new_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostic.json"
            with patch.object(manifest_maker, "print") as printed:
                self.assertEqual(manifest_maker.main([
                    "--account-id", "synthetic-account", "--output", str(path),
                    "--release-id", "d" * 32]), 0)
            manifest, digest = diagnostic.load_diagnostic_manifest(
                path, release_id="d" * 32, account_id="synthetic-account")
            self.assertEqual(manifest["max_charged_bytes"], 270030113)
            self.assertEqual(manifest["max_duration_seconds"], 1200)
            self.assertEqual(json.loads(printed.call_args.args[0])["manifest_sha256"], digest)
            with self.assertRaises(FileExistsError):
                manifest_maker.main(["--account-id", "synthetic-account", "--output", str(path)])


class WatchdogTests(unittest.TestCase):
    @staticmethod
    def synthetic_kill_group(child, _signal):
        # macOS task sandbox denies killpg; emulate delivery to the direct
        # worker. The Linux OS group operation still needs deployment review.
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)

    def test_persisted_release_deadline_bounds_monotonic_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            db_path = Path(temporary) / "synthetic-authority.sqlite"
            with contextlib.closing(sqlite3.connect(db_path)) as db:
                db.execute("CREATE TABLE release(id TEXT PRIMARY KEY, started_at REAL, deadline_at REAL)")
                now = time.time()
                db.execute("INSERT INTO release VALUES (?,?,?)", ("release", now - 10, now + 30))
                db.commit()
            before = time.monotonic()
            cutoff = diagnostic.release_watchdog_cutoff(db_path, "release", 60)
            self.assertLessEqual(cutoff - before, 30 - diagnostic.WATCHDOG_MARGIN_SECONDS)
            self.assertGreater(cutoff - before, 25)
            with self.assertRaisesRegex(Conflict, "unavailable"):
                diagnostic.release_watchdog_cutoff(db_path, "other-release", 60)
            with self.assertRaisesRegex(Conflict, "duration or deadline"):
                diagnostic.release_watchdog_cutoff(db_path, "release", 20)

    def test_supervisor_kills_trickling_worker_before_late_side_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            late_marker = Path(temporary) / "late-marker"
            def stalled_worker():
                time.sleep(0.25)  # Synthetic stalled I/O; no cloud or network.
                late_marker.write_text("late")
                return 0
            with patch.object(diagnostic, "_arm_parent_death"), \
                 patch.object(diagnostic.os, "killpg", side_effect=self.synthetic_kill_group), \
                 contextlib.redirect_stderr(io.StringIO()) as stderr:
                result = diagnostic.supervise_worker(stalled_worker, time.monotonic() + 0.05)
            self.assertEqual(result, 124)
            self.assertFalse(late_marker.exists())
            self.assertIn("drive_diagnostic_watchdog_stop", stderr.getvalue())

    def test_supervisor_kills_worker_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            late_marker = Path(temporary) / "grandchild-marker"
            descendant_pid = Path(temporary) / "grandchild-pid"
            def spawned_worker():
                descendant = diagnostic.os.fork()
                if descendant == 0:
                    time.sleep(0.25)
                    late_marker.write_text("late")
                    diagnostic.os._exit(0)
                descendant_pid.write_text(str(descendant))
                time.sleep(0.25)
                return 0
            def synthetic_group_kill(child, sig):
                if descendant_pid.exists():
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(int(descendant_pid.read_text()), sig)
                self.synthetic_kill_group(child, sig)
            with patch.object(diagnostic, "_arm_parent_death"), \
                 patch.object(diagnostic.os, "killpg", side_effect=synthetic_group_kill), \
                 contextlib.redirect_stderr(io.StringIO()):
                result = diagnostic.supervise_worker(spawned_worker, time.monotonic() + 0.05)
            self.assertEqual(result, 124)
            time.sleep(0.28)
            self.assertFalse(late_marker.exists())

    def test_worker_success_still_stops_early_exit_descendant(self):
        with tempfile.TemporaryDirectory() as temporary:
            descendant_pid = Path(temporary) / "grandchild-pid"
            late_marker = Path(temporary) / "late-marker"
            def early_exit_worker():
                descendant = os.fork()
                if descendant == 0:
                    time.sleep(0.25)
                    late_marker.write_text("late")
                    os._exit(0)
                descendant_pid.write_text(str(descendant))
                return 0
            calls = []
            def synthetic_group_kill(child, sig):
                calls.append(child)
                if descendant_pid.exists():
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(int(descendant_pid.read_text()), sig)
                self.synthetic_kill_group(child, sig)
            with patch.object(diagnostic, "_arm_parent_death"), \
                 patch.object(diagnostic.os, "killpg", side_effect=synthetic_group_kill):
                result = diagnostic.supervise_worker(early_exit_worker,
                                                      time.monotonic() + 0.2)
            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 1)
            time.sleep(0.28)
            self.assertFalse(late_marker.exists())

    def test_finished_after_cutoff_never_reports_success(self):
        # A slow waitpid response can reveal a worker that completed late.
        original_waitpid = diagnostic.os.waitpid
        def delayed_waitpid(pid, flags):
            outcome = original_waitpid(pid, 0)
            time.sleep(0.08)
            return outcome
        with patch.object(diagnostic, "_arm_parent_death"), \
             patch.object(diagnostic.os, "killpg", side_effect=self.synthetic_kill_group), \
             patch.object(diagnostic.os, "waitpid", side_effect=delayed_waitpid):
            with contextlib.redirect_stderr(io.StringIO()):
                result = diagnostic.supervise_worker(lambda: 0, time.monotonic() + 0.05)
        self.assertEqual(result, 124)

    def test_supervisor_sigterm_stops_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            late_marker = Path(temporary) / "late-marker"
            def worker():
                until = time.monotonic() + 0.25
                while time.monotonic() < until:
                    pass
                late_marker.write_text("late")
                return 0
            def interrupt_parent(_seconds):
                signal.raise_signal(signal.SIGTERM)
            with patch.object(diagnostic, "_arm_parent_death"), \
                 patch.object(diagnostic.os, "killpg", side_effect=self.synthetic_kill_group), \
                 patch.object(diagnostic.time, "sleep", side_effect=interrupt_parent):
                with self.assertRaises(KeyboardInterrupt):
                    diagnostic.supervise_worker(worker, time.monotonic() + 0.2)
            self.assertFalse(late_marker.exists())


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name) / "diagnostic"
        self.release_id = "b" * 32
        self.payload_data = b"tiny sealed synthetic segment"
        sha = hashlib.sha256(self.payload_data).hexdigest()
        name = "seg_00000002u5.jss"
        payload_name = (f"sources/{diagnostic.SOURCE_ID}/{diagnostic.SOURCE_EPOCH}/segments/"
                        f"0000000e/{name}/{sha}.jss")
        self.cloud = Cloud()
        payload = self.cloud.add(payload_name, self.payload_data, 11,
                                 {"kind": "payload", "sha256": sha})
        receipt_body = {"schema": "catbird.jetstream.archive.v1", "name": name,
                        "index": 3677, "sha256": sha, "bytes": payload.size,
                        "source_id": diagnostic.SOURCE_ID,
                        "source_epoch": diagnostic.SOURCE_EPOCH,
                        "gcs": {"bucket": diagnostic.BUCKET, "object": payload_name,
                                "generation": 11, "bytes": payload.size, "sha256": sha,
                                "md5_hash": base64.b64encode(
                                    hashlib.md5(self.payload_data).digest()).decode()}}
        receipt_data = json.dumps(receipt_body).encode()
        receipt_name = payload_name + ".g11.receipt.json"
        receipt = self.cloud.add(receipt_name, receipt_data, 12,
                                 {"kind": "receipt", "sha256": hashlib.sha256(receipt_data).hexdigest(),
                                  "segment_name": name, "payload_object": payload_name,
                                  "payload_generation": "11"})
        self.payload = {"object": payload.name, "generation": payload.generation,
                        "bytes": payload.size, "sha256": sha}
        self.receipt = {"object": receipt.name, "generation": receipt.generation,
                        "bytes": receipt.size, "sha256": receipt.metadata["sha256"]}
        self.control = 1024
        self.cap = payload.size + receipt.size + self.control
        for key, value in (("PAYLOAD", self.payload), ("RECEIPT", self.receipt),
                           ("CONTROL_ALLOWANCE", self.control), ("TOTAL_CAP", self.cap)):
            patcher = patch.object(diagnostic, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.manifest = {"schema": "catbird.drive.diagnostic.v1",
                         "release_id": self.release_id, "account_id": "synthetic-account",
                         "bucket": diagnostic.BUCKET, "source_id": diagnostic.SOURCE_ID,
                         "source_epoch": diagnostic.SOURCE_EPOCH,
                         "drive_root": diagnostic.DRIVE_ROOT,
                         "payload": self.payload, "receipt": self.receipt,
                         "folder_name": "catbird-drive-diagnostic-" + self.release_id,
                         "control_allowance_bytes": self.control,
                         "max_charged_bytes": self.cap,
                         "max_duration_seconds": 1200,
                         "retain_gcs": True, "retain_local": True,
                         "archive_qualified": False}
        self.manifest_path = Path(self.temporary.name) / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.manifest, self.manifest_sha = diagnostic.load_diagnostic_manifest(
            self.manifest_path, release_id=self.release_id, account_id="synthetic-account")
        self.authority = Authority(payload.size + receipt.size, self.control)
        self.drive = Drive()

    def new_drain(self):
        drain = diagnostic.DiagnosticDrain(self.cloud, self.drive, self.authority,
                                           self.work, self.manifest, self.manifest_sha)
        self.drive.drain = drain
        return drain

    def test_exact_pair_in_isolated_child_is_verified_but_never_archive_qualified(self):
        drain = self.new_drain()
        with patch.object(drain, "finish", side_effect=AssertionError("archive finish called")), \
             patch.object(drain, "clean_verified_temp", side_effect=AssertionError("local cleanup called")):
            result = drain.run_diagnostic()
        self.assertEqual(result["status"], diagnostic.FINAL_STATUS)
        self.assertIs(result["archive_qualified"], False)
        self.assertEqual(len(self.drive.ids), 3)
        self.assertEqual(len(self.drive.folder_posts), 1)
        self.assertEqual(self.drive.folder_posts[0]["parents"], [diagnostic.DRIVE_ROOT])
        self.assertEqual(self.drive.folder_posts[0]["name"], self.manifest["folder_name"])
        self.assertEqual([name for _, name in self.drive.uploads],
                         ["seg_00000002u5.jss", "seg_00000002u5.jss.receipt.json"])
        self.assertEqual(len(self.authority.closed), 1)
        self.assertEqual(self.cloud.deleted, [])
        self.assertTrue(list((self.work / "stage").rglob("*.jss")))
        self.assertTrue(list((self.work / "stage").rglob("*.receipt.json")))
        entry = next(iter(drain.state["segments"].values()))
        self.assertIs(entry["archive_qualified"], False)
        self.assertEqual(entry["status"], diagnostic.FINAL_STATUS)

    def test_lost_folder_create_response_reconciles_only_pinned_id_and_stops(self):
        self.drive.ambiguous_create = True
        self.drive.create_visible = True
        drain = self.new_drain()
        with self.assertRaises(urllib.error.URLError):
            drain.run_diagnostic()
        entry = next(iter(drain.state["segments"].values()))
        self.assertEqual(entry["status"], "folder_create_inflight")
        ids = tuple(self.drive.ids)
        self.assertEqual(len(self.drive.folder_posts), 1)
        replay = self.new_drain()
        with self.assertRaisesRegex(Conflict, "uncertain create charge needs review"):
            replay.run_diagnostic()
        self.assertEqual(tuple(self.drive.ids), ids)
        self.assertEqual(len(self.drive.folder_posts), 1)
        self.assertEqual(self.drive.uploads, [])
        self.assertEqual(self.authority.closed, [])
        self.assertEqual(next(iter(replay.state["segments"].values()))["status"],
                         "folder_reconciled_unsettled")

    def test_missing_uncertain_folder_never_reposts(self):
        self.drive.ambiguous_create = True
        with self.assertRaises(urllib.error.URLError):
            self.new_drain().run_diagnostic()
        with self.assertRaisesRegex(Conflict, "not visible after uncertain create"):
            self.new_drain().run_diagnostic()
        self.assertEqual(len(self.drive.folder_posts), 1)
        self.assertEqual(len(self.drive.ids), 3)
        self.assertEqual(self.drive.uploads, [])
        self.assertEqual(self.cloud.deleted, [])

    def test_mismatched_uncertain_folder_never_reposts(self):
        self.drive.ambiguous_create = True
        self.drive.create_visible = True
        with self.assertRaises(urllib.error.URLError):
            self.new_drain().run_diagnostic()
        self.drive.files[self.drive.ids[0]]["name"] = "wrong-folder"
        with self.assertRaisesRegex(Conflict, "folder identity differs"):
            self.new_drain().run_diagnostic()
        self.assertEqual(len(self.drive.folder_posts), 1)
        self.assertEqual(len(self.drive.ids), 3)
        self.assertEqual(self.drive.uploads, [])
        self.assertEqual(self.cloud.deleted, [])

    def test_uncertain_file_send_never_replays(self):
        self.drive.fail_upload = True
        with self.assertRaises(urllib.error.URLError):
            self.new_drain().run_diagnostic()
        self.assertEqual(next(iter(self.new_drain().state["segments"].values()))["status"],
                         "file_upload_inflight")
        counts = (len(self.drive.ids), len(self.drive.folder_posts), len(self.drive.uploads))
        with self.assertRaisesRegex(Conflict, "no automatic retry"):
            self.new_drain().run_diagnostic()
        self.assertEqual((len(self.drive.ids), len(self.drive.folder_posts), len(self.drive.uploads)),
                         counts)
        self.assertEqual(self.authority.closed, [])

    def test_completed_diagnostic_cannot_replay_or_qualify_archive(self):
        self.new_drain().run_diagnostic()
        original_posts, original_ids, original_uploads = (
            len(self.drive.folder_posts), len(self.drive.ids), len(self.drive.uploads))
        with self.assertRaisesRegex(Conflict, "no automatic retry"):
            self.new_drain().run_diagnostic()
        self.assertEqual((len(self.drive.folder_posts), len(self.drive.ids), len(self.drive.uploads)),
                         (original_posts, original_ids, original_uploads))

    def test_manifest_or_journal_release_replay_is_rejected(self):
        self.new_drain().run_diagnostic()
        other = dict(self.manifest)
        other["release_id"] = "c" * 32
        other["folder_name"] = "catbird-drive-diagnostic-" + other["release_id"]
        other_sha = hashlib.sha256(json.dumps(other).encode()).hexdigest()
        with self.assertRaisesRegex(Conflict, "identity differs"):
            diagnostic.DiagnosticDrain(self.cloud, self.drive, self.authority,
                                       self.work, other, other_sha).run_diagnostic()

    def test_folder_create_body_is_charged_by_safe_drive(self):
        drive = object.__new__(SafeDrive)
        events = []
        class Meter:
            def clock(self):
                return time.time()
            def assert_active(self, timeout, *, slot_at=None):
                return None
            def start_request(self, body_bytes, timeout, kind, **identity):
                events.append(("start", body_bytes, kind, identity))
                return "attempt", time.time()
            def mark_dispatch(self, attempt_id):
                events.append(("dispatch", attempt_id))
            def finish_request(self, attempt_id, status):
                events.append(("finish", status))
        drive.authority = Meter()
        drive.auth = type("Auth", (), {"token": lambda self: "synthetic-token"})()
        class Response(io.BytesIO):
            status = 200
        def opened(request, timeout=60):
            events.append(("open",))
            return Response(b'{"id":"synthetic-generated-1"}')
        with patch.object(_NO_REDIRECT_OPENER, "open", side_effect=opened):
            info = diagnostic.create_folder_by_pinned_id(
                drive, folder_id="synthetic-generated-1",
                name=self.manifest["folder_name"], root=diagnostic.DRIVE_ROOT)
        self.assertEqual(info["id"], "synthetic-generated-1")
        self.assertGreater(events[0][1], 0)
        self.assertLess(events[0][1], self.control)
        self.assertEqual(events[0][2], "diagnostic_folder_create")
        self.assertEqual([event[0] for event in events],
                         ["start", "dispatch", "open", "finish"])
        self.assertRegex(events[0][3]["operation_key"], r"^[0-9a-f]{64}$")


class RealAuthorityIntegrationTests(unittest.TestCase):
    def test_observation_authority_accepts_exact_synthetic_diagnostic_flow(self):
        """Real SQLite reservation/closure; synthetic source integrity avoids 269 MB fixture."""
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            release_id = "e" * 32
            account_id = "synthetic-account"
            manifest = diagnostic.build_diagnostic_manifest(
                release_id=release_id, account_id=account_id)
            raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            digest = hashlib.sha256(raw).hexdigest()
            now = time.time()
            evidence = {"account_id": account_id, "account_identity_verified": True,
                        "all_writers_excluded": True, "history_basis": "unproven",
                        "quota_rejection_accepted": True,
                        "exclusion_valid_until": now + 700, "observed_at": now,
                        "evidence_sha256": "a" * 64}
            db_path = work / "observation.sqlite"
            TransferAuthority.initialize_diagnostic_quota_observation(
                db_path, release_id=release_id, writer_id="synthetic-writer",
                account_id=account_id, manifest_bytes=raw, evidence=evidence,
                deadline_at=now + 600)
            authority = TransferAuthority.open_existing(
                db_path, release_id=release_id, writer_id="synthetic-writer",
                account_id=account_id, manifest_sha256=digest,
                allowed_mode=DIAGNOSTIC_QUOTA_OBSERVATION)
            self.assertEqual(authority.admission_mode, DIAGNOSTIC_QUOTA_OBSERVATION)
            self.assertIs(authority.snapshot()["history_proven"], False)
            self.assertIsNone(authority.snapshot()["effective_rolling_cap_bytes"])
            cloud = Cloud()
            payload = manifest["payload"]
            receipt = manifest["receipt"]
            payload_ref = ObjectRef(payload["object"], payload["generation"], payload["bytes"],
                                    {"kind": "payload", "sha256": payload["sha256"]})
            receipt_ref = ObjectRef(receipt["object"], receipt["generation"], receipt["bytes"],
                                    {"kind": "receipt", "sha256": receipt["sha256"],
                                     "segment_name": "seg_00000002u5.jss",
                                     "payload_object": payload["object"],
                                     "payload_generation": str(payload["generation"])})
            cloud.objects[(receipt_ref.name, receipt_ref.generation)] = (receipt_ref, b"synthetic")
            drive = Drive()
            class SyntheticVerifiedSource(diagnostic.DiagnosticDrain):
                def download_checked(self, ref, path, expected_sha, limit):
                    path.write_bytes(b"synthetic")
                    return {"size": ref.size, "sha256": expected_sha, "md5": "00" * 16}
                def parse_receipt(self, ref, path):
                    return ({"sha256": payload["sha256"], "name": "seg_00000002u5.jss",
                             "gcs": {"md5_hash": base64.b64encode(bytes(16)).decode()}},
                            payload_ref)
            drain = SyntheticVerifiedSource(cloud, drive, authority, work / "diagnostic",
                                            manifest, digest)
            drive.drain = drain
            result = drain.run_diagnostic()
            self.assertEqual(result["status"], diagnostic.FINAL_STATUS)
            self.assertIs(result["archive_qualified"], False)
            self.assertEqual(len(drive.folder_posts), 1)
            self.assertEqual(len(drive.uploads), 2)
            with contextlib.closing(sqlite3.connect(db_path)) as db:
                self.assertEqual(db.execute("SELECT status FROM release").fetchone()[0], "closed")
            authority.close()


class HttpSourceTests(unittest.TestCase):
    def test_oauth_403_is_preserved_without_drive_dispatch(self):
        drive = object.__new__(diagnostic.DiagnosticSafeDrive)
        events = []
        class Meter:
            def assert_active(self, timeout, *, slot_at=None):
                return None
            def start_request(self, *args, **kwargs):
                events.append("start")
                return "attempt", time.time()
        class Auth:
            def token(self):
                raise urllib.error.HTTPError("https://oauth.invalid/token", 403,
                                             "forbidden", {}, io.BytesIO(b"secret"))
        drive.authority = Meter()
        drive.auth = diagnostic._TokenProxy(Auth())
        with patch.object(_NO_REDIRECT_OPENER, "open", side_effect=AssertionError("Drive sent")):
            with self.assertRaises(diagnostic.OAuthRefreshBlocked):
                drive.request("https://www.googleapis.com/drive/v3/about", kind="account")
        self.assertEqual(events, [])

    def test_drive_403_is_reported_as_drive_http_with_bounded_reason(self):
        drive = object.__new__(diagnostic.DiagnosticSafeDrive)
        events = []
        class Meter:
            def clock(self):
                return time.time()
            def assert_active(self, timeout, *, slot_at=None):
                return None
            def start_request(self, *args, **kwargs):
                events.append("start")
                return "attempt", time.time()
            def mark_dispatch(self, attempt):
                events.append("dispatch")
            def finish_request(self, attempt, status):
                events.append(("finish", status))
        drive.authority = Meter()
        drive.auth = diagnostic._TokenProxy(type("Auth", (), {"token": lambda self: "synthetic"})())
        error = urllib.error.HTTPError(
            "https://www.googleapis.com/drive/v3/files", 403, "forbidden", {},
            io.BytesIO(b'{"error":{"errors":[{"reason":"insufficientFilePermissions"}]}}'))
        with patch.object(_NO_REDIRECT_OPENER, "open", side_effect=error):
            with self.assertRaises(diagnostic.DriveHttpStop) as stopped:
                drive.request("https://www.googleapis.com/drive/v3/files", kind="metadata")
        self.assertEqual(stopped.exception.status, 403)
        self.assertEqual(stopped.exception.reason, "insufficientFilePermissions")
        self.assertEqual(events, ["start", "dispatch", ("finish", 403)])
        with patch.object(diagnostic, "_run_worker", side_effect=stopped.exception), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(diagnostic._run_worker_guarded(None, {}, ""), 75)
        report = json.loads(output.getvalue())
        self.assertEqual(report["event"], "drive_http_stop")
        self.assertEqual(report["drive_reason"], "insufficientFilePermissions")
        self.assertIs(report["history_proven"], False)

    def test_gcs_403_is_preserved_without_drive_quota_label(self):
        error = urllib.error.HTTPError("https://storage.invalid/object", 403,
                                       "forbidden", {}, io.BytesIO(b"secret"))
        with patch.object(diagnostic, "_run_worker", side_effect=error), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(diagnostic._run_worker_guarded(None, {}, ""), 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["event"], "drive_diagnostic_preserved")
        self.assertNotIn("secret", output.getvalue())


if __name__ == "__main__":
    unittest.main()
