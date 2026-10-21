"""Synthetic-only retained runner checks; no credentials or cloud calls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from gcs_drive_drain import Conflict
import gcs_drive_retained_runner as runner
from test_gcs_drive_retained_overflow import ACCOUNT, RELEASE, fixture_manifest


class FakeLedger:
    def __init__(self):
        self.status = "unadmitted"
        self.unresolved = 0
        self.cooldown = 0
        self.mapping_remaining = 32 * (1 << 20)
        self.inspections = 0

    def inspect_account(self, *_args, **_kwargs):
        self.inspections += 1
        return {"admission_mode": "retained_pair_observation", "history_proven": False,
                "diagnostic_imported": True, "cooldown_until": self.cooldown,
                "next_slot": 0, "last_clock": 0,
                "unresolved_attempts": self.unresolved,
                "mapping_partition_remaining_bytes": self.mapping_remaining,
                "known_local_rolling_bytes": 123,
                "nonterminal_run_id": None if self.status in ("unadmitted", "closed") else RELEASE,
                "nonterminal_status": None if self.status in ("unadmitted", "closed") else self.status}

    def list_pairs(self, *_args, **_kwargs):
        return ({"run_id": "diagnostic-import", "pair_key": None,
                 "status": "diagnostic_verified_unqualified",
                 "bucket": "synthetic-bucket", "payload_object": "other",
                 "payload_generation": 42},)

    def inspect_pair(self, *_args, **_kwargs):
        return {"status": self.status, "mapping_sha256": "a" * 64}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot_dir = self.root / "snapshot"
        self.snapshot_dir.mkdir()
        self.evidence_dir = self.root / "evidence"
        self.evidence_dir.mkdir()
        self.capture_dir = self.root / "segments"
        self.capture_dir.mkdir()
        self.oauth = self.root / "existing-oauth.json"
        self.oauth.write_text("synthetic-only")
        (self.root / "ledger.sqlite").write_bytes(b"synthetic ledger identity only")
        self.ledger = FakeLedger()
        self.manifest, _ = fixture_manifest()
        self.manifest_path = self.snapshot_dir / "manifest.json"
        raw = json.dumps(self.manifest, sort_keys=True, separators=(",", ":")).encode()
        self.manifest_path.write_bytes(raw)
        self.manifest_digest = hashlib.sha256(raw).hexdigest()
        self.candidate = {"pair_key": runner.pair_key(self.manifest),
                          "release_id": RELEASE, "manifest_path": str(self.manifest_path),
                          "manifest_sha256": self.manifest_digest,
                          "reserved_bytes": self.manifest["max_charged_bytes"]}
        self.snapshot = self.snapshot_dir / "snapshot.json"
        self._save_snapshot()
        self.status = self.root / "retained_runner_status.jsonl"
        self.evidence = self.evidence_dir / f"{RELEASE}.json"
        self._save_evidence()

    def _save_snapshot(self):
        raw = json.dumps({"schema": runner.SNAPSHOT_SCHEMA,
                          "candidates": [self.candidate]}, sort_keys=True).encode()
        self.snapshot.write_bytes(raw)
        self.snapshot_digest = hashlib.sha256(raw).hexdigest()

    def _save_evidence(self, **changes):
        obj = {"run_id": RELEASE, "writer_id": "synthetic-writer",
               "account_id": ACCOUNT, "reviewed_manifest_sha256": self.manifest_digest,
               "history_basis": "unproven", "account_identity_verified": True,
               "all_writers_excluded": True, "quota_rejection_accepted": True,
               "exclusion_valid_until": time.time() + 7200,
               "observed_at": time.time(), "evidence_sha256": "b" * 64}
        obj.update(changes)
        self.evidence.write_text(json.dumps(obj))

    def _run(self, child):
        with (mock.patch.object(runner, "STOP_AT", time.time() + 3600),
              mock.patch.object(runner, "DATA_ROOT", self.root.resolve()),
              mock.patch.object(runner.sys, "platform", "linux"),
              mock.patch.object(runner.TransferAuthority, "inspect_retained_account",
                                side_effect=self.ledger.inspect_account),
              mock.patch.object(runner.TransferAuthority, "list_retained_pairs",
                                side_effect=self.ledger.list_pairs),
              mock.patch.object(runner.TransferAuthority, "inspect_retained_pair",
                                side_effect=self.ledger.inspect_pair),
              mock.patch.object(runner, "check_disk", return_value={"free_after_reservation_bytes": 1})):
            return runner.run_snapshot(
                snapshot_json=self.snapshot, snapshot_sha256=self.snapshot_digest,
                snapshot_dir=self.snapshot_dir, authority_db=self.root / "ledger.sqlite",
                evidence_dir=self.evidence_dir, account_id=ACCOUNT,
                writer_id="synthetic-writer", oauth_file=self.oauth,
                status_file=self.status, data_root=self.root,
                capture_dir=self.capture_dir, child_runner=child)

    def _latest_status(self):
        lines = [json.loads(line) for line in self.status.read_text().splitlines()
                 if line.strip()]
        self.assertGreaterEqual(len(lines), 2)
        return lines[-1]

    def test_serial_pair_requires_durable_closed_mapping_and_account_gate(self):
        invoked = []
        def child(command, **kwargs):
            invoked.append(command)
            self.ledger.status = "closed"
            return 0
        result = self._run(child)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["pairs_closed"], 1)
        self.assertEqual(result["last_mapping_sha256"], "a" * 64)
        self.assertEqual(len(invoked), 1)
        self.assertIn("--retain-gcs", invoked[0])
        self.assertIn("--quota-observation", invoked[0])
        self.assertGreaterEqual(self.ledger.inspections, 2)
        self.assertEqual(self._latest_status()["state"], "completed")

    def test_unresolved_after_successful_child_stops_sequence(self):
        def child(_command, **_kwargs):
            self.ledger.status = "closed"
            self.ledger.unresolved = 1
            return 0
        with self.assertRaises(Conflict):
            self._run(child)
        self.assertEqual(self._latest_status()["state"], "stopped")

    def test_nonzero_child_never_claims_pair_closed(self):
        with self.assertRaises(Conflict):
            self._run(lambda *_args, **_kwargs: 75)
        state = self._latest_status()
        self.assertEqual(state["pairs_closed"], 0)
        self.assertEqual(state["state"], "stopped")

    def test_403_status_is_explicit_and_never_retried(self):
        calls = []
        def child(_command, **kwargs):
            calls.append(1)
            kwargs["log_path"].write_text('{"event":"overflow_drive_http_stop","http_status":403}\n')
            return 75
        with self.assertRaises(Conflict):
            self._run(child)
        self.assertEqual(calls, [1])
        self.assertEqual(self._latest_status()["stop_reason"], "drive_http_403")

    def test_stale_evidence_stops_before_child(self):
        self._save_evidence(observed_at=time.time() - 301)
        result = self._run(lambda *_args, **_kwargs: self.fail("child must not start"))
        self.assertEqual(result["state"], "awaiting_fresh_evidence")

    def test_uncertain_account_or_partition_stops_before_child(self):
        for name, value in (("unresolved", 1), ("cooldown", time.time() + 500),
                            ("mapping_remaining", 1)):
            with self.subTest(name=name):
                setattr(self.ledger, name, value)
                with self.assertRaises(Conflict):
                    self._run(lambda *_args, **_kwargs: self.fail("child must not start"))
                setattr(self.ledger, name, 0 if name != "mapping_remaining" else 32 * (1 << 20))

    def test_snapshot_hash_manifest_mutation_and_duplicate_rejected(self):
        with self.assertRaises(Conflict):
            runner.load_snapshot(self.snapshot, "0" * 64, self.snapshot_dir, ACCOUNT)
        changed = dict(self.manifest)
        changed["folder_name"] = "tampered"
        self.manifest_path.write_text(json.dumps(changed))
        with self.assertRaises(Conflict):
            runner.load_snapshot(self.snapshot, self.snapshot_digest, self.snapshot_dir, ACCOUNT)

    def test_disk_guard_uses_capture_shared_filesystem_and_cumulative_stage(self):
        item = runner.load_snapshot(self.snapshot, self.snapshot_digest,
                                    self.snapshot_dir, ACCOUNT)[0]
        with mock.patch.object(runner.shutil, "disk_usage") as disk:
            disk.return_value.free = runner.CAPTURE_RESTART_FLOOR_BYTES + 1
            with self.assertRaises(Conflict):
                runner.check_disk(item, data_root=self.root, capture_dir=self.capture_dir)
            disk.return_value.free = runner.CAPTURE_RESTART_FLOOR_BYTES + 100 * (1 << 20)
            result = runner.check_disk(item, data_root=self.root, capture_dir=self.capture_dir)
            self.assertGreater(result["free_after_reservation_bytes"], runner.CAPTURE_RESTART_FLOOR_BYTES)
        work = self.root / ("gcs-drive-retained-" + RELEASE)
        work.mkdir()
        (work / "large-synthetic.stage").write_bytes(b"x" * 1024)
        with mock.patch.object(runner.shutil, "disk_usage") as disk:
            disk.return_value.free = runner.CAPTURE_RESTART_FLOOR_BYTES + 100 * (1 << 20)
            with self.assertRaises(Conflict):
                runner.check_disk(item, data_root=self.root, capture_dir=self.capture_dir,
                                  max_staged_bytes=1024)

    def test_unreadable_stage_tree_fails_closed(self):
        (self.root / ("gcs-drive-retained-" + RELEASE)).mkdir()
        def unreadable_walk(_root, *, followlinks, onerror):
            self.assertFalse(followlinks)
            onerror(OSError("synthetic unreadable directory"))
            yield from ()
        with mock.patch.object(runner.os, "walk", side_effect=unreadable_walk):
            with self.assertRaisesRegex(Conflict, "unreadable"):
                runner._stage_bytes(self.root)

    def test_child_heartbeat_stops_on_low_shared_disk(self):
        class FakeChild:
            pid = 81234
            returncode = None
            def poll(self):
                return self.returncode
        child = FakeChild()
        def kill(_child):
            child.returncode = -9
        with (mock.patch.object(runner.sys, "platform", "linux"),
              mock.patch.object(runner.subprocess, "Popen", return_value=child),
              mock.patch.object(runner, "_kill_child_tree", side_effect=kill) as killed,
              mock.patch.object(runner.shutil, "disk_usage") as disk):
            disk.return_value.free = runner.CAPTURE_RESTART_FLOOR_BYTES + 9
            with self.assertRaisesRegex(Conflict, "In-flight"):
                runner._run_child(["synthetic-no-cloud"], deadline=time.time() + 100,
                                  status=mock.Mock(), log_path=self.root / "child.log",
                                  runner_stop_monotonic=time.monotonic() + 100,
                                  capture_dir=self.capture_dir, inflight_allowance=10)
            killed.assert_called_once_with(child)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux descendant kill smoke")
    def test_linux_child_timeout_kills_nested_session(self):
        marker = self.root / "nested.pid"
        script = ("import os,time,pathlib\n"
                  "p=os.fork()\n"
                  "if p==0:\n"
                  " os.setsid()\n"
                  f" pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
                  " while True: time.sleep(60)\n"
                  "while True: time.sleep(60)\n")
        fake_status = mock.Mock()
        with mock.patch.object(runner, "CAPTURE_RESTART_FLOOR_BYTES", 1):
            with self.assertRaises(Conflict):
                runner._run_child([sys.executable, "-c", script],
                                  deadline=time.time() + 30,
                                  status=fake_status, log_path=self.root / "child.log",
                                  runner_stop_monotonic=time.monotonic() + 31,
                                  capture_dir=self.capture_dir, inflight_allowance=1)
        self.assertTrue(marker.exists())
        pid = int(marker.read_text())
        for _ in range(50):
            proc_stat = Path(f"/proc/{pid}/stat")
            try:
                state = proc_stat.read_text().split()[2]
            except (FileNotFoundError, ProcessLookupError):
                # Linux may reap the child between lookup and read.
                break
            if state == "Z":
                break
            time.sleep(0.02)
        else:
            self.fail("Runner watchdog left nested-session child active")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux parent-death smoke")
    def test_linux_runner_death_kills_supervisor(self):
        marker = self.root / "orphan-supervisor.pid"
        helper_code = (
            "import os,subprocess,sys,time,pathlib\n"
            "from gcs_drive_retained_runner import _arm_parent_death\n"
            "parent=os.getpid()\n"
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
            "start_new_session=True,preexec_fn=lambda:_arm_parent_death(parent))\n"
            f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n"
            "while True: time.sleep(60)\n"
        )
        helper = subprocess.Popen([sys.executable, "-c", helper_code],
                                  start_new_session=True)
        child_pid = None
        try:
            for _ in range(100):
                if marker.exists():
                    child_pid = int(marker.read_text())
                    break
                time.sleep(0.01)
            self.assertIsNotNone(child_pid, "Linux orphan smoke did not start child")
            os.kill(helper.pid, signal.SIGKILL)
            helper.wait(timeout=2)
            for _ in range(100):
                proc_stat = Path(f"/proc/{child_pid}/stat")
                try:
                    state = proc_stat.read_text().split()[2]
                except (FileNotFoundError, ProcessLookupError):
                    # Linux may reap the child between lookup and read.
                    break
                if state == "Z":
                    break
                time.sleep(0.01)
            else:
                self.fail("Parent-death guard left overflow supervisor running")
        finally:
            if helper.poll() is None:
                helper.kill()
                helper.wait(timeout=2)
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
