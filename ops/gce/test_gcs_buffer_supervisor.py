import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import json
import tempfile
import gcs_buffer_supervisor as supervisor
from gcs_buffer_supervisor import collector_may_start, utc_timestamp


class SupervisorGates(unittest.TestCase):
    def test_only_verified_drained_backlog_and_headroom_can_start(self):
        self.assertTrue(collector_may_start({"pending_sealed": 0}, 101*1024**3, 3))
        for status, disk, count in [
            ({}, 101*1024**3, 3),
            ({"pending_sealed": 1}, 101*1024**3, 3),
            ({"pending_sealed": 0}, 99*1024**3, 3),
            ({"pending_sealed": 0}, 101*1024**3, 4),
        ]:
            self.assertFalse(collector_may_start(status, disk, count))

    def test_absolute_deadline_requires_timezone(self):
        self.assertEqual(utc_timestamp("2026-10-21T05:29:29Z"),
                         utc_timestamp("2026-10-21T01:29:29-04:00"))
        with self.assertRaises(ValueError):
            utc_timestamp("2026-10-21T05:29:29")

    def test_uploader_failure_stops_collector_before_poweroff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"status.json").write_text(json.dumps({"pending_sealed": 0}))
            child = Mock()
            child.returncode = 7
            child.poll.side_effect = [None, 7, 7]
            calls = []

            def run(command, **_kwargs):
                calls.append(command)
                return Mock(returncode=3 if command[:2] == ["systemctl", "is-active"] else 0)

            argv = ["supervisor", "--bucket", "test", "--deadline", "2030-01-01T00:00:00Z",
                    "--work-dir", str(root), "--segments-dir", str(root), "--poweroff-on-exit"]
            with patch.object(supervisor.sys, "argv", argv), \
                 patch.object(supervisor.subprocess, "Popen", return_value=child), \
                 patch.object(supervisor.subprocess, "run", side_effect=run), \
                 patch.object(supervisor.subprocess, "check_output", side_effect=["exited no", "true"]), \
                 patch.object(supervisor.shutil, "disk_usage", return_value=Mock(free=101*1024**3)), \
                 patch.object(supervisor.time, "time", return_value=1000), \
                 patch.object(supervisor.time, "sleep"), \
                 patch.object(supervisor.signal, "signal"), \
                 patch.object(supervisor.os, "sync", create=True):
                self.assertEqual(supervisor.main(), 1)
            self.assertIn(["docker", "start", "js"], calls)
            stop = calls.index(["docker", "stop", "--time", "120", "js"])
            poweroff = calls.index(["systemctl", "poweroff"])
            self.assertLess(stop, poweroff)
            receipt = json.loads((root/"supervisor_status.json").read_text())
            self.assertEqual(receipt["stop_reason"], "buffer_exited_7")


if __name__ == "__main__":
    unittest.main()
