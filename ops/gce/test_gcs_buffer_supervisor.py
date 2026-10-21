import contextlib
import datetime as dt
import io
import json
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gcs_buffer_supervisor as supervisor


class Clock:
    def __init__(self): self.elapsed = 0
    def wall(self): return 1800000000 + self.elapsed
    def monotonic(self): return self.elapsed
    def advance(self, seconds): self.elapsed += seconds


class Event:
    def __init__(self, clock): self.clock, self.stopped = clock, False
    def set(self): self.stopped = True
    def is_set(self): return self.stopped
    def wait(self, seconds): self.clock.advance(seconds)


class SupervisorGates(unittest.TestCase):
    def test_only_verified_drained_backlog_and_headroom_can_start(self):
        self.assertTrue(supervisor.collector_may_start({"pending_sealed": 0}, 101*1024**3, 3))
        for status, disk, count in [({}, 101*1024**3, 3), ({"pending_sealed": 1}, 101*1024**3, 3),
                                    ({"pending_sealed": 0}, 99*1024**3, 3), ({"pending_sealed": 0}, 101*1024**3, 4)]:
            self.assertFalse(supervisor.collector_may_start(status, disk, count))

    def test_absolute_deadline_requires_timezone(self):
        self.assertEqual(supervisor.utc_timestamp("2026-10-21T05:29:29Z"),
                         supervisor.utc_timestamp("2026-10-21T01:29:29-04:00"))
        with self.assertRaises(ValueError): supervisor.utc_timestamp("2026-10-21T05:29:29")

    def run_supervisor(self, root, *, container="exited no", polls=None, heartbeat=True,
                       remaining=3600, stale=900, prior=False, stop_record_error=False):
        clock = Clock()
        event = Event(clock)
        calls = []
        if prior:
            (root / "status.json").write_text('{"old":"buffer"}')
            (root / "supervisor_status.json").write_text('{"old":"supervisor"}')
        child = Mock(pid=99999999)
        child.poll.side_effect = polls
        if polls is None: child.poll.return_value = None
        child.returncode = 7
        def start(command, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            if heartbeat:
                run_id = command[command.index('--run-id') + 1]
                (root / 'status.json').write_text(json.dumps({'run_id': run_id, 'reason': 'ready', 'pending_sealed': 0}))
            self.assertLessEqual(float(command[command.index('--run-hours') + 1]), 72)
            self.assertEqual(command[command.index('--max-bytes') + 1], str(5 * supervisor.TIB // 2))
            return child
        def run(command, **kwargs):
            calls.append(command)
            self.assertIn('timeout', kwargs)
            return Mock(returncode=3 if command[:2] == ['systemctl', 'is-active'] else 0)
        original_atomic = supervisor.atomic_json
        def atomic(path, value):
            if stop_record_error and 'stop_reason' in value: raise OSError('disk unavailable')
            original_atomic(path, value)
        deadline = dt.datetime.fromtimestamp(clock.wall() + remaining, dt.timezone.utc).isoformat()
        argv = ['supervisor', '--bucket', 'fixture', '--deadline', deadline,
                '--max-bytes', str(5 * supervisor.TIB // 2), '--stale-seconds', str(stale),
                '--work-dir', str(root), '--segments-dir', str(root), '--poweroff-on-exit']
        with contextlib.ExitStack() as stack:
            for obj, name, value in [
                (supervisor.sys, 'argv', argv),
                (supervisor.subprocess, 'Popen', Mock(side_effect=start)),
                (supervisor.subprocess, 'run', Mock(side_effect=run)),
                (supervisor.subprocess, 'check_output', Mock(side_effect=lambda cmd, **kw: container if 'RestartPolicy' in cmd[-1] else 'true')),
                (supervisor.shutil, 'disk_usage', Mock(return_value=Mock(free=101*1024**3))),
                (supervisor.time, 'time', clock.wall), (supervisor.time, 'monotonic', clock.monotonic),
                (supervisor.threading, 'Event', lambda: event),
                (supervisor.signal, 'signal', Mock()), (supervisor.os, 'killpg', Mock()),
                (supervisor, 'atomic_json', atomic),
            ]: stack.enter_context(patch.object(obj, name, value))
            if stop_record_error:
                with self.assertRaisesRegex(OSError, 'disk unavailable'): supervisor.main()
                result = None
            else:
                result = supervisor.main()
        return result, calls, clock

    def test_uploader_failure_stops_collector_before_poweroff_and_preserves_prior_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, calls, _ = self.run_supervisor(root, polls=[None, 7, 7], prior=True)
            self.assertEqual(result, 1)
            self.assertIn(['docker', 'start', 'js'], calls)
            self.assertLess(calls.index(['docker', 'stop', '--time', '60', 'js']), calls.index(['systemctl', 'poweroff']))
            status = json.loads((root / 'supervisor_status.json').read_text())
            self.assertEqual(status['stop_reason'], 'buffer_exited_7')
            run = root / 'runs' / status['run_id']
            self.assertEqual((run / 'previous_status.json').read_text(), '{"old":"buffer"}')
            self.assertEqual((run / 'previous_supervisor_status.json').read_text(), '{"old":"supervisor"}')
            self.assertEqual(json.loads((run / 'start.json').read_text())['max_bytes'], 5 * supervisor.TIB // 2)
            self.assertTrue((run / 'stop.json').exists())

    def test_created_container_waits_for_verified_ready_then_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            _, calls, _ = self.run_supervisor(Path(directory), container='created no', polls=[None, 7, 7])
            self.assertIn(['docker', 'start', 'js'], calls)

    def test_running_or_restart_enabled_container_rejected(self):
        for state in ['running no', 'restarting no', 'paused no', 'exited always']:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(RuntimeError, 'expected stopped collector'):
                    self.run_supervisor(Path(directory), container=state)

    def test_stale_status_stops_alive_uploader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, calls, clock = self.run_supervisor(root, stale=10)
            self.assertEqual(result, 1)
            self.assertEqual(clock.elapsed, 10)
            self.assertEqual(json.loads((root/'supervisor_status.json').read_text())['stop_reason'], 'buffer_status_stale')
            self.assertIn(['systemctl', 'poweroff'], calls)

    def test_prior_run_status_cannot_start_collector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'status.json').write_text(json.dumps({'run_id': 'old-run', 'reason': 'ready', 'pending_sealed': 0}))
            _, calls, _ = self.run_supervisor(root, heartbeat=False, stale=10)
            self.assertNotIn(['docker', 'start', 'js'], calls)

    def test_deadline_stops_alive_network_blocked_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, calls, clock = self.run_supervisor(root, remaining=160)
            self.assertEqual(result, 0)
            self.assertEqual(clock.elapsed, 10)
            self.assertEqual(json.loads((root/'supervisor_status.json').read_text())['stop_reason'], 'authorized_deadline')
            self.assertIn(['systemctl', 'poweroff'], calls)

    def test_failed_stop_status_write_still_powers_off(self):
        with tempfile.TemporaryDirectory() as directory:
            _, calls, _ = self.run_supervisor(Path(directory), polls=[7, 7], stop_record_error=True)
            self.assertIn(['systemctl', 'poweroff'], calls)

    def test_monotonic_deadline_survives_wall_clock_rollback(self):
        clock = Clock()
        with patch.object(supervisor.time, 'time', clock.wall), patch.object(supervisor.time, 'monotonic', clock.monotonic):
            guard = supervisor.Watchdog(clock.wall() + 160, 900, 'run')
            clock.advance(10)
            self.assertEqual(guard.stop_reason(1, 1800000160), 'authorized_deadline')

    def test_stale_or_other_run_heartbeat_cannot_refresh_watchdog(self):
        clock = Clock()
        with patch.object(supervisor.time, 'time', clock.wall), patch.object(supervisor.time, 'monotonic', clock.monotonic):
            guard = supervisor.Watchdog(clock.wall() + 3600, 10, 'run')
            self.assertTrue(guard.observe({'run_id': 'run'}, (1, 1, 1)))
            clock.advance(9)
            guard.observe({'run_id': 'run'}, (1, 1, 1))
            self.assertFalse(guard.observe({'run_id': 'old'}, (2, 2, 2)))
            clock.advance(1)
            self.assertEqual(guard.stop_reason(clock.wall(), clock.wall()+3500), 'buffer_status_stale')

    def test_shutdown_timeouts_kill_process_group_and_return(self):
        child = Mock(pid=12345)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired('buffer', 5), 0]
        def run(command, **kwargs):
            if command[:2] == ['docker', 'stop']: raise subprocess.TimeoutExpired(command, kwargs['timeout'])
            return Mock(returncode=0)
        with patch.object(supervisor.subprocess, 'run', side_effect=run) as invoke, patch.object(supervisor.os, 'killpg') as kill:
            errors = supervisor.stop_runtime(child)
        self.assertEqual(errors, ['collector_stop_TimeoutExpired'])
        self.assertEqual([call.args for call in kill.call_args_list], [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])
        self.assertTrue(all('timeout' in call.kwargs for call in invoke.call_args_list))

    def test_exited_uploader_leader_still_signals_its_verifier_group(self):
        child = Mock(pid=12345)
        child.poll.return_value = 7
        with patch.object(supervisor.subprocess, 'run', return_value=Mock(returncode=0)), patch.object(supervisor.os, 'killpg') as kill:
            self.assertEqual(supervisor.stop_runtime(child), [])
        self.assertEqual([call.args for call in kill.call_args_list], [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])

    def test_exclusive_owner_powers_off_on_expired_deadline_or_preflight_error(self):
        for error in [ValueError('expired deadline'), subprocess.TimeoutExpired('docker inspect', 10)]:
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as directory:
                argv = ['supervisor', '--bucket', 'fixture', '--deadline', '2020-01-01T00:00:00Z',
                        '--work-dir', directory, '--poweroff-on-exit']
                with patch.object(supervisor.sys, 'argv', argv), patch.object(supervisor, 'supervise', side_effect=error), \
                     patch.object(supervisor.subprocess, 'run', return_value=Mock(returncode=0)) as run:
                    with self.assertRaises(type(error)): supervisor.main()
                run.assert_called_once_with(['systemctl', 'poweroff'], check=True, timeout=10)

    def test_duplicate_supervisor_cannot_power_off_existing_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['supervisor', '--bucket', 'fixture', '--deadline', '2030-01-01T00:00:00Z',
                    '--work-dir', directory, '--poweroff-on-exit']
            with patch.object(supervisor.sys, 'argv', argv), \
                 patch.object(supervisor.fcntl, 'flock', side_effect=BlockingIOError('owned')), \
                 patch.object(supervisor.subprocess, 'run') as run:
                with self.assertRaises(BlockingIOError): supervisor.main()
            run.assert_not_called()

    def test_malformed_status_is_preserved_and_not_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            path.write_bytes(b'{partial')
            self.assertEqual(supervisor.status_snapshot(path), ({}, None))
            self.assertEqual(path.read_bytes(), b'{partial')


if __name__ == '__main__': unittest.main()
