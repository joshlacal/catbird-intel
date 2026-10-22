"""Synthetic supervisor tests: no GCS, Drive, OAuth refresh, or VM mutation."""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import warnings
from unittest import mock

import gcs_drive_retained_continuous as continuous
from gcs_drive_drain import Conflict
from gcs_drive_retained_queue import build_snapshot
from test_gcs_drive_retained_queue import FakeCloud, fixture, PAYLOAD_NAME, PAYLOAD_GEN


ACCOUNT = "synthetic-permission-id"
WRITER = "synthetic-sole-writer"
RUN = "a" * 32
MANIFEST = "b" * 64


class ContinuousTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.db = self.root / "authority.sqlite"
        self.db.write_bytes(b"synthetic-ledger-placeholder")
        self.oauth = self.root / "oauth.json"
        self.oauth.write_bytes(b"synthetic-only-not-a-credential")
        self.now = time.time()
        envelope = {
            "schema": continuous.ENVELOPE_SCHEMA,
            "account_id": ACCOUNT, "writer_id": WRITER,
            "drive_root": continuous.DRIVE_ROOT,
            "bucket": continuous.BUCKET,
            "authority_db": str(self.db.resolve()),
            "owner_reference": "synthetic-owner-decision",
            "issued_at": self.now - 10,
            "account_identity_verified_at": self.now - 100,
            "manual_colab_hold_until": continuous.STOP_AT,
            "netcup_nonwriter_hold_until": continuous.STOP_AT,
            "other_writer_hold_until": continuous.STOP_AT,
            "oauth_file_sha256": hashlib.sha256(self.oauth.read_bytes()).hexdigest(),
        }
        self.envelope = self.root / "standing.json"
        self.envelope.write_text(json.dumps(envelope, sort_keys=True, separators=(",", ":")))
        self.digest = hashlib.sha256(self.envelope.read_bytes()).hexdigest()

    def _standing(self):
        return continuous.standing_envelope(
            self.envelope, self.digest, authority_db=self.db, oauth_file=self.oauth,
            account_id=ACCOUNT, writer_id=WRITER, now=self.now,
            require_root=False)

    def _account(self):
        return {"diagnostic_imported": True,
                "admission_mode": "retained_pair_observation",
                "history_proven": False, "nonterminal_run_id": None,
                "nonterminal_status": None, "unresolved_attempts": 0,
                "mapping_partition_remaining_bytes": 32 * (1 << 20),
                "cooldown_until": 0, "last_clock": 0}

    def _run(self, *, snapshots, outcomes, max_cycles=4):
        snapshots = iter(snapshots)
        outcomes = iter(outcomes)
        calls = []
        def snapshot_fn(**kwargs):
            calls.append("snapshot")
            rows = next(snapshots)
            value = {"schema": continuous.SNAPSHOT_SCHEMA, "candidates": rows}
            return value, json.dumps(value, sort_keys=True).encode()
        def runner_fn(**kwargs):
            calls.append("runner")
            return next(outcomes)
        with mock.patch.object(continuous.TransferAuthority, "inspect_retained_account",
                               return_value=self._account()), \
             mock.patch.object(continuous, "STOP_AT", self.now + 7200), \
             mock.patch.object(continuous, "CONTINUOUS_ROOT", self.work.resolve()), \
             mock.patch("gcs_drive_retained_continuous.sys.platform", "linux"):
            result = continuous.run_cycles(
                envelope_path=self.envelope, envelope_sha256=self.digest,
                authority_db=self.db, oauth_file=self.oauth,
                account_id=ACCOUNT, writer_id=WRITER, work_root=self.work,
                snapshot_fn=snapshot_fn, runner_fn=runner_fn,
                local_check=lambda: {"legacy_units": {}, "processes_inspected": 7},
                clock=lambda: self.now, sleep=lambda _: calls.append("idle"),
                capture_dir=self.work, cache_check=lambda *_: {},
                require_root=False, max_cycles=max_cycles)
        return result, calls

    def test_standing_envelope_binds_exact_oauth_and_owner_hold(self):
        envelope, digest = self._standing()
        self.assertEqual(digest, self.digest)
        self.assertEqual(envelope["account_id"], ACCOUNT)
        self.oauth.write_bytes(b"changed-synthetic-oauth")
        with self.assertRaisesRegex(Conflict, "OAuth bytes differ"):
            self._standing()

    def test_standing_envelope_rejects_stale_nonlocal_hold(self):
        value = json.loads(self.envelope.read_text())
        value["netcup_nonwriter_hold_until"] = self.now + 10
        self.envelope.write_text(json.dumps(value, sort_keys=True))
        self.digest = hashlib.sha256(self.envelope.read_bytes()).hexdigest()
        with self.assertRaisesRegex(Conflict, "nonlocal exclusion"):
            self._standing()

    def test_standing_envelope_one_hour_cap_and_pinned_renewal(self):
        value = json.loads(self.envelope.read_text())
        issued = value["issued_at"]
        # Even a multi-day claimed hold cannot authorize another 20-minute
        # child when less than that remains in the one-hour reviewed window.
        with self.assertRaisesRegex(Conflict, "nonlocal exclusion"):
            continuous.standing_envelope(
                self.envelope, self.digest, authority_db=self.db,
                oauth_file=self.oauth, account_id=ACCOUNT, writer_id=WRITER,
                now=issued + continuous.MAX_STANDING_SECONDS -
                    continuous.MIN_WINDOW_SECONDS + 1,
                require_root=False)
        # A changed file cannot renew the running supervisor's pinned digest.
        value["issued_at"] = self.now
        value["account_identity_verified_at"] = self.now
        self.envelope.write_text(json.dumps(value, sort_keys=True))
        with self.assertRaisesRegex(Conflict, "bytes differ from pinned review"):
            self._standing()

    def test_local_writer_preflight_rejects_active_unit_or_process(self):
        healthy = lambda _: {"ActiveState": "inactive", "UnitFileState": "disabled"}
        report = continuous.local_writer_preflight(unit_state=healthy,
                                                    proc_cmdlines=lambda: [b"collector.py"])
        self.assertEqual(report["processes_inspected"], 1)
        with self.assertRaisesRegex(Conflict, "not held"):
            continuous.local_writer_preflight(
                unit_state=lambda _: {"ActiveState": "active", "UnitFileState": "enabled"},
                proc_cmdlines=lambda: [])
        with self.assertRaisesRegex(Conflict, "writer process"):
            continuous.local_writer_preflight(unit_state=healthy,
                                                proc_cmdlines=lambda: [b"rclone upload"])

    def test_evidence_binds_fresh_local_facts_and_pinned_manifest(self):
        value = continuous.review_evidence(
            {"release_id": RUN, "manifest_sha256": MANIFEST},
            account_id=ACCOUNT, writer_id=WRITER,
            envelope_sha256=self.digest, local_report={"processes_inspected": 7},
            exclusion_until=continuous.STOP_AT, now=self.now)
        self.assertEqual(value["reviewed_manifest_sha256"], MANIFEST)
        self.assertEqual(value["standing_envelope_sha256"], self.digest)
        self.assertRegex(value["evidence_sha256"], r"^[0-9a-f]{64}$")

    def test_distinct_successful_batches_continue_without_human_refresh(self):
        first = {"release_id": RUN, "manifest_sha256": MANIFEST}
        second = {"release_id": "c" * 32, "manifest_sha256": "d" * 64}
        result, calls = self._run(
            snapshots=[[first], [second]],
            outcomes=[(0, {"state": "completed", "pairs_closed": 1}),
                      (0, {"state": "completed", "pairs_closed": 1})],
            max_cycles=2)
        self.assertEqual(result["pairs_closed"], 2)
        self.assertEqual(result["reason"], "cycle_bound_reached")
        self.assertEqual(calls, ["snapshot", "runner", "snapshot", "runner"])

    def test_only_fresh_evidence_stop_can_resnapshot_automatically(self):
        row = {"release_id": RUN, "manifest_sha256": MANIFEST}
        result, calls = self._run(
            snapshots=[[row], [row]],
            outcomes=[(75, {"state": "awaiting_fresh_evidence", "pairs_closed": 0}),
                      (0, {"state": "completed", "pairs_closed": 1})],
            max_cycles=2)
        self.assertEqual(result["pairs_closed"], 1)
        self.assertEqual(calls, ["snapshot", "runner", "snapshot", "runner"])

    def test_actual_selector_skips_source_after_first_closed_batch(self):
        raw, payload_ref, receipt_ref = fixture()
        cloud = FakeCloud(raw, payload_ref, receipt_ref)
        ledger_rows = []
        runner_calls = []
        def snapshot_fn(**kwargs):
            selected = build_snapshot(
                cloud, account_id=ACCOUNT, work=kwargs["work"],
                ledger_pairs=ledger_rows, max_scan=4, max_pairs=1,
                max_reserved_bytes=8 << 20)
            value = {"schema": continuous.SNAPSHOT_SCHEMA,
                     "candidates": [{"pair_key": item.pair_key,
                                     "release_id": item.release_id,
                                     "manifest_path": str(item.manifest_path),
                                     "manifest_sha256": item.manifest_sha256,
                                     "reserved_bytes": item.reserved_bytes}
                                    for item in selected]}
            return value, json.dumps(value, sort_keys=True).encode()
        def runner_fn(**kwargs):
            runner_calls.append(kwargs["snapshot_path"])
            candidate = json.loads(kwargs["snapshot_path"].read_text())["candidates"][0]
            ledger_rows.append({"pair_key": candidate["pair_key"], "status": "closed",
                                "bucket": continuous.BUCKET,
                                "payload_object": PAYLOAD_NAME,
                                "payload_generation": PAYLOAD_GEN})
            return 0, {"state": "completed", "pairs_closed": 1}
        with mock.patch.object(continuous.TransferAuthority, "inspect_retained_account",
                               return_value=self._account()), \
             mock.patch.object(continuous, "STOP_AT", self.now + 7200), \
             mock.patch.object(continuous, "CONTINUOUS_ROOT", self.work.resolve()), \
             mock.patch("gcs_drive_retained_continuous.sys.platform", "linux"):
            result = continuous.run_cycles(
                envelope_path=self.envelope, envelope_sha256=self.digest,
                authority_db=self.db, oauth_file=self.oauth,
                account_id=ACCOUNT, writer_id=WRITER, work_root=self.work,
                snapshot_fn=snapshot_fn, runner_fn=runner_fn,
                local_check=lambda: {}, clock=lambda: self.now,
                sleep=lambda _: None, capture_dir=self.work,
                cache_check=lambda *_: {}, require_root=False, max_cycles=2)
        self.assertEqual(result["pairs_closed"], 1)
        self.assertEqual(len(runner_calls), 1)
        self.assertEqual(cloud.download_calls, 1)

    def test_quota_stop_does_not_resnapshot_or_retry(self):
        row = {"release_id": RUN, "manifest_sha256": MANIFEST}
        result, calls = self._run(
            snapshots=[[row]],
            outcomes=[(75, {"state": "stopped", "stop_reason": "drive_http_403",
                            "pairs_closed": 0})])
        self.assertEqual(result["reason"], "drive_http_403")
        self.assertEqual(calls, ["snapshot", "runner"])

    def test_false_completed_count_stops_before_next_snapshot(self):
        row = {"release_id": RUN, "manifest_sha256": MANIFEST}
        with self.assertRaisesRegex(Conflict, "completed without all"):
            self._run(snapshots=[[row]],
                      outcomes=[(0, {"state": "completed", "pairs_closed": 0})])

    def test_malformed_candidate_cannot_write_outside_evidence_dir(self):
        row = {"release_id": "../escape", "manifest_sha256": MANIFEST}
        with self.assertRaisesRegex(Conflict, "candidate identity"):
            self._run(snapshots=[[row]], outcomes=[])
        self.assertFalse((self.root / "escape.json").exists())

    def test_snapshot_cache_preserves_capture_floor_and_cumulative_budget(self):
        cache = self.work / "cycle-prior"
        cache.mkdir()
        (cache / "receipt.json").write_bytes(b"synthetic")
        size = continuous.CAPTURE_RESTART_FLOOR_BYTES + \
            continuous.SNAPSHOT_RESERVATION_BYTES - 1
        statvfs = mock.Mock(f_bavail=size, f_frsize=1)
        with mock.patch.object(continuous.os, "statvfs", return_value=statvfs):
            with self.assertRaisesRegex(Conflict, "restart headroom"):
                continuous.check_snapshot_cache(self.work, self.work)
        statvfs.f_bavail = 10 * (1 << 30)
        with mock.patch.object(continuous.os, "statvfs", return_value=statvfs), \
             mock.patch.object(continuous, "MAX_CACHE_BYTES", 8):
            with self.assertRaisesRegex(Conflict, "Cumulative snapshot cache"):
                continuous.check_snapshot_cache(self.work, self.work)

    def test_backward_wall_jump_cannot_extend_monotonic_stop(self):
        mutable = {"wall": self.now, "mono": 0.0}
        snapshots = []
        def snapshot_fn(**kwargs):
            snapshots.append("snapshot")
            mutable["wall"] -= 3600
            mutable["mono"] += 1500
            return {"schema": continuous.SNAPSHOT_SCHEMA, "candidates": []}, b'{}'
        with mock.patch.object(continuous.TransferAuthority, "inspect_retained_account",
                               return_value=self._account()), \
             mock.patch.object(continuous, "STOP_AT", self.now + 1500), \
             mock.patch.object(continuous, "CONTINUOUS_ROOT", self.work.resolve()), \
             mock.patch("gcs_drive_retained_continuous.sys.platform", "linux"):
            result = continuous.run_cycles(
                envelope_path=self.envelope, envelope_sha256=self.digest,
                authority_db=self.db, oauth_file=self.oauth,
                account_id=ACCOUNT, writer_id=WRITER, work_root=self.work,
                snapshot_fn=snapshot_fn,
                runner_fn=lambda **_: self.fail("runner must not start"),
                local_check=lambda: {}, clock=lambda: mutable["wall"],
                monotonic=lambda: mutable["mono"],
                capture_dir=self.work, cache_check=lambda *_: {},
                require_root=False, max_cycles=2)
        self.assertEqual(result["reason"], "capture_stop_approaching")
        self.assertEqual(snapshots, ["snapshot"])

    def test_earlier_hold_rollback_during_snapshot_stops_naturally(self):
        value = json.loads(self.envelope.read_text())
        value["netcup_nonwriter_hold_until"] = self.now + 2000
        self.envelope.write_text(json.dumps(value, sort_keys=True))
        self.digest = hashlib.sha256(self.envelope.read_bytes()).hexdigest()
        mutable = {"wall": self.now, "mono": 0.0}
        def snapshot_fn(**kwargs):
            mutable["wall"] -= 3600
            mutable["mono"] += 800  # 1,200 seconds remain on the shorter hold.
            kwargs["monitor"]()
            self.fail("snapshot must stop before returning")
        with mock.patch.object(continuous.TransferAuthority, "inspect_retained_account",
                               return_value=self._account()), \
             mock.patch.object(continuous, "STOP_AT", self.now + 7200), \
             mock.patch.object(continuous, "CONTINUOUS_ROOT", self.work.resolve()), \
             mock.patch("gcs_drive_retained_continuous.sys.platform", "linux"):
            result = continuous.run_cycles(
                envelope_path=self.envelope, envelope_sha256=self.digest,
                authority_db=self.db, oauth_file=self.oauth,
                account_id=ACCOUNT, writer_id=WRITER, work_root=self.work,
                snapshot_fn=snapshot_fn,
                runner_fn=lambda **_: self.fail("runner must not start"),
                local_check=lambda: {}, clock=lambda: mutable["wall"],
                monotonic=lambda: mutable["mono"], capture_dir=self.work,
                cache_check=lambda *_: {}, require_root=False, max_cycles=2)
        self.assertEqual(result["reason"], "standing_exclusion_approaching")

    def test_earlier_hold_rollback_before_runner_stops_naturally(self):
        value = json.loads(self.envelope.read_text())
        value["manual_colab_hold_until"] = self.now + 2000
        self.envelope.write_text(json.dumps(value, sort_keys=True))
        self.digest = hashlib.sha256(self.envelope.read_bytes()).hexdigest()
        mutable = {"wall": self.now, "mono": 0.0}
        row = {"release_id": RUN, "manifest_sha256": MANIFEST}
        def snapshot_fn(**kwargs):
            value = {"schema": continuous.SNAPSHOT_SCHEMA, "candidates": [row]}
            return value, json.dumps(value, sort_keys=True).encode()
        def local_check():
            mutable["wall"] -= 3600
            mutable["mono"] += 800
            return {"processes_inspected": 1}
        with mock.patch.object(continuous.TransferAuthority, "inspect_retained_account",
                               return_value=self._account()), \
             mock.patch.object(continuous, "STOP_AT", self.now + 7200), \
             mock.patch.object(continuous, "CONTINUOUS_ROOT", self.work.resolve()), \
             mock.patch("gcs_drive_retained_continuous.sys.platform", "linux"):
            result = continuous.run_cycles(
                envelope_path=self.envelope, envelope_sha256=self.digest,
                authority_db=self.db, oauth_file=self.oauth,
                account_id=ACCOUNT, writer_id=WRITER, work_root=self.work,
                snapshot_fn=snapshot_fn,
                runner_fn=lambda **_: self.fail("runner must not start"),
                local_check=local_check, clock=lambda: mutable["wall"],
                monotonic=lambda: mutable["mono"], capture_dir=self.work,
                cache_check=lambda *_: {}, require_root=False, max_cycles=2)
        self.assertEqual(result["reason"], "standing_exclusion_approaching")

    def test_backward_wall_jump_cannot_extend_one_hour_exclusion(self):
        mutable = {"wall": self.now, "mono": 0.0}
        snapshots = []
        def snapshot_fn(**kwargs):
            snapshots.append("snapshot")
            return {"schema": continuous.SNAPSHOT_SCHEMA, "candidates": []}, b'{}'
        def advance_during_idle(_seconds):
            mutable["wall"] -= 3600
            mutable["mono"] += continuous.MAX_STANDING_SECONDS - 1200
        with mock.patch.object(continuous.TransferAuthority, "inspect_retained_account",
                               return_value=self._account()), \
             mock.patch.object(continuous, "STOP_AT", self.now + 7200), \
             mock.patch.object(continuous, "CONTINUOUS_ROOT", self.work.resolve()), \
             mock.patch("gcs_drive_retained_continuous.sys.platform", "linux"):
            result = continuous.run_cycles(
                envelope_path=self.envelope, envelope_sha256=self.digest,
                authority_db=self.db, oauth_file=self.oauth,
                account_id=ACCOUNT, writer_id=WRITER, work_root=self.work,
                snapshot_fn=snapshot_fn,
                runner_fn=lambda **_: self.fail("runner must not start"),
                local_check=lambda: {}, clock=lambda: mutable["wall"],
                monotonic=lambda: mutable["mono"], sleep=advance_during_idle,
                capture_dir=self.work, cache_check=lambda *_: {},
                require_root=False, max_cycles=3)
        self.assertEqual(result["reason"], "standing_exclusion_approaching")
        self.assertEqual(snapshots, ["snapshot"])

    def test_snapshot_monitor_stops_when_capture_disk_changes_mid_scan(self):
        checks = []
        def cache_check(*_):
            checks.append("check")
            if len(checks) > 1:
                raise Conflict("synthetic capture floor breach")
            return {}
        def snapshot_fn(**kwargs):
            kwargs["monitor"]()
            self.fail("scan must stop before returning")
        with mock.patch.object(continuous.TransferAuthority, "inspect_retained_account",
                               return_value=self._account()), \
             mock.patch.object(continuous, "STOP_AT", self.now + 7200), \
             mock.patch.object(continuous, "CONTINUOUS_ROOT", self.work.resolve()), \
             mock.patch("gcs_drive_retained_continuous.sys.platform", "linux"):
            with self.assertRaisesRegex(Conflict, "capture floor breach"):
                continuous.run_cycles(
                    envelope_path=self.envelope, envelope_sha256=self.digest,
                    authority_db=self.db, oauth_file=self.oauth,
                    account_id=ACCOUNT, writer_id=WRITER, work_root=self.work,
                    snapshot_fn=snapshot_fn,
                    runner_fn=lambda **_: self.fail("runner must not start"),
                    local_check=lambda: {}, clock=lambda: self.now,
                    capture_dir=self.work, cache_check=cache_check,
                    require_root=False, max_cycles=2)
        self.assertEqual(checks, ["check", "check"])

    def test_bounded_child_closes_stdout_on_timeout_or_monitor_failure(self):
        command = [sys.executable, "-c", "import time; time.sleep(10)"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with mock.patch("gcs_drive_retained_continuous.sys.platform", "linux"), \
                 mock.patch.object(continuous, "_arm_parent_death", lambda _: None):
                with self.assertRaises(subprocess.TimeoutExpired):
                    continuous._run_bounded(command, timeout=0.05)
                with self.assertRaisesRegex(Conflict, "synthetic monitor stop"):
                    continuous._run_bounded(
                        command, timeout=0.05,
                        monitor=lambda: (_ for _ in ()).throw(
                            Conflict("synthetic monitor stop")))
            gc.collect()
        self.assertFalse([item for item in caught
                          if issubclass(item.category, ResourceWarning)])

    def test_standing_envelope_is_24_hours(self):
        self.assertEqual(continuous.MAX_STANDING_SECONDS, 86400)

    def test_renew_standing_envelope_checks_single_writer_conditions_and_chains_past_initial_24h(self):
        healthy_unit = lambda _: {"ActiveState": "inactive", "UnitFileState": "disabled"}
        canonical_cmd = (
            f"python3 gcs_drive_retained_continuous.py "
            f"--account-id {ACCOUNT} --writer-id {WRITER} "
            f"--authority-db {self.db.resolve()}"
        ).encode()
        with mock.patch.object(continuous.TransferAuthority, "inspect_retained_account",
                               return_value=self._account()):
            renewed_1, digest_1 = continuous.renew_standing_envelope(
                self.envelope, self.digest, authority_db=self.db,
                oauth_file=self.oauth, account_id=ACCOUNT, writer_id=WRITER,
                now=self.now + 12 * 3600, unit_state=healthy_unit,
                proc_cmdlines=lambda: [canonical_cmd], require_root=False)
            self.assertEqual(renewed_1["root_envelope_sha256"], self.digest)
            self.assertEqual(renewed_1["renewed_from_sha256"], self.digest)
            self.assertEqual(renewed_1["issued_at"], self.now + 12 * 3600)

            # Second renewal at +30h (past initial 24h window, within first renewal's 24h window)
            renewed_2, digest_2 = continuous.renew_standing_envelope(
                self.envelope, self.digest, authority_db=self.db,
                oauth_file=self.oauth, account_id=ACCOUNT, writer_id=WRITER,
                now=self.now + 30 * 3600, unit_state=healthy_unit,
                proc_cmdlines=lambda: [canonical_cmd], require_root=False)
            self.assertEqual(renewed_2["root_envelope_sha256"], self.digest)
            self.assertEqual(renewed_2["renewed_from_sha256"], digest_1)
            self.assertNotEqual(digest_1, digest_2)

            # Reject foreign writer process
            with self.assertRaisesRegex(Conflict, "writer process"):
                continuous.renew_standing_envelope(
                    self.envelope, self.digest, authority_db=self.db,
                    oauth_file=self.oauth, account_id=ACCOUNT, writer_id=WRITER,
                    now=self.now + 31 * 3600, unit_state=healthy_unit,
                    proc_cmdlines=lambda: [b"rclone copy"], require_root=False)

            # Reject canonical script bound to a different writer_id
            foreign_retained = (
                f"python3 gcs_drive_retained_runner.py "
                f"--account-id {ACCOUNT} --writer-id other-writer "
                f"--authority-db {self.db.resolve()}"
            ).encode()
            with self.assertRaisesRegex(Conflict, "writer process"):
                continuous.renew_standing_envelope(
                    self.envelope, self.digest, authority_db=self.db,
                    oauth_file=self.oauth, account_id=ACCOUNT, writer_id=WRITER,
                    now=self.now + 31 * 3600, unit_state=healthy_unit,
                    proc_cmdlines=lambda: [foreign_retained], require_root=False)

        # Reject unresolved ledger attempts
        bad_account = dict(self._account(), unresolved_attempts=1)
        with mock.patch.object(continuous.TransferAuthority, "inspect_retained_account",
                               return_value=bad_account):
            with self.assertRaisesRegex(Conflict, "unresolved"):
                continuous.renew_standing_envelope(
                    self.envelope, self.digest, authority_db=self.db,
                    oauth_file=self.oauth, account_id=ACCOUNT, writer_id=WRITER,
                    now=self.now + 31 * 3600, unit_state=healthy_unit,
                    proc_cmdlines=lambda: [], require_root=False)

    def test_run_cycles_adopts_renewed_standing_envelope_without_restart(self):
        mutable = {"wall": self.now, "mono": 0.0}
        first = {"release_id": RUN, "manifest_sha256": MANIFEST}
        second = {"release_id": "c" * 32, "manifest_sha256": "d" * 64}
        snapshots = iter([[first], [second]])
        healthy_unit = lambda _: {"ActiveState": "inactive", "UnitFileState": "disabled"}

        def snapshot_fn(**_kwargs):
            rows = next(snapshots)
            value = {"schema": continuous.SNAPSHOT_SCHEMA, "candidates": rows}
            return value, json.dumps(value, sort_keys=True).encode()

        calls = []
        def runner_fn(**_kwargs):
            calls.append("runner")
            if len(calls) == 1:
                # Advance 23.5 hours (within initial 24h, leaving <1290s without renewal)
                mutable["wall"] += 23 * 3600 + 1800
                mutable["mono"] += 23 * 3600 + 1800
                continuous.renew_standing_envelope(
                    self.envelope, self.digest, authority_db=self.db,
                    oauth_file=self.oauth, account_id=ACCOUNT, writer_id=WRITER,
                    now=mutable["wall"], stop_at=self.now + 72 * 3600,
                    unit_state=healthy_unit, proc_cmdlines=lambda: [],
                    require_root=False)
            return 0, {"state": "completed", "pairs_closed": 1}

        with mock.patch.object(continuous.TransferAuthority, "inspect_retained_account",
                               return_value=self._account()), \
             mock.patch.object(continuous, "STOP_AT", self.now + 72 * 3600), \
             mock.patch.object(continuous, "CONTINUOUS_ROOT", self.work.resolve()), \
             mock.patch("gcs_drive_retained_continuous.sys.platform", "linux"):
            result = continuous.run_cycles(
                envelope_path=self.envelope, envelope_sha256=self.digest,
                authority_db=self.db, oauth_file=self.oauth,
                account_id=ACCOUNT, writer_id=WRITER, work_root=self.work,
                snapshot_fn=snapshot_fn, runner_fn=runner_fn,
                local_check=lambda: {"legacy_units": {}, "processes_inspected": 2},
                clock=lambda: mutable["wall"], monotonic=lambda: mutable["mono"],
                sleep=lambda _: None, capture_dir=self.work,
                cache_check=lambda *_: {}, require_root=False, max_cycles=2,
                stop_at=self.now + 72 * 3600)
        self.assertEqual(result["pairs_closed"], 2)
        self.assertEqual(result["reason"], "cycle_bound_reached")
        status_lines = [
            json.loads(line)
            for line in (self.work / "continuous_status.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.assertTrue(any(row["state"] == "standing_renewed" for row in status_lines))


if __name__ == "__main__":
    unittest.main()

