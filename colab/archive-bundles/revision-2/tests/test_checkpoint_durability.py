"""Checkpoint object tests only: no model imports, training data, or inference."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_pipeline import checkpoints


class CheckpointDurabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.config = {"seed": 1729, "batch_size": 2}
        self.loaded = []

    @staticmethod
    def serialize(payload, path):
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path):
        self.loaded.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def checkpoint(self, step, config=None):
        path = self.directory / f"checkpoint-{step:07d}.pt"
        config = self.config if config is None else config
        checkpoints.save_checkpoint(
            path, {"step": step, "config": config, "state": {"counter": step}},
            step=step, config=config, serializer=self.serialize,
        )
        return path

    def steps(self, records, tail=b""):
        path = self.directory / "steps.jsonl"
        path.write_bytes(b"".join(json.dumps(record).encode() + b"\n" for record in records) + tail)
        return path

    def recover(self, **kwargs):
        return checkpoints.recover_checkpoint(
            self.directory, config=self.config, loader=self.load, **kwargs,
        )

    def assert_archived(self, result, name):
        recovery = Path(result.info["recovery_directory"])
        self.assertTrue((recovery / name).exists(), name)
        return recovery / name

    def test_save_publishes_verified_metadata_and_refuses_overwrite(self):
        path = self.checkpoint(2)
        metadata = json.loads(checkpoints.integrity_path(path).read_text())
        self.assertEqual(metadata["size_bytes"], len(path.read_bytes()))
        self.assertEqual(metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(metadata["config_sha256"], checkpoints.config_sha256(self.config))
        self.assertEqual(metadata["step"], 2)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.checkpoint(2)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.directory.glob(".*.partial")), [])

    def test_atomic_write_order_flushes_bytes_before_publishing_marker(self):
        events = []
        real_replace = checkpoints.os.replace
        real_fsync = checkpoints.os.fsync

        def record_replace(source, target):
            events.append(("replace", Path(target).name))
            return real_replace(source, target)

        def record_fsync(descriptor):
            events.append(("fsync", None))
            return real_fsync(descriptor)

        with patch.object(checkpoints.os, "replace", side_effect=record_replace), patch.object(
            checkpoints.os, "fsync", side_effect=record_fsync,
        ):
            path = self.checkpoint(1)
        publication = events.index(("replace", path.name))
        marker = events.index(("replace", checkpoints.integrity_path(path).name))
        self.assertIn(("fsync", None), events[:publication])
        self.assertIn(("fsync", None), events[publication + 1:marker])
        self.assertIn(("fsync", None), events[marker + 1:])

    def test_latest_valid_checkpoint_selected_numerically(self):
        self.checkpoint(9)
        latest = self.checkpoint(10)
        result = self.recover()
        self.assertEqual(result.path, latest)
        self.assertEqual(result.step, 10)
        self.assertEqual(self.loaded, [latest])
        self.assertFalse(result.info["log_complete_through_checkpoint"])

    def test_partial_serializer_failure_retains_then_archives_bytes(self):
        older = self.checkpoint(2)
        failed = self.directory / "checkpoint-0000003.pt"

        def interrupted(payload, path):
            path.write_bytes(b'{"step":')
            raise OSError("injected interrupted write")

        with self.assertRaisesRegex(OSError, "injected"):
            checkpoints.save_checkpoint(
                failed, {"step": 3, "config": self.config}, step=3,
                config=self.config, serializer=interrupted,
            )
        partial = list(self.directory.glob(".*.partial"))
        self.assertEqual(len(partial), 1)
        result = self.recover(requested=failed)
        self.assertEqual(result.path, older)
        self.assertEqual(self.assert_archived(result, partial[0].name).read_bytes(), b'{"step":')
        self.assertEqual(list(self.directory.glob(".*.partial")), [])
        self.assertEqual(self.loaded, [older])

    def test_marker_rename_failure_falls_back_without_loading_unverified_file(self):
        older = self.checkpoint(2)
        original_replace = checkpoints.os.replace

        def fail_marker(source, target):
            if str(target).endswith(".integrity.json"):
                raise OSError("injected marker interruption")
            return original_replace(source, target)

        with patch.object(checkpoints.os, "replace", side_effect=fail_marker):
            with self.assertRaisesRegex(OSError, "injected"):
                self.checkpoint(3)
        newest = self.directory / "checkpoint-0000003.pt"
        self.assertTrue(newest.exists())
        self.assertFalse(checkpoints.integrity_path(newest).exists())
        result = self.recover(requested=newest)
        self.assertEqual(result.path, older)
        self.assertTrue(result.info["fell_back"])
        self.assert_archived(result, newest.name)
        self.assertEqual(self.loaded, [older])
        # Archiving makes the failed step's name available for a fresh write.
        self.assertTrue(self.checkpoint(3).exists())

    def test_truncated_checkpoint_falls_back_and_archives_sidecar(self):
        older = self.checkpoint(2)
        newest = self.checkpoint(3)
        newest.write_bytes(newest.read_bytes()[:12])
        result = self.recover(requested=newest)
        self.assertEqual(result.path, older)
        self.assertTrue(result.info["fell_back"])
        self.assert_archived(result, newest.name)
        self.assert_archived(result, checkpoints.integrity_path(newest).name)
        self.assertEqual(self.loaded, [older])

    def test_wrong_digest_same_size_falls_back(self):
        older = self.checkpoint(2)
        newest = self.checkpoint(3)
        data = bytearray(newest.read_bytes())
        data[-2] ^= 1
        newest.write_bytes(data)
        result = self.recover()
        self.assertEqual(result.path, older)
        self.assert_archived(result, newest.name)
        self.assertEqual(self.loaded, [older])

    def test_loader_failure_with_verified_bytes_falls_back(self):
        older = self.checkpoint(2)
        newest = self.directory / "checkpoint-0000003.pt"
        checkpoints.save_checkpoint(
            newest, {"step": 3, "config": self.config}, step=3,
            config=self.config, serializer=lambda payload, path: path.write_bytes(b"not JSON"),
        )
        result = self.recover()
        self.assertEqual(result.path, older)
        self.assert_archived(result, newest.name)
        self.assertEqual(self.loaded, [newest, older])

    def test_valid_configuration_mismatch_is_refused_before_mutation(self):
        self.checkpoint(2)
        newest = self.checkpoint(3, {"seed": 99})
        log = self.steps([{"step": 1}, {"step": 2}, {"step": 3}])
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        with self.assertRaises(checkpoints.ConfigurationMismatchError):
            self.recover(requested=newest)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.directory.iterdir()})
        self.assertTrue(log.exists())
        self.assertEqual(self.loaded, [])

    def test_explicit_older_mismatched_request_is_not_silently_ignored(self):
        older = self.checkpoint(2, {"seed": 99})
        self.checkpoint(3)
        with self.assertRaises(checkpoints.ConfigurationMismatchError):
            self.recover(requested=older)
        self.assertFalse((self.directory / "recovery").exists())

    def test_corrupt_newer_file_is_not_archived_before_intact_mismatch_refusal(self):
        self.checkpoint(2, {"seed": 99})
        newest = self.checkpoint(3)
        newest.write_bytes(b"interrupted")
        with self.assertRaises(checkpoints.ConfigurationMismatchError):
            self.recover()
        self.assertEqual(newest.read_bytes(), b"interrupted")
        self.assertTrue(checkpoints.integrity_path(newest).exists())
        self.assertFalse((self.directory / "recovery").exists())

    def test_payload_configuration_mismatch_is_refused_before_mutation(self):
        path = self.checkpoint(2)
        bad_payload = {"step": 2, "config": {"seed": 99}}

        def incompatible_loader(_):
            return bad_payload

        with self.assertRaises(checkpoints.ConfigurationMismatchError):
            checkpoints.recover_checkpoint(
                self.directory, config=self.config, loader=incompatible_loader,
            )
        self.assertTrue(path.exists())
        self.assertFalse((self.directory / "recovery").exists())

    def test_legacy_checkpoint_without_integrity_is_never_loaded(self):
        path = self.directory / "checkpoint-0000001.pt"
        self.serialize({"step": 1, "config": self.config}, path)
        with self.assertRaises(checkpoints.NoValidCheckpointError) as raised:
            self.recover(requested=path)
        self.assertEqual(self.loaded, [])
        recovery = Path(raised.exception.info["recovery_directory"])
        self.assertTrue((recovery / path.name).exists())
        self.assertFalse(path.exists())

    def test_original_log_archived_and_tail_deduplicated_through_restored_step(self):
        self.checkpoint(3)
        records = [
            {"step": 1, "loss": 0.8}, {"step": 2, "loss": 0.7},
            {"step": 2, "loss": 0.7}, {"step": 3, "loss": 0.6},
            {"step": 4, "loss": 0.5},
        ]
        log = self.steps(records, tail=b'{"step":5')
        before = log.read_bytes()
        result = self.recover()
        self.assertEqual(self.assert_archived(result, "steps.jsonl").read_bytes(), before)
        repaired = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(repaired, [records[0], records[1], records[3]])
        self.assertEqual(result.info["log_last_step"], 3)
        self.assertTrue(result.info["log_complete_through_checkpoint"])
        self.assertEqual(result.info["log_duplicate_records_removed"], 1)
        persisted = json.loads((Path(result.info["recovery_directory"]) / "recovery.json").read_text())
        self.assertEqual(persisted, result.info)

    def test_partial_log_tail_preserves_existing_contiguous_records_only(self):
        self.checkpoint(3)
        log = self.steps([{"step": 1}, {"step": 2}], tail=b'{"step":3')
        result = self.recover()
        self.assertEqual([json.loads(line)["step"] for line in log.read_text().splitlines()], [1, 2])
        self.assertFalse(result.info["log_complete_through_checkpoint"])
        self.assertEqual(result.info["log_last_step"], 2)
        self.assertEqual(result.info["log_stopped_reason"], "invalid_json_at_line_3")

    def test_log_gap_does_not_fabricate_or_reorder_history(self):
        self.checkpoint(3)
        log = self.steps([{"step": 1}, {"step": 3}, {"step": 2}])
        result = self.recover()
        self.assertEqual([json.loads(line)["step"] for line in log.read_text().splitlines()], [1])
        self.assertEqual(result.info["log_stopped_reason"], "noncontiguous_step_at_line_2")

    def test_repeated_recovery_keeps_single_contiguous_log_history(self):
        self.checkpoint(2)
        log = self.steps([{"step": 1}, {"step": 2}, {"step": 2}])
        first = self.recover()
        second = self.recover()
        self.assertNotEqual(first.info["recovery_directory"], second.info["recovery_directory"])
        self.assertEqual([json.loads(line)["step"] for line in log.read_text().splitlines()], [1, 2])
        self.assertTrue(second.info["log_complete_through_checkpoint"])

    def test_log_replacement_failure_preserves_original_and_new_partial(self):
        self.checkpoint(2)
        log = self.steps([{"step": 1}, {"step": 2}, {"step": 3}])
        before = log.read_bytes()
        real_replace = checkpoints.os.replace

        def fail_log_replace(source, target):
            if Path(target) == log:
                raise OSError("injected log interruption")
            return real_replace(source, target)

        with patch.object(checkpoints.os, "replace", side_effect=fail_log_replace):
            with self.assertRaisesRegex(OSError, "injected"):
                self.recover()
        self.assertEqual(log.read_bytes(), before)
        self.assertEqual(len(list(self.directory.glob(".steps.jsonl.*.partial"))), 1)
        archives = list((self.directory / "recovery").glob("*/steps.jsonl"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_bytes(), before)
        result = self.recover()
        self.assertEqual(result.step, 2)
        self.assertEqual(list(self.directory.glob(".steps.jsonl.*.partial")), [])

    def test_orphan_sidecar_and_partial_do_not_block_intact_checkpoint(self):
        self.checkpoint(2)
        orphan = self.directory / "checkpoint-0000003.pt.integrity.json"
        orphan.write_text("{}")
        partial = self.directory / ".checkpoint-0000004.pt.integrity.json.example.partial"
        partial.write_text("{")
        result = self.recover()
        self.assertEqual(result.step, 2)
        self.assert_archived(result, orphan.name)
        self.assert_archived(result, partial.name)


if __name__ == "__main__":
    unittest.main()
