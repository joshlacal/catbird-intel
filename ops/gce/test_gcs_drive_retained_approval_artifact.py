"""Synthetic local approval-artifact checks; no cloud, release, or deletion."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gcs_drive_drain import Conflict
from gcs_drive_retained_approval_artifact import (
    OwnerPinnedApproval, SCOPE, _create_owned_file,
    create_owner_artifact_after_human_yes,
)
from gcs_drive_retained_cache_release import APPROVAL_SCHEMA
from gcs_drive_retained_overflow import _canonical
from gcs_drive_retained_runner import STOP_AT


RUN = "a" * 32
PAIR = "e" * 64
POLICY = "b" * 64
REFERENCE = "Josh-authorization-thread-reference"


class ApprovalArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "private-evidence"
        self.root.mkdir(mode=0o700)
        self.root.chmod(0o700)
        self.path = self.root / ("approval-" + RUN + "-" + "f" * 32 + ".json")
        self.inventory = {
            "run_id": RUN, "pair_key": PAIR,
            "eligible_for_approval": True, "gcs_retained": True,
            "drive_media_verified": True, "current_cache_state": "present",
            "stage": {
                "payload": {"path": "/data/gcs-drive-retained-" + RUN + "/stage/" + PAIR + "/payload"},
                "receipt": {"path": "/data/gcs-drive-retained-" + RUN + "/stage/" + PAIR + "/receipt"},
            },
        }
        self.approval = {
            "schema": APPROVAL_SCHEMA, "approved_by": "Josh",
            "approval_reference": REFERENCE,
            "authorization_scope": SCOPE,
            "policy_sha256": POLICY,
            "inventory_sha256": hashlib.sha256(_canonical(self.inventory)).hexdigest(),
            "run_id": RUN, "pair_key": PAIR,
            "paths": [self.inventory["stage"][kind]["path"]
                      for kind in ("payload", "receipt")],
            "retain_gcs": True, "retain_drive": True,
            "issued_at": STOP_AT - 7200, "valid_until": STOP_AT - 1800,
        }
        self.now = STOP_AT - 3600
        self._write()

    def _write(self, raw=None):
        self.path.write_bytes(_canonical(self.approval) if raw is None else raw)
        self.path.chmod(0o600)
        self.pinned = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.verifier = OwnerPinnedApproval(
            path=self.path, root=self.root, expected_sha256=self.pinned,
            inventory=self.inventory, authorization_reference=REFERENCE,
            authorization_epoch=STOP_AT - 8000,
            expected_policy_sha256=POLICY)

    def _verify_fixture(self):
        with mock.patch.object(self.verifier, "_read_root_file",
                               side_effect=lambda: self.verifier._read_owned_file(os.getuid())), \
             mock.patch("gcs_drive_retained_approval_artifact.time.time", return_value=self.now):
            return self.verifier.verify(self.approval)

    def test_exact_private_artifact_returns_v2_proof_shape(self):
        proof = self._verify_fixture()
        self.assertEqual(proof, {
            "authenticated": True, "approved_by": "Josh",
            "approval_reference": REFERENCE,
            "inventory_sha256": self.approval["inventory_sha256"],
        })

    def test_production_reader_requires_root_owner_even_in_fixture(self):
        if os.getuid() == 0:
            self.skipTest("test requires nonroot synthetic fixture")
        with self.assertRaisesRegex(Conflict, "root-owned"):
            self.verifier._read_root_file()

    def test_rejects_tamper_after_hash_pin(self):
        self.path.write_bytes(self.path.read_bytes() + b" ")
        with self.assertRaisesRegex(Conflict, "pinned hash"):
            self.verifier._read_owned_file(os.getuid())

    def test_rejects_world_readable_or_symlinked_file(self):
        self.path.chmod(0o644)
        with self.assertRaisesRegex(Conflict, "private evidence"):
            self.verifier._read_owned_file(os.getuid())
        other = self.root / "other"
        other.write_bytes(_canonical(self.approval))
        self.path.rename(self.root / "original")
        self.path.symlink_to(other)
        with self.assertRaises(OSError):
            self.verifier._read_owned_file(os.getuid())

    def test_rejects_root_directory_that_is_not_private(self):
        self.root.chmod(0o755)
        with self.assertRaisesRegex(Conflict, "private storage"):
            self.verifier._read_owned_file(os.getuid())

    def test_rejects_wrong_inventory_scope_or_authorization(self):
        for key, replacement in (
            ("paths", ["/data/other", self.approval["paths"][1]]),
            ("inventory_sha256", "c" * 64),
            ("authorization_scope", "all-local-files"),
            ("approval_reference", "unreviewed"),
            ("retain_gcs", False),
            ("retain_drive", False),
            ("policy_sha256", "d" * 64),
        ):
            with self.subTest(key=key):
                self.approval[key] = replacement
                self._write()
                with self.assertRaisesRegex(Conflict, "does not bind"):
                    self._verify_fixture()
                self.approval[key] = {
                    "paths": [self.inventory["stage"][kind]["path"]
                              for kind in ("payload", "receipt")],
                    "inventory_sha256": hashlib.sha256(_canonical(self.inventory)).hexdigest(),
                    "authorization_scope": SCOPE,
                    "approval_reference": REFERENCE,
                    "retain_gcs": True, "retain_drive": True,
                    "policy_sha256": POLICY,
                }[key]

    def test_rejects_creation_from_unqualified_local_only_inventory(self):
        unqualified = dict(self.inventory, drive_media_verified=False,
                           eligible_for_approval=False)
        with self.assertRaisesRegex(Conflict, "Pinned approval inventory"):
            OwnerPinnedApproval(path=self.path, root=self.root,
                                expected_sha256=self.pinned, inventory=unqualified,
                                authorization_reference=REFERENCE,
                                authorization_epoch=STOP_AT - 8000,
                                expected_policy_sha256=POLICY)

    def test_rejects_expired_or_post_stop_artifact(self):
        self.approval["valid_until"] = self.now
        self._write()
        with self.assertRaisesRegex(Conflict, "bounded policy window"):
            self._verify_fixture()
        self.approval["valid_until"] = STOP_AT + 1
        self._write()
        with self.assertRaisesRegex(Conflict, "bounded policy window"):
            self._verify_fixture()

    def test_rejects_artifact_predating_owner_attested_human_reply(self):
        self.approval["issued_at"] = STOP_AT - 9000
        self._write()
        with self.assertRaisesRegex(Conflict, "bounded policy window"):
            self._verify_fixture()

    def test_rejects_duplicate_keys_and_unknown_fields(self):
        raw = _canonical(self.approval)
        duplicate = raw[:-1] + b',"approved_by":"Josh"}'
        self._write(duplicate)
        with self.assertRaisesRegex(Conflict, "duplicate"):
            self.verifier._read_owned_file(os.getuid())
        self.approval["extra"] = "anything"
        self._write()
        with self.assertRaisesRegex(Conflict, "canonical schema"):
            self.verifier._read_owned_file(os.getuid())

    def test_passed_object_must_equal_pinned_file(self):
        changed = dict(self.approval, approval_reference="other")
        with mock.patch.object(self.verifier, "_read_root_file",
                               side_effect=lambda: self.verifier._read_owned_file(os.getuid())):
            with self.assertRaisesRegex(Conflict, "Passed approval differs"):
                self.verifier.verify(changed)

    def test_backward_wall_jump_cannot_extend_rechecked_expiry(self):
        with mock.patch.object(self.verifier, "_read_root_file", return_value=self.approval), \
             mock.patch("gcs_drive_retained_approval_artifact.time.time",
                        side_effect=[self.now, self.now - 300]), \
             mock.patch("gcs_drive_retained_approval_artifact.time.monotonic",
                        side_effect=[100.0, 101.0]):
            self.verifier.verify(self.approval)
            with self.assertRaisesRegex(Conflict, "bounded policy window"):
                self.verifier.verify(self.approval)

    def test_private_creator_is_exclusive_and_fsynced_file_is_verifiable(self):
        other_root = Path(self.tmp.name) / "new-private"
        other_root.mkdir(mode=0o700)
        other_root.chmod(0o700)
        path, digest = _create_owned_file(root=other_root, record=self.approval,
                                          expected_uid=os.getuid())
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        second, _ = _create_owned_file(root=other_root, record=self.approval,
                                       expected_uid=os.getuid())
        self.assertNotEqual(path, second)
        self.assertTrue(path.is_file())  # Earlier immutable attempt is retained.
        verifier = OwnerPinnedApproval(
            path=path, root=other_root, expected_sha256=digest,
            inventory=self.inventory, authorization_reference=REFERENCE,
            authorization_epoch=STOP_AT - 8000,
            expected_policy_sha256=POLICY)
        self.assertEqual(verifier._read_owned_file(os.getuid()), self.approval)

    def test_public_creator_refuses_nonroot_before_any_file_write(self):
        if os.getuid() == 0:
            self.skipTest("test requires nonroot synthetic fixture")
        other_root = Path(self.tmp.name) / "new-private"
        other_root.mkdir(mode=0o700)
        with self.assertRaisesRegex(Conflict, "Root operator"):
            create_owner_artifact_after_human_yes(
                inventory=self.inventory, authorization_reference=REFERENCE,
                authorization_epoch=STOP_AT - 8000, policy_sha256=POLICY,
                valid_until=STOP_AT - 1800, root=other_root)
        self.assertEqual(list(other_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
