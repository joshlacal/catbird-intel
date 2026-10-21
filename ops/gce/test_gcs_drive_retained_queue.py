"""Synthetic-only selection tests; no GCS or Drive client is constructed."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gcs_drive_drain import Conflict, ObjectRef
from gcs_drive_retained_overflow import BUCKET, SOURCE_EPOCH, SOURCE_ID, load_overflow_manifest
from gcs_drive_retained_queue import PREFIX, build_snapshot
from retained_transfer_authority import LEGACY_DIAGNOSTIC_PAYLOAD_GENERATION
from drive_transfer_authority import DIAGNOSTIC_PAYLOAD_OBJECT


ACCOUNT = "synthetic-account"
PAYLOAD = b"sealed synthetic payload"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
NAME = "seg_0000000001.jss"
PAYLOAD_NAME = f"{PREFIX}00000000/{NAME}/{PAYLOAD_SHA}.jss"
PAYLOAD_GEN = 101
RECEIPT_NAME = PAYLOAD_NAME + f".g{PAYLOAD_GEN}.receipt.json"
RECEIPT_GEN = 102


def fixture():
    receipt = {
        "schema": "catbird.jetstream.archive.v1", "name": NAME, "index": 1,
        "source_id": SOURCE_ID, "source_epoch": SOURCE_EPOCH,
        "sha256": PAYLOAD_SHA, "bytes": len(PAYLOAD),
        "gcs": {"bucket": BUCKET, "object": PAYLOAD_NAME,
                "generation": str(PAYLOAD_GEN), "sha256": PAYLOAD_SHA,
                "bytes": len(PAYLOAD),
                "md5_hash": base64.b64encode(hashlib.md5(PAYLOAD).digest()).decode()},
    }
    raw = json.dumps(receipt, sort_keys=True).encode()
    payload_ref = ObjectRef(PAYLOAD_NAME, PAYLOAD_GEN, len(PAYLOAD),
                            {"sha256": PAYLOAD_SHA, "kind": "payload"})
    receipt_ref = ObjectRef(RECEIPT_NAME, RECEIPT_GEN, len(raw),
                            {"sha256": hashlib.sha256(raw).hexdigest(),
                             "kind": "receipt", "segment_name": NAME,
                             "payload_object": PAYLOAD_NAME,
                             "payload_generation": str(PAYLOAD_GEN)})
    return raw, payload_ref, receipt_ref


class FakeCloud:
    bucket_name = BUCKET

    def __init__(self, raw, payload_ref, receipt_ref):
        self.raw = raw
        self.payload_ref = payload_ref
        self.receipt_ref = receipt_ref
        self.list_calls = 0
        self.download_calls = 0
        self.delete_calls = 0

    def receipts(self, prefix):
        self.list_calls += 1
        assert prefix == PREFIX
        yield self.receipt_ref

    def lookup(self, name, generation=None):
        if name == self.payload_ref.name and generation == self.payload_ref.generation:
            return self.payload_ref
        return None

    def download(self, ref, path):
        self.download_calls += 1
        assert ref == self.receipt_ref
        Path(path).write_bytes(self.raw)

    def delete(self, _ref):
        self.delete_calls += 1
        raise AssertionError("snapshot selector may not delete GCS originals")


class QueueSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.raw, self.payload_ref, self.receipt_ref = fixture()
        self.cloud = FakeCloud(self.raw, self.payload_ref, self.receipt_ref)

    def select(self, **overrides):
        params = {"account_id": ACCOUNT, "work": self.work, "ledger_pairs": [],
                  "max_scan": 4, "max_pairs": 1, "max_reserved_bytes": 8 << 20,
                  "max_run_seconds": 1200}
        params.update(overrides)
        return build_snapshot(self.cloud, **params)

    def test_pins_generation_and_raw_manifest_with_bounded_reservation(self):
        selected = self.select()
        self.assertEqual(len(selected), 1)
        item = selected[0]
        manifest, digest = load_overflow_manifest(item.manifest_path,
                                                  release_id=item.release_id,
                                                  account_id=ACCOUNT)
        self.assertEqual(digest, item.manifest_sha256)
        self.assertEqual(manifest["payload"]["generation"], PAYLOAD_GEN)
        self.assertEqual(manifest["receipt"]["generation"], RECEIPT_GEN)
        self.assertEqual(manifest["max_charged_bytes"], item.reserved_bytes)
        self.assertEqual(self.cloud.delete_calls, 0)
        self.assertEqual(self.select(), selected)

    def test_closed_pair_is_skipped_and_nonterminal_halts_before_gcs_read(self):
        key = self.select()[0].pair_key
        source = {"bucket": BUCKET, "payload_object": PAYLOAD_NAME,
                  "payload_generation": PAYLOAD_GEN}
        self.assertEqual(self.select(ledger_pairs=[{"pair_key": key, "status": "closed",
                                                    **source}]), [])
        before_downloads = self.cloud.download_calls
        self.assertEqual(self.select(ledger_pairs=[{"pair_key": None,
                                                    "status": "diagnostic_verified_unqualified",
                                                    **source}]), [])
        self.assertEqual(self.cloud.download_calls, before_downloads)
        before = self.cloud.list_calls
        with self.assertRaisesRegex(Conflict, "needs recovery"):
            self.select(ledger_pairs=[{"pair_key": key, "status": "verified",
                                       **source}])
        self.assertEqual(self.cloud.list_calls, before)

    def test_imported_segment_3677_is_skipped_without_second_download(self):
        pilot_ref = ObjectRef(
            DIAGNOSTIC_PAYLOAD_OBJECT +
            f".g{LEGACY_DIAGNOSTIC_PAYLOAD_GENERATION}.receipt.json",
            999, 100, {"kind": "receipt"})

        class PilotCloud:
            bucket_name = BUCKET

            def receipts(self, _prefix):
                yield pilot_ref

            def download(self, _ref, _path):
                raise AssertionError("already verified diagnostic must not be recopied")

        marker = {"pair_key": None, "status": "diagnostic_verified_unqualified",
                  "bucket": BUCKET, "payload_object": DIAGNOSTIC_PAYLOAD_OBJECT,
                  "payload_generation": LEGACY_DIAGNOSTIC_PAYLOAD_GENERATION}
        self.assertEqual(build_snapshot(PilotCloud(), account_id=ACCOUNT,
                                        work=self.work, ledger_pairs=[marker],
                                        max_scan=1, max_pairs=1,
                                        max_reserved_bytes=8 << 20), [])

    def test_duplicate_or_nonfinite_receipt_json_is_rejected(self):
        for trailer, message in ((b',"schema":"catbird.jetstream.archive.v1"}',
                                  "duplicate keys"),
                                 (b',"extra":NaN}', "nonfinite")):
            with self.subTest(message=message):
                raw = self.raw[:-1] + trailer
                self.cloud.raw = raw
                self.cloud.receipt_ref = ObjectRef(
                    RECEIPT_NAME, RECEIPT_GEN, len(raw),
                    {**self.receipt_ref.metadata,
                     "sha256": hashlib.sha256(raw).hexdigest()})
                self.work = Path(self.temp.name) / message
                with self.assertRaisesRegex(Conflict, message):
                    self.select()

    def test_changed_receipt_or_payload_generation_blocks(self):
        self.cloud.raw = self.raw + b" "
        with self.assertRaisesRegex(Conflict, "bytes differ"):
            self.select()
        self.cloud.raw = self.raw
        self.cloud.payload_ref = ObjectRef(PAYLOAD_NAME, PAYLOAD_GEN + 1, len(PAYLOAD),
                                           {"sha256": PAYLOAD_SHA})
        self.work = self.work / "new-snapshot"
        with self.assertRaisesRegex(Conflict, "missing or changed"):
            self.select()

    def test_budget_and_scan_limits_fail_closed(self):
        self.assertEqual(self.select(max_reserved_bytes=1), [])
        with self.assertRaisesRegex(Conflict, "Invalid max scan"):
            self.select(max_scan=0)
        with self.assertRaisesRegex(Conflict, "controls"):
            self.select(max_pairs=5, max_scan=4)
        with self.assertRaisesRegex(Conflict, "controls"):
            self.select(max_pairs=9, max_scan=9)
        with self.assertRaisesRegex(Conflict, "controls"):
            self.select(max_scan=5001)

    def test_two_pair_snapshot_respects_total_reservation(self):
        first_raw, first_payload, first_receipt = fixture()
        second_name = "seg_0000000002.jss"
        second_payload_name = f"{PREFIX}00000000/{second_name}/{PAYLOAD_SHA}.jss"
        second_receipt_name = second_payload_name + ".g201.receipt.json"
        second = json.loads(first_raw)
        second["name"] = second_name
        second["index"] = 2
        second["gcs"]["object"] = second_payload_name
        second["gcs"]["generation"] = "201"
        second_raw = json.dumps(second, sort_keys=True).encode()
        second_payload = ObjectRef(second_payload_name, 201, len(PAYLOAD),
                                   {"sha256": PAYLOAD_SHA})
        second_receipt = ObjectRef(second_receipt_name, 202, len(second_raw),
                                   {"sha256": hashlib.sha256(second_raw).hexdigest(),
                                    "kind": "receipt", "segment_name": second_name,
                                    "payload_object": second_payload_name,
                                    "payload_generation": "201"})

        class TwoCloud:
            bucket_name = BUCKET

            def receipts(self, _prefix):
                yield first_receipt
                yield second_receipt

            def lookup(self, name, generation=None):
                return {first_payload.name: first_payload,
                        second_payload.name: second_payload}.get(name) if generation in (101, 201) else None

            def download(self, ref, path):
                Path(path).write_bytes(first_raw if ref.name == first_receipt.name else second_raw)

            def delete(self, _ref):
                raise AssertionError("GCS deletion is forbidden")

        cloud = TwoCloud()
        reserve_one = len(PAYLOAD) + len(first_raw) + (1 << 20)
        one = build_snapshot(cloud, account_id=ACCOUNT, work=self.work,
                             ledger_pairs=[], max_scan=2, max_pairs=2,
                             max_reserved_bytes=reserve_one, max_run_seconds=1200)
        self.assertEqual(len(one), 1)
        two = build_snapshot(cloud, account_id=ACCOUNT, work=self.work,
                             ledger_pairs=[], max_scan=2, max_pairs=2,
                             max_reserved_bytes=reserve_one * 2 + 1024,
                             max_run_seconds=1200)
        self.assertEqual(len(two), 2)
        self.assertNotEqual(two[0].pair_key, two[1].pair_key)

    def test_existing_snapshot_manifest_cannot_change(self):
        item = self.select()[0]
        item.manifest_path.write_bytes(b"changed")
        with self.assertRaisesRegex(Conflict, "manifest changed"):
            self.select()


if __name__ == "__main__":
    unittest.main()
