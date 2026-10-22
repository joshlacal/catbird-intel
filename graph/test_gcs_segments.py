"""Listing, pairing and verification rules for the Cloud Storage segment source."""
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import gcs_segments

PREFIX = "gs://bucket/sources/gce-selfhost-001/buffer-admission-20261020/segments"
SHA_A = "a" * 64
SHA_B = "b" * 64


def line(size, path):
    return f"{size:>11}  2026-10-20T18:13:22Z  {PREFIX}/{path}"


class ParseListing(unittest.TestCase):
    def test_pairs_payload_with_its_receipt_and_ignores_release_marker(self):
        listing = "\n".join([
            line(272951056, f"0000000f/seg_00000002zu.jss/{SHA_A}.jss"),
            line(1093, f"0000000f/seg_00000002zu.jss/{SHA_A}.jss.g17925.receipt.json"),
            line(581, f"0000000f/seg_00000002zu.jss/{SHA_A}.jss.g17925.receipt.json.local-release.json"),
            "TOTAL: 3 objects, 272952730 bytes (260.31MiB)",
        ])
        segs = gcs_segments.parse_listing(listing)
        self.assertEqual(list(segs), [int("2zu", 36)])
        seg = segs[int("2zu", 36)]
        self.assertEqual(seg["group"], "0000000f")
        self.assertEqual(seg["sha256"], SHA_A)
        self.assertEqual(seg["jss"]["name"], "seg_00000002zu.jss")
        self.assertEqual(seg["jss"]["size"], 272951056)
        self.assertTrue(seg["receipt"]["uri"].endswith(".g17925.receipt.json"))

    def test_payload_without_receipt_is_not_ready(self):
        listing = line(10, f"0000000f/seg_00000002zv.jss/{SHA_A}.jss")
        self.assertEqual(gcs_segments.parse_listing(listing), {})

    def test_two_receipted_payloads_for_one_segment_are_refused(self):
        listing = "\n".join([
            line(10, f"0000000f/seg_00000002zu.jss/{SHA_A}.jss"),
            line(5, f"0000000f/seg_00000002zu.jss/{SHA_A}.jss.g1.receipt.json"),
            line(10, f"0000000f/seg_00000002zu.jss/{SHA_B}.jss"),
            line(5, f"0000000f/seg_00000002zu.jss/{SHA_B}.jss.g2.receipt.json"),
        ])
        with self.assertRaises(RuntimeError):
            gcs_segments.parse_listing(listing)


class Download(unittest.TestCase):
    def setUp(self):
        self.payload = b"segment-bytes"
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.seg = {
            "group": "0000000f", "sha256": self.sha,
            "jss": {"name": "seg_00000002zu.jss", "size": len(self.payload), "uri": "gs://b/p.jss"},
            "receipt": {"uri": "gs://b/p.jss.g1.receipt.json"},
        }
        self.dest = pathlib.Path(tempfile.mkdtemp())

    def fake_gcloud(self, payload, receipt):
        def run(*args):
            if args[0] == "cat":
                return json.dumps(receipt)
            pathlib.Path(args[2]).write_bytes(payload)
            return ""
        return run

    def test_verified_download_writes_decoder_layout(self):
        receipt = {"name": "seg_00000002zu.jss", "sha256": self.sha, "bytes": len(self.payload)}
        with mock.patch.object(gcs_segments, "_gcloud", self.fake_gcloud(self.payload, receipt)):
            gcs_segments.download_segment(self.seg, self.dest)
        self.assertEqual((self.dest / "seg_00000002zu.jss").read_bytes(), self.payload)
        self.assertEqual(json.loads((self.dest / "seg_00000002zu.jss.receipt.json").read_text())["sha256"], self.sha)

    def test_corrupt_download_is_removed_and_raises(self):
        receipt = {"name": "seg_00000002zu.jss", "sha256": self.sha, "bytes": len(self.payload)}
        with mock.patch.object(gcs_segments, "_gcloud", self.fake_gcloud(b"segment-bytez", receipt)):
            with self.assertRaises(RuntimeError):
                gcs_segments.download_segment(self.seg, self.dest)
        self.assertFalse((self.dest / "seg_00000002zu.jss").exists())
        self.assertFalse((self.dest / "seg_00000002zu.jss.receipt.json").exists())

    def test_receipt_for_another_payload_is_refused(self):
        receipt = {"name": "seg_00000002zu.jss", "sha256": SHA_B, "bytes": len(self.payload)}
        with mock.patch.object(gcs_segments, "_gcloud", self.fake_gcloud(self.payload, receipt)):
            with self.assertRaises(RuntimeError):
                gcs_segments.download_segment(self.seg, self.dest)


if __name__ == "__main__":
    unittest.main()
