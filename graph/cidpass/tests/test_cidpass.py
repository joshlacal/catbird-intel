"""Tests for the image-CID sidecar pass.

Run:  graph/.venv/bin/python -m unittest discover -s graph/cidpass/tests -v

Unit tests cover cid_rows. The end-to-end tests build synthetic .jss segments with
the decoder's own encoder (make_segments.mjs), run `cidpass.py segments` through the
real node decoder into a local store, kill-and-resume it, then fold with
clickhouse-local. They need CIDPASS_DECODER_DIR (a decoder copy with node_modules,
made by setup_decoder.sh) and `clickhouse` on PATH; otherwise they are skipped.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
CIDPASS = HERE.parent
sys.path.insert(0, str(CIDPASS))
sys.path.insert(0, str(CIDPASS.parent / "vmpass"))

import pyarrow.parquet as pq  # noqa: E402

import cid_rows  # noqa: E402
import cidpass  # noqa: E402
from cityhash import city_hash64  # noqa: E402

CID_A = "bafkreidtt757okdivxau7bl5jgwtmbaq62s7s67cd63greeaqbiq33msae"
CID_B = "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
CID_C = "bafkreib3kn55ex3owf6bvt6fgiffpoi6rkv4tsik7ntl4qtu3awffieocu"
CID_D = "bafkreif2aggqwlyu25hvlksuqttll4fxnxjg3ns3jhmjnid4wzloqjdmzm"
DECODER_DIR = os.environ.get("CIDPASS_DECODER_DIR", "")
HAVE_DECODER = bool(DECODER_DIR) and (pathlib.Path(DECODER_DIR) / "node_modules" / "@bsky" / "jetstream").is_dir()
HAVE_CH = shutil.which("clickhouse") is not None


def blob(cid, mime="image/jpeg"):
    return {"$type": "blob", "ref": {"$link": cid}, "mimeType": mime, "size": 10}


def ev(record=None, op="create", rkey="3lido2jgaih2k", rev="3lido2jgaih2k", did="did:plc:a", seq=5):
    e = {"$type": "network.bsky.jetstream.subscribeEvents#commit", "operation": op,
         "collection": "app.bsky.feed.post", "rkey": rkey, "rev": rev, "seq": seq, "did": did}
    if record is not None:
        e["record"] = record
    return e


class Rows(unittest.TestCase):
    def test_images_with_alt_aspect_mime_and_index(self):
        rec = {"embed": {"$type": "app.bsky.embed.images", "images": [
            {"alt": "x" * 400, "image": blob(CID_A), "aspectRatio": {"width": 4, "height": 3}},
            {"alt": "bad cid", "image": blob("not-a-cid")},
            {"alt": "", "image": blob(CID_B, "image/PNG"), "aspectRatio": {"width": 0, "height": 3}}]}}
        rows = cid_rows.rows_for_event(ev(rec))
        self.assertEqual([(r["image_index"], r["cid"]) for r in rows], [(0, CID_A), (2, CID_B)])
        self.assertEqual(len(rows[0]["alt"]), 300)
        self.assertEqual((rows[0]["aspect_w"], rows[0]["aspect_h"]), (4, 3))
        self.assertEqual((rows[1]["aspect_w"], rows[1]["aspect_h"]), (None, None))
        self.assertEqual((rows[0]["mime"], rows[1]["mime"]), ("image/jpeg", ""))  # uppercase mime rejected
        self.assertTrue(all(r["source"] == "images" and r["kind"] == "image" and r["row_type"] == "media" for r in rows))

    def test_record_with_media_images_and_external(self):
        rwm = {"$type": "app.bsky.embed.recordWithMedia", "record": {"record": {"uri": "at://x"}},
               "media": {"$type": "app.bsky.embed.images", "images": [{"alt": "a", "image": blob(CID_A)}]}}
        self.assertEqual([(r["kind"], r["source"]) for r in cid_rows.rows_for_event(ev({"embed": rwm}))],
                         [("image", "recordWithMedia")])
        ext = {"$type": "app.bsky.embed.external", "external": {"uri": "https://e.test", "title": "T",
                                                                 "thumb": blob(CID_C)}}
        r = cid_rows.rows_for_event(ev({"embed": ext}))
        self.assertEqual([(x["kind"], x["source"], x["cid"], x["alt"]) for x in r], [("external_thumb", "external", CID_C, "T")])
        rwm_ext = {"$type": "app.bsky.embed.recordWithMedia", "record": {"record": {"uri": "at://x"}}, "media": ext}
        self.assertEqual([(x["kind"], x["source"]) for x in cid_rows.rows_for_event(ev({"embed": rwm_ext}))],
                         [("external_thumb", "recordWithMedia")])

    def test_gallery_items_direct_and_under_record_with_media(self):
        gal = {"$type": "app.bsky.embed.gallery", "items": [
            {"$type": "app.bsky.embed.gallery#image", "alt": "g0", "image": blob(CID_A),
             "aspectRatio": {"width": 1, "height": 1}},
            {"$type": "app.bsky.embed.gallery#video", "video": blob(CID_B, "video/mp4")},  # unknown item type
            {"alt": "g2", "image": blob(CID_C)}]}
        rows = cid_rows.rows_for_event(ev({"embed": gal}))
        self.assertEqual([(r["image_index"], r["cid"], r["source"], r["alt"]) for r in rows],
                         [(0, CID_A, "gallery", "g0"), (2, CID_C, "gallery", "g2")])
        rwm = {"$type": "app.bsky.embed.recordWithMedia", "record": {"record": {"uri": "at://x"}}, "media": gal}
        self.assertEqual({r["source"] for r in cid_rows.rows_for_event(ev({"embed": rwm}))}, {"recordWithMedia"})

    def test_legacy_blob_and_no_video(self):
        legacy = {"embed": {"$type": "app.bsky.embed.images", "images": [{"image": {"cid": CID_B, "mimeType": "image/png"}}]}}
        self.assertEqual([r["cid"] for r in cid_rows.rows_for_event(ev(legacy))], [CID_B])
        video = {"embed": {"$type": "app.bsky.embed.video", "video": blob(CID_A, "video/mp4")}}
        self.assertEqual(cid_rows.rows_for_event(ev(video)), [])  # create without media: no row

    def test_markers(self):
        self.assertEqual([r["row_type"] for r in cid_rows.rows_for_event(ev(op="delete"))], ["marker"])
        self.assertEqual([r["row_type"] for r in cid_rows.rows_for_event(ev({"text": "x"}, op="update"))], ["marker"])
        self.assertEqual(cid_rows.rows_for_event({**ev(op="delete"), "collection": "app.bsky.feed.like"}), [])

    def test_tid_decode(self):
        us = cid_rows.tid_to_us("3lido2jgaih2k")
        self.assertTrue(us and cid_rows.us_to_iso(us) == "2025-02-17T02:08:43.751885Z")
        for bad in ("3abc", "zzzzzzzzzzzzz", "3lido2jgaih2!", 7, "2222222222222"):
            self.assertIsNone(cid_rows.tid_to_us(bad), bad)

    def test_prefilter(self):
        create = json.dumps(ev({"text": "x"})).encode()
        self.assertTrue(create.startswith(b'{"$type"'))
        line = b'{"$type":"network.bsky.jetstream.subscribeEvents#commit","operation":"create","collection":"app.bsky.feed.post"}'
        self.assertFalse(cid_rows.needs_parse(line))
        self.assertTrue(cid_rows.needs_parse(line.replace(b"create", b"delete")))
        self.assertTrue(cid_rows.needs_parse(line[:-1] + b',"record":{"embed":{}}}'))


# ---------------------------------------------------------------- end to end

DID1 = "did:plc:aaaaaaaaaaaaaaaaaaaaaaaa"
DID2 = "did:plc:bbbbbbbbbbbbbbbbbbbbbbbb"
DID3 = "did:plc:cccccccccccccccccccccccc"
R1, R2, R3, R4, R5 = "3lido2jgaih2a", "3lido2jgaih2b", "3lido2jgaih2c", "3lido2jgaih2d", "3lido2jgaih2e"


def jblob(cid, mime="image/jpeg"):
    return {"$blob": cid, "mimeType": mime, "size": 100}


def post(seq, did, rkey, rev, embed=None, op="create", text="t"):
    e = {"seq": seq, "kind": "commit", "operation": op, "did": did, "collection": "app.bsky.feed.post",
         "rkey": rkey, "rev": rev}
    if op != "delete":
        e["record"] = {"$type": "app.bsky.feed.post", "text": text, "createdAt": "2026-10-20T00:00:00Z"}
        if embed:
            e["record"]["embed"] = embed
    return e


def images(*cids, alt="a"):
    return {"$type": "app.bsky.embed.images",
            "images": [{"alt": alt, "image": jblob(c), "aspectRatio": {"width": 3, "height": 2}} for c in cids]}


SPEC = {"segments": [
    {"index": 3671, "events": [
        post(1, DID1, R1, "3lido2jgaih2a", images(CID_A, CID_B)),            # later updated to one image (seg 3672)
        post(2, DID1, R2, "3lido2jgaih2b", images(CID_A)),                   # later deleted
        post(3, DID2, R3, "3lido2jgaih2c", images(CID_C)),                   # survives, duplicated in seg 3672
        post(4, DID2, R4, "3lido2jgaih2d", images(CID_D)),                   # later updated with no media
        {"seq": 5, "kind": "commit", "operation": "create", "did": DID3, "collection": "app.bsky.feed.like",
         "rkey": "3lido2jgaih2z", "rev": "3lido2jgaih2z",
         "record": {"$type": "app.bsky.feed.like", "createdAt": "2026-10-20T00:00:00Z",
                    "subject": {"uri": f"at://{DID1}/app.bsky.feed.post/{R1}", "cid": CID_A.replace("bafkrei", "bafyrei")}}},
    ]},
    {"index": 3672, "group": "catbird-overflow-0bfdcbe5b8a6c6a1e2e6738836de482a", "events": [
        post(10, DID1, R1, "3lido2jgaih3a", images(CID_C), op="update"),
        post(11, DID1, R2, "3lido2jgaih3b", op="delete"),
        post(12, DID2, R3, "3lido2jgaih2c", images(CID_C)),                  # same rev re-delivered: dedup
        post(13, DID2, R4, "3lido2jgaih3d", op="update"),
        post(14, DID3, R5, "3lido2jgaih3e", {"$type": "app.bsky.embed.recordWithMedia",
                                              "record": {"record": {"uri": f"at://{DID1}/app.bsky.feed.post/{R1}",
                                                                    "cid": CID_A.replace("bafkrei", "bafyrei")}},
                                              "media": {"$type": "app.bsky.embed.external", "external": {
                                                  "uri": "https://e.test", "title": "card", "description": "d",
                                                  "thumb": jblob(CID_B)}}}),
        post(15, DID3, "3lido2jgaih2f", "3lido2jgaih3f", text="no media"),
    ]},
    {"index": 3673, "events": [
        post(20, DID3, "3lido2jgaih2g", "3lido2jgaih3g", images(CID_D), text="third"),
        post(21, DID2, "3lido2jgaih2h", "3lido2jgaih3h", {"$type": "app.bsky.embed.gallery", "items": [
            {"$type": "app.bsky.embed.gallery#image", "alt": "gal", "image": jblob(CID_A),
             "aspectRatio": {"width": 5, "height": 4}}]}, text="gallery"),
    ]},
]}


@unittest.skipUnless(HAVE_DECODER, "set CIDPASS_DECODER_DIR to a decoder copy with node_modules (setup_decoder.sh)")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="cidpass-test-"))
        (cls.tmp / "spec.json").write_text(json.dumps(SPEC))
        r = subprocess.run(["node", str(CIDPASS / "make_segments.mjs"), "--decoder-dir", DECODER_DIR,
                            "--spec", str(cls.tmp / "spec.json"), "--out", str(cls.tmp / "archive")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        cls.store30 = cls.tmp / "store30"
        cls.store60 = cls.tmp / "store60"
        cls.decoder = str(pathlib.Path(DECODER_DIR) / "decode_segments.mjs")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def segs(self, *extra, run="r1"):
        return cidpass.main(["segments", "--source", "local", "--src", str(self.tmp / "archive"), "--store30",
                             str(self.store30), "--run-id", run, "--work", str(self.tmp / f"work-{run}"),
                             "--decoder", self.decoder, "--workers", "2", "--progress-seconds", "0", *extra])

    def test_1_segments_resume_and_fold(self):
        # a "preempted" first run that finishes one segment, then a resume that does the rest
        self.assertEqual(self.segs("--max-segments", "1"), 0)
        self.assertEqual(cidpass.done_indices(cidpass.open_store(str(self.store30)), "r1"), {3671})
        self.assertEqual(self.segs(), 0)
        done = cidpass.done_indices(cidpass.open_store(str(self.store30)), "r1")
        self.assertEqual(done, {3671, 3672, 3673})
        led = json.loads((self.store30 / cidpass.ledger_key("r1", 3672)).read_text())
        self.assertEqual((led["post_events"], led["media_rows"], led["marker_rows"], led["deletes"]), (6, 3, 2, 1))  # delete + no-media update; a no-media create emits nothing
        self.assertEqual(len(led["files"]), 16)
        # every row sits in partition cityHash64(did) % 16
        for f in led["files"]:
            t = pq.read_table(self.store30 / led["parts_prefix"] / f["file"])
            self.assertTrue(all(city_hash64(d) % 16 == f["part"] for d in t.column("did").to_pylist()))
        # a third run reads nothing
        self.assertEqual(self.segs(), 0)

        if not HAVE_CH:
            self.skipTest("clickhouse not on PATH: fold not tested")
        # refuses an incomplete range, then folds the complete one
        rc = cidpass.main(["fold", "--store30", str(self.store30), "--store60", str(self.store60), "--run-id", "r1",
                           "--work", str(self.tmp / "fold"), "--expect-range", "3671-3674"])
        self.assertEqual(rc, 4)
        rc = cidpass.main(["fold", "--store30", str(self.store30), "--store60", str(self.store60), "--run-id", "r1",
                           "--work", str(self.tmp / "fold"), "--expect-range", "3671-3673"])
        self.assertEqual(rc, 0)
        out = self.store60 / "sidecar" / "image-cids" / "r1"
        rows = []
        for p in range(16):
            t = pq.read_table(out / f"image_cids-p{p:02d}.parquet")
            self.assertEqual(t.schema.names, [f.name for f in cid_rows.SIDECAR_SCHEMA])
            for r in t.to_pylist():
                self.assertEqual(city_hash64(r["did"]) % 16, p)
                rows.append(r)
        got = sorted((r["did"], r["rkey"], r["kind"], r["image_index"], r["cid"], r["source"], r["rev"]) for r in rows)
        self.assertEqual(got, sorted([
            (DID1, R1, "image", 0, CID_C, "images", "3lido2jgaih3a"),          # latest revision only
            (DID2, R3, "image", 0, CID_C, "images", "3lido2jgaih2c"),          # duplicate collapsed
            (DID3, R5, "external_thumb", 0, CID_B, "recordWithMedia", "3lido2jgaih3e"),
            (DID3, "3lido2jgaih2g", "image", 0, CID_D, "images", "3lido2jgaih3g"),
            (DID2, "3lido2jgaih2h", "image", 0, CID_A, "gallery", "3lido2jgaih3h"),
        ]))
        r1 = next(r for r in rows if r["rkey"] == R1)
        self.assertEqual((r1["aspect_w"], r1["aspect_h"], r1["alt"], r1["mime"]), (3, 2, "a", "image/jpeg"))
        self.assertEqual(int(r1["post_created"].timestamp() * 1e6), cid_rows.tid_to_us(R1))
        man = json.loads((out / "manifest.json").read_text())
        self.assertEqual(sum(v["rows"] for v in man["parts"].values()), 5)
        self.assertEqual(man["segment_totals"]["post_events"], 12)  # 4 + 6 + 2 post commits
        self.assertEqual(man["segment_totals"]["skipped_invalid_records"], 0)  # the decoder accepts gallery posts
        self.assertEqual(man["sources"], ["local"])

    def test_2_failed_segment_is_retried_not_marked(self):
        bad = self.tmp / "bad-archive"
        shutil.copytree(self.tmp / "archive", bad)
        seg = next(bad.glob("segments/*/seg_00000002tz.jss"))  # index 3671
        seg.write_bytes(seg.read_bytes()[:-40])  # truncated: header/footer checks fail in the decoder
        rc = cidpass.main(["segments", "--source", "local", "--src", str(bad), "--store30", str(self.store30),
                           "--run-id", "r2", "--work", str(self.tmp / "work-r2"), "--decoder", self.decoder,
                           "--workers", "1", "--max-index", "3671", "--retry-sleep", "0"])
        self.assertEqual(rc, 3)
        self.assertEqual(cidpass.done_indices(cidpass.open_store(str(self.store30)), "r2"), set())


if __name__ == "__main__":
    unittest.main()
