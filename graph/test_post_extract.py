"""Post rows for archive_graph_extract.py --content posts."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import pyarrow.parquet as pq

import archive_graph_extract as age
from post_extract import POST_SCHEMA

HERE = pathlib.Path(__file__).parent


def commit(record, operation="create", collection="app.bsky.feed.post", rkey="3abc"):
    ev = {"kind": "commit", "did": "did:plc:a", "time_us": 1, "seq": 7,
          "commit": {"rev": "3lido2jgaih2k", "operation": operation,
                     "collection": collection, "rkey": rkey}}
    if record is not None:
        ev["commit"]["record"] = record
    return ev


class PostRows(unittest.TestCase):
    def test_quote_with_images_reply_facets_and_labels(self):
        rec = {
            "$type": "app.bsky.feed.post", "text": "rust 1.90 is out", "createdAt": "2026-10-01T00:00:00Z",
            "langs": ["en"], "tags": ["release"],
            "reply": {"root": {"uri": "at://did:plc:r/app.bsky.feed.post/1"},
                      "parent": {"uri": "at://did:plc:p/app.bsky.feed.post/2"}},
            "embed": {"$type": "app.bsky.embed.recordWithMedia",
                      "record": {"record": {"uri": "at://did:plc:q/app.bsky.feed.post/3"}},
                      "media": {"$type": "app.bsky.embed.images",
                                "images": [{"alt": "crab"}, {"alt": ""}]}},
            "facets": [{"features": [
                {"$type": "app.bsky.richtext.facet#link", "uri": "https://blog.rust-lang.org"},
                {"$type": "app.bsky.richtext.facet#tag", "tag": "rustlang"},
                {"$type": "app.bsky.richtext.facet#mention", "did": "did:plc:m"}]}],
            "labels": {"values": [{"val": "nudity"}]},
        }
        row = age.post_row(commit(rec), "gce_main")
        self.assertEqual(row["text"], "rust 1.90 is out")
        self.assertEqual(row["reply_parent"], "at://did:plc:p/app.bsky.feed.post/2")
        self.assertEqual(row["reply_root"], "at://did:plc:r/app.bsky.feed.post/1")
        self.assertEqual(row["quote_uri"], "at://did:plc:q/app.bsky.feed.post/3")
        self.assertEqual((row["image_count"], row["image_alts"]), (2, ["crab", ""]))
        self.assertEqual(row["facet_links"], ["https://blog.rust-lang.org"])
        self.assertEqual(row["facet_tags"], ["rustlang", "release"])
        self.assertEqual(row["facet_mentions"], ["did:plc:m"])
        self.assertEqual(row["self_labels"], ["nudity"])
        self.assertEqual((row["did"], row["rkey"], row["rev"], row["source_id"]),
                         ("did:plc:a", "3abc", "3lido2jgaih2k", "gce_main"))

    def test_delete_is_kept_without_record(self):
        row = age.post_row(commit(None, operation="delete"), "gce_main")
        self.assertEqual((row["operation"], row["text"], row["image_count"]), ("delete", "", 0))

    def test_non_post_and_non_commit_events_are_skipped(self):
        self.assertIsNone(age.post_row(commit({"subject": "did:plc:b"}, collection="app.bsky.graph.follow"), "gce_main"))
        self.assertIsNone(age.post_row({"kind": "account", "did": "did:plc:a", "account": {"active": False}}, "gce_main"))

    def test_malformed_fields_do_not_raise(self):
        rec = {"text": 5, "langs": "en", "embed": {"$type": "app.bsky.embed.images", "images": 3},
               "facets": "x", "labels": [], "reply": {"root": "nope"}}
        row = age.post_row(commit(rec), "gce_main")
        self.assertEqual((row["text"], row["langs"], row["image_count"], row["reply_root"]), ("", [], 0, ""))

    def test_convert_shards_writes_post_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            dec, out = pathlib.Path(tmp, "dec"), pathlib.Path(tmp, "out")
            dec.mkdir()
            lines = [commit({"text": "hi", "createdAt": "2026-10-01T00:00:00Z"}),
                     commit({"subject": "did:plc:b"}, collection="app.bsky.graph.follow"),
                     commit(None, operation="delete", rkey="3abd")]
            (dec / "s.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
            (dec / "checkpoint.json").write_text(json.dumps({"shards": [{"path": "s.jsonl", "sha256": "s"}]}))
            n = age.convert_shards(dec, out, set(), "000001-000001", source_id="gce_main", posts=True)
            self.assertEqual(n, 2)
            [f] = list(out.glob("posts-*.parquet"))
            table = pq.read_table(f)
            self.assertEqual(table.schema, POST_SCHEMA)
            self.assertEqual(table.column("operation").to_pylist(), ["create", "delete"])


CID_A = "bafkreidtt757okdivxau7bl5jgwtmbaq62s7s67cd63greeaqbiq33msae"
CID_B = "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"


def blob(cid):
    return {"$type": "blob", "ref": {"$link": cid}, "mimeType": "image/jpeg", "size": 1234}


class ImageCids(unittest.TestCase):
    def test_images_embed_in_record_order(self):
        rec = {"text": "two", "embed": {"$type": "app.bsky.embed.images", "images": [
            {"alt": "a", "image": blob(CID_A)}, {"alt": "b", "image": blob(CID_B)}]}}
        row = age.post_row(commit(rec), "gce_main")
        self.assertEqual(row["image_cids"], [CID_A, CID_B])
        self.assertEqual((row["image_count"], row["image_alts"]), (2, ["a", "b"]))

    def test_record_with_media_images(self):
        rec = {"embed": {"$type": "app.bsky.embed.recordWithMedia",
                         "record": {"record": {"uri": "at://did:plc:q/app.bsky.feed.post/3"}},
                         "media": {"$type": "app.bsky.embed.images", "images": [{"alt": "", "image": blob(CID_A)}]}}}
        self.assertEqual(age.post_row(commit(rec), "gce_main")["image_cids"], [CID_A])

    def test_legacy_blob_cid_form(self):
        rec = {"embed": {"$type": "app.bsky.embed.images",
                         "images": [{"alt": "", "image": {"cid": CID_B, "mimeType": "image/png"}}]}}
        self.assertEqual(age.post_row(commit(rec), "gce_main")["image_cids"], [CID_B])

    def test_video_external_and_record_have_none(self):
        for embed in ({"$type": "app.bsky.embed.video", "video": blob(CID_A), "alt": "v"},
                      {"$type": "app.bsky.embed.external",
                       "external": {"uri": "https://x.test", "thumb": blob(CID_A)}},
                      {"$type": "app.bsky.embed.recordWithMedia", "record": {"record": {"uri": "at://x"}},
                       "media": {"$type": "app.bsky.embed.video", "video": blob(CID_A)}},
                      {"$type": "app.bsky.embed.record", "record": {"uri": "at://x"}}):
            self.assertEqual(age.post_row(commit({"embed": embed}), "gce_main")["image_cids"], [], embed["$type"])

    def test_malformed_or_unsafe_cids_are_dropped(self):
        rec = {"embed": {"$type": "app.bsky.embed.images", "images": [
            {"image": blob("../../avatar/plain/did:plc:x/" + CID_A)},  # path injection
            {"image": blob(CID_A.upper())},                            # not lowercase base32
            {"image": blob("Qm" + "a" * 44)},                          # CIDv0
            {"image": {"ref": 5}}, {"image": "nope"}, {"alt": "no image"}, 7,
            {"image": blob(CID_B)}]}}
        row = age.post_row(commit(rec), "gce_main")
        self.assertEqual(row["image_cids"], [CID_B])

    def test_delete_and_no_embed_have_empty_list(self):
        self.assertEqual(age.post_row(commit(None, operation="delete"), "gce_main")["image_cids"], [])
        self.assertEqual(age.post_row(commit({"text": "x"}), "gce_main")["image_cids"], [])

    def test_new_column_is_appended_last(self):
        names = POST_SCHEMA.names
        self.assertEqual(names[-1], "image_cids")
        self.assertEqual(names[:-1], ["source_id", "seq", "time", "did", "rkey", "rev", "operation", "created_at",
                                      "text", "langs", "reply_root", "reply_parent", "embed_type", "quote_uri",
                                      "external_uri", "image_count", "image_alts", "facet_links", "facet_tags",
                                      "facet_mentions", "self_labels"])


class AllContent(unittest.TestCase):
    def test_split_writes_posts_and_events_and_keeps_generic_records(self):
        age._ACTIVE_COLLECTIONS = set(age.ALL_COLLECTIONS)
        self.addCleanup(setattr, age, "_ACTIVE_COLLECTIONS", set(age.COLLECTIONS))
        with tempfile.TemporaryDirectory() as tmp:
            dec, out = pathlib.Path(tmp, "dec"), pathlib.Path(tmp, "out")
            dec.mkdir()
            lines = [commit({"text": "hi"}),
                     commit({"subject": {"uri": "at://did:plc:b/app.bsky.feed.post/1", "cid": "x"}},
                            collection="app.bsky.feed.like"),
                     commit({"post": "at://did:plc:a/app.bsky.feed.post/1", "allow": []},
                            collection="app.bsky.feed.threadgate"),
                     {"kind": "account", "did": "did:plc:a", "account": {"active": False, "status": "deactivated"}}]
            (dec / "s.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
            (dec / "checkpoint.json").write_text(json.dumps({"shards": [{"path": "s.jsonl", "sha256": "s"}]}))
            n = age.convert_shards(dec, out, set(), "000001-000001", source_id="gce_main", split_posts=True)
            self.assertEqual(n, 4)
            [pf] = list(out.glob("posts-*.parquet"))
            [ef] = list(out.glob("events-*.parquet"))
            self.assertEqual(pq.read_table(pf).column("text").to_pylist(), ["hi"])
            ev = pq.read_table(ef).to_pylist()
            by = {r["collection"] or r["kind"]: r for r in ev}
            self.assertEqual(by["app.bsky.feed.like"]["subject"], "at://did:plc:b/app.bsky.feed.post/1")
            self.assertEqual(json.loads(by["app.bsky.feed.threadgate"]["extra"])["post"], "at://did:plc:a/app.bsky.feed.post/1")
            self.assertEqual(by["account"]["account_status"], "deactivated")

    def test_graph_rows_for_existing_collections_are_unchanged(self):
        row = age.row_from_event(commit({"subject": "did:plc:b"}, collection="app.bsky.graph.follow"), "gce_main")
        self.assertEqual((row["subject"], row["extra"]), ("did:plc:b", ""))


class ReadOnlyShim(unittest.TestCase):
    def test_refuses_anything_but_list_and_get(self):
        for argv in (["drive", "files", "create", "--params", "{}"],
                     ["drive", "files", "update", "--params", "{}"],
                     ["drive", "permissions", "list", "--params", "{}"]):
            r = subprocess.run([sys.executable, str(HERE / "drive_readonly_shim.py"), *argv],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, argv)


if __name__ == "__main__":
    unittest.main()
