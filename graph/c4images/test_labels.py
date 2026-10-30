"""labels.py on stores written by the collector's own writer (needs cbor2: run with label-backfill-venv).

  ~/catbird-lake-local/label-backfill-venv/bin/python -m unittest graph/c4images/test_labels.py
"""
import json
import pathlib
import sys
import tempfile
import unittest

import cbor2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import labels  # noqa: E402

C = labels.collector
SRC = "did:plc:ar7c4by46qjdydhdevvrndac"


def frame(seq, *labs):
    return cbor2.dumps({"op": 1, "t": "#labels"}) + cbor2.dumps({"seq": seq, "labels": list(labs)})


def lab(uri, val, cts, **kw):
    return {"src": SRC, "uri": uri, "val": val, "cts": cts, **kw}


def packed_store(root, batches, torn_tail=False, cursor=0):
    root.mkdir(parents=True)
    identity = {"did": SRC, "endpoint": "https://mod.example"}
    (root / "identity.json").write_text(json.dumps(identity))
    base = {**C.compact_initial(), "cursor": cursor}
    manifest = {"schema": C.PACKED_SCHEMA, "identity": identity, "target_seq": 100, "start_cursor": cursor,
                "base_state": base, "parent": None}
    C.immutable(root / "manifest.json", C.json_bytes(manifest))
    store = C.PackedStore(root, manifest)
    state = json.loads(json.dumps(store.state))
    for frames in batches:
        for raw in frames:
            C.compact_apply(state, raw, C.decode_frame(raw, SRC))
        store.commit(frames, state)
    if torn_tail:
        with open(root / "chunks.jsonl", "ab") as f:
            f.write(b'{"previous_sha256": "torn')
    return root


def legacy_store(root, frames):
    (root / "raw").mkdir(parents=True)
    (root / "identity.json").write_text(json.dumps({"did": SRC}))
    with open(root / "journal.jsonl", "w") as j:
        for raw in frames:
            h = C.sha(raw)
            (root / "raw" / (h + ".cbor")).write_bytes(raw)
            j.write(json.dumps({"kind": "labels", "raw_path": f"raw/{h}.cbor", "sha256": h, "bytes": len(raw)}) + "\n")
    return root


NOW = "2026-10-28T00:00:00Z"
A, B = "did:plc:aaaaaaaaaaaaaaaaaaaaaaaa", "did:plc:bbbbbbbbbbbbbbbbbbbbbbbb"
POST = f"at://{B}/app.bsky.feed.post/3lxyzabcdefgh"


class Labels(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def resolve(self, stores, record_dids=None):
        return labels.resolve((x for s in stores for x in labels.store_labels(s)), NOW, record_dids)

    def test_subject_scopes(self):
        self.assertEqual(labels.subject(A), (A, "account"))
        self.assertEqual(labels.subject(f"at://{A}/app.bsky.actor.profile/self"), (A, "account"))
        self.assertEqual(labels.subject(POST), (B, "record"))
        self.assertEqual(labels.subject("https://x"), ("", ""))

    def test_negation_expiry_cid_and_order(self):
        s = packed_store(self.tmp / "p", [
            [frame(1, lab(A, "porn", "2024-01-01T00:00:00Z")),
             frame(2, lab(A, "spam", "2024-01-01T00:00:00Z"))],
            [frame(3, lab(A, "spam", "2024-02-01T00:00:00Z", neg=True)),          # negated later
             frame(4, lab(B, "!hide", "2024-01-01T00:00:00Z", exp="2025-01-01T00:00:00Z")),  # expired
             frame(5, lab(POST, "nudity", "2024-01-01T00:00:00Z", cid="bafyfakecid")),     # cid-scoped
             frame(6, lab(B, "!warn", "2024-01-01T00:00:00Z", exp="2027-01-01T00:00:00Z"))],  # still in force
            [frame(7, lab(A, "spam", "2024-03-01T00:00:00Z"))],                      # re-applied after negation
        ])
        active = self.resolve([s])
        got = sorted((r["uri"], r["val"]) for r in active.values())
        self.assertEqual(got, sorted([(A, "porn"), (A, "spam"), (B, "!warn"), (POST, "nudity")]))
        self.assertEqual(active[(SRC, POST, "nudity")]["scope"], "record")

    def test_older_negation_does_not_remove_newer_label(self):
        s = packed_store(self.tmp / "p", [[frame(1, lab(A, "spam", "2024-05-01T00:00:00Z")),
                                           frame(2, lab(A, "spam", "2024-04-01T00:00:00Z", neg=True))]])
        self.assertIn((SRC, A, "spam"), self.resolve([s]))

    def test_record_filter_keeps_all_account_labels(self):
        s = packed_store(self.tmp / "p", [[frame(1, lab(POST, "porn", "2024-01-01T00:00:00Z"),
                                                 lab(B, "spam", "2024-01-01T00:00:00Z"))]])
        active = self.resolve([s], record_dids=set())
        self.assertEqual([(r["uri"], r["scope"]) for r in active.values()], [(B, "account")])
        self.assertEqual(len(self.resolve([s], record_dids={B})), 2)

    def test_torn_tail_ignored_and_store_untouched(self):
        s = packed_store(self.tmp / "p", [[frame(1, lab(A, "porn", "2024-01-01T00:00:00Z"))]], torn_tail=True)
        before = {p.name: p.read_bytes() for p in s.rglob("*") if p.is_file()}
        self.assertEqual(len(self.resolve([s])), 1)
        after = {p.name: p.read_bytes() for p in s.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_corrupt_chunk_raises(self):
        s = packed_store(self.tmp / "p", [[frame(1, lab(A, "porn", "2024-01-01T00:00:00Z"))]])
        chunk = next((s / "chunks").iterdir())
        data = bytearray(chunk.read_bytes())
        data[len(data) // 2] ^= 0xFF
        chunk.write_bytes(bytes(data))
        with self.assertRaises(labels.StoreError):
            self.resolve([s])

    def test_legacy_plus_packed_and_wrong_source_rejected(self):
        old = legacy_store(self.tmp / "v1", [frame(1, lab(A, "spam", "2023-05-01T00:00:00Z"))])
        new = packed_store(self.tmp / "v2", [[frame(2, lab(A, "spam", "2024-01-01T00:00:00Z", neg=True))]])
        self.assertEqual(self.resolve([old, new]), {})
        bad = legacy_store(self.tmp / "bad", [cbor2.dumps({"op": 1, "t": "#labels"}) + cbor2.dumps(
            {"seq": 1, "labels": [{"src": "did:plc:other", "uri": A, "val": "x", "cts": "2024-01-01T00:00:00Z"}]})])
        with self.assertRaises(ValueError):
            self.resolve([bad])

    def test_cli_writes_tsv_and_summary(self):
        s = packed_store(self.tmp / "p", [[frame(1, lab(A, "porn", "2024-01-01T00:00:00Z"),
                                                 lab(POST, "nudity", "2024-01-01T00:00:00Z"))]])
        out = self.tmp / "out.tsv"
        labels.main([str(out), str(s), "--now", NOW])
        rows = [l.split("\t") for l in out.read_text().splitlines()]
        self.assertEqual(sorted((r[0], r[1], r[3]) for r in rows), [(A, "account", "porn"), (B, "record", "nudity")])
        summary = json.loads((self.tmp / "out.tsv.summary.json").read_text())
        self.assertEqual(summary["active_labels"], 2)


class UnionOfSplitStores(unittest.TestCase):
    """Parallel replay streams (bluesky-v2 + bluesky-v3-s*): overlaps deduped by seq, holes refused."""

    def stores(self, tmp, starts_and_seqs):
        out = []
        for i, (cur, seqs) in enumerate(starts_and_seqs):
            frames = [frame(q, lab(f"did:plc:{chr(97 + i) * 24}", f"v{q}", f"2026-01-01T00:00:{q % 60:02d}Z"))
                      for q in seqs]
            out.append(packed_store(tmp / f"s{i}", [frames], cursor=cur))
        return out

    def test_overlap_dedupe_and_labeler_gaps(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = pathlib.Path(t)
            # s0 covers 1..12 with labeler gap 5; s1 starts at cursor 10 (overlap 11, 12), gap 15; s2 at 14.
            st = self.stores(tmp, [(0, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]), (10, [11, 12, 13, 14, 16]),
                                   (14, [16, 17, 18])])
            per, union = {}, {}
            seqs = [q for q, _l in labels.union_labels(list(reversed(st)), per, union)]
            self.assertEqual(seqs, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18])
            self.assertEqual(union["duplicate_frames"], 3)       # 11, 12 from s1 and 16 from s2
            self.assertEqual(union["gap_seqs"], 2)               # 5 and 15: the labeler's own
            self.assertEqual(union["max_seq"], 18)
            self.assertTrue(union["gaps_within_store_gaps"])
            self.assertEqual(union["order"][0], str(st[0]))

    def test_hole_between_streams_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = pathlib.Path(t)
            st = self.stores(tmp, [(0, [1, 2, 3]), (5, [6, 7])])   # seqs 4, 5 never requested by anyone
            with self.assertRaises(labels.StoreError):
                list(labels.union_labels(st, {}, {}))

    def test_label_root_discovery_and_coverage(self):
        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t)
            packed_store(root / "bluesky-v2", [[frame(1, lab(A, "spam", "2026-01-01T00:00:00Z"))]])
            packed_store(root / "bluesky-v3-s1", [[frame(2, lab(A, "porn", "2026-01-01T00:00:01Z"))]], cursor=1)
            packed_store(root / "speed-probe-x", [[frame(1, lab(A, "x", "2026-01-01T00:00:00Z"))]])
            self.assertEqual([p.name for p in labels.label_stores(root)], ["bluesky-v2", "bluesky-v3-s1"])
            out = root / "o.tsv"
            labels.main([str(out), "--label-root", str(root), "--now", NOW, "--coverage-to", "2"])
            self.assertEqual(len(out.read_text().splitlines()), 2)
            with self.assertRaises(labels.StoreError):
                labels.main([str(out), "--label-root", str(root), "--now", NOW, "--coverage-to", "3"])


if __name__ == "__main__":
    unittest.main()
