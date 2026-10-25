"""Storage, resume, undo and context-join tests for the labeling tool (offline, temp dirs only)."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from labeling.agreement import load_gold
from labeling.axes import axis_spec, load_rubric, normalize_value
from labeling.pools import load_context_index, make_card
from labeling.server import LabelingHandler
from labeling.service import NotesStore, new_session, session_payload
from labeling.store import Session, seeded_order

KEYS = [f"at://did:plc:a{i}/app.bsky.feed.post/{i}" for i in range(6)]


def make_session(root: Path, axis: str = "politicalness", seed: int = 7) -> Session:
    return Session.create(
        kind="axis", axis=axis, pool="test", subset=None, item_keys=KEYS, pool_sha256="p",
        context_sha256={}, rubric_sha256="r", seed=seed, root=root,
    )


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_order_is_seeded_and_recorded(self) -> None:
        s = make_session(self.root, seed=7)
        self.assertEqual(s.order, seeded_order(KEYS, 7))
        self.assertEqual(s.meta["seed"], 7)
        self.assertNotEqual(s.order, sorted(KEYS))  # shuffled, not file order
        self.assertEqual(make_session(self.root, seed=7).order, s.order)

    def test_label_rows_are_append_only_with_required_fields(self) -> None:
        s = make_session(self.root)
        s.label(s.order[0], "2", ms_to_decide=1500, note_ids=["n-000000000001"])
        before = s.labels_path.read_bytes()
        s.label(s.order[1], 0, ms_to_decide=900)
        self.assertTrue(s.labels_path.read_bytes().startswith(before))
        row = json.loads(before)
        for field in ("uri", "axis", "value", "ms_to_decide", "ts", "session_id", "note_ids"):
            self.assertIn(field, row)
        self.assertEqual((row["value"], row["axis"], row["row_id"]), (2, "politicalness", 1))
        self.assertTrue(s.labels_path.name == f"labels-politicalness-{s.id}.jsonl")

    def test_invalid_values_are_refused(self) -> None:
        s = make_session(self.root)
        for bad in (4, "x", "", True, "2.0"):
            with self.assertRaises(ValueError):
                s.label(s.order[0], bad, ms_to_decide=1)
        b = make_session(self.root, axis="spam")
        with self.assertRaises(ValueError):
            b.label(b.order[0], 2, ms_to_decide=1)  # binary axis
        b.label(b.order[0], "?", ms_to_decide=1)
        self.assertEqual(normalize_value("spam", "1"), 1)

    def test_resume_starts_at_first_unlabeled_item(self) -> None:
        s = make_session(self.root)
        s.label(s.order[0], 1, ms_to_decide=10)
        s.label(s.order[1], 3, ms_to_decide=10)
        s.label(s.order[3], 0, ms_to_decide=10)  # skipped item 2
        reopened = Session.open(s.id, root=self.root)
        self.assertEqual(reopened.state().cursor, 2)
        self.assertEqual(reopened.order, s.order)

    def test_undo_appends_correction_and_walks_back(self) -> None:
        s = make_session(self.root)
        s.label(s.order[0], 1, ms_to_decide=10)
        s.label(s.order[1], 2, ms_to_decide=10)
        n_bytes = len(s.labels_path.read_bytes())
        undo = s.undo()
        self.assertEqual((undo["op"], undo["undoes_row_id"], undo["key"]), ("undo", 2, s.order[1]))
        self.assertGreater(len(s.labels_path.read_bytes()), n_bytes)
        self.assertEqual(s.state().cursor, 1)
        s.undo()
        self.assertEqual(s.state().cursor, 0)
        self.assertIsNone(s.undo())
        self.assertEqual(len(s.rows()), 4)

    def test_relabel_supersedes_and_undo_restores_previous(self) -> None:
        s = make_session(self.root)
        s.label(s.order[0], 1, ms_to_decide=10)
        s.label(s.order[0], 3, ms_to_decide=10)
        self.assertEqual(s.state().effective[s.order[0]]["value"], 3)
        s.undo()
        self.assertEqual(s.state().effective[s.order[0]]["value"], 1)

    def test_torn_write_is_detected(self) -> None:
        s = make_session(self.root)
        s.label(s.order[0], 1, ms_to_decide=10)
        with s.labels_path.open("a") as fh:
            fh.write('{"op": "label"')
        with self.assertRaises(ValueError):
            s.state()

    def test_agreement_loader_reads_store_output(self) -> None:
        s = make_session(self.root)
        s.label(s.order[0], 2, ms_to_decide=10)
        s.label(s.order[1], "?", ms_to_decide=10)
        s.label(s.order[2], 1, ms_to_decide=10)
        s.undo()
        gold = load_gold([s.labels_path])
        self.assertEqual(gold.values("politicalness"), {s.order[0]: 2, s.order[1]: "?"})

    def test_session_id_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            Session.open("../../etc", root=self.root)


class ContextJoinTests(unittest.TestCase):
    def test_card_joins_parent_and_quote_and_hides_author(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Path(tmp) / "ctx.jsonl"
            ctx.write_text(
                json.dumps({"uri": "at://p/app.bsky.feed.post/1", "text": "parent says hi"}) + "\n"
                + json.dumps({"uri": "at://q/app.bsky.feed.post/2", "text": "x" * 1500}) + "\n"
            )
            index, hashes = load_context_index([ctx])
        self.assertEqual(len(hashes), 1)
        row = {
            "uri": "at://did:plc:author/app.bsky.feed.post/3", "author_did": "did:plc:author", "text": "reply",
            "post_type": "reply", "reply_parent_uri": "at://p/app.bsky.feed.post/1",
            "quote_uri": "at://q/app.bsky.feed.post/2", "image_count": 2, "external_uri": "https://example.com/a",
            "stratum": "random_reply",
        }
        card = make_card(row, index)
        self.assertEqual(card["reply_parent"], {"available": True, "text": "parent says hi", "truncated": False})
        self.assertTrue(card["quoted"]["truncated"])
        self.assertEqual(len(card["quoted"]["text"]), 1200)
        self.assertEqual(card["media"]["image_count"], 2)
        self.assertEqual(card["media"]["external_host"], "example.com")
        self.assertFalse(card["media"]["alt_text_exported"])
        self.assertNotIn("did:plc:author", json.dumps({k: v for k, v in card.items() if k != "uri"}))

    def test_missing_context_is_marked_unavailable(self) -> None:
        card = make_card({"uri": "u", "text": "t", "reply_parent_uri": "at://gone/app.bsky.feed.post/9"}, {})
        self.assertEqual(card["reply_parent"], {"available": False, "text": None})
        self.assertIsNone(card["quoted"])

    def test_axis_spec_uses_rubric_definitions(self) -> None:
        rubric, _ = load_rubric()
        spec = axis_spec("outraged", rubric)
        self.assertEqual(spec["definition"], rubric["components"]["outraged"]["definition"])
        self.assertEqual([k["key"] for k in spec["keys"]], ["0", "1", "2", "3", "?"])
        self.assertEqual([k["key"] for k in axis_spec("spam", rubric)["keys"]], ["0", "1", "?"])


class RealPoolTests(unittest.TestCase):
    """Uses the committed frozen holdout; skipped only if the file is missing."""

    def test_holdout_session_payload_resumes_and_joins_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            s = new_session(kind="axis", axis="critical", pool="holdout600", seed=1, root=Path(tmp))
            payload = session_payload(s)
            self.assertEqual(len(payload["cards"]), 600)
            replies = [c for c in payload["cards"] if c["reply_parent"]]
            available = sum(c["reply_parent"]["available"] for c in replies)
            self.assertGreater(available / len(replies), 0.95)
            s.label(payload["cards"][0]["key"], 2, ms_to_decide=1200)
            self.assertEqual(session_payload(s)["cursor"], 1)


class ServerTests(unittest.TestCase):
    def test_label_undo_and_note_over_http(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcribed = threading.Event()

            def fake_transcribe(path: Path) -> dict:
                transcribed.set()
                return {"text": "sarcastic dunk", "engine": "fake"}

            handler = type("H", (LabelingHandler,), {"sessions_root": root,
                                                      "notes": NotesStore(root / "notes", transcriber=fake_transcribe)})
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def call(path: str, body: object = None, ctype: str = "application/json") -> dict:
                data = body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None)
                req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET",
                                             headers={"Content-Type": ctype})
                with urllib.request.urlopen(req) as res:
                    return json.loads(res.read())

            try:
                sid = call("/api/sessions", {"kind": "axis", "axis": "spam", "pool": "holdout600"})["session_id"]
                cards = call(f"/api/session/{sid}")["cards"]
                key = cards[0]["key"]
                note = call(f"/api/session/{sid}/note?key={urllib.request.quote(key)}&duration_ms=900", b"\x1a\x45audio", "audio/webm")
                res = call(f"/api/session/{sid}/label", {"key": key, "value": 1, "ms_to_decide": 1800, "note_ids": [note["note_id"]]})
                self.assertEqual((res["row"]["note_ids"], res["cursor"]), ([note["note_id"]], 1))
                self.assertEqual(call(f"/api/session/{sid}/undo", {})["cursor"], 0)
                self.assertTrue(transcribed.wait(5))
                for _ in range(50):
                    notes = call(f"/api/notes?session={sid}")["notes"]
                    if notes and notes[0].get("transcript"):
                        break
                    threading.Event().wait(0.05)
                self.assertEqual(notes[0]["transcript"], "sarcastic dunk")
                self.assertTrue((root / "notes" / sid / f"{note['note_id']}.webm").exists())
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
