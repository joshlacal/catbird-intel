"""Unit tests for the S6 picker's keyword screen, S5 reader and verifier (no network, no lake files)."""
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from s6picker import keywords, live  # noqa: E402
from s6picker.build import verify  # noqa: E402


class S6Picker(unittest.TestCase):
    def test_keywords_whole_word_and_hashtag(self):
        assert keywords.classes("News") == ["news"]
        assert keywords.classes("#resist") == ["politics"]
        assert keywords.classes("trans artists") == ["sexuality_gender"]
        assert keywords.classes("18+ only") == ["adult"]
        assert keywords.classes("ニュース速報") == ["news"]
        assert keywords.classes("Brazilian artists") == ["ethnicity_nationality"]
        assert keywords.classes("high school students") == ["likely_minors"]
        assert keywords.classes("students") == []  # student(s) needs a school-age cue
        assert keywords.classes("newsletter writers") == []  # whole word only
        assert keywords.classes("birding") == []


    @staticmethod
    def _s5(tmp, receipt):
        d = tmp / "s5"
        (d / "work/profiles").mkdir(parents=True)
        (d / "receipt.json").write_text(json.dumps(receipt))
        rows = [{"did": "did:plc:ok", "found": True, "handle": "ok.test", "postsCount": 10, "followersCount": 5, "labels": []},
                {"did": "did:plc:gone", "found": False},
                {"did": "did:plc:nsfw", "found": True, "handle": "n.test", "postsCount": 9, "followersCount": 1,
                 "labels": [{"src": "did:plc:mod", "val": "porn", "neg": False}]},
                {"did": "did:plc:nu", "found": True, "handle": "nu.test", "postsCount": 9, "followersCount": 1,
                 "labels": [{"src": "did:plc:nu", "val": "!no-unauthenticated", "neg": False}]},
                {"did": "did:plc:negated", "found": True, "handle": "neg.test", "postsCount": 9, "followersCount": 1,
                 "labels": [{"src": "did:plc:mod", "val": "spam", "neg": True}]},
                {"did": "did:plc:bad", "found": True, "handle": "handle.invalid", "postsCount": 9, "followersCount": 1, "labels": []}]
        lines = [{"batch": 0, "rows": rows}, {"batch": 1, "error": 502, "detail": {}, "dids": ["did:plc:err"]}]
        (d / "work/profiles/profiles.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
        (d / "work/other.jsonl").write_text(json.dumps({"did": "did:plc:x", "items": 3}) + "\n")
        return d


    def test_live_statuses(self):
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        lv = live.load(self._s5(tmp_path, {"controls": {"labelled": True, "taken_down": True}}))
        assert lv.present and lv.controls == {"controls.labelled": True, "controls.taken_down": True}
        assert lv.status("did:plc:ok") == "ok"
        assert lv.status("did:plc:gone") == "missing"
        assert lv.status("did:plc:nsfw") == "labelled"
        assert lv.status("did:plc:nu") == "no_unauth"
        assert lv.status("did:plc:negated") == "ok"
        assert lv.status("did:plc:bad") == "invalid_handle"
        assert lv.status("did:plc:err") == "unchecked"  # a failed call is not a "not found"
        assert lv.status("did:plc:never") == "unchecked"
        assert lv.status("did:plc:x") == "unchecked"  # other files are skipped
        assert len(lv.response_hashes) == 1 and len(lv.files) == 1


    def test_live_absent(self):
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        lv = live.load(tmp_path / "nothing")
        assert not lv.present and lv.status("did:plc:ok") == "unchecked"


    def test_verify_catches_violations(self):
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        lv = live.load(self._s5(tmp_path, {}))
        m = lambda d, f=10: {"did": d, "followers": f}
        ok = [m("did:plc:ok")] * 25
        data = {"cards": {
            "1": {"community": 1, "sensitive": False, "members": ok, "packs": [], "feeds": []},
            "2": {"community": 2, "sensitive": True, "members": [m("did:plc:ok")], "packs": [], "feeds": []},
            "3": {"community": 3, "sensitive": False, "members": ok[:24] + [m("did:plc:nsfw")], "packs": [], "feeds": []},
            "4": {"community": 4, "sensitive": False, "members": ok, "packs": [], "feeds": [], },
            "5": {"community": 5, "sensitive": False, "members": [m("did:plc:ok", 600_000)] * 3 + ok[:22], "packs": [], "feeds": []},
        }}
        v = verify(data, lv, set(), {}, {4: {"sensitivity": ["likely_minors"]}}, {})["violations"]
        assert not any(x.startswith("card 1:") for x in v)
        assert any(x.startswith("card 2: sensitive") for x in v)
        assert any(x.startswith("card 3: member did:plc:nsfw live status labelled") for x in v)
        assert any(x.startswith("card 4: Claude likely_minors") for x in v)
        assert any(x.startswith("card 5: popularity ceiling") for x in v)


if __name__ == "__main__":
    unittest.main()
