#!/usr/bin/env python3
"""Tests for why_picked.py, its use in build_intel_serving_v2.classify_member, and --link-stats-out."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_intel_serving_v2 as B  # noqa: E402
import build_onboarding_signals as S  # noqa: E402
import test_link_feed_screen as TL  # noqa: E402
import why_picked as W  # noqa: E402


def test_anchor_why_states_only_measured_facts():
    assert W.anchor_why(10, (46, 43)) == "Shares articles and links · ~10/wk"
    assert W.anchor_why(10, (10, 5)) == "Shares articles and links · ~10/wk"  # half linked
    assert W.anchor_why(10, (10, 4)) == "Core scene anchor · mostly original posts · ~10/wk"
    assert W.anchor_why(3, (4, 4)) == "Core scene anchor · mostly original posts · ~3/wk"  # too few posts
    assert W.anchor_why(3, None) == "Core scene anchor · mostly original posts · ~3/wk"
    assert "strong" not in W.anchor_why(3, None)


def test_classify_member_uses_link_stats():
    meta = {"lf": {"rp": 0.0, "ry": 0.05}, "img": 0.05, "ppd": 1.45}
    try:
        B.LINK_STATS.clear()
        assert B.classify_member(meta, None, "did:plc:a")[5] == "Core scene anchor · mostly original posts · ~10/wk"
        B.LINK_STATS["did:plc:a"] = (46, 43)
        role, *_, why = B.classify_member(meta, None, "did:plc:a")
        assert role == "anchor_creator" and why == "Shares articles and links · ~10/wk"
        assert B.classify_member(meta, None, "did:plc:other")[5].startswith("Core scene anchor")
    finally:
        B.LINK_STATS.clear()


def test_relabel_touches_only_anchor_labels():
    doc = {"communities": [{"members": [
        {"did": "did:plc:feed", "role": "anchor_creator", "whyPicked": "Core scene anchor · strong original posts · ~10/wk"},
        {"did": "did:plc:essay", "role": "anchor_creator", "whyPicked": "Core scene anchor · strong original posts · ~3/wk"},
        {"did": "did:plc:cur", "role": "anchor_creator",
         "whyPicked": "Core scene anchor · strong original posts · ~7/wk · Originals Only suggested"},
        {"did": "did:plc:art", "role": "visual_maker", "whyPicked": "Visual maker · 80% media posts · ~23/wk"},
    ]}]}
    stats = {"did:plc:feed": (46, 43), "did:plc:essay": (12, 1), "did:plc:art": (30, 30)}
    assert W.relabel(doc, stats) == {"anchors": 3, "linkSharers": 1, "changed": 3}
    whys = [m["whyPicked"] for m in doc["communities"][0]["members"]]
    assert whys == ["Shares articles and links · ~10/wk", "Core scene anchor · mostly original posts · ~3/wk",
                    "Core scene anchor · mostly original posts · ~7/wk · Originals Only suggested",
                    "Visual maker · 80% media posts · ~23/wk"]
    assert W.relabel(doc, stats)["changed"] == 0  # idempotent


def test_signals_builder_writes_link_stats():
    tmp = pathlib.Path(tempfile.mkdtemp())
    root, pool = TL.scenario(tmp)
    out, stats = tmp / "signals.json", tmp / "link_stats.json"
    rc = S.main(["--export-root", str(root), "--pool", str(pool), "--as-of", "2026-11-10", "--activity-days", "3",
                 "--follow-from", "2026-11-08", "--follow-to", "2026-11-09", "--removal-days", "1", "--k", "1",
                 "--min-eligible", "5", "--max-dormant-share", "1", "--max-link-feed-share", "1",
                 "--out", str(out), "--link-stats-out", str(stats)])
    assert rc == 0
    doc = json.loads(stats.read_text())
    assert doc["accounts"][TL.FEED] == [12, 11] and doc["accounts"][TL.SHORT] == [9, 9]
    assert W.load_link_stats(stats)[TL.FEED] == (12, 11)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"{len(tests)} passed")
