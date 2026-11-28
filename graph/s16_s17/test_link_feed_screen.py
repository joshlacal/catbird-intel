#!/usr/bin/env python3
"""Tests for link_feed_screen.py and the link-feed gate in build_onboarding_signals.py."""
from __future__ import annotations

import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_onboarding_signals as S  # noqa: E402
import link_feed_screen as LF  # noqa: E402
import test_onboarding_signals_builder as TB  # noqa: E402

T = TB.T
FEED, ANSWERED, MIXED, CHATTY, SHORT, OUTLET = POOL = [
    "did:plc:linkfeed", "did:plc:answered", "did:plc:mixedsites", "did:plc:chatty", "did:plc:short", "did:plc:outlet"]
TEXT = "a headline that is long enough to sample"


def test_link_host():
    assert LF.link_host("https://www.Shashi.co/2026/11/x.html#gsc") == "shashi.co"
    assert LF.link_host("http://user@buff.ly:8080/abc") == "buff.ly"
    assert LF.link_host("at://did:plc:x/app.bsky.feed.post/1") == "did"  # never passed in practice; still parses
    assert LF.link_host("") == "" and LF.link_host(None) == "" and LF.link_host("not a url") == ""


def test_mask_thresholds():
    m = LF.link_feed_mask(
        np.array([10, 9, 10, 10, 10, 20, 20]),
        np.array([10, 9, 6, 10, 10, 14, 20]),
        np.array([10, 9, 6, 5, 10, 9, 20]),
        np.array([0, 0, 0, 0, 1, 0, 1]))
    # ok, too few posts, links 60%, top site 50%, 1 responder per 10 posts, top site 64% of 14 links, 1 per 20
    assert m.tolist() == [True, False, False, False, False, True, False]


def scenario(tmp: pathlib.Path):
    ex = TB.Export()
    base = T("2026-11-08T00:00:00")
    for i in range(12):
        c = base + i * 3 * 3600 * 1_000_000
        site = f"https://www.myblog.example/2026/11/p{i}.html"
        ex.post(FEED, c, TEXT, link=site if i < 10 else "", facets=(site,) if i == 10 else (), clock=1)
        ex.post(ANSWERED, c, TEXT, link=site, clock=2)
        ex.post(MIXED, c, TEXT, link=f"https://site{i % 4}.example/x", clock=3)
        ex.post(CHATTY, c, TEXT, link=site, clock=4)
        ex.post(OUTLET, c, TEXT, link=site, clock=6)
        if i < 9:
            ex.post(SHORT, c, TEXT, link=site, clock=5)
    # FEED: its own replies to others do not count as top-level, and a self-reply is not a responder
    ex.post(FEED, base + 3600 * 1_000_000, "reply", reply="at://did:plc:someone/app.bsky.feed.post/3abc", clock=9)
    ex.post(FEED, base + 2 * 3600 * 1_000_000, "thread", reply=f"at://{FEED}/app.bsky.feed.post/3abc", clock=8)
    # ANSWERED: three different people reply or quote (3 / 12 posts >= 0.05)
    for j, who in enumerate(["did:plc:r1", "did:plc:r2", "did:plc:r3"]):
        kw = {"quote": f"at://{ANSWERED}/app.bsky.feed.post/3xyz"} if j == 2 else \
             {"reply": f"at://{ANSWERED}/app.bsky.feed.post/3xyz"}
        ex.post(who, base + j * 60_000_000, "nice", clock=j, **kw)
    # OUTLET: one person replies five times -> one responder (1 / 12 < 0.05 is false: 0.083) -> kept
    for j in range(5):
        ex.post("did:plc:fan", base + j * 60_000_000, "hm", reply=f"at://{OUTLET}/app.bsky.feed.post/3q{j}", clock=j)
    # CHATTY: 2 of its posts are replies to others -> 10 top-level... still a feed by links; give it 1 responder
    # per 10 posts so it is kept
    ex.post("did:plc:pal", base, "hey", reply=f"at://{CHATTY}/app.bsky.feed.post/3abc", clock=1)
    root = tmp / "export"
    ex.write(root, "2026-11-08", "2026-11-10")
    pool = tmp / "pool.json"
    pool.write_text(S.json.dumps({"poolVersion": "lf", "communities": [
        {"id": "c", "members": [{"did": d, "role": "anchor_creator", "followers": 500} for d in POOL]}]}))
    return root, pool


def run(**kw):
    tmp = pathlib.Path(tempfile.mkdtemp())
    root, pool = scenario(tmp)
    args = dict(activity_days=3, follow_from="2026-11-08", follow_to="2026-11-09", removal_days=1, k=1.0,
                min_eligible=5, viewer_day_cap=3, max_dormant_share=1.0, max_link_feed_share=1.0, sanity_n=0)
    return S.build(root, pool, "2026-11-10", **{**args, **kw})


def test_builder_gates_only_the_unanswered_single_site_feed():
    res = run()
    assert res["code"] == 0, res["error"]
    d = res["detail"]
    i = {x: n for n, x in enumerate(d["dids"])}
    assert d["originals14"][i[FEED]] == 12 and d["linked14"][i[FEED]] == 11 and d["topDomain14"][i[FEED]] == 11
    assert d["responders14"][i[FEED]] == 0  # the self-reply is not counted
    assert d["responders14"][i[ANSWERED]] == 3 and d["responders14"][i[OUTLET]] == 1
    assert d["topDomain14"][i[MIXED]] == 3
    assert d["linkFeed"].tolist() == [x == FEED for x in d["dids"]]
    assert res["signals"]["accounts"][FEED] == {"multiplier": 0.0}
    for did in (ANSWERED, MIXED, CHATTY, SHORT, OUTLET):
        assert res["signals"]["accounts"].get(did, {}).get("multiplier") != 0.0
    lf = res["report"]["linkFeed"]
    assert lf["flagged"]["total"] == 1
    assert lf["accounts"] == [{"did": FEED, "role": "anchor_creator", "followers": 500, "topLevel": 12, "linked": 11,
                               "topHost": "myblog.example", "topHostPosts": 11, "responders": 0}]
    assert res["report"]["output"]["linkFeedZero"] == 1


def test_builder_stops_when_too_many_look_like_link_feeds():
    res = run(max_link_feed_share=0.1)  # 1 of 6 = 0.167
    assert res["code"] == 4 and res["signals"] is None and "link-feed share" in res["report"]["stopped"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"{len(tests)} passed")
