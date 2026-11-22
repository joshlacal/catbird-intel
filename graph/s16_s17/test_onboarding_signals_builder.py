#!/usr/bin/env python3
"""Tests for build_onboarding_signals.py on tiny synthetic export day folders.

Covers: dedupe by rev, tombstones, the dormant gate (incl. backdated imports), the status gate (max seq, not
file order), sample-post choice, burst exclusion, the per-viewer-day removal cap, 14-day censoring, the
dormant-share stop rule, the live-path refusal, determinism, and the output shape the consumer
(src/onboarding_signals.ts) accepts.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_onboarding_signals as S  # noqa: E402

B32 = "234567abcdefghijklmnopqrstuvwxyz"
SECRET_TEXT = "a long enough sample post text body"


def T(s: str) -> int:
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp() * 1_000_000)


def tid(us: int, clock: int = 0) -> str:
    v = (us << 10) | clock
    return "".join(B32[(v >> (5 * (12 - i))) & 31] for i in range(13))


def day_of(us: int) -> str:
    return dt.datetime.fromtimestamp(us / 1e6, tz=dt.timezone.utc).date().isoformat()


TS = pa.timestamp("ms", tz="UTC")
FOLLOW_SCHEMA = pa.schema([("did", pa.string()), ("rkey", pa.string()), ("rev", pa.uint64()), ("is_deleted", pa.uint8()),
                           ("seq", pa.uint64()), ("indexed_at", TS), ("subject", pa.string())])
POST_SCHEMA = pa.schema([("did", pa.string()), ("rkey", pa.string()), ("rev", pa.uint64()), ("is_deleted", pa.uint8()),
                         ("indexed_at", TS), ("created_at", TS), ("text", pa.string()), ("reply_parent", pa.string()),
                         ("image_count", pa.uint8()), ("has_video", pa.uint8()), ("self_labels", pa.list_(pa.string())),
                         ("politicalness", pa.float32()), ("heat", pa.float32())])
STATUS_SCHEMA = pa.schema([("did", pa.string()), ("seq", pa.uint64()), ("indexed_at", TS), ("active", pa.uint8()),
                           ("status", pa.string())])

POOL = ["did:plc:active1", "did:plc:churn", "did:plc:dormant1", "did:plc:deleted", "did:plc:taken", "did:plc:react",
        "did:plc:same", "did:plc:backdated"]
A, CH, DO, DE, TK, RE, SA, BD = POOL


class Export:
    def __init__(self):
        self.rows = {"follows": {}, "posts": {}, "account_status": {}}
        self.seq = 0

    def _add(self, table, day, row):
        self.seq += 1
        row.setdefault("seq", self.seq)
        self.rows[table].setdefault(day, []).append(row)

    def follow(self, viewer, subject, create_us, clock, rev=10, delete_us=None, delete_rev=None, extra_copy=False):
        rk = tid(create_us, clock)
        row = {"did": viewer, "rkey": rk, "rev": rev, "is_deleted": 0, "indexed_at": (create_us + 1_000_000) // 1000,
               "subject": subject}
        self._add("follows", day_of(create_us + 1_000_000), row)
        if extra_copy:
            self._add("follows", day_of(create_us + 86_400_000_000), dict(row))
        if delete_us is not None:
            self._add("follows", day_of(delete_us), {"did": viewer, "rkey": rk, "rev": delete_rev or rev + 5,
                                                     "is_deleted": 1, "indexed_at": delete_us // 1000, "subject": ""})
        return rk

    def post(self, did, created_us, text="", reply="", images=0, video=0, labels=(), political=None, heat=None,
             rev=10, indexed_us=None, rkey=None, deleted=False):
        idx = indexed_us if indexed_us is not None else created_us + 2_000_000
        rk = rkey or tid(created_us, 7)
        row = {"did": did, "rkey": rk, "rev": rev, "is_deleted": int(deleted), "indexed_at": idx // 1000,
               "created_at": 0 if deleted else created_us // 1000, "text": "" if deleted else text,
               "reply_parent": "" if deleted else reply, "image_count": images, "has_video": video,
               "self_labels": [] if deleted else list(labels), "politicalness": political, "heat": heat}
        self._add("posts", day_of(idx), row)
        return rk

    def status(self, did, day, seq, active, status):
        self._add("account_status", day, {"did": did, "seq": seq, "indexed_at": T(day + "T06:00:00") // 1000,
                                          "active": active, "status": status})

    def write(self, root: pathlib.Path, first: str, last: str):
        schemas = {"follows": FOLLOW_SCHEMA, "posts": POST_SCHEMA, "account_status": STATUS_SCHEMA}
        for d in S.days(dt.date.fromisoformat(first), dt.date.fromisoformat(last)):
            p = root / d.isoformat()
            p.mkdir(parents=True, exist_ok=True)
            for name, schema in schemas.items():
                rows = self.rows[name].get(d.isoformat(), [])
                pq.write_table(pa.Table.from_pylist(rows, schema=schema), p / f"{name}.parquet")


def scenario(tmp: pathlib.Path):
    ex = Export()
    uris = {}
    # posts: window for as-of 11-10 with 3 activity days = 11-08..11-10
    ex.post(A, T("2026-11-10T09:00:00"), SECRET_TEXT, reply="at://did:plc:x/app.bsky.feed.post/3abc")
    ex.post(A, T("2026-11-10T08:00:00"), "hi")
    ex.post(A, T("2026-11-10T07:00:00"), SECRET_TEXT, labels=["nudity"])
    ex.post(A, T("2026-11-10T06:00:00"), SECRET_TEXT, political=0.9)
    uris["media"] = ex.post(A, T("2026-11-09T10:00:00"), "", images=1)
    long_rk = ex.post(A, T("2026-11-09T08:00:00"), SECRET_TEXT)
    ex.post(A, T("2026-11-09T08:00:00"), SECRET_TEXT, rkey=long_rk, indexed_us=T("2026-11-10T01:00:00"))  # same rev dup
    del_rk = ex.post(A, T("2026-11-10T10:00:00"), SECRET_TEXT, rev=1)
    ex.post(A, T("2026-11-10T10:00:00"), rev=2, rkey=del_rk, deleted=True, indexed_us=T("2026-11-10T11:00:00"))
    ex.post(CH, T("2026-11-09T12:00:00"), "ok")
    ex.post(DO, T("2026-11-05T12:00:00"), SECRET_TEXT)  # only before the window
    d_rk = tid(T("2026-11-09T12:00:00"), 7)
    ex.post(DE, T("2026-11-09T12:00:00"), rev=2, rkey=d_rk, deleted=True, indexed_us=T("2026-11-09T12:00:05"))  # first
    ex.post(DE, T("2026-11-09T12:00:00"), SECRET_TEXT, rev=1, rkey=d_rk)
    ex.post(TK, T("2026-11-09T12:00:00"), SECRET_TEXT)
    uris["react"] = ex.post(RE, T("2026-11-09T15:00:00"), "", video=1)
    ex.post(SA, T("2026-11-08T10:00:00"), SECRET_TEXT)
    same_rk = ex.post(SA, T("2026-11-09T16:00:00"), SECRET_TEXT)
    ex.post(BD, T("2024-01-01T00:00:00"), SECRET_TEXT, indexed_us=T("2026-11-09T09:00:00"))  # backdated import
    ex.post("did:plc:outsider", T("2026-11-09T09:00:00"), SECRET_TEXT)
    # posts inside the censoring run window (11-04..11-06) for a few accounts
    ex.post(TK, T("2026-11-04T12:00:00"), SECRET_TEXT)
    # status: taken's max seq is takendown although a later file holds a lower-seq active row
    ex.status(TK, "2026-10-30", 3, 1, "")
    ex.status(TK, "2026-11-05", 7, 0, "takendown")
    ex.status(TK, "2026-11-09", 6, 1, "")
    ex.status(RE, "2026-11-02", 5, 0, "takendown")
    ex.status(RE, "2026-11-03", 9, 1, "")
    ex.status("did:plc:outsider", "2026-11-03", 4, 0, "deleted")
    # follows (create window 11-01..11-04)
    for i in range(20):
        ex.follow(f"did:plc:v{i:02d}", A, T("2026-11-01T12:00:00") + i * 60_000_000, i, extra_copy=(i == 0))
    for i in range(20):
        c = T("2026-11-01T13:00:00") + i * 60_000_000
        if i < 10:
            ex.follow(f"did:plc:w{i:02d}", CH, c, i, delete_us=c + 6 * 3600 * 1_000_000)
        elif i < 12:
            ex.follow(f"did:plc:w{i:02d}", CH, c, i, delete_us=c + 5 * 86_400_000_000)  # after the 3-day window
        elif i == 12:
            ex.follow(f"did:plc:w{i:02d}", CH, c, i, rev=100, delete_us=c + 3600 * 1_000_000, delete_rev=50)
        else:
            ex.follow(f"did:plc:w{i:02d}", CH, c, i)
    b = T("2026-11-01T15:00:00")
    for j in range(12):  # burst: 12 follows within 16.5 s
        subj = CH if j == 0 else f"did:plc:x{j:02d}"
        ex.follow("did:plc:b1", subj, b + j * 1_500_000, j, delete_us=(b + 3600 * 1_000_000) if j == 0 else None)
    b = T("2026-11-01T15:10:00")
    for j in range(9):  # 9 follows within 16 s: not a burst
        ex.follow("did:plc:b2", A if j == 0 else f"did:plc:y{j:02d}", b + j * 2_000_000, j)
    for j, subj in enumerate([CH, A, SA, TK, RE]):  # heavy unfollower: 5 removals on one day
        ex.follow("did:plc:h1", subj, T("2026-11-01T14:00:00") + j * 60_000_000, j,
                  delete_us=T("2026-11-02T10:00:00") + j * 60_000_000)
    for i in range(5):  # created late: censored when as-of is 11-06
        ex.follow(f"did:plc:c{i:02d}", A, T("2026-11-04T12:00:00") + i * 60_000_000, i)
    root = tmp / "export"
    ex.write(root, "2026-10-30", "2026-11-10")
    old_rk = tid(T("2026-10-20T10:00:00"), 7)
    members = []
    for k, d in enumerate(POOL):
        m = {"did": d, "role": "maker" if k % 2 else "talker", "followers": [500, 5000, 50000][k % 3],
             "estimatedPostsPerWeek": k, "handle": "secret-handle", "displayName": "Secret Name"}
        if d == SA:
            m["samplePostUri"] = f"at://{SA}/app.bsky.feed.post/{same_rk}"
        elif d != RE:
            m["samplePostUri"] = f"at://{d}/app.bsky.feed.post/{old_rk}"
        members.append(m)
    pool = tmp / "pool.json"
    pool.write_text(json.dumps({"poolVersion": "test-pool", "communities": [{"id": "c1", "members": members[:5]},
                                                                            {"id": "c2", "members": members[3:]}]}))
    return root, pool, uris


KW = dict(activity_days=3, follow_from="2026-11-01", follow_to="2026-11-04", removal_days=3, k=1.0, min_eligible=5,
          viewer_day_cap=3, max_dormant_share=1.0, sanity_n=50)


def run(as_of="2026-11-10", **kw):
    tmp = pathlib.Path(tempfile.mkdtemp())
    root, pool, uris = scenario(tmp)
    res = S.build(root, pool, as_of, **{**KW, **kw})
    return res, uris, root, pool, tmp


def consumer_accepts(doc: dict) -> None:
    """Mirror of parseOnboardingSignals in src/onboarding_signals.ts plus the builder's own stricter contract."""
    assert set(doc) == {"signalsVersion", "generatedAt", "accounts"}
    assert isinstance(doc["signalsVersion"], str) and doc["signalsVersion"]
    assert dt.datetime.fromisoformat(doc["generatedAt"].replace("Z", "+00:00"))
    assert isinstance(doc["accounts"], dict)
    for did, ent in doc["accounts"].items():
        assert did.startswith("did:") and did in POOL
        assert set(ent) <= {"multiplier", "samplePostUri"} and ent
        if "multiplier" in ent:
            m = ent["multiplier"]
            assert isinstance(m, (int, float)) and math.isfinite(m) and 0 <= m <= 4
        if "samplePostUri" in ent:
            assert ent["samplePostUri"].startswith(f"at://{did}/app.bsky.feed.post/")


def test_activity_dedupe_tombstones_and_dormant_gate():
    res, _, _, _, _ = run()
    d = res["detail"]
    i = {x: n for n, x in enumerate(d["dids"])}
    assert d["posts14"][i[A]] == 6 and d["originals14"][i[A]] == 5  # dup counted once, tombstoned post dropped
    assert d["lastPostUs"][i[A]] == T("2026-11-10T09:00:00")
    assert d["posts14"][i[CH]] == 1
    assert d["dormant"].tolist() == [x in (DO, DE, BD) for x in d["dids"]]
    assert d["raw14"][i[DE]] == 2  # rows exist, but the tombstone has the higher rev
    act = res["report"]["activity"]
    assert act["dormant"]["total"] == 3 and act["dormantWithRawRows"] == 2  # deleted + backdated
    san = {s["did"]: s for s in act["sanity"]["sample"]}
    assert set(san) == {DO, DE, BD}
    assert san[DO]["rawRowsInWindow"] == 0 and san[DO]["rawRowsEarlier"] == 1
    for did in (DO, DE, BD):
        assert res["signals"]["accounts"][did] == {"multiplier": 0.0}


def test_status_gate_uses_max_seq():
    res, _, _, _, _ = run()
    st = res["report"]["status"]
    assert st == {"inactive": 1, "byStatus": {"takendown": 1}}
    assert res["signals"]["accounts"][TK] == {"multiplier": 0.0}
    assert res["detail"]["inactive"].tolist() == [x == TK for x in res["detail"]["dids"]]


def test_sample_post_choice():
    res, uris, _, _, _ = run()
    acc = res["signals"]["accounts"]
    assert acc[A]["samplePostUri"] == f"at://{A}/app.bsky.feed.post/{uris['media']}"  # newer ones are reply/short/labelled/political
    assert acc[RE] == {"samplePostUri": f"at://{RE}/app.bsky.feed.post/{uris['react']}"}  # video, pool had none
    assert SA not in acc  # same as the pool's sample, too little follow evidence
    assert "samplePostUri" not in acc[CH]  # "ok" is too short
    assert res["report"]["samplePost"]["emitted"] == 2 and res["report"]["samplePost"]["sameAsPool"] == 1


def test_kept_follow_burst_tombstones_and_cap():
    res, _, _, _, _ = run()
    kf = res["report"]["keptFollow"]
    assert kf["popCreatesRaw"] == 53 and kf["popCreatesDeduped"] == 52
    assert kf["burstExcluded"] == 1  # b1 -> churn; b2 (9 in 30 s) kept
    assert kf["censored"] == 0
    assert kf["tombRevNotHigher"] == 1 and kf["removedAfterWindow"] == 2
    assert kf["uncapped"]["eligibleFollows"] == 51 and kf["uncapped"]["removed14d"] == 15
    assert kf["capDropped"] == 2  # h1's 4th and 5th removal that day
    assert kf["capped"]["eligibleFollows"] == 49 and kf["capped"]["removed14d"] == 13
    assert kf["concentration"]["maxPerViewer"] == 5 and kf["concentration"]["removingViewers"] == 11
    d = res["detail"]
    i = {x: n for n, x in enumerate(d["dids"])}
    assert d["n"][i[CH]] == 21 and d["o"][i[CH]] == 11
    assert d["n"][i[A]] == 27 and d["o"][i[A]] == 1
    assert d["n"][i[RE]] == 0  # h1's removal of react was capped away
    acc = res["signals"]["accounts"]
    assert 0.6 <= acc[CH]["multiplier"] < 0.9
    assert 1.1 < acc[A]["multiplier"] <= 1.25
    no_cap, _, _, _, _ = run(viewer_day_cap=0)
    assert "capped" not in no_cap["report"]["keptFollow"] and no_cap["detail"]["n"][i[RE]] == 1


def test_removal_window_censoring():
    res, _, _, _, _ = run(as_of="2026-11-06")
    kf = res["report"]["keptFollow"]
    assert kf["censored"] == 5  # created 11-04 + 3 days runs past the end of 11-06
    assert kf["uncapped"]["eligibleFollows"] == 46
    assert res["detail"]["dormant"].tolist() == [x not in (TK, DO) for x in res["detail"]["dids"]]


def test_dormant_share_stop_rule_writes_nothing():
    res, _, _, _, _ = run(max_dormant_share=0.25)
    assert res["code"] == 3 and res["signals"] is None and "stopped" in res["report"]
    tmp = pathlib.Path(tempfile.mkdtemp())
    root, pool, _ = scenario(tmp)
    out = tmp / "out.json"
    rc = S.main(["--export-root", str(root), "--pool", str(pool), "--as-of", "2026-11-10", "--out", str(out),
                 "--activity-days", "3", "--follow-from", "2026-11-01", "--follow-to", "2026-11-04",
                 "--removal-days", "3"])
    assert rc == 3 and not out.exists()


def test_missing_files_refused():
    res, _, _, _, _ = run(as_of="2026-11-12")
    assert res["code"] == 2 and res["report"]["missingFiles"]


def test_live_path_refused():
    tmp = pathlib.Path(tempfile.mkdtemp())
    root, pool, _ = scenario(tmp)
    out = tmp / S.LIVE_PATH_NAME
    assert S.main(["--export-root", str(root), "--pool", str(pool), "--as-of", "2026-11-10", "--out", str(out)]) == 2
    assert not out.exists()


def test_main_output_shape_and_determinism():
    tmp = pathlib.Path(tempfile.mkdtemp())
    root, pool, _ = scenario(tmp)
    args = ["--export-root", str(root), "--pool", str(pool), "--as-of", "2026-11-10", "--activity-days", "3",
            "--follow-from", "2026-11-01", "--follow-to", "2026-11-04", "--removal-days", "3", "--k", "1",
            "--min-eligible", "5", "--max-dormant-share", "1"]
    outs = []
    for n in (1, 2):
        out, rep = tmp / f"o{n}.json", tmp / f"r{n}.json"
        assert S.main(args + ["--out", str(out), "--report", str(rep)]) == 0
        r = json.loads(rep.read_bytes())
        r.pop("seconds")
        outs.append((out.read_bytes(), json.dumps(r, sort_keys=True).encode()))
    assert outs[0] == outs[1]
    doc = json.loads(outs[0][0])
    consumer_accepts(doc)
    assert doc["generatedAt"] == "2026-11-11T00:00:00Z"
    assert set(doc["accounts"]) == {A, CH, DO, DE, TK, RE, BD}
    blob = outs[0][0] + outs[0][1]
    for bad in (b"secret-handle", b"Secret Name", SECRET_TEXT.encode(), b"outsider"):
        assert bad not in blob


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"{len(tests)} passed")
