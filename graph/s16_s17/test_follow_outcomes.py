"""Tests for build_follow_outcomes.py on a tiny synthetic follows file and pool.

Covers: the observation window and age bucketing (the censoring rules), age standardisation, shrinkage,
burst exclusion, multiplier bounds and monotonicity, output shape and determinism.

Run: python test_follow_outcomes.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_follow_outcomes as B  # noqa: E402

DAY = B.DAY_US
HOUR = 3_600_000_000
OBS = 1_792_300_000_000_000          # first valid delete time in the synthetic file
A, BB, C, D, E = (f"did:plc:{x * 24}" for x in "abcde")


def _rows() -> list[tuple]:
    """(viewer, subject, c_us, burst_b, del_us). E is not in the pool."""
    rows: list[tuple] = []
    v = iter(f"did:plc:viewer{i:04d}" for i in range(1000))
    young, old = OBS - DAY - HOUR, OBS - 40 * DAY - HOUR
    for j in range(6):                                   # A: 6 young follows, 3 removed (first at OBS)
        rows.append((next(v), A, young + j, 1, OBS + j * HOUR if j < 3 else 0))
    rows.append((next(v), A, OBS + HOUR, 1, 0))          # A: created after observation start -> age 0
    for j in range(4):                                   # B: 4 young, 2 removed
        rows.append((next(v), BB, young + j, 2, OBS + 2 * HOUR if j < 2 else 0))
    for j in range(20):                                  # B: 20 old, 2 removed
        rows.append((next(v), BB, old + j, 3, OBS + 3 * HOUR if j < 2 else 0))
    for j in range(5):                                   # C: 5 ordinary follows at age 20, none removed
        rows.append((next(v), C, OBS - 20 * DAY - j, 1, 0))
    for j in range(5):                                   # C: 5 bulk follows (starter pack), all removed
        rows.append((next(v), C, OBS - 20 * DAY - j, 12, OBS + 5 * HOUR))
    for j in range(3):                                   # D: bulk follows only
        rows.append((next(v), D, OBS - 10 * DAY - j, 40, 0))
    rows.append((next(v), E, old, 1, 7))                 # delete rev with no usable time (decodes to ~1970)
    rows.append((next(v), E, old, 1, OBS + 4 * DAY))     # last delete in the file
    return rows


def _write(tmp: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    rows = _rows()
    cols = list(zip(*rows))
    t = pa.table({
        "did": pa.array(cols[0], pa.string()),
        "subject": pa.array(cols[1], pa.string()),
        "c_us": pa.array(cols[2], pa.int64()),
        "burst_b": pa.array(cols[3], pa.uint16()),
        "del_us": pa.array(cols[4], pa.int64()),
    })
    follows = tmp / "follows.parquet"
    pq.write_table(t, follows, row_group_size=17)        # several row groups on purpose
    pool = tmp / "pool.json"
    members = [{"did": d, "followers": 1, "role": "anchor"} for d in (A, BB, C, D)]
    pool.write_text(json.dumps({"poolVersion": "pool_test", "communities": [
        {"members": members[:3]}, {"members": members[1:]}]}))  # B, C appear twice
    return follows, pool


def test_multiplier_bounds_and_monotone() -> None:
    assert B.multiplier(1.0, 10_000, 1000) == 1.0
    assert B.multiplier(10.0, 999, 1000) == 1.0          # below the evidence threshold: neutral
    assert B.multiplier(0.1, 999, 1000) == 1.0
    assert B.multiplier(float("nan"), 10_000, 1000) == 1.0
    assert B.multiplier(0.0, 10_000, 1000) == 1.0        # no estimate: neutral
    assert B.multiplier(B.RR_FLOOR, 10_000, 1000) == B.MULT_MIN
    assert B.multiplier(1e9, 10_000, 1000) == B.MULT_MIN
    assert math.isclose(B.multiplier(1 / B.RR_BOOST, 10_000, 1000), B.MULT_MAX)
    assert B.multiplier(1e-9, 10_000, 1000) == B.MULT_MAX
    grid = np.exp(np.linspace(math.log(1e-4), math.log(1e4), 2001))
    m = [B.multiplier(float(r), 10_000, 1000) for r in grid]
    assert all(B.MULT_MIN <= x <= B.MULT_MAX for x in m)
    assert all(a >= b for a, b in zip(m, m[1:])), "multiplier must not increase with relative risk"


def test_shrinkage_pulls_thin_evidence_to_neutral() -> None:
    cnt = np.zeros((3, B.AGE_BUCKETS), dtype=np.int64)
    rem = np.zeros_like(cnt)
    cnt[:, 5] = [100, 10_000, 100_000]
    rem[:, 5] = [5, 500, 500]
    raw = B.score(cnt, rem, 0)
    assert math.isclose(raw["rr"][0], raw["rr"][1]) and raw["rr"][0] > 5
    s = B.score(cnt, rem, 1500)
    assert 1 < s["rr"][0] < s["rr"][1] < raw["rr"][1]    # same raw rate, less evidence -> nearer 1
    prev = raw["rr"][0]
    for k in (20, 200, 2000, 20_000, 2_000_000):
        r = B.score(cnt, rem, k)["rr"][0]
        assert 1 <= r < prev
        prev = r
    assert abs(B.score(cnt, rem, 1e9)["keep"][0] - (1 - raw["g"])) < 1e-6  # infinite prior -> global keep rate
    assert math.isclose(s["keep"][1], 1 - s["g"] * s["rr"][1])


def test_age_standardisation() -> None:
    # Young follows churn 5x faster; an account that is only young must not look worse for it.
    cnt = np.zeros((2, B.AGE_BUCKETS), dtype=np.int64)
    rem = np.zeros_like(cnt)
    cnt[0, 1], rem[0, 1] = 1000, 50                      # young only, at the young hazard (0.05)
    cnt[1, 40], rem[1, 40] = 9000, 90                    # old only, at the old hazard (0.01)
    s = B.score(cnt, rem, 0)
    assert np.allclose(s["rr"], [1.0, 1.0])
    assert np.allclose(s["e"], [50, 90])


def test_scan_window_and_age_buckets() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        follows, pool = _write(tmp)
        _, dids = B.load_pool(pool)
        assert dids == sorted({A, BB, C, D})
        s = B.scan(follows, dids)
        assert s["obs_from"] == OBS, "the ~1970 delete time must not open the observation window"
        assert s["obs_to"] == OBS + 4 * DAY
        ia, ib = dids.index(A), dids.index(BB)
        assert s["cnt"][ia, 0] == 1 and s["cnt"][ia, 1] == 6   # created after OBS -> age 0
        assert s["cnt"][ib, 40] == 20 and s["rem"][ib, 40] == 2
        # A follow older than the last bucket is clipped into it rather than dropped.
        t = pq.read_table(follows)
        old = t.slice(0, 1).set_column(2, "c_us", pa.array([OBS - 200 * DAY], pa.int64()))
        pq.write_table(pa.concat_tables([t, old]), tmp / "f2.parquet")
        s2 = B.scan(tmp / "f2.parquet", dids)
        assert s2["cnt"][ia, B.AGE_BUCKETS - 1] == 1


def test_burst_exclusion() -> None:
    with tempfile.TemporaryDirectory() as td:
        follows, pool = _write(pathlib.Path(td))
        _, dids = B.load_pool(pool)
        ic, id_ = dids.index(C), dids.index(D)
        s = B.scan(follows, dids)
        assert s["gained"][ic] == 10 and s["cnt"][ic].sum() == 5 and s["rem"][ic].sum() == 0
        assert s["gained"][id_] == 3 and s["cnt"][id_].sum() == 0
        s_all = B.scan(follows, dids, include_bursts=True)
        assert s_all["cnt"][ic].sum() == 10 and s_all["rem"][ic].sum() == 5
        s_ex = B.scan(follows, dids, exclude_viewers={_rows()[0][0]})
        assert s_ex["cnt"][dids.index(A)].sum() == 6


def test_build_output_shape_and_determinism() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        follows, pool = _write(tmp)
        args = ["--follows", str(follows), "--pool", str(pool), "--k", "20", "--min-eligible", "1"]
        assert B.main(args + ["--out", str(tmp / "a.json")]) == 0
        assert B.main(args + ["--out", str(tmp / "b.json")]) == 0
        a_bytes = (tmp / "a.json").read_bytes()
        assert a_bytes == (tmp / "b.json").read_bytes()
        out = json.loads(a_bytes)
        for key in ("version", "window", "k", "minEligible", "globalKeepRate", "sourcePoolVersion", "accounts"):
            assert key in out
        assert out["sourcePoolVersion"] == "pool_test" and out["k"] == 20 and out["minEligible"] == 1
        assert out["observation"]["from"] < out["observation"]["to"]
        assert set(out["accounts"]) == {A, BB, C}, "D has no eligible follows; E is not in the pool"
        fields = {"gained", "eligible", "removedObs", "expectedRemoved", "keepRate", "multiplier"}
        assert all(set(v) == fields for v in out["accounts"].values())
        a, b, c = out["accounts"][A], out["accounts"][BB], out["accounts"][C]
        assert (a["gained"], a["eligible"], a["removedObs"]) == (7, 7, 3)
        assert (b["gained"], b["eligible"], b["removedObs"]) == (24, 24, 4)
        assert (c["gained"], c["eligible"], c["removedObs"]) == (10, 5, 0)
        g = 7 / 36
        assert math.isclose(out["globalKeepRate"], 1 - g, abs_tol=1e-8)
        # A (raw 3/7 removed) and B (raw 4/24) both sit exactly at their age-expected removals.
        assert a["expectedRemoved"] == 3.0 and b["expectedRemoved"] == 4.0
        assert a["multiplier"] == b["multiplier"] == 1.0
        assert math.isclose(a["keepRate"], 1 - g, abs_tol=1e-6)
        for v in out["accounts"].values():
            assert B.MULT_MIN <= v["multiplier"] <= B.MULT_MAX
        assert "handle" not in a_bytes.decode() and "displayName" not in a_bytes.decode()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"{len(tests)} passed")
