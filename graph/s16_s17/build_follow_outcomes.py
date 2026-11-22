"""Per-account "kept follow" prior for the onboarding candidate pool.

For every pool account, how often do the people who recently followed it stop following it? Accounts
whose new followers leave faster than usual get a multiplier below 1 on their onboarding draw weight;
accounts whose followers stay get a mild boost. Output feeds `LiveAccountSignals.accountMultiplier`.

What the source can and cannot measure (read before changing the metric)
-----------------------------------------------------------------------
Input is the S2a `test_follows.parquet` (graph/s2prep/s2a_test.sql): follow records whose rkey TID falls
in 2026-08-16 .. 2026-10-18, as they stood in the fold (Aug 4 seed + the October GCE capture + netcup
exports Oct 19-23). There was no live capture between the Aug 4 seed and the October capture, so a follow
created and deleted inside the window before the October snapshot never appears at all, and every
observed delete (`del_us > 0`) falls in the October live window (2026-10-18 .. 2026-10-23). "Removed within
14 days of creation" is therefore structurally zero for any follow old enough to be uncensored, and is not
computed here.

What it does measure: among an account's follows that were still alive when live capture began, the
share deleted during the live window (about 4.5 days). The deletion hazard depends on follow age (young
follows churn more), so each account's observed removals O are compared with the removals E expected
from the pool-wide hazard at each follow's age:

    h[a]     = removed / eligible over all pool-subject eligible follows of age a days (a = 0..63)
    E_i      = sum over account i's eligible follows of h[age]
    RR_i     = (O_i + k*g) / (E_i + k*g)          g = pool-wide removal rate; k = pseudo-follows of prior
    keepRate = 1 - g * RR_i                        age-standardised, shrunk window keep rate

Eligible follow: not a bulk follow (burst_b >= 10: ten or more follows by the same account within +-30 s,
S1 R8, eval_temporal.py BURST_MIN; starter packs and follow-all) unless --include-bursts. Gained counts
every follow in the window.

Multiplier: 1.0 below --min-eligible eligible follows; otherwise log-linear in RR, 1.25 at RR <= 1/RR_BOOST
(0.25), 0.6 at RR >= RR_FLOOR (4), 1.0 at RR = 1 (the global rate). Monotone non-increasing in RR.

Run: python build_follow_outcomes.py --follows <test_follows.parquet> --pool <pool.json> --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DAY_US = 86_400_000_000
AGE_BUCKETS = 64            # 0..63 days; the create window is 63 days long
BURST_MIN = 10              # eval_temporal.py:91, s2a_test.sql:6 (S1 R8)
VERSION = "follow_outcomes_v1"
MULT_MIN, MULT_MAX = 0.6, 1.25
RR_FLOOR = 4.0              # RR at which the multiplier reaches MULT_MIN
RR_BOOST = 4.0              # 1/RR at which the multiplier reaches MULT_MAX (symmetric in log RR with
                            # RR_FLOOR, so 0 removed of ~2.5 expected gives ~1.12, not the cap)
DEFAULT_K = 1500            # pseudo-follows of prior. Method of moments on the Oct 2026 data
                            # (var_theta = sum((O-E)^2 - O) / sum(E^2), accounts with E >= 1) gives
                            # alpha ~ 2.6 expected removals, i.e. k ~ 1544 at g ~ 0.0017.
DEFAULT_MIN_ELIGIBLE = 1000


def multiplier(rr: float, eligible: int, min_eligible: int) -> float:
    """Monotone map from shrunk relative removal risk to a draw-weight multiplier in [0.6, 1.25]."""
    if eligible < min_eligible or not (rr > 0) or math.isnan(rr):
        return 1.0
    if rr >= 1.0:
        frac = min(math.log(rr) / math.log(RR_FLOOR), 1.0)
        return 1.0 - (1.0 - MULT_MIN) * frac
    frac = min(math.log(1.0 / rr) / math.log(RR_BOOST), 1.0)
    return 1.0 + (MULT_MAX - 1.0) * frac


def load_pool(path: pathlib.Path) -> tuple[str, list[str]]:
    pool = json.loads(path.read_text())
    dids = sorted({m["did"] for c in pool["communities"] for m in c["members"]})
    return pool["poolVersion"], dids


def scan(follows: pathlib.Path, dids: list[str], include_bursts: bool = False,
         exclude_viewers: set[str] | None = None, split_us: int | None = None) -> dict:
    """One pass over the follows file. Returns per-account counts by age bucket.

    cnt[i, a] eligible follows of age a; rem[i, a] of those deleted; rem_late[i, a] deleted at or after
    split_us (for an observation-window split; zero when split_us is None). Ages are measured from
    obs_from, the first observed delete time, which is found in a cheap numeric pre-pass.
    """
    pf = pq.ParquetFile(str(follows))
    c_min, c_max, d_min, d_max = None, None, None, 0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["c_us", "del_us"])
        c = t["c_us"].to_numpy()
        d = t["del_us"].to_numpy()
        c_min = int(c.min()) if c_min is None else min(c_min, int(c.min()))
        c_max = int(c.max()) if c_max is None else max(c_max, int(c.max()))
        dv = d[d >= c_min]  # a delete rev without a usable time decodes to ~1970; ignore it for the window
        if len(dv):
            d_min = int(dv.min()) if d_min is None else min(d_min, int(dv.min()))
            d_max = max(d_max, int(dv.max()))
    obs_from = d_min if d_min is not None else c_max
    value_set = pa.array(dids, type=pa.string())
    viewers = pa.array(sorted(exclude_viewers), type=pa.string()) if exclude_viewers else None
    P = len(dids)
    gained = np.zeros(P, dtype=np.int64)
    cnt = np.zeros((P, AGE_BUCKETS), dtype=np.int64)
    rem = np.zeros((P, AGE_BUCKETS), dtype=np.int64)
    rem_late = np.zeros((P, AGE_BUCKETS), dtype=np.int64)
    cols = ["subject", "c_us", "burst_b", "del_us"] + (["did"] if viewers is not None else [])
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=cols)
        idx = pc.index_in(t["subject"], value_set=value_set)
        keep = pc.is_valid(idx)
        if not pc.any(keep).as_py():
            continue
        i = pc.filter(idx, keep).to_numpy().astype(np.int64)
        c = pc.filter(t["c_us"], keep).to_numpy().astype(np.int64)
        b = pc.filter(t["burst_b"], keep).to_numpy()
        d = pc.filter(t["del_us"], keep).to_numpy().astype(np.int64)
        gained += np.bincount(i, minlength=P)
        el = np.ones(len(i), dtype=bool) if include_bursts else (b < BURST_MIN)
        if viewers is not None:
            v = pc.filter(t["did"], keep)
            el &= ~pc.is_in(v, value_set=viewers).to_numpy(zero_copy_only=False)
        i, c, d = i[el], c[el], d[el]
        age = np.clip((obs_from - c) // DAY_US, 0, AGE_BUCKETS - 1)
        flat = i * AGE_BUCKETS + age
        cnt += np.bincount(flat, minlength=P * AGE_BUCKETS).reshape(P, AGE_BUCKETS)
        r = d > 0
        rem += np.bincount(flat[r], minlength=P * AGE_BUCKETS).reshape(P, AGE_BUCKETS)
        if split_us is not None:
            rl = r & (d >= split_us)
            rem_late += np.bincount(flat[rl], minlength=P * AGE_BUCKETS).reshape(P, AGE_BUCKETS)
    return {"gained": gained, "cnt": cnt, "rem": rem, "rem_late": rem_late, "c_min": c_min, "c_max": c_max,
            "obs_from": obs_from, "obs_to": d_max}


def score(cnt: np.ndarray, rem: np.ndarray, k: float, h: np.ndarray | None = None) -> dict:
    """Age-standardised shrunk relative risk and keep rate per account (rows of cnt/rem)."""
    tot_n, tot_r = cnt.sum(), rem.sum()
    g = tot_r / tot_n if tot_n else 0.0
    if h is None:
        col_n, col_r = cnt.sum(axis=0), rem.sum(axis=0)
        h = np.where(col_n > 0, col_r / np.maximum(col_n, 1), g)
    n = cnt.sum(axis=1)
    o = rem.sum(axis=1)
    e = cnt @ h
    prior = k * g
    rr = (o + prior) / np.maximum(e + prior, 1e-12)
    return {"g": float(g), "h": h, "n": n, "o": o, "e": e, "rr": rr, "keep": 1.0 - g * rr}


def build(follows: pathlib.Path, pool: pathlib.Path, k: float, min_eligible: int,
          include_bursts: bool = False) -> dict:
    pool_version, dids = load_pool(pool)
    s = scan(follows, dids, include_bursts=include_bursts)
    sc = score(s["cnt"], s["rem"], k)
    iso = lambda us: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(us / 1e6))  # noqa: E731
    accounts = {}
    for j, did in enumerate(dids):
        n = int(sc["n"][j])
        if n < 1:
            continue
        rr = float(sc["rr"][j])
        accounts[did] = {
            "gained": int(s["gained"][j]),
            "eligible": n,
            "removedObs": int(sc["o"][j]),
            "expectedRemoved": round(float(sc["e"][j]), 3),
            "keepRate": round(float(sc["keep"][j]), 6),
            "multiplier": round(multiplier(rr, n, min_eligible), 4),
        }
    return {
        "version": VERSION,
        "metric": "age-standardised share of window-alive follows deleted during the live observation window",
        "window": {"from": iso(s["c_min"]), "to": iso(s["c_max"])},
        "observation": {"from": iso(s["obs_from"]), "to": iso(s["obs_to"])},
        "k": k,
        "minEligible": min_eligible,
        "burstsExcluded": not include_bursts,
        "burstMin": BURST_MIN,
        "multiplierMap": {"min": MULT_MIN, "max": MULT_MAX, "rrFloor": RR_FLOOR, "rrBoost": RR_BOOST},
        "globalRemovalRate": round(sc["g"], 8),
        "globalKeepRate": round(1.0 - sc["g"], 8),
        "sourcePoolVersion": pool_version,
        "accounts": accounts,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--follows", required=True, type=pathlib.Path)
    ap.add_argument("--pool", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--k", type=float, default=DEFAULT_K)
    ap.add_argument("--min-eligible", type=int, default=DEFAULT_MIN_ELIGIBLE)
    ap.add_argument("--include-bursts", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    pa.set_cpu_count(a.threads)
    pa.set_io_thread_count(a.threads)
    t0 = time.time()
    out = build(a.follows, a.pool, a.k, a.min_eligible, a.include_bursts)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    tmp.write_text(json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(a.out)
    m = [v["multiplier"] for v in out["accounts"].values()]
    print(f"wrote {a.out}: {len(m)} accounts, {sum(x != 1.0 for x in m)} non-neutral, "
          f"g={out['globalRemovalRate']:.5f}, {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
