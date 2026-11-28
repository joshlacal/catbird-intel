#!/usr/bin/env python3
"""Build the onboarding account-signals file (src/onboarding_signals.ts) from the nightly firehose export.

Three signals per pool DID, reference time = end of the --as-of UTC day:
  1. activity gate   latest account_status (max seq) not active, or no live post in the last
                     --activity-days days  -> multiplier 0 (the account stays out of new lists)
  2. sample post     most recent live, non-reply, unlabelled post in the window with >= 20 chars of
                     text or media; emitted only when it differs from the pool's samplePostUri
  3. kept-follow     follows of pool DIDs created --follow-from..--follow-to (TID time), burst
                     follows excluded, removed14d = a tombstone with a higher rev indexed within
                     --removal-days of creation. Shrunk, create-day-standardised relative risk and
                     the [0.6, 1.25] multiplier come from build_follow_outcomes.py.
  4. link-feed gate  own posts in the activity window are mostly links to one site and almost nobody
                     replies to or quotes them (link_feed_screen.py) -> multiplier 0

Export rows are raw (tombstones included, not deduplicated): every record is reduced to its max-rev row.
Output holds only DIDs, numbers and at:// URIs. Diagnostics go to stderr and --report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_follow_outcomes as FO  # noqa: E402
import link_feed_screen as LF  # noqa: E402

VERSION = "onboarding_signals_v1"
LIVE_PATH_NAME = "intel_onboarding_signals_v1.candidate.json"  # watched by a running local service
DAY_US = 86_400_000_000
B32 = "234567abcdefghijklmnopqrstuvwxyz"
MIN_TID_US = 1_672_531_200_000_000  # 2023-01-01
BURST_MIN = FO.BURST_MIN            # >= 10 follows by one viewer within +-30 s (s2a_test.sql:6,23)
BURST_HALF_US = 30_000_000
SAFE_SCORE_MAX = 0.45               # build_intel_serving_v2.py: politicalness/heat > 0.45 never sampled
MIN_SAMPLE_TEXT = 20
EVIDENCE_EXPECTED = 10.0            # default min_eligible = ceil(10 / g): ~10 expected removals
STRONG_EXPECTED = 20.0              # "strong evidence" for the worst-accounts list
TIERS = ((0, 1500, "<1500"), (1500, 20000, "1500-20k"), (20000, math.inf, ">=20k"))
KEY_VID_BITS, KEY_REL_BITS = 23, 41

_LUT = np.full(256, 255, dtype=np.uint8)
for _i, _c in enumerate(B32):
    _LUT[ord(_c)] = _i
_SHIFTS = np.array([5 * (12 - i) for i in range(13)], dtype=np.uint64)


# ---------------------------------------------------------------- helpers

def tid_us(rkey, arrival_us: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised TID decode (src/tid.ts): microseconds and a validity mask."""
    a = rkey.combine_chunks() if isinstance(rkey, pa.ChunkedArray) else rkey
    a = pc.fill_null(a.cast(pa.string()), "")
    n = len(a)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, bool)
    ok = pc.equal(pc.binary_length(a), 13).to_numpy(zero_copy_only=False)
    fixed = pc.if_else(pa.array(ok), a, pa.scalar("2" * 13))
    bufs = fixed.buffers()
    offs = np.frombuffer(bufs[1], dtype=np.int32)[fixed.offset:fixed.offset + n + 1]
    if offs[-1] - offs[0] != 13 * n:
        raise RuntimeError("non-contiguous rkey buffer")
    mat = np.frombuffer(bufs[2], dtype=np.uint8)[offs[0]:offs[0] + 13 * n].reshape(n, 13)
    v = _LUT[mat]
    ok &= ~(v == 255).any(axis=1) & (v[:, 0] < 16)
    us = ((v.astype(np.uint64) << _SHIFTS).sum(axis=1) >> np.uint64(10)).astype(np.int64)
    ok &= us >= MIN_TID_US
    if arrival_us is not None:
        ok &= us <= arrival_us + DAY_US
    return us, ok


_UNIT_US = {"s": 1_000_000, "ms": 1000, "us": 1, "ns": None}


def ts_us(col) -> np.ndarray:
    unit = col.type.unit
    v = pc.fill_null(pc.cast(col, pa.int64()), 0).to_numpy(zero_copy_only=False)
    return v // 1000 if unit == "ns" else v * _UNIT_US[unit]


def map_ids(arr, table: dict, add: bool) -> np.ndarray:
    a = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    enc = pc.dictionary_encode(a)
    dic = enc.dictionary.to_pylist()
    get = table.setdefault if add else table.get
    lut = np.fromiter((get(d, len(table)) if add else get(d, -1) for d in dic), dtype=np.int64, count=len(dic))
    return lut[enc.indices.to_numpy(zero_copy_only=False)]


def first_of_runs(*cols: np.ndarray) -> np.ndarray:
    n = len(cols[0])
    m = np.zeros(n, bool)
    if n:
        m[0] = True
        for c in cols:
            m[1:] |= c[1:] != c[:-1]
    return m


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), as Spearman needs."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    start = np.r_[0, np.flatnonzero(xs[1:] != xs[:-1]) + 1]
    end = np.r_[start[1:], len(x)]
    r = np.empty(len(x))
    r[order] = np.repeat((start + end - 1) / 2.0, end - start)
    return r


def _finite_or_none(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        return _finite_or_none(np.corrcoef(rank(a), rank(b))[0, 1])


def partial_spearman(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> float | None:
    if len(a) < 4:
        return None
    ra, rb, rz = rank(a), rank(b), rank(z)
    x = np.column_stack([np.ones(len(rz)), rz])
    res_a = ra - x @ np.linalg.lstsq(x, ra, rcond=None)[0]
    res_b = rb - x @ np.linalg.lstsq(x, rb, rcond=None)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return _finite_or_none(np.corrcoef(res_a, res_b)[0, 1])


def days(a: dt.date, b: dt.date) -> list[dt.date]:
    return [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]


def day_dirs(root: pathlib.Path, last: dt.date) -> list[pathlib.Path]:
    out = []
    for p in sorted(root.iterdir()):
        try:
            d = dt.date.fromisoformat(p.name)
        except ValueError:
            continue
        if p.is_dir() and d <= last:
            out.append(p)
    return out


def load_pool(path: pathlib.Path) -> tuple[str, list[str], dict]:
    doc = json.loads(path.read_text())
    meta: dict[str, dict] = {}
    for c in doc["communities"]:
        for m in c["members"]:
            meta.setdefault(m["did"], m)
    return str(doc.get("poolVersion", "")), sorted(meta), meta


def tier(f: int) -> str:
    for lo, hi, name in TIERS:
        if lo <= f < hi:
            return name
    return TIERS[-1][2]


# ---------------------------------------------------------------- scans

def scan_status(root: pathlib.Path, last: dt.date, pool_arr: pa.Array, npool: int):
    parts = []
    for d in day_dirs(root, last):
        f = d / "account_status.parquet"
        if not f.exists():
            continue
        t = pq.read_table(f, columns=["did", "seq", "active", "status"])
        pidx = pc.index_in(t["did"], value_set=pool_arr)
        m = pc.is_valid(pidx)
        if not pc.any(m).as_py():
            continue
        parts.append(pa.table({"pidx": pidx.filter(m), "seq": t["seq"].filter(m),
                               "active": t["active"].filter(m), "status": t["status"].filter(m)}))
    inactive = np.zeros(npool, bool)
    status = {}
    if parts:
        t = pa.concat_tables(parts).sort_by([("pidx", "ascending"), ("seq", "descending")])
        p = t["pidx"].to_numpy()
        keep = first_of_runs(p)
        t = t.filter(pa.array(keep))
        p = t["pidx"].to_numpy()
        act = t["active"].to_numpy()
        st = t["status"].to_pylist()
        inactive[p[act == 0]] = True
        for i, a, s in zip(p, act, st):
            if a == 0:
                status[int(i)] = s or "inactive"
    return inactive, status


POST_COLS = ["rkey", "rev", "is_deleted", "created_at", "reply_parent", "text", "image_count", "has_video",
             "self_labels", "politicalness", "heat", "external_uri", "facet_links"]
_TARGET_DID = r"^at://(?P<did>did:[^/]+)/"


def _targets(col, pool_arr: pa.Array) -> pa.Array:
    """Pool index of the account an at:// URI column points at (null when outside the pool)."""
    did = pc.struct_field(pc.extract_regex(pc.fill_null(col, ""), _TARGET_DID), [0])
    return pc.index_in(did, value_set=pool_arr)


def scan_posts(files: list[pathlib.Path], pool_arr: pa.Array, npool: int):
    """Pool accounts' own posts, plus how many distinct other accounts replied to or quoted each one."""
    raw = np.zeros(npool, np.int64)
    parts = []
    responder_ids: dict[str, int] = {}
    pairs = []
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            head = pf.read_row_group(rg, columns=["did", "reply_parent", "quote_uri"])
            for col in ("reply_parent", "quote_uri"):
                tgt = _targets(head[col], pool_arr)
                hit = pc.fill_null(pc.not_equal(pc.take(pool_arr, tgt), head["did"]), False)
                if pc.any(hit).as_py():
                    src = map_ids(head["did"].filter(hit), responder_ids, add=True)
                    tg = tgt.filter(hit).to_numpy(zero_copy_only=False).astype(np.int64)
                    pairs.append((tg << 32) | src)
            pidx = pc.index_in(head["did"], value_set=pool_arr)
            m = pc.is_valid(pidx)
            sel = pidx.filter(m).to_numpy()
            if len(sel) == 0:
                continue
            raw += np.bincount(sel, minlength=npool)
            t = pf.read_row_group(rg, columns=POST_COLS).filter(m)
            unsafe = pc.or_(
                pc.or_(pc.fill_null(pc.greater(t["politicalness"], SAFE_SCORE_MAX), False),
                       pc.fill_null(pc.greater(t["heat"], SAFE_SCORE_MAX), False)),
                pc.greater(pc.fill_null(pc.list_value_length(t["self_labels"]), 0), 0))
            media = pc.or_(pc.greater(pc.fill_null(t["image_count"], 0), 0),
                           pc.equal(pc.fill_null(t["has_video"], 0), 1))
            hosts = [LF.link_host(e or (fl[0] if fl else ""))
                     for e, fl in zip(t["external_uri"].to_pylist(), t["facet_links"].to_pylist())]
            parts.append(pa.table({
                "pidx": pa.array(sel.astype(np.int32)),
                "rkey": t["rkey"],
                "rev": t["rev"],
                "del": t["is_deleted"],
                "created_us": pa.array(ts_us(t["created_at"])),
                "orig": pc.equal(pc.fill_null(t["reply_parent"], ""), ""),
                "tlen": pc.utf8_length(pc.fill_null(t["text"], "")),
                "media": media,
                "safe": pc.invert(unsafe),
                "host": pa.array(hosts, pa.string()),
            }))
    keys = np.unique(np.concatenate(pairs)) if pairs else np.zeros(0, np.int64)
    responders = np.bincount((keys >> 32).astype(np.int64), minlength=npool)
    return raw, parts, responders


def count_rows(files: list[pathlib.Path], pool_arr: pa.Array, npool: int) -> np.ndarray:
    raw = np.zeros(npool, np.int64)
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            pidx = pc.index_in(pf.read_row_group(rg, columns=["did"])["did"], value_set=pool_arr)
            sel = pidx.filter(pc.is_valid(pidx)).to_numpy()
            if len(sel):
                raw += np.bincount(sel, minlength=npool)
    return raw


def scan_follows(files: list[pathlib.Path], pool_arr: pa.Array, w0: int, w1: int, origin: int):
    """Creates (all subjects, for burst counting) and pool-subject population, plus tombstones, by TID time."""
    vids: dict[str, int] = {}
    burst, pk, pp, prev, pus = [], [], [], [], []
    tdid, trel, trev, tidx = [], [], [], []
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["did", "rkey", "rev", "is_deleted", "indexed_at", "subject"])
            idx_us = ts_us(t["indexed_at"])
            us, ok = tid_us(t["rkey"], idx_us)
            isdel = t["is_deleted"].to_numpy() == 1
            rev = t["rev"].to_numpy()
            cre = ok & ~isdel & (us >= w0 - BURST_HALF_US) & (us < w1 + BURST_HALF_US)
            if cre.any():
                cm = pa.array(cre)
                vid = map_ids(t["did"].filter(cm), vids, add=True)
                key = (vid.astype(np.uint64) << np.uint64(KEY_REL_BITS)) | (us[cre] - origin).astype(np.uint64)
                burst.append(key)
                sidx = pc.fill_null(pc.index_in(t["subject"].filter(cm), value_set=pool_arr), -1).to_numpy()
                pm = (sidx >= 0) & (us[cre] >= w0) & (us[cre] < w1)
                if pm.any():
                    pk.append(key[pm])
                    pp.append(sidx[pm].astype(np.int32))
                    prev.append(rev[cre][pm])
                    pus.append(us[cre][pm])
            tm = ok & isdel & (us >= w0) & (us < w1)
            if tm.any():
                tdid.append(t["did"].filter(pa.array(tm)))
                trel.append(us[tm] - origin)
                trev.append(rev[tm])
                tidx.append(idx_us[tm])
    if len(vids) >= 1 << KEY_VID_BITS:
        raise RuntimeError(f"{len(vids)} viewers exceed the {KEY_VID_BITS}-bit key space")
    cat = (lambda xs, dtp: np.concatenate(xs) if xs else np.zeros(0, dtp))
    tomb_vid = map_ids(pa.chunked_array([c for x in tdid for c in x.chunks], pa.string()), vids, add=False) \
        if tdid else np.zeros(0, np.int64)
    trel_a = cat(trel, np.int64)
    good = tomb_vid >= 0
    tkey = (tomb_vid[good].astype(np.uint64) << np.uint64(KEY_REL_BITS)) | trel_a[good].astype(np.uint64)
    return {
        "viewers": len(vids),
        "burst_keys": np.unique(cat(burst, np.uint64)),
        "pop_key": cat(pk, np.uint64), "pop_pidx": cat(pp, np.int32),
        "pop_rev": cat(prev, np.uint64), "pop_us": cat(pus, np.int64),
        "tomb_key": tkey, "tomb_rev": cat(trev, np.uint64)[good], "tomb_idx": cat(tidx, np.int64)[good],
        "tomb_unmatched_viewer": int((~good).sum()),
    }


# ---------------------------------------------------------------- kept-follow scoring

def estimate_k(cnt: np.ndarray, rem: np.ndarray) -> tuple[float, float | None]:
    s = FO.score(cnt, rem, 1.0)
    g, e, o = s["g"], s["e"], s["o"]
    m = e >= 1
    if g <= 0 or m.sum() < 10:
        return float(FO.DEFAULT_K), None
    var = float((((o[m] - e[m]) ** 2) - o[m]).sum() / (e[m] ** 2).sum())
    if var <= 0:
        return float(FO.DEFAULT_K), var
    return (1.0 / var) / g, var


def matrix(pidx, bucket, removed, mask, npool, nb):
    flat = pidx[mask].astype(np.int64) * nb + bucket[mask]
    cnt = np.bincount(flat, minlength=npool * nb).reshape(npool, nb).astype(np.float64)
    rem = np.bincount(flat[removed[mask]], minlength=npool * nb).reshape(npool, nb).astype(np.float64)
    return cnt, rem


def kept_follow(fo: dict, npool: int, nb: int, w0: int, ref: int, removal_days: int, k_arg, min_elig_arg,
                cap: int, meta_rows: list[dict]) -> dict:
    key, pidx, prev, pus = fo["pop_key"], fo["pop_pidx"], fo["pop_rev"], fo["pop_us"]
    raw_creates = len(key)
    o = np.argsort(key, kind="stable")
    key, pidx, prev, pus = key[o], pidx[o], prev[o], pus[o]
    u = first_of_runs(key)
    key, pidx, prev, pus = key[u], pidx[u], prev[u], pus[u]
    bk = fo["burst_keys"]
    half = np.uint64(BURST_HALF_US)
    near = np.searchsorted(bk, key + half, "right") - np.searchsorted(bk, key - half, "left")
    burst = near >= BURST_MIN
    # earliest tombstone per record key
    to = np.lexsort((fo["tomb_idx"], fo["tomb_key"]))
    tk, tr, ti = fo["tomb_key"][to], fo["tomb_rev"][to], fo["tomb_idx"][to]
    tf = first_of_runs(tk)
    tk, tr, ti = tk[tf], tr[tf], ti[tf]
    pos = np.minimum(np.searchsorted(tk, key), max(len(tk) - 1, 0))
    found = (len(tk) > 0) & (tk[pos] == key) if len(tk) else np.zeros(len(key), bool)
    rev_ok = found & (tr[pos] > prev) if len(tk) else found
    rem_us = np.where(rev_ok, ti[pos] if len(tk) else 0, 0)
    win = removal_days * DAY_US
    removed = rev_ok & (rem_us - pus <= win)
    censored = pus + win > ref
    elig = ~burst & ~censored
    bucket = ((pus - w0) // DAY_US).astype(np.int64)
    vid = (key >> np.uint64(KEY_REL_BITS)).astype(np.int64)
    # heavy unfollowers (eligible removals, before any cap)
    er = elig & removed
    per_viewer = np.bincount(vid[er]) if er.any() else np.zeros(0, np.int64)
    per_viewer = np.sort(per_viewer[per_viewer > 0])[::-1]
    tot_rem = int(per_viewer.sum())
    conc = {"removingViewers": int(len(per_viewer)), "removals": tot_rem,
            "top100Share": round(float(per_viewer[:100].sum() / tot_rem), 4) if tot_rem else None,
            "top500Share": round(float(per_viewer[:500].sum() / tot_rem), 4) if tot_rem else None,
            "top1000Share": round(float(per_viewer[:1000].sum() / tot_rem), 4) if tot_rem else None,
            "viewersWith10plus": int((per_viewer >= 10).sum()), "viewersWith50plus": int((per_viewer >= 50).sum()),
            "maxPerViewer": int(per_viewer[0]) if len(per_viewer) else 0}
    # per (viewer, removal day) cap: removals beyond the cap leave numerator and denominator
    drop = np.zeros(len(key), bool)
    if cap > 0 and er.any():
        idx = np.nonzero(er)[0]
        rday = rem_us[idx] // DAY_US
        oo = np.lexsort((rem_us[idx], rday, vid[idx]))
        idx, rday = idx[oo], rday[oo]
        g0 = first_of_runs(vid[idx], rday)
        start = np.maximum.accumulate(np.where(g0, np.arange(len(idx)), 0))
        drop[idx[(np.arange(len(idx)) - start) >= cap]] = True

    def evaluate(mask, k=None, min_elig=None):
        cnt, rem = matrix(pidx, bucket, removed, mask, npool, nb)
        kk, var = (k, None) if k is not None else estimate_k(cnt, rem)
        s = FO.score(cnt, rem, kk)
        me = min_elig if min_elig is not None else max(1, math.ceil(EVIDENCE_EXPECTED / s["g"])) if s["g"] > 0 else 10 ** 9
        mult = np.array([FO.multiplier(float(s["rr"][i]), int(s["n"][i]), me) for i in range(npool)])
        return {"s": s, "k": kk, "var": var, "min_elig": me, "mult": mult}

    def split_half(mask, k, me):
        hb = nb // 2
        a = evaluate(mask & (bucket < hb), k, me)["s"]
        b = evaluate(mask & (bucket >= hb), k, me)["s"]
        thr = max(1, math.ceil(me / 2))
        both = (a["n"] >= thr) & (b["n"] >= thr)
        with np.errstate(divide="ignore"):  # rr == 0 only when a half has g == 0; ranks still order it
            la, lb = np.log(a["rr"][both]), np.log(b["rr"][both])
        raw_a = (a["o"][both] + 0.5) / (a["e"][both] + 0.5)
        raw_b = (b["o"][both] + 0.5) / (b["e"][both] + 0.5)
        ntot = a["n"][both] + b["n"][both]
        return {"halfThreshold": thr, "accountsBothHalves": int(both.sum()),
                "gHalves": [round(a["g"], 5), round(b["g"], 5)],
                "spearmanShrunkRR": spearman(la, lb), "partialSpearmanShrunkRR_sizeNeutral": partial_spearman(la, lb, ntot),
                "spearmanRawOE": spearman(raw_a, raw_b), "partialSpearmanRawOE_sizeNeutral": partial_spearman(raw_a, raw_b, ntot),
                "spearmanRRvsSize": spearman(la, ntot)}

    def summary(ev, mask):
        s, me = ev["s"], ev["min_elig"]
        enough = s["n"] >= me
        strong = enough & (s["e"] >= STRONG_EXPECTED)
        worst = sorted(np.nonzero(strong)[0], key=lambda i: (-s["rr"][i], i))[:10]
        m = ev["mult"]
        return {"eligibleFollows": int(mask.sum()), "removed14d": int((mask & removed).sum()), "g": round(s["g"], 6),
                "k": round(ev["k"], 1), "momVarTheta": ev["var"], "minEligible": me,
                "accountsWithAnyFollow": int((s["n"] > 0).sum()), "accountsEnoughEvidence": int(enough.sum()),
                "accountsStrongEvidence": int(strong.sum()),
                "multiplierPct": {p: round(float(np.percentile(m[enough], p)), 4) for p in (1, 10, 50, 90, 99)} if enough.any() else None,
                "multBelow0p8": int((m < 0.8).sum()), "multAt0p6": int((m <= 0.6 + 1e-9).sum()), "multAbove1p1": int((m > 1.1).sum()),
                "splitHalf": split_half(mask, ev["k"], me),
                "worst10": [{"did": meta_rows[i]["did"], "followers": int(meta_rows[i].get("followers") or 0),
                             "role": meta_rows[i].get("role"), "n": int(s["n"][i]), "removed": int(s["o"][i]),
                             "expected": round(float(s["e"][i]), 1), "rate": round(float(s["o"][i] / s["n"][i]), 4),
                             "rr": round(float(s["rr"][i]), 3), "multiplier": round(float(m[i]), 4)} for i in worst]}

    uncapped = evaluate(elig, k_arg, min_elig_arg)
    final_mask = elig & ~drop
    final = evaluate(final_mask, k_arg, min_elig_arg) if cap > 0 else uncapped
    report = {"popCreatesRaw": raw_creates, "popCreatesDeduped": int(len(key)), "burstExcluded": int(burst.sum()),
              "censored": int((censored & ~burst).sum()), "tombRevNotHigher": int((found & ~rev_ok).sum()),
              "removedAfterWindow": int((rev_ok & ~removed & elig).sum()), "viewers": fo["viewers"],
              "tombUnmatchedViewer": fo["tomb_unmatched_viewer"], "viewerDayCap": cap, "capDropped": int((drop & elig).sum()),
              "concentration": conc, "uncapped": summary(uncapped, elig)}
    if cap > 0:
        report["capped"] = summary(final, final_mask)
        d = np.abs(final["mult"] - uncapped["mult"])
        report["capEffect"] = {"multChangedOver0p05": int((d > 0.05).sum()), "multChangedOver0p1": int((d > 0.1).sum()),
                               "maxAbsChange": round(float(d.max()), 4) if len(d) else 0.0}
    return {"mult": final["mult"], "n": final["s"]["n"], "o": final["s"]["o"], "report": report}


# ---------------------------------------------------------------- build

def build(export_root: pathlib.Path, pool_path: pathlib.Path, as_of: str, *, activity_days: int = 14,
          follow_from: str = "2026-10-19", follow_to: str = "2026-11-07", removal_days: int = 14,
          k: float | None = None, min_eligible: int | None = None, viewer_day_cap: int = 3,
          max_dormant_share: float = 0.25, max_link_feed_share: float = 0.05, sanity_n: int = 15,
          allow_missing: bool = False) -> dict:
    t0 = time.time()
    last = dt.date.fromisoformat(as_of)
    ref = int(dt.datetime(last.year, last.month, last.day, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000 + DAY_US
    ws = ref - activity_days * DAY_US
    f0, f1 = dt.date.fromisoformat(follow_from), dt.date.fromisoformat(follow_to)
    w0 = int(dt.datetime(f0.year, f0.month, f0.day, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000
    w1 = int(dt.datetime(f1.year, f1.month, f1.day, tzinfo=dt.timezone.utc).timestamp()) * 1_000_000 + DAY_US
    nb = (f1 - f0).days + 1
    pool_version, dids, meta = load_pool(pool_path)
    rows = [meta[d] for d in dids]
    npool = len(dids)
    pool_arr = pa.array(dids, pa.string())
    rep: dict = {"version": VERSION, "asOf": as_of, "referenceTime": iso(ref), "poolVersion": pool_version,
                 "poolDids": npool}

    act_days = days(last - dt.timedelta(days=activity_days - 1), last)
    post_files = [export_root / d.isoformat() / "posts.parquet" for d in act_days]
    fol_files = [export_root / d.isoformat() / "follows.parquet" for d in days(f0, last)]
    missing = [str(p) for p in post_files + fol_files if not p.exists()]
    rep["missingFiles"] = missing
    if missing and not allow_missing:
        return {"code": 2, "report": rep, "signals": None, "detail": None, "error": f"missing {len(missing)} files"}
    post_files = [p for p in post_files if p.exists()]
    fol_files = [p for p in fol_files if p.exists()]

    # (a) status gate
    inactive, status = scan_status(export_root, last, pool_arr, npool)
    st_counts: dict[str, int] = {}
    for s in status.values():
        st_counts[s] = st_counts.get(s, 0) + 1
    rep["status"] = {"inactive": int(inactive.sum()), "byStatus": dict(sorted(st_counts.items()))}
    t_status = time.time()

    # (b) activity + (c) sample post
    raw14, parts, responders14 = scan_posts(post_files, pool_arr, npool)
    posts14 = np.zeros(npool, np.int64)
    orig14 = np.zeros(npool, np.int64)
    linked14 = np.zeros(npool, np.int64)
    top_domain14 = np.zeros(npool, np.int64)
    top_host: list[str] = [""] * npool
    last_us = np.full(npool, -1, np.int64)
    sample: list[str | None] = [None] * npool
    if parts:
        t = pa.concat_tables(parts).sort_by([("pidx", "ascending"), ("rkey", "ascending"), ("rev", "descending")])
        p = t["pidx"].to_numpy()
        rk = np.asarray(t["rkey"].to_pylist(), dtype=object)
        keep = first_of_runs(p, rk)
        t = t.filter(pa.array(keep))
        p, rk = p[keep], rk[keep]
        cu = t["created_us"].to_numpy()
        live = (t["del"].to_numpy() == 0) & (cu >= ws - DAY_US) & (cu < ref + DAY_US)
        orig = t["orig"].to_numpy(zero_copy_only=False)
        posts14 = np.bincount(p[live], minlength=npool)
        orig14 = np.bincount(p[live & orig], minlength=npool)
        np.maximum.at(last_us, p[live], np.minimum(cu[live], ref))
        host = np.asarray(t["host"].to_pylist(), dtype=object)
        li = np.nonzero(live & orig & (host != ""))[0]
        linked14 = np.bincount(p[li], minlength=npool)
        if len(li):
            host_ids: dict[str, int] = {}
            hid = np.fromiter((host_ids.setdefault(h, len(host_ids)) for h in host[li]), np.int64, count=len(li))
            pair, cnt = np.unique((p[li].astype(np.int64) << 32) | hid, return_counts=True)
            names = {v: k for k, v in host_ids.items()}
            for o, h, c in zip((pair >> 32).tolist(), (pair & 0xFFFFFFFF).tolist(), cnt.tolist()):
                if c > top_domain14[o]:
                    top_domain14[o] = c
                    top_host[o] = names[h]
        cand = live & orig & t["safe"].to_numpy(zero_copy_only=False) & (
            (t["tlen"].to_numpy() >= MIN_SAMPLE_TEXT) | t["media"].to_numpy(zero_copy_only=False))
        ci = np.nonzero(cand)[0]
        if len(ci):
            rks = rk[ci].astype(str)
            oo = np.lexsort((rks, cu[ci], p[ci]))[::-1]  # pidx desc, created desc, rkey desc
            ci, rks = ci[oo], rks[oo]
            firsts = first_of_runs(p[ci])
            for i, r in zip(p[ci][firsts], rks[firsts]):
                sample[int(i)] = f"at://{dids[i]}/app.bsky.feed.post/{r}"
    dormant = posts14 == 0
    t_posts = time.time()

    def dist(mask):
        out = {"total": int(mask.sum()), "share": round(float(mask.mean()), 4) if npool else 0.0, "byRole": {}, "byTier": {}}
        for key, fn in (("byRole", lambda r: r.get("role") or "?"), ("byTier", lambda r: tier(int(r.get("followers") or 0)))):
            agg: dict[str, list[int]] = {}
            for i, r in enumerate(rows):
                a = agg.setdefault(fn(r), [0, 0])
                a[0] += 1
                a[1] += int(mask[i])
            out[key] = {g: {"n": a[0], "dormant": a[1], "share": round(a[1] / a[0], 4)} for g, a in sorted(agg.items())}
        return out

    rep["activity"] = {"windowStart": iso(ws), "postDays": [d.isoformat() for d in act_days],
                       "dormant": dist(dormant), "inactiveAndDormant": int((inactive & dormant).sum()),
                       "dormantWithRawRows": int((dormant & (raw14 > 0)).sum()),
                       "posts14Pct": {q: int(np.percentile(posts14, q)) for q in (10, 25, 50, 75, 90)} if npool else None,
                       "spearmanPosts14VsPoolEstimatedPPW": spearman(
                           posts14.astype(float), np.array([float(r.get("estimatedPostsPerWeek") or 0) for r in rows]))}
    # dormant sanity sample: raw rows in the window files and in any earlier posts files present
    early_files = [d / "posts.parquet" for d in day_dirs(export_root, act_days[0] - dt.timedelta(days=1))
                   if (d / "posts.parquet").exists()]
    early = count_rows(early_files, pool_arr, npool) if early_files and sanity_n else np.zeros(npool, np.int64)
    rng = np.random.default_rng(20261122)
    di = np.nonzero(dormant)[0]
    pick = sorted(rng.choice(di, size=min(sanity_n, len(di)), replace=False).tolist()) if len(di) else []
    rep["activity"]["sanity"] = {
        "earlierPostDays": [f.parent.name for f in early_files],
        "dormantWithEarlierRows": int((dormant & (early > 0)).sum()),
        "sample": [{"did": dids[i], "role": rows[i].get("role"), "followers": int(rows[i].get("followers") or 0),
                    "rawRowsInWindow": int(raw14[i]), "rawRowsEarlier": int(early[i])} for i in pick]}
    share = float(dormant.mean()) if npool else 0.0
    if share > max_dormant_share:
        rep["stopped"] = f"dormant share {share:.4f} > {max_dormant_share}"
        return {"code": 3, "report": rep, "signals": None, "detail": None, "error": rep["stopped"]}

    # link-feed gate: own posts mostly link to one site and almost nobody replies to or quotes them
    link_feed = LF.link_feed_mask(orig14, linked14, top_domain14, responders14) & ~inactive & ~dormant
    judged = orig14 >= LF.MIN_TOP_LEVEL
    rep["linkFeed"] = {
        "reason": LF.REASON,
        "thresholds": {"minTopLevel": LF.MIN_TOP_LEVEL, "minLinkShare": LF.MIN_LINK_SHARE,
                       "minTopDomainShare": LF.MIN_TOP_DOMAIN_SHARE,
                       "maxRespondersPerPost": LF.MAX_RESPONDERS_PER_POST},
        "flagged": dist(link_feed),
        "judged": int(judged.sum()),
        "accountsWithResponders": int((responders14 > 0).sum()),
        "respondersPerPostPct": {q: round(float(np.percentile(responders14[judged] / orig14[judged], q)), 4)
                                 for q in (5, 10, 25, 50)} if judged.any() else None,
        "accounts": [{"did": dids[i], "role": rows[i].get("role"), "followers": int(rows[i].get("followers") or 0),
                      "topLevel": int(orig14[i]), "linked": int(linked14[i]), "topHost": top_host[i],
                      "topHostPosts": int(top_domain14[i]), "responders": int(responders14[i])}
                     for i in sorted(np.nonzero(link_feed)[0].tolist(), key=lambda i: (-int(orig14[i]), dids[i]))]}
    lf_share = float(link_feed.mean()) if npool else 0.0
    if lf_share > max_link_feed_share:
        rep["stopped"] = f"link-feed share {lf_share:.4f} > {max_link_feed_share}"
        return {"code": 4, "report": rep, "signals": None, "detail": None, "error": rep["stopped"]}

    # sample freshness vs the pool's
    def uri_age_days(u):
        if not u:
            return None
        us, ok = tid_us(pa.array([u.rsplit("/", 1)[-1]]))
        return (ref - int(us[0])) / DAY_US if ok[0] else None

    gated = inactive | dormant | link_feed
    differs = [i for i in range(npool) if not gated[i] and sample[i] and sample[i] != rows[i].get("samplePostUri")]
    new_age = [a for a in (uri_age_days(sample[i]) for i in differs) if a is not None]
    old_age = [a for a in (uri_age_days(rows[i].get("samplePostUri")) for i in differs) if a is not None]
    rep["samplePost"] = {"ungatedWithCandidate": int(sum(1 for i in range(npool) if not gated[i] and sample[i])),
                         "ungatedWithoutCandidate": int(sum(1 for i in range(npool) if not gated[i] and not sample[i])),
                         "emitted": len(differs),
                         "sameAsPool": int(sum(1 for i in range(npool) if not gated[i] and sample[i]
                                               and sample[i] == rows[i].get("samplePostUri"))),
                         "medianAgeDaysNew": round(float(np.median(new_age)), 2) if new_age else None,
                         "medianAgeDaysPool": round(float(np.median(old_age)), 2) if old_age else None}

    # (d) kept-follow
    origin = w0 - 2 * BURST_HALF_US
    fo = scan_follows(fol_files, pool_arr, w0, w1, origin)
    kf = kept_follow(fo, npool, nb, w0, ref, removal_days, k, min_eligible, viewer_day_cap, rows)
    rep["keptFollow"] = {"createWindow": [follow_from, follow_to], "removalDays": removal_days,
                         "followDays": [f.parent.name for f in fol_files], **kf["report"]}
    t_follows = time.time()

    emit_sample = set(differs)
    accounts: dict[str, dict] = {}
    mult_out = np.ones(npool)
    for i, did in enumerate(dids):
        ent: dict = {}
        if gated[i]:
            ent["multiplier"] = 0.0
            mult_out[i] = 0.0
        else:
            m = round(float(kf["mult"][i]), 4)
            mult_out[i] = m
            if m != 1.0:
                ent["multiplier"] = m
            if i in emit_sample:
                ent["samplePostUri"] = sample[i]
        if ent:
            accounts[did] = ent
    rep["output"] = {"accounts": len(accounts), "multiplierZero": int(gated.sum()),
                     "linkFeedZero": int(link_feed.sum()),
                     "multiplierKeep": int(((mult_out != 1.0) & ~gated).sum()),
                     "samplePostUri": int(sum(1 for e in accounts.values() if "samplePostUri" in e))}
    rep["seconds"] = {"status": round(t_status - t0, 1), "posts": round(t_posts - t_status, 1),
                      "follows": round(t_follows - t_posts, 1), "total": round(time.time() - t0, 1)}
    signals = {"signalsVersion": VERSION, "generatedAt": iso(ref), "accounts": accounts}
    detail = {"dids": dids, "posts14": posts14, "originals14": orig14, "lastPostUs": last_us, "raw14": raw14,
              "inactive": inactive, "dormant": dormant, "linkFeed": link_feed, "linked14": linked14,
              "topDomain14": top_domain14, "responders14": responders14, "sample": sample, "n": kf["n"], "o": kf["o"], "mult": mult_out}
    return {"code": 0, "report": rep, "signals": signals, "detail": detail, "error": None}


def iso(us: int) -> str:
    return dt.datetime.fromtimestamp(us / 1_000_000, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_atomic(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export-root", type=pathlib.Path, required=True)
    ap.add_argument("--pool", type=pathlib.Path, required=True)
    ap.add_argument("--as-of", required=True, help="last export day (UTC), YYYY-MM-DD")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--report", type=pathlib.Path, default=None)
    ap.add_argument("--activity-days", type=int, default=14)
    ap.add_argument("--follow-from", default="2026-10-19")
    ap.add_argument("--follow-to", default="2026-11-07")
    ap.add_argument("--removal-days", type=int, default=14)
    ap.add_argument("--k", type=float, default=None, help="default: method of moments")
    ap.add_argument("--min-eligible", type=int, default=None, help="default: ceil(10 / g)")
    ap.add_argument("--viewer-day-cap", type=int, default=3, help="max removals per viewer per day; 0 = off")
    ap.add_argument("--max-dormant-share", type=float, default=0.25)
    ap.add_argument("--link-stats-out", type=pathlib.Path, default=None,
                    help="also write {did: [own posts, linked own posts]} for the window (why_picked.py input)")
    ap.add_argument("--max-link-feed-share", type=float, default=0.05,
                    help="stop without writing when a larger share of the pool looks like link feeds")
    ap.add_argument("--sanity-n", type=int, default=15)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--allow-live-path", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    if a.out.name == LIVE_PATH_NAME and not a.allow_live_path:
        print(f"refusing to write {LIVE_PATH_NAME}: a running service watches it", file=sys.stderr)
        return 2
    pa.set_cpu_count(a.threads)
    pa.set_io_thread_count(a.threads)
    res = build(a.export_root, a.pool, a.as_of, activity_days=a.activity_days, follow_from=a.follow_from,
                follow_to=a.follow_to, removal_days=a.removal_days, k=a.k, min_eligible=a.min_eligible,
                viewer_day_cap=a.viewer_day_cap, max_dormant_share=a.max_dormant_share,
                max_link_feed_share=a.max_link_feed_share, sanity_n=a.sanity_n,
                allow_missing=a.allow_missing)
    rep_text = json.dumps(res["report"], indent=1, sort_keys=True) + "\n"
    if a.report:
        write_atomic(a.report, rep_text)
    print(rep_text, file=sys.stderr)
    if res["code"] != 0:
        print(f"not written: {res['error']}", file=sys.stderr)
        return res["code"]
    write_atomic(a.out, json.dumps(res["signals"], sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"wrote {a.out} ({len(res['signals']['accounts'])} accounts)", file=sys.stderr)
    if a.link_stats_out:
        d = res["detail"]
        stats = {did: [int(n), int(k)] for did, n, k in zip(d["dids"], d["originals14"], d["linked14"]) if n > 0}
        write_atomic(a.link_stats_out, json.dumps({"asOf": a.as_of, "activityDays": a.activity_days,
                                                   "accounts": stats}, sort_keys=True, separators=(",", ":")) + "\n")
        print(f"wrote {a.link_stats_out} ({len(stats)} accounts)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
