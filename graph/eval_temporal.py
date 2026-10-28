#!/usr/bin/env python3
"""Temporal evaluation M1 and M2 at the cutoff T (whole-network plan S3, §6).

Plan: docs/execution-plans/2026-10-25-whole-network-signals-plan.md §3.1 S3 and S16, §4.7, §6.
Rules: docs/execution-plans/2026-10-28-whole-network-preregistration.md (S1); points it leaves
open are logged in docs/DECISIONS.md (2026-10-28, "eval_temporal.py ... readings").

  m1 --universe DIR --m1-dir DIR --arm NAME=DIR [--arm ...] [--item-sim NAME=DIR]
     [--exclude-dids FILE ...] [--compare A:B ...] [--freeze BASE:CHALLENGER --b1a-lists TSV]
     [--full-producers TSV] [--workers N] [--boot 2000] [--seed 20261028] --out JSON
  m2 --universe DIR --m2-dir DIR --arm NAME=DIR [--flags NAME=NPZ] [--ranker NAME=PARQUET]
     [--exclude-dids FILE ...] [--supplement-dids FILE] [--boot 2000] [--seed 20261028] --out JSON

Times are UTC microseconds (TID time, `tid >> 10`). Windows follow the S1 rules file
(docs/execution-plans/2026-10-28-whole-network-preregistration.md, R1): day-granular UTC, so
pre-T means created_us < 2026-08-16T00:00Z. Inputs (S2a writes the parquet files; S3 the partitions):
  --universe DIR       producers.tsv of the train-only fold (pid, did, followers at T). Every arm
                       must carry a byte-identical producers.tsv (leakage guard).
  --arm NAME=DIR       a train-only partition: producers.tsv, known_for.parquet (pid, did,
                       community, score, rank); communities.parquet for the m1 freeze's
                       granularity check and for m2 --flags (with membership.npy).
  --item-sim NAME=DIR  item-item cosine baseline: DIR/sim.bin (simgraph on the train edges).
  --m1-dir DIR
    m1_users.parquet         did [, lang]: M1 users, cityHash64(did) % 100 == 13 (checked here;
                             --m1-mod 200 after the registered halving).
    m1_train_follows.parquet did, subject, created_us: M1 users' follows live at T (all pre-T).
    m1_test_follows.parquet  did, subject, created_us, burst, live [, mutual]: post-T follows to
                             Oct 18; burst = the user's follows within ±30 s, itself included (R8).
    m1_engagement.parquet    optional; did, subject, created_us [, kind in {like, repost,
                             confirmed_reply}; without `kind` every row counts]; covers
                             (T - 90 d, 2026-09-12 + 30 d]. May be pre-aggregated to one row per
                             (did, subject, UTC day) holding that day's LAST qualifying event.
  --m2-dir DIR
    m2_cohort.parquet        did, created_us (PLC genesis; first record TID for did:web) [, split,
                             checked against the registered hash: test iff cityHash64(did||'m2') % 2 == 1].
    m2_follows.parquet       did, subject, created_us, burst, live: cohort follows to day 30.
    m2_engagement.parquet    optional; as m1_engagement.
  --exclude-dids FILE  one DID per line (purges, taken-down/labelled accounts, user-intents
                       opt-outs): removed as users, candidates, positives and list members. Not
                       the hide-from-recs union: amendment A1 keeps those accounts as candidates.
  --flags NAME=NPZ     community_flags.npz for that arm, keyed by RAW Leiden label
                       (membership.npy); mapped to community ids via top_dids majority.
  --ranker NAME=PARQUET  did, score: an in-card rank variant for M2 (S13/S16 supply these).

M1 ranks every eligible producer for each M1 user by InterestedIn·KnownFor
(score(u,q) = sum_c II_u[c]·KF[q,c], II_u[c] = sum over u's follows at T of KF[p,c]),
excluding producers u followed at T and u itself; zero-score ties fall back to a seeded
permutation, never to popularity. Recall@K = hits in top K / positives; NDCG@100 binary.
M2 picks each new user's top-3 cards by day-1-follow KnownFor mass and fills a 25-account
bundle (even split, popularity ceiling, co-membership fill rule); precision@25 = hits / 25.
CIs are percentile bootstraps (2,000 resamples, default_rng(20261028)) over users, or lists for
B1a; 95% unless a bar says 98.75%.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import s3_prep  # noqa: E402

DAY_US = 86_400 * 1_000_000


def utc_us(y: int, m: int, d: int) -> int:
    return int(datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc).timestamp()) * 1_000_000


# S1 R1: date windows are UTC and day-granular, "(a, b]" = a < UTC-date(TID) <= b, so every
# window below is a half-open [start, end) of day starts. T = 2026-08-15; pre-T = TID < Aug 16.
TRAIN_END_US = utc_us(2026, 8, 16)       # train / "followed at T": t < TRAIN_END
M1_START_US = TRAIN_END_US               # M1 primary positives (08-15, 09-12]
M1_END_US = utc_us(2026, 9, 13)
M1_HIST_US = utc_us(2026, 5, 18)         # discovery: no engagement in (05-17, 08-15]
M2_COHORT_START_US = utc_us(2026, 8, 15)  # M2 cohort created in [08-15, 09-08]
M2_COHORT_END_US = utc_us(2026, 9, 9)
LABEL_CUTOFF_US = utc_us(2026, 10, 19)   # every label window ends by Oct 18
BOOT_N, BOOT_SEED = 2000, 20261028       # S1 conventions: 2,000 resamples, default_rng(20261028)
KS = (10, 50, 100)
BURST_MIN = 10
ENGAGED_DAYS_US = 30 * DAY_US
BUNDLE = 25
POP_CEILING = (500_000, 2)               # at most 2 accounts over 500k followers per bundle
CARD_MIN = 25
N_PICKS = 3
M1_HASH = (100, 13)
FREEZE_REL = 0.05
FREEZE_B1A_LEVEL = 0.9875
GRANULARITY_TOL = 0.15                   # plan §6: matched median community size within ±15%
M2_BAR_RATIO = 1.2
M2_POWER_FLOOR = 2000
ENGAGE_KINDS = ("like", "repost", "confirmed_reply")
assert M1_END_US + ENGAGED_DAYS_US <= LABEL_CUTOFF_US and M2_COHORT_END_US + ENGAGED_DAYS_US <= LABEL_CUTOFF_US
assert M1_HIST_US == TRAIN_END_US - 90 * DAY_US


# ---------- shared structures ----------

def group(keys: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR offsets and stable order of rows grouped by integer key in [0, n)."""
    order = np.argsort(keys, kind="stable")
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(keys, minlength=n), out=off[1:])
    return off, order


def gather(off: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Positions of all CSR entries of `rows`, concatenated in row order."""
    starts, lens = off[rows], off[rows + 1] - off[rows]
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    return np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(total)


def read_dids(paths: list[str]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        with open(p) as fh:
            out |= {line.strip() for line in fh if line.strip() and not line.startswith("#")}
    return out


class Universe:
    """Train-only producers at T, eligibility and the seeded tie-break order."""

    def __init__(self, producers: pathlib.Path, exclude: set[str], seed: int):
        self.path = producers
        self.dids, self.followers = s3_prep.read_producers(producers)
        self.n = len(self.dids)
        self.did_array = pa.array(self.dids, pa.string())
        self.eligible = np.array([d not in exclude for d in self.dids], dtype=bool)
        self.elig_idx = np.flatnonzero(self.eligible)
        self.tiebreak = np.random.default_rng([seed, 1]).permutation(self.n)
        self.tie_order = self.elig_idx[np.argsort(self.tiebreak[self.elig_idx], kind="stable")]
        self.pop_order = self.elig_idx[np.lexsort((self.tiebreak[self.elig_idx], -self.followers[self.elig_idx]))]

    def index(self, values: pa.Array) -> np.ndarray:
        """Universe index per DID, -1 where the DID is not a producer at T."""
        return pc.fill_null(pc.index_in(values, value_set=self.did_array), -1).to_numpy().astype(np.int64)


class Partition:
    """A train-only partition's KnownFor as CSRs over universe indices."""

    def __init__(self, name: str, d: pathlib.Path, uni: Universe, flags: pathlib.Path | None = None):
        self.name, self.dir = name, d
        if s3_prep.sha256_file(d / "producers.tsv") != s3_prep.sha256_file(uni.path):
            raise SystemExit(f"leakage guard: arm {name} producers.tsv differs from the train-only universe")
        kf = pq.read_table(d / "known_for.parquet", columns=["did", "community", "score", "rank"])
        idx = uni.index(kf["did"].combine_chunks())
        if (idx < 0).any():
            raise SystemExit(f"arm {name}: known_for has DIDs outside producers.tsv")
        comm = kf["community"].to_numpy().astype(np.int64)
        score = kf["score"].to_numpy().astype(np.float64)
        rank = kf["rank"].to_numpy()
        self.n_comm = int(comm.max()) + 1 if len(comm) else 0
        off, order = group(idx, uni.n)
        self.prod_off, self.prod_comm, self.prod_score = off, comm[order], score[order]
        el = uni.eligible[idx]
        off, order = group(comm[el], self.n_comm)
        self.comm_off, self.comm_idx, self.comm_score = off, idx[el][order], score[el][order]
        self.primary = np.full(uni.n, -1, dtype=np.int64)
        self.primary_score = np.zeros(uni.n)
        r0 = rank == 0
        self.primary[idx[r0]] = comm[r0]
        self.primary_score[idx[r0]] = score[r0]
        self.flagged = np.zeros(self.n_comm, dtype=bool)
        if flags is not None:
            communities = pq.read_table(d / "communities.parquet", columns=["community", "top_dids"])
            membership = np.load(d / "membership.npy")
            raw = s3_prep.raw_label_by_community(communities, membership, {x: i for i, x in enumerate(uni.dids)})
            fl = np.load(flags)
            bad = np.zeros(len(fl["follow_back"]), dtype=bool)
            for key in ("follow_back", "follow_train", "low_post_follow_heavy"):
                if key in fl.files:
                    bad |= fl[key].astype(bool)
            for c, lab in raw.items():
                if c < self.n_comm:
                    self.flagged[c] = bool(bad[lab])
        self._co: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def interest(self, follows: np.ndarray) -> np.ndarray:
        pos = gather(self.prod_off, follows)
        return np.bincount(self.prod_comm[pos], weights=self.prod_score[pos], minlength=self.n_comm)

    def co_membership(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Community pairs sharing a producer in KnownFor: CSR (off, neighbour, count)."""
        if self._co is None:
            lens = np.diff(self.prod_off)
            a, b = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)]
            for i in range(int(lens.max(initial=0))):
                for j in range(int(lens.max(initial=0))):
                    rows = np.flatnonzero(lens > max(i, j))
                    if i != j and len(rows):
                        a.append(self.prod_comm[self.prod_off[rows] + i])
                        b.append(self.prod_comm[self.prod_off[rows] + j])
            key = np.concatenate(a) * max(self.n_comm, 1) + np.concatenate(b)
            uk, cnt = np.unique(key, return_counts=True)
            src, dst = uk // max(self.n_comm, 1), uk % max(self.n_comm, 1)
            off, order = group(src, self.n_comm)
            self._co = (off, dst[order], cnt[order])
        return self._co


def top_k(uni: Universe, cand: np.ndarray, w: np.ndarray, excl: np.ndarray, k: int) -> np.ndarray:
    """Top k distinct eligible candidates by (-score, tie-break), filled from the tie-break order."""
    top = np.zeros(0, dtype=np.int64)
    if len(cand):
        uniq, inv = np.unique(cand, return_inverse=True)
        s = np.bincount(inv, weights=w)
        keep = (s > 0) & uni.eligible[uniq] & ~np.isin(uniq, excl)
        uniq, s = uniq[keep], s[keep]
        if len(uniq) > k:
            thr = np.partition(s, len(s) - k)[len(s) - k]
            uniq, s = uniq[s >= thr], s[s >= thr]
        top = uniq[np.lexsort((uni.tiebreak[uniq], -s))][:k]
    if len(top) < k:
        top = np.concatenate([top, fill(uni.tie_order, np.union1d(excl, top), k - len(top))])
    return top


def fill(order: np.ndarray, excl: np.ndarray, n: int) -> np.ndarray:
    head = order[: n + len(excl)]
    return head[~np.isin(head, excl)][:n]


# ---------- M1 ----------

def read_pairs(path: pathlib.Path, users: pa.Array, uni: Universe, cols: list[str]) -> dict[str, np.ndarray]:
    t = pq.read_table(path)
    missing = [c for c in ["did", "subject", "created_us"] + cols if c not in t.column_names]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    out = {"user": pc.fill_null(pc.index_in(t["did"], value_set=users), -1).to_numpy().astype(np.int64),
           "item": uni.index(t["subject"].combine_chunks()),
           "created_us": t["created_us"].to_numpy().astype(np.int64)}
    for c in cols:
        out[c] = t[c].to_numpy(zero_copy_only=False)
    return out


def dedupe_first(user: np.ndarray, item: np.ndarray, t: np.ndarray, n_items: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = user * n_items + item
    order = np.lexsort((t, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    sel = order[first]
    return user[sel], item[sel], t[sel]


def pair_keys(user: np.ndarray, item: np.ndarray, n_items: int) -> np.ndarray:
    return user.astype(np.int64) * n_items + item.astype(np.int64)


def read_engagement(path: pathlib.Path, users: pa.Array, uni: Universe) -> dict[str, np.ndarray] | None:
    """Qualifying engagement rows mapped to (user, producer); `kind`, if present, is filtered."""
    if not path.exists():
        return None
    has_kind = "kind" in pq.read_schema(path).names
    eng = read_pairs(path, users, uni, ["kind"] if has_kind else [])
    ok = (eng["user"] >= 0) & (eng["item"] >= 0)
    if has_kind:
        ok &= np.isin(eng["kind"].astype(str), ENGAGE_KINDS)
    return {"key": pair_keys(eng["user"][ok], eng["item"][ok], uni.n), "created_us": eng["created_us"][ok]}


def engaged_within(eng: dict[str, np.ndarray], keys: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """For each (key, lo, hi): is there an engagement with that key and lo < t <= hi?"""
    sel = np.isin(eng["key"], keys)
    ek, et_ = eng["key"][sel], eng["created_us"][sel]
    order = np.lexsort((et_, ek))
    ek, et_ = ek[order], et_[order]
    out = np.zeros(len(keys), dtype=bool)
    starts, stops = np.searchsorted(ek, keys, "left"), np.searchsorted(ek, keys, "right")
    for j in np.flatnonzero(stops > starts):
        ts = et_[starts[j]:stops[j]]
        first_after = np.searchsorted(ts, lo[j], "right")
        out[j] = first_after < len(ts) and ts[first_after] <= hi[j]
    return out


def keys_with_days(eng: dict[str, np.ndarray], lo: int, hi: int, min_days: int) -> np.ndarray:
    """Keys engaged on >= min_days distinct UTC days with lo <= t < hi."""
    m = (eng["created_us"] >= lo) & (eng["created_us"] < hi)
    k, day = eng["key"][m], eng["created_us"][m] // DAY_US
    if not len(k):
        return np.zeros(0, dtype=np.int64)
    kd = np.unique(np.stack([k, day], axis=1), axis=0)
    uk, cnt = np.unique(kd[:, 0], return_counts=True)
    return uk[cnt >= min_days]


class M1Data:
    def __init__(self, d: pathlib.Path, uni: Universe, exclude: set[str], mod: int, clickhouse: str):
        users_t = pq.read_table(d / "m1_users.parquet")
        dids = [x for x in users_t["did"].to_pylist() if x not in exclude]
        self.excluded_users = users_t.num_rows - len(dids)
        residues = s3_prep.city_hash_mod(dids, mod, clickhouse=clickhouse)
        self.hash_mismatches = int((residues != M1_HASH[1]).sum())
        if self.hash_mismatches:
            raise SystemExit(f"sample guard: {self.hash_mismatches} M1 users fail cityHash64(did) % {mod} == {M1_HASH[1]}")
        lang = dict(zip(users_t["did"].to_pylist(), users_t["lang"].to_pylist())) if "lang" in users_t.column_names else {}
        self.dids, self.lang = dids, [lang.get(x) for x in dids]
        users = pa.array(dids, pa.string())
        n = self.n = len(dids)
        self.self_idx = uni.index(users)
        tr = read_pairs(d / "m1_train_follows.parquet", users, uni, [])
        if len(tr["created_us"]) and tr["created_us"].max() >= TRAIN_END_US:
            raise SystemExit("leakage guard: m1_train_follows has follows on or after 2026-08-16 (post-T)")
        m = (tr["user"] >= 0) & (tr["item"] >= 0)
        tu, ti, _ = dedupe_first(tr["user"][m], tr["item"][m], tr["created_us"][m], uni.n)
        self.f_off, order = group(tu, n)
        self.f_items = ti[order]
        followed = np.unique(pair_keys(tu, ti, uni.n))
        te = read_pairs(d / "m1_test_follows.parquet", users, uni, ["burst", "live"] +
                        (["mutual"] if "mutual" in pq.read_schema(d / "m1_test_follows.parquet").names else []))
        win = (te["created_us"] >= M1_START_US) & (te["created_us"] < M1_END_US)
        self.test_rows_outside_window = int((~win).sum())
        in_window = (te["user"] >= 0) & win
        # Follows of accounts that are not producers at T cannot be ranked by any arm.
        self.test_follows_outside_universe = int((in_window & (te["item"] < 0)).sum())
        m = ((te["user"] >= 0) & (te["item"] >= 0) & win
             & te["live"].astype(bool) & (te["burst"].astype(np.int64) < BURST_MIN))
        m &= uni.eligible[np.maximum(te["item"], 0)] & (te["item"] != self.self_idx[np.maximum(te["user"], 0)])
        pu, pi, pt = dedupe_first(te["user"][m], te["item"][m], te["created_us"][m], uni.n)
        keep = ~np.isin(pair_keys(pu, pi, uni.n), followed)
        self.burst_or_dead_dropped = int(((te["user"] >= 0) & (te["item"] >= 0) & win).sum() - m.sum())
        pu, pi, pt = pu[keep], pi[keep], pt[keep]
        mutual = None
        if "mutual" in te:
            mk = pair_keys(te["user"], te["item"], uni.n)
            order = np.argsort(mk, kind="stable")
            j = np.searchsorted(mk[order], pair_keys(pu, pi, uni.n))
            mutual = te["mutual"].astype(bool)[order][j]
        eng = read_engagement(d / "m1_engagement.parquet", users, uni)
        self.has_engagement = eng is not None
        sets: dict[str, tuple[np.ndarray, np.ndarray]] = {"primary": (pu, pi)}
        self.pos_attr = {"primary": {"follower_bucket": np.digitize(uni.followers[pi], [1_000, 10_000, 100_000, 1_000_000])}}
        if mutual is not None:
            self.pos_attr["primary"]["mutual"] = mutual
        if eng is not None:
            sec = engaged_within(eng, pair_keys(pu, pi, uni.n), pt, pt + ENGAGED_DAYS_US)
            sets["secondary"] = (pu[sec], pi[sec])
            before = keys_with_days(eng, M1_HIST_US, TRAIN_END_US, 1)
            disc = np.setdiff1d(keys_with_days(eng, M1_START_US, M1_END_US, 2), np.union1d(before, followed))
            du, di = disc // uni.n, disc % uni.n
            ok = uni.eligible[di] & (di != self.self_idx[du])
            sets["discovery"] = (du[ok], di[ok])
        self.sets = sets

    def follows(self, u: int) -> np.ndarray:
        return self.f_items[self.f_off[u]:self.f_off[u + 1]]

    def degree(self) -> np.ndarray:
        return np.diff(self.f_off)


_STATE: dict = {}


def rank_user(u: int) -> dict[str, np.ndarray]:
    uni, data, arms, sims, seed, k = (_STATE[x] for x in ("uni", "data", "arms", "sims", "seed", "k"))
    f = data.follows(u)
    excl = np.union1d(f, [data.self_idx[u]] if data.self_idx[u] >= 0 else [])
    out = {}
    for part in arms:
        ii = part.interest(f)
        nz = np.flatnonzero(ii)
        pos = gather(part.comm_off, nz)
        w = part.comm_score[pos] * np.repeat(ii[nz], np.diff(part.comm_off)[nz])
        out[part.name] = top_k(uni, part.comm_idx[pos], w, excl, k)
    for name, (off, q, wq) in sims.items():
        pos = gather(off, f)
        out[name] = top_k(uni, q[pos], wq[pos].astype(np.float64), excl, k)
    out["popularity"] = fill(uni.pop_order, excl, k)
    rng = np.random.default_rng([seed, 7, u])
    draw = uni.elig_idx[rng.choice(len(uni.elig_idx), size=min(len(uni.elig_idx), k + len(excl)), replace=False)]
    out["random"] = draw[~np.isin(draw, excl)][:k]
    pu, pi = _STATE["oracle"]
    mine = pi[np.searchsorted(pu, u, "left"):np.searchsorted(pu, u, "right")]
    mine = mine[np.argsort(uni.tiebreak[mine], kind="stable")][:k]
    out["oracle"] = np.concatenate([mine, fill(uni.tie_order, np.union1d(excl, mine), k - len(mine))])
    return out


def rank_chunk(users: range) -> dict[str, np.ndarray]:
    k = _STATE["k"]
    rows = [rank_user(u) for u in users]
    names = rows[0].keys() if rows else []
    return {n: np.array([np.pad(r[n], (0, k - len(r[n])), constant_values=-1) for r in rows], dtype=np.int64).reshape(-1, k)
            for n in names}


def rank_all(n_users: int, workers: int) -> dict[str, np.ndarray]:
    chunks = [range(s, min(s + 2000, n_users)) for s in range(0, n_users, 2000)]
    if workers <= 1:
        parts = [rank_chunk(c) for c in chunks]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(rank_chunk, chunks)
    names = parts[0].keys()
    return {n: np.concatenate([p[n] for p in parts]) for n in names}


def hit_ranks(top: np.ndarray, pu: np.ndarray, pi: np.ndarray, n_items: int) -> np.ndarray:
    """Rank (0-based) of each positive (pu, pi) in its user's top list, -1 if absent."""
    rows, cols = np.nonzero(top >= 0)
    keys = rows.astype(np.int64) * n_items + top[rows, cols]
    order = np.argsort(keys, kind="stable")
    fs, rs = keys[order], cols[order]
    if not len(fs):
        return np.full(len(pu), -1, dtype=np.int64)
    q = pu * n_items + pi
    j = np.minimum(np.searchsorted(fs, q), len(fs) - 1)
    return np.where(fs[j] == q, rs[j], -1)


def user_metrics(ranks: np.ndarray, pu: np.ndarray, n_users: int) -> dict[str, np.ndarray]:
    npos = np.bincount(pu, minlength=n_users).astype(float)
    out = {}
    for k in KS:
        out[f"recall@{k}"] = np.bincount(pu, weights=((ranks >= 0) & (ranks < k)).astype(float), minlength=n_users) / np.maximum(npos, 1)
    gain = np.where((ranks >= 0) & (ranks < 100), 1.0 / np.log2(np.maximum(ranks, 0) + 2.0), 0.0)
    ideal = np.r_[0.0, np.cumsum(1.0 / np.log2(np.arange(100) + 2.0))]
    out["ndcg@100"] = np.bincount(pu, weights=gain, minlength=n_users) / np.maximum(ideal[np.minimum(npos, 100).astype(int)], 1e-300)
    out["_mask"] = npos > 0
    out["_npos"] = npos
    return out


# ---------- statistics ----------

def bootstrap(X: np.ndarray, M: np.ndarray, n_boot: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Masked column means and their bootstrap replicates over resampled rows (users or lists)."""
    XM, Mf = np.where(M, X, 0.0), M.astype(float)
    n = X.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        point = XM.sum(0) / Mf.sum(0)
        rng = np.random.default_rng(seed)
        boots = np.empty((n_boot, X.shape[1]))
        for b0 in range(0, n_boot, 50):
            nb = min(50, n_boot - b0)
            draws = rng.integers(0, n, (nb, n))
            counts = np.stack([np.bincount(row, minlength=n) for row in draws]).astype(float)
            boots[b0:b0 + nb] = (counts @ XM) / (counts @ Mf)
    return point, boots


def ci(values: np.ndarray, level: float = 0.95) -> list[float]:
    a = (1 - level) / 2
    if not np.isfinite(values).any():
        return [float("nan"), float("nan")]
    return [float(np.nanpercentile(values, 100 * a)), float(np.nanpercentile(values, 100 * (1 - a)))]


def summary(point: float, boots: np.ndarray, n: int, level: float = 0.95) -> dict:
    return {"mean": float(point), "ci": ci(boots, level), "level": level, "n": int(n)}


def finish(report: dict) -> dict:
    body = json.dumps(report, sort_keys=True, default=float)
    report["metrics_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    return report


# ---------- B1a lists (S3 freeze) ----------

def b1a_lists(path: pathlib.Path) -> list[list[str]]:
    """S1 R11, exactly as audit/list_purity.py: rows sorted, identical member sets deduped,
    <= 3 lists per owner chosen with default_rng(7); lists of <= 500 items only."""
    rng = np.random.default_rng(7)
    rows = sorted((uri, owner, name, sorted(set(members)))
                  for uri, owner, name, members, total in s3_prep.read_lists(path, with_total=True) if total <= 500)
    seen, by_owner = set(), {}
    for r in rows:
        key = tuple(r[3])
        if len(key) >= 3 and key not in seen:
            seen.add(key)
            by_owner.setdefault(r[1], []).append(r)
    out = []
    for rs in by_owner.values():
        out += [rs[i][3] for i in sorted(rng.choice(len(rs), min(3, len(rs)), replace=False))]
    return out


def list_top1(lists: list[list[str]], parts: list[Partition], uni: Universe) -> tuple[np.ndarray, np.ndarray]:
    """Partition-form macro top-1 per list (S1 R12): the share of member pairs with the same
    primary community, over members that are eligible producers with a primary in every arm;
    a list counts when >= 3 such members remain (R11)."""
    pid = {d: i for i, d in enumerate(uni.dids)}
    vals = np.zeros((len(lists), len(parts)))
    mask = np.zeros((len(lists), len(parts)), dtype=bool)
    for li, members in enumerate(lists):
        idx = [pid[d] for d in members if d in pid and uni.eligible[pid[d]]]
        idx = [i for i in idx if all(p.primary[i] >= 0 for p in parts)]
        if len(idx) < 3:
            continue
        for j, p in enumerate(parts):
            cnt = np.unique(p.primary[idx], return_counts=True)[1]
            vals[li, j] = float((cnt * (cnt - 1)).sum()) / (len(idx) * (len(idx) - 1))
            mask[li, j] = True
    return vals, mask


def same_primary_null(part: Partition, uni: Universe) -> float:
    """Uniform same-size null: P(two random assigned producers share a primary community)."""
    cnt = np.bincount(part.primary[(part.primary >= 0) & uni.eligible]).astype(float)
    n = cnt.sum()
    return float((cnt * (cnt - 1)).sum() / (n * (n - 1))) if n > 1 else float("nan")


def median_community_size(d: pathlib.Path) -> float | None:
    path = d / "communities.parquet"
    return float(np.median(pq.read_table(path, columns=["size"])["size"].to_numpy())) if path.exists() else None


# ---------- M1 driver ----------

def run_m1(args) -> dict:
    exclude = read_dids(args.exclude_dids)
    uni = Universe(args.universe / "producers.tsv", exclude, args.seed)
    if args.full_producers and s3_prep.sha256_file(args.full_producers) == s3_prep.sha256_file(uni.path):
        raise SystemExit("leakage guard: the universe is the full-data producers.tsv, not the train-only one")
    arms = [Partition(n, pathlib.Path(p), uni) for n, p in (a.split("=", 1) for a in args.arm)]
    sims = {}
    for spec in args.item_sim:
        name, p = spec.split("=", 1)
        if s3_prep.sha256_file(pathlib.Path(p) / "producers.tsv") != s3_prep.sha256_file(uni.path):
            raise SystemExit(f"leakage guard: item-sim {name} producers.tsv differs from the universe")
        raw = np.fromfile(pathlib.Path(p) / "sim.bin", dtype=s3_prep.SIM_DTYPE)
        off, order = group(raw["p"].astype(np.int64), uni.n)
        sims[name] = (off, raw["q"][order].astype(np.int64), raw["w"][order])
    data = M1Data(args.m1_dir, uni, exclude, args.m1_mod, args.clickhouse)
    pu, pi = data.sets["primary"]
    _STATE.update(uni=uni, data=data, arms=arms, sims=sims, seed=args.seed, k=max(KS), oracle=(pu, pi))
    tops = rank_all(data.n, args.workers)
    names = list(tops)
    cols, masks, labels = [], [], []
    deg = data.degree()
    elig_follow = np.array([uni.eligible[data.follows(u)].sum() for u in range(data.n)])
    n_u = len(uni.elig_idx) - elig_follow - (data.self_idx >= 0) * uni.eligible[np.maximum(data.self_idx, 0)]
    user_slices = {"all": np.ones(data.n, dtype=bool), "deg0": deg == 0, "deg1-9": (deg >= 1) & (deg < 10),
                   "deg10-99": (deg >= 10) & (deg < 100), "deg100-999": (deg >= 100) & (deg < 1000),
                   "deg1000-2999": (deg >= 1000) & (deg < 3000), "mass_follower_3000+": deg >= 3000,
                   "follows_le20": deg <= 20}
    langs = collections.Counter(x for x in data.lang if x)
    for lang in sorted(langs, key=lambda x: (-langs[x], x))[:10]:
        user_slices[f"lang={lang}"] = np.array([x == lang for x in data.lang])
    for set_name, (su, si) in data.sets.items():
        attr = data.pos_attr.get(set_name, {})
        pos_slices = {"all": np.ones(len(su), dtype=bool)}
        for key, values in attr.items():
            for v in np.unique(values):
                pos_slices[f"{key}={int(v)}"] = values == v
        for arm in names:
            if arm == "oracle" and set_name != "primary":
                continue
            ranks = hit_ranks(tops[arm], su, si, uni.n)
            for pname, pm in pos_slices.items():
                um = user_metrics(ranks[pm], su[pm], data.n)
                for uname, umask in user_slices.items():
                    if pname != "all" and uname != "all":
                        continue
                    for metric in ("recall@10", "recall@50", "recall@100", "ndcg@100"):
                        cols.append(um[metric])
                        masks.append(um["_mask"] & umask)
                        labels.append((set_name, arm, f"{pname}|{uname}" if pname != "all" else uname, metric))
    npos_all = np.bincount(pu, minlength=data.n).astype(float)
    ceiling = {k: np.minimum(npos_all, k) / np.maximum(npos_all, 1) for k in KS}
    X, M = np.column_stack(cols), np.column_stack(masks)
    point, boots = bootstrap(X, M, args.boot, args.seed)
    col = {lab: j for j, lab in enumerate(labels)}
    metrics: dict = {}
    for (s, arm, sl, metric), j in col.items():
        metrics.setdefault(s, {}).setdefault(arm, {}).setdefault(sl, {})[metric] = summary(point[j], boots[:, j], M[:, j].sum())
    prim = M[:, col[("primary", "oracle", "all", "recall@50")]]
    checks = {"oracle": {}, "random_vs_base_rate": {}}
    for k in KS:
        j = col[("primary", "oracle", "all", f"recall@{k}")]
        checks["oracle"][f"recall@{k}_equals_ceiling"] = bool(np.allclose(X[prim, j], ceiling[k][prim]))
        jr = col[("primary", "random", "all", f"recall@{k}")]
        npos = np.bincount(pu, minlength=data.n)[prim].astype(float)
        nu = np.maximum(n_u[prim], 1).astype(float)
        base = np.minimum(k / nu, 1.0)
        var = (k * (npos / nu) * (1 - npos / nu) * np.maximum(nu - k, 0) / np.maximum(nu - 1, 1)) / npos ** 2
        diff, se = float(X[prim, jr].mean() - base.mean()), float(np.sqrt(var.sum()) / max(prim.sum(), 1))
        checks["random_vs_base_rate"][f"recall@{k}"] = {"random": float(X[prim, jr].mean()), "base_rate": float(base.mean()),
                                                        "z": diff / se if se > 0 else 0.0, "pass": bool(abs(diff) <= 3.29 * se + 1e-12)}
    checks["oracle"]["ndcg@100_is_1"] = bool(np.allclose(X[prim, col[("primary", "oracle", "all", "ndcg@100")]], 1.0))
    report = {
        "step": "S3/M1", "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md §6",
        "constants": {"train_end_us": TRAIN_END_US, "m1_window_us": [M1_START_US, M1_END_US], "hist_start_us": M1_HIST_US, "ks": KS,
                      "burst_min": BURST_MIN, "boot": args.boot, "seed": args.seed, "m1_hash": [args.m1_mod, M1_HASH[1]]},
        "inputs": {"universe": s3_prep.sha256_file(uni.path),
                   **{f"m1/{p.name}": s3_prep.sha256_file(p) for p in sorted(args.m1_dir.glob("m1_*.parquet"))},
                   **{f"arm/{a.name}/known_for": s3_prep.sha256_file(a.dir / "known_for.parquet") for a in arms}},
        "guards": {"train_follows_pre_T": True, "arms_share_train_universe": True, "m1_hash_mismatches": data.hash_mismatches,
                   "excluded_users": data.excluded_users, "test_rows_outside_window": data.test_rows_outside_window,
                   "test_follows_outside_universe": data.test_follows_outside_universe,
                   "burst_dead_or_ineligible_dropped": data.burst_or_dead_dropped, "eligible_producers": int(len(uni.elig_idx)),
                   "producers": uni.n},
        "users": {"m1_users": data.n, **{f"with_{s}_positive": int(np.unique(su).size) for s, (su, _si) in data.sets.items()}},
        "positives": {s: int(len(su)) for s, (su, _si) in data.sets.items()},
        "engagement_sets": "computed" if data.has_engagement else "unavailable: no m1_engagement.parquet",
        "self_checks": checks, "metrics": metrics,
    }
    report["compare"] = {}
    for spec in args.compare + ([args.freeze] if args.freeze else []):
        a, b = spec.split(":")
        out = {}
        for s in data.sets:
            for metric in ("recall@50", "recall@100"):
                ja, jb = col[(s, a, "all", metric)], col[(s, b, "all", metric)]
                rel_b = boots[:, jb] / boots[:, ja] - 1
                out[f"{s}/{metric}"] = {"rel": float(point[jb] / point[ja] - 1), "ci": ci(rel_b), "level": 0.95}
        report["compare"][spec] = out
    if args.freeze:
        base, chal = args.freeze.split(":")
        m1 = report["compare"][args.freeze]["primary/recall@50"]
        freeze = {"rule": " ".join(freeze_decision.__doc__.split()), "m1_follow_recall@50": m1}
        by_name = {p.name: p for p in arms}
        sizes = {n: median_community_size(by_name[n].dir) for n in (base, chal)}
        granularity_ok = (None if None in sizes.values()
                          else bool(abs(sizes[chal] / sizes[base] - 1) <= GRANULARITY_TOL))
        freeze["median_community_size"] = {**sizes, "tolerance": GRANULARITY_TOL}
        dci = None
        if args.b1a_lists:
            vals, lm = list_top1(b1a_lists(args.b1a_lists), [by_name[base], by_name[chal]], uni)
            lp, lb = bootstrap(vals, lm, args.boot, args.seed)
            dci = ci(lb[:, 1] - lb[:, 0], FREEZE_B1A_LEVEL)
            nulls = {n: same_primary_null(by_name[n], uni) for n in (base, chal)}
            freeze["b1a_macro_top1"] = {base: float(lp[0]), chal: float(lp[1]), "diff": float(lp[1] - lp[0]), "ci": dci,
                                        "level": FREEZE_B1A_LEVEL, "lists": int(lm[:, 0].sum()),
                                        "lift_over_null": {n: float(lp[j]) / nulls[n] for j, n in enumerate((base, chal))},
                                        "lists_sha256": s3_prep.sha256_file(args.b1a_lists)}
        freeze.update(freeze_decision(base, chal, m1["rel"], m1["ci"], dci, granularity_ok))
        report["freeze"] = freeze
    return finish(report)


# ---------- M2 ----------

class M2Data:
    def __init__(self, d: pathlib.Path, uni: Universe, exclude: set[str], clickhouse: str):
        ct = pq.read_table(d / "m2_cohort.parquet")
        dids, created = ct["did"].to_pylist(), ct["created_us"].to_numpy().astype(np.int64)
        inside = (created >= M2_COHORT_START_US) & (created < M2_COHORT_END_US)
        keep = np.array([x not in exclude for x in dids], dtype=bool) & inside
        self.dropped = {"excluded": int(sum(x in exclude for x in dids)), "created_outside_window": int((~inside).sum())}
        self.dids = [x for x, k in zip(dids, keep) if k]
        self.created = created[keep]
        # S1 R3: test iff cityHash64(did || 'm2') % 2 == 1. An input `split` column must agree.
        self.split = np.where(s3_prep.city_hash_mod(self.dids, 2, salt="m2", clickhouse=clickhouse) == 1, "test", "val")
        if "split" in ct.column_names:
            given = np.array([x for x, k in zip(ct["split"].to_pylist(), keep) if k])
            if (given != self.split).any():
                raise SystemExit("split guard: m2_cohort.parquet `split` disagrees with cityHash64(did || 'm2') % 2")
        users = pa.array(self.dids, pa.string())
        self.n = n = len(self.dids)
        self.self_idx = uni.index(users)
        fo = read_pairs(d / "m2_follows.parquet", users, uni, ["burst", "live"])
        m = fo["user"] >= 0
        u, i, t = fo["user"][m], fo["item"][m], fo["created_us"][m]
        burst, live = fo["burst"][m].astype(np.int64), fo["live"][m].astype(bool)
        c = self.created[u]
        day1_any = (t >= c) & (t < c + DAY_US)
        day1 = day1_any & (i >= 0)
        self.seed_off, order = group(u[day1], n)
        self.seed_items = i[day1][order]
        self.pack_burst = np.zeros(n, dtype=bool)
        self.pack_burst[u[day1_any & (burst >= BURST_MIN)]] = True
        out = ((t >= c + DAY_US) & (t < c + ENGAGED_DAYS_US) & (i >= 0) & live & (burst < BURST_MIN))
        out &= uni.eligible[np.maximum(i, 0)] & (i != self.self_idx[u])
        ou, oi, _ot = dedupe_first(u[out], i[out], t[out], uni.n)
        k = ~np.isin(pair_keys(ou, oi, uni.n), pair_keys(u[day1], i[day1], uni.n))
        self.outcomes = (ou[k], oi[k])
        self.engaged = None
        eng = read_engagement(d / "m2_engagement.parquet", users, uni)
        if eng is not None:
            # "Engaged by day 30": any qualifying engagement from account creation to day 30.
            c0 = self.created[self.outcomes[0]]
            e = engaged_within(eng, pair_keys(*self.outcomes, uni.n), c0 - 1, c0 + ENGAGED_DAYS_US - 1)
            self.engaged = (self.outcomes[0][e], self.outcomes[1][e])

    def seeds(self, u: int) -> np.ndarray:
        return self.seed_items[self.seed_off[u]:self.seed_off[u + 1]]


def card_rankings(part: Partition, uni: Universe, rankers: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Eligible primary members per community, ordered per ranker by (-key, tie-break)."""
    members = np.flatnonzero((part.primary >= 0) & uni.eligible)
    off, _ = group(part.primary[members], part.n_comm)
    out = {}
    for name, key in rankers.items():
        order = np.lexsort((uni.tiebreak[members], -key[members], part.primary[members]))
        out[name] = members[order]
    return off, out


def allocate(n_picks: int, size: int | None = None) -> list[int]:
    """Even split, remainder to the higher-mass picks (≥3 per card; ≤40% per card for 3 picks)."""
    size = BUNDLE if size is None else size
    return [size // n_picks + (1 if j < size % n_picks else 0) for j in range(n_picks)]


def build_bundle(picks: list[int], part: Partition, off: np.ndarray, ranked: np.ndarray, uni: Universe,
                 excl: set[int], size: int | None = None) -> tuple[list[int], int]:
    """Bundle per plan §3.3: even split over picks, popularity ceiling, then the fill rule
    (nearest non-flagged communities by KnownFor co-membership) on any shortfall."""
    size = BUNDLE if size is None else size
    chosen: list[int] = []
    big = 0

    def take(c: int, n: int) -> None:
        nonlocal big
        got = 0
        for p in ranked[off[c]:off[c + 1]]:
            if got >= n:
                break
            p = int(p)
            if p in excl or p in chosen:
                continue
            if uni.followers[p] > POP_CEILING[0]:
                if big >= POP_CEILING[1]:
                    continue
                big += 1
            chosen.append(p)
            got += 1

    for c, n in zip(picks, allocate(len(picks), size)):
        take(c, n)
    filled = 0
    if len(chosen) < size:
        co_off, co_dst, co_cnt = part.co_membership()
        score: dict[int, int] = {}
        for c in picks:
            for nb, cnt in zip(co_dst[co_off[c]:co_off[c + 1]], co_cnt[co_off[c]:co_off[c + 1]]):
                if int(nb) not in picks and not part.flagged[nb]:
                    score[int(nb)] = score.get(int(nb), 0) + int(cnt)
        for nb in sorted(score, key=lambda x: (-score[x], x)):
            if len(chosen) >= size:
                break
            before = len(chosen)
            take(nb, size - len(chosen))
            filled += len(chosen) - before
    return chosen, filled


def freeze_decision(base: str, challenger: str, m1_rel: float, m1_ci: list[float],
                    b1a_ci: list[float] | None, granularity_ok: bool | None) -> dict:
    """S3 freeze (S1 §9.2): the challenger only if M1-follow Recall@50 is >= +5% relative with
    its 95% CI above 0 (R2), the 98.75% CI lower bound of B1a macro top-1 (challenger - base)
    is >= 0 (R14), and the two partitions have matched granularity (plan §6); else the base."""
    m1_pass = bool(m1_rel >= FREEZE_REL and m1_ci[0] > 0)
    if b1a_ci is None or granularity_ok is None:
        return {"m1_pass": m1_pass, "b1a_pass": None,
                "decision": "undecided: needs --b1a-lists and communities.parquet in both arms"}
    b1a_pass = bool(b1a_ci[0] >= 0)
    ok = m1_pass and b1a_pass and granularity_ok
    return {"m1_pass": m1_pass, "b1a_pass": b1a_pass, "granularity_matched": granularity_ok,
            "decision": challenger if ok else base}


def pick_cards(part: Partition, seeds: np.ndarray, card_ok: np.ndarray) -> list[int]:
    mass = part.interest(seeds)
    cand = np.flatnonzero((mass > 0) & card_ok)
    return [int(c) for c in cand[np.lexsort((cand, -mass[cand]))][:N_PICKS]]


def run_m2(args) -> dict:
    exclude = read_dids(args.exclude_dids)
    uni = Universe(args.universe / "producers.tsv", exclude, args.seed)
    flags = dict(f.split("=", 1) for f in args.flags)
    arms = [Partition(n, pathlib.Path(p), uni, pathlib.Path(flags[n]) if n in flags else None)
            for n, p in (a.split("=", 1) for a in args.arm)]
    data = M2Data(args.m2_dir, uni, exclude, args.clickhouse)
    supplement = read_dids([args.supplement_dids]) if args.supplement_dids else set()
    sup_mask = np.array([d in supplement for d in uni.dids], dtype=bool)
    custom = {}
    for spec in args.ranker:
        name, p = spec.split("=", 1)
        t = pq.read_table(p, columns=["did", "score"])
        key = np.full(uni.n, -np.inf)
        idx = uni.index(t["did"].combine_chunks())
        key[idx[idx >= 0]] = t["score"].to_numpy()[idx >= 0]
        custom[name] = key
    report = {"step": "S3/M2", "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md §6, §3.3",
              "constants": {"cohort_window_us": [M2_COHORT_START_US, M2_COHORT_END_US], "bundle": BUNDLE, "pop_ceiling": POP_CEILING,
                            "card_min_eligible": CARD_MIN, "picks": N_PICKS, "burst_min": BURST_MIN, "boot": args.boot,
                            "seed": args.seed, "bar_ratio": M2_BAR_RATIO, "power_floor": M2_POWER_FLOOR},
              "inputs": {"universe": s3_prep.sha256_file(uni.path),
                         **{f"m2/{p.name}": s3_prep.sha256_file(p) for p in sorted(args.m2_dir.glob("m2_*.parquet"))}},
              "cohort": {"users": data.n, **data.dropped, "val": int((data.split == "val").sum()),
                         "test": int((data.split == "test").sum())},
              "arms": {}}
    ou, oi = data.outcomes
    for part in arms:
        rankers = {"community_popularity": uni.followers.astype(float), "knownfor": part.primary_score, **custom}
        off, ranked = card_rankings(part, uni, rankers)
        card_ok = (np.diff(off) >= CARD_MIN) & ~part.flagged
        picks = [pick_cards(part, data.seeds(u), card_ok) for u in range(data.n)]
        has_pick = np.array([len(p) > 0 for p in picks])
        outcome_sets = {"follow": (ou, oi)}
        if data.engaged is not None:
            outcome_sets["engaged"] = data.engaged
        o_by_user = {name: {} for name in outcome_sets}
        for name, (a, b) in outcome_sets.items():
            for x, y in zip(a.tolist(), b.tolist()):
                o_by_user[name].setdefault(x, set()).add(y)
        follow_out = o_by_user["follow"]
        evaluated = has_pick & np.array([u in follow_out for u in range(data.n)])
        cols, masks, labels, fill_counts = [], [], [], {}
        oracle_ok = True
        for rname in list(rankers) + ["oracle"]:
            hits = {f"{o}/{s}": np.zeros(data.n) for o in outcome_sets for s in ("all", "le100k", "supplement")}
            filled = 0
            for u in np.flatnonzero(evaluated):
                excl = set(data.seeds(u).tolist()) | ({int(data.self_idx[u])} if data.self_idx[u] >= 0 else set())
                if rname == "oracle":
                    outs = sorted(follow_out[u], key=lambda p: uni.tiebreak[p])
                    bundle = outs[:BUNDLE]
                else:
                    bundle, f = build_bundle(picks[u], part, off, ranked[rname], uni, excl)
                    filled += f
                for o in outcome_sets:
                    got = [p for p in bundle if p in o_by_user[o].get(u, ())]
                    hits[f"{o}/all"][u] = len(got) / BUNDLE
                    hits[f"{o}/le100k"][u] = sum(uni.followers[p] <= 100_000 for p in got) / BUNDLE
                    hits[f"{o}/supplement"][u] = sum(sup_mask[p] for p in got) / BUNDLE
                if rname == "oracle":
                    hits["follow/all"][u] = len(bundle) / min(BUNDLE, len(follow_out[u]))
                    oracle_ok &= hits["follow/all"][u] == 1.0
            fill_counts[rname] = filled
            for key, values in hits.items():
                if rname == "oracle" and key != "follow/all":
                    continue
                if key.endswith("supplement") and not supplement:
                    continue
                for split in ("val", "test"):
                    for sl, sm in (("all", None), ("pack_burst", True), ("no_pack_burst", False)):
                        m = evaluated & (data.split == split)
                        if sm is not None:
                            m = m & (data.pack_burst == sm)
                        cols.append(values)
                        masks.append(m)
                        labels.append((rname, key, split, sl))
        X, M = np.column_stack(cols), np.column_stack(masks)
        point, boots = bootstrap(X, M, args.boot, args.seed)
        col = {lab: j for j, lab in enumerate(labels)}
        metrics: dict = {}
        for (r, key, split, sl), j in col.items():
            name = "precision@25_normalised" if r == "oracle" else "precision@25"
            metrics.setdefault(r, {}).setdefault(split, {}).setdefault(sl, {})[f"{key}/{name}"] = summary(point[j], boots[:, j], M[:, j].sum())
        n_test = int((evaluated & (data.split == "test")).sum())
        bars = {}
        for r in [x for x in rankers if x != "community_popularity"]:
            jb, jr = col[("community_popularity", "follow/all", "test", "all")], col[(r, "follow/all", "test", "all")]
            ratio, rb = float(point[jr] / point[jb]), boots[:, jr] / boots[:, jb]
            bars[r] = {"ratio_vs_community_popularity": ratio, "ci": ci(rb), "level": 0.95,
                       "pass": bool(ratio >= M2_BAR_RATIO and ci(rb)[0] > 1.0 and n_test >= M2_POWER_FLOOR),
                       "bar": "M2-test precision@25 >= 1.2x community popularity, 95% CI lower > 1.0 (plan §3.1 S16)"}
        report["arms"][part.name] = {
            "users_with_picks": int(has_pick.sum()), "users_evaluated": int(evaluated.sum()), "test_users_evaluated": n_test,
            "power_floor": {"test_users": n_test, "floor": M2_POWER_FLOOR, "met": n_test >= M2_POWER_FLOOR,
                            "if_not_met": "M2 is diagnostic only; the S16 bar moves to M1-follow on users with <= 20 follows at T (slice follows_le20)"},
            "cards": int(card_ok.sum()), "flagged_communities": int(part.flagged.sum()),
            "fill_rule_accounts": fill_counts,
            "self_checks": {"oracle_normalised_precision_is_1": bool(oracle_ok)},
            "metrics": metrics, "bars": bars,
        }
    return finish(report)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("m1", "m2"):
        s = sub.add_parser(name)
        s.add_argument("--universe", type=pathlib.Path, required=True)
        s.add_argument("--arm", action="append", required=True, help="NAME=DIR of a train-only partition")
        s.add_argument("--exclude-dids", action="append", default=[])
        s.add_argument("--boot", type=int, default=BOOT_N)
        s.add_argument("--seed", type=int, default=BOOT_SEED, help="bootstrap seed; tie-breaks use [seed, 1]")
        s.add_argument("--clickhouse", default="clickhouse")
        s.add_argument("--out", type=pathlib.Path, required=True)
        if name == "m1":
            s.add_argument("--m1-dir", type=pathlib.Path, required=True)
            s.add_argument("--m1-mod", type=int, default=M1_HASH[0])
            s.add_argument("--item-sim", action="append", default=[])
            s.add_argument("--compare", action="append", default=[])
            s.add_argument("--freeze", default=None, help="BASE:CHALLENGER, e.g. cand1:cand5")
            s.add_argument("--b1a-lists", type=pathlib.Path)
            s.add_argument("--full-producers", type=pathlib.Path)
            s.add_argument("--workers", type=int, default=1)
        else:
            s.add_argument("--m2-dir", type=pathlib.Path, required=True)
            s.add_argument("--flags", action="append", default=[])
            s.add_argument("--ranker", action="append", default=[])
            s.add_argument("--supplement-dids")
    args = ap.parse_args()
    report = run_m1(args) if args.cmd == "m1" else run_m2(args)
    args.out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in report if k in ("users", "cohort", "self_checks", "freeze", "metrics_sha256")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
