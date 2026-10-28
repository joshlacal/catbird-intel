"""Tests for eval_temporal.py (M1/M2 at T) and s3_prep.py on hand-built fixtures.

Every expected number below is worked out by hand in the comments; none comes from the code.
"""
import contextlib
import io
import json
import math
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import eval_temporal as et
import s3_prep

# T here is the first post-T instant: S1 R1 makes pre-T mean TID < 2026-08-16T00:00Z.
T, DAY = et.TRAIN_END_US, et.DAY_US
# Universe at T: pid, did, followers. g is excluded (hide-from-recs).
PRODUCERS = [("a", 1000), ("b", 900), ("c", 800), ("d", 700), ("e", 600), ("f", 500), ("g", 400), ("h", 300)]
P = {name: i for i, (name, _n) in enumerate(PRODUCERS)}
# Arm x KnownFor (did, community, score, rank). b and e belong to two communities.
KF_X = [("a", 0, 1.0, 0), ("b", 0, 0.8, 0), ("b", 1, 0.2, 1), ("c", 0, 0.6, 0), ("d", 1, 1.0, 0),
        ("e", 1, 0.5, 0), ("e", 2, 0.3, 1), ("f", 2, 1.0, 0), ("g", 0, 0.9, 0), ("h", 2, 0.7, 0)]
# Arm y puts a..d in one community.
KF_Y = [("a", 0, 1.0, 0), ("b", 0, 1.0, 0), ("c", 0, 1.0, 0), ("d", 0, 1.0, 0), ("e", 1, 1.0, 0),
        ("f", 1, 1.0, 0), ("g", 1, 1.0, 0), ("h", 1, 1.0, 0)]


def did(name):
    return f"did:p:{name}"


def fake_hash(values, mod, salt="", clickhouse="clickhouse"):
    """M1 users all sit in the M1 residue; every M2 user hashes to test (1)."""
    return np.full(len(values), 1 if salt == "m2" else 13, dtype=np.int64)


def write_producers(path, rows=PRODUCERS):
    path.write_text("".join(f"{i}\t{did(n)}\t{f}\n" for i, (n, f) in enumerate(rows)))


def write_partition(d, kf, communities=None, membership=None):
    d.mkdir(parents=True, exist_ok=True)
    write_producers(d / "producers.tsv")
    pq.write_table(pa.table({
        "pid": pa.array([P[n] for n, *_ in kf], pa.uint32()), "did": [did(n) for n, *_ in kf],
        "community": pa.array([c for _, c, _, _ in kf], pa.uint32()), "score": pa.array([s for *_, s, _ in kf], pa.float32()),
        "rank": pa.array([r for *_, r in kf], pa.uint8())}), d / "known_for.parquet")
    if communities is not None:
        pq.write_table(pa.table({"community": pa.array([c for c, _ in communities], pa.uint32()),
                                 "size": pa.array([len(tops) for _, tops in communities], pa.uint32()),
                                 "top_dids": [[did(x) for x in tops] for _, tops in communities]}), d / "communities.parquet")
    if membership is not None:
        np.save(d / "membership.npy", np.array(membership, dtype=np.int64))


def pairs_table(rows, extra=()):
    cols = {"did": [r[0] for r in rows], "subject": [r[1] for r in rows],
            "created_us": pa.array([r[2] for r in rows], pa.int64())}
    for j, name in enumerate(extra):
        cols[name] = [r[3 + j] for r in rows]
    return pa.table(cols)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="eval-temporal-"))
        self.uni_dir = self.dir / "universe"
        self.uni_dir.mkdir()
        write_producers(self.uni_dir / "producers.tsv")
        # Median community size: x 2 (sizes 4, 2, 2), y 4 (sizes 4, 4), so granularity is not matched.
        write_partition(self.dir / "x", KF_X, [(0, "abcg"), (1, "de"), (2, "fh")])
        write_partition(self.dir / "y", KF_Y, [(0, "abcd"), (1, "efgh")])
        (self.dir / "exclude.txt").write_text(did("g") + "\ndid:u:excl\n")
        self.patch = mock.patch.object(s3_prep, "city_hash_mod", fake_hash)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write_m1(self, train=None):
        m = self.dir / "m1"
        m.mkdir(exist_ok=True)
        pq.write_table(pa.table({"did": ["did:u:1", "did:u:2", "did:u:excl"], "lang": ["en", "ja", "en"]}), m / "m1_users.parquet")
        train = train or [("did:u:1", did("a"), T - 5 * DAY), ("did:u:1", did("d"), T - 1), ("did:u:2", did("f"), T - DAY)]
        pq.write_table(pairs_table(train), m / "m1_train_follows.parquet")
        test = [
            ("did:u:1", did("e"), T + DAY, 1, True),        # primary
            ("did:u:1", did("e"), T + 5 * DAY, 1, True),    # duplicate, deduped
            ("did:u:1", did("h"), T + 2 * DAY, 1, True),    # primary, zero InterestedIn score
            ("did:u:1", did("c"), T + DAY, 12, True),       # burst of >= 10: dropped
            ("did:u:1", did("b"), T + 3 * DAY, 1, False),   # not live at the snapshot: dropped
            ("did:u:1", did("a"), T + DAY, 1, True),        # already followed at T: dropped
            ("did:u:1", did("g"), T + DAY, 1, True),        # excluded producer: dropped
            ("did:u:1", did("c"), T - DAY, 1, True),        # before T: outside the window
            ("did:u:2", did("h"), T + DAY, 1, True),        # primary
            ("did:u:2", did("b"), et.M1_END_US + DAY, 1, True),  # after Sep 12: outside the window
        ]
        pq.write_table(pairs_table(test, ("burst", "live")), m / "m1_test_follows.parquet")
        eng = [
            ("did:u:1", did("e"), T + 2 * DAY, "like"),     # after the T+1d follow of e: secondary
            ("did:u:1", did("h"), T + DAY, "like"),         # before the follow of h: not secondary
            ("did:u:1", did("f"), T + DAY, "like"),         # f on two days after T, none before: discovery
            ("did:u:1", did("f"), T + 3 * DAY, "repost"),
            ("did:u:1", did("c"), T + DAY, "like"),         # c on two days after T but also before T
            ("did:u:1", did("c"), T + 4 * DAY, "like"),
            ("did:u:1", did("c"), T - 10 * DAY, "like"),
            ("did:u:2", did("b"), T + DAY, "like"),         # one day only
            ("did:u:2", did("b"), T + DAY + 5, "view"),     # not a qualifying kind
        ]
        pq.write_table(pairs_table(eng, ("kind",)), m / "m1_engagement.parquet")
        return m

    def run_cli(self, argv):
        out = self.dir / "out.json"
        with mock.patch.object(sys, "argv", ["eval_temporal.py"] + argv + ["--out", str(out)]), \
                contextlib.redirect_stdout(io.StringIO()):
            et.main()
        return json.loads(out.read_text())


class CsrAndRanking(Fixture):
    def test_group_and_gather(self):
        off, order = et.group(np.array([2, 0, 2, 1]), 3)
        self.assertEqual(off.tolist(), [0, 1, 2, 4])
        self.assertEqual(order.tolist(), [1, 3, 0, 2])
        self.assertEqual(et.gather(off, np.array([2, 0])).tolist(), [2, 3, 0])
        self.assertEqual(et.gather(off, np.array([], dtype=np.int64)).tolist(), [])

    def test_hit_ranks_ignore_padding(self):
        top = np.array([[3, 1, -1], [7, -1, -1]])
        # Item 7 is the padded placeholder's neighbour; padding must never match it.
        ranks = et.hit_ranks(top, np.array([0, 0, 0, 1, 1]), np.array([1, 3, 7, 7, 2]), 8)
        self.assertEqual(ranks.tolist(), [1, 0, -1, 0, -1])

    def test_user_metrics_hand_computed(self):
        # One user, positives at ranks 2 and 60, a third never ranked.
        m = et.user_metrics(np.array([2, 60, -1]), np.array([0, 0, 0]), 1)
        self.assertAlmostEqual(m["recall@10"][0], 1 / 3)
        self.assertAlmostEqual(m["recall@50"][0], 1 / 3)
        self.assertAlmostEqual(m["recall@100"][0], 2 / 3)
        ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(m["ndcg@100"][0], (1 / math.log2(4) + 1 / math.log2(62)) / ideal)

    def test_interestedin_knownfor_ranking_hand_computed(self):
        uni = et.Universe(self.uni_dir / "producers.tsv", {did("g")}, 42)
        x = et.Partition("x", self.dir / "x", uni)
        data = mock.Mock()
        data.follows = lambda u: np.array([P["a"], P["d"]])
        data.self_idx = np.array([-1])
        et._STATE.update(uni=uni, data=data, arms=[x], sims={}, seed=42, k=100,
                         oracle=(np.array([0]), np.array([P["e"]])))
        top = et.rank_user(0)
        # u follows a (c0: 1.0) and d (c1: 1.0): b = 0.8 + 0.2 = 1.0, c = 0.6, e = 0.5 + 0 (c2 unseen);
        # g is excluded, a and d are followed, f and h score 0 and follow the seeded tie-break order.
        zero = sorted([P["f"], P["h"]], key=lambda p: uni.tiebreak[p])
        self.assertEqual(top["x"].tolist(), [P["b"], P["c"], P["e"]] + zero)
        self.assertEqual(top["popularity"].tolist(), [P["b"], P["c"], P["e"], P["f"], P["h"]])
        self.assertEqual(sorted(top["random"].tolist()), sorted([P["b"], P["c"], P["e"], P["f"], P["h"]]))
        self.assertEqual(top["oracle"][0], P["e"])

    def test_bootstrap_masked_means(self):
        X = np.array([[1.0, 5.0], [3.0, 7.0], [2.0, 9.0]])
        M = np.array([[True, True], [True, False], [False, True]])
        point, boots = et.bootstrap(X, M, 50, 1)
        self.assertEqual(point.tolist(), [2.0, 7.0])
        const, cb = et.bootstrap(np.full((4, 1), 0.25), np.ones((4, 1), bool), 20, 1)
        self.assertEqual(et.ci(cb[:, 0]), [0.25, 0.25])


class M1(Fixture):
    def test_positive_sets_follow_the_plan_definitions(self):
        uni = et.Universe(self.uni_dir / "producers.tsv", {did("g"), "did:u:excl"}, 42)
        data = et.M1Data(self.write_m1(), uni, {did("g"), "did:u:excl"}, 100, "clickhouse")
        self.assertEqual(data.dids, ["did:u:1", "did:u:2"])
        pairs = lambda s: sorted((data.dids[u], uni.dids[i]) for u, i in zip(*data.sets[s]))
        self.assertEqual(pairs("primary"), [("did:u:1", did("e")), ("did:u:1", did("h")), ("did:u:2", did("h"))])
        self.assertEqual(pairs("secondary"), [("did:u:1", did("e"))])
        self.assertEqual(pairs("discovery"), [("did:u:1", did("f"))])
        self.assertEqual(data.test_rows_outside_window, 2)
        self.assertEqual(data.test_follows_outside_universe, 0)
        self.assertEqual(data.degree().tolist(), [2, 1])

    def test_end_to_end_report_hand_computed(self):
        m1 = self.write_m1()
        lists = self.dir / "b1a.tsv"
        a, b, c, d = (did(n) for n in "abcd")
        g = did("g")
        lists.write_text(f"at://o1/l/1\tdid:o:1\tn\t5\t{a},{b},{c},{d},{g}\n"
                         f"at://o2/l/1\tdid:o:2\tn\t5\t{g},{d},{c},{b},{a}\n"   # same member set: deduped
                         f"at://o3/l/1\tdid:o:3\tn\t2\t{a},{b}\n"               # < 3 members: dropped
                         f"at://o4/l/1\tdid:o:4\tn\t501\t{a},{b},{c}\n")        # > 500 items: dropped
        argv = ["m1", "--universe", str(self.uni_dir), "--m1-dir", str(m1), "--arm", f"x={self.dir / 'x'}",
                "--arm", f"y={self.dir / 'y'}", "--exclude-dids", str(self.dir / "exclude.txt"), "--boot", "200",
                "--compare", "popularity:x", "--freeze", "x:y", "--b1a-lists", str(lists)]
        r = self.run_cli(argv)
        self.assertEqual(r["users"], {"m1_users": 2, "with_primary_positive": 2, "with_secondary_positive": 1,
                                      "with_discovery_positive": 1})
        uni = et.Universe(self.uni_dir / "producers.tsv", {did("g")}, et.BOOT_SEED)
        # u1's top list in x is b, c, e, then f and h in tie-break order: e at rank 2, h at 3 or 4.
        rh = 3 if uni.tiebreak[P["h"]] < uni.tiebreak[P["f"]] else 4
        ideal = 1 + 1 / math.log2(3)
        u1 = (1 / math.log2(4) + 1 / math.log2(rh + 2)) / ideal
        # u2 follows f (c2): h = 0.7 tops the list, so NDCG = 1.
        x = r["metrics"]["primary"]["x"]["all"]
        self.assertAlmostEqual(x["ndcg@100"]["mean"], (u1 + 1.0) / 2)
        self.assertEqual(x["recall@10"]["mean"], 1.0)
        # Popularity for u1: b, c, e, f, h (h at rank 4); for u2: a, b, c, d, e, h (h at rank 5).
        pop = r["metrics"]["primary"]["popularity"]["all"]["ndcg@100"]["mean"]
        expected = ((1 / math.log2(4) + 1 / math.log2(6)) / ideal + 1 / math.log2(7)) / 2
        self.assertAlmostEqual(pop, expected)
        self.assertTrue(all(r["self_checks"]["oracle"].values()))
        self.assertTrue(all(v["pass"] for v in r["self_checks"]["random_vs_base_rate"].values()))
        # B1a: in x, a, b, c share c0 and d is c1 -> 3 of 6 pairs; in y all four share c0 -> 1.
        fz = r["freeze"]
        self.assertEqual(fz["b1a_macro_top1"]["x"], 0.5)
        self.assertEqual(fz["b1a_macro_top1"]["y"], 1.0)
        self.assertEqual(fz["b1a_macro_top1"]["lists"], 1)
        self.assertTrue(fz["b1a_pass"])   # one list, so the 98.75% CI is [0.5, 0.5]
        # Null: x primaries over eligible producers are c0 x3, c1 x2, c2 x2 -> 10/42 same-primary pairs;
        # y has c0 x4, c1 x3 -> 18/42.
        self.assertAlmostEqual(fz["b1a_macro_top1"]["lift_over_null"]["x"], 0.5 / (10 / 42))
        self.assertAlmostEqual(fz["b1a_macro_top1"]["lift_over_null"]["y"], 1.0 / (18 / 42))
        self.assertEqual(fz["median_community_size"]["x"], 2.0)
        self.assertFalse(fz["granularity_matched"])
        # Recall@50 is 1.0 in both arms, so +5% is impossible and cand1's role (x) is kept.
        self.assertEqual(fz["m1_follow_recall@50"]["rel"], 0.0)
        self.assertFalse(fz["m1_pass"])
        self.assertEqual(fz["decision"], "x")
        self.assertIn("lang=en", r["metrics"]["primary"]["x"])
        self.assertIn("follower_bucket=0|all", r["metrics"]["primary"]["x"])
        # Same seed, same metrics; a forked worker pool gives the same bytes.
        again = self.run_cli(argv + ["--workers", "2"])
        self.assertEqual(again["metrics_sha256"], r["metrics_sha256"])

    def test_freeze_decision_truth_table(self):
        decide = lambda *a: et.freeze_decision("c1", "c5", *a)["decision"]
        self.assertEqual(decide(0.06, [0.01, 0.1], [0.0, 0.02], True), "c5")
        self.assertEqual(decide(0.04, [0.01, 0.1], [0.0, 0.02], True), "c1")    # below +5%
        self.assertEqual(decide(0.06, [-0.01, 0.1], [0.0, 0.02], True), "c1")   # M1 CI includes 0
        # R14: a B1a CI that merely includes 0 is not enough; its lower bound must be >= 0.
        self.assertEqual(decide(0.06, [0.01, 0.1], [-0.01, 0.01], True), "c1")
        self.assertEqual(decide(0.06, [0.01, 0.1], [0.0, 0.02], False), "c1")   # granularity not matched
        self.assertIsNone(et.freeze_decision("c1", "c5", 0.06, [0.01, 0.1], None, True)["b1a_pass"])

    def test_leakage_guards(self):
        uni_args = ["--universe", str(self.uni_dir), "--exclude-dids", str(self.dir / "exclude.txt"), "--boot", "10"]
        leaky = self.write_m1(train=[("did:u:1", did("a"), T)])
        with self.assertRaisesRegex(SystemExit, "on or after 2026-08-16"):
            self.run_cli(["m1", "--m1-dir", str(leaky), "--arm", f"x={self.dir / 'x'}"] + uni_args)
        m1 = self.write_m1()
        full = self.dir / "full"
        write_partition(full, KF_X)
        (full / "producers.tsv").write_text((full / "producers.tsv").read_text().replace("\t300\n", "\t301\n"))
        with self.assertRaisesRegex(SystemExit, "differs from the train-only universe"):
            self.run_cli(["m1", "--m1-dir", str(m1), "--arm", f"full={full}"] + uni_args)
        with self.assertRaisesRegex(SystemExit, "full-data producers"):
            self.run_cli(["m1", "--m1-dir", str(m1), "--arm", f"x={self.dir / 'x'}",
                          "--full-producers", str(self.uni_dir / "producers.tsv")] + uni_args)
        with mock.patch.object(s3_prep, "city_hash_mod", lambda v, m, **k: np.full(len(v), 12)):
            with self.assertRaisesRegex(SystemExit, "sample guard"):
                self.run_cli(["m1", "--m1-dir", str(m1), "--arm", f"x={self.dir / 'x'}"] + uni_args)


class MembershipTrap(Fixture):
    def trap_partition(self, name, communities):
        # Raw Leiden labels: {a, b} = 0, {c, d} = 1, everything else 2.
        d = self.dir / name
        write_partition(d, KF_X, communities, [0, 0, 1, 1, 2, 2, 2, 2])
        return d

    def test_flags_keyed_by_raw_label_land_on_the_right_community(self):
        # Equal sizes: cluster.py numbered raw label 1 as community 0 and raw label 0 as community 1.
        d = self.trap_partition("trap", [(0, "cd"), (1, "ab"), (2, "efh")])
        np.savez(d / "flags.npz", follow_back=np.array([True, False, False]), follow_train=np.zeros(3, bool))
        uni = et.Universe(self.uni_dir / "producers.tsv", set(), 42)
        part = et.Partition("trap", d, uni, d / "flags.npz")
        # Raw label 0 ({a, b}) is flagged, which is community 1; a naive join would flag community 0.
        self.assertEqual(part.flagged.tolist(), [False, True, False])

    def test_mapping_rejects_ambiguous_or_colliding_labels(self):
        pid = {did(n): i for i, (n, _f) in enumerate(PRODUCERS)}
        memb = np.array([0, 0, 1, 1, 2, 2, 2, 2])
        tie = pa.table({"community": [0], "top_dids": [[did("a"), did("c")]]})
        with self.assertRaisesRegex(ValueError, "strict-majority"):
            s3_prep.raw_label_by_community(tie, memb, pid)
        dup = pa.table({"community": [0, 1], "top_dids": [[did("a")], [did("b")]]})
        with self.assertRaisesRegex(ValueError, "same raw label"):
            s3_prep.raw_label_by_community(dup, memb, pid)
        ok = pa.table({"community": [0, 1], "top_dids": [[did("c"), did("d"), did("a")], [did("a"), did("b")]]})
        self.assertEqual(s3_prep.raw_label_by_community(ok, memb, pid), {0: 1, 1: 0})

    def test_repro_compares_communities_through_raw_labels(self):
        ref = self.trap_partition("ref", [(0, "ab"), (1, "cd"), (2, "efh")])
        same = self.trap_partition("same", [(0, "cd"), (1, "ab"), (2, "efh")])
        r = s3_prep.repro(ref, same)
        self.assertTrue(r["identical_membership"] and r["pass"])
        self.assertEqual(r["communities_same_members"], 3)
        self.assertEqual(r["communities_same_members_and_same_id"], 1)   # only community 2 kept its id
        relabel = self.dir / "relabel"
        write_partition(relabel, KF_X, [(0, "ab"), (1, "cd"), (2, "efh")], [2, 2, 0, 0, 1, 1, 1, 1])
        r = s3_prep.repro(ref, relabel)
        self.assertFalse(r["identical_membership"])
        self.assertAlmostEqual(r["nmi_arithmetic"], 1.0)
        self.assertTrue(r["pass"])
        self.assertEqual(r["communities_same_members"], 3)

    def test_nmi_hand_computed(self):
        # a = [0,0,1,1], b = [0,0,0,1]: H(a) = ln 2, H(b) = -(.75 ln .75 + .25 ln .25),
        # H(a,b) = -(.5 ln .5 + 2 * .25 ln .25); NMI = (H(a) + H(b) - H(a,b)) / mean(H(a), H(b)).
        ha, hb = math.log(2), -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
        hab = -(0.5 * math.log(0.5) + 0.5 * math.log(0.25))
        self.assertAlmostEqual(s3_prep.nmi([0, 0, 1, 1], [0, 0, 0, 1]), (ha + hb - hab) / ((ha + hb) / 2))
        self.assertAlmostEqual(s3_prep.nmi([5, 5, 9], [1, 1, 0]), 1.0)


class M2(Fixture):
    def setUp(self):
        super().setUp()
        for name, value in (("BUNDLE", 4), ("CARD_MIN", 1)):
            p = mock.patch.object(et, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_allocation_and_bundle_rules(self):
        self.assertEqual(et.allocate(3, 25), [9, 8, 8])
        self.assertEqual(et.allocate(2, 25), [13, 12])
        self.assertEqual(et.allocate(1, 25), [25])
        uni = et.Universe(self.uni_dir / "producers.tsv", {did("g")}, 42)
        x = et.Partition("x", self.dir / "x", uni)
        off, ranked = et.card_rankings(x, uni, {"pop": uni.followers.astype(float)})
        # Picks c0, c1 with a, b already followed: c0 gives c only, c1 gives d, e; the shortfall is
        # filled from c2 (e is in c1 and c2), whose most-followed member is f.
        bundle, filled = et.build_bundle([0, 1], x, off, ranked["pop"], uni, {P["a"], P["b"]})
        self.assertEqual(bundle, [P["c"], P["d"], P["e"], P["f"]])
        self.assertEqual(filled, 1)
        x.flagged[2] = True    # a flagged neighbour is never used to fill
        self.assertEqual(et.build_bundle([0, 1], x, off, ranked["pop"], uni, {P["a"], P["b"]})[0],
                         [P["c"], P["d"], P["e"]])
        x.flagged[2] = False
        with mock.patch.object(et, "POP_CEILING", (850, 1)):
            # a (1000) and b (900) are both over 850; only one may enter, so c (800) follows a.
            self.assertEqual(et.build_bundle([0], x, off, ranked["pop"], uni, set(), size=2)[0], [P["a"], P["c"]])

    def test_end_to_end_hand_computed(self):
        m = self.dir / "m2"
        m.mkdir()
        c1, c2 = T + 3_600_000_000, T + 2 * DAY
        pq.write_table(pa.table({"did": ["did:n:1", "did:n:2", "did:n:old", "did:u:excl"],
                                 "created_us": pa.array([c1, c2, T - 5 * DAY, c1], pa.int64())}), m / "m2_cohort.parquet")
        follows = [
            ("did:n:1", did("a"), c1 + 1, 1, True),             # day 1 seed
            ("did:n:1", did("b"), c1 + 2, 11, True),            # day 1 seed in a burst: pack-burst user
            ("did:n:1", did("e"), c1 + 3 * DAY, 1, True),       # outcome
            ("did:n:1", did("h"), c1 + 4 * DAY, 1, True),       # outcome
            ("did:n:1", did("c"), c1 + 5 * DAY, 12, True),      # burst: not an outcome
            ("did:n:1", did("f"), c1 + 40 * DAY, 1, True),      # after day 30
            ("did:n:1", did("a"), c1 + 6 * DAY, 1, True),       # already a seed
            ("did:n:2", did("f"), c2 + 10, 1, True),            # day 1 seed
            ("did:n:2", did("h"), c2 + 2 * DAY, 1, True),       # outcome
            ("did:n:2", did("d"), c2 + 3 * DAY, 1, False),      # not live
        ]
        pq.write_table(pairs_table(follows, ("burst", "live")), m / "m2_follows.parquet")
        pq.write_table(pairs_table([("did:n:1", did("e"), c1 + 5 * DAY, "like")], ("kind",)), m / "m2_engagement.parquet")
        r = self.run_cli(["m2", "--universe", str(self.uni_dir), "--m2-dir", str(m), "--arm", f"x={self.dir / 'x'}",
                          "--exclude-dids", str(self.dir / "exclude.txt"), "--boot", "100"])
        self.assertEqual(r["cohort"], {"users": 2, "excluded": 1, "created_outside_window": 1, "val": 0, "test": 2})
        x = r["arms"]["x"]
        self.assertEqual(x["users_evaluated"], 2)
        # n1: seeds a (c0 1.0) and b (c0 0.8, c1 0.2) -> picks c0, c1 -> bundle c, d, e + fill f; hits {e}.
        # n2: seed f (c2) -> pick c2 -> h, then fill from c1 (shares e) -> d, e; hits {h}. Both 1/4.
        for ranker in ("community_popularity", "knownfor"):
            self.assertEqual(x["metrics"][ranker]["test"]["all"]["follow/all/precision@25"]["mean"], 0.25)
        self.assertEqual(x["metrics"]["community_popularity"]["test"]["all"]["engaged/all/precision@25"]["mean"], 0.125)
        self.assertEqual(x["metrics"]["community_popularity"]["test"]["pack_burst"]["follow/all/precision@25"]["n"], 1)
        self.assertEqual(x["fill_rule_accounts"]["community_popularity"], 3)
        self.assertTrue(x["self_checks"]["oracle_normalised_precision_is_1"])
        bar = x["bars"]["knownfor"]
        self.assertEqual(bar["ratio_vs_community_popularity"], 1.0)
        self.assertFalse(bar["pass"])
        self.assertFalse(x["power_floor"]["met"])


class S3Prep(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(tempfile.mkdtemp(prefix="s3-prep-"))

    @unittest.skipUnless(shutil.which("clickhouse"), "needs the clickhouse binary")
    def test_city_hash_matches_clickhouse(self):
        # Values printed by `clickhouse local` 26.9 for cityHash64(did) % 8.
        self.assertEqual(s3_prep.city_hash_mod(["did:plc:abc", "did:plc:xyz"], 8).tolist(), [2, 4])

    def test_split_lists_by_owner_class(self):
        tsv = self.dir / "curatelists.tsv"
        tsv.write_text("at://1\tdid:o:a\tx\t5\tdid:p:1\nat://2\tdid:o:b\ty\t5\tdid:p:2\n"
                       "at://3\tdid:o:c\tz\t5\tdid:p:3\nat://4\tdid:o:a\tw\t5\tdid:p:4\n")
        classes = {"did:o:a": 2, "did:o:b": 6, "did:o:c": 5}
        with mock.patch.object(s3_prep, "city_hash_mod", lambda v, m, **k: np.array([classes[x] for x in v])):
            receipt = s3_prep.split_lists(tsv, self.dir / "out")
        self.assertEqual(receipt["lists"], {"A": 1, "B1a": 2, "B1b": 1, "B2": 0})
        self.assertEqual((self.dir / "out" / "b1_owners.txt").read_text(), "did:o:a\ndid:o:b\n")
        self.assertEqual((self.dir / "out" / "lists_B1a.tsv").read_text().count("\n"), 2)

    def test_extract_knn_top_k_per_list_member(self):
        write_producers(self.dir / "producers.tsv")
        rows = [(0, 1, 0.9), (0, 2, 0.5), (0, 3, 0.7), (2, 0, 0.4), (5, 0, 0.3)]
        np.array(rows, dtype=s3_prep.SIM_DTYPE).tofile(self.dir / "sim.bin")
        lists = self.dir / "lists.tsv"
        lists.write_text(f"at://1\tdid:o:a\tx\t3\t{did('a')},{did('c')},did:not:producer\n")
        receipt = s3_prep.extract_knn(self.dir / "sim.bin", self.dir / "producers.tsv", [lists], self.dir / "knn.parquet", k=2)
        t = pq.read_table(self.dir / "knn.parquet").to_pylist()
        self.assertEqual([(r["did"], r["rank"], r["nbr_did"]) for r in t],
                         [(did("a"), 0, did("b")), (did("a"), 1, did("d")), (did("c"), 0, did("a"))])
        self.assertEqual(receipt["list_member_producers"], 2)


if __name__ == "__main__":
    unittest.main()
