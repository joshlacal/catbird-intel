"""Tests for labeling.agreement (stdlib unittest)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from labeling.agreement import (
    GOLD_SCHEMA,
    GoldFormatError,
    binarize,
    bootstrap_kappa_ci,
    cohen_kappa,
    compare_axis,
    confusion_matrix,
    load_gold,
    resolve_session_file,
)


# Row shapes mirror labeling/store.py: every row carries "key" (post uri, or
# pair_id for trap sessions), "uri" and "position".


def label_row(row_id, key, axis, value, ts, session_id, *, uri=None, **extra):
    row = {
        "schema": GOLD_SCHEMA,
        "op": "label",
        "row_id": row_id,
        "key": key,
        "uri": uri or key,
        "axis": axis,
        "ts": ts,
        "session_id": session_id,
        "position": row_id,
        "value": value,
        "ms_to_decide": 900,
        "note_ids": [],
    }
    if axis == "trap":
        row["pair_id"] = key
    row.update(extra)
    return row


def view_row(row_id, key, ts, session_id):
    return {
        "schema": GOLD_SCHEMA,
        "op": "view",
        "row_id": row_id,
        "key": key,
        "uri": key,
        "axis": "narrate",
        "ts": ts,
        "session_id": session_id,
        "position": row_id,
        "ms_on_card": 4000,
        "note_ids": ["n1"],
    }


def undo_row(row_id, undoes, key, axis, ts, session_id):
    row = {
        "schema": GOLD_SCHEMA,
        "op": "undo",
        "row_id": row_id,
        "key": key,
        "uri": key,
        "axis": axis,
        "ts": ts,
        "session_id": session_id,
        "position": 0,
        "undoes_row_id": undoes,
    }
    if axis == "trap":
        row["pair_id"] = key
    return row


class GoldFileMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, axis, session_id, rows):
        path = self.dir / f"labels-{axis}-{session_id}.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path


class LoadGoldTests(GoldFileMixin, unittest.TestCase):
    def test_undo_revokes_and_last_label_wins(self):
        s = "s1"
        path = self.write("outraged", s, [
            label_row(1, "at://a", "outraged", 2, "2026-10-23T10:00:00Z", s),
            label_row(2, "at://b", "outraged", 1, "2026-10-23T10:00:01Z", s),
            label_row(3, "at://a", "outraged", 3, "2026-10-23T10:00:02Z", s),
            undo_row(4, 3, "at://a", "outraged", "2026-10-23T10:00:03Z", s),
            undo_row(5, 2, "at://b", "outraged", "2026-10-23T10:00:04Z", s),
            label_row(6, "at://c", "outraged", "?", "2026-10-23T10:00:05Z", s),
            label_row(7, "at://c", "outraged", 0, "2026-10-23T10:00:06Z", s),
        ])
        gold = load_gold([path])
        # a: row 3 undone -> falls back to row 1; b: only label undone -> absent;
        # c: last label wins.
        self.assertEqual(gold.values("outraged"), {"at://a": 2, "at://c": 0})
        self.assertEqual(gold.labels["outraged"]["at://a"].row_id, 1)
        expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(gold.file_sha256, {str(path): expected_sha})
        self.assertEqual(gold.repeats["outraged"].n_repeated, 0)

    def test_multi_session_latest_ts_and_intra_rater_disagreement(self):
        p1 = self.write("spam", "early", [
            label_row(1, "at://a", "spam", 1, "2026-10-20T09:00:00Z", "early"),
            label_row(2, "at://b", "spam", 0, "2026-10-20T09:00:01Z", "early"),
            label_row(3, "at://c", "spam", "?", "2026-10-20T09:00:02Z", "early"),
            label_row(4, "at://only-early", "spam", 1, "2026-10-20T09:00:03Z", "early"),
        ])
        p2 = self.write("spam", "late", [
            label_row(1, "at://a", "spam", 0, "2026-10-22T09:00:00+00:00", "late"),
            label_row(2, "at://b", "spam", 0, "2026-10-22T09:00:01+00:00", "late"),
            label_row(3, "at://c", "spam", 1, "2026-10-22T09:00:02+00:00", "late"),
        ])
        # Path order must not matter.
        gold = load_gold([p2, p1])
        self.assertEqual(
            gold.values("spam"),
            {"at://a": 0, "at://b": 0, "at://c": 1, "at://only-early": 1},
        )
        self.assertEqual(gold.labels["spam"]["at://a"].session_id, "late")
        repeats = gold.repeats["spam"]
        self.assertEqual(repeats.n_repeated, 3)
        self.assertEqual(repeats.n_disagreed, 2)
        self.assertEqual(repeats.disagreed_keys, ("at://a", "at://c"))
        self.assertEqual(len(gold.file_sha256), 2)

    def test_latest_ts_uses_time_not_session_order(self):
        # "zzz" sorts after "aaa" but was labelled earlier.
        p1 = self.write("critical", "zzz", [
            label_row(1, "at://a", "critical", 3, "2026-10-01T00:00:00Z", "zzz"),
        ])
        p2 = self.write("critical", "aaa", [
            label_row(1, "at://a", "critical", 1, "2026-10-02T00:00:00Z", "aaa"),
        ])
        gold = load_gold([p1, p2])
        self.assertEqual(gold.values("critical"), {"at://a": 1})

    def test_trap_rows_keyed_by_pair_id(self):
        s = "t1"
        path = self.write("trap", s, [
            label_row(1, "pair-1", "trap", 2, "2026-10-23T10:00:00Z", s, uri="at://x"),
            label_row(2, "pair-2", "trap", 0, "2026-10-23T10:00:01Z", s, uri="at://y"),
            undo_row(3, 2, "pair-2", "trap", "2026-10-23T10:00:02Z", s),
            label_row(4, "pair-2", "trap", "?", "2026-10-23T10:00:03Z", s, uri="at://y"),
        ])
        other = self.write("trap", "t2", [
            label_row(1, "pair-1", "trap", 1, "2026-10-24T10:00:00Z", "t2", uri="at://x"),
        ])
        gold = load_gold([path, other])
        self.assertEqual(gold.values("trap"), {"pair-1": 1, "pair-2": "?"})
        self.assertEqual(gold.repeats["trap"].n_repeated, 1)
        self.assertEqual(gold.repeats["trap"].n_disagreed, 1)

    def test_trap_rejects_value_three_and_key_pair_id_mismatch(self):
        bad_value = self.write("trap", "t1", [
            label_row(1, "p", "trap", 3, "2026-10-23T10:00:00Z", "t1", uri="at://x"),
        ])
        with self.assertRaises(GoldFormatError):
            resolve_session_file(bad_value)
        mismatch = self.write("trap", "t2", [
            label_row(1, "p", "trap", 1, "2026-10-23T10:00:00Z", "t2", uri="at://x", pair_id="q"),
        ])
        with self.assertRaises(GoldFormatError):
            resolve_session_file(mismatch)

    def test_resolves_by_key_not_uri(self):
        s = "k"
        path = self.write("spam", s, [
            label_row(1, "at://a", "spam", 1, "2026-10-23T00:00:00Z", s, uri="at://shared"),
            label_row(2, "at://b", "spam", 0, "2026-10-23T00:00:01Z", s, uri="at://shared"),
        ])
        self.assertEqual(load_gold([path]).values("spam"), {"at://a": 1, "at://b": 0})

    def test_narrate_files_ignored_and_view_rows_undoable(self):
        s = "n1"
        narrate = self.write("narrate", s, [
            view_row(1, "at://a", "2026-10-23T00:00:00Z", s),
            view_row(2, "at://b", "2026-10-23T00:00:01Z", s),
            undo_row(3, 2, "at://b", "narrate", "2026-10-23T00:00:02Z", s),
        ])
        labels = self.write("outraged", "s1", [
            label_row(1, "at://a", "outraged", 1, "2026-10-23T00:00:00Z", "s1"),
        ])
        gold = load_gold([narrate, labels])
        self.assertEqual(set(gold.labels), {"outraged"})
        self.assertEqual(gold.ignored_files, [str(narrate)])
        self.assertEqual(list(gold.file_sha256), [str(labels)])
        self.assertEqual(resolve_session_file(narrate), ("narrate", s, {}))

    def test_torn_write_rejected(self):
        path = self.write("spam", "torn", [
            label_row(1, "u", "spam", 1, "2026-10-23T00:00:00Z", "torn"),
        ])
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"schema": "catbird-gold')
        with self.assertRaises(GoldFormatError):
            resolve_session_file(path)

    def test_format_errors(self):
        s = "s"
        cases = {
            "binary_value_2": [label_row(1, "u", "spam", 2, "2026-10-23T00:00:00Z", s)],
            "non_increasing": [
                label_row(2, "u", "spam", 1, "2026-10-23T00:00:00Z", s),
                label_row(2, "v", "spam", 1, "2026-10-23T00:00:00Z", s),
            ],
            "undo_unknown": [undo_row(1, 9, "u", "spam", "2026-10-23T00:00:00Z", s)],
            "undo_key_mismatch": [
                label_row(1, "u", "spam", 1, "2026-10-23T00:00:00Z", s),
                undo_row(2, 1, "v", "spam", "2026-10-23T00:00:00Z", s),
            ],
            "missing_key": [
                {k: v for k, v in label_row(1, "u", "spam", 1, "2026-10-23T00:00:00Z", s).items() if k != "key"}
            ],
            "label_in_narrate": [label_row(1, "u", "narrate", 1, "2026-10-23T00:00:00Z", s)],
            "double_undo": [
                label_row(1, "u", "spam", 1, "2026-10-23T00:00:00Z", s),
                undo_row(2, 1, "u", "spam", "2026-10-23T00:00:00Z", s),
                undo_row(3, 1, "u", "spam", "2026-10-23T00:00:00Z", s),
            ],
            "bad_schema": [dict(label_row(1, "u", "spam", 1, "2026-10-23T00:00:00Z", s), schema="x")],
            "bool_value": [label_row(1, "u", "spam", True, "2026-10-23T00:00:00Z", s)],
            "unknown_axis": [label_row(1, "u", "vibes", 1, "2026-10-23T00:00:00Z", s)],
        }
        for name, rows in cases.items():
            with self.subTest(name):
                path = self.write("spam", name, rows)
                with self.assertRaises(GoldFormatError):
                    resolve_session_file(path)

    def test_duplicate_session_axis_rejected(self):
        a = self.write("spam", "dup", [label_row(1, "u", "spam", 1, "2026-10-23T00:00:00Z", "dup")])
        b = self.dir / "copy.jsonl"
        b.write_bytes(a.read_bytes())
        with self.assertRaises(GoldFormatError):
            load_gold([a, b])


class KappaTests(unittest.TestCase):
    def test_sklearn_doc_example_unweighted(self):
        # sklearn.metrics.cohen_kappa_score docstring example: 0.6875.
        y1 = ["negative", "positive", "negative", "neutral", "positive"]
        y2 = ["negative", "positive", "negative", "neutral", "negative"]
        cats = ["negative", "neutral", "positive"]
        self.assertAlmostEqual(cohen_kappa(list(zip(y1, y2)), cats), 0.6875, delta=1e-9)

    def test_sklearn_confusion_matrix_example(self):
        # sklearn.metrics.confusion_matrix docstring example.
        y_true = [2, 0, 2, 2, 0, 1]
        y_pred = [0, 0, 2, 2, 0, 2]
        pairs = list(zip(y_true, y_pred))
        self.assertEqual(confusion_matrix(pairs, [0, 1, 2]), [[2, 0, 0], [0, 0, 1], [1, 0, 2]])
        # po = 4/6, pe = (2*3 + 1*0 + 3*3)/36 = 15/36 -> kappa = 9/21 = 3/7.
        self.assertAlmostEqual(cohen_kappa(pairs, [0, 1, 2]), 3 / 7, delta=1e-9)

    def test_hand_computed_unweighted_linear_quadratic(self):
        # gold marginals [.5,.25,.25], model marginals [.25,.25,.5].
        pairs = [(0, 0), (1, 2), (2, 2), (0, 1)]
        cats = [0, 1, 2]
        # Unweighted: po=.5, pe=.3125 -> .1875/.6875 = 3/11.
        self.assertAlmostEqual(cohen_kappa(pairs, cats), 3 / 11, delta=1e-9)
        # Linear: observed disagreement .25, expected .5 -> 0.5.
        self.assertAlmostEqual(cohen_kappa(pairs, cats, "linear"), 0.5, delta=1e-9)
        # Quadratic: observed .125, expected .40625 -> 1 - 8/26 = 9/13.
        self.assertAlmostEqual(cohen_kappa(pairs, cats, "quadratic"), 9 / 13, delta=1e-9)

    def test_weights_use_category_span(self):
        # Unused category 3 widens the span: linear distance 1 -> 1/3.
        pairs = [(0, 0), (1, 1), (0, 1), (1, 0)]
        k3 = cohen_kappa(pairs, [0, 1, 2, 3], "linear")
        k1 = cohen_kappa(pairs, [0, 1], "linear")
        # Both are 0 here (chance-level), but the matrix must accept unused cats.
        self.assertAlmostEqual(k3, 0.0, delta=1e-9)
        self.assertAlmostEqual(k1, 0.0, delta=1e-9)

    def test_perfect_and_opposite_agreement(self):
        cats = [0, 1, 2, 3]
        perfect = [(0, 0), (1, 1), (2, 2), (3, 3)]
        for w in (None, "linear", "quadratic"):
            self.assertAlmostEqual(cohen_kappa(perfect, cats, w), 1.0, delta=1e-12)
        # Binary total disagreement with balanced marginals -> -1.
        self.assertAlmostEqual(cohen_kappa([(0, 1), (1, 0)], [0, 1]), -1.0, delta=1e-12)

    def test_degenerate_cases(self):
        # Both raters identical, single category used -> 1.0 by convention.
        self.assertEqual(cohen_kappa([(1, 1)] * 5, [0, 1]), 1.0)
        self.assertEqual(cohen_kappa([(2, 2)] * 3, [0, 1, 2, 3], "quadratic"), 1.0)
        # Each rater constant but on different categories: pe = 0, po = 0 -> 0.
        self.assertEqual(cohen_kappa([(0, 1)] * 4, [0, 1]), 0.0)
        # No pairs -> nan.
        self.assertTrue(math.isnan(cohen_kappa([], [0, 1])))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            cohen_kappa([(0, 5)], [0, 1])
        with self.assertRaises(ValueError):
            cohen_kappa([(0, 1)], [0, 1], "cubic")


class BootstrapTests(unittest.TestCase):
    def test_deterministic_and_brackets_point(self):
        pairs = [(i % 4, (i + (i % 3 == 0)) % 4) for i in range(200)]
        a = bootstrap_kappa_ci(pairs, [0, 1, 2, 3], "quadratic", n_resamples=300, seed=7)
        b = bootstrap_kappa_ci(pairs, [0, 1, 2, 3], "quadratic", n_resamples=300, seed=7)
        c = bootstrap_kappa_ci(pairs, [0, 1, 2, 3], "quadratic", n_resamples=300, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual((a["lower"], a["upper"]), (c["lower"], c["upper"]))
        self.assertLessEqual(a["lower"], a["kappa"])
        self.assertGreaterEqual(a["upper"], a["kappa"])
        self.assertLess(a["lower"], a["upper"])

    def test_perfect_agreement_ci_is_one(self):
        pairs = [(0, 0), (1, 1)] * 20
        ci = bootstrap_kappa_ci(pairs, [0, 1], n_resamples=50, seed=1)
        self.assertEqual((ci["lower"], ci["upper"]), (1.0, 1.0))

    def test_empty(self):
        ci = bootstrap_kappa_ci([], [0, 1], n_resamples=10)
        self.assertTrue(math.isnan(ci["lower"]))
        self.assertEqual(ci["n_nan"], 10)


class CompareAxisTests(unittest.TestCase):
    def test_binarize_threshold(self):
        self.assertEqual([binarize(s) for s in range(4)], [0, 0, 1, 1])
        self.assertEqual([binarize(s, 1) for s in range(4)], [0, 1, 1, 1])
        self.assertEqual([binarize(s, 3) for s in range(4)], [0, 0, 0, 1])

    def test_binary_axis_threshold_changes_result(self):
        gold = {"a": 1, "b": 0, "c": 1, "d": 0}
        model = {"a": 3, "b": 1, "c": 1, "d": 0}
        default = compare_axis(gold, model, "spam")
        self.assertEqual(default["kappa_weights"], "unweighted")
        self.assertEqual(default["binary_threshold"], 2)
        # threshold 2: model -> a1 b0 c0 d0 -> 3/4 exact.
        self.assertEqual(default["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertAlmostEqual(default["exact_agreement"], 0.75)
        lowered = compare_axis(gold, model, "spam", binary_threshold=1)
        # threshold 1: model -> a1 b1 c1 d0.
        self.assertEqual(lowered["confusion_matrix"], [[1, 1], [0, 2]])

    def test_exclusions_and_cooccurrence(self):
        gold = {
            "a": 3, "b": 2, "c": "?", "d": "?", "e": 1, "f": 0, "g": 2,
        }
        model = {
            "a": 3, "b": "insufficient", "c": "insufficient", "d": 1, "e": 1,
            # f missing
            "g": 0, "model-only": 2,
        }
        out = compare_axis(gold, model, "politicalness")
        self.assertEqual(out["kind"], "ordinal")
        self.assertEqual(out["kappa_weights"], "quadratic")
        self.assertEqual(out["n_gold"], 7)
        self.assertEqual(out["n_compared"], 3)  # a, e, g
        self.assertEqual(out["n_gold_unsure"], 2)
        self.assertEqual(out["n_model_insufficient"], 2)
        self.assertEqual(out["n_missing_model"], 1)
        self.assertEqual(out["n_model_only"], 1)
        self.assertEqual(
            out["unsure_cooccurrence"],
            {"both": 1, "gold_unsure_only": 1, "model_insufficient_only": 1, "neither": 3},
        )
        pairs = [(3, 3), (1, 1), (2, 0)]
        self.assertAlmostEqual(
            out["kappa"], cohen_kappa(pairs, [0, 1, 2, 3], "quadratic"), delta=1e-12
        )
        self.assertEqual(sum(map(sum, out["confusion_matrix"])), 3)
        self.assertEqual(out["confusion_matrix"][2][0], 1)

    def test_per_stratum_breakdown(self):
        gold = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 0, "f": 3}
        model = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 3, "f": 0}
        strata = {"a": "news", "b": "news", "c": "news", "d": "news", "e": "rare"}
        out = compare_axis(gold, model, "critical", strata_by_uri=strata)
        self.assertEqual(set(out["strata"]), {"news", "rare", "unstratified"})
        news = out["strata"]["news"]
        self.assertEqual(news["n"], 4)
        self.assertAlmostEqual(news["kappa"], 1.0)
        self.assertAlmostEqual(news["exact_agreement"], 1.0)
        self.assertEqual(out["strata"]["rare"]["n"], 1)
        self.assertEqual(out["strata"]["rare"]["exact_agreement"], 0.0)
        self.assertEqual(out["strata"]["unstratified"]["n"], 1)
        self.assertEqual(sum(s["n"] for s in out["strata"].values()), out["n_compared"])

    def test_trap_axis_is_nominal(self):
        out = compare_axis({"p1": 2, "p2": 0, "p3": "?"}, {"p1": 2, "p2": 1, "p3": 0}, "trap")
        self.assertEqual(out["kind"], "nominal")
        self.assertEqual(out["kappa_weights"], "unweighted")
        self.assertEqual(out["n_compared"], 2)
        self.assertEqual(out["n_gold_unsure"], 1)

    def test_bootstrap_attached(self):
        gold = {str(i): i % 4 for i in range(40)}
        model = {str(i): (i + (i % 5 == 0)) % 4 for i in range(40)}
        out = compare_axis(gold, model, "outraged", bootstrap_resamples=100, bootstrap_seed=3)
        self.assertIn("kappa_ci", out)
        self.assertEqual(out["kappa_ci"]["kappa"], out["kappa"])

    def test_invalid_values_raise(self):
        with self.assertRaises(ValueError):
            compare_axis({"a": 2}, {"a": 1}, "spam")  # gold 2 invalid on binary
        with self.assertRaises(ValueError):
            compare_axis({"a": 1}, {"a": 4}, "outraged")
        with self.assertRaises(ValueError):
            compare_axis({"a": 1}, {"a": "?"}, "outraged")  # model can't say "?"
        with self.assertRaises(ValueError):
            compare_axis({"a": 1}, {"a": 1}, "vibes")


if __name__ == "__main__":
    unittest.main()
