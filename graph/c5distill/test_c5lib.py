"""c5lib checks against sklearn (run: python -m pytest graph/c5distill/test_c5lib.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c5lib  # noqa: E402

skm = pytest.importorskip("sklearn.metrics")


def _data(k=5, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    p = np.where(rng.random(n) < 0.7, y, rng.integers(0, k, n))
    return y, p


def test_kappa_and_macro_f1_match_sklearn():
    for k in (2, 5, 9):
        y, p = _data(k)
        cm = c5lib.confusion(y, p, k)
        assert c5lib.kappa_cm(cm) == pytest.approx(skm.cohen_kappa_score(y, p))
        assert c5lib.macro_f1_cm(cm) == pytest.approx(skm.f1_score(y, p, average="macro", labels=list(range(k))))


def test_absent_class_scores_zero():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 0, 1, 1])
    assert c5lib.macro_f1_cm(c5lib.confusion(y, p, 3)) == pytest.approx(2 / 3)


def test_bootstrap_is_deterministic_and_brackets_point():
    y, p = _data(2)
    cl = np.arange(len(y)) // 4
    a = c5lib.bootstrap(y, p, 2, c5lib.kappa_cm, cl, n_boot=300)
    b = c5lib.bootstrap(y, p, 2, c5lib.kappa_cm, cl, n_boot=300)
    k = c5lib.kappa_cm(c5lib.confusion(y, p, 2))
    assert a == b and a[0] < k < a[1]


def test_cluster_bootstrap_is_wider_when_clusters_are_duplicates():
    y, p = _data(2, n=1000)
    yy, pp = np.repeat(y, 5), np.repeat(p, 5)
    rows = c5lib.bootstrap(yy, pp, 2, c5lib.kappa_cm, None, n_boot=400)
    clus = c5lib.bootstrap(yy, pp, 2, c5lib.kappa_cm, np.repeat(np.arange(1000), 5), n_boot=400)
    assert (clus[1] - clus[0]) > 1.5 * (rows[1] - rows[0])


def test_unit_guard():
    g = c5lib.UnitGuard(cap=1.0, rate=1.71, preamble_min=30)
    assert g.allows(60) and not g.allows(3600)
    assert c5lib.gpu_rate("NVIDIA L4") == ("L4", 1.71)
    assert c5lib.gpu_rate("NVIDIA A100-SXM4-40GB") == ("A100", 5.4)
    assert c5lib.gpu_rate("NVIDIA H100 80GB HBM3") == (None, None)
