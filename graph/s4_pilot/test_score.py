"""Unit checks for score.py's kNN macro top-1, same-size null, pooling and bootstrap (synthetic data)."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import score  # noqa: E402


def clustered(n_clusters=20, per=15, dim=32, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = score.normalise(rng.normal(size=(n_clusters, dim)))
    x = np.repeat(centres, per, axis=0) + noise * rng.normal(size=(n_clusters * per, dim))
    return score.normalise(x).astype(np.float32), np.repeat(np.arange(n_clusters), per)


def test_nearest_excludes_self_and_finds_cluster_mate():
    x, lab = clustered()
    nn = score.nearest(x, chunk=37)
    assert (nn != np.arange(len(x))).all()
    assert (lab[nn] == lab).mean() > 0.99


def test_list_hits_vs_null():
    x, lab = clustered()
    nn = score.nearest(x)
    groups = [np.flatnonzero(lab == c) for c in range(20)]
    obs = score.list_hits(groups, nn)
    nul = score.null_hits(groups, len(x), nn)
    assert obs.mean() > 0.99 and nul.mean() < 0.2
    lo, hi = score.boot_ci(obs - nul, 0.9875)
    assert lo > 0.7 and hi <= 1.0


def test_list_hits_counts_members_only():
    nn = np.array([1, 0, 3, 2, 0])
    assert np.allclose(score.list_hits([np.array([0, 1, 2])], nn), [2 / 3])
    assert np.allclose(score.list_hits([np.array([2, 3, 4])], nn), [2 / 3])


def test_pool_groups_by_pid():
    pids = np.array([5, 2, 5, 2, 9])
    vec = np.array([[1, 0], [0, 1], [1, 0], [0, 3], [3, 4]], dtype=np.float16)
    u, p = score.pool(pids, vec)
    assert u.tolist() == [2, 5, 9]
    assert np.allclose(p, [[0, 1], [1, 0], [0.6, 0.8]], atol=1e-3)


def test_boot_ci_is_deterministic():
    d = np.random.default_rng(1).normal(0.1, 1, 500)
    assert score.boot_ci(d, 0.95) == score.boot_ci(d, 0.95)
