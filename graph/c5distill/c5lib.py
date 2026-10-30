"""C5 metrics, bootstrap and the unit guard. numpy only, so it runs unchanged on the Mac and on Colab."""
import time

import numpy as np

SEED = 20261028
N_BOOT = 2000


def confusion(y, p, k, w=None):
    return np.bincount(y * k + p, weights=w, minlength=k * k).reshape(k, k)


def kappa_cm(cm):
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def per_class_cm(cm):
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    f1 = np.where(d > 0, 2 * tp / np.maximum(d, 1e-12), 0.0)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0)
    rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-12), 0.0)
    return prec, rec, f1, cm.sum(1)


def macro_f1_cm(cm):
    return float(per_class_cm(cm)[2].mean())


def bootstrap(y, p, k, stat, clusters=None, n_boot=N_BOOT, seed=SEED):
    """Percentile 95% CI of stat(confusion) over resamples of rows, or of clusters when given."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    if clusters is None:
        idx = np.arange(len(y))
    else:
        _, idx = np.unique(np.asarray(clusters), return_inverse=True)
    m = idx.max() + 1 if len(idx) else 0
    vals = np.empty(n_boot)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, m, m), minlength=m)
        vals[b] = stat(confusion(y, p, k, cnt[idx].astype(float)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def report(y, p, classes, clusters=None, with_ci=True):
    k = len(classes)
    y, p = np.asarray(y), np.asarray(p)
    cm = confusion(y, p, k)
    prec, rec, f1, sup = per_class_cm(cm)
    out = {"n": int(len(y)), "kappa": kappa_cm(cm), "macro_f1": macro_f1_cm(cm),
           "accuracy": float(np.trace(cm) / max(cm.sum(), 1)),
           "per_class": {c: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                             "f1": round(float(f1[i]), 4), "support": int(sup[i])} for i, c in enumerate(classes)},
           "confusion": cm.astype(int).tolist()}
    if with_ci:
        out["kappa_ci95"] = bootstrap(y, p, k, kappa_cm, clusters)
        out["macro_f1_ci95"] = bootstrap(y, p, k, macro_f1_cm, clusters)
    return out


class UnitGuard:
    """Projects Colab units from wall time. rate = units/hour of the GPU actually assigned."""

    def __init__(self, cap, rate, preamble_min=10.0):
        self.cap, self.rate, self.t0 = cap, rate, time.time() - preamble_min * 60

    def spent(self):
        return (time.time() - self.t0) / 3600 * self.rate

    def allows(self, next_seconds):
        return self.spent() + next_seconds / 3600 * self.rate <= self.cap

    def state(self):
        return {"units_spent_est": round(self.spent(), 3), "cap": self.cap, "rate_per_hour": self.rate}


# Colab units/hour (handoff §1: one blogger's table; read the live rate from `colab usage`). H100 unknown.
RATES = {"T4": 1.19, "L4": 1.71, "A100": 5.4, "RTX PRO 6000": 8.71}


def gpu_rate(name):
    """(gpu, units/h) for a torch device name, or (None, None) when unknown: the job then needs --rate."""
    for k, v in RATES.items():
        if k in (name or "").upper():
            return ("G4" if k == "RTX PRO 6000" else k), v
    return None, None
