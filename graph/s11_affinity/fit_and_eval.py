#!/usr/bin/env python3
"""Step S11b: Fit learned affinity weights (Tasks A, B', C', theta_serve), run V1-V6 & V8(i), and emit W_lin.

References:
  - docs/execution-plans/2026-10-25-whole-network-signals-plan.md §4.3-§4.8, S11
  - docs/execution-plans/2026-10-28-whole-network-preregistration.md §3, §4, §9.3, §9.4
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import pathlib
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.optimize
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

GRAPH_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GRAPH_DIR))
import eval_temporal as et  # noqa: E402
import s3_prep  # noqa: E402


# ---------- Pre-registered Ablation Ladder (S1 §3 line 75) ----------
F_FOLLOW = ["FOL", "FAGE", "FBURST", "MUT"]
G_USER = ["DU", "DVP", "HEAVY", "MISS"]
H_PROD = ["PFOL", "PORIG", "POSH", "PCUR", "PBLK", "PBLKN"]


def type_feats(types: list[str], half_lives: tuple[int, ...] = (14, 45, 120)) -> list[str]:
    out: list[str] = []
    for t in types:
        if t in ("LU", "DL"):
            for h in half_lives:
                out.append(f"{t}_L{h}")
            out.append(f"{t}_ND")
        else:
            for h in half_lives:
                out.append(f"{t}_L{h}")
            out.extend([f"{t}_ND", f"{t}_DP", f"{t}_DSL", f"{t}_DSF", f"{t}_C7", f"{t}_ES", f"{t}_LS"])
    return out


def build_ladder(include_qn: bool, v7_precision_pass: bool, half_lives: tuple[int, ...] = (14, 45, 120)) -> dict[str, list[str]]:
    r1 = F_FOLLOW + G_USER + H_PROD
    r2 = r1 + type_feats(["LO", "RP", "LU", "DL"], half_lives)
    r3_types = ["LR", "CR", "URf", "URn", "Qf"] + (["Qn"] if include_qn else [])
    r3 = r2 + type_feats(r3_types, half_lives) + ["LST_N", "RNL"]
    r4 = r3 + ["ACKS", "MAINT"]
    # S1 §9.3 V7: if V7 precision fails, QH/RH revert out of hostile A- features
    e_block = ["QH_L45", "RH_L45", "BLK", "LBLK"] if v7_precision_pass else ["BLK", "LBLK"]
    r5 = r4 + e_block
    return {"R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5, "R5_full_E": r4 + ["QH_L45", "RH_L45", "BLK", "LBLK"]}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------- Data Loading ----------

def load_o3a_split(
    split_dir: pathlib.Path,
    feature_names: list[str],
    task_filter: int | None = 1,
    heldout_filter: int | None = None,
    require_elig_a: bool = True,
) -> dict[str, np.ndarray]:
    cols = ["u", "p", "task", "ck", "pi", "ft", "heldout", "y_a", "y_b", "y_c", "elig_p", "elig_u", "offset"] + feature_names
    tables = []
    for p in sorted(split_dir.glob("j*.parquet")):
        t = pq.read_table(p, columns=cols)
        mask = None
        if task_filter is not None:
            m = pc.equal(t["task"], task_filter)
            mask = m if mask is None else pc.and_(mask, m)
        if heldout_filter is not None:
            m = pc.equal(t["heldout"], heldout_filter)
            mask = m if mask is None else pc.and_(mask, m)
        if require_elig_a:
            m = pc.and_(pc.and_(pc.equal(t["elig_u"], 1), pc.equal(t["elig_p"], 1)), pc.is_valid(t["y_a"]))
            mask = m if mask is None else pc.and_(mask, m)
        if mask is not None:
            t = t.filter(mask)
        tables.append(t)
    full = pa.concat_tables(tables)
    n = full.num_rows
    X = np.empty((n, len(feature_names)), dtype=np.float32)
    for j, fn in enumerate(feature_names):
        col_np = full[fn].to_numpy(zero_copy_only=False).astype(np.float32)
        # O3a features are stored as halffloat (fp16 max 65504); clamp rare PBLK fp16 overflows cleanly.
        X[:, j] = np.nan_to_num(col_np, nan=0.0, posinf=65504.0, neginf=-65504.0)
    pi_np = np.nan_to_num(full["pi"].to_numpy(zero_copy_only=False).astype(np.float64), nan=1.0, posinf=1.0, neginf=1e-6)
    off_np = np.nan_to_num(full["offset"].to_numpy(zero_copy_only=False).astype(np.float32), nan=0.0, posinf=20.0, neginf=0.0)
    out = {
        "u": full["u"].to_numpy(zero_copy_only=False).astype(np.uint32),
        "p": full["p"].to_numpy(zero_copy_only=False).astype(np.uint32),
        "ck": full["ck"].to_numpy(zero_copy_only=False).astype(np.uint8),
        "pi": np.maximum(pi_np, 1e-6),
        "ft": full["ft"].to_numpy(zero_copy_only=False).astype(np.uint32),
        "heldout": full["heldout"].to_numpy(zero_copy_only=False).astype(np.uint8),
        "offset": off_np,
        "y_c": full["y_c"].to_numpy(zero_copy_only=False).astype(np.uint8),
        "X": X,
    }
    if require_elig_a:
        out["y_a"] = full["y_a"].to_numpy(zero_copy_only=False).astype(np.float64)
    else:
        out["y_a_valid"] = pc.is_valid(full["y_a"]).to_numpy(zero_copy_only=False)
        out["y_a"] = pc.fill_null(full["y_a"], 0).to_numpy(zero_copy_only=False).astype(np.float64)
        out["y_b_valid"] = pc.is_valid(full["y_b"]).to_numpy(zero_copy_only=False)
        out["y_b"] = pc.fill_null(full["y_b"], 0).to_numpy(zero_copy_only=False).astype(np.float64)
    return out


# ---------- Fast Exact Weighted L2 Logistic Regression with Offset ----------

class OffsetLogReg:
    def __init__(self, feat_idx: list[int], feat_names: list[str], theta0: float, theta: np.ndarray, beta: float, use_offset: bool = True):
        self.feat_idx = np.array(feat_idx, dtype=np.int64)
        self.feat_names = feat_names
        self.theta0 = float(theta0)
        self.theta = theta.astype(np.float64)
        self.beta = float(beta)
        self.use_offset = use_offset
        self.calibrator: IsotonicRegression | None = None

    def linear(self, X_all: np.ndarray, offset: np.ndarray | float) -> np.ndarray:
        X = X_all[:, self.feat_idx] if X_all.shape[1] != len(self.feat_idx) else X_all
        z = self.theta0 + (X @ self.theta.astype(np.float32)).astype(np.float64)
        if self.use_offset:
            z = z + self.beta * offset
        return z

    def predict_proba(self, X_all: np.ndarray, offset: np.ndarray | float, calibrated: bool = False) -> np.ndarray:
        z = self.linear(X_all, offset)
        z_clip = np.clip(z, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-z_clip))
        if calibrated and self.calibrator is not None:
            p = self.calibrator.predict(p)
        return p


def fit_offset_logreg(
    X_all: np.ndarray,
    feat_idx: list[int],
    feat_names: list[str],
    y: np.ndarray,
    w: np.ndarray,
    offset: np.ndarray | None,
    l2: float = 1e-4,
    maxiter: int = 120,
) -> OffsetLogReg:
    X_sub = X_all[:, feat_idx].astype(np.float32, copy=True)
    w64 = w.astype(np.float64)
    w_sum = float(w64.sum())
    wn = (w64 / w_sum).astype(np.float64)

    mean = (wn @ X_sub).astype(np.float32)
    X_sub -= mean
    var = (wn @ (X_sub * X_sub)).astype(np.float32)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    # Keep constant columns safe
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    X_sub /= std

    use_offset = offset is not None
    if use_offset:
        off32 = offset.astype(np.float32)
        off_mean = float(wn @ off32)
        off_c = (off32 - off_mean).astype(np.float32)
    else:
        off_mean = 0.0
        off_c = None

    k = len(feat_idx)
    n_params = 1 + k + (1 if use_offset else 0)
    p_bar = float(np.clip(wn @ y, 1e-6, 1.0 - 1e-6))
    init = np.zeros(n_params, dtype=np.float64)
    init[0] = math.log(p_bar / (1.0 - p_bar))
    if use_offset:
        init[-1] = 0.5

    wy = wn * y

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        b0 = params[0]
        th = params[1:1 + k].astype(np.float32)
        z = b0 + (X_sub @ th).astype(np.float64)
        if use_offset:
            beta = params[-1]
            z = z + beta * off_c
        # Stable weighted BCEWithLogits: max(z, 0) - z*y + log1p(exp(-|z|))
        pos_z = np.maximum(z, 0.0)
        nll = float(wn @ (pos_z + np.log1p(np.exp(-np.abs(z)))) - wy @ z)
        sig = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40))), np.exp(np.clip(z, -40, 40)) / (1.0 + np.exp(np.clip(z, -40, 40))))
        diff = (wn * (sig - y)).astype(np.float32)

        grad = np.empty(n_params, dtype=np.float64)
        grad[0] = float(diff.sum(dtype=np.float64))
        grad_th = (diff @ X_sub).astype(np.float64) + l2 * params[1:1 + k]
        grad[1:1 + k] = grad_th
        reg = 0.5 * l2 * float(np.dot(params[1:1 + k], params[1:1 + k]))
        if use_offset:
            beta = params[-1]
            grad[-1] = float(np.dot(diff, off_c)) + 1e-4 * l2 * beta
            reg += 0.5 * 1e-4 * l2 * beta * beta
        return nll + reg, grad

    res = scipy.optimize.minimize(
        loss_and_grad,
        init,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
    )
    opt = res.x
    th_raw = opt[1:1 + k] / std.astype(np.float64)
    beta_raw = float(opt[-1]) if use_offset else 0.0
    th0_raw = float(opt[0] - np.dot(th_raw, mean.astype(np.float64)) - beta_raw * off_mean)
    return OffsetLogReg(feat_idx, feat_names, th0_raw, th_raw, beta_raw, use_offset=use_offset)


# ---------- Fast User-Resampled Bootstrap PR-AUC ----------

def binned_user_matrices(
    u: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    scores: np.ndarray,
    n_users: int,
    u_inv: np.ndarray,
    max_pos_bins: int = 24000,
) -> tuple[sp.csr_matrix, sp.csr_matrix, float]:
    exact = float(average_precision_score(y, scores, sample_weight=w))
    up = np.unique(scores[y > 0])
    if len(up) > max_pos_bins:
        up = np.unique(np.quantile(scores[y > 0], np.linspace(0.0, 1.0, max_pos_bins)))
    if len(up) == 0:
        bin_idx = np.zeros(len(scores), dtype=np.int64)
        nb = 1
    else:
        idx_r = np.searchsorted(up, scores, side="right") - 1
        idx_l = np.searchsorted(up, scores, side="left")
        is_tie = (idx_l < len(up)) & (up[np.minimum(idx_l, len(up) - 1)] == scores)
        bin_idx = np.where(idx_r < 0, 0, np.where(is_tie, 2 * idx_l + 1, 2 * idx_r + 2))
        nb = 2 * len(up) + 1
    wy = w * y
    P = sp.csr_matrix((wy, (u_inv, bin_idx)), shape=(n_users, nb), dtype=np.float64)
    N = sp.csr_matrix((w, (u_inv, bin_idx)), shape=(n_users, nb), dtype=np.float64)
    return P, N, exact


def boot_ap_from_bins(
    arms: dict[str, tuple[sp.csr_matrix, sp.csr_matrix, float]],
    n_users: int,
    n_boot: int = 2000,
    seed: int = 20261028,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    points = {k: v[2] for k, v in arms.items()}
    boots = {k: np.empty(n_boot, dtype=np.float64) for k in arms}
    # Check binned AP accuracy on full data
    ones = np.ones((1, n_users), dtype=np.float64)
    for k, (P, N, exact) in arms.items():
        pb = (ones @ P)[:, ::-1]
        tb = (ones @ N)[:, ::-1]
        cp = np.cumsum(pb, axis=1)
        ct = np.cumsum(tb, axis=1)
        ap_bin = float(((cp / np.maximum(ct, 1e-12)) * (pb / np.maximum(cp[:, -1:], 1e-12))).sum())
        if abs(ap_bin - exact) > 2e-3:
            raise RuntimeError(f"binned AP drift for {k}: bin={ap_bin:.6f} exact={exact:.6f}")

    rng = np.random.default_rng(seed)
    for b0 in range(0, n_boot, 50):
        nb = min(50, n_boot - b0)
        draws = rng.integers(0, n_users, size=(nb, n_users))
        counts = np.stack([np.bincount(row, minlength=n_users) for row in draws]).astype(np.float64)
        for k, (P, N, _) in arms.items():
            pb = (counts @ P)[:, ::-1]
            tb = (counts @ N)[:, ::-1]
            cp = np.cumsum(pb, axis=1)
            ct = np.cumsum(tb, axis=1)
            ap = ((cp / np.maximum(ct, 1e-12)) * (pb / np.maximum(cp[:, -1:], 1e-12))).sum(axis=1)
            boots[k][b0:b0 + nb] = ap
    return points, boots


def evaluate_slice_ap(
    u: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    arm_scores: dict[str, np.ndarray],
    mask: np.ndarray,
    comparisons: list[tuple[str, str]],
    n_boot: int = 2000,
    seed: int = 20261028,
) -> dict:
    m = mask & np.isfinite(y)
    if int(m.sum()) < 50 or float(y[m].sum()) < 5:
        return {"rows": int(m.sum()), "positives": int(y[m].sum()), "insufficient": True}
    u_sub = u[m]
    uniq_u, u_inv = np.unique(u_sub, return_inverse=True)
    n_u = len(uniq_u)
    arms_bin = {
        name: binned_user_matrices(u_sub, y[m], w[m], sc[m], n_u, u_inv)
        for name, sc in arm_scores.items()
    }
    pts, bts = boot_ap_from_bins(arms_bin, n_u, n_boot=n_boot, seed=seed)
    out = {
        "rows": int(m.sum()),
        "users": int(n_u),
        "positives": int(y[m].sum()),
        "weighted_base_rate": float((w[m] * y[m]).sum() / w[m].sum()),
        "arms": {
            k: {"pr_auc": pts[k], "ci_95": et.ci(bts[k], 0.95)}
            for k in arm_scores
        },
        "diffs": {},
    }
    for challenger, baseline in comparisons:
        d_pt = pts[challenger] - pts[baseline]
        d_bt = bts[challenger] - bts[baseline]
        d_ci = et.ci(d_bt, 0.95)
        out["diffs"][f"{challenger}_minus_{baseline}"] = {
            "diff": float(d_pt),
            "ci_95": d_ci,
            "pass_gt_0": bool(d_pt > 0 and d_ci[0] > 0),
        }
    return out


# ---------- Normalisation Selection on T_val (S1 §4) ----------

def select_normalisation_on_tval(
    tval: dict[str, np.ndarray],
    s_plus: np.ndarray,
    feat_map: dict[str, int],
    n_actors: float,
) -> dict:
    """Select (mode, alpha, kappa / K1) on T_val heldout==1 using non-followed / evidenced pairs (labels <= T)."""
    # Restrict to T_val heldout == 1 so selection is out-of-sample wrt T_fit fit
    m = tval["heldout"] == 1
    u = tval["u"][m]
    y = tval["y_a"][m]
    w = 1.0 / tval["pi"][m]
    sp_m = s_plus[m]
    du = np.expm1(tval["X"][m, feat_map["DU"]].astype(np.float64))
    np_fol = np.expm1(tval["X"][m, feat_map["PFOL"]].astype(np.float64))
    idf = np.log1p(n_actors / (1.0 + np_fol))

    uniq_u, u_inv = np.unique(u, return_inverse=True)
    n_u = len(uniq_u)

    cand_scores: dict[str, np.ndarray] = {}
    for alpha in (0.5, 1.0):
        au = np.power(50.0 / (50.0 + np.maximum(du, 0.0)), alpha)
        num = sp_m * au * idf
        u_sum = np.bincount(u_inv, weights=num, minlength=n_u)
        for kappa in (1.0, 3.0, 10.0):
            cand_scores[f"budget_a{alpha}_k{int(kappa)}"] = num / (kappa + u_sum[u_inv])

    # BM25 alternative (K1 in {1, 3}, B = 0.75)
    row_len = np.bincount(u_inv, weights=sp_m, minlength=n_u)
    avg_len = max(float(row_len.mean()), 1e-6)
    B = 0.75
    for k1 in (1.0, 3.0):
        denom = k1 * (1.0 - B + B * (row_len[u_inv] / avg_len)) + sp_m
        cand_scores[f"bm25_k1_{int(k1)}_b075"] = (sp_m * (k1 + 1.0) / np.maximum(denom, 1e-9)) * idf

    # Score AP on non-mutual T_val heldout==1
    nonmut = tval["X"][m, feat_map["MUT"]] == 0
    res = {}
    for name, sc in cand_scores.items():
        res[name] = float(average_precision_score(y[nonmut], sc[nonmut], sample_weight=w[nonmut]))

    best_ii = max(res, key=lambda k: res[k])
    # Also pick alpha in {0.5, 1.0} for W_lin (Vf41: only a_u and IDF allowed for W_lin)
    wlin_alpha_scores = {}
    for alpha in (0.5, 1.0):
        au = np.power(50.0 / (50.0 + np.maximum(du, 0.0)), alpha)
        sc = sp_m * au * idf
        wlin_alpha_scores[str(alpha)] = float(average_precision_score(y[nonmut], sc[nonmut], sample_weight=w[nonmut]))
    best_alpha_wlin = 0.5 if wlin_alpha_scores["0.5"] >= wlin_alpha_scores["1.0"] else 1.0

    return {
        "tval_nonmutual_ap_by_norm": res,
        "chosen_ii_norm": best_ii,
        "wlin_alpha_tval_ap": wlin_alpha_scores,
        "chosen_wlin_alpha": best_alpha_wlin,
    }


# ---------- V3, V4, V5, V6 Outcome Validations ----------

def run_v3_v4_v5_v6(
    otest: dict[str, np.ndarray],
    feat_map: dict[str, int],
    s10_dir: pathlib.Path,
    s3_dir: pathlib.Path,
    purge: set[str],
    exclude_s3: set[str],
    scales: dict[str, float],
    full_dids: list[str],
) -> dict:
    out: dict = {}

    # --- V3 (H1 courtesy likes): 14d top-level conversion after pure ACK vs size-matched LR interest ---
    X = otest["X"]
    y = otest["y_a"]
    w = 1.0 / otest["pi"]
    u = otest["u"]
    acks = X[:, feat_map["ACKS"]]
    lo_nd = X[:, feat_map["LO_ND"]]
    lr_nd = X[:, feat_map["LR_ND"]]
    pfol = X[:, feat_map["PFOL"]]
    decile = np.clip(np.digitize(pfol, np.quantile(pfol, np.linspace(0.1, 0.9, 9))), 0, 9)

    is_ack_only = (acks >= 0.999) & (lo_nd == 0) & (lr_nd == 0)
    is_lr_only = (acks == 0) & (lo_nd == 0) & (lr_nd > 0)

    # Size-match interest weights to ACK's producer-follower decile distribution
    ack_dec_w = np.bincount(decile[is_ack_only], weights=w[is_ack_only], minlength=10)
    lr_dec_w = np.bincount(decile[is_lr_only], weights=w[is_lr_only], minlength=10)
    dec_ratio = np.where(lr_dec_w > 0, ack_dec_w / np.maximum(lr_dec_w, 1e-12), 0.0)

    sel = is_ack_only | is_lr_only
    u_v3, inv_v3 = np.unique(u[sel], return_inverse=True)
    n_u3 = len(u_v3)
    m_ack = is_ack_only[sel]
    m_lr = is_lr_only[sel]
    w_sub = w[sel]
    y_sub = y[sel]
    w_lr_matched = w_sub * dec_ratio[decile[sel]]

    ack_num = np.bincount(inv_v3, weights=np.where(m_ack, w_sub * y_sub, 0.0), minlength=n_u3)
    ack_den = np.bincount(inv_v3, weights=np.where(m_ack, w_sub, 0.0), minlength=n_u3)
    lr_num = np.bincount(inv_v3, weights=np.where(m_lr, w_lr_matched * y_sub, 0.0), minlength=n_u3)
    lr_den = np.bincount(inv_v3, weights=np.where(m_lr, w_lr_matched, 0.0), minlength=n_u3)

    rng = np.random.default_rng(20261028)
    r_ack_pt = float(ack_num.sum() / max(ack_den.sum(), 1e-12))
    r_lr_pt = float(lr_num.sum() / max(lr_dec_w_sum := lr_den.sum(), 1e-12))
    diffs_v3 = np.empty(2000, dtype=np.float64)
    for b0 in range(0, 2000, 100):
        draws = rng.integers(0, n_u3, size=(100, n_u3))
        cnt = np.stack([np.bincount(row, minlength=n_u3) for row in draws]).astype(np.float64)
        ra = (cnt @ ack_num) / np.maximum(cnt @ ack_den, 1e-12)
        rl = (cnt @ lr_num) / np.maximum(cnt @ lr_den, 1e-12)
        diffs_v3[b0:b0 + 100] = ra - rl
    ci_v3 = et.ci(diffs_v3, 0.95)
    out["V3_courtesy_likes"] = {
        "ack_pairs": int(is_ack_only.sum()),
        "interest_lr_pairs": int(is_lr_only.sum()),
        "ack_14d_rate": r_ack_pt,
        "size_matched_interest_14d_rate": r_lr_pt,
        "diff_ack_minus_interest": float(r_ack_pt - r_lr_pt),
        "ci_95": ci_v3,
        "pass_ack_lower": bool(r_ack_pt < r_lr_pt and ci_v3[1] < 0),
        "note": "Whole-network 14-day top-level conversion at T_test (size-matched on producer follower decile).",
    }

    # --- V4 (H2 maintenance): (1) same-community share on producer-producer mutual pairs; (2) FB vs CLEAN gate ---
    cand5_full = s3_dir / "matched/cand5-full-matched"
    memb_full = np.load(cand5_full / "membership.npy").astype(np.int64)
    n_prod_full = len(memb_full)
    mut_lo = (X[:, feat_map["MUT"]] == 1) & (X[:, feat_map["LO_ND"]] > 0) & (otest["u"] < n_prod_full) & (otest["p"] < n_prod_full)
    u_pp = otest["u"][mut_lo]
    p_pp = otest["p"][mut_lo]
    maint_pp = X[mut_lo, feat_map["MAINT"]] == 1
    valid_c = (memb_pp_u := memb_full[u_pp]) >= 0
    valid_c &= (memb_pp_p := memb_full[p_pp]) >= 0
    same_c = (memb_pp_u == memb_pp_p).astype(np.float64)
    f_m = valid_c & maint_pp
    u_m = valid_c & (~maint_pp)
    same_flagged = float(same_c[f_m].mean()) if f_m.any() else float("nan")
    same_unflagged = float(same_c[u_m].mean()) if u_m.any() else float("nan")

    # Part 2 of V4: excluding MAINT from W/light-engagers on FB vs CLEAN (90d O2/prod)
    fl = np.load(cand5_full / "community_flags.npz")
    audit_dir = pathlib.Path.home() / "catbird-lake-local/communities-cand1-hubdemote/audit-mass-followers"
    if not audit_dir.exists():
        audit_dir = cand5_full / "audit-mass-followers"
    fd = np.load(audit_dir / "follower_degree.npz")
    st = np.load(audit_dir / "producer_stats.npz")
    packs = np.load(audit_dir / "pack_counts.npy")
    mass = fd["mass3000"] / np.maximum(st["followers"], 1)
    fb_mask = fl["follow_back"][memb_full]
    ft_mask = fl["follow_train"][memb_full]
    clean_mask = (packs >= 1) & (mass < 0.4) & (~fb_mask) & (~ft_mask)

    aud_excl_maint = np.zeros(n_prod_full, dtype=np.int64)
    maint_cnt = np.zeros(n_prod_full, dtype=np.int64)
    for p_pq in sorted((s10_dir / "O2/prod").glob("j*.parquet")):
        t = pq.read_table(p_pq, columns=["p", "light_eng_nonlu", "maint_pairs"])
        pids = t["p"].to_numpy()
        aud_excl_maint[pids] = t["light_eng_nonlu"].to_numpy()
        maint_cnt[pids] = t["maint_pairs"].to_numpy()
    # Note: light_eng_nonlu counts all light engagers; audience excludes MAINT pairs (`aud = light=1 & maint=0 & ...`).
    # Compare eligibility >= 10 engagers when subtracting maint_pairs vs keeping them:
    elig_no_maint = np.maximum(aud_excl_maint - maint_cnt, 0) >= 10
    elig_with_maint = aud_excl_maint >= 10
    rm_fb_nomaint = float((~elig_no_maint[fb_mask]).mean())
    rm_fb_withmaint = float((~elig_with_maint[fb_mask]).mean())
    rm_clean_nomaint = float((~elig_no_maint[clean_mask]).mean())
    out["V4_maintenance"] = {
        "pp_flagged_pairs": int(f_m.sum()),
        "pp_unflagged_mutual_pairs": int(u_m.sum()),
        "same_comm_flagged": same_flagged,
        "same_comm_unflagged": same_unflagged,
        "rm_fb_excl_maint_ge10": rm_fb_nomaint,
        "rm_fb_incl_maint_ge10": rm_fb_withmaint,
        "rm_clean_excl_maint_ge10": rm_clean_nomaint,
        "pass": bool(
            (not math.isnan(same_flagged))
            and same_flagged < same_unflagged
            and rm_fb_nomaint >= rm_fb_withmaint
            and rm_clean_nomaint <= 0.10
        ),
    }

    # --- V5 (H5 discovery likes): O3b at T (FOL==0) + m1_test_follows within 14d (live==True) ---
    excl_all = purge | exclude_s3
    uni_T = et.Universe(s3_dir / "matched/cand5-T-matched/producers.tsv", excl_all, 20261028)
    m1_users_t = pq.read_table(s3_dir / "m1/m1_users.parquet")
    m1_dids = [d for d in m1_users_t["did"].to_pylist() if d not in excl_all]
    m1_did_arr = pa.array(m1_dids, pa.string())
    # Map uid -> m1 user index via cityHash64
    ch_vals = _city_hash64_list(m1_dids)
    h_to_m1idx = {int(h): i for i, h in enumerate(ch_vals)}
    uh_t = pq.read_table(s10_dir / "s5p/eval_sample_uh.parquet", columns=["uid", "h", "m1"])
    uh_t = uh_t.filter(pc.equal(uh_t["m1"], 1))
    uid_to_m1 = {
        int(uid): h_to_m1idx[int(h)]
        for uid, h in zip(uh_t["uid"].to_pylist(), uh_t["h"].to_pylist())
        if int(h) in h_to_m1idx
    }
    full_to_uniT = uni_T.index(pa.array(full_dids, pa.string()))

    # 14d post-T live non-burst follows from m1_test_follows.parquet
    te_f = et.read_pairs(s3_dir / "m1/m1_test_follows.parquet", m1_did_arr, uni_T, ["burst", "live"])
    t_14d_end = et.M1_START_US + 14 * et.DAY_US
    m14 = (
        (te_f["user"] >= 0)
        & (te_f["item"] >= 0)
        & (te_f["created_us"] >= et.M1_START_US)
        & (te_f["created_us"] < t_14d_end)
        & te_f["live"].astype(bool)
    )
    conv14_keys = np.unique(et.pair_keys(te_f["user"][m14], te_f["item"][m14], uni_T.n))

    dl_scale = float(scales["DL_ND"])
    fol_scale = float(scales["FOL"])
    # Scan O3b for non-followed pairs at T (FOL == 0): dl_nd >= 2 vs dl_nd == 1
    u_d2, c_d2, b_d2 = [], [], []
    u_d1, c_d1, b_d1 = [], [], []
    fol_buckets = np.digitize(uni_T.followers, [1_000, 10_000, 100_000])
    for p_pq in sorted((s10_dir / "O3b").glob("features_j*.parquet")):
        t = pq.read_table(p_pq, columns=["u", "p", "FOL", "DL_ND"])
        u_raw = t["u"].to_numpy()
        p_raw = t["p"].to_numpy()
        fol_q = t["FOL"].to_numpy()
        dl_q = t["DL_ND"].to_numpy()
        nf = fol_q == 0
        if not nf.any():
            continue
        u_raw, p_raw, dl_q = u_raw[nf], p_raw[nf], dl_q[nf]
        m_idx = np.array([uid_to_m1.get(int(x), -1) for x in u_raw], dtype=np.int64)
        t_idx = full_to_uniT[p_raw]
        ok = (m_idx >= 0) & (t_idx >= 0) & uni_T.eligible[np.maximum(t_idx, 0)]
        m_idx, t_idx, dl_q = m_idx[ok], t_idx[ok], dl_q[ok]
        dl_val = dl_q.astype(np.float64) * (dl_scale / 255.0)
        # log1p(1) = 0.6931, log1p(2) = 1.0986 -> midpoint 0.90 separates >=2 days from 1 day
        keys = et.pair_keys(m_idx, t_idx, uni_T.n)
        hit = np.isin(keys, conv14_keys).astype(np.float64)
        bk = fol_buckets[t_idx]
        m2d = dl_val >= 0.90
        m1d = (dl_val > 0.30) & (dl_val < 0.90)
        if m2d.any():
            u_d2.append(m_idx[m2d]); c_d2.append(hit[m2d]); b_d2.append(bk[m2d])
        if m1d.any():
            u_d1.append(m_idx[m1d]); c_d1.append(hit[m1d]); b_d1.append(bk[m1d])

    u2 = np.concatenate(u_d2); c2 = np.concatenate(c_d2); b2 = np.concatenate(b_d2)
    u1 = np.concatenate(u_d1); c1 = np.concatenate(c_d1); b1 = np.concatenate(b_d1)
    b2_cnt = np.bincount(b2, minlength=4).astype(np.float64)
    b1_cnt = np.bincount(b1, minlength=4).astype(np.float64)
    w1_match = np.where(b1_cnt > 0, b2_cnt / np.maximum(b1_cnt, 1.0), 0.0)[b1]

    n_m1 = len(m1_dids)
    n2_u = np.bincount(u2, weights=c2, minlength=n_m1)
    d2_u = np.bincount(u2, minlength=n_m1).astype(np.float64)
    n1_u = np.bincount(u1, weights=c1 * w1_match, minlength=n_m1)
    d1_u = np.bincount(u1, weights=w1_match, minlength=n_m1)
    rate_2d = float(n2_u.sum() / max(d2_u.sum(), 1.0))
    rate_1d = float(n1_u.sum() / max(d1_u.sum(), 1.0))
    lift_pt = rate_2d / max(rate_1d, 1e-12)
    lifts_v5 = np.empty(2000, dtype=np.float64)
    rng5 = np.random.default_rng(20261028)
    for b0 in range(0, 2000, 100):
        draws = rng5.integers(0, n_m1, size=(100, n_m1))
        cnt = np.stack([np.bincount(row, minlength=n_m1) for row in draws]).astype(np.float64)
        r2 = (cnt @ n2_u) / np.maximum(cnt @ d2_u, 1e-12)
        r1 = (cnt @ n1_u) / np.maximum(cnt @ d1_u, 1e-12)
        lifts_v5[b0:b0 + 100] = r2 / np.maximum(r1, 1e-12)
    ci_v5 = et.ci(lifts_v5, 0.95)
    out["V5_discovery_likes"] = {
        "treated_ge2_days_pairs": int(len(u2)),
        "control_1_day_pairs": int(len(u1)),
        "conv_14d_live_ge2_days": rate_2d,
        "conv_14d_live_1_day_matched": rate_1d,
        "lift_vs_1_day_matched": lift_pt,
        "lift_ci_95": ci_v5,
        "pass_lift_gt_1": bool(lift_pt > 1.0 and ci_v5[0] > 1.0),
    }

    # --- V6 (H11 bulk follows): Task B' positive rate on bulk (burst>=10) vs size-matched non-bulk follows ---
    eng_m1 = et.read_engagement(s3_dir / "m1/m1_engagement.parquet", m1_did_arr, uni_T)
    win_b = (
        (te_f["user"] >= 0)
        & (te_f["item"] >= 0)
        & (te_f["created_us"] >= et.M1_START_US)
        & (te_f["created_us"] < et.M1_END_US)
        & te_f["live"].astype(bool)
    )
    bu = te_f["user"][win_b]
    bi = te_f["item"][win_b]
    bt = te_f["created_us"][win_b]
    bburst = te_f["burst"][win_b].astype(np.int64)
    # Task B' positive: >= 1 qualifying engagement in (follow + 7d, follow + 30d]
    b_keys = et.pair_keys(bu, bi, uni_T.n)
    b_pos = et.engaged_within(eng_m1, b_keys, bt + 7 * et.DAY_US, bt + 30 * et.DAY_US).astype(np.float64)
    is_bulk = bburst >= 10
    bk_v6 = fol_buckets[bi]
    bulk_bk = np.bincount(bk_v6[is_bulk], minlength=4).astype(np.float64)
    nonbulk_bk = np.bincount(bk_v6[~is_bulk], minlength=4).astype(np.float64)
    w_nb = np.where(nonbulk_bk > 0, bulk_bk / np.maximum(nonbulk_bk, 1.0), 0.0)[bk_v6[~is_bulk]]

    nb_u = np.bincount(bu[is_bulk], weights=b_pos[is_bulk], minlength=n_m1)
    db_u = np.bincount(bu[is_bulk], minlength=n_m1).astype(np.float64)
    nn_u = np.bincount(bu[~is_bulk], weights=b_pos[~is_bulk] * w_nb, minlength=n_m1)
    dn_u = np.bincount(bu[~is_bulk], weights=w_nb, minlength=n_m1)
    r_bulk = float(nb_u.sum() / max(db_u.sum(), 1.0))
    r_nonbulk = float(nn_u.sum() / max(dn_u.sum(), 1.0))
    diffs_v6 = np.empty(2000, dtype=np.float64)
    rng6 = np.random.default_rng(20261028)
    for b0 in range(0, 2000, 100):
        draws = rng6.integers(0, n_m1, size=(100, n_m1))
        cnt = np.stack([np.bincount(row, minlength=n_m1) for row in draws]).astype(np.float64)
        rb = (cnt @ nb_u) / np.maximum(cnt @ db_u, 1e-12)
        rn = (cnt @ nn_u) / np.maximum(cnt @ dn_u, 1e-12)
        diffs_v6[b0:b0 + 100] = rb - rn
    ci_v6 = et.ci(diffs_v6, 0.95)
    # Also compute empirical engagement rate by burst bucket to verify m_bulk curve
    burst_curve = {}
    for label, lo_b, hi_b in (("1", 1, 2), ("2-4", 2, 5), ("5-9", 5, 10), ("10-29", 10, 30), ("30+", 30, 10**9)):
        mb = (bburst >= lo_b) & (bburst < hi_b)
        burst_curve[label] = {"n": int(mb.sum()), "task_b_pos_rate": float(b_pos[mb].mean()) if mb.any() else float("nan")}
    out["V6_bulk_follows"] = {
        "bulk_follows_n": int(is_bulk.sum()),
        "nonbulk_follows_n": int((~is_bulk).sum()),
        "bulk_task_b_rate": r_bulk,
        "size_matched_nonbulk_task_b_rate": r_nonbulk,
        "diff_bulk_minus_nonbulk": float(r_bulk - r_nonbulk),
        "ci_95": ci_v6,
        "pass_bulk_lower": bool(r_bulk < r_nonbulk and ci_v6[1] < 0),
        "empirical_burst_curve": burst_curve,
    }
    return out


def _city_hash64_list(dids: list[str]) -> list[int]:
    """Exact ClickHouse cityHash64(did) for a list of DIDs."""
    import subprocess
    inp = "\n".join(dids).encode()
    res = subprocess.run(
        ["clickhouse", "local", "--input-format", "TSV", "--query", "SELECT cityHash64(c1) FROM table FORMAT TSV"],
        input=inp,
        capture_output=True,
        check=True,
    )
    return [int(x) for x in res.stdout.decode().splitlines() if x]


# ---------- V8(i): M1 Evaluation of Learned InterestedIn vs 2/1/2 vs Follows-Only ----------

_V8_STATE: dict = {}


def _v8_rank_user(u: int) -> dict[str, np.ndarray]:
    uni, data, part, arm_edges, k = (
        _V8_STATE[x] for x in ("uni", "data", "part", "arm_edges", "k")
    )
    f = data.follows(u)
    excl = np.union1d(f, [data.self_idx[u]] if data.self_idx[u] >= 0 else [])
    out = {}
    # Follows-only baseline (exact Partition.interest)
    ii_f = part.interest(f)
    nz = np.flatnonzero(ii_f)
    pos = et.gather(part.comm_off, nz)
    w = part.comm_score[pos] * np.repeat(ii_f[nz], np.diff(part.comm_off)[nz])
    out["follows_only"] = et.top_k(uni, part.comm_idx[pos], w, excl, k)

    for aname, (off, items, weights) in arm_edges.items():
        it = items[off[u]:off[u + 1]]
        wt = weights[off[u]:off[u + 1]]
        if len(it) == 0:
            out[aname] = et.fill(uni.tie_order, excl, k)
            continue
        p_pos = et.gather(part.prod_off, it)
        p_lens = np.diff(part.prod_off)[it]
        ii = np.bincount(
            part.prod_comm[p_pos],
            weights=part.prod_score[p_pos] * np.repeat(wt, p_lens),
            minlength=part.n_comm,
        )
        nz_a = np.flatnonzero(ii)
        pos_a = et.gather(part.comm_off, nz_a)
        w_a = part.comm_score[pos_a] * np.repeat(ii[nz_a], np.diff(part.comm_off)[nz_a])
        out[aname] = et.top_k(uni, part.comm_idx[pos_a], w_a, excl, k)
    return out


def _v8_rank_chunk(users: range) -> dict[str, np.ndarray]:
    k = _V8_STATE["k"]
    rows = [_v8_rank_user(u) for u in users]
    names = list(rows[0].keys())
    return {
        n: np.array([np.pad(r[n], (0, k - len(r[n])), constant_values=-1) for r in rows], dtype=np.int64).reshape(-1, k)
        for n in names
    }


def run_v8_m1(
    model_preT: OffsetLogReg,
    o_ref_eval: float,
    chosen_ii_norm: str,
    s10_dir: pathlib.Path,
    s3_dir: pathlib.Path,
    purge: set[str],
    exclude_s3: set[str],
    feature_names: list[str],
    scales: dict[str, float],
    full_dids: list[str],
    n_actors: float,
    workers: int = 8,
) -> dict:
    excl_all = purge | exclude_s3
    uni_T = et.Universe(s3_dir / "matched/cand5-T-matched/producers.tsv", excl_all, 20261028)
    part = et.Partition("cand5_T", s3_dir / "matched/cand5-T-matched", uni_T)
    data = et.M1Data(s3_dir / "m1", uni_T, excl_all, 100, "clickhouse")

    ch_vals = _city_hash64_list(data.dids)
    h_to_m1 = {int(h): i for i, h in enumerate(ch_vals)}
    uh_t = pq.read_table(s10_dir / "s5p/eval_sample_uh.parquet", columns=["uid", "h", "m1"])
    uh_t = uh_t.filter(pc.equal(uh_t["m1"], 1))
    uid_to_m1 = {
        int(uid): h_to_m1[int(h)]
        for uid, h in zip(uh_t["uid"].to_pylist(), uh_t["h"].to_pylist())
        if int(h) in h_to_m1
    }
    full_to_uniT = uni_T.index(pa.array(full_dids, pa.string()))
    feat_map = {fn: i for i, fn in enumerate(feature_names)}
    scale_vec = np.array([float(scales[fn]) / 255.0 for fn in feature_names], dtype=np.float32)

    # Load O3b and score s+ and 2/1/2 in chunks per shard (low RAM!)
    u_list, item_list, splus_list, h212_list, du_list = [], [], [], [], []
    for p_pq in sorted((s10_dir / "O3b").glob("features_j*.parquet")):
        t = pq.read_table(p_pq)
        u_raw = t["u"].to_numpy()
        p_raw = t["p"].to_numpy()
        m_idx = np.array([uid_to_m1.get(int(x), -1) for x in u_raw], dtype=np.int64)
        t_idx = full_to_uniT[p_raw]
        ok = (m_idx >= 0) & (t_idx >= 0)
        if not ok.any():
            continue
        m_idx = m_idx[ok]
        t_idx = t_idx[ok]
        n_ok = int(ok.sum())
        X_sh = np.empty((n_ok, len(feature_names)), dtype=np.float32)
        for j, fn in enumerate(feature_names):
            X_sh[:, j] = t[fn].to_numpy(zero_copy_only=False)[ok].astype(np.float32) * scale_vec[j]
        sp_sh = model_preT.predict_proba(X_sh, o_ref_eval, calibrated=True)
        # 2.0 * FOL + 1.0 * likes + 2.0 * reposts
        fol_v = (X_sh[:, feat_map["FOL"]] > 0.5).astype(np.float64)
        likes_v = np.expm1(X_sh[:, feat_map["LO_DP"]]) + np.expm1(X_sh[:, feat_map["LR_DP"]]) + np.expm1(X_sh[:, feat_map["LU_ND"]])
        rp_v = np.expm1(X_sh[:, feat_map["RP_DP"]])
        h212_sh = 2.0 * fol_v + 1.0 * likes_v + 2.0 * rp_v
        du_sh = np.expm1(X_sh[:, feat_map["DU"]].astype(np.float64))

        u_list.append(m_idx)
        item_list.append(t_idx)
        splus_list.append(sp_sh)
        h212_list.append(h212_sh)
        du_list.append(du_sh)

    u_all = np.concatenate(u_list)
    it_all = np.concatenate(item_list)
    sp_all = np.concatenate(splus_list)
    h212_all = np.concatenate(h212_list)
    du_all = np.concatenate(du_list)

    # Supplement any pre-T train follows beyond O3b's 300-pair cap so heavy-degree users lose 0 follows
    o3b_keys = np.unique(et.pair_keys(u_all, it_all, uni_T.n))
    deg = data.degree()
    fu = np.repeat(np.arange(data.n, dtype=np.int64), deg)
    fi = data.f_items
    fkeys = et.pair_keys(fu, fi, uni_T.n)
    miss = ~np.isin(fkeys, o3b_keys)
    if miss.any():
        mu, mi = fu[miss], fi[miss]
        # Construct quiet-follow baseline feature vector (FOL=1, FAGE=log1p(45),DSL/DSF=90, DU, PFOL)
        X_qf = np.zeros((len(mu), len(feature_names)), dtype=np.float32)
        for fn in feature_names:
            if fn.endswith("_DSL") or fn.endswith("_DSF"):
                X_qf[:, feat_map[fn]] = 90.0
        X_qf[:, feat_map["FOL"]] = 1.0
        X_qf[:, feat_map["FAGE"]] = math.log1p(60.0)
        X_qf[:, feat_map["DU"]] = np.log1p(deg[mu]).astype(np.float32)
        X_qf[:, feat_map["HEAVY"]] = (deg[mu] >= 3000).astype(np.float32)
        X_qf[:, feat_map["PFOL"]] = np.log1p(uni_T.followers[mi]).astype(np.float32)
        X_qf[:, feat_map["PBLKN"]] = 1.0
        sp_qf = model_preT.predict_proba(X_qf, o_ref_eval, calibrated=True)
        h212_qf = np.full(len(mu), 2.0, dtype=np.float64)
        du_qf = deg[mu].astype(np.float64)

        u_all = np.concatenate([u_all, mu])
        it_all = np.concatenate([it_all, mi])
        sp_all = np.concatenate([sp_all, sp_qf])
        h212_all = np.concatenate([h212_all, h212_qf])
        du_all = np.concatenate([du_all, du_qf])

    # Apply chosen normalisation to learned_s11 (and keep top 200 per user per §4.6 step 7)
    idf_p = np.log1p(n_actors / (1.0 + uni_T.followers[it_all].astype(np.float64)))
    if chosen_ii_norm.startswith("budget_"):
        parts_n = chosen_ii_norm.split("_")
        alpha = float(parts_n[1][1:])
        kappa = float(parts_n[2][1:])
        au = np.power(50.0 / (50.0 + np.maximum(du_all, 0.0)), alpha)
        num = sp_all * au * idf_p
        u_sum = np.bincount(u_all, weights=num, minlength=data.n)
        w_learned = num / (kappa + u_sum[u_all])
    else:
        k1 = float(chosen_ii_norm.split("_")[2])
        B = 0.75
        rlen = np.bincount(u_all, weights=sp_all, minlength=data.n)
        avg_l = max(float(rlen[rlen > 0].mean()), 1e-6)
        denom = k1 * (1.0 - B + B * (rlen[u_all] / avg_l)) + sp_all
        w_learned = (sp_all * (k1 + 1.0) / np.maximum(denom, 1e-9)) * idf_p

    # Also build raw unnormalized s+ arm as diagnostic, and keep top-200 per user (§4.6 step 7)
    def build_csr_top200(u_arr: np.ndarray, it_arr: np.ndarray, w_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = w_arr > 0
        u_k, i_k, w_k = u_arr[keep], it_arr[keep], w_arr[keep]
        order = np.lexsort((-w_k, u_k))
        u_s, i_s, w_s = u_k[order], i_k[order], w_k[order]
        off = np.zeros(data.n + 1, dtype=np.int64)
        np.cumsum(np.bincount(u_s, minlength=data.n), out=off[1:])
        # Cap at 200 per user
        sel_chunks = [np.arange(off[u], min(off[u + 1], off[u] + 200), dtype=np.int64) for u in range(data.n) if off[u + 1] > off[u]]
        sel = np.concatenate(sel_chunks) if sel_chunks else np.zeros(0, dtype=np.int64)
        u_f, i_f, w_f = u_s[sel], i_s[sel], w_s[sel]
        off2 = np.zeros(data.n + 1, dtype=np.int64)
        np.cumsum(np.bincount(u_f, minlength=data.n), out=off2[1:])
        return off2, i_f, w_f

    arm_edges = {
        "learned_s11": build_csr_top200(u_all, it_all, w_learned),
        "heuristic_2_1_2": build_csr_top200(u_all, it_all, h212_all),
    }

    import multiprocessing as mp
    _V8_STATE.update(uni=uni_T, data=data, part=part, arm_edges=arm_edges, k=100)
    chunks = [range(s, min(s + 2000, data.n)) for s in range(0, data.n, 2000)]
    with mp.get_context("fork").Pool(workers) as pool:
        parts_r = pool.map(_v8_rank_chunk, chunks)
    tops = {n: np.concatenate([p[n] for p in parts_r]) for n in parts_r[0]}

    cols, masks, labels = [], [], []
    for set_name in ("primary", "secondary", "discovery"):
        if set_name not in data.sets:
            continue
        su, si = data.sets[set_name]
        for arm in ("follows_only", "heuristic_2_1_2", "learned_s11"):
            ranks = et.hit_ranks(tops[arm], su, si, uni_T.n)
            um = et.user_metrics(ranks, su, data.n)
            for metric in ("recall@10", "recall@50", "recall@100", "ndcg@100"):
                cols.append(um[metric])
                masks.append(um["_mask"])
                labels.append((set_name, arm, metric))

    X_m, M_m = np.column_stack(cols), np.column_stack(masks)
    point, boots = et.bootstrap(X_m, M_m, 2000, 20261028)
    col = {lab: j for j, lab in enumerate(labels)}

    metrics: dict = {}
    for (s, arm, metric), j in col.items():
        metrics.setdefault(s, {}).setdefault(arm, {})[metric] = et.summary(point[j], boots[:, j], int(M_m[:, j].sum()))

    comparisons = {}
    for base in ("follows_only", "heuristic_2_1_2"):
        comp = {}
        for s in ("primary", "secondary", "discovery"):
            if s not in data.sets:
                continue
            for metric in ("recall@50", "recall@100"):
                ja, jb = col[(s, base, metric)], col[(s, "learned_s11", metric)]
                rel_b = boots[:, jb] / boots[:, ja] - 1.0
                comp[f"{s}/{metric}"] = {
                    "rel": float(point[jb] / point[ja] - 1.0),
                    "ci_95": et.ci(rel_b, 0.95),
                }
        comparisons[f"learned_s11_vs_{base}"] = comp

    # S1 §9.3 V8(i): non-inferior on M1-follow (CI lower bound >= -2% relative) and better on M1-engaged (secondary)
    f50_vs_fol = comparisons["learned_s11_vs_follows_only"]["primary/recall@50"]
    f50_vs_212 = comparisons["learned_s11_vs_heuristic_2_1_2"]["primary/recall@50"]
    e50_vs_fol = comparisons["learned_s11_vs_follows_only"]["secondary/recall@50"]
    e50_vs_212 = comparisons["learned_s11_vs_heuristic_2_1_2"]["secondary/recall@50"]
    v8i_pass = bool(
        f50_vs_fol["ci_95"][0] >= -0.02
        and f50_vs_212["ci_95"][0] >= -0.02
        and e50_vs_fol["rel"] > 0 and e50_vs_fol["ci_95"][0] > 0
        and e50_vs_212["rel"] > 0 and e50_vs_212["ci_95"][0] > 0
    )
    return {
        "chosen_ii_norm": chosen_ii_norm,
        "o3b_pairs_scored": int(len(u_all)),
        "supplemented_quiet_follows_beyond_300": int(miss.sum()),
        "metrics": metrics,
        "comparisons": comparisons,
        "v8_i_pass": v8i_pass,
    }


# ---------- Exact Producer Score W_lin from O2 (plan §4.6, Vf41) ----------

def compute_and_write_wlin(
    o2_dir: pathlib.Path,
    lu_disc_path: pathlib.Path,
    model: OffsetLogReg,
    alpha: float,
    full_dids: list[str],
    n_actors: float,
    out_parquet: pathlib.Path,
) -> dict:
    sfx = "__a05" if abs(alpha - 0.5) < 1e-6 else "__a10"
    # Also read auxiliary columns so any fp16-overflowed O2 halffloat sum (>65,504 on mega-accounts
    # or 90*S0 on *_DSL/*_DSF when S0 > 728) is reconstructed cleanly in float64.
    aux_cols = [
        "p", "S0__a05", "S0__a10", "audience", "followers", "originals_90d", "replies_90d", "quotes_90d",
        "a_half_curators", "live_blockers_raw", "blockers_per_1k_burst_discounted",
    ]
    dsl_bases = {fn[:-4] for fn in model.feat_names if fn.endswith("_DSL") or fn.endswith("_DSF")}
    extra_sum_cols = set()
    for fn in model.feat_names:
        extra_sum_cols.add(f"{fn}{sfx}")
    for b in dsl_bases:
        for st_s in ("L14", "L45", "L120", "ND", "DP", "C7"):
            extra_sum_cols.add(f"{b}_{st_s}{sfx}")
    needed_cols = aux_cols + sorted(extra_sum_cols)
    tables = [pq.read_table(p, columns=needed_cols) for p in sorted(o2_dir.glob("j*.parquet"))]
    t = pa.concat_tables(tables)
    pids = t["p"].to_numpy().astype(np.int64)
    orig90 = t["originals_90d"].to_numpy().astype(np.float64)
    repl90 = t["replies_90d"].to_numpy().astype(np.float64)
    quot90 = t["quotes_90d"].to_numpy().astype(np.float64)
    posts90 = np.maximum(orig90 + repl90 + quot90, 1.0)
    elig_mask = orig90 > 0
    med_orig_14d = float(np.median(orig90[elig_mask] * (14.0 / 90.0))) if elig_mask.any() else 1.0
    o_ref = math.log1p(med_orig_14d)

    aud = t["audience"].to_numpy().astype(np.float64)
    s0_raw = t[f"S0{sfx}"].to_numpy().astype(np.float64)
    s0_a10 = t["S0__a10"].to_numpy().astype(np.float64)
    fin_s0_large = np.isfinite(s0_raw) & (aud >= 2000)
    med_s0_per_aud = float(np.median(s0_raw[fin_s0_large] / aud[fin_s0_large])) if fin_s0_large.any() else 0.32
    s0 = s0_raw.copy()
    bad_s0 = ~np.isfinite(s0)
    if bad_s0.any():
        if sfx == "__a05":
            fin_both = np.isfinite(s0_raw) & np.isfinite(s0_a10) & (s0_raw >= 20000)
            r05_10 = float(np.median(s0_raw[fin_both] / np.maximum(s0_a10[fin_both], 1.0))) if fin_both.any() else 1.86
            use_a10 = bad_s0 & np.isfinite(s0_a10)
            s0[use_a10] = np.clip(r05_10 * s0_a10[use_a10], 65504.0, aud[use_a10])
        still_bad = ~np.isfinite(s0)
        s0[still_bad] = np.clip(med_s0_per_aud * aud[still_bad], 65504.0, np.maximum(aud[still_bad], 65504.0))

    fol_raw = t["followers"].to_numpy().astype(np.float64)
    fol = np.maximum(fol_raw, 1.0)
    cur = t["a_half_curators"].to_numpy().astype(np.float64)
    blk_raw = t["live_blockers_raw"].to_numpy().astype(np.float64)
    blk1k = pc.fill_null(t["blockers_per_1k_burst_discounted"], 0.0).to_numpy(zero_copy_only=False).astype(np.float64)
    blk1k = np.nan_to_num(blk1k, nan=0.0, posinf=65504.0, neginf=0.0)
    idf_p = np.log1p(n_actors / (1.0 + aud))

    # Exact producer-constant feature values f_p (where S_k(p) == S0(p) * f_p identically in sql_p2.py)
    prod_const = {
        "PFOL": np.log1p(fol_raw),
        "PORIG": np.log1p(orig90),
        "POSH": orig90 / posts90,
        "PCUR": np.log1p(cur),
        "PBLK": np.where(blk_raw >= 50, blk1k, 0.0),
        "PBLKN": (blk_raw < 50).astype(np.float64),
        "HEAVY": np.zeros_like(s0),
        "MAINT": np.zeros_like(s0),
    }

    def clean_aux_sum(cname: str) -> np.ndarray:
        arr = t[cname].to_numpy().astype(np.float64)
        bad = ~np.isfinite(arr)
        if bad.any():
            fin_hi = np.isfinite(arr) & (s0 >= 10000) & (s0 <= 60000)
            ratio = float(np.median(arr[fin_hi] / np.maximum(s0[fin_hi], 1.0))) if fin_hi.any() else 0.5
            arr = arr.copy()
            arr[bad] = np.maximum(65504.0, ratio * s0[bad])
        return arr

    # Exact registered formula: W_lin_raw(p) = ((theta0 + beta * o_ref) * S0 + sum_k theta_k * S_k(p)) * idf_p
    lin_sum = (model.theta0 + model.beta * o_ref) * s0
    # Also compute null-referenced log-odds lift sum_u a_u * (eta(u,p) - eta(x_null)) * idf_p
    # where x_null has 90 for _DSL/_DSF, 1 for PBLKN, and 0 for active counts/FOL
    null_feat = np.zeros(len(model.feat_names), dtype=np.float64)
    for j, fn in enumerate(model.feat_names):
        if fn.endswith("_DSL") or fn.endswith("_DSF"):
            null_feat[j] = 90.0
        elif fn == "PBLKN":
            null_feat[j] = 1.0
    eta_null = model.theta0 + model.beta * o_ref + float(np.dot(model.theta, null_feat))

    reconstructed_cols = {}
    for j, fn in enumerate(model.feat_names):
        sk = t[f"{fn}{sfx}"].to_numpy().astype(np.float64)
        bad = ~np.isfinite(sk)
        if fn in prod_const:
            # Exact float64 identity S_k(p) = S0(p) * f_p avoids fp16 rounding and overflow
            sk = np.where(bad, s0 * prod_const[fn], sk)
            if bad.any():
                reconstructed_cols[fn] = int(bad.sum())
        elif bad.any() and (fn.endswith("_DSL") or fn.endswith("_DSF")):
            base_t = fn[:-4]
            X_b = np.column_stack([clean_aux_sum(f"{base_t}_{s}{sfx}") for s in ("L14", "L45", "L120", "ND", "DP", "C7")])
            fin_fit = (~bad) & np.isfinite(s0_raw) & (s0_raw > 20.0) & (s0_raw <= 700.0)
            if int(fin_fit.sum()) >= 100:
                y_rec = np.clip(90.0 * s0[fin_fit] - sk[fin_fit], 0.0, None)
                coef, *_ = np.linalg.lstsq(X_b[fin_fit], y_rec, rcond=None)
                rec_hat = np.clip(X_b[bad] @ coef, 0.0, 90.0 * s0[bad])
            else:
                rec_hat = np.zeros(int(bad.sum()), dtype=np.float64)
            sk = sk.copy()
            sk[bad] = 90.0 * s0[bad] - rec_hat
            reconstructed_cols[fn] = int(bad.sum())
        elif bad.any():
            sk = clean_aux_sum(f"{fn}{sfx}")
            reconstructed_cols[fn] = int(bad.sum())
        lin_sum = lin_sum + model.theta[j] * sk

    w_lin_literal = lin_sum * idf_p
    w_lin_lift = (lin_sum - eta_null * s0) * idf_p
    if not (np.isfinite(w_lin_literal).all() and np.isfinite(w_lin_lift).all()):
        raise RuntimeError("Non-finite values remain in W_lin after halffloat overflow reconstruction!")

    # Load D_discounted and light_engagers_discounted
    lu_t = pq.read_table(lu_disc_path, columns=["p", "light_engagers_discounted", "D_discounted"])
    lu_p = lu_t["p"].to_numpy().astype(np.int64)
    d_map = np.zeros(len(full_dids), dtype=np.float64)
    le_map = np.zeros(len(full_dids), dtype=np.float64)
    d_map[lu_p] = lu_t["D_discounted"].to_numpy().astype(np.float64)
    le_map[lu_p] = lu_t["light_engagers_discounted"].to_numpy().astype(np.float64)

    d_disc = d_map[pids]
    le_disc = le_map[pids]
    sqrt_fol = np.sqrt(fol)

    dids_col = [full_dids[int(p)] for p in pids]
    out_tbl = pa.table({
        "did": pa.array(dids_col, pa.string()),
        "pid_full": pa.array(pids, pa.int32()),
        "score": pa.array(w_lin_lift / sqrt_fol, pa.float64()),
        "w_lin_literal": pa.array(w_lin_literal, pa.float64()),
        "w_lin_lift": pa.array(w_lin_lift, pa.float64()),
        "w_lin_literal_per_sqrt_fol": pa.array(w_lin_literal / sqrt_fol, pa.float64()),
        "w_lin_lift_per_sqrt_fol": pa.array(w_lin_lift / sqrt_fol, pa.float64()),
        "D_discounted": pa.array(d_disc, pa.float64()),
        "D_per_sqrt_fol": pa.array(d_disc / sqrt_fol, pa.float64()),
        "light_engagers_discounted": pa.array(le_disc, pa.float64()),
        "a_half_curators": pa.array(cur, pa.float64()),
        "followers": pa.array(fol, pa.float64()),
        "S0": pa.array(s0, pa.float64()),
    })
    pq.write_table(out_tbl, out_parquet, compression="zstd")
    return {
        "rows": int(len(pids)),
        "alpha": alpha,
        "o_ref": o_ref,
        "eta_null": eta_null,
        "s0_fp16_overflows_reconstructed": int(bad_s0.sum()),
        "feature_fp16_overflows_reconstructed": reconstructed_cols,
        "w_lin_literal_negative_share": float((w_lin_literal < 0).mean()),
        "w_lin_lift_positive_share": float((w_lin_lift > 0).mean()),
        "sha256": sha256_file(out_parquet),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s10-dir", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s10-outputs-20261030")
    ap.add_argument("--s3-dir", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s3")
    ap.add_argument("--purge-dids", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s5-sweep-20261030/purge_dids.txt")
    ap.add_argument("--qn-report", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s11-affinity-20261030")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rules = json.loads((args.s10_dir / "meta/rules.json").read_text())
    scales = json.loads((args.s10_dir / "meta/o3b_scales.json").read_text())
    o1_sum = json.loads((args.s10_dir / "O1/summary.json").read_text())
    n_actors = float(o1_sum["o1"]["actors"]["||"])

    qn_rep = json.loads(args.qn_report.read_text())
    chosen_qn = qn_rep["R17_qn_gate"]["chosen_variant"]  # 'noqn' or 'withqn'
    v7_prec_pass = bool(qn_rep["V7_hostile_strata"]["precision_pass"])
    include_qn_chosen = chosen_qn == "withqn"

    feature_names = [f["name"] for f in rules["features"]]
    feat_map = {fn: i for i, fn in enumerate(feature_names)}
    print(f"[s11] Features={len(feature_names)}, chosen_qn={chosen_qn}, v7_precision_pass={v7_prec_pass}", flush=True)

    purge = {l.strip() for l in open(args.purge_dids) if l.strip()}
    exclude_s3 = et.read_dids([str(args.s3_dir / "exclude/exclude_dids_s3.txt")])
    full_dids, _ = s3_prep.read_producers(args.s3_dir / "matched/cand5-full-matched/producers.tsv")

    # 1. Load O3a Task A splits
    print("[s11] Loading O3a Task A splits (T_fit, T_val, T_test)...", flush=True)
    t_fit = load_o3a_split(args.s10_dir / "O3a/T_fit", feature_names, task_filter=1, heldout_filter=0, require_elig_a=True)
    t_val = load_o3a_split(args.s10_dir / "O3a/T_val", feature_names, task_filter=1, heldout_filter=None, require_elig_a=True)
    t_test_ho = load_o3a_split(args.s10_dir / "O3a/T_test", feature_names, task_filter=1, heldout_filter=1, require_elig_a=True)
    print(
        f"[s11] Loaded T_fit(ho=0)={len(t_fit['u']):,}, T_val(all)={len(t_val['u']):,}, T_test(ho=1)={len(t_test_ho['u']):,}",
        flush=True,
    )

    val_fit_m = t_val["heldout"] == 0
    val_ho_m = t_val["heldout"] == 1
    nonmut_val = val_ho_m & (t_val["X"][:, feat_map["MUT"]] == 0)
    w_fit = 1.0 / t_fit["pi"]
    w_val_ho = 1.0 / t_val["pi"][nonmut_val]

    # 2. Tune L2 and half-life subset on T_val (heldout == 1, non-mutual)
    ladder_chosen = build_ladder(include_qn_chosen, v7_prec_pass)
    cand_r5_idx = [feat_map[f] for f in ladder_chosen["R5"]]
    l2_grid = {}
    for l2 in (1e-5, 1e-4, 1e-3, 1e-2):
        m_tmp = fit_offset_logreg(t_fit["X"], cand_r5_idx, ladder_chosen["R5"], t_fit["y_a"], w_fit, t_fit["offset"], l2=l2)
        p_tmp = m_tmp.predict_proba(t_val["X"][nonmut_val], t_val["offset"][nonmut_val])
        l2_grid[str(l2)] = float(average_precision_score(t_val["y_a"][nonmut_val], p_tmp, sample_weight=w_val_ho))
    best_l2 = float(max(l2_grid, key=lambda k: l2_grid[k]))

    hl_grid = {}
    for hl_name, hls in (("all_14_45_120", (14, 45, 120)), ("h14_only", (14,)), ("h45_only", (45,)), ("h120_only", (120,))):
        lad_h = build_ladder(include_qn_chosen, v7_prec_pass, hls)
        idx_h = [feat_map[f] for f in lad_h["R5"]]
        m_h = fit_offset_logreg(t_fit["X"], idx_h, lad_h["R5"], t_fit["y_a"], w_fit, t_fit["offset"], l2=best_l2)
        p_h = m_h.predict_proba(t_val["X"][nonmut_val], t_val["offset"][nonmut_val])
        hl_grid[hl_name] = float(average_precision_score(t_val["y_a"][nonmut_val], p_h, sample_weight=w_val_ho))
    best_hl_name = max(hl_grid, key=lambda k: hl_grid[k])
    best_hls = {"all_14_45_120": (14, 45, 120), "h14_only": (14,), "h45_only": (45,), "h120_only": (120,)}[best_hl_name]
    print(f"[s11] T_val selection: best_l2={best_l2} ({l2_grid}), best_hl={best_hl_name} ({hl_grid})", flush=True)

    # 3. Fit calibrator & select InterestedIn / W_lin normalisation on T_val (before touching T_test or M1!)
    ladder_chosen = build_ladder(include_qn_chosen, v7_prec_pass, best_hls)
    ladder_withqn = build_ladder(True, v7_prec_pass, best_hls)
    ladder_noqn = build_ladder(False, v7_prec_pass, best_hls)

    # Pre-T ladder check on T_val (heldout == 1) to select active rung <= T without touching T_test
    tval_rung_ap = {}
    tval_models_fit = {}
    for rname in ("R1", "R2", "R3", "R4", "R5"):
        f_list = ladder_chosen[rname]
        f_idx = [feat_map[f] for f in f_list]
        mod = fit_offset_logreg(t_fit["X"], f_idx, f_list, t_fit["y_a"], w_fit, t_fit["offset"], l2=best_l2)
        sc_v = mod.predict_proba(t_val["X"], t_val["offset"])
        tval_models_fit[rname] = sc_v
        tval_rung_ap[rname] = float(average_precision_score(t_val["y_a"][nonmut_val], sc_v[nonmut_val], sample_weight=w_val_ho))

    tval_ladder_eval = evaluate_slice_ap(
        t_val["u"],
        t_val["y_a"],
        1.0 / t_val["pi"],
        tval_models_fit,
        nonmut_val,
        [("R2", "R1"), ("R3", "R2"), ("R4", "R3"), ("R5", "R4")],
    )
    chosen_preT_rung = "R1"
    for rk, rprev in (("R2", "R1"), ("R3", "R2"), ("R4", "R3"), ("R5", "R4")):
        if tval_ladder_eval["diffs"][f"{rk}_minus_{rprev}"]["pass_gt_0"]:
            chosen_preT_rung = rk
        else:
            break
    print(f"[s11] Pre-T ladder chosen on T_val: {chosen_preT_rung} (AP={tval_rung_ap})", flush=True)

    # Calibrator + normalisation selection on T_val using T_fit-fitted chosen_preT_rung
    f_preT = ladder_chosen[chosen_preT_rung]
    idx_preT = [feat_map[f] for f in f_preT]
    mod_tfit_chosen = fit_offset_logreg(t_fit["X"], idx_preT, f_preT, t_fit["y_a"], w_fit, t_fit["offset"], l2=best_l2)
    o_ref_tval = float(np.median(t_val["offset"]))
    raw_p_val = mod_tfit_chosen.predict_proba(t_val["X"], o_ref_tval, calibrated=False)
    iso_tval = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
    iso_tval.fit(raw_p_val[val_fit_m], t_val["y_a"][val_fit_m], sample_weight=1.0 / t_val["pi"][val_fit_m])
    mod_tfit_chosen.calibrator = iso_tval
    s_plus_tval = mod_tfit_chosen.predict_proba(t_val["X"], o_ref_tval, calibrated=True)
    norm_sel = select_normalisation_on_tval(t_val, s_plus_tval, feat_map, n_actors)
    print(f"[s11] Normalisation selected on T_val: {norm_sel}", flush=True)

    # 4. Fit full pre-T models on (T_fit + T_val)[heldout == 0] (strictly labels <= T = 2026-08-15!)
    X_fv = np.concatenate([t_fit["X"], t_val["X"][val_fit_m]], axis=0)
    y_fv = np.concatenate([t_fit["y_a"], t_val["y_a"][val_fit_m]], axis=0)
    w_fv = np.concatenate([w_fit, 1.0 / t_val["pi"][val_fit_m]], axis=0)
    off_fv = np.concatenate([t_fit["offset"], t_val["offset"][val_fit_m]], axis=0)
    del t_fit  # free RAM

    models_fv: dict[str, OffsetLogReg] = {}
    for rname in ("R1", "R2", "R3", "R4", "R5", "R5_full_E"):
        flist = ladder_chosen[rname]
        models_fv[rname] = fit_offset_logreg(X_fv, [feat_map[f] for f in flist], flist, y_fv, w_fv, off_fv, l2=best_l2)

    # Fit both Qn variants (withqn and noqn) at chosen_preT_rung and R5
    mod_withqn = fit_offset_logreg(
        X_fv, [feat_map[f] for f in ladder_withqn[chosen_preT_rung]], ladder_withqn[chosen_preT_rung], y_fv, w_fv, off_fv, l2=best_l2
    )
    mod_noqn = fit_offset_logreg(
        X_fv, [feat_map[f] for f in ladder_noqn[chosen_preT_rung]], ladder_noqn[chosen_preT_rung], y_fv, w_fv, off_fv, l2=best_l2
    )

    # Baselines for V1: follows_only (FOL + offset) & popularity_offset (PFOL + PORIG + offset)
    mod_fol_only = fit_offset_logreg(X_fv, [feat_map["FOL"]], ["FOL"], y_fv, w_fv, off_fv, l2=best_l2)
    mod_pop_off = fit_offset_logreg(X_fv, [feat_map["PFOL"], feat_map["PORIG"]], ["PFOL", "PORIG"], y_fv, w_fv, off_fv, l2=best_l2)

    # Serving-compatible variant theta_serve: FOL, LO_L14, LR_L14, LU_L14, RP_L14
    serve_feats = ["FOL", "LO_L14", "LR_L14", "LU_L14", "RP_L14"]
    mod_serve = fit_offset_logreg(X_fv, [feat_map[f] for f in serve_feats], serve_feats, y_fv, w_fv, off_fv, l2=best_l2)

    # LightGBM / HistGradientBoosting check on subsample of X_fv (1.5M rows)
    rng_gb = np.random.default_rng(20261028)
    n_sub = min(1_500_000, len(y_fv))
    sub_idx = rng_gb.choice(len(y_fv), size=n_sub, replace=False)
    X_gb = np.column_stack([X_fv[sub_idx][:, idx_preT], off_fv[sub_idx]])
    gbdt = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=31, learning_rate=0.05, random_state=20261028)
    gbdt.fit(X_gb, y_fv[sub_idx].astype(int), sample_weight=w_fv[sub_idx])
    del X_gb

    # Calibrate chosen pre-T model on T_val heldout==1 (still <= T!)
    mod_preT_final = models_fv[chosen_preT_rung]
    raw_val_ho = mod_preT_final.predict_proba(t_val["X"][val_ho_m], o_ref_tval, calibrated=False)
    iso_final = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
    iso_final.fit(raw_val_ho, t_val["y_a"][val_ho_m], sample_weight=1.0 / t_val["pi"][val_ho_m])
    mod_preT_final.calibrator = iso_final
    del t_val, X_fv, y_fv, w_fv, off_fv

    # 5. Report-Once V1 & V2 Evaluation on T_test (heldout == 1)
    print("[s11] Scoring V1 and V2 on T_test (heldout == 1)...", flush=True)
    X_te = t_test_ho["X"]
    off_te = t_test_ho["offset"]
    y_te = t_test_ho["y_a"]
    w_te = 1.0 / t_test_ho["pi"]
    u_te = t_test_ho["u"]

    X_te_gb = np.column_stack([X_te[:, idx_preT], off_te])
    sc_gbdt = gbdt.predict_proba(X_te_gb)[:, 1]
    del X_te_gb

    arm_scores_te = {
        "follows_only": mod_fol_only.predict_proba(X_te, off_te),
        "popularity_offset": mod_pop_off.predict_proba(X_te, off_te),
        "R1": models_fv["R1"].predict_proba(X_te, off_te),
        "R2": models_fv["R2"].predict_proba(X_te, off_te),
        "R3": models_fv["R3"].predict_proba(X_te, off_te),
        "R4": models_fv["R4"].predict_proba(X_te, off_te),
        "R5": models_fv["R5"].predict_proba(X_te, off_te),
        "R5_full_E": models_fv["R5_full_E"].predict_proba(X_te, off_te),
        "chosen_model": mod_preT_final.predict_proba(X_te, off_te),
        "chosen_fixed_oref": mod_preT_final.predict_proba(X_te, float(np.median(off_te))),
        "withqn": mod_withqn.predict_proba(X_te, off_te),
        "noqn": mod_noqn.predict_proba(X_te, off_te),
        "theta_serve": mod_serve.predict_proba(X_te, off_te),
        "gbdt_check": sc_gbdt,
    }

    comps_v1_v2 = [
        ("chosen_model", "follows_only"),
        ("chosen_model", "popularity_offset"),
        ("R2", "R1"),
        ("R3", "R2"),
        ("R4", "R3"),
        ("R5", "R4"),
        ("R5_full_E", "R4"),
        ("withqn", "noqn"),
        ("theta_serve", "follows_only"),
        ("gbdt_check", "chosen_model"),
    ]

    nonmut_te = X_te[:, feat_map["MUT"]] == 0
    v1_primary = evaluate_slice_ap(u_te, y_te, w_te, arm_scores_te, nonmut_te, comps_v1_v2)
    v1_all = evaluate_slice_ap(u_te, y_te, w_te, arm_scores_te, np.ones(len(u_te), dtype=bool), comps_v1_v2)

    # Required V1 slices: discovery, follower bucket, light/heavy, language, supplement
    small_arms = {
        k: arm_scores_te[k] for k in ("follows_only", "popularity_offset", "chosen_model", "theta_serve")
    }
    small_comps = [("chosen_model", "follows_only"), ("chosen_model", "popularity_offset")]
    v1_slices = {
        "non_mutual_primary": v1_primary,
        "all_pairs": v1_all,
        "discovery_nonfollowed": evaluate_slice_ap(u_te, y_te, w_te, small_arms, X_te[:, feat_map["FOL"]] == 0, small_comps),
        "light_users": evaluate_slice_ap(u_te, y_te, w_te, small_arms, X_te[:, feat_map["HEAVY"]] == 0, small_comps),
        "heavy_users": evaluate_slice_ap(u_te, y_te, w_te, small_arms, X_te[:, feat_map["HEAVY"]] == 1, small_comps),
    }
    pfol_cnt = np.expm1(X_te[:, feat_map["PFOL"]])
    for bname, lo_f, hi_f in (("fol_lt_1k", 0, 1000), ("fol_1k_10k", 1000, 10000), ("fol_10k_100k", 10000, 100000), ("fol_ge_100k", 100000, 1e12)):
        v1_slices[bname] = evaluate_slice_ap(u_te, y_te, w_te, small_arms, (pfol_cnt >= lo_f) & (pfol_cnt < hi_f), small_comps)

    # Producer top language & supplement slice from O2/prod
    prod_lang = [""] * len(full_dids)
    prod_sup = np.zeros(len(full_dids), dtype=np.uint8)
    for p_pq in sorted((args.s10_dir / "O2/prod").glob("j*.parquet")):
        tl = pq.read_table(p_pq, columns=["p", "top_langs", "supplement"])
        for pv, lg, sv in zip(tl["p"].to_pylist(), tl["top_langs"].to_pylist(), tl["supplement"].to_pylist()):
            prod_lang[pv] = lg[0] if lg else ""
            prod_sup[pv] = int(sv or 0)
    te_langs = np.array([prod_lang[int(p)] for p in t_test_ho["p"]])
    from collections import Counter
    top5_langs = [lg for lg, _ in Counter(x for x in te_langs if x).most_common(5)]
    for lg in top5_langs:
        v1_slices[f"lang_{lg}"] = evaluate_slice_ap(u_te, y_te, w_te, small_arms, te_langs == lg, small_comps)
    v1_slices["supplement_producers"] = {
        "rows": int((prod_sup[t_test_ho["p"]] == 1).sum()),
        "supplement_producers_in_O2": int(prod_sup.sum()),
    }

    v1_pass = bool(
        v1_primary["diffs"]["chosen_model_minus_follows_only"]["pass_gt_0"]
        and v1_primary["diffs"]["chosen_model_minus_popularity_offset"]["pass_gt_0"]
    )

    # Check V2 rung progression on T_test primary slice
    v2_rungs = {}
    highest_v2_rung = "R1"
    for rk, rprev in (("R2", "R1"), ("R3", "R2"), ("R4", "R3"), ("R5", "R4")):
        d = v1_primary["diffs"][f"{rk}_minus_{rprev}"]
        v2_rungs[f"{rk}_vs_{rprev}"] = d
        if d["pass_gt_0"]:
            highest_v2_rung = rk

    # 6. Run V3, V4, V5, V6
    print("[s11] Running V3, V4, V5, V6 outcome checks...", flush=True)
    v3_v6 = run_v3_v4_v5_v6(t_test_ho, feat_map, args.s10_dir, args.s3_dir, purge, exclude_s3, scales, full_dids)
    del t_test_ho

    # 7. Fit Task B' and Task C'
    print("[s11] Fitting Task B' (engaged follow) and Task C' (persistent negative outcome)...", flush=True)
    b_fit = load_o3a_split(args.s10_dir / "O3a/T_fit", feature_names, task_filter=2, heldout_filter=0, require_elig_a=False)
    b_val = load_o3a_split(args.s10_dir / "O3a/T_val", feature_names, task_filter=2, heldout_filter=0, require_elig_a=False)
    b_te = load_o3a_split(args.s10_dir / "O3a/T_test", feature_names, task_filter=2, heldout_filter=1, require_elig_a=False)
    xb_tr = np.concatenate([b_fit["X"][b_fit["y_b_valid"]], b_val["X"][b_val["y_b_valid"]]], axis=0)
    yb_tr = np.concatenate([b_fit["y_b"][b_fit["y_b_valid"]], b_val["y_b"][b_val["y_b_valid"]]], axis=0)
    wb_tr = np.ones(len(yb_tr), dtype=np.float64)
    b_feats = [f for f in ladder_chosen["R4"] if f not in ("FOL", "FAGE", "FBURST")]
    mod_b = fit_offset_logreg(xb_tr, [feat_map[f] for f in b_feats], b_feats, yb_tr, wb_tr, None, l2=1e-3)
    mod_b_pop = fit_offset_logreg(xb_tr, [feat_map["PFOL"], feat_map["PORIG"]], ["PFOL", "PORIG"], yb_tr, wb_tr, None, l2=1e-3)
    mb_te = b_te["y_b_valid"]
    task_b_eval = evaluate_slice_ap(
        b_te["u"],
        b_te["y_b"],
        np.ones(len(b_te["u"])),
        {"task_b_model": mod_b.predict_proba(b_te["X"], 0.0), "popularity": mod_b_pop.predict_proba(b_te["X"], 0.0)},
        mb_te,
        [("task_b_model", "popularity")],
    )
    del b_fit, b_val, b_te, xb_tr, yb_tr

    # Task C': ck in (1, 2) (pairs with interaction in window), leakage trap 11 excludes same-window block features
    c_fit = load_o3a_split(args.s10_dir / "O3a/T_fit", feature_names, task_filter=1, heldout_filter=0, require_elig_a=False)
    c_te = load_o3a_split(args.s10_dir / "O3a/T_test", feature_names, task_filter=1, heldout_filter=1, require_elig_a=False)
    mc_fit = np.isin(c_fit["ck"], [1, 2])
    mc_te = np.isin(c_te["ck"], [1, 2])
    c_feats = [f for f in ladder_chosen["R4"] if f not in ("QH_L45", "RH_L45", "BLK", "LBLK", "PBLK", "PBLKN")]
    yc_fit = c_fit["y_c"][mc_fit].astype(np.float64)
    pos_wt = max(1.0, float((len(yc_fit) - yc_fit.sum()) / max(yc_fit.sum(), 1.0)))
    wc_fit = (1.0 / c_fit["pi"][mc_fit]) * np.where(yc_fit > 0.5, pos_wt, 1.0)
    mod_c = fit_offset_logreg(c_fit["X"][mc_fit], [feat_map[f] for f in c_feats], c_feats, yc_fit, wc_fit, None, l2=1e-3)
    mod_c_pop = fit_offset_logreg(c_fit["X"][mc_fit], [feat_map["PFOL"], feat_map["PORIG"]], ["PFOL", "PORIG"], yc_fit, wc_fit, None, l2=1e-3)
    task_c_eval = evaluate_slice_ap(
        c_te["u"],
        c_te["y_c"].astype(np.float64),
        1.0 / c_te["pi"],
        {"task_c_model": mod_c.predict_proba(c_te["X"], 0.0), "popularity": mod_c_pop.predict_proba(c_te["X"], 0.0)},
        mc_te,
        [("task_c_model", "popularity")],
    )
    del c_fit, c_te

    # 8. Emit W_lin(p) for eval (at T, using strictly <=T model) and prod (at T_now)
    print("[s11] Computing exact W_lin(p) on O2/eval and O2/prod...", flush=True)
    wlin_alpha = float(norm_sel["chosen_wlin_alpha"])
    wlin_eval_meta = compute_and_write_wlin(
        args.s10_dir / "O2/eval",
        args.s10_dir / "O2/lu_discounted_eval.parquet",
        mod_preT_final,
        wlin_alpha,
        full_dids,
        n_actors,
        args.out_dir / "w_lin_eval.parquet",
    )
    # Prod model uses highest validated rung from V2 (or preT model if same)
    mod_prod_final = models_fv[highest_v2_rung]
    wlin_prod_meta = compute_and_write_wlin(
        args.s10_dir / "O2/prod",
        args.s10_dir / "O2/lu_discounted_prod.parquet",
        mod_prod_final,
        wlin_alpha,
        full_dids,
        n_actors,
        args.out_dir / "w_lin_prod.parquet",
    )

    # 9. Run V8(i) on M1 (cand5-T-matched)
    print("[s11] Running V8(i) on M1 (cand5-T-matched)...", flush=True)
    v8_m1 = run_v8_m1(
        mod_preT_final,
        wlin_eval_meta["o_ref"],
        norm_sel["chosen_ii_norm"],
        args.s10_dir,
        args.s3_dir,
        purge,
        exclude_s3,
        feature_names,
        scales,
        full_dids,
        n_actors,
        workers=args.workers,
    )

    # Save fitted weight vectors
    weights_json = {
        "chosen_qn_variant": chosen_qn,
        "v7_precision_pass": v7_prec_pass,
        "chosen_preT_rung": chosen_preT_rung,
        "highest_v2_rung_on_ttest": highest_v2_rung,
        "l2": best_l2,
        "half_lives": list(best_hls),
        "preT_model": {
            "theta0": mod_preT_final.theta0,
            "beta_offset": mod_preT_final.beta,
            "weights": dict(zip(mod_preT_final.feat_names, [float(x) for x in mod_preT_final.theta])),
        },
        "prod_model": {
            "theta0": mod_prod_final.theta0,
            "beta_offset": mod_prod_final.beta,
            "weights": dict(zip(mod_prod_final.feat_names, [float(x) for x in mod_prod_final.theta])),
        },
        "theta_serve": {
            "theta0": mod_serve.theta0,
            "beta_offset": mod_serve.beta,
            "weights": dict(zip(mod_serve.feat_names, [float(x) for x in mod_serve.theta])),
            "deployment_status": "evaluated_only_pending_D7",
        },
    }
    (args.out_dir / "weights.json").write_text(json.dumps(weights_json, indent=2))

    receipt = {
        "step": "S11_affinity_weights",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 2),
        "chosen_qn_variant": chosen_qn,
        "v7_precision_pass": v7_prec_pass,
        "t_val_selection": {
            "l2_grid": l2_grid,
            "best_l2": best_l2,
            "half_life_grid": hl_grid,
            "best_half_lives": best_hl_name,
            "tval_rung_ap": tval_rung_ap,
            "chosen_preT_rung": chosen_preT_rung,
            "normalisation": norm_sel,
        },
        "V1_task_a_ttest": {
            "pass": v1_pass,
            "slices": v1_slices,
        },
        "V2_ablation_ladder_ttest": {
            "highest_passing_rung": highest_v2_rung,
            "rungs": v2_rungs,
        },
        **v3_v6,
        "Task_B_prime": task_b_eval,
        "Task_C_prime": task_c_eval,
        "W_lin": {
            "eval_at_T": wlin_eval_meta,
            "prod_at_T_now": wlin_prod_meta,
        },
        "V8_i_M1": v8_m1,
    }
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(f"[s11] Done in {receipt['elapsed_s']}s. V1={v1_pass}, V2={highest_v2_rung}, V8(i)={v8_m1['v8_i_pass']}", flush=True)


if __name__ == "__main__":
    main()
