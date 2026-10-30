#!/usr/bin/env python3
"""Step S16 Assembly & Report-Once Evaluation:
1. Tier-1 projection head (256->512->256 residual MLP on MPS, 0 units) + Y10 dunk-negative ablation
2. Gates G-T1, G-T2 trigger, G-slab, and G-img trigger (A2 report-only)
3. M2-test REPORT-ONCE (locked S13 (k=3, m=3, theta=0.5, w_lin_plus_curators) vs Base S3 & S13 community_popularity, V8(ii), slices) + M1 follows_le20 diagnostic
4. B2 report-once (partition macro top-1 & content-kNN macro top-1)
5. Top-10 language Y3 sensitivity-coverage loss table & R13 trap status
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import eval_temporal as et  # noqa: E402
import s3_prep  # noqa: E402
from s13_elig import run_s13_grid as s13  # noqa: E402
from s6picker import keywords as s6_kw  # noqa: E402

B1_SEED = 20261028
B1_BOOT = 2000
B1_LEVEL = 0.9875
Y3_CLASSES = (
    "news",
    "politics",
    "religion",
    "sexuality_gender",
    "health",
    "ethnicity_nationality",
    "adult",
    "likely_minors",
)


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def read_filtered_lists(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    """Exact R11 list filter (matching eval_temporal.b1a_lists & s4_pilot/score.py), preserving list URI."""
    rng = np.random.default_rng(7)
    rows = sorted(
        (uri, owner, name, sorted(set(members)))
        for uri, owner, name, members, total in s3_prep.read_lists(path, with_total=True)
        if max(total, len(set(members))) <= 500
    )
    seen, by_owner = set(), collections.defaultdict(list)
    for r in rows:
        key = tuple(r[3])
        if len(key) >= 3 and key not in seen:
            seen.add(key)
            by_owner[r[1]].append(r)
    out = []
    for rs in by_owner.values():
        for idx in sorted(rng.choice(len(rs), min(3, len(rs)), replace=False)):
            out.append((rs[idx][0], rs[idx][3]))
    return out


def exact_nn_for_queries(
    corpus_vecs: np.ndarray,
    query_indices: np.ndarray,
    device: torch.device,
    chunk: int = 1024,
) -> np.ndarray:
    """For each row index in `query_indices` (into `corpus_vecs`), return its nearest other corpus index by cosine."""
    X = torch.from_numpy(np.ascontiguousarray(corpus_vecs, dtype=np.float32)).to(device)
    out = np.empty(len(query_indices), dtype=np.int64)
    q_arr = np.asarray(query_indices, dtype=np.int64)
    for s in range(0, len(q_arr), chunk):
        qb = q_arr[s : s + chunk]
        q_t = torch.from_numpy(qb).to(device)
        sims = X[q_t] @ X.T
        sims[torch.arange(len(qb), device=device), q_t] = -float("inf")
        out[s : s + chunk] = sims.argmax(dim=1).cpu().numpy()
    return out


def boot_list_ci(vals: np.ndarray, level: float, seed: int = B1_SEED, n_boot: int = B1_BOOT) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(vals)
    boots = np.empty(n_boot, dtype=np.float64)
    for b0 in range(0, n_boot, 100):
        nb = min(100, n_boot - b0)
        draws = rng.integers(0, n, (nb, n))
        boots[b0 : b0 + nb] = vals[draws].mean(axis=1)
    a = (1.0 - level) / 2.0
    return [float(np.quantile(boots, a)), float(np.quantile(boots, 1.0 - a))]


def eval_knn_lists_on_universe(
    lists: list[tuple[str, list[str]]],
    did_to_u: dict[str, int],
    vecs_by_arm: dict[str, np.ndarray],
    device: torch.device,
    with_null_for: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], list[str], int]:
    """Compute per-list kNN macro top-1 across arms on a shared universe (len = V)."""
    kept_uris = []
    groups = []
    q_set = set()
    for uri, members in lists:
        g = np.array([did_to_u[d] for d in members if d in did_to_u], dtype=np.int64)
        if len(g) >= 3:
            kept_uris.append(uri)
            groups.append(g)
            q_set.update(g.tolist())

    n_u = len(did_to_u)
    if not groups:
        return {}, [], 0

    # If we also compute same-size null, add random-group members to q_set
    null_groups = None
    if with_null_for:
        rng_null = np.random.default_rng(B1_SEED)
        null_groups = [rng_null.choice(n_u, len(g), replace=False) for g in groups]
        for ng in null_groups:
            q_set.update(ng.tolist())

    q_sorted = np.array(sorted(q_set), dtype=np.int64)
    out_hits: dict[str, np.ndarray] = {}
    for arm, V in vecs_by_arm.items():
        nn_q = exact_nn_for_queries(V, q_sorted, device=device)
        nn_map = np.full(n_u, -1, dtype=np.int64)
        nn_map[q_sorted] = nn_q
        hits = np.empty(len(groups), dtype=np.float64)
        for i, g in enumerate(groups):
            s = set(g.tolist())
            hits[i] = float(np.mean([nn_map[m] in s for m in g]))
        out_hits[arm] = hits
        if with_null_for and arm in with_null_for and null_groups is not None:
            nhits = np.empty(len(null_groups), dtype=np.float64)
            for i, ng in enumerate(null_groups):
                s = set(ng.tolist())
                nhits[i] = float(np.mean([nn_map[m] in s for m in ng]))
            out_hits[f"{arm}__null"] = nhits

    return out_hits, kept_uris, int(sum(len(g) for g in groups))


class Tier1Head(nn.Module):
    """Pre-registered Tier-1 projection head: 2-layer MLP 256 -> 512 -> 256 anchored to stock vector."""

    def __init__(self, d_in: int = 256, d_hid: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_in)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fc2(F.gelu(self.fc1(x)))
        return F.normalize(x + delta, p=2, dim=-1)


def train_tier1_head(
    slab_pids: np.ndarray,
    slab_vecs: np.ndarray,
    full_dids: list[str],
    memb_for_resample: np.ndarray,
    o4_parquet: pathlib.Path,
    s16a_parquet: pathlib.Path,
    train_exclude_dids: set[str],
    heldout_t1_mask_full: np.ndarray,
    val_anchor_mask_full: np.ndarray,
    device: torch.device,
    extra_neg_pairs_full: list[tuple[int, int]] | None = None,
    fixed_lam: float | None = None,
    fixed_epochs: int | None = None,
) -> tuple[Tier1Head, dict]:
    """Train Tier-1 head excluding heldout_t1 (10%) and train_exclude_dids (hide_from_recs, optout, purge, rule2)."""
    t0 = time.time()
    n_full = len(full_dids)
    pid_to_row = np.full(n_full, -1, dtype=np.int32)
    valid_slab = slab_pids < n_full
    pid_to_row[slab_pids[valid_slab]] = np.arange(len(slab_pids), dtype=np.int32)[valid_slab]

    allowed_full = np.array([d not in train_exclude_dids for d in full_dids], dtype=bool) & (~heldout_t1_mask_full) & (pid_to_row >= 0)

    # Read O4 top-20 positives per anchor
    t_o4 = pq.read_table(o4_parquet, columns=["p", "q", "w"])
    p_o4 = t_o4["p"].to_numpy().astype(np.int64)
    q_o4 = t_o4["q"].to_numpy().astype(np.int64)
    w_o4 = t_o4["w"].to_numpy().astype(np.float32)
    ok_pair = (p_o4 < n_full) & (q_o4 < n_full)
    p_o4, q_o4, w_o4 = p_o4[ok_pair], q_o4[ok_pair], w_o4[ok_pair]
    ok_allow = allowed_full[p_o4] & allowed_full[q_o4] & (p_o4 != q_o4)
    p_o4, q_o4, w_o4 = p_o4[ok_allow], q_o4[ok_allow], w_o4[ok_allow]

    # Sort by (p, -w) and keep top-20 per anchor
    ord_o4 = np.lexsort((-w_o4, p_o4))
    p_o4, q_o4 = p_o4[ord_o4], q_o4[ord_o4]
    u_p, idx_start, counts = np.unique(p_o4, return_index=True, return_counts=True)

    pos_by_anchor: dict[int, np.ndarray] = {}
    for ap, st, cnt in zip(u_p.tolist(), idx_start.tolist(), counts.tolist()):
        pos_by_anchor[ap] = pid_to_row[q_o4[st : st + min(cnt, 20)]]

    # Read S16a zero_shared_any1 hard negatives
    t_hn = pq.read_table(s16a_parquet, columns=["p", "q", "zero_shared_any1"])
    p_hn = t_hn["p"].to_numpy().astype(np.int64)
    q_hn = t_hn["q"].to_numpy().astype(np.int64)
    z_hn = t_hn["zero_shared_any1"].to_numpy().astype(bool)
    ok_hn = z_hn & (p_hn < n_full) & (q_hn < n_full)
    p_hn, q_hn = p_hn[ok_hn], q_hn[ok_hn]
    ok_hn2 = allowed_full[p_hn] & allowed_full[q_hn] & (p_hn != q_hn)
    p_hn, q_hn = p_hn[ok_hn2], q_hn[ok_hn2]

    hn_lists: dict[int, list[int]] = collections.defaultdict(list)
    for ap, nq in zip(p_hn.tolist(), pid_to_row[q_hn].tolist()):
        hn_lists[ap].append(nq)

    extra_neg_added = 0
    if extra_neg_pairs_full:
        for a_f, b_f in extra_neg_pairs_full:
            if a_f < n_full and b_f < n_full and allowed_full[a_f] and allowed_full[b_f] and a_f != b_f:
                ra, rb = int(pid_to_row[a_f]), int(pid_to_row[b_f])
                if a_f in pos_by_anchor:
                    pos_set = set(pos_by_anchor[a_f].tolist())
                    if rb not in pos_set:
                        hn_lists[a_f].append(rb)
                        extra_neg_added += 1
                if b_f in pos_by_anchor:
                    pos_set = set(pos_by_anchor[b_f].tolist())
                    if ra not in pos_set:
                        hn_lists[b_f].append(ra)
                        extra_neg_added += 1

    train_anchors = [ap for ap in pos_by_anchor if not val_anchor_mask_full[ap]]
    val_anchors = [ap for ap in pos_by_anchor if val_anchor_mask_full[ap]]

    # Community size^0.7 resampling weights for train_anchors (per anchor weight ~ size^(0.7 - 1) = size^-0.3)
    c_max = int(max(memb_for_resample.max(), 0)) + 1
    comm_sizes = np.bincount(memb_for_resample[memb_for_resample >= 0], minlength=c_max)
    anchor_w = np.empty(len(train_anchors), dtype=np.float64)
    for i, ap in enumerate(train_anchors):
        c = int(memb_for_resample[ap])
        sz = int(comm_sizes[c]) if c >= 0 else 25
        anchor_w[i] = float(max(sz, 10)) ** (-0.3)
    anchor_prob = anchor_w / anchor_w.sum()

    X_gpu = torch.from_numpy(l2_norm(slab_vecs.astype(np.float32))).to(device)
    n_slab = len(slab_vecs)

    # Prepack padded positive & hard-negative index arrays for fast vectorized sampling
    max_pos = 20
    max_hn = 16
    tr_a_rows = np.array([pid_to_row[ap] for ap in train_anchors], dtype=np.int64)
    tr_pos_mat = np.zeros((len(train_anchors), max_pos), dtype=np.int64)
    tr_pos_len = np.zeros(len(train_anchors), dtype=np.int64)
    tr_hn_mat = np.full((len(train_anchors), max_hn), -1, dtype=np.int64)
    tr_hn_len = np.zeros(len(train_anchors), dtype=np.int64)
    for i, ap in enumerate(train_anchors):
        pa_r = pos_by_anchor[ap]
        tr_pos_len[i] = len(pa_r)
        tr_pos_mat[i, : len(pa_r)] = pa_r
        hl = hn_lists.get(ap)
        if hl:
            hl_u = hl[:max_hn]
            tr_hn_len[i] = len(hl_u)
            tr_hn_mat[i, : len(hl_u)] = hl_u

    # Build fixed teacher-val evaluation triplets (anchor_row, pos_row, up to 8 hn_rows)
    rng_v = np.random.default_rng(20261030)
    val_sub = val_anchors if len(val_anchors) <= 8000 else [val_anchors[i] for i in rng_v.choice(len(val_anchors), 8000, replace=False)]
    val_a_rows = np.array([pid_to_row[ap] for ap in val_sub], dtype=np.int64)
    val_q_rows = np.array([int(pos_by_anchor[ap][0]) for ap in val_sub], dtype=np.int64)

    def eval_teacher_val(model: Tier1Head) -> float:
        model.eval()
        with torch.no_grad():
            va = model(X_gpu[torch.from_numpy(val_a_rows).to(device)])
            vq = model(X_gpu[torch.from_numpy(val_q_rows).to(device)])
            # Chunked 512-way InfoNCE retrieval accuracy on teacher-val pairs
            hits = 0
            tot = len(val_sub)
            for s in range(0, tot, 512):
                e = min(tot, s + 512)
                if e - s < 32:
                    continue
                logits = (va[s:e] @ vq[s:e].T) / 0.1
                pred = logits.argmax(dim=1)
                hits += int((pred == torch.arange(e - s, device=device)).sum().item())
            return hits / max(tot, 1)

    tau = 0.1
    batch_size = 512
    steps_per_epoch = max(1, len(train_anchors) // batch_size)
    lams = [fixed_lam] if fixed_lam is not None else [0.1, 0.25, 0.5]
    max_epochs = fixed_epochs if fixed_epochs is not None else 4

    best_val_acc = -1.0
    best_state = None
    best_cfg = {}
    val_grid_log = []

    for lam in lams:
        torch.manual_seed(20261030)
        rng_tr = np.random.default_rng(20261030)
        model = Tier1Head(256, 512).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        for ep in range(1, max_epochs + 1):
            model.train()
            for _ in range(steps_per_epoch):
                b_idx = rng_tr.choice(len(train_anchors), size=batch_size, replace=False, p=anchor_prob)
                a_r = tr_a_rows[b_idx]
                p_offsets = (rng_tr.random(batch_size) * tr_pos_len[b_idx]).astype(np.int64)
                p_r = tr_pos_mat[b_idx, p_offsets]

                # Gather up to 4 explicit S16a hard negatives per anchor in the batch
                hn_mask_b = tr_hn_len[b_idx] > 0
                hn_r_list = []
                for k_neg in range(4):
                    sel_i = b_idx[hn_mask_b]
                    if len(sel_i):
                        offs = (k_neg + rng_tr.integers(0, 16, size=len(sel_i))) % tr_hn_len[sel_i]
                        hn_r_list.append(tr_hn_mat[sel_i, offs])
                hn_flat = np.unique(np.concatenate(hn_r_list)) if hn_r_list else np.zeros(0, dtype=np.int64)
                hn_flat = hn_flat[hn_flat >= 0]

                xa_raw = X_gpu[torch.from_numpy(a_r).to(device)]
                xp_raw = X_gpu[torch.from_numpy(p_r).to(device)]
                za = model(xa_raw)
                zp = model(xp_raw)

                if len(hn_flat):
                    xhn_raw = X_gpu[torch.from_numpy(hn_flat).to(device)]
                    zhn = model(xhn_raw)
                    all_targets = torch.cat([zp, zhn], dim=0)
                else:
                    all_targets = zp

                logits = (za @ all_targets.T) / tau
                # Mask accidental self-collisions in targets
                if len(hn_flat):
                    all_t_rows = np.concatenate([p_r, hn_flat])
                else:
                    all_t_rows = p_r
                same_mask = torch.from_numpy(a_r[:, None] == all_t_rows[None, :]).to(device)
                same_mask[:, :batch_size].fill_diagonal_(False)
                logits = logits.masked_fill(same_mask, -1e4)

                targets = torch.arange(batch_size, device=device)
                loss_nce = F.cross_entropy(logits, targets)
                loss_anc = ((za - xa_raw).pow(2).sum(dim=-1).mean() + (zp - xp_raw).pow(2).sum(dim=-1).mean()) * 0.5
                loss = loss_nce + float(lam) * loss_anc

                opt.zero_grad()
                loss.backward()
                opt.step()

            val_acc = eval_teacher_val(model)
            val_grid_log.append({"lambda": float(lam), "epoch": ep, "teacher_val_acc512": round(val_acc, 6)})
            if fixed_lam is not None and fixed_epochs is not None:
                if ep == fixed_epochs:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_cfg = {"lambda": float(lam), "epoch": ep, "tau": tau}
            elif val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_cfg = {"lambda": float(lam), "epoch": ep, "tau": tau}

    assert best_state is not None
    final_model = Tier1Head(256, 512).to(device)
    final_model.load_state_dict(best_state)
    final_model.eval()
    stock_val_acc = 0.0
    with torch.no_grad():
        va0 = X_gpu[torch.from_numpy(val_a_rows).to(device)]
        vq0 = X_gpu[torch.from_numpy(val_q_rows).to(device)]
        hits0 = 0
        for s in range(0, len(val_sub), 512):
            e = min(len(val_sub), s + 512)
            if e - s < 32:
                continue
            hits0 += int(((va0[s:e] @ vq0[s:e].T).argmax(dim=1) == torch.arange(e - s, device=device)).sum().item())
        stock_val_acc = hits0 / max(len(val_sub), 1)

    meta = {
        "train_anchors": len(train_anchors),
        "val_anchors": len(val_anchors),
        "anchors_with_s16a_hardnegs": int((tr_hn_len > 0).sum()),
        "extra_dunk_neg_edges_added": extra_neg_added,
        "stock_teacher_val_acc512": round(stock_val_acc, 6),
        "best_head_teacher_val_acc512": round(best_val_acc, 6),
        "chosen_hyperparams": best_cfg,
        "val_grid": val_grid_log,
        "train_seconds": round(time.time() - t0, 2),
    }
    return final_model, meta


def project_all_vecs(model: Tier1Head, vecs: np.ndarray, device: torch.device, batch: int = 16384) -> np.ndarray:
    model.eval()
    out = np.empty((len(vecs), vecs.shape[1]), dtype=np.float32)
    v_norm = l2_norm(vecs.astype(np.float32))
    with torch.no_grad():
        for s in range(0, len(v_norm), batch):
            xb = torch.from_numpy(v_norm[s : s + batch]).to(device)
            out[s : s + batch] = model(xb).cpu().numpy()
    return out


def compute_card_coherence(
    part_primary: np.ndarray,
    n_comm: int,
    elig_mask: np.ndarray,
    pid_to_vec_row: np.ndarray,
    vecs_norm: np.ndarray,
) -> np.ndarray:
    """Cosine of each producer to its primary community's eligible centroid (median fill where missing)."""
    coh = np.full(len(part_primary), np.nan, dtype=np.float64)
    has_v = (part_primary >= 0) & elig_mask & (pid_to_vec_row >= 0)
    p_idx = np.flatnonzero(has_v)
    c_ids = part_primary[p_idx]
    v_rows = pid_to_vec_row[p_idx]
    V = vecs_norm[v_rows].astype(np.float32)

    centroids = np.zeros((n_comm, V.shape[1]), dtype=np.float32)
    np.add.at(centroids, c_ids, V)
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    has_c = (c_norms[:, 0] > 1e-8)
    centroids[has_c] /= c_norms[has_c]

    # Score every producer with a vector and a valid community centroid
    all_v = np.flatnonzero((part_primary >= 0) & (pid_to_vec_row >= 0))
    all_c = part_primary[all_v]
    ok_c = has_c[all_c]
    all_v = all_v[ok_c]
    all_c = all_c[ok_c]
    dots = (vecs_norm[pid_to_vec_row[all_v]] * centroids[all_c]).sum(axis=1)
    coh[all_v] = np.clip(dots, 0.0, 1.0)

    med = float(np.nanmedian(coh[elig_mask & (part_primary >= 0)])) if np.isfinite(coh).any() else 0.5
    return np.where(np.isfinite(coh), coh, med)


def score_m2_split_full_slices(
    part: et.Partition,
    uni: et.Universe,
    m2_data: et.M2Data,
    split_name: str,
    elig_mask: np.ndarray,
    raw_comm_flagged: np.ndarray,
    memb_T: np.ndarray,
    comm_flagged: np.ndarray,
    sup_mask: np.ndarray,
    rankers: dict[str, np.ndarray],
    high_h_pairs: set[tuple[int, int]],
    co_off: np.ndarray,
    co_dst: np.ndarray,
    co_cnt: np.ndarray,
) -> dict:
    """Exact M2 evaluation on a single split ('val' or 'test') with all pre-registered slices."""
    cand_ok = (part.primary >= 0) & uni.eligible & elig_mask & (~raw_comm_flagged[np.maximum(memb_T, 0)])
    members = np.flatnonzero(cand_ok)
    off, _ = et.group(part.primary[members], part.n_comm)
    card_ok = (np.diff(off) >= et.CARD_MIN) & (~comm_flagged)

    top_per_card: dict[str, list[list[int]]] = {}
    for rname, key in rankers.items():
        order = np.lexsort((uni.tiebreak[members], -key[members], part.primary[members]))
        ranked_mem = members[order]
        top_per_card[rname] = [
            ranked_mem[int(off[c]) : min(int(off[c + 1]), int(off[c]) + 45)].tolist()
            if off[c + 1] > off[c]
            else []
            for c in range(part.n_comm)
        ]

    split_users = np.flatnonzero(m2_data.split == split_name)
    ou, oi = m2_data.outcomes
    follow_out: dict[int, set[int]] = {}
    for u_i, p_i in zip(ou.tolist(), oi.tolist()):
        if m2_data.split[u_i] == split_name:
            follow_out.setdefault(u_i, set()).add(p_i)
    engaged_out: dict[int, set[int]] = {}
    if m2_data.engaged is not None:
        for u_i, p_i in zip(m2_data.engaged[0].tolist(), m2_data.engaged[1].tolist()):
            if m2_data.split[u_i] == split_name:
                engaged_out.setdefault(u_i, set()).add(p_i)

    picks_u: dict[int, list[int]] = {}
    for u in split_users.tolist():
        s_arr = m2_data.seeds(u)
        if len(s_arr) == 0:
            continue
        mass = part.interest(s_arr)
        cand_c = np.flatnonzero((mass > 0) & card_ok)
        if len(cand_c):
            picks_u[u] = [int(c) for c in cand_c[np.lexsort((cand_c, -mass[cand_c]))][: et.N_PICKS]]

    eval_u = np.array([u for u in split_users.tolist() if (u in picks_u) and (u in follow_out)], dtype=np.int64)
    fol_arr = uni.followers
    pop_lim, pop_max = et.POP_CEILING

    fill_cache: dict[tuple[int, ...], list[int]] = {}

    def get_fill_nbs(pk: tuple[int, ...]) -> list[int]:
        res = fill_cache.get(pk)
        if res is not None:
            return res
        sc: dict[int, int] = {}
        pk_set = set(pk)
        for c in pk:
            for nb, cnt in zip(co_dst[co_off[c] : co_off[c + 1]], co_cnt[co_off[c] : co_off[c + 1]]):
                nb_i = int(nb)
                if nb_i in pk_set or not card_ok[nb_i]:
                    continue
                if any((c, nb_i) in high_h_pairs for c in pk):
                    continue
                sc[nb_i] = sc.get(nb_i, 0) + int(cnt)
        res = sorted(sc, key=lambda x: (-sc[x], x))
        fill_cache[pk] = res
        return res

    vecs_by_r: dict[str, dict[str, np.ndarray]] = {}
    fill_counts: dict[str, int] = {}
    for rname, c_lists in top_per_card.items():
        h_f_all = np.zeros(len(eval_u), dtype=np.float64)
        h_f_le100k = np.zeros(len(eval_u), dtype=np.float64)
        h_f_sup = np.zeros(len(eval_u), dtype=np.float64)
        h_e_all = np.zeros(len(eval_u), dtype=np.float64)
        h_e_le100k = np.zeros(len(eval_u), dtype=np.float64)
        filled_cnt = 0
        for idx_u, u in enumerate(eval_u.tolist()):
            pks = picks_u[u]
            excl = set(m2_data.seeds(u).tolist()) | ({int(m2_data.self_idx[u])} if m2_data.self_idx[u] >= 0 else set())
            chosen: list[int] = []
            chosen_set: set[int] = set()
            big = 0
            for c, quota in zip(pks, et.allocate(len(pks), et.BUNDLE)):
                got = 0
                for p in c_lists[c]:
                    if got >= quota:
                        break
                    if p in excl or p in chosen_set:
                        continue
                    if fol_arr[p] > pop_lim:
                        if big >= pop_max:
                            continue
                        big += 1
                    chosen.append(p)
                    chosen_set.add(p)
                    got += 1
            if len(chosen) < et.BUNDLE:
                for nb in get_fill_nbs(tuple(pks)):
                    if len(chosen) >= et.BUNDLE:
                        break
                    need = et.BUNDLE - len(chosen)
                    got = 0
                    for p in c_lists[nb]:
                        if got >= need:
                            break
                        if p in excl or p in chosen_set:
                            continue
                        if fol_arr[p] > pop_lim:
                            if big >= pop_max:
                                continue
                            big += 1
                        chosen.append(p)
                        chosen_set.add(p)
                        got += 1
                        filled_cnt += 1
            fset = follow_out[u]
            fgot = [p for p in chosen if p in fset]
            h_f_all[idx_u] = len(fgot) / et.BUNDLE
            h_f_le100k[idx_u] = sum(fol_arr[p] <= 100_000 for p in fgot) / et.BUNDLE
            h_f_sup[idx_u] = sum(bool(sup_mask[p]) for p in fgot) / et.BUNDLE
            eset = engaged_out.get(u, ())
            egot = [p for p in chosen if p in eset]
            h_e_all[idx_u] = len(egot) / et.BUNDLE
            h_e_le100k[idx_u] = sum(fol_arr[p] <= 100_000 for p in egot) / et.BUNDLE
        vecs_by_r[rname] = {
            "follow/all": h_f_all,
            "follow/le100k": h_f_le100k,
            "follow/supplement": h_f_sup,
            "engaged/all": h_e_all,
            "engaged/le100k": h_e_le100k,
        }
        fill_counts[rname] = filled_cnt

    return {
        "n_cards": int(card_ok.sum()),
        "n_eligible_producers": int(len(members)),
        "eval_users": eval_u,
        "pack_burst_mask": m2_data.pack_burst[eval_u],
        "vecs": vecs_by_r,
        "fill_counts": fill_counts,
    }


def main() -> None:
    t0 = time.time()
    lake = pathlib.Path.home() / "catbird-lake-local"
    out_dir = lake / "s16-s17-20261030"
    out_dir.mkdir(parents=True, exist_ok=True)
    s3_dir = lake / "s3"
    s5_dir = lake / "s5-sweep-20261030"
    s10_dir = lake / "s10-outputs-20261030"
    s12_dir = lake / "s12-closeout-20261030"
    s13_dir = lake / "s13-elig-20261030"
    s14_dir = lake / "s14-hostility-20261030"
    s16a_dir = out_dir / "s16a-outputs"
    train_t_dir = lake / "s2-prep-20261028/out/s2a/train_T"
    cand5_t_dir = s3_dir / "matched/cand5-T-matched"
    cand5_full_dir = s3_dir / "matched/cand5-full-matched"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[s16] starting on device={device}", flush=True)

    # 1. Load universes, purge/safety exclusions, and Tier-1 10% held-out producer split
    base_exclude = {ln.strip() for ln in open(s13_dir / "base_safety_exclude_dids.txt") if ln.startswith("did:")}
    hide_dids = {
        ln.split("\t")[0].strip()
        for ln in open(lake / "s5-declarations-20261027/hide_from_recs_union.tsv")
        if ln.startswith("did:")
    }
    train_exclude_dids = base_exclude | hide_dids

    full_dids = [ln.rstrip("\n").split("\t")[1] for ln in open(cand5_full_dir / "producers.tsv")]
    n_full = len(full_dids)
    did2full = {d: i for i, d in enumerate(full_dids)}

    # S1 §5.2: 10% held-out producers by hash, excluded from every training pair and negative mining
    heldout_t1_full = s3_prep.city_hash_mod(full_dids, 10, salt="t1", clickhouse="clickhouse") == 0
    val_anchor_full = (~heldout_t1_full) & (
        s3_prep.city_hash_mod(full_dids, 10, salt="t1val", clickhouse="clickhouse") == 0
    )
    print(
        f"[s16] full producers={n_full:,}, heldout_t1={int(heldout_t1_full.sum()):,} ({heldout_t1_full.mean()*100:.2f}%), "
        f"val_anchors_pool={int(val_anchor_full.sum()):,}",
        flush=True,
    )

    # 2. Load pooled pre-T and all producer vectors & train Tier-1 projection heads + Y10 dunk-negative ablation
    z_pre = np.load(s12_dir / "pooled_preT_fp16.npz")
    pre_pids = z_pre["mm_pids"].astype(np.int64)
    pre_vecs = l2_norm(z_pre["mm_fused"].astype(np.float32))

    z_all = np.load(s12_dir / "pooled_all_fp16.npz")
    all_pids = z_all["mm_pids"].astype(np.int64)
    all_vecs = l2_norm(z_all["mm_fused"].astype(np.float32))
    all_text_pids = z_all["text_pids"].astype(np.int64)
    all_text_vecs = l2_norm(z_all["text_fused"].astype(np.float32))

    # Map cand5-T-matched primary community onto full_dids for pre-T community size^0.7 resampling (leak-free!)
    uni_T = et.Universe(train_t_dir / "producers.tsv", base_exclude, et.BOOT_SEED)
    part_T = et.Partition("cand5", cand5_t_dir, uni_T, None)
    full_to_T = uni_T.index(pa.array(full_dids, pa.string()))
    prim_T_on_full = np.where(full_to_T >= 0, part_T.primary[np.maximum(full_to_T, 0)], -1)

    head_preT, head_preT_meta = train_tier1_head(
        pre_pids,
        pre_vecs,
        full_dids,
        prim_T_on_full,
        s10_dir / "O4/preT_noqn.parquet",
        s16a_dir / "s16a_pairs_preT.parquet",
        train_exclude_dids,
        heldout_t1_full,
        val_anchor_full,
        device=device,
    )
    print(f"[s16] trained preT Tier-1 default head: {json.dumps({k: v for k, v in head_preT_meta.items() if k != 'val_grid'})}", flush=True)

    # Load validated hostile/dunk pairs for Y10 dunk-negative ablation
    cl_dir = lake / "claude-labels-20261027"
    p_map_pairs: dict[str, tuple[str, str]] = {}
    for fn in ("reply_pairs.parquet", "reply_pairs_topup.parquet"):
        tp = pq.read_table(cl_dir / fn, columns=["pair_id", "child_did", "parent_did"])
        for pid_s, cd, pd_s in zip(*(tp[c].to_pylist() for c in ("pair_id", "child_did", "parent_did"))):
            p_map_pairs[pid_s] = (cd, pd_s)
    dunk_pairs_full: list[tuple[int, int]] = []
    tl = pq.read_table(cl_dir / "pair_labels_v2.parquet", columns=["pair_id", "sonnet_stance", "sonnet_dunk"])
    for pid_s, st_s, dk_b in zip(*(tl[c].to_pylist() for c in ("pair_id", "sonnet_stance", "sonnet_dunk"))):
        if st_s == "hostile" or dk_b:
            pr = p_map_pairs.get(pid_s)
            if pr:
                ia, ib = did2full.get(pr[0]), did2full.get(pr[1])
                if ia is not None and ib is not None and ia != ib:
                    dunk_pairs_full.append((ia, ib))

    head_dunk_preT, head_dunk_meta = train_tier1_head(
        pre_pids,
        pre_vecs,
        full_dids,
        prim_T_on_full,
        s10_dir / "O4/preT_noqn.parquet",
        s16a_dir / "s16a_pairs_preT.parquet",
        train_exclude_dids,
        heldout_t1_full,
        val_anchor_full,
        device=device,
        extra_neg_pairs_full=dunk_pairs_full,
        fixed_lam=head_preT_meta["chosen_hyperparams"]["lambda"],
        fixed_epochs=head_preT_meta["chosen_hyperparams"]["epoch"],
    )
    print(f"[s16] trained Y10 dunk-negative ablation head (dunk producer pairs={len(dunk_pairs_full):,}, added={head_dunk_meta['extra_dunk_neg_edges_added']:,})", flush=True)

    # Also train production Tier-1 head on prod_noqn + s16a_pairs_prod with the chosen hyperparams
    kf_full = pq.read_table(cand5_full_dir / "known_for.parquet", columns=["pid", "community", "rank"])
    prim_full = np.full(n_full, -1, dtype=np.int64)
    pf_kf = kf_full["pid"].to_numpy().astype(np.int64)
    rf_kf = kf_full["rank"].to_numpy()
    prim_full[pf_kf[rf_kf == 0]] = kf_full["community"].to_numpy().astype(np.int64)[rf_kf == 0]

    head_prod, head_prod_meta = train_tier1_head(
        all_pids,
        all_vecs,
        full_dids,
        prim_full,
        s10_dir / "O4/prod_noqn.parquet",
        s16a_dir / "s16a_pairs_prod.parquet",
        train_exclude_dids,
        heldout_t1_full,
        val_anchor_full,
        device=device,
        fixed_lam=head_preT_meta["chosen_hyperparams"]["lambda"],
        fixed_epochs=head_preT_meta["chosen_hyperparams"]["epoch"],
    )
    torch.save(
        {
            "preT_state_dict": head_preT.state_dict(),
            "prod_state_dict": head_prod.state_dict(),
            "dunk_ablation_state_dict": head_dunk_preT.state_dict(),
            "preT_meta": head_preT_meta,
            "prod_meta": head_prod_meta,
            "dunk_meta": head_dunk_meta,
        },
        out_dir / "tier1_heads.pt",
    )

    pre_head_vecs = project_all_vecs(head_preT, pre_vecs, device=device)
    pre_dunk_vecs = project_all_vecs(head_dunk_preT, pre_vecs, device=device)
    all_head_vecs = project_all_vecs(head_prod, all_vecs, device=device)

    # 3. Evaluate G-T1 Clause 1 & Y10 Dunk-Negative Ablation on B1a held-out producers + G-T2 trigger
    b1a_lists = read_filtered_lists(s3_dir / "lists/lists_B1a.tsv")
    # Held-out preT slab universe (eligible, non-excluded, in heldout_t1_full)
    ho_mask_rows = np.array(
        [
            int(p) < n_full
            and bool(heldout_t1_full[int(p)])
            and (full_dids[int(p)] not in base_exclude)
            for p in pre_pids
        ],
        dtype=bool,
    )
    ho_rows = np.flatnonzero(ho_mask_rows)
    ho_did_to_u = {full_dids[int(pre_pids[r])]: idx for idx, r in enumerate(ho_rows.tolist())}

    b1a_ho_hits, b1a_ho_uris, b1a_ho_members = eval_knn_lists_on_universe(
        b1a_lists,
        ho_did_to_u,
        {
            "stock": pre_vecs[ho_rows],
            "head_default": pre_head_vecs[ho_rows],
            "head_dunkneg": pre_dunk_vecs[ho_rows],
        },
        device=device,
        with_null_for=["stock", "head_default"],
    )
    diff_gt1 = b1a_ho_hits["head_default"] - b1a_ho_hits["stock"]
    ci_gt1 = boot_list_ci(diff_gt1, B1_LEVEL)
    gt1_clause1_pass = bool(diff_gt1.mean() > 0.0 and ci_gt1[0] > 0.0)

    diff_dunk = b1a_ho_hits["head_dunkneg"] - b1a_ho_hits["head_default"]
    ci_dunk = boot_list_ci(diff_dunk, B1_LEVEL)

    # Rule-generated same-scene anti-trap check on held-out producers (top 20 non-sensitive cand5-T communities with >=15 held-out slab members)
    cl_sens_map = {
        int(r["community"]): list(r.get("sensitivity") or [])
        for r in pq.read_table(lake / "claude-labels-cand5-20261028/community_labels.parquet").to_pylist()
    }
    ho_by_comm_T: dict[int, list[int]] = collections.defaultdict(list)
    for u_idx, r in enumerate(ho_rows.tolist()):
        pf = int(pre_pids[r])
        c_t = int(prim_T_on_full[pf])
        if c_t >= 0 and not cl_sens_map.get(c_t):
            ho_by_comm_T[c_t].append(u_idx)
    anti_trap_comms = sorted([c for c, us in ho_by_comm_T.items() if 15 <= len(us) <= 80], key=lambda c: (-len(ho_by_comm_T[c]), c))[:20]
    anti_trap_splits = 0
    if anti_trap_comms:
        q_at = np.array(sorted({u for c in anti_trap_comms for u in ho_by_comm_T[c]}), dtype=np.int64)
        nn_def_at = exact_nn_for_queries(pre_head_vecs[ho_rows], q_at, device=device)
        nn_dnk_at = exact_nn_for_queries(pre_dunk_vecs[ho_rows], q_at, device=device)
        m_def = np.full(len(ho_rows), -1, dtype=np.int64)
        m_dnk = np.full(len(ho_rows), -1, dtype=np.int64)
        m_def[q_at] = nn_def_at
        m_dnk[q_at] = nn_dnk_at
        for c in anti_trap_comms:
            s_c = set(ho_by_comm_T[c])
            coh_def = float(np.mean([m_def[u] in s_c for u in s_c]))
            coh_dnk = float(np.mean([m_dnk[u] in s_c for u in s_c]))
            if coh_dnk < coh_def - 0.05:
                anti_trap_splits += 1

    y10_keep_dunkneg = bool(diff_dunk.mean() > 0.0 and ci_dunk[0] > 0.0 and anti_trap_splits == 0)
    print(
        f"[s16] G-T1 B1a held-out ({len(b1a_ho_uris):,} lists, {b1a_ho_members:,} members): "
        f"stock={b1a_ho_hits['stock'].mean():.6f}, head={b1a_ho_hits['head_default'].mean():.6f}, "
        f"diff={diff_gt1.mean():+.6f} CI98.75={ci_gt1} pass={gt1_clause1_pass} | "
        f"Y10 dunk-default={diff_dunk.mean():+.6f} CI98.75={ci_dunk} splits={anti_trap_splits}",
        flush=True,
    )

    # G-T2 trigger: recall@50 of O4 teacher top-50 recovered by head_default on held-out anchors
    pid_to_pre_row = np.full(n_full, -1, dtype=np.int32)
    ok_pre_f = (pre_pids < n_full) & np.array([full_dids[int(p)] not in base_exclude for p in pre_pids], dtype=bool)
    pid_to_pre_row[pre_pids[ok_pre_f]] = np.arange(len(pre_pids), dtype=np.int32)[ok_pre_f]

    t_o4_all = pq.read_table(s10_dir / "O4/preT_noqn.parquet", columns=["p", "q", "w"])
    po4 = t_o4_all["p"].to_numpy().astype(np.int64)
    qo4 = t_o4_all["q"].to_numpy().astype(np.int64)
    wo4 = t_o4_all["w"].to_numpy().astype(np.float32)
    m_ho_o4 = (po4 < n_full) & (qo4 < n_full)
    po4, qo4, wo4 = po4[m_ho_o4], qo4[m_ho_o4], wo4[m_ho_o4]
    m_ho_o4_2 = heldout_t1_full[po4] & (pid_to_pre_row[po4] >= 0) & (pid_to_pre_row[qo4] >= 0) & (po4 != qo4)
    po4, qo4, wo4 = po4[m_ho_o4_2], qo4[m_ho_o4_2], wo4[m_ho_o4_2]
    ord_ho = np.lexsort((-wo4, po4))
    po4, qo4 = po4[ord_ho], qo4[ord_ho]
    up_ho, st_ho, cnt_ho = np.unique(po4, return_index=True, return_counts=True)

    rng_gt2 = np.random.default_rng(20261030)
    ho_anchor_indices = np.arange(len(up_ho))
    if len(ho_anchor_indices) > 5000:
        ho_anchor_indices = np.sort(rng_gt2.choice(len(up_ho), 5000, replace=False))
    gt2_q_rows = np.array([int(pid_to_pre_row[up_ho[i]]) for i in ho_anchor_indices], dtype=np.int64)
    gt2_teacher_sets = [
        set(pid_to_pre_row[qo4[st_ho[i] : st_ho[i] + min(int(cnt_ho[i]), 50)]].tolist())
        for i in ho_anchor_indices
    ]

    valid_corpus_rows = np.flatnonzero(ok_pre_f)
    V_head_corpus = torch.from_numpy(pre_head_vecs[valid_corpus_rows]).to(device)
    V_stock_corpus = torch.from_numpy(pre_vecs[valid_corpus_rows]).to(device)
    rec50_head = np.empty(len(gt2_q_rows), dtype=np.float64)
    rec50_stock = np.empty(len(gt2_q_rows), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(gt2_q_rows), 512):
            qb = gt2_q_rows[s : s + 512]
            qh = torch.from_numpy(pre_head_vecs[qb]).to(device)
            qs = torch.from_numpy(pre_vecs[qb]).to(device)
            sh = qh @ V_head_corpus.T
            ss = qs @ V_stock_corpus.T
            # Exclude self
            self_pos = np.searchsorted(valid_corpus_rows, qb)
            sh[torch.arange(len(qb), device=device), torch.from_numpy(self_pos).to(device)] = -float("inf")
            ss[torch.arange(len(qb), device=device), torch.from_numpy(self_pos).to(device)] = -float("inf")
            top50_h = valid_corpus_rows[sh.topk(50, dim=1).indices.cpu().numpy()]
            top50_s = valid_corpus_rows[ss.topk(50, dim=1).indices.cpu().numpy()]
            for j in range(len(qb)):
                tset = gt2_teacher_sets[s + j]
                rec50_head[s + j] = sum(1 for r in top50_h[j] if r in tset) / max(len(tset), 1)
                rec50_stock[s + j] = sum(1 for r in top50_s[j] if r in tset) / max(len(tset), 1)

    print(
        f"[s16] G-T2 teacher top-50 recall@50 on {len(gt2_q_rows):,} held-out anchors: "
        f"stock={rec50_stock.mean():.4f}, head={rec50_head.mean():.4f} (<0.60={bool(rec50_head.mean() < 0.60)})",
        flush=True,
    )

    # 4. Load locked S13 eval scores & M2Data; score G-T1 Clause 2 & G-slab on M2-val first, then M2-test REPORT-ONCE!
    s13_eval_t = pq.read_table(s13_dir / "s13_eval_T_eligibility_and_scores.parquet")
    s13_elig_T = s13_eval_t["eligible"].to_numpy().astype(bool)
    s13_sup_T = s13_eval_t["supplement"].to_numpy().astype(bool)
    s13_win_score_T = s13_eval_t["score"].to_numpy().astype(np.float64)
    s13_d_score_T = s13_eval_t["D_per_sqrt_fol"].to_numpy().astype(np.float64)
    s13_wlin_score_T = s13_eval_t["w_lin_per_sqrt_fol"].to_numpy().astype(np.float64)

    fl05_T = np.load(s13_dir / "flags_T_theta05.npz")
    memb_T = np.load(cand5_t_dir / "membership.npy").astype(np.int64)
    raw_bad_T = fl05_T["follow_back"].astype(bool) | fl05_T["follow_train"].astype(bool) | fl05_T["low_post_follow_heavy"].astype(bool)
    base_raw_bad_T = fl05_T["follow_back"].astype(bool) | fl05_T["follow_train"].astype(bool)
    comm_table_T = pq.read_table(cand5_t_dir / "communities.parquet", columns=["community", "top_dids"])
    raw_by_comm_T = s3_prep.raw_label_by_community(comm_table_T, memb_T, {x: i for i, x in enumerate(uni_T.dids)})
    comm_bad_T = np.zeros(part_T.n_comm, dtype=bool)
    base_comm_bad_T = np.zeros(part_T.n_comm, dtype=bool)
    for c_id, rlab in raw_by_comm_T.items():
        if c_id < part_T.n_comm:
            comm_bad_T[c_id] = bool(raw_bad_T[rlab])
            base_comm_bad_T[c_id] = bool(base_raw_bad_T[rlab])

    co_off_T, co_dst_T, co_cnt_T = part_T.co_membership()
    high_h_T = s13.load_high_H_pairs_T(s14_dir / "s14a-outputs")
    m2_data = et.M2Data(s3_dir / "m2", uni_T, base_exclude, "clickhouse")

    # Map pid_T -> pre_pids row for topical coherence at T
    T_to_full = np.array([did2full.get(d, -1) for d in uni_T.dids], dtype=np.int64)
    pid_T_to_pre_row = np.where(T_to_full >= 0, pid_to_pre_row[np.maximum(T_to_full, 0)], -1)

    coh_stock_T = compute_card_coherence(part_T.primary, part_T.n_comm, s13_elig_T, pid_T_to_pre_row, pre_vecs)
    coh_head_T = compute_card_coherence(part_T.primary, part_T.n_comm, s13_elig_T, pid_T_to_pre_row, pre_head_vecs)

    # Evaluate G-slab & G-T1 Clause 2 on M2-val ONLY first
    val_coh_rankers = {
        "s13_locked": s13_win_score_T,
        **{f"stock_coh_g{g}": s13_win_score_T * (1.0 + g * coh_stock_T) for g in (0.1, 0.25, 0.5, 1.0)},
        **{f"head_coh_g{g}": s13_win_score_T * (1.0 + g * coh_head_T) for g in (0.1, 0.25, 0.5, 1.0)},
    }
    val_coh_res = score_m2_split_full_slices(
        part_T, uni_T, m2_data, "val", s13_elig_T, raw_bad_T, memb_T, comm_bad_T, s13_sup_T,
        val_coh_rankers, high_h_T, co_off_T, co_dst_T, co_cnt_T,
    )
    best_stock_g = max((0.1, 0.25, 0.5, 1.0), key=lambda g: float(val_coh_res["vecs"][f"stock_coh_g{g}"]["follow/all"].mean()))
    best_head_g = max((0.1, 0.25, 0.5, 1.0), key=lambda g: float(val_coh_res["vecs"][f"head_coh_g{g}"]["follow/all"].mean()))

    X_vc = np.column_stack([
        val_coh_res["vecs"]["s13_locked"]["follow/all"],
        val_coh_res["vecs"][f"stock_coh_g{best_stock_g}"]["follow/all"],
        val_coh_res["vecs"][f"head_coh_g{best_head_g}"]["follow/all"],
    ])
    pt_vc, bt_vc = et.bootstrap(X_vc, np.ones_like(X_vc, dtype=bool), et.BOOT_N, et.BOOT_SEED)
    gt1_clause2_pass = bool(pt_vc[2] >= pt_vc[1] - 1e-12)
    gt1_pass = bool(gt1_clause1_pass and gt1_clause2_pass)
    gt2_trigger_pass = bool(gt1_pass and float(rec50_head.mean()) < 0.60)

    best_coh_idx = 2 if pt_vc[2] >= pt_vc[1] else 1
    g_slab_val_rel = float(pt_vc[best_coh_idx] / pt_vc[0] - 1.0)
    g_slab_val_ci = et.ci(bt_vc[:, best_coh_idx] / bt_vc[:, 0] - 1.0)
    g_slab_val_pass = bool(g_slab_val_rel >= 0.05 and g_slab_val_ci[0] > 0.0)
    print(
        f"[s16] G-T1 clause2 (M2-val head {pt_vc[2]:.6f} >= stock {pt_vc[1]:.6f}): {gt1_clause2_pass} -> G-T1={gt1_pass}, G-T2_trigger={gt2_trigger_pass} | "
        f"G-slab M2-val rel={g_slab_val_rel:+.4%} CI95={g_slab_val_ci} pass={g_slab_val_pass}",
        flush=True,
    )

    # 5. M2-test REPORT-ONCE (first and only touch of split == 'test'!)
    test_rankers_s13 = {
        "s13_winner_w_lin_plus_curators": s13_win_score_T,
        "w_lin_per_sqrt_fol": s13_wlin_score_T,
        "D_per_sqrt_fol": s13_d_score_T,
        "s13_cell_community_popularity": uni_T.followers.astype(np.float64),
        "head_coh_best": s13_win_score_T * (1.0 + best_head_g * coh_head_T),
    }
    m2_test_s13 = score_m2_split_full_slices(
        part_T, uni_T, m2_data, "test", s13_elig_T, raw_bad_T, memb_T, comm_bad_T, s13_sup_T,
        test_rankers_s13, high_h_T, co_off_T, co_dst_T, co_cnt_T,
    )
    m2_test_base = score_m2_split_full_slices(
        part_T, uni_T, m2_data, "test", np.ones(uni_T.n, dtype=bool), np.zeros_like(base_raw_bad_T), memb_T, base_comm_bad_T, s13_sup_T,
        {"base_s3_community_popularity": uni_T.followers.astype(np.float64), "base_s3_knownfor": part_T.primary_score},
        set(), co_off_T, co_dst_T, co_cnt_T,
    )

    # Align Base S3 community_popularity on the evaluated M2-test users AND report full Base S3 M2-test
    base_u_map = {
        int(u): i for i, u in enumerate(m2_test_base["eval_users"].tolist())
    }
    test_u = m2_test_s13["eval_users"]
    n_test = len(test_u)
    pb_mask = m2_test_s13["pack_burst_mask"]

    # Build bootstrap matrix over M2-test users for all primary ratios & slices
    cols_t, masks_t, labels_t = [], [], []
    for rn, sdict in m2_test_s13["vecs"].items():
        for k_sl, arr in sdict.items():
            for u_sl, um in (("all", np.ones(n_test, dtype=bool)), ("pack_burst", pb_mask), ("no_pack_burst", ~pb_mask)):
                cols_t.append(arr)
                masks_t.append(um)
                labels_t.append((rn, k_sl, u_sl))
    # Aligned base_s3_community_popularity on the exact same test_u
    for b_rn in ("base_s3_community_popularity", "base_s3_knownfor"):
        for k_sl in ("follow/all", "follow/le100k", "follow/supplement", "engaged/all"):
            aligned = np.array(
                [m2_test_base["vecs"][b_rn][k_sl][base_u_map[int(u)]] if int(u) in base_u_map else 0.0 for u in test_u.tolist()],
                dtype=np.float64,
            )
            for u_sl, um in (("all", np.ones(n_test, dtype=bool)), ("pack_burst", pb_mask), ("no_pack_burst", ~pb_mask)):
                cols_t.append(aligned)
                masks_t.append(um)
                labels_t.append((b_rn, k_sl, u_sl))

    Xt = np.column_stack(cols_t)
    Mt = np.column_stack(masks_t)
    pt_t, bt_t = et.bootstrap(Xt, Mt, et.BOOT_N, et.BOOT_SEED)
    t_col = {lab: j for j, lab in enumerate(labels_t)}

    # Also bootstrap unaligned full Base S3 M2-test on its own 21k+ test users
    Xb_full = np.column_stack([
        m2_test_base["vecs"]["base_s3_community_popularity"]["follow/all"],
        m2_test_base["vecs"]["base_s3_knownfor"]["follow/all"],
    ])
    pt_bf, bt_bf = et.bootstrap(Xb_full, np.ones_like(Xb_full, dtype=bool), et.BOOT_N, et.BOOT_SEED)

    j_win = t_col[("s13_winner_w_lin_plus_curators", "follow/all", "all")]
    j_wlin = t_col[("w_lin_per_sqrt_fol", "follow/all", "all")]
    j_d = t_col[("D_per_sqrt_fol", "follow/all", "all")]
    j_cpop = t_col[("s13_cell_community_popularity", "follow/all", "all")]
    j_bpop = t_col[("base_s3_community_popularity", "follow/all", "all")]
    j_hcoh = t_col[("head_coh_best", "follow/all", "all")]

    ratio_vs_base = float(pt_t[j_win] / pt_t[j_bpop])
    ratio_vs_base_ci = et.ci(bt_t[:, j_win] / bt_t[:, j_bpop])
    ratio_vs_base_unaligned = float(pt_t[j_win] / pt_bf[0])
    ratio_vs_s13_cpop = float(pt_t[j_win] / pt_t[j_cpop])
    ratio_vs_s13_cpop_ci = et.ci(bt_t[:, j_win] / bt_t[:, j_cpop])

    v8ii_test_winner_vs_D_rel = float(pt_t[j_win] / pt_t[j_d] - 1.0)
    v8ii_test_winner_vs_D_ci = et.ci(bt_t[:, j_win] / bt_t[:, j_d] - 1.0)
    v8ii_test_wlin_vs_D_rel = float(pt_t[j_wlin] / pt_t[j_d] - 1.0)
    v8ii_test_wlin_vs_D_ci = et.ci(bt_t[:, j_wlin] / bt_t[:, j_d] - 1.0)

    g_slab_test_rel = float(pt_t[j_hcoh] / pt_t[j_win] - 1.0)
    g_slab_test_ci = et.ci(bt_t[:, j_hcoh] / bt_t[:, j_win] - 1.0)
    g_slab_pass = bool(g_slab_val_pass and g_slab_test_rel >= 0.05 and g_slab_test_ci[0] > 0.0)

    m2_test_slices_summary: dict = {}
    for (rn, k_sl, u_sl), j in t_col.items():
        m2_test_slices_summary.setdefault(rn, {}).setdefault(u_sl, {})[k_sl] = et.summary(
            pt_t[j], bt_t[:, j], int(Mt[:, j].sum())
        )

    print(
        f"[s16] M2-TEST REPORT-ONCE (n_test={n_test:,}, power_floor_ge_2000={n_test >= 2000}): "
        f"winner_p@25={pt_t[j_win]:.6f} | vs_base_s3_pop={ratio_vs_base:.4f}x CI95={ratio_vs_base_ci} (unaligned={ratio_vs_base_unaligned:.4f}x) | "
        f"vs_s13_cell_pop={ratio_vs_s13_cpop:.4f}x CI95={ratio_vs_s13_cpop_ci} | "
        f"V8(ii) winner_vs_D={v8ii_test_winner_vs_D_rel:+.4%} CI95={v8ii_test_winner_vs_D_ci}",
        flush=True,
    )

    # Read M1 follows_le20 diagnostic from s3/m1-T.run1.json
    m1_diag = {}
    if (s3_dir / "m1-T.run1.json").exists():
        m1_j = json.loads((s3_dir / "m1-T.run1.json").read_text())
        for arm_nm in ("cand5", "cand1", "popularity"):
            sl_d = m1_j.get("metrics", {}).get("primary", {}).get(arm_nm, {}).get("follows_le20", {})
            if sl_d:
                m1_diag[arm_nm] = sl_d

    # 6. G-img trigger (A2 report-only): text-kNN vs B1-owner-excluded follow-kNN on image-heavy B1b lists
    b1b_lists = read_filtered_lists(s3_dir / "lists/lists_B1b.tsv")
    img_heavy_uris = {ln.split("\t")[0].strip() for ln in open(lake / "c4-prep-20261028/q20/q20_lists_k5_at50.tsv") if ln.strip()}
    b1b_img_lists = [(u, ms) for u, ms in b1b_lists if u in img_heavy_uris]

    # Intersect non-excluded text_fused producers with knn_b1x_top50 producers
    ok_txt = (all_text_pids < n_full) & np.array([full_dids[int(p)] not in base_exclude for p in all_text_pids], dtype=bool)
    txt_did_to_row = {full_dids[int(all_text_pids[r])]: r for r in np.flatnonzero(ok_txt).tolist()}

    t_b1x = pq.read_table(s3_dir / "knn_b1x_top50.parquet", columns=["did", "rank", "nbr_did"])
    b1x_d = t_b1x["did"].to_pylist()
    b1x_r = t_b1x["rank"].to_numpy()
    b1x_nd = t_b1x["nbr_did"].to_pylist()

    # Keep DIDs present in txt_did_to_row; find first neighbour also in txt_did_to_row & not self
    cand_u_dids = sorted(set(b1x_d) & set(txt_did_to_row.keys()))
    cand_u_set = set(cand_u_dids)
    best_rank_for: dict[str, int] = {}
    follow_nn_did: dict[str, str] = {}
    for d, rk, nd in zip(b1x_d, b1x_r.tolist(), b1x_nd):
        if d in cand_u_set and nd in cand_u_set and d != nd:
            prev = best_rank_for.get(d, 999)
            if rk < prev:
                best_rank_for[d] = rk
                follow_nn_did[d] = nd

    shared_gimg_dids = sorted(follow_nn_did.keys())
    gimg_did_to_u = {d: i for i, d in enumerate(shared_gimg_dids)}
    # Ensure follow_nn_did target is still in shared_gimg_dids (second pass)
    shared_gimg_dids = [d for d in shared_gimg_dids if follow_nn_did[d] in gimg_did_to_u]
    gimg_did_to_u = {d: i for i, d in enumerate(shared_gimg_dids)}
    # Recompute exact first neighbour inside final shared_gimg_dids
    best_rank_for2: dict[str, int] = {}
    follow_nn_u = np.full(len(shared_gimg_dids), -1, dtype=np.int64)
    for d, rk, nd in zip(b1x_d, b1x_r.tolist(), b1x_nd):
        ui = gimg_did_to_u.get(d)
        ni = gimg_did_to_u.get(nd)
        if ui is not None and ni is not None and ui != ni:
            if rk < best_rank_for2.get(d, 999):
                best_rank_for2[d] = rk
                follow_nn_u[ui] = ni
    valid_both = follow_nn_u >= 0
    if not valid_both.all():
        shared_gimg_dids = [d for d, ok in zip(shared_gimg_dids, valid_both.tolist()) if ok]
        gimg_did_to_u = {d: i for i, d in enumerate(shared_gimg_dids)}
        best_rank_for3: dict[str, int] = {}
        follow_nn_u = np.full(len(shared_gimg_dids), -1, dtype=np.int64)
        for d, rk, nd in zip(b1x_d, b1x_r.tolist(), b1x_nd):
            ui = gimg_did_to_u.get(d)
            ni = gimg_did_to_u.get(nd)
            if ui is not None and ni is not None and ui != ni:
                if rk < best_rank_for3.get(d, 999):
                    best_rank_for3[d] = rk
                    follow_nn_u[ui] = ni

    gimg_txt_vecs = np.stack([all_text_vecs[txt_did_to_row[d]] for d in shared_gimg_dids], axis=0)
    gimg_img_groups = []
    gimg_q_set = set()
    for uri, ms in b1b_img_lists:
        g = np.array([gimg_did_to_u[d] for d in ms if d in gimg_did_to_u], dtype=np.int64)
        if len(g) >= 3:
            gimg_img_groups.append(g)
            gimg_q_set.update(g.tolist())
    q_gimg = np.array(sorted(gimg_q_set), dtype=np.int64)
    txt_nn_q = exact_nn_for_queries(gimg_txt_vecs, q_gimg, device=device)
    txt_nn_u = np.full(len(shared_gimg_dids), -1, dtype=np.int64)
    txt_nn_u[q_gimg] = txt_nn_q

    fol_hits_img = np.array([np.mean([follow_nn_u[m] in set(g.tolist()) for m in g]) for g in gimg_img_groups], dtype=np.float64)
    txt_hits_img = np.array([np.mean([txt_nn_u[m] in set(g.tolist()) for m in g]) for g in gimg_img_groups], dtype=np.float64)
    lag_fol_minus_txt = fol_hits_img - txt_hits_img
    lag_ci = boot_list_ci(lag_fol_minus_txt, B1_LEVEL)
    gimg_trigger_would_fire = bool(len(gimg_img_groups) >= 30 and lag_fol_minus_txt.mean() >= 0.10)
    print(
        f"[s16] G-img trigger (A2 report-only, {len(gimg_img_groups):,} image-heavy B1b lists): "
        f"follow_b1x={fol_hits_img.mean():.6f}, text_knn={txt_hits_img.mean():.6f}, "
        f"lag={lag_fol_minus_txt.mean():+.6f} CI98.75={lag_ci}",
        flush=True,
    )

    # 7. B2 Report-Once (partition macro top-1 & content-kNN macro top-1)
    b2_lists_with_uri = read_filtered_lists(s3_dir / "lists/lists_B2.tsv")
    b2_lists_members = [ms for _, ms in b2_lists_with_uri]
    uni_full = et.Universe(cand5_full_dir / "producers.tsv", base_exclude, et.BOOT_SEED)
    part_full = et.Partition("cand5_full", cand5_full_dir, uni_full, None)

    b2_part_T_vals, b2_part_T_m = et.list_top1(b2_lists_members, [part_T], uni_T)
    b2_part_T_pt, b2_part_T_bt = et.bootstrap(b2_part_T_vals, b2_part_T_m, et.BOOT_N, et.BOOT_SEED)
    b2_part_F_vals, b2_part_F_m = et.list_top1(b2_lists_members, [part_full], uni_full)
    b2_part_F_pt, b2_part_F_bt = et.bootstrap(b2_part_F_vals, b2_part_F_m, et.BOOT_N, et.BOOT_SEED)

    ok_all_mm = (all_pids < n_full) & np.array([full_dids[int(p)] not in base_exclude for p in all_pids], dtype=bool)
    all_mm_rows = np.flatnonzero(ok_all_mm)
    b2_did_to_u = {full_dids[int(all_pids[r])]: idx for idx, r in enumerate(all_mm_rows.tolist())}
    b2_knn_hits, b2_knn_uris, b2_knn_members = eval_knn_lists_on_universe(
        b2_lists_with_uri,
        b2_did_to_u,
        {
            "s12m_stock": all_vecs[all_mm_rows],
            "s16_tier1_head": all_head_vecs[all_mm_rows],
        },
        device=device,
        with_null_for=["s12m_stock", "s16_tier1_head"],
    )
    print(
        f"[s16] B2 report-once: part_full_top1={float(b2_part_F_pt[0]):.6f} (n={int(b2_part_F_m[:,0].sum()):,}), "
        f"s12m_stock_knn={b2_knn_hits['s12m_stock'].mean():.6f}, tier1_head_knn={b2_knn_hits['s16_tier1_head'].mean():.6f} (n={len(b2_knn_uris):,})",
        flush=True,
    )

    # 8. Top-10 Language Y3 Sensitivity-Coverage Loss Table (S1 §10) & R13 Trap Status
    s13_prod_t = pq.read_table(s13_dir / "s13_prod_eligibility_and_scores.parquet", columns=["pid", "eligible"])
    elig_prod = s13_prod_t["eligible"].to_numpy().astype(bool)
    by_card_prod: dict[int, int] = collections.Counter()
    for p in np.flatnonzero(elig_prod & (prim_full >= 0)).tolist():
        by_card_prod[int(prim_full[p])] += 1
    displayable_cards = {c: cnt for c, cnt in by_card_prod.items() if cnt >= 25}

    # Load top list names per community (A-half + community_list_names.tsv, matching s6picker/build.py lines 97-100;
    # never run keyword demonyms on Opus's English language-prefixed label like "French tabletop RPG scene")
    from s6picker import inputs as s6_in
    s6_in.set_partition("cand5", str(lake / "claude-labels-cand5-20261028/community_labels.parquet"))
    home_full = {full_dids[p]: int(prim_full[p]) for p in range(n_full) if prim_full[p] >= 0}
    _a_own, a_names_full = s6_in.a_lists(out_dir / "s6_cache", home_full)
    all_names_full = s6_in.all_list_names()

    cl_rows = {
        int(r["community"]): r
        for r in pq.read_table(lake / "claude-labels-cand5-20261028/community_labels.parquet").to_pylist()
    }

    lang_stats: dict[str, dict] = {}
    eth_lang_only_violations = 0
    for c, n_cand in displayable_cards.items():
        cr = cl_rows.get(c, {})
        lang = (cr.get("primary_language") or "und").strip() or "und"
        opus_sens = {s.replace("/", "_").replace("-", "_") for s in (cr.get("sensitivity") or []) if s and s != "none"}
        a_top = [n for n, _k in a_names_full.get(c, [])]
        names_screened = a_top[:6] + all_names_full.get(c, [])[:6]
        kw_sens = {kcl.replace("/", "_").replace("-", "_") for nm in names_screened for kcl in s6_kw.classes(nm)}
        all_sens = (opus_sens | kw_sens) & set(Y3_CLASSES)

        # Check ethnicity_nationality never inferred from non-en language alone without identity evidence
        if "ethnicity_nationality" in opus_sens and lang != "en":
            lbl_low = f"{cr.get('label', '')} {cr.get('description', '')}".lower()
            if not lbl_low.strip():
                eth_lang_only_violations += 1

        st = lang_stats.setdefault(
            lang,
            {
                "language": lang,
                "displayable_cards": 0,
                "kept_v0_cards": 0,
                "excluded_y3_cards": 0,
                "displayable_candidates": 0,
                "kept_v0_candidates": 0,
                "excluded_y3_candidates": 0,
                "by_y3_class_cards": {cl: 0 for cl in Y3_CLASSES},
            },
        )
        st["displayable_cards"] += 1
        st["displayable_candidates"] += int(n_cand)
        if all_sens:
            st["excluded_y3_cards"] += 1
            st["excluded_y3_candidates"] += int(n_cand)
            for cl in all_sens:
                st["by_y3_class_cards"][cl] += 1
        else:
            st["kept_v0_cards"] += 1
            st["kept_v0_candidates"] += int(n_cand)

    top10_langs = sorted(lang_stats.values(), key=lambda d: (-d["displayable_cards"], d["language"]))[:10]
    for d in top10_langs:
        d["card_coverage_loss_pct"] = round(100.0 * d["excluded_y3_cards"] / max(d["displayable_cards"], 1), 2)
        d["candidate_coverage_loss_pct"] = round(
            100.0 * d["excluded_y3_candidates"] / max(d["displayable_candidates"], 1), 2
        )

    # R13 frozen seed & trap check (must verify exact sha256 6fa85f0c... before reporting, else 'not run')
    trap_path = lake / "graph-build/seed_and_traps_manifest.json"
    expected_trap_sha = "6fa85f0c99a4674e40488396212623b71bcac8367d06b5231c16cae99c074edb"
    if trap_path.exists() and sha256_file(trap_path) == expected_trap_sha:
        r13_traps = {"status": "available", "sha256": expected_trap_sha}
    else:
        r13_traps = {
            "status": "not run",
            "reason": "seed_and_traps_manifest.json (expected sha256 6fa85f0c99a4674e40488396212623b71bcac8367d06b5231c16cae99c074edb) resides on unmounted external drive 'My Passport' since D14; per S1 §8 / R13, reported 'not run' and never re-created.",
            "local_file_exists": trap_path.exists(),
        }

    # Assemble complete S16 report
    primary_m2_bar_pass = bool(ratio_vs_base >= 1.20 and ratio_vs_base_ci[0] > 1.0 and n_test >= 2000)
    report = {
        "step": "S16 Catalog, Tier-1 Head, Content Gates, and M2-test / B2 Report-Once",
        "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md row S16, §5.2, §5.4, §6",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md §9.5, §9.8, §10, Amendments A1 & A2",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t0, 2),
        "inputs_sha256": {
            "s13_eval_T": sha256_file(s13_dir / "s13_eval_T_eligibility_and_scores.parquet"),
            "s13_prod": sha256_file(s13_dir / "s13_prod_eligibility_and_scores.parquet"),
            "pooled_preT_fp16": sha256_file(s12_dir / "pooled_preT_fp16.npz"),
            "pooled_all_fp16": sha256_file(s12_dir / "pooled_all_fp16.npz"),
            "s16a_pairs_preT": sha256_file(s16a_dir / "s16a_pairs_preT.parquet"),
            "s16a_pairs_prod": sha256_file(s16a_dir / "s16a_pairs_prod.parquet"),
            "lists_B1a": sha256_file(s3_dir / "lists/lists_B1a.tsv"),
            "lists_B1b": sha256_file(s3_dir / "lists/lists_B1b.tsv"),
            "lists_B2": sha256_file(s3_dir / "lists/lists_B2.tsv"),
        },
        "tier1_head": {
            "architecture": "f(x) = normalize(x + MLP_256_512_256(x))",
            "heldout_t1_producers_10pct": int(heldout_t1_full.sum()),
            "preT_default": head_preT_meta,
            "prod_default": head_prod_meta,
            "y10_dunk_negative_ablation": {
                **head_dunk_meta,
                "b1a_heldout_diff_dunk_minus_default": float(diff_dunk.mean()),
                "b1a_heldout_ci9875": ci_dunk,
                "anti_trap_communities_checked": len(anti_trap_comms),
                "anti_trap_splits": anti_trap_splits,
                "adopted": y10_keep_dunkneg,
                "decision": "adopt_dunkneg" if y10_keep_dunkneg else "keep_default_no_dunk_negatives",
            },
        },
        "gates_5_4": {
            "G_T1": {
                "pass": gt1_pass,
                "clause1_b1a_heldout_knn": {
                    "lists": len(b1a_ho_uris),
                    "heldout_members": b1a_ho_members,
                    "stock_macro_top1": float(b1a_ho_hits["stock"].mean()),
                    "stock_null_macro_top1": float(b1a_ho_hits["stock__null"].mean()),
                    "head_macro_top1": float(b1a_ho_hits["head_default"].mean()),
                    "head_null_macro_top1": float(b1a_ho_hits["head_default__null"].mean()),
                    "diff_head_minus_stock": float(diff_gt1.mean()),
                    "ci9875": ci_gt1,
                    "pass": gt1_clause1_pass,
                },
                "clause2_m2_val_coherence_noninferior": {
                    "stock_best_gamma": best_stock_g,
                    "head_best_gamma": best_head_g,
                    "m2_val_stock_p25": float(pt_vc[1]),
                    "m2_val_head_p25": float(pt_vc[2]),
                    "pass": gt1_clause2_pass,
                },
                "decision": "adopt_tier1_head" if gt1_pass else "keep_stock_eg2",
            },
            "G_T2_trigger": {
                "pass": gt2_trigger_pass,
                "gt1_passed": gt1_pass,
                "heldout_anchors_evaluated": len(gt2_q_rows),
                "stock_teacher_top50_recall50": float(rec50_stock.mean()),
                "head_teacher_top50_recall50": float(rec50_head.mean()),
                "bar": "G-T1 passes AND head teacher top-50 recall@50 < 0.60 on held-out producers",
                "decision": "C1_LoRA_eligible_if_requested" if gt2_trigger_pass else "no_C1",
            },
            "G_slab": {
                "pass": g_slab_pass,
                "m2_val_no_coherence_p25": float(pt_vc[0]),
                "m2_val_best_coherence_p25": float(pt_vc[best_coh_idx]),
                "m2_val_rel_lift": g_slab_val_rel,
                "m2_val_rel_ci95": g_slab_val_ci,
                "m2_val_pass_ge_5pct": g_slab_val_pass,
                "m2_test_rel_lift": g_slab_test_rel,
                "m2_test_rel_ci95": g_slab_test_ci,
                "decision": "include_topical_coherence_in_bundle_ranker"
                if g_slab_pass
                else "vectors_used_for_medoid_posts_and_preview_only_C1_C3_off",
            },
            "G_img_trigger_A2_report_only": {
                "status": "report_only_per_amendment_A2",
                "image_heavy_b1b_lists_evaluated": len(gimg_img_groups),
                "shared_universe_producers": len(shared_gimg_dids),
                "floor_ge_30_met": len(gimg_img_groups) >= 30,
                "follow_b1x_macro_top1": float(fol_hits_img.mean()),
                "text_knn_macro_top1": float(txt_hits_img.mean()),
                "lag_follow_minus_text": float(lag_fol_minus_txt.mean()),
                "lag_ci9875": lag_ci,
                "would_have_triggered_pre_A2_ge_10pts": gimg_trigger_would_fire,
            },
        },
        "m2_test_report_once": {
            "locked_s13_spec": {"k": 3, "m": 3, "theta": 0.5, "ranker": "w_lin_plus_curators"},
            "test_users_evaluated": n_test,
            "power_floor_ge_2000_met": bool(n_test >= 2000),
            "survivorship_caveat_Y12": "M2 cohort consists of PLC genesis [Aug 15, Sep 8] accounts still live at Oct 24 snapshot (survivors only per plan §6 / Y12).",
            "primary_bar_vs_base_s3_community_popularity": {
                "pass": primary_m2_bar_pass,
                "bar": "M2-test precision@25 >= 1.2x community_popularity baseline with 95% CI > 1.0",
                "winner_p25": float(pt_t[j_win]),
                "winner_ci95": et.ci(bt_t[:, j_win]),
                "base_s3_comm_pop_aligned_p25": float(pt_t[j_bpop]),
                "base_s3_comm_pop_aligned_ci95": et.ci(bt_t[:, j_bpop]),
                "ratio_vs_base_s3_aligned": ratio_vs_base,
                "ratio_vs_base_s3_aligned_ci95": ratio_vs_base_ci,
                "base_s3_comm_pop_full_cohort_n": int(len(m2_test_base["eval_users"])),
                "base_s3_comm_pop_full_cohort_p25": float(pt_bf[0]),
                "ratio_vs_base_s3_full_cohort": ratio_vs_base_unaligned,
            },
            "within_s13_cell_comparison": {
                "s13_cell_comm_pop_p25": float(pt_t[j_cpop]),
                "s13_cell_comm_pop_ci95": et.ci(bt_t[:, j_cpop]),
                "ratio_vs_s13_cell_comm_pop": ratio_vs_s13_cpop,
                "ratio_vs_s13_cell_comm_pop_ci95": ratio_vs_s13_cpop_ci,
                "ci95_excludes_1_above": bool(ratio_vs_s13_cpop_ci[0] > 1.0),
            },
            "v8_part_ii_m2_test_confirmation": {
                "pass": bool(v8ii_test_winner_vs_D_ci[0] > 0.0 and v8ii_test_wlin_vs_D_ci[0] > 0.0),
                "D_per_sqrt_fol_p25": float(pt_t[j_d]),
                "w_lin_per_sqrt_fol_p25": float(pt_t[j_wlin]),
                "w_lin_plus_curators_p25": float(pt_t[j_win]),
                "w_lin_vs_D_rel": v8ii_test_wlin_vs_D_rel,
                "w_lin_vs_D_rel_ci95": v8ii_test_wlin_vs_D_ci,
                "winner_vs_D_rel": v8ii_test_winner_vs_D_rel,
                "winner_vs_D_rel_ci95": v8ii_test_winner_vs_D_ci,
            },
            "slices": m2_test_slices_summary,
            "fill_rule_accounts": m2_test_s13["fill_counts"],
            "m1_follows_le20_diagnostic": m1_diag,
        },
        "b2_report_once": {
            "partition_cand5_T_matched": {
                "lists": int(b2_part_T_m[:, 0].sum()),
                "macro_top1": float(b2_part_T_pt[0]),
                "ci95": et.ci(b2_part_T_bt[:, 0]),
                "same_size_null": et.same_primary_null(part_T, uni_T),
            },
            "partition_cand5_full_matched": {
                "lists": int(b2_part_F_m[:, 0].sum()),
                "macro_top1": float(b2_part_F_pt[0]),
                "ci95": et.ci(b2_part_F_bt[:, 0]),
                "same_size_null": et.same_primary_null(part_full, uni_full),
            },
            "content_knn_shared_slab_universe": {
                "lists": len(b2_knn_uris),
                "members": b2_knn_members,
                "universe_producers": len(b2_did_to_u),
                "s12m_stock_macro_top1": float(b2_knn_hits["s12m_stock"].mean()),
                "s12m_stock_ci95": boot_list_ci(b2_knn_hits["s12m_stock"], 0.95),
                "s12m_stock_null": float(b2_knn_hits["s12m_stock__null"].mean()),
                "s16_tier1_head_macro_top1": float(b2_knn_hits["s16_tier1_head"].mean()),
                "s16_tier1_head_ci95": boot_list_ci(b2_knn_hits["s16_tier1_head"], 0.95),
                "s16_tier1_head_null": float(b2_knn_hits["s16_tier1_head__null"].mean()),
            },
        },
        "y3_sensitivity_coverage_top10_languages": {
            "total_displayable_cards_ge25": len(displayable_cards),
            "total_kept_v0_cards": sum(d["kept_v0_cards"] for d in lang_stats.values()),
            "total_excluded_y3_cards": sum(d["excluded_y3_cards"] for d in lang_stats.values()),
            "ethnicity_nationality_from_language_alone_violations": eth_lang_only_violations,
            "top10_languages": top10_langs,
        },
        "r13_seeds_and_traps": r13_traps,
    }
    report["metrics_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, default=float).encode()).hexdigest()
    (out_dir / "s16_report.json").write_text(json.dumps(report, indent=2))
    print(f"[s16] DONE in {time.time() - t0:.1f}s -> {out_dir / 's16_report.json'}", flush=True)


if __name__ == "__main__":
    main()
