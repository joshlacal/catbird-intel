#!/usr/bin/env python3
"""Compute full-data S12m + S16 Tier-1 + per-post slab duplication + S5/S13/C5/curatelist anti-spam signals
for all S13-eligible organic candidate producers across cand5 communities (S17 rev 4).

Outputs in `--out` (`~/catbird-lake-local/s17-picker-v1-20261030`):
- `candidate_signals.parquet`: per-candidate semantic, Tier-1 social, slab self-similarity, follow-churn,
  cadence, v9-demotion, `producer_quality` (`mass3k_share`, `mutual_share`, `modlists`, `blockers`,
  `pq_posts`, `pq_originals`, `pq_light_engagers`), negative curated-list consensus (`bot_list_owners`,
  `spam_list_owners`), Claude teacher `profile_labels`, and distilled `C5` `account_type_model` probabilities.
- `card_centroids.npz`: global mean vectors (`mu_fused`, `mu_posts`, `mu_text_posts`, `mu_t1`) and
  trimmed unit centroids per kept/candidate community (`cids`, `cent_fused`, `cent_posts`, `cent_text_posts`, `cent_t1`).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import json
import os
import pathlib
import re
import sys
import tarfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from s6picker import inputs as s6_in  # noqa: E402

BUCKET_RE = re.compile(r"^u-(b\d{4})-\d{4}-\d+\.parquet$")

BOT_LIST_RE = re.compile(
    r"^(?:cool\s+|hourly\s+|daily\s+|quote\s+|art\s+|museum\s+|photo\s+|bluesky\s+|good\s+|fun\s+|cute\s+|mostly\s+|a\s+bunch\s+of\s+|🤖\s*)*"
    r"(?:bots?|automated\s+accounts?|bot\s+union|bot\s+world|bbdq\s+bots?)(?:\s*[.|+!🤖\d]+|\s+and\s+automated.*)?$",
    re.I,
)
SPAM_LIST_RE = re.compile(
    r"\b(?:spammers?|spam\s+(?:accounts?|bots?|ring)|scammers?|e-?begging|e-?solicitors?|ai[- ]slop|slop\s+(?:accounts?|bots?)|"
    r"follow\s*for\s*follow|\bf4f\b|follow\s*farm(?:ers?|ing)?|engagement\s+(?:bait|farm)|"
    r"fakes?,?\s*bots?\s*(?:&|and)\s*scrapers?|promo\s+spam)\b",
    re.I,
)
GUARD_LIST_RE = re.compile(
    r"\b(?:no\s+bots?|not\s+bots?|and\s+nots|influencers|posters|robotics|battlebots|aimbot|bot\s+makers?|"
    r"anti[- ]ai|no[- ]ai|slop[- ]free|free\s+of\s+ai)\b",
    re.I,
)


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class Tier1Head(nn.Module):
    def __init__(self, d_in: int = 256, d_hid: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.fc2(F.gelu(self.fc1(x))), p=2, dim=-1)


class C5AccountTypeStudent(nn.Module):
    def __init__(self, model_dir: pathlib.Path, n_out: int):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(str(model_dir))
        self.enc = AutoModel.from_config(cfg)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(cfg.hidden_size, n_out)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))


def _scan_bucket_worker(args: tuple[str, list[str], set[int], np.ndarray]) -> dict[int, tuple[int, int, float, float]]:
    """Scan one DID bucket (`u-bXXXX-*.parquet`) for wanted PIDs.
    Returns pid -> (n_posts, n_img_posts, mean_centered_nn_cos, frac_centered_nn_gt_090).
    """
    _bname, files, wanted_pids, mu_raw = args
    by_p_vecs: dict[int, list[np.ndarray]] = collections.defaultdict(list)
    by_p_imgs: collections.Counter = collections.Counter()
    by_p_tot: collections.Counter = collections.Counter()

    wanted_arr = pa.array(sorted(wanted_pids), type=pa.int64())

    for fp in files:
        tb = pq.read_table(fp, columns=["p", "modality", "n_images", "vec"])
        mask = pc.is_in(tb["p"], value_set=wanted_arr)
        tb = tb.filter(mask)
        if len(tb) == 0:
            continue
        ps = tb["p"].to_numpy()
        mods = tb["modality"].to_numpy()
        nimgs = tb["n_images"].to_numpy()
        blobs = tb["vec"].to_pylist()

        valid_idx = [i for i, m in enumerate(mods) if m != 4 and blobs[i] is not None]
        if not valid_idx:
            continue
        raw_bytes = b"".join(blobs[i] for i in valid_idx)
        mat = np.frombuffer(raw_bytes, dtype="<f2").reshape(len(valid_idx), 256).astype(np.float32)

        for row_i, orig_i in enumerate(valid_idx):
            pid = int(ps[orig_i])
            by_p_tot[pid] += 1
            if nimgs[orig_i] > 0 or mods[orig_i] in (1, 2):
                by_p_imgs[pid] += 1
            if len(by_p_vecs[pid]) < 80:
                by_p_vecs[pid].append(mat[row_i])

    out: dict[int, tuple[int, int, float, float]] = {}
    for pid, n_tot in by_p_tot.items():
        vecs = by_p_vecs[pid]
        n_im = int(by_p_imgs[pid])
        if len(vecs) < 4:
            out[pid] = (int(n_tot), n_im, 0.0, 0.0)
            continue
        V = np.stack(vecs, axis=0)
        V = l2_norm(V - mu_raw[None, :])
        S = V @ V.T
        np.fill_diagonal(S, -2.0)
        nn_max = S.max(axis=1)
        out[pid] = (
            int(n_tot),
            n_im,
            float(nn_max.mean()),
            float((nn_max >= 0.90).mean()),
        )
    return out


def compute_trimmed_centroids(
    cids: list[int],
    pids_by_c: dict[int, list[int]],
    pid_to_row: dict[int, int],
    mat_centered: np.ndarray,
    trim_q: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2-pass iteratively trimmed community centroid (drops bottom 15% outliers on pass 2)."""
    cents = np.zeros((len(cids), mat_centered.shape[1]), dtype=np.float32)
    cohs = np.zeros(len(cids), dtype=np.float32)
    for i, c in enumerate(cids):
        rows = [pid_to_row[p] for p in pids_by_c.get(c, []) if p in pid_to_row]
        if not rows:
            continue
        M = mat_centered[rows]
        c1 = l2_norm(M.mean(axis=0))
        sims1 = M @ c1
        if len(rows) >= 8:
            thresh = float(np.quantile(sims1, trim_q))
            keep = sims1 >= thresh
            if keep.sum() >= 5:
                c2 = l2_norm(M[keep].mean(axis=0))
                sims2 = M[keep] @ c2
                cents[i] = c2
                cohs[i] = float(sims2.mean())
                continue
        cents[i] = c1
        cohs[i] = float(sims1.mean())
    return cents, cohs


def score_c5_account_types(
    lake: pathlib.Path,
    out_dir: pathlib.Path,
    target_pids: list[int],
    full_dids: list[str],
    pid_home: np.ndarray,
    comm_label_by_cid: dict[int, str],
    snap: dict[str, tuple[str, str]],
) -> dict[int, tuple[str, float, float, float]]:
    """Run local `C5` `account_type_model` (`mmBERT-base` student) on `target_pids`.
    Returns `pid -> (pred_class, p_bot, p_adult, p_brand)`.
    """
    tar_p = lake / "c5-distill-20261028/colab-results/c5_models.tar"
    ext_dir = out_dir / "cache/c5_account_type_model"
    if not (ext_dir / "student.safetensors").exists():
        if not tar_p.exists():
            return {}
        ext_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_p) as tf:
            for m in tf.getmembers():
                if m.name.startswith("account_type_model/"):
                    rel = m.name.split("/", 1)[1]
                    if rel:
                        (ext_dir / rel).write_bytes(tf.extractfile(m).read())

    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    head_meta = json.loads((ext_dir / "head.json").read_text())
    classes: list[str] = head_meta["classes"]
    max_len = int(head_meta.get("max_len") or 128)
    idx_bot = classes.index("bot_or_automated")
    idx_adult = classes.index("adult_creator")
    idx_brand = classes.index("brand_or_commercial")

    texts: list[str] = []
    for p in target_pids:
        d = full_dids[p]
        nm, bio = snap.get(d, ("", ""))
        clbl = comm_label_by_cid.get(int(pid_home[p]), "")
        txt = f"{nm}\n{bio}\ncommunity: {clbl}".strip()
        texts.append(txt)

    tok = AutoTokenizer.from_pretrained(str(ext_dir))
    model = C5AccountTypeStudent(ext_dir, len(classes))
    model.load_state_dict(load_file(str(ext_dir / "student.safetensors")))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()

    order = sorted(range(len(target_pids)), key=lambda i: len(texts[i]))
    probs = np.zeros((len(target_pids), len(classes)), dtype=np.float32)
    bs = 192
    with torch.no_grad():
        for s in range(0, len(order), bs):
            chunk_idx = order[s : s + bs]
            chunk_tx = [texts[i] for i in chunk_idx]
            enc = tok(chunk_tx, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "mps")):
                lo = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            probs[chunk_idx] = torch.softmax(lo.float(), dim=-1).cpu().numpy()
            if (s // bs) % 80 == 79 and device.type == "mps":
                torch.mps.empty_cache()

    if device.type == "mps":
        torch.mps.empty_cache()

    pred_idx = probs.argmax(axis=1)
    out: dict[int, tuple[str, float, float, float]] = {}
    for i, p in enumerate(target_pids):
        out[p] = (
            classes[int(pred_idx[i])],
            float(probs[i, idx_bot]),
            float(probs[i, idx_adult]),
            float(probs[i, idx_brand]),
        )
    return out


def main() -> None:
    t0 = time.time()
    lake = pathlib.Path.home() / "catbird-lake-local"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(lake / "s17-picker-v1-20261030"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-c5", action="store_true")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    cand5_dir = lake / "s3/matched/cand5-full-matched"
    s5_dir = lake / "s5-sweep-20261030"
    s12_dir = lake / "s12-closeout-20261030"
    s12_units_dir = lake / "s12mm-colab-s12m-20261029a/vectors/units"
    s13_dir = lake / "s13-elig-20261030"
    s16_dir = lake / "s16-s17-20261030"
    claude_comm_path = lake / "claude-labels-cand5-20261028/community_labels.parquet"
    claude_prof_path = lake / "claude-labels-20261027/profile_labels.parquet"

    s6_in.set_partition("cand5", str(claude_comm_path))

    print("[audit-signals] 1/6 Loading cand5 partition, S13 scores, S5 live profiles, and quality tables...", flush=True)
    full_dids = [ln.rstrip("\n").split("\t")[1] for ln in open(cand5_dir / "producers.tsv")]
    n_prod = len(full_dids)
    did2pid = {d: i for i, d in enumerate(full_dids)}

    t_kf = pq.read_table(cand5_dir / "known_for.parquet", columns=["pid", "community", "rank"])
    pid_kf = t_kf["pid"].to_numpy().astype(np.int64)
    com_kf = t_kf["community"].to_numpy().astype(np.int64)
    rk_kf = t_kf["rank"].to_numpy()
    pid_home = np.full(n_prod, -1, dtype=np.int64)
    pid_home[pid_kf[rk_kf == 0]] = com_kf[rk_kf == 0]
    del t_kf

    comm_label_by_cid: dict[int, str] = {}
    if claude_comm_path.exists():
        t_cl_c = pq.read_table(claude_comm_path, columns=["community", "label"])
        for rc in t_cl_c.to_pylist():
            comm_label_by_cid[int(rc["community"])] = (rc.get("label") or "").strip()
        del t_cl_c

    t_s13 = pq.read_table(
        s13_dir / "s13_prod_eligibility_and_scores.parquet",
        columns=[
            "pid",
            "eligible",
            "supplement",
            "score",
            "w_lin_per_sqrt_fol",
            "w_lin_plus_curators",
            "w_lin_plus_curators_v9dem",
        ],
    )
    s13_elig = t_s13["eligible"].to_numpy().astype(bool)
    s13_sup = t_s13["supplement"].to_numpy().astype(bool)
    s13_score = t_s13["score"].to_numpy().astype(np.float64)
    s13_wlin = t_s13["w_lin_per_sqrt_fol"].to_numpy().astype(np.float64)
    s13_wcur = t_s13["w_lin_plus_curators"].to_numpy().astype(np.float64)
    s13_v9 = t_s13["w_lin_plus_curators_v9dem"].to_numpy().astype(np.float64)
    del t_s13

    purge_dids = s6_in.did_set(s5_dir / "purge_dids.txt")
    base_excl = s6_in.did_set(s13_dir / "base_safety_exclude_dids.txt")
    rule2_excl = s6_in.did_set(s5_dir / "eligibility_rule2_exclude_dids.txt")
    optout_dids = s6_in.did_set(lake / "s5-declarations-20261027/user_intents_optout_dids.tsv")
    hard_excl = purge_dids | base_excl | rule2_excl | optout_dids

    t_prof = pq.read_table(
        s5_dir / "profiles.parquet",
        columns=[
            "did",
            "found",
            "handle",
            "postsCount",
            "followersCount",
            "followsCount",
            "createdAt",
            "no_unauthenticated",
            "any_mod_label",
            "hard_label",
            "adult_graphic",
            "self_labels",
            "repo_status",
        ],
    )
    now_dt = dt.datetime(2026, 10, 30, 12, 0, 0, tzinfo=dt.timezone.utc)
    prof_ok: dict[int, tuple[str, int, int, int, float]] = {}
    for r in t_prof.to_pylist():
        d = r["did"]
        p_i = did2pid.get(d)
        if p_i is None or not s13_elig[p_i] or pid_home[p_i] < 0 or d in hard_excl:
            continue
        if not r["found"] or r["repo_status"] != "active" or r["no_unauthenticated"]:
            continue
        if r["any_mod_label"] or r["hard_label"] or r["adult_graphic"] or len(r["self_labels"] or []) > 0:
            continue
        hdl = r["handle"] or ""
        if not hdl or hdl == "handle.invalid":
            continue
        ca_str = r.get("createdAt") or ""
        age_days = 365.0
        if ca_str:
            try:
                ca = dt.datetime.fromisoformat(ca_str.replace("Z", "+00:00"))
                if ca.tzinfo is None:
                    ca = ca.replace(tzinfo=dt.timezone.utc)
                age_days = max((now_dt - ca).total_seconds() / 86400.0, 1.0)
            except Exception:
                age_days = 365.0
        prof_ok[p_i] = (
            hdl,
            int(r["postsCount"] or 0),
            int(r["followersCount"] or 0),
            int(r["followsCount"] or 0),
            float(age_days),
        )
    del t_prof

    pids_by_c: dict[int, list[int]] = collections.defaultdict(list)
    for p_i in prof_ok:
        if prof_ok[p_i][1] >= 3:
            pids_by_c[int(pid_home[p_i])].append(p_i)

    active_cids = sorted(c for c, ps in pids_by_c.items() if len(ps) >= 15)
    cand_pids = sorted(p_i for c in active_cids for p_i in pids_by_c[c])
    cand_dids = [full_dids[p] for p in cand_pids]
    cand_did_set = set(cand_dids)
    print(
        f"[audit-signals] Active communities (>=15 live elig): {len(active_cids):,}; candidate producers: {len(cand_pids):,}",
        flush=True,
    )

    # Load producer_quality.parquet for cand_pids
    pq_file = cand5_dir / "audit-quality/producer_quality.parquet"
    t_pq = pq.read_table(
        pq_file,
        columns=[
            "pid",
            "mass3k_share",
            "mutual_share",
            "modlists",
            "blockers",
            "posts",
            "originals",
            "light_engagers",
        ],
    )
    t_pq = t_pq.filter(pc.is_in(t_pq["pid"], value_set=pa.array(cand_pids, type=pa.int64())))
    pq_by_pid: dict[int, tuple[float, float, int, int, int, int, int]] = {}
    for r in t_pq.to_pylist():
        pq_by_pid[int(r["pid"])] = (
            float(r["mass3k_share"] or 0.0),
            float(r["mutual_share"] or 0.0),
            int(r["modlists"] or 0),
            int(r["blockers"] or 0),
            int(r["posts"] or 0),
            int(r["originals"] or 0),
            int(r["light_engagers"] or 0),
        )
    del t_pq

    # Scan curatelists.tsv for independent bot-list & spam/scam/slop-list owner consensus
    bot_owners_by_did: dict[str, set[str]] = collections.defaultdict(set)
    spam_owners_by_did: dict[str, set[str]] = collections.defaultdict(set)
    cl_tsv = cand5_dir / "audit-quality/curatelists.tsv"
    if cl_tsv.exists():
        with open(cl_tsv) as fh:
            for ln in fh:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                owner, name, cnt_s, mems_s = parts[1], parts[2].strip(), parts[3], parts[4]
                n_mems = int(cnt_s or 0)
                if n_mems < 2 or n_mems > 1500:
                    continue
                if GUARD_LIST_RE.search(name):
                    continue
                is_bot_l = bool(BOT_LIST_RE.match(name))
                is_spam_l = bool(SPAM_LIST_RE.search(name))
                if not (is_bot_l or is_spam_l):
                    continue
                for d in mems_s.split(","):
                    if d in cand_did_set:
                        if is_bot_l:
                            bot_owners_by_did[d].add(owner)
                        if is_spam_l:
                            spam_owners_by_did[d].add(owner)

    # Load Claude teacher profile_labels.parquet
    claude_prof_by_did: dict[str, tuple[str, str, str]] = {}
    if claude_prof_path.exists():
        t_pl = pq.read_table(
            claude_prof_path,
            columns=[
                "did",
                "haiku_account_type",
                "haiku_sensitivity",
                "haiku_confidence",
                "sonnet_account_type",
                "sonnet_sensitivity",
                "sonnet_confidence",
            ],
        )
        t_pl = t_pl.filter(pc.is_in(t_pl["did"], value_set=pa.array(cand_dids, type=pa.string())))
        for r in t_pl.to_pylist():
            at = (r.get("sonnet_account_type") or r.get("haiku_account_type") or "").strip()
            cf_v = (r.get("sonnet_confidence") or r.get("haiku_confidence") or "").strip()
            sens_raw = r.get("sonnet_sensitivity") if r.get("sonnet_sensitivity") is not None else r.get("haiku_sensitivity")
            sens_clean = sorted({str(s).strip() for s in (sens_raw or []) if s and str(s).strip() != "none"})
            claude_prof_by_did[r["did"]] = (at, cf_v, ",".join(sens_clean))
        del t_pl

    print("[audit-signals] 2/6 Computing centered S12m semantic + S16 Tier-1 social embeddings...", flush=True)
    npz = np.load(s12_dir / "pooled_all_fp16.npz")
    mm_pids = npz["mm_pids"].astype(np.int64)
    mm_fused_raw = l2_norm(npz["mm_fused"].astype(np.float32))
    mm_posts_raw = l2_norm(npz["mm_posts"].astype(np.float32))
    mm_counts = npz["mm_counts"].astype(np.int32)

    text_pids = npz["text_pids"].astype(np.int64)
    text_posts_raw = l2_norm(npz["text_posts"].astype(np.float32))

    mu_fused = mm_fused_raw.mean(axis=0)
    mu_posts = mm_posts_raw.mean(axis=0)
    mu_text_posts = text_posts_raw.mean(axis=0)

    mm_fused_c = l2_norm(mm_fused_raw - mu_fused[None, :])
    mm_posts_c = l2_norm(mm_posts_raw - mu_posts[None, :])
    text_posts_c = l2_norm(text_posts_raw - mu_text_posts[None, :])

    # Apply S16 Tier-1 prod projection head
    head = Tier1Head()
    t1_ckpt = torch.load(s16_dir / "tier1_heads.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(t1_ckpt["prod_state_dict"])
    head.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    head.to(device)
    t1_raw = np.empty_like(mm_fused_raw)
    with torch.no_grad():
        for s in range(0, len(mm_fused_raw), 32768):
            xb = torch.from_numpy(mm_fused_raw[s : s + 32768]).to(device)
            t1_raw[s : s + 32768] = head(xb).cpu().numpy()
    mu_t1 = t1_raw.mean(axis=0)
    t1_c = l2_norm(t1_raw - mu_t1[None, :])

    del head, mm_fused_raw, mm_posts_raw, text_posts_raw, t1_raw
    if device.type == "mps":
        torch.mps.empty_cache()

    pid_to_mm = {int(p): i for i, p in enumerate(mm_pids)}
    pid_to_tx = {int(p): i for i, p in enumerate(text_pids)}

    cent_fused, coh_fused = compute_trimmed_centroids(active_cids, pids_by_c, pid_to_mm, mm_fused_c)
    cent_posts, coh_posts = compute_trimmed_centroids(active_cids, pids_by_c, pid_to_mm, mm_posts_c)
    cent_tx_posts, coh_tx_posts = compute_trimmed_centroids(active_cids, pids_by_c, pid_to_tx, text_posts_c)
    cent_t1, coh_t1 = compute_trimmed_centroids(active_cids, pids_by_c, pid_to_mm, t1_c)

    np.savez_compressed(
        out_dir / "card_centroids.npz",
        cids=np.array(active_cids, dtype=np.int32),
        mu_fused=mu_fused.astype(np.float32),
        mu_posts=mu_posts.astype(np.float32),
        mu_text_posts=mu_text_posts.astype(np.float32),
        mu_t1=mu_t1.astype(np.float32),
        cent_fused=cent_fused,
        cent_posts=cent_posts,
        cent_text_posts=cent_tx_posts,
        cent_t1=cent_t1,
        coh_fused=coh_fused,
        coh_posts=coh_posts,
        coh_t1=coh_t1,
    )

    print("[audit-signals] 3/6 Computing per-candidate home & best-other-card similarity...", flush=True)
    cid_to_idx = {c: i for i, c in enumerate(active_cids)}
    substantial_mask = np.array([len(pids_by_c[c]) >= 25 for c in active_cids], dtype=bool)
    sub_indices = np.flatnonzero(substantial_mask)
    sub_cids = np.array(active_cids, dtype=np.int32)[sub_indices]
    C_posts_sub = cent_posts[sub_indices]
    C_t1_sub = cent_t1[sub_indices]

    n_c = len(cand_pids)
    out_pid = np.array(cand_pids, dtype=np.int32)
    out_did = cand_dids
    out_cid = np.array([pid_home[p] for p in cand_pids], dtype=np.int32)
    out_has_vec = np.zeros(n_c, dtype=bool)
    out_mm_cnt = np.zeros(n_c, dtype=np.int32)
    out_sem_fused = np.full(n_c, np.nan, dtype=np.float32)
    out_sem_posts = np.full(n_c, np.nan, dtype=np.float32)
    out_t1_cos = np.full(n_c, np.nan, dtype=np.float32)
    out_best_other_cid = np.full(n_c, -1, dtype=np.int32)
    out_best_other_sem = np.full(n_c, np.nan, dtype=np.float32)
    out_best_other_t1 = np.full(n_c, np.nan, dtype=np.float32)

    vec_pos = [idx for idx, p in enumerate(cand_pids) if p in pid_to_mm]
    vec_mm_rows = np.array([pid_to_mm[cand_pids[idx]] for idx in vec_pos], dtype=np.int64)
    vec_home_cidx = np.array([cid_to_idx[int(out_cid[idx])] for idx in vec_pos], dtype=np.int64)

    Vf = mm_fused_c[vec_mm_rows]
    Vp = mm_posts_c[vec_mm_rows]
    Vt = t1_c[vec_mm_rows]

    sf = (Vf * cent_fused[vec_home_cidx]).sum(axis=1)
    sp = (Vp * cent_posts[vec_home_cidx]).sum(axis=1)
    st = (Vt * cent_t1[vec_home_cidx]).sum(axis=1)

    out_has_vec[vec_pos] = True
    out_mm_cnt[vec_pos] = mm_counts[vec_mm_rows]
    out_sem_fused[vec_pos] = sf
    out_sem_posts[vec_pos] = sp
    out_t1_cos[vec_pos] = st

    for s in range(0, len(vec_pos), 8192):
        sl = slice(s, min(s + 8192, len(vec_pos)))
        pos_b = vec_pos[sl]
        sims_p = Vp[sl] @ C_posts_sub.T
        sims_t = Vt[sl] @ C_t1_sub.T
        for bi, idx_global in enumerate(pos_b):
            hc = int(out_cid[idx_global])
            mask_self = sub_cids == hc
            sims_p[bi, mask_self] = -2.0
            sims_t[bi, mask_self] = -2.0
        best_j = sims_p.argmax(axis=1)
        out_best_other_cid[pos_b] = sub_cids[best_j]
        out_best_other_sem[pos_b] = sims_p[np.arange(len(pos_b)), best_j]
        out_best_other_t1[pos_b] = sims_t.max(axis=1)

    del mm_fused_c, mm_posts_c, text_posts_c, t1_c, Vf, Vp, Vt

    # Top-60 ranked per community get both slab post-duplication scan AND C5 account_type_model MPS scoring
    top_scan_pids_set: set[int] = set()
    for c in active_cids:
        ps_sorted = sorted(pids_by_c[c], key=lambda p_i: (-float(s13_score[p_i]), -prof_ok[p_i][2], full_dids[p_i]))
        top_scan_pids_set.update(ps_sorted[:60])

    print(
        f"[audit-signals] 4/6 Scanning 64 S12m post buckets for template/near-duplicate spam on {len(top_scan_pids_set):,} top candidates...",
        flush=True,
    )
    by_bucket: dict[str, list[str]] = collections.defaultdict(list)
    for fp in sorted(s12_units_dir.glob("u-b*.parquet")):
        m = BUCKET_RE.match(fp.name)
        if m:
            by_bucket[m.group(1)].append(str(fp))

    slab_stats: dict[int, tuple[int, int, float, float]] = {}
    t_scan = time.time()
    tasks = [(b, fs, top_scan_pids_set, mu_posts) for b, fs in sorted(by_bucket.items())]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_scan_bucket_worker, tasks):
            slab_stats.update(res)
    print(f"[audit-signals] Slab scan finished in {time.time() - t_scan:.1f}s ({len(slab_stats):,} producers scored)", flush=True)

    c5_by_pid: dict[int, tuple[str, float, float, float]] = {}
    if not args.skip_c5:
        t_c5 = time.time()
        snap = s6_in.profile_snapshot()
        sorted_top_pids = sorted(top_scan_pids_set)
        print(
            f"[audit-signals] 5/6 Running local C5 account_type_model (mmBERT-base) on MPS for {len(sorted_top_pids):,} top candidates...",
            flush=True,
        )
        c5_by_pid = score_c5_account_types(
            lake, out_dir, sorted_top_pids, full_dids, pid_home, comm_label_by_cid, snap
        )
        del snap
        print(f"[audit-signals] C5 account_type scoring finished in {time.time() - t_c5:.1f}s ({len(c5_by_pid):,} scored)", flush=True)

    print("[audit-signals] 6/6 Writing candidate_signals.parquet...", flush=True)
    out_slab_n = np.zeros(n_c, dtype=np.int32)
    out_slab_img_frac = np.zeros(n_c, dtype=np.float32)
    out_slab_nn_mean = np.zeros(n_c, dtype=np.float32)
    out_slab_dup_frac = np.zeros(n_c, dtype=np.float32)

    out_posts_cnt = np.zeros(n_c, dtype=np.int32)
    out_followers = np.zeros(n_c, dtype=np.int32)
    out_follows = np.zeros(n_c, dtype=np.int32)
    out_follow_ratio = np.zeros(n_c, dtype=np.float32)
    out_posts_per_day = np.zeros(n_c, dtype=np.float32)
    out_v9_gap = np.zeros(n_c, dtype=np.float32)
    out_curator_boost = np.zeros(n_c, dtype=np.float32)

    out_m3k = np.zeros(n_c, dtype=np.float32)
    out_mut = np.zeros(n_c, dtype=np.float32)
    out_modlists = np.zeros(n_c, dtype=np.int32)
    out_blockers = np.zeros(n_c, dtype=np.int32)
    out_modlists_p1k = np.zeros(n_c, dtype=np.float32)
    out_blockers_p1k = np.zeros(n_c, dtype=np.float32)
    out_pq_posts = np.zeros(n_c, dtype=np.int32)
    out_pq_orig = np.zeros(n_c, dtype=np.int32)
    out_pq_leng = np.zeros(n_c, dtype=np.int32)

    out_bot_lists = np.zeros(n_c, dtype=np.int32)
    out_spam_lists = np.zeros(n_c, dtype=np.int32)

    out_cl_type: list[str] = []
    out_cl_conf: list[str] = []
    out_cl_sens: list[str] = []

    out_c5_type: list[str] = []
    out_c5_p_bot = np.full(n_c, np.nan, dtype=np.float32)
    out_c5_p_adult = np.full(n_c, np.nan, dtype=np.float32)
    out_c5_p_brand = np.full(n_c, np.nan, dtype=np.float32)

    for idx, p in enumerate(cand_pids):
        d = out_did[idx]
        _hdl, pcnt, fol, fwing, age_d = prof_ok[p]
        out_posts_cnt[idx] = pcnt
        out_followers[idx] = fol
        out_follows[idx] = fwing
        out_follow_ratio[idx] = float(fwing) / max(float(fol), 50.0)
        out_posts_per_day[idx] = float(pcnt) / max(age_d, 14.0)
        wcur = float(s13_wcur[p])
        wlin = float(s13_wlin[p])
        wv9 = float(s13_v9[p])
        out_v9_gap[idx] = max(0.0, 1.0 - (wv9 / max(wcur, 1e-9))) if wcur > 0 else 0.0
        out_curator_boost[idx] = max(0.0, wcur - wlin)

        st_tup = slab_stats.get(p)
        if st_tup is not None:
            sn, simg, snn, sdup = st_tup
            out_slab_n[idx] = sn
            out_slab_img_frac[idx] = float(simg) / max(sn, 1)
            out_slab_nn_mean[idx] = snn
            out_slab_dup_frac[idx] = sdup

        pq_tup = pq_by_pid.get(p)
        if pq_tup is not None:
            m3k, mut, modl, blk, pqp, pqo, pql = pq_tup
            denom_1k = max(float(fol), 100.0) / 1000.0
            out_m3k[idx] = m3k
            out_mut[idx] = mut
            out_modlists[idx] = modl
            out_blockers[idx] = blk
            out_modlists_p1k[idx] = float(modl) / denom_1k
            out_blockers_p1k[idx] = float(blk) / denom_1k
            out_pq_posts[idx] = pqp
            out_pq_orig[idx] = pqo
            out_pq_leng[idx] = pql

        out_bot_lists[idx] = len(bot_owners_by_did.get(d, ()))
        out_spam_lists[idx] = len(spam_owners_by_did.get(d, ()))

        cl_tup = claude_prof_by_did.get(d)
        if cl_tup is not None:
            out_cl_type.append(cl_tup[0])
            out_cl_conf.append(cl_tup[1])
            out_cl_sens.append(cl_tup[2])
        else:
            out_cl_type.append("")
            out_cl_conf.append("")
            out_cl_sens.append("")

        c5_tup = c5_by_pid.get(p)
        if c5_tup is not None:
            out_c5_type.append(c5_tup[0])
            out_c5_p_bot[idx] = c5_tup[1]
            out_c5_p_adult[idx] = c5_tup[2]
            out_c5_p_brand[idx] = c5_tup[3]
        else:
            out_c5_type.append("")

    tbl = pa.table(
        {
            "pid": out_pid,
            "did": out_did,
            "community": out_cid,
            "has_vec": out_has_vec,
            "mm_post_count": out_mm_cnt,
            "sem_cos_fused": out_sem_fused,
            "sem_cos_posts": out_sem_posts,
            "t1_cos": out_t1_cos,
            "best_other_cid": out_best_other_cid,
            "best_other_sem_cos": out_best_other_sem,
            "best_other_t1_cos": out_best_other_t1,
            "slab_n_posts": out_slab_n,
            "slab_img_frac": out_slab_img_frac,
            "slab_nn_mean_cos": out_slab_nn_mean,
            "slab_near_dup_frac": out_slab_dup_frac,
            "posts_count": out_posts_cnt,
            "followers_count": out_followers,
            "follows_count": out_follows,
            "follow_ratio": out_follow_ratio,
            "posts_per_day": out_posts_per_day,
            "v9_demotion_gap": out_v9_gap,
            "curator_boost": out_curator_boost,
            "pq_mass3k_share": out_m3k,
            "pq_mutual_share": out_mut,
            "pq_modlists": out_modlists,
            "pq_blockers": out_blockers,
            "pq_modlists_per_1k": out_modlists_p1k,
            "pq_blockers_per_1k": out_blockers_p1k,
            "pq_posts": out_pq_posts,
            "pq_originals": out_pq_orig,
            "pq_light_engagers": out_pq_leng,
            "bot_list_owners": out_bot_lists,
            "spam_list_owners": out_spam_lists,
            "claude_acct_type": pa.array(out_cl_type, type=pa.string()),
            "claude_acct_conf": pa.array(out_cl_conf, type=pa.string()),
            "claude_sens": pa.array(out_cl_sens, type=pa.string()),
            "c5_pred_type": pa.array(out_c5_type, type=pa.string()),
            "c5_p_bot": out_c5_p_bot,
            "c5_p_adult": out_c5_p_adult,
            "c5_p_brand": out_c5_p_brand,
            "supplement": np.array([bool(s13_sup[p]) for p in cand_pids], dtype=bool),
            "s13_score": np.array([float(s13_score[p]) for p in cand_pids], dtype=np.float32),
        }
    )
    pq.write_table(tbl, out_dir / "candidate_signals.parquet", compression="zstd")
    print(f"[audit-signals] Done in {time.time() - t0:.1f}s -> {out_dir / 'candidate_signals.parquet'}", flush=True)


if __name__ == "__main__":
    main()
