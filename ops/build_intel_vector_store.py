#!/usr/bin/env python3
"""Compile S12m + S16 Tier-1 + S17 centroids + S13/Candidate-Signals into a flat,
isotropic-centered, memory-mappable serving store (`~/catbird-lake-local/intel-serving-v4/`).

Zero external DB required; loads in ~2s in Bun via TypedArray views over file buffers.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
import torch.nn.functional as F


class Tier1Head(nn.Module):
    def __init__(self, d_in: int = 256, d_hid: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.fc2(F.gelu(self.fc1(x))), p=2, dim=-1)


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def quantize_int8(mat_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize (N, 256) float32 vectors to signed int8 + uint16 raw IEEE-754 fp16 scale bits."""
    abs_max = np.maximum(np.abs(mat_f32).max(axis=1), 1e-8)
    scale_f16 = (abs_max / 127.0).astype(np.float16)
    scale_f32 = np.maximum(scale_f16.astype(np.float32), 1e-8)
    q8 = np.clip(np.rint(mat_f32 / scale_f32[:, None]), -127, 127).astype(np.int8)
    scale_u16 = scale_f16.view(np.uint16)
    return q8, scale_u16


def quantize_single_b64(vec_f32: np.ndarray) -> tuple[str, int]:
    q8, sc = quantize_int8(l2_norm(vec_f32[None, :].astype(np.float32)))
    return base64.b64encode(q8[0].tobytes()).decode("ascii"), int(sc[0])


def build_store(lake: pathlib.Path, out_dir: pathlib.Path) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    cc = np.load(lake / "s17-picker-v1-20261030/card_centroids.npz")
    cids = cc["cids"].astype(np.int32)
    cent_posts = cc["cent_posts"].astype(np.float32)
    cent_fused = cc["cent_fused"].astype(np.float32)
    cent_t1 = cc["cent_t1"].astype(np.float32)
    mu_posts_arr = cc["mu_posts"].astype(np.float32)
    mu_fused_arr = cc["mu_fused"].astype(np.float32)
    mu_t1_arr = cc["mu_t1"].astype(np.float32)

    print("[1/6] Loading S13 producer directory & quality scores...", flush=True)
    s13_path = lake / "s13-elig-20261030/s13_prod_eligibility_and_scores.parquet"
    s13 = pq.read_table(s13_path, columns=["pid", "did", "eligible", "score"])
    pids_s13 = s13["pid"].to_numpy()
    n_prod = int(pids_s13.max()) + 1
    assert n_prod == len(pids_s13), f"Expected dense PIDs 0..{n_prod - 1}, got {len(pids_s13)}"
    order = np.argsort(pids_s13)
    dids = s13["did"].to_pylist()
    if not np.all(order == np.arange(n_prod)):
        dids = [dids[i] for i in order]
    eligible = s13["eligible"].to_numpy()[order].astype(np.bool_)
    s13_score = np.nan_to_num(s13["score"].to_numpy()[order], nan=0.0, posinf=1.0, neginf=0.0)
    qual_bps = np.clip(np.rint(10000.0 / (1.0 + np.exp(-s13_score))), 0, 10000).astype(np.uint16)

    print("[2/6] Loading KnownFor top-2 communities per producer...", flush=True)
    kf = pq.read_table(
        lake / "communities-cand5-damped/known_for.parquet",
        columns=["pid", "community", "rank", "hide_from_recs"],
    )
    kf_pid = kf["pid"].to_numpy()
    kf_comm = kf["community"].to_numpy()
    kf_rank = kf["rank"].to_numpy()
    kf_hide = kf["hide_from_recs"].to_numpy()

    top_c0 = np.full(n_prod, 0xFFFFFFFF, dtype=np.uint32)
    top_c1 = np.full(n_prod, 0xFFFFFFFF, dtype=np.uint32)
    hide_recs = np.zeros(n_prod, dtype=np.bool_)

    valid_kf = kf_pid < n_prod
    m0 = valid_kf & (kf_rank == 0)
    top_c0[kf_pid[m0]] = kf_comm[m0]
    hide_recs[kf_pid[m0]] = kf_hide[m0]
    m1 = valid_kf & (kf_rank == 1)
    top_c1[kf_pid[m1]] = kf_comm[m1]

    print("[3/6] Loading S17 candidate_signals (rage-bait, spam, bot, original share)...", flush=True)
    rage_pm = np.full(n_prod, 80, dtype=np.uint16)
    spam_pm = np.full(n_prod, 30, dtype=np.uint16)
    bot_pm = np.full(n_prod, 20, dtype=np.uint16)
    orig_pm = np.full(n_prod, 500, dtype=np.uint16)
    has_c5 = np.zeros(n_prod, dtype=np.bool_)

    cs_path = lake / "s17-picker-v1-20261030/candidate_signals.parquet"
    if cs_path.exists():
        cs = pq.read_table(
            cs_path,
            columns=[
                "pid",
                "pq_modlists_per_1k",
                "pq_blockers_per_1k",
                "v9_demotion_gap",
                "spam_list_owners",
                "bot_list_owners",
                "c5_p_bot",
                "pq_posts",
                "pq_originals",
                "slab_near_dup_frac",
            ],
        )
        cpid = cs["pid"].to_numpy()
        ok = (cpid >= 0) & (cpid < n_prod)
        cpid = cpid[ok]
        mod_1k = np.nan_to_num(cs["pq_modlists_per_1k"].to_numpy()[ok], nan=0.0)
        blk_1k = np.nan_to_num(cs["pq_blockers_per_1k"].to_numpy()[ok], nan=0.0)
        v9_gap = np.nan_to_num(cs["v9_demotion_gap"].to_numpy()[ok], nan=0.0)
        spam_own = np.nan_to_num(cs["spam_list_owners"].to_numpy()[ok], nan=0.0)
        bot_own = np.nan_to_num(cs["bot_list_owners"].to_numpy()[ok], nan=0.0)
        p_bot = np.nan_to_num(cs["c5_p_bot"].to_numpy()[ok], nan=0.0)
        pq_p = np.maximum(np.nan_to_num(cs["pq_posts"].to_numpy()[ok], nan=1.0), 1.0)
        pq_o = np.clip(np.nan_to_num(cs["pq_originals"].to_numpy()[ok], nan=0.5), 0.0, pq_p)
        dup_frac = np.nan_to_num(cs["slab_near_dup_frac"].to_numpy()[ok], nan=0.0)

        rb_raw = (
            np.tanh(blk_1k / 12.0) * 0.45
            + np.tanh(mod_1k / 8.0) * 0.30
            + np.clip(v9_gap, 0.0, 1.0) * 0.25
        )
        sp_raw = np.clip(np.tanh(spam_own / 3.0) * 0.65 + dup_frac * 0.35, 0.0, 1.0)
        bt_raw = np.clip(np.maximum(p_bot, np.tanh(bot_own / 3.0)), 0.0, 1.0)

        rage_pm[cpid] = np.clip(np.rint(rb_raw * 1000.0), 0, 1000).astype(np.uint16)
        spam_pm[cpid] = np.clip(np.rint(sp_raw * 1000.0), 0, 1000).astype(np.uint16)
        bot_pm[cpid] = np.clip(np.rint(bt_raw * 1000.0), 0, 1000).astype(np.uint16)
        orig_pm[cpid] = np.clip(np.rint((pq_o / pq_p) * 1000.0), 0, 1000).astype(np.uint16)
        has_c5[cpid] = True

    print("[4/6] Centering S12m all & preT vectors and computing centered S16 Tier-1 social vectors...", flush=True)
    z_all = np.load(lake / "s12-closeout-20261030/pooled_all_int8.npz")
    all_pids = z_all["pids"].astype(np.int32)
    all_raw_unit = l2_norm(z_all["vec_i8"].astype(np.float32) * z_all["scale"].astype(np.float32)[:, None])
    all_centered = l2_norm(all_raw_unit - mu_fused_arr[None, :])
    all_i8, all_scale_u16 = quantize_int8(all_centered)
    all_counts = np.clip(z_all["counts"], 0, 65535).astype(np.uint16)
    n_vec = len(all_pids)

    vec_row_by_pid = np.full(n_prod, -1, dtype=np.int32)
    valid_all = (all_pids >= 0) & (all_pids < n_prod)
    vec_row_by_pid[all_pids[valid_all]] = np.arange(n_vec, dtype=np.int32)[valid_all]

    z_pre = np.load(lake / "s12-closeout-20261030/pooled_preT_int8.npz")
    pre_pids = z_pre["pids"].astype(np.int32)
    pre_raw_unit = l2_norm(z_pre["vec_i8"].astype(np.float32) * z_pre["scale"].astype(np.float32)[:, None])
    pre_centered = l2_norm(pre_raw_unit - mu_fused_arr[None, :])
    pre_i8_src, pre_scale_u16_src = quantize_int8(pre_centered)
    pre_counts_src = np.clip(z_pre["counts"], 0, 65535).astype(np.uint16)

    pre_i8_aligned = np.zeros((n_vec, 256), dtype=np.int8)
    pre_scale_aligned = np.zeros(n_vec, dtype=np.uint16)
    pre_counts_aligned = np.zeros(n_vec, dtype=np.uint16)
    has_pre_by_pid = np.zeros(n_prod, dtype=np.bool_)

    valid_pre = (pre_pids >= 0) & (pre_pids < n_prod)
    mapped_rows = vec_row_by_pid[pre_pids[valid_pre]]
    in_all = mapped_rows >= 0
    target_rows = mapped_rows[in_all]
    src_rows = np.nonzero(valid_pre)[0][in_all]
    pre_i8_aligned[target_rows] = pre_i8_src[src_rows]
    pre_scale_aligned[target_rows] = pre_scale_u16_src[src_rows]
    pre_counts_aligned[target_rows] = pre_counts_src[src_rows]
    has_pre_by_pid[pre_pids[valid_pre][in_all]] = True

    head = Tier1Head()
    ckpt = torch.load(lake / "s16-s17-20261030/tier1_heads.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["prod_state_dict"])
    head.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    head.to(device)

    t1_i8 = np.empty((n_vec, 256), dtype=np.int8)
    t1_scale_u16 = np.empty(n_vec, dtype=np.uint16)
    batch = 65536
    with torch.no_grad():
        for s in range(0, n_vec, batch):
            e = min(s + batch, n_vec)
            yb = head(torch.from_numpy(all_raw_unit[s:e]).to(device)).cpu().numpy()
            yb_c = l2_norm(yb - mu_t1_arr[None, :])
            qb, sb = quantize_int8(yb_c)
            t1_i8[s:e] = qb
            t1_scale_u16[s:e] = sb
    if device.type == "mps":
        torch.mps.empty_cache()

    print("[5/6] Packing flat binary tables...", flush=True)
    (out_dir / "dids.txt").write_bytes(("\n".join(dids) + "\n").encode("utf-8"))

    has_all_by_pid = vec_row_by_pid >= 0
    flags = (
        ((eligible & ~hide_recs).astype(np.uint16) << 0)
        | (has_all_by_pid.astype(np.uint16) << 1)
        | (has_pre_by_pid.astype(np.uint16) << 2)
        | (has_all_by_pid.astype(np.uint16) << 3)
        | (has_c5.astype(np.uint16) << 4)
        | (hide_recs.astype(np.uint16) << 5)
    )

    sig_dtype = np.dtype(
        [
            ("flags", "<u2"),
            ("qualityScoreBps", "<u2"),
            ("rageBaitPermille", "<u2"),
            ("spamPermille", "<u2"),
            ("botPermille", "<u2"),
            ("originalSharePermille", "<u2"),
            ("topCommunity0", "<u4"),
            ("topCommunity1", "<u4"),
            ("vecRow", "<i4"),
        ]
    )
    signals = np.empty(n_prod, dtype=sig_dtype)
    signals["flags"] = flags
    signals["qualityScoreBps"] = qual_bps
    signals["rageBaitPermille"] = rage_pm
    signals["spamPermille"] = spam_pm
    signals["botPermille"] = bot_pm
    signals["originalSharePermille"] = orig_pm
    signals["topCommunity0"] = top_c0
    signals["topCommunity1"] = top_c1
    signals["vecRow"] = vec_row_by_pid
    (out_dir / "signals.bin").write_bytes(signals.tobytes())

    meta_dtype = np.dtype([("scaleFp16", "<u2"), ("count", "<u2")])
    for prefix, q_arr, s_arr, c_arr in [
        ("vec_all", all_i8, all_scale_u16, all_counts),
        ("vec_preT", pre_i8_aligned, pre_scale_aligned, pre_counts_aligned),
        ("vec_t1", t1_i8, t1_scale_u16, all_counts),
    ]:
        meta = np.empty(n_vec, dtype=meta_dtype)
        meta["scaleFp16"] = s_arr
        meta["count"] = c_arr
        (out_dir / f"{prefix}_i8.bin").write_bytes(q_arr.tobytes())
        (out_dir / f"{prefix}_meta.bin").write_bytes(meta.tobytes())

    print("[6/6] Packing all Claude-labeled + S17 community centroids and hostility pairs...", flush=True)
    labels_by_cid: dict[int, dict] = {}
    lbl_path = lake / "claude-labels-cand5-20261028/community_labels.parquet"
    if lbl_path.exists():
        for row in pq.read_table(lbl_path).to_pylist():
            labels_by_cid[int(row["community"])] = row

    s17_by_cid: dict[int, dict] = {}
    s17_bundle_path = lake / "s17-picker-v1-20261030/intel_serving_v2.json"
    if s17_bundle_path.exists():
        bdata = json.loads(s17_bundle_path.read_text("utf-8"))
        for c in bdata.get("communities", []):
            s17_by_cid[int(c["communityId"])] = c

    hostile_by_cid: dict[int, list[int]] = {}
    hp_path = lake / "s14-hostility-20261030/H_prod_cand5_full.parquet"
    if hp_path.exists():
        ht = pq.read_table(hp_path, columns=["c_actor", "c_target", "is_high_top1pct"])
        hi = ht["is_high_top1pct"].to_numpy()
        for a, b in zip(ht["c_actor"].to_numpy()[hi], ht["c_target"].to_numpy()[hi]):
            hostile_by_cid.setdefault(int(a), []).append(int(b))

    communities_out = []
    s17_cid_set = set()
    for idx, cid_raw in enumerate(cids):
        cid = int(cid_raw)
        s17_cid_set.add(cid)
        lbl = labels_by_cid.get(cid, {})
        s17 = s17_by_cid.get(cid, {})
        p_b64, p_sc = quantize_single_b64(cent_posts[idx])
        f_b64, f_sc = quantize_single_b64(cent_fused[idx])
        t_b64, t_sc = quantize_single_b64(cent_t1[idx])
        sens = [str(x) for x in (lbl.get("sensitivity") or []) if x and str(x) != "none"]
        communities_out.append(
            {
                "communityId": cid,
                "label": s17.get("title") or lbl.get("label") or f"Community #{cid}",
                "description": s17.get("description") or lbl.get("description") or "",
                "primaryLanguage": lbl.get("primary_language") or "en",
                "communityType": lbl.get("community_type") or "organic",
                "onboardingFit": lbl.get("onboarding_fit") or "ok",
                "macroCategory": s17.get("macroCategory") or "",
                "sceneSlug": s17.get("sceneSlug") or "",
                "isS17Clean": cid in s17_by_cid,
                "sensitivity": sens,
                "samplePostUris": s17.get("samplePostUris") or [],
                "sampleAccountDids": s17.get("sampleAccountDids") or [],
                "centPostsQ8B64": p_b64,
                "centPostsScaleFp16": p_sc,
                "centFusedQ8B64": f_b64,
                "centFusedScaleFp16": f_sc,
                "centT1Q8B64": t_b64,
                "centT1ScaleFp16": t_sc,
                "hostileTargets": hostile_by_cid.get(cid, [])[:32],
            }
        )

    rows_by_cid: dict[int, list[int]] = {}
    for c0, vr in zip(top_c0[has_all_by_pid], vec_row_by_pid[has_all_by_pid]):
        ic = int(c0)
        if ic not in s17_cid_set and ic in labels_by_cid and len(rows_by_cid.get(ic, ())) < 256:
            rows_by_cid.setdefault(ic, []).append(int(vr))

    t1_scale_f32 = t1_scale_u16.view(np.float16).astype(np.float32)
    for cid in sorted(set(labels_by_cid.keys()) - s17_cid_set):
        rrows = rows_by_cid.get(cid)
        if not rrows:
            continue
        lbl = labels_by_cid[cid]
        vp = l2_norm(all_centered[rrows].mean(axis=0, keepdims=True))[0]
        vt = l2_norm((t1_i8[rrows].astype(np.float32) * t1_scale_f32[rrows, None]).mean(axis=0, keepdims=True))[0]
        p_b64, p_sc = quantize_single_b64(vp)
        t_b64, t_sc = quantize_single_b64(vt)
        sens = [str(x) for x in (lbl.get("sensitivity") or []) if x and str(x) != "none"]
        communities_out.append(
            {
                "communityId": cid,
                "label": lbl.get("label") or f"Community #{cid}",
                "description": lbl.get("description") or "",
                "primaryLanguage": lbl.get("primary_language") or "en",
                "communityType": lbl.get("community_type") or "organic",
                "onboardingFit": lbl.get("onboarding_fit") or "extra",
                "macroCategory": "",
                "sceneSlug": "",
                "isS17Clean": False,
                "sensitivity": sens,
                "samplePostUris": [],
                "sampleAccountDids": [],
                "centPostsQ8B64": p_b64,
                "centPostsScaleFp16": p_sc,
                "centFusedQ8B64": p_b64,
                "centFusedScaleFp16": p_sc,
                "centT1Q8B64": t_b64,
                "centT1ScaleFp16": t_sc,
                "hostileTargets": hostile_by_cid.get(cid, [])[:32],
            }
        )

    (out_dir / "communities.json").write_text(json.dumps(communities_out, separators=(",", ":")), encoding="utf-8")

    t1_sd = ckpt["prod_state_dict"]
    t1_export = {
        "d_in": 256,
        "d_hid": 512,
        "fc1_weight_f16_b64": base64.b64encode(t1_sd["fc1.weight"].cpu().numpy().astype(np.float16).tobytes()).decode("ascii"),
        "fc1_bias_f16_b64": base64.b64encode(t1_sd["fc1.bias"].cpu().numpy().astype(np.float16).tobytes()).decode("ascii"),
        "fc2_weight_f16_b64": base64.b64encode(t1_sd["fc2.weight"].cpu().numpy().astype(np.float16).tobytes()).decode("ascii"),
        "fc2_bias_f16_b64": base64.b64encode(t1_sd["fc2.bias"].cpu().numpy().astype(np.float16).tobytes()).decode("ascii"),
    }
    (out_dir / "tier1_head_f16.json").write_text(json.dumps(t1_export), encoding="utf-8")

    h = hashlib.sha256()
    for fname in [
        "dids.txt",
        "signals.bin",
        "vec_all_i8.bin",
        "vec_all_meta.bin",
        "vec_preT_i8.bin",
        "vec_preT_meta.bin",
        "vec_t1_i8.bin",
        "vec_t1_meta.bin",
        "communities.json",
        "tier1_head_f16.json",
    ]:
        fp = out_dir / fname
        h.update(f"{fname}:{fp.stat().st_size}:".encode())
        with fp.open("rb") as f:
            h.update(f.read(65536))
    gen = f"v4-{h.hexdigest()[:16]}"

    manifest = {
        "version": "catbird.intel.vectors.v4",
        "generation": gen,
        "isCentered": True,
        "builtAtUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wallSeconds": round(time.time() - t0, 2),
        "modelId": "eg2-256-q8",
        "socialModelId": "s16-t1-256-q8",
        "dims": 256,
        "producerCount": n_prod,
        "vectorizedProducerCount": n_vec,
        "preTProducerCount": int(has_pre_by_pid.sum()),
        "communityCentroidCount": len(communities_out),
        "signalRecordBytes": 24,
        "muPosts": mu_posts_arr.tolist(),
        "muFused": mu_fused_arr.tolist(),
        "muT1": mu_t1_arr.tolist(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done in {manifest['wallSeconds']}s -> generation {gen}", flush=True)
    return manifest


if __name__ == "__main__":
    lake_dir = pathlib.Path(os.environ.get("CATBIRD_LAKE_DIR", "/Users/joshlacalamito/catbird-lake-local"))
    out = pathlib.Path(os.environ.get("INTEL_VECTOR_STORE_DIR", str(lake_dir / "intel-serving-v4")))
    print(json.dumps(build_store(lake_dir, out), indent=2))
