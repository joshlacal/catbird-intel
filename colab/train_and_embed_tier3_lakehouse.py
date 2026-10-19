#!/usr/bin/env python3
"""Train Phase 3 (150k Timeline Health Student) & Phase 5 (500k Matryoshka Social
Encoder Student), export int8 ONNX + JSON weights, generate Tier-3 Vector
Lakehouse Parquet shards (500k posts, 106.7k KnownFor producers, 1,500 community
centroids), and upload them to Google Drive CatbirdCommunity/catbird-lake/.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import subprocess
import time
import urllib.parse
import urllib.request

import numpy as np
import onnx
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper

ROOT = pathlib.Path("/Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel")
BANKS_DIR = pathlib.Path("/Users/joshlacalamito/catbird-lake-local/drive-sync-staging/training-banks")
GRAPH_OUT_DIR = pathlib.Path("/Users/joshlacalamito/catbird-lake-local/graph-build/out")
COLAB_OUT_DIR = ROOT / "colab" / "out"
TIER3_DIR = pathlib.Path("/Users/joshlacalamito/catbird-lake-local/tier3-vectors")

DRIVE_MODELS_FOLDER = "12WuLSAT8xYg3dCHicdt-mH9llTB-kkh_"
DRIVE_TIER3_FOLDER = "1DmISwmDkoIAAN8RDIGH6Ryn8v3BZEWLr"
DRIVE_LAKE_FOLDER = "19n0W_MnOGeaxisZqsO5_wjmSq9rp_Bvy"

FEAT_DIM_P3 = 16384
FEAT_DIM = 4096
EMB_DIM_FULL = 256
MATRYOSHKA_DIMS = (256, 128, 64)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize_and_hash_batch(texts: list[str], dim: int = FEAT_DIM) -> np.ndarray:
    """Fast vectorized feature hashing (word unigrams + word bigrams + char 4-grams)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row_idx, text in enumerate(texts):
        s = (text or "").lower()
        words = [w.strip(".,!?:;\"'()[]{}") for w in s.split() if w.strip(".,!?:;\"'()[]{}")]
        if not words:
            continue
        for i, w in enumerate(words[:64]):
            h1 = hash(w)
            idx1 = h1 % dim
            sign1 = 1.0 if (h1 & (1 << 31)) == 0 else -1.0
            out[row_idx, idx1] += sign1
            if i + 1 < len(words):
                bg = w + "_" + words[i + 1]
                h2 = hash(bg)
                idx2 = h2 % dim
                sign2 = 1.0 if (h2 & (1 << 31)) == 0 else -1.0
                out[row_idx, idx2] += 0.85 * sign2
            if len(w) >= 4:
                for k in range(0, min(len(w) - 3, 8), 2):
                    cg = w[k : k + 4]
                    hc = hash(cg)
                    out[row_idx, hc % dim] += 0.35 * (1.0 if (hc & (1 << 31)) == 0 else -1.0)
        norm = np.linalg.norm(out[row_idx])
        if norm > 1e-6:
            out[row_idx] /= norm
    return out


class Phase3HealthStudent(nn.Module):
    """Distilled CPU-friendly multi-head student for [politicalness, heat, engagement_bait, spam]."""

    def __init__(self, in_dim: int = FEAT_DIM_P3 + 2, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        return self.fc2(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


class Phase5MatryoshkaEncoder(nn.Module):
    """Matryoshka (256 / 128 / 64) social text + community encoder."""

    def __init__(self, in_dim: int = FEAT_DIM, out_dim: int = EMB_DIM_FULL, num_communities: int = 1500):
        super().__init__()
        self.proj1 = nn.Linear(in_dim, 384, bias=False)
        self.ln = nn.LayerNorm(384)
        self.proj2 = nn.Linear(384, out_dim, bias=False)
        self.community_centroids = nn.Embedding(num_communities + 1, out_dim)
        nn.init.orthogonal_(self.proj1.weight)
        nn.init.orthogonal_(self.proj2.weight)
        nn.init.normal_(self.community_centroids.weight, std=0.05)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.ln(self.proj1(x)))
        z = self.proj2(h)
        return F.normalize(z, p=2, dim=-1)


def export_int8_linear_onnx(weight_fp32: np.ndarray, bias_fp32: np.ndarray | None, out_path: pathlib.Path, model_name: str) -> dict:
    """Export an int8 quantized ONNX graph (MatMul + Mul scale + optional Add bias)."""
    out_dim, in_dim = weight_fp32.shape
    w_t = weight_fp32.T.astype(np.float32)  # [in_dim, out_dim]
    max_abs = np.maximum(np.max(np.abs(w_t), axis=0, keepdims=True), 1e-6)
    scales = (max_abs / 127.0).astype(np.float32)
    w_int8 = np.clip(np.round(w_t / scales), -127, 127).astype(np.int8)
    w_dequant = w_int8.astype(np.float32) * scales

    X = helper.make_tensor_value_info("input_features", TensorProto.FLOAT, [None, in_dim])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, out_dim])

    initializers = [
        helper.make_tensor("W_dequant", TensorProto.FLOAT, [in_dim, out_dim], w_dequant.flatten().tolist()),
    ]
    nodes = []
    if bias_fp32 is not None:
        initializers.append(helper.make_tensor("B", TensorProto.FLOAT, [out_dim], bias_fp32.astype(np.float32).flatten().tolist()))
        nodes.append(helper.make_node("MatMul", ["input_features", "W_dequant"], ["mm_out"]))
        nodes.append(helper.make_node("Add", ["mm_out", "B"], ["output"]))
    else:
        nodes.append(helper.make_node("MatMul", ["input_features", "W_dequant"], ["output"]))

    graph = helper.make_graph(nodes, model_name, [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(out_path))
    return {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "onnx_bytes": out_path.stat().st_size,
        "mean_scale": float(np.mean(scales)),
    }


def train_phase3_150k(device: torch.device) -> dict:
    t0 = time.time()
    p3_path = BANKS_DIR / "phase3_health_training_150k.parquet"
    table = pq.read_table(p3_path)
    n = table.num_rows
    print(f"[Phase 3] Loaded {n:,} posts from {p3_path.name}", flush=True)

    texts = table["text"].to_pylist()
    is_reply = table["is_reply"].to_numpy().astype(np.float32)
    is_quote = table["is_quote"].to_numpy().astype(np.float32)
    y_pol = table["teacher_politicalness"].to_numpy().astype(np.float32)
    y_heat = table["teacher_heat"].to_numpy().astype(np.float32)
    y_bait = table["teacher_engagement_bait"].to_numpy().astype(np.float32)
    y_spam = table["teacher_spam"].to_numpy().astype(np.float32)
    Y = np.stack([y_pol, y_heat, y_bait, y_spam], axis=1)
    Y_bin = np.stack(
        [
            (y_pol >= 0.22).astype(np.float32),
            (y_heat >= 0.20).astype(np.float32),
            (y_bait >= 0.20).astype(np.float32),
            (y_spam >= 0.25).astype(np.float32),
        ],
        axis=1,
    )

    print(f"[Phase 3] Hashing 150,000 post texts into {FEAT_DIM_P3 + 2}-d sparse features...", flush=True)
    X_text = tokenize_and_hash_batch(texts, FEAT_DIM_P3)
    X = np.concatenate([X_text, is_reply[:, None], is_quote[:, None]], axis=1)

    split = int(0.9 * n)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    Yb_train, Yb_val = Y_bin[:split], Y_bin[split:]

    model = Phase3HealthStudent(in_dim=FEAT_DIM_P3 + 2, hidden_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=5e-5)
    pos_weight = torch.tensor([6.0, 8.0, 8.0, 10.0], dtype=torch.float32, device=device)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = 2048

    for epoch in range(8):
        model.train()
        perm = np.random.permutation(len(X_train))
        total_loss = 0.0
        steps = 0
        for i in range(0, len(X_train), batch_size):
            idx = perm[i : i + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(Y_train[idx]).to(device)
            ybb = torch.from_numpy(Yb_train[idx]).to(device)
            logits = model.forward_logits(xb)
            pred = torch.sigmoid(logits)
            loss = 0.6 * bce_fn(logits, ybb) + 0.4 * F.smooth_l1_loss(pred, yb, beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            steps += 1
        print(f"  [Phase 3] Epoch {epoch + 1}/8 loss={total_loss / max(steps, 1):.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        val_preds = []
        for i in range(0, len(X_val), batch_size):
            xb = torch.from_numpy(X_val[i : i + batch_size]).to(device)
            val_preds.append(model(xb).cpu().numpy())
        P_val = np.concatenate(val_preds, axis=0)

    mae_per_head = np.mean(np.abs(P_val - Y_val), axis=0)

    def best_f1(scores: np.ndarray, truth_bin: np.ndarray) -> float:
        best = 0.0
        for thr in np.linspace(0.15, 0.85, 29):
            p_bin = scores >= thr
            t_bin = truth_bin >= 0.5
            tp = int(np.sum(p_bin & t_bin))
            fp = int(np.sum(p_bin & ~t_bin))
            fn = int(np.sum(~p_bin & t_bin))
            f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
            if f1 > best:
                best = f1
        return best

    pol_f1 = best_f1(P_val[:, 0], Yb_val[:, 0])
    heat_f1 = best_f1(P_val[:, 1], Yb_val[:, 1])

    # Collapse 2-layer approximation for single-MatMul int8 ONNX export + save full 2-layer JSON weights
    with torch.no_grad():
        w1 = model.fc1.weight.detach().cpu().numpy()
        b1 = model.fc1.bias.detach().cpu().numpy()
        w2 = model.fc2.weight.detach().cpu().numpy()
        b2 = model.fc2.bias.detach().cpu().numpy()
        w_eff = w2 @ w1
        b_eff = w2 @ b1 + b2

    COLAB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = COLAB_OUT_DIR / "phase3_health_student_150k_int8.onnx"
    onnx_meta = export_int8_linear_onnx(w_eff, b_eff, onnx_path, "catbird_phase3_health_student_150k")

    weights_json_path = COLAB_OUT_DIR / "phase3_health_student_150k_weights.json"
    weights_json_path.write_text(
        json.dumps(
            {
                "model": "catbird_phase3_health_student_150k_int8",
                "trained_rows": int(len(X_train)),
                "val_rows": int(len(X_val)),
                "feature_dim": FEAT_DIM_P3 + 2,
                "heads": ["politicalness", "heat", "engagement_bait", "spam"],
                "val_mae": {
                    "politicalness": round(float(mae_per_head[0]), 5),
                    "heat": round(float(mae_per_head[1]), 5),
                    "engagement_bait": round(float(mae_per_head[2]), 5),
                    "spam": round(float(mae_per_head[3]), 5),
                },
                "val_binary_f1": {
                    "politicalness_f1": round(float(pol_f1), 4),
                    "heat_f1": round(float(heat_f1), 4),
                },
                "effective_linear_weights_head0_top32": [round(float(x), 5) for x in w_eff[0, :32]],
                "bias": [round(float(x), 5) for x in b_eff],
            },
            indent=2,
        )
    )

    report = {
        "generated_at": time.strftime("%FT%TZ", time.gmtime()),
        "dataset": str(p3_path),
        "total_rows": n,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "device": str(device),
        "elapsed_s": round(time.time() - t0, 2),
        "val_mae": {
            "politicalness": round(float(mae_per_head[0]), 5),
            "heat": round(float(mae_per_head[1]), 5),
            "engagement_bait": round(float(mae_per_head[2]), 5),
            "spam": round(float(mae_per_head[3]), 5),
        },
        "val_f1": {
            "politicalness_f1": round(float(pol_f1), 4),
            "heat_f1": round(float(heat_f1), 4),
        },
        "onnx_export": onnx_meta,
    }
    (COLAB_OUT_DIR / "phase3_150k_eval_report.json").write_text(json.dumps(report, indent=2))
    print(f"[Phase 3] Done in {report['elapsed_s']}s: val MAE={report['val_mae']} F1={report['val_f1']}", flush=True)
    return {"report": report, "model": model}


def train_phase5_and_build_tier3(device: torch.device, p3_model: Phase3HealthStudent) -> dict:
    t0 = time.time()
    p5_path = BANKS_DIR / "phase5_social_encoder_pairs_500k.parquet"
    table = pq.read_table(p5_path, columns=["uri", "author_did", "community_id"])
    n = table.num_rows
    print(f"[Phase 5] Loaded {n:,} community-labeled posts from {p5_path.name}", flush=True)

    TIER3_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = COLAB_OUT_DIR / "phase5_social_student_matryoshka_int8.onnx"
    part_paths = [TIER3_DIR / f"post_vectors_256d_fp16_part{i:02d}.parquet" for i in range(5)]
    serving_path = TIER3_DIR / "post_vectors_64d_int8_serving.parquet"

    producer_vec_sum: dict[str, np.ndarray] = {}
    producer_vec_cnt: dict[str, int] = {}
    comm_vec_sum: dict[int, np.ndarray] = {}
    comm_vec_cnt: dict[int, int] = {}

    if all(p.exists() for p in part_paths) and serving_path.exists() and onnx_path.exists():
        print("[Phase 5] Reusing existing 500,000 embedded post shards to aggregate producer & community vectors...", flush=True)
        for p in part_paths:
            st = pq.read_table(p, columns=["author_did", "community_id", "vec_256d_fp16"])
            dids = st["author_did"].to_pylist()
            cids = st["community_id"].to_numpy()
            raw_bytes = st["vec_256d_fp16"].to_pylist()
            buf = b"".join(raw_bytes)
            vecs = np.frombuffer(buf, dtype=np.float16).reshape(len(dids), EMB_DIM_FULL).astype(np.float32)
            for j, (did, cid) in enumerate(zip(dids, cids)):
                cid_i = int(cid)
                v = vecs[j]
                if did not in producer_vec_sum:
                    producer_vec_sum[did] = v.copy()
                    producer_vec_cnt[did] = 1
                else:
                    producer_vec_sum[did] += v
                    producer_vec_cnt[did] += 1
                if cid_i not in comm_vec_sum:
                    comm_vec_sum[cid_i] = v.copy()
                    comm_vec_cnt[cid_i] = 1
                else:
                    comm_vec_sum[cid_i] += v
                    comm_vec_cnt[cid_i] += 1
        onnx_meta = {
            "in_dim": FEAT_DIM,
            "out_dim": EMB_DIM_FULL,
            "onnx_bytes": onnx_path.stat().st_size,
        }
    else:
        raise RuntimeError("Expected post_vectors_256d_fp16_part00..04.parquet to exist")

    # Build community_centroids_256d.parquet for all 1,500 communities
    comm_table = pq.read_table(GRAPH_OUT_DIR / "communities_labeled.parquet")
    c_rows = comm_table.to_pylist()

    cent_ids, cent_labels, cent_scenes, cent_domains, cent_posts, cent_fp16, cent_int8_64 = [], [], [], [], [], [], []
    rng = np.random.default_rng(42)
    for row in c_rows:
        cid = int(row["community"])
        cent_ids.append(cid)
        cent_labels.append(str(row.get("label") or f"community_{cid}"))
        cent_scenes.append(int(row.get("scene_id") or 0))
        cent_domains.append(int(row.get("domain_id") or 0))
        cnt = comm_vec_cnt.get(cid, 0)
        cent_posts.append(cnt)
        if cnt > 0:
            v = comm_vec_sum[cid] / cnt
            v = v / max(float(np.linalg.norm(v)), 1e-6)
        else:
            v = rng.standard_normal(EMB_DIM_FULL).astype(np.float32)
            v = v / max(float(np.linalg.norm(v)), 1e-6)
        cent_fp16.append(v.astype(np.float16).tobytes())
        v64 = v[:64]
        v64 = v64 / max(float(np.linalg.norm(v64)), 1e-6)
        s64 = max(float(np.max(np.abs(v64))), 1e-6) / 127.0
        q64 = np.clip(np.round(v64 / s64), -127, 127).astype(np.int8)
        cent_int8_64.append(q64.tobytes())

    comm_vec_table = pa.table(
        {
            "community_id": pa.array(cent_ids, type=pa.uint32()),
            "label": pa.array(cent_labels, type=pa.string()),
            "scene_id": pa.array(cent_scenes, type=pa.uint16()),
            "domain_id": pa.array(cent_domains, type=pa.uint8()),
            "embedded_posts": pa.array(cent_posts, type=pa.uint32()),
            "centroid_256d_fp16": pa.array(cent_fp16, type=pa.binary(512)),
            "centroid_64d_int8": pa.array(cent_int8_64, type=pa.binary(64)),
        }
    )
    pq.write_table(
        comm_vec_table,
        TIER3_DIR / "community_centroids_256d.parquet",
        compression="zstd",
        compression_level=6,
    )
    print(f"  wrote community_centroids_256d.parquet ({len(cent_ids):,} communities)", flush=True)

    # Build producer_vectors_256d.parquet for all 106,744 KnownFor producers
    kf_table = pq.read_table(GRAPH_OUT_DIR / "known_for.parquet")
    kf_rows = kf_table.to_pylist()
    by_prod: dict[str, list[tuple[int, float, int, int]]] = {}
    for r in kf_rows:
        by_prod.setdefault(r["did"], []).append(
            (int(r["community"]), float(r["score"]), int(r.get("scene_id") or 0), int(r.get("domain_id") or 0))
        )

    cent_lookup = {cid: np.frombuffer(fp16_b, dtype=np.float16).astype(np.float32) for cid, fp16_b in zip(cent_ids, cent_fp16)}
    prod_dids, prod_c1, prod_w1, prod_scene, prod_dom, prod_posts, prod_fp16 = [], [], [], [], [], [], []
    for did, memberships in by_prod.items():
        memberships.sort(key=lambda x: -x[1])
        c1, w1, sc, dm = memberships[0]
        prod_dids.append(did)
        prod_c1.append(c1)
        prod_w1.append(w1)
        prod_scene.append(sc)
        prod_dom.append(dm)
        cnt = producer_vec_cnt.get(did, 0)
        prod_posts.append(cnt)
        if cnt > 0:
            v = producer_vec_sum[did] / cnt
        else:
            v = np.zeros(EMB_DIM_FULL, dtype=np.float32)
            for cid, w, _, _ in memberships:
                v += w * cent_lookup.get(cid, np.zeros(EMB_DIM_FULL, dtype=np.float32))
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            v = v / norm
        prod_fp16.append(v.astype(np.float16).tobytes())

    prod_vec_table = pa.table(
        {
            "producer_did": pa.array(prod_dids, type=pa.string()),
            "primary_community_id": pa.array(prod_c1, type=pa.uint32()),
            "primary_weight": pa.array(prod_w1, type=pa.float32()),
            "scene_id": pa.array(prod_scene, type=pa.uint16()),
            "domain_id": pa.array(prod_dom, type=pa.uint8()),
            "embedded_posts": pa.array(prod_posts, type=pa.uint32()),
            "vec_256d_fp16": pa.array(prod_fp16, type=pa.binary(512)),
        }
    )
    pq.write_table(
        prod_vec_table,
        TIER3_DIR / "producer_vectors_256d.parquet",
        compression="zstd",
        compression_level=6,
    )
    print(f"  wrote producer_vectors_256d.parquet ({len(prod_dids):,} producers)", flush=True)

    summary = {
        "generated_at": time.strftime("%FT%TZ", time.gmtime()),
        "posts_embedded": n,
        "producers_embedded": len(prod_dids),
        "communities_embedded": len(cent_ids),
        "matryoshka_dims": list(MATRYOSHKA_DIMS),
        "onnx_export": onnx_meta,
        "tier3_files": [p.name for p in sorted(TIER3_DIR.glob("*.parquet"))],
        "elapsed_s": round(time.time() - t0, 2),
    }
    (COLAB_OUT_DIR / "tier3_lakehouse_manifest.json").write_text(json.dumps(summary, indent=2))
    return summary


def get_access_token() -> str:
    raw = subprocess.check_output(["/usr/local/bin/gws", "auth", "export", "--unmasked"], text=True)
    creds = json.loads(raw[raw.find("{") :])
    data = urllib.parse.urlencode(
        {
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "refresh_token": creds["refresh_token"],
            "grant_type": "refresh_token",
        }
    ).encode()
    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())["access_token"]


def list_drive_folder(token: str, folder_id: str) -> dict[str, dict]:
    q = urllib.parse.quote(f"'{folder_id}' in parents and trashed = false")
    url = f"https://www.googleapis.com/drive/v3/files?q={q}&pageSize=1000&fields=files(id,name,size)"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        files = json.loads(resp.read().decode()).get("files", [])
    return {f["name"]: f for f in files}


def upload_to_drive(token: str, local_path: pathlib.Path, folder_id: str, existing: dict[str, dict]) -> dict:
    name = local_path.name
    size = local_path.stat().st_size
    if name in existing and int(existing[name].get("size") or 0) == size:
        return existing[name]
    meta = json.dumps({"name": name, "parents": [folder_id]}).encode()
    init_req = urllib.request.Request(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&fields=id,name,size",
        data=meta,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "application/octet-stream",
            "X-Upload-Content-Length": str(size),
        },
        method="POST",
    )
    with urllib.request.urlopen(init_req, timeout=60) as resp:
        upload_url = resp.headers["Location"]
    data = local_path.read_bytes()
    put_req = urllib.request.Request(
        upload_url,
        data=data,
        headers={"Content-Length": str(size), "Content-Type": "application/octet-stream"},
        method="PUT",
    )
    with urllib.request.urlopen(put_req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def sync_outputs_to_drive() -> dict:
    token = get_access_token()
    existing_models = list_drive_folder(token, DRIVE_MODELS_FOLDER)
    uploaded_models = 0
    for p in sorted(COLAB_OUT_DIR.glob("*")):
        if p.is_file():
            upload_to_drive(token, p, DRIVE_MODELS_FOLDER, existing_models)
            uploaded_models += 1

    existing_tier3 = list_drive_folder(token, DRIVE_TIER3_FOLDER)
    uploaded_tier3 = 0
    total_tier3_bytes = 0
    for p in sorted(TIER3_DIR.glob("*.parquet")):
        upload_to_drive(token, p, DRIVE_TIER3_FOLDER, existing_tier3)
        uploaded_tier3 += 1
        total_tier3_bytes += p.stat().st_size

    nb_path = ROOT / "colab" / "Catbird_Phase3_Phase5_Colab_GPU_Trainer.ipynb"
    if nb_path.exists():
        existing_lake = list_drive_folder(token, DRIVE_LAKE_FOLDER)
        upload_to_drive(token, nb_path, DRIVE_LAKE_FOLDER, existing_lake)

    return {
        "uploaded_models": uploaded_models,
        "uploaded_tier3_shards": uploaded_tier3,
        "total_tier3_mb": round(total_tier3_bytes / (1024 * 1024), 2),
    }


def main() -> int:
    device = get_device()
    print(f"Using device: {device}", flush=True)
    p3_res = train_phase3_150k(device)
    p5_res = train_phase5_and_build_tier3(device, p3_res["model"])
    drive_res = sync_outputs_to_drive()
    print(json.dumps({"phase3": p3_res["report"], "phase5_tier3": p5_res, "drive_sync": drive_res}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
