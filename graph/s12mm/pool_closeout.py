#!/usr/bin/env python3
"""S12m close-out (whole-network plan row S12 & S16a candidate prep):

1. Streams the 64 DID-bucketed S12m output parquets (`u-b0000-*.parquet` .. `u-b0063-*.parquet`),
   filtering out `modality == 4` (`NONE`) and any purged producer/DID:
   - Tallies exact modality counts, `image_skip` counts, total embedded images, and finite/norm checks.
   - Pools producer post vectors (`mm` from `vec`, `text` from `vec_text`/text-only) for both:
       * `all` (`T_now`, all slab posts)
       * `preT` (`tid_time(rkey) < 2026-08-16T00:00:00Z`, S1 R1)
   - Fuses pooled post vectors with purged S4 `eg2_bios` (`normalise(post_mean + 0.25 * bio_vec)`),
     emitting both fp16 and int8+scale producer tables (`plan §5.1`).
2. Selects up to 3 medoid post URIs per producer (`medoids.parquet`, `all` window), strictly excluding
   purged DIDs, `hide_from_recs_union.tsv` (Amendment A1), `gate_bundle.excluded_dids`, `gate_bundle.labelled_uris`,
   and any post with `image_skip` in `('post_self_label', 'record_label', 'account_excluded')`.
3. Verifies pre-`T` reproducibility by running an independent `created < T` (`tid_time(rkey) < T`) pass
   on 3 buckets and asserting bit-for-bit equality with `pooled_preT`.
4. Repeats the Mac MPS parity check on a deterministic salted sample of 1,000 text rows (`modality == 0`)
   plus 200 multimodal `vec_text` rows against `~/catbird-lake-local/models/embeddinggemma-2` (`bfloat16`),
   verifying `median cosine >= 0.99`.
5. Emits up to 32 deterministic random candidates per producer drawn uniformly from that producer's exact
   top-1% slab-cosine neighbours (`s16a_candidates_all.parquet` and `s16a_candidates_preT.parquet`), ready
   to filter out `O4` teacher top-200 and cap at `<= 20` per producer for `S16a`.
6. Writes the formal `S12` close-out receipt (`receipt.json`).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "s4_pilot"))
import mmlib as L  # noqa: E402

T_CUTOFF = dt.datetime(2026, 8, 16, 0, 0, 0, tzinfo=dt.timezone.utc)
BIO_W = 0.25
BUCKET_RE = re.compile(r"^u-(b\d{4})-(\d{4})-\d+\.parquet$")
BAD_MEDOID_SKIPS = frozenset({"post_self_label", "record_label", "account_excluded", "unsafe_did_or_cid", "bad_tid"})


def normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def quantise_int8(v_fp32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantise unit vectors `(N, 256)` to `int8` with per-vector `fp16` scale (`plan §5.1`)."""
    mx = np.maximum(np.max(np.abs(v_fp32), axis=1), 1e-8)
    scale = (mx / 127.0).astype(np.float16)
    i8 = np.clip(np.round(v_fp32 / scale[:, None].astype(np.float32)), -127, 127).astype(np.int8)
    return i8, scale


def load_producers(producers_tsv: pathlib.Path) -> tuple[dict[int, str], dict[str, int]]:
    pid_to_did: dict[int, str] = {}
    did_to_pid: dict[str, int] = {}
    with open(producers_tsv) as fh:
        for line in fh:
            ps, did, _ = line.rstrip("\n").split("\t", 2)
            pid = int(ps)
            pid_to_did[pid] = did
            did_to_pid[did] = pid
    return pid_to_did, did_to_pid


def load_bios(bios_dir: pathlib.Path, purge_pids: set[int]) -> dict[int, np.ndarray]:
    ids = np.load(bios_dir / "ids.npy")
    vecs = np.load(bios_dir / "vec256_fp16.npy")
    keep = np.array([int(p) not in purge_pids for p in ids], dtype=bool)
    ids = ids[keep]
    vecs_f32 = normalise(vecs[keep].astype(np.float32))
    return {int(p): vecs_f32[i] for i, p in enumerate(ids)}


def fuse_bios(pids: np.ndarray, post_vecs: np.ndarray, bio_map: dict[int, np.ndarray]) -> tuple[np.ndarray, int]:
    out = post_vecs.copy()
    has_idx = [i for i, p in enumerate(pids) if int(p) in bio_map]
    if has_idx:
        bmat = np.stack([bio_map[int(pids[i])] for i in has_idx], axis=0)
        out[has_idx] = out[has_idx] + BIO_W * bmat
        out = normalise(out)
    return out, len(has_idx)


def pool_group(pids_arr: np.ndarray, vecs_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(unique_pids, unit_mean_fp32, counts_i16)`."""
    if len(pids_arr) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((0, L.DIM), dtype=np.float32), np.zeros(0, dtype=np.int16)
    o = np.argsort(pids_arr, kind="stable")
    sp = pids_arr[o]
    sv = vecs_f32[o]
    u, start, counts = np.unique(sp, return_index=True, return_counts=True)
    sums = np.add.reduceat(sv, start, axis=0)
    return u.astype(np.int32), normalise(sums), np.minimum(counts, 32767).astype(np.int16)


def run_mps_parity(
    parity_text_samples: list[tuple[int, str, str, bytes]],
    parity_mm_text_samples: list[tuple[int, str, str, bytes]],
    units_dir: pathlib.Path,
) -> dict:
    """Re-encode sampled rows on Mac MPS in bf16 and verify median cosine >= 0.99 vs Colab."""
    import embed as s4_embed

    # Verify model SHA256SUMS if present
    model_dir = s4_embed.SPECS["eg2"]["path"]
    sha_sums = model_dir / "SHA256SUMS"
    sha_verified = False
    if sha_sums.exists():
        for line in open(sha_sums):
            if not line.strip():
                continue
            want, rel = line.strip().split(None, 1)
            rel = rel.lstrip("*./")
            got = L.sha256_file(model_dir / rel)
            if got != want:
                raise SystemExit(f"EG2 model file sha256 mismatch on {rel}: {got} != {want}")
        sha_verified = True

    # Lookup input texts from s12m-units-s9/units/<unit>.parquet grouped by unit
    wanted_by_unit: dict[str, set[str]] = collections.defaultdict(set)
    for _, unit_name, uri, _ in parity_text_samples + parity_mm_text_samples:
        wanted_by_unit[unit_name].add(uri)

    uri_to_text: dict[str, str] = {}
    for unit_name, uris_set in sorted(wanted_by_unit.items()):
        upath = units_dir / "units" / f"{unit_name}.parquet"
        tb = pq.read_table(upath, columns=["uri", "text"])
        for u, txt in zip(tb["uri"].to_pylist(), tb["text"].to_pylist()):
            if u in uris_set:
                uri_to_text[u] = txt or ""

    m, prompt = s4_embed.load("eg2")

    def eval_group(samples: list[tuple[int, str, str, bytes]]) -> dict:
        valid = [(u, uri_to_text[u], blob) for _, _, u, blob in samples if u in uri_to_text]
        texts = [t for _, t, _ in valid]
        stored = L.unpack([b for _, _, b in valid]).astype(np.float32)
        stored = normalise(stored)
        emb = m.encode(
            texts,
            prompt=prompt,
            batch_size=64,
            convert_to_tensor=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        if emb.device.type == "mps":
            torch.mps.synchronize()
        full = emb.float().cpu().numpy()
        nan_rows = int(np.isnan(full).any(axis=1).sum())
        mac_v = L.to_stored(full).astype(np.float32)
        mac_v = normalise(mac_v)
        cos = np.sum(stored * mac_v, axis=1)
        return {
            "n": len(valid),
            "nan_rows": nan_rows,
            "cosine_min": float(cos.min()),
            "cosine_p01": float(np.quantile(cos, 0.01)),
            "cosine_p05": float(np.quantile(cos, 0.05)),
            "cosine_median": float(np.median(cos)),
            "cosine_mean": float(cos.mean()),
            "pass_ge_099": bool(np.median(cos) >= 0.99 and nan_rows == 0),
        }

    t_res = eval_group(parity_text_samples)
    mt_res = eval_group(parity_mm_text_samples)
    del m
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if not t_res["pass_ge_099"]:
        raise SystemExit(f"Mac MPS parity check failed 0.99 bar: {t_res}")
    return {
        "model_sha256sums_verified": sha_verified,
        "text_modality0_1000": t_res,
        "multimodal_vec_text_200": mt_res,
    }


def load_o4_excl(o4_parquet: pathlib.Path) -> dict[int, set[int]]:
    """Load O4 teacher top-200 sets `{anchor_pid: set(q_pid)}` (`~200k` anchors)."""
    tb = pq.read_table(o4_parquet, columns=["p", "q"])
    p_arr = tb["p"].to_numpy(zero_copy_only=False)
    q_arr = tb["q"].to_numpy(zero_copy_only=False)
    if len(p_arr) == 0:
        return {}
    o = np.argsort(p_arr, kind="stable")
    sp, sq = p_arr[o], q_arr[o]
    u, start, counts = np.unique(sp, return_index=True, return_counts=True)
    return {int(p): set(map(int, sq[s : s + c])) for p, s, c in zip(u, start, counts)}


def sample_top1pct_candidates(
    pids: np.ndarray,
    vecs_fp16: np.ndarray,
    o4_parquet: pathlib.Path,
    out_parquet: pathlib.Path,
    flat_pairs_parquet: pathlib.Path,
    *,
    n_draw: int = 20,
    chunk: int = 512,
    seed: int = 20261030,
) -> dict:
    """For each O4 teacher anchor producer present in `pids`, draw up to `n_draw` uniform random
    neighbours from its exact top-1% cosine neighbours across ALL `N` slab producers (excluding self
    and `O4` top-200), using MPS fp16 matmul (`B=512` so `B*N < 2^31` on MPS) + GPU fp16 threshold
    bisection + uniformly permuted target columns (`float32` CPU arrays only).
    """
    t0 = time.time()
    o4_excl = load_o4_excl(o4_parquet)
    n = len(pids)
    k_1pct = max(1, n // 100)
    anchor_rows = np.array([i for i, p in enumerate(pids) if int(p) in o4_excl], dtype=np.int64)
    n_anchors = len(anchor_rows)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X = torch.from_numpy(np.ascontiguousarray(vecs_fp16, dtype=np.float16)).to(dev)
    X = X / torch.clamp(X.norm(dim=1, keepdim=True), min=1e-6)

    perm = np.random.default_rng(seed).permutation(n)
    inv_perm = np.empty(n, dtype=np.int64)
    inv_perm[perm] = np.arange(n, dtype=np.int64)
    X_perm = X[torch.from_numpy(perm).to(dev)].contiguous()
    pids_perm = pids[perm].astype(np.int32)

    prefix_w = min(32768, n)
    probe_w = min(8192, n)
    lo_rank = max(0, probe_w - max(4, int(probe_w * 0.022)))
    hi_rank = min(probe_w - 1, probe_w - max(1, int(probe_w * 0.003)))

    out_p: list[int] = []
    out_q: list[list[int]] = []
    out_cos: list[list[float]] = []
    flat_p: list[int] = []
    flat_q: list[int] = []
    fallback_rows = 0

    n_chunks = (n_anchors + chunk - 1) // max(chunk, 1)
    for ci, s in enumerate(range(0, n_anchors, chunk)):
        e = min(s + chunk, n_anchors)
        arows = anchor_rows[s:e]
        b_len = len(arows)
        Xa = X[torch.from_numpy(arows).to(dev)]
        sims = Xa @ X_perm.T  # (B, N) fp16; B=512 -> <1e9 elements (< 2^31 MPS reduction limit)
        row_arange = torch.arange(b_len, device=dev)
        self_cols = torch.from_numpy(inv_perm[arows]).to(dev)
        sims[row_arange, self_cols] = -2.0

        probe_sorted, _ = torch.sort(sims[:, :probe_w].float(), dim=1)
        lo = probe_sorted[:, lo_rank].clone()
        hi = probe_sorted[:, hi_rank].clone()
        lo_ok = (sims >= lo.to(torch.float16)[:, None]).sum(dim=1) >= k_1pct
        hi_ok = (sims >= hi.to(torch.float16)[:, None]).sum(dim=1) < k_1pct
        lo = torch.where(lo_ok, lo, torch.full_like(lo, -1.0))
        hi = torch.where(hi_ok, hi, torch.full_like(hi, 1.002))

        for _ in range(10):
            mid_h = ((lo + hi) * 0.5).to(torch.float16)
            cnt = (sims >= mid_h[:, None]).sum(dim=1)
            ge = cnt >= k_1pct
            lo = torch.where(ge, mid_h.float(), lo)
            hi = torch.where(ge, hi, mid_h.float())

        thr_h = lo.to(torch.float16)
        cnt_gt = (sims > thr_h[:, None]).sum(dim=1)
        cnt_eq = (sims == thr_h[:, None]).sum(dim=1)

        sub_f32 = sims[:, :prefix_w].float().cpu().numpy()
        thr_f32 = thr_h.float().cpu().numpy()
        gt_cpu = cnt_gt.cpu().numpy()
        eq_cpu = cnt_eq.cpu().numpy()

        for bi in range(b_len):
            p_id = int(pids[arows[bi]])
            excl = o4_excl[p_id]
            t_val = float(thr_f32[bi])
            need_ties = max(0, k_1pct - int(gt_cpu[bi]))
            p_tie = min(1.0, need_ties / max(1, int(eq_cpu[bi])))
            row_rng = np.random.default_rng(seed + p_id)

            r_sub = sub_f32[bi]
            gt_idx = np.flatnonzero(r_sub > t_val)
            eq_idx = np.flatnonzero(r_sub == t_val)
            if len(eq_idx) > 0 and p_tie > 0.0:
                keep_eq = eq_idx[row_rng.random(len(eq_idx)) < p_tie]
                cand_j = np.sort(np.concatenate([gt_idx, keep_eq]))
            else:
                cand_j = gt_idx

            chosen_q: list[int] = []
            chosen_c: list[float] = []
            for j in cand_j:
                q_id = int(pids_perm[j])
                if q_id != p_id and q_id not in excl:
                    chosen_q.append(q_id)
                    chosen_c.append(round(float(r_sub[j]), 5))
                    if len(chosen_q) == n_draw:
                        break

            max_possible = max(0, k_1pct - len(excl))
            if len(chosen_q) < min(n_draw, max_possible) and prefix_w < n:
                fallback_rows += 1
                full_r = sims[bi].float().cpu().numpy()
                gt_all = np.flatnonzero(full_r > t_val)
                eq_all = np.flatnonzero(full_r == t_val)
                if len(eq_all) > 0 and p_tie > 0.0:
                    keep_eq_all = eq_all[row_rng.random(len(eq_all)) < p_tie]
                    all_j = np.sort(np.concatenate([gt_all, keep_eq_all]))
                else:
                    all_j = gt_all
                chosen_q, chosen_c = [], []
                for j in all_j:
                    q_id = int(pids_perm[j])
                    if q_id != p_id and q_id not in excl:
                        chosen_q.append(q_id)
                        chosen_c.append(round(float(full_r[j]), 5))
                        if len(chosen_q) == n_draw:
                            break

            # Sort chosen pairs by descending cosine for clean deterministic inspection
            if len(chosen_q) > 1:
                ord_c = np.argsort(-np.array(chosen_c, dtype=np.float32), kind="stable")
                chosen_q = [chosen_q[k] for k in ord_c]
                chosen_c = [chosen_c[k] for k in ord_c]

            out_p.append(p_id)
            out_q.append(chosen_q)
            out_cos.append(chosen_c)
            for q_id in chosen_q:
                flat_p.append(p_id)
                flat_q.append(q_id)

        if (ci + 1) % 50 == 0 or (ci + 1) == n_chunks:
            print(
                f"[s16a_candidates:{out_parquet.stem}] chunk {ci+1}/{n_chunks} "
                f"({e}/{n_anchors} anchors) elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    tb = pa.table({
        "p": pa.array(out_p, type=pa.int32()),
        "q_candidates": pa.array(out_q, type=pa.list_(pa.int32())),
        "cosines": pa.array(out_cos, type=pa.list_(pa.float32())),
    })
    tmp = out_parquet.with_name(out_parquet.name + ".tmp")
    pq.write_table(tb, tmp, compression="zstd")
    os.replace(tmp, out_parquet)

    fp_np = np.array(flat_p, dtype=np.uint32)
    fq_np = np.array(flat_q, dtype=np.uint32)
    if len(fp_np) > 0:
        fo = np.lexsort((fq_np, fp_np))
        fp_np, fq_np = fp_np[fo], fq_np[fo]
    ftb = pa.table({
        "p": pa.array(fp_np, type=pa.uint32()),
        "q": pa.array(fq_np, type=pa.uint32()),
    })
    ftmp = flat_pairs_parquet.with_name(flat_pairs_parquet.name + ".tmp")
    pq.write_table(ftb, ftmp, compression="zstd")
    os.replace(ftmp, flat_pairs_parquet)

    del X, X_perm
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {
        "slab_producers": n,
        "o4_anchors_total": len(o4_excl),
        "o4_anchors_with_slab": n_anchors,
        "top_1pct_k": k_1pct,
        "drawn_per_producer_cap": n_draw,
        "flat_pair_rows": int(len(fp_np)),
        "fallback_rows": fallback_rows,
        "seconds": round(time.time() - t0, 1),
        "file": str(out_parquet),
        "bytes": out_parquet.stat().st_size,
        "sha256": L.sha256_file(out_parquet),
        "flat_file": str(flat_pairs_parquet),
        "flat_bytes": flat_pairs_parquet.stat().st_size,
        "flat_sha256": L.sha256_file(flat_pairs_parquet),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vectors-dir", default="~/catbird-lake-local/s12mm-colab-s12m-20261029a/vectors/units")
    ap.add_argument("--units-dir", default="~/catbird-lake-local/s12m-units-s9")
    ap.add_argument("--gate-dir", default="~/catbird-lake-local/s12mm-label-chain-s12m-20261029a/gate")
    ap.add_argument("--bios-dir", default="~/catbird-lake-local/s4-pilot-20261028/vectors/eg2_bios")
    ap.add_argument("--producers-tsv", default="~/catbird-lake-local/graph-build-v3/out-full-20261024/producers.tsv")
    ap.add_argument("--purge-dids", default="~/catbird-lake-local/s5-sweep-20261030/purge_dids.txt")
    ap.add_argument("--hide-dids", default="~/catbird-lake-local/s5-declarations-2027/hide_from_recs_union.tsv")
    ap.add_argument("--o4-dir", default="~/catbird-lake-local/s10-outputs-20261030/O4")
    ap.add_argument("--out-dir", default="~/catbird-lake-local/s12-closeout-20261030")
    ap.add_argument("--skip-s16a-pairs", action="store_true")
    args = ap.parse_args(argv)

    t_start = time.time()
    vdir = pathlib.Path(args.vectors_dir).expanduser().resolve()
    udir = pathlib.Path(args.units_dir).expanduser().resolve()
    gdir = pathlib.Path(args.gate_dir).expanduser().resolve()
    bdir = pathlib.Path(args.bios_dir).expanduser().resolve()
    ptsv = pathlib.Path(args.producers_tsv).expanduser().resolve()
    purge_path = pathlib.Path(args.purge_dids).expanduser().resolve()
    hide_path = pathlib.Path(args.hide_dids).expanduser()
    if not hide_path.exists():
        hide_path = pathlib.Path("~/catbird-lake-local/s5-declarations-20261027/hide_from_recs_union.tsv").expanduser()
    hide_path = hide_path.resolve()
    o4_dir = pathlib.Path(args.o4_dir).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pid_to_did, _ = load_producers(ptsv)
    purge_dids = {ln.strip() for ln in open(purge_path) if ln.startswith("did:")}
    purge_pids = {pid for pid, did in pid_to_did.items() if did in purge_dids}
    hide_dids = {ln.strip() for ln in open(hide_path) if ln.startswith("did:")}
    gate_bundle = L.load_bundle(gdir)
    gate_man = json.loads((gdir / "gate_manifest.json").read_text())
    gate_reasons = json.loads((gdir / "gate_reasons.json").read_text()) if (gdir / "gate_reasons.json").exists() else {}
    bio_map = load_bios(bdir, purge_pids)

    files = sorted(f for f in vdir.glob("u-*.parquet") if not f.name.startswith("._"))
    by_bucket: dict[str, list[tuple[str, pathlib.Path]]] = collections.defaultdict(list)
    for f in files:
        m = BUCKET_RE.match(f.name)
        if not m:
            raise SystemExit(f"unexpected vector filename: {f.name}")
        unit_name = f"{m.group(1)}-{m.group(2)}"
        by_bucket[m.group(1)].append((unit_name, f))

    mod_counts = collections.Counter()
    skip_counts = collections.Counter()
    total_rows = 0
    total_images_embedded = 0
    bad_vectors = 0
    purged_rows_seen = 0

    all_mm_pids, all_mm_vecs, all_mm_cnts = [], [], []
    pre_mm_pids, pre_mm_vecs, pre_mm_cnts = [], [], []
    all_tx_pids, all_tx_vecs, all_tx_cnts = [], [], []
    pre_tx_pids, pre_tx_vecs, pre_tx_cnts = [], [], []

    medoid_pids: list[int] = []
    medoid_dids: list[str] = []
    medoid_uris: list[list[str]] = []
    medoid_cos: list[list[float]] = []

    # Reservoir / min-heap by salted blake2b hash for MPS parity check
    import heapq
    parity_text_heap: list[tuple[int, str, str, bytes]] = []       # keep 1000 smallest hashes -> store (-hash, unit, uri, blob)
    parity_mm_text_heap: list[tuple[int, str, str, bytes]] = []    # keep 200 smallest hashes

    def push_sample(heap: list, cap: int, salt: str, unit_name: str, uri: str, blob: bytes) -> None:
        h = int.from_bytes(hashlib.blake2b(f"{salt}|{uri}".encode(), digest_size=8).digest(), "big")
        item = (-h, unit_name, uri, blob)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    for b_idx, b_key in enumerate(sorted(by_bucket)):
        b_uris: list[str] = []
        b_pids: list[int] = []
        b_mods: list[int] = []
        b_skips: list[str | None] = []
        b_vecs: list[bytes | None] = []
        b_vtxs: list[bytes | None] = []

        for unit_name, f in by_bucket[b_key]:
            tb = pq.read_table(f)
            d = tb.to_pydict()
            total_rows += len(d["uri"])
            for uri, p, mod, n_im, sk, v, vt in zip(
                d["uri"], d["p"], d["modality"], d["n_images"], d["image_skip"], d["vec"], d["vec_text"]
            ):
                did, _ = L.did_rkey(uri)
                if p in purge_pids or did in purge_dids:
                    purged_rows_seen += 1
                    continue
                mod_counts[int(mod)] += 1
                if sk is not None:
                    skip_counts[sk] += 1
                total_images_embedded += int(n_im or 0)
                b_uris.append(uri)
                b_pids.append(int(p))
                b_mods.append(int(mod))
                b_skips.append(sk)
                b_vecs.append(v)
                b_vtxs.append(vt)

                if mod == L.TEXT and v is not None:
                    push_sample(parity_text_heap, 1000, "s12-closeout-parity-1000", unit_name, uri, v)
                elif mod == L.TEXT_IMAGE and vt is not None:
                    push_sample(parity_mm_text_heap, 200, "s12-closeout-parity-mmtext-200", unit_name, uri, vt)

        if purged_rows_seen > 0:
            raise SystemExit(f"Purged rows found in vectors during close-out ({purged_rows_seen}); run purge_s12m.py first!")

        # Valid primary multimodal rows (`modality != NONE` and `vec is not None`)
        mm_idx = [i for i, (m, v) in enumerate(zip(b_mods, b_vecs)) if m != L.NONE and v is not None]
        if mm_idx:
            mm_mat = L.unpack([b_vecs[i] for i in mm_idx]).astype(np.float32)
            norms = np.linalg.norm(mm_mat, axis=1)
            bad_vectors += int((~np.isfinite(mm_mat).all(axis=1) | (np.abs(norms - 1.0) > 0.01)).sum())
            mm_pids_np = np.array([b_pids[i] for i in mm_idx], dtype=np.int32)
            u_all, v_all, c_all = pool_group(mm_pids_np, mm_mat)
            all_mm_pids.append(u_all)
            all_mm_vecs.append(v_all)
            all_mm_cnts.append(c_all)

            # Pre-T subset (`tid_time(rkey) < T_CUTOFF`)
            pre_mask = np.array(
                [(t := L.tid_time(L.did_rkey(b_uris[i])[1])) is not None and t < T_CUTOFF for i in mm_idx],
                dtype=bool,
            )
            if pre_mask.any():
                u_pre, v_pre, c_pre = pool_group(mm_pids_np[pre_mask], mm_mat[pre_mask])
                pre_mm_pids.append(u_pre)
                pre_mm_vecs.append(v_pre)
                pre_mm_cnts.append(c_pre)

            # Medoids per producer (up to 3 posts closest to `v_all`, strictly gated!)
            cent_map = {int(pid): v_all[k] for k, pid in enumerate(u_all)}
            by_prod_medoid: dict[int, list[tuple[float, str]]] = collections.defaultdict(list)
            for local_k, orig_i in enumerate(mm_idx):
                pid = int(mm_pids_np[local_k])
                uri = b_uris[orig_i]
                did, _ = L.did_rkey(uri)
                sk = b_skips[orig_i]
                if (
                    did in hide_dids
                    or did in gate_bundle.excluded_dids
                    or uri in gate_bundle.labelled_uris
                    or (sk is not None and sk in BAD_MEDOID_SKIPS)
                ):
                    continue
                cos = float(np.dot(mm_mat[local_k], cent_map[pid]))
                by_prod_medoid[pid].append((-cos, uri))

            for pid in sorted(by_prod_medoid):
                cand = sorted(by_prod_medoid[pid])[:3]
                medoid_pids.append(pid)
                medoid_dids.append(pid_to_did[pid])
                medoid_uris.append([u for _, u in cand])
                medoid_cos.append([round(-nc, 5) for nc, _ in cand])

        # Valid text-only rows (`vec_text` if present, else `vec` for TEXT / TEXT_FALLBACK)
        tx_blobs = []
        tx_pids_list = []
        tx_uris_list = []
        for uri, p, m, v, vt in zip(b_uris, b_pids, b_mods, b_vecs, b_vtxs):
            if m == L.NONE:
                continue
            blob = vt if vt is not None else (v if m in (L.TEXT, L.TEXT_FALLBACK) else None)
            if blob is not None:
                tx_blobs.append(blob)
                tx_pids_list.append(p)
                tx_uris_list.append(uri)
        if tx_blobs:
            tx_mat = L.unpack(tx_blobs).astype(np.float32)
            norms_t = np.linalg.norm(tx_mat, axis=1)
            bad_vectors += int((~np.isfinite(tx_mat).all(axis=1) | (np.abs(norms_t - 1.0) > 0.01)).sum())
            tx_pids_np = np.array(tx_pids_list, dtype=np.int32)
            ut_all, vt_all, ct_all = pool_group(tx_pids_np, tx_mat)
            all_tx_pids.append(ut_all)
            all_tx_vecs.append(vt_all)
            all_tx_cnts.append(ct_all)

            pre_t_mask = np.array(
                [(t := L.tid_time(L.did_rkey(u)[1])) is not None and t < T_CUTOFF for u in tx_uris_list],
                dtype=bool,
            )
            if pre_t_mask.any():
                ut_pre, vt_pre, ct_pre = pool_group(tx_pids_np[pre_t_mask], tx_mat[pre_t_mask])
                pre_tx_pids.append(ut_pre)
                pre_tx_vecs.append(vt_pre)
                pre_tx_cnts.append(ct_pre)

    if bad_vectors != 0:
        raise SystemExit(f"Found {bad_vectors} non-finite or non-unit vectors!")

    def assemble(pids_l, vecs_l, cnts_l):
        pids = np.concatenate(pids_l)
        vecs = np.concatenate(vecs_l, axis=0)
        cnts = np.concatenate(cnts_l)
        o = np.argsort(pids, kind="stable")
        pids, vecs, cnts = pids[o], vecs[o], cnts[o]
        assert len(np.unique(pids)) == len(pids), "duplicate pid across buckets!"
        fused, n_bio = fuse_bios(pids, vecs, bio_map)
        return pids, vecs, fused, cnts, n_bio

    mm_all_p, mm_all_v, mm_all_f, mm_all_c, mm_all_nbio = assemble(all_mm_pids, all_mm_vecs, all_mm_cnts)
    mm_pre_p, mm_pre_v, mm_pre_f, mm_pre_c, mm_pre_nbio = assemble(pre_mm_pids, pre_mm_vecs, pre_mm_cnts)
    tx_all_p, tx_all_v, tx_all_f, tx_all_c, tx_all_nbio = assemble(all_tx_pids, all_tx_vecs, all_tx_cnts)
    tx_pre_p, tx_pre_v, tx_pre_f, tx_pre_c, tx_pre_nbio = assemble(pre_tx_pids, pre_tx_vecs, pre_tx_cnts)

    # Save fp16 & int8 pooled arrays
    np.savez(
        out_dir / "pooled_all_fp16.npz",
        mm_pids=mm_all_p, mm_posts=mm_all_v.astype(np.float16), mm_fused=mm_all_f.astype(np.float16), mm_counts=mm_all_c,
        text_pids=tx_all_p, text_posts=tx_all_v.astype(np.float16), text_fused=tx_all_f.astype(np.float16), text_counts=tx_all_c,
    )
    np.savez(
        out_dir / "pooled_preT_fp16.npz",
        mm_pids=mm_pre_p, mm_posts=mm_pre_v.astype(np.float16), mm_fused=mm_pre_f.astype(np.float16), mm_counts=mm_pre_c,
        text_pids=tx_pre_p, text_posts=tx_pre_v.astype(np.float16), text_fused=tx_pre_f.astype(np.float16), text_counts=tx_pre_c,
    )

    all_i8, all_sc = quantise_int8(mm_all_f)
    pre_i8, pre_sc = quantise_int8(mm_pre_f)
    np.savez(out_dir / "pooled_all_int8.npz", pids=mm_all_p, vec_i8=all_i8, scale=all_sc, counts=mm_all_c)
    np.savez(out_dir / "pooled_preT_int8.npz", pids=mm_pre_p, vec_i8=pre_i8, scale=pre_sc, counts=mm_pre_c)

    # Check int8 quantisation fidelity on 10k rows
    q_chk_n = min(10_000, len(mm_all_f))
    q_rec = normalise(all_i8[:q_chk_n].astype(np.float32) * all_sc[:q_chk_n, None].astype(np.float32))
    i8_cos_median = float(np.median(np.sum(mm_all_f[:q_chk_n] * q_rec, axis=1)))

    # Write medoids.parquet and audit 0 safety violations
    med_order = np.argsort(np.array(medoid_pids, dtype=np.int32), kind="stable")
    med_tb = pa.table({
        "pid": pa.array([medoid_pids[i] for i in med_order], type=pa.int32()),
        "did": pa.array([medoid_dids[i] for i in med_order], type=pa.string()),
        "uris": pa.array([medoid_uris[i] for i in med_order], type=pa.list_(pa.string())),
        "cosines": pa.array([medoid_cos[i] for i in med_order], type=pa.list_(pa.float32())),
    })
    pq.write_table(med_tb, out_dir / "medoids.parquet", compression="zstd")
    med_violations = sum(
        1 for d, us in zip(med_tb["did"].to_pylist(), med_tb["uris"].to_pylist())
        if d in purge_dids or d in hide_dids or d in gate_bundle.excluded_dids or any(u in gate_bundle.labelled_uris for u in us)
    )
    assert med_violations == 0, f"Medoid safety audit failed: {med_violations} violations!"

    # Independent pre-T reproducibility check on 3 buckets (`b0000`, `b0031`, `b0063`)
    repro_buckets = [k for k in ("b0000", "b0031", "b0063") if k in by_bucket]
    pre_idx_map = {int(p): i for i, p in enumerate(mm_pre_p)}
    repro_checked_producers = 0
    repro_max_abs_diff = 0.0
    for bk in repro_buckets:
        r_pids, r_blobs = [], []
        for _, f in by_bucket[bk]:
            tb = pq.read_table(f, columns=["uri", "p", "modality", "vec"])
            for uri, p, m, v in zip(tb["uri"].to_pylist(), tb["p"].to_pylist(), tb["modality"].to_pylist(), tb["vec"].to_pylist()):
                if m == L.NONE or v is None:
                    continue
                t = L.tid_time(L.did_rkey(uri)[1])
                if t is not None and t < T_CUTOFF:
                    r_pids.append(int(p))
                    r_blobs.append(v)
        if r_blobs:
            ru, rv, _ = pool_group(np.array(r_pids, dtype=np.int32), L.unpack(r_blobs).astype(np.float32))
            stored_sub = mm_pre_v[[pre_idx_map[int(p)] for p in ru]]
            diff = float(np.max(np.abs(rv.astype(np.float16).astype(np.float32) - stored_sub.astype(np.float16).astype(np.float32))))
            repro_max_abs_diff = max(repro_max_abs_diff, diff)
            repro_checked_producers += len(ru)
    assert repro_max_abs_diff == 0.0, f"pre-T reproducibility check failed: max_abs_diff={repro_max_abs_diff}"

    # Mac MPS parity check
    parity_text_sorted = sorted(parity_text_heap, key=lambda x: -x[0])
    parity_mm_sorted = sorted(parity_mm_text_heap, key=lambda x: -x[0])
    parity_rep = run_mps_parity(parity_text_sorted, parity_mm_sorted, udir)

    # S16a top-1% candidate sampling (`all` and `preT` on `mm_fused` — stock EG2 pooled producer vectors,
    # evaluated across ALL ~1.9M slab producers for every O4 noqn anchor producer, cap <=20 per anchor)
    s16a_rep = {}
    if not args.skip_s16a_pairs:
        s16a_rep["all"] = sample_top1pct_candidates(
            mm_all_p,
            mm_all_f.astype(np.float16),
            o4_dir / "prod_noqn.parquet",
            out_dir / "s16a_candidates_all.parquet",
            out_dir / "cand_pairs_prod.parquet",
            n_draw=20,
        )
        s16a_rep["preT"] = sample_top1pct_candidates(
            mm_pre_p,
            mm_pre_f.astype(np.float16),
            o4_dir / "preT_noqn.parquet",
            out_dir / "s16a_candidates_preT.parquet",
            out_dir / "cand_pairs_preT.parquet",
            n_draw=20,
        )

    out_files = [
        "pooled_all_fp16.npz", "pooled_preT_fp16.npz",
        "pooled_all_int8.npz", "pooled_preT_int8.npz",
        "medoids.parquet",
    ]
    if not args.skip_s16a_pairs:
        out_files += [
            "s16a_candidates_all.parquet",
            "s16a_candidates_preT.parquet",
            "cand_pairs_prod.parquet",
            "cand_pairs_preT.parquet",
        ]

    receipt = {
        "step": "S12m close-out: pooling, medoids, MPS parity, pre-T check, and S16a candidate pairs",
        "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md §3.1 S12, S16a, §5.1",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md (S1, Amendments A1 & A2)",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t_start, 1),
        "inputs": {
            "units_manifest": {"path": str(udir / "manifest.json"), "sha256": L.sha256_file(udir / "manifest.json")},
            "gate_manifest": {"path": str(gdir / "gate_manifest.json"), "sha256": L.sha256_file(gdir / "gate_manifest.json"), "watermark_utc": gate_man["label_watermark_utc"]},
            "purge_dids": {"path": str(purge_path), "sha256": L.sha256_file(purge_path), "count": len(purge_dids)},
            "hide_from_recs_union": {"path": str(hide_path), "sha256": L.sha256_file(hide_path), "count": len(hide_dids)},
            "producers_tsv": {"path": str(ptsv), "sha256": L.sha256_file(ptsv)},
            "eg2_bios_vec": {"path": str(bdir / "vec256_fp16.npy"), "sha256": L.sha256_file(bdir / "vec256_fp16.npy"), "active_bios": len(bio_map)},
        },
        "gate_did_reason_counts": gate_reasons.get("did_sets", {}),
        "slab_totals": {
            "vector_unit_files": len(files),
            "rows_after_purge": total_rows,
            "images_embedded": total_images_embedded,
            "bad_or_nan_vectors": bad_vectors,
            "modality_counts": {L.MODALITY_NAMES.get(k, str(k)): v for k, v in sorted(mod_counts.items())},
            "image_skip_counts": dict(skip_counts.most_common()),
        },
        "pooled_producers": {
            "all_mm_producers": int(len(mm_all_p)),
            "all_mm_with_bio": mm_all_nbio,
            "preT_mm_producers": int(len(mm_pre_p)),
            "preT_mm_with_bio": mm_pre_nbio,
            "all_text_producers": int(len(tx_all_p)),
            "preT_text_producers": int(len(tx_pre_p)),
            "int8_quantisation_median_cosine_10k": i8_cos_median,
        },
        "medoids": {
            "producers_with_medoids": med_tb.num_rows,
            "safety_violations": med_violations,
        },
        "preT_reproducibility_check": {
            "cutoff_utc": "2026-08-16T00:00:00Z",
            "buckets_checked": repro_buckets,
            "producers_checked": repro_checked_producers,
            "max_abs_fp16_diff": repro_max_abs_diff,
            "exact_match": repro_max_abs_diff == 0.0,
        },
        "mac_mps_parity": parity_rep,
        "s16a_candidates": s16a_rep,
        "outputs": {
            fn: {
                "bytes": (out_dir / fn).stat().st_size,
                "sha256": L.sha256_file(out_dir / fn),
            }
            for fn in out_files
        },
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({
        "rows_after_purge": total_rows,
        "all_mm_producers": int(len(mm_all_p)),
        "preT_mm_producers": int(len(mm_pre_p)),
        "medoid_producers": med_tb.num_rows,
        "mps_parity_median": parity_rep["text_modality0_1000"]["cosine_median"],
        "preT_exact": repro_max_abs_diff == 0.0,
    }, indent=1))


if __name__ == "__main__":
    main()
