"""S7 EmbeddingGemma 2 text benchmark: throughput, tokens/post, NaN check and parity vs the Mac.

  bench.py --model-dir D --data DIR --out DIR [--device cuda|mps|cpu]
           [--parity-ref parity_mac.npz | --emit-parity parity_mac.npz --parity-only]
           [--limit-random N] [--limit-slab N] [--batch-sizes 64,128,256,512] [--sweep-n 10000]
           [--unit-cap U --rate R --preamble-min M]

Same model stack as S4 (graph/s4_pilot/embed.py): sentence-transformers, bfloat16 (never float16), text-only
(vision/audio configs dropped, media-URL modality sniffing disabled), max_seq 512, `Clustering` prompt
(`task: clustering | query: `), raw (unnormalised) full-dim outputs, then 256-d truncation + renormalisation +
fp16, the stored format. Measured separately for the random and slab rows: posts/s (wall time of `encode`,
which includes tokenisation, device sync, and the 256-d fp16 conversion), tokens/post (with and without the
prompt). The batch size is picked by a short sweep on random rows. Parity: rows 0..999 (random) and the first
250 slab rows are encoded on their own at batch 64 (S4's batch), and compared by cosine with the Mac reference
(full dim and 256-d). Writes OUT/bench.json; exits 3 on any NaN.
"""
import argparse
import json
import os
import pathlib
import platform
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import s7lib  # noqa: E402

PROMPT_NAME, MAX_SEQ, DIM = "Clustering", 512, 256
PARITY_RANDOM, PARITY_SLAB, PARITY_BATCH = 1000, 250, 64
SLAB_ROW0 = 1_000_000


def force_text_modality():
    """sentence-transformers 6 treats a post whose whole text is a media URL (or an existing local media path) as
    an image/video/audio input and would fetch it. Every S7 input is text; never fetch URLs from post text."""
    from sentence_transformers.base import modality as stm
    stm.is_image_url_or_path = stm.is_video_url_or_path = stm.is_audio_url_or_path = lambda _text: False
    assert stm.infer_modality("https://www.youtube.com/watch?v=x") == "text"
    assert stm.infer_modality("https://example.com/a.jpg") == "text"


def sync(dev):
    import torch
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def load(model_dir, dev):
    import torch
    from sentence_transformers import SentenceTransformer
    force_text_modality()
    m = SentenceTransformer(str(model_dir), device=dev, local_files_only=True,
                            model_kwargs={"dtype": torch.bfloat16},
                            config_kwargs={"vision_config": None, "audio_config": None})
    m.max_seq_length = MAX_SEQ
    dtypes = sorted({str(p.dtype) for p in m.parameters()})
    assert dtypes == ["torch.bfloat16"], f"model not pure bf16: {dtypes}"
    return m, m.prompts[PROMPT_NAME], dtypes


def tokenizer_of(m):
    tok = getattr(m, "tokenizer", None)
    if tok is None or not callable(tok):
        tok = m[0].processor.tokenizer
    return getattr(tok, "tokenizer", tok)


def token_counts(tok, texts, prompt):
    import numpy as np
    n_text = np.array([len(x) for x in tok(texts, add_special_tokens=True, truncation=True, max_length=MAX_SEQ)["input_ids"]])
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    full = np.minimum(n_text + n_prompt, MAX_SEQ)
    return {"n": int(len(full)), "tokens_per_post_with_prompt": float(full.mean()),
            "tokens_per_post_text_only": float(n_text.mean()), "prompt_tokens": n_prompt,
            "p99_with_prompt": float(np.quantile(full, 0.99)), "truncated_at_512": int((n_text + n_prompt > MAX_SEQ).sum())}


def encode(m, texts, prompt, bs, dev):
    """Timed encode -> (seconds, full fp32 numpy, stored 256-d fp16 numpy). The clock covers what S12 pays per
    shard: tokenisation, the forward passes, the copy to host and the 256-d fp16 conversion."""
    import numpy as np
    sync(dev)
    t0 = time.time()
    emb = m.encode(texts, prompt=prompt, batch_size=bs, convert_to_tensor=True, normalize_embeddings=False,
                   show_progress_bar=False)
    full = emb.float().cpu().numpy()
    cut = full[:, :DIM]
    v = (cut / np.maximum(np.linalg.norm(cut, axis=1), 1e-12)[:, None]).astype(np.float16)
    sync(dev)
    return time.time() - t0, full, v


def nan_stats(full, v):
    import numpy as np
    nrm = np.linalg.norm(full, axis=1)
    return {"nan_rows_full": int(np.isnan(full).any(axis=1).sum()), "nan_rows_stored": int(np.isnan(v.astype(np.float32)).any(axis=1).sum()),
            "inf_rows": int(np.isinf(full).any(axis=1).sum()), "full_norm_min": float(np.nanmin(nrm)), "full_norm_max": float(np.nanmax(nrm))}


def mem(dev):
    import torch
    if dev == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if dev == "mps":
        return round(torch.mps.driver_allocated_memory() / 2**30, 2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--parity-ref")
    ap.add_argument("--emit-parity")
    ap.add_argument("--parity-only", action="store_true")
    ap.add_argument("--limit-random", type=int, default=0)
    ap.add_argument("--limit-slab", type=int, default=0)
    ap.add_argument("--batch-sizes", default="64,128,256,512")
    ap.add_argument("--sweep-n", type=int, default=10_000)
    ap.add_argument("--unit-cap", type=float, default=0.0)
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--preamble-min", type=float, default=0.0)
    ap.add_argument("--mps-gb", type=float, default=7.0, help="MPS allocator cap for Mac runs")
    a = ap.parse_args()

    import numpy as np
    import pyarrow.parquet as pq
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "mps" and hasattr(torch.mps, "recommended_max_memory"):
        torch.mps.set_per_process_memory_fraction(min(1.0, a.mps_gb * 2**30 / torch.mps.recommended_max_memory()))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    guard = s7lib.UnitGuard(a.unit_cap, a.rate, a.preamble_min) if a.unit_cap and a.rate else None

    def allows(seconds):
        return guard is None or guard.allows(seconds)

    rnd = pq.read_table(pathlib.Path(a.data) / "s7_random.parquet").to_pandas().sort_values("row_id")
    slb = pq.read_table(pathlib.Path(a.data) / "s7_slab.parquet").to_pandas().sort_values("row_id")
    if a.limit_random:
        rnd = rnd.iloc[: a.limit_random]
    if a.limit_slab:
        slb = slb.iloc[: a.limit_slab]
    par = __import__("pandas").concat([rnd.iloc[:PARITY_RANDOM], slb.iloc[:PARITY_SLAB]])

    import sentence_transformers
    import transformers
    res = {"device": dev, "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
           "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                        "sentence_transformers": sentence_transformers.__version__, "numpy": np.__version__,
                        "python": platform.python_version()},
           "dtype": "bfloat16", "max_seq": MAX_SEQ, "dim_stored": DIM, "n_random": int(len(rnd)), "n_slab": int(len(slb))}
    t0 = time.time()
    m, prompt, dtypes = load(a.model_dir, dev)
    res.update(load_seconds=round(time.time() - t0, 2), prompt=prompt, param_dtypes=dtypes)
    tok = tokenizer_of(m)

    # Parity rows on their own, at S4's batch size (same procedure on the Mac and on Colab).
    ps, pfull, pv = encode(m, par.text.tolist(), prompt, PARITY_BATCH, dev)
    res["parity_encode"] = {"rows": int(len(par)), "seconds": round(ps, 2), **nan_stats(pfull, pv)}
    pids = par.row_id.to_numpy(np.int64)
    if a.emit_parity:
        np.savez(a.emit_parity, row_id=pids, vec=pfull.astype(np.float32))
        res["parity_emitted"] = a.emit_parity
    if a.parity_ref:
        ref = np.load(a.parity_ref)
        pos = {int(r): i for i, r in enumerate(ref["row_id"])}
        sel = [i for i, r in enumerate(pids) if int(r) in pos]
        rv = ref["vec"][[pos[int(pids[i])] for i in sel]]
        mine = pfull[sel]
        is_r = pids[sel] < SLAB_ROW0
        res["parity"] = {"ref": a.parity_ref,
                         "random_full": s7lib.cosine_stats(mine[is_r], rv[is_r]),
                         "random_256": s7lib.cosine_stats(mine[is_r], rv[is_r], DIM),
                         "slab_full": s7lib.cosine_stats(mine[~is_r], rv[~is_r]) if (~is_r).any() else None,
                         "slab_256": s7lib.cosine_stats(mine[~is_r], rv[~is_r], DIM) if (~is_r).any() else None}
        res["parity"]["bar_median_ge_0_99_on_random"] = res["parity"]["random_full"]["median"] >= 0.99
    if a.parity_only:
        json.dump(res, open(out / "bench.json", "w"), indent=1)
        print(s7lib.dumps(res))
        return

    res["tokens"] = {"random": token_counts(tok, rnd.text.tolist(), prompt), "slab": token_counts(tok, slb.text.tolist(), prompt)}
    encode(m, rnd.text.tolist()[:256], prompt, 64, dev)  # warm-up (kernels, allocator)

    sweep_texts = rnd.text.tolist()[PARITY_RANDOM:PARITY_RANDOM + a.sweep_n] or rnd.text.tolist()[: a.sweep_n]
    res["batch_sweep"], best = [], None
    for bs in [int(x) for x in a.batch_sizes.split(",") if x]:
        est = len(sweep_texts) / (best[1] if best else 200.0) * 1.5
        if not allows(est):
            res["batch_sweep"].append({"batch": bs, "skipped": "unit cap", **(guard.state() if guard else {})})
            break
        try:
            s, full, v = encode(m, sweep_texts, prompt, bs, dev)
        except torch.OutOfMemoryError:
            res["batch_sweep"].append({"batch": bs, "oom": True})
            torch.cuda.empty_cache() if dev == "cuda" else None
            break
        pps = len(sweep_texts) / s
        res["batch_sweep"].append({"batch": bs, "n": len(sweep_texts), "seconds": round(s, 2), "posts_per_s": round(pps, 1),
                                   "peak_mem_gb": mem(dev), "nan_rows": nan_stats(full, v)["nan_rows_full"]})
        print("[bench] sweep", json.dumps(res["batch_sweep"][-1]), flush=True)
        if best is None or pps > best[1]:
            best = (bs, pps)
    if best is None:
        sys.exit("[bench] no batch size completed")
    res["batch_chosen"] = best[0]

    nan_total = 0
    for name, df in [("random", rnd), ("slab", slb)]:
        est = len(df) / best[1] * 1.5
        if not allows(est):
            res[name] = {"skipped": f"unit cap: estimate {est:.0f}s", **guard.state()}
            continue
        s, full, v = encode(m, df.text.tolist(), prompt, best[0], dev)
        ns = nan_stats(full, v)
        nan_total += ns["nan_rows_full"] + ns["nan_rows_stored"] + ns["inf_rows"]
        tk = res["tokens"][name]["tokens_per_post_with_prompt"]
        r = {"n": int(len(df)), "batch": best[0], "seconds": round(s, 2), "posts_per_s": round(len(df) / s, 1),
             "tokens_per_s_with_prompt": round(len(df) * tk / s, 0), "peak_mem_gb": mem(dev), **ns}
        if a.rate:
            u, arith = s7lib.units_per_10m(r["posts_per_s"], a.rate)
            u15, arith15 = s7lib.units_per_10m(r["posts_per_s"], a.rate, 1.5)
            r.update(units_per_10m=round(u, 3), units_per_10m_arith=arith, units_per_10m_x1_5=round(u15, 3),
                     units_per_10m_x1_5_arith=arith15)
        res[name] = r
        print("[bench]", name, json.dumps(r), flush=True)
    res["nan_or_inf_rows_total"] = nan_total
    res["units"] = guard.state() if guard else None
    json.dump(res, open(out / "bench.json", "w"), indent=1)
    print(s7lib.dumps(res))
    if nan_total:
        sys.exit(3)


if __name__ == "__main__":
    main()
