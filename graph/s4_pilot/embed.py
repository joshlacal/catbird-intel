"""Embed S4 pilot texts on the Mac (MPS) and store 256-d fp16 vectors (plan Y17).

  embed.py --model eg2|harrier --input PARQUET --id-col post_idx --out DIR [--only-bakeoff] [--limit N]

Models load from ~/catbird-lake-local/models with local_files_only (no downloads). EG2 runs in
bfloat16 (its card: never float16) with its `Clustering` prompt, text-only encoders. harrier runs in
bfloat16 with a fixed clustering instruction in its card's `Instruct: ...\\nQuery: ` form. Both are cut
to 512 tokens. Full-dim outputs are truncated to the first 256 dims and re-normalised in fp32, then
stored as fp16. Work is chunked and resumable: each chunk is written as chunk_*.npz and skipped on
rerun. Per-chunk wall time, token counts and norm statistics go to DIR/chunks.jsonl; the run summary
(throughput posts/s, tokens/post, NaN count) to DIR/run_summary.json.
"""
import argparse
import json
import os
import pathlib
import platform
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pyarrow.parquet as pq
import torch

MODELS = pathlib.Path.home() / "catbird-lake-local/models"
SPECS = {
    "eg2": dict(path=MODELS / "embeddinggemma-2", prompt_name="Clustering", prompt=None,
                config_kwargs={"vision_config": None, "audio_config": None}),
    "harrier": dict(path=MODELS / "harrier-oss-v1-0.6b", prompt_name=None,
                    prompt="Instruct: Identify the topic or theme of the given social media post\nQuery: ",
                    config_kwargs=None),
}
DIM, MAX_SEQ = 256, 512


def load(name):
    from sentence_transformers import SentenceTransformer
    s = SPECS[name]
    kw = dict(device="mps", local_files_only=True, model_kwargs={"dtype": torch.bfloat16})
    if s["config_kwargs"]:
        kw["config_kwargs"] = s["config_kwargs"]
    m = SentenceTransformer(str(s["path"]), **kw)
    m.max_seq_length = MAX_SEQ
    prompt = s["prompt"] if s["prompt"] is not None else m.prompts[s["prompt_name"]]
    return m, prompt


def tokenizer_of(m):
    tok = getattr(m, "tokenizer", None)
    if tok is None or not callable(tok):
        tok = m[0].processor.tokenizer
    return getattr(tok, "tokenizer", tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(SPECS))
    ap.add_argument("--input", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-bakeoff", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cols = [a.id_col, a.text_col] + (["bakeoff"] if a.only_bakeoff else [])
    t = pq.read_table(a.input, columns=cols).to_pandas()
    if a.only_bakeoff:
        t = t[t.bakeoff]
    t = t.sort_values(a.id_col).reset_index(drop=True)
    if a.limit:
        t = t.iloc[: a.limit]
    ids = t[a.id_col].to_numpy(np.int64)
    texts = t[a.text_col].tolist()

    t0 = time.time()
    m, prompt = load(a.model)
    load_s = time.time() - t0
    tok = tokenizer_of(m)
    t1 = time.time()
    n_text = np.array([len(x) for x in tok(texts, add_special_tokens=True, truncation=True, max_length=MAX_SEQ)["input_ids"]])
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    n_full = np.minimum(n_text + n_prompt, MAX_SEQ)
    tokenize_s = time.time() - t1
    order = np.argsort(-n_full, kind="stable")  # longest first: memory problems show up at once

    log = open(out / "chunks.jsonl", "a")
    run_start = time.time()
    embed_s, embedded, nan_total = 0.0, 0, 0
    for ci, start in enumerate(range(0, len(order), a.chunk)):
        f = out / f"chunk_{ci:05d}.npz"
        sel = order[start:start + a.chunk]
        if f.exists():
            continue
        c0 = time.time()
        emb = m.encode([texts[i] for i in sel], prompt=prompt, batch_size=a.batch_size, convert_to_tensor=True,
                       normalize_embeddings=False, show_progress_bar=False)
        if emb.device.type == "mps":
            torch.mps.synchronize()
        dt = time.time() - c0
        full = emb.float().cpu().numpy()
        nan = int(np.isnan(full).any(axis=1).sum())
        full_norm = np.linalg.norm(full, axis=1)
        cut = full[:, :DIM]
        cut_norm = np.linalg.norm(cut, axis=1)
        v = (cut / np.maximum(cut_norm, 1e-12)[:, None]).astype(np.float16)
        np.savez(f, ids=ids[sel], vec=v)
        nan_total += nan
        embed_s += dt
        embedded += len(sel)
        rec = dict(chunk=ci, n=len(sel), seconds=round(dt, 3), posts_per_s=round(len(sel) / dt, 1),
                   tokens_mean=float(n_full[sel].mean()), tokens_max=int(n_full[sel].max()), nan_rows=nan,
                   full_norm_min=float(full_norm.min()), full_norm_max=float(full_norm.max()),
                   cut_norm_p01=float(np.quantile(cut_norm, 0.01)), cut_norm_p50=float(np.median(cut_norm)),
                   cut_norm_p99=float(np.quantile(cut_norm, 0.99)),
                   mps_driver_gb=round(torch.mps.driver_allocated_memory() / 2**30, 2))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if nan:
            raise SystemExit(f"NaN rows in chunk {ci}: stopping")

    # Assemble in id order.
    parts = [np.load(f) for f in sorted(out.glob("chunk_*.npz"))]
    all_ids = np.concatenate([p["ids"] for p in parts])
    all_vec = np.concatenate([p["vec"] for p in parts])
    o = np.argsort(all_ids)
    all_ids, all_vec = all_ids[o], all_vec[o]
    assert np.array_equal(all_ids, ids), "assembled ids do not match the input"
    np.save(out / "ids.npy", all_ids)
    np.save(out / "vec256_fp16.npy", all_vec)
    import sentence_transformers
    import transformers
    summary = dict(model=a.model, model_path=str(SPECS[a.model]["path"]), prompt=prompt, dtype="bfloat16",
                   max_seq=MAX_SEQ, dim_stored=DIM, n=len(ids), embedded_this_run=embedded,
                   load_seconds=round(load_s, 2), tokenize_seconds=round(tokenize_s, 2),
                   embed_seconds_this_run=round(embed_s, 2), wall_seconds_this_run=round(time.time() - run_start, 2),
                   posts_per_s_this_run=round(embedded / embed_s, 2) if embed_s else None,
                   tokens_per_post_with_prompt=float(n_full.mean()), tokens_per_post_text_only=float(n_text.mean()),
                   prompt_tokens=n_prompt, truncated_at_max_seq=int((n_text + n_prompt > MAX_SEQ).sum()),
                   nan_rows_this_run=nan_total, batch_size=a.batch_size,
                   versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                 sentence_transformers=sentence_transformers.__version__, numpy=np.__version__,
                                 python=platform.python_version(), macos=platform.mac_ver()[0]))
    json.dump(summary, open(out / "run_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
