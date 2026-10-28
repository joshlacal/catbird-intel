"""S4 diagnostics that touch no curated list: truncation loss and bf16 parity, in memory only.

  diag_dims.py POSTS_CORPUS_PARQUET OUT_JSON

On the first 10,000 bake-off posts (post_idx order), per model: top-10 nearest-neighbour overlap
between full-dim vectors and the stored 256-d truncation (harrier is not Matryoshka-trained, EG2 is).
For EG2 also: bf16 vs fp32 cosine on 1,000 posts (both 256-d, re-normalised). Full-dim vectors are
never written to disk (plan Y17 stores 256-d fp16 only).
"""
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import embed  # noqa: E402

N, K = 10_000, 10


def topk(x):
    s = x @ x.T
    np.fill_diagonal(s, -np.inf)
    return np.argpartition(-s, K, axis=1)[:, :K]


def overlap(a, b):
    return float(np.mean([len(set(i) & set(j)) / K for i, j in zip(a, b)]))


def norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main(corpus, out):
    t = pq.read_table(corpus, columns=["post_idx", "text", "bakeoff"]).to_pandas()
    texts = t[t.bakeoff].sort_values("post_idx").text.tolist()[:N]
    res = {}
    for name in ["eg2", "harrier"]:
        m, prompt = embed.load(name)
        full = m.encode(texts, prompt=prompt, batch_size=32, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        cut = norm(full[:, :embed.DIM])
        res[name] = {"full_dim": int(full.shape[1]), "top10_overlap_full_vs_256": overlap(topk(full), topk(cut)),
                     "nan_rows": int(np.isnan(full).any(axis=1).sum())}
        if name == "eg2":
            from sentence_transformers import SentenceTransformer
            s = embed.SPECS["eg2"]
            m32 = SentenceTransformer(str(s["path"]), device="mps", local_files_only=True,
                                      model_kwargs={"dtype": torch.float32}, config_kwargs=s["config_kwargs"])
            m32.max_seq_length = embed.MAX_SEQ
            f32 = m32.encode(texts[:1000], prompt=prompt, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
            c32 = norm(f32[:, :embed.DIM].astype(np.float32))
            cos = (c32 * cut[:1000]).sum(1)
            res[name]["bf16_vs_fp32_cos_256d"] = {"median": float(np.median(cos)), "min": float(cos.min()),
                                                  "p01": float(np.quantile(cos, 0.01))}
            del m32
        del m
        torch.mps.empty_cache()
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
