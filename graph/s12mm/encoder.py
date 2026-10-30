"""EmbeddingGemma 2 with text + vision (audio never loaded), bf16 only, for S12m.

Text-only posts are encoded exactly as S4/S7 did (prompt passed as `prompt=`, max_seq 512, media-URL modality
sniffing disabled so a post whose text is a bare URL stays text), so text vectors stay comparable with S4's
and with the Mac parity reference. Posts with fetched images are one interleaved input,
`{"text": prompt + text + " <|image|>"*n, "image": [PIL, ...]}`, with max_seq raised so image soft tokens are
never truncated; the image budget is `processor.image_processor.max_soft_tokens` (70 or 280; EG2 accepts
70-1120). Outputs are full-dim fp32 (unnormalised); mmlib.to_stored makes the stored 256-d fp16.
"""
from __future__ import annotations

import io
import os
import pathlib
import subprocess
import threading
import time

from mmlib import mm_text

PROMPT_NAME, TEXT_MAX_SEQ, MM_MAX_SEQ = "Clustering", 512, 2048
MAX_PIXELS = 40_000_000


def force_text_modality():
    from sentence_transformers.base import modality as stm
    stm.is_image_url_or_path = stm.is_video_url_or_path = stm.is_audio_url_or_path = lambda _t: False
    assert stm.infer_modality("https://example.com/a.jpg") == "text"


def decode(data: bytes):
    """RGB PIL image from in-memory bytes; raises on non-images and decompression bombs. Never touches disk."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    with Image.open(io.BytesIO(data)) as im:
        if im.format not in {"JPEG", "PNG", "WEBP", "GIF"}:
            raise ValueError(f"unsupported image format {im.format}")
        im.load()
        return im.convert("RGB")


class EG2:
    def __init__(self, model_dir, device=None, soft_tokens=70, mps_gb=7.0):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if self.dev == "mps" and hasattr(torch.mps, "recommended_max_memory"):
            torch.mps.set_per_process_memory_fraction(min(1.0, mps_gb * 2**30 / torch.mps.recommended_max_memory()))
        force_text_modality()
        t0 = time.time()
        self.m = SentenceTransformer(str(model_dir), device=self.dev, local_files_only=True,
                                     model_kwargs={"dtype": torch.bfloat16}, config_kwargs={"audio_config": None})
        self.load_seconds = time.time() - t0
        self.param_dtypes = sorted({str(p.dtype) for p in self.m.parameters()})
        assert self.param_dtypes == ["torch.bfloat16"], f"model not pure bf16: {self.param_dtypes}"
        self.params_m = sum(p.numel() for p in self.m.parameters()) / 1e6
        self.prompt = self.m.prompts[PROMPT_NAME]
        self.proc = self.m[0].processor
        self.prompt_tokens = len(self.proc.tokenizer(self.prompt, add_special_tokens=False)["input_ids"]) + 2
        self.set_soft_tokens(soft_tokens)

    def set_soft_tokens(self, s):
        if not 70 <= s <= 1120:
            raise ValueError("EG2 image budget is 70-1120 soft tokens")
        self.proc.image_processor.max_soft_tokens = s
        self.soft_tokens = s

    def sync(self):
        if self.dev == "cuda":
            self.torch.cuda.synchronize()
        elif self.dev == "mps":
            self.torch.mps.synchronize()

    def _encode(self, inputs, bs, **kw):
        import numpy as np
        if not inputs:
            return np.zeros((0, 768), np.float32)
        emb = self.m.encode(inputs, batch_size=bs, convert_to_tensor=True, normalize_embeddings=False,
                            show_progress_bar=False, **kw)
        return emb.float().cpu().numpy()

    def encode_text(self, texts, bs=256):
        self.m.max_seq_length = TEXT_MAX_SEQ
        return self._encode(list(texts), bs, prompt=self.prompt)

    def clip_mm_text(self, text, n_images):
        """Cap the text part of an interleaved input so truncation at MM_MAX_SEQ can never reach image tokens
        (the processor raises on a partly truncated image). Each image is reserved soft_tokens + 8 for its
        begin/end markers; text that already fits is returned unchanged, so earlier vectors stay valid."""
        t = text or ""
        budget = MM_MAX_SEQ - self.prompt_tokens - n_images * (self.soft_tokens + 8) - 8
        if len(t.encode()) <= budget:
            return t
        tok = self.proc.tokenizer
        ids = tok(t, add_special_tokens=False)["input_ids"]
        return t if len(ids) <= budget else tok.decode(ids[:budget])

    def encode_mm(self, posts, bs=16):
        """posts: [(text, [PIL images])]; one interleaved forward input per post."""
        self.m.max_seq_length = MM_MAX_SEQ
        inputs = [{"text": mm_text(self.prompt, self.clip_mm_text(t, len(ims)), len(ims)), "image": list(ims)}
                  for t, ims in posts]
        return self._encode(inputs, bs)

    def mm_tokens(self, text, images):
        """Sequence length of one interleaved input (text + soft tokens), from the processor."""
        self.m.max_seq_length = MM_MAX_SEQ
        f = self.m.tokenize([{"text": mm_text(self.prompt, text, len(images)), "image": list(images)}])
        return int(f["attention_mask"].sum())

    def empty_cache(self):
        if self.dev == "cuda":
            self.torch.cuda.empty_cache()
        elif self.dev == "mps":
            self.torch.mps.empty_cache()

    def peak_mem_gb(self):
        if self.dev == "cuda":
            return round(self.torch.cuda.max_memory_allocated() / 2**30, 2)
        if self.dev == "mps":
            return round(self.torch.mps.driver_allocated_memory() / 2**30, 2)
        return None


class GpuSampler:
    """Samples nvidia-smi every `period` s in a thread; `window()` returns mean utilisation since `mark()`.
    On machines without nvidia-smi (the Mac) every reading is None."""

    def __init__(self, period=1.0):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self.available = _which("nvidia-smi")
        self._t = threading.Thread(target=self._run, daemon=True)
        if self.available:
            self._t.start()

    def _run(self):
        while not self._stop.wait(self.period):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                u, m, p = [float(x) for x in out.strip().splitlines()[0].split(",")]
                self.samples.append((time.time(), u, m, p))
            except Exception:
                pass

    def mark(self):
        return time.time()

    def window(self, since):
        s = [x for x in self.samples if x[0] >= since]
        if not s:
            return {"gpu_util_mean": None, "gpu_mem_mib_max": None, "power_w_mean": None, "samples": 0}
        return {"gpu_util_mean": round(sum(x[1] for x in s) / len(s), 1), "gpu_mem_mib_max": max(x[2] for x in s),
                "power_w_mean": round(sum(x[3] for x in s) / len(s), 1), "samples": len(s)}

    def stop(self):
        self._stop.set()


def _which(cmd):
    return any(os.access(pathlib.Path(d) / cmd, os.X_OK) for d in os.environ.get("PATH", "").split(os.pathsep) if d)
