#!/usr/bin/env python3
"""Local text-only EmbeddingGemma-2 sidecar for Catbird Intelligence."""

from __future__ import annotations

import base64
import json
import os
import struct
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from graph.s4_pilot.embed import force_text_modality

HOST = "127.0.0.1"
PORT = 3211
DIMS = 256
MAX_BODY_BYTES = 64 * 1024
MAX_TEXTS = 64
MODEL_PATH = Path.home() / "catbird-lake-local/models/embeddinggemma-2"
VECTOR_MANIFEST = Path.home() / "catbird-lake-local/intel-serving-v4/manifest.json"


def load_model() -> tuple[SentenceTransformer, str]:
    force_text_modality()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    model = SentenceTransformer(
        str(MODEL_PATH),
        device=device,
        local_files_only=True,
        model_kwargs={"dtype": dtype},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    model.max_seq_length = 512
    return model, model.prompts["Clustering"]


def quantize(vector: np.ndarray) -> dict[str, int | str]:
    normalized = vector.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(normalized))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Embedding model returned an invalid vector")
    normalized = normalized / norm
    maximum = float(np.max(np.abs(normalized)))
    scale = maximum / 127.0 if maximum else 0.0
    q8 = np.rint(normalized / scale).clip(-127, 127).astype(np.int8) if scale else np.zeros(DIMS, dtype=np.int8)
    scale_bits = struct.unpack("<H", np.float16(scale).tobytes())[0]
    return {"scaleFp16": scale_bits, "q8B64": base64.b64encode(q8.tobytes()).decode("ascii")}


MODEL, PROMPT = load_model()
MU_POSTS = np.asarray(json.loads(VECTOR_MANIFEST.read_text("utf-8"))["muPosts"], dtype=np.float32)
if MU_POSTS.shape != (DIMS,) or not np.all(np.isfinite(MU_POSTS)):
    raise RuntimeError("Intel vector manifest has an invalid muPosts centroid")


def embed_texts(texts: list[str]) -> list[dict[str, int | str]]:
    with torch.inference_mode():
        embeddings = MODEL.encode(
            texts,
            prompt=PROMPT,
            batch_size=min(32, len(texts)),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
    centered: list[np.ndarray] = []
    for vector in embeddings:
        unit = vector[:DIMS].astype(np.float32, copy=False)
        norm = float(np.linalg.norm(unit))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Embedding model returned an invalid vector")
        centered.append(unit / norm - MU_POSTS)
    return [quantize(vector) for vector in centered]


class Handler(BaseHTTPRequestHandler):
    server_version = "CatbirdEG2/1"

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(404, {"error": "NotFound"})
            return
        self._json(200, {"ok": True, "model": "embeddinggemma-2", "dims": DIMS})

    def do_POST(self) -> None:
        if self.path != "/embed":
            self._json(404, {"error": "NotFound"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            if length < 2 or length > MAX_BODY_BYTES:
                raise ValueError("Invalid request body size")
            body: Any = json.loads(self.rfile.read(length))
            texts = body.get("texts") if isinstance(body, dict) else None
            if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_TEXTS:
                raise ValueError(f"texts must contain 1..{MAX_TEXTS} strings")
            if any(not isinstance(text, str) or not text.strip() or len(text) > 4_000 for text in texts):
                raise ValueError("Each text must be a non-empty string of at most 4000 characters")
            self._json(200, {"embeddings": embed_texts(texts)})
        except (ValueError, json.JSONDecodeError) as error:
            self._json(400, {"error": "InvalidRequest", "message": str(error)})
        except Exception:
            self._json(500, {"error": "EmbeddingFailed"})

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} {format % args}", flush=True)

    def _json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    print(f"EmbeddingGemma-2 sidecar listening on http://{HOST}:{PORT}", flush=True)
    HTTPServer((HOST, PORT), Handler).serve_forever()
