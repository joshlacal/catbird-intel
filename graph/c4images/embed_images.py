"""C4 embed step (skeleton): fetch -> decode in memory -> EG2 vision -> vectors only.

Plan §5.2 multimodal variant / §5.3 pilot, both variants from the same in-memory bytes:
  (a) native interleaved: one EG2 embedding of `Clustering prompt + post text + " <|image|>" per image`
      with the images, vision tower frozen, S = 70 soft tokens per image;
  (b) late fusion: the text-only vector t and one vector per image (no prompt, as the EG2 card says);
      the post vector normalise(t + β · mean(image vectors)) is formed at scoring time, with β chosen on
      teacher-validation pairs, so β is not baked in here (`late_fusion` is the formula).
The text string is the same in (a) and t: post text plus " [image: alt]" for each non-empty alt, the
form the VM pass uses for the text arm, so the only difference between the arms is the pixels.

Storage rule: image bytes never touch disk. `run` keeps a post's bytes in a local list, decodes them
with PIL from BytesIO, embeds, then drops every reference before the next post. A post with any image
that is not fetched cleanly (non-200, not an image, undecodable) is skipped whole and logged. Vectors are
256-d (Matryoshka truncation, re-normalised in fp32) stored as fp16, with the post URI and blob CIDs;
`audit` lists the output directory, refuses unexpected file types, and scans every written file for
JPEG/PNG/GIF/WebP signatures.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import time

import numpy as np

DIM = 256
SOFT_TOKENS = 70
MODEL_DIR = pathlib.Path.home() / "catbird-lake-local/models/embeddinggemma-2"
IMAGE_TOKEN = "<|image|>"
ALLOWED_SUFFIXES = {".npz", ".jsonl", ".json"}
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"RIFF")
MAX_PIXELS = 40_000_000


def normalise(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def truncate(v, dim=DIM):
    """Matryoshka truncation then re-normalisation in fp32, stored fp16 (plan Y17)."""
    return normalise(np.asarray(v, dtype=np.float32)[..., :dim]).astype(np.float16)


def late_fusion(t, image_vecs, beta):
    """Variant (b): normalise(t + β · mean(image vectors)); t may be zeros for a post without text."""
    t = np.asarray(t, dtype=np.float32)
    m = np.asarray(image_vecs, dtype=np.float32).mean(axis=0)
    return normalise(t + beta * m)


def decode_image(data: bytes):
    """PIL image (RGB) from bytes in memory. Raises on non-image data or decompression bombs."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    with Image.open(io.BytesIO(data)) as im:
        if im.format not in {"JPEG", "PNG", "WEBP", "GIF"}:
            raise ValueError(f"unsupported image format {im.format}")
        im.load()
        return im.convert("RGB")


def post_text(text: str, alts) -> str:
    return (text or "") + "".join(f" [image: {a}]" for a in (alts or []) if a)


class EG2Backend:
    """EmbeddingGemma 2 with text + vision (audio not loaded), bf16 on MPS, local files only."""
    def __init__(self, model_dir=MODEL_DIR, device="mps", soft_tokens=SOFT_TOKENS):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.base import modality as stm
        # Text never becomes a URL fetch: a post whose text is a bare image URL stays text (as S4 embed.py).
        stm.is_image_url_or_path = stm.is_video_url_or_path = stm.is_audio_url_or_path = lambda _t: False
        self.model = SentenceTransformer(str(model_dir), device=device, local_files_only=True,
                                         model_kwargs={"dtype": torch.bfloat16},
                                         config_kwargs={"audio_config": None})
        self.model[0].processor.image_processor.max_soft_tokens = soft_tokens
        self.prompt = self.model.prompts["Clustering"]

    def encode(self, inputs):
        return normalise(self.model.encode(inputs, convert_to_numpy=True, batch_size=len(inputs)))


def embed_posts(backend, posts):
    """posts: list of (text, [PIL images]). Returns per post dict(text, interleaved, images) at full dim."""
    inputs, spans = [], []
    for text, images in posts:
        start = len(inputs)
        has_text = bool(text.strip())
        if has_text:
            inputs.append(backend.prompt + text)
        inputs.append({"text": backend.prompt + text + "".join(f" {IMAGE_TOKEN}" for _ in images),
                       "image": list(images)})
        inputs.extend({"image": im} for im in images)
        spans.append((start, has_text, len(images)))
    vecs = backend.encode(inputs)
    out = []
    for start, has_text, n in spans:
        i = start
        t = vecs[i] if has_text else np.zeros(vecs.shape[1], np.float32)
        i += has_text
        out.append({"text": t, "has_text": has_text, "interleaved": vecs[i], "images": vecs[i + 1:i + 1 + n]})
    return out


class VectorSink:
    """Writes chunk_NNNNN.npz files holding only ids and vectors, plus a JSONL log without bodies."""
    def __init__(self, out_dir, chunk_posts=2000):
        self.dir = pathlib.Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunk_posts, self.buf, self.n_chunks = chunk_posts, [], 0
        self.log = open(self.dir / "log.jsonl", "a")

    def add(self, uri, cids, vec):
        self.buf.append((uri, cids, vec))
        if len(self.buf) >= self.chunk_posts:
            self.flush()

    def note(self, record):
        self.log.write(json.dumps(record, sort_keys=True) + "\n")

    def flush(self):
        if not self.buf:
            return
        uris = np.array([b[0] for b in self.buf])
        cids = np.array(["|".join(b[1]) for b in self.buf])
        n_img = np.array([len(b[1]) for b in self.buf], np.int32)
        text = np.stack([truncate(b[2]["text"]) if b[2]["has_text"] else np.zeros(DIM, np.float16) for b in self.buf])
        inter = np.stack([truncate(b[2]["interleaved"]) for b in self.buf])
        imgs = np.concatenate([truncate(b[2]["images"]) for b in self.buf])
        np.savez(self.dir / f"chunk_{self.n_chunks:05d}.npz", uri=uris, cids=cids, n_images=n_img,
                 has_text=np.array([b[2]["has_text"] for b in self.buf]), text=text, interleaved=inter,
                 image=imgs)
        self.n_chunks += 1
        self.buf = []

    def close(self):
        self.flush()
        self.log.close()


def group_by_post(fetch_rows):
    posts, order = {}, []
    for r in fetch_rows:
        if r["uri"] not in posts:
            posts[r["uri"]] = {"uri": r["uri"], "text": r.get("text") or "", "alts": [], "items": []}
            order.append(r["uri"])
        posts[r["uri"]]["items"].append((r["image_index"], r["cid"], r["url"]))
        posts[r["uri"]]["alts"].append(r.get("alt") or "")
    return [posts[u] for u in order]


def run(fetch_rows, fetcher, backend, sink, batch_posts=16):
    """Fetch, decode in memory, embed both variants, write vectors. Returns counts by outcome."""
    counts = {"posts_embedded": 0, "images_embedded": 0, "posts_skipped": 0}
    batch = []

    def drain():
        if not batch:
            return
        vecs = embed_posts(backend, [(post_text(p["text"], p["alts"]), ims) for p, ims in batch])
        for (p, ims), v in zip(batch, vecs):
            sink.add(p["uri"], [c for _i, c, _u in sorted(p["items"])], v)
            counts["posts_embedded"] += 1
            counts["images_embedded"] += len(ims)
        batch.clear()  # drops the last references to the decoded images

    for p in group_by_post(fetch_rows):
        images, reason = [], None
        for _i, cid, url in sorted(p["items"]):
            data, log = fetcher.get(url)
            sink.note({"uri": p["uri"], "cid": cid, **{k: v for k, v in log.items() if k != "url"}})
            if data is None:
                reason = log["skip"]
                break
            try:
                images.append(decode_image(data))
            except Exception as e:  # undecodable bytes are a skip, never a write
                reason = f"decode_{type(e).__name__}"
                break
            finally:
                del data
        if reason:
            counts["posts_skipped"] += 1
            counts[f"skip_{reason}"] = counts.get(f"skip_{reason}", 0) + 1
            sink.note({"uri": p["uri"], "post_skip": reason})
            images.clear()
            continue
        batch.append((p, images))
        if len(batch) >= batch_posts:
            drain()
    drain()
    sink.close()
    return counts


def listing(out_dir):
    d = pathlib.Path(out_dir)
    return {str(p.relative_to(d)): p.stat().st_size for p in sorted(d.rglob("*")) if p.is_file()} if d.exists() else {}


def audit(out_dir, before: dict):
    """Receipt: files added or changed since `before`, total bytes, and any file breaking the rule."""
    after = listing(out_dir)
    changed = {k: v for k, v in after.items() if before.get(k) != v}
    bad_suffix = [k for k in changed if pathlib.Path(k).suffix not in ALLOWED_SUFFIXES]
    magic = []
    for k in changed:
        data = (pathlib.Path(out_dir) / k).read_bytes()
        if pathlib.Path(k).suffix == ".npz":
            # npz members are .npy arrays; scan the decompressed members, not just the zip container.
            # An npz that will not load as plain arrays is judged on its raw bytes alone.
            try:
                with np.load(pathlib.Path(out_dir) / k, allow_pickle=False) as z:
                    data = data + b"".join(z[m].tobytes() for m in z.files)
            except (ValueError, OSError):
                pass
        if any(sig in data for sig in IMAGE_MAGIC if sig != b"RIFF") or b"RIFF" in data and b"WEBP" in data:
            magic.append(k)
    return {"files_changed": changed, "bytes_added": sum(after.values()) - sum(before.values()),
            "unexpected_suffix": bad_suffix, "image_signature_found": magic,
            "ok": not bad_suffix and not magic, "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
