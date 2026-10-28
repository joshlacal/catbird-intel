"""Rehearsal fixtures for S12m: synthetic thumbnails, a loopback image server, O8-like shards and a gate bundle.

Nothing here touches the network beyond 127.0.0.1. Images are generated in memory from a seed (smooth colour
fields plus shapes, JPEG ~40-120 KB at a 1000-px long side, like Bluesky feed thumbnails) and served from
memory; the server never writes a file. Synthetic DIDs/CIDs are well-formed but random (`did:plc:` + 24 base32).
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mmlib as L

B32 = "abcdefghijklmnopqrstuvwxyz234567"
TID_ALPH = "234567abcdefghijklmnopqrstuvwxyz"


def _b32(seed: str, n: int) -> str:
    h = hashlib.sha256(seed.encode()).digest() * 4
    return "".join(B32[b % 32] for b in h[:n])


def synthetic_did(i) -> str:
    return "did:plc:" + _b32(f"did{i}", 24)


def synthetic_cid(i) -> str:
    return "bafkrei" + _b32(f"cid{i}", 52)


def tid_for(dt: datetime, clock=7) -> str:
    v = (int(dt.timestamp() * 1e6) << 10) | clock
    return "".join(TID_ALPH[(v >> ((12 - i) * 5)) & 31] for i in range(13))


def synthetic_array(seed: int, w: int, h: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (max(2, h // 64), max(2, w // 64), 3), dtype=np.uint8)
    from PIL import Image
    im = Image.fromarray(small).resize((w, h), Image.BICUBIC)
    a = np.asarray(im).astype(np.int16)
    for _ in range(6):    # a few hard-edged shapes so the JPEG is not trivially small
        x0, y0 = rng.integers(0, w - 1), rng.integers(0, h - 1)
        a[y0:y0 + rng.integers(20, h // 2), x0:x0 + rng.integers(20, w // 2)] = rng.integers(0, 256, 3)
    a += rng.integers(-12, 13, a.shape, dtype=np.int16)
    return np.clip(a, 0, 255).astype(np.uint8)


def synthetic_jpeg(seed: int) -> bytes:
    from PIL import Image
    r = random.Random(seed)
    w, h = r.choice([(1000, 750), (750, 1000), (1000, 1000), (1000, 562), (800, 1000), (1000, 667)])
    buf = io.BytesIO()
    Image.fromarray(synthetic_array(seed, w, h)).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def parity_images(n=48):
    """Deterministic PIL images (no JPEG step) for the Mac-vs-GPU multimodal parity check."""
    from PIL import Image
    sizes = [(1000, 750), (750, 1000), (1000, 1000), (1000, 562)]
    return [Image.fromarray(synthetic_array(10_000 + i, *sizes[i % 4])) for i in range(n)]


class ImageServer:
    """ThreadingHTTPServer on 127.0.0.1 serving /img/feed_thumbnail/plain/{did}/{cid} from memory.
    CIDs whose last char is in `fail404` answer 404; `latency_ms` adds a per-request delay."""

    def __init__(self, latency_ms=30, fail404="7"):
        self.cache, self.lock, self.requests = {}, threading.Lock(), 0
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                with srv.lock:
                    srv.requests += 1
                parts = self.path.split("/")
                if len(parts) != 6 or self.path[:len("/img/feed_thumbnail/plain/")] != "/img/feed_thumbnail/plain/":
                    return self._send(404, b"not found", "text/plain")
                cid = parts[5]
                time.sleep(latency_ms / 1000)
                if cid[-1] in fail404:
                    return self._send(404, b"blob not found", "text/plain")
                with srv.lock:
                    body = srv.cache.get(cid)
                if body is None:
                    body = synthetic_jpeg(int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
                    with srv.lock:
                        srv.cache[cid] = body
                self._send(200, body, "image/jpeg")

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def synthetic_posts(texts, now, image_share=0.23, seed=0, young_share=0.02, label_share=0.02, excluded_share=0.02):
    """O8-like rows (uri, p, text, self_labels, image_cids, did, rkey) with ~image_share image posts (1-4 images,
    mean ~1.23) and a few posts that the gate must skip. Returns (rows, excluded_dids, labelled_uris)."""
    r = random.Random(seed)
    rows, excl, labelled = [], set(), set()
    n_img_dist = [1] * 85 + [2] * 8 + [3] * 3 + [4] * 4
    for i, t in enumerate(texts):
        pid = i // 8
        did = synthetic_did(pid)
        age = timedelta(days=1) if r.random() < young_share else timedelta(days=r.uniform(4, 360))
        rkey = tid_for(now - age, clock=i % 1024)
        cids = [synthetic_cid(i * 4 + j) for j in range(r.choice(n_img_dist))] if r.random() < image_share else []
        row = {"uri": L.post_uri(did, rkey), "did": did, "rkey": rkey, "p": pid, "text": t, "image_cids": cids,
               "self_labels": ["sexual"] if cids and r.random() < label_share else []}
        if cids and r.random() < label_share:
            labelled.add(row["uri"])
        if r.random() < excluded_share / 8:
            excl.add(did)
        rows.append(row)
    return rows, excl, labelled


def write_bundle(d, excluded, labelled, watermark: datetime):
    d = pathlib.Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "excluded_dids.txt").write_text("".join(f"{x}\n" for x in sorted(excluded)))
    (d / "labelled_uris.txt").write_text("".join(f"{x}\n" for x in sorted(labelled)))
    man = {"label_watermark_utc": L.iso(watermark), "excluded_dids": len(excluded), "labelled_uris": len(labelled),
           "sha256": {n: L.sha256_file(d / n) for n in ("excluded_dids.txt", "labelled_uris.txt")}}
    (d / "gate_manifest.json").write_text(json.dumps(man, indent=1))
    return man


def write_shards(rows, out_dir, n_shards=3):
    """Input shards without image_cids (O8-like) plus a separate sidecar, so the job's join is exercised."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = pathlib.Path(out_dir)
    (out / "posts").mkdir(parents=True, exist_ok=True)
    (out / "sidecar").mkdir(parents=True, exist_ok=True)
    per = (len(rows) + n_shards - 1) // n_shards
    for j in range(n_shards):
        part = rows[j * per:(j + 1) * per]
        pq.write_table(pa.Table.from_pylist([{k: r[k] for k in ("uri", "p", "text", "self_labels")} for r in part]),
                       out / "posts" / f"j{j:02d}.parquet")
        side = [{"did": r["did"], "rkey": r["rkey"], "image_cids": r["image_cids"]} for r in part if r["image_cids"]]
        pq.write_table(pa.Table.from_pylist(side, schema=pa.schema([("did", pa.string()), ("rkey", pa.string()),
                                                                     ("image_cids", pa.list_(pa.string()))])),
                       out / "sidecar" / f"j{j:02d}.parquet")


def utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def main(argv=None):
    """synth.py make DIR --texts s7_slab.parquet --n N [--shards K]: rehearsal inputs (posts, sidecar, gate)."""
    import argparse
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make"])
    ap.add_argument("dir")
    ap.add_argument("--texts", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--shards", type=int, default=2)
    ap.add_argument("--now", required=True)
    a = ap.parse_args(argv)
    now = utc(a.now)
    texts = pq.read_table(a.texts).sort_by("row_id")["text"].to_pylist()[: a.n]
    rows, excl, lab = synthetic_posts(texts, now)
    write_shards(rows, pathlib.Path(a.dir) / "in", a.shards)
    man = write_bundle(pathlib.Path(a.dir) / "gate", excl, lab, now - timedelta(hours=2))
    print(json.dumps({"rows": len(rows), "image_posts": sum(bool(r["image_cids"]) for r in rows), **man}))


if __name__ == "__main__":
    main()
