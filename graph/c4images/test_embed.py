"""Embed skeleton on synthetic images generated in memory. Set C4_EG2_SMOKE=1 to also run the real
EG2 vision path on the Mac (local weights only, ~2.5 GB RAM)."""
import io
import os
import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cdn  # noqa: E402
import embed_images as em  # noqa: E402


def synthetic(fmt="PNG", colour=(200, 30, 30), size=(64, 48), noise=False):
    if noise:
        arr = (np.random.default_rng(0).random((size[1], size[0], 3)) * 255).astype("uint8")
        im = Image.fromarray(arr)
    else:
        im = Image.new("RGB", size, colour)
    buf = io.BytesIO()
    im.save(buf, fmt)
    return buf.getvalue()


class FakeBackend:
    """Deterministic 768-d vectors: text hashed, images from mean colour, so tests can check routing."""
    prompt = "task: clustering | query: "

    def __init__(self):
        self.calls = []

    def encode(self, inputs):
        self.calls.append(inputs)
        out = []
        for x in inputs:
            v = np.zeros(768, np.float32)
            if isinstance(x, str):
                v[hash(x) % 700] = 1
            else:
                for im in x["image"] if isinstance(x["image"], list) else [x["image"]]:
                    v[700 + int(np.asarray(im).mean()) % 60] += 1
                if "text" in x:
                    v[hash(x["text"]) % 700] += 1
            out.append(v)
        return em.normalise(np.stack(out))


class FakeFetcher:
    def __init__(self, bodies):
        self.bodies = bodies

    def get(self, url):
        b = self.bodies.get(url)
        if isinstance(b, int):
            return None, {"url": url, "status": b, "skip": f"cdn_{b}"}
        return b, {"url": url, "status": 200, "skip": None}


def rows(uri, n, text="hello"):
    return [{"uri": uri, "image_index": i, "cid": f"bafk{uri[-1]}{i}", "url": f"https://cdn/{uri}/{i}", "text": text,
             "alt": "a cat" if i == 0 else ""} for i in range(n)]


def test_decode_in_memory_formats_and_rejects():
    for fmt in ("PNG", "JPEG", "WEBP", "GIF"):
        assert em.decode_image(synthetic(fmt)).mode == "RGB"
    with pytest.raises(Exception):
        em.decode_image(b"not an image")
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "BMP")
    with pytest.raises(ValueError):
        em.decode_image(buf.getvalue())


def test_truncate_and_late_fusion():
    v = em.normalise(np.arange(1, 769, dtype=np.float32))
    t = em.truncate(v)
    assert t.dtype == np.float16 and t.shape == (256,) and abs(float(np.linalg.norm(t.astype(np.float32))) - 1) < 1e-3
    a, b = np.eye(4)[0], np.eye(4)[1]
    np.testing.assert_allclose(em.late_fusion(a, [b, b], 1.0), em.normalise(a + b))
    np.testing.assert_allclose(em.late_fusion(np.zeros(4), [b], 0.5), b)


def test_run_both_variants_skip_whole_post_and_audit(tmp_path):
    fetch = rows("at://p/1", 2) + rows("at://p/2", 1, text="") + rows("at://p/3", 2)
    fetch[2]["alt"] = ""  # post 2: no text and no alt -> no text vector (an alt alone would count as text)
    bodies = {r["url"]: synthetic(colour=(10 * i, 0, 0)) for i, r in enumerate(fetch)}
    bodies[fetch[-1]["url"]] = 404  # second image of post 3 taken down -> whole post skipped
    backend, out = FakeBackend(), tmp_path / "vec"
    before = em.listing(out)
    counts = em.run(fetch, FakeFetcher(bodies), backend, em.VectorSink(out), batch_posts=8)
    assert counts["posts_embedded"] == 2 and counts["images_embedded"] == 3 and counts["skip_cdn_404"] == 1
    # post 1: text, interleaved (with 2 image placeholders, alt folded into the text), 2 images; post 2: no text
    inputs = backend.calls[0]
    assert inputs[0] == backend.prompt + "hello [image: a cat]"
    assert inputs[1]["text"].endswith("hello [image: a cat] <|image|> <|image|>") and len(inputs[1]["image"]) == 2
    assert "text" not in inputs[2] and "text" not in inputs[3]
    assert "text" not in inputs[4] or inputs[4]["text"].endswith("<|image|>")
    [chunk] = sorted(out.glob("chunk_*.npz"))
    z = np.load(chunk)
    assert list(z["uri"]) == ["at://p/1", "at://p/2"] and list(z["n_images"]) == [2, 1]
    assert z["text"].shape == (2, 256) and z["interleaved"].dtype == np.float16 and z["image"].shape == (3, 256)
    assert list(z["has_text"]) == [True, False] and not z["text"][1].any()
    receipt = em.audit(out, before)
    assert receipt["ok"], receipt
    assert set(receipt["files_changed"]) == {"chunk_00000.npz", "log.jsonl"}
    assert b"\xff\xd8" not in (out / "log.jsonl").read_bytes()


def test_audit_catches_image_bytes_and_bad_suffix(tmp_path):
    before = em.listing(tmp_path)
    (tmp_path / "x.npz").write_bytes(b"zip" + synthetic("JPEG"))
    (tmp_path / "y.bin").write_bytes(b"\x00")
    r = em.audit(tmp_path, before)
    assert not r["ok"] and r["unexpected_suffix"] == ["y.bin"] and "y.bin" not in r["image_signature_found"]
    (tmp_path / "y.bin").unlink()
    (tmp_path / "x.npz").unlink()
    np.savez(tmp_path / "z.npz", leaked=np.frombuffer(synthetic("PNG"), np.uint8))
    assert em.audit(tmp_path, before)["image_signature_found"] == ["z.npz"]


def test_run_with_real_cdn_fetcher_and_fake_transport(tmp_path):
    jpg = synthetic("JPEG", noise=True)
    url = "https://cdn.bsky.app/img/feed_thumbnail/plain/did:plc:" + "a" * 24 + "/bafkreiaaaaaaaaaaaaaaaaaaaa"
    transport = lambda m, u, h, t: cdn.Response(200, {"content-type": "image/jpeg"}, jpg)  # noqa: E731
    f = cdn.CdnFetcher(transport, max_rps=50, sleep=lambda s: None)
    counts = em.run([{"uri": "at://q/1", "image_index": 0, "cid": "bafkreiaaaaaaaaaaaaaaaaaaaa", "url": url,
                      "text": "t", "alt": ""}], f, FakeBackend(), em.VectorSink(tmp_path))
    assert counts["posts_embedded"] == 1 and em.audit(tmp_path, {})["ok"]


@pytest.mark.skipif(os.environ.get("C4_EG2_SMOKE") != "1", reason="set C4_EG2_SMOKE=1 to load EG2 on MPS")
def test_real_eg2_vision_smoke(tmp_path):
    backend = em.EG2Backend()
    assert backend.model[0].processor.image_processor.max_soft_tokens == em.SOFT_TOKENS
    red, noise = em.decode_image(synthetic("PNG", (220, 20, 20), (256, 256))), em.decode_image(synthetic("JPEG", noise=True, size=(400, 300)))
    out = em.embed_posts(backend, [("a red square", [red]), ("", [noise, red])])
    for p in out:
        assert np.isfinite(p["interleaved"]).all() and abs(np.linalg.norm(p["interleaved"]) - 1) < 1e-2
        assert all(abs(np.linalg.norm(v) - 1) < 1e-2 for v in p["images"])
    assert out[0]["has_text"] and not out[1]["has_text"] and len(out[1]["images"]) == 2
    assert float(out[0]["text"] @ out[0]["interleaved"]) < 0.999  # pixels change the interleaved vector
    np.testing.assert_allclose(out[0]["images"][0], out[1]["images"][1], atol=2e-2)  # same red image, same vector
