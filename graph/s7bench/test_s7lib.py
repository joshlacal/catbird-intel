"""Tests for the pure S7 pieces (s7lib) and a local run of roundtrip_job.py.

  ~/catbird-lake-local/s4-venv/bin/python -I -m pytest -q graph/s7bench/test_s7lib.py
"""
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s7lib  # noqa: E402

C5_USAGE = """
[colab] A new version of Colab CLI is available: 0.7.5 (current: 0.7.4)

Current balance: 4996.34 compute units
Usage rate: 1.71/hr
Active assignments: 1
"""


def test_hash64_matches_s4_rule():
    def s4(salt, keys):  # graph/s4_pilot/prep_corpus.py hash64
        return [int.from_bytes(hashlib.blake2b(f"{salt}|{k}".encode(), digest_size=8).digest(), "big") for k in keys]
    keys = ["did:plc:a/3l", "did:plc:b/3m", "x"]
    assert s7lib.hash64("s4-cap32", keys) == s4("s4-cap32", keys)


def test_draw_smallest_is_deterministic_and_salted():
    keys = [f"k{i}" for i in range(1000)]
    a = s7lib.draw_smallest(keys, 50, "s")
    assert a == s7lib.draw_smallest(keys, 50, "s") and len(set(a)) == 50
    assert a != s7lib.draw_smallest(keys, 50, "t")
    h = s7lib.hash64("s", keys)
    assert max(h[i] for i in a) < min(h[i] for i in range(1000) if i not in set(a))


def test_draw_one_per_group():
    groups = ["p1", "p1", "p1", "p2", "p2", "p3"]
    keys = ["a", "b", "c", "d", "e", "f"]
    got = s7lib.draw_one_per_group(groups, keys, "g", "i", exclude=frozenset({"f"}))
    assert sorted(g for g, _ in got) == ["p1", "p2"]  # p3's only post is excluded
    hi = dict(zip(keys, s7lib.hash64("i", keys)))
    for g, i in got:
        assert hi[keys[i]] == min(hi[k] for gg, k in zip(groups, keys) if gg == g)
    hg = s7lib.hash64("g", [g for g, _ in got])
    assert hg == sorted(hg)


def test_slab_text_matches_o8_sql():
    # sql_p2.py: concat(text, arrayStringConcat(arrayMap(x -> concat(' [image: ', x, ']'), arrayFilter(x -> x != '', alts)), ''))
    assert s7lib.slab_text("hi", ["a cat", "", "a dog"]) == "hi [image: a cat] [image: a dog]"
    assert s7lib.slab_text("hi", []) == "hi"
    assert s7lib.slab_text("", ["x"]) == " [image: x]"


def test_alts_from_record():
    img = {"embed": {"$type": "app.bsky.embed.images", "images": [{"alt": "one"}, {"alt": ""}, {}]}}
    assert s7lib.alts_from_record(img) == ["one", "", ""]
    rwm = {"embed": {"$type": "app.bsky.embed.recordWithMedia", "record": {},
                     "media": {"$type": "app.bsky.embed.images", "images": [{"alt": "q"}]}}}
    assert s7lib.alts_from_record(rwm) == ["q"]
    vid = {"embed": {"$type": "app.bsky.embed.video", "alt": "clip"}}
    assert s7lib.alts_from_record(vid) == ["clip"]
    assert s7lib.alts_from_record({"embed": {"$type": "app.bsky.embed.external"}}) == []
    assert s7lib.alts_from_record({}) == [] and s7lib.alts_from_record(None) == []


@pytest.mark.parametrize("raw,leak", [
    ("GET https://8080-gpu-l4-s-abc.prod.colab.dev/api/contents?authuser=0&token=SeCrEtTokenValue123 500", "SeCrEtTokenValue123"),
    ("Authorization: Bearer ya29.a0AfB_byC-abcdefghijklmnop", "ya29.a0AfB_byC"),
    ('{"access_token": "ya29.zzzzzzzzzzzzzzzzzzzzzz", "x": 1}', "zzzzzzzzzz"),
    ('{"refresh_token": "1//0gAbCdEfGhIjKlMnOpQrStUvWxYz"}', "AbCdEfGhIjKlMnOp"),
    ("Enter authorization code: 4/0AeanS0abcdefghijklmnopqrstuvw", "abcdefghijklmnopqrstuvw"),
    ("x-colab-token=AAAAbbbbCCCCdddd done", "AAAAbbbbCCCCdddd"),
])
def test_redact_removes_secrets(raw, leak):
    out = s7lib.redact(raw)
    assert leak not in out and "<redacted>" in out


def test_redact_keeps_ordinary_text():
    s = "[s7job] RESULT slab 2500.0 posts/s 47.1 tok/post"
    assert s7lib.redact(s) == s
    assert s7lib.redact("see https://huggingface.co/google/embeddinggemma-2 ok") == "see https://huggingface.co/google/embeddinggemma-2 ok"


def test_gpu_rate():
    assert s7lib.gpu_rate("NVIDIA L4") == ("L4", 1.71)
    assert s7lib.gpu_rate("NVIDIA A100-SXM4-40GB") == ("A100", 5.4)
    assert s7lib.gpu_rate("Tesla T4") == ("T4", 1.19)
    assert s7lib.gpu_rate("NVIDIA H100 80GB HBM3") == (None, None)  # unknown rate: the job stops unless S7_RATE


def test_units_per_10m_arithmetic():
    u, arith = s7lib.units_per_10m(2500.0, 1.71)
    assert u == pytest.approx(1e7 / 2500 / 3600 * 1.71)
    assert "1e7 / 2,500 posts/s = 4,000 s = 1.111 h x 1.71 units/h = 1.90 units" == arith
    u15, a15 = s7lib.units_per_10m(2500.0, 1.71, 1.5)
    assert u15 == pytest.approx(u * 1.5) and "x 1.5" in a15


def test_budget_seconds():
    assert s7lib.budget_seconds(3.0, 1.71) == pytest.approx(3.0 / 1.71 * 3600)
    assert s7lib.budget_seconds(1.0, 5.4, spent_units=2.0) == 0.0


def test_unit_guard_with_fake_clock():
    t = [1000.0]
    g = s7lib.UnitGuard(1.0, 3.6, preamble_min=10, clock=lambda: t[0])  # 3.6 units/h = 0.001 unit/s
    assert g.spent() == pytest.approx(0.6)
    assert g.allows(399) and not g.allows(401)
    t[0] += 400
    assert g.spent() == pytest.approx(1.0) and not g.allows(1)


def test_parse_usage():
    assert s7lib.parse_usage(C5_USAGE) == {"balance": 4996.34, "rate_per_hour": 1.71, "active": 1}
    assert s7lib.parse_usage("nothing") == {"balance": None, "rate_per_hour": None, "active": None}


def test_cosine_stats():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, 768))
    s = s7lib.cosine_stats(a, a * 3.0)
    assert s["median"] == pytest.approx(1.0) and s["below_0_99"] == 0
    b = a.copy()
    b[:, 256:] = rng.normal(size=(50, 512))  # differs only beyond 256-d
    assert s7lib.cosine_stats(a, b, 256)["min"] == pytest.approx(1.0)
    assert s7lib.cosine_stats(a, b)["median"] < 0.9
    assert s7lib.cosine_stats(a, -a)["below_0_99"] == 50


def test_rt_allowed():
    ok = s7lib.rt_allowed
    assert ok("catbird-436600-plan-30d", "colab-test/synthetic-2e9-random-20261028.bin")
    assert not ok("catbird-436600-plan-60d", "colab-test/x")
    assert not ok("catbird-436600-plan-30d", "intermediates/x")
    assert not ok("catbird-436600-plan-30d", "colab-test/../vectors/x")
    assert not ok("catbird-436600-plan-30d", "colab-test/")
    assert not ok("catbird-436600-plan-30d", "colab-test//x")


def test_scopes_from_tokeninfo():
    info = {"scope": "openid https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/drive"}
    assert s7lib.scopes_from_tokeninfo(info)[0] == "https://www.googleapis.com/auth/cloud-platform"
    assert s7lib.scopes_from_tokeninfo({}) == []


def test_randbytes(tmp_path):
    buf = io.BytesIO()
    h = s7lib.randbytes_to(buf, 3 * 1024 + 7, tmp_path / "sha", block=1024)
    assert len(buf.getvalue()) == 3 * 1024 + 7
    assert hashlib.sha256(buf.getvalue()).hexdigest() == h == (tmp_path / "sha").read_text().strip()


def run_rt(tmp_path, obj, sha, n):
    env = dict(os.environ, S7_RT_LOCAL_DIR=str(tmp_path / "bucket"), S7_ROOT=str(HERE), S7_OUT=str(tmp_path / "out"),
               S7_RT_OBJECT=obj, S7_RT_SHA256=sha, S7_RT_BYTES=str(n), S7_SESSION="test")
    p = subprocess.run([sys.executable, "-I", str(HERE / "roundtrip_job.py")], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return json.load(open(tmp_path / "out" / "roundtrip.json")), p.stdout


def test_roundtrip_job_local(tmp_path):
    (tmp_path / "bucket/colab-test").mkdir(parents=True)
    data = os.urandom(5 * (1 << 20) + 3)
    (tmp_path / "bucket/colab-test/obj.bin").write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    rec, out = run_rt(tmp_path, "colab-test/obj.bin", sha, len(data))
    assert rec["ok"] and rec["ok_read"] and rec["ok_write"] and "ROUNDTRIP_OK" in out
    assert rec["write_object"].startswith("colab-test/roundtrip-test-")
    assert json.loads((tmp_path / "bucket" / rec["write_object"]).read_text())["read_sha256"] == sha
    bad, out = run_rt(tmp_path, "colab-test/obj.bin", "0" * 64, len(data))
    assert not bad["ok"] and "ROUNDTRIP_FAILED" in out
    outside, out = run_rt(tmp_path, "intermediates/obj.bin", sha, len(data))
    assert not outside["ok"] and "refusing" in outside["error"]
