"""S12m tests without the model or any network beyond 127.0.0.1: gate, fetcher, pipeline, job resume, audit."""
import builtins
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow.parquet as pq
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fetcher as F  # noqa: E402
import job  # noqa: E402
import mmlib as L  # noqa: E402
import pipeline as P  # noqa: E402
import projection  # noqa: E402
import synth  # noqa: E402

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
DID = synth.synthetic_did(1)
CID = synth.synthetic_cid(1)


def bundle(**kw):
    return L.GateBundle(kw.get("excl", set()), kw.get("lab", set()), NOW - timedelta(hours=1))


# --- gate -----------------------------------------------------------------------------------------------------

def test_gate_reasons_in_order():
    old = synth.tid_for(NOW - timedelta(days=5))
    assert L.gate_images(DID, old, [], [], bundle(), NOW) == ([], None)
    urls, r = L.gate_images(DID, old, [CID], [], bundle(), NOW)
    assert r is None and urls == [f"https://cdn.bsky.app/img/feed_thumbnail/plain/{DID}/{CID}"]
    assert L.gate_images(DID, synth.tid_for(NOW - timedelta(days=2)), [CID], [], bundle(), NOW)[1] == "too_young"
    assert L.gate_images(DID, synth.tid_for(NOW + timedelta(days=1)), [CID], [], bundle(), NOW)[1] == "bad_tid"
    assert L.gate_images(DID, "notatid", [CID], [], bundle(), NOW)[1] == "bad_tid"
    assert L.gate_images(DID, old, [CID], ["porn"], bundle(), NOW)[1] == "post_self_label"
    assert L.gate_images(DID, old, [CID], [], bundle(excl={DID}), NOW)[1] == "account_excluded"
    assert L.gate_images(DID, old, [CID], [], bundle(lab={L.post_uri(DID, old)}), NOW)[1] == "record_label"
    assert L.gate_images(DID, old, [CID] * 5, [], bundle(), NOW)[1] == "too_many_cids"
    assert L.gate_images(DID, old, ["../../x"], [], bundle(), NOW)[1] == "unsafe_did_or_cid"


def test_bundle_staleness_and_hash(tmp_path):
    synth.write_bundle(tmp_path, {DID}, set(), NOW - timedelta(hours=40))
    b = L.load_bundle(tmp_path)
    assert b.excluded_dids == {DID}
    assert b.stale(NOW, timedelta(hours=36)) and not b.stale(NOW, timedelta(hours=48))
    (tmp_path / "excluded_dids.txt").write_text("did:plc:tampered\n")
    with pytest.raises(ValueError):
        L.load_bundle(tmp_path)


def test_mm_text_layout():
    assert L.mm_text("P: ", "hello [image: a cat]", 2) == "P: hello [image: a cat] <|image|> <|image|>"
    assert L.mm_text("P: ", "  ", 1) == "P: <|image|>"


def test_pack_roundtrip_and_storage_rule():
    full = np.random.default_rng(0).normal(size=(3, 768)).astype(np.float32)
    v = L.to_stored(full)
    assert v.dtype == np.float16 and v.shape == (3, 256)
    assert np.allclose(np.linalg.norm(v.astype(np.float32), axis=1), 1, atol=1e-3)
    assert np.array_equal(L.unpack(L.pack(v)), v)


def test_dollar_guard():
    t = [0.0]
    g = L.DollarGuard(1.0, 0.5, prior_usd=0.25, t0=0.0, clock=lambda: t[0])
    t[0] = 3600
    assert abs(g.spent() - 0.75) < 1e-9
    assert g.allows(1800) and not g.allows(1801)


def test_g2_price_and_projection():
    assert 0.5 < L.g2_standard_8_spot_usd_per_hour() < 0.6
    p = projection.project(projection.PRIOR)
    r50 = next(x for x in p["rows"] if x["soft_tokens"] == 70 and x["rps"] == 50)
    assert r50["bound"] == "fetch" and 70 < r50["fetch_h"] < 72


# --- fetcher --------------------------------------------------------------------------------------------------

def test_fetcher_policy_refusals():
    with pytest.raises(ValueError):
        F.AsyncFetcher(rps=60)
    with pytest.raises(ValueError):
        F.AsyncFetcher(base="https://example.com")
    with pytest.raises(ValueError):
        F.AsyncFetcher(base="http://10.0.0.1:80", rehearsal=True)
    with pytest.raises(ValueError):
        F.AsyncFetcher(user_agent="bot/1.0")
    f = F.AsyncFetcher(rps=150, allow_above_etiquette=True)
    f.close()


def test_fetcher_loopback_rate_and_skips():
    with synth.ImageServer(latency_ms=20, fail404="7") as srv:
        f = F.AsyncFetcher(rps=40, max_in_flight=8, base=srv.base, rehearsal=True)
        cids = [synth.synthetic_cid(i) for i in range(60)]
        items = [(i, f"https://cdn.bsky.app/img/feed_thumbnail/plain/{DID}/{c}") for i, c in enumerate(cids)]
        items.append(("bad", "https://cdn.bsky.app/img/avatar/plain/x/y"))
        t0 = time.monotonic()
        res = f.fetch_sync(items)
        dt = time.monotonic() - t0
        f.close()
    assert res["bad"] == (None, {"status": None, "skip": "not_feed_thumbnail_url"})
    n404 = sum(c[-1] == "7" for c in cids)
    assert sum(b is None for k, (b, _) in res.items() if k != "bad") == n404
    assert all(lg["skip"] == "cdn_404" for k, (b, lg) in res.items() if k != "bad" and b is None)
    assert all(L.has_image_header(b) for b, _ in res.values() if b)
    assert dt >= 59 / 40 * 0.95           # the token bucket held the rate
    assert srv.requests == 60             # the refused URL was never requested


class _Fake429:
    def __call__(self, path, headers):
        return F.Resp(429, "text/plain", b"", "0")


def test_circuit_breaker_trips():
    f = F.AsyncFetcher(rps=50, max_in_flight=4, transport=_Fake429(), window=200)
    f._pause_until = 0
    items = [(i, f"https://cdn.bsky.app/img/feed_thumbnail/plain/{DID}/{synth.synthetic_cid(i)}") for i in range(250)]
    with pytest.raises(F.CircuitOpen):
        f.fetch_sync(items)
    f.close()


# --- pipeline + job -------------------------------------------------------------------------------------------

class FakeEnc:
    def __init__(self, *_a, **_k):
        self.soft_tokens, self.dev, self.params_m, self.param_dtypes = 70, "cpu", 438.8, ["torch.bfloat16"]
        self.prompt, self.mm_calls = "task: clustering | query: ", 0

    def _v(self, keys):
        return np.stack([np.random.default_rng(abs(hash(k)) % 2**32).normal(size=768) for k in keys]).astype(np.float32)

    def encode_text(self, texts, bs=256):
        return self._v(texts)

    def encode_mm(self, posts, bs=16):
        from PIL import Image
        assert all(isinstance(im, Image.Image) for _t, ims in posts for im in ims)
        self.mm_calls += 1
        return self._v([t + str(len(ims)) for t, ims in posts])

    def sync(self):
        pass

    def peak_mem_gb(self):
        return None


def _fixture_rows(n=300):
    texts = [f"post number {i}" if i % 17 else "" for i in range(n)]
    return synth.synthetic_posts(texts, NOW, image_share=0.5, seed=3)


def test_encode_unit_modalities_and_no_file_writes(monkeypatch):
    rows, excl, lab = _fixture_rows()
    u = P.prepare("t", rows, L.GateBundle(excl, lab, NOW), NOW)
    with synth.ImageServer(latency_ms=0) as srv:
        f = F.AsyncFetcher(rps=50, base=srv.base, rehearsal=True)
        fetched = f.fetch_sync(u.fetch_items())
        f.close()
    real_open = builtins.open

    def guarded(file, mode="r", *a, **k):
        assert not any(c in mode for c in "wax+"), f"file write during encode: {file}"
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded)
    cols, c = P.encode_unit(u, fetched, FakeEnc(), mm_bs=4)
    monkeypatch.setattr(builtins, "open", real_open)
    assert fetched == {}                                  # every byte reference dropped
    mods = cols["modality"]
    for i, m in enumerate(mods):
        has_t = bool(u.texts[i].strip())
        if not u.n_cids[i]:
            assert m == (L.TEXT if has_t else L.NONE)
        elif m in (L.TEXT_IMAGE, L.IMAGE_ONLY):
            assert u.urls[i] and cols["n_images"][i] == len(u.urls[i]) and cols["image_skip"][i] is None
            assert (cols["vec_text"][i] is not None) == has_t
        else:
            assert m in (L.TEXT_FALLBACK, L.NONE) and cols["image_skip"][i]
    assert c["modality_text+image"] > 0 and c["modality_text_fallback"] > 0
    assert any(s == "cdn_404" for s in cols["image_skip"])
    assert any(s in ("too_young", "post_self_label", "record_label") for s in cols["image_skip"])


def _args(tmp, srv, **kw):
    a = dict(model_dir="x", work=str(tmp / "work"), run_id="t-run", out=str(tmp / "bucket" / "vectors"),
             gate_bundle=str(tmp / "gate"), rps=50.0, allow_above_etiquette=False, max_in_flight=16, rehearsal=True,
             cdn_base=srv.base, now=NOW.isoformat(), cap_usd=0.0, usd_per_hour=0.53, vm_start_epoch=None,
             max_label_lag_h=36.0, text_bs=64, mm_bs=8, device="cpu", mps_gb=1.0, posts=str(tmp / "in" / "posts"),
             sidecar=str(tmp / "in" / "sidecar"), soft_tokens=70, unit_rows=100, max_units=0)
    a.update(kw)
    import argparse
    return argparse.Namespace(**a)


def test_job_run_resume_and_receipt(tmp_path):
    rows, excl, lab = _fixture_rows(450)
    synth.write_shards(rows, tmp_path / "in", n_shards=2)
    synth.write_bundle(tmp_path / "gate", excl, lab, NOW - timedelta(hours=2))
    with synth.ImageServer(latency_ms=0) as srv:
        assert job.cmd_run(_args(tmp_path, srv, max_units=2), make_encoder=FakeEnc) == 0
        led = sorted((tmp_path / "bucket/vectors/_ledger").glob("u-*.json"))
        assert len(led) == 2
        assert job.cmd_run(_args(tmp_path, srv), make_encoder=FakeEnc) == 0
    units = sorted((tmp_path / "bucket/vectors/units").glob("*.parquet"))
    assert len(units) == 6                       # 2 shards x 225 rows / 100 per unit -> 3 units each
    t = pq.read_table(units[0])
    assert t.schema.equals(L.output_schema())
    total = sum(pq.read_table(u).num_rows for u in units)
    assert total == 450
    recs = [json.loads(p.read_text()) for p in (tmp_path / "bucket/vectors/receipt").glob("*.json")]
    assert sorted(r["units_this_vm"] for r in recs) == [2, 4]       # one receipt per VM boot
    assert all(r["audit"]["ok"] for r in recs)
    assert len(list((tmp_path / "bucket/vectors/_ledger").glob("vm-*.json"))) == 2
    assert not list((tmp_path / "work" / "out").glob("*.parquet"))     # local vector copies removed
    run = json.loads((tmp_path / "bucket/vectors/_run.json").read_text())
    assert run["config"]["soft_tokens"] == 70


def test_job_refuses_stale_labels_and_gcs_rehearsal(tmp_path):
    rows, excl, lab = _fixture_rows(50)
    synth.write_shards(rows, tmp_path / "in", n_shards=1)
    synth.write_bundle(tmp_path / "gate", excl, lab, NOW - timedelta(hours=50))
    with synth.ImageServer(latency_ms=0) as srv:
        assert job.cmd_run(_args(tmp_path, srv), make_encoder=FakeEnc) == job.EXIT_STALE_LABELS
        with pytest.raises(SystemExit):
            job.cmd_run(_args(tmp_path, srv, out="gs://catbird-436600-plan-60d/vectors/s12mm/x"), make_encoder=FakeEnc)
        with pytest.raises(SystemExit):
            job.check_out(_args(tmp_path, srv, rehearsal=False, out="gs://catbird-436600-plan-30d/inputs/x"))


def test_audit_catches_an_image_file(tmp_path):
    import audit
    m = audit.start_marker(tmp_path)
    (tmp_path / "leak.bin").write_bytes(synth.synthetic_jpeg(1))
    (tmp_path / "embedded.log").write_bytes(b"log line " + synth.synthetic_jpeg(2)[:64])
    rep = audit.finish(tmp_path, m)
    assert not rep["ok"]
    assert any("leak.bin" in p for p in rep["scan"]["image_header"])
    assert any("embedded.log" in p for p in rep["scan"]["image_signature"])



def test_audit_input_dirs_header_only(tmp_path):
    """A prefetched input tar part with a chance JPEG marker passes; a real image file there still fails."""
    import audit
    inputs = tmp_path / "batches"
    inputs.mkdir()
    m = audit.start_marker(tmp_path / "work")
    (inputs / "b003.tar.part.0001").write_bytes(b"parquet bytes \xff\xd8\xff more bytes")
    rep = audit.finish(tmp_path / "work", m, input_dirs=[str(inputs)])
    assert any("b003.tar.part.0001" in f for f in rep["listing_head"]) and rep["scan"]["ok"]
    assert not audit.finish(tmp_path / "work", m)["scan"]["ok"]   # without the input dirs: the full scan trips
    (inputs / "leak.jpg").write_bytes(synth.synthetic_jpeg(3))
    rep = audit.finish(tmp_path / "work", m, input_dirs=[str(inputs)])
    assert not rep["ok"]

# --- launcher -------------------------------------------------------------------------------------------------

def test_launcher_print_only_spot_l4(tmp_path):
    fake = tmp_path / "fakebin"
    fake.mkdir()
    (fake / "gcloud").write_text("#!/bin/sh\necho CALLED >> \"$(dirname \"$0\")/calls\"\nexit 1\n")
    (fake / "gcloud").chmod(0o755)
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "SESSIONS": str(tmp_path / "s.tsv"),
           "POSTS": "gs://catbird-436600-plan-60d/x/O8", "SIDECAR": "gs://catbird-436600-plan-60d/x/cids"}
    for mode, maxrun in (("bench", "90m"), ("s12", "112h"), ("resume", "112h")):
        r = subprocess.run(["bash", str(HERE / "launch_s12mm.sh"), mode, "t-run"], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        o = r.stdout
        for s in ("--provisioning-model=SPOT", "--instance-termination-action=DELETE", f"--max-run-duration='{maxrun}'",
                  "--machine-type='g2-standard-8'", "plan=whole-network-signals", "dry run: nothing was executed",
                  "deeplearning-platform-release", "--maintenance-policy=TERMINATE"):
            assert s in o, (mode, s)
    assert not (fake / "calls").exists()
    r = subprocess.run(["bash", str(HERE / "launch_s12mm.sh"), "delete", "nope"], capture_output=True, text=True, env=env)
    assert r.returncode != 0


def test_plan60_writes_need_custom_time_and_prefix_listing(tmp_path):
    import stores as S
    with pytest.raises(RuntimeError):
        S.write_text(S.GcsStore("gs://catbird-436600-plan-60d/vectors/s12mm/x"), "_run.json", "{}", tmp_path)
    assert not S.allowed_write("gs://catbird-436600-plan-60d/O8/x")
    assert S.allowed_write("gs://catbird-436600-plan-60d/vectors/s12mm/r1")
    st = S.LocalStore(tmp_path / "b")
    for k in ("_ledger/u-a.json", "_ledger/vm-1.json", "units/u-a.parquet"):
        st.write_text(k, "x")
    assert S.list_keys(st, "_ledger/u-", ".json") == ["_ledger/u-a.json"]
