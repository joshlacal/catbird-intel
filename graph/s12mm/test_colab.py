"""S12m on Colab: store abstraction, Mac ledger, splitter, CDN ramp, Colab audit mounts, and full driver
rehearsals against a FAKE colab CLI (a run, a session drop, a resume, a crashed driver, a stale gate, the bench).

Nothing leaves the Mac: the fake CLI keeps each "session" as a directory, runs `exec -f` locally with the
session's /content mapped into it, and serves thumbnails from 127.0.0.1 (synth.ImageServer). The job runs with
synth.FakeEncoder (no model), so a whole multi-session run takes seconds.
"""
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit  # noqa: E402
import colab_ledger as CL  # noqa: E402
import fetcher as F  # noqa: E402
import mmlib as L  # noqa: E402
import projection  # noqa: E402
import ramp  # noqa: E402
import split_units  # noqa: E402
import stores as S  # noqa: E402
import synth  # noqa: E402

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
NOW_S = "2026-11-01T12:00:00Z"


# --- store abstraction ----------------------------------------------------------------------------------------

def test_local_dir_store_interface(tmp_path):
    st = S.open_store(str(tmp_path / "root"))
    assert isinstance(st, S.LocalDirStore) and S.LocalStore is S.LocalDirStore
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    st.put(f, "units/u-a.parquet")
    st.write_text("_ledger/u-a.json", '{"a": 1}')
    st.write_text("_ledger/vm-1.json", "{}")
    assert st.exists("units/u-a.parquet") and not st.exists("units/u-b.parquet")
    assert st.read_text("_ledger/u-a.json") == '{"a": 1}'
    st.get("units/u-a.parquet", tmp_path / "back" / "y.bin")
    assert (tmp_path / "back/y.bin").read_bytes() == b"abc"
    assert S.list_keys(st, "_ledger/u-", ".json") == ["_ledger/u-a.json"]
    assert st.list("", ".json") == ["_ledger/u-a.json", "_ledger/vm-1.json"]
    assert st.list("nothing/", "") == []
    st.write_text("_ledger/u-a.json", "v2")                        # overwrite is atomic and complete
    assert st.read_text("_ledger/u-a.json") == "v2"
    (tmp_path / "root/_ledger/.tmp-abc").write_text("half")         # an interrupted write is never listed
    assert ".tmp" not in "".join(st.list("_ledger/"))
    for bad in ("../escape", "/abs/key", "a/../../b", ""):
        with pytest.raises(ValueError):
            st.write_text(bad, "x")
    assert st.ops["put"] >= 3 and st.ops["list"] >= 3


def test_store_write_failure_leaves_no_partial(tmp_path, monkeypatch):
    st = S.LocalDirStore(tmp_path)
    st.write_text("k.json", "old")

    class Boom(Exception):
        pass

    def bad_write(fh):
        fh.write(b"partial")
        raise Boom()
    with pytest.raises(Boom):
        st._atomic(st._path("k.json"), bad_write)
    assert st.read_text("k.json") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["k.json"]


def test_gcs_store_is_lazy_and_guarded(tmp_path):
    assert S.open_store(str(tmp_path)).custom_time is None
    g = S.GcsStore("gs://catbird-436600-plan-60d/vectors/s12mm/x")       # built on first use, no gcloud call
    assert hasattr(g, "list") and g.url.endswith("/x")
    with pytest.raises(RuntimeError):
        S.write_text(g, "_run.json", "{}", tmp_path)
    assert S.allowed_write("/content/s12mm/out/b1") and not S.allowed_write("gs://catbird-436600-plan-60d/O8/x")


# --- ledger ---------------------------------------------------------------------------------------------------

def _parquet(path, n):
    t = pa.table({"uri": [f"at://did:plc:a/app.bsky.feed.post/{i}" for i in range(n)]})
    pq.write_table(t, path)
    return CL.sha256_file(path)


def test_ledger_marks_only_verified_units(tmp_path):
    led = CL.RunLedger(tmp_path / "run")
    man = {"units": [{"unit": "b0000-0000", "rows": 5, "sha256": "in0"}, {"unit": "b0000-0001", "rows": 7, "sha256": "in1"}]}
    assert [u["unit"] for u in led.todo(man)] == ["b0000-0000", "b0000-0001"]
    f = tmp_path / "u.parquet"
    h = _parquet(f, 5)
    assert led.verify_unit(man["units"][0], {"sha256": "0" * 64, "rows": 5}, f) == (False, f"sha256 {h[:12]} != VM 000000000000")
    assert led.verify_unit(man["units"][0], {"sha256": h, "rows": 4}, f)[0] is False
    assert led.verify_unit(man["units"][0], {"sha256": h, "rows": 5}, f, audit=lambda p: {"ok": False, "rows": 5})[0] is False
    assert led.verify_unit(man["units"][0], {"sha256": h, "rows": 5}, tmp_path / "missing.parquet") == (False, "missing")
    ok, why = led.verify_unit(man["units"][0], {"sha256": h, "rows": 5}, f, audit=lambda p: {"ok": True, "rows": 5})
    assert ok and why is None and not f.exists()
    assert (led.dir / "vectors/units/u-b0000-0000-00000.parquet").exists()
    assert led.is_done("b0000-0000") and [u["unit"] for u in led.todo(man)] == ["b0000-0001"]
    assert [u["unit"] for u in led.todo(man, exclude={"b0000-0001"})] == []
    (led.dir / "ledger" / ".tmp-zz").write_text("{")            # a torn write never counts
    (led.dir / "ledger" / "junk.json").write_text("{not json")
    assert set(led.done()) == {"b0000-0000"}
    with pytest.raises(ValueError):
        led.mark("x", {"rows": 1})


def test_ledger_sessions_and_spend(tmp_path):
    clock = [1000.0]
    led = CL.RunLedger(tmp_path / "run", clock=lambda: clock[0])
    led.record("s1", "created", "L4")
    led.spend_update("s1", rate=1.71, t_created=0.0, t_last_seen=3600.0, balance_before=4990.0)
    led.record("s1", "stopped")
    led.spend_update("s1", t_stopped=3600.0)
    led.record("s2", "created", "L4")
    led.spend_update("s2", rate=1.71, t_created=7200.0, t_last_seen=7200.0 + 1800)
    led.record("s3", "created", "L4")
    led.spend_update("s3", rate=1.71, t_created=10000.0, t_last_seen=10000.0)
    assert led.open_sessions() == ["s2", "s3"]
    led.record("s2", "ended(unobserved)")
    led.spend_update("s2", closed_unobserved=True)
    clock[0] = 10000.0 + 1800
    guard, det = led.spent_units(balance_now=4989.0)
    want = 1.71 + (1800 + CL.GRACE_S) / 3600 * 1.71 + 0.5 * 1.71   # stopped + unobserved(last seen + grace) + live
    assert abs(det["wall_estimate_units"] - want) < 1e-3 and guard == pytest.approx(want, abs=1e-3)
    guard2, det2 = led.spent_units(balance_now=4980.0)            # the account balance fell more: it wins
    assert guard2 == pytest.approx(10.0) and det2["balance_drop_units"] == 10.0
    assert led.open_sessions() == ["s3"]
    assert CL.parse_usage("Current balance: 4993.86 compute units\nUsage rate: 1.71/hr\nActive assignments: 1") == \
        {"balance": 4993.86, "rate_per_hour": 1.71, "active": 1}


# --- splitter -------------------------------------------------------------------------------------------------

def make_o8_and_sidecar(d, n=240, seed=5, extra_noise=True):
    texts = [f"slab post {i} about things" if i % 11 else "" for i in range(n)]
    rows, excl, lab = synth.synthetic_posts(texts, NOW, image_share=0.4, seed=seed)
    o8 = d / "o8"
    side = d / "sidecar"
    o8.mkdir(parents=True)
    side.mkdir(parents=True)
    half = n // 2
    for j, part in enumerate((rows[:half], rows[half:])):
        pq.write_table(pa.table({"p": [r["p"] for r in part], "uri": [r["uri"] for r in part],
                                 "created": ["2026-10-01T00:00:00Z"] * len(part), "text": [r["text"] for r in part],
                                 "langs": [["en"]] * len(part), "image_count": [len(r["image_cids"]) for r in part],
                                 "self_labels": [r["self_labels"] for r in part]}), o8 / f"j{j:02d}.parquet")
    sr = []
    for r in rows:
        for i, c in reversed(list(enumerate(r["image_cids"]))):        # out of order on purpose
            sr.append({"did": r["did"], "rkey": r["rkey"], "kind": "image", "image_index": i, "cid": c,
                       "source": "images" if i % 2 == 0 else "recordWithMedia"})
        if extra_noise:
            sr.append({"did": r["did"], "rkey": r["rkey"], "kind": "external_thumb", "image_index": 0,
                       "cid": synth.synthetic_cid(99999), "source": "external"})
    sr.append({"did": rows[0]["did"], "rkey": rows[0]["rkey"], "kind": "image", "image_index": 9,
               "cid": synth.synthetic_cid(88888), "source": "gallery"})          # gallery: never fetched
    sr.append({"did": synth.synthetic_did(777), "rkey": rows[0]["rkey"], "kind": "image", "image_index": 0,
               "cid": synth.synthetic_cid(77777), "source": "images"})          # not a slab post
    sch = pa.schema([("did", pa.string()), ("rkey", pa.string()), ("post_created", pa.timestamp("us", tz="UTC")),
                     ("rev", pa.string()), ("kind", pa.string()), ("image_index", pa.int16()), ("cid", pa.string()),
                     ("mime", pa.string()), ("alt", pa.string()), ("aspect_w", pa.int32()), ("aspect_h", pa.int32()),
                     ("source", pa.string())])
    full = [{**x, "post_created": None, "rev": "r1", "mime": "image/jpeg", "alt": "", "aspect_w": 1, "aspect_h": 1} for x in sr]
    pq.write_table(pa.Table.from_pylist(full[: len(full) // 2], schema=sch), side / "image_cids-p00.parquet")
    pq.write_table(pa.Table.from_pylist(full[len(full) // 2:], schema=sch), side / "image_cids-p01.parquet")
    return rows, excl, lab


def test_split_units_join_and_format(tmp_path):
    rows, _e, _l = make_o8_and_sidecar(tmp_path)
    bad = tmp_path / "o8" / "j09.parquet"     # one bad uri and one duplicate of an existing post
    pq.write_table(pa.table({"p": [1, rows[3]["p"]], "uri": ["https://not-a-post", rows[3]["uri"]], "text": ["x", "dup"],
                             "self_labels": [[], []], "image_count": [0, 0]}), bad)
    out = tmp_path / "units"
    split_units.main([str(out), "--o8", str(tmp_path / "o8"), "--sidecar", str(tmp_path / "sidecar"),
                      "--unit-rows", "50", "--buckets", "4", "--batch-rows", "37"])
    man = json.loads((out / "manifest.json").read_text())
    rec = json.loads((out / "split_receipt.json").read_text())
    assert man["format"] == "s12mm-colab-units/1" and man["unit_rows"] == 50
    assert man["totals"]["posts"] == len(rows) and rec["counts"]["o8_bad_uri_dropped"] == 1
    assert rec["counts"]["o8_duplicate_uris_dropped"] == 1
    assert not (out / "_tmp").exists()
    got = {}
    for u in man["units"]:
        f = out / u["file"]
        assert CL.sha256_file(f) == u["sha256"] and 0 < u["rows"] <= 50
        t = pq.read_table(f)
        assert t.schema.equals(split_units.unit_schema())
        ps = t["p"].to_pylist()
        assert ps == sorted(ps)
        for r in t.to_pylist():
            got[r["uri"]] = r
    assert len(got) == len(rows)
    for r in rows:
        g = got[r["uri"]]
        assert (g["image_cids"] or []) == r["image_cids"]                # ordered, no gallery/thumb/foreign CIDs
        assert g["text"] == r["text"] and g["p"] == r["p"] and (g["self_labels"] or []) == r["self_labels"]
    assert man["totals"]["images"] == sum(len(r["image_cids"]) for r in rows)
    assert (out / "record_dids.txt").read_text().split() == sorted({r["did"] for r in rows if r["image_cids"]})
    with pytest.raises(SystemExit):           # units are immutable once written
        split_units.main([str(out), "--o8", str(tmp_path / "o8"), "--sidecar", str(tmp_path / "sidecar")])


# --- ramp + fetcher -------------------------------------------------------------------------------------------

def test_ramp_plan_and_verdict():
    assert ramp.plan(100_000) == [(25, 2250, 90.0), (50, 4500, 90.0), (100, 9000, 90.0), (200, 18000, 90.0)]
    p = ramp.plan(21_377)                       # the real S7g pool: every step shortened to ~57 s, 200 kept
    assert [r for r, _n, _s in p] == [25, 50, 100, 200] and all(56 <= s <= 57.1 for _r, _n, s in p)
    assert sum(n for _r, n, _s in p) <= 21_377
    assert ramp.plan(9_000) == [(25, 750, 30.0), (50, 1500, 30.0), (100, 3000, 30.0)]   # 30 s floor: 200 dropped
    base = {"requests": 1000, "statuses": {"200": 1000}, "latency_ms_p95": 100.0}
    assert ramp.verdict(base, 100.0) is None
    assert ramp.verdict({**base, "statuses": {"200": 999, "429": 1}}, 100.0) == "429"
    assert ramp.verdict({**base, "statuses": {"200": 985, "503": 15}}, 100.0).startswith("5xx")
    assert ramp.verdict({**base, "statuses": {"200": 995, "503": 5}}, 100.0) is None      # 0.5% is fine
    assert ramp.verdict({**base, "latency_ms_p95": 301.0}, 100.0).startswith("p95")
    assert ramp.in_flight_for(200) >= 200 and ramp.in_flight_for(25) == 38


def test_ramp_stops_at_first_429_and_counts_cache():
    did = synth.synthetic_did(3)
    items = [(i, f"https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{synth.synthetic_cid(i)}") for i in range(400)]
    with synth.ImageServer(latency_ms=1, fail404="", limit_rps=30) as srv:
        def make(rate, in_flight, max_5xx):
            return F.AsyncFetcher(rps=rate, max_in_flight=in_flight, base=srv.base, rehearsal=True,
                                  discard_bodies=True, stop_on_429=True, max_5xx=max_5xx)
        rep = ramp.run(items, make, steps=(10, 20, 40, 80), step_s=2.0, min_step_s=1.0, log=lambda *a: None)
        n_req = srv.requests
    assert [s["verdict"] for s in rep["steps"]][:2] == ["clean", "clean"]
    assert rep["stopped"]["reason"] == "429" and rep["stopped"]["at_target_rps"] == 40
    assert rep["max_clean_target_rps"] == 20 and rep["achieved_rps_at_max_clean"] > 10
    s0 = rep["steps"][0]
    assert s0["cdn_cache"] == {"MISS": 20} and s0["miss_share"] == 1.0 and s0["hit_share"] == 0.0
    assert n_req < 20 + 40 + 80 + 20          # the 429 step was cut short; the 80 step never ran


def test_fetcher_discards_bodies_and_records_cache():
    did = synth.synthetic_did(4)
    url = f"https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{synth.synthetic_cid(5)}"
    with synth.ImageServer(latency_ms=0, fail404="") as srv:
        f = F.AsyncFetcher(rps=50, base=srv.base, rehearsal=True)
        res = f.fetch_sync([(0, url), (1, url)])
        f.close()
        g = F.AsyncFetcher(rps=50, base=srv.base, rehearsal=True, discard_bodies=True)
        res2 = g.fetch_sync([(0, url)])
        g.close()
    assert res[0][0] and res[1][0] and f.stats.cache == {"MISS": 1, "HIT": 1}
    assert res2[0][0] is None and res2[0][1]["discarded"] and g.stats.ok == 1 and g.stats.bytes > 0


# --- Colab audit + projection ---------------------------------------------------------------------------------

COLAB_MOUNTS = """overlay / overlay rw,relatime,lowerdir=/x 0 0
proc /proc proc rw 0 0
tmpfs /dev tmpfs rw 0 0
devpts /dev/pts devpts rw 0 0
shm /dev/shm tmpfs rw 0 0
sysfs /sys sysfs ro 0 0
cgroup /sys/fs/cgroup cgroup2 ro 0 0
/dev/root /usr/lib64-nvidia ext2 ro 0 0
/dev/sda1 /etc/hosts ext4 rw 0 0
/dev/sda1 /kaggle/input ext4 rw 0 0
tmpfs /run tmpfs rw 0 0
drive /content/drive fuse.drive rw 0 0
/dev/sdb1 /content ext4 rw 0 0
"""


def test_audit_mounts_cover_colab_container():
    walk, fuse = audit.scan_mounts(COLAB_MOUNTS)
    assert "/" in walk and "/content" in walk and "/dev/shm" in walk and "/etc/hosts" in walk
    assert not any(m in walk for m in ("/proc", "/dev", "/dev/pts", "/sys", "/sys/fs/cgroup", "/run"))
    assert fuse == ["/content/drive"]


def test_colab_costs_block():
    res = {"text": {"posts_per_s": 2000.0}, "mm70": {"posts_per_s": 120.0}, "mm280": {"posts_per_s": 30.0},
           "image_selection": {"gate_counts": {"posts_with_cids": 100, "gate_pass": 80}},
           "cdn_ramp": {"achieved_rps_at_max_clean": 157.0}}
    c = projection.colab_costs(res)
    assert c["gate_pass"] == 0.8
    l4 = next(r for r in c["per_10m_posts"] if r["gpu"] == "L4" and r["rps"] == "100")
    img = 1e7 * 0.23 * 0.8
    assert l4["fetch_h"] == round(img * 1.232 / 100 / 3600, 2) and l4["bound"] == "fetch"
    assert l4["units"] == round(l4["wall_h"] * 1.71, 1) and l4["usd"] == round(l4["units"] * 0.0999, 2)
    a100 = next(r for r in c["per_10m_posts"] if r["gpu"] == "A100" and r["rps"] == "100")
    assert a100["units"] > l4["units"]            # fetch-bound: the A100 only costs more
    assert {r["rps"] for r in c["full_slab"]} == {"etiquette_50", "100", "200", "ramp_max_clean_achieved"}


# --- driver rehearsals against a fake colab CLI ---------------------------------------------------------------

FAKE_COLAB = r'''#!PYTHON
"""Fake Colab CLI: sessions are directories under $FAKE_DIR/vm/<name>; every call is logged to calls.jsonl."""
import fcntl, json, os, pathlib, shutil, subprocess, sys
D = pathlib.Path(os.environ["FAKE_DIR"])
argv = sys.argv[1:]
if argv[:1] == ["--config"]:
    argv = argv[2:]
with open(D / "calls.jsonl", "a") as f:
    f.write(json.dumps(argv) + "\n")
cmd, rest = argv[0], argv[1:]
def opt(name, default=None):
    return rest[rest.index(name) + 1] if name in rest else default
sess = opt("-s")
vm = D / "vm" / (sess or "_")
active = lambda: sorted(p.name for p in (D / "vm").iterdir()) if (D / "vm").exists() else []
def remote(p):
    assert p.startswith("/content/"), p
    return vm / p.lstrip("/")
if cmd == "usage":
    n = len(active())
    calls = sum(1 for _ in open(D / "calls.jsonl"))
    print(f"Current balance: {4990 - 0.001 * calls:.3f} compute units\nUsage rate: {1.71 * (n + 1):.2f}/hr\nActive assignments: {n + 1}")
elif cmd == "sessions":
    print("c5-distill-foreign  L4  BUSY")
    for s in active():
        print(f"{s}  L4  IDLE")
elif cmd == "new":
    m = os.environ.get("FAKE_FAIL_NEW_MATCH")
    if os.environ.get("FAKE_FAIL_NEW") == "1" or (m and m in sess):
        print("[colab] Allocation refused (precondition failed). This can mean too many active sessions", file=sys.stderr)
        sys.exit(1)
    (vm / "content").mkdir(parents=True)
    with open(D / "peak.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        peak = D / "peak_active"
        n = len(active())
        if not peak.exists() or int(peak.read_text()) < n:
            peak.write_text(str(n))
    print("created", sess)
elif cmd == "stop":
    shutil.rmtree(vm, ignore_errors=True)
    print("stopped", sess)
elif cmd == "upload":
    src, dst = [x for x in rest if not x.startswith("-") and x != sess][:2]
    t = remote(dst)
    if not vm.exists() or not t.parent.exists():
        print("500 Internal Server Error: parent directory missing", file=sys.stderr); sys.exit(1)
    shutil.copyfile(src, t)
elif cmd == "download":
    src, dst = [x for x in rest if not x.startswith("-") and x != sess][:2]
    if not remote(src).exists():
        print("404", file=sys.stderr); sys.exit(1)
    shutil.copyfile(remote(src), dst)
elif cmd == "exec":
    if not vm.exists():
        print("error: no such session", file=sys.stderr); sys.exit(1)
    env = dict(os.environ)
    for i, x in enumerate(rest):
        if x == "--env":
            k, v = rest[i + 1].split("=", 1); env[k] = v
    env["S12C_CONTENT"] = str(vm / "content")
    (vm / "tmp").mkdir(exist_ok=True)
    env["TMPDIR"] = str(vm / "tmp")          # each VM has its own /tmp (the Mac audit scans work dir + TMPDIR)
    if env.get("S12C_MODE") == "batch":
        with open(D / "count.lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            cnt = D / "batch_execs"
            n = int(cnt.read_text()) + 1 if cnt.exists() else 1
            cnt.write_text(str(n))
        if str(n) in os.environ.get("FAKE_DROP_EXEC", "").split(","):
            shutil.rmtree(vm, ignore_errors=True)          # the session dies mid-exec
            print("Kernel connection lost: runtime disconnected", file=sys.stderr); sys.exit(1)
    with open(D / f"kernel-{sess}.lock", "a") as lk:   # one exec at a time per session, as on a real kernel
        fcntl.flock(lk, fcntl.LOCK_EX)
        r = subprocess.run([sys.executable, opt("-f")], env=env, capture_output=True, text=True, cwd=str(vm))
    sys.stdout.write(r.stdout); sys.stderr.write(r.stderr[-4000:])
    sys.exit(0)
elif cmd == "ls":
    print("\n".join(os.listdir(remote(rest[-1]))))
else:
    print("fake: unknown", cmd, file=sys.stderr); sys.exit(2)
'''


def fake_env(tmp_path):
    fake = tmp_path / "fake"
    (fake / "vm").mkdir(parents=True)
    exe = fake / "colab"
    exe.write_text(FAKE_COLAB.replace("#!PYTHON", "#!" + sys.executable))
    exe.chmod(0o755)
    models = tmp_path / "models"
    (models / "embeddinggemma-2").mkdir(parents=True)
    (models / "SHA256SUMS").write_text("ab" * 32 + "  ./embeddinggemma-2/model.safetensors\n")
    synth.write_bundle(tmp_path / "gate", set(), set(), NOW - timedelta(hours=2))
    env = {**os.environ, "S12C_COLAB_BIN": str(exe), "FAKE_DIR": str(fake), "S12C_NOW": NOW_S,
           "S12C_UPLOAD_CHUNK": "65536",                 # batch, code and gate tars go up in parts, joined by the job
           "S12C_REHEARSAL_ENV": f"S12C_FAKE_ENCODER=1;S12C_NOW={NOW_S};S12C_DEVICE=cpu;S12C_FAKE_CDN_MS=0"}
    return env, fake, models


def calls(fake):
    p = fake / "calls.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def drive(tmp_path, mode, *args, env=None, run_dir=None):
    cmd = [sys.executable, str(HERE / "colab_driver.py"), mode, "--run-id", "rehearsal-1", "--execute", "--rehearsal",
           "--run-dir", str(run_dir or tmp_path / "run"), "--gate", str(tmp_path / "gate"),
           "--models", str(tmp_path / "models"), *args]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)


@pytest.fixture
def units(tmp_path):
    make_o8_and_sidecar(tmp_path / "src", n=300)
    out = tmp_path / "units"
    split_units.main([str(out), "--o8", str(tmp_path / "src/o8"), "--sidecar", str(tmp_path / "src/sidecar"),
                      "--unit-rows", "40", "--buckets", "3"])
    return out


def check_run_complete(tmp_path, units, fake):
    man = json.loads((units / "manifest.json").read_text())
    led = CL.RunLedger(tmp_path / "run")
    done = led.done()
    assert set(done) == {u["unit"] for u in man["units"]}
    total = 0
    for u in man["units"]:
        f = tmp_path / "run/vectors/units" / f"u-{u['unit']}-00000.parquet"
        t = pq.read_table(f)
        assert t.schema.equals(L.output_schema()) and t.num_rows == u["rows"]
        assert L.audit_output_table(f)["ok"]
        total += t.num_rows
    assert total == man["totals"]["posts"]
    stops = [c for c in calls(fake) if c[0] == "stop"]
    assert not any("c5-distill-foreign" in c for c in stops)                    # never someone else's session
    assert not os.listdir(fake / "vm")                                         # nothing left running
    receipts = list((tmp_path / "run/vectors/receipt").glob("*.json"))
    assert receipts and all(json.loads(p.read_text())["audit"]["ok"] for p in receipts)
    assert all(json.loads(p.read_text())["rate_unit"] == "colab_units" for p in receipts)
    return man, led


def test_driver_run_with_session_drop_and_auto_resume(tmp_path, units):
    env, fake, _m = fake_env(tmp_path)
    env["FAKE_DROP_EXEC"] = "2"                          # the 2nd batch exec loses the session
    p = drive(tmp_path, "run", "--units", str(units), "--batch-units", "2", env=env)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-3000:]
    man, led = check_run_complete(tmp_path, units, fake)
    ev = led.sessions()
    assert len(ev) == 2
    first, second = sorted(ev)
    assert ev[first][:2] == ["created", "dropped"] and ev[second] == ["created", "stopped"]
    news = [c for c in calls(fake) if c[0] == "new"]
    assert all("--gpu" in c and c[c.index("--gpu") + 1] == "L4" for c in news)
    up = [c for c in calls(fake) if c[0] == "upload"]
    mk = [i for i, c in enumerate(calls(fake)) if c[0] == "exec" and "/_mkdir-" in c[c.index("-f") + 1]]
    assert mk and min(i for i, c in enumerate(calls(fake)) if c[0] == "upload") > mk[0]      # mkdir before upload
    assert any(c[-1] == "/content/s12mm/gate.tar.gz" for c in up)
    assert len(list((tmp_path / "run/receipts").glob("usage_before_*.txt"))) == 1
    assert len(list((tmp_path / "run/receipts").glob("sessions_after_*.txt"))) == 1
    spent, det = led.spent_units()
    assert det["sessions"] == 2 and spent > 0
    for tf in (tmp_path / "run/stage").glob("*.tar.gz"):     # no AppleDouble anywhere in what went up
        import tarfile
        with tarfile.open(tf) as t:
            assert not any(os.path.basename(m.name).startswith("._") for m in t.getmembers())


def test_driver_resume_across_invocations(tmp_path, units):
    env, fake, _m = fake_env(tmp_path)
    p1 = drive(tmp_path, "run", "--units", str(units), "--batch-units", "2", "--max-sessions", "1",
               "--max-batches", "2", env=env)
    assert p1.returncode == 0, p1.stdout[-3000:] + p1.stderr[-2000:]
    led = CL.RunLedger(tmp_path / "run")
    first_done = set(led.done())
    man = json.loads((units / "manifest.json").read_text())
    assert 0 < len(first_done) < len(man["units"])
    n_up = len([c for c in calls(fake) if c[0] == "upload" and "/batches/" in c[-1]])
    p2 = drive(tmp_path, "run", "--units", str(units), "--batch-units", "3", env=env)
    assert p2.returncode == 0, p2.stdout[-3000:] + p2.stderr[-2000:]
    check_run_complete(tmp_path, units, fake)
    # the second driver uploaded only units that were not yet in the ledger
    later = [c for c in calls(fake) if c[0] == "upload" and "/batches/" in c[-1]][n_up:]
    sent = set()
    for c in later:
        name = pathlib.Path(c[-1]).stem
        meta = json.loads((tmp_path / "run/stage/batches" / name / "batch.json").read_text())
        sent |= {u["unit"] for u in meta["units"]}
    assert sent and not (sent & first_done)
    p3 = drive(tmp_path, "run", "--units", str(units), env=env)
    assert p3.returncode == 0 and "nothing to do" in p3.stdout


def test_driver_reconciles_a_crashed_drivers_session_and_respects_cap(tmp_path, units):
    env, fake, _m = fake_env(tmp_path)
    led = CL.RunLedger(tmp_path / "run")
    (fake / "vm" / "s12c-run-old/content").mkdir(parents=True)         # still running on "Colab"
    led.record("s12c-run-old", "created", "L4")
    led.spend_update("s12c-run-old", rate=1.71, t_created=time.time() - 3600, t_last_seen=time.time() - 3500)
    led.record("s12c-run-older", "created", "L4")                         # gone without a trace
    led.spend_update("s12c-run-older", rate=1.71, t_created=time.time() - 7200, t_last_seen=time.time() - 7000)
    p = drive(tmp_path, "run", "--units", str(units), "--cap", "1.5", env=env)
    assert p.returncode == 4, p.stdout[-3000:]                            # ~2.0 units already spent > cap
    assert "unit cap" in p.stdout
    ev = led.sessions()
    assert "stopped(reconcile)" in ev["s12c-run-old"] and "ended(unobserved)" in ev["s12c-run-older"]
    assert not (fake / "vm" / "s12c-run-old").exists()
    assert not [c for c in calls(fake) if c[0] == "new"]                  # no new session past the cap


def test_driver_stops_on_stale_gate_without_a_session(tmp_path, units):
    env, fake, _m = fake_env(tmp_path)
    synth.write_bundle(tmp_path / "gate", set(), set(), NOW - timedelta(hours=40))
    p = drive(tmp_path, "run", "--units", str(units), env=env)
    assert p.returncode == 6 and "refresh it" in p.stdout
    assert not [c for c in calls(fake) if c[0] in ("new", "upload")]


def test_driver_print_only_by_default(tmp_path, units):
    env, fake, _m = fake_env(tmp_path)
    cmd = [sys.executable, str(HERE / "colab_driver.py"), "run", "--run-id", "dry", "--units", str(units),
           "--run-dir", str(tmp_path / "run"), "--gate", str(tmp_path / "gate"), "--models", str(tmp_path / "models"),
           "--rehearsal"]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "dry run: nothing was executed" in p.stdout and "+ " in p.stdout and " new -s s12c-run-dry" in p.stdout
    assert not calls(fake)
    p = subprocess.run(cmd[:-1] + ["--cap", "701", "--execute"], env=env, capture_output=True, text=True)
    assert p.returncode == 2 and "700 units" in p.stderr
    p = subprocess.run(cmd[:-1] + ["--sessions", "4", "--execute"], env=env, capture_output=True, text=True)
    assert p.returncode == 2 and "--sessions" in p.stderr
    p = subprocess.run(cmd + ["--rps", "100", "--execute"], env=env, capture_output=True, text=True)
    assert p.returncode == 2 and "allow-above-etiquette" in p.stderr


def bench_lake(tmp_path):
    """The bench inputs the driver uploads, made with FakeEncoder so the parity checks pass exactly."""
    lake = tmp_path / "lake"
    enc = synth.FakeEncoder()
    texts = [f"slab text {i} with words" for i in range(400)]
    up = lake / "s7-bench-20261028/upload"
    up.mkdir(parents=True)
    pq.write_table(pa.table({"row_id": list(range(1_000_000, 1_000_400)), "text": texts}), up / "s7_slab.parquet")
    rnd = [f"random text {i}" for i in range(1000)]
    pq.write_table(pa.table({"row_id": list(range(1000)), "text": rnd}), up / "s7_random.parquet")
    ids = np.array(list(range(1000)) + list(range(1_000_000, 1_000_250)), np.int64)
    np.savez(up / "parity_mac.npz", row_id=ids, vec=enc.encode_text(rnd + texts[:250]))
    ims = synth.parity_images()
    ptexts = [f"parity post {i}: a picture" if i % 3 else "" for i in range(len(ims))]
    mm = {}
    for s in (70, 280):
        enc.set_soft_tokens(s)
        mm[f"s{s}"] = enc.encode_mm([(t, [im]) for t, im in zip(ptexts, ims)])
    (lake / "s12mm-rehearsal-20261028").mkdir(parents=True)
    np.savez(lake / "s12mm-rehearsal-20261028/mm_parity_mac.npz", **mm)
    ip = lake / "s7g-input-20261028/out/image_posts"
    ip.mkdir(parents=True)
    pq.write_table(pa.table({"uri": ["at://did:plc:x/app.bsky.feed.post/y"], "image_cids": [["c"]]}), ip / "part-0.parquet")
    (lake / "s7g-input-20261028/out/bench_producers.txt").write_text("")
    return lake


def test_driver_bench_with_ramp(tmp_path):
    env, fake, _m = fake_env(tmp_path)
    lake = bench_lake(tmp_path)
    env["S12C_REHEARSAL_ENV"] += (";S12C_SYNTHETIC_IMAGES=1;S12C_SYNTHETIC_LIMIT_RPS=25;S12C_RAMP_STEP_S=2;"
                                  "S12C_RAMP_MIN_STEP_S=1;S12C_TEXT_BS=64,128;S12C_MM_BS=4,8;S12C_SWEEP_N_MM=16")
    p = drive(tmp_path, "bench", "--lake", str(lake), "--ramp", "5,10,20,40", "--n-text", "300",
              "--n-image-posts", "24", env=env)
    assert p.returncode == 0, p.stdout[-5000:] + p.stderr[-3000:]
    res = tmp_path / "run/results"
    b = json.loads((res / "bench.json").read_text())
    assert json.loads((res / "driver_receipt.json").read_text())["tar_sha256_ok"]
    assert b["parity_text"]["bar_median_ge_0_99"] and b["parity_mm"]["bar_median_ge_0_98"]
    assert b["audit"]["ok"] and b["nan_or_inf_rows_total"] == 0 and b["rate_unit"] == "colab_units"
    r = b["cdn_ramp"]
    # 429 at the 40 step; on a loaded Mac the loopback p95 rule can fire first (both are real stop rules)
    assert r["stopped"]["reason"] == "429" or r["stopped"]["reason"].startswith("p95")
    assert r["max_clean_target_rps"] in (5, 10, 20)
    assert all("cdn_cache" in s and s["miss_share"] is not None for s in r["steps"])
    assert "MISS" in b["fetch"]["cdn_cache"]
    assert {x["gpu"] for x in b["colab"]["per_10m_posts"]} == {"L4", "A100"}
    ev = CL.RunLedger(tmp_path / "run").sessions()
    assert len(ev) == 1 and list(ev.values())[0] == ["created", "stopped"]
    assert not os.listdir(fake / "vm")


# --- parallel sessions, leases, fetch concurrency -------------------------------------------------------------

def test_leases_are_disjoint_expire_and_release(tmp_path):
    clock = [0.0]
    led = CL.RunLedger(tmp_path / "run", clock=lambda: clock[0])
    man = {"units": [{"unit": f"u{i}", "rows": 1} for i in range(6)]}
    a = led.claim("A", man, 2, ttl_s=100)
    b = led.claim("B", man, 3, ttl_s=100)
    assert [u["unit"] for u in a] == ["u0", "u1"] and [u["unit"] for u in b] == ["u2", "u3", "u4"]
    assert [u["unit"] for u in led.unclaimed_todo(man)] == ["u5"]
    assert [u["unit"] for u in led.claim("A", man, 9, ttl_s=100)] == ["u5"]   # never its own in-flight units again
    clock[0] = 150                                                    # B's session died silently: leases lapse
    led.renew("A", ["u0", "u1", "u5"], ttl_s=100)
    assert [u["unit"] for u in led.unclaimed_todo(man)] == ["u2", "u3", "u4"]
    assert [u["unit"] for u in led.claim("C", man, 9, ttl_s=100)] == ["u2", "u3", "u4"]
    assert led.release("C", ["u3"]) == 1 and [u["unit"] for u in led.unclaimed_todo(man)] == ["u3"]
    led.mark("u0", {"sha256": "x", "rows": 1, "file": "f"})
    assert led.release("A") == 3                                       # u0 is done, so it left the pool anyway
    assert "u0" not in [u["unit"] for u in led.claim("D", man, 9, ttl_s=100)]


def test_leases_hold_across_threads(tmp_path):
    import threading
    led = CL.RunLedger(tmp_path / "run")
    man = {"units": [{"unit": f"u{i:03d}", "rows": 1} for i in range(300)]}
    got = {}

    def take(o):
        mine = []
        while True:
            c = [u["unit"] for u in led.claim(o, man, 7, ttl_s=1000)]
            if not c:
                break
            mine += c
        got[o] = mine
    ts = [threading.Thread(target=take, args=(f"L{i}",)) for i in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    allu = [u for v in got.values() for u in v]
    assert len(allu) == 300 and len(set(allu)) == 300


def test_fetcher_auto_in_flight_reaches_rps_at_slow_latency():
    assert F.auto_in_flight(50) == 150 and F.auto_in_flight(5) == 32 and F.auto_in_flight(500) == 256
    did = synth.synthetic_did(6)
    items = [(i, f"https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{synth.synthetic_cid(i)}") for i in range(300)]
    walls = {}
    with synth.ImageServer(latency_ms=1000, fail404="") as srv:
        for name, n_inflight, chunk in (("old32", 32, items[:150]), ("auto", F.auto_in_flight(50), items[150:])):
            f = F.AsyncFetcher(rps=50, max_in_flight=n_inflight, base=srv.base, rehearsal=True, discard_bodies=True)
            t0 = time.monotonic()
            f.fetch_sync(chunk)
            walls[name] = time.monotonic() - t0
            f.close()
    # 150 requests at 50 req/s and 1 s per answer: the token bucket alone needs ~3 s (+1 s for the last answer);
    # 32 in flight caps the rate at 32 req/s (~4.7 s + 1 s), as on Colab (35 req/s achieved at a 50 cap)
    assert walls["old32"] > 5.2 and walls["auto"] < 4.7, walls


def test_driver_three_parallel_sessions_one_drops(tmp_path, units):
    env, fake, _m = fake_env(tmp_path)
    env["FAKE_DROP_EXEC"] = "3"                      # the 3rd batch exec (whichever lane) loses its session
    p = drive(tmp_path, "run", "--units", str(units), "--batch-units", "1", "--sessions", "3", env=env)
    assert p.returncode == 0, p.stdout[-5000:] + p.stderr[-3000:]
    man, led = check_run_complete(tmp_path, units, fake)
    ev = led.sessions()
    lanes = {n.split("-l")[1][0] for n in ev}
    assert lanes == {"0", "1", "2"}
    assert int((fake / "peak_active").read_text()) == 3      # three sessions really ran at the same time
    dropped = [n for n, e in ev.items() if "dropped" in e]
    assert len(dropped) == 1 and all(e[-1] in ("stopped", "dropped") for e in ev.values())
    # every unit was verified once; units of the dropped batch came back and were done by some session
    done = led.done()
    assert set(done) == {u["unit"] for u in man["units"]}
    assert not list((tmp_path / "run/ledger/claims").glob("*.json"))   # no lease left behind
    seen = {}
    for bj in (tmp_path / "run/stage/batches").glob("*/batch.json"):
        for u in json.loads(bj.read_text())["units"]:
            seen.setdefault(u["unit"], []).append(bj.parent.name)
    redo = {u: b for u, b in seen.items() if len(b) > 1}
    assert len(redo) <= 2, redo                       # only the dropped batch (and its pre-uploaded next) redone
    # each new session got its own CLI state file
    news = [c for c in calls(fake) if c[0] == "new"]
    assert len(news) == len(ev) >= 4
    spend = [json.loads(x.read_text()) for x in (tmp_path / "run/spend").glob("*.json")]
    assert len(spend) == len(ev) and all(r.get("t_stopped") for r in spend)


def test_driver_degrades_when_a_parallel_new_is_refused(tmp_path, units):
    env, fake, _m = fake_env(tmp_path)
    env["FAKE_FAIL_NEW_MATCH"] = "-l2-"              # the account refuses a third session
    p = drive(tmp_path, "run", "--units", str(units), "--batch-units", "2", "--sessions", "3", env=env)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-2000:]
    check_run_complete(tmp_path, units, fake)
    assert "continuing with fewer sessions" in p.stdout
    ev = CL.RunLedger(tmp_path / "run").sessions()
    assert {n.split("-l")[1][0] for n in ev} == {"0", "1"}
