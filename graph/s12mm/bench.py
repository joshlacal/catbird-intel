"""S12m benchmark / smoke test (replaces the Colab S7; S1 A2 makes C4 an engineering smoke test).

  job.py bench --model-dir D --work W --run-id ID --out URL --bench-text s7_slab.parquet
               [--bench-random s7_random.parquet --parity-ref parity_mac.npz] [--mm-parity-ref mm_parity_mac.npz]
               (--image-posts URL [--sidecar URL] --gate-bundle DIR [--bench-producers FILE]
                | --synthetic-images)   # rehearsal only: loopback server, generated JPEGs
               [--n-text 10000] [--n-image-posts 10000] [--soft-tokens-list 70,280] [--rps 50] [--cap-usd C]
               [--emit-parity F --emit-mm-parity F]   # Mac: write the references instead of comparing

Phases, each timed and each with nvidia-smi utilisation (None on the Mac):
  1 parity, text: S7's rows (random 0-999 + slab 0-249) at batch 64 vs the Mac reference (median cosine >= 0.99);
  2 parity, multimodal: 48 deterministic synthetic images + fixed texts at S = 70 and 280 vs the Mac reference;
  3 text throughput: batch sweep on the slab-format rows, then all --n-text rows; tokens/post;
  4 fetch: every thumbnail of --n-image-posts gate-passing posts at --rps into memory; achieved req/s, latency,
    statuses; the GPU is idle here;
  5 multimodal throughput at each soft-token budget: batch sweep, then all fetched posts (decode included, as in
    the run); tokens/post including soft tokens; text-only companion vectors for the same posts;
  6 dollars per 10M posts each way and the S12m projection (projection.py);
  7 G-img inputs (report-only under A2): per image post, vec_text, vec_mm70, vec_mm280 -> bench/vectors.parquet
    for gimg_report.py on the Mac; 8 the no-image-bytes audit.
Writes OUT/bench/bench.json (+ vectors.parquet); exits 3 on any NaN, 7 on a failed audit.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import audit
import mmlib as L
import projection
import stores as S
from fetcher import CircuitOpen

PARITY_RANDOM, PARITY_SLAB, PARITY_BATCH = 1000, 250, 64


def add_args(b):
    b.add_argument("--bench-text", required=True)
    b.add_argument("--bench-random")
    b.add_argument("--parity-ref")
    b.add_argument("--emit-parity")
    b.add_argument("--mm-parity-ref")
    b.add_argument("--emit-mm-parity")
    b.add_argument("--image-posts")
    b.add_argument("--sidecar")
    b.add_argument("--bench-producers")
    b.add_argument("--synthetic-images", action="store_true")
    b.add_argument("--n-text", type=int, default=10_000)
    b.add_argument("--n-image-posts", type=int, default=10_000)
    b.add_argument("--soft-tokens-list", default="70,280")
    b.add_argument("--text-batch-sizes", default="64,128,256,512")
    b.add_argument("--mm-batch-sizes", default="8,16,32")
    b.add_argument("--sweep-n-mm", type=int, default=512)
    b.add_argument("--parity-only", action="store_true")


def _s7(name):
    """Load graph/s7bench/<name>.py under a private module name (s7bench/bench.py would shadow this bench)."""
    import importlib.util
    key = f"_s7_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, L.HERE.parent / "s7bench" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def fetch_local(url, work):
    """A bench input given as a local path or a gs:// object, as a local path."""
    if not url.startswith("gs://"):
        return pathlib.Path(url)
    base, name = url.rsplit("/", 1)
    dst = work / "bench_in" / name
    S.open_store(base).get(name, dst)
    return dst


def read_rows(path, n=None):
    import pyarrow.parquet as pq
    t = pq.read_table(path).sort_by("row_id")
    return t.slice(0, n).to_pylist() if n else t.to_pylist()


def timed(enc, fn):
    enc.sync()
    t0 = time.time()
    r = fn()
    enc.sync()
    return time.time() - t0, r


def nan_rows(full):
    import numpy as np
    return int((~np.isfinite(full)).any(axis=1).sum())


def parity_text(a, enc, work, res):
    import numpy as np
    rnd = read_rows(fetch_local(a.bench_random, work), PARITY_RANDOM)
    slab = read_rows(fetch_local(a.bench_text, work), PARITY_SLAB)
    rows = rnd + slab
    full = enc.encode_text([r["text"] for r in rows], bs=PARITY_BATCH)
    ids = np.array([r["row_id"] for r in rows], np.int64)
    if a.emit_parity:
        np.savez(a.emit_parity, row_id=ids, vec=full.astype(np.float32))
    if a.parity_ref:
        s7lib = _s7("s7lib")
        ref = np.load(fetch_local(a.parity_ref, work))
        pos = {int(r): i for i, r in enumerate(ref["row_id"])}
        sel = [i for i, r in enumerate(ids) if int(r) in pos]
        rv, mine = ref["vec"][[pos[int(ids[i])] for i in sel]], full[sel]
        is_r = ids[sel] < 1_000_000
        res["parity_text"] = {"random_full": s7lib.cosine_stats(mine[is_r], rv[is_r]),
                              "random_256": s7lib.cosine_stats(mine[is_r], rv[is_r], L.DIM),
                              "slab_256": s7lib.cosine_stats(mine[~is_r], rv[~is_r], L.DIM) if (~is_r).any() else None}
        res["parity_text"]["bar_median_ge_0_99"] = res["parity_text"]["random_full"]["median"] >= 0.99
    res["parity_text_nan_rows"] = nan_rows(full)


def parity_mm(a, enc, work, res, softs):
    import numpy as np
    from synth import parity_images
    ims = parity_images()
    texts = [f"parity post {i}: a picture" if i % 3 else "" for i in range(len(ims))]
    posts = [(t, [im]) for t, im in zip(texts, ims)]
    out = {}
    for s in softs:
        enc.set_soft_tokens(s)
        out[s] = enc.encode_mm(posts, bs=8)
    if a.emit_mm_parity:
        np.savez(a.emit_mm_parity, **{f"s{s}": v.astype(np.float32) for s, v in out.items()})
    if a.mm_parity_ref:
        s7lib = _s7("s7lib")
        ref = np.load(fetch_local(a.mm_parity_ref, work))
        res["parity_mm"] = {str(s): s7lib.cosine_stats(out[s], ref[f"s{s}"], L.DIM) for s in softs if f"s{s}" in ref}
        res["parity_mm"]["bar_median_ge_0_98"] = all(v["median"] >= 0.98 for k, v in res["parity_mm"].items()
                                                     if isinstance(v, dict))
    res["parity_mm_nan_rows"] = sum(nan_rows(v) for v in out.values())


def text_phase(a, enc, gpu, work, res, guard):
    s7bench = _s7("bench")   # S7's token counter (same tokenizer handling)
    rows = read_rows(fetch_local(a.bench_text, work), a.n_text)
    texts = [r["text"] for r in rows]
    res["text_tokens"] = s7bench.token_counts(s7bench.tokenizer_of(enc.m), texts, enc.prompt)
    enc.encode_text(texts[:256], bs=64)   # warm-up
    sweep, best = [], None
    for bs in [int(x) for x in a.text_batch_sizes.split(",") if x]:
        if not guard.allows(60):
            sweep.append({"batch": bs, "skipped": "dollar cap"})
            break
        g0 = gpu.mark()
        try:
            s, full = timed(enc, lambda: enc.encode_text(texts[:4000], bs=bs))
        except Exception as e:
            if "out of memory" not in str(e).lower():
                raise
            sweep.append({"batch": bs, "oom": True})
            enc.empty_cache()
            break
        sweep.append({"batch": bs, "n": min(4000, len(texts)), "posts_per_s": round(min(4000, len(texts)) / s, 1),
                      "nan_rows": nan_rows(full), **gpu.window(g0)})
        if best is None or sweep[-1]["posts_per_s"] > best[1]:
            best = (bs, sweep[-1]["posts_per_s"])
    g0 = gpu.mark()
    s, full = timed(enc, lambda: enc.encode_text(texts, bs=best[0]))
    res["text"] = {"n": len(texts), "batch": best[0], "seconds": round(s, 2), "posts_per_s": round(len(texts) / s, 1),
                   "nan_rows": nan_rows(full), "peak_mem_gb": enc.peak_mem_gb(), "sweep": sweep, **gpu.window(g0)}


def image_posts(a, work, now):
    """Real: gate-passing image posts from the S12m inputs, ordered by (not in --bench-producers, input `tier`
    (bench_input.py: 0 = S7 slab producer, 1 = S4 corpus, 2 = other), blake2b('s12mm-bench-image|uri')). Rehearsal: synthetic posts with S7 slab texts. Returns (posts, gate counts)."""
    import collections
    import pipeline as P
    if a.synthetic_images:
        from synth import synthetic_posts
        texts = [r["text"] for r in read_rows(fetch_local(a.bench_text, work))]
        rows, excl, lab = synthetic_posts(texts, now, image_share=1.0, seed=1)
        bundle = L.GateBundle(excl, lab, now)
    else:
        import job
        bundle = L.load_bundle(a.gate_bundle)
        src, keys, side = job.load_inputs(argparse_ns(posts=a.image_posts, sidecar=a.sidecar), work)
        rows = []
        for k in keys:   # enough shards for a 20x candidate pool (shards are producer-partitioned, ~uniform)
            rows += [r for _n, rs in job.unit_rows(src, k, work, side, 10**9) for r in rs if r.get("image_cids")]
            if len(rows) >= 20 * a.n_image_posts:
                break
    u = P.prepare("bench", rows, bundle, now)
    ok = [i for i in range(len(u.uris)) if u.urls[i]]
    pref = set()
    if a.bench_producers:
        pref = {x.strip() for x in open(fetch_local(a.bench_producers, work)) if x.strip()}
    order = L.hash_order("s12mm-bench-image", [u.uris[i] for i in ok])
    tier = {r.get("uri") or L.post_uri(r["did"], r["rkey"]): r.get("tier", 9) for r in rows}   # bench_input.py tiers
    pos = {j: n for n, j in enumerate(order)}
    ranked = sorted(order, key=lambda j: (L.did_rkey(u.uris[ok[j]])[0] not in pref, tier.get(u.uris[ok[j]], 9), pos[j]))
    keep = [ok[j] for j in ranked[:a.n_image_posts]]
    sub = P.Unit("bench", [u.uris[i] for i in keep], [u.texts[i] for i in keep], [u.pids[i] for i in keep],
                 [u.n_cids[i] for i in keep], [u.urls[i] for i in keep], [None] * len(keep))
    return sub, dict(collections.Counter({k: v for k, v in u.counts.items()})), len(pref)


def argparse_ns(**kw):
    import argparse
    return argparse.Namespace(**kw)


def mm_phase(a, enc, gpu, unit, fetched, res, guard, softs):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from encoder import decode
    good = [i for i in range(len(unit.uris)) if all(fetched.get((i, j), (None,))[0] is not None
                                                     for j in range(len(unit.urls[i])))]
    good.sort(key=lambda i: len(unit.urls[i]))
    res["mm_posts"] = len(good)
    res["mm_images"] = sum(len(unit.urls[i]) for i in good)
    pool = ThreadPoolExecutor(6)
    vecs = {}

    def run(idx, bs):
        out = []
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            ims = list(pool.map(lambda i: [decode(fetched[(i, j)][0]) for j in range(len(unit.urls[i]))], b))
            out.append(enc.encode_mm([(unit.texts[i], im) for i, im in zip(b, ims)], bs=len(b)))
            del ims
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    hasT = [i for i in good if unit.texts[i].strip()]
    s, tfull = timed(enc, lambda: enc.encode_text([unit.texts[i] for i in hasT], bs=res.get("text", {}).get("batch", 256)))
    vecs["text"] = dict(zip(hasT, L.pack(L.to_stored(tfull)))) if hasT else {}
    res["image_posts_text_companion"] = {"n": len(hasT), "seconds": round(s, 2)}
    for soft in softs:
        enc.set_soft_tokens(soft)
        r = {"soft_tokens": soft}
        sample = good[:: max(1, len(good) // 200)][:200]
        toks = [enc.mm_tokens(unit.texts[i], [decode(fetched[(i, j)][0]) for j in range(len(unit.urls[i]))])
                for i in sample]
        r["tokens_per_post_mean"] = round(float(np.mean(toks)), 1) if toks else None
        r["tokens_per_post_p99"] = float(np.quantile(toks, 0.99)) if toks else None
        sweep_idx = good[: a.sweep_n_mm]
        run(sweep_idx[:16], 8)   # warm-up
        sweep, best = [], None
        for bs in [int(x) for x in a.mm_batch_sizes.split(",") if x]:
            if not guard.allows(120):
                sweep.append({"batch": bs, "skipped": "dollar cap"})
                break
            g0 = gpu.mark()
            try:
                t, full = timed(enc, lambda: run(sweep_idx, bs))
            except Exception as e:
                if "out of memory" not in str(e).lower():
                    raise
                sweep.append({"batch": bs, "oom": True})
                enc.empty_cache()
                break
            sweep.append({"batch": bs, "n": len(sweep_idx), "posts_per_s": round(len(sweep_idx) / t, 2),
                          "nan_rows": nan_rows(full), **gpu.window(g0)})
            if best is None or sweep[-1]["posts_per_s"] > best[1]:
                best = (bs, sweep[-1]["posts_per_s"])
        r["sweep"] = sweep
        if best is None:
            res[f"mm{soft}"] = r
            continue
        est = len(good) / best[1] * 1.2
        if not guard.allows(est):
            r["full"] = {"skipped": f"dollar cap: estimate {est:.0f}s", **guard.state()}
            res[f"mm{soft}"] = r
            continue
        g0 = gpu.mark()
        t, full = timed(enc, lambda: run(good, best[0]))
        r.update(n=len(good), batch=best[0], seconds=round(t, 2), posts_per_s=round(len(good) / t, 2),
                 images_per_s=round(res["mm_images"] / t, 2), nan_rows=nan_rows(full), peak_mem_gb=enc.peak_mem_gb(),
                 **gpu.window(g0))
        vecs[soft] = dict(zip(good, L.pack(L.to_stored(full))))
        res[f"mm{soft}"] = r
        print("[bench]", f"mm{soft}", json.dumps({k: v for k, v in r.items() if k != "sweep"}), flush=True)
    pool.shutdown()
    return good, vecs


def write_vectors(path, unit, good, vecs, softs):
    import pyarrow as pa
    import pyarrow.parquet as pq
    b = pa.binary(2 * L.DIM)
    cols = {"uri": [unit.uris[i] for i in good], "p": [unit.pids[i] for i in good],
            "n_images": [len(unit.urls[i]) for i in good], "vec_text": [vecs["text"].get(i) for i in good]}
    sch = [("uri", pa.string()), ("p", pa.int64()), ("n_images", pa.int8()), ("vec_text", b)]
    for s in softs:
        cols[f"vec_mm{s}"] = [vecs.get(s, {}).get(i) for i in good]
        sch.append((f"vec_mm{s}", b))
    pq.write_table(pa.Table.from_pydict(cols, schema=pa.schema(sch)), path, compression="zstd")


def cmd_bench(a):
    import job
    from encoder import EG2, GpuSampler
    job.check_out(a)
    if a.synthetic_images and not a.rehearsal:
        sys.exit("--synthetic-images is rehearsal-only")
    if not a.synthetic_images and not (a.image_posts and a.gate_bundle) and not a.parity_only:
        sys.exit("bench needs --image-posts + --gate-bundle (or --synthetic-images in a rehearsal)")
    work = pathlib.Path(a.work)
    (work / "out").mkdir(parents=True, exist_ok=True)
    out = S.open_store(a.out, custom_time=L.iso(L.utcnow()))   # plan-60d lifecycle keys on customTime
    softs = [int(x) for x in a.soft_tokens_list.split(",")]
    guard = L.DollarGuard(a.cap_usd or float("inf"), a.usd_per_hour, 0.0, t0=a.vm_start_epoch or time.time())
    res = {"run_id": a.run_id, "rehearsal": a.rehearsal, "rps_cap": a.rps}
    enc = EG2(a.model_dir, a.device, softs[0], a.mps_gb)
    import torch
    import sentence_transformers
    import transformers
    res["stack"] = {"device": enc.dev, "device_name": torch.cuda.get_device_name(0) if enc.dev == "cuda" else enc.dev,
                    "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
                    "sentence_transformers": sentence_transformers.__version__, "params_m": round(enc.params_m, 1),
                    "dtypes": enc.param_dtypes, "load_seconds": round(enc.load_seconds, 1)}
    gpu = GpuSampler()
    if a.bench_random:
        parity_text(a, enc, work, res)
    parity_mm(a, enc, work, res, softs)
    if a.parity_only:
        (work / "bench.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return 0
    text_phase(a, enc, gpu, work, res, guard)
    print("[bench] text", json.dumps({k: v for k, v in res["text"].items() if k != "sweep"}), flush=True)

    now = job.clock_now(a)
    unit, gate_counts, n_pref = image_posts(a, work, now)
    res["image_selection"] = {"gate_counts": gate_counts, "selected_posts": len(unit.uris),
                              "selected_images": sum(len(x) for x in unit.urls), "bench_producers": n_pref}
    marker = audit.start_marker(work)
    from contextlib import nullcontext
    from synth import ImageServer
    ctx = ImageServer() if a.synthetic_images else nullcontext()
    with ctx as srv:
        if srv is not None:
            a.cdn_base = srv.base
        fetcher = job.make_fetcher(a)
        g0 = gpu.mark()
        t0 = time.time()
        try:
            fetched = fetcher.fetch_sync(unit.fetch_items())
        except CircuitOpen as e:
            res["fetch"] = {"circuit_open": str(e), **fetcher.stats.summary()}
            fetched = {}
        res["fetch"] = {**res.get("fetch", {}), "seconds": round(time.time() - t0, 2), **fetcher.stats.summary(),
                        **gpu.window(g0)}
        fetcher.close()
    print("[bench] fetch", json.dumps(res["fetch"]), flush=True)
    good, vecs = mm_phase(a, enc, gpu, unit, fetched, res, guard, softs)
    fetched.clear()
    vpath = work / "out" / "bench_vectors.parquet"
    write_vectors(vpath, unit, good, vecs, softs)
    res["dollars"] = projection.dollars_from_bench(res, a.usd_per_hour)
    res["projection"] = projection.project_from_bench(res, a.usd_per_hour)
    res["nan_or_inf_rows_total"] = (res.get("parity_text_nan_rows", 0) + res.get("parity_mm_nan_rows", 0)
                                    + res["text"]["nan_rows"] + sum(res.get(f"mm{s}", {}).get("nan_rows", 0) for s in softs))
    res["guard"] = guard.state()
    res["audit"] = audit.finish(work, marker)
    res["audit"]["bench_vectors_rows"] = len(good)
    gpu.stop()
    (work / "bench.json").write_text(json.dumps(res, indent=1, default=str))
    out.put(vpath, "bench/vectors.parquet")
    out.put(work / "bench.json", "bench/bench.json")
    print(json.dumps({k: v for k, v in res.items() if k not in ("audit",)}, indent=1, default=str))
    print("[bench] audit ok:", res["audit"]["ok"], flush=True)
    if res["nan_or_inf_rows_total"]:
        return 3
    return 0 if res["audit"]["ok"] else 7

