"""S12m main run: natively multimodal producer-slab embeddings on one GCE Spot L4 (S1 amendment A2).

  job.py run --posts URL --out URL --gate-bundle DIR --model-dir DIR --work DIR --run-id ID
             [--sidecar URL] [--soft-tokens 70] [--rps 50] [--max-in-flight 32] [--unit-rows 20000]
             [--cap-usd C --usd-per-hour R] [--max-label-lag-h 36] [--max-units N]
             [--rehearsal --cdn-base http://127.0.0.1:PORT --now ISO]

--posts: prefix of input parquet shards (O8 joined with image CIDs; schema in mmlib's docstring); --sidecar: an
optional (uri | did+rkey, image_cids) prefix left-joined on uri. --out: the run's output root, on GCS
gs://catbird-436600-plan-60d/vectors/s12mm/<RUN_ID> (60-day lifecycle on customTime, set once per run in
_run.json and reused by every resumed VM so a rewrite never extends an object's life).

Work units = (input file, slice of --unit-rows rows), processed in order. A unit's output parquet goes up first,
then its ledger marker `_ledger/<unit>.json`; a resumed VM lists the ledger once and skips marked units (a unit
with an output but no marker is redone and overwritten). The fetch of unit k+1 overlaps the GPU work of unit k.
Stops cleanly (exit 0, VM self-deletes) when the dollar guard would cross --cap-usd before the next unit; exits 5
if the fetch circuit breaker trips (429s/errors), 6 if the label bundle is older than --max-label-lag-h, 7 if
any output fails the no-image-bytes audit. Progress: one line per unit and a heartbeat every 60 s.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit  # noqa: E402
import mmlib as L  # noqa: E402
import pipeline as P  # noqa: E402
import stores as S  # noqa: E402
from fetcher import AsyncFetcher, CircuitOpen  # noqa: E402

EXIT_CIRCUIT, EXIT_STALE_LABELS, EXIT_AUDIT = 5, 6, 7


def log(*a):
    print(f"[{L.iso(L.utcnow())}]", *a, flush=True)


def common_args(ap):
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-bundle", help="local dir with gate_manifest.json, excluded_dids.txt, labelled_uris.txt")
    ap.add_argument("--rps", type=float, default=50.0)
    ap.add_argument("--allow-above-etiquette", action="store_true",
                    help="permit up to 200 req/s; only with Josh's written approval and Bluesky's OK")
    ap.add_argument("--max-in-flight", type=int, default=32)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--cdn-base", default="https://cdn.bsky.app")
    ap.add_argument("--now", help="rehearsal only: fixed clock for the age rule")
    ap.add_argument("--cap-usd", type=float, default=0.0)
    ap.add_argument("--usd-per-hour", type=float, default=L.g2_standard_8_spot_usd_per_hour())
    ap.add_argument("--vm-start-epoch", type=float, help="VM creation time (default: now)")
    ap.add_argument("--max-label-lag-h", type=float, default=36.0)
    ap.add_argument("--text-bs", type=int, default=256)
    ap.add_argument("--mm-bs", type=int, default=16)
    ap.add_argument("--device")
    ap.add_argument("--mps-gb", type=float, default=7.0)


def make_fetcher(a):
    return AsyncFetcher(rps=a.rps, max_in_flight=a.max_in_flight, base=a.cdn_base, rehearsal=a.rehearsal,
                        allow_above_etiquette=a.allow_above_etiquette)


def clock_now(a):
    if a.now:
        if not a.rehearsal:
            sys.exit("--now is rehearsal-only")
        return datetime.fromisoformat(a.now.replace("Z", "+00:00"))
    return L.utcnow()


def check_out(a):
    if not S.allowed_write(a.out):
        sys.exit(f"refusing to write outside plan-60d/vectors/s12mm/ or plan-30d/intermediates/: {a.out}")
    if a.rehearsal and a.out.startswith("gs://"):
        sys.exit("a rehearsal never writes to GCS")


def prior_spend(out, vm):
    tot = 0.0
    for k in S.list_keys(out, "_ledger/vm-", ".json"):
        if k != f"_ledger/vm-{vm}.json":
            tot += json.loads(out.read_text(k)).get("usd_spent_this_vm", 0.0)
    return tot


def load_inputs(a, work):
    src = S.open_store(a.posts)
    keys = S.list_keys(src, "", ".parquet")
    if not keys:
        sys.exit(f"no parquet under {a.posts}")
    side = None
    if a.sidecar:
        import pyarrow as pa
        import pyarrow.parquet as pq
        ss = S.open_store(a.sidecar)
        parts = []
        for k in S.list_keys(ss, "", ".parquet"):
            ss.get(k, work / "sidecar" / k)
            t = pq.read_table(work / "sidecar" / k)
            if "uri" not in t.column_names:
                t = t.append_column("uri", pa.array([L.post_uri(d, r) for d, r in zip(t["did"].to_pylist(), t["rkey"].to_pylist())]))
            parts.append(t.select(["uri", "image_cids"]))
        side = pa.concat_tables(parts).combine_chunks() if parts else None
    return src, keys, side


def unit_rows(src, key, work, side, unit_rows_n):
    import pyarrow.parquet as pq
    local = work / "in" / key
    if not local.exists():
        src.get(key, local)
    t = pq.read_table(local)
    if side is not None:
        if "image_cids" in t.column_names:
            t = t.drop(["image_cids"])
        if "uri" not in t.column_names:
            import pyarrow as pa
            t = t.append_column("uri", pa.array([L.post_uri(d, r) for d, r in zip(t["did"].to_pylist(), t["rkey"].to_pylist())]))
        import pyarrow as pa
        import pyarrow.compute as pc
        idx = pc.index_in(t["uri"], value_set=side["uri"])     # vectorised left join on uri (lists can't be join payload)
        cids = side["image_cids"].take(idx.fill_null(0))
        cids = pc.if_else(pc.is_null(idx), pa.nulls(len(t), cids.type), cids)
        t = t.append_column("image_cids", cids)
    stem = pathlib.PurePosixPath(key).stem
    for s in range(0, t.num_rows, unit_rows_n):
        yield f"{stem}-{s // unit_rows_n:05d}", t.slice(s, unit_rows_n).to_pylist()


def cmd_run(a, make_encoder=None):
    import pyarrow.parquet as pq
    from encoder import EG2, GpuSampler
    check_out(a)
    work = pathlib.Path(a.work)
    (work / "out").mkdir(parents=True, exist_ok=True)
    out = S.open_store(a.out)
    t_vm0 = a.vm_start_epoch or time.time()
    vm = f"{socket.gethostname()}-{int(t_vm0)}"   # unique per VM boot, so resumed VMs never overwrite spend records

    run_key = "_run.json"
    cfg = {"soft_tokens": a.soft_tokens, "unit_rows": a.unit_rows, "posts": a.posts, "sidecar": a.sidecar,
           "rps": a.rps, "text_prompt": "Clustering", "dim": L.DIM, "dtype": "bfloat16", "modalities": "text+vision"}
    if out.exists(run_key):
        run = json.loads(out.read_text(run_key))
        if {k: run["config"].get(k) for k in ("soft_tokens", "unit_rows", "posts", "sidecar")} != \
                {k: cfg[k] for k in ("soft_tokens", "unit_rows", "posts", "sidecar")}:
            sys.exit(f"config differs from the run's _run.json: {run['config']} vs {cfg}")
        log("resuming run", a.run_id, "customTime", run["custom_time"])
    else:
        run = {"run_id": a.run_id, "custom_time": L.iso(L.utcnow()), "config": cfg, "started_by_vm": vm}
    if hasattr(out, "custom_time"):
        out.custom_time = run["custom_time"]   # set once per run; every resumed VM reuses it
    if not out.exists(run_key):
        S.write_text(out, run_key, json.dumps(run, indent=1), work / "tmp")

    bundle = L.load_bundle(a.gate_bundle)
    max_lag = timedelta(hours=a.max_label_lag_h)
    if bundle.stale(clock_now(a), max_lag):
        log("label bundle stale or in the future:", bundle.lag(clock_now(a)))
        return EXIT_STALE_LABELS
    guard = L.DollarGuard(a.cap_usd or float("inf"), a.usd_per_hour, prior_spend(out, vm), t0=t_vm0)
    done = {pathlib.PurePosixPath(k).stem for k in S.list_keys(out, "_ledger/u-", ".json")}
    log(f"{len(done)} units already done; guard {guard.state()}")

    src, keys, side = load_inputs(a, work)
    enc = (make_encoder or EG2)(a.model_dir, a.device, a.soft_tokens, a.mps_gb)
    log(f"EG2 loaded on {enc.dev}: {enc.params_m:.1f}M params, {enc.param_dtypes}, soft tokens {enc.soft_tokens}")
    gpu = GpuSampler()
    fetcher = make_fetcher(a)
    totals = collections.Counter()
    unit_secs = []
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(60):
            log("heartbeat", json.dumps({"fetch": fetcher.stats.summary() | {"latency_ms": None}, "totals": dict(totals),
                                         "guard": guard.state(), "gpu": gpu.window(time.time() - 60)}, default=str))
    threading.Thread(target=heartbeat, daemon=True).start()

    def todo():
        for key in keys:
            for name, rows in unit_rows(src, key, work, side, a.unit_rows):
                if f"u-{name}" not in done:
                    yield name, rows

    marker = audit.start_marker(work)   # after model load: everything written from here on is audited
    pool = ThreadPoolExecutor(1, thread_name_prefix="fetch-unit")
    it = todo()
    rc, n_units = 0, 0
    try:
        cur = next(it, None)
        if cur is None:
            log("nothing to do")
        cur_u = P.prepare(*cur, bundle, clock_now(a)) if cur else None
        fut = pool.submit(fetcher.fetch_sync, cur_u.fetch_items()) if cur_u else None
        while cur_u is not None:
            t0 = time.time()
            est = (sum(unit_secs[-5:]) / len(unit_secs[-5:])) if unit_secs else 0
            if not guard.allows(est):
                log("dollar guard: next unit would cross the cap; stopping", guard.state())
                break
            if bundle.stale(clock_now(a), max_lag):
                log("label bundle went stale; stopping before", cur_u.name)
                rc = EXIT_STALE_LABELS
                break
            nxt = next(it, None) if (a.max_units == 0 or n_units + 1 < a.max_units) else None
            nxt_u = P.prepare(*nxt, bundle, clock_now(a)) if nxt else None
            g0 = time.time()
            try:
                fetched = fut.result()
            except CircuitOpen as e:
                log("fetch circuit breaker tripped:", e)
                rc = EXIT_CIRCUIT
                break
            fetch_wait = time.time() - g0
            if nxt_u and not guard.allows(2 * (est or (time.time() - t0))):
                log("dollar guard: not fetching the next unit; this is the last one", guard.state())
                nxt_u = None
            fut = pool.submit(fetcher.fetch_sync, nxt_u.fetch_items()) if nxt_u else None
            g1 = time.time()
            cols, counts = P.encode_unit(cur_u, fetched, enc, text_bs=a.text_bs, mm_bs=a.mm_bs)
            gpu_s = time.time() - g1
            local = work / "out" / f"u-{cur_u.name}.parquet"
            pq.write_table(P.to_table(cols), local, compression="zstd")
            chk = L.audit_output_table(local)
            if not chk["ok"]:
                log("output audit failed", chk)
                rc = EXIT_AUDIT
                break
            out.put(local, f"units/u-{cur_u.name}.parquet")
            rec = {"unit": cur_u.name, "rows": chk["rows"], "sha256": L.sha256_file(local), "counts": dict(counts),
                   "fetch_wait_s": round(fetch_wait, 2), "gpu_s": round(gpu_s, 2), "wall_s": round(time.time() - t0, 2),
                   "gpu": gpu.window(g1), "vm": vm, "at": L.iso(L.utcnow())}
            S.write_text(out, f"_ledger/u-{cur_u.name}.json", json.dumps(rec, sort_keys=True), work / "tmp")
            local.unlink()   # vectors are in the bucket; the local copy is not kept
            totals.update(counts)
            n_units += 1
            unit_secs.append(time.time() - t0)
            S.write_text(out, f"_ledger/vm-{vm}.json", json.dumps({"vm": vm, "usd_spent_this_vm": guard.spent() - guard.prior,
                                                                    "units": n_units, "at": L.iso(L.utcnow())}), work / "tmp")
            log(f"unit {cur_u.name}: {chk['rows']} posts, {counts['images_embedded']} images, "
                f"wait {fetch_wait:.0f}s gpu {gpu_s:.0f}s, fetch {fetcher.stats.rate() or 0:.1f} req/s, "
                f"gpu util {rec['gpu']['gpu_util_mean']}, {guard.state()['usd_spent_est']} USD")
            cur_u = nxt_u
    finally:
        stop.set()
        pool.shutdown(wait=True)
        fetcher.close()
        gpu.stop()
    rep = audit.finish(work, marker)
    summary = {"run_id": a.run_id, "vm": vm, "exit": rc, "units_this_vm": n_units, "totals": dict(totals),
               "fetch": fetcher.stats.summary(), "guard": guard.state(), "audit": rep,
               "label_watermark": L.iso(bundle.watermark), "gate_bundle_sha256": bundle.sha256}
    S.write_text(out, f"receipt/vm-{vm}.json", json.dumps(summary, indent=1, default=str), work / "tmp")
    log("receipt", json.dumps({k: v for k, v in summary.items() if k != "audit"} | {"audit_ok": rep["ok"]}, default=str))
    if not rep["ok"]:
        return EXIT_AUDIT
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    common_args(r)
    r.add_argument("--posts", required=True)
    r.add_argument("--sidecar")
    r.add_argument("--soft-tokens", type=int, default=70)
    r.add_argument("--unit-rows", type=int, default=20_000)
    r.add_argument("--max-units", type=int, default=0)
    b = sub.add_parser("bench")
    common_args(b)
    import bench
    bench.add_args(b)
    a = ap.parse_args(argv)
    if a.cmd == "run" and not a.gate_bundle:
        ap.error("run needs --gate-bundle")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return cmd_run(a) if a.cmd == "run" else bench.cmd_bench(a)


if __name__ == "__main__":
    sys.exit(main())
