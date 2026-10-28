#!/usr/bin/env python3
"""The us-west1 two-phase extraction pass (plan steps S8 and S9).

  passrun.py run  --mode s9 --stage /mnt/ssd/stage --work /mnt/ssd/work \\
      --store30 gs://BUCKET-30d --store60 gs://BUCKET-60d --run-id RUN \\
      --inputs gs://BUCKET-30d/inputs/RUN --rules RULES.json --rules-sha256 SHA \\
      --simgraph /opt/catbird/simgraph [--manifest manifest.tsv]
  passrun.py run  --mode s8 ...            (smoke test + projection for R-S9)

Every step's files go to the 30-day store (intermediates/<run>/...) or, for
O8/O9/the DID map, the 60-day store, followed by a ledger marker. A rerun with
the same --run-id (a resumed restart on a fresh Spot VM) skips every step whose
marker exists and downloads only the intermediates later steps read. The
bucketing step is local-only (post text never leaves the VM) and reruns on a
fresh VM.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import chrun  # noqa: E402
import rules as rules_mod  # noqa: E402
import sql_global  # noqa: E402
import sql_p1  # noqa: E402
import sql_p2  # noqa: E402
from store import Ledger, open_store, sha256_file  # noqa: E402

SIDE_REQUIRED = ("producers.tsv", "hide.tsv", "intents.tsv")
SIDE_OPTIONAL = ("status.tsv", "plc.tsv", "comm_cand1.tsv", "comm_cand5.tsv", "m2_cohort.parquet", "m2_cohort.tsv")
SPOT_PER_HOUR = 2.715  # plan §7.1: n2-highmem-64 Spot + 8 x 375 GiB local SSD + boot


def select_shards(dirs: list[str], k: int) -> list[str]:
    """k evenly spaced staged directories (all when k <= 0): S8's bucketing subset."""
    dirs = sorted(dirs)
    if k <= 0 or k >= len(dirs):
        return dirs
    return [dirs[(i * len(dirs)) // k] for i in range(k)]


def manifest_dirs(manifest: str) -> dict[str, int]:
    """Rows per top-level extract directory, from the manifest (path = /all|posts/<dir>/...)."""
    rows: dict[str, int] = {}
    for line in pathlib.Path(manifest).read_text().splitlines()[1:]:
        if line.strip():
            path, _b, n = line.split("\t")
            d = path.strip("/").split("/")[1]
            rows[d] = rows.get(d, 0) + int(n)
    return rows


class Pass:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.mode = a.mode
        self.rules = rules_mod.load(a.rules)
        if a.mode == "s9" and self.rules.draft and not a.allow_draft_rules:
            raise SystemExit("refusing S9 with a DRAFT rules file; S1 must commit the registered rules (or pass --allow-draft-rules)")
        if a.rules_sha256 and a.rules_sha256 != self.rules.sha256:
            raise SystemExit(f"rules sha256 mismatch: file {self.rules.sha256} != --rules-sha256 {a.rules_sha256}")
        if self.rules.N != 16 and not a.test_mode:
            raise SystemExit("the plan registers 16 partitions per phase")
        self.stage = pathlib.Path(a.stage).resolve()
        self.work = pathlib.Path(a.work).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.s30 = open_store(a.store30)
        # customTime = first write; S5r copies it on rewrite, so a rewrite never extends an object's life
        self.s60 = open_store(a.store60, custom_time=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.run_id = a.run_id
        self.ledger = Ledger(self.s30, self.run_id)
        self.local_ledger = self.work / "_local_ledger"
        self.local_ledger.mkdir(exist_ok=True)
        self.stats: dict[str, dict] = {}
        self.fail_after = os.environ.get("VMPASS_FAIL_AFTER")  # tests: simulate a preemption after this step
        self.t_start = time.time()
        self.out30 = self.work / "out30"
        self.out60 = self.work / "out60"
        self.side = self.work / "side"
        self.g = self.work / "g"
        self.p1dir = self.work / "p1"
        self.p2dir = self.work / "p2"
        self.buckets = self.work / "buckets"
        self.have: dict[str, bool] = {}
        import threading
        self._log_lock = threading.Lock()
        self.progress_log = self.work / "progress.log"
        self._last_log_upload = 0.0
        self._n_shards = 0
        self.hb_s = float(os.environ.get("VMPASS_HEARTBEAT_S", "120"))
        self.subset_fraction = 1.0
        self.bucket_shards = int(getattr(a, "bucket_shards", 0) or 0)
        if self.bucket_shards and a.mode != "s8":
            raise SystemExit("--bucket-shards is an S8 measurement option; S9 buckets every staged file")
        self.log_upload_s = float(os.environ.get("VMPASS_LOG_UPLOAD_S", "180"))

    # ---------- ledger plumbing ----------
    def _key30(self, rel: str) -> str:
        return f"intermediates/{self.run_id}/{rel}"

    def _okey(self, rel: str) -> str:
        return f"outputs/{self.run_id}/{rel}"

    def step(self, name: str, fn, local_only: bool = False, uploads: tuple = ()) -> dict:
        """Run fn() unless a marker says the step finished. `uploads` lists local
        directories (relative to work) whose new files belong to this step."""
        marker = self.local_ledger / f"{name}.json" if local_only else None
        if local_only:
            if marker.exists():
                rec = json.loads(marker.read_text())
                if all((self.work / f).exists() for f in rec["files"]):
                    self.stats[name] = {**rec["stats"], "skipped": True}
                    if name == "bucket":
                        self.subset_fraction = float(rec["stats"].get("subset_fraction", 1.0))
                    return rec
        else:
            rec = self.ledger.done(name)
            if rec:
                self.stats[name] = {**rec["stats"], "skipped": True}
                return rec
        t0 = time.time()
        self.log(f"step {name}: start")
        try:
            stats = fn() or {}
        except Exception as e:
            self.report_failure(name, e)
            raise
        self.log(f"step {name}: done in {time.time() - t0:.0f}s" + (
            f", peak RSS {stats['peak_rss_bytes'] / 2**30:.1f} GiB" if isinstance(stats.get("peak_rss_bytes"), (int, float)) else ""))
        files = []
        for d in uploads:
            base = self.work / d
            for p in sorted(base.rglob("*")) if base.exists() else []:
                if p.is_file() and p.stat().st_mtime >= t0 - 1:
                    rel = str(p.relative_to(self.work))
                    files.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        if not local_only:
            self._upload([f["path"] for f in files])
        rec = {"step": name, "files": [f["path"] for f in files], "manifest": files, "stats": stats,
               "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "rules_sha256": self.rules.sha256}
        if local_only:
            marker.write_text(json.dumps(rec, indent=1))
        else:
            self.ledger.mark(name, rec)
        self.stats[name] = stats
        if self.mode == "s8" and name not in ("projection", "finalize"):
            self.incremental_projection(after=name)
        self.flush_log(force=True)
        if self.fail_after and self.fail_after == name:
            raise SystemExit(f"VMPASS_FAIL_AFTER={name}: simulated preemption")
        return rec

    def _remote(self, rel: str):
        """(store, key) for a work-relative path."""
        if rel.startswith("out60/"):
            return self.s60, self._okey(rel[len("out60/"):])
        if rel.startswith("out30/"):
            return self.s30, self._okey(rel[len("out30/"):])
        if rel.startswith("g/didmap"):
            return self.s60, f"didmap/{self.run_id}/{pathlib.Path(rel).name}"
        return self.s30, self._key30(rel)

    def _upload(self, rels: list[str]) -> None:
        by: dict = {}
        for rel in rels:
            st, key = self._remote(rel)
            by.setdefault(id(st), (st, []))[1].append((self.work / rel, key))
        for st, pairs in by.values():
            st.put_files(pairs)

    def ensure(self, step: str) -> None:
        """Make the files a finished step wrote present locally (fresh-VM resume),
        one download call per directory."""
        rec = self.ledger.done(step)
        if not rec:
            return
        by: dict = {}
        for rel in rec["files"]:
            p = self.work / rel
            if not p.exists():
                st, key = self._remote(rel)
                by.setdefault(id(st), (st, []))[1].append((key, p))
        for st, pairs in by.values():
            st.get_files(pairs)

    def sql(self, name: str, text: str, max_threads: int | None = None) -> dict:
        r = chrun.run_script(text, self.work, name, heartbeat=self.heartbeat, heartbeat_s=self.hb_s, max_threads=max_threads,
                             marks=True)
        slow = sorted(r["statements"], key=lambda x: -x["seconds"])[:8]
        for st in slow[:5]:
            self.log(f"  {name} slowest: {st['seconds']:.1f}s stmt {st['i'] + 1}: {st['what'][:80]}")
        return {"seconds": r["seconds"], "peak_rss_bytes": r["peak_rss_bytes"], "mean_cpu_pct": r["mean_cpu_pct"],
                "clickhouse_progress": r["last_progress"], "slowest_statements": slow, "stdout": r["stdout"][-2000:]}

    # ---------- progress log (streamed to the 30-day store, so an overrun still shows where time went) ----------
    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] +{time.time() - self.t_start:.0f}s {msg}"
        print(line, file=sys.stderr, flush=True)
        with self._log_lock:
            with open(self.progress_log, "a") as fh:
                fh.write(line + "\n")
        self.flush_log()

    def report_failure(self, name: str, err: Exception) -> None:
        """A failed step: its error into the progress log, and every ClickHouse stderr it
        wrote (progress redraws removed, plus the raw tail) to intermediates/<run>/_logs/."""
        self.log(f"step {name}: FAILED: {str(err)[:6000]}")
        logs = self.work / "logs"
        for f in sorted(logs.glob(f"{name}*.stderr")) if logs.exists() else []:
            try:
                clean = chrun.stderr_text(f)
                out = logs / (f.name + ".txt")
                out.write_text(clean)
                self.s30.put(out, self._key30(f"_logs/{out.name}"))
                self.log(f"  {f.name} ({f.stat().st_size} bytes): {clean[-3000:]}")
            except Exception as e2:  # noqa: BLE001
                self.log(f"  could not upload {f.name}: {e2!r}")
        self.flush_log(force=True)

    def heartbeat(self, name: str, info: dict) -> None:
        extra = ""
        if name.startswith("bucket"):
            done = len(list((self.buckets / "_done").glob("*.json"))) if (self.buckets / "_done").exists() else 0
            extra = f" shards_done={done}/{self._n_shards}"
        self.log(f"  {name}: {info['elapsed_s']}s rss={info['rss_gb']}GiB cpu={info['cpu_pct']}%{extra} "
                 f"{info.get('statement') or ''} ch: {info['clickhouse_progress'] or '-'}")

    def flush_log(self, force: bool = False) -> None:
        if not force and time.time() - self._last_log_upload < self.log_upload_s:
            return
        self._last_log_upload = time.time()
        try:
            self.s30.put(self.progress_log, self._key30("_logs/progress.log"))
        except Exception as e:  # noqa: BLE001 - logging must never stop the pass
            print(f"progress-log upload failed: {e}", file=sys.stderr, flush=True)

    def incremental_projection(self, after: str) -> None:
        """S8: rewrite S8/projection.json after every finished step with what is known
        so far, so a max-run kill still leaves a partial projection in the store."""
        import projection
        try:
            proj = projection.incremental(self, after)
            d = self.out30 / "S8"
            d.mkdir(parents=True, exist_ok=True)
            (d / "projection.json").write_text(json.dumps(proj, indent=1, sort_keys=True, default=str))
            (d / "projection.txt").write_text(projection.render_partial(proj))
            self._upload(["out30/S8/projection.json", "out30/S8/projection.txt"])
        except Exception as e:  # noqa: BLE001
            self.log(f"incremental projection after {after} failed: {e!r}"[:500])

    # ---------- sharded bucketing ----------
    def bucket_step(self, actor_filter: str) -> dict:
        """One clickhouse-local process per top-level staged directory, `jobs` at a time."""
        staged = sorted({p.relative_to(self.stage).parts[0] for p in self.stage.glob("*/**/*.parquet")})
        shards = select_shards(staged, self.bucket_shards)
        if self.a.manifest:
            per = manifest_dirs(self.a.manifest)
            self.subset_fraction = sum(per.get(s, 0) for s in shards) / max(1, sum(per.values()))
        else:
            self.subset_fraction = len(shards) / max(1, len(staged))
        if len(shards) < len(staged) or self.subset_fraction < 1:
            self.log(f"bucket: S8 subset of {len(shards)}/{len(staged)} staged dirs, f = {self.subset_fraction:.4f} of manifest rows")
        self._n_shards = len(shards)
        donedir = self.buckets / "_done"
        donedir.mkdir(parents=True, exist_ok=True)
        todo = [s for s in shards if not (donedir / f"{s}.json").exists()]
        ncpu = os.cpu_count() or 8
        cfg = self.rules.d.get("bucket", {})
        jobs = int(cfg.get("jobs") or max(1, min(len(todo) or 1, ncpu // int(cfg.get("threads_per_job", 5)))))
        threads = int(cfg.get("threads_per_job") or max(2, ncpu // jobs))
        self.log(f"bucket: {len(shards)} shards ({len(todo)} to do), {jobs} parallel x {threads} threads")
        t0 = time.time()

        def done(name: str, st: dict) -> None:
            shard = name[len("bucket_"):]
            (donedir / f"{shard}.json").write_text(json.dumps({k: v for k, v in st.items() if k != "stdout"}))
            n = len(list(donedir.glob("*.json")))
            self.log(f"bucket shard {shard} done in {st['seconds']:.0f}s ({n}/{len(shards)} shards, {time.time() - t0:.0f}s elapsed)")
            if self.mode == "s8" and (n == len(shards) or n % max(1, len(shards) // 8) == 0):
                self.incremental_projection(after=f"bucket {n}/{len(shards)}")

        dm = str(self.g / "didmap.parquet")
        sqls = [(f"bucket_{s}", sql_global.bucket(self.rules, str(self.stage), dm, str(self.buckets), actor_filter, shard=s)) for s in todo]
        res = chrun.run_many(sqls, self.work, jobs, threads, on_done=done, heartbeat=self.heartbeat, heartbeat_s=self.hb_s)
        per = [json.loads(p.read_text()) for p in donedir.glob("*.json")]
        return {"seconds": round(time.time() - t0, 3), "shards": len(shards), "jobs": jobs, "threads_per_job": threads,
                "subset_fraction": self.subset_fraction, "shard_names": shards,
                "peak_rss_bytes": sum(sorted((x["peak_rss_bytes"] for x in per), reverse=True)[:jobs]),
                "shard_seconds_sum": round(sum(x["seconds"] for x in per), 1), "ran_now": len(res)}

    # ---------- inputs ----------
    def fetch_inputs(self) -> dict:
        self.side.mkdir(parents=True, exist_ok=True)
        src = self.a.inputs
        got = {}
        for name in SIDE_REQUIRED + SIDE_OPTIONAL:
            dst = self.side / name
            if src.startswith("gs://"):
                r = subprocess.run(["gcloud", "storage", "cp", "--quiet", f"{src.rstrip('/')}/{name}", str(dst)], capture_output=True, text=True)
                ok = r.returncode == 0
            else:
                s = pathlib.Path(src) / name
                ok = s.exists()
                if ok:
                    shutil.copyfile(s, dst)
            if not ok and name in SIDE_REQUIRED:
                raise SystemExit(f"missing required input {name} under {src}")
            if ok:
                got[name] = sha256_file(dst)
        return {"inputs_sha256": got}

    def detect_inputs(self) -> None:
        self.have = {"status": (self.side / "status.tsv").exists() and (self.side / "status.tsv").stat().st_size > 0,
                     "plc": (self.side / "plc.tsv").exists() and (self.side / "plc.tsv").stat().st_size > 0,
                     "maps": (self.side / "comm_cand1.tsv").exists() and (self.side / "comm_cand5.tsv").exists(),
                     "m2_parquet": (self.side / "m2_cohort.parquet").exists(),
                     "m2_tsv": (self.side / "m2_cohort.tsv").exists() and (self.side / "m2_cohort.tsv").stat().st_size > 0}

    # ---------- steps ----------
    def run(self) -> dict:
        r = self.rules
        N = r.N
        s8 = self.mode == "s8"
        self.step("inputs", self.fetch_inputs, uploads=("side",))
        self.ensure("inputs")
        self.detect_inputs()
        self.step("inventory", lambda: self.inventory(), uploads=("inv",))
        self.ensure("inventory")
        if s8:
            self.step("read_rate", lambda: self.sql("read_rate", sql_global.read_rate(str(self.stage))))
            self.step("hash_crosscheck", self.hash_crosscheck, uploads=("out30/S8",))
        self.step("didmap", lambda: self.sql("didmap", sql_global.didmap(r, str(self.stage), str(self.side), str(self.g), self.have)),
                  uploads=("g",))
        self.ensure("didmap")
        hf = "cityHash64(ifNull(did, ''))"
        actor_filter = f"({hf} % {r['samples']['s8_actor_mod'][0]} = 0 OR {hf} % {N} = 0)" if s8 else "1"
        self.step("bucket", lambda: self.bucket_step(actor_filter), local_only=True, uploads=("buckets",))
        p1opts = {"have_maps": self.have["maps"]}
        if s8:  # O3 sizing: every sampled actor stands in for a fit / M1 user
            p1opts.update(fit_flag="s200", m1_flag="s200")
        final1 = {"o5raw": str(self.out30 / "O5-raw"), "o3b": str(self.out30 / "O3b"), "o3c": str(self.out30 / "O3c"),
                  "pc": str(self.out30 / "positive-control")}
        for i in range(N):
            name = f"p1_{i:02d}"
            o = self.p1dir / f"i{i:02d}"
            self.step(name, lambda i=i, o=o, name=name: self._p1(name, i, o, final1, p1opts),
                uploads=(f"p1/i{i:02d}", "out30/O5-raw", "out30/O3b", "out30/O3c", "out30/positive-control"))
        for i in range(N):
            self.ensure(f"p1_{i:02d}")
        self.step("ustats_merge", lambda: self.sql("ustats_merge", sql_global.ustats_merge(r, str(self.p1dir), str(self.g))),
                  uploads=("g",))
        self.step("gates_merge", lambda: self.sql("gates_merge", sql_global.gates_merge(r, str(self.p1dir), str(self.g / 'didmap.parquet'), str(self.g))),
                  uploads=("g",))
        self.step("lblk", lambda: self.sql("lblk", sql_global.lblk(r, str(self.p1dir), str(self.g), str(self.out30 / "O5-raw"))),
                  uploads=("g", "out30/O5-raw"))
        for s in ("ustats_merge", "gates_merge", "lblk"):
            self.ensure(s)
        self.step("o5prod", lambda: self.sql("o5prod", sql_global.o5prod(r, str(self.p1dir), str(self.g / "gates"), str(self.g / "ustats.parquet"),
                                                                          str(self.side), str(self.out30 / "O5-prod"), self.have["maps"])),
                  uploads=("out30/O5-prod",))
        self.step("o7", lambda: self.sql("o7", sql_global.o7_final(r, str(self.p1dir), str(self.out30 / "O7"))), uploads=("out30/O7",))
        final2 = {"o2": str(self.out30 / "O2"), "o3a": str(self.out30 / "O3a"), "o3b": str(self.out30 / "O3b"), "o6": str(self.out30 / "O6"),
                  "o8": str(self.out60 / "O8"), "o9": str(self.out60 / "O9"), "o11": str(self.out30 / "O11")}
        p2opts = {"fit_flag": p1opts.get("fit_flag", "fit"), "o13_flag": "s200" if s8 else "o13m1"}
        if self.a.o3b_scales:  # S1 R4: p99.9 of |x_k| on S8's sample, hashed into R-S9
            p2opts["o3b_scales"] = json.loads(pathlib.Path(self.a.o3b_scales).read_text())
            self.stats["o3b_scales"] = {"path": self.a.o3b_scales, "sha256": sha256_file(pathlib.Path(self.a.o3b_scales))}
        if s8:
            runs = [("s400", 0), ("s200", 0)]
        else:
            runs = [(None, j) for j in range(N)]
        p2names = []
        for samp, j in runs:
            tag = f"p2{'_' + samp if samp else ''}_{j:02d}"
            sub = f"p2{'_' + samp if samp else ''}"
            o = self.work / sub / f"j{j:02d}"
            fin = {k: (v if not samp else v.replace("/out30/", f"/out30/S8-{samp}/").replace("/out60/", f"/out60/S8-{samp}/")) for k, v in final2.items()}
            opts = {**p2opts, "inbound_sample": samp}
            self.step(tag, lambda j=j, o=o, fin=fin, opts=opts, tag=tag: self._fp16(fin, j, (self._clean(o), self.sql(tag, sql_p2.p2(
                r, j, str(self.p1dir), str(self.g), str(self.buckets), str(o), fin, opts)))[1]),
                uploads=(f"{sub}/j{j:02d}", "out30", "out60"))
            p2names.append((tag, sub))
        main_sub = "p2_s200" if s8 else "p2"
        for tag, sub in p2names:
            if sub == main_sub:
                self.ensure(tag)
        self.step("maint", lambda: self.sql("maint", sql_global.maint_shuffle(r, str(self.work / main_sub), str(self.out30 / "O11-side")))
                  | self._o11_users(), uploads=("out30/O11-side",))
        self.step("o13", lambda: self.sql("o13", sql_global.o13_merge(r, str(self.work / main_sub), str(self.out60 / "O13"))),
                  uploads=("out60/O13",))
        self.step("teacher", lambda: self.teacher(main_sub), uploads=("out30/O4",))
        if r["o12"]["emit"]:
            self.step("o12", lambda: self.o12(main_sub), uploads=("out30/O12",))
        if s8:
            self.step("projection", lambda: self.projection(), uploads=("out30/S8",))
        self.step("finalize", self.finalize, uploads=("out30",))
        return self.stats

    def p1_chunks(self, i: int) -> tuple[int, int]:
        """Actor sub-chunks for phase-1 partition i, sized from the machine's RAM: the dedupe
        GROUP BY measured ~350 B of memory per input row (synthetic, 96M rows -> 33 GiB), so a
        chunk holds about 15% of RAM worth of rows, under the 20% external-GROUP-BY threshold."""
        from sqlc import ram_bytes
        rows = 0
        for kind in ("ev", "po"):
            pat = self.buckets / kind / f"p{i}" / "*.parquet"
            if glob.glob(str(pat)):
                rows += int(chrun.query(f"SELECT count() FROM file('{pat}', Parquet) FORMAT TSV", self.work, f"p1rows_{i}").strip() or 0)
        per = int(os.environ.get("VMPASS_P1_ROWS_PER_CHUNK") or max(1_000_000, int(ram_bytes() * 0.15 / 350)))
        return rows, max(1, -(-rows // per))

    def _p1(self, name: str, i: int, o: pathlib.Path, final1: dict, p1opts: dict) -> dict:
        self._clean(o)
        rows, k = self.p1_chunks(i)
        self.log(f"{name}: {rows} bucket rows -> {k} actor sub-chunk(s)")
        st = self.sql(name, sql_p1.p1(self.rules, i, str(self.buckets / f"ev/p{i}/*.parquet"), str(self.buckets / f"po/p{i}/*.parquet"),
                                      str(self.g / "didmap.parquet"), str(self.side), str(o), final1, {**p1opts, "chunks": k}))
        return {**st, "bucket_rows": rows, "chunks": k}

    def _clean(self, d: pathlib.Path) -> None:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    def _fp16(self, fin: dict, j: int, stats: dict) -> dict:
        """Store O2's producer sums and O3a's features as fp16 (plan O2/O3a byte budgets)."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        files = glob.glob(f"{fin['o2']}/*/j{j:02d}.parquet") + glob.glob(f"{fin['o3a']}/*/j{j:02d}.parquet")
        keep = {"pi", "offset"}
        for f in files:
            t = pq.read_table(f)
            o2 = "/O2/" in f
            cols = []
            for name, col in zip(t.column_names, t.columns):
                if pa.types.is_float32(col.type) and ((o2 and name.endswith(("__a05", "__a10"))) or (not o2 and name not in keep)):
                    col = col.cast(pa.float16())
                cols.append(col)
            pq.write_table(pa.Table.from_arrays(cols, names=t.column_names), f, compression="zstd")
        return stats

    def _o11_users(self) -> dict:
        self.sql("o11_users", sql_global.o11_users(self.rules, str(self.g / "ustats.parquet"), str(self.out30 / "O11-side")))
        return {}

    def inventory(self) -> dict:
        st = self.sql("inventory", sql_global.inventory(self.rules, str(self.stage), str(self.work / "inv")))
        q = chrun.query(f"""SELECT kind, count(), sum(rows) FROM (SELECT * FROM file('{self.work}/inv/files_ev.parquet') UNION ALL
            SELECT * FROM file('{self.work}/inv/files_po.parquet')) GROUP BY kind FORMAT TSV""", self.work, "inv_sum")
        tot = {}
        for line in q.strip().splitlines():
            k, n, rows = line.split("\t")
            tot[k] = {"files": int(n), "rows": int(rows)}
        rec = {**st, "staged": tot}
        if self.a.manifest:
            man = {"events": {"files": 0, "rows": 0}, "posts": {"files": 0, "rows": 0}}
            per = {}
            for line in pathlib.Path(self.a.manifest).read_text().splitlines()[1:]:
                if not line.strip():
                    continue
                path, _b, rows = line.split("\t")
                name = path.rsplit("/", 1)[1]
                k = "posts" if name.startswith("posts-") else "events"
                man[k]["files"] += 1
                man[k]["rows"] += int(rows)
                per[name] = int(rows)
            got = {}
            for f in ("files_ev", "files_po"):
                out = chrun.query(f"SELECT file, rows FROM file('{self.work}/inv/{f}.parquet') FORMAT TSV", self.work, "inv_files")
                for line in out.strip().splitlines():
                    n, rows = line.split("\t")
                    got[n] = int(rows)
            mism = [n for n in per if got.get(n) != per[n]]
            rec["manifest"] = man
            rec["check_a_rows_equal_manifest"] = (man == tot and not mism)
            rec["files_mismatched"] = mism[:50]
            rec["files_missing"] = len([n for n in per if n not in got])
        return rec

    def hash_crosscheck(self) -> dict:
        """S8: 1,000 synthetic DID/URI-shaped strings hashed by this VM's clickhouse;
        crosscheck_hash.py regenerates them on the Mac and compares."""
        import crosscheck_hash
        d = self.out30 / "S8"
        d.mkdir(parents=True, exist_ok=True)
        vals = crosscheck_hash.strings()
        (self.work / "xc_in.txt").write_text("\n".join(vals) + "\n")
        out = subprocess.run([chrun.CLICKHOUSE, "local", "--input-format", "LineAsString", "--structure", "s String",
                              "-q", "SELECT s, cityHash64(s) FROM table FORMAT TabSeparatedRaw"],
                             input=(self.work / "xc_in.txt").read_bytes(), capture_output=True, check=True).stdout.decode()
        (d / "cityhash64_crosscheck.tsv").write_text(out)
        ver = subprocess.run([chrun.CLICKHOUSE, "local", "--version"], capture_output=True, text=True).stdout.strip()
        mism = crosscheck_hash.compare_text(out)
        return {"values": len(out.splitlines()), "python_mismatches_on_vm": mism, "clickhouse_version": ver}

    def teacher(self, sub: str) -> dict:
        """O4: E0 -> simgraph (built from source on the VM) -> top-200 neighbours per
        producer with >= 50 light engagers, for pre-T and production, with and without Qn."""
        r = self.rules
        t = r["teacher"]
        dst = self.out30 / "O4"
        dst.mkdir(parents=True, exist_ok=True)
        stats = {}
        np_ = int(chrun.query(f"SELECT count() FROM file('{self.g}/didmap.parquet') WHERE is_prod = 1 FORMAT TSV", self.work, "np").strip())
        e0 = f"{self.work}/{sub}/j*/e0_*.parquet"
        if not glob.glob(e0):
            return {"skipped": "no E0 edges"}
        for tag, cidx in (("preT", r.cut_names.index("T")), ("prod", r.cut_names.index("T_now"))):
            for var in ("withqn", "noqn"):
                d = self.work / "teacher" / f"{tag}_{var}"
                self._clean(d)
                chrun.run_script(f"""
SELECT u, p FROM file('{e0}', Parquet) WHERE cidx = {cidx} AND {var} = 1 ORDER BY u, p
INTO OUTFILE '{d}/edges.bin' TRUNCATE FORMAT RowBinary;
SELECT n.number AS pid, '-' AS did, toUInt32(ifNull(c.deg, 0)) AS deg FROM numbers({np_}) AS n
LEFT JOIN (SELECT p, uniqExact(u) AS deg FROM file('{e0}', Parquet) WHERE cidx = {cidx} AND {var} = 1 GROUP BY p) AS c ON c.p = n.number
ORDER BY pid INTO OUTFILE '{d}/producers.tsv' TRUNCATE FORMAT TSV;""", self.work, f"teacher_in_{tag}_{var}")
                if not self.a.simgraph or not pathlib.Path(self.a.simgraph).exists():
                    stats[f"{tag}_{var}"] = {"skipped": "simgraph binary not given"}
                    continue
                t0 = time.time()
                res = subprocess.run([self.a.simgraph, "--dir", str(d), "--sample", str(t["sample"]), "--cap", str(t["cap"]),
                                      "--min-cos", str(t["min_cos"]), "--top-k", str(t["top_k"]), "--seed", str(t["seed"]),
                                      "--follower-idf-alpha", str(t["follower_idf_alpha"])], capture_output=True, text=True)
                if res.returncode != 0:
                    raise chrun.StepError(f"simgraph failed: {res.stderr[-2000:]}")
                chrun.run_script(f"""
{sql_global.out(str(dst / f'{tag}_{var}.parquet'))}
SELECT s.p AS p, s.q AS q, s.w AS w FROM file('{d}/sim.bin', RowBinary, 'p UInt32, q UInt32, w Float32') AS s
INNER JOIN (SELECT pid, deg FROM file('{d}/producers.tsv', TSV, 'pid UInt32, did String, deg UInt32')) AS pr ON pr.pid = s.p
WHERE pr.deg >= {t['min_light_engagers']} ORDER BY p, w DESC;""", self.work, f"teacher_out_{tag}_{var}")
                stats[f"{tag}_{var}"] = {"seconds": round(time.time() - t0, 2), "simgraph": res.stdout.strip()[-400:]}
        return stats

    def o12(self, sub: str) -> dict:
        r = self.rules
        per = r["o12"]["per_stratum"]
        nq = r["o12"]["random_nonfollower_quotes"]
        d = self.out30 / "O12"
        d.mkdir(parents=True, exist_ok=True)
        cand = f"{self.work}/{sub}/j*/o12c.parquet"
        if not glob.glob(cand):
            return {"skipped": "no candidates"}
        st = self.sql("o12", sql_global.PRELUDE + f"""
CREATE TABLE c ENGINE = Memory AS SELECT * FROM (
  SELECT *, row_number() OVER (PARTITION BY stratum ORDER BY hsh) AS rn FROM file('{cand}', Parquet, '{sql_global.S['o12c']}'))
  WHERE rn <= if(stratum = 6, {nq}, {per});
CREATE TABLE dmd ENGINE = Join(ANY, LEFT, uid) AS SELECT uid, did FROM file('{self.g}/didmap.parquet', Parquet, '{sql_global.S['didmap']}')
  WHERE uid IN (SELECT c_a FROM c UNION ALL SELECT x_a FROM c);
CREATE TABLE tx ENGINE = Join(ANY, LEFT, ph) AS SELECT ph, argMax(text, (rev, seq)) AS text, argMax(rkey, (rev, seq)) AS rkey
  FROM file('{self.buckets}/po/p*/*.parquet', Parquet, '{sql_global.S['po']}') WHERE op != 3 AND ph IN (SELECT c_ph FROM c UNION ALL SELECT x_ph FROM c) GROUP BY ph;
{sql_global.out(str(d / 'pairs.parquet'))}
SELECT transform(stratum, [1, 2, 3, 4, 5], ['detached', 'block_after', 'non_follower', 'follower', 'mutual'], '') AS stratum,
  if(kind = 1, 'reply', 'quote') AS kind,
  concat('at://', joinGet('dmd', 'did', c_a), '/app.bsky.feed.post/', joinGet('tx', 'rkey', c_ph)) AS child_uri, joinGet('tx', 'text', c_ph) AS child_text,
  concat('at://', joinGet('dmd', 'did', x_a), '/app.bsky.feed.post/', joinGet('tx', 'rkey', x_ph)) AS parent_uri, joinGet('tx', 'text', x_ph) AS parent_text
FROM c WHERE c.stratum != 6 AND joinGet('tx', 'rkey', c_ph) != '' AND joinGet('tx', 'rkey', x_ph) != '';
{sql_global.out(str(d / 'random_nonfollower_quotes.parquet'))}
SELECT concat('at://', joinGet('dmd', 'did', c_a), '/app.bsky.feed.post/', joinGet('tx', 'rkey', c_ph)) AS uri FROM c
WHERE stratum = 6 AND joinGet('tx', 'rkey', c_ph) != '';
""")
        return st

    # ---------- S8 projection and finalize ----------
    def projection(self) -> dict:
        import projection
        proj = {"partial": False, "after_step": "projection", "known": projection._known(self), **projection.project(self)}
        d = self.out30 / "S8"
        d.mkdir(parents=True, exist_ok=True)
        (d / "projection.json").write_text(json.dumps(proj, indent=1, sort_keys=True))
        (d / "projection.txt").write_text(projection.render(proj))
        return {"projection": proj}

    def finalize(self) -> dict:
        import finalize
        return finalize.finalize(self)


def selfcheck(a: argparse.Namespace) -> int:
    """Preflight on the VM's own interpreter, ClickHouse and simgraph: build the synthetic
    fixture world and run the pass end to end (local stores), so a syntax, import, SQL or
    tooling error fails in ~1-2 min instead of after 10 min of staging."""
    t0 = time.time()
    root = pathlib.Path(a.work).resolve()
    if root.exists():
        shutil.rmtree(root)
    sys.path.insert(0, str(HERE / "tests"))
    import fixtures  # noqa: E402 - test fixture generator, shipped in the code tarball
    for mod in ("chrun", "cityhash", "crosscheck_hash", "finalize", "projection", "rules", "sql_global", "sql_p1", "sql_p2", "sqlc", "store"):
        __import__(mod)
    fixtures.build(root / "stage", root / "side")
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "run", "--mode", a.mode, "--stage", str(root / "stage"),
           "--work", str(root / "work"), "--store30", str(root / "s30"), "--store60", str(root / "s60"), "--run-id", "selfcheck",
           "--inputs", str(root / "side"), "--manifest", str(root / "stage" / "manifest.tsv"), "--rules", a.rules, "--allow-draft-rules"]
    if a.bucket_shards:
        cmd += ["--bucket-shards", str(a.bucket_shards)]
    if a.simgraph:
        cmd += ["--simgraph", a.simgraph]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "VMPASS_HEARTBEAT_S": "5"})
    ok = r.returncode == 0
    want = root / "s30" / "outputs" / "selfcheck" / ("S8/projection.json" if a.mode == "s8" else "receipt/receipt.json")
    ok = ok and want.exists()
    print(f"selfcheck {'OK' if ok else 'FAILED'}: python {sys.version.split()[0]}, mode {a.mode}, {time.time() - t0:.0f}s", flush=True)
    if not ok:
        print(r.stderr[-6000:], file=sys.stderr, flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--mode", choices=("s8", "s9"), required=True)
    p.add_argument("--stage", required=True, help="local dir holding the staged v2-*/events/{events,posts}-*.parquet")
    p.add_argument("--work", required=True, help="local SSD work dir")
    p.add_argument("--store30", required=True, help="gs://plan-30d bucket (or a local dir in tests)")
    p.add_argument("--store60", required=True, help="gs://plan-60d bucket (or a local dir in tests)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--inputs", required=True, help="dir or gs:// prefix with producers.tsv, hide.tsv, intents.tsv [status.tsv plc.tsv comm_cand{1,5}.tsv m2_cohort.parquet or .tsv]")
    p.add_argument("--rules", default=str(rules_mod.DRAFT))
    p.add_argument("--rules-sha256", default="")
    p.add_argument("--allow-draft-rules", action="store_true")
    p.add_argument("--manifest", default="")
    p.add_argument("--simgraph", default="")
    p.add_argument("--o3b-scales", default="", help="JSON {feature: scale} from S8's projection (S1 R4)")
    p.add_argument("--bucket-shards", type=int, default=0,
                   help="S8 only: bucket K evenly spaced staged directories and scale the projection by the row fraction")
    p.add_argument("--stage-seconds", type=float, default=0, help="staging wall time from the bootstrap, for the S8 projection")
    p.add_argument("--test-mode", action="store_true", help="allow a non-16 partition count (tests only)")
    sp = sub.add_parser("shards", help="print the staged directories --bucket-shards K selects (for the VM bootstrap)")
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--k", type=int, required=True)
    sc = sub.add_parser("selfcheck", help="run the whole pass on the synthetic fixture world (VM preflight, before staging)")
    sc.add_argument("--work", required=True, help="empty scratch dir")
    sc.add_argument("--mode", choices=("s8", "s9"), default="s8")
    sc.add_argument("--rules", default=str(rules_mod.S1))
    sc.add_argument("--bucket-shards", type=int, default=0)
    sc.add_argument("--simgraph", default="")
    a = ap.parse_args()
    if a.cmd == "shards":
        print("\n".join(select_shards(list(manifest_dirs(a.manifest)), a.k)))
        return 0
    if a.cmd == "selfcheck":
        return selfcheck(a)
    stats = Pass(a).run()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "stdout"} for k, v in stats.items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
