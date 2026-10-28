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
                    return rec
        else:
            rec = self.ledger.done(name)
            if rec:
                self.stats[name] = {**rec["stats"], "skipped": True}
                return rec
        t0 = time.time()
        stats = fn() or {}
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

    def sql(self, name: str, text: str) -> dict:
        r = chrun.run_script(text, self.work, name)
        return {"seconds": r["seconds"], "peak_rss_bytes": r["peak_rss_bytes"], "stdout": r["stdout"][-2000:]}

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
        self.step("bucket", lambda: self.sql("bucket", sql_global.bucket(r, str(self.stage), str(self.g / 'didmap.parquet'),
                                                                          str(self.buckets), actor_filter)),
                  local_only=True, uploads=("buckets",))
        p1opts = {"have_maps": self.have["maps"]}
        if s8:  # O3 sizing: every sampled actor stands in for a fit / M1 user
            p1opts.update(fit_flag="s200", m1_flag="s200")
        final1 = {"o5raw": str(self.out30 / "O5-raw"), "o3b": str(self.out30 / "O3b"), "o3c": str(self.out30 / "O3c"),
                  "pc": str(self.out30 / "positive-control")}
        for i in range(N):
            name = f"p1_{i:02d}"
            o = self.p1dir / f"i{i:02d}"
            self.step(name, lambda i=i, o=o: (self._clean(o), self.sql(name, sql_p1.p1(
                r, i, str(self.buckets / f"ev/p{i}.parquet"), str(self.buckets / f"po/p{i}.parquet"), str(self.g / "didmap.parquet"),
                str(self.side), str(o), final1, p1opts)))[1],
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
  FROM file('{self.buckets}/po/p*.parquet', Parquet, '{sql_global.S['po']}') WHERE op != 3 AND ph IN (SELECT c_ph FROM c UNION ALL SELECT x_ph FROM c) GROUP BY ph;
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
        proj = projection.project(self)
        d = self.out30 / "S8"
        d.mkdir(parents=True, exist_ok=True)
        (d / "projection.json").write_text(json.dumps(proj, indent=1, sort_keys=True))
        (d / "projection.txt").write_text(projection.render(proj))
        return {"projection": proj}

    def finalize(self) -> dict:
        import finalize
        return finalize.finalize(self)


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
    p.add_argument("--stage-seconds", type=float, default=0, help="staging wall time from the bootstrap, for the S8 projection")
    p.add_argument("--test-mode", action="store_true", help="allow a non-16 partition count (tests only)")
    a = ap.parse_args()
    stats = Pass(a).run()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "stdout"} for k, v in stats.items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
