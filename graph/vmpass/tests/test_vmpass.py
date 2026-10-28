"""Tests for the us-west1 pass on the synthetic extract (tests/fixtures.py).

Run:  graph/.venv/bin/python -m unittest discover -s graph/vmpass/tests -v

One S9-mode run, one S8-mode run and one killed-then-resumed run (on a fresh
work dir, as a new Spot VM would see it) are made once in setUpModule; the
tests then check both phases, resume-after-kill, partition-hash parity,
exclusion handling and the output schemas. Needs `clickhouse` on PATH; builds
simgraph with cargo into a temp dir when cargo is present (else O4 is skipped).
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import random
import shutil
import string
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
VMPASS = HERE.parent
sys.path.insert(0, str(VMPASS))
sys.path.insert(0, str(HERE))

import fixtures  # noqa: E402
import rules as rules_mod  # noqa: E402
from cityhash import city_hash64  # noqa: E402

PY = sys.executable
CH = shutil.which("clickhouse")
NONE = 4294967295
ENV: dict = {}


def ch(sql: str) -> list[list[str]]:
    out = subprocess.run([CH, "local", "-q", sql + " FORMAT TSV"], capture_output=True, text=True)
    if out.returncode != 0:
        raise AssertionError(out.stderr[-2000:])
    return [line.split("\t") for line in out.stdout.strip().splitlines() if line]


def run_pass(mode: str, root: pathlib.Path, work: str, run_id: str, rules: str | None = None, env: dict | None = None,
             extra: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [PY, str(VMPASS / "passrun.py"), "run", "--mode", mode, "--stage", str(root / "stage"), "--work", str(root / work),
           "--store30", str(root / "s30"), "--store60", str(root / "s60"), "--run-id", run_id, "--inputs", str(root / "side"),
           "--manifest", str(root / "stage" / "manifest.tsv")]
    if mode == "s9":
        cmd.append("--allow-draft-rules")
    if rules:
        cmd += ["--rules", rules]
    if ENV.get("simgraph"):
        cmd += ["--simgraph", ENV["simgraph"]]
    cmd += extra or []
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})


def setUpModule() -> None:  # noqa: N802
    if not CH:
        raise unittest.SkipTest("clickhouse not on PATH")
    root = pathlib.Path(tempfile.mkdtemp(prefix="vmpass-test-"))
    ENV["root"] = root
    ENV["world"] = fixtures.build(root / "stage", root / "side")
    if shutil.which("cargo"):
        tgt = root / "cargo-target"
        b = subprocess.run(["cargo", "build", "--release", "--quiet"], cwd=VMPASS.parent / "simgraph",
                           env={**os.environ, "CARGO_TARGET_DIR": str(tgt)}, capture_output=True, text=True)
        if b.returncode == 0:
            ENV["simgraph"] = str(tgt / "release" / "simgraph")
    r = run_pass("s9", root, "work", "base")
    assert r.returncode == 0, r.stderr[-4000:]
    # the same world under the S1-registered rules (pre-registration §3 feature list, §6 list split)
    r = run_pass("s9", root, "work1", "s1", rules=str(rules_mod.S1))
    assert r.returncode == 0, r.stderr[-4000:]
    # S8 with a denser actor sample so the tiny world has sampled users (1/4 and 1/8 instead of 1/200 and 1/400)
    rules = json.loads(rules_mod.DRAFT.read_text())
    rules["samples"]["s8_actor_mod"] = [4, 8]
    (root / "rules_s8.json").write_text(json.dumps(rules))
    r = run_pass("s8", root, "work8", "s8", rules=str(root / "rules_s8.json"))
    assert r.returncode == 0, r.stderr[-4000:]
    # killed after p1_05, then resumed on a fresh work dir (a new Spot VM)
    r = run_pass("s9", root, "workk", "kill", env={"VMPASS_FAIL_AFTER": "p1_05"})
    ENV["kill_rc"] = r.returncode
    ENV["kill_err"] = r.stderr
    shutil.rmtree(root / "workk")
    r = run_pass("s9", root, "workk", "kill")
    assert r.returncode == 0, r.stderr[-4000:]
    ENV["resume_out"] = r.stdout


def tearDownModule() -> None:  # noqa: N802
    if ENV.get("root") and not os.environ.get("VMPASS_KEEP"):
        shutil.rmtree(ENV["root"], ignore_errors=True)


def out30(run: str = "base") -> pathlib.Path:
    return ENV["root"] / "s30" / "outputs" / run


def out60(run: str = "base") -> pathlib.Path:
    return ENV["root"] / "s60" / "outputs" / run


def dm() -> str:
    return str(ENV["root"] / "work" / "g" / "didmap.parquet")


def uid(d: str) -> int:
    return int(ch(f"SELECT uid FROM file('{dm()}') WHERE did = '{d}'")[0][0])


class HashParity(unittest.TestCase):
    def test_python_cityhash_matches_clickhouse(self):
        rnd = random.Random(5)
        vals = ["x" * n for n in range(0, 140)] + ["".join(rnd.choice(string.printable[:94] + "é日") for _ in range(rnd.randint(0, 400)))
                                                   .replace("\\", "/") for _ in range(1500)]
        vals = [v for v in vals if "\t" not in v and "\n" not in v and "\r" not in v and "\x0b" not in v and "\x0c" not in v]
        out = subprocess.run([CH, "local", "--input-format", "LineAsString", "--structure", "s String", "-q",
                              "SELECT s, cityHash64(s) FROM table FORMAT TabSeparatedRaw"],
                             input=("\n".join(vals) + "\n").encode(), capture_output=True, check=True).stdout.decode()
        bad = [line for line in out.splitlines() if city_hash64(line.rsplit("\t", 1)[0]) != int(line.rsplit("\t", 1)[1])]
        self.assertEqual(len(out.splitlines()), len(vals))
        self.assertEqual(bad, [])

    def test_didmap_hash_and_partition(self):
        rows = ch(f"SELECT did, h, part FROM file('{dm()}')")
        self.assertEqual(len(rows), 300)
        for d, h, part in rows:
            self.assertEqual(city_hash64(d), int(h))
            self.assertEqual(city_hash64(d) % 16, int(part))

    def test_buckets_hold_only_their_partition(self):
        for kind in ("ev", "po"):
            for f in glob.glob(str(ENV["root"] / "work" / "buckets" / kind / "p*.parquet")):
                k = int(pathlib.Path(f).stem[1:])
                bad = ch(f"SELECT count() FROM file('{f}', Parquet) WHERE part != {k} OR "
                         f"u NOT IN (SELECT uid FROM file('{dm()}') WHERE part = {k})")[0][0]
                self.assertEqual(int(bad), 0, f)

    def test_s8_crosscheck_file(self):
        f = out30("s8") / "S8" / "cityhash64_crosscheck.tsv"
        self.assertEqual(len(f.read_text().splitlines()), 1000)
        r = subprocess.run([PY, str(VMPASS / "crosscheck_hash.py"), str(f)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("python_mismatches=0", r.stdout)

    def test_tid_udf_matches_python(self):
        sys.path.insert(0, str(VMPASS.parent))
        from archive_graph_extract import tid_from_uint64, uint64_from_tid
        import sqlc
        rnd = random.Random(3)
        tids = [tid_from_uint64((rnd.randint(1_600_000_000_000_000, 1_800_000_000_000_000) << 10) | rnd.randint(0, 1023)) for _ in range(200)]
        arr = "[" + ",".join(f"'{t}'" for t in tids) + "]"
        got = subprocess.run([CH, "local", "-q", sqlc.PRELUDE + f"SELECT arrayJoin(arrayMap(x -> (x, tid_us(x)), {arr})) AS r FORMAT TSV"],
                             capture_output=True, text=True, check=True).stdout
        for line in got.strip().splitlines():
            t, v = line.strip("()").replace("'", "").split(",")
            self.assertEqual(int(v), uint64_from_tid(t) >> 10)


class Phase1(unittest.TestCase):
    def test_rows_reconcile_with_manifest(self):
        s = json.loads((out30() / "O1" / "summary.json").read_text())["checks"]
        self.assertIs(s["a_rows_equal_manifest"], True)

    def test_cross_shard_duplicate_counted_once(self):
        o1 = ch(f"SELECT k1, sum(v) FROM file('{out30()}/O1/o1_counts.parquet') WHERE metric IN ('dup_rows') GROUP BY k1 ORDER BY k1")
        o2 = dict(ch(f"SELECT k1, sum(v) FROM file('{out30()}/O1/o1_counts.parquet') WHERE metric = 'dup_key_revs' GROUP BY k1"))
        extra = {k: float(v) - float(o2[k]) for k, v in o1}
        self.assertEqual(extra.get("3"), 1.0)  # the one like row duplicated across two shard files
        self.assertEqual(sum(extra.values()), 1.0)

    def test_deleted_like_and_deleted_post_are_gone(self):
        w = ENV["world"]
        n = ch(f"SELECT count() FROM file('{out30()}/O11/*.parquet') WHERE liker = {uid(w.users[0])}")
        self.assertGreater(int(n[0][0]), 0)
        deleted = ch(f"SELECT count() FROM file('{ENV['root']}/work/p1/i*/pi.parquet') WHERE rkey = '3mzzzzzzzzzz2'")
        self.assertEqual(int(deleted[0][0]), 0)
        dl = ch(f"SELECT sum(v) FROM file('{out30()}/O1/o1_counts.parquet') WHERE metric = 'deleted_keys' AND k1 = '3'")
        self.assertEqual(float(dl[0][0]), 1.0)

    def test_deleted_follow_has_a_delete_time(self):
        w = ENV["world"]
        rows = ch(f"SELECT dt FROM file('{ENV['root']}/work/p1/i*/fol_own.parquet') WHERE u = {uid(w.users[20])} AND dt != {NONE}")
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0][0]) // 86400, int(fixtures.ts("2026-10-22")) // 86400)

    def test_follow_state_at_like_time(self):
        w = ENV["world"]
        u, p = uid(w.late_follower), uid(w.p_main)
        rows = ch(f"SELECT t, fol FROM file('{ENV['root']}/work/p1/i*/eng/*.parquet') WHERE u = {u} AND a = {p} ORDER BY t")
        before = [int(f) for t, f in rows if int(t) < fixtures.ts("2026-08-25")]
        self.assertIn(0, before)  # liked before following: a discovery like

    def test_block_burst(self):
        w = ENV["world"]
        rows = ch(f"SELECT min(b), max(bulk) FROM file('{out30()}/O5-raw/blocks/*.parquet') WHERE u = {uid(w.burst_blocker)}")
        self.assertGreaterEqual(int(rows[0][0]), 10)
        self.assertEqual(int(rows[0][1]), 1)
        rows = ch(f"SELECT max(b) FROM file('{out30()}/O5-raw/blocks/*.parquet') WHERE u = {uid(w.users[100])}")
        self.assertEqual(int(rows[0][0]), 1)

    def test_ustats_light_and_damping(self):
        rows = ch(f"SELECT count(), countIf(light = 1), min(a05), max(a05), countIf(abs(a10 - pow(50 / (50 + d_u), 1)) > 1e-5) "
                  f"FROM file('{ENV['root']}/work/g/ustats.parquet')")
        n, light, lo, hi, bad = rows[0]
        self.assertEqual(int(n), 298 * 5)
        self.assertEqual(int(light), int(n))
        self.assertLessEqual(float(hi), 1.0)
        self.assertEqual(int(bad), 0)

    def test_positive_control_hash_list(self):
        rows = ch(f"SELECT h, bucket FROM file('{out30()}/positive-control/*.parquet')")
        hs = {city_hash64(d): d for d in ENV["world"].everyone}
        for h, b in rows:
            self.assertIn(int(h), hs)
            self.assertLess(int(b), 20)
            self.assertEqual(int(h) % 1000, int(b))

    def test_o3_candidates_and_ipw(self):
        rows = ch(f"SELECT ck, min(pi), max(pi), count() FROM file('{ENV['root']}/work/p1/i*/cand/*.parquet') GROUP BY ck ORDER BY ck")
        cks = {int(r[0]) for r in rows}
        self.assertTrue({1, 2, 3}.issubset(cks))
        for ck, lo, hi, _ in rows:
            self.assertGreater(float(lo), 0)
            self.assertLessEqual(float(hi), 1.0)
        fit = ch(f"SELECT count() FROM file('{ENV['root']}/work/p1/i*/cand/*.parquet') WHERE ck IN (1, 2, 3) AND "
                 f"u NOT IN (SELECT uid FROM file('{dm()}') WHERE fit = 1)")
        self.assertEqual(int(fit[0][0]), 0)


class Phase2(unittest.TestCase):
    def test_o11_type_codes(self):
        w = ENV["world"]
        types = dict(ch(f"SELECT type, count() FROM file('{out30()}/O11/*.parquet') GROUP BY type"))
        for t in ("1", "2", "3", "4", "5"):
            self.assertIn(t, types, f"type {t} missing: {types}")
        ack = ch(f"SELECT liker FROM file('{out30()}/O11/*.parquet') WHERE type = 3")
        self.assertEqual({int(x[0]) for x in ack}, {uid(w.p_main), uid(w.o13_prod)})  # parent authors liking replies
        lu = ch(f"SELECT DISTINCT author FROM file('{out30()}/O11/*.parquet') WHERE type = 4")
        self.assertEqual({int(x[0]) for x in lu}, {uid(w.sup_prod)})

    def test_confirmed_and_hostile_replies(self):
        rows = dict((r[0] + "|" + r[1] + "|" + r[2], float(r[3])) for r in ch(
            f"SELECT metric, k1, k3, sum(v) FROM file('{out30()}/O1/o1_counts.parquet') WHERE metric IN ('replies_quotes', 'hostile_within_48h') GROUP BY ALL"))
        self.assertGreaterEqual(rows.get("replies_quotes|1|1", 0), 1)  # a reply confirmed by the parent author's like
        self.assertGreaterEqual(rows.get("hostile_within_48h|1|", 0), 1)  # a reply followed by a block within 48 h

    def test_o2_sums_are_a_u_weighted(self):
        p = uid(ENV["world"].p_main)
        rows = ch(f"SELECT audience, S0__a10, light_eng_nonlu, followers, fol__a10 FROM file('{out30()}/O2/prod/*.parquet') WHERE p = {p}")
        aud, s0, light, fol, fs = rows[0]
        self.assertGreater(int(aud), 10)
        self.assertGreater(float(s0), 0)
        self.assertLessEqual(float(s0), int(aud))  # each a_u <= 1
        self.assertLessEqual(float(fs), float(s0) + 1e-2)
        self.assertGreater(int(fol), 0)

    def test_blockers_and_listblock_aggregates(self):
        w = ENV["world"]
        b = ch(f"SELECT live_blockers_raw, blockers_per_1k FROM file('{out30()}/O2/prod/*.parquet') WHERE p = {uid(w.producers[1])}")
        self.assertGreaterEqual(int(b[0][0]), 50)
        self.assertNotEqual(b[0][1], "\\N")
        lb = ch(f"SELECT lblk_subscribers FROM file('{out30()}/O2/prod/*.parquet') WHERE p = {uid(w.producers[2])}")
        self.assertGreaterEqual(int(lb[0][0]), 50)
        small = ch(f"SELECT count() FROM file('{out30()}/O2/prod/*.parquet') WHERE live_blockers_raw < 50 AND blockers_per_1k IS NOT NULL")
        self.assertEqual(int(small[0][0]), 0)  # k >= 50 or null

    def test_o3a_labels_and_tasks(self):
        rows = ch(f"SELECT cidx, task, count(), countIf(y_a = 1) FROM file('{out30()}/O3a/*/*.parquet') GROUP BY cidx, task ORDER BY ALL")
        self.assertEqual({int(r[0]) for r in rows}, {0, 1, 3})
        self.assertTrue(any(int(r[3]) > 0 for r in rows))

    def test_gates_are_labels_only(self):
        rows = ch(f"SELECT kind, count() FROM file('{ENV['root']}/work/g/gates/*.parquet') GROUP BY kind ORDER BY kind")
        self.assertEqual([r[0] for r in rows], ["1", "2", "3"])

    def test_teacher_o4(self):
        if not ENV.get("simgraph"):
            self.skipTest("cargo/simgraph unavailable")
        files = sorted(p.name for p in (out30() / "O4").glob("*.parquet"))
        self.assertEqual(files, ["preT_noqn.parquet", "preT_withqn.parquet", "prod_noqn.parquet", "prod_withqn.parquet"])


class Exclusions(unittest.TestCase):
    def _all_uids(self, run="base"):
        """Every account id in every per-account output of a run."""
        o, o6 = out30(run), out60(run)
        q = [f"SELECT liker FROM file('{o}/O11/*.parquet')", f"SELECT author FROM file('{o}/O11/*.parquet')",
             f"SELECT p FROM file('{o}/O2/*/*.parquet')", f"SELECT u FROM file('{o}/O3a/*/*.parquet')",
             f"SELECT p FROM file('{o}/O3a/*/*.parquet')", f"SELECT u FROM file('{o}/O3c/*.parquet')",
             f"SELECT u FROM file('{o}/O5-raw/blocks/*.parquet')", f"SELECT p FROM file('{o}/O5-raw/blocks/*.parquet')",
             f"SELECT p FROM file('{o}/O6/*.parquet')", f"SELECT p FROM file('{o6}/O8/*.parquet')", f"SELECT p FROM file('{o6}/O9/*.parquet')"]
        got = set()
        for s in q:
            if glob.glob(s.split("'")[1]):
                got |= {int(r[0]) for r in ch(s)}
        return got

    def test_purged_and_bulk_false_accounts_vanish(self):
        w = ENV["world"]
        ids = self._all_uids()
        for d in (w.purged, w.bulk_no):
            self.assertNotIn(uid(d), ids)
            n = ch(f"SELECT count() FROM file('{ENV['root']}/work/buckets/ev/*.parquet') WHERE u = {uid(d)} OR subj = {uid(d)}")
            self.assertEqual(int(n[0][0]), 0)
        for f in (out30() / "O12").glob("*.parquet"):
            text = " ".join(" ".join(r) for r in ch(f"SELECT * FROM file('{f}')"))
            self.assertNotIn(w.purged, text)
            self.assertNotIn(w.bulk_no, text)

    def test_hide_from_recs_kept_as_signal_but_no_post_text_or_node(self):
        """Amendment A1 (S1, 2026-10-28): the flag governs posts only; bios (O9) stay."""
        w = ENV["world"]
        h = uid(w.hide_prod)
        self.assertEqual(len(ch(f"SELECT p FROM file('{out30()}/O2/prod/*.parquet') WHERE p = {h}")), 1)
        self.assertGreater(int(ch(f"SELECT count() FROM file('{out30()}/O11/*.parquet') WHERE author = {h}")[0][0]), 0)
        self.assertEqual(int(ch(f"SELECT count() FROM file('{out60()}/O9/*.parquet') WHERE p = {h}")[0][0]), 1)
        for s in (f"SELECT count() FROM file('{out30()}/O6/*.parquet') WHERE p = {h}",
                  f"SELECT count() FROM file('{out60()}/O8/*.parquet') WHERE p = {h}"):
            self.assertEqual(int(ch(s)[0][0]), 0, s)
        if ENV.get("simgraph"):
            self.assertEqual(int(ch(f"SELECT count() FROM file('{out30()}/O4/*.parquet') WHERE p = {h} OR q = {h}")[0][0]), 0)
        self.assertEqual(int(ch(f"SELECT count() FROM file('{ENV['root']}/work/p2/j*/e0_*.parquet') WHERE p = {h}")[0][0]), 0)
        for f in (out30() / "O12").glob("*.parquet"):
            self.assertNotIn(w.hide_prod, " ".join(" ".join(r) for r in ch(f"SELECT * FROM file('{f}')")))

    def test_synthetic_false_out_of_teacher_inputs(self):
        w = ENV["world"]
        s = uid(w.synth_prod)
        self.assertEqual(len(ch(f"SELECT p FROM file('{out30()}/O2/prod/*.parquet') WHERE p = {s}")), 1)
        for q in (f"SELECT count() FROM file('{out30()}/O11/*.parquet') WHERE author = {s} OR liker = {s}",
                  f"SELECT count() FROM file('{ENV['root']}/work/p2/j*/e0_*.parquet') WHERE p = {s} OR u = {s}",
                  f"SELECT count() FROM file('{out30()}/O6/*.parquet') WHERE p = {s}",
                  f"SELECT count() FROM file('{out60()}/O8/*.parquet') WHERE p = {s}",
                  f"SELECT count() FROM file('{out60()}/O9/*.parquet') WHERE p = {s}"):
            self.assertEqual(int(ch(q)[0][0]), 0, q)

    def test_b1_owners_out_of_teacher(self):
        b1 = ch(f"SELECT DISTINCT u FROM file('{ENV['root']}/work/g/ustats.parquet') WHERE b1own = 1")
        self.assertGreater(len(b1), 0)
        ids = ",".join(r[0] for r in b1)
        self.assertEqual(int(ch(f"SELECT count() FROM file('{ENV['root']}/work/p2/j*/e0_*.parquet') WHERE u IN ({ids})")[0][0]), 0)

    def test_exclusion_counts_reported(self):
        g = json.loads((out30() / "O1" / "summary.json").read_text())["checks"]["g_exclusions"]
        self.assertEqual(g["extract account inactive"]["accounts"], 1)
        self.assertEqual(g["bulkDataset:false"]["accounts"], 1)
        self.assertEqual(g["kept"]["hide_from_recs"], 1)
        self.assertEqual(g["kept"]["synthetic_false"], 1)


class Schemas(unittest.TestCase):
    def test_o11(self):
        import pyarrow.parquet as pq
        sch = pq.read_schema(glob.glob(f"{out30()}/O11/*.parquet")[0])
        self.assertEqual([(f.name, str(f.type)) for f in sch],
                         [("liker", "uint32"), ("post_hash", "uint64"), ("author", "uint32"), ("t", "uint32"), ("type", "uint8")])

    def test_o2_has_every_registered_feature_sum_in_fp16(self):
        import pyarrow.parquet as pq
        r = rules_mod.load()
        f = glob.glob(f"{out30()}/O2/prod/*.parquet")[0]
        sch = pq.read_schema(f)
        names = set(sch.names)
        for feat in r.features + [{"name": "S0"}]:
            for a in ("a05", "a10"):
                self.assertIn(f"{feat['name']}__{a}", names)
                self.assertEqual(str(sch.field(f"{feat['name']}__{a}").type), "halffloat")
        for c in ("p", "cidx", "light_eng_nonlu", "light_eng_luonly", "D_nonlu", "D_luonly", "reposters", "confirmed_reply_share",
                  "courtesy_like_share", "originals_30d", "originals_90d", "originals_12mo", "lifetime_posts", "last_post_t",
                  "image_post_share", "blockers_per_1k", "blockers_per_1k_burst_discounted", "lblk_subscribers", "a_half_curators",
                  "combativeness", "top_langs", "supplement"):
            self.assertIn(c, names)
        self.assertTrue(glob.glob(f"{out30()}/O2/lu_discounted_prod.parquet"))

    def test_o3(self):
        import pyarrow.parquet as pq
        r = rules_mod.load()
        fa = pq.read_schema(glob.glob(f"{out30()}/O3a/T_fit/*.parquet")[0])
        self.assertEqual(fa.names[:14], ["u", "p", "cidx", "task", "ck", "pi", "ft", "heldout", "y_a", "y_b", "y_c", "elig_p", "elig_u", "offset"])
        self.assertEqual(fa.names[14:], [f["name"] for f in r.features])
        self.assertEqual(str(fa.field(r.features[0]["name"]).type), "halffloat")
        fb = pq.read_schema(glob.glob(f"{out30()}/O3b/features_*.parquet")[0])
        self.assertEqual(fb.names, ["u", "p"] + [f["name"] for f in r.features])
        self.assertEqual(str(fb.field(r.features[0]["name"]).type), "uint8")
        self.assertEqual(pq.read_schema(glob.glob(f"{out30()}/O3b/labels_*.parquet")[0]).names, ["u", "p", "kind", "t"])
        self.assertEqual(pq.read_schema(glob.glob(f"{out30()}/O3c/*.parquet")[0]).names, ["u", "p", "kind", "ft", "engaged30"])

    def test_small_outputs(self):
        import pyarrow.parquet as pq
        want = {
            f"{out30()}/O5-raw/blocks/*.parquet": ["u", "p", "t", "b", "bulk"],
            f"{out30()}/O5-raw/follows/*.parquet": ["u", "p", "ft", "dt", "b"],
            f"{out30()}/O5-raw/listblock_subs.parquet": ["u", "lh", "t"],
            f"{out30()}/O5-raw/listblock_members_at_T.parquet": ["lh", "member", "t"],
            f"{out30()}/O5-prod/O5_prod.parquet": ["map", "c_actor", "c_target", "what", "n_actors"],
            f"{out30()}/O6/*.parquet": ["p", "uri", "light_engagers", "rank"],
            f"{out30()}/O7/O7.parquet": ["feed_uri", "partition_map", "community", "n_likers"],
            f"{out60()}/O8/*.parquet": ["p", "uri", "created", "text", "langs", "image_count", "self_labels"],
            f"{out60()}/O9/*.parquet": ["p", "display_name", "description"],
            f"{out30()}/O11-side/maint_pairs.parquet": ["u1", "u2", "cidx", "flag_12", "flag_21", "n_12", "n_21"],
            f"{out30()}/O11-side/users_light.parquet": ["u", "cidx", "light"],
            f"{out30()}/O12/pairs.parquet": ["stratum", "kind", "child_uri", "child_text", "parent_uri", "parent_text"],
            f"{out30()}/O12/random_nonfollower_quotes.parquet": ["uri"],
            f"{out30()}/positive-control/*.parquet": ["h", "bucket", "n", "e_new", "e_old", "e_fb", "used_fb"],
        }
        for pat, names in want.items():
            files = glob.glob(pat)
            self.assertTrue(files, pat)
            self.assertEqual(pq.read_schema(files[0]).names, names, pat)
        if ENV.get("simgraph"):
            self.assertEqual(pq.read_schema(glob.glob(f"{out30()}/O4/*.parquet")[0]).names, ["p", "q", "w"])

    def test_o8_text_has_alt_marker_format_and_cap(self):
        rows = ch(f"SELECT p, count() FROM file('{out60()}/O8/*.parquet') GROUP BY p ORDER BY 2 DESC LIMIT 1")
        self.assertLessEqual(int(rows[0][1]), 64)
        per_month = ch(f"SELECT max(c) FROM (SELECT p, substring(created, 1, 7) m, count() c FROM file('{out60()}/O8/*.parquet') GROUP BY p, m)")
        self.assertLessEqual(int(per_month[0][0]), 6)

    def test_receipt_lists_outputs_with_hashes(self):
        rec = json.loads((out30() / "receipt" / "receipt.json").read_text())
        self.assertEqual(rec["rules_sha256"], rules_mod.load().sha256)
        paths = {o["path"] for o in rec["outputs"]}
        self.assertTrue(any(p.startswith("out30/O11/") for p in paths))
        self.assertTrue(any(p.startswith("out60/O8/") for p in paths))
        self.assertTrue(all(len(o["sha256"]) == 64 for o in rec["outputs"]))


class Resume(unittest.TestCase):
    def test_kill_then_resume_on_fresh_vm(self):
        self.assertNotEqual(ENV["kill_rc"], 0)
        self.assertIn("simulated preemption", ENV["kill_err"])
        led = ENV["root"] / "s30" / "intermediates" / "kill" / "_ledger"
        self.assertTrue((led / "p1_05.json").exists())
        stats = json.loads(ENV["resume_out"][ENV["resume_out"].index("{"):])
        for i in range(6):
            self.assertTrue(stats[f"p1_{i:02d}"].get("skipped"), f"p1_{i:02d} should be skipped on resume")
        self.assertFalse(stats["p1_06"].get("skipped", False))
        self.assertFalse(stats["bucket"].get("skipped", False))  # local-only: rebuilt on the fresh VM
        self.assertTrue(stats["didmap"].get("skipped"))

    def test_resumed_outputs_equal_uninterrupted(self):
        for pat, key in (("O11/*.parquet", "liker, post_hash, author, t, type"),
                         ("O2/prod/*.parquet", "p, audience, light_eng_nonlu, D_nonlu, followers, round(toFloat64(LO_dec14__a05), 2)"),
                         ("O3a/*/*.parquet", "u, p, cidx, task, ck, y_a, y_c"),
                         ("O5-raw/blocks/*.parquet", "u, p, t, b")):
            a = ch(f"SELECT {key} FROM file('{out30()}/{pat}') ORDER BY ALL")
            b = ch(f"SELECT {key} FROM file('{out30('kill')}/{pat}') ORDER BY ALL")
            self.assertEqual(a, b, pat)


class S8(unittest.TestCase):
    def test_projection_has_what_r_s9_needs(self):
        p = json.loads((out30("s8") / "S8" / "projection.json").read_text())
        pr = p["projected"]
        for k in ("bucket_hours", "phase1_hours", "phase2_hours", "main_run_hours", "peak_ram_phase1_bytes", "peak_ram_phase2_bytes",
                  "intermediate_bytes", "maint_shuffle_bytes", "cross_shard_duplicate_rate", "spend_usd_at_spot"):
            self.assertIn(k, pr)
        for k in ("O3a_bytes", "O3b_bytes", "O3c_bytes", "O3a_rows", "O3b_rows", "O3c_rows"):
            self.assertIn(k, pr["O3_after_shrink"])
        self.assertGreater(pr["O3_rows_per_sampled_user"]["O3a"], 0)
        self.assertGreater(pr["O3_rows_per_sampled_user"]["O3b"], 0)
        self.assertGreater(pr["fit_users_full"], 0)
        self.assertIn("distinct_key_growth", p["measured"])
        self.assertTrue((out30("s8") / "S8" / "projection.txt").exists())

    def test_two_inbound_rates_and_one_author_partition(self):
        self.assertTrue(glob.glob(f"{out30('s8')}/S8-s400/O2/prod/j00.parquet"))
        self.assertTrue(glob.glob(f"{out30('s8')}/S8-s200/O2/prod/j00.parquet"))
        self.assertFalse(glob.glob(f"{out30('s8')}/S8-s200/O2/prod/j01.parquet"))
        r400 = int(ch(f"SELECT count() FROM file('{out30('s8')}/S8-s400/O11/*.parquet')")[0][0])
        r200 = int(ch(f"SELECT count() FROM file('{out30('s8')}/S8-s200/O11/*.parquet')")[0][0])
        self.assertLess(r400, r200)

    def test_sample_bucket_holds_sampled_actors_and_partition_zero(self):
        d = str(ENV["root"] / "work8" / "g" / "didmap.parquet")
        bad = ch(f"SELECT count() FROM file('{ENV['root']}/work8/buckets/ev/*.parquet', Parquet) WHERE u NOT IN "
                 f"(SELECT uid FROM file('{d}') WHERE h % 4 = 0 OR part = 0)")
        self.assertEqual(int(bad[0][0]), 0)

    def test_kind_and_source_listing(self):
        o1 = ch(f"SELECT DISTINCT metric FROM file('{out30('s8')}/O1/o1_counts.parquet') WHERE startsWith(metric, 'backfill_probe')")
        self.assertEqual({r[0] for r in o1}, {"backfill_probe", "backfill_probe_modal_rev", "backfill_probe_ing_min", "backfill_probe_ing_max"})
        self.assertTrue((ENV["root"] / "work8" / "inv" / "ev_skew.parquet").exists())
        ks = json.loads((out30("s8") / "S8" / "projection.json").read_text())["kind_source_listing"]
        self.assertEqual({(x["source_id"], x["kind"]) for x in ks["events"]}, {("gce_main", "commit"), ("gce_main", "account"), ("repo_supplement", "commit")})


class GcsBatching(unittest.TestCase):
    def test_put_and_get_are_one_call_per_directory_with_custom_time(self):
        from store import GcsStore
        fake = ENV["root"] / "fakegcs"
        fake.mkdir(exist_ok=True)
        log = fake / "argv"
        (fake / "gcloud").write_text(f"#!/bin/sh\necho \"$@\" >> {log}\n")
        (fake / "gcloud").chmod(0o755)
        old = os.environ["PATH"]
        os.environ["PATH"] = f"{fake}:{old}"
        try:
            st = GcsStore("gs://b", custom_time="2026-10-28T00:00:00Z")
            files = []
            for d in ("x", "y"):
                for k in range(3):
                    p = fake / d / f"f{k}.parquet"
                    p.parent.mkdir(exist_ok=True)
                    p.write_text("z")
                    files.append((p, f"outputs/r/{d}/f{k}.parquet"))
            st.put_files(files)
            st.get_files([(k, fake / "dl" / pathlib.Path(k).name) for _, k in files[:3]])
        finally:
            os.environ["PATH"] = old
        calls = log.read_text().splitlines()
        self.assertEqual(len(calls), 3)  # two uploads (one per directory) and one download
        self.assertTrue(all("--custom-time=2026-10-28T00:00:00Z" in c for c in calls[:2]))
        self.assertTrue(calls[0].endswith("gs://b/outputs/r/x/"))


class O13(unittest.TestCase):
    """eval_temporal.py's m1_/m2_engagement.parquet: one row per (did, subject, kind, UTC day), last event."""

    def _rows(self, f, did=None, subject=None):
        w = f" WHERE did = '{did}'" + (f" AND subject = '{subject}'" if subject else "") if did else ""
        return ch(f"SELECT did, subject, created_us, kind FROM file('{out60()}/O13/{f}'){w} ORDER BY created_us, kind")

    def test_schema_matches_eval_temporal(self):
        import pyarrow.parquet as pq
        for f in ("m1_engagement.parquet", "m2_engagement.parquet"):
            sch = pq.read_schema(out60() / "O13" / f)
            self.assertEqual([(x.name, str(x.type)) for x in sch],
                             [("did", "string"), ("subject", "string"), ("created_us", "int64"), ("kind", "string")])
        sys.path.insert(0, str(VMPASS.parent))
        import eval_temporal
        kinds = {r[0] for r in ch(f"SELECT DISTINCT kind FROM file('{out60()}/O13/*.parquet')")}
        self.assertTrue(kinds <= set(eval_temporal.ENGAGE_KINDS))
        self.assertEqual(kinds, set(eval_temporal.ENGAGE_KINDS))

    def test_window_matches_eval_temporal_constants(self):
        sys.path.insert(0, str(VMPASS.parent))
        import eval_temporal as et
        lo, hi = ch(f"SELECT min(created_us), max(created_us) FROM file('{out60()}/O13/*.parquet')")[0]
        self.assertGreaterEqual(int(lo), et.M1_HIST_US)                      # (T - 90 d, ...: from 05-18 00:00
        self.assertLess(int(hi), et.M1_END_US + et.ENGAGED_DAYS_US)          # ..., 09-12 + 30 d]: before 10-13 00:00
        self.assertLessEqual(et.M2_COHORT_END_US + et.ENGAGED_DAYS_US, et.M1_END_US + et.ENGAGED_DAYS_US)

    def test_day_aggregation_and_window_edges(self):
        w = ENV["world"]
        rows = self._rows("m1_engagement.parquet", w.o13_user, w.o13_prod)
        got = {(fixtures.dt.datetime.fromtimestamp(int(r[2]) / 1e6, fixtures.dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), r[3]) for r in rows}
        self.assertIn(("2026-05-18T00:00:01", "like"), got)                  # first included second
        self.assertNotIn("2026-05-17", " ".join(x[0] for x in got))         # last excluded second
        self.assertIn(("2026-10-12T23:59:59", "like"), got)                  # last included second
        self.assertNotIn("2026-10-13", " ".join(x[0] for x in got))
        jul = [x for x in got if x[0].startswith("2026-07-01")]
        self.assertEqual(sorted(jul), [("2026-07-01T12:00:00", "repost"), ("2026-07-01T15:00:00", "like")])  # two likes -> the later
        self.assertTrue(any(k == "confirmed_reply" and d.startswith("2026-09-01") for d, k in got))
        for f in ("m1_engagement.parquet", "m2_engagement.parquet"):
            dup = ch(f"SELECT count() FROM (SELECT did, subject, kind, intDiv(created_us, 86400000000) d, count() c FROM "
                     f"file('{out60()}/O13/{f}') GROUP BY ALL HAVING c > 1)")
            self.assertEqual(int(dup[0][0]), 0, f)

    def test_users_m1_hash_and_m2_list(self):
        w = ENV["world"]
        m1 = {r[0] for r in ch(f"SELECT DISTINCT did FROM file('{out60()}/O13/m1_engagement.parquet')")}
        self.assertTrue(m1)
        self.assertTrue(all(city_hash64(d) % 100 == 13 for d in m1))
        m2 = {r[0] for r in ch(f"SELECT DISTINCT did FROM file('{out60()}/O13/m2_engagement.parquet')")}
        self.assertTrue(m2 and m2 <= set(w.m2_users))
        self.assertIn(w.o13_user, m2)

    def test_exclusions(self):
        w = ENV["world"]
        dids = " ".join(" ".join(r) for r in ch(f"SELECT did, subject FROM file('{out60()}/O13/*.parquet')"))
        self.assertNotIn(w.purged, dids)
        self.assertNotIn(w.bulk_no, dids)
        hid = ch(f"SELECT count() FROM file('{ENV['root']}/work/p2/j*/o13.parquet', Parquet) WHERE subject = '{w.hide_prod}'")
        self.assertGreaterEqual(int(hid[0][0]), 0)  # A1: hide-from-recs subjects are not removed (behaviour, not post text)

    def test_s8_projects_o13(self):
        p = json.loads((out30("s8") / "S8" / "projection.json").read_text())["projected"]["O13"]
        for k in ("m1_rows", "m1_bytes", "rows_per_user", "m2_rows"):
            self.assertIn(k, p)
        self.assertGreater(p["rows_per_user"], 0)


class S1Rules(unittest.TestCase):
    """The pass under rules_s1.json: S1 pre-registration §3's 110 features, §6's list split."""

    def test_s1_feature_list_reaches_o2_and_o3(self):
        import pyarrow.parquet as pq
        r = rules_mod.load(rules_mod.S1)
        self.assertFalse(r.draft)
        names = [f["name"] for f in r.features]
        self.assertEqual(len(names), 110)
        o2 = pq.read_schema(glob.glob(f"{out30('s1')}/O2/prod/*.parquet")[0]).names
        for n in names:
            self.assertIn(f"{n}__a05", o2)
        o3 = pq.read_schema(glob.glob(f"{out30('s1')}/O3a/T_fit/*.parquet")[0]).names
        self.assertEqual(o3[14:], names)

    def test_s1_feature_values(self):
        o3 = f"{out30('s1')}/O3a/*/*.parquet"
        row = ch(f"SELECT min(FOL), max(FOL), max(LO_DSL), max(HEAVY), max(MUT), min(DL_L14), max(PBLKN), max(FBURST) FROM file('{o3}')")[0]
        fmin, fmax, dsl, heavy, mut, dl, pblkn, fb = (float(x) for x in row)
        self.assertEqual((fmin, fmax), (0.0, 1.0))
        self.assertLessEqual(dsl, 90.0)
        self.assertEqual(heavy, 0.0)  # nobody in the fixture follows >= 3,000 producers
        self.assertEqual(pblkn, 1.0)
        self.assertGreaterEqual(dl, 0.0)
        w = ENV["world"]
        o2 = f"{out30('s1')}/O2/prod/*.parquet"
        def one(col, d):
            return float(ch(f"SELECT {col} FROM file('{o2}') WHERE p = (SELECT uid FROM file('{ENV['root']}/work1/g/didmap.parquet') WHERE did = '{d}')")[0][0])
        self.assertGreater(one("BLK__a10", w.producers[1]), 0)      # blockers of producers[1] who also engage it
        self.assertGreater(one("LBLK__a10", w.producers[2]), 0)     # list-block subscribers covering producers[2]
        self.assertGreater(float(ch(f"SELECT sum(LST_N__a10), sum(DL_L14__a10), sum(MAINT__a10) + 1 FROM file('{o2}')")[0][0]), 0)

    def test_s1_list_split(self):
        rows = ch(f"SELECT did, ls FROM file('{ENV['root']}/work1/g/didmap.parquet')")
        code = {0: 1, 1: 1, 4: 1, 5: 1, 2: 2, 6: 3, 3: 4, 7: 4}
        for d, ls in rows:
            self.assertEqual(int(ls), code[city_hash64(d) % 8])

    def test_s8_writes_o3b_scales(self):
        sc = json.loads((out30("s8") / "S8" / "o3b_scales.json").read_text())
        self.assertEqual(sorted(sc), sorted(f["name"] for f in rules_mod.load().features))
        self.assertTrue(all(v > 0 for v in sc.values()))


class Guards(unittest.TestCase):
    def test_s9_refuses_draft_rules_and_bad_sha(self):
        root = ENV["root"]
        cmd = [PY, str(VMPASS / "passrun.py"), "run", "--mode", "s9", "--stage", str(root / "stage"), "--work", str(root / "wg"),
               "--store30", str(root / "sg30"), "--store60", str(root / "sg60"), "--run-id", "g", "--inputs", str(root / "side")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("DRAFT", r.stderr)
        r = subprocess.run(cmd + ["--allow-draft-rules", "--rules-sha256", "0" * 64], capture_output=True, text=True)
        self.assertIn("sha256 mismatch", r.stderr)

    def test_launcher_is_print_only_and_spot_only(self):
        fake = ENV["root"] / "fakebin"
        fake.mkdir(exist_ok=True)
        (fake / "gcloud").write_text("#!/bin/sh\necho CALLED >> \"$(dirname \"$0\")/calls\"\nexit 1\n")
        (fake / "gcloud").chmod(0o755)
        env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}", "SESSIONS": str(ENV['root'] / "sessions.tsv")}
        for mode, maxrun in (("s8", "90m"), ("s9", "9h"), ("resume", "165m")):
            r = subprocess.run(["bash", str(VMPASS / "launch_vm.sh"), mode, "t-run"], capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stderr)
            o = r.stdout
            self.assertIn("--provisioning-model=SPOT", o)
            self.assertIn("--instance-termination-action=DELETE", o)
            self.assertIn(f"--max-run-duration='{maxrun}'", o)
            self.assertIn("--machine-type='n2-highmem-64'", o)
            self.assertEqual(o.count("--local-ssd=interface=NVME"), 8)
            self.assertIn("catbird-plan=whole-network-20261025", o)
            self.assertIn("dry run: nothing was executed", o)
        self.assertFalse((fake / "calls").exists(), "the dry run must not call gcloud")
        r = subprocess.run(["bash", str(VMPASS / "launch_vm.sh"), "delete", "unknown-run"], capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)  # never deletes a VM it did not record


if __name__ == "__main__":
    unittest.main()
