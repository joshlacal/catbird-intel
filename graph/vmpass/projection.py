"""S8 projection for the R-S9 ask (plan §3.1 S8, §3.2 "Peak RAM and hours come from S8", §4.7 O3 budget).

Measured in S8 (one VM session):
  - phase 1 on partition 0 in full plus the 1/200 DID-hash sample of the other 15;
  - phase 2 on one full author partition (0) joined against the 1/400 and 1/200
    actor samples.
Projected to the full S9 run:
  - phase 1 = t(P1, partition 0) x sum_j size_j / size_0 (partition sizes from the
    full inventory scan); peak = peak(P1_0) x max_j size_j / size_0;
  - phase 2 at rate 1 from the two rates: rows/keys by the distinct-key growth
    exponent beta = log(K2/K1)/log(r2/r1), time and memory linear and by beta
    (the larger is used), then x 16 partitions with the post-file skew;
  - bucketing = read cost (decode-only scan) + per-row write cost x all rows;
  - O3a/O3b/O3c rows and bytes separately, then the registered shrink order.
"""
from __future__ import annotations

import glob
import json
import math

import chrun
from sqlc import PRELUDE

SPOT = 2.715
LIMIT_H, LIMIT_RAM, O3_BUDGET = 8.5, 400e9, 8e9


def _q(ps, sql: str) -> list[list[str]]:
    txt = chrun.query(PRELUDE + sql + " FORMAT TSV", ps.work, "proj_q")
    return [line.split("\t") for line in txt.strip().splitlines() if line]


def _bytes(ps, step: str) -> int:
    m = ps.ledger.done(step) or {}
    return sum(f["bytes"] for f in m.get("manifest", []))


def _st(ps, step: str) -> dict:
    s = ps.stats.get(step) or (ps.ledger.done(step) or {}).get("stats", {}) or {}
    return s


def _o1(ps, path_glob: str) -> dict:
    if not glob.glob(path_glob):
        return {}
    rows = _q(ps, f"SELECT metric, k1, k2, sum(v) FROM file('{path_glob}', Parquet) GROUP BY metric, k1, k2")
    d: dict = {}
    for m, k1, k2, v in rows:
        d.setdefault(m, {})[f"{k1}|{k2}"] = float(v)
    return d


def _beta(k1: float, k2: float, r1: float, r2: float) -> float:
    if k1 <= 0 or k2 <= 0:
        return 1.0
    return math.log(k2 / k1) / math.log(r2 / r1)


def project(ps) -> dict:
    r = ps.rules
    N = r.N
    nf = len(r.features)
    inv = ps.work / "inv"
    ev = {int(p): int(n) for p, n in _q(ps, f"SELECT part, sum(rows) FROM file('{inv}/ev_skew.parquet') GROUP BY part")}
    po = {int(p): int(n) for p, n in _q(ps, f"SELECT part, sum(rows) FROM file('{inv}/po_skew.parquet') GROUP BY part")}
    size = {j: ev.get(j, 0) + po.get(j, 0) for j in range(N)}
    s0 = max(1, size.get(0, 1))
    skew1 = sum(size.values()) / s0
    maxskew1 = max(size.values()) / s0 if size else 1
    po0 = max(1, po.get(0, 1))
    skew2 = sum(po.values()) / po0
    maxskew2 = max(po.values()) / po0 if po else 1
    total_rows = sum(ev.values()) + sum(po.values())
    # S8 on a bucketing subset (--bucket-shards): measured work covers a fraction f of the rows;
    # every row-proportional measurement below is scaled by F = 1/f (memory too, conservatively).
    f_sub = float(getattr(ps, "subset_fraction", 1.0) or 1.0)
    F = 1.0 / f_sub
    man = (_st(ps, "inventory").get("manifest") or {})
    if man:
        total_rows = sum(v["rows"] for v in man.values())

    # bucketing: read cost + per-row write cost
    b = _st(ps, "bucket")
    rr = _st(ps, "read_rate")
    smp = f"u IN (SELECT uid FROM file('{ps.g}/didmap.parquet') WHERE s200 = 1)"
    evb, pob = f"file('{ps.buckets}/ev/*/*.parquet', Parquet, 'u UInt32')", f"file('{ps.buckets}/po/*/*.parquet', Parquet, 'u UInt32')"
    brows = _q(ps, f"SELECT (SELECT count() FROM {evb}) + (SELECT count() FROM {pob}), "
                   f"(SELECT count() FROM {evb} WHERE {smp}) + (SELECT count() FROM {pob} WHERE {smp})")
    written, sample_rows = (int(brows[0][0]), int(brows[0][1])) if brows else (0, 0)
    t_read = rr.get("seconds", 0) or 0
    t_b = b.get("seconds", 0) or 0
    per_row = max(0.0, t_b - t_read * f_sub) / max(1, written)
    bucket_full_s = t_read + per_row * total_rows

    # phase 1
    p1_0 = _st(ps, "p1_00")
    p1_s = sum((_st(ps, f"p1_{i:02d}").get("seconds", 0) or 0) for i in range(1, N))
    p1_full_s = (p1_0.get("seconds", 0) or 0) * skew1 * F
    p1_from_sample_s = p1_s * r["samples"]["s8_actor_mod"][0] * N / max(1, N - 1) * F
    p1_peak = (p1_0.get("peak_rss_bytes", 0) or 0) * maxskew1 * F
    p1_bytes = _bytes(ps, "p1_00") * F

    # phase 2 at two inbound rates
    r1, r2 = 1 / r["samples"]["s8_actor_mod"][1], 1 / r["samples"]["s8_actor_mod"][0]
    a, c = _st(ps, "p2_s400_00"), _st(ps, "p2_s200_00")
    o1a, o1c = _o1(ps, f"{ps.work}/p2_s400/j00/o1.parquet"), _o1(ps, f"{ps.work}/p2_s200/j00/o1.parquet")

    def tot(d, m):
        return sum(d.get(m, {}).values())

    keys = {}
    for m in ("pairs_prod", "pairs_eval", "aud_pairs_prod", "typed_engagement", "replies_quotes"):
        k1, k2 = tot(o1a, m), tot(o1c, m)
        beta = _beta(k1, k2, r1, r2)
        keys[m] = {"r1": k1, "r2": k2, "beta": round(beta, 3), "at_rate_1": k2 * (1 / r2) ** beta}

    def extrap(x1, x2):
        lin = x2 + (x2 - x1) / (r2 - r1) * (1 - r2)
        bt = _beta(x1, x2, r1, r2)
        powr = x2 * (1 / r2) ** bt if x2 > 0 else 0
        return max(lin, powr), lin, powr

    t2, t2lin, t2pow = (x * F for x in extrap(a.get("seconds", 0) or 0, c.get("seconds", 0) or 0))
    m2, m2lin, m2pow = (x * F for x in extrap(a.get("peak_rss_bytes", 0) or 0, c.get("peak_rss_bytes", 0) or 0))
    # Phase 2 at rate 1. The two sample rates differ only in the inbound rows, so extrapolating the
    # whole step's seconds or RSS from them multiplies noise by ~400 (S8: -11.5 h and 564 GB). Instead:
    # pairs grow linearly in the inbound sample (exact counts), statements are split into fixed,
    # inbound-scaled and pair-scaled ones, and the pair aggregate's memory is per pair x pairs per chunk.
    model = phase2_model(ps, o1a, o1c, r1, r2, F)
    p2_full_s = model["seconds_rate1"] * skew2 if model.get("seconds_rate1") else t2 * skew2
    p2_peak = model["peak_bytes_rate1"] * maxskew2 if model.get("peak_bytes_rate1") else m2 * maxskew2
    b2a, b2c = _bytes(ps, "p2_s400_00"), _bytes(ps, "p2_s200_00")
    p2_bytes_1 = extrap(b2a, b2c)[0] * F

    # globals between/after phases, scaled by the phase-1 output growth
    p1_growth = skew1 * 1.0
    glob_s = sum((_st(ps, s).get("seconds", 0) or 0) for s in ("ustats_merge", "gates_merge", "lblk", "o5prod", "o7")) * p1_growth
    e0_rows = 0
    e0g = f"{ps.work}/p2_s200/j00/e0_*.parquet"
    if glob.glob(e0g):
        e0_rows = int(_q(ps, f"SELECT count() FROM file('{e0g}', Parquet)")[0][0])
    e0_full = e0_rows / r2 * skew2 * F
    teacher_s = 4 * 420 * max(0.05, e0_full / 2.04e9)  # simgraph: 7 min per 2.04B edges, 4 teachers
    sum(_bytes(ps, s) for s in ("maint",))
    maintd = glob.glob(f"{ps.work}/p2_s200/j00/maintd/*.parquet")
    maintd_bytes_full = 0
    if maintd:
        import os
        maintd_bytes_full = sum(os.path.getsize(f) for f in maintd) / r2 * skew2 * F
    didmap_s = _st(ps, "didmap").get("seconds", 0) or 0
    inv_s = _st(ps, "inventory").get("seconds", 0) or 0
    stage_s = float(getattr(ps.a, "stage_seconds", 0) or 0)
    finalize_s = 600
    total_s = stage_s + inv_s + didmap_s + bucket_full_s + p1_full_s + glob_s + p2_full_s + teacher_s + finalize_s

    # O3 sizes (rows per sampled user x full user counts) and the shrink order
    p1o1 = _o1(ps, f"{ps.work}/p1/i*/o1.parquet")
    fit_cidx = [0, 1] + ([] if r["o3"]["drop_t_test"] else [3])
    cand_rows = p1o1.get("cand_rows", {})
    o3a_rows_s = sum(v for k, v in cand_rows.items() if int(k.split("|")[0]) in fit_cidx and int(k.split("|")[1]) in (1, 2, 3))
    o3b_rows_s = sum(v for k, v in cand_rows.items() if int(k.split("|")[1]) == 4)
    users_s = int(_q(ps, f"SELECT uniqExact(u) FROM file('{ps.g}/ustats.parquet') WHERE u IN (SELECT uid FROM file('{ps.g}/didmap.parquet') WHERE s200 = 1)")[0][0])
    # full fit / M1 user counts: DIDs passing the predicate x the share of partition-0 DIDs that are actors
    dm_rows = _q(ps, f"SELECT countIf(fit = 1), countIf(m1 = 1), countIf(part = 0) FROM file('{ps.g}/didmap.parquet')")[0]
    actors_p0 = int(_q(ps, f"SELECT uniqExact(u) FROM file('{ps.p1dir}/i00/ustats.parquet')")[0][0])
    frac_actor = actors_p0 / max(1, int(dm_rows[2]))
    n_fit, n_m1 = int(dm_rows[0]) * frac_actor, int(dm_rows[1]) * frac_actor
    b_rows = 0
    o3ag = f"{ps.out30}/S8-s200/O3a/*/*.parquet"
    if glob.glob(o3ag):
        # B' rows exist only for partition-0 producers here: x post skew for all producers, per sampled user
        b_rows = int(_q(ps, f"SELECT countIf(task = 2) FROM file('{o3ag}', Parquet)")[0][0]) * skew2 / max(1, users_s)
    rpu_a = (o3a_rows_s / max(1, users_s) + b_rows) * F
    rpu_b = o3b_rows_s / max(1, users_s) * F
    o3c_rows = 0
    o3cf = glob.glob(f"{ps.out30}/O3c/i00.parquet")
    if o3cf:
        o3c_rows = int(_q(ps, f"SELECT count() FROM file('{o3cf[0]}', Parquet)")[0][0]) * skew1 * F
    _bpr_a, _bpr_b, bpr_c = 2 * nf + 16, nf + 16, 14

    def o3(nfit, nm1, nfeat, drop_test):
        cuts = len(fit_cidx) - (1 if drop_test and 3 in fit_cidx else 0)
        ra = rpu_a / len(fit_cidx) * cuts
        return {"O3a_rows": ra * nfit, "O3a_bytes": ra * nfit * (2 * nfeat + 16), "O3b_rows": rpu_b * nm1,
                "O3b_bytes": rpu_b * nm1 * (nfeat + 16), "O3c_rows": o3c_rows, "O3c_bytes": o3c_rows * bpr_c}

    shrink = []
    cur = {"fit_halvings": 0, "m1_halvings": 0, "trimmed": 0, "drop_t_test": False}
    nfit_c, nm1_c, nfeat_c = n_fit, n_m1, nf
    est = o3(nfit_c, nm1_c, nfeat_c, False)
    trim = list(r["o3"]["trim_order"])

    def total_b(e):
        return e["O3a_bytes"] + e["O3b_bytes"] + e["O3c_bytes"]

    while total_b(est) > O3_BUDGET:
        if nfit_c / 2 >= r["o3"]["min_fit_users"]:
            nfit_c /= 2
            cur["fit_halvings"] += 1
            shrink.append(f"halve O3a fit users -> {int(nfit_c)}")
        elif cur["m1_halvings"] == 0 and nm1_c / 2 >= r["o3"]["min_m1_users"]:
            nm1_c /= 2
            cur["m1_halvings"] = 1
            shrink.append(f"halve M1 users -> {int(nm1_c)}")
        elif trim:
            trim.pop(0)
            nfeat_c -= 1
            cur["trimmed"] += 1
            if cur["trimmed"] % 9 == 1:
                shrink.append("trim features in registered order (120-day variants first)")
        elif not cur["drop_t_test"]:
            cur["drop_t_test"] = True
            shrink.append("drop T_test from O3a")
        else:
            shrink.append("O3 still over budget after the registered order")
            break
        est = o3(nfit_c, nm1_c, nfeat_c, cur["drop_t_test"])

    # S8 check (c): distinct kind / source_id over every events file, and over the
    # sampled repos the share of rows carrying the repo's modal rev (backfill rows
    # share one snapshot rev; live-tail commits each have their own)
    ks = _q(ps, f"SELECT source_id, kind, sum(rows), min(tmin), max(tmax) FROM file('{inv}/ev_skew.parquet') GROUP BY ALL ORDER BY ALL")
    ps_ = _q(ps, f"SELECT source_id, sum(rows), min(tmin), max(tmax) FROM file('{inv}/po_skew.parquet') GROUP BY ALL ORDER BY ALL")
    probe = {k: {"rows": v, "modal_rev_rows": p1o1.get("backfill_probe_modal_rev", {}).get(k, 0)} for k, v in p1o1.get("backfill_probe", {}).items()}
    kind_source = {"events": [{"source_id": a, "kind": b, "rows": int(c), "time_min": d, "time_max": e} for a, b, c, d, e in ks],
                   "posts": [{"source_id": a, "rows": int(b), "time_min": c, "time_max": d} for a, b, c, d in ps_],
                   "sample_probe_by_src_kind_codes": probe,
                   "codes": {"src": {"public_archive": 1, "gce_main": 2, "gce_bootstrap_live": 3, "netcup_live": 4, "repo_supplement": 5},
                             "kind": {"commit": 1, "account": 2, "identity": 3, "sync": 4}},
                   "next": "set rules.backfill_filter (SQL over src/kind codes) for S9 if a value separates backfill from the live tail; else the fallback stands"}
    o1s8 = _o1(ps, f"{ps.work}/p1/i*/o1.parquet")
    rows_d, revs_d = tot(o1s8, "dup_rows"), tot(o1s8, "dup_key_revs")
    hours = total_s / 3600
    proj = {
        "measured": {
            "inventory_s": inv_s, "read_rate_s": t_read, "bucket_s": t_b, "didmap_s": didmap_s,
            "p1_partition0_full": p1_0, "p1_sampled_partitions_s": p1_s, "p2_rate_1_400": {k: a.get(k) for k in ("seconds", "peak_rss_bytes")},
            "p2_rate_1_200": {k: c.get(k) for k in ("seconds", "peak_rss_bytes")},
            "bucket_rows_written": written, "sample_rows": sample_rows, "sample_share_of_all_rows": sample_rows / max(1, total_rows),
            "rows_total": total_rows, "partition_sizes": size, "post_partition_sizes": po,
            "subset_fraction": f_sub, "bucket_shards": getattr(ps, "bucket_shards", None),
            "distinct_key_growth": keys, "sampled_users": users_s,
        },
        "projected": {
            "bucket_hours": bucket_full_s / 3600, "phase1_hours": p1_full_s / 3600, "phase1_hours_from_sample": p1_from_sample_s / 3600,
            "phase2_hours": p2_full_s / 3600, "phase2_hours_linear_vs_power": [t2lin * skew2 / 3600, t2pow * skew2 / 3600],
            "globals_hours": glob_s / 3600, "teacher_hours": teacher_s / 3600, "didmap_hours": didmap_s / 3600,
            "main_run_hours_excl_staging": (total_s - stage_s) / 3600, "main_run_hours": hours,
            "peak_ram_phase1_bytes": p1_peak, "peak_ram_phase2_bytes": p2_peak,
            "peak_ram_phase2_linear_vs_power": [m2lin * maxskew2, m2pow * maxskew2], "phase2_model": model,
            "intermediate_bytes": p1_bytes * skew1 + p2_bytes_1 * skew2,
            "intermediate_bytes_phase1": p1_bytes * skew1, "intermediate_bytes_phase2": p2_bytes_1 * skew2,
            "maint_shuffle_bytes": maintd_bytes_full, "e0_edges": e0_full,
            "O3_before_shrink": o3(n_fit, n_m1, nf, False), "O3_after_shrink": est, "O3_shrink_steps": shrink,
            "O3_rules_changes": cur, "fit_users_full": n_fit, "m1_users_full": n_m1,
            "O3_rows_per_sampled_user": {"O3a": rpu_a, "O3b": rpu_b},
            "cross_shard_duplicate_rate": (rows_d - revs_d) / rows_d if rows_d else None,
            "spend_usd_at_spot": hours * SPOT,
        },
    }
    over = []
    if hours > LIMIT_H:
        over.append(f"main run {hours:.2f} h > {LIMIT_H} h")
    if max(p1_peak, p2_peak) > LIMIT_RAM:
        over.append(f"peak RAM {max(p1_peak, p2_peak) / 1e9:.0f} GB > 400 GB")
    if total_b(o3(n_fit, n_m1, nf, False)) > O3_BUDGET:
        over.append("O3 over 8 GB before the shrink order")
    proj["kind_source_listing"] = kind_source
    # O13 (eval_temporal.py engagement): rows for the sampled users toward partition-0 authors,
    # x post-file skew for all authors, per sampled user, x the full M1 user count (h % m1_mod == m1_rem)
    o13f = f"{ps.work}/p2_s200/j00/o13.parquet"
    if glob.glob(o13f):
        import os
        o13_rows = int(_q(ps, f"SELECT count() FROM file('{o13f}', Parquet)")[0][0])
        o13_bytes = os.path.getsize(o13f)
        n13 = int(_q(ps, f"SELECT countIf(o13m1 = 1) FROM file('{ps.g}/didmap.parquet')")[0][0]) * frac_actor
        rpu13 = o13_rows * skew2 / max(1, users_s) * F
        bpr13 = o13_bytes / max(1, o13_rows)
        m2n = int(_q(ps, f"SELECT countIf(m2c = 1) FROM file('{ps.g}/didmap.parquet')")[0][0])
        proj["projected"]["O13"] = {
            "rows_per_user": rpu13, "bytes_per_row_parquet": bpr13, "m1_users": n13,
            "m1_rows": rpu13 * n13, "m1_bytes": rpu13 * n13 * bpr13,
            "m2_users_listed": m2n, "m2_rows": rpu13 * m2n, "m2_bytes": rpu13 * m2n * bpr13,
            "note": "M2 cohort users are new (Aug 15 - Sep 8), so rows_per_user overstates their rows",
        }
    # S1 R4: O3b u8 scales = p99.9 of |x_k| on S8's sample (the S8-s200 O3a rows), written for R-S9
    if glob.glob(o3ag):
        names = [f["name"] for f in r.features]
        q = _q(ps, "SELECT " + ", ".join(f"quantileExact(0.999)(abs(toFloat64(`{n}`)))" for n in names) + f" FROM file('{o3ag}', Parquet)")
        scales = {n: (float(v) if float(v) > 0 else 1.0) for n, v in zip(names, q[0])}
        (ps.out30 / "S8").mkdir(parents=True, exist_ok=True)
        (ps.out30 / "S8" / "o3b_scales.json").write_text(json.dumps(scales, indent=1, sort_keys=True) + "\n")
        proj["o3b_scales_file"] = "S8/o3b_scales.json (pass to S9 with --o3b-scales; its sha256 goes in R-S9)"
    proj["descoping_triggered"] = over
    proj["descoping_order"] = ["O3 shrink order (above)", "drop T_test from O3a", "cap O8 at 32 posts per producer",
                               "cut O12 to 20k pairs", "drop O11", "split into two Spot sessions (fresh ask)"] if over else []
    proj["notes"] = ["phase 2 hours/RAM: max of linear and distinct-key power-law extrapolation, x post-file skew",
                     "staging time comes from the launcher's receipt (pass --stage-seconds to include it)"]
    return json.loads(json.dumps(proj, default=float))


def render(p: dict) -> str:
    pr = p["projected"]
    gb = 1e9
    lines = [
        "S8 projection for R-S9",
        f"  main run (excl. staging): {pr['main_run_hours_excl_staging']:.2f} h; spend at Spot ${pr['spend_usd_at_spot']:.2f}",
        f"  bucketing {pr['bucket_hours']:.2f} h | phase 1 {pr['phase1_hours']:.2f} h (from sample {pr['phase1_hours_from_sample']:.2f} h)"
        f" | phase 2 {pr['phase2_hours']:.2f} h | globals {pr['globals_hours']:.2f} h | teacher {pr['teacher_hours']:.2f} h",
        f"  peak RAM: phase 1 {pr['peak_ram_phase1_bytes'] / gb:.1f} GB, phase 2 {pr['peak_ram_phase2_bytes'] / gb:.1f} GB",
        f"  intermediates {pr['intermediate_bytes'] / gb:.1f} GB (MAINT shuffle {pr['maint_shuffle_bytes'] / gb:.2f} GB)",
        f"  O3a {pr['O3_after_shrink']['O3a_bytes'] / gb:.2f} GB / {pr['O3_after_shrink']['O3a_rows']:.0f} rows; "
        f"O3b {pr['O3_after_shrink']['O3b_bytes'] / gb:.2f} GB; O3c {pr['O3_after_shrink']['O3c_bytes'] / gb:.3f} GB "
        f"(before shrink: {sum(v for k, v in pr['O3_before_shrink'].items() if k.endswith('bytes')) / gb:.2f} GB)",
        f"  shrink steps: {pr['O3_shrink_steps'] or 'none'}",
        f"  cross-shard duplicate rate: {pr['cross_shard_duplicate_rate']}",
        f"  sample share of rows: {p['measured']['sample_share_of_all_rows']:.4f}",
        f"  descoping triggered: {p['descoping_triggered'] or 'no'}",
        "  O13: " + (f"{pr['O13']['m1_rows']:.0f} M1 rows, {pr['O13']['m1_bytes'] / gb:.2f} GB; {pr['O13']['m2_rows']:.0f} M2 rows "
                     f"({pr['O13']['rows_per_user']:.1f} rows/user)" if "O13" in pr else "not measured"),
        "  kind / source_id (events): " + "; ".join(f"{x['source_id']}/{x['kind']} {x['rows']} rows {x['time_min']}..{x['time_max']}"
                                                     for x in p["kind_source_listing"]["events"]),
        "  modal-rev share by (src, kind) code on the sample: " + "; ".join(
            f"{k} {v['modal_rev_rows'] / max(1, v['rows']):.3f}" for k, v in p["kind_source_listing"]["sample_probe_by_src_kind_codes"].items()),
    ]
    return "\n".join(lines) + "\n"


def _known(ps) -> dict:
    """What a partial S8 run already measured (no step depends on another here)."""
    import datetime as dt
    steps = {}
    for name, st in ps.stats.items():
        if isinstance(st, dict) and "seconds" in st:
            steps[name] = {k: st.get(k) for k in ("seconds", "peak_rss_bytes", "mean_cpu_pct", "skipped") if k in st}
    k = {"stage_seconds": float(getattr(ps.a, "stage_seconds", 0) or 0), "steps": steps,
         "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    inv = (ps.stats.get("inventory") or {}).get("staged")
    if inv:
        k["staged"] = inv
    donedir = ps.buckets / "_done"
    if donedir.exists():
        per = [json.loads(p.read_text()) for p in donedir.glob("*.json")]
        k["bucket_progress"] = {"shards_done": len(per), "shards_total": ps._n_shards or None,
                                "shard_seconds": sorted(round(x["seconds"]) for x in per)}
    rr = (ps.stats.get("read_rate") or {}).get("seconds")
    b = ps.stats.get("bucket") or {}
    if rr and b.get("seconds") and inv:
        # S9 bucketing writes every row; S8 wrote ~7%. Write throughput from the time S8 spent
        # beyond a full read (bucketing reads ~1.5x read_rate's columns), applied to all rows.
        evb, pob = f"file('{ps.buckets}/ev/*/*.parquet', Parquet, 'u UInt32')", f"file('{ps.buckets}/po/*/*.parquet', Parquet, 'u UInt32')"
        written = int(_q(ps, f"SELECT (SELECT count() FROM {evb}) + (SELECT count() FROM {pob})")[0][0])
        total = sum(v["rows"] for v in inv.values())
        read_s = 1.5 * rr
        write_rate = written / max(60.0, b["seconds"] - read_s)
        k["s9_bucket_estimate"] = {"rows_written_s8": written, "rows_total": total, "read_component_s": read_s,
                                   "write_rows_per_s": write_rate, "s9_bucket_hours": (read_s + total / write_rate) / 3600,
                                   "method": "1.5 x read_rate + all rows / S8's write rate beyond the read (upper-bound style)"}
    p10 = ps.stats.get("p1_00") or {}
    if p10.get("seconds") and p10.get("bucket_rows") and not p10.get("skipped"):
        # S9 phase 1 from this run: partition 0 ran in full, so rows/s at full partition size is measured
        # directly; S9 runs 16 partitions of the sizes the inventory counted, one after another.
        sizes = {}
        try:
            inv = ps.work / "inv"
            for part, n in _q(ps, f"SELECT part, sum(rows) FROM (SELECT part, rows FROM file('{inv}/ev_skew.parquet') "
                                  f"UNION ALL SELECT part, rows FROM file('{inv}/po_skew.parquet')) GROUP BY part"):
                sizes[int(part)] = int(n)
        except Exception:  # noqa: BLE001
            pass
        rate = p10["bucket_rows"] / p10["seconds"]
        k["s9_phase1_estimate"] = {
            "p1_00_rows": p10["bucket_rows"], "p1_00_seconds": p10["seconds"], "p1_00_chunks": p10.get("chunks"),
            "p1_00_peak_rss_bytes": p10.get("peak_rss_bytes"), "rows_per_s": rate,
            "s9_phase1_hours": (sum(sizes.values()) / rate / 3600) if sizes else None,
            "largest_partition_hours": (max(sizes.values()) / rate / 3600) if sizes else None,
            "method": "full-size partition 0 rows/s x every partition's inventory row count (sequential)"}
    return k


def incremental(ps, after: str) -> dict:
    out = {"partial": True, "after_step": after, "known": _known(ps)}
    try:
        full = project(ps)
        out.update(full)
        out["partial"] = False
    except Exception as e:  # noqa: BLE001 - steps not run yet
        out["full_projection_unavailable"] = repr(e)[:300]
    return out


def render_partial(p: dict) -> str:
    if not p.get("partial"):
        return render(p)
    k = p["known"]
    lines = [f"S8 PARTIAL projection (after {p['after_step']}, {k['updated_utc']})",
             f"  staging {k['stage_seconds']:.0f}s"]
    for name, st in k["steps"].items():
        lines.append(f"  {name}: {st.get('seconds', 0):.0f}s, peak {((st.get('peak_rss_bytes') or 0) / 2**30):.1f} GiB"
                     + (" (skipped: done earlier)" if st.get("skipped") else ""))
    if "bucket_progress" in k:
        bp = k["bucket_progress"]
        lines.append(f"  bucketing: {bp['shards_done']}/{bp['shards_total']} shards")
    if "s9_phase1_estimate" in k and k["s9_phase1_estimate"].get("s9_phase1_hours"):
        e = k["s9_phase1_estimate"]
        lines.append(f"  S9 phase-1 estimate: {e['s9_phase1_hours']:.2f} h ({e['rows_per_s'] / 1e6:.2f} M rows/s on partition 0, "
                     f"{e['p1_00_chunks']} sub-chunks)")
    if "s9_bucket_estimate" in k:
        lines.append(f"  S9 bucketing estimate: {k['s9_bucket_estimate']['s9_bucket_hours']:.2f} h")
    return "\n".join(lines) + "\n"


PAIR_BYTES_DEFAULT = 2400.0  # synthetic: 12 GiB per 5M pairs for the 153-column per-pair aggregate
PAIR_SCALED = ("CREATE TABLE l2_", "INSERT INTO acc_", "CREATE TABLE x_maint", "INSERT INTO FUNCTION file( %e0_", "DROP TABLE l2_")
INBOUND_SCALED = ("CREATE TABLE te ", "CREATE TABLE r0 ", "CREATE TABLE tr ", "CREATE TABLE crj ", "CREATE TABLE evx ",
                  "CREATE TABLE bp ", "CREATE TABLE bpj ", "CREATE TABLE eng_light ", "CREATE TABLE lev ", "CREATE TABLE levj ")


def phase2_model(ps, o1a: dict, o1c: dict, r1: float, r2: float, F: float = 1.0) -> dict:
    from sqlc import ram_bytes
    pa, pc = sum(o1a.get("pairs_prod", {}).values()), sum(o1c.get("pairs_prod", {}).values())
    tc = sum(o1c.get("typed_engagement", {}).values())
    if not pc or not tc:
        return {}
    inbound_pairs = max(0.0, (pc - pa) / (r2 - r1)) * F           # pairs contributed by all actors at rate 1
    own_pairs = max(0.0, pc - inbound_pairs * r2 / F)              # the partition's own rows (rate-independent)
    pairs1 = own_pairs * F + inbound_pairs
    eng_ratio = (tc / r2 * F) / tc                                 # inbound engagement growth s200 -> rate 1
    pair_ratio = pairs1 / pc
    st = _st(ps, "p2_s200_00")
    stmts = st.get("statements") or st.get("slowest_statements") or []
    k_s8 = st.get("subchunks") or ps.rules["p2_user_subchunks"]
    ram = ram_bytes()
    # S9's own sub-chunk rule (passrun.p2_chunks), applied to the projected pair count
    k9 = max(int(ps.rules["p2_user_subchunks"]), -(-int(pairs1 * PAIR_BYTES_DEFAULT) // int(ram * 0.25)))
    secs = {"fixed": 0.0, "inbound": 0.0, "pairs": 0.0}
    for x in stmts:
        cat = "pairs" if x["what"].startswith(PAIR_SCALED) else "inbound" if x["what"].startswith(INBOUND_SCALED) else "fixed"
        secs[cat] += x["seconds"]
    known = sum(secs.values())
    rest = max(0.0, (st.get("seconds") or 0) - known)
    seconds1 = (secs["fixed"] + rest) * F + secs["inbound"] * eng_ratio + secs["pairs"] * pair_ratio * F if stmts else None
    fixed_peak = max([x.get("peak_rss_bytes", 0) for x in stmts if not x["what"].startswith(PAIR_SCALED)] or [0])
    l2 = [x for x in stmts if x["what"].startswith("CREATE TABLE l2_") and x.get("peak_rss_bytes")]
    if l2 and pc:
        base = min(x.get("peak_rss_bytes", 0) for x in stmts if x.get("peak_rss_bytes")) if stmts else 0
        per_pair = max(PAIR_BYTES_DEFAULT, (max(x["peak_rss_bytes"] for x in l2) - base) / (pc / k_s8))
    else:
        per_pair = PAIR_BYTES_DEFAULT
    l2_chunk_bytes = per_pair * pairs1 / k9
    peak1 = max(fixed_peak * F, fixed_peak * F + l2_chunk_bytes) if fixed_peak else None
    return {"pairs_s400": pa, "pairs_s200": pc, "own_pairs": own_pairs, "inbound_pairs_rate1": inbound_pairs, "pairs_rate1": pairs1,
            "pair_bytes": per_pair, "subchunks_s9": k9, "l2_chunk_bytes_rate1": l2_chunk_bytes, "fixed_peak_bytes_s200": fixed_peak,
            "seconds_by_category_s200": secs, "seconds_rate1": seconds1, "peak_bytes_rate1": peak1,
            "note": "fixed statements as measured; inbound ones x inbound-engagement growth; pair ones x pair growth; "
                    "peak = largest fixed statement + one sub-chunk of the per-pair aggregate (conservative: adds them)"}
