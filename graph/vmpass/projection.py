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

    # bucketing: read cost + per-row write cost
    b = _st(ps, "bucket")
    rr = _st(ps, "read_rate")
    smp = f"u IN (SELECT uid FROM file('{ps.g}/didmap.parquet') WHERE s200 = 1)"
    evb, pob = f"file('{ps.buckets}/ev/*.parquet', Parquet, 'u UInt32')", f"file('{ps.buckets}/po/*.parquet', Parquet, 'u UInt32')"
    brows = _q(ps, f"SELECT (SELECT count() FROM {evb}) + (SELECT count() FROM {pob}), "
                   f"(SELECT count() FROM {evb} WHERE {smp}) + (SELECT count() FROM {pob} WHERE {smp})")
    written, sample_rows = (int(brows[0][0]), int(brows[0][1])) if brows else (0, 0)
    t_read = rr.get("seconds", 0) or 0
    t_b = b.get("seconds", 0) or 0
    per_row = max(0.0, t_b - t_read) / max(1, written)
    bucket_full_s = t_read + per_row * total_rows

    # phase 1
    p1_0 = _st(ps, "p1_00")
    p1_s = sum((_st(ps, f"p1_{i:02d}").get("seconds", 0) or 0) for i in range(1, N))
    p1_full_s = (p1_0.get("seconds", 0) or 0) * skew1
    p1_from_sample_s = p1_s * r["samples"]["s8_actor_mod"][0] * N / max(1, N - 1)
    p1_peak = (p1_0.get("peak_rss_bytes", 0) or 0) * maxskew1
    p1_bytes = _bytes(ps, "p1_00")

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

    t2, t2lin, t2pow = extrap(a.get("seconds", 0) or 0, c.get("seconds", 0) or 0)
    m2, m2lin, m2pow = extrap(a.get("peak_rss_bytes", 0) or 0, c.get("peak_rss_bytes", 0) or 0)
    p2_full_s = t2 * skew2
    p2_peak = m2 * maxskew2
    b2a, b2c = _bytes(ps, "p2_s400_00"), _bytes(ps, "p2_s200_00")
    p2_bytes_1 = extrap(b2a, b2c)[0]

    # globals between/after phases, scaled by the phase-1 output growth
    p1_growth = skew1 * 1.0
    glob_s = sum((_st(ps, s).get("seconds", 0) or 0) for s in ("ustats_merge", "gates_merge", "lblk", "o5prod", "o7")) * p1_growth
    e0_rows = 0
    e0g = f"{ps.work}/p2_s200/j00/e0_*.parquet"
    if glob.glob(e0g):
        e0_rows = int(_q(ps, f"SELECT count() FROM file('{e0g}', Parquet)")[0][0])
    e0_full = e0_rows / r2 * skew2
    teacher_s = 4 * 420 * max(0.05, e0_full / 2.04e9)  # simgraph: 7 min per 2.04B edges, 4 teachers
    sum(_bytes(ps, s) for s in ("maint",))
    maintd = glob.glob(f"{ps.work}/p2_s200/j00/maintd/*.parquet")
    maintd_bytes_full = 0
    if maintd:
        import os
        maintd_bytes_full = sum(os.path.getsize(f) for f in maintd) / r2 * skew2
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
    rpu_a = o3a_rows_s / max(1, users_s) + b_rows
    rpu_b = o3b_rows_s / max(1, users_s)
    o3c_rows = 0
    o3cf = glob.glob(f"{ps.out30}/O3c/i00.parquet")
    if o3cf:
        o3c_rows = int(_q(ps, f"SELECT count() FROM file('{o3cf[0]}', Parquet)")[0][0]) * skew1
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
            "distinct_key_growth": keys, "sampled_users": users_s,
        },
        "projected": {
            "bucket_hours": bucket_full_s / 3600, "phase1_hours": p1_full_s / 3600, "phase1_hours_from_sample": p1_from_sample_s / 3600,
            "phase2_hours": p2_full_s / 3600, "phase2_hours_linear_vs_power": [t2lin * skew2 / 3600, t2pow * skew2 / 3600],
            "globals_hours": glob_s / 3600, "teacher_hours": teacher_s / 3600, "didmap_hours": didmap_s / 3600,
            "main_run_hours_excl_staging": (total_s - stage_s) / 3600, "main_run_hours": hours,
            "peak_ram_phase1_bytes": p1_peak, "peak_ram_phase2_bytes": p2_peak,
            "peak_ram_phase2_linear_vs_power": [m2lin * maxskew2, m2pow * maxskew2],
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
        "  kind / source_id (events): " + "; ".join(f"{x['source_id']}/{x['kind']} {x['rows']} rows {x['time_min']}..{x['time_max']}"
                                                     for x in p["kind_source_listing"]["events"]),
        "  modal-rev share by (src, kind) code on the sample: " + "; ".join(
            f"{k} {v['modal_rev_rows'] / max(1, v['rows']):.3f}" for k, v in p["kind_source_listing"]["sample_probe_by_src_kind_codes"].items()),
    ]
    return "\n".join(lines) + "\n"
