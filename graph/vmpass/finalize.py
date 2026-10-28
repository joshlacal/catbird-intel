"""Finalize: merge O1, apply the LU courtesy discount to O2, run checks (a)-(k)
(plan §3.2) and write the receipt (every output's path, bytes and sha256, from
the ledger markers, so it is complete after a resumed restart too)."""
from __future__ import annotations

import datetime as dt
import glob
import json
import pathlib

import chrun
from sqlc import PRELUDE, S, out

LIKE_BAND = (17.15e9, 17.57e9)
REPOST_BAND = (2.31e9, 2.49e9)


def _tsv(ps, sql: str, name: str) -> list[list[str]]:
    txt = chrun.query(PRELUDE + sql + " FORMAT TSV", ps.work, name)
    return [line.split("\t") for line in txt.strip().splitlines() if line]


def finalize(ps) -> dict:
    r = ps.rules
    s8 = ps.mode == "s8"
    sub = "p2_s200" if s8 else "p2"
    o1dir = ps.out30 / "O1"
    o1dir.mkdir(parents=True, exist_ok=True)
    frags = sorted(glob.glob(f"{ps.work}/p1/i*/o1.parquet")) + sorted(glob.glob(f"{ps.work}/{sub}/j*/o1.parquet"))
    srcs = " UNION ALL ".join(f"SELECT * FROM file('{g}', Parquet, '{S['o1']}')" for g in
                              (f"{ps.work}/p1/i*/o1.parquet", f"{ps.work}/{sub}/j*/o1.parquet") if glob.glob(g))
    if frags:
        chrun.run_script(PRELUDE + f"""
{out(str(o1dir / 'o1_counts.parquet'))}
SELECT metric, k1, k2, k3, multiIf(endsWith(metric, '_min'), min(v), endsWith(metric, '_max'), max(v), sum(v)) AS v
FROM ({srcs}) GROUP BY metric, k1, k2, k3 ORDER BY metric, k1, k2, k3;""", ps.work, "o1_merge")
    o1 = {}
    for m, k1, k2, k3, v in _tsv(ps, f"SELECT * FROM file('{o1dir}/o1_counts.parquet')", "o1_read") if frags else []:
        o1.setdefault(m, {})[f"{k1}|{k2}|{k3}"] = float(v)

    def g(m, key=None, default=0.0):
        d = o1.get(m, {})
        if key is None:
            return sum(d.values())
        return d.get(key, default)

    checks: dict = {}
    inv = ps.stats.get("inventory", {}) or (ps.ledger.done("inventory") or {}).get("stats", {})
    checks["a_rows_equal_manifest"] = inv.get("check_a_rows_equal_manifest", "no manifest given")
    checks["a_staged"] = inv.get("staged")
    like_live = g("dup_keys", "3||") - g("deleted_keys", "3||")
    rp_live = g("dup_keys", "4||") - g("deleted_keys", "4||")
    checks["b_likes_live_records"] = like_live
    checks["b_reposts_live_records"] = rp_live
    if not s8:
        checks["b_likes_in_band"] = LIKE_BAND[0] <= like_live <= LIKE_BAND[1]
        checks["b_reposts_in_band"] = REPOST_BAND[0] <= rp_live <= REPOST_BAND[1]
    checks["c_positive_control"] = {
        "likers_by_old_new_fallback": o1.get("pc_likers", {}), "likes_by_old_new_fallback": o1.get("pc_likes", {}),
        "likers_on_fallback_definition": g("pc_used_fallback"),
        "backfill_filter": r["backfill_filter"] or "none: fallback min(time) over all rows",
        "hash_list": "outputs/<run>/positive-control/i*.parquet (h, bucket<20, n, e_new, e_old, e_fb); compare on the Mac",
    }
    rows, revs = g("dup_rows"), g("dup_key_revs")
    checks["d_duplicate_share"] = {"rows": rows, "distinct_key_revisions": revs, "keys": g("dup_keys"),
                                   "cross_shard_duplicate_rate": (rows - revs) / rows if rows else None}
    hist = o1.get("follow_created_minus_tid", {})
    tot = sum(hist.values())
    over = sum(v for k, v in hist.items() if int(k.split("|")[0]) > 86400)

    def q(pct):
        acc = 0.0
        for k, v in sorted(hist.items(), key=lambda kv: int(kv[0].split("|")[0])):
            acc += v
            if tot and acc >= pct * tot:
                return f"<= {k.split('|')[0]} s"
        return None

    checks["e_follow_backdating"] = {"follows": tot, "median_abs_bucket": q(0.5), "p99_abs_bucket": q(0.99),
                                     "share_over_1_day": over / tot if tot else None}
    sup_prod = _tsv(ps, f"SELECT count() FROM file('{ps.out30}/{'S8-s200/' if s8 else ''}O2/prod/*.parquet') WHERE supplement = 1", "sup") \
        if glob.glob(f"{ps.out30}/{'S8-s200/' if s8 else ''}O2/prod/*.parquet") else [["0"]]
    checks["f_supplement"] = {"supplement_actors": g("supplement_actors"), "supplement_producers_in_O2": int(sup_prod[0][0]),
                              "LU_typed_engagements": g("typed_engagement", "4||")}
    excl = _tsv(ps, f"""SELECT purge_reason, count(), sum(hide), sum(synth_no), sum(bulk_no), sum(is_prod)
                       FROM file('{ps.g}/didmap.parquet', Parquet, '{S['didmap']}') GROUP BY purge_reason ORDER BY 1""", "excl")
    reasons = {0: "kept", 1: "relay active=false", 2: "absent from complete relay listing", 3: "extract account inactive", 4: "bulkDataset:false"}
    checks["g_exclusions"] = {reasons[int(x[0])]: {"accounts": int(x[1]), "hide_from_recs": int(x[2]), "synthetic_false": int(x[3]),
                                                   "bulk_false": int(x[4]), "producers": int(x[5])} for x in excl}
    hours = (sum(v.get("seconds", 0) or 0 for v in ps.stats.values() if isinstance(v, dict) and not v.get("skipped"))) / 3600
    hours_all = (sum(v.get("seconds", 0) or 0 for v in ps.stats.values() if isinstance(v, dict))) / 3600
    checks["h_spend"] = {"pass_hours_this_session": round(hours, 3), "pass_hours_all_sessions": round(hours_all, 3),
                         "estimate_usd_at_spot": round(hours_all * 2.715, 2),
                         "note": "VM wall time incl. staging is in the launcher receipt; label-filtered Billing -> Reports is authoritative"}
    checks["i_posts_live_12mo"] = {"total": g("posts_live_12mo"), "by_is_reply_is_quote": o1.get("posts_live_12mo", {})}
    checks["j_truncated_gates"] = {"truncated_gate_records": g("gate_records_truncated"), "gate_rows": o1.get("gates", {}),
                                   "note": "hide and detach counts are lower bounds (P8)"}
    checks["k_gcs_ops"] = {"store30": ps.s30.ops, "store60": ps.s60.ops}

    # LU courtesy discount (§4.3 LU, F4/Vf18): c per (follower decile, top language) from normal producers,
    # emitted as a table plus discounted O2 columns written beside O2.
    o2g = f"{ps.out30}/{'S8-s200/' if s8 else ''}O2"
    if glob.glob(f"{o2g}/prod/*.parquet"):
        for tag in ("eval", "prod"):
            chrun.run_script(PRELUDE + f"""
CREATE TABLE o2 ENGINE = Memory AS SELECT p, followers, if(empty(top_langs), '', top_langs[1]) AS lang, light_eng_nonlu, light_eng_luonly,
  D_nonlu, D_luonly, courtesy_likes, likes_all, supplement FROM file('{o2g}/{tag}/*.parquet');
CREATE TABLE dec ENGINE = Memory AS SELECT p, toUInt8(intDiv((row_number() OVER (ORDER BY followers, p) - 1) * 10, count() OVER ())) AS d FROM o2;
CREATE TABLE ccm ENGINE = Memory AS
SELECT d, lang, toFloat32(sum(courtesy_likes) / greatest(1, sum(likes_all))) AS c FROM o2 INNER JOIN dec USING (p) WHERE supplement = 0 GROUP BY d, lang;
CREATE TABLE cc ENGINE = Join(ANY, LEFT, d, lang) AS SELECT * FROM ccm;
{out(str(pathlib.Path(o2g) / f'courtesy_share_{tag}.parquet'))} SELECT d AS follower_decile, lang, c FROM ccm;
{out(str(pathlib.Path(o2g) / f'lu_discounted_{tag}.parquet'))}
SELECT p, toFloat32(light_eng_nonlu + (1 - joinGet('cc', 'c', d, lang)) * light_eng_luonly) AS light_engagers_discounted,
  toFloat32(D_nonlu + (1 - joinGet('cc', 'c', d, lang)) * D_luonly) AS D_discounted
FROM o2 INNER JOIN dec USING (p);""", ps.work, f"lu_{tag}")

    rec = {"run_id": ps.run_id, "mode": ps.mode, "rules_path": r.path, "rules_sha256": r.sha256, "rules_draft": r.draft,
           "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "checks": checks,
           "steps": {k: {kk: vv for kk, vv in v.items() if kk not in ("stdout", "projection")} for k, v in ps.stats.items() if isinstance(v, dict)},
           "outputs": []}
    for marker in sorted(ps.stats):
        m = ps.ledger.done(marker)
        if m:
            rec["outputs"] += [f for f in m.get("manifest", []) if f["path"].startswith(("out30/", "out60/"))]
    d = ps.out30 / "receipt"
    d.mkdir(parents=True, exist_ok=True)
    (d / "receipt.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    (o1dir / "summary.json").write_text(json.dumps({"checks": checks, "o1": o1}, indent=1, sort_keys=True, default=str))
    return {"checks": checks}
