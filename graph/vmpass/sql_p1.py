"""Phase 1, actor-partitioned (`cityHash64(did) % 16`), plan §3.2.

Everything an actor did is in its own partition, so this step does the
actor-side work: latest-revision dedupe per (did, collection, rkey), one like
per (liker, subject), follow state at like time, daily and per-pair caps,
follow and block burst sizes, per-user stats (d_u, light/heavy, a_u), the O3
candidate draw, and re-keys the deduplicated, integer-keyed rows by the
target's partition for phase 2. Outputs that are final (O3b labels, O3c,
O5-raw blocks and follows, the positive-control list) are written here too.
"""
from __future__ import annotations

from rules import day_start, epoch
from sqlc import NONE, PRELUDE, S, out, settings_sql, src

URI_RE = r"at://[^\"\\\\]+"  # SQL literal 'at://[^"\\]+' -> regex at://[^"\]+


def _arr(xs) -> str:
    return "[" + ", ".join(str(x) for x in xs) + "]"


def p1(rules, i: int, evf: str, pof: str, dm: str, side: str, o: str, final: dict, opts: dict) -> str:
    r = rules
    _N, W = r.N, r.W
    wd = r["window_days"]
    cuts = [r.cut[k] for k in r.cut_names]
    cdays = [c // 86400 for c in cuts]
    C = _arr(cuts)
    CD = _arr(cdays)
    bw = r["burst_window_s"]
    thr = r["burst_threshold"]
    t12 = day_start(r["twelve_month_start"])
    tfit_lo = cuts[0] - W
    T = r.cut["T"]
    cr_w = r["cr_window_s"]
    o3 = r["o3"]
    fit_cidx = [0, 1] + ([] if o3["drop_t_test"] else [3])
    fitflag = opts.get("fit_flag", "fit")
    m1flag = opts.get("m1_flag", "m1")
    bf = r["backfill_filter"] or "0"
    pc = r["positive_control"]
    pc0 = day_start(pc["like_day"])
    snapmin = epoch(pc["snapshot_min"])
    m2a, m2b = day_start(r["m2_cohort"][0]), epoch(r["m2_cohort"][1])
    m2o = r["m2_outcome_days"] * 86400
    m1end = epoch(r["m1_label_end"])
    o5a, o5b = epoch(r["o5_follow_window"][0]), epoch(r["o5_follow_window"][1])
    cur = r["curate"]
    maps = opts.get("have_maps", False)
    tn = r.cut["T_now"]
    is_prod = "joinGet('dmu', 'is_prod', {x}) = 1"

    def prod(x: str) -> str:
        return is_prod.format(x=x)

    if maps:
        comm_sql = f"""
CREATE TABLE m1 ENGINE = Join(ANY, LEFT, pid) AS SELECT pid, comm + 1 AS c FROM file('{side}/comm_cand1.tsv', TSV, 'pid UInt32, comm Int32');
CREATE TABLE m5 ENGINE = Join(ANY, LEFT, pid) AS SELECT pid, comm + 1 AS c FROM file('{side}/comm_cand5.tsv', TSV, 'pid UInt32, comm Int32');
CREATE TABLE ucomm ENGINE = Join(ANY, LEFT, u) AS
SELECT u, toInt32(argMaxIf(c1, n1, c1 > 0)) AS cc1, toInt32(argMaxIf(c5, n5, c5 > 0)) AS cc5 FROM (
  SELECT u, joinGet('m1', 'c', p) AS c1, joinGet('m5', 'c', p) AS c5,
    count() OVER (PARTITION BY u, c1) AS n1, count() OVER (PARTITION BY u, c5) AS n5
  FROM fol WHERE ft <= {tn} AND {tn} < dt) GROUP BY u;"""
    else:
        comm_sql = "CREATE TABLE ucomm (u UInt32, cc1 Int32, cc5 Int32) ENGINE = Join(ANY, LEFT, u);"

    return PRELUDE + settings_sql(r) + f"""
CREATE TABLE dmu ENGINE = Join(ANY, LEFT, uid) AS
SELECT uid, part, is_prod, ls, fit, m1, pc, h, hide, synth_no, plc_t, web, followers, s200, s400 FROM file('{dm}', Parquet, '{S['didmap']}');
CREATE TABLE ev ENGINE = MergeTree ORDER BY (coll, u, rk) AS SELECT * FROM {src(evf, 'ev')};
CREATE TABLE st ENGINE = MergeTree ORDER BY (coll, u, rk) AS
SELECT coll, u, rk, argMax(op, (rev, seq)) AS lop, max(rev) AS lrev, count() AS nrows, uniqExact(rev) AS nrevs,
  countIf(op != 3) AS ncr,
  argMaxIf(tuple(t, tok, subj, spart, sph, scoll, st, lh, gph, created, extra, trunc, name, descr, feed), (rev, seq), op != 3) AS pl
FROM ev WHERE kind = 1 GROUP BY coll, u, rk;
CREATE TABLE live ENGINE = MergeTree ORDER BY (coll, u, rk) AS
SELECT coll, u, rk, lop, lrev, pl.1 AS t, pl.2 AS tok, pl.3 AS subj, pl.4 AS spart, pl.5 AS sph, pl.6 AS scoll, pl.7 AS st,
  pl.8 AS lh, pl.9 AS gph, pl.10 AS created, pl.11 AS extra, pl.12 AS trunc, pl.13 AS name, pl.14 AS descr, pl.15 AS feed
FROM st WHERE ncr > 0;

-- follows: one row per follow record; dt = delete time (TID of the delete commit) or never
CREATE TABLE fol ENGINE = MergeTree ORDER BY (u, p) AS
SELECT u, subj AS p, t AS ft, toUInt32(if(lop = 3, intDiv(lrev, 1000000), {NONE})) AS dt,
  toUInt16(count() OVER (PARTITION BY u ORDER BY t RANGE BETWEEN {bw} PRECEDING AND {bw} FOLLOWING)) AS b, spart AS ppart
FROM live WHERE coll = 1 AND subj != {NONE} AND subj != u;
CREATE TABLE fiv ENGINE = Join(ANY, LEFT, u, p) AS SELECT u, p, groupArray((ft, dt)) AS iv FROM fol GROUP BY u, p;

-- likes and reposts: one per (actor, subject); caps; follow state at event time
CREATE TABLE lk ENGINE = MergeTree ORDER BY (u, a) AS
SELECT u, a, ph, t, s_t, apart,
  toFloat32(least(1, {r['like_day_cap']} / count() OVER (PARTITION BY u, intDiv(t, 86400)))
          * least(1, {r['like_pair_day_cap']} / count() OVER (PARTITION BY u, a, intDiv(t, 86400)))) AS m,
  toUInt8(arrayExists(x -> x.1 <= t AND t < x.2, joinGet('fiv', 'iv', u, a))) AS fol
FROM (SELECT u, sph AS ph, any(subj) AS a, any(spart) AS apart, min(t) AS t, any(st) AS s_t FROM live
      WHERE coll = 3 AND lop != 3 AND scoll = 1 AND subj != {NONE} AND subj != u GROUP BY u, sph);
CREATE TABLE rp ENGINE = MergeTree ORDER BY (u, a) AS
SELECT u, a, ph, t, s_t, apart,
  toFloat32(least(1, {r['repost_day_cap']} / count() OVER (PARTITION BY u, intDiv(t, 86400)))) AS m,
  toUInt8(arrayExists(x -> x.1 <= t AND t < x.2, joinGet('fiv', 'iv', u, a))) AS fol
FROM (SELECT u, sph AS ph, any(subj) AS a, any(spart) AS apart, min(t) AS t, any(st) AS s_t FROM live
      WHERE coll = 4 AND lop != 3 AND scoll = 1 AND subj != {NONE} AND subj != u GROUP BY u, sph);

-- the actor's own posts (post index for phase 2) and its outbound replies and quotes
CREATE TABLE po ENGINE = MergeTree ORDER BY (u, rk) AS
SELECT u, rk, rkey, t, tok, rev, seq, op, ing, src, ph, par_ph, par_a, par_t, root_ph, root_a, q_ph, q_a, q_t, img, langs FROM {src(pof, 'po')};
CREATE TABLE pi ENGINE = MergeTree ORDER BY (u, ph) AS
SELECT u, pl.1 AS ph, pl.2 AS rkey, pl.3 AS t, toUInt8(pl.4 != 0) AS is_reply, pl.4 AS par_ph, pl.5 AS par_a, pl.6 AS par_t,
  pl.7 AS root_ph, pl.8 AS root_a, toUInt8(pl.9 != 0) AS is_quote, pl.9 AS q_ph, pl.10 AS q_a, pl.11 AS q_t, pl.12 AS img, pl.13 AS langs
FROM (SELECT u, rk, argMax(op, (rev, seq)) AS lop, countIf(op != 3) AS ncr,
        argMaxIf(tuple(ph, rkey, t, par_ph, par_a, par_t, root_ph, root_a, q_ph, q_a, q_t, img, langs), (rev, seq), op != 3) AS pl
      FROM po GROUP BY u, rk) WHERE lop != 3 AND ncr > 0;
CREATE TABLE rep ENGINE = MergeTree ORDER BY (u, a) AS
SELECT u, a, kind, r_ph, t, x_ph, x_t, root_ph, root_a,
  toUInt8(arrayExists(x -> x.1 <= t AND t < x.2, joinGet('fiv', 'iv', u, a))) AS fol, toUInt8(joinGet('dmu', 'part', a)) AS tpart
FROM (
  SELECT u, par_a AS a, toUInt8(1) AS kind, ph AS r_ph, t, par_ph AS x_ph, par_t AS x_t, root_ph, root_a FROM pi
  WHERE is_reply = 1 AND par_a != {NONE} AND par_a != u
  UNION ALL
  SELECT u, q_a, toUInt8(2), ph, t, q_ph, q_t, toUInt64(0), toUInt32({NONE}) FROM pi WHERE is_quote = 1 AND q_a != {NONE} AND q_a != u
) WHERE t > {t12};

-- blocks with burst sizes
CREATE TABLE blk ENGINE = MergeTree ORDER BY (u, p) AS
SELECT u, subj AS p, t, toUInt16(count() OVER (PARTITION BY u ORDER BY t RANGE BETWEEN {bw} PRECEDING AND {bw} FOLLOWING)) AS b, spart AS ppart
FROM live WHERE coll = 2 AND lop != 3 AND subj != {NONE} AND subj != u;

-- lists: curate lists (LST, curators), mod lists and list-block subscriptions (LBLK)
CREATE TABLE lists ENGINE = MergeTree ORDER BY (u, lh) AS SELECT u, lh, extra AS purpose FROM live WHERE coll = 5 AND lop != 3;
CREATE TABLE li ENGINE = MergeTree ORDER BY (u, lh) AS SELECT u, lh, subj AS member, t FROM live WHERE coll = 6 AND lop != 3 AND subj != {NONE};
CREATE TABLE cl ENGINE = MergeTree ORDER BY (u, lh) AS
SELECT l.u AS u, l.lh AS lh, count() AS nm, cityHash64(arraySort(groupArray(li.member))) AS sethash
FROM lists AS l INNER JOIN li ON li.lh = l.lh AND li.u = l.u
WHERE endsWith(l.purpose, '#curatelist') GROUP BY u, lh HAVING nm >= {cur.get('min_members', 1)} AND nm <= {cur['max_members']};
CREATE TABLE clsel ENGINE = Join(ANY, LEFT, u, lh) AS
SELECT u, lh, toUInt8(1) AS keep FROM (
  SELECT u, lh, row_number() OVER (PARTITION BY u ORDER BY cityHash64(concat(toString(lh), '{cur['seed']}'))) AS rn
  FROM (SELECT u, argMin(lh, cityHash64(concat(toString(lh), '{cur['seed']}'))) AS lh FROM cl GROUP BY u, sethash)
) WHERE rn <= {cur['max_lists_per_owner']};
CREATE TABLE cur ENGINE = MergeTree ORDER BY (u, p) AS
SELECT u, member AS p, min(t) AS t, toUInt8(joinGet('dmu', 'part', member)) AS ppart FROM li
WHERE joinGet('clsel', 'keep', u, lh) = 1 AND joinGet('dmu', 'ls', u) = 1 AND {prod('member')} AND member != u GROUP BY u, lh, member;

-- threadgate hiddenReplies / postgate detachedEmbeddingUris (labels and outcomes only, P7); extra is cut at 4,000 chars (P8)
CREATE TABLE g0 ENGINE = Memory AS SELECT u AS p, coll, gph, trunc, extra FROM live WHERE coll IN (11, 12) AND lop != 3;
CREATE TABLE gates ENGINE = Memory AS
SELECT p, toUInt8(if(coll = 11, 1, 2)) AS kind, cityHash64(c) AS c_ph, cityHash64(uri_did(c)) AS c_h,
  toUInt32(intDiv(tid_us(uri_rkey(c)), 1000000)) AS c_t, gph AS g_ph, trunc, toUInt8(0) AS allow_empty, toUInt8({i}) AS gpart
FROM (SELECT p, coll, gph, trunc, if(coll = 11, 'hiddenReplies', 'detachedEmbeddingUris') AS key,
        if(isValidJSON(extra), JSONExtract(extra, key, 'Array(String)'),
           if(position(extra, concat('"', key, '"')) = 0, [], extractAll(substring(extra, position(extra, concat('"', key, '"'))), '{URI_RE}'))) AS uris
      FROM g0) ARRAY JOIN uris AS c WHERE startsWith(c, 'at://')
UNION ALL
SELECT p, toUInt8(3), toUInt64(0), toUInt64(0), toUInt32(0), gph, trunc, toUInt8(1), toUInt8({i}) FROM g0
WHERE coll = 11 AND isValidJSON(extra) AND JSONHas(extra, 'allow') AND JSONLength(extra, 'allow') = 0;

-- per-user stats at every cutoff
CREATE TABLE ue ENGINE = MergeTree ORDER BY (u, a, d) AS
SELECT u, a, intDiv(t, 86400) AS d, count() AS n FROM (SELECT u, a, t FROM lk UNION ALL SELECT u, a, t FROM rp UNION ALL SELECT u, a, t FROM rep)
WHERE {prod('a')} GROUP BY u, a, d;
CREATE TABLE actors ENGINE = MergeTree ORDER BY u AS
SELECT u, max(src = 5) AS has_sup_src, max(from_po) AS has_po FROM (SELECT u, src, 0 AS from_po FROM ev UNION ALL SELECT u, src, 1 FROM po) GROUP BY u;
CREATE TABLE engever ENGINE = Join(ANY, LEFT, u) AS SELECT DISTINCT u, toUInt8(1) AS f FROM (SELECT u FROM lk UNION ALL SELECT u FROM rp UNION ALL SELECT u FROM pi);
CREATE TABLE b1 ENGINE = Join(ANY, LEFT, u) AS SELECT DISTINCT u, toUInt8(1) AS f FROM cl WHERE joinGet('dmu', 'ls', u) IN (2, 3);
CREATE TABLE du ENGINE = Join(ANY, LEFT, u, cidx) AS
SELECT u, cidx, toUInt32(uniqExact(a)) AS d_u FROM (
  SELECT u, a, cidx FROM ue ARRAY JOIN {CD} AS cd, range(5) AS cidx WHERE d > cd - {wd} AND d <= cd
  UNION ALL SELECT u, p, cidx FROM fol ARRAY JOIN {C} AS C, range(5) AS cidx WHERE ft <= C AND C < dt AND {prod('p')}
) GROUP BY u, cidx;
CREATE TABLE nf ENGINE = Join(ANY, LEFT, u, cidx) AS
SELECT u, cidx, toUInt32(uniqExact(p)) AS n FROM fol ARRAY JOIN {C} AS C, range(5) AS cidx WHERE ft <= C AND C < dt AND {prod('p')} GROUP BY u, cidx;
CREATE TABLE vol ENGINE = Join(ANY, LEFT, u, cidx) AS
SELECT u, cidx, toFloat32(count() / {wd}) AS v FROM (SELECT u, t FROM lk UNION ALL SELECT u, t FROM rp)
ARRAY JOIN {C} AS C, range(5) AS cidx WHERE t > C - {W} AND t <= C GROUP BY u, cidx;
CREATE TABLE el ENGINE = Join(ANY, LEFT, u, cidx) AS
SELECT u, cidx, toUInt8(1) AS f FROM ue ARRAY JOIN {CD} AS cd, range(5) AS cidx WHERE d > cd AND d <= cd + {r['label_days_A']} GROUP BY u, cidx;
{comm_sql}
CREATE TABLE ustats ENGINE = MergeTree ORDER BY (u, cidx) AS
SELECT u, cidx, joinGet('nf', 'n', u, cidx) AS n_fprod, joinGet('du', 'd_u', u, cidx) AS d_u,
  toUInt8(n_fprod < {r['light_max_producer_follows']}) AS light,
  toFloat32(pow({r['a_u_k']} / ({r['a_u_k']} + d_u), {r['a_u_alphas'][0]})) AS a05,
  toFloat32(pow({r['a_u_k']} / ({r['a_u_k']} + d_u), {r['a_u_alphas'][1]})) AS a10,
  joinGet('vol', 'v', u, cidx) AS vol,
  toUInt8(joinGet('engever', 'f', u) = 0 OR sup = 1) AS missing, joinGet('el', 'f', u, cidx) AS elig_u,
  toUInt8(has_sup_src = 1 AND has_po = 0) AS sup, joinGet('b1', 'f', u) AS b1own,
  toInt32(joinGet('ucomm', 'cc1', u) - 1) AS comm1, toInt32(joinGet('ucomm', 'cc5', u) - 1) AS comm5
FROM actors ARRAY JOIN range(5) AS cidx;

-- O3 candidates (§4.7 Task A): <=100 followed (IPW), <=30 evidenced non-followed, 20 popularity-matched random
CREATE TABLE fitu ENGINE = Join(ANY, LEFT, u) AS SELECT u, toUInt8(1) AS f FROM actors WHERE joinGet('dmu', '{fitflag}', u) = 1;
CREATE TABLE m1u ENGINE = Join(ANY, LEFT, u) AS SELECT u, toUInt8(1) AS f FROM actors WHERE joinGet('dmu', '{m1flag}', u) = 1;
CREATE TABLE cf ENGINE = MergeTree ORDER BY (u, cidx, p) AS
SELECT DISTINCT u, p, toUInt8(cidx) AS cidx FROM fol ARRAY JOIN {C} AS C, range(5) AS cidx
WHERE cidx IN {tuple(fit_cidx)} AND joinGet('fitu', 'f', u) = 1 AND ft <= C AND C < dt AND {prod('p')};
CREATE TABLE ce ENGINE = MergeTree ORDER BY (u, cidx, p) AS
SELECT DISTINCT u, a AS p, toUInt8(cidx) AS cidx FROM ue ARRAY JOIN {CD} AS cd, range(5) AS cidx
WHERE cidx IN {tuple(fit_cidx)} AND joinGet('fitu', 'f', u) = 1 AND d > cd - {wd} AND d <= cd AND (u, a, toUInt8(cidx)) NOT IN (SELECT u, p, cidx FROM cf);
CREATE TABLE cand0 ENGINE = MergeTree ORDER BY (u, cidx) AS
SELECT u, p, cidx, toUInt8(1) AS ck, toFloat32(least(1, {o3['max_followed']} / n)) AS pi FROM (
  SELECT u, p, cidx, count() OVER (PARTITION BY u, cidx) AS n, row_number() OVER (PARTITION BY u, cidx ORDER BY cityHash64(u, p, cidx)) AS rn FROM cf
) WHERE rn <= {o3['max_followed']}
UNION ALL
SELECT u, p, cidx, toUInt8(2), toFloat32(least(1, {o3['max_evidenced']} / n)) FROM (
  SELECT u, p, cidx, count() OVER (PARTITION BY u, cidx) AS n, row_number() OVER (PARTITION BY u, cidx ORDER BY cityHash64(u, p, cidx)) AS rn FROM ce
) WHERE rn <= {o3['max_evidenced']};
CREATE TABLE pdec ENGINE = Memory AS
SELECT uid AS p, toUInt8(intDiv((rn - 1) * 10, n)) AS dec FROM (
  SELECT uid, row_number() OVER (ORDER BY followers, uid) AS rn, count() OVER () AS n FROM file('{dm}', Parquet, '{S['didmap']}') WHERE is_prod = 1);
CREATE TABLE pdj ENGINE = Join(ANY, LEFT, p) AS SELECT p, dec FROM pdec;
CREATE TABLE pdr ENGINE = Join(ANY, LEFT, dec, k) AS SELECT dec, toUInt32(row_number() OVER (PARTITION BY dec ORDER BY p) - 1) AS k, p FROM pdec;
CREATE TABLE pdn ENGINE = Join(ANY, LEFT, dec) AS SELECT dec, toUInt32(count()) AS n FROM pdec GROUP BY dec;
CREATE TABLE cdec ENGINE = Join(ANY, LEFT, u, cidx) AS
SELECT u, cidx, arrayMap(x -> x.1, arraySort(x -> x.2, groupArray((joinGet('pdj', 'dec', p), cityHash64(u, p, cidx))))) AS decs FROM cand0 GROUP BY u, cidx;
CREATE TABLE cand ENGINE = MergeTree ORDER BY (u, cidx, p) AS
SELECT u, p, cidx, ck, pi FROM cand0
UNION ALL
SELECT u, p, cidx, toUInt8(3) AS ck, toFloat32(1) AS pi FROM (
  SELECT u, cidx, s, joinGet('cdec', 'decs', u, cidx) AS decs,
    toUInt8(if(length(decs) > 0, decs[(s % length(decs)) + 1], s % 10)) AS ds,
    joinGet('pdr', 'p', ds, toUInt32(cityHash64(u, cidx, s, 'rnd') % greatest(1, joinGet('pdn', 'n', ds)))) AS p
  FROM (SELECT u FROM actors WHERE joinGet('fitu', 'f', u) = 1) ARRAY JOIN {_arr(fit_cidx)} AS cidx ARRAY JOIN range({o3['n_random']}) AS s
) WHERE joinGet('pdn', 'n', ds) > 0 AND p != u AND (u, p, cidx) NOT IN (SELECT u, p, cidx FROM cand0)
UNION ALL
SELECT u, p, toUInt8(2) AS cidx, toUInt8(4) AS ck, toFloat32(1) AS pi FROM (
  SELECT u, p, max(lt) AS last, row_number() OVER (PARTITION BY u ORDER BY max(lt) DESC, cityHash64(u, p)) AS rn FROM (
    SELECT u, p, ft AS lt FROM fol WHERE joinGet('m1u', 'f', u) = 1 AND ft <= {T} AND {T} < dt AND {prod('p')}
    UNION ALL SELECT u, a, toUInt32(d * 86400) FROM ue WHERE joinGet('m1u', 'f', u) = 1 AND d > {T // 86400} - {wd} AND d <= {T // 86400}
  ) GROUP BY u, p
) WHERE rn <= {o3['m1_max_pairs']};

-- MAINT input: the actor's likes to accounts it follows, per cutoff window
CREATE TABLE lkf ENGINE = MergeTree ORDER BY (u, a) AS
SELECT u, a, toUInt8(cidx) AS cidx, toUInt32(count()) AS n FROM (SELECT u, a, t FROM lk WHERE t > {tfit_lo})
ARRAY JOIN {C} AS C, range(5) AS cidx
WHERE t > C - {W} AND t <= C AND arrayExists(x -> x.1 <= C AND C < x.2, joinGet('fiv', 'iv', u, a)) GROUP BY u, a, cidx;

-- ===== outputs re-keyed by the target's partition (phase 2 inputs) =====
{out(o + '/eng/j{_partition_id}.parquet')} PARTITION BY apart
SELECT u, a, ph, t, s_t, typ, fol, m, apart FROM (
  SELECT u, a, ph, t, s_t, toUInt8(1) AS typ, fol, m, apart FROM lk WHERE t > {t12}
  UNION ALL SELECT u, a, ph, t, s_t, toUInt8(2), fol, m, apart FROM rp WHERE t > {t12});
{out(o + '/eng_own.parquet')} SELECT u, ph, t FROM lk WHERE t > {t12} AND t >= s_t AND t - s_t <= {cr_w};
{out(o + '/rep/j{_partition_id}.parquet')} PARTITION BY tpart SELECT * FROM rep;
{out(o + '/fol/j{_partition_id}.parquet')} PARTITION BY ppart SELECT * FROM fol;
{out(o + '/fol_own.parquet')} SELECT * FROM fol;
{out(o + '/blk/j{_partition_id}.parquet')} PARTITION BY ppart SELECT * FROM blk;
{out(o + '/blk_own.parquet')} SELECT * FROM blk;
{out(o + '/cur/j{_partition_id}.parquet')} PARTITION BY ppart SELECT * FROM cur;
{out(o + '/cand/j{_partition_id}.parquet')} PARTITION BY ppart SELECT u, p, cidx, ck, pi, toUInt8(joinGet('dmu', 'part', p)) AS ppart FROM cand;
{out(o + '/pi.parquet')} SELECT * FROM pi;
{out(o + '/lk_own_fol.parquet')} SELECT * FROM lkf;
{out(o + '/gates.parquet')} SELECT * FROM gates;
{out(o + '/modli.parquet')} SELECT li.lh AS lh, li.u AS owner, member, t FROM li INNER JOIN lists AS l ON l.lh = li.lh AND l.u = li.u WHERE endsWith(l.purpose, '#modlist');
{out(o + '/lbsub.parquet')} SELECT u, lh, t FROM live WHERE coll = 7 AND lop != 3;
{out(o + '/ustats.parquet')} SELECT * FROM ustats;
{out(o + '/o7p.parquet')} SELECT feed, toUInt8(1) AS map, toInt32(joinGet('ucomm', 'cc1', u) - 1) AS comm, uniqExact(u) AS n FROM live
  WHERE coll = 3 AND lop != 3 AND scoll = 2 AND feed != '' AND {1 if maps else 0} GROUP BY feed, comm
  UNION ALL SELECT feed, toUInt8(5), toInt32(joinGet('ucomm', 'cc5', u) - 1), uniqExact(u) FROM live
  WHERE coll = 3 AND lop != 3 AND scoll = 2 AND feed != '' AND {1 if maps else 0} GROUP BY feed, 3;

-- ===== final outputs written by phase 1 =====
{out(final['o5raw'] + f'/blocks/i{i:02d}.parquet')} SELECT u, p, t, b, toUInt8(b >= {thr}) AS bulk FROM blk;
{out(final['o5raw'] + f'/follows/i{i:02d}.parquet')} SELECT u, p, ft, dt, b FROM fol WHERE b < {thr} AND ft > {o5a} AND ft <= {o5b};
{out(final['o3b'] + f'/labels_i{i:02d}.parquet')}
SELECT u, p, kind, t FROM (
  SELECT u, p, toUInt8(1) AS kind, ft AS t FROM fol WHERE {prod('p')}
  UNION ALL SELECT u, a, toUInt8(2), t FROM lk WHERE {prod('a')}
  UNION ALL SELECT u, a, toUInt8(3), t FROM rp WHERE {prod('a')}
  UNION ALL SELECT u, a, if(kind = 1, toUInt8(4), toUInt8(5)), t FROM rep WHERE {prod('a')}
) WHERE joinGet('m1u', 'f', u) = 1 AND t > {T} AND t <= {m1end};
CREATE TABLE first ENGINE = Join(ANY, LEFT, u) AS
SELECT u, min(t) AS first_t FROM (SELECT u, t FROM ev WHERE kind = 1 AND tok = 1 UNION ALL SELECT u, t FROM po WHERE tok = 1) GROUP BY u;
CREATE TABLE coh ENGINE = Join(ANY, LEFT, u) AS
SELECT u, c0 FROM (SELECT u, if(joinGet('dmu', 'plc_t', u) > 0, joinGet('dmu', 'plc_t', u),
                               if(joinGet('dmu', 'web', u) = 1, joinGet('first', 'first_t', u), 0)) AS c0 FROM actors)
WHERE c0 >= {m2a} AND c0 <= {m2b};
CREATE TABLE eng30 ENGINE = Join(ANY, LEFT, u, a) AS
SELECT u, a, min(t) AS t0 FROM (SELECT u, a, t FROM lk UNION ALL SELECT u, a, t FROM rp UNION ALL SELECT u, a, t FROM rep)
WHERE joinGet('coh', 'c0', u) > 0 GROUP BY u, a;
{out(final['o3c'] + f'/i{i:02d}.parquet')}
SELECT u, p, kind, ft, toUInt8(e0 > 0 AND e0 <= c0 + {m2o}) AS engaged30 FROM (
  SELECT u, p, ft, joinGet('coh', 'c0', u) AS c0, joinGet('eng30', 't0', u, p) AS e0,
    multiIf(ft >= c0 AND ft < c0 + 86400, 1, ft >= c0 + 86400 AND ft < c0 + {m2o} AND dt = {NONE}, 2, 0) AS kind
  FROM fol WHERE joinGet('coh', 'c0', u) > 0) WHERE kind > 0;

-- positive control (check c): Oct 20 likes of posts, snapshot time from backfill rows (rules.backfill_filter)
CREATE TABLE snap ENGINE = Join(ANY, LEFT, u) AS
SELECT u, minIf(ing, {bf}) AS s_bf, countIf({bf}) AS n_bf, max(ing) AS s_old, min(ing) AS s_fb FROM (
  SELECT u, ing, src, kind FROM ev WHERE ing > 0 UNION ALL SELECT u, ing, src, toUInt8(1) FROM po WHERE ing > 0) GROUP BY u;
CREATE TABLE pcl ENGINE = MergeTree ORDER BY u AS
SELECT u, toUInt32(uniqExact(rk)) AS n,
  toUInt8(if(joinGet('snap', 'n_bf', u) > 0, joinGet('snap', 's_bf', u), joinGet('snap', 's_fb', u)) >= {snapmin}) AS e_new,
  toUInt8(joinGet('snap', 's_old', u) >= {snapmin}) AS e_old, toUInt8(joinGet('snap', 's_fb', u) >= {snapmin}) AS e_fb,
  toUInt8(joinGet('snap', 'n_bf', u) = 0) AS used_fb
FROM live WHERE coll = 3 AND lop != 3 AND scoll = 1 AND tok = 1 AND t >= {pc0} AND t < {pc0 + 86400} GROUP BY u;
{out(final['pc'] + f'/i{i:02d}.parquet')}
SELECT joinGet('dmu', 'h', u) AS h, toUInt16(h % 1000) AS bucket, n, e_new, e_old, e_fb, used_fb FROM pcl WHERE joinGet('dmu', 'pc', u) = 1;

-- ===== O1 fragments =====
CREATE TABLE o1 (metric String, k1 String, k2 String, k3 String, v Float64) ENGINE = Memory;
INSERT INTO o1 SELECT 'rows_by_coll_op_month', toString(coll), toString(op), toString(toYYYYMM(toDateTime(t))), count() FROM ev WHERE kind = 1 GROUP BY coll, op, 4;
INSERT INTO o1 SELECT 'rows_by_coll_op_month', '19', toString(op), toString(toYYYYMM(toDateTime(t))), count() FROM po GROUP BY op, 4;
INSERT INTO o1 SELECT 'rows_by_kind', toString(kind), toString(src), '', count() FROM ev GROUP BY kind, src;
INSERT INTO o1 SELECT m.1, toString(coll), '', '', m.2 FROM (
  SELECT coll, [('dup_rows', toFloat64(sum(nrows))), ('dup_keys', toFloat64(count())), ('dup_key_revs', toFloat64(sum(nrevs))),
                ('deleted_keys', toFloat64(countIf(lop = 3)))] AS arr FROM st GROUP BY coll) ARRAY JOIN arr AS m;
INSERT INTO o1 SELECT 'bad_tid', toString(coll), '', '', countIf(tok = 0) FROM ev WHERE kind = 1 GROUP BY coll;
INSERT INTO o1 SELECT 'follow_created_minus_tid', toString(multiIf(dd <= 1, 1, dd <= 10, 10, dd <= 60, 60, dd <= 3600, 3600, dd <= 86400, 86400, dd <= 2592000, 2592000, 999999999)),
  toString(sign(sd)), '', count() FROM (SELECT toInt64(created) - toInt64(t) AS sd, abs(sd) AS dd FROM live WHERE coll = 1 AND created > 0 AND tok = 1) GROUP BY 2, 3;
INSERT INTO o1 SELECT 'backfill_probe', toString(src), toString(kind), '', count() FROM ev GROUP BY src, kind;
INSERT INTO o1 SELECT 'backfill_probe_modal_rev', toString(src), toString(kind), '', countIf(rev = mr) FROM ev
  INNER JOIN (SELECT u, topK(1)(rev)[1] AS mr FROM ev WHERE rev > 0 GROUP BY u) AS m USING (u) GROUP BY src, kind;
INSERT INTO o1 SELECT 'backfill_probe_ing_min', toString(src), toString(kind), '', min(ing) FROM ev WHERE ing > 0 GROUP BY src, kind;
INSERT INTO o1 SELECT 'backfill_probe_ing_max', toString(src), toString(kind), '', max(ing) FROM ev WHERE ing > 0 GROUP BY src, kind;
INSERT INTO o1 SELECT 'pc_likers', toString(e_old), toString(e_new), toString(e_fb), count() FROM pcl GROUP BY e_old, e_new, e_fb;
INSERT INTO o1 SELECT 'pc_likes', toString(e_old), toString(e_new), toString(e_fb), sum(n) FROM pcl GROUP BY e_old, e_new, e_fb;
INSERT INTO o1 SELECT 'pc_used_fallback', '', '', '', countIf(used_fb = 1) FROM pcl;
INSERT INTO o1 SELECT 'plc_vs_first_record', multiIf(plc = 0 AND web = 1, 'web', plc = 0, 'no_plc', fr = 0, 'no_record',
    fr + 86400 < plc, 'first_before_plc', fr > plc + 86400, 'first_after_plc', 'agree'),
  if(plc = 0 OR fr = 0, '', toString(multiIf(dd < -30, -30, dd < -7, -7, dd < -1, -1, dd <= 1, 0, dd <= 7, 7, dd <= 30, 30, dd <= 90, 90, dd <= 365, 365, 9999))), '', count()
FROM (SELECT u, joinGet('dmu', 'plc_t', u) AS plc, joinGet('dmu', 'web', u) AS web, joinGet('first', 'first_t', u) AS fr,
        (toInt64(fr) - toInt64(plc)) / 86400 AS dd FROM actors) GROUP BY 2, 3;
INSERT INTO o1 SELECT 'm2_cohort_users', '', '', '', count() FROM (SELECT DISTINCT u FROM actors WHERE joinGet('coh', 'c0', u) > 0);
INSERT INTO o1 SELECT 'gates', toString(kind), toString(trunc), '', count() FROM gates GROUP BY kind, trunc;
INSERT INTO o1 SELECT 'gate_records_truncated', toString(coll), '', '', countIf(trunc = 1) FROM g0 GROUP BY coll;
INSERT INTO o1 SELECT 'actors', '', '', '', count() FROM actors;
INSERT INTO o1 SELECT 'supplement_actors', '', '', '', countIf(has_sup_src = 1 AND has_po = 0) FROM actors;
INSERT INTO o1 SELECT 'likes_dedup', '', '', '', count() FROM lk;
INSERT INTO o1 SELECT 'reposts_dedup', '', '', '', count() FROM rp;
INSERT INTO o1 SELECT 'cand_rows', toString(cidx), toString(ck), '', count() FROM cand GROUP BY cidx, ck;
INSERT INTO o1 SELECT 'cand_users', toString(cidx), toString(ck), '', uniqExact(u) FROM cand GROUP BY cidx, ck;
INSERT INTO o1 SELECT 'posts_live_12mo', toString(is_reply), toString(is_quote), '', count() FROM pi WHERE t > {t12} AND t <= {tn} GROUP BY is_reply, is_quote;
{out(o + '/o1.parquet')} SELECT * FROM o1;
"""
