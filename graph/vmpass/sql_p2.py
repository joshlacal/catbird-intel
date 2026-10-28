"""Phase 2, author-partitioned (`cityHash64(author) % 16`), plan §3.2 and §4.3-§4.7.

For the producers/authors in partition j this joins the inbound rows phase 1
re-keyed here with the post index, types every interaction (LO/LR/ACK/LU/RP,
CR/URf/URn/RH, Qf/Qn/QH, LST, follows), and computes, per cutoff:
  - pair aggregates (two levels: per (u,p,type,day), then per (u,p)), in user
    sub-chunks so the hash tables stay bounded;
  - O2 producer sums S_k(p) = sum_u a_u x_k(u,p) over the light, non-MAINT,
    evidenced audience, at both alphas, plus the base aggregates;
  - O3a/O3b pair-feature rows for the sampled users, with labels;
  - O4's E0 edges, O6, O8, O9, O11, the O12 candidates and the directed MAINT
    rows for the pair shuffle.
"""
from __future__ import annotations

from rules import A_TYPES, day_start
from sqlc import NONE, PRELUDE, S, TY, out, settings_sql, src

TYPES2 = list(A_TYPES) + ["ACK", "RH", "QH"]
LIKES = ("LO", "LR", "LU")
DL_TYPES = ("LO", "LR", "LU", "RP")  # S1 §3 C: discovery likes and reposts
Q_WITH = ("LO", "LR", "LU", "RP", "CR", "Qf", "Qn")
Q_NO = ("LO", "LR", "LU", "RP", "CR", "Qf")
APLUS = list(A_TYPES)
F_ROW, RF_ROW = 20, 21
REPLY_RAW, QUOTE_RAW = 100, 101


def _in(types) -> str:
    return "(" + ", ".join(str(TY[t]) for t in types) + ")"


def half_lives(rules) -> list[int]:
    return sorted({int(f["h"]) for f in rules.features if f.get("stat") in ("dec", "L")})


def lvl1(rules, C: int, where: str) -> str:
    """Per (u, p, resolved type, day) partial aggregates at cutoff C."""
    W = rules.W
    hl = half_lives(rules)
    typ = (f"multiIf(ty = {REPLY_RAW}, multiIf(hb_t > 0 AND hb_t <= {C}, {TY['RH']}, cr_t > 0 AND cr_t <= {C}, {TY['CR']}, "
           f"fol = 1, {TY['URf']}, {TY['URn']}), ty = {QUOTE_RAW}, multiIf(hb_t > 0 AND hb_t <= {C}, {TY['QH']}, fol = 1, {TY['Qf']}, {TY['Qn']}), ty)")
    window = (f"((ty IN ({F_ROW}, {RF_ROW}) AND t <= {C} AND {C} < dt) OR (ty = {TY['LST']} AND t <= {C}) "
              f"OR (ty NOT IN ({F_ROW}, {RF_ROW}, {TY['LST']}) AND t > {C - W} AND t <= {C}))")
    decs = ", ".join(f"sum(m * exp2(-(toInt64({C}) - toInt64(t)) / {h * 86400})) AS s{h}, "
                     f"sumIf(m * exp2(-(toInt64({C}) - toInt64(t)) / {h * 86400}), fol = 0) AS sdl{h}" for h in hl)
    return f"""SELECT u, p, toUInt8(tyc) AS tyc, toUInt32(if(tyc IN ({F_ROW}, {RF_ROW}, {TY['LST']}), 0, intDiv(t, 86400))) AS d,
  toUInt32(count()) AS n, {decs}, max(t) AS tmax, min(t) AS tmin, toUInt32(countIf(t > {C - 7 * 86400})) AS n7,
  toUInt32(countIf(lat < {rules['early_s']})) AS ne, toUInt32(countIf(lat >= {rules['late_s']})) AS nl,
  toUInt32(countIf(fol = 0)) AS ndl, max(m) AS mx, max(lat) AS mxlat
FROM (SELECT u, p, {typ} AS tyc, t, m, lat, fol FROM evx WHERE {window} AND ({where}))
GROUP BY u, p, tyc, d"""


def lvl2_cols(rules) -> list[str]:
    hl = half_lives(rules)
    c = []
    for x in TYPES2:
        k = TY[x]
        c += [f"sumIf(n, tyc = {k}) AS n_{x}", f"countIf(tyc = {k}) AS nz_{x}", f"maxIf(tmax, tyc = {k}) AS tmax_{x}",
              f"minIf(tmin, tyc = {k}) AS tmin_{x}", f"sumIf(n7, tyc = {k}) AS n7_{x}", f"sumIf(ne, tyc = {k}) AS ne_{x}",
              f"sumIf(nl, tyc = {k}) AS nl_{x}", f"sumIf(ndl, tyc = {k}) AS ndl_{x}"]
        c += [f"sumIf(s{h}, tyc = {k}) AS dec_{x}_{h}" for h in hl]
    c += [f"countIf(tyc = {F_ROW}) AS f_n", f"minIf(tmin, tyc = {F_ROW}) AS f_tmin", f"maxIf(mx, tyc = {F_ROW}) AS f_mx",
          f"countIf(tyc = {RF_ROW}) AS rv_n", f"sumIf(n, tyc = {TY['LST']}) AS lst_n",
          f"sumIf(ndl, tyc IN {_in(LIKES)}) AS dl_n", f"minIf(d, tyc IN {_in(LIKES)} AND ndl > 0) AS dl_dmin",
          f"maxIf(d, tyc IN {_in(LIKES)} AND ndl > 0) AS dl_dmax",
          f"sumIf(n, tyc IN {_in(Q_WITH)}) AS qa_n", f"minIf(d, tyc IN {_in(Q_WITH)}) AS qa_dmin", f"maxIf(d, tyc IN {_in(Q_WITH)}) AS qa_dmax",
          f"sumIf(n, tyc IN {_in(Q_NO)}) AS qb_n", f"minIf(d, tyc IN {_in(Q_NO)}) AS qb_dmin", f"maxIf(d, tyc IN {_in(Q_NO)}) AS qb_dmax",
          f"sumIf(n, tyc IN {_in(('LO', 'LR', 'ACK', 'LU'))}) AS nlk", f"sumIf(n, tyc IN {_in(APLUS)}) AS nap",
          f"maxIf(mxlat, tyc = {F_ROW}) AS f_b",
          f"uniqExactIf(d, ndl > 0 AND tyc IN {_in(DL_TYPES)}) AS dl_nd"]
    c += [f"sumIf(sdl{h}, tyc IN {_in(DL_TYPES)}) AS dl_s{h}" for h in hl]
    return c


def derived(rules, C: int, cidx: int) -> list[str]:
    """Per-pair derived columns: user stats, producer stats, MAINT, mutual."""
    m = rules["maint"]
    return [
        f"joinGet('usj', 'light', u, toUInt8({cidx})) AS light", f"joinGet('usj', 'a05', u, toUInt8({cidx})) AS a05",
        f"joinGet('usj', 'a10', u, toUInt8({cidx})) AS a10", f"joinGet('usj', 'd_u', u, toUInt8({cidx})) AS d_u",
        f"joinGet('usj', 'vol_pct', u, toUInt8({cidx})) AS vol_pct", f"joinGet('usj', 'missing', u, toUInt8({cidx})) AS missing",
        f"joinGet('usj', 'b1own', u, toUInt8({cidx})) AS b1own",
        f"joinGet('prodc', 'followers', p, toUInt8({cidx})) AS pc_followers", f"joinGet('prodc', 'orig_w', p, toUInt8({cidx})) AS pc_orig",
        f"joinGet('prodc', 'posts_w', p, toUInt8({cidx})) AS pc_posts", f"joinGet('prodc', 'curators', p, toUInt8({cidx})) AS pc_cur",
        f"joinGet('prodc', 'blk_1k', p, toUInt8({cidx})) AS pc_blk1k", f"joinGet('prodc', 'blockers', p, toUInt8({cidx})) AS pc_blockers",
        "joinGet('supj', 'sup', u) AS sup_u",
        f"toUInt8(arrayExists(x -> x <= {C}, joinGet('bpj', 'ts', u, p))) AS blk_pair",
        f"toUInt8(hasAny(arrayMap(x -> x.1, arrayFilter(x -> x.2 <= {C}, joinGet('plists', 'ls', p))), "
        f"arrayMap(x -> x.1, arrayFilter(x -> x.2 <= {C}, joinGet('ulists', 'ls', u))))) AS lblk_pair",
        "toUInt8(f_n > 0) AS fol", "toUInt8(f_n > 0 AND rv_n > 0) AS mutual",
        f"joinGet('lkfj', 'n', u, p, toUInt8({cidx})) AS n_pu",
        f"toUInt8(mutual = 1 AND nlk >= {m['n_min']} AND n_pu >= {m['n_min']} AND least(nlk, n_pu) / greatest(nlk, n_pu) >= {m['recip_min']} "
        f"AND nlk / greatest(1, pc_orig) >= {m['cov_min']} AND pc_orig >= {m['orig_min']}) AS maint",
        "toUInt8(light = 1 AND maint = 0 AND (f_n > 0 OR nap > 0 OR lst_n > 0) AND u != p) AS aud",
    ]


K_MIN = [50]  # set from rules in p2()


def fexpr(f: dict, C: int) -> str:
    """SQL for one registered feature, from the level-2 columns and derived()."""
    if "type" in f and f["type"] == "DL":
        return {"L": f"log1p(dl_s{f.get('h')})", "ND": "log1p(dl_nd)"}[f["stat"]]
    if "type" in f and f["stat"] in ("L", "ND", "DP", "DSL", "DSF", "C7", "ES", "LS"):
        x, s = f["type"], f["stat"]
        n = f"n_{x}"
        return {
            "L": f"log1p(dec_{x}_{f.get('h')})", "ND": f"log1p(nz_{x})", "DP": f"log1p({n})",
            "DSL": f"if({n} = 0, 90, least(90, ({C} - tmax_{x}) / 86400))",
            "DSF": f"if({n} = 0, 90, least(90, ({C} - tmin_{x}) / 86400))",
            "C7": f"log1p(n7_{x})", "ES": f"if({n} = 0, 0, ne_{x} / {n})", "LS": f"if({n} = 0, 0, nl_{x} / {n})",
        }[s]
    if "type" in f:
        x, s = f["type"], f["stat"]
        n = f"n_{x}"
        return {
            "dec": f"log1p(dec_{x}_{f.get('h')})",
            "nzdays": f"nz_{x}",
            "nposts": n,
            "dsl": f"if({n} = 0, 91, ({C} - tmax_{x}) / 86400)",
            "dsf": f"if({n} = 0, 91, ({C} - tmin_{x}) / 86400)",
            "n7d": f"n7_{x}",
            "early": f"if({n} = 0, 0, ne_{x} / {n})",
            "late": f"if({n} = 0, 0, nl_{x} / {n})",
            "dl": f"if({n} = 0, 0, ndl_{x} / {n})",
        }[s]
    name = f["name"]
    return {
        "fol": "fol", "fol_age": f"if(fol = 1, ({C} - f_tmin) / 86400, 0)", "fol_mbulk": "if(fol = 1, f_mx, 0)",
        "mutual": "mutual", "ack_share": "n_ACK / greatest(1, nlk)",
        "reply_nolike_share": "(n_CR + n_URf + n_URn + n_RH) / greatest(1, n_CR + n_URf + n_URn + n_RH + nlk)",
        "maint": "maint", "lst": "log1p(lst_n)",
        "d_u_log": "log1p(d_u)", "vol_pct": "vol_pct", "light": "light", "missing": "missing",
        "followers_log": "log1p(pc_followers)", "originals_log": "log1p(pc_orig)",
        "originals_share": "pc_orig / greatest(1, pc_posts)", "curators_log": "log1p(pc_cur)", "blockers_1k": "pc_blk1k",
        # S1 §3 D-H
        "LST_N": "log1p(lst_n)", "BLK": "blk_pair", "LBLK": "lblk_pair",
        "FOL": "fol", "FAGE": f"if(fol = 1, log1p(({C} - f_tmin) / 86400), 0)", "FBURST": "if(fol = 1, log(greatest(1, f_b)), 0)",
        "MUT": "mutual", "ACKS": "n_ACK / greatest(1, nlk)",
        "RNL": "(n_URf + n_URn + n_RH) / greatest(1, n_CR + n_URf + n_URn + n_RH)", "MAINT": "maint",
        "DU": "log1p(d_u)", "DVP": "vol_pct", "HEAVY": "1 - light", "MISS": "sup_u",
        "PFOL": "log1p(pc_followers)", "PORIG": "log1p(pc_orig)", "POSH": "pc_orig / greatest(1, pc_posts)",
        "PCUR": "log1p(pc_cur)", "PBLK": "pc_blk1k", "PBLKN": f"toUInt8(pc_blockers < {K_MIN[0]})",
    }[name]


def p2(rules, j: int, P1: str, G: str, buckets: str, o: str, final: dict, opts: dict) -> str:
    r = rules
    N, W = r.N, r.W
    cut = r.cut
    names = r.cut_names
    K = int(opts.get("subchunks", r["p2_user_subchunks"]))
    samp = opts.get("inbound_sample")  # None, 's200' or 's400' (S8 only)
    ins = (lambda col: f"joinGet('dmu', '{samp}', {col}) = 1") if samp else (lambda col: "1")
    thr, div = r["burst_threshold"], r["bulk_div"]
    crw, hw = r["cr_window_s"], r["hostile_window_s"]
    k_min = r["k_min"]
    K_MIN[0] = k_min
    o13flag = opts.get("o13_flag", "o13m1")
    o13lo, o13hi = day_start(r["o13"]["window"][0]) + 86400, day_start(r["o13"]["window"][1]) + 86400
    t12 = day_start(r["twelve_month_start"])
    tn = cut["T_now"]
    LA = r["label_days_A"] * 86400
    LB0, LB1 = r["label_days_B"][0] * 86400, r["label_days_B"][1] * 86400
    LC = r["label_days_C"] * 86400
    fit_cidx = [0, 1] + ([] if r["o3"]["drop_t_test"] else [3])
    feats = r.features
    fn = [f["name"] for f in feats]
    cuts_arr = "[" + ", ".join(str(cut[k]) for k in names) + "]"
    sc = r["o3b_default_scales"]
    scales = {**r["o3b_scales"], **(opts.get("o3b_scales") or {})}

    def scale(f):
        if f["name"] in scales:
            return float(scales[f["name"]])
        return float(sc[f["stat"]] if "stat" in f else sc[f["level"]])

    def i_glob(name):
        return f"{P1}/i*/{name}/j{j}.parquet"

    # ---------- shared tables ----------
    sql = PRELUDE + settings_sql(r) + f"""
CREATE TABLE dmu ENGINE = Join(ANY, LEFT, uid) AS
SELECT uid, part, is_prod, hide, synth_no, fit, m1, h, ls, s200, s400, ho FROM file('{G}/didmap.parquet', Parquet, '{S['didmap']}');
CREATE TABLE dmd ENGINE = Join(ANY, LEFT, uid) AS SELECT uid, did FROM file('{G}/didmap.parquet', Parquet, '{S['didmap']}') WHERE part = {j};
CREATE TABLE usj ENGINE = Join(ANY, LEFT, u, cidx) AS
SELECT u, cidx, light, a05, a10, d_u, vol_pct, missing, elig_u, b1own FROM file('{G}/ustats.parquet', Parquet, '{S['ustats_m']}');
CREATE TABLE supj ENGINE = Join(ANY, LEFT, u) AS SELECT u, sup FROM file('{G}/ustats.parquet', Parquet, '{S['ustats_m']}') WHERE cidx = 0;
CREATE TABLE pi ENGINE = MergeTree ORDER BY ph AS SELECT * FROM {src(f'{P1}/i{j:02d}/pi.parquet', 'pi')};
CREATE TABLE pij ENGINE = Join(ANY, LEFT, ph) AS SELECT ph, toUInt8(1) AS one, t AS tp, is_reply, par_a, root_a FROM pi;

-- typed inbound likes and reposts
CREATE TABLE te ENGINE = MergeTree ORDER BY (u, a) AS
SELECT u, a, ph, t, s_t, fol, m, ty, isr, typ, toInt64(t) - toInt64(if(has = 1, tp, s_t)) AS lat FROM (
  SELECT e.*, joinGet('pij', 'one', ph) AS has, joinGet('pij', 'tp', ph) AS tp, joinGet('pij', 'is_reply', ph) AS isr,
    joinGet('pij', 'par_a', ph) AS pa, joinGet('pij', 'root_a', ph) AS ra, joinGet('supj', 'sup', a) AS asup,
    toUInt8(multiIf(has = 0 AND asup = 1, {TY['LU']}, has = 0, 0, typ = 2, {TY['RP']}, isr = 0, {TY['LO']}, u = pa OR u = ra, {TY['ACK']}, {TY['LR']})) AS ty
  FROM {src(i_glob('eng'), 'eng')} AS e WHERE {ins('u')}
);
-- inbound replies and quotes; confirmation (CR), hostility (block within 48 h), gates
CREATE TABLE r0 ENGINE = MergeTree ORDER BY (u, a) AS SELECT * FROM {src(i_glob('rep'), 'rep')} WHERE {ins('u')};
CREATE TABLE crj ENGINE = Join(ANY, LEFT, r_ph, liker) AS
SELECT r_ph, liker, groupArray(ct) AS cts FROM (
  SELECT ph AS r_ph, u AS liker, t AS ct FROM {src(f'{P1}/i{j:02d}/eng_own.parquet', 'eng_own')} WHERE ph IN (SELECT r_ph FROM r0 WHERE kind = 1)
  UNION ALL SELECT par_ph, u, t FROM pi WHERE is_reply = 1 AND par_ph IN (SELECT r_ph FROM r0 WHERE kind = 1)
) GROUP BY r_ph, liker;
CREATE TABLE bp ENGINE = MergeTree ORDER BY (u, p) AS
SELECT u, p, t, b FROM {src(i_glob('blk'), 'blk')} WHERE {ins('u')}
UNION ALL SELECT bo.p AS u, bo.u AS p, bo.t AS t, bo.b AS b FROM {src(f'{P1}/i{j:02d}/blk_own.parquet', 'blk')} AS bo;
CREATE TABLE bpj ENGINE = Join(ANY, LEFT, u, p) AS SELECT u, p, groupArray(t) AS ts FROM bp GROUP BY u, p;
CREATE TABLE gt ENGINE = MergeTree ORDER BY (p, c_ph) AS SELECT * FROM {src(f'{G}/gates/j{j}.parquet', 'gates')};
CREATE TABLE hidj ENGINE = Join(ANY, LEFT, p, c_ph) AS SELECT p, c_ph, toUInt8(max(kind = 1)) AS hid, toUInt8(max(kind = 2)) AS det FROM gt WHERE kind IN (1, 2) GROUP BY p, c_ph;
CREATE TABLE aej ENGINE = Join(ANY, LEFT, p, g_ph) AS SELECT p, g_ph, toUInt8(1) AS ae FROM gt WHERE kind = 3 GROUP BY p, g_ph;
CREATE TABLE tr ENGINE = MergeTree ORDER BY (u, a) AS
SELECT u, a, kind, r_ph, t, x_ph, x_t, root_ph, root_a, fol,
  toUInt32(arrayMin(arrayFilter(c -> c >= t AND c <= t + {crw}, joinGet('crj', 'cts', r_ph, a)))) AS cr_t0,
  toUInt32(if(kind = 1, cr_t0, 0)) AS cr_t,
  toUInt32(arrayMin(arrayFilter(c -> c > t AND c <= t + {hw}, joinGet('bpj', 'ts', u, a)))) AS hb_t,
  joinGet('hidj', 'hid', a, r_ph) AS hid, joinGet('hidj', 'det', a, r_ph) AS det,
  toUInt8(kind = 1 AND root_a = a AND joinGet('aej', 'ae', a, root_ph) = 1) AS gated,
  joinGet('pij', 'is_reply', x_ph) AS x_isreply, joinGet('pij', 'one', x_ph) AS x_has
FROM r0;
-- evidence rows for the pair aggregates
CREATE TABLE evx ENGINE = MergeTree ORDER BY (u, p) AS
SELECT * FROM (
SELECT u, a AS p, ty, t, m, lat, fol, toUInt32(0) AS cr_t, toUInt32(0) AS hb_t, toUInt32({NONE}) AS dt FROM te WHERE ty > 0
UNION ALL SELECT u, a, toUInt8(if(kind = 1, {REPLY_RAW}, {QUOTE_RAW})), t, toFloat32(1), toInt64(t) - toInt64(x_t), fol, cr_t, hb_t, toUInt32({NONE}) FROM tr WHERE gated = 0
UNION ALL SELECT u, p, toUInt8({TY['LST']}), t, toFloat32(1), toInt64(0), toUInt8(1), toUInt32(0), toUInt32(0), toUInt32({NONE}) FROM {src(i_glob('cur'), 'cur')} WHERE {ins('u')}
UNION ALL SELECT u, p, toUInt8({F_ROW}), ft, toFloat32(if(b >= {thr}, 1 / (1 + (b - 1) / {div}), 1)), toInt64(b), toUInt8(1), toUInt32(0), toUInt32(0), dt
  FROM {src(i_glob('fol'), 'fol')} WHERE {ins('u')}
UNION ALL SELECT fo.p AS u, fo.u AS p, toUInt8({RF_ROW}), fo.ft, toFloat32(1), toInt64(0), toUInt8(1), toUInt32(0), toUInt32(0), fo.dt FROM {src(f'{P1}/i{j:02d}/fol_own.parquet', 'fol')} AS fo
) WHERE joinGet('dmu', 'is_prod', p) = 1 AND u != p;
CREATE TABLE ulists ENGINE = Join(ANY, LEFT, u) AS SELECT u, groupArray((lh, t)) AS ls FROM {src(f'{P1}/i*/lbsub.parquet', 'lbsub')} GROUP BY u;
CREATE TABLE plists ENGINE = Join(ANY, LEFT, p) AS
SELECT member AS p, groupArray((lh, t)) AS ls FROM {src(f'{G}/modli_by_part/j{j}.parquet', 'lh UInt64, member UInt32, t UInt32, mpart UInt8')} GROUP BY member;
CREATE TABLE lkfj ENGINE = Join(ANY, LEFT, a, u, cidx) AS SELECT a, u, cidx, n FROM {src(f'{P1}/i{j:02d}/lk_own_fol.parquet', 'lk_own_fol')};

-- producer stats per cutoff (window (C-90d, C])
CREATE TABLE lb ENGINE = Join(ANY, LEFT, p, cidx) AS SELECT p, cidx, n_sub FROM {src(G + '/lblk.parquet', 'p UInt32, cidx UInt8, n_sub UInt64')};
CREATE TABLE pcs ENGINE = MergeTree ORDER BY (p, cidx) AS
SELECT p, toUInt8(cidx) AS cidx, sum(fl) AS followers, sum(ow) AS orig_w, sum(pw) AS posts_w, sum(cu) AS curators, sum(bl) AS blockers, sum(bw) AS blockers_w,
  sum(o30) AS orig_30, sum(o12) AS orig_12, sum(r30) AS repl_30, sum(rw) AS repl_w, sum(r12) AS repl_12, sum(q30) AS quote_30, sum(qw) AS quote_w,
  sum(q12) AS quote_12, sum(lt) AS lifetime, max(lastp) AS last_post, sum(iw) AS img_w, sum(olab) AS orig_label
FROM (
  SELECT p, cidx, uniqExact(u) AS fl, 0 AS ow, 0 AS pw, 0 AS cu, 0 AS bl, 0 AS bw, 0 AS o30, 0 AS o12, 0 AS r30, 0 AS rw, 0 AS r12, 0 AS q30,
    0 AS qw, 0 AS q12, 0 AS lt, toUInt32(0) AS lastp, 0 AS iw, 0 AS olab
  FROM {src(i_glob('fol'), 'fol')} ARRAY JOIN {cuts_arr} AS C, range(5) AS cidx WHERE {ins('u')} AND ft <= C AND C < dt GROUP BY p, cidx
  UNION ALL
  SELECT u AS p, cidx, 0, countIf(is_reply = 0 AND t > C - {W} AND t <= C), countIf(t > C - {W} AND t <= C), 0, 0, 0,
    countIf(is_reply = 0 AND t > C - 2592000 AND t <= C), countIf(is_reply = 0 AND t > C - 31536000 AND t <= C),
    countIf(is_reply = 1 AND t > C - 2592000 AND t <= C), countIf(is_reply = 1 AND t > C - {W} AND t <= C), countIf(is_reply = 1 AND t > C - 31536000 AND t <= C),
    countIf(is_quote = 1 AND t > C - 2592000 AND t <= C), countIf(is_quote = 1 AND t > C - {W} AND t <= C), countIf(is_quote = 1 AND t > C - 31536000 AND t <= C),
    countIf(t <= C), maxIf(t, t <= C), countIf(img > 0 AND t > C - {W} AND t <= C), countIf(is_reply = 0 AND t > C AND t <= C + {LA})
  FROM pi ARRAY JOIN {cuts_arr} AS C, range(5) AS cidx GROUP BY p, cidx
  UNION ALL
  SELECT p, cidx, 0, 0, 0, uniqExact(u), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, toUInt32(0), 0, 0
  FROM {src(i_glob('cur'), 'cur')} ARRAY JOIN {cuts_arr} AS C, range(5) AS cidx WHERE {ins('u')} AND t <= C GROUP BY p, cidx
  UNION ALL
  SELECT p, cidx, 0, 0, 0, 0, uniqExact(u), sum(w), 0, 0, 0, 0, 0, 0, 0, 0, 0, toUInt32(0), 0, 0 FROM (
    SELECT u, p, min(t) AS t, max(if(b >= {thr}, 1 / (1 + (b - 1) / {div}), 1)) AS w FROM {src(i_glob('blk'), 'blk')} WHERE {ins('u')} GROUP BY u, p)
  ARRAY JOIN {cuts_arr} AS C, range(5) AS cidx WHERE t <= C GROUP BY p, cidx
) GROUP BY p, cidx;
CREATE TABLE prodc ENGINE = Join(ANY, LEFT, p, cidx) AS
SELECT p, cidx, followers, orig_w, posts_w, curators, orig_label, blockers,
  toFloat32(if(blockers >= {k_min}, 1000 * blockers_w / greatest(1, followers), 0)) AS blk_1k FROM pcs;
CREATE TABLE fown ENGINE = Join(ANY, LEFT, u, p) AS
SELECT u, p, groupArray((ft, dt)) AS iv FROM {src(f'{P1}/i{j:02d}/fol_own.parquet', 'fol')} GROUP BY u, p;

-- ===== O11: deduplicated 12-month like/repost table with type codes =====
{out(final['o11'] + f'/j{j:02d}.parquet')}
SELECT u AS liker, ph AS post_hash, a AS author, t, ty AS type FROM te
WHERE ty IN (1, 2, 3, 4, 5) AND t > {t12} AND joinGet('dmu', 'synth_no', u) = 0 AND joinGet('dmu', 'synth_no', a) = 0;

CREATE TABLE o1 (metric String, k1 String, k2 String, k3 String, v Float64) ENGINE = Memory;
INSERT INTO o1 SELECT 'typed_engagement', toString(ty), '', '', count() FROM te GROUP BY ty;
INSERT INTO o1 SELECT 'replies_quotes', toString(kind), toString(gated), toString(cr_t > 0), count() FROM tr GROUP BY kind, gated, cr_t > 0;
INSERT INTO o1 SELECT 'hostile_within_48h', toString(kind), '', '', countIf(hb_t > 0) FROM tr GROUP BY kind;
INSERT INTO o1 SELECT 'hidden_detached_matched', toString(kind), '', '', countIf(hid = 1 OR det = 1) FROM tr GROUP BY kind;
INSERT INTO o1 SELECT 'supplement_producers', '', '', '', count() FROM (SELECT DISTINCT a FROM te WHERE ty = {TY['LU']});
"""
    # ---------- full pair aggregates at T (eval) and T_now (prod), in user sub-chunks ----------
    l2 = ",\n  ".join(lvl2_cols(r))
    sums = []
    for f in feats:
        fexpr(f, 0)
        sums.append(f"toFloat32(sumIf(a05 * ({{e_{f['name']}}}), aud = 1)) AS `{f['name']}__a05`")
        sums.append(f"toFloat32(sumIf(a10 * ({{e_{f['name']}}}), aud = 1)) AS `{f['name']}__a10`")
    base = [
        "toFloat32(sumIf(a05, aud = 1)) AS `S0__a05`", "toFloat32(sumIf(a10, aud = 1)) AS `S0__a10`",
        "toUInt64(countIf(aud = 1)) AS audience",
        "toUInt64(countIf(light = 1 AND (n_LO + n_LR + n_RP) > 0)) AS light_eng_nonlu",
        "toUInt64(countIf(light = 1 AND n_LU > 0 AND (n_LO + n_LR + n_RP) = 0)) AS light_eng_luonly",
        "toUInt64(countIf(light = 1 AND dl_n >= 2 AND dl_dmin != dl_dmax AND (n_LO + n_LR) > 0)) AS D_nonlu",
        "toUInt64(countIf(light = 1 AND dl_n >= 2 AND dl_dmin != dl_dmax AND (n_LO + n_LR) = 0 AND n_LU > 0)) AS D_luonly",
        "toUInt64(countIf(n_RP > 0)) AS reposters", "toUInt64(sum(n_CR)) AS replies_confirmed",
        "toUInt64(sum(n_CR + n_URf + n_URn + n_RH)) AS replies_all", "toUInt64(sum(n_ACK)) AS courtesy_likes",
        "toUInt64(sum(nlk)) AS likes_all", "toUInt64(countIf(maint = 1)) AS maint_pairs",
    ]
    for cidx, tag in ((names.index("T"), "eval"), (names.index("T_now"), "prod")):
        C = cut[names[cidx]]
        der = ",\n    ".join(derived(r, C, cidx))
        sumsC = ",\n  ".join(s.format(**{f"e_{f['name']}": fexpr(f, C) for f in feats}) for s in sums)
        sql += f"""
CREATE TABLE acc_{tag} (p UInt32, k UInt8, {', '.join(f'`{f}__a05` Float32, `{f}__a10` Float32' for f in fn)}, `S0__a05` Float32, `S0__a10` Float32,
  audience UInt64, light_eng_nonlu UInt64, light_eng_luonly UInt64, D_nonlu UInt64, D_luonly UInt64, reposters UInt64, replies_confirmed UInt64,
  replies_all UInt64, courtesy_likes UInt64, likes_all UInt64, maint_pairs UInt64) ENGINE = MergeTree ORDER BY p;
"""
        for k in range(K):
            sql += f"""
CREATE TABLE l2_{tag}_{k} ENGINE = MergeTree ORDER BY (u, p) AS
SELECT *, {der}
FROM (SELECT u, p,
  {l2}
  FROM ({lvl1(r, C, f'cityHash64(u) % {K} = {k}')}) GROUP BY u, p);
INSERT INTO acc_{tag} SELECT p, {k}, {sumsC},
  {', '.join(base)}
FROM l2_{tag}_{k} WHERE joinGet('dmu', 'is_prod', p) = 1 GROUP BY p;
{out(o + f'/e0_{tag}_{k}.parquet')}
SELECT u, p, toUInt8({cidx}) AS cidx, toUInt8(qa_n >= {r['teacher']['min_engagements']} AND qa_dmin != qa_dmax) AS withqn,
  toUInt8(qb_n >= {r['teacher']['min_engagements']} AND qb_dmin != qb_dmax) AS noqn
FROM l2_{tag}_{k}
WHERE (withqn = 1 OR noqn = 1) AND light = 1 AND maint = 0 AND b1own = 0 AND u != p AND joinGet('dmu', 'is_prod', p) = 1
  AND joinGet('dmu', 'synth_no', u) = 0 AND joinGet('dmu', 'hide', p) = 0 AND joinGet('dmu', 'synth_no', p) = 0;
{out(o + f'/maintd/k{{_partition_id}}_{tag}_{k}.parquet')} PARTITION BY kpart
SELECT u, p, toUInt8({cidx}) AS cidx, toUInt32(nlk) AS n_up, toUInt32(n_pu) AS n_pu, maint AS flag,
  toUInt8(cityHash64(least(u, p), greatest(u, p)) % {N}) AS kpart FROM l2_{tag}_{k} WHERE mutual = 1;
INSERT INTO o1 SELECT 'pairs_{tag}', toString({k}), '', '', count() FROM l2_{tag}_{k};
INSERT INTO o1 SELECT 'aud_pairs_{tag}', toString({k}), '', '', countIf(aud = 1) FROM l2_{tag}_{k};
DROP TABLE l2_{tag}_{k};
"""
    # ---------- O2 ----------
    comb = []
    for cidx, tag in ((names.index("T"), "eval"), (names.index("T_now"), "prod")):
        C = cut[names[cidx]]
        comb.append(f"""
CREATE TABLE comb_{tag} ENGINE = Join(ANY, LEFT, p) AS
SELECT p, uniqExact(x) AS targets, countIf(hb > 0) / greatest(1, count()) AS share FROM (
  SELECT u AS p, x, t, toUInt32(arrayMin(arrayFilter(c -> c > t AND c <= t + {hw} AND c <= {C}, joinGet('bpj', 'ts', x, u)))) AS hb
  FROM (SELECT u, if(is_reply = 1, par_a, q_a) AS x, t FROM pi WHERE ((is_reply = 1 AND par_a != {NONE}) OR (is_quote = 1 AND q_a != {NONE}))
        AND t > {C - W} AND t <= {C})
  WHERE x != u AND NOT arrayExists(iv -> iv.1 <= t AND t < iv.2, joinGet('fown', 'iv', u, x))
) GROUP BY p;
CREATE TABLE lang_{tag} ENGINE = Join(ANY, LEFT, p) AS
SELECT u AS p, topK(3)(lang) AS langs FROM (SELECT u, arrayJoin(langs) AS lang FROM pi WHERE t > {C - 31536000} AND t <= {C}) GROUP BY u;""")
    sql += "\n".join(comb)
    for cidx, tag in ((names.index("T"), "eval"), (names.index("T_now"), "prod")):
        sumcols = ", ".join(f"toFloat32(sum(`{f}__a05`)) AS `{f}__a05`, toFloat32(sum(`{f}__a10`)) AS `{f}__a10`" for f in fn + ["S0"])
        basecols = ", ".join(f"sum({c}) AS {c}" for c in ["audience", "light_eng_nonlu", "light_eng_luonly", "D_nonlu", "D_luonly", "reposters",
                                                         "replies_confirmed", "replies_all", "courtesy_likes", "likes_all", "maint_pairs"])
        sql += f"""
{out(final['o2'] + f'/{tag}/j{j:02d}.parquet')}
SELECT a.p AS p, {cidx} AS cidx, a.* EXCEPT (p),
  toFloat32(replies_confirmed / greatest(1, replies_all)) AS confirmed_reply_share, toFloat32(courtesy_likes / greatest(1, likes_all)) AS courtesy_like_share,
  s.followers AS followers, s.orig_30 AS originals_30d, s.orig_w AS originals_90d, s.orig_12 AS originals_12mo, s.repl_30 AS replies_30d, s.repl_w AS replies_90d,
  s.repl_12 AS replies_12mo, s.quote_30 AS quotes_30d, s.quote_w AS quotes_90d, s.quote_12 AS quotes_12mo, s.lifetime AS lifetime_posts, s.last_post AS last_post_t,
  toFloat32(s.img_w / greatest(1, s.posts_w)) AS image_post_share, s.blockers AS live_blockers_raw,
  if(s.blockers >= {k_min}, toNullable(toFloat32(1000 * s.blockers / greatest(1, s.followers))), NULL) AS blockers_per_1k,
  if(s.blockers >= {k_min}, toNullable(toFloat32(1000 * s.blockers_w / greatest(1, s.followers))), NULL) AS blockers_per_1k_burst_discounted,
  if(joinGet('lb', 'n_sub', a.p, toUInt8({cidx})) >= {k_min}, toNullable(joinGet('lb', 'n_sub', a.p, toUInt8({cidx}))), NULL) AS lblk_subscribers,
  s.curators AS a_half_curators,
  if(joinGet('comb_{tag}', 'targets', a.p) >= {k_min}, toNullable(toFloat32(joinGet('comb_{tag}', 'share', a.p))), NULL) AS combativeness,
  joinGet('lang_{tag}', 'langs', a.p) AS top_langs, joinGet('supj', 'sup', a.p) AS supplement
FROM (SELECT p, {sumcols}, {basecols} FROM acc_{tag} GROUP BY p) AS a
LEFT JOIN (SELECT * FROM pcs WHERE cidx = {cidx}) AS s ON s.p = a.p
WHERE joinGet('dmu', 'is_prod', a.p) = 1;
"""
    # ---------- O3a / O3b: candidate pairs, features and labels ----------
    sql += f"""
CREATE TABLE cs ENGINE = MergeTree ORDER BY (cidx, u, p) AS
SELECT u, p, cidx, ck, pi, toUInt8(1) AS task, toUInt32(0) AS ft FROM {src(i_glob('cand'), 'cand')}
UNION ALL
SELECT u, p, toUInt8(cidx), toUInt8(0), toFloat32(1), toUInt8(2), ft FROM {src(i_glob('fol'), 'fol')}
ARRAY JOIN {cuts_arr} AS C, range(5) AS cidx
WHERE cidx IN {tuple(fit_cidx)} AND joinGet('dmu', '{opts.get('fit_flag', 'fit')}', u) = 1 AND b < {thr} AND ft > C AND ft <= C + {LB0};
CREATE TABLE lev ENGINE = MergeTree ORDER BY (u, p) AS
SELECT u, a AS p, toUInt8(if(ty = {TY['LU']}, 9, 1)) AS lk, t FROM te WHERE (ty = {TY['LO']} OR (ty = {TY['RP']} AND isr = 0) OR ty = {TY['LU']})
  AND (u, a) IN (SELECT u, p FROM cs)
UNION ALL SELECT u, a, toUInt8(1), greatest(t, cr_t) FROM tr WHERE kind = 1 AND cr_t > 0 AND x_has = 1 AND x_isreply = 0 AND (u, a) IN (SELECT u, p FROM cs)
UNION ALL SELECT u, p, toUInt8(2), t FROM bp WHERE (u, p) IN (SELECT u, p FROM cs)
UNION ALL SELECT c_a, p, toUInt8(3), c_t FROM gt WHERE kind IN (1, 2) AND (c_a, p) IN (SELECT u, p FROM cs);
CREATE TABLE levj ENGINE = Join(ANY, LEFT, u, p) AS SELECT u, p, groupArray((lk, t)) AS evs FROM lev GROUP BY u, p;
"""
    fcols = ", ".join(f"toFloat32({fexpr(f, '{C}')}) AS `{f['name']}`" for f in feats)
    qcols = ", ".join(f"toUInt8(least(255, greatest(0, round(({fexpr(f, '{C}')}) / {scale(f)} * 255)))) AS `{f['name']}`" for f in feats)
    for cidx in sorted(set(fit_cidx + [names.index("T")])):
        cname = names[cidx]
        C = cut[cname]
        cap = r.label_cap(cname)
        der = ",\n    ".join(derived(r, C, cidx))
        lA1 = C + LA
        lC1 = min(C + LC, cap)
        lvl = lvl1(r, C, f"(u, p) IN (SELECT u, p FROM cs WHERE cidx = {cidx})")
        sql += f"""
CREATE TABLE l2c_{cidx} ENGINE = MergeTree ORDER BY (u, p) AS
SELECT u, p, {l2} FROM ({lvl}) GROUP BY u, p;
CREATE TABLE f_{cidx} ENGINE = MergeTree ORDER BY (u, p) AS
SELECT x.*, {der} FROM (
  SELECT c.u AS u, c.p AS p, c.ck AS ck, c.pi AS pi, c.task AS task, c.ft AS ft, l.* EXCEPT (u, p)
  FROM (SELECT * FROM cs WHERE cidx = {cidx}) AS c LEFT JOIN l2c_{cidx} AS l ON l.u = c.u AND l.p = c.p
) AS x;
"""
        if cidx in fit_cidx:
            sql += f"""
{out(final['o3a'] + f'/{cname}/j{j:02d}.parquet')}
SELECT u, p, toUInt8({cidx}) AS cidx, task, ck, pi, ft, joinGet('dmu', 'ho', u) AS heldout,
  if(task = 1, if(arrayExists(e -> e.1 = 9 AND e.2 > {C} AND e.2 <= {lA1}, ev) AND NOT arrayExists(e -> e.1 = 1 AND e.2 > {C} AND e.2 <= {lA1}, ev), NULL,
     toNullable(toUInt8(arrayExists(e -> e.1 = 1 AND e.2 > {C} AND e.2 <= {lA1}, ev)))), NULL) AS y_a,
  if(task = 2, toNullable(toUInt8(arrayExists(e -> e.1 = 1 AND e.2 > ft + {LB0} AND e.2 <= least(ft + {LB1}, {cap}), ev))), NULL) AS y_b,
  toUInt8(arrayExists(e -> e.1 IN (2, 3) AND e.2 > {C} AND e.2 <= {lC1}, ev)) AS y_c,
  toUInt8(joinGet('prodc', 'orig_label', p, toUInt8({cidx})) > 0) AS elig_p, joinGet('usj', 'elig_u', u, toUInt8({cidx})) AS elig_u,
  toFloat32(log1p(joinGet('prodc', 'orig_label', p, toUInt8({cidx})))) AS offset,
  {fcols.format(C=C)}
FROM (SELECT *, joinGet('levj', 'evs', u, p) AS ev FROM f_{cidx});
"""
        if cidx == names.index("T"):
            sql += f"""
{out(final['o3b'] + f'/features_j{j:02d}.parquet')}
SELECT u, p, {qcols.format(C=C)} FROM f_{cidx} WHERE ck = 4;
"""
    # ---------- O6, O8, O9, O12 candidates (author-side, text from the local buckets) ----------
    o8 = r["o8"]
    recent = r["o6_recent_days"] * 86400
    o12 = r["o12"]
    per = (o12["per_stratum"] + N - 1) // N * 2
    nq = (o12["random_nonfollower_quotes"] + N - 1) // N * 2
    sql += f"""
CREATE TABLE eng_light ENGINE = Join(ANY, LEFT, ph) AS
SELECT ph, uniqExactIf(u, joinGet('usj', 'light', u, toUInt8(4)) = 1) AS nl FROM te WHERE ty IN ({TY['LO']}, {TY['RP']}, {TY['LR']}) GROUP BY ph;
{out(final['o6'] + f'/j{j:02d}.parquet')}
SELECT u AS p, concat('at://', joinGet('dmd', 'did', u), '/app.bsky.feed.post/', rkey) AS uri, nl AS light_engagers, rn AS rank FROM (
  SELECT u, rkey, joinGet('eng_light', 'nl', ph) AS nl, row_number() OVER (PARTITION BY u ORDER BY nl DESC, cityHash64(ph)) AS rn
  FROM pi WHERE is_reply = 0 AND t > {tn - recent} AND t <= {tn} AND joinGet('dmu', 'is_prod', u) = 1
    AND joinGet('dmu', 'hide', u) = 0 AND joinGet('dmu', 'synth_no', u) = 0
) WHERE rn <= 3 AND nl > 0;
CREATE TABLE ptxt ENGINE = Join(ANY, LEFT, ph) AS
SELECT ph, argMax(text, (rev, seq)) AS text, argMax(alts, (rev, seq)) AS alts, argMax(labels, (rev, seq)) AS labels
FROM {src(f'{buckets}/po/p{j}.parquet', 'po')} WHERE op != 3 GROUP BY ph;
{out(final['o8'] + f'/j{j:02d}.parquet')}
SELECT u AS p, concat('at://', joinGet('dmd', 'did', u), '/app.bsky.feed.post/', rkey) AS uri,
  formatDateTime(toDateTime(t), '%Y-%m-%dT%H:%i:%SZ') AS created,
  concat(joinGet('ptxt', 'text', ph), arrayStringConcat(arrayMap(x -> concat(' [image: ', x, ']'), arrayFilter(x -> x != '', joinGet('ptxt', 'alts', ph))), '')) AS text,
  langs, img AS image_count, joinGet('ptxt', 'labels', ph) AS self_labels
FROM (
  SELECT *, row_number() OVER (PARTITION BY u ORDER BY t DESC) AS rall FROM (
    SELECT *, row_number() OVER (PARTITION BY u, toYYYYMM(toDateTime(t)) ORDER BY joinGet('eng_light', 'nl', ph) DESC, cityHash64(ph)) AS rm
    FROM pi WHERE is_reply = 0 AND t > {t12} AND joinGet('dmu', 'is_prod', u) = 1 AND joinGet('dmu', 'hide', u) = 0 AND joinGet('dmu', 'synth_no', u) = 0
  ) WHERE rm <= {o8['per_month']}
) WHERE rall <= {o8['max_posts']};
{out(final['o9'] + f'/j{j:02d}.parquet')}
SELECT u AS p, argMax(name, (rev, seq)) AS display_name, argMax(descr, (rev, seq)) AS description
FROM {src(f'{buckets}/ev/p{j}.parquet', 'ev')} WHERE coll = 10 AND kind = 1 AND op != 3 AND joinGet('dmu', 'is_prod', u) = 1 AND joinGet('dmu', 'synth_no', u) = 0
GROUP BY u;
{out(o + '/o12c.parquet')}
SELECT stratum, kind, c_ph, c_a, x_ph, x_a, hsh FROM (
  SELECT *, row_number() OVER (PARTITION BY stratum ORDER BY hsh) AS rn FROM (
    SELECT toUInt8(multiIf(kind = 2 AND det = 1, 1, hb_t > 0, 2, fol = 0, 3,
                           arrayExists(iv -> iv.1 <= t AND t < iv.2, joinGet('fown', 'iv', a, u)), 5, 4)) AS stratum,
      kind, r_ph AS c_ph, u AS c_a, x_ph, a AS x_a, cityHash64(r_ph, 'o12') AS hsh
    FROM tr WHERE t > {t12} AND joinGet('dmu', 'hide', u) = 0 AND joinGet('dmu', 'hide', a) = 0
      AND joinGet('dmu', 'synth_no', u) = 0 AND joinGet('dmu', 'synth_no', a) = 0
    UNION ALL
    SELECT toUInt8(6), kind, r_ph, u, x_ph, a, cityHash64(r_ph, 'o12q') FROM tr
    WHERE kind = 2 AND fol = 0 AND t > {t12} AND joinGet('dmu', 'hide', u) = 0 AND joinGet('dmu', 'hide', a) = 0
      AND joinGet('dmu', 'synth_no', u) = 0 AND joinGet('dmu', 'synth_no', a) = 0)
) WHERE rn <= if(stratum = 6, {nq}, {per});
-- O13: the M1/M2 users' own qualifying engagement toward this partition's authors (eval_temporal.py inputs).
-- Kinds: like = LO/LR/LU likes, repost = RP/LU reposts, confirmed_reply = CR; ACK (courtesy) likes excluded.
CREATE TABLE o13u ENGINE = Join(ANY, LEFT, uid) AS
SELECT uid, did, {o13flag} AS m1x, m2c FROM file('{G}/didmap.parquet', Parquet, '{S['didmap']}') WHERE {o13flag} = 1 OR m2c = 1;
{out(o + '/o13.parquet')}
SELECT joinGet('o13u', 'm1x', u) AS m1, joinGet('o13u', 'm2c', u) AS m2, joinGet('o13u', 'did', u) AS did,
  joinGet('dmd', 'did', a) AS subject, kind, toInt64(max(t)) * 1000000 AS created_us
FROM (
  SELECT u, a, if(typ = 2, 'repost', 'like') AS kind, t FROM te WHERE ty IN ({TY['LO']}, {TY['LR']}, {TY['LU']}, {TY['RP']})
  UNION ALL SELECT u, a, 'confirmed_reply', t FROM tr WHERE kind = 1 AND cr_t > 0 AND gated = 0
) WHERE (joinGet('o13u', 'm1x', u) = 1 OR joinGet('o13u', 'm2c', u) = 1) AND t >= {o13lo} AND t < {o13hi}
GROUP BY u, a, kind, intDiv(t, 86400);
INSERT INTO o1 SELECT 'o13_rows', toString(m1), toString(m2), '', count() FROM file('{o}/o13.parquet', Parquet, '{S['o13p']}') GROUP BY m1, m2;
{out(o + '/o1.parquet')} SELECT * FROM o1;
"""
    return sql
