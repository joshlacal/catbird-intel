"""Global (not partitioned) steps: inventory, DID map, bucketing, and the merges
between and after the two phases."""
from __future__ import annotations

from sqlc import (C_BLOCK, C_FOLLOW, C_LIKE, C_LIST, C_LISTBLOCK, C_LISTITEM, C_POSTGATE, C_REPOST,
                  C_THREADGATE, COLL, KIND, NONE, OP, PRELUDE, S, SRC, map_expr, out, settings_sql, src)

EV_GLOB = "{stage}/**/events-*.parquet"
PO_GLOB = "{stage}/**/posts-*.parquet"


def inventory(rules, stage: str, o: str) -> str:
    n = rules.N
    ev, po = EV_GLOB.format(stage=stage), PO_GLOB.format(stage=stage)
    return PRELUDE + settings_sql(rules) + f"""
{out(o + '/files_ev.parquet')} SELECT 'events' AS kind, _file AS file, count() AS rows FROM file('{ev}', Parquet) GROUP BY _file;
{out(o + '/files_po.parquet')} SELECT 'posts' AS kind, _file AS file, count() AS rows FROM file('{po}', Parquet) GROUP BY _file;
{out(o + '/ev_skew.parquet')} SELECT toUInt8(cityHash64(did) % {n}) AS part, ifNull(source_id, '') AS source_id,
  ifNull(kind, '') AS kind, count() AS rows, min(time) AS tmin, max(time) AS tmax
FROM file('{ev}', Parquet) GROUP BY ALL;
{out(o + '/po_skew.parquet')} SELECT toUInt8(cityHash64(did) % {n}) AS part, ifNull(source_id, '') AS source_id,
  count() AS rows, min(time) AS tmin, max(time) AS tmax FROM file('{po}', Parquet) GROUP BY ALL;
"""


def read_rate(stage: str) -> str:
    """Decode-only scan of the bucketing columns, so S8 can split the bucket step's
    time into read cost and per-row write cost."""
    ev, po = EV_GLOB.format(stage=stage), PO_GLOB.format(stage=stage)
    return PRELUDE + f"""
SELECT 'events', count(), sum(cityHash64(did, collection, rkey, rev, subject, list_uri) % 7), sum(length(extra) + length(name) + length(description))
FROM file('{ev}', Parquet) FORMAT TSV;
SELECT 'posts', count(), sum(cityHash64(did, rkey, rev, reply_parent, reply_root, quote_uri) % 7), sum(length(text))
FROM file('{po}', Parquet) FORMAT TSV;
"""


def didmap(rules, stage: str, side: str, o: str, have: dict) -> str:
    """One DID -> u32 map for every account in the data. Producers keep their
    edges.bin pid; everyone else gets NP + rank by DID. Exclusion flags ride along."""
    ev, po = EV_GLOB.format(stage=stage), PO_GLOB.format(stage=stage)
    smp = rules["samples"]
    ls = rules["list_split"]
    n = rules.N

    def halv(base: str, salt0: int, k: int) -> str:
        parts = [base] + [f"cityHash64(concat(did, 'h{salt0 + i}')) % 2 = 0" for i in range(k)]
        return " AND ".join(f"({p})" for p in parts)

    fit = halv(f"h % {smp['fit_mod']} = {smp['fit_rem']}", 2, int(smp["fit_halvings"]))
    m1 = f"h % {smp['m1_mod'] * (2 ** int(smp['m1_halvings']))} = {smp['m1_rem']}"
    lsx = f"cityHash64(concat(did, '{ls['salt']}')) % {ls['mod']}"
    def _in(vals):
        return "(" + ", ".join(str(v) for v in vals) + ")"
    # S1 §6: cityHash64(owner) % 8 -> {0,1,4,5} A, 2 B1a, 6 B1b, {3,7} B2
    lscode = (f"multiIf({lsx} IN {_in(ls['A'])}, 1, {lsx} IN {_in(ls['B1a'])}, 2, {lsx} IN {_in(ls['B1b'])}, 3, "
              f"{lsx} IN {_in(ls['B2'])}, 4, 0)")
    hox = f"cityHash64(concat(did, '{smp['holdout_salt']}')) % {smp['holdout_mod']} = {smp['holdout_rem']}"
    status_join = "LEFT JOIN st ON st.did = a.did" if have.get("status") else ""
    status_cols = "st.did != '' AS in_listing, st.active AS relay_active" if have.get("status") else "0 AS in_listing, 1 AS relay_active"
    complete = 1 if (have.get("status") and rules["status_list_complete"]) else 0
    plc = (f"CREATE TABLE plc ENGINE = Join(ANY, LEFT, did) AS SELECT did, toUInt32(parseDateTimeBestEffortOrZero(created)) AS plc_t "
           f"FROM file('{side}/plc.tsv', TSV, 'did String, created String');") if have.get("plc") else \
        "CREATE TABLE plc (did String, plc_t UInt32) ENGINE = Join(ANY, LEFT, did);"
    status = (f"CREATE TABLE st ENGINE = Memory AS SELECT did, toUInt8(lower(active) IN ('true','1')) AS active, status "
              f"FROM file('{side}/status.tsv', TSV, 'did String, active String, status String');") if have.get("status") else ""
    return PRELUDE + settings_sql(rules) + f"""
CREATE TABLE prod ENGINE = Memory AS SELECT pid, did, followers FROM file('{side}/producers.tsv', TSV, 'pid UInt32, did String, followers UInt32');
CREATE TABLE hide ENGINE = Join(ANY, LEFT, did) AS SELECT DISTINCT did, toUInt8(1) AS f FROM file('{side}/hide.tsv', TSV, 'did String') WHERE did != '';
CREATE TABLE intents ENGINE = Join(ANY, LEFT, did) AS SELECT did, max(lower(bulk) = 'false') AS bulk_no, max(lower(synth) = 'false') AS synth_no
  FROM file('{side}/intents.tsv', TSV, 'did String, bulk String, bridging String, archive String, synth String') WHERE startsWith(did, 'did:') GROUP BY did;
{plc}
{status}
CREATE TABLE ids ENGINE = MergeTree ORDER BY did AS
SELECT d AS did,
  argMaxIf(aa, (ti, sq), isacct) AS aact, argMaxIf(ast, (ti, sq), isacct) AS astat, max(isacct) AS has_acct
FROM (
  SELECT x.1 AS d, toUInt8(kd = 'account' AND x.2 = 0) AS isacct, aa, ast, ti, sq FROM (
    SELECT arrayJoin(arrayFilter(y -> startsWith(y.1, 'did:') AND length(y.1) <= 256, [
        (ifNull(did, ''), 0),
        (if(collection IN ('app.bsky.graph.follow', 'app.bsky.graph.block', 'app.bsky.graph.listitem'), ifNull(subject, ''), ''), 1),
        (uri_did(ifNull(subject, '')), 1), (uri_did(ifNull(list_uri, '')), 1)])) AS x,
      ifNull(kind, '') AS kd, toInt8(if(account_active IS NULL, -1, toInt8(account_active))) AS aa, ifNull(account_status, '') AS ast,
      ifNull(time, '') AS ti, ifNull(seq, 0) AS sq
    FROM file('{ev}', Parquet))
  UNION ALL
  SELECT arrayJoin(arrayFilter(x -> startsWith(x, 'did:') AND length(x) <= 256,
      [ifNull(did, ''), uri_did(ifNull(reply_parent, '')), uri_did(ifNull(reply_root, '')), uri_did(ifNull(quote_uri, ''))])) AS d,
    toUInt8(0), toInt8(-1), '', '', toUInt64(0)
  FROM file('{po}', Parquet)
) GROUP BY d;
CREATE TABLE np ENGINE = Memory AS SELECT toUInt32(count()) AS n FROM prod;
CREATE TABLE a ENGINE = MergeTree ORDER BY did AS
SELECT did, toUInt32(if(is_prod, pid, (SELECT n FROM np) + rn - 1)) AS uid, is_prod, followers, aact, astat, has_acct FROM (
  SELECT u.did AS did, p.did != '' AS is_prod, p.pid AS pid, p.followers AS followers, u.aact AS aact, u.astat AS astat, u.has_acct AS has_acct,
    if(p.did != '', 0, row_number() OVER (PARTITION BY p.did != '' ORDER BY u.did)) AS rn
  FROM (SELECT did, aact, astat, has_acct FROM ids UNION ALL SELECT did, -1, '', 0 FROM prod WHERE did NOT IN (SELECT did FROM ids)) AS u
  LEFT JOIN prod AS p ON p.did = u.did
);
{out(o + '/didmap.parquet')}
SELECT did, uid, h, toUInt8(h % {n}) AS part, toUInt8(is_prod) AS is_prod, toUInt8({lscode}) AS ls,
  toUInt8({fit}) AS fit, toUInt8({m1}) AS m1, toUInt8(h % {smp['pc_mod']} < {smp['pc_lt']}) AS pc,
  toUInt8({hox}) AS ho,
  toUInt8(h % {smp['s8_actor_mod'][0]} = 0) AS s200, toUInt8(h % {smp['s8_actor_mod'][1]} = 0) AS s400,
  toUInt8(joinGet('hide', 'f', did)) AS hide, toUInt8(joinGet('intents', 'synth_no', did)) AS synth_no,
  toUInt8(joinGet('intents', 'bulk_no', did)) AS bulk_no,
  toUInt8(purge_reason > 0) AS purge, purge_reason, joinGet('plc', 'plc_t', did) AS plc_t,
  toUInt8(startsWith(did, 'did:web:')) AS web, toUInt32(followers) AS followers
FROM (
  SELECT a.*, cityHash64(a.did) AS h, {status_cols},
    toUInt8(multiIf(relay_active = 0 AND in_listing, 1, {complete} = 1 AND NOT in_listing, 2,
                    has_acct = 1 AND aact = 0, 3, joinGet('intents', 'bulk_no', a.did) = 1, 4, 0)) AS purge_reason
  FROM a {status_join}
) ORDER BY uid;
"""


def bucket(rules, stage: str, dm: str, o: str, actor_filter: str) -> str:
    """Scan every staged file once; map DIDs to u32; drop purged and bulkDataset:false
    accounts; write 16 actor/author-keyed buckets to local SSD (never uploaded:
    post text stays on the VM)."""
    from rules import epoch
    ev, po = EV_GLOB.format(stage=stage), PO_GLOB.format(stage=stage)
    lo, hi = rules["tid_valid"]
    tlo, thi = epoch(lo) * 1_000_000, epoch(hi) * 1_000_000
    coll = map_expr("ifNull(collection, '')", COLL)
    kind = map_expr("ifNull(kind, '')", KIND)
    op = map_expr("ifNull(operation, '')", OP)
    srcc = map_expr("ifNull(source_id, '')", SRC)
    scoll = "transform(uri_coll(subj_s), ['app.bsky.feed.post', 'app.bsky.feed.generator', 'app.bsky.graph.list'], [1, 2, 3], 0)"
    tsec = (f"if(tus >= {tlo} AND tus < {thi}, toUInt32(intDiv(tus, 1000000)), "
            f"toUInt32(parseDateTimeBestEffortOrZero(ifNull(created_at, ''))))")
    return PRELUDE + settings_sql(rules) + f"""
CREATE TABLE dmj ENGINE = Join(ANY, LEFT, did) AS SELECT did, uid + 1 AS uid1, purge, part FROM file('{dm}', Parquet, '{S['didmap']}');
{out(o + '/ev/p{_partition_id}.parquet')} PARTITION BY part
SELECT toUInt32(u1 - 1) AS u, pt AS part, kc AS kind, cc AS coll, oc AS op, rk, t, tok, rv AS rev, sq AS seq, ing, sr AS src,
  toUInt32(if(s1 > 0, s1 - 1, {NONE})) AS subj, toUInt8(spart0) AS spart, sph, scoll, st, lh, gph, cr AS created,
  xx AS extra, trunc, nm AS name, ds AS descr, feed, aact, astat
FROM (
  SELECT ifNull(did, '') AS d,
    joinGet('dmj', 'uid1', d) AS u1, joinGet('dmj', 'purge', d) AS px, joinGet('dmj', 'part', d) AS pt,
    {kind} AS kc, {coll} AS cc, {op} AS oc, cityHash64(ifNull(rkey, '')) AS rk,
    tid_us(ifNull(rkey, '')) AS tus, toUInt8(tus >= {tlo} AND tus < {thi}) AS tok, {tsec} AS t,
    tid_us(ifNull(rev, '')) AS rv, ifNull(seq, 0) AS sq, toUInt32(parseDateTimeBestEffortOrZero(ifNull(time, ''))) AS ing,
    {srcc} AS sr, ifNull(subject, '') AS subj_s,
    multiIf(cc IN ({C_FOLLOW}, {C_BLOCK}, {C_LISTITEM}), subj_s, startsWith(subj_s, 'at://'), uri_did(subj_s), '') AS sdid,
    joinGet('dmj', 'uid1', sdid) AS s1, joinGet('dmj', 'purge', sdid) AS spx, joinGet('dmj', 'part', sdid) AS spart0,
    if(cc IN ({C_LIKE}, {C_REPOST}, {C_LISTBLOCK}), cityHash64(subj_s), 0) AS sph, {scoll} AS scoll,
    if(cc IN ({C_LIKE}, {C_REPOST}), toUInt32(intDiv(tid_us(uri_rkey(subj_s)), 1000000)), 0) AS st,
    multiIf(cc = {C_LISTITEM}, cityHash64(ifNull(list_uri, '')), cc = {C_LIST}, cityHash64(concat('at://', d, '/app.bsky.graph.list/', ifNull(rkey, ''))),
            cc = {C_LISTBLOCK}, cityHash64(subj_s), 0) AS lh,
    if(cc IN ({C_THREADGATE}, {C_POSTGATE}), cityHash64(concat('at://', d, '/app.bsky.feed.post/', ifNull(rkey, ''))), 0) AS gph,
    toUInt32(parseDateTimeBestEffortOrZero(ifNull(created_at, ''))) AS cr,
    if(cc IN ({C_LIST}, {C_THREADGATE}, {C_POSTGATE}, 13), ifNull(extra, ''), '') AS xx,
    toUInt8(length(ifNull(extra, '')) >= 4000) AS trunc,
    if(cc = 10, ifNull(name, ''), '') AS nm, if(cc = 10, ifNull(description, ''), '') AS ds,
    if(scoll = 2, subj_s, '') AS feed,
    toInt8(if(account_active IS NULL, -1, toInt8(account_active))) AS aact, ifNull(account_status, '') AS astat
  FROM file('{ev}', Parquet)
  WHERE {actor_filter}
) WHERE u1 > 0 AND px = 0 AND (s1 = 0 OR spx = 0) AND NOT (sdid != '' AND s1 = 0 AND cc IN ({C_FOLLOW}, {C_BLOCK}, {C_LISTITEM}));
{out(o + '/po/p{_partition_id}.parquet')} PARTITION BY part
SELECT toUInt32(u1 - 1) AS u, pt AS part, rk, rks AS rkey, t, tok, rv AS rev, sq AS seq, oc AS op, ing, sr AS src, ph, cr AS created,
  par_ph, toUInt32(if(pa1 > 0 AND papx = 0, pa1 - 1, {NONE})) AS par_a, par_t, root_ph,
  toUInt32(if(ra1 > 0 AND rapx = 0, ra1 - 1, {NONE})) AS root_a,
  q_ph, toUInt32(if(qa1 > 0 AND qapx = 0, qa1 - 1, {NONE})) AS q_a, q_t, img, la AS langs, lb AS labels, tx AS text,
  al AS alts, em AS embed
FROM (
  SELECT ifNull(did, '') AS d, joinGet('dmj', 'uid1', d) AS u1, joinGet('dmj', 'purge', d) AS px, joinGet('dmj', 'part', d) AS pt,
    ifNull(rkey, '') AS rks, cityHash64(rks) AS rk,
    tid_us(rks) AS tus, toUInt8(tus >= {tlo} AND tus < {thi}) AS tok, {tsec} AS t,
    tid_us(ifNull(rev, '')) AS rv, ifNull(seq, 0) AS sq, {op} AS oc,
    toUInt32(parseDateTimeBestEffortOrZero(ifNull(time, ''))) AS ing, {srcc} AS sr,
    cityHash64(concat('at://', d, '/app.bsky.feed.post/', rks)) AS ph,
    toUInt32(parseDateTimeBestEffortOrZero(ifNull(created_at, ''))) AS cr,
    ifNull(reply_parent, '') AS rp_s, ifNull(reply_root, '') AS rr_s, ifNull(quote_uri, '') AS q_s,
    if(rp_s != '', cityHash64(rp_s), 0) AS par_ph, joinGet('dmj', 'uid1', uri_did(rp_s)) AS pa1, joinGet('dmj', 'purge', uri_did(rp_s)) AS papx,
    toUInt32(intDiv(tid_us(uri_rkey(rp_s)), 1000000)) AS par_t,
    if(rr_s != '', cityHash64(rr_s), 0) AS root_ph, joinGet('dmj', 'uid1', uri_did(rr_s)) AS ra1, joinGet('dmj', 'purge', uri_did(rr_s)) AS rapx,
    if(uri_coll(q_s) = 'app.bsky.feed.post', cityHash64(q_s), 0) AS q_ph,
    if(q_ph != 0, joinGet('dmj', 'uid1', uri_did(q_s)), 0) AS qa1, joinGet('dmj', 'purge', uri_did(q_s)) AS qapx,
    toUInt32(intDiv(tid_us(uri_rkey(q_s)), 1000000)) AS q_t,
    toUInt8(least(255, ifNull(image_count, 0))) AS img, ifNull(langs, []) AS la, ifNull(self_labels, []) AS lb,
    ifNull(text, '') AS tx, ifNull(image_alts, []) AS al, ifNull(embed_type, '') AS em
  FROM file('{po}', Parquet)
  WHERE {actor_filter}
) WHERE u1 > 0 AND px = 0;
"""


def ustats_merge(rules, p1: str, o: str) -> str:
    """Concatenate per-partition user stats; add each user's daily-volume percentile
    per cutoff (needs the global distribution)."""
    return PRELUDE + settings_sql(rules) + f"""
CREATE TABLE us ENGINE = MergeTree ORDER BY (cidx, u) AS SELECT * FROM {src(p1 + '/i*/ustats.parquet', 'ustats')};
CREATE TABLE q ENGINE = Memory AS SELECT cidx, quantilesExact({', '.join(f'{i / 100:.2f}' for i in range(1, 100))})(vol) AS qs FROM us WHERE vol > 0 GROUP BY cidx;
{out(o + '/ustats.parquet')} SELECT us.*, toFloat32(if(vol = 0, 0, arrayCount(x -> x <= vol, q.qs) / 100.0)) AS vol_pct
FROM us LEFT JOIN q USING (cidx) ORDER BY cidx, u;
"""


def gates_merge(rules, p1: str, dm: str, o: str) -> str:
    """Resolve hidden/detached content authors (hashed in phase 1 to keep the
    intermediates integer-keyed) to u32 and key the rows by the gate owner's partition."""
    return PRELUDE + settings_sql(rules) + f"""
CREATE TABLE hmap ENGINE = Join(ANY, LEFT, h) AS SELECT h, uid + 1 AS uid1 FROM file('{dm}', Parquet, '{S['didmap']}');
{out(o + '/gates/j{_partition_id}.parquet')} PARTITION BY ppart
SELECT p, kind, c_ph, toUInt32(if(c1 > 0, c1 - 1, {NONE})) AS c_a, c_t, g_ph, trunc, allow_empty, gpart AS ppart
FROM (SELECT *, joinGet('hmap', 'uid1', c_h) AS c1 FROM {src(p1 + '/i*/gates.parquet', 'gates1')});
"""


def lblk(rules, p1: str, o: str, o5raw: str, dm: str = "") -> str:
    """List-blocks (LBLK): per producer, distinct list-block subscribers whose
    subscribed mod lists cover it at each cutoff, counted once per list [F19];
    plus O5-raw's list-block subscription rows and those lists' members at T."""
    c = rules.cut
    cut_arr = "[" + ", ".join(str(c[k]) for k in rules.cut_names) + "]"
    return PRELUDE + settings_sql(rules) + f"""
CREATE TABLE dmp ENGINE = Join(ANY, LEFT, uid) AS SELECT uid, part FROM file('{dm or o + "/didmap.parquet"}', Parquet, '{S['didmap']}');
CREATE TABLE li ENGINE = MergeTree ORDER BY lh AS SELECT * FROM {src(p1 + '/i*/modli.parquet', 'modli')};
CREATE TABLE sb ENGINE = MergeTree ORDER BY lh AS SELECT * FROM {src(p1 + '/i*/lbsub.parquet', 'lbsub')};
{out(o + '/lblk.parquet')}
SELECT member AS p, toUInt8(cidx) AS cidx, uniqExact(sb.u) AS n_sub
FROM sb INNER JOIN li ON li.lh = sb.lh ARRAY JOIN {cut_arr} AS C, range(5) AS cidx
WHERE sb.t <= C AND li.t <= C AND sb.u != member GROUP BY p, cidx;
{out(o + '/modli_by_part/j{_partition_id}.parquet')} PARTITION BY mpart
SELECT lh, member, t, toUInt8(joinGet('dmp', 'part', member)) AS mpart FROM li WHERE lh IN (SELECT lh FROM sb);
{out(o5raw + '/listblock_subs.parquet')} SELECT u, lh, t FROM sb;
{out(o5raw + '/listblock_members_at_T.parquet')} SELECT lh, member, t FROM li WHERE lh IN (SELECT lh FROM sb) AND t <= {c['T']};
"""


def maint_shuffle(rules, p2: str, o: str) -> str:
    """The third, small shuffle keyed by the unordered pair: combine both
    directions of every mutual pair's MAINT counts into O11's side table."""
    return PRELUDE + settings_sql(rules) + f"""
{out(o + '/maint_pairs.parquet')}
SELECT least(u, p) AS u1, greatest(u, p) AS u2, cidx,
  maxIf(flag, u = least(u, p)) AS flag_12, maxIf(flag, u != least(u, p)) AS flag_21,
  maxIf(n_up, u = least(u, p)) AS n_12, maxIf(n_up, u != least(u, p)) AS n_21
FROM {src(p2 + '/j*/maintd/k*.parquet', 'maintd')} GROUP BY u1, u2, cidx;
"""


def o11_users(rules, us: str, o: str) -> str:
    return PRELUDE + f"""
{out(o + '/users_light.parquet')} SELECT u, cidx, light FROM file('{us}', Parquet) WHERE cidx IN (2, 4);
"""


def o7_final(rules, p1: str, o: str) -> str:
    return PRELUDE + f"""
{out(o + '/O7.parquet')} SELECT feed AS feed_uri, if(map = 1, 'cand1', 'cand5') AS partition_map, comm AS community, sum(n) AS n_likers
FROM {src(p1 + '/i*/o7p.parquet', 'o7p')} WHERE comm >= 0 GROUP BY feed, map, comm HAVING n_likers >= 10 ORDER BY feed, map, comm;
"""


def o5prod(rules, p1: str, gates: str, us: str, side: str, o: str, have_maps: bool) -> str:
    """Community x community block and hide/detach aggregates on the full-data
    cand1/cand5 rank-0 maps at T_now; only cells with >= k distinct actors."""
    k = rules["k_min"]
    tn = rules.cut["T_now"]
    if not have_maps:
        return PRELUDE + f"{out(o + '/O5_prod.parquet')} SELECT '' AS map, 0 AS c_actor, 0 AS c_target, '' AS what, toUInt64(0) AS n_actors WHERE 0;"
    return PRELUDE + settings_sql(rules) + f"""
CREATE TABLE m1 ENGINE = Join(ANY, LEFT, pid) AS SELECT pid, comm + 1 AS c FROM file('{side}/comm_cand1.tsv', TSV, 'pid UInt32, comm Int32');
CREATE TABLE m5 ENGINE = Join(ANY, LEFT, pid) AS SELECT pid, comm + 1 AS c FROM file('{side}/comm_cand5.tsv', TSV, 'pid UInt32, comm Int32');
CREATE TABLE uc ENGINE = Join(ANY, LEFT, u) AS SELECT u, comm1, comm5 FROM file('{us}', Parquet, '{S['ustats_m']}') WHERE cidx = 4;
CREATE OR REPLACE FUNCTION c1 AS (x) -> if(joinGet('m1', 'c', x) > 0, joinGet('m1', 'c', x) - 1, joinGet('uc', 'comm1', x));
CREATE OR REPLACE FUNCTION c5 AS (x) -> if(joinGet('m5', 'c', x) > 0, joinGet('m5', 'c', x) - 1, joinGet('uc', 'comm5', x));
CREATE TABLE ev ENGINE = MergeTree ORDER BY tuple() AS
SELECT u AS actor, p AS target, 'block' AS what FROM {src(p1 + '/i*/blk_own.parquet', 'blk')} WHERE t <= {tn}
UNION ALL SELECT p AS actor, c_a AS target, if(kind = 1, 'hide', 'detach') AS what FROM {src(gates + '/j*.parquet', 'gates')} WHERE kind IN (1, 2) AND c_a != {NONE} AND c_t <= {tn};
{out(o + '/O5_prod.parquet')}
SELECT * FROM (
  SELECT 'cand1' AS map, c1(actor) AS c_actor, c1(target) AS c_target, what, uniqExact(actor) AS n_actors FROM ev GROUP BY c_actor, c_target, what
  UNION ALL
  SELECT 'cand5', c5(actor), c5(target), what, uniqExact(actor) FROM ev GROUP BY 2, 3, 4
) WHERE n_actors >= {k} AND c_actor >= 0 AND c_target >= 0;
"""
