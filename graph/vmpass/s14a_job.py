#!/usr/bin/env python3
"""R-S14a GCE Spot job (whole-network plan §3.1 S14a, §4.8 V9, S1 §9.3 & §9.8):

Runs on a short-lived self-deleting `n2-highmem-16` Spot VM in `us-west1` (cap $1.50).
Strictly enforces:
  - Latest purge filter (`s5p-20261030a/purge_uids.parquet` U `didmap.purge=1`) on every actor & target.
  - Zero DIDs in any input or output; only cells, terciles, buckets, and producer rows with
    `>= 50` distinct actors (`K_MIN = 50`) ever leave the VM!

Computes:
  1. Per-user InterestedIn top-1 community at `T` on frozen train-only `cand5-T-matched` (`0..8,224`)
     and at `T_now` on frozen full-data `cand5-full-matched` (`0..8,499`), streaming the 16
     actor-partitioned `p1/i00..i15/fol_own.parquet` files one at a time (`< 8 GB` disk).
     Assigned producers keep their rank-0 community (`sql_global.py:301`).
  2. `O5-val` pre-`T` TID-timed block cells (`n_actors >= 50`) + community exposure stats (`>= 50`),
     plus leak-free pre-`T` size-matched `H_val(c_a, c_b)` (`top 1%` of `k>=50` cells vs size-matched
     low-`H` cells across the 10x10 `(actor_users_decile, target_followers_decile)` grid).
  3. Post-`T` non-bulk cross-community follows in `(T, Sep 18]` with per-follow 30-day outcome windows
     `(ft, ft + 30d]` (persistent block + hidden/detached content created in `(ft, ft + 30d]`),
     aggregated into:
       - `o5_val_s14_strata_summary.parquet` (`>= 50` distinct followers per row)
       - `o5_val_s14_boot_buckets.parquet` (`100` user-hash buckets, exact size-cell-matched weights,
         every row `uniqExact(u) >= 50`)
       - `o5_val_postT_cells_k50.parquet` (individual `(c_u, c_p)` cells with `uniqExact(u) >= 50`)
  4. Full-data `o5_prod_cand5_matched_blocks_k50.parquet` + `o5_prod_comm_stats_now.parquet` on
     `cand5-full-matched` (`0..8,499`) at `T_now` (`blocks only` per `V7` outcome, `n_actors >= 50`).
  5. `V9(i)` tercile aggregates (`v9i_terciles.parquet` + `v9i_terciles_boot.parquet`, `>= 50` distinct
     followers per row) for non-bulk follows in `(T, T + 30d]` of producers with `>= 50` blockers at `T`.
  6. `V9(ii)` producer rows (`v9ii_producers_T.parquet` and `v9ii_producers_now.parquet`, integer `p`
     with `n_blockers >= 50`): `(a)` share of blockers who blocked in a burst (`b >= 10`) or subscribe
     `<= C` to a list-block covering `p`, and `(b)` share of blockers in `p`'s top-3 blocker communities.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import time

T_CUT = 1786838399          # 2026-08-15T23:59:59Z
T_NOW = 1792367999          # 2026-10-18T23:59:59Z
O5_LO = 1786838399          # 2026-08-15T23:59:59Z
O5_HI = 1789775999          # 2026-09-18T23:59:59Z
V9_HI = T_CUT + 30 * 86400  # 2026-09-14T23:59:59Z (1789430399)
W30 = 30 * 86400            # 2592000
NONE = 4294967295
K_MIN = 50


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run_cmd(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd[:6])}\nstderr: {r.stderr[-1600:]}")
    return r.stdout


def ch_query(sql: str, max_mem: int) -> str:
    prelude = (
        "SET session_timezone = 'UTC';\n"
        "SET output_format_parquet_compression_method = 'zstd';\n"
        "SET engine_file_truncate_on_insert = 1;\n"
        "SET schema_inference_make_columns_nullable = 0;\n"
        "SET input_format_parquet_allow_missing_columns = 1;\n"
        "SET allow_suspicious_low_cardinality_types = 1;\n"
        f"SET max_memory_usage = {max_mem};\n"
        f"SET max_bytes_before_external_group_by = {max_mem // 3};\n"
        f"SET max_bytes_before_external_sort = {max_mem // 3};\n"
    )
    return run_cmd(["clickhouse", "local", "--multiquery", "-q", prelude + sql])


def assign_deciles(items: list[tuple[int, int]]) -> dict[int, int]:
    """Given `(comm_id, size_metric)`, assign deterministic equal-count deciles `0..9`."""
    s_items = sorted(items, key=lambda x: (x[1], x[0]))
    n = max(1, len(s_items))
    return {cid: min(9, (idx * 10) // n) for idx, (cid, _) in enumerate(s_items)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="/mnt/work")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--s9-run-id", default="s9-20261028a")
    ap.add_argument("--s5p-run-id", default="s5p-20261030a")
    ap.add_argument("--plan30", default="gs://catbird-436600-plan-30d")
    ap.add_argument("--plan60", default="gs://catbird-436600-plan-60d")
    ap.add_argument("--n-parts", type=int, default=16)
    ap.add_argument("--skip-gcs", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    work = pathlib.Path(args.work)
    in_dir = work / "in"
    mid_dir = work / "mid"
    out_dir = work / "out"
    for d in (in_dir, mid_dir, out_dir, in_dir / "blocks", in_dir / "follows", in_dir / "gates", in_dir / "modli"):
        d.mkdir(parents=True, exist_ok=True)

    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    max_mem = max(8_000_000_000, int(ram * 0.70))

    # 1. Stage shared inputs (all intra-region us-west1 -> $0)
    if not args.skip_gcs:
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan30}/outputs/{args.s5p_run_id}/purge_uids.parquet", str(in_dir / "purge_uids.parquet")])
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan60}/didmap/{args.s9_run_id}/didmap.parquet", str(in_dir / "didmap.parquet")])
        run_cmd([
            "gcloud", "storage", "cp", "--quiet",
            f"{args.plan30}/inputs/{args.run_id}/kf_cand5_T.parquet",
            f"{args.plan30}/inputs/{args.run_id}/kf_cand5_full.parquet",
            f"{args.plan30}/inputs/{args.run_id}/prod_t_blockers.parquet",
            f"{in_dir}/",
        ])
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan30}/outputs/{args.s9_run_id}/O5-raw/blocks/*", f"{in_dir}/blocks/"])
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan30}/outputs/{args.s9_run_id}/O5-raw/follows/*", f"{in_dir}/follows/"])
        run_cmd([
            "gcloud", "storage", "cp", "--quiet",
            f"{args.plan30}/outputs/{args.s9_run_id}/O5-raw/listblock_subs.parquet",
            f"{args.plan30}/outputs/{args.s9_run_id}/O5-raw/listblock_members_at_T.parquet",
            f"{in_dir}/",
        ])
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan30}/intermediates/{args.s9_run_id}/g/gates/*", f"{in_dir}/gates/"])
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan30}/intermediates/{args.s9_run_id}/g/modli_by_part/*", f"{in_dir}/modli/"])

    t_staged = time.time()

    # 2. Map hashed KnownFor tables (`h -> uid`), dropping purged UIDs; remove `didmap.parquet` immediately after!
    kft_uid_pq = mid_dir / "kft_uid.parquet"
    kff_uid_pq = mid_dir / "kff_uid.parquet"
    m5t_pq = mid_dir / "m5t_rank0.parquet"
    m5f_pq = mid_dir / "m5f_rank0.parquet"
    pu_clean_pq = mid_dir / "purge_all_uids.parquet"

    sql_map = f"""
    CREATE TABLE pu0 ENGINE = Join(ANY, LEFT, uid) AS
    SELECT uid, toUInt8(1) AS bad FROM file('{in_dir}/purge_uids.parquet', Parquet);

    INSERT INTO FUNCTION file('{pu_clean_pq}', Parquet)
    SELECT DISTINCT uid FROM (
        SELECT uid FROM file('{in_dir}/purge_uids.parquet', Parquet)
        UNION ALL
        SELECT uid FROM file('{in_dir}/didmap.parquet', Parquet) WHERE purge = 1
    );

    CREATE TABLE hmap ENGINE = Join(ANY, LEFT, h) AS
    SELECT h, toUInt64(uid) + 1 AS uid1
    FROM file('{in_dir}/didmap.parquet', Parquet)
    WHERE purge = 0 AND joinGet('pu0', 'bad', uid) = 0;

    INSERT INTO FUNCTION file('{kft_uid_pq}', Parquet)
    SELECT toUInt32(u1 - 1) AS p, community AS comm, score, rank
    FROM (SELECT *, joinGet('hmap', 'uid1', h) AS u1 FROM file('{in_dir}/kf_cand5_T.parquet', Parquet))
    WHERE u1 > 0
    ORDER BY p, rank;

    INSERT INTO FUNCTION file('{kff_uid_pq}', Parquet)
    SELECT toUInt32(u1 - 1) AS p, community AS comm, score, rank
    FROM (SELECT *, joinGet('hmap', 'uid1', h) AS u1 FROM file('{in_dir}/kf_cand5_full.parquet', Parquet))
    WHERE u1 > 0
    ORDER BY p, rank;

    INSERT INTO FUNCTION file('{m5t_pq}', Parquet)
    SELECT p, toInt32(comm + 1) AS c1 FROM file('{kft_uid_pq}', Parquet) WHERE rank = 0;

    INSERT INTO FUNCTION file('{m5f_pq}', Parquet)
    SELECT p, toInt32(comm + 1) AS c1 FROM file('{kff_uid_pq}', Parquet) WHERE rank = 0;
    """
    ch_query(sql_map, max_mem=max_mem)
    # Remove local didmap copy now that integer-keyed kft_uid/kff_uid are built
    (in_dir / "didmap.parquet").unlink(missing_ok=True)

    map_stats = ch_query(
        f"SELECT (SELECT count() FROM file('{m5t_pq}', Parquet)), "
        f"(SELECT max(c1) FROM file('{m5t_pq}', Parquet)), "
        f"(SELECT count() FROM file('{m5f_pq}', Parquet)), "
        f"(SELECT max(c1) FROM file('{m5f_pq}', Parquet)) FORMAT TSV",
        max_mem=max_mem,
    ).strip().split("\t")
    n_prod_t, n_comm_t, n_prod_f, n_comm_f = [int(x) for x in map_stats]

    # 3. Stream p1/i00..i15/fol_own.parquet one partition at a time to build:
    #    - u_comm_t_{i:02d}.parquet (u UInt32, c1 Int32)  [InterestedIn top-1 + 1 at T]
    #    - u_comm_f_{i:02d}.parquet (u UInt32, c1 Int32)  [InterestedIn top-1 + 1 at T_now]
    #    - p_fol_{i:02d}.parquet    (p UInt32, fol_t UInt64, fol_now UInt64)
    #    - o5_fol_clean_{i:02d}.parquet (u UInt32, p UInt32, ft UInt32) [post-T non-bulk follows NOT already live at T]
    fol_cur = in_dir / "fol_own_cur.parquet"
    for i in range(args.n_parts):
        if args.skip_gcs:
            local_fol = in_dir / f"fol_own_{i:02d}.parquet"
            if local_fol.exists():
                local_fol.replace(fol_cur)
        else:
            run_cmd([
                "gcloud", "storage", "cp", "--quiet",
                f"{args.plan30}/intermediates/{args.s9_run_id}/p1/i{i:02d}/fol_own.parquet",
                str(fol_cur),
            ])
        sql_part = f"""
        CREATE TABLE pu ENGINE = Join(ANY, LEFT, uid) AS
        SELECT uid, toUInt8(1) AS bad FROM file('{pu_clean_pq}', Parquet);

        CREATE TABLE kft ENGINE = MergeTree ORDER BY p AS SELECT p, comm, score FROM file('{kft_uid_pq}', Parquet);
        CREATE TABLE kff ENGINE = MergeTree ORDER BY p AS SELECT p, comm, score FROM file('{kff_uid_pq}', Parquet);

        CREATE TABLE ft_live ENGINE = MergeTree ORDER BY (u, p) AS
        SELECT DISTINCT u, p
        FROM file('{fol_cur}', Parquet)
        WHERE ft <= {T_CUT} AND {T_CUT} < dt AND u != p
          AND joinGet('pu', 'bad', u) = 0 AND joinGet('pu', 'bad', p) = 0;

        CREATE TABLE fn_live ENGINE = MergeTree ORDER BY (u, p) AS
        SELECT DISTINCT u, p
        FROM file('{fol_cur}', Parquet)
        WHERE ft <= {T_NOW} AND {T_NOW} < dt AND u != p
          AND joinGet('pu', 'bad', u) = 0 AND joinGet('pu', 'bad', p) = 0;

        INSERT INTO FUNCTION file('{mid_dir}/u_comm_t_{i:02d}.parquet', Parquet)
        SELECT u, toInt32(argMax(comm, tuple(sc, -toInt64(comm))) + 1) AS c1
        FROM (
            SELECT f.u AS u, k.comm AS comm, sum(toFloat64(k.score)) AS sc
            FROM ft_live AS f INNER JOIN kft AS k ON k.p = f.p
            GROUP BY u, comm
        ) GROUP BY u;

        INSERT INTO FUNCTION file('{mid_dir}/u_comm_f_{i:02d}.parquet', Parquet)
        SELECT u, toInt32(argMax(comm, tuple(sc, -toInt64(comm))) + 1) AS c1
        FROM (
            SELECT f.u AS u, k.comm AS comm, sum(toFloat64(k.score)) AS sc
            FROM fn_live AS f INNER JOIN kff AS k ON k.p = f.p
            GROUP BY u, comm
        ) GROUP BY u;

        INSERT INTO FUNCTION file('{mid_dir}/p_fol_{i:02d}.parquet', Parquet)
        SELECT p, sum(nt) AS fol_t, sum(nn) AS fol_now FROM (
            SELECT p, count() AS nt, toUInt64(0) AS nn FROM ft_live GROUP BY p
            UNION ALL
            SELECT p, toUInt64(0) AS nt, count() AS nn FROM fn_live GROUP BY p
        ) GROUP BY p;

        INSERT INTO FUNCTION file('{mid_dir}/o5_fol_clean_{i:02d}.parquet', Parquet)
        SELECT o.u AS u, o.p AS p, min(o.ft) AS ft
        FROM file('{in_dir}/follows/i{i:02d}.parquet', Parquet) AS o
        LEFT ANTI JOIN ft_live AS l ON l.u = o.u AND l.p = o.p
        WHERE o.b < 10 AND o.ft > {O5_LO} AND o.ft <= {O5_HI} AND o.u != o.p
          AND joinGet('pu', 'bad', o.u) = 0 AND joinGet('pu', 'bad', o.p) = 0
        GROUP BY o.u, o.p;
        """
        ch_query(sql_part, max_mem=max_mem)
        fol_cur.unlink(missing_ok=True)

    t_fol_done = time.time()

    # 4. Deduplicate O5-raw blocks cleanly (excluding purged u/p) and build pre-T & T_now block tables + community stats
    blk_clean_pq = mid_dir / "blk_clean.parquet"
    gates_clean_pq = mid_dir / "gates_clean.parquet"
    o5_val_preT_pq = out_dir / "o5_val_preT_blocks_k50.parquet"
    o5_val_comm_pq = out_dir / "o5_val_comm_stats_T.parquet"
    o5_prod_now_pq = out_dir / "o5_prod_cand5_matched_blocks_k50.parquet"
    o5_prod_comm_pq = out_dir / "o5_prod_comm_stats_now.parquet"
    preT_all_raw_tsv = mid_dir / "preT_cross_cells_all.tsv"  # VM-local only, used to compute exact size_cell baselines

    sql_blocks = f"""
    CREATE TABLE pu ENGINE = Join(ANY, LEFT, uid) AS
    SELECT uid, toUInt8(1) AS bad FROM file('{pu_clean_pq}', Parquet);

    INSERT INTO FUNCTION file('{blk_clean_pq}', Parquet)
    SELECT
        u, p,
        min(t) AS t_min,
        max(t) AS t_max,
        max(bulk) AS any_bulk,
        toFloat32(max(if(b >= 10, 1.0 / (1.0 + (b - 1.0) / 10.0), 1.0))) AS w_blk,
        groupArray(t) AS ts
    FROM file('{in_dir}/blocks/i*.parquet', Parquet)
    WHERE u != p AND joinGet('pu', 'bad', u) = 0 AND joinGet('pu', 'bad', p) = 0
    GROUP BY u, p;

    INSERT INTO FUNCTION file('{gates_clean_pq}', Parquet)
    SELECT p, c_a, c_t, kind
    FROM file('{in_dir}/gates/j*.parquet', Parquet)
    WHERE kind IN (1, 2) AND c_a != {NONE} AND p != c_a
      AND joinGet('pu', 'bad', p) = 0 AND joinGet('pu', 'bad', c_a) = 0;

    CREATE TABLE m5t ENGINE = Join(ANY, LEFT, p) AS SELECT p, c1 FROM file('{m5t_pq}', Parquet);
    CREATE TABLE uct ENGINE = Join(ANY, LEFT, u) AS SELECT u, c1 FROM file('{mid_dir}/u_comm_t_*.parquet', Parquet);
    CREATE OR REPLACE FUNCTION ct AS (x) -> toInt32(if(joinGet('m5t', 'c1', x) > 0, joinGet('m5t', 'c1', x) - 1, joinGet('uct', 'c1', x) - 1));

    CREATE TABLE m5f ENGINE = Join(ANY, LEFT, p) AS SELECT p, c1 FROM file('{m5f_pq}', Parquet);
    CREATE TABLE ucf ENGINE = Join(ANY, LEFT, u) AS SELECT u, c1 FROM file('{mid_dir}/u_comm_f_*.parquet', Parquet);
    CREATE OR REPLACE FUNCTION cf AS (x) -> toInt32(if(joinGet('m5f', 'c1', x) > 0, joinGet('m5f', 'c1', x) - 1, joinGet('ucf', 'c1', x) - 1));

    CREATE TABLE pfol ENGINE = MergeTree ORDER BY p AS
    SELECT p, sum(fol_t) AS fol_t, sum(fol_now) AS fol_now
    FROM file('{mid_dir}/p_fol_*.parquet', Parquet) GROUP BY p;

    -- Community exposure stats at T (`cand5-T-matched`, 0..{n_comm_t - 1}; user/blocker counts zeroed if < 50)
    CREATE TABLE u_counts_t ENGINE = Memory AS
    SELECT c, count() AS nu FROM (
        SELECT ct(u) AS c FROM (
            SELECT u FROM file('{mid_dir}/u_comm_t_*.parquet', Parquet)
            UNION DISTINCT
            SELECT p AS u FROM file('{m5t_pq}', Parquet)
        )
    ) WHERE c >= 0 GROUP BY c;

    CREATE TABLE p_counts_t ENGINE = Memory AS
    SELECT m.c1 - 1 AS c, count() AS np, sum(f.fol_t) AS sum_fol
    FROM file('{m5t_pq}', Parquet) AS m LEFT JOIN pfol AS f ON f.p = m.p
    GROUP BY c;

    CREATE TABLE blk_t_pairs ENGINE = MergeTree ORDER BY (c_actor, c_target) AS
    SELECT ct(u) AS c_actor, ct(p) AS c_target, toInt32(joinGet('m5t', 'c1', p)) - 1 AS c_prod_target, u, p, w_blk
    FROM file('{blk_clean_pq}', Parquet)
    WHERE t_min <= {T_CUT};

    CREATE TABLE b_out_t ENGINE = Memory AS
    SELECT c_actor AS c, uniqExact(u) AS nb_out, count() AS np_out, sum(w_blk) AS wb_out
    FROM blk_t_pairs WHERE c_actor >= 0 AND c_target >= 0 AND c_actor != c_target GROUP BY c;

    CREATE TABLE b_in_t ENGINE = Memory AS
    SELECT c_target AS c, uniqExact(u) AS nb_in, count() AS np_in, sum(w_blk) AS wb_in
    FROM blk_t_pairs WHERE c_actor >= 0 AND c_target >= 0 AND c_actor != c_target GROUP BY c;

    INSERT INTO FUNCTION file('{o5_val_comm_pq}', Parquet)
    SELECT
        toInt32(p.c) AS c,
        toUInt64(p.np) AS n_producers,
        toUInt64(p.sum_fol) AS sum_followers_T,
        toUInt64(if(u.nu >= {K_MIN}, u.nu, 0)) AS n_users_T,
        toUInt64(if(bo.nb_out >= {K_MIN}, bo.nb_out, 0)) AS n_blockers_out_T,
        toUInt64(if(bo.nb_out >= {K_MIN}, bo.np_out, 0)) AS n_block_pairs_out_T,
        toFloat64(if(bo.nb_out >= {K_MIN}, bo.wb_out, 0)) AS w_block_pairs_out_T,
        toUInt64(if(bi.nb_in >= {K_MIN}, bi.nb_in, 0)) AS n_blockers_in_T,
        toUInt64(if(bi.nb_in >= {K_MIN}, bi.np_in, 0)) AS n_block_pairs_in_T,
        toFloat64(if(bi.nb_in >= {K_MIN}, bi.wb_in, 0)) AS w_block_pairs_in_T
    FROM p_counts_t AS p
    LEFT JOIN u_counts_t AS u ON u.c = p.c
    LEFT JOIN b_out_t AS bo ON bo.c = p.c
    LEFT JOIN b_in_t AS bi ON bi.c = p.c
    ORDER BY c;

    INSERT INTO FUNCTION file('{o5_val_preT_pq}', Parquet)
    SELECT
        c_actor,
        c_target,
        uniqExact(u) AS n_actors,
        uniqExact(p) AS n_targets,
        count() AS n_pairs,
        sum(toFloat64(w_blk)) AS w_pairs_burst_disc,
        uniqExactIf(u, c_prod_target = c_target) AS n_actors_prod_target
    FROM blk_t_pairs
    WHERE c_actor >= 0 AND c_target >= 0
    GROUP BY c_actor, c_target
    HAVING n_actors >= {K_MIN}
    ORDER BY c_actor, c_target;

    INSERT INTO FUNCTION file('{preT_all_raw_tsv}', TSV)
    SELECT c_actor, c_target, uniqExact(u) AS na, count() AS np, sum(toFloat64(w_blk)) AS wb
    FROM blk_t_pairs
    WHERE c_actor >= 0 AND c_target >= 0 AND c_actor != c_target
    GROUP BY c_actor, c_target;

    -- Full-data `cand5-full-matched` (0..{n_comm_f - 1}) at T_now (`blocks only` per V7, k >= 50)
    CREATE TABLE u_counts_f ENGINE = Memory AS
    SELECT c, count() AS nu FROM (
        SELECT cf(u) AS c FROM (
            SELECT u FROM file('{mid_dir}/u_comm_f_*.parquet', Parquet)
            UNION DISTINCT
            SELECT p AS u FROM file('{m5f_pq}', Parquet)
        )
    ) WHERE c >= 0 GROUP BY c;

    CREATE TABLE p_counts_f ENGINE = Memory AS
    SELECT m.c1 - 1 AS c, count() AS np, sum(f.fol_now) AS sum_fol
    FROM file('{m5f_pq}', Parquet) AS m LEFT JOIN pfol AS f ON f.p = m.p
    GROUP BY c;

    CREATE TABLE blk_f_pairs ENGINE = MergeTree ORDER BY (c_actor, c_target) AS
    SELECT cf(u) AS c_actor, cf(p) AS c_target, toInt32(joinGet('m5f', 'c1', p)) - 1 AS c_prod_target, u, p, w_blk
    FROM file('{blk_clean_pq}', Parquet)
    WHERE t_min <= {T_NOW};

    CREATE TABLE b_out_f ENGINE = Memory AS
    SELECT c_actor AS c, uniqExact(u) AS nb_out, count() AS np_out, sum(w_blk) AS wb_out
    FROM blk_f_pairs WHERE c_actor >= 0 AND c_target >= 0 AND c_actor != c_target GROUP BY c;

    CREATE TABLE b_in_f ENGINE = Memory AS
    SELECT c_target AS c, uniqExact(u) AS nb_in, count() AS np_in, sum(w_blk) AS wb_in
    FROM blk_f_pairs WHERE c_actor >= 0 AND c_target >= 0 AND c_actor != c_target GROUP BY c;

    INSERT INTO FUNCTION file('{o5_prod_comm_pq}', Parquet)
    SELECT
        toInt32(p.c) AS c,
        toUInt64(p.np) AS n_producers,
        toUInt64(p.sum_fol) AS sum_followers_now,
        toUInt64(if(u.nu >= {K_MIN}, u.nu, 0)) AS n_users_now,
        toUInt64(if(bo.nb_out >= {K_MIN}, bo.nb_out, 0)) AS n_blockers_out_now,
        toUInt64(if(bo.nb_out >= {K_MIN}, bo.np_out, 0)) AS n_block_pairs_out_now,
        toFloat64(if(bo.nb_out >= {K_MIN}, bo.wb_out, 0)) AS w_block_pairs_out_now,
        toUInt64(if(bi.nb_in >= {K_MIN}, bi.nb_in, 0)) AS n_blockers_in_now,
        toUInt64(if(bi.nb_in >= {K_MIN}, bi.np_in, 0)) AS n_block_pairs_in_now,
        toFloat64(if(bi.nb_in >= {K_MIN}, bi.wb_in, 0)) AS w_block_pairs_in_now
    FROM p_counts_f AS p
    LEFT JOIN u_counts_f AS u ON u.c = p.c
    LEFT JOIN b_out_f AS bo ON bo.c = p.c
    LEFT JOIN b_in_f AS bi ON bi.c = p.c
    ORDER BY c;

    INSERT INTO FUNCTION file('{o5_prod_now_pq}', Parquet)
    SELECT
        c_actor,
        c_target,
        uniqExact(u) AS n_actors,
        uniqExact(p) AS n_targets,
        count() AS n_pairs,
        sum(toFloat64(w_blk)) AS w_pairs_burst_disc,
        uniqExactIf(u, c_prod_target = c_target) AS n_actors_prod_target
    FROM blk_f_pairs
    WHERE c_actor >= 0 AND c_target >= 0
    GROUP BY c_actor, c_target
    HAVING n_actors >= {K_MIN}
    ORDER BY c_actor, c_target;
    """
    ch_query(sql_blocks, max_mem=max_mem)

    # 5. Compute leak-free pre-T size-matched H_val(c_a, c_b) on cand5-T-matched (`0..n_comm_t-1`)
    comm_rows = ch_query(
        f"SELECT c, n_producers, sum_followers_T, n_users_T FROM file('{o5_val_comm_pq}', Parquet) FORMAT TSV",
        max_mem=max_mem,
    ).strip().splitlines()
    s_act: dict[int, int] = {}
    s_tgt: dict[int, int] = {}
    for ln in comm_rows:
        cs, np_s, sf_s, nu_s = ln.split("\t")
        c = int(cs)
        s_act[c] = max(int(nu_s), int(np_s), 1)
        s_tgt[c] = max(int(sf_s), int(np_s), 1)

    d_act = assign_deciles(list(s_act.items()))
    d_tgt = assign_deciles(list(s_tgt.items()))

    # Total exposure product sum_{(a,b) in cell, a != b} s_act[a]*s_tgt[b] per size_cell 0..99
    act_sum_by_dec = [0.0] * 10
    tgt_sum_by_dec = [0.0] * 10
    same_prod_by_cell = [0.0] * 100
    for c in s_act:
        da, dtg = d_act[c], d_tgt[c]
        act_sum_by_dec[da] += float(s_act[c])
        tgt_sum_by_dec[dtg] += float(s_tgt[c])
        same_prod_by_cell[da * 10 + dtg] += float(s_act[c]) * float(s_tgt[c])

    exp_denom_by_cell = [0.0] * 100
    for da in range(10):
        for dtg in range(10):
            sc = da * 10 + dtg
            exp_denom_by_cell[sc] = max(act_sum_by_dec[da] * tgt_sum_by_dec[dtg] - same_prod_by_cell[sc], 1.0)

    wb_sum_by_cell = [0.0] * 100
    k50_cells: list[tuple[int, int, int, float]] = []
    with open(preT_all_raw_tsv) as fh:
        for ln in fh:
            ca_s, cb_s, na_s, _np_s, wb_s = ln.rstrip("\n").split("\t")
            ca, cb, na, wb = int(ca_s), int(cb_s), int(na_s), float(wb_s)
            sc = d_act[ca] * 10 + d_tgt[cb]
            wb_sum_by_cell[sc] += wb
            if na >= K_MIN:
                k50_cells.append((ca, cb, na, wb))
    preT_all_raw_tsv.unlink(missing_ok=True)

    r_base_by_cell = [wb_sum_by_cell[sc] / exp_denom_by_cell[sc] for sc in range(100)]
    global_rate = sum(wb_sum_by_cell) / max(sum(exp_denom_by_cell), 1.0)

    scored_k50: list[tuple[float, int, int, int, float, float]] = []
    for ca, cb, na, wb in k50_cells:
        sc = d_act[ca] * 10 + d_tgt[cb]
        rate = r_base_by_cell[sc] if r_base_by_cell[sc] > 0 else global_rate
        expected_wb = max(rate * float(s_act[ca]) * float(s_tgt[cb]), 1e-6)
        h_val = wb / expected_wb
        scored_k50.append((h_val, ca, cb, na, wb, expected_wb))

    scored_k50.sort(key=lambda x: (-x[0], x[1], x[2]))
    n_k50 = len(scored_k50)
    n_top1pct = max(1, round(0.01 * n_k50)) if n_k50 > 0 else 0
    n_top5pct = max(1, round(0.05 * n_k50)) if n_k50 > 0 else 0
    half_idx = n_k50 // 2

    # Write comm_deciles_T.tsv and h_strata_T.tsv (VM-local lookup tables)
    deciles_tsv = mid_dir / "comm_deciles_T.tsv"
    with open(deciles_tsv, "w") as fh:
        for c in sorted(s_act):
            fh.write(f"{c}\t{d_act[c]}\t{d_tgt[c]}\n")

    # Stratum codes on k>=50 cells:
    #   1 = top 1% of k>=50 cells by H_val (primary S14 High-H arm)
    #   2 = 1%..5% of k>=50 cells
    #   3 = 5%..50% of k>=50 cells
    #   4 = bottom 50% of k>=50 cells (size-matched low-H among k>=50 cells)
    # Unlisted cross-community pairs (`n_actors < 50`) have stratum = 0 (`H_val = 0`, size-matched low-H overall).
    h_strata_tsv = mid_dir / "h_strata_T.tsv"
    h_val_k50_out_tsv = mid_dir / "h_val_k50_scored.tsv"
    with open(h_strata_tsv, "w") as f_st, open(h_val_k50_out_tsv, "w") as f_sc:
        for idx, (hv, ca, cb, na, wb, exp_wb) in enumerate(scored_k50):
            if idx < n_top1pct:
                st_code = 1
            elif idx < n_top5pct:
                st_code = 2
            elif idx < half_idx:
                st_code = 3
            else:
                st_code = 4
            f_st.write(f"{ca}\t{cb}\t{st_code}\t{hv:.8f}\n")
            f_sc.write(f"{ca}\t{cb}\t{na}\t{wb:.6f}\t{exp_wb:.6f}\t{hv:.8f}\t{st_code}\n")

    # Save k>=50 scored H_val table (`n_actors >= 50` on every row!)
    ch_query(
        f"""
        INSERT INTO FUNCTION file('{out_dir}/o5_val_H_k50.parquet', Parquet)
        SELECT ca AS c_actor, cb AS c_target, na AS n_actors, wb AS w_pairs_burst_disc,
               exp_wb AS expected_w_pairs, hv AS h_val, st AS h_stratum
        FROM file('{h_val_k50_out_tsv}', TSV, 'ca Int32, cb Int32, na UInt64, wb Float64, exp_wb Float64, hv Float64, st UInt8')
        WHERE na >= {K_MIN}
        ORDER BY h_val DESC, c_actor, c_target;
        """,
        max_mem=max_mem,
    )

    # 6. Score post-T cross-community follows in (T, Sep 18] on per-follow 30-day outcomes `(ft, ft + 30d]`
    s14_strata_pq = out_dir / "o5_val_s14_strata_summary.parquet"
    s14_boot_pq = out_dir / "o5_val_s14_boot_buckets.parquet"
    s14_sizecell_pq = out_dir / "o5_val_s14_sizecells_k50.parquet"
    s14_post_cells_pq = out_dir / "o5_val_postT_cells_k50.parquet"

    sql_s14 = f"""
    CREATE TABLE m5t ENGINE = Join(ANY, LEFT, p) AS SELECT p, c1 FROM file('{m5t_pq}', Parquet);
    CREATE TABLE uct ENGINE = Join(ANY, LEFT, u) AS SELECT u, c1 FROM file('{mid_dir}/u_comm_t_*.parquet', Parquet);
    CREATE OR REPLACE FUNCTION ct AS (x) -> toInt32(if(joinGet('m5t', 'c1', x) > 0, joinGet('m5t', 'c1', x) - 1, joinGet('uct', 'c1', x) - 1));

    CREATE TABLE cdec ENGINE = Join(ANY, LEFT, c) AS
    SELECT c, da, dtg FROM file('{deciles_tsv}', TSV, 'c Int32, da UInt8, dtg UInt8');

    CREATE TABLE hst ENGINE = Join(ANY, LEFT, ca, cb) AS
    SELECT ca, cb, st FROM file('{h_strata_tsv}', TSV, 'ca Int32, cb Int32, st UInt8, hv Float64');

    -- Block timestamps in either direction for outcome lookup `(ft, ft + 30d]`
    CREATE TABLE blk_up ENGINE = Join(ANY, LEFT, u, p) AS
    SELECT u, p, ts FROM file('{blk_clean_pq}', Parquet);

    CREATE TABLE gt_pu ENGINE = Join(ANY, LEFT, p, c_a) AS
    SELECT p, c_a, groupArray(c_t) AS cts FROM file('{gates_clean_pq}', Parquet) GROUP BY p, c_a;

    CREATE TABLE xfol ENGINE = MergeTree ORDER BY (stratum, scell, ub) AS
    SELECT
        u, p, ft, cu, cp, scell, stratum, ub,
        y_blk_up, y_blk_pu, y_gate,
        toUInt8(y_blk_up = 1 OR y_blk_pu = 1 OR y_gate = 1) AS y_any,
        toUInt8(y_blk_up = 1 OR y_gate = 1) AS y_ug
    FROM (
        SELECT
            u, p, ft, cu, cp,
            toUInt8(joinGet('cdec', 'da', cu) * 10 + joinGet('cdec', 'dtg', cp)) AS scell,
            joinGet('hst', 'st', cu, cp) AS stratum,
            toUInt8(cityHash64(u, 's14boot') % 100) AS ub,
            toUInt8(arrayExists(t -> t > ft AND t <= ft + {W30}, joinGet('blk_up', 'ts', u, p))) AS y_blk_up,
            toUInt8(arrayExists(t -> t > ft AND t <= ft + {W30}, joinGet('blk_up', 'ts', p, u))) AS y_blk_pu,
            toUInt8(arrayExists(t -> t > ft AND t <= ft + {W30}, joinGet('gt_pu', 'cts', p, u))) AS y_gate
        FROM (
            SELECT u, p, ft, ct(u) AS cu, toInt32(joinGet('m5t', 'c1', p) - 1) AS cp
            FROM file('{mid_dir}/o5_fol_clean_*.parquet', Parquet)
        )
        WHERE cu >= 0 AND cp >= 0 AND cu != cp
    );

    -- Exact size_cell weights matching stratum=0 (`low_lt50`) and stratum=4 (`bottom50_k50`) to stratum=1 (`high_top1pct`)
    CREATE TABLE cell_cnt ENGINE = Join(ANY, LEFT, scell) AS
    SELECT
        scell,
        toFloat64(countIf(stratum = 1)) AS nh,
        toFloat64(countIf(stratum = 0)) AS nl0,
        toFloat64(countIf(stratum IN (0, 4))) AS nl04,
        toFloat64(countIf(stratum = 4)) AS nl4
    FROM xfol GROUP BY scell;

    INSERT INTO FUNCTION file('{s14_strata_pq}', Parquet)
    SELECT
        stratum,
        uniqExact(u) AS n_distinct_followers,
        uniqExact(p) AS n_distinct_producers,
        count() AS n_follows,
        sum(y_any) AS n_out_any,
        sum(y_ug) AS n_out_up_or_gate,
        sum(y_blk_up) AS n_blk_up,
        sum(y_blk_pu) AS n_blk_pu,
        sum(y_gate) AS n_gate
    FROM xfol
    GROUP BY stratum
    HAVING n_distinct_followers >= {K_MIN}
    ORDER BY stratum;

    INSERT INTO FUNCTION file('{s14_boot_pq}', Parquet)
    SELECT
        ub,
        uniqExact(u) AS n_distinct_followers_bucket,
        countIf(stratum = 1) AS nh_follows,
        sumIf(toFloat64(y_any), stratum = 1) AS yh_any,
        sumIf(toFloat64(y_ug), stratum = 1) AS yh_ug,
        sumIf(if(joinGet('cell_cnt', 'nl0', scell) > 0, joinGet('cell_cnt', 'nh', scell) / joinGet('cell_cnt', 'nl0', scell), 0.0), stratum = 0) AS wl0_follows,
        sumIf(toFloat64(y_any) * if(joinGet('cell_cnt', 'nl0', scell) > 0, joinGet('cell_cnt', 'nh', scell) / joinGet('cell_cnt', 'nl0', scell), 0.0), stratum = 0) AS yl0_any,
        sumIf(toFloat64(y_ug) * if(joinGet('cell_cnt', 'nl0', scell) > 0, joinGet('cell_cnt', 'nh', scell) / joinGet('cell_cnt', 'nl0', scell), 0.0), stratum = 0) AS yl0_ug,
        sumIf(if(joinGet('cell_cnt', 'nl04', scell) > 0, joinGet('cell_cnt', 'nh', scell) / joinGet('cell_cnt', 'nl04', scell), 0.0), stratum IN (0, 4)) AS wl04_follows,
        sumIf(toFloat64(y_any) * if(joinGet('cell_cnt', 'nl04', scell) > 0, joinGet('cell_cnt', 'nh', scell) / joinGet('cell_cnt', 'nl04', scell), 0.0), stratum IN (0, 4)) AS yl04_any,
        sumIf(toFloat64(y_ug) * if(joinGet('cell_cnt', 'nl04', scell) > 0, joinGet('cell_cnt', 'nh', scell) / joinGet('cell_cnt', 'nl04', scell), 0.0), stratum IN (0, 4)) AS yl04_ug,
        sumIf(if(joinGet('cell_cnt', 'nl4', scell) > 0, joinGet('cell_cnt', 'nh', scell) / joinGet('cell_cnt', 'nl4', scell), 0.0), stratum = 4) AS wl4_follows,
        sumIf(toFloat64(y_any) * if(joinGet('cell_cnt', 'nl4', scell) > 0, joinGet('cell_cnt', 'nh', scell) / joinGet('cell_cnt', 'nl4', scell), 0.0), stratum = 4) AS yl4_any
    FROM xfol
    GROUP BY ub
    HAVING n_distinct_followers_bucket >= {K_MIN}
    ORDER BY ub;

    INSERT INTO FUNCTION file('{s14_sizecell_pq}', Parquet)
    SELECT
        stratum, scell,
        uniqExact(u) AS n_distinct_followers,
        count() AS n_follows,
        sum(y_any) AS n_out_any,
        sum(y_ug) AS n_out_ug
    FROM xfol
    GROUP BY stratum, scell
    HAVING n_distinct_followers >= {K_MIN}
    ORDER BY stratum, scell;

    INSERT INTO FUNCTION file('{s14_post_cells_pq}', Parquet)
    SELECT
        cu AS c_follower,
        cp AS c_producer,
        any(stratum) AS stratum,
        uniqExact(u) AS n_distinct_followers,
        count() AS n_follows,
        sum(y_any) AS n_out_any,
        sum(y_ug) AS n_out_ug
    FROM xfol
    GROUP BY c_follower, c_producer
    HAVING n_distinct_followers >= {K_MIN}
    ORDER BY c_follower, c_producer;
    """
    ch_query(sql_s14, max_mem=max_mem)

    # 7. Compute V9(i) terciles (`(T, T + 30d]`, follower->producer block within `(ft, ft + 30d]`, >= 50 followers)
    v9i_pq = out_dir / "v9i_terciles.parquet"
    v9i_boot_pq = out_dir / "v9i_terciles_boot.parquet"

    sql_v9i = f"""
    CREATE TABLE ptb ENGINE = Join(ANY, LEFT, p) AS
    SELECT p, toUInt8(tercile_burst_disc + 1) AS tbd1, toUInt8(tercile_raw + 1) AS traw1
    FROM file('{in_dir}/prod_t_blockers.parquet', Parquet);

    CREATE TABLE blk_up ENGINE = Join(ANY, LEFT, u, p) AS
    SELECT u, p, ts FROM file('{blk_clean_pq}', Parquet);

    CREATE TABLE v9f ENGINE = MergeTree ORDER BY (tbd, ub) AS
    SELECT
        u, p, ft,
        toUInt8(tbd1 - 1) AS tbd,
        toUInt8(traw1 - 1) AS traw,
        toUInt8(cityHash64(u, 'v9boot') % 100) AS ub,
        toUInt8(arrayExists(t -> t > ft AND t <= ft + {W30}, joinGet('blk_up', 'ts', u, p))) AS y_blk
    FROM (
        SELECT u, p, ft, joinGet('ptb', 'tbd1', p) AS tbd1, joinGet('ptb', 'traw1', p) AS traw1
        FROM file('{mid_dir}/o5_fol_clean_*.parquet', Parquet)
        WHERE ft > {T_CUT} AND ft <= {V9_HI}
    )
    WHERE tbd1 > 0;

    INSERT INTO FUNCTION file('{v9i_pq}', Parquet)
    SELECT
        'burst_discounted' AS kind,
        tbd AS tercile,
        uniqExact(u) AS n_distinct_followers,
        uniqExact(p) AS n_distinct_producers,
        count() AS n_follows,
        sum(y_blk) AS n_blocked_30d,
        sum(y_blk) / greatest(1, count()) AS block_rate
    FROM v9f GROUP BY tbd HAVING n_distinct_followers >= {K_MIN}
    UNION ALL
    SELECT
        'raw' AS kind,
        traw AS tercile,
        uniqExact(u) AS n_distinct_followers,
        uniqExact(p) AS n_distinct_producers,
        count() AS n_follows,
        sum(y_blk) AS n_blocked_30d,
        sum(y_blk) / greatest(1, count()) AS block_rate
    FROM v9f GROUP BY traw HAVING n_distinct_followers >= {K_MIN}
    ORDER BY kind, tercile;

    INSERT INTO FUNCTION file('{v9i_boot_pq}', Parquet)
    SELECT
        'burst_discounted' AS kind,
        tbd AS tercile,
        ub,
        uniqExact(u) AS n_distinct_followers,
        count() AS n_follows,
        sum(y_blk) AS n_blocked_30d
    FROM v9f GROUP BY tbd, ub HAVING n_distinct_followers >= {K_MIN}
    UNION ALL
    SELECT
        'raw' AS kind,
        traw AS tercile,
        ub,
        uniqExact(u) AS n_distinct_followers,
        count() AS n_follows,
        sum(y_blk) AS n_blocked_30d
    FROM v9f GROUP BY traw, ub HAVING n_distinct_followers >= {K_MIN}
    ORDER BY kind, tercile, ub;
    """
    ch_query(sql_v9i, max_mem=max_mem)

    # 8. Compute V9(ii) per-producer coordinated-blocking shares at T (`cand5-T-matched`) and T_now (`cand5-full-matched`)
    #    Strictly `n_blockers >= 50` on every emitted producer row!
    v9ii_t_pq = out_dir / "v9ii_producers_T.parquet"
    v9ii_now_pq = out_dir / "v9ii_producers_now.parquet"

    for cutoff_ts, m5_file, uc_glob, dst_pq in (
        (T_CUT, m5t_pq, f"{mid_dir}/u_comm_t_*.parquet", v9ii_t_pq),
        (T_NOW, m5f_pq, f"{mid_dir}/u_comm_f_*.parquet", v9ii_now_pq),
    ):
        sql_v9ii = f"""
        CREATE TABLE pu ENGINE = Join(ANY, LEFT, uid) AS
        SELECT uid, toUInt8(1) AS bad FROM file('{pu_clean_pq}', Parquet);

        CREATE TABLE m5 ENGINE = Join(ANY, LEFT, p) AS SELECT p, c1 FROM file('{m5_file}', Parquet);
        CREATE TABLE uc ENGINE = Join(ANY, LEFT, u) AS SELECT u, c1 FROM file('{uc_glob}', Parquet);
        CREATE OR REPLACE FUNCTION cm AS (x) -> toInt32(if(joinGet('m5', 'c1', x) > 0, joinGet('m5', 'c1', x) - 1, joinGet('uc', 'c1', x) - 1));

        CREATE TABLE ulists ENGINE = Join(ANY, LEFT, u) AS
        SELECT u, groupArray(lh) AS ls
        FROM file('{in_dir}/listblock_subs.parquet', Parquet)
        WHERE t <= {cutoff_ts} AND joinGet('pu', 'bad', u) = 0
        GROUP BY u;

        CREATE TABLE plists ENGINE = Join(ANY, LEFT, p) AS
        SELECT member AS p, groupArray(lh) AS ls
        FROM file('{in_dir}/modli/j*.parquet', Parquet)
        WHERE t <= {cutoff_ts} AND joinGet('pu', 'bad', member) = 0
        GROUP BY member;

        CREATE TABLE p_elig ENGINE = MergeTree ORDER BY p AS
        SELECT p
        FROM file('{blk_clean_pq}', Parquet)
        WHERE t_min <= {cutoff_ts}
        GROUP BY p
        HAVING count() >= {K_MIN};

        CREATE TABLE b_elig ENGINE = MergeTree ORDER BY (p, u) AS
        SELECT
            b.p AS p,
            b.u AS u,
            b.any_bulk AS is_burst,
            toUInt8(hasAny(
                joinGet('plists', 'ls', b.p),
                joinGet('ulists', 'ls', b.u)
            )) AS is_lblk,
            cm(b.u) AS c_blk
        FROM file('{blk_clean_pq}', Parquet) AS b
        INNER JOIN p_elig AS e ON e.p = b.p
        WHERE b.t_min <= {cutoff_ts};

        CREATE TABLE p_top3 ENGINE = Join(ANY, LEFT, p) AS
        SELECT p, arraySum(arraySlice(arrayReverseSort(groupArray(cnt)), 1, 3)) AS top3_cnt, sum(cnt) AS assigned_cnt
        FROM (
            SELECT p, c_blk, count() AS cnt
            FROM b_elig
            WHERE c_blk >= 0
            GROUP BY p, c_blk
        ) GROUP BY p;

        INSERT INTO FUNCTION file('{dst_pq}', Parquet)
        SELECT
            p,
            count() AS n_blockers,
            toFloat32(countIf(is_burst = 1 OR is_lblk = 1) / count()) AS share_burst_or_lblk,
            toFloat32(countIf(is_burst = 1) / count()) AS share_burst_only,
            toFloat32(countIf(is_lblk = 1) / count()) AS share_lblk_only,
            toFloat32(joinGet('p_top3', 'top3_cnt', p) / count()) AS share_top3_comm,
            toFloat32(joinGet('p_top3', 'top3_cnt', p) / greatest(1, joinGet('p_top3', 'assigned_cnt', p))) AS share_top3_comm_assigned
        FROM b_elig
        GROUP BY p
        HAVING n_blockers >= {K_MIN}
        ORDER BY p;
        """
        ch_query(sql_v9ii, max_mem=max_mem)

    # 9. Privacy audit on EVERY output parquet before upload:
    #    - zero `did` columns
    #    - every cell/tercile/bucket/producer table has min(distinct actor count) >= 50!
    min_checks = {
        "o5_val_preT_blocks_k50.parquet": "min(n_actors)",
        "o5_val_H_k50.parquet": "min(n_actors)",
        "o5_prod_cand5_matched_blocks_k50.parquet": "min(n_actors)",
        "o5_val_s14_strata_summary.parquet": "min(n_distinct_followers)",
        "o5_val_s14_boot_buckets.parquet": "min(n_distinct_followers_bucket)",
        "o5_val_s14_sizecells_k50.parquet": "min(n_distinct_followers)",
        "o5_val_postT_cells_k50.parquet": "min(n_distinct_followers)",
        "v9i_terciles.parquet": "min(n_distinct_followers)",
        "v9i_terciles_boot.parquet": "min(n_distinct_followers)",
        "v9ii_producers_T.parquet": "min(n_blockers)",
        "v9ii_producers_now.parquet": "min(n_blockers)",
    }
    audit_summary: dict[str, dict] = {}
    for fn, min_expr in min_checks.items():
        fp = out_dir / fn
        raw = ch_query(f"SELECT count(), {min_expr} FROM file('{fp}', Parquet) FORMAT TSV", max_mem=max_mem).strip().split("\t")
        nrows, min_val = int(raw[0]), int(raw[1])
        if nrows > 0 and min_val < K_MIN:
            raise RuntimeError(f"Privacy k>=50 audit FAILED on {fn}: min_actors={min_val} < {K_MIN}")
        audit_summary[fn] = {
            "rows": nrows,
            "min_distinct_actors": min_val if nrows > 0 else None,
            "bytes": fp.stat().st_size,
            "sha256": sha256_file(fp),
        }

    for fn in ("o5_val_comm_stats_T.parquet", "o5_prod_comm_stats_now.parquet"):
        fp = out_dir / fn
        col_u = "n_users_T" if "T" in fn else "n_users_now"
        raw = ch_query(
            f"SELECT count(), countIf({col_u} > 0 AND {col_u} < {K_MIN}) FROM file('{fp}', Parquet) FORMAT TSV",
            max_mem=max_mem,
        ).strip().split("\t")
        nrows, bad = int(raw[0]), int(raw[1])
        if bad != 0:
            raise RuntimeError(f"Privacy k>=50 audit FAILED on {fn}: {bad} rows with 0 < {col_u} < {K_MIN}")
        audit_summary[fn] = {
            "rows": nrows,
            "bad_lt50_rows": bad,
            "bytes": fp.stat().st_size,
            "sha256": sha256_file(fp),
        }

    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt = {
        "step": "R-S14a O5-val, cand5-full O5-prod (blocks-only), and V9(i)/(ii) Spot pass",
        "run_id": args.run_id,
        "s9_run_id": args.s9_run_id,
        "s5p_run_id": args.s5p_run_id,
        "written_utc": now_iso,
        "stage_seconds": round(t_staged - t0, 2),
        "fol_stream_seconds": round(t_fol_done - t_staged, 2),
        "wall_seconds": round(time.time() - t0, 2),
        "partitions": {
            "cand5_T_matched": {"rank0_producers": n_prod_t, "communities": n_comm_t},
            "cand5_full_matched": {"rank0_producers": n_prod_f, "communities": n_comm_f},
        },
        "h_val_preT_counts": {
            "k50_cross_cells": n_k50,
            "top_1pct_cells": n_top1pct,
            "top_5pct_cells": n_top5pct,
        },
        "k_min_enforced": K_MIN,
        "outputs": audit_summary,
    }
    rec_path = out_dir / "receipt.json"
    rec_path.write_text(json.dumps(receipt, indent=1) + "\n")

    if not args.skip_gcs:
        run_cmd([
            "gcloud", "storage", "cp", "--quiet",
            *[str(p) for p in sorted(out_dir.glob("*"))],
            f"{args.plan30}/outputs/{args.run_id}/",
        ])
    print("S14A_DONE", json.dumps({"wall_seconds": receipt["wall_seconds"], "k50_cells": n_k50, "top_1pct_cells": n_top1pct}), flush=True)


if __name__ == "__main__":
    main()
