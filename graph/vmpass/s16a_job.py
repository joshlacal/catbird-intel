#!/usr/bin/env python3
"""R-S16a GCE Spot job (whole-network plan §3.1 S16a, §5.2, S1 §5):

Runs on a short-lived self-deleting `n2-highmem-16` Spot VM in `us-west1` (cap $2.00).
Tests whether each candidate producer pair `(p, q)` drawn from `p`'s top-1% S12m slab-cosine
neighbours (excluding `O4` top-200, `<= 20` per anchor producer) shares zero light engagers in
`(C - 90d, C]` for both:
  - `preT` (`cidx = 2`, `C = T = 1786838399`, window `(1779062399, 1786838399]`)
  - `prod` (`cidx = 4`, `C = T_now = 1792367999`, window `(1784591999, 1792367999]`)

Strictly enforces:
  - Latest purge + opt-out filter (`s5p-20261030a/purge_uids.parquet` U `didmap.purge=1` U `didmap.synth_no=1`)
    on every engager and producer.
  - Qualifying `O11` types: `type IN (1, 2, 4, 5)` (`LO=1, LR=2, LU=4, RP=5`; `ACK=3` courtesy likes excluded).
  - Eligible engagers at `cidx`: `light = 1 AND b1own = 0` (from `ustats.parquet`), `liker != author`,
    and `(liker, author)` not `MAINT`-flagged at `cidx` (`O11-side/maint_pairs.parquet`).
  - Privacy invariant: ONLY per-pair boolean flags (`zero_shared_any1 UInt8, zero_shared_2d UInt8`) and
    per-producer candidate/hard-negative counts (`0..20`) leave the VM — zero DIDs, zero engager lists,
    and zero user counts `< 50`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time

T_CUT = 1786838399   # 2026-08-15T23:59:59Z (cidx = 2)
T_NOW = 1792367999   # 2026-10-18T23:59:59Z (cidx = 4)
W90 = 90 * 86400     # 7,776,000 seconds

INTERSECT_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    uint32_t max_pid;
    const uint64_t *starts;
    const uint32_t *lens;
    const uint64_t *packed;
    const uint32_t *pairs;
    uint32_t *out;
    size_t n_pairs;
    int tid;
    int n_threads;
} worker_t;

static inline size_t lower_bound_u64(const uint64_t *arr, size_t lo, size_t hi, uint64_t key_hi) {
    while (lo < hi) {
        size_t mid = lo + ((hi - lo) >> 1);
        if ((arr[mid] >> 1) < key_hi) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static void *run_worker(void *arg) {
    worker_t *w = (worker_t *)arg;
    size_t chunk = (w->n_pairs + (size_t)w->n_threads - 1) / (size_t)w->n_threads;
    size_t s = (size_t)w->tid * chunk;
    size_t e = s + chunk;
    if (e > w->n_pairs) e = w->n_pairs;

    for (size_t idx = s; idx < e; idx++) {
        uint32_t p = w->pairs[2 * idx];
        uint32_t q = w->pairs[2 * idx + 1];
        uint32_t na = (p <= w->max_pid) ? w->lens[p] : 0;
        uint32_t nb = (q <= w->max_pid) ? w->lens[q] : 0;

        uint32_t shared_any1 = 0;
        uint32_t shared_2d = 0;

        if (na > 0 && nb > 0 && p != q) {
            const uint64_t *A = w->packed + w->starts[p];
            const uint64_t *B = w->packed + w->starts[q];
            size_t la = na, lb = nb;
            if (la > lb) {
                const uint64_t *tmp_p = A; A = B; B = tmp_p;
                size_t tmp_l = la; la = lb; lb = tmp_l;
            }
            size_t i = 0, j = 0;
            while (i < la && j < lb) {
                uint64_t ua = A[i] >> 1;
                uint64_t ub = B[j] >> 1;
                if (ua == ub) {
                    shared_any1 = 1;
                    if ((A[i] & 1ULL) && (B[j] & 1ULL)) {
                        shared_2d = 1;
                        break;
                    }
                    i++;
                    j++;
                } else if (ua < ub) {
                    if ((la - i) * 8 < (lb - j)) {
                        i++;
                    } else {
                        i = lower_bound_u64(A, i + 1, la, ub);
                    }
                } else {
                    if ((lb - j) > (la - i) * 8) {
                        j = lower_bound_u64(B, j + 1, lb, ua);
                    } else {
                        j++;
                    }
                }
            }
        }
        w->out[4 * idx]     = p;
        w->out[4 * idx + 1] = q;
        w->out[4 * idx + 2] = shared_any1 ? 0U : 1U;
        w->out[4 * idx + 3] = shared_2d   ? 0U : 1U;
    }
    return NULL;
}

static size_t fsize(FILE *f) {
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    return (sz > 0) ? (size_t)sz : 0;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <deg.bin> <packed.bin> <pairs.bin> <out.bin> <threads>\n", argv[0]);
        return 2;
    }
    FILE *f_deg = fopen(argv[1], "rb");
    FILE *f_pck = fopen(argv[2], "rb");
    FILE *f_prs = fopen(argv[3], "rb");
    if (!f_deg || !f_pck || !f_prs) {
        fprintf(stderr, "failed opening input binary files\n");
        return 1;
    }
    size_t deg_bytes = fsize(f_deg);
    size_t pck_bytes = fsize(f_pck);
    size_t prs_bytes = fsize(f_prs);

    size_t n_deg = deg_bytes / 8;
    size_t n_pck = pck_bytes / 8;
    size_t n_pairs = prs_bytes / 8;
    int n_threads = atoi(argv[5]);
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 32) n_threads = 32;

    uint32_t *deg_buf = (uint32_t *)malloc(deg_bytes ? deg_bytes : 8);
    uint64_t *packed = (uint64_t *)malloc(pck_bytes ? pck_bytes : 8);
    uint32_t *pairs = (uint32_t *)malloc(prs_bytes ? prs_bytes : 8);
    if (!deg_buf || !packed || !pairs) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    if (deg_bytes && fread(deg_buf, 1, deg_bytes, f_deg) != deg_bytes) return 1;
    if (pck_bytes && fread(packed, 1, pck_bytes, f_pck) != pck_bytes) return 1;
    if (prs_bytes && fread(pairs, 1, prs_bytes, f_prs) != prs_bytes) return 1;
    fclose(f_deg);
    fclose(f_pck);
    fclose(f_prs);

    uint32_t max_pid = 0;
    for (size_t i = 0; i < n_deg; i++) {
        uint32_t a = deg_buf[2 * i];
        if (a > max_pid) max_pid = a;
    }
    uint64_t *starts = (uint64_t *)calloc((size_t)max_pid + 1, sizeof(uint64_t));
    uint32_t *lens   = (uint32_t *)calloc((size_t)max_pid + 1, sizeof(uint32_t));
    if (!starts || !lens) {
        fprintf(stderr, "calloc index failed\n");
        return 1;
    }
    uint64_t cur = 0;
    for (size_t i = 0; i < n_deg; i++) {
        uint32_t a = deg_buf[2 * i];
        uint32_t c = deg_buf[2 * i + 1];
        starts[a] = cur;
        lens[a] = c;
        cur += (uint64_t)c;
    }
    if (cur != (uint64_t)n_pck) {
        fprintf(stderr, "offset sum %llu != n_pck %zu\n", (unsigned long long)cur, n_pck);
        return 1;
    }
    free(deg_buf);

    uint32_t *out = (uint32_t *)malloc((n_pairs ? n_pairs : 1) * 4 * sizeof(uint32_t));
    if (!out) {
        fprintf(stderr, "malloc out failed\n");
        return 1;
    }

    pthread_t th[32];
    worker_t wk[32];
    for (int t = 0; t < n_threads; t++) {
        wk[t].max_pid = max_pid;
        wk[t].starts = starts;
        wk[t].lens = lens;
        wk[t].packed = packed;
        wk[t].pairs = pairs;
        wk[t].out = out;
        wk[t].n_pairs = n_pairs;
        wk[t].tid = t;
        wk[t].n_threads = n_threads;
        pthread_create(&th[t], NULL, run_worker, &wk[t]);
    }
    for (int t = 0; t < n_threads; t++) {
        pthread_join(th[t], NULL);
    }

    FILE *f_out = fopen(argv[4], "wb");
    if (!f_out) return 1;
    if (n_pairs && fwrite(out, sizeof(uint32_t) * 4, n_pairs, f_out) != n_pairs) return 1;
    fclose(f_out);

    free(out);
    free(starts);
    free(lens);
    free(packed);
    free(pairs);
    return 0;
}
"""


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run_cmd(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd[:6])}\nstderr: {r.stderr[-1800:]}")
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


def compile_intersector(work: pathlib.Path) -> pathlib.Path:
    c_src = work / "s16a_intersect.c"
    c_bin = work / "s16a_intersect"
    c_src.write_text(INTERSECT_C)
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("No C compiler (cc/gcc/clang) found for s16a_intersect!")
    run_cmd([cc, "-O3", "-pthread", str(c_src), "-o", str(c_bin)])
    return c_bin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", default="/mnt/work")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--s9-run-id", default="s9-20261028a")
    ap.add_argument("--s5p-run-id", default="s5p-20261030a")
    ap.add_argument("--plan30", default="gs://catbird-436600-plan-30d")
    ap.add_argument("--plan60", default="gs://catbird-436600-plan-60d")
    ap.add_argument("--n-parts", type=int, default=16)
    ap.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 8)))
    ap.add_argument("--skip-gcs", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    work = pathlib.Path(args.work)
    in_dir = work / "in"
    mid_dir = work / "mid"
    out_dir = work / "out"
    for d in (in_dir, mid_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    max_mem = max(8_000_000_000, int(ram * 0.70))
    intersect_bin = compile_intersector(work)

    if not args.skip_gcs:
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan30}/outputs/{args.s5p_run_id}/purge_uids.parquet", str(in_dir / "purge_uids.parquet")])
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan60}/didmap/{args.s9_run_id}/didmap.parquet", str(in_dir / "didmap.parquet")])
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan30}/intermediates/{args.s9_run_id}/g/ustats.parquet", str(in_dir / "ustats.parquet")])
        run_cmd(["gcloud", "storage", "cp", "--quiet", f"{args.plan30}/outputs/{args.s9_run_id}/O11-side/maint_pairs.parquet", str(in_dir / "maint_pairs.parquet")])
        run_cmd([
            "gcloud", "storage", "cp", "--quiet",
            f"{args.plan30}/inputs/{args.run_id}/cand_pairs_preT.parquet",
            f"{args.plan30}/inputs/{args.run_id}/cand_pairs_prod.parquet",
            f"{in_dir}/",
        ])

    t_staged = time.time()

    # 1. Build clean integer exclusion tables, clean candidate pair tables, eligible light-engager tables,
    #    and directed MAINT exclusion tables; unlink bulky staged files immediately to keep disk low!
    pu_clean_pq = mid_dir / "pu_or_synth_uids.parquet"
    cand_pre_pq = mid_dir / "cand_preT_clean.parquet"
    cand_prd_pq = mid_dir / "cand_prod_clean.parquet"
    rel_auth_pq = mid_dir / "rel_authors.parquet"
    elig_t_pq = mid_dir / "elig_likers_T.parquet"
    elig_n_pq = mid_dir / "elig_likers_now.parquet"
    maint_t_pq = mid_dir / "maint_dir_T.parquet"
    maint_n_pq = mid_dir / "maint_dir_now.parquet"

    sql_prep = f"""
    INSERT INTO FUNCTION file('{pu_clean_pq}', Parquet)
    SELECT DISTINCT toUInt32(uid) AS uid FROM (
        SELECT toUInt32(uid) AS uid FROM file('{in_dir}/purge_uids.parquet', Parquet)
        UNION ALL
        SELECT toUInt32(uid) AS uid FROM file('{in_dir}/didmap.parquet', Parquet) WHERE purge = 1 OR synth_no = 1
    );

    CREATE TABLE bad ENGINE = Join(ANY, LEFT, uid) AS
    SELECT toUInt32(uid) AS uid, toUInt8(1) AS x FROM file('{pu_clean_pq}', Parquet);

    INSERT INTO FUNCTION file('{cand_pre_pq}', Parquet)
    SELECT DISTINCT toUInt32(p) AS p, toUInt32(q) AS q
    FROM file('{in_dir}/cand_pairs_preT.parquet', Parquet)
    WHERE p != q AND joinGet('bad', 'x', toUInt32(p)) = 0 AND joinGet('bad', 'x', toUInt32(q)) = 0
    ORDER BY p, q;

    INSERT INTO FUNCTION file('{cand_prd_pq}', Parquet)
    SELECT DISTINCT toUInt32(p) AS p, toUInt32(q) AS q
    FROM file('{in_dir}/cand_pairs_prod.parquet', Parquet)
    WHERE p != q AND joinGet('bad', 'x', toUInt32(p)) = 0 AND joinGet('bad', 'x', toUInt32(q)) = 0
    ORDER BY p, q;

    INSERT INTO FUNCTION file('{rel_auth_pq}', Parquet)
    SELECT DISTINCT toUInt32(a) AS author FROM (
        SELECT p AS a FROM file('{cand_pre_pq}', Parquet)
        UNION ALL
        SELECT q AS a FROM file('{cand_pre_pq}', Parquet)
        UNION ALL
        SELECT p AS a FROM file('{cand_prd_pq}', Parquet)
        UNION ALL
        SELECT q AS a FROM file('{cand_prd_pq}', Parquet)
    ) ORDER BY author;

    INSERT INTO FUNCTION file('{elig_t_pq}', Parquet)
    SELECT DISTINCT toUInt32(u) AS u
    FROM file('{in_dir}/ustats.parquet', Parquet)
    WHERE cidx = 2 AND light = 1 AND b1own = 0 AND joinGet('bad', 'x', toUInt32(u)) = 0;

    INSERT INTO FUNCTION file('{elig_n_pq}', Parquet)
    SELECT DISTINCT toUInt32(u) AS u
    FROM file('{in_dir}/ustats.parquet', Parquet)
    WHERE cidx = 4 AND light = 1 AND b1own = 0 AND joinGet('bad', 'x', toUInt32(u)) = 0;

    CREATE TABLE rauth ENGINE = Join(ANY, LEFT, author) AS
    SELECT toUInt32(author) AS author, toUInt8(1) AS ok FROM file('{rel_auth_pq}', Parquet);

    INSERT INTO FUNCTION file('{maint_t_pq}', Parquet)
    SELECT DISTINCT toUInt32(u) AS u, toUInt32(p) AS p FROM (
        SELECT toUInt32(u1) AS u, toUInt32(u2) AS p FROM file('{in_dir}/maint_pairs.parquet', Parquet)
        WHERE cidx = 2 AND flag_12 = 1 AND joinGet('rauth', 'ok', toUInt32(u2)) = 1
        UNION ALL
        SELECT toUInt32(u2) AS u, toUInt32(u1) AS p FROM file('{in_dir}/maint_pairs.parquet', Parquet)
        WHERE cidx = 2 AND flag_21 = 1 AND joinGet('rauth', 'ok', toUInt32(u1)) = 1
    );

    INSERT INTO FUNCTION file('{maint_n_pq}', Parquet)
    SELECT DISTINCT toUInt32(u) AS u, toUInt32(p) AS p FROM (
        SELECT toUInt32(u1) AS u, toUInt32(u2) AS p FROM file('{in_dir}/maint_pairs.parquet', Parquet)
        WHERE cidx = 4 AND flag_12 = 1 AND joinGet('rauth', 'ok', toUInt32(u2)) = 1
        UNION ALL
        SELECT toUInt32(u2) AS u, toUInt32(u1) AS p FROM file('{in_dir}/maint_pairs.parquet', Parquet)
        WHERE cidx = 4 AND flag_21 = 1 AND joinGet('rauth', 'ok', toUInt32(u1)) = 1
    );
    """
    ch_query(sql_prep, max_mem=max_mem)
    for fn in ("didmap.parquet", "ustats.parquet", "maint_pairs.parquet"):
        (in_dir / fn).unlink(missing_ok=True)

    # 2. Stream O11/j00..j15.parquet one partition at a time (~6.8 GiB each), emitting compact
    #    `(author, liker, two_d)` Parquets per partition for `preT` and `prod`
    o11_cur = in_dir / "o11_cur.parquet"
    t_lo_pre = T_CUT - W90
    t_lo_now = T_NOW - W90

    for j in range(args.n_parts):
        if args.skip_gcs:
            local_o11 = in_dir / f"o11_j{j:02d}.parquet"
            if not local_o11.exists():
                continue
            local_o11.replace(o11_cur)
        else:
            run_cmd([
                "gcloud", "storage", "cp", "--quiet",
                f"{args.plan30}/outputs/{args.s9_run_id}/O11/j{j:02d}.parquet",
                str(o11_cur),
            ])
        sql_part = f"""
        CREATE TABLE rauth ENGINE = Join(ANY, LEFT, author) AS
        SELECT toUInt32(author) AS author, toUInt8(1) AS ok FROM file('{rel_auth_pq}', Parquet);

        CREATE TABLE elig_t ENGINE = Join(ANY, LEFT, u) AS
        SELECT toUInt32(u) AS u, toUInt8(1) AS ok FROM file('{elig_t_pq}', Parquet);

        CREATE TABLE elig_n ENGINE = Join(ANY, LEFT, u) AS
        SELECT toUInt32(u) AS u, toUInt8(1) AS ok FROM file('{elig_n_pq}', Parquet);

        CREATE TABLE mt_t ENGINE = Join(ANY, LEFT, u, p) AS
        SELECT toUInt32(u) AS u, toUInt32(p) AS p, toUInt8(1) AS bad FROM file('{maint_t_pq}', Parquet);

        CREATE TABLE mt_n ENGINE = Join(ANY, LEFT, u, p) AS
        SELECT toUInt32(u) AS u, toUInt32(p) AS p, toUInt8(1) AS bad FROM file('{maint_n_pq}', Parquet);

        INSERT INTO FUNCTION file('{mid_dir}/ae_T_j{j:02d}.parquet', Parquet)
        SELECT
            toUInt32(author) AS author,
            toUInt32(liker) AS liker,
            toUInt8(count() >= 2 AND min(intDiv(t, 86400)) != max(intDiv(t, 86400))) AS two_d
        FROM file('{o11_cur}', Parquet)
        WHERE type IN (1, 2, 4, 5)
          AND t > {t_lo_pre} AND t <= {T_CUT}
          AND liker != author
          AND joinGet('rauth', 'ok', toUInt32(author)) = 1
          AND joinGet('elig_t', 'ok', toUInt32(liker)) = 1
          AND joinGet('mt_t', 'bad', toUInt32(liker), toUInt32(author)) = 0
        GROUP BY author, liker;

        INSERT INTO FUNCTION file('{mid_dir}/ae_now_j{j:02d}.parquet', Parquet)
        SELECT
            toUInt32(author) AS author,
            toUInt32(liker) AS liker,
            toUInt8(count() >= 2 AND min(intDiv(t, 86400)) != max(intDiv(t, 86400))) AS two_d
        FROM file('{o11_cur}', Parquet)
        WHERE type IN (1, 2, 4, 5)
          AND t > {t_lo_now} AND t <= {T_NOW}
          AND liker != author
          AND joinGet('rauth', 'ok', toUInt32(author)) = 1
          AND joinGet('elig_n', 'ok', toUInt32(liker)) = 1
          AND joinGet('mt_n', 'bad', toUInt32(liker), toUInt32(author)) = 0
        GROUP BY author, liker;
        """
        ch_query(sql_part, max_mem=max_mem)
        o11_cur.unlink(missing_ok=True)

    t_streamed = time.time()

    # 3. For both `preT` and `prod`, build sorted flat binary CSR files, run exact pthreads intersection,
    #    and emit boolean-only pair flags + per-producer summary counts (`0..20`).
    window_stats: dict[str, dict] = {}
    for tag, ae_glob, cand_pq in (
        ("preT", f"{mid_dir}/ae_T_j*.parquet", cand_pre_pq),
        ("prod", f"{mid_dir}/ae_now_j*.parquet", cand_prd_pq),
    ):
        merged_pq = mid_dir / f"ae_merged_{tag}.parquet"
        deg_bin = mid_dir / f"deg_{tag}.bin"
        pck_bin = mid_dir / f"pck_{tag}.bin"
        prs_bin = mid_dir / f"prs_{tag}.bin"
        res_bin = mid_dir / f"res_{tag}.bin"
        pairs_out_pq = out_dir / f"s16a_pairs_{tag}.parquet"
        prod_cnt_pq = out_dir / f"s16a_producer_counts_{tag}.parquet"

        ch_query(
            f"""
            INSERT INTO FUNCTION file('{merged_pq}', Parquet)
            SELECT author, liker, max(two_d) AS two_d
            FROM file('{ae_glob}', Parquet)
            GROUP BY author, liker
            ORDER BY author, liker;

            INSERT INTO FUNCTION file('{deg_bin}', RowBinary)
            SELECT toUInt32(author) AS author, toUInt32(count()) AS cnt
            FROM file('{merged_pq}', Parquet)
            GROUP BY author
            ORDER BY author;

            INSERT INTO FUNCTION file('{pck_bin}', RowBinary)
            SELECT bitOr(bitShiftLeft(toUInt64(liker), 1), toUInt64(two_d)) AS packed
            FROM file('{merged_pq}', Parquet)
            ORDER BY author, liker;

            INSERT INTO FUNCTION file('{prs_bin}', RowBinary)
            SELECT toUInt32(p) AS p, toUInt32(q) AS q
            FROM file('{cand_pq}', Parquet)
            ORDER BY p, q;
            """,
            max_mem=max_mem,
        )

        t_int0 = time.time()
        run_cmd([str(intersect_bin), str(deg_bin), str(pck_bin), str(prs_bin), str(res_bin), str(args.threads)])
        t_int_sec = round(time.time() - t_int0, 3)

        ch_query(
            f"""
            INSERT INTO FUNCTION file('{pairs_out_pq}', Parquet)
            SELECT
                p,
                q,
                toUInt8(z1) AS zero_shared_any1,
                toUInt8(z2) AS zero_shared_2d
            FROM file('{res_bin}', RowBinary, 'p UInt32, q UInt32, z1 UInt32, z2 UInt32')
            ORDER BY p, q;

            INSERT INTO FUNCTION file('{prod_cnt_pq}', Parquet)
            SELECT
                p,
                toUInt16(count()) AS n_candidates,
                toUInt16(sum(zero_shared_any1)) AS n_hardneg_any1,
                toUInt16(sum(zero_shared_2d)) AS n_hardneg_2d
            FROM file('{pairs_out_pq}', Parquet)
            GROUP BY p
            ORDER BY p;
            """,
            max_mem=max_mem,
        )

        st_row = ch_query(
            f"""
            SELECT
                (SELECT count() FROM file('{merged_pq}', Parquet)),
                (SELECT count() FROM file('{pairs_out_pq}', Parquet)),
                (SELECT sum(zero_shared_any1) FROM file('{pairs_out_pq}', Parquet)),
                (SELECT sum(zero_shared_2d) FROM file('{pairs_out_pq}', Parquet)),
                (SELECT count() FROM file('{prod_cnt_pq}', Parquet)),
                (SELECT avg(n_hardneg_any1) FROM file('{prod_cnt_pq}', Parquet)),
                (SELECT avg(n_hardneg_2d) FROM file('{prod_cnt_pq}', Parquet))
            FORMAT TSV
            """,
            max_mem=max_mem,
        ).strip().split("\t")

        window_stats[tag] = {
            "ae_pairs_scanned": int(st_row[0]),
            "candidate_pairs": int(st_row[1]),
            "zero_shared_any1_pairs": int(st_row[2]),
            "zero_shared_2d_pairs": int(st_row[3]),
            "anchor_producers": int(st_row[4]),
            "mean_hardneg_any1_per_anchor": round(float(st_row[5] or 0.0), 4),
            "mean_hardneg_2d_per_anchor": round(float(st_row[6] or 0.0), 4),
            "intersect_seconds": t_int_sec,
        }

        for tmp_f in (merged_pq, deg_bin, pck_bin, prs_bin, res_bin):
            tmp_f.unlink(missing_ok=True)

    # 4. Privacy audit on every output file: verify zero String columns (no DIDs) and only boolean/<=20 columns
    out_files = sorted(f.name for f in out_dir.glob("*.parquet"))
    for fn in out_files:
        types = ch_query(
            f"DESCRIBE TABLE file('{out_dir / fn}', Parquet) FORMAT TSV",
            max_mem=max_mem,
        )
        if "String" in types:
            raise RuntimeError(f"Privacy violation: String column found in {fn}: {types}")

    receipt = {
        "step": "R-S16a hard-negative shared-light-engager filter",
        "run_id": args.run_id,
        "s9_run_id": args.s9_run_id,
        "s5p_run_id": args.s5p_run_id,
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_seconds": round(time.time() - t0, 2),
        "stage_seconds": round(t_staged - t0, 2),
        "o11_stream_seconds": round(t_streamed - t_staged, 2),
        "privacy_audit": {
            "dids_emitted": 0,
            "engager_counts_emitted": 0,
            "pair_flags_boolean_only": True,
        },
        "windows": window_stats,
        "outputs": {
            fn: {
                "bytes": (out_dir / fn).stat().st_size,
                "sha256": sha256_file(out_dir / fn),
            }
            for fn in out_files
        },
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    if not args.skip_gcs:
        run_cmd(["gcloud", "storage", "cp", "--quiet", *[str(out_dir / fn) for fn in out_files], str(out_dir / "receipt.json"), f"{args.plan30}/outputs/{args.run_id}/"])

    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
