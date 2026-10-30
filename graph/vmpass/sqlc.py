"""Shared ClickHouse SQL pieces: codes, UDFs, table structures, source helpers."""
from __future__ import annotations

import glob
import os

NONE = 4294967295  # u32 "no account" sentinel

COLL = {
    "app.bsky.graph.follow": 1, "app.bsky.graph.block": 2, "app.bsky.feed.like": 3,
    "app.bsky.feed.repost": 4, "app.bsky.graph.list": 5, "app.bsky.graph.listitem": 6,
    "app.bsky.graph.listblock": 7, "app.bsky.graph.starterpack": 8, "app.bsky.feed.generator": 9,
    "app.bsky.actor.profile": 10, "app.bsky.feed.threadgate": 11, "app.bsky.feed.postgate": 12,
    "app.bsky.actor.contentVisibilityDeclaration": 13, "app.bsky.graph.verification": 14,
    "app.bsky.actor.status": 15, "app.bsky.labeler.service": 16,
    "app.bsky.notification.declaration": 17, "chat.bsky.actor.declaration": 18,
    "app.bsky.feed.post": 19,
}
C_FOLLOW, C_BLOCK, C_LIKE, C_REPOST, C_LIST, C_LISTITEM, C_LISTBLOCK = 1, 2, 3, 4, 5, 6, 7
C_PROFILE, C_THREADGATE, C_POSTGATE = 10, 11, 12
KIND = {"commit": 1, "account": 2, "identity": 3, "sync": 4}
OP = {"create": 1, "update": 2, "delete": 3}
SRC = {"public_archive": 1, "gce_main": 2, "gce_bootstrap_live": 3, "netcup_live": 4, "repo_supplement": 5}
# Evidence type codes (O11 uses 1..5 as its u8 type code).
TY = {"LO": 1, "LR": 2, "ACK": 3, "LU": 4, "RP": 5, "CR": 6, "URf": 7, "URn": 8, "Qf": 9, "Qn": 10,
      "RH": 11, "QH": 12, "LST": 13}


def map_expr(col: str, mapping: dict, default: int = 0) -> str:
    keys = ", ".join(f"'{k}'" for k in mapping)
    vals = ", ".join(str(v) for v in mapping.values())
    return f"transform({col}, [{keys}], [{vals}], {default})"


PRELUDE = r"""
SET session_timezone = 'UTC';
SET output_format_parquet_compression_method = 'zstd';
SET engine_file_truncate_on_insert = 1;
SET input_format_parquet_allow_missing_columns = 1;
SET allow_suspicious_low_cardinality_types = 1;
-- base32-sortable TID -> microseconds ('2'..'7' = 0..5, 'a'..'z' = 6..31), unrolled: no lambda per row
CREATE OR REPLACE FUNCTION tid_us AS (k) -> if(match(k, '^[2-7a-j][2-7a-z]{12}$'),
  toUInt64(intDiv(bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 1, 1)) - if(reinterpretAsUInt8(substring(k, 1, 1)) <= 55, 50, 91)), 60) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 2, 1)) - if(reinterpretAsUInt8(substring(k, 2, 1)) <= 55, 50, 91)), 55) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 3, 1)) - if(reinterpretAsUInt8(substring(k, 3, 1)) <= 55, 50, 91)), 50) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 4, 1)) - if(reinterpretAsUInt8(substring(k, 4, 1)) <= 55, 50, 91)), 45) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 5, 1)) - if(reinterpretAsUInt8(substring(k, 5, 1)) <= 55, 50, 91)), 40) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 6, 1)) - if(reinterpretAsUInt8(substring(k, 6, 1)) <= 55, 50, 91)), 35) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 7, 1)) - if(reinterpretAsUInt8(substring(k, 7, 1)) <= 55, 50, 91)), 30) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 8, 1)) - if(reinterpretAsUInt8(substring(k, 8, 1)) <= 55, 50, 91)), 25) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 9, 1)) - if(reinterpretAsUInt8(substring(k, 9, 1)) <= 55, 50, 91)), 20) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 10, 1)) - if(reinterpretAsUInt8(substring(k, 10, 1)) <= 55, 50, 91)), 15) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 11, 1)) - if(reinterpretAsUInt8(substring(k, 11, 1)) <= 55, 50, 91)), 10) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 12, 1)) - if(reinterpretAsUInt8(substring(k, 12, 1)) <= 55, 50, 91)), 5) + bitShiftLeft(toUInt64(reinterpretAsUInt8(substring(k, 13, 1)) - if(reinterpretAsUInt8(substring(k, 13, 1)) <= 55, 50, 91)), 0), 1024)),
  toUInt64(0));
-- the extract's `time` column is fixed ISO-8601 (YYYY-MM-DDTHH:MM:SS[.ffffff]Z): no best-effort parser per row
CREATE OR REPLACE FUNCTION iso_s AS (x) -> toUInt32(parseDateTimeOrZero(substring(x, 1, 19), '%Y-%m-%dT%H:%i:%S'));
SET use_variant_as_common_type = 0;
CREATE OR REPLACE FUNCTION uri_did AS (u) -> if(startsWith(u, 'at://'), splitByChar('/', u)[3], '');
CREATE OR REPLACE FUNCTION uri_coll AS (u) -> if(startsWith(u, 'at://'), splitByChar('/', u)[4], '');
CREATE OR REPLACE FUNCTION uri_rkey AS (u) -> if(startsWith(u, 'at://'), splitByChar('/', u)[5], '');
"""


def ram_bytes() -> int:
    """Physical RAM of this machine (VMPASS_RAM_BYTES overrides, for tests)."""
    if os.environ.get("VMPASS_RAM_BYTES"):
        return int(os.environ["VMPASS_RAM_BYTES"])
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 64 << 30


def memory_settings(ram: int | None = None) -> dict:
    """Per-query limits from the machine's RAM, not a fixed 512 GB. clickhouse-local's own
    server-wide cap is 0.9 x RAM, so a query limit must sit under it; spilling starts early
    enough that the spill buffers, Join tables and sort runs still fit."""
    ram = ram or ram_bytes()
    return {"max_memory_usage": int(ram * 0.70), "max_bytes_before_external_group_by": int(ram * 0.20),
            "max_bytes_before_external_sort": int(ram * 0.20)}


def settings_sql(rules) -> str:
    ch = {**memory_settings(), **{k: v for k, v in rules["clickhouse"].items() if k not in ("max_memory_usage",
          "max_bytes_before_external_group_by", "max_bytes_before_external_sort")}}
    return "\n".join(f"SET {k} = {v};" for k, v in ch.items()) + "\n"


# Structures of every intermediate (ClickHouse types). Used for null() stand-ins
# when a partition produced no rows, and by the schema tests.
S = {
    "didmap": "did String, uid UInt32, h UInt64, part UInt8, is_prod UInt8, ls UInt8, fit UInt8, m1 UInt8, "
              "pc UInt8, ho UInt8, s200 UInt8, s400 UInt8, hide UInt8, synth_no UInt8, bulk_no UInt8, purge UInt8, "
              "purge_reason UInt8, plc_t UInt32, web UInt8, followers UInt32, o13m1 UInt8, m2c UInt8",
    "ev": "u UInt32, part UInt8, kind UInt8, coll UInt8, op UInt8, rk UInt64, t UInt32, tok UInt8, rev UInt64, "
          "seq UInt64, ing UInt32, src UInt8, subj UInt32, spart UInt8, sph UInt64, scoll UInt8, st UInt32, "
          "lh UInt64, gph UInt64, created UInt32, extra String, trunc UInt8, name String, descr String, feed String, "
          "aact Int8, astat String",
    "po": "u UInt32, part UInt8, rk UInt64, rkey String, t UInt32, tok UInt8, rev UInt64, seq UInt64, op UInt8, "
          "ing UInt32, src UInt8, ph UInt64, created UInt32, par_ph UInt64, par_a UInt32, par_t UInt32, "
          "root_ph UInt64, root_a UInt32, q_ph UInt64, q_a UInt32, q_t UInt32, img UInt8, langs Array(String), "
          "labels Array(String), text String, alts Array(String), embed String",
    # phase 1 outputs
    "pi": "u UInt32, ph UInt64, rkey String, t UInt32, is_reply UInt8, par_ph UInt64, par_a UInt32, par_t UInt32, "
          "root_ph UInt64, root_a UInt32, is_quote UInt8, q_ph UInt64, q_a UInt32, q_t UInt32, img UInt8, langs Array(String)",
    "eng": "u UInt32, a UInt32, ph UInt64, t UInt32, s_t UInt32, typ UInt8, fol UInt8, m Float32, apart UInt8",
    "eng_own": "u UInt32, ph UInt64, t UInt32",
    "rep": "u UInt32, a UInt32, kind UInt8, r_ph UInt64, t UInt32, x_ph UInt64, x_t UInt32, root_ph UInt64, "
           "root_a UInt32, fol UInt8, tpart UInt8",
    "fol": "u UInt32, p UInt32, ft UInt32, dt UInt32, b UInt16, ppart UInt8",
    "blk": "u UInt32, p UInt32, t UInt32, b UInt16, ppart UInt8",
    "cur": "u UInt32, p UInt32, t UInt32, ppart UInt8",
    "lk_own_fol": "u UInt32, a UInt32, cidx UInt8, n UInt32",
    "gates1": "p UInt32, kind UInt8, c_ph UInt64, c_h UInt64, c_t UInt32, g_ph UInt64, trunc UInt8, allow_empty UInt8, gpart UInt8",
    "gates": "p UInt32, kind UInt8, c_ph UInt64, c_a UInt32, c_t UInt32, g_ph UInt64, trunc UInt8, allow_empty UInt8, ppart UInt8",
    "modli": "lh UInt64, owner UInt32, member UInt32, t UInt32",
    "lbsub": "u UInt32, lh UInt64, t UInt32",
    "cand": "u UInt32, p UInt32, cidx UInt8, ck UInt8, pi Float32, ppart UInt8",
    "ustats": "u UInt32, cidx UInt8, n_fprod UInt32, d_u UInt32, light UInt8, a05 Float32, a10 Float32, "
              "vol Float32, missing UInt8, elig_u UInt8, sup UInt8, b1own UInt8, comm1 Int32, comm5 Int32",
    "ustats_m": "u UInt32, cidx UInt8, n_fprod UInt32, d_u UInt32, light UInt8, a05 Float32, a10 Float32, "
                "vol Float32, missing UInt8, elig_u UInt8, sup UInt8, b1own UInt8, comm1 Int32, comm5 Int32, vol_pct Float32",
    "o1": "metric String, k1 String, k2 String, k3 String, v Float64",
    "o7p": "feed String, map UInt8, comm Int32, n UInt64",
    # phase 2 outputs
    "e0": "u UInt32, p UInt32, cidx UInt8, withqn UInt8, noqn UInt8",
    "maintd": "u UInt32, p UInt32, cidx UInt8, n_up UInt32, n_pu UInt32, flag UInt8, kpart UInt8",
    "o12c": "stratum UInt8, kind UInt8, c_ph UInt64, c_a UInt32, x_ph UInt64, x_a UInt32, hsh UInt64",
    "o13p": "m1 UInt8, m2 UInt8, did String, subject String, kind String, created_us Int64",
}


def cols(name: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in S[name]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip().split(" ")[0])
            cur = ""
        else:
            cur += ch
    out.append(cur.strip().split(" ")[0])
    return out


def src(pattern: str, struct: str) -> str:
    """file(pattern) if anything matches, else an empty null() table of that structure."""
    if glob.glob(pattern, recursive=True) and any(os.path.getsize(p) > 0 for p in glob.glob(pattern, recursive=True)):
        return f"file('{pattern}', Parquet, '{S.get(struct, struct)}')"
    return f"null('{S.get(struct, struct)}')"


def out(path: str, fmt: str = "Parquet") -> str:
    return f"INSERT INTO FUNCTION file('{path}', {fmt})"


def split_out(path_fmt: str, key: str, n: int, select_sql: str, tmp: str) -> str:
    """Write select_sql's rows to n files path_fmt.format(k=k), one per value of `key`.

    Replaces `INSERT INTO FUNCTION file(...) PARTITION BY key`: a partitioned file()
    insert drives every partition's Parquet writer from one serial sink (measured ~4x
    slower than an unpartitioned write). Here the rows land in a MergeTree ordered by
    the key (parallel insert), and each file is an unpartitioned, parallel-encoded
    write that reads only its key range through the primary index."""
    s = f"CREATE TABLE {tmp} ENGINE = MergeTree ORDER BY {key} AS {select_sql};\n"
    for k in range(n):
        s += f"{out(path_fmt.format(k=k))} SELECT * FROM {tmp} WHERE {key} = {k};\n"
    return s + f"DROP TABLE {tmp};\n"
