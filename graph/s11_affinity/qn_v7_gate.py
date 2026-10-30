#!/usr/bin/env python3
"""Step S11a: Evaluate the registered Qn gate (R17) and V7 hostile-strata precision/recall (R15, R16).

References:
  - docs/execution-plans/2026-10-25-whole-network-signals-plan.md §4.3, §4.4, §4.8 V7, S11, S15b
  - docs/execution-plans/2026-10-28-whole-network-preregistration.md §9.3, §9.4, R15, R16, R17

Method:
  1. Reconstruct the exact candidate population counts from netcup-export 2026-10-{19,20,21,22}
     using build_reply_pairs.py's exact live-post + 48h block_after rules (plus author-cap counts),
     filtering Oct 30 purge_dids.txt on both parent_did and child_did.
  2. Join reply_pairs.parquet (50,000) + reply_pairs_topup.parquet (33,301) to pair_labels_v2.parquet
     (83,301) on pair_id, dropping any pair touching Oct 30 purge_dids.txt.
  3. Query exact follow state at child post TID `follows(child_did, parent_did, child_tid_us)` from
     the read-only APFS clone of g.follow_events + g.floors via graph/s2prep/chq.sh --fold (P6).
  4. Compute:
     - R17 Qn hostility rate on non-follower quotes (`sonnet_stance == 'hostile' or sonnet_dunk`,
       `sonnet_stance == 'unclear'` excluded from denominator), inverse-inclusion weighted across
       random-quote and block-quote strata, with Wilson 95% CI and 2,000-resample bootstrap 95% CI.
       Decision: zero Qn (`noqn`) iff point estimate >= 0.15, else `withqn`.
     - R16 V7 precision stand-in on `block_after` stratum (overall, RH replies, QH quotes) vs 0.60 bar.
     - R15 V7 recall of `block_after` among validated hostile non-follower quotes (inverse-inclusion
       weighted, 95% bootstrap CI).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import subprocess
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

TID_ALPHA = "234567abcdefghijklmnopqrstuvwxyz"
TID_MAP = {ch: i for i, ch in enumerate(TID_ALPHA)}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def tid_to_us(rkey: str) -> int:
    if len(rkey) != 13 or rkey[0] not in "234567abcdefghij":
        return 0
    val = 0
    for ch in rkey:
        v = TID_MAP.get(ch, -1)
        if v < 0:
            return 0
        val = (val << 5) | v
    return val >> 10


def wilson_ci(p: float, n: float, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [float("nan"), float("nan")]
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n)))
    return [max(0.0, center - half), min(1.0, center + half)]


def weighted_boot_ci(y: np.ndarray, w: np.ndarray, n_boot: int = 2000, seed: int = 20261028) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0 or w.sum() <= 0:
        return [float("nan"), float("nan")]
    yw = y * w
    est = np.empty(n_boot, dtype=np.float64)
    for b0 in range(0, n_boot, 100):
        nb = min(100, n_boot - b0)
        idx = rng.integers(0, n, size=(nb, n))
        counts = np.stack([np.bincount(row, minlength=n) for row in idx]).astype(np.float64)
        num = counts @ yw
        den = counts @ w
        est[b0:b0 + nb] = np.where(den > 0, num / den, np.nan)
    return [float(np.nanpercentile(est, 2.5)), float(np.nanpercentile(est, 97.5))]


def count_netcup_population(days: list[pathlib.Path], excl_path: pathlib.Path, purge: set[str]) -> dict:
    """Exact candidate population counts matching build_reply_pairs.py, minus Oct 30 purge_dids."""
    excl = {l.strip() for l in open(excl_path) if l.strip()}
    for d in days:
        v = pq.read_table(d / "visibility.parquet", columns=["did", "hide_from_algorithmic"])
        excl.update(v.filter(pc.equal(v["hide_from_algorithmic"], 1))["did"].to_pylist())

    deleted: set[tuple[str, str]] = set()
    posts: dict[tuple[str, str], tuple] = {}
    cols = ["did", "rkey", "rev", "is_deleted", "created_at", "text", "reply_parent", "quote_uri"]
    for d in days:
        dl = pq.read_table(d / "deletions.parquet", columns=["collection", "did", "rkey"])
        dl = dl.filter(pc.equal(dl["collection"], "app.bsky.feed.post"))
        deleted.update(zip(dl["did"].to_pylist(), dl["rkey"].to_pylist()))
        t = pq.read_table(d / "posts.parquet", columns=cols)
        for did, rkey, rev, isdel, ca, text, rp, qu in zip(*(t[c].to_pylist() for c in cols)):
            k = (did, rkey)
            if k not in posts or rev > posts[k][0]:
                posts[k] = (rev, isdel, ca, text, rp, qu)

    live = {
        f"at://{d}/app.bsky.feed.post/{r}": (d, v)
        for (d, r), v in posts.items()
        if not v[1] and (d, r) not in deleted and d not in excl
    }
    blocks: dict[tuple[str, str], object] = {}
    for d in days:
        b = pq.read_table(d / "blocks.parquet", columns=["did", "subject", "is_deleted", "indexed_at"])
        for blocker, subj, isdel, at in zip(
            *(b[c].to_pylist() for c in ["did", "subject", "is_deleted", "indexed_at"])
        ):
            if not isdel:
                blocks.setdefault((blocker, subj), at)

    counts_all = {"quote_nonblock": 0, "quote_block": 0, "reply_nonblock": 0, "reply_block": 0}
    counts_clean = {"quote_nonblock": 0, "quote_block": 0, "reply_nonblock": 0, "reply_block": 0}
    for curi, (cdid, (_, _, cca, ctext, rp, qu)) in live.items():
        for kind, puri in (("reply", rp), ("quote", qu)):
            if not puri or puri not in live or len(ctext.strip()) < 3:
                continue
            pdid, pv = live[puri]
            if pdid == cdid or not pv[3].strip():
                continue
            bat = blocks.get((pdid, cdid))
            after = bat is not None and cca is not None and 0 <= (bat - cca).total_seconds() <= 48 * 3600
            key = f"{kind}_{'block' if after else 'nonblock'}"
            counts_all[key] += 1
            if cdid not in purge and pdid not in purge:
                counts_clean[key] += 1
    return {"all_pre_purge": counts_all, "clean_post_purge": counts_clean}


def query_follows_at_tid(
    pairs: list[dict],
    chq_script: pathlib.Path,
    work_dir: pathlib.Path,
) -> dict[str, int]:
    """Query exact follows(child_did, parent_did, child_tid_us) on s2-prep chclone via chq.sh --fold."""
    qpairs_tsv = work_dir / "qpairs_input.tsv"
    out_tsv = work_dir / "qpairs_follows.tsv"
    sql_file = work_dir / "qpairs_follows.sql"

    with open(qpairs_tsv, "w") as fh:
        for r in pairs:
            fh.write(f"{r['pair_id']}\t{r['child_did']}\t{r['parent_did']}\t{r['child_tid_us']}\n")

    sql = f"""
CREATE TABLE qp ENGINE = MergeTree ORDER BY (child_did, parent_did) AS
SELECT * FROM file('{qpairs_tsv}', TSV, 'pair_id String, child_did String, parent_did String, t_us UInt64');

CREATE TABLE qdids ENGINE = Set SETTINGS persistent = 0 AS SELECT DISTINCT child_did FROM qp;
CREATE TABLE qhash ENGINE = Set SETTINGS persistent = 0 AS SELECT DISTINCT cityHash64(child_did, parent_did) FROM qp;
CREATE TABLE floor_lookup ENGINE = Join(ANY, LEFT, did) SETTINGS persistent = 0 AS
SELECT did, floor FROM g.floors WHERE did IN default.qdids;

CREATE TABLE fr ENGINE = MergeTree ORDER BY h AS
SELECT cityHash64(did, subject) AS h, c_us, snap_live, del_us
FROM (
    SELECT did, rkey, bitShiftRight(tid_u64(rkey), 10) AS c_us,
           argMaxIf(subject, rev, is_delete = 0) AS subject,
           toUInt8(argMax(is_delete, (rev, is_delete)) = 0 AND max(rev) >= joinGet('default.floor_lookup', 'floor', did)) AS snap_live,
           bitShiftRight(minIf(rev, is_delete = 1), 10) AS del_us
    FROM g.follow_events
    WHERE did IN default.qdids
    GROUP BY did, rkey
)
WHERE c_us > 0 AND cityHash64(did, subject) IN default.qhash
SETTINGS optimize_aggregation_in_order = 1, read_in_order_two_level_merge_threshold = 0;

SELECT q.pair_id,
       max(toUInt8(f.c_us > 0 AND f.c_us <= q.t_us AND (f.del_us = 0 OR f.del_us > q.t_us) AND (f.snap_live = 1 OR f.del_us > q.t_us))) AS follows_at_t,
       max(toUInt8(f.snap_live = 1)) AS follows_at_snap
FROM qp AS q
LEFT JOIN fr AS f ON f.h = cityHash64(q.child_did, q.parent_did)
GROUP BY q.pair_id
INTO OUTFILE '{out_tsv}' TRUNCATE FORMAT TSV;
"""
    sql_file.write_text(sql)
    subprocess.run(
        [str(chq_script), "s11_qn_follows", str(sql_file), "--fold", "--memory", "20000000000", "--threads", "8"],
        check=True,
    )
    res: dict[str, tuple[int, int]] = {}
    with open(out_tsv) as fh:
        for line in fh:
            pid, ft, fs = line.rstrip("\n").split("\t")
            res[pid] = (int(ft), int(fs))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/claude-labels-20261027")
    ap.add_argument("--netcup-dir", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/netcup-export")
    ap.add_argument("--purge-dids", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s5-sweep-20261030/purge_dids.txt")
    ap.add_argument("--chq", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path.home() / "catbird-lake-local/s11-affinity-20261030")
    args = ap.parse_args()

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.out_dir / "work_qn"
    work_dir.mkdir(parents=True, exist_ok=True)

    purge = {l.strip() for l in open(args.purge_dids) if l.strip()}
    print(f"[qn_v7] Loaded {len(purge):,} purge DIDs", flush=True)

    days = [args.netcup_dir / f"2026-10-{d}" for d in (19, 20, 21, 22)]
    pop = count_netcup_population(days, args.labels_dir / "exclude_dids.txt", purge)
    print(f"[qn_v7] Netcup candidate population: {json.dumps(pop)}", flush=True)

    rp1 = pq.read_table(
        args.labels_dir / "reply_pairs.parquet",
        columns=["pair_id", "stratum", "kind", "block_after", "parent_uri", "child_uri", "parent_did", "child_did"],
    )
    rp2 = pq.read_table(
        args.labels_dir / "reply_pairs_topup.parquet",
        columns=["pair_id", "stratum", "kind", "block_after", "parent_uri", "child_uri", "parent_did", "child_did"],
    )
    pl = pq.read_table(args.labels_dir / "pair_labels_v2.parquet")
    lbl_by_id = {
        r["pair_id"]: r
        for r in pl.select(["pair_id", "sonnet_stance", "sonnet_dunk", "sonnet_confidence"]).to_pylist()
    }

    raw_rows = rp1.to_pylist() + rp2.to_pylist()
    assert len(raw_rows) == len(lbl_by_id), f"mismatch {len(raw_rows)} vs {len(lbl_by_id)}"

    clean_rows = []
    purged_pairs = 0
    bad_tid_pairs = 0
    for r in raw_rows:
        if r["parent_did"] in purge or r["child_did"] in purge:
            purged_pairs += 1
            continue
        rkey = r["child_uri"].rsplit("/", 1)[-1]
        tus = tid_to_us(rkey)
        if tus <= 0:
            bad_tid_pairs += 1
            continue
        lbl = lbl_by_id[r["pair_id"]]
        r["child_tid_us"] = tus
        r["sonnet_stance"] = lbl["sonnet_stance"]
        r["sonnet_dunk"] = bool(lbl["sonnet_dunk"]) if lbl["sonnet_dunk"] is not None else False
        clean_rows.append(r)

    print(f"[qn_v7] Clean pairs={len(clean_rows):,} (purged={purged_pairs:,}, bad_tid={bad_tid_pairs:,})", flush=True)

    fol_map = query_follows_at_tid(clean_rows, args.chq, work_dir)
    for r in clean_rows:
        ft, fs = fol_map.get(r["pair_id"], (0, 0))
        r["follows_at_t"] = ft
        r["follows_at_snap"] = fs

    # Sampled clean stratum counts
    n_q_nonblock = sum(1 for r in clean_rows if r["stratum"] == "quote" and r["kind"] == "quote" and not r["block_after"])
    n_q_block = sum(1 for r in clean_rows if r["stratum"] == "block" and r["kind"] == "quote" and r["block_after"])
    n_r_nonblock = sum(1 for r in clean_rows if r["stratum"] == "reply" and r["kind"] == "reply" and not r["block_after"])
    n_r_block = sum(1 for r in clean_rows if r["stratum"] == "block" and r["kind"] == "reply" and r["block_after"])

    N_q_nonblock = pop["clean_post_purge"]["quote_nonblock"]
    N_q_block = pop["clean_post_purge"]["quote_block"]
    N_r_nonblock = pop["clean_post_purge"]["reply_nonblock"]
    N_r_block = pop["clean_post_purge"]["reply_block"]

    w_q_nonblock = N_q_nonblock / max(1, n_q_nonblock)
    w_q_block = N_q_block / max(1, n_q_block)
    w_r_nonblock = N_r_nonblock / max(1, n_r_nonblock)
    w_r_block = N_r_block / max(1, n_r_block)

    # ---------- R17: Qn hostility rate on non-follower quotes ----------
    nf_quotes = [
        r for r in clean_rows
        if r["kind"] == "quote"
        and r["stratum"] in ("quote", "block")
        and r["follows_at_t"] == 0
    ]
    # R17: Hostile = sonnet_stance == 'hostile' or sonnet_dunk == true; unclear excluded from denominator
    nf_valid = [r for r in nf_quotes if r["sonnet_stance"] is not None and r["sonnet_stance"] != "unclear"]
    y_qn = np.array([1.0 if (r["sonnet_stance"] == "hostile" or r["sonnet_dunk"]) else 0.0 for r in nf_valid])
    w_qn = np.array([w_q_block if r["stratum"] == "block" else w_q_nonblock for r in nf_valid])
    b_qn = np.array([1.0 if r["block_after"] else 0.0 for r in nf_valid])

    qn_rate = float((y_qn * w_qn).sum() / w_qn.sum())
    n_eff_qn = float((w_qn.sum() ** 2) / (w_qn ** 2).sum())
    qn_wilson_eff = wilson_ci(qn_rate, n_eff_qn)
    qn_boot_ci = weighted_boot_ci(y_qn, w_qn)

    # Also check random-quote stratum alone & follower quotes for context
    nf_rand = [r for r in nf_valid if r["stratum"] == "quote"]
    qn_rand_rate = sum(1 for r in nf_rand if r["sonnet_stance"] == "hostile" or r["sonnet_dunk"]) / max(1, len(nf_rand))
    fol_quotes_valid = [
        r for r in clean_rows
        if r["kind"] == "quote" and r["stratum"] in ("quote", "block")
        and r["follows_at_t"] == 1 and r["sonnet_stance"] is not None and r["sonnet_stance"] != "unclear"
    ]
    y_qf = np.array([1.0 if (r["sonnet_stance"] == "hostile" or r["sonnet_dunk"]) else 0.0 for r in fol_quotes_valid])
    w_qf = np.array([w_q_block if r["stratum"] == "block" else w_q_nonblock for r in fol_quotes_valid])
    qf_rate = float((y_qf * w_qf).sum() / w_qf.sum())

    qn_zeroed = bool(qn_rate >= 0.15)
    chosen_variant = "noqn" if qn_zeroed else "withqn"

    # ---------- R15: V7 structural block_after recall on validated hostile non-follower quotes ----------
    hostile_mask = y_qn > 0.5
    w_host = w_qn[hostile_mask]
    b_host = b_qn[hostile_mask]
    v7_recall_nf_quotes = float((b_host * w_host).sum() / w_host.sum()) if w_host.sum() > 0 else float("nan")
    v7_recall_ci = weighted_boot_ci(b_host, w_host)

    # ---------- R16: V7 precision stand-in on block_after stratum ----------
    blk_valid = [
        r for r in clean_rows
        if r["stratum"] == "block" and r["sonnet_stance"] is not None and r["sonnet_stance"] != "unclear"
    ]
    blk_q = [r for r in blk_valid if r["kind"] == "quote"]
    blk_r = [r for r in blk_valid if r["kind"] == "reply"]

    def prec_stats(rows_sub: list[dict], weights: np.ndarray | None = None) -> dict:
        if not rows_sub:
            return {"n": 0, "rate": float("nan"), "wilson_95_ci": [float("nan"), float("nan")]}
        y = np.array([1.0 if (r["sonnet_stance"] == "hostile" or r["sonnet_dunk"]) else 0.0 for r in rows_sub])
        if weights is None:
            p = float(y.mean())
            return {
                "n": len(rows_sub),
                "hostile_or_dunk": int(y.sum()),
                "stance_hostile_only": int(sum(1 for r in rows_sub if r["sonnet_stance"] == "hostile")),
                "dunk_only_nonhostile_stance": int(sum(1 for r in rows_sub if r["sonnet_stance"] != "hostile" and r["sonnet_dunk"])),
                "rate": p,
                "wilson_95_ci": wilson_ci(p, len(rows_sub)),
            }
        p = float((y * weights).sum() / weights.sum())
        neff = float((weights.sum() ** 2) / (weights ** 2).sum())
        return {
            "n": len(rows_sub),
            "n_eff": neff,
            "rate": p,
            "wilson_95_ci": wilson_ci(p, neff),
            "boot_95_ci": weighted_boot_ci(y, weights),
        }

    w_blk = np.array([w_q_block if r["kind"] == "quote" else w_r_block for r in blk_valid])
    v7_prec_unweighted = prec_stats(blk_valid)
    v7_prec_weighted = prec_stats(blk_valid, w_blk)
    v7_prec_qh = prec_stats(blk_q)
    v7_prec_rh = prec_stats(blk_r)
    v7_precision_pass = bool(v7_prec_unweighted["rate"] >= 0.60)

    report = {
        "step": "S11a_Qn_and_V7_gates",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 2),
        "inputs": {
            "pair_labels_v2_sha256": sha256_file(args.labels_dir / "pair_labels_v2.parquet"),
            "reply_pairs_sha256": sha256_file(args.labels_dir / "reply_pairs.parquet"),
            "reply_pairs_topup_sha256": sha256_file(args.labels_dir / "reply_pairs_topup.parquet"),
            "purge_dids_sha256": sha256_file(args.purge_dids),
        },
        "caveat": "Sonnet pair sample comes from the 4 local netcup hot-export days (2026-10-19..22), not O12's 12-month whole-network bank (S1 §9.4).",
        "population_and_sampling": {
            "netcup_candidates": pop,
            "sampled_clean": {
                "quote_nonblock": n_q_nonblock,
                "quote_block": n_q_block,
                "reply_nonblock": n_r_nonblock,
                "reply_block": n_r_block,
                "purged_dropped": purged_pairs,
                "bad_tid_dropped": bad_tid_pairs,
            },
            "inclusion_weights": {
                "w_quote_nonblock": w_q_nonblock,
                "w_quote_block": w_q_block,
                "w_reply_nonblock": w_r_nonblock,
                "w_reply_block": w_r_block,
            },
        },
        "follow_state_check": {
            "quotes_total_clean": sum(1 for r in clean_rows if r["kind"] == "quote"),
            "quotes_following_at_tid": sum(1 for r in clean_rows if r["kind"] == "quote" and r["follows_at_t"] == 1),
            "quotes_following_at_snapshot": sum(1 for r in clean_rows if r["kind"] == "quote" and r["follows_at_snap"] == 1),
            "quotes_nonfollower_at_tid": len(nf_quotes),
        },
        "R17_qn_gate": {
            "bar": "Zero Qn in v0 (use noqn Task A fit and noqn O4 teacher) iff non-follower quote hostility >= 0.15 (S1 §9.4, R17)",
            "nonfollower_quotes_total": len(nf_quotes),
            "nonfollower_quotes_unclear_excluded": len(nf_quotes) - len(nf_valid),
            "nonfollower_quotes_valid": len(nf_valid),
            "n_eff": n_eff_qn,
            "weighted_hostile_or_dunk_rate": qn_rate,
            "wilson_95_ci_neff": qn_wilson_eff,
            "bootstrap_95_ci": qn_boot_ci,
            "random_stratum_only_rate": qn_rand_rate,
            "follower_quotes_weighted_rate_for_comparison": qf_rate,
            "qn_zeroed_in_v0": qn_zeroed,
            "chosen_variant": chosen_variant,
        },
        "V7_hostile_strata": {
            "precision_bar": 0.60,
            "R16_standin_block_after_all_unweighted": v7_prec_unweighted,
            "R16_standin_block_after_all_weighted": v7_prec_weighted,
            "R16_block_after_quotes_QH": v7_prec_qh,
            "R16_block_after_replies_RH": v7_prec_rh,
            "precision_pass": v7_precision_pass,
            "consequence_if_precision_fails": "QH/RH revert to ordinary Q/UR strata (R5 hostile-quote/reply features not adopted) and H uses blocks only (S1 §9.3 V7)",
            "R15_structural_block_recall_on_hostile_nonfollower_quotes": {
                "weighted_recall": v7_recall_nf_quotes,
                "bootstrap_95_ci": v7_recall_ci,
                "raw_hostile_in_block_stratum": int(b_host.sum()),
                "raw_hostile_total": int(hostile_mask.sum()),
            },
        },
    }

    out_json = args.out_dir / "qn_v7_gate_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
