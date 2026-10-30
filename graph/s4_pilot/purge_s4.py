#!/usr/bin/env python3
"""Purge S4 pilot artifacts (`~/catbird-lake-local/s4-pilot-20261028/`) under the §4.9 7-day Mac SLA.

Atomically rewrites (`*.tmp` + `os.replace`) every S4 input table, DID list, assembled vector array,
and per-chunk `.npz` shard so zero rows or vectors belonging to `purge_dids.txt` remain on disk.
Preserves `post_idx` and `pid` identifiers so `graph/s4_pilot/score.py` continues to run cleanly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_purge_sets(purge_txt: pathlib.Path, producers_tsv: pathlib.Path) -> tuple[set[str], set[int]]:
    purge_dids = {line.strip() for line in open(purge_txt) if line.startswith("did:")}
    purge_pids: set[int] = set()
    with open(producers_tsv) as fh:
        for line in fh:
            pid_s, did, _ = line.rstrip("\n").split("\t", 2)
            if did in purge_dids:
                purge_pids.add(int(pid_s))
    return purge_dids, purge_pids


def purge_vector_dir(vdir: pathlib.Path, bad_ids: np.ndarray) -> dict:
    ids_path = vdir / "ids.npy"
    vec_path = vdir / "vec256_fp16.npy"
    pre_ids_sha = sha256_file(ids_path)
    pre_vec_sha = sha256_file(vec_path)

    ids = np.load(ids_path)
    vecs = np.load(vec_path)
    assert len(ids) == len(vecs), f"{vdir}: len(ids)={len(ids)} != len(vecs)={len(vecs)}"

    keep = ~np.isin(ids, bad_ids)
    removed = int((~keep).sum())
    new_ids = ids[keep]
    new_vecs = vecs[keep]

    # Atomically rewrite any chunk_*.npz containing a purged id.
    chunks_rewritten = 0
    chunks_total = 0
    for cf in sorted(vdir.glob("chunk_*.npz")):
        chunks_total += 1
        with np.load(cf) as z:
            c_ids = z["ids"]
            c_vec = z["vec"]
        c_keep = ~np.isin(c_ids, bad_ids)
        if not bool(c_keep.all()):
            tmp = cf.with_name(cf.name + ".tmp.npz")
            np.savez(tmp, ids=c_ids[c_keep], vec=c_vec[c_keep])
            os.replace(tmp, cf)
            chunks_rewritten += 1

    if removed > 0:
        ids_tmp = vdir / "ids.tmp.npy"
        vec_tmp = vdir / "vec256_fp16.tmp.npy"
        np.save(ids_tmp, new_ids)
        np.save(vec_tmp, new_vecs)
        os.replace(ids_tmp, ids_path)
        os.replace(vec_tmp, vec_path)

    # Post-condition check across assembled arrays AND chunks.
    chk_ids = np.load(ids_path)
    assert int(np.isin(chk_ids, bad_ids).sum()) == 0, f"{vdir}: purged ids remain in ids.npy"
    for cf in sorted(vdir.glob("chunk_*.npz")):
        with np.load(cf) as z:
            assert int(np.isin(z["ids"], bad_ids).sum()) == 0, f"{cf}: purged ids remain"

    return {
        "rows_before": int(len(ids)),
        "rows_removed": removed,
        "rows_after": int(len(new_ids)),
        "chunks_total": chunks_total,
        "chunks_rewritten": chunks_rewritten,
        "pre_sha256": {"ids.npy": pre_ids_sha, "vec256_fp16.npy": pre_vec_sha},
        "post_sha256": {"ids.npy": sha256_file(ids_path), "vec256_fp16.npy": sha256_file(vec_path)},
    }


def purge_parquet(path: pathlib.Path, purge_dids_arr: pa.Array, purge_pids_arr: pa.Array | None = None) -> tuple[dict, np.ndarray | None]:
    pre_sha = sha256_file(path)
    t = pq.read_table(path)
    n0 = t.num_rows
    bad = pc.is_in(t["did"], value_set=purge_dids_arr)
    if purge_pids_arr is not None and "pid" in t.column_names:
        bad = pc.or_(bad, pc.is_in(t["pid"], value_set=purge_pids_arr))
    removed_post_idx = None
    if "post_idx" in t.column_names:
        removed_post_idx = t.filter(bad)["post_idx"].to_numpy()
    kept = t.filter(pc.invert(bad))
    removed = n0 - kept.num_rows
    if removed > 0:
        tmp = path.with_name(path.name + ".tmp")
        pq.write_table(kept, tmp, compression="zstd")
        os.replace(tmp, path)
    # Verify 0 remain
    t_chk = pq.read_table(path, columns=["did"] + (["pid"] if "pid" in t.column_names else []))
    rem_chk = pc.sum(pc.cast(pc.is_in(t_chk["did"], value_set=purge_dids_arr), pa.int64())).as_py() or 0
    assert rem_chk == 0, f"{path}: {rem_chk} purged DIDs remain"
    info = {
        "rows_before": n0,
        "rows_removed": removed,
        "rows_after": kept.num_rows,
        "pre_sha256": pre_sha,
        "post_sha256": sha256_file(path),
    }
    return info, removed_post_idx


def purge_text_dids(path: pathlib.Path, purge_dids: set[str]) -> dict:
    pre_sha = sha256_file(path)
    lines = open(path).readlines()
    kept = [ln for ln in lines if ln.strip().split("\t", 1)[0] not in purge_dids]
    removed = len(lines) - len(kept)
    if removed > 0:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w") as fh:
            fh.writelines(kept)
        os.replace(tmp, path)
    return {
        "rows_before": len(lines),
        "rows_removed": removed,
        "rows_after": len(kept),
        "pre_sha256": pre_sha,
        "post_sha256": sha256_file(path),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", default="~/catbird-lake-local/s4-pilot-20261028")
    ap.add_argument("--purge-dids", required=True)
    ap.add_argument("--producers-tsv", default="~/catbird-lake-local/graph-build-v3/out-full-20261024/producers.tsv")
    ap.add_argument("--receipt-out", default="")
    args = ap.parse_args(argv)

    pilot = pathlib.Path(args.pilot_dir).expanduser().resolve()
    purge_txt = pathlib.Path(args.purge_dids).expanduser().resolve()
    producers_tsv = pathlib.Path(args.producers_tsv).expanduser().resolve()
    receipt_out = pathlib.Path(args.receipt_out).expanduser().resolve() if args.receipt_out else (pilot / "purge_receipt_20261030.json")

    purge_dids, purge_pids = load_purge_sets(purge_txt, producers_tsv)
    purge_dids_arr = pa.array(sorted(purge_dids), type=pa.string())
    purge_pids_arr = pa.array(sorted(purge_pids), type=pa.int64())
    purge_pids_np = np.array(sorted(purge_pids), dtype=np.int64)

    # Identify all post_idx belonging to purged DIDs/pids BEFORE rewriting posts_corpus.parquet
    pc_pre = pq.read_table(pilot / "inputs/posts_corpus.parquet", columns=["post_idx", "did", "pid"])
    bad_mask = pc.or_(
        pc.is_in(pc_pre["did"], value_set=purge_dids_arr),
        pc.is_in(pc_pre["pid"], value_set=purge_pids_arr),
    )
    bad_post_idx = pc_pre.filter(bad_mask)["post_idx"].to_numpy()
    purged_embedded_producers = len(set(pc_pre.filter(bad_mask)["did"].to_pylist()))

    inputs_rep = {}
    for fn in ("posts_corpus.parquet", "bios_corpus.parquet", "bios_raw_all_producers.parquet"):
        p = pilot / "inputs" / fn
        if p.exists():
            info, _ = purge_parquet(p, purge_dids_arr, purge_pids_arr)
            inputs_rep[fn] = info

    for fn in ("producer_dids.txt", "list_owners.txt", "list_owner_split.tsv"):
        p = pilot / "inputs" / fn
        if p.exists():
            inputs_rep[fn] = purge_text_dids(p, purge_dids)

    vectors_rep = {
        "eg2_posts": purge_vector_dir(pilot / "vectors/eg2_posts", bad_post_idx),
        "eg2_bios": purge_vector_dir(pilot / "vectors/eg2_bios", purge_pids_np),
        "harrier_bakeoff": purge_vector_dir(pilot / "vectors/harrier_bakeoff", bad_post_idx),
    }

    # Final cross-consistency assertion matching score.py
    pc_post = pq.read_table(pilot / "inputs/posts_corpus.parquet", columns=["post_idx", "bakeoff"])
    eg2_ids = np.load(pilot / "vectors/eg2_posts/ids.npy")
    har_ids = np.load(pilot / "vectors/harrier_bakeoff/ids.npy")
    assert np.array_equal(eg2_ids, pc_post["post_idx"].to_numpy()), "eg2_posts ids mismatch after purge"
    bmask = pc_post["bakeoff"].to_numpy()
    assert np.array_equal(har_ids, pc_post["post_idx"].to_numpy()[bmask]), "harrier_bakeoff ids mismatch after purge"

    bc_post = pq.read_table(pilot / "inputs/bios_corpus.parquet", columns=["pid"])
    bio_ids = np.load(pilot / "vectors/eg2_bios/ids.npy")
    assert np.array_equal(bio_ids, bc_post["pid"].to_numpy()), "eg2_bios ids mismatch after purge"

    receipt = {
        "step": "S4 pilot purge (§4.9 7-day Mac SLA)",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pilot_dir": str(pilot),
        "purge_dids_file": str(purge_txt),
        "purge_dids_sha256": sha256_file(purge_txt),
        "purge_dids_count": len(purge_dids),
        "purge_producer_pids_count": len(purge_pids),
        "embedded_producers_removed": purged_embedded_producers,
        "inputs": inputs_rep,
        "vectors": vectors_rep,
        "post_check_score_alignment_ok": True,
    }
    receipt_out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({
        "embedded_producers_removed": purged_embedded_producers,
        "posts_removed": inputs_rep["posts_corpus.parquet"]["rows_removed"],
        "bios_removed": inputs_rep["bios_corpus.parquet"]["rows_removed"],
        "raw_bios_removed": inputs_rep["bios_raw_all_producers.parquet"]["rows_removed"],
        "harrier_removed": vectors_rep["harrier_bakeoff"]["rows_removed"],
    }, indent=1))


if __name__ == "__main__":
    main()
