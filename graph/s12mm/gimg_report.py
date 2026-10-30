"""G-img, report-only under S1 amendment A2: B1b text-only vs multimodal on image-heavy lists (Mac).

  gimg_report.py VECTORS OUT_JSON [--split list_owner_split.tsv] [--heavy q20_lists_k5_at50.tsv]

VECTORS: a directory of S12m unit parquets (vec + vec_text + modality) or the bench's vectors.parquet
(vec_text, vec_mm70, vec_mm280). Arms, per producer (DID from the post URI), each the normalised mean of its
posts' 256-d vectors:
  text   the text-only vector of every post that has one (S12m: vec_text for image posts, vec otherwise;
         modality == NONE is always excluded);
  mm     the stored primary vector (S12m: vec for modality != NONE; bench: vec_mm70 and, separately, vec_mm280).
Universe = producers present in both arms (identical for both). Lists, families, the kNN macro top-1 and the
percentile bootstrap are S4's (`graph/s4_pilot/score.py`, imported, not copied); B1b at 98.75%.
Reported, never gating: (a) image-heavy B1b lists (Q20's qualifying list file), mm - text, with the frozen
G-img bar (>= +0.01 and CI excluding 0) shown for reference; (b) every other B1b list ("no loss elsewhere").

Memory note: S12m output directories hold 1,853 unit files (36.3M posts) bucketed by `blake2b(did) % 64`
(`u-b0000-*.parquet` .. `u-b0063-*.parquet`), so all posts of any producer reside in a single bucket. Streaming
one bucket at a time produces bit-for-bit identical pooled producer vectors in < 2 GB RAM.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "s4_pilot"))
import mmlib as L  # noqa: E402

LAKE = pathlib.Path.home() / "catbird-lake-local"
SPLIT = LAKE / "s4-pilot-20261028/inputs/list_owner_split.tsv"
HEAVY = LAKE / "c4-prep-20261028/q20/q20_lists_k5_at50.tsv"
BUCKET_RE = re.compile(r"^u-b(\d{4})-\d{4}-\d+\.parquet$")


def _extract_table_arms(tb) -> tuple[list[str], dict[str, list]]:
    cols = set(tb.column_names)
    d = tb.to_pydict()
    uris = d["uri"]
    arms: dict[str, list] = {}
    if "vec" in cols:  # S12m units
        text = [
            None if m == L.NONE else (vt if vt is not None else (v if m in (L.TEXT, L.TEXT_FALLBACK) else None))
            for v, vt, m in zip(d["vec"], d["vec_text"], d["modality"])
        ]
        mm = [None if m == L.NONE else v for v, m in zip(d["vec"], d["modality"])]
        arms["text"] = text
        arms["mm"] = mm
    else:  # bench vectors
        arms["text"] = d["vec_text"]
        for c in sorted(cols):
            if c.startswith("vec_mm"):
                arms[c[4:]] = d[c]
    return [L.did_rkey(u)[0] for u in uris], arms


def load_vectors(path):
    p = pathlib.Path(path)
    files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
    dids_all: list[str] = []
    arms_all: dict[str, list] = {}
    for f in files:
        dids, arms = _extract_table_arms(pq.read_table(f))
        dids_all.extend(dids)
        for k, v in arms.items():
            arms_all.setdefault(k, []).extend(v)
    return dids_all, arms_all


def pooled_arrays(dids: list[str], blobs: list) -> tuple[list[str], np.ndarray]:
    keep = [i for i, b in enumerate(blobs) if b is not None]
    if not keep:
        return [], np.zeros((0, L.DIM), dtype=np.float32)
    v = L.unpack([blobs[i] for i in keep]).astype(np.float32)
    ds = np.array([dids[i] for i in keep])
    o = np.argsort(ds, kind="stable")
    u, start = np.unique(ds[o], return_index=True)
    m = np.add.reduceat(v[o], start, axis=0)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return u.tolist(), m


def pooled(dids, blobs):
    u, m = pooled_arrays(dids, blobs)
    return dict(zip(u, m))


def load_and_pool(path) -> tuple[int, dict[str, tuple[list[str], np.ndarray]]]:
    """Return `(total_posts, {arm_name: (sorted_dids, unit_vecs_fp32)})`.

    Uses 64-bucket streaming when `path` is a directory of `u-bXXXX-*.parquet` files so peak RAM stays < 2 GB,
    and falls back to single-pass `load_vectors` otherwise.
    """
    p = pathlib.Path(path)
    if p.is_dir():
        files = sorted(f for f in p.glob("*.parquet") if not f.name.startswith("._"))
        by_bucket: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
        all_bucketed = bool(files)
        for f in files:
            m = BUCKET_RE.match(f.name)
            if not m:
                all_bucketed = False
                break
            by_bucket[m.group(1)].append(f)
        if all_bucketed:
            total_posts = 0
            arm_dids: dict[str, list[str]] = collections.defaultdict(list)
            arm_vecs: dict[str, list[np.ndarray]] = collections.defaultdict(list)
            seen_dids: set[str] = set()
            for b in sorted(by_bucket):
                b_dids: list[str] = []
                b_arms: dict[str, list] = collections.defaultdict(list)
                for f in by_bucket[b]:
                    dids, arms = _extract_table_arms(pq.read_table(f, columns=["uri", "modality", "vec", "vec_text"]))
                    total_posts += len(dids)
                    b_dids.extend(dids)
                    for k, v in arms.items():
                        b_arms[k].extend(v)
                b_unique = set(b_dids)
                overlap = seen_dids & b_unique
                if overlap:
                    raise ValueError(f"DID appeared in multiple buckets (sample: {next(iter(overlap))})")
                seen_dids |= b_unique
                for k, blobs in b_arms.items():
                    u, mat = pooled_arrays(b_dids, blobs)
                    if u:
                        arm_dids[k].extend(u)
                        arm_vecs[k].append(mat)
            out: dict[str, tuple[list[str], np.ndarray]] = {}
            for k in arm_dids:
                dids_k = arm_dids[k]
                vecs_k = np.concatenate(arm_vecs[k], axis=0) if arm_vecs[k] else np.zeros((0, L.DIM), dtype=np.float32)
                o = np.argsort(np.array(dids_k), kind="stable")
                out[k] = ([dids_k[i] for i in o], vecs_k[o])
            return total_posts, out

    dids, arms = load_vectors(p)
    return len(dids), {k: pooled_arrays(dids, blobs) for k, blobs in arms.items()}


def compare(fam_lists, uni: list[str], text_mat: np.ndarray, mm_mat: np.ndarray, score):
    index = {d: i for i, d in enumerate(uni)}
    nn_t = score.nearest(text_mat, chunk=512)
    nn_m = score.nearest(mm_mat, chunk=512)
    _u, groups = score.groups_in(fam_lists, index)
    if not groups:
        return {"lists": 0}
    a, b = score.list_hits(groups, nn_t), score.list_hits(groups, nn_m)
    d = b - a
    lo, hi = score.boot_ci(d, score.LEVELS["B1b"])
    return {
        "lists": len(groups),
        "members": int(sum(len(g) for g in groups)),
        "text_macro_top1": float(a.mean()),
        "mm_macro_top1": float(b.mean()),
        "diff_mm_minus_text": float(d.mean()),
        "ci_level": score.LEVELS["B1b"],
        "ci": [lo, hi],
        "frozen_bar_met_for_reference": bool(d.mean() >= 0.01 and lo > 0),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vectors")
    ap.add_argument("out_json")
    ap.add_argument("--split", default=str(SPLIT))
    ap.add_argument("--heavy", default=str(HEAVY))
    a = ap.parse_args(argv)
    import score

    fam, _n, _m = score.load_lists(a.split)
    heavy = {ln.split("\t", 1)[0] for ln in open(a.heavy) if ln.strip()}
    b1b_heavy = [x for x in fam["B1b"] if x[0] in heavy]
    b1b_rest = [x for x in fam["B1b"] if x[0] not in heavy]
    total_posts, pooled_arms = load_and_pool(a.vectors)
    text_dids, text_mat = pooled_arms["text"]
    text_idx = {d: i for i, d in enumerate(text_dids)}
    res = {
        "status": "report-only (S1 amendment A2)",
        "vectors": a.vectors,
        "posts": total_posts,
        "b1b_lists_image_heavy": len(b1b_heavy),
        "b1b_lists_other": len(b1b_rest),
        "arms": {},
    }
    for arm, (mm_dids, mm_mat) in pooled_arms.items():
        if arm == "text":
            continue
        mm_idx = {d: i for i, d in enumerate(mm_dids)}
        uni = sorted(set(text_dids) & set(mm_dids))
        if len(uni) > 2:
            t_sub = text_mat[[text_idx[d] for d in uni]]
            m_sub = mm_mat[[mm_idx[d] for d in uni]]
            ih = compare(b1b_heavy, uni, t_sub, m_sub, score)
            ot = compare(b1b_rest, uni, t_sub, m_sub, score)
        else:
            ih, ot = {"lists": 0}, {"lists": 0}
        res["arms"][arm] = {
            "producers": len(uni),
            "image_heavy_lists": ih,
            "other_b1b_lists": ot,
        }
    pathlib.Path(a.out_json).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
