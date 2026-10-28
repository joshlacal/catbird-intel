"""G-img, report-only under S1 amendment A2: B1b text-only vs multimodal on image-heavy lists (Mac).

  gimg_report.py VECTORS OUT_JSON [--split list_owner_split.tsv] [--heavy q20_lists_k5_at50.tsv]

VECTORS: a directory of S12m unit parquets (vec + vec_text + modality) or the bench's vectors.parquet
(vec_text, vec_mm70, vec_mm280). Arms, per producer (DID from the post URI), each the normalised mean of its
posts' 256-d vectors:
  text   the text-only vector of every post that has one (S12m: vec_text for image posts, vec otherwise);
  mm     the stored primary vector (S12m: vec; bench: vec_mm70 and, separately, vec_mm280).
Universe = producers present in both arms (identical for both). Lists, families, the kNN macro top-1 and the
percentile bootstrap are S4's (`graph/s4_pilot/score.py`, imported, not copied); B1b at 98.75%.
Reported, never gating: (a) image-heavy B1b lists (Q20's qualifying list file), mm - text, with the frozen
G-img bar (>= +0.01 and CI excluding 0) shown for reference; (b) every other B1b list ("no loss elsewhere").
"""
from __future__ import annotations

import argparse
import json
import pathlib
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


def load_vectors(path):
    p = pathlib.Path(path)
    files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
    t = [pq.read_table(f) for f in files]
    cols = set(t[0].column_names)
    uris, arms = [], {}
    for tb in t:
        d = tb.to_pydict()
        uris += d["uri"]
        if "vec" in cols:   # S12m units
            text = [vt if vt is not None else (v if m in (L.TEXT, L.TEXT_FALLBACK) else None)
                    for v, vt, m in zip(d["vec"], d["vec_text"], d["modality"])]
            arms.setdefault("text", []).extend(text)
            arms.setdefault("mm", []).extend(d["vec"])
        else:               # bench vectors
            arms.setdefault("text", []).extend(d["vec_text"])
            for c in cols:
                if c.startswith("vec_mm"):
                    arms.setdefault(c[4:], []).extend(d[c])
    return [L.did_rkey(u)[0] for u in uris], arms


def pooled(dids, blobs):
    keep = [i for i, b in enumerate(blobs) if b is not None]
    if not keep:
        return {}
    v = L.unpack([blobs[i] for i in keep]).astype(np.float32)
    ds = np.array([dids[i] for i in keep])
    o = np.argsort(ds, kind="stable")
    u, start = np.unique(ds[o], return_index=True)
    m = np.add.reduceat(v[o], start, axis=0)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)
    return dict(zip(u.tolist(), m))


def compare(fam_lists, uni, text, mm, score):
    index = {d: i for i, d in enumerate(uni)}
    nn_t = score.nearest(np.stack([text[d] for d in uni]))
    nn_m = score.nearest(np.stack([mm[d] for d in uni]))
    _u, groups = score.groups_in(fam_lists, index)
    if not groups:
        return {"lists": 0}
    a, b = score.list_hits(groups, nn_t), score.list_hits(groups, nn_m)
    d = b - a
    lo, hi = score.boot_ci(d, score.LEVELS["B1b"])
    return {"lists": len(groups), "members": int(sum(len(g) for g in groups)), "text_macro_top1": float(a.mean()),
            "mm_macro_top1": float(b.mean()), "diff_mm_minus_text": float(d.mean()), "ci_level": score.LEVELS["B1b"],
            "ci": [lo, hi], "frozen_bar_met_for_reference": bool(d.mean() >= 0.01 and lo > 0)}


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
    dids, arms = load_vectors(a.vectors)
    text = pooled(dids, arms["text"])
    res = {"status": "report-only (S1 amendment A2)", "vectors": a.vectors, "posts": len(dids),
           "b1b_lists_image_heavy": len(b1b_heavy), "b1b_lists_other": len(b1b_rest), "arms": {}}
    for arm, blobs in arms.items():
        if arm == "text":
            continue
        mm = pooled(dids, blobs)
        uni = sorted(set(text) & set(mm))
        res["arms"][arm] = {"producers": len(uni),
                            "image_heavy_lists": compare(b1b_heavy, uni, text, mm, score) if len(uni) > 2 else {"lists": 0},
                            "other_b1b_lists": compare(b1b_rest, uni, text, mm, score) if len(uni) > 2 else {"lists": 0}}
    pathlib.Path(a.out_json).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
