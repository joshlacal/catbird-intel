"""Step 4: export the selected collaborative variant in an engine-loadable layout (only if 03 passed the gate).

follow_vec_i8.bin  N x 256 int8, row i = round(unit(item vector) * 127), zero-padded past d
follow_vec_dids.txt line i = producer did of row i (own alignment; NOT the intel-serving dids.txt / vecRow scheme)
manifest.json      method, selected variant, train cut, counts, gate + eval table, sha256 of both files

usage: python 04_export.py [--force]   (--force writes even when the gate failed; manifest records gate.pass=false)
"""
import hashlib
import json
import os
import sys

import common
from common import DATA, SCRATCH, log

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

R = json.load(open(os.path.join(DATA, "eval_results.json")))
if not R["gate"]["pass"] and "--force" not in sys.argv:
    log("gate failed; nothing exported (use --force to export anyway)")
    sys.exit(0)
sel = R["selected_variant"]
d, power, centred = int(sel["d"]), float(sel["sv_power"]), bool(sel["centred"])

Z = np.load(os.path.join(DATA, "svd.npz"))
V, S, item_pid = Z["V"], Z["S"], Z["item_pid"]


def unit(a):
    nrm = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, nrm, out=np.zeros_like(a), where=nrm > 0)


U = unit(V[:, :d] * (S[:d] ** power)[None, :])
if centred:
    U = unit(U - U.mean(axis=0, keepdims=True))
Q = np.zeros((len(U), 256), np.int8)
Q[:, :d] = np.round(U * 127).clip(-127, 127).astype(np.int8)
zero_rows = int((np.abs(Q).sum(1) == 0).sum())

prod = pacsv.read_csv(common.PRODUCERS, read_options=pacsv.ReadOptions(column_names=["pid", "did", "followers"]),
                      parse_options=pacsv.ParseOptions(delimiter="\t"),
                      convert_options=pacsv.ConvertOptions(column_types={"pid": pa.int64(), "did": pa.string(), "followers": pa.int64()}))
assert np.array_equal(prod["pid"].to_numpy(), np.arange(prod.num_rows))
dids = np.asarray(prod["did"].to_pylist(), dtype=object)[item_pid]

vec_path = os.path.join(SCRATCH, "follow_vec_i8.bin")
did_path = os.path.join(SCRATCH, "follow_vec_dids.txt")
Q.tofile(vec_path)
with open(did_path, "w") as fh:
    fh.write("\n".join(dids) + "\n")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


svd_stats = json.load(open(os.path.join(DATA, "svd_stats.json")))
prep = json.load(open(os.path.join(DATA, "prep_stats.json")))
keep = ["B0", "B1", "B1t", "B2", "B3@0.35", sel["name"] + ":cos", sel["name"] + ":blend@0.35", sel["name"] + ":both@0.35",
        "N1q:cos", "N1q:blend@0.35", "N1q:both@0.35"]
manifest = {
    "method": "randomized truncated SVD (subspace iteration, K=%d, q=%d, oversample %d) of row-L2-normalised IDF-weighted binary viewer x producer follow matrix; item vector = V[:, :d] * S[:d]^power, unit, optional mean-centring, unit"
              % (svd_stats["K"], svd_stats["q_iters"], svd_stats["oversample"]),
    "selected_variant": sel, "d": d, "zero_padded_to": 256, "zero_padded": d < 256,
    "layout": {"vectors": "follow_vec_i8.bin", "dids": "follow_vec_dids.txt", "rows": int(len(Q)), "dim": 256, "dtype": "int8",
               "scale": "round(unit * 127); re-unit after load", "alignment": "row i <-> line i of follow_vec_dids.txt", "zero_rows": zero_rows},
    "train_cut": "follow edges with created <= 2026-08-15 (s2a train_T); producers = accounts with >= 100 followers at T",
    "train_sample": {"viewer_follow_range": [10, 3000], "rows": svd_stats["rows"], "nnz": svd_stats["nnz"], "items_min_sample_df": svd_stats["min_df"],
                     "items": svd_stats["items"]},
    "eval": {"window": "test follows 2026-08-16..2026-10-18, non-burst, alive 14 d, positive in onboarding pool", "events": prep.get("events_sampled"),
             "gate": R["gate"], "coverage": R["coverage"], "table": {k: R["table"][k] for k in keep if k in R["table"]}},
    "sha256": {"follow_vec_i8.bin": sha(vec_path), "follow_vec_dids.txt": sha(did_path)},
}
json.dump(manifest, open(os.path.join(SCRATCH, "manifest.json"), "w"), indent=1)
log("exported", len(Q), "rows d", d, "zero rows", zero_rows, "->", SCRATCH)
