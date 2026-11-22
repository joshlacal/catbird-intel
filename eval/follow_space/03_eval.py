"""Step 3: temporal ranking eval of onboarding scene candidates (engine serving shape, deterministic).

Event = (viewer v, scene s, positive p) from 01_prep. Candidates = scene members minus v's train follows.
Score (log-weight, higher first, ties -> better rankScoreBps):
  B0  rankScoreBps                     B1  pool followers          B1t followers at T (producers.tsv)
  B2  t1 cosine (known first, rest by rank; null taste -> B0)
  B3  -1.5*log(rank+3) + beta*z_t1     (onboarding_live.ts rotation 1/(rank+3)^1.5 x exp(personalBeta*z))
  N1  collaborative cosine             N2  B3 with z_N1            N3  -1.5*log(rank+3) + beta*(z_t1 + z_N1)
z = (aff-mean)/popstd over scene members with known affinity, clamp +-2; needs >=4 known and std>1e-6 (:550-575).
Taste = unit mean of unit vectors of the viewer's follows (random cap 400, shared), null if <8 have vectors (:489-511).
Variants of N1: d in {64,128,256} x singular-value power {0,0.5,1} x {raw, mean-centred}; picked on the dev half
(viewer hash), reported on the report half. N1q = chosen variant padded to 256, int8 round(unit*127), re-unit.

Reads: $SCRATCH/data/{pool,events,viewer_follows,svd}.npz, degrees.npy; producers.tsv; intel-serving-v4 t1 vectors.
Writes: $SCRATCH/data/eval_results.json, $SCRATCH/eval_table.md
"""
import json
import os

import common
from common import DATA, SCRATCH, SEED, log

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import scipy.sparse as sp

CAP = 400
TAG = os.environ.get("OUT_TAG", "")
MIN_TASTE = 8
BETAS = [0.35, 0.7, 1.0, 1.5]
GATE_BETA = 0.35
DIMS, POWERS, CENTRED = [64, 128, 256], [0.0, 0.5, 1.0], [False, True]
VB = [("10-49", 10, 50), ("50-199", 50, 200), ("200+", 200, 1 << 62)]
PB = [("<1500", 0, 1500), ("1500-20k", 1500, 20000), (">=20k", 20000, 1 << 62)]


def unit(a):
    nrm = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, nrm, out=np.zeros_like(a), where=nrm > 0)


# ---- load ----
P = np.load(os.path.join(DATA, "pool.npz"))
m_scene, m_fol, m_bps, m_pid = P["scene"], P["followers"], P["rank_bps"], P["pid_t"]
scene_off = P["scene_off"]; n_scenes = len(scene_off) - 1
E = np.load(os.path.join(DATA, "events.npz"))
e_uid, e_mem = E["uid"], E["member"]
n_ev = len(e_uid)
VF = np.load(os.path.join(DATA, "viewer_follows.npz"))
v_uids, v_indptr, v_pids = VF["uids"], VF["indptr"], VF["pids"]
n_v = len(v_uids)
deg = np.load(os.path.join(DATA, "degrees.npy"))
e_v = np.searchsorted(v_uids, e_uid)
assert np.array_equal(v_uids[e_v], e_uid)
prod = pacsv.read_csv(common.PRODUCERS, read_options=pacsv.ReadOptions(column_names=["pid", "did", "followers"]),
                      parse_options=pacsv.ParseOptions(delimiter="\t"),
                      convert_options=pacsv.ConvertOptions(column_types={"pid": pa.int64(), "did": pa.string(), "followers": pa.int64()}))
n_prod = prod.num_rows
prod_fol = prod["followers"].to_numpy()
m_fol_t = np.where(m_pid >= 0, prod_fol[m_pid.clip(0)], 0)
log("events", n_ev, "viewers", n_v, "members", len(m_pid))

# ---- shared random follow cap; full-follow keys for the candidate mask ----
rng = np.random.default_rng(SEED + 3)
cnt = np.diff(v_indptr)
row_of = np.repeat(np.arange(n_v, dtype=np.int64), cnt)
o = np.lexsort((rng.random(len(v_pids)), row_of))
pos_in_row = np.arange(len(v_pids)) - v_indptr[row_of[o]]
kept = np.sort(o[pos_in_row < CAP])
F = sp.csr_matrix((np.ones(len(kept), np.float32), v_pids[kept].astype(np.int64), np.concatenate([[0], np.cumsum(np.minimum(cnt, CAP))])),
                  shape=(n_v, n_prod))
fkeys = np.sort((row_of.astype(np.uint64) << np.uint64(32)) | v_pids.astype(np.uint64))
del row_of, o, pos_in_row, kept
log("follow cap built; capped nnz", F.nnz, "of", len(v_pids))


def taste(col_map, U):
    """Viewer taste per source: unit(sum of unit vectors of capped follows that have one); null if < MIN_TASTE."""
    Fc = F.tocoo()
    r = col_map[Fc.col]
    ok = r >= 0
    M = sp.csr_matrix((Fc.data[ok], (Fc.row[ok], r[ok])), shape=(n_v, U.shape[0]))
    n_has = np.diff(M.indptr)
    C = unit(np.asarray(M @ U, dtype=np.float32))
    return C, n_has >= MIN_TASTE


# ---- t1 source (intel-serving-v4; rows via signals.bin vecRow + t1 flag 8, vector_store.ts:410,428-433,572) ----
with open(os.path.join(common.SERVING, "dids.txt")) as fh:
    sdids = fh.read().split("\n")
if sdids and sdids[-1] == "":
    sdids.pop()
sig = np.fromfile(os.path.join(common.SERVING, "signals.bin"), dtype=np.dtype({"names": ["flags", "vecRow"], "formats": ["<u2", "<i4"], "offsets": [0, 20], "itemsize": 24}))
assert len(sig) == len(sdids)
srow = np.where(((sig["flags"] & 8) != 0) & (sig["vecRow"] >= 0), sig["vecRow"], -1).astype(np.int64)
sdid_arr = pa.array(sdids, pa.string())
prod_srow = srow[pc.fill_null(pc.index_in(prod["did"], value_set=sdid_arr), 0).to_numpy()]
prod_srow[pc.is_null(pc.index_in(prod["did"], value_set=sdid_arr)).to_numpy(zero_copy_only=False)] = -1
mem_si = pc.index_in(pa.array(P["did"].tolist(), pa.string()), value_set=sdid_arr)
mem_srow = np.where(pc.is_valid(mem_si).to_numpy(zero_copy_only=False), srow[pc.fill_null(mem_si, 0).to_numpy()], -1)
vec = np.memmap(os.path.join(common.SERVING, "vec_t1_i8.bin"), dtype=np.int8, mode="r").reshape(-1, 256)
need = np.unique(np.concatenate([prod_srow[np.unique(F.indices)], mem_srow]))
need = need[need >= 0]
Ut1 = unit(np.asarray(vec[need], dtype=np.float32))
t1_local = np.full(vec.shape[0], -1, np.int64); t1_local[need] = np.arange(len(need))
prod_t1 = np.where(prod_srow >= 0, t1_local[prod_srow.clip(0)], -1)
mem_t1 = np.where(mem_srow >= 0, t1_local[mem_srow.clip(0)], -1)
C_t1, has_t1 = taste(prod_t1, Ut1)
Um_t1 = np.where((mem_t1 >= 0)[:, None], Ut1[mem_t1.clip(0)], 0).astype(np.float32)
SRC_T1 = (C_t1, has_t1, Um_t1, mem_t1 >= 0)
log("t1 rows", vec.shape[0], "producers with t1", int((prod_t1 >= 0).sum()), "members with t1", int((mem_t1 >= 0).sum()),
    "viewers with t1 taste", int(has_t1.sum()), "/", n_v)

# ---- collaborative source ----
Z = np.load(os.environ.get("SVD_NPZ", os.path.join(DATA, "svd.npz")))
V, S, item_pid = Z["V"], Z["S"], Z["item_pid"]
prod_svd = np.full(n_prod, -1, np.int64); prod_svd[item_pid] = np.arange(len(item_pid))
mem_svd = np.where(m_pid >= 0, prod_svd[m_pid.clip(0)], -1)
log("svd items", len(item_pid), "members with N1 vector", int((mem_svd >= 0).sum()))


def item_matrix(d, power, centred):
    U = unit(V[:, :d] * (S[:d] ** power)[None, :])
    if centred:
        U = unit(U - U.mean(axis=0, keepdims=True))
    return U.astype(np.float32)


def quantise(U):
    return unit(np.round(U * 127).clip(-127, 127).astype(np.int8).astype(np.float32))


def svd_source(U):
    C, has = taste(prod_svd, U)
    Um = np.where((mem_svd >= 0)[:, None], U[mem_svd.clip(0)], 0).astype(np.float32)
    return (C, has, Um, mem_svd >= 0)


# ---- per-scene evaluation ----
ev_scene = m_scene[e_mem]
ev_order = np.argsort(ev_scene, kind="stable")
ev_bounds = np.searchsorted(ev_scene[ev_order], np.arange(n_scenes + 1))
n_cand = np.zeros(n_ev, np.int32)


def run_eval(srcs):
    """srcs: name -> (taste (n_v x d), has_taste (n_v), member vecs (n_mem x d), member_has (n_mem)); 't1' required."""
    out = {}
    cov = {k: {"null_taste": 0, "pos_no_vec": 0, "cand_no_vec_sum": 0.0} for k in srcs}

    def put(name, idx, r):
        out.setdefault(name, np.zeros(n_ev, np.int16))[idx] = r

    for s in range(int(os.environ.get("SMOKE_SCENES", n_scenes))):
        idx = ev_order[ev_bounds[s]:ev_bounds[s + 1]]
        if len(idx) == 0:
            continue
        a, b = scene_off[s], scene_off[s + 1]; ms = b - a
        ne = len(idx); ar = np.arange(ne)
        rnk = np.empty(ms, np.int64); rnk[np.argsort(-m_bps[a:b], kind="stable")] = np.arange(ms)
        pos = e_mem[idx] - a
        vi = e_v[idx]
        pid = m_pid[a:b]
        keys = ((vi.astype(np.uint64) << np.uint64(32))[:, None] | pid.clip(0).astype(np.uint64)[None, :]).ravel()
        loc = np.searchsorted(fkeys, keys).clip(0, len(fkeys) - 1)
        followed = ((fkeys[loc] == keys).reshape(ne, ms)) & (pid >= 0)[None, :]
        assert not followed[ar, pos].any()
        cand = ~followed; cand[ar, pos] = False
        n_cand[idx] = cand.sum(1)
        rp = rnk[pos][:, None]

        def position(score):
            score = np.broadcast_to(score, (ne, ms))
            sp_ = score[ar, pos][:, None]
            return ((score > sp_) | ((score == sp_) & (rnk[None, :] < rp))) & cand

        def pos_count(score):
            return position(score).sum(1)

        rot = -1.5 * np.log(rnk + 3.0)
        put("B0", idx, pos_count(-rnk.astype(np.float64)))
        put("B1", idx, pos_count(m_fol[a:b].astype(np.float64)))
        put("B1t", idx, pos_count(m_fol_t[a:b].astype(np.float64)))
        zs = {}
        for name, (C, has_c, Um, has_m) in srcs.items():
            hc = has_c[vi]; hm = has_m[a:b]
            aff = C[vi] @ Um[a:b].T
            cs = np.where(hm[None, :], 2.0 + aff, -1.0 - rnk[None, :] * 1e-3)
            cs = np.where(hc[:, None], cs, -rnk[None, :].astype(np.float64))
            put("B2" if name == "t1" else name + ":cos", idx, pos_count(cs))
            known = hm[None, :] & hc[:, None]
            k = known.sum(1); kd = np.maximum(k, 1)
            mean = (aff * known).sum(1) / kd
            std = np.sqrt((((aff - mean[:, None]) ** 2) * known).sum(1) / kd)
            ok = (k >= 4) & (std > 1e-6)
            zs[name] = np.where(known & ok[:, None], np.clip((aff - mean[:, None]) / np.where(std > 0, std, 1)[:, None], -2, 2), 0.0)
            cov[name]["null_taste"] += int((~hc).sum())
            cov[name]["pos_no_vec"] += int((~hm[pos]).sum())
            cov[name]["cand_no_vec_sum"] += float(((cand & ~hm[None, :]).sum(1) / np.maximum(cand.sum(1), 1)).sum())
        for beta in BETAS:
            for name in srcs:
                put(("B3" if name == "t1" else name + ":blend") + f"@{beta}", idx, pos_count(rot[None, :] + beta * zs[name]))
                if name != "t1":
                    put(name + f":both@{beta}", idx, pos_count(rot[None, :] + beta * (zs["t1"] + zs[name])))
    for k in cov:
        cov[k] = {"null_taste_share": round(cov[k]["null_taste"] / n_ev, 4), "positive_no_vec_share": round(cov[k]["pos_no_vec"] / n_ev, 4),
                  "mean_candidate_no_vec_share": round(cov[k]["cand_no_vec_sum"] / n_ev, 4)}
    return out, cov


# ---- slices ----
h = (e_uid.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(1 << 32)
is_dev = ((h >> np.uint64(16)) & np.uint64(1)) == 0
vdeg = deg[e_uid]
pfol = m_fol[e_mem]
SL = {"full": np.ones(n_ev, bool), "dev": is_dev, "report": ~is_dev}
for half in ["full", "report"]:
    for lab, lo, hi in VB:
        SL[f"{half}|viewer {lab}"] = SL[half] & (vdeg >= lo) & (vdeg < hi)
    for lab, lo, hi in PB:
        SL[f"{half}|pos {lab}"] = SL[half] & (pfol >= lo) & (pfol < hi)


def metrics(r, mask):
    rr = r[mask].astype(np.float64)
    return {"n": int(mask.sum()), "hit6": round(float((rr < 6).mean()), 4), "hit12": round(float((rr < 12).mean()), 4),
            "mrr": round(float((1.0 / (rr + 1)).mean()), 4)}


def table(res):
    return {name: {sl: metrics(r, m) for sl, m in SL.items()} for name, r in res.items()}


# ---- phase 1: t1 + 18 collaborative variants ----
variants = [(d, p, c) for d in DIMS for p in POWERS for c in CENTRED]
srcs = {"t1": SRC_T1}
for d, p, c in variants:
    srcs[f"N[d{d},p{p},{'ctr' if c else 'raw'}]"] = svd_source(item_matrix(d, p, c))
log("built", len(srcs), "sources")
res, cov = run_eval(srcs)
del srcs
log("phase 1 eval done; scorers", len(res))
T = table(res)
rand = {sl: round(float(np.mean(np.minimum(12, n_cand[m] + 1) / (n_cand[m] + 1.0))), 4) for sl, m in SL.items()}

best_v, best_dev = None, -1
var_rows = []
for d, p, c in variants:
    nm = f"N[d{d},p{p},{'ctr' if c else 'raw'}]"
    dv = {k: T[k]["dev"]["hit12"] for k in [nm + ":cos", nm + f":blend@{GATE_BETA}", nm + f":both@{GATE_BETA}"]}
    var_rows.append((nm, dv))
    if max(dv.values()) > best_dev:
        best_dev, best_v = max(dv.values()), (d, p, c, nm)
d_sel, p_sel, c_sel, nm_sel = best_v
log("selected", nm_sel, "dev hit12", best_dev)

# ---- phase 2: quantised chosen variant ----
Uq = quantise(item_matrix(d_sel, p_sel, c_sel))
res2, cov2 = run_eval({"t1": SRC_T1, "N1q": svd_source(Uq)})
T2 = table({k: v for k, v in res2.items() if k.startswith("N1q")})
cov.update({"N1q": cov2["N1q"]})
log("phase 2 eval done")

# ---- gate (report half, beta = engine personalBeta 0.35) ----
base = ["B0", "B1", "B1t", "B2", f"B3@{GATE_BETA}"]
nsel = {"N1": nm_sel + ":cos", "N2": nm_sel + f":blend@{GATE_BETA}", "N3": nm_sel + f":both@{GATE_BETA}"}
nq = {"N1q": "N1q:cos", "N2q": f"N1q:blend@{GATE_BETA}", "N3q": f"N1q:both@{GATE_BETA}"}
ALL = dict(T); ALL.update(T2)
bb = max(base, key=lambda k: ALL[k]["report"]["hit12"])
bn = max(nsel.values(), key=lambda k: ALL[k]["report"]["hit12"])
bb_low = max(ALL[k]["report|viewer 10-49"]["hit12"] for k in base)
ratio = ALL[bn]["report"]["hit12"] / ALL[bb]["report"]["hit12"]
low_ok = ALL[bn]["report|viewer 10-49"]["hit12"] >= bb_low
gate = {"best_baseline": bb, "best_baseline_hit12": ALL[bb]["report"]["hit12"], "best_N": bn, "best_N_hit12": ALL[bn]["report"]["hit12"],
        "ratio": round(ratio, 4), "need_ratio": 1.10, "bucket_10_49_best_N": ALL[bn]["report|viewer 10-49"]["hit12"],
        "bucket_10_49_best_baseline": bb_low, "pass": bool(ratio >= 1.10 and low_ok)}
bnq = nq[{v: k for k, v in nsel.items()}[bn] + "q"]
gate["best_N_quantised"] = bnq
gate["best_N_quantised_hit12"] = ALL[bnq]["report"]["hit12"]
log("GATE", json.dumps(gate))

out = {"selected_variant": {"d": d_sel, "sv_power": p_sel, "centred": c_sel, "name": nm_sel, "dev_hit12": best_dev},
       "gate": gate, "coverage": cov, "random_expected_hit12": rand, "variants_dev_hit12": dict(var_rows),
       "n_candidates_mean": round(float(n_cand.mean()), 2), "table": ALL}
json.dump(out, open(os.path.join(DATA, "eval_results" + TAG + ".json"), "w"), indent=1)

# ---- markdown table ----
rows = [("B0 rankScoreBps", "B0"), ("B1 pool followers", "B1"), ("B1t followers@T", "B1t"), ("B2 t1 cosine", "B2"),
        ("B3 t1 blend b.35", f"B3@{GATE_BETA}"), ("N1 collab cosine", nsel["N1"]), ("N2 collab blend b.35", nsel["N2"]),
        ("N3 t1+collab blend b.35", nsel["N3"]), ("N1q int8", nq["N1q"]), ("N2q int8", nq["N2q"]), ("N3q int8", nq["N3q"])]
cols = ["report", "report|viewer 10-49", "report|viewer 50-199", "report|viewer 200+", "report|pos <1500", "report|pos 1500-20k", "report|pos >=20k", "full"]
L = [f"selected N variant: {nm_sel} (dev hit@12 {best_dev}); events {n_ev}, report half n={SL['report'].sum()}, mean candidates {out['n_candidates_mean']}", "",
     "| scorer | rep hit@6 | rep hit@12 | rep MRR | " + " | ".join(c.replace("report|", "h12 ") for c in cols[1:]) + " |",
     "|---" * (4 + len(cols) - 1) + "|"]
for lab, k in rows:
    t = ALL[k]
    L.append(f"| {lab} | {t['report']['hit6']} | {t['report']['hit12']} | {t['report']['mrr']} | " + " | ".join(str(t[c]['hit12']) for c in cols[1:]) + " |")
L.append(f"| random (expected) | | {rand['report']} | | " + " | ".join(str(rand[c]) for c in cols[1:]) + " |")
L += ["", "slice n: " + ", ".join(f"{c}={SL[c].sum()}" for c in cols), "", "beta sweep (report hit@12):", "",
      "| beta | B3 t1 | N2 collab | N3 both | N2q | N3q |", "|---|---|---|---|---|---|"]
for beta in BETAS:
    L.append(f"| {beta} | {ALL[f'B3@{beta}']['report']['hit12']} | {ALL[nm_sel + f':blend@{beta}']['report']['hit12']} | "
             f"{ALL[nm_sel + f':both@{beta}']['report']['hit12']} | {ALL[f'N1q:blend@{beta}']['report']['hit12']} | {ALL[f'N1q:both@{beta}']['report']['hit12']} |")
L += ["", "variants (dev hit@12: cos / blend.35 / both.35):", ""]
for nm, dv in var_rows:
    L.append(f"- {nm}: " + " / ".join(str(x) for x in dv.values()))
L += ["", "coverage (share of events): " + json.dumps(cov), "", "gate: " + json.dumps(gate)]
open(os.path.join(SCRATCH, "eval_table" + TAG + ".md"), "w").write("\n".join(L) + "\n")
log("done; peak", round(common.peak_gb(), 2), "GB")
