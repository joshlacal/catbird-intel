"""Step 1: build the eval events, the eval viewers' train follows, and the SVD training sample.

Reads (read-only): train_T/edges.bin, producers.tsv, follower_weights.tsv, test_follows.parquet, the pool JSON.
Writes: $SCRATCH/data/{pool.npz, degrees.npy, events.npz, viewer_follows.npz, train_csr.npz, prep_stats.json}

usage: python 01_prep.py [train_nnz_target=600000000]
"""
import json
import os
import sys

import common
from common import DATA, DAY_US, SEED, log

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

TRAIN_NNZ_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000_000
MIN_TRAIN_FOLLOWS = 10
BURST_MIN = 10  # graph/eval_temporal.py:91,334 (burst_b >= 10 is a bulk follow)
DELETE_WINDOW_US = 14 * DAY_US
MAX_EVENTS = 150_000
MAX_PER_SCENE = 300
TRAIN_MIN_DEG, TRAIN_MAX_DEG = 10, 3000
CHUNK_PAIRS = 250_000_000

os.makedirs(DATA, exist_ok=True)
stats = {}


def read_pairs(f, a, b):
    """(uid, pid) pairs a..b of edges.bin via pread (keeps the 16 GB file out of RSS)."""
    f.seek(8 * int(a))
    return np.fromfile(f, dtype=np.uint32, count=2 * int(b - a)).reshape(-1, 2)


# ---- producers at T -------------------------------------------------------------------------------
prod = pacsv.read_csv(common.PRODUCERS, read_options=pacsv.ReadOptions(column_names=["pid", "did", "followers"]),
                      parse_options=pacsv.ParseOptions(delimiter="\t"),
                      convert_options=pacsv.ConvertOptions(column_types={"pid": pa.uint32(), "did": pa.string(), "followers": pa.int64()}))
assert np.array_equal(prod["pid"].to_numpy(), np.arange(prod.num_rows)), "producers.tsv pid != row"
n_prod = prod.num_rows
prod_did = prod["did"].combine_chunks()
log("producers", n_prod)

# ---- pool -----------------------------------------------------------------------------------------
pool = json.load(open(common.POOL))
m_did, m_scene, m_fol, m_rank, scene_ids = [], [], [], [], []
for s, c in enumerate(pool["communities"]):
    scene_ids.append(int(c["communityId"]))
    for m in c["members"]:
        m_did.append(m["did"]); m_scene.append(s); m_fol.append(int(m["followers"])); m_rank.append(int(m["rankScoreBps"]))
m_did_arr = pa.array(m_did, pa.string())
m_scene = np.asarray(m_scene, np.int32)
assert np.all(np.diff(m_scene) >= 0)
m_pid = pc.fill_null(pc.index_in(m_did_arr, value_set=prod_did), -1).to_numpy().astype(np.int64)
scene_off = np.zeros(len(scene_ids) + 1, np.int64)
np.add.at(scene_off, m_scene + 1, 1)
scene_off = np.cumsum(scene_off)
np.savez(os.path.join(DATA, "pool.npz"), did=np.asarray(m_did), scene=m_scene, followers=np.asarray(m_fol, np.int64),
         rank_bps=np.asarray(m_rank, np.int64), pid_t=m_pid, scene_off=scene_off, scene_ids=np.asarray(scene_ids, np.int64))
stats["pool_members"] = len(m_did); stats["pool_scenes"] = len(scene_ids)
stats["pool_members_without_pid_t"] = int((m_pid < 0).sum())
log("pool members", len(m_did), "scenes", len(scene_ids), "no pid_t", stats["pool_members_without_pid_t"])

# ---- degrees (pass 1 over edges.bin; also checks uid order) -------------------------------------------
n_pairs = os.path.getsize(common.EDGES_BIN) // 8
deg = np.zeros(40_000_000, np.int64)
last_u = 0
with open(common.EDGES_BIN, "rb") as f:
    for a in range(0, n_pairs, CHUNK_PAIRS):
        u = read_pairs(f, a, min(a + CHUNK_PAIRS, n_pairs))[:, 0]
        assert u[0] >= last_u and np.all(u[1:] >= u[:-1]), "edges.bin not sorted by uid"
        last_u = int(u[-1])
        bc = np.bincount(u)
        deg[: len(bc)] += bc
        del u
n_users = last_u + 1
deg = deg[:n_users].copy()
indptr = np.zeros(n_users + 1, np.int64); np.cumsum(deg, out=indptr[1:])
assert indptr[-1] == n_pairs
np.save(os.path.join(DATA, "degrees.npy"), deg)
stats["train_edges"] = int(n_pairs); stats["train_users"] = int(n_users)
log("pass1 degrees users", n_users, "edges", n_pairs)

# ---- test follows -> candidate events --------------------------------------------------------------
pf = pq.ParquetFile(common.TEST_FOLLOWS)
parts = []
tot = n_pool = n_pool_nb = 0
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=["did", "subject", "c_us", "burst_b", "del_us"])
    tot += t.num_rows
    midx = pc.index_in(t["subject"], value_set=m_did_arr)
    keep = pc.is_valid(midx)
    n_pool += int(pc.sum(keep).as_py() or 0)
    keep = pc.and_(keep, pc.less(t["burst_b"], BURST_MIN))
    n_pool_nb += int(pc.sum(keep).as_py() or 0)
    c_us = pc.cast(t["c_us"], pa.int64()); del_us = pc.cast(t["del_us"], pa.int64())
    alive = pc.or_(pc.equal(del_us, 0), pc.greater_equal(pc.subtract(del_us, c_us), DELETE_WINDOW_US))
    keep = pc.and_(keep, alive)
    parts.append(pa.table({"did": t["did"], "m": midx, "c_us": c_us}).filter(keep))
ev = pa.concat_tables(parts)
stats.update(test_follows_rows=tot, test_follows_to_pool=n_pool, to_pool_nonburst=n_pool_nb, to_pool_nonburst_alive14d=ev.num_rows)
log("test follows", tot, "to pool", n_pool, "non-burst", n_pool_nb, "& not deleted <14d", ev.num_rows)

fw = pacsv.read_csv(common.FOLLOWER_WEIGHTS, read_options=pacsv.ReadOptions(column_names=["uid", "did", "is_stale", "weight"]),
                    parse_options=pacsv.ParseOptions(delimiter="\t"),
                    convert_options=pacsv.ConvertOptions(include_columns=["uid", "did"], column_types={"uid": pa.int64(), "did": pa.string()}))
fw_uid = fw["uid"].to_numpy()
assert fw.num_rows == n_users and fw_uid.max() == n_users - 1
uidx = pc.index_in(ev["did"], value_set=fw["did"].combine_chunks())
del fw
has_uid = pc.is_valid(uidx).to_numpy(zero_copy_only=False)
e_uid = fw_uid[pc.fill_null(uidx, 0).to_numpy()]
e_m = ev["m"].to_numpy(zero_copy_only=False).astype(np.int64)
e_c = ev["c_us"].to_numpy(zero_copy_only=False).astype(np.int64)
del ev, fw_uid
stats["events_viewer_not_in_train"] = int((~has_uid).sum())
ok = has_uid & (deg[e_uid] >= MIN_TRAIN_FOLLOWS)
stats["events_viewer_lt10_train_follows"] = int(has_uid.sum() - ok.sum())
e_uid, e_m, e_c = e_uid[ok], e_m[ok], e_c[ok]
o = np.lexsort((e_c, e_m, e_uid))
e_uid, e_m, e_c = e_uid[o], e_m[o], e_c[o]
first = np.ones(len(e_uid), bool); first[1:] = (e_uid[1:] != e_uid[:-1]) | (e_m[1:] != e_m[:-1])
stats["events_dup_viewer_member_dropped"] = int((~first).sum())
e_uid, e_m, e_c = e_uid[first], e_m[first], e_c[first]
log("events after viewer>=10 + dedupe", len(e_uid))

# ---- pass 2: drop positives already followed at T, extract the SVD training sample -------------------
rng = np.random.default_rng(SEED)
elig = (deg >= TRAIN_MIN_DEG) & (deg <= TRAIN_MAX_DEG)
elig_nnz = int(deg[elig].sum())
p_keep = min(1.0, TRAIN_NNZ_TARGET / max(1, elig_nnz))
sel = elig & (rng.random(n_users) < p_keep)
qs = np.percentile(deg[deg > 0], [10, 25, 50, 75, 90, 99, 99.9])
stats.update(deg_percentiles_10_25_50_75_90_99_999=[int(x) for x in qs], train_eligible_viewers=int(elig.sum()),
             train_eligible_nnz=elig_nnz, train_sample_p=p_keep, train_sample_viewers=int(sel.sum()),
             train_sample_nnz=int(deg[sel].sum()))
log("deg pct", qs, "train sample viewers", int(sel.sum()), "nnz", int(deg[sel].sum()), "p", round(p_keep, 4))

ev_pid = m_pid[e_m]
key_ok = ev_pid >= 0
ev_key = (e_uid.astype(np.uint64) << np.uint64(32)) | ev_pid.clip(0).astype(np.uint64)
ks = np.sort(ev_key[key_ok])
in_train_sorted = np.zeros(len(ks), bool)
viewer_flag = np.zeros(n_users, bool); viewer_flag[e_uid] = True

tr_idx = np.empty(int(deg[sel].sum()), np.int32)
fill = 0
with open(common.EDGES_BIN, "rb") as f:
    for a in range(0, n_pairs, CHUNK_PAIRS):
        b = min(a + CHUNK_PAIRS, n_pairs)
        blk = read_pairs(f, a, b)
        u = blk[:, 0]; p = blk[:, 1]
        ms = sel[u]
        k = int(ms.sum()); tr_idx[fill: fill + k] = p[ms]; fill += k
        mv = viewer_flag[u]
        kk = (u[mv].astype(np.uint64) << np.uint64(32)) | p[mv].astype(np.uint64)
        pos = np.searchsorted(ks, kk).clip(0, len(ks) - 1)
        hit = ks[pos] == kk
        in_train_sorted[pos[hit]] = True
        del blk, u, p, ms, mv, kk, pos, hit
        log("pass2", b, "/", n_pairs)
assert fill == len(tr_idx)
followed = np.zeros(len(e_uid), bool)
followed[np.flatnonzero(key_ok)] = in_train_sorted[np.searchsorted(ks, ev_key[key_ok])]
stats["events_positive_already_followed_at_T"] = int(followed.sum())
e_uid, e_m, e_c = e_uid[~followed], e_m[~followed], e_c[~followed]
stats["eligible_events"] = int(len(e_uid)); stats["eligible_viewers"] = int(len(np.unique(e_uid)))
log("eligible events", len(e_uid), "viewers", stats["eligible_viewers"])

sel_uid = np.flatnonzero(sel)
tr_indptr = np.zeros(len(sel_uid) + 1, np.int64); np.cumsum(deg[sel_uid], out=tr_indptr[1:])
np.savez(os.path.join(DATA, "train_csr.npz"), indptr=tr_indptr, indices=tr_idx, uids=sel_uid, n_cols=np.int64(n_prod))
del tr_idx

# ---- sample events: cap per scene, then overall cap, fixed seed --------------------------------------
perm = rng.permutation(len(e_uid))
sc = m_scene[e_m[perm]]
o = np.argsort(sc, kind="stable")  # within a scene keeps the permuted order
sc_sorted = sc[o]
rank_in_scene = np.arange(len(o)) - np.searchsorted(sc_sorted, sc_sorted, side="left")
keep_pos = np.zeros(len(perm), bool); keep_pos[o[rank_in_scene < MAX_PER_SCENE]] = True
chosen = np.sort(perm[keep_pos][:MAX_EVENTS])
e_uid, e_m, e_c = e_uid[chosen], e_m[chosen], e_c[chosen]
stats["sampled_events"] = int(len(e_uid)); stats["sampled_viewers"] = int(len(np.unique(e_uid)))
stats["sampled_scenes"] = int(len(np.unique(m_scene[e_m])))
log("sampled events", len(e_uid), "viewers", stats["sampled_viewers"], "scenes", stats["sampled_scenes"])
np.savez(os.path.join(DATA, "events.npz"), uid=e_uid, member=e_m, c_us=e_c)

# ---- eval viewers' full train follows (random-access slices) -----------------------------------------
vu = np.unique(e_uid)
v_indptr = np.zeros(len(vu) + 1, np.int64); np.cumsum(deg[vu], out=v_indptr[1:])
v_pids = np.empty(v_indptr[-1], np.int32)
with open(common.EDGES_BIN, "rb") as f:
    for i, u in enumerate(vu):
        blk = read_pairs(f, indptr[u], indptr[u + 1])
        v_pids[v_indptr[i]: v_indptr[i + 1]] = blk[:, 1]
np.savez(os.path.join(DATA, "viewer_follows.npz"), uids=vu, indptr=v_indptr, pids=v_pids)
stats["viewer_follow_edges"] = int(v_indptr[-1])
stats["peak_rss_gb"] = round(common.peak_gb(), 2)
json.dump(stats, open(os.path.join(DATA, "prep_stats.json"), "w"), indent=1)
log("done", json.dumps(stats))
