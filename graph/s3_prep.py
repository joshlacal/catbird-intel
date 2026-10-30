#!/usr/bin/env python3
"""S3 helpers for the whole-network plan: list split, B1 follow-kNN extract, reproduction check.

Plan: docs/execution-plans/2026-10-25-whole-network-signals-plan.md §3.1 S3, §5.4, §6.

  split-lists CURATELISTS_TSV OUT_DIR
      Curated lists split by cityHash64(owner) % 8 (plan §6): {0,1,4,5} A, 2 B1a, 6 B1b, {3,7} B2.
      Writes lists_{A,B1a,B1b,B2}.tsv (input rows unchanged), b1_owners.txt (owner DIDs of
      B1a and B1b lists, the input for `simgraph --drop-users`) and split_receipt.json.
  extract-knn --sim SIM_BIN --producers PRODUCERS_TSV --lists TSV [--lists TSV ...] --out PARQUET [--k 50]
      Top-k neighbours, from a sim.bin, of every list member that is a producer. S3 runs it on
      the B1-owner-excluded follow-kNN before the rotating slot is reused (cluster.py never reads
      that sim.bin). Output columns: pid, did, rank, nbr_pid, nbr_did, cos; plus <out>.json receipt.
  repro --ref DIR --new DIR [--out JSON]
      S3 reproduction check: identical membership.npy, or NMI >= 0.99. Both dirs need
      producers.tsv, membership.npy and communities.parquet.

CURATELISTS_TSV is audit/producer_features.py's curatelists.tsv:
uri \t owner \t name \t total_items \t member_dids (comma-separated producer DIDs).

membership.npy holds raw Leiden labels. communities.parquet and known_for.parquet renumber
communities by size with an unstable sort, so equal-size communities swap ids: never join
membership.npy to community ids directly. `raw_label_by_community` maps each community to the
strict-majority raw label of its top_dids (memory: membership.npy ids differ from communities).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LIST_CLASSES = {0: "A", 1: "A", 4: "A", 5: "A", 2: "B1a", 6: "B1b", 3: "B2", 7: "B2"}
SIM_DTYPE = np.dtype([("p", "<u4"), ("q", "<u4"), ("w", "<f4")])
REPRO_NMI_BAR = 0.99


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def city_hash_mod(values: list[str], mod: int, salt: str = "", clickhouse: str = "clickhouse") -> np.ndarray:
    """ClickHouse `cityHash64(concat(value, salt)) % mod`, so splits match the VM and the fold DB."""
    if not values:
        return np.zeros(0, dtype=np.int64)
    if any(("\t" in v) or ("\n" in v) or ("\\" in v) for v in values):
        raise ValueError("values must not contain tabs, newlines or backslashes")
    query = f"SELECT cityHash64(concat(v, '{salt}')) % {int(mod)} FROM table"
    out = subprocess.run(
        [clickhouse, "local", "--input-format", "TSV", "--structure", "v String", "--query", query],
        input="\n".join(values) + "\n", capture_output=True, text=True, check=True,
    ).stdout.split()
    if len(out) != len(values):
        raise RuntimeError(f"clickhouse returned {len(out)} hashes for {len(values)} values")
    return np.array(out, dtype=np.int64)


def read_producers(path: pathlib.Path) -> tuple[list[str], np.ndarray]:
    dids, followers = [], []
    with open(path) as fh:
        for i, line in enumerate(fh):
            pid, did, n = line.rstrip("\n").split("\t")
            if int(pid) != i:
                raise ValueError(f"{path}: pids must be dense and ordered")
            dids.append(did)
            followers.append(int(n))
    return dids, np.array(followers, dtype=np.int64)


def read_lists(path: pathlib.Path, with_total: bool = False) -> list[tuple]:
    """Rows of a curatelists.tsv: (uri, owner, name, members[, total_items])."""
    rows = []
    with open(path) as fh:
        for line in fh:
            uri, owner, name, total, members = line.rstrip("\n").split("\t")
            row = (uri, owner, name, [m for m in members.split(",") if m])
            rows.append(row + (int(total),) if with_total else row)
    return rows


def split_lists(curatelists: pathlib.Path, out_dir: pathlib.Path, clickhouse: str = "clickhouse") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = curatelists.read_text().splitlines()
    owners = sorted({line.split("\t")[1] for line in lines})
    cls = dict(zip(owners, (LIST_CLASSES[int(h)] for h in city_hash_mod(owners, 8, clickhouse=clickhouse))))
    files = {c: open(out_dir / f"lists_{c}.tsv", "w") for c in ("A", "B1a", "B1b", "B2")}
    counts = {c: 0 for c in files}
    for line in lines:
        c = cls[line.split("\t")[1]]
        files[c].write(line + "\n")
        counts[c] += 1
    for fh in files.values():
        fh.close()
    b1_owners = sorted(o for o, c in cls.items() if c in ("B1a", "B1b"))
    (out_dir / "b1_owners.txt").write_text("".join(o + "\n" for o in b1_owners))
    receipt = {
        "rule": "cityHash64(owner) % 8: {0,1,4,5}=A, 2=B1a, 6=B1b, {3,7}=B2 (plan §6)",
        "input": str(curatelists), "input_sha256": sha256_file(curatelists),
        "lists": counts, "owners": len(owners),
        "owners_by_class": {c: sum(1 for v in cls.values() if v == c) for c in counts},
        "b1_owners": len(b1_owners),
        "outputs": {p.name: sha256_file(p) for p in sorted(out_dir.glob("lists_*.tsv")) + [out_dir / "b1_owners.txt"]},
    }
    (out_dir / "split_receipt.json").write_text(json.dumps(receipt, indent=1))
    return receipt


def extract_knn(sim: pathlib.Path, producers: pathlib.Path, list_files: list[pathlib.Path],
                out: pathlib.Path, k: int = 50) -> dict:
    dids, _followers = read_producers(producers)
    pid_of = {d: i for i, d in enumerate(dids)}
    members = sorted({pid_of[m] for f in list_files for _u, _o, _n, ms in read_lists(f) for m in ms if m in pid_of})
    raw = np.memmap(sim, dtype=SIM_DTYPE, mode="r")
    # raw["p"] is a strided field view; np.searchsorted copies a non-contiguous haystack on
    # every call (~3 GB per member on a full sim.bin), so take one contiguous copy and
    # look all members up at once.
    p_col = np.ascontiguousarray(raw["p"])
    if len(p_col) > 1 and (p_col[1:] < p_col[:-1]).any():
        raise ValueError("sim.bin is not grouped by p in ascending order")
    member_arr = np.array(members, dtype=p_col.dtype)
    los = np.searchsorted(p_col, member_arr, "left")
    his = np.searchsorted(p_col, member_arr, "right")
    del p_col
    cols: dict[str, list] = {"pid": [], "did": [], "rank": [], "nbr_pid": [], "nbr_did": [], "cos": []}
    for pid, lo, hi in zip(members, los.tolist(), his.tolist()):
        rows = np.array(raw[lo:hi])
        if len(rows) and (rows["p"] != pid).any():
            raise ValueError("sim.bin is not grouped by p in ascending order")
        rows = rows[np.lexsort((rows["q"], -rows["w"]))][:k]
        for r, row in enumerate(rows):
            cols["pid"].append(pid)
            cols["did"].append(dids[pid])
            cols["rank"].append(r)
            cols["nbr_pid"].append(int(row["q"]))
            cols["nbr_did"].append(dids[int(row["q"])])
            cols["cos"].append(float(row["w"]))
    table = pa.table({
        "pid": pa.array(cols["pid"], pa.uint32()), "did": pa.array(cols["did"], pa.string()),
        "rank": pa.array(cols["rank"], pa.uint8()), "nbr_pid": pa.array(cols["nbr_pid"], pa.uint32()),
        "nbr_did": pa.array(cols["nbr_did"], pa.string()), "cos": pa.array(cols["cos"], pa.float32()),
    })
    pq.write_table(table, out, compression="zstd")
    receipt = {
        "sim": str(sim), "sim_sha256": sha256_file(sim), "producers_sha256": sha256_file(producers),
        "lists": {str(f): sha256_file(f) for f in list_files}, "k": k,
        "list_member_producers": len(members), "rows": table.num_rows,
        "members_without_neighbours": len(members) - len(set(cols["pid"])),
        "out_sha256": sha256_file(out),
    }
    pathlib.Path(str(out) + ".json").write_text(json.dumps(receipt, indent=1))
    return receipt


def raw_label_by_community(communities: pa.Table, membership: np.ndarray, pid_of: dict[str, int]) -> dict[int, int]:
    """community id -> raw Leiden label: the strict majority of its top_dids' membership.npy labels."""
    out: dict[int, int] = {}
    seen: dict[int, int] = {}
    for c, tops in zip(communities["community"].to_pylist(), communities["top_dids"].to_pylist()):
        labels = np.array([membership[pid_of[d]] for d in tops if d in pid_of], dtype=np.int64)
        if not len(labels):
            raise ValueError(f"community {c}: no top_dids in producers.tsv")
        vals, counts = np.unique(labels, return_counts=True)
        best = int(np.argmax(counts))
        if counts[best] * 2 <= len(labels):
            raise ValueError(f"community {c}: no strict-majority raw label among its top_dids")
        label = int(vals[best])
        if label in seen:
            raise ValueError(f"communities {seen[label]} and {c} map to the same raw label {label}")
        out[int(c)] = label
        seen[label] = int(c)
    return out


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized mutual information with the arithmetic-mean normalisation (sklearn's default)."""
    a = np.unique(np.asarray(a), return_inverse=True)[1].astype(np.int64)
    b = np.unique(np.asarray(b), return_inverse=True)[1].astype(np.int64)
    n = len(a)
    _keys, joint = np.unique(a * (int(b.max()) + 1) + b, return_counts=True)
    ha, hb, hab = entropy(np.bincount(a)), entropy(np.bincount(b)), entropy(joint)
    if ha == 0.0 and hb == 0.0:
        return 1.0
    mi = ha + hb - hab
    return float(max(0.0, min(1.0, mi / ((ha + hb) / 2)))) if n else 1.0


def repro(ref: pathlib.Path, new: pathlib.Path) -> dict:
    ref_dids, _ = read_producers(ref / "producers.tsv")
    new_dids, _ = read_producers(new / "producers.tsv")
    if ref_dids != new_dids:
        raise SystemExit("producers.tsv differs: a reproduction must use the same producers")
    pid_of = {d: i for i, d in enumerate(ref_dids)}
    m_ref, m_new = np.load(ref / "membership.npy"), np.load(new / "membership.npy")
    identical = bool(m_ref.shape == m_new.shape and np.array_equal(m_ref, m_new))
    score = 1.0 if identical else nmi(m_ref, m_new)
    c_ref, c_new = pq.read_table(ref / "communities.parquet"), pq.read_table(new / "communities.parquet")
    lab_ref = raw_label_by_community(c_ref, m_ref, pid_of)
    lab_new = raw_label_by_community(c_new, m_new, pid_of)
    by_label_new = {lab: c for c, lab in lab_new.items()}
    # A reference community is reproduced if its raw member set is exactly one new raw community.
    order = np.argsort(m_ref, kind="stable")
    bounds = np.searchsorted(m_ref[order], np.arange(int(m_ref.max()) + 2))
    new_sizes = np.bincount(m_new)
    same_members = same_id = 0
    for c, lab in lab_ref.items():
        members = order[bounds[lab]:bounds[lab + 1]]
        labs = np.unique(m_new[members])
        if len(labs) == 1 and new_sizes[labs[0]] == len(members):
            same_members += 1
            same_id += int(by_label_new.get(int(labs[0])) == c)
    return {
        "ref": str(ref), "new": str(new),
        "ref_membership_sha256": sha256_file(ref / "membership.npy"),
        "new_membership_sha256": sha256_file(new / "membership.npy"),
        "identical_membership": identical, "nmi_arithmetic": score, "nmi_bar": REPRO_NMI_BAR,
        "pass": bool(identical or score >= REPRO_NMI_BAR),
        "ref_communities": len(lab_ref),
        "communities_same_members": same_members,
        "communities_same_members_and_same_id": same_id,
        "note": "ids are compared through raw labels; equal-size communities may swap ids between runs",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split-lists")
    s.add_argument("curatelists", type=pathlib.Path)
    s.add_argument("out_dir", type=pathlib.Path)
    s.add_argument("--clickhouse", default="clickhouse")
    k = sub.add_parser("extract-knn")
    k.add_argument("--sim", type=pathlib.Path, required=True)
    k.add_argument("--producers", type=pathlib.Path, required=True)
    k.add_argument("--lists", type=pathlib.Path, action="append", required=True)
    k.add_argument("--out", type=pathlib.Path, required=True)
    k.add_argument("--k", type=int, default=50)
    r = sub.add_parser("repro")
    r.add_argument("--ref", type=pathlib.Path, required=True)
    r.add_argument("--new", type=pathlib.Path, required=True)
    r.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    if args.cmd == "split-lists":
        result = split_lists(args.curatelists, args.out_dir, args.clickhouse)
    elif args.cmd == "extract-knn":
        result = extract_knn(args.sim, args.producers, args.lists, args.out, args.k)
    else:
        result = repro(args.ref, args.new)
        if args.out:
            args.out.write_text(json.dumps(result, indent=1))
    json.dump(result, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
