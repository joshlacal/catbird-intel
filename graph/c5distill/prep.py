"""C5 prep (Mac only): filter the Claude-labelled rows, assign the registered splits, write the upload set.

  prep.py --labels ~/catbird-lake-local/claude-labels-20261027 \
          --s5 ~/catbird-lake-local/s5-declarations-20261027 \
          --out ~/catbird-lake-local/c5-distill-20261028/inputs

Writes three parquet files that are the ONLY data a C5 Colab run receives, plus prep_receipt.json:
  c5_pairs.parquet        pair_id, split, cluster (int, dense id of the child author), kind, stratum,
                          parent_text, child_text, child_langs, sonnet_stance, haiku_stance
  c5_profiles.parquet     pid, split, text, label, label_source, sonnet_account_type, haiku_account_type,
                          primary_language
  c5_communities.parquet  community, split, text, label, confidence
No DID, URI or handle leaves the Mac; the cluster ids are a per-run dense rank and are not saved.

Splits (rules_c5.json, S1 9.7): test = ClickHouse cityHash64(key) % 5 == 0, computed by `clickhouse local`
so the hash is ClickHouse's own; val = not test and cityHash64(concat('c5val|', key)) % 8 == 0.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

POSTS_MARK = "Sample recent original posts by members:"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_lines(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    out = set()
    with open(path) as f:
        for line in f:
            s = line.strip().split("\t")[0]
            if s.startswith("did:"):
                out.add(s)
    return out


def ch_splits(keys):
    """{key: 'test'|'val'|'train'} via clickhouse local (ClickHouse cityHash64 is the registered hash)."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "keys.parquet")
        pq.write_table(pa.table({"k": pa.array(sorted(set(keys)), pa.string())}), src)
        q = (f"SELECT k, cityHash64(k) % 5 = 0 AS t, cityHash64(concat('c5val|', k)) % 8 = 0 AS v "
             f"FROM file('{src}', Parquet) FORMAT TSV")
        r = subprocess.run(["clickhouse", "local", "-q", q], capture_output=True, text=True, check=True)
    out = {}
    for line in r.stdout.splitlines():
        k, t, v = line.split("\t")
        out[k] = "test" if t == "1" else ("val" if v == "1" else "train")
    if len(out) != len(set(keys)):
        sys.exit("split hash lost keys")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--s5", required=True)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    L, S5 = os.path.expanduser(A.labels), os.path.expanduser(A.s5)
    os.makedirs(A.out, exist_ok=True)
    f = lambda n: os.path.join(L, n)
    inputs = {n: f(n) for n in ["pair_labels_v2.parquet", "reply_pairs.parquet", "reply_pairs_topup.parquet",
                                "profile_labels.parquet", "profile_inputs.parquet", "community_labels_v2.parquet",
                                "community_prompts.jsonl", "exclude_dids_v2.txt"]}
    inputs["hide_from_recs_union.tsv"] = os.path.join(S5, "hide_from_recs_union.tsv")
    inputs["user_intents_optout_dids.tsv"] = os.path.join(S5, "user_intents_optout_dids.tsv")
    receipt = {"inputs": {n: {"path": p, "sha256": sha256(p)} for n, p in inputs.items()},
               "rules_c5_sha256": sha256(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules_c5.json"))}

    hide = read_lines(inputs["hide_from_recs_union.tsv"]) | read_lines(inputs["exclude_dids_v2.txt"])
    optout = read_lines(inputs["user_intents_optout_dids.tsv"])
    comm = pq.read_table(inputs["community_labels_v2.parquet"]).to_pylist()
    minor_comm = {c["community"] for c in comm if "likely_minors" in (c["sensitivity"] or [])}
    prof_lab = pq.read_table(inputs["profile_labels.parquet"]).to_pylist()
    minors = set()
    for r in prof_lab:
        flags = set(r["haiku_sensitivity"] or []) | set(r["sonnet_sensitivity"] or [])
        if "likely_minors" in flags or r["community"] in minor_comm:
            minors.add(r["did"])
    receipt["exclusion_sets"] = {"hide_from_recs_union": len(hide), "user_intents_optout": len(optout),
                                 "likely_minors_communities": len(minor_comm), "likely_minors_profiles": len(minors)}

    # ---- pairs (posts: A1 applies) ----
    rp = pa.concat_tables([pq.read_table(inputs["reply_pairs.parquet"]),
                           pq.read_table(inputs["reply_pairs_topup.parquet"])]).to_pylist()
    lab = {r["pair_id"]: r for r in pq.read_table(inputs["pair_labels_v2.parquet"]).to_pylist()}
    drop = Counter()
    keep = []
    for r in rp:
        a, b = r["parent_did"], r["child_did"]
        lb = lab.get(r["pair_id"])
        if a in hide or b in hide:
            drop["hide_from_recs_author"] += 1
        elif a in optout or b in optout:
            drop["user_intents_optout"] += 1
        elif a in minors or b in minors:
            drop["likely_minors_author"] += 1
        elif lb is None or lb["sonnet_stance"] is None:
            drop["no_sonnet_label"] += 1
        else:
            keep.append((r, lb))
    sp = ch_splits([r["child_did"] for r, _ in keep])
    cl = {d: i for i, d in enumerate(sorted({r["child_did"] for r, _ in keep}))}
    pairs = pa.table({
        "pair_id": [r["pair_id"] for r, _ in keep],
        "split": [sp[r["child_did"]] for r, _ in keep],
        "cluster": pa.array([cl[r["child_did"]] for r, _ in keep], pa.int32()),
        "kind": [r["kind"] for r, _ in keep],
        "stratum": [r["stratum"] for r, _ in keep],
        "parent_text": [r["parent_text"] or "" for r, _ in keep],
        "child_text": [r["child_text"] or "" for r, _ in keep],
        "child_langs": [r["child_langs"] or "" for r, _ in keep],
        "sonnet_stance": [lb["sonnet_stance"] for _, lb in keep],
        "haiku_stance": [lb["haiku_stance"] for _, lb in keep],
    })
    receipt["pairs"] = {"rows_in": len(rp), "dropped": dict(drop), "rows_out": pairs.num_rows,
                        "split": dict(Counter(pairs["split"].to_pylist())),
                        "hostile_by_split": dict(Counter(s for s, h in zip(
                            pairs["split"].to_pylist(), pairs["sonnet_stance"].to_pylist()) if h == "hostile")),
                        "test_with_haiku": int(pc.sum(pc.and_(pc.equal(pairs["split"], "test"),
                                                              pc.is_valid(pairs["haiku_stance"]))).as_py())}

    # ---- profiles (bios: A1 keeps hide-from-recs bios; minors rule and opt-outs apply) ----
    pin = {r["pid"]: r for r in pq.read_table(inputs["profile_inputs.parquet"]).to_pylist()}
    drop = Counter()
    keep = []
    for r in prof_lab:
        d = r["did"]
        label, src = (r["sonnet_account_type"], "sonnet") if r["sonnet_account_type"] else (r["haiku_account_type"], "haiku")
        if d in optout:
            drop["user_intents_optout"] += 1
        elif d in minors:
            drop["likely_minors"] += 1
        elif not label or label == "null":
            drop["no_label"] += 1
        else:
            keep.append((r, pin[r["pid"]], label, src))
    sp = ch_splits([r["did"] for r, *_ in keep])
    text = lambda i: f"{i['name'] or ''}\n{i['description'] or ''}\ncommunity: {i['community_label'] or ''}"
    profiles = pa.table({
        "pid": [r["pid"] for r, *_ in keep],
        "split": [sp[r["did"]] for r, *_ in keep],
        "text": [text(i) for _, i, *_ in keep],
        "label": [lb for *_, lb, _ in keep],
        "label_source": [s for *_, s in keep],
        "sonnet_account_type": [r["sonnet_account_type"] for r, *_ in keep],
        "haiku_account_type": [r["haiku_account_type"] for r, *_ in keep],
        "primary_language": [r["sonnet_primary_language"] or r["haiku_primary_language"] for r, *_ in keep],
    })
    receipt["profiles"] = {"rows_in": len(prof_lab), "dropped": dict(drop), "rows_out": profiles.num_rows,
                           "hide_from_recs_bios_kept": sum(1 for r, *_ in keep if r["did"] in hide),
                           "split": dict(Counter(profiles["split"].to_pylist())),
                           "test_with_sonnet": sum(1 for r, *_ in keep if sp[r["did"]] == "test" and r["sonnet_account_type"])}

    # ---- communities (Opus evidence, posts section cut) ----
    prompts = {}
    with open(inputs["community_prompts.jsonl"]) as fh:
        for line in fh:
            d = json.loads(line)
            prompts[d["community"]] = d["prompt"].split(POSTS_MARK)[0].rstrip()
    drop = Counter()
    keep = []
    for c in comm:
        if c["community"] in minor_comm:
            drop["likely_minors"] += 1
        elif not c["community_type"] or c["community"] not in prompts:
            drop["no_label_or_prompt"] += 1
        else:
            keep.append(c)
    sp = ch_splits([str(c["community"]) for c in keep])
    communities = pa.table({
        "community": pa.array([c["community"] for c in keep], pa.int64()),
        "split": [sp[str(c["community"])] for c in keep],
        "text": [prompts[c["community"]] for c in keep],
        "label": [c["community_type"] for c in keep],
        "confidence": [c["confidence"] for c in keep],
    })
    receipt["communities"] = {"rows_in": len(comm), "dropped": dict(drop), "rows_out": communities.num_rows,
                              "split": dict(Counter(communities["split"].to_pylist()))}

    outs = {}
    for name, t in [("c5_pairs.parquet", pairs), ("c5_profiles.parquet", profiles), ("c5_communities.parquet", communities)]:
        p = os.path.join(A.out, name)
        pq.write_table(t, p, compression="zstd")
        outs[name] = {"sha256": sha256(p), "bytes": os.path.getsize(p), "rows": t.num_rows}
    receipt["outputs"] = outs
    with open(os.path.join(A.out, "prep_receipt.json"), "w") as fh:
        json.dump(receipt, fh, indent=1)
    print(json.dumps({k: receipt[k] for k in ["exclusion_sets", "pairs", "profiles", "communities", "outputs"]}, indent=1))


if __name__ == "__main__":
    main()
