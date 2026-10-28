"""Merge the second-round Claude batches into new label files (originals left untouched)."""
import collections, hashlib, json, pathlib
import pyarrow as pa, pyarrow.parquet as pq

W = pathlib.Path.home() / "catbird-lake-local/claude-labels-20261027"
B = W / "batches"
CH, TOP, PS, H1, H2 = (B / f"msgbatch_{x}.results.jsonl" for x in (
    "0159wMdtdktGpvZdVYTx6eAi", "01T2zCyoXrh5pLJnc3XdU1hz", "01RtxGLjZgbuVyYTvmrmugLM",
    "01RJqoQBGCqVBGVovnMPta3s", "01Um2ucNhQgCmVpwS6MLF2iF"))

def rows(p):
    for l in open(p):
        r = json.loads(l)
        yield r

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

rep = {}
# communities: Opus high-effort relabels replace the low-confidence rows
old = pq.read_table(W / "community_labels.parquet").to_pylist()
new = {int(r["custom_id"][1:]): r for r in rows(CH) if r["type"] == "succeeded"}
lowconf = {int(x) for x in (W / "lowconf_communities.txt").read_text().split()}
assert set(new) <= lowconf, "relabel outside low-confidence set"
conf_before = collections.Counter(r["confidence"] for r in old if r["community"] in new)
out, changed = [], collections.Counter()
for r in old:
    r = dict(r, relabelled_high_effort=False)
    if r["community"] in new:
        n = new[r["community"]]["label"]
        for k in ("onboarding_fit", "community_type", "confidence"):
            changed[k] += n[k] != r[k]
        r.update(n, model=new[r["community"]]["model"], relabelled_high_effort=True)
    out.append(r)
pq.write_table(pa.Table.from_pylist(out), W / "community_labels_v2.parquet")
rep["communities"] = {
    "rows": len(out), "relabelled": len(new), "lowconf_listed": len(lowconf),
    "confidence_before": dict(conf_before),
    "confidence_after": dict(collections.Counter(new[c]["label"]["confidence"] for c in new)),
    "changed_fields": dict(changed),
    "fit_all_after": dict(collections.Counter(r["onboarding_fit"] for r in out)),
    "type_all_after": dict(collections.Counter(r["community_type"] for r in out)),
    "likely_minors_after": sum("likely_minors" in (r["sensitivity"] or []) for r in out),
}

# pairs: append Sonnet top-up labels (no Haiku labels for the top-up)
old = pq.read_table(W / "pair_labels.parquet").to_pylist()
top = [r for r in rows(TOP) if r["type"] == "succeeded"]
for r in top:
    l = r["label"]
    old.append({"pair_id": r["custom_id"], "sonnet_stance": l["stance"], "sonnet_dunk": l["dunk"],
                "sonnet_target": l["target"], "sonnet_confidence": l["confidence"],
                "haiku_stance": None, "haiku_dunk": None, "haiku_target": None, "haiku_confidence": None})
pq.write_table(pa.Table.from_pylist(old), W / "pair_labels_v2.parquet")
tp = {r["pair_id"]: r for r in pq.read_table(W / "reply_pairs_topup.parquet").to_pylist()}
by = collections.defaultdict(collections.Counter)
for r in top:
    p = tp[r["custom_id"]]
    by[f'{p["stratum"]}/{p["kind"]}'][r["label"]["stance"]] += 1
rep["pairs"] = {"rows": len(old), "topup_labelled": len(top), "topup_inputs": len(tp),
                "topup_stance_by_stratum": {k: dict(v) for k, v in by.items()}}

# profiles: Haiku for all, Sonnet for the 4,000 check sample
inp = {r["pid"]: r for r in pq.read_table(W / "profile_inputs.parquet", columns=["pid", "did", "community"]).to_pylist()}
hk, refused = {}, []
for p in (H1, H2):
    for r in rows(p):
        if r["type"] == "succeeded" and r.get("label"):
            hk[r["custom_id"]] = r["label"]
        else:
            refused.append((r["custom_id"], r["type"], r.get("stop_reason")))
sn = {r["custom_id"]: r["label"] for r in rows(PS) if r["type"] == "succeeded"}
F = ("account_type", "primary_language", "topics", "sensitivity", "confidence")
prof = []
for pid, b in inp.items():
    h, s = hk.get(pid), sn.get(pid)
    row = dict(b)
    for pre, l in (("haiku", h), ("sonnet", s)):
        for f in F:
            row[f"{pre}_{f}"] = l.get(f) if l else None
    prof.append(row)
pq.write_table(pa.Table.from_pylist(prof), W / "profile_labels.parquet")
both = [p for p in prof if p["sonnet_account_type"] and p["haiku_account_type"]]
agree = lambda f: sum(p[f"haiku_{f}"] == p[f"sonnet_{f}"] for p in both) / len(both)
sens = lambda p, pre: set(p[f"{pre}_sensitivity"] or [])
conf = collections.Counter((p["sonnet_account_type"], p["haiku_account_type"]) for p in both
                           if p["sonnet_account_type"] != p["haiku_account_type"])
def flag_stats(cls):
    s = [cls in sens(p, "sonnet") for p in both]; h = [cls in sens(p, "haiku") for p in both]
    tp_ = sum(a and b for a, b in zip(s, h))
    return {"sonnet": sum(s), "haiku": sum(h), "both": tp_,
            "haiku_recall_vs_sonnet": round(tp_ / max(1, sum(s)), 3),
            "haiku_precision_vs_sonnet": round(tp_ / max(1, sum(h)), 3)}
allsens = collections.Counter(c for p in both for c in sens(p, "sonnet") | sens(p, "haiku"))
rep["profiles"] = {
    "inputs": len(inp), "haiku_labelled": len(hk), "haiku_not_labelled": refused,
    "sonnet_labelled": len(sn), "both": len(both),
    "agree_account_type": round(agree("account_type"), 3),
    "agree_primary_language": round(agree("primary_language"), 3),
    "agree_confidence": round(agree("confidence"), 3),
    "top_account_type_disagreements": [[f"{a} (sonnet) vs {b} (haiku)", n] for (a, b), n in conf.most_common(8)],
    "sensitivity_flags": {c: flag_stats(c) for c, _ in allsens.most_common(12)},
    "haiku_account_type_all": dict(collections.Counter(p["haiku_account_type"] for p in prof)),
    "haiku_likely_minors_all": sum("likely_minors" in (p["haiku_sensitivity"] or []) for p in prof),
}
cost = 0.0
for p in (CH, TOP, PS, H1, H2):
    u = json.loads(p.with_name(p.name.replace(".results.jsonl", ".usage.json")).read_text())
    cost += u["cost"]
rep["round2_cost_usd"] = round(cost, 2)
rep["sha256"] = {f: sha(W / f) for f in ("community_labels_v2.parquet", "pair_labels_v2.parquet", "profile_labels.parquet")}
(W / "round2_merge_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
print(json.dumps(rep, indent=1, ensure_ascii=False))
