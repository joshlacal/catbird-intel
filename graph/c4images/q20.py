"""Q20 / C4 power floor: how many B1b lists have >=50% image-heavy members (pre-registration §11).

  q20.py OUT_DIR [--account-labels TSV]

Window: the four netcup export days 2026-10-19..22 (the S4 corpus window), every account, not just the
S4 cap-32 sample. Originals = posts that are not replies (quotes included), latest revision, not
deleted, not in that day's deletions ledger, exactly as S4 prep_corpus.py counts them (it reproduces
S4's 377,304 producers / 3,528,877 originals). Image original = image_count >= 1 (`images` embeds and
recordWithMedia images; video and `gallery` embeds are not images here).

Definitions (logged as Yellow in docs/DECISIONS.md; the frozen rules give the 50% thresholds only):
  member       a distinct DID on the list that is a producer (graph-build-v3 producers.tsv), as R11's
               ">=3 members that are producers in the evaluated universe";
  known        a member with >= K originals in the window (K = 5 primary; 3 and 10 reported);
               members below K are unknown, never text-only;
  image-heavy  a known member with >= 50% image originals (plan §5.3 "≥50% image originals");
  qualifies    a B1b evaluation list (S4 load_lists filter: dedupe, <=3 lists per owner with
               default_rng(7), <=500 members, cityHash64(owner) % 8 == 6) with >= 3 known members of
               which >= 50% are image-heavy. Sensitivity: 40% / 60% (reported only), and a stricter
               denominator counting unknown members as not image-heavy.
Fetch coverage: image-heavy members of qualifying lists that pass the gate's DID-level exclusions
(gate.load_did_exclusions), against the pilot's ~12,500 producers (100k images, <=8 each).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gate  # noqa: E402

LAKE = pathlib.Path.home() / "catbird-lake-local"
LISTS = LAKE / "communities-cand1-hubdemote/audit-quality/curatelists.tsv"
LISTS_SHA = "11099805e1ec25a78ee2b0cde2450b3074a3d865daf053863fe39d6eee607113"
SPLIT = LAKE / "s4-pilot-20261028/inputs/list_owner_split.tsv"
PRODUCERS = LAKE / "graph-build-v3/out-full-20261024/producers.tsv"
NETCUP = LAKE / "netcup-export"
DAYS = "2026-10-{19,20,21,22}"
B1B = 6
KS = (3, 5, 10)
K_PRIMARY = 5
LIST_THRESHOLDS = (0.4, 0.5, 0.6)
PILOT_PRODUCERS = 100_000 // 8

COUNTS_SQL = f"""
WITH p AS (
  SELECT did, rkey, argMax(is_deleted, rev) AS del, argMax(reply_parent, rev) AS rp, argMax(image_count, rev) AS ic
  FROM file('{NETCUP}/{DAYS}/posts.parquet', Parquet,
            'did String, rkey String, rev UInt64, is_deleted UInt8, reply_parent String, image_count UInt8')
  GROUP BY did, rkey
), d AS (
  SELECT DISTINCT did, rkey FROM file('{NETCUP}/{DAYS}/deletions.parquet', Parquet,
                                      'collection String, did String, rkey String')
  WHERE collection = 'app.bsky.feed.post'
)
SELECT did, count() AS originals, countIf(ic >= 1) AS image_originals
FROM p LEFT ANTI JOIN d USING (did, rkey)
WHERE del = 0 AND rp = ''
GROUP BY did ORDER BY did
"""


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_lists():
    """S4 score.py load_lists, B1b only (same filter, same rng, same order)."""
    assert sha256(LISTS) == LISTS_SHA
    rng = np.random.default_rng(7)
    rows = []
    for line in open(LISTS):
        uri, owner, name, total, members = line.rstrip("\n").split("\t")
        rows.append((uri, owner, name, sorted(set(m for m in members.split(",") if m)), int(total)))
    rows.sort()
    seen, by_owner = set(), collections.defaultdict(list)
    for r in rows:
        key = tuple(r[3])
        if len(key) >= 3 and key not in seen:
            seen.add(key)
            by_owner[r[1]].append(r)
    lists = []
    for owner, rs in by_owner.items():
        lists += [rs[i] for i in sorted(rng.choice(len(rs), min(3, len(rs)), replace=False))]
    bucket = dict(l.rstrip("\n").split("\t") for l in open(SPLIT))
    return [(uri, name, members) for uri, owner, name, members, total in lists
            if max(total, len(members)) <= 500 and int(bucket[owner]) == B1B]


def member_counts(out_dir: pathlib.Path):
    path = out_dir / "member_image_counts.tsv"
    if not path.exists():
        tmp = path.with_suffix(".tsv.partial")
        with open(tmp, "w") as f:
            subprocess.run(["clickhouse", "local", "--max_memory_usage", "6000000000", "--max_threads", "6",
                            "--query", COUNTS_SQL + " FORMAT TSV"], stdout=f, check=True)
        tmp.replace(path)
    counts = {}
    for line in open(path):
        d, o, i = line.rstrip("\n").split("\t")
        counts[d] = (int(o), int(i))
    return counts, path


def evaluate(lists, producers, counts, k, eligible=None):
    """Per list: (known members, image-heavy members, producer members). eligible restricts members."""
    out = []
    for uri, name, members in lists:
        mem = [m for m in members if m in producers and (eligible is None or m in eligible)]
        known = [m for m in mem if counts.get(m, (0, 0))[0] >= k]
        heavy = [m for m in known if counts[m][1] / counts[m][0] >= 0.5]
        out.append((uri, name, mem, known, heavy))
    return out


def summarise(ev, threshold, min_known=3, strict_denominator=False):
    q = [e for e in ev if len(e[3]) >= min_known and len(e[2]) >= 3
         and len(e[4]) / (len(e[2]) if strict_denominator else len(e[3])) >= threshold]
    return q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--account-labels", help="labels.py TSV for the fetch-coverage exclusions")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lists = load_lists()
    producers = {l.split("\t")[1] for l in open(PRODUCERS)}
    counts, counts_path = member_counts(out)
    res = {"window": "netcup export 2026-10-19..22 (4 days), all accounts", "b1b_eval_lists": len(lists),
           "b1b_lists_ge3_producer_members": sum(1 for _u, _n, m in lists if sum(x in producers for x in m) >= 3),
           "accounts_with_originals": len(counts), "by_k": {}}
    for k in KS:
        ev = evaluate(lists, producers, counts, k)
        r = {"lists_ge3_known": sum(1 for e in ev if len(e[3]) >= 3 and len(e[2]) >= 3)}
        for t in LIST_THRESHOLDS:
            q = summarise(ev, t)
            qs = summarise(ev, t, strict_denominator=True)
            r[f"qualifying_at_{int(t * 100)}"] = len(q)
            r[f"qualifying_at_{int(t * 100)}_unknown_as_text"] = len(qs)
            if t == 0.5:
                prod = {m for e in q for m in e[2]}
                known = {m for e in q for m in e[3]}
                heavy = {m for e in q for m in e[4]}
                origs = np.array([counts[m][0] for m in known]) if known else np.zeros(1)
                r["at_50"] = {
                    "lists": len(q), "distinct_producer_members": len(prod), "distinct_known_members": len(known),
                    "distinct_image_heavy_members": len(heavy),
                    "member_slots_known": sum(len(e[3]) for e in q),
                    "originals_per_known_member": {"median": float(np.median(origs)), "p10": float(np.percentile(origs, 10)),
                                                   "p90": float(np.percentile(origs, 90)), "min": int(origs.min())},
                    "known_share_of_producer_members_median": float(np.median([len(e[3]) / len(e[2]) for e in q])) if q else None,
                    "list_names_sample": sorted({e[1] for e in q})[:40]}
                if k == K_PRIMARY:
                    res["_primary"] = (q, heavy)
        res["by_k"][str(k)] = r
    q, heavy = res.pop("_primary")
    excl = gate.load_did_exclusions(account_labels_tsv=a.account_labels)
    reasons = collections.Counter(excl.reason(m) or "eligible" for m in heavy)
    eligible_heavy = {m for m in heavy if excl.reason(m) is None}
    ev_elig = evaluate(lists, producers, counts, K_PRIMARY, eligible={m for m in producers if excl.reason(m) is None})
    res["fetch_coverage_k5_at_50"] = {
        "image_heavy_members": len(heavy), "by_first_exclusion_reason": dict(reasons),
        "fetch_eligible_image_heavy": len(eligible_heavy), "pilot_producer_capacity": PILOT_PRODUCERS,
        "pilot_covers_all_eligible": len(eligible_heavy) <= PILOT_PRODUCERS,
        "lists_qualifying_on_eligible_members_only": len(summarise(ev_elig, 0.5)),
        "did_exclusion_sizes": excl.sizes(),
        "note": "pre-live: CDN status at fetch time and post-level labels/self-labels/age are not applied here"}
    res["k_primary"] = K_PRIMARY
    res["floor"] = 30
    res["passes_floor"] = res["by_k"][str(K_PRIMARY)]["qualifying_at_50"] >= 30
    res["inputs_sha256"] = {str(p): sha256(p) for p in [LISTS, SPLIT, PRODUCERS, counts_path]
                            + ([pathlib.Path(a.account_labels)] if a.account_labels else [])}
    with open(out / "q20_lists_k5_at50.tsv", "w") as f:
        for uri, name, mem, known, hv in q:
            f.write(f"{uri}\t{name}\t{len(mem)}\t{len(known)}\t{len(hv)}\n")
    (out / "q20.json").write_text(json.dumps(res, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "inputs_sha256"}, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
