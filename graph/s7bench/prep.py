"""S7 inputs: 100k random posts and 10k slab-format rows from S4's filtered corpus, text only + a row id.

  prep.py OUT_DIR [--no-fetch]          (run with ~/catbird-lake-local/s4-venv/bin/python -I; read-only on inputs)

Whole-network plan §3.1 row S7 and §5.1; frozen rules 2026-10-28-whole-network-preregistration.md (amendment A1:
hide-from-recs accounts' POSTS are excluded). Start: S4's posts_corpus.parquet (four-day originals, non-empty text,
<=32 per producer, already without hide-from-recs union, visibility hide_from_algorithmic, user-intents opt-outs and
likely-minors-home producers). Re-applied here, with counts, before any draw:
  hide_from_recs_union       S5 declarations union (sha-pinned, the file S4 used)
  user_intents_optout        any DID whose org.user-intents.declaration has bulkDataset:false or
                             syntheticContentGeneration:false (from the records, not only S4's derived list)
  s5_purge                   ~/catbird-lake-local/s5-sweep-20261028/purge_dids.txt (inactive/absent repos)
  likely_minors              S4's rule (rank-0 home in a likely_minors community = S4 bio_exclude_dids.txt, since the
                             opt-out list is empty) UNION the C5 rule (Haiku or Sonnet profile flag, or the profile's
                             community flagged in community_labels_v2)
Draws (blake2b-64 salted hashes, recorded): random = the 100,000 smallest hash64('s7-random-100k', did/rkey);
slab = producers in hash64('s7-slab-producer', did) order, one post each (smallest hash64('s7-slab-post', key),
never a random-set post), kept until 10,000 rows exist. Slab text = O8's format (graph/vmpass/sql_p2.py): text +
' [image: <alt>]' per non-empty alt. S4's corpus has no alt strings, so alt texts for slab candidates that carry
media (netcup image_count > 0 or has_video) are read live from the public AppView (app.bsky.feed.getPosts, <=25
URIs per call, ~8 req/s); a media candidate the AppView no longer returns is dropped (deleted / taken down).
Outputs in OUT_DIR:
  upload/s7_random.parquet, upload/s7_slab.parquet   row_id, text, n_alts  (THE ONLY post data that leaves the Mac)
  local/s7_rowmap.parquet                            row_id -> did, rkey, post_idx  (never uploaded)
  local/alts_cache.jsonl                             AppView responses, slimmed to uri + alts (resumable)
  prep_receipt.json                                  input sha256s, exclusion counts, draw salts, token stats
"""
import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import s7lib  # noqa: E402

LAKE = pathlib.Path.home() / "catbird-lake-local"
S4IN = LAKE / "s4-pilot-20261028/inputs"
CORPUS = S4IN / "posts_corpus.parquet"
S4_MINORS = S4IN / "bio_exclude_dids.txt"
HIDE = LAKE / "s5-declarations-20261027/hide_from_recs_union.tsv"
INTENT_RECORDS = LAKE / "s5-declarations-20261027/user_intents_records.tsv"
PURGE = LAKE / "s5-sweep-20261028/purge_dids.txt"
PROFILE_LABELS = LAKE / "claude-labels-20261027/profile_labels.parquet"
COMMUNITY_V2 = LAKE / "claude-labels-20261027/community_labels_v2.parquet"
DAYS = ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]
MODEL = LAKE / "models/embeddinggemma-2"
EXPECTED_SHA = {CORPUS: "6748c893e7242671aa1f44b99a1db80cd81ff2c89d2313a0a92c21b0d6e6085f",
                HIDE: "7d369fa5ced8ca5f22d8cf21b4b2fd0de549a8dcf887f4c92c0217eb0d51ca52",
                S4_MINORS: "3d4e5964c3306fc1886dc3878694ba50ae03d21550ec38a826f87b865cde361e"}
N_RANDOM, N_SLAB, SLAB_SPARE = 100_000, 10_000, 1_000
SALTS = {"random": "s7-random-100k", "slab_producer": "s7-slab-producer", "slab_post": "s7-slab-post"}
SLAB_ROW0 = 1_000_000
APPVIEW = "https://public.api.bsky.app"
UA = "catbird-community-research/0.1 (+https://catbird.blue; contact joshlacal@gmail.com)"
PROMPT = "task: clustering | query: "
MAX_SEQ = 512


def read_dids(p):
    return {l.split("\t")[0].strip() for l in open(p) if l.strip() and not l.startswith("did\t")}


def optout_dids():
    t = pd.read_csv(INTENT_RECORDS, sep="\t", dtype=str)
    bad = (t.bulkDataset.str.lower() == "false") | (t.syntheticContentGeneration.str.lower() == "false")
    return set(t.did[bad]), int(len(t))


def minors_dids():
    s4 = read_dids(S4_MINORS)  # S4: rank-0 home in a likely_minors community (opt-outs are empty, so this is minors only)
    comm = pq.read_table(COMMUNITY_V2, columns=["community", "sensitivity"]).to_pylist()
    flagged = {c["community"] for c in comm if "likely_minors" in (c["sensitivity"] or [])}
    pl = pq.read_table(PROFILE_LABELS, columns=["did", "community", "haiku_sensitivity", "sonnet_sensitivity"]).to_pylist()
    c5 = {r["did"] for r in pl if "likely_minors" in (r["haiku_sensitivity"] or []) + (r["sonnet_sensitivity"] or [])
          or r["community"] in flagged}
    return s4 | c5, {"s4_rule": len(s4), "c5_rule": len(c5), "union": len(s4 | c5),
                     "likely_minors_communities_v2": len(flagged)}


def media_flags(dids):
    """Latest-revision media flags from the netcup posts for the given DIDs: (did, rkey) -> (image_count, has_video)."""
    ds = pa.array(sorted(dids))
    parts = []
    for day in DAYS:
        t = pq.read_table(LAKE / "netcup-export" / day / "posts.parquet",
                          columns=["did", "rkey", "rev", "image_count", "has_alt_text", "has_video"])
        parts.append(t.filter(pc.is_in(t["did"], value_set=ds)))
    df = pa.concat_tables(parts).to_pandas().sort_values("rev").drop_duplicates(["did", "rkey"], keep="last")
    return {(d, r): (int(i), int(v), int(a)) for d, r, i, v, a in
            zip(df.did, df.rkey, df.image_count, df.has_video, df.has_alt_text)}


def fetch_alts(uris, cache_path, rps=8.0):
    """uri -> alts (list) for live posts, or absent if the AppView did not return the post. Resumable."""
    done = {}
    if cache_path.exists():
        for line in open(cache_path):
            r = json.loads(line)
            done[r["uri"]] = r
    todo = [u for u in uris if u not in done]
    calls = errors = 0
    with open(cache_path, "a") as out:
        for i in range(0, len(todo), 25):
            batch = todo[i:i + 25]
            url = f"{APPVIEW}/xrpc/app.bsky.feed.getPosts?" + urllib.parse.urlencode({"uris": batch}, doseq=True)
            for attempt in range(5):
                t0 = time.time()
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        posts = json.loads(r.read())["posts"]
                    break
                except urllib.error.HTTPError as e:
                    errors += 1
                    if e.code == 429 or e.code >= 500:
                        time.sleep(2 ** attempt * 2)
                        continue
                    raise
                except urllib.error.URLError:
                    errors += 1
                    time.sleep(2 ** attempt * 2)
            else:
                raise SystemExit(f"getPosts failed 5 times at batch {i // 25}")
            calls += 1
            got = {p["uri"]: s7lib.alts_from_record(p.get("record")) for p in posts}
            for u in batch:
                rec = {"uri": u, "live": u in got, "alts": got.get(u, [])}
                done[u] = rec
                out.write(json.dumps(rec) + "\n")
            out.flush()
            time.sleep(max(0.0, 1.0 / rps - (time.time() - t0)))
    return done, {"calls_this_run": calls, "http_errors_retried": errors}


def token_stats(texts, tok, n_prompt):
    n_text = np.array([len(x) for x in tok(list(texts), add_special_tokens=True, truncation=True,
                                             max_length=MAX_SEQ)["input_ids"]])
    full = np.minimum(n_text + n_prompt, MAX_SEQ)
    return {"n": int(len(full)), "tokens_per_post_with_prompt": float(full.mean()),
            "tokens_per_post_text_only": float(n_text.mean()), "p50_with_prompt": float(np.median(full)),
            "p99_with_prompt": float(np.quantile(full, 0.99)), "max_with_prompt": int(full.max()),
            "truncated_at_512": int((n_text + n_prompt > MAX_SEQ).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--no-fetch", action="store_true", help="skip the AppView alt fetch (tests); media rows keep no alt")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "upload").mkdir(parents=True, exist_ok=True)
    (out / "local").mkdir(parents=True, exist_ok=True)
    rec = {"step": "S7 prep", "inputs": {}, "counts": {}, "salts": SALTS}
    for p, want in EXPECTED_SHA.items():
        got = s7lib.sha256_file(p)
        assert got == want, f"{p}: sha256 {got} != registered {want}"
    for p in [CORPUS, S4_MINORS, HIDE, INTENT_RECORDS, PURGE, PROFILE_LABELS, COMMUNITY_V2]:
        rec["inputs"][str(p)] = s7lib.sha256_file(p)

    df = pq.read_table(CORPUS, columns=["post_idx", "did", "rkey", "text"]).to_pandas()
    c = rec["counts"]
    c["s4_corpus_posts"], c["s4_corpus_producers"] = int(len(df)), int(df.did.nunique())
    opt, n_records = optout_dids()
    minors, minors_info = minors_dids()
    rec["minors_rule"] = minors_info
    rec["user_intents_records"] = n_records
    excl = {"hide_from_recs_union": read_dids(HIDE), "user_intents_optout": opt,
            "s5_purge": read_dids(PURGE), "likely_minors": minors}
    drop = np.zeros(len(df), bool)
    c["exclusions"] = {}
    for name, s in excl.items():
        hit = df.did.isin(s).to_numpy()
        c["exclusions"][name] = {"list_dids": len(s), "posts_hit": int(hit.sum()), "producers_hit": int(df.did[hit].nunique()),
                                 "posts_newly_dropped": int((hit & ~drop).sum())}
        drop |= hit
    df = df[~drop].reset_index(drop=True)
    c["eligible_posts"], c["eligible_producers"] = int(len(df)), int(df.did.nunique())
    keys = (df.did + "/" + df.rkey).tolist()

    ridx = s7lib.draw_smallest(keys, N_RANDOM, SALTS["random"])
    rand = df.iloc[ridx].reset_index(drop=True)
    rand["row_id"] = np.arange(len(rand), dtype=np.int64)
    rand["n_alts"] = np.int8(0)
    rand_keys = frozenset(keys[i] for i in ridx)

    cand = s7lib.draw_one_per_group(df.did.tolist(), keys, SALTS["slab_producer"], SALTS["slab_post"], exclude=rand_keys)
    cand = cand[:N_SLAB + SLAB_SPARE]
    sl = df.iloc[[i for _, i in cand]].reset_index(drop=True)
    flags = media_flags(set(sl.did))
    f = [flags.get((d, r), (0, 0, 0)) for d, r in zip(sl.did, sl.rkey)]
    sl["image_count"] = [x[0] for x in f]
    sl["has_video"] = [x[1] for x in f]
    sl["has_alt_flag"] = [x[2] for x in f]
    sl["uri"] = "at://" + sl.did + "/app.bsky.feed.post/" + sl.rkey
    media = (sl.image_count > 0) | (sl.has_video == 1)
    c["slab_candidates"] = int(len(sl))
    c["slab_candidates_with_media"] = int(media.sum())
    c["slab_candidates_netcup_has_alt_flag"] = int((sl.has_alt_flag == 1).sum())
    if a.no_fetch:
        live = {}
        rec["alt_fetch"] = "skipped (--no-fetch)"
    else:
        live, stats = fetch_alts(sl.uri[media].tolist(), out / "local/alts_cache.jsonl")
        rec["alt_fetch"] = {"endpoint": f"{APPVIEW} app.bsky.feed.getPosts", "user_agent": UA, "uris": int(media.sum()),
                            **stats, "returned_live": sum(1 for u in sl.uri[media] if live.get(u, {}).get("live"))}
    keep, alts = [], []
    for u, m in zip(sl.uri, media):
        r = live.get(u)
        if m and not a.no_fetch and not (r and r["live"]):
            keep.append(False)
            alts.append([])
            continue
        keep.append(True)
        alts.append(r["alts"] if r else [])
    sl["alts"] = alts
    c["slab_media_candidates_dropped_not_live"] = int((~np.array(keep)).sum())
    sl = sl[np.array(keep)].iloc[:N_SLAB].reset_index(drop=True)
    assert len(sl) == N_SLAB, f"only {len(sl)} slab rows; raise SLAB_SPARE"
    sl["text"] = [s7lib.slab_text(t, al) for t, al in zip(sl.text, sl.alts)]
    sl["n_alts"] = np.array([sum(1 for x in al if x) for al in sl.alts], dtype=np.int8)
    sl["row_id"] = np.arange(SLAB_ROW0, SLAB_ROW0 + len(sl), dtype=np.int64)
    c["slab_rows"] = int(len(sl))
    c["slab_rows_with_alt_text"] = int((sl.n_alts > 0).sum())
    c["slab_alt_texts_appended"] = int(sl.n_alts.sum())
    c["slab_rows_with_media"] = int(((sl.image_count > 0) | (sl.has_video == 1)).sum())
    c["slab_producers"] = int(sl.did.nunique())
    c["random_rows"], c["random_producers"] = int(len(rand)), int(rand.did.nunique())
    c["random_slab_overlap_posts"] = int(len(set(zip(rand.did, rand.rkey)) & set(zip(sl.did, sl.rkey))))

    up = {"s7_random.parquet": rand, "s7_slab.parquet": sl}
    for name, t in up.items():
        pq.write_table(pa.Table.from_pandas(t[["row_id", "text", "n_alts"]], preserve_index=False), out / "upload" / name)
    rowmap = pd.concat([rand[["row_id", "did", "rkey", "post_idx"]], sl[["row_id", "did", "rkey", "post_idx"]]])
    pq.write_table(pa.Table.from_pandas(rowmap, preserve_index=False), out / "local/s7_rowmap.parquet")
    for name in up:  # no DID/URI column may leave the Mac (a DID typed inside a post's text is post text)
        assert pq.read_schema(out / "upload" / name).names == ["row_id", "text", "n_alts"]
    rec["upload_columns"] = ["row_id", "text", "n_alts"]
    rec["upload_texts_containing_did_string"] = {
        n: int(pc.sum(pc.cast(pc.match_substring(pq.read_table(out / "upload" / n)["text"], "did:"), pa.int64())).as_py() or 0)
        for n in up}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    n_prompt = len(tok(PROMPT, add_special_tokens=False)["input_ids"])
    rec["tokens"] = {"tokenizer": str(MODEL), "prompt": PROMPT, "prompt_tokens": n_prompt,
                     "random": token_stats(rand.text, tok, n_prompt), "slab": token_stats(sl.text, tok, n_prompt),
                     "slab_rows_with_alt": token_stats(sl.text[sl.n_alts > 0], tok, n_prompt) if (sl.n_alts > 0).any() else None}
    rec["outputs"] = {str(p.relative_to(out)): s7lib.sha256_file(p) for p in sorted(out.glob("*/*.parquet"))}
    json.dump(rec, open(out / "prep_receipt.json", "w"), indent=1)
    print(s7lib.dumps(rec))


if __name__ == "__main__":
    main()
