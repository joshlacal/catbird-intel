"""C4 fetch gate: candidate posts -> CDN thumbnail fetch list under the D9 image policy.

  gate.py CANDIDATES_PARQUET[,...] OUT_DIR --record-labels TSV [--now ISO] [--producers FILE] [--no-scope]

D9 (docs/DECISIONS.md 2026-10-28, Josh) as amended by Josh the same day: no per-post getPosts hydration;
the thumbnail URL is built from the author DID and the blob CID that `post_extract.py` now keeps
(`image_cids`), and the CDN itself is the live check (it enforces Bluesky takedowns; any non-200 is a
skip, see cdn.py). The offline gate, in order, giving each dropped post the first reason that hits:

  1  latest_rev_deleted   latest revision is a delete
     reply                not an original (plan §5.3 pilots image originals); off with originals_only=False
     no_image_cids        no `app.bsky.embed.images` blob CID (videos, links, galleries never fetched)
  2  bad_tid / too_young  age from the rkey TID < 3 days (D9 item 3); unparseable or future TIDs dropped
  3  post_self_label      any self-label on the post (D9 item 2, "any value")
  4  purge                S5 purge list (inactive / absent accounts)
     hide_from_recs       hide-from-recs union (amendment A1: their posts are never embedded)
     account_label_s5     S5 getProfiles: any moderation or self label, adult/graphic, hard label,
                          !no-unauthenticated, not found, or repo status other than active
     account_label_replay account-level label in force in our Bluesky labeler replay (a DID subject or
                          the account's profile record)
     claude_profile       Haiku or Sonnet flags the account adult or likely_minors, or adult_creator
     claude_community     KnownFor membership (any rank) in an Opus-labelled adult or likely-minors
                          community (sensitivity adult/likely_minors or type adult_content)
     scope_community      plan §5.3 pilot scope (on by default): rank-0 home community is roleplay,
                          follow-back, follow-train, or carries a Y3 sensitive class
  5  record_label_replay  record-level label in force on the post URI in the replay
  6  not_selected_producer  outside --producers (e.g. the image-heavy eligible producers)
  7  producer_cap         beyond 8 images per producer (posts taken in sha256(uri) order)

Avatars and banners are never fetched: URLs are only ever built as `feed_thumbnail` from a post's
`embed.images` CIDs, and any CID in `avatar_banner_cids` (when such a list exists) is dropped as
`avatar_or_banner_cid`. Outputs: fetch_list.parquet (one row per image), skips.parquet (uri, reason),
gate_receipt.json (counts by reason and input hashes). Nothing is fetched here.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq

LAKE = pathlib.Path.home() / "catbird-lake-local"
CDN_HOST = "cdn.bsky.app"
THUMB_URL = "https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{cid}"
MIN_AGE = timedelta(days=3)
PER_PRODUCER_IMAGES = 8
_TID_ALPHABET = "234567abcdefghijklmnopqrstuvwxyz"
_TID_LOOKUP = {c: i for i, c in enumerate(_TID_ALPHABET)}
_DID_RE = re.compile(r"^(did:plc:[a-z2-7]{24}|did:web:[a-z0-9.-]{1,253})$")
_CID_RE = re.compile(r"^b[a-z2-7]{7,127}$")
D9_SENSITIVITY = frozenset({"adult", "likely_minors"})
Y3_SENSITIVITY = frozenset({"news", "politics", "religion", "sexuality_gender", "health",
                            "ethnicity_nationality", "adult", "likely_minors"})
SCOPE_TYPES = frozenset({"roleplay", "follow_back_circle", "follow_train_or_political_amplification",
                         "adult_content"})
DID_REASONS = ("purge", "hide_from_recs", "account_label_s5", "account_label_replay", "claude_profile",
               "claude_community", "scope_community")


@dataclass(frozen=True)
class Paths:
    purge: pathlib.Path = LAKE / "s5-sweep-20261028/purge_dids.txt"
    hide_from_recs: pathlib.Path = LAKE / "s5-declarations-20261027/hide_from_recs_union.tsv"
    s5_profiles: pathlib.Path = LAKE / "s5-sweep-20261028/profiles.parquet"
    profile_labels: pathlib.Path = LAKE / "claude-labels-20261027/profile_labels.parquet"
    community_labels: pathlib.Path = LAKE / "claude-labels-20261027/community_labels_v2.parquet"
    known_for: pathlib.Path = LAKE / "communities-cand1-hubdemote/known_for.parquet"


def tid_time(rkey: str) -> datetime | None:
    """Creation time encoded in a TID record key (53-bit microseconds); None if not a valid TID."""
    if not isinstance(rkey, str) or len(rkey) != 13 or rkey[0] not in "234567abcdefghij":
        return None
    v = 0
    for ch in rkey:
        if ch not in _TID_LOOKUP:
            return None
        v = (v << 5) | _TID_LOOKUP[ch]
    return datetime.fromtimestamp((v >> 10) / 1e6, tz=timezone.utc)


def thumb_url(did: str, cid: str) -> str:
    """The only URL form this pipeline fetches. Raises on anything that could escape the path."""
    if not _DID_RE.match(did or "") or not _CID_RE.match(cid or ""):
        raise ValueError(f"refusing to build a CDN URL from did={did!r} cid={cid!r}")
    return THUMB_URL.format(did=did, cid=cid)


def _lines(path) -> set[str]:
    out = set()
    with open(path) as f:
        for line in f:
            v = line.split("\t", 1)[0].strip()
            if v:
                out.add(v)
    return out


@dataclass
class DidExclusions:
    """DID -> first reason in DID_REASONS order. Build with `load_did_exclusions` or directly in tests."""
    sets: dict[str, set[str]] = field(default_factory=dict)

    def reason(self, did: str) -> str | None:
        for r in DID_REASONS:
            s = self.sets.get(r)
            if s and did in s:
                return r
        return None

    def sizes(self) -> dict[str, int]:
        return {r: len(self.sets.get(r, ())) for r in DID_REASONS}


def s5_flagged(rows) -> set[str]:
    out = set()
    for r in rows:
        if (r.get("found") is not True or r.get("any_mod_label") or r.get("mod_labels") or r.get("self_labels")
                or r.get("adult_graphic") or r.get("hard_label") or r.get("no_unauthenticated")
                or (r.get("repo_status") or "active") != "active"):
            out.add(r["did"])
    return out


def claude_profile_flagged(rows) -> set[str]:
    out = set()
    for r in rows:
        sens = set(r.get("haiku_sensitivity") or []) | set(r.get("sonnet_sensitivity") or [])
        if sens & D9_SENSITIVITY or "adult_creator" in (r.get("haiku_account_type"), r.get("sonnet_account_type")):
            out.add(r["did"])
    return out


def community_classes(rows) -> tuple[set[int], set[int]]:
    """(D9 adult/likely-minors communities, plan §5.3 scope-excluded communities)."""
    d9, scope = set(), set()
    for r in rows:
        sens = set(r.get("sensitivity") or [])
        c = r["community"]
        if sens & D9_SENSITIVITY or r.get("community_type") == "adult_content":
            d9.add(c)
        if sens & Y3_SENSITIVITY or r.get("community_type") in SCOPE_TYPES:
            scope.add(c)
    return d9, scope


def load_did_exclusions(paths: Paths = Paths(), account_labels_tsv=None, scope=True) -> DidExclusions:
    sets = {"purge": _lines(paths.purge), "hide_from_recs": _lines(paths.hide_from_recs)}
    sets["account_label_s5"] = s5_flagged(pq.read_table(paths.s5_profiles).to_pylist())
    if account_labels_tsv:
        sets["account_label_replay"] = account_labelled(account_labels_tsv)
    sets["claude_profile"] = claude_profile_flagged(pq.read_table(
        paths.profile_labels, columns=["did", "haiku_account_type", "haiku_sensitivity", "sonnet_account_type",
                                       "sonnet_sensitivity"]).to_pylist())
    d9, sc = community_classes(pq.read_table(
        paths.community_labels, columns=["community", "community_type", "sensitivity"]).to_pylist())
    kf = pq.read_table(paths.known_for, columns=["did", "community", "rank"]).to_pydict()
    sets["claude_community"] = {d for d, c in zip(kf["did"], kf["community"]) if c in d9}
    if scope:
        sets["scope_community"] = {d for d, c, k in zip(kf["did"], kf["community"], kf["rank"]) if k == 0 and c in sc}
    return DidExclusions(sets)


def account_labelled(tsv) -> set[str]:
    out = set()
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1] == "account":
                out.add(p[0])
    return out


def record_labelled(tsv) -> set[str]:
    out = set()
    with open(tsv) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[1] == "record":
                out.add(p[2])
    return out


def cids_in_order(r) -> list[str]:
    """CIDs aligned with `image_alts` position by position, or [] when a malformed CID was dropped at
    extraction (then the alignment is unknown and no alt text is attached)."""
    cids, alts = list(r.get("image_cids") or []), list(r.get("image_alts") or [])
    return cids if len(cids) == len(alts) else []


def latest_revisions(rows):
    """Latest revision per (did, rkey) by `rev` (TID revs sort as strings), then `seq` as tie-break."""
    best = {}
    for r in rows:
        k = (r["did"], r["rkey"])
        rank = (r.get("rev") or "", r.get("seq") or 0)
        if k not in best or rank >= best[k][0]:
            best[k] = (rank, r)
    return [v[1] for v in best.values()]


def gate(rows, did_excl: DidExclusions, labelled_uris: set[str], now: datetime, *, originals_only=True,
         producers: set[str] | None = None, per_producer_images=PER_PRODUCER_IMAGES,
         avatar_banner_cids: frozenset[str] = frozenset(), min_age=MIN_AGE):
    """rows: dicts in POST_SCHEMA (any revisions). Returns (fetch rows, skip rows, Counter of reasons)."""
    skips, counts, kept = [], collections.Counter(), []

    def skip(uri, reason):
        skips.append({"uri": uri, "reason": reason})
        counts[reason] += 1

    for r in latest_revisions(rows):
        did, rkey = r["did"], r["rkey"]
        uri = f"at://{did}/app.bsky.feed.post/{rkey}"
        if r.get("operation") == "delete":
            skip(uri, "latest_rev_deleted")
            continue
        if originals_only and r.get("reply_parent"):
            skip(uri, "reply")
            continue
        cids = list(r.get("image_cids") or [])
        if not cids:
            skip(uri, "no_image_cids")
            continue
        t = tid_time(rkey)
        if t is None or t > now:
            skip(uri, "bad_tid")
            continue
        if now - t < min_age:
            skip(uri, "too_young")
            continue
        if r.get("self_labels"):
            skip(uri, "post_self_label")
            continue
        why = did_excl.reason(did)
        if why:
            skip(uri, why)
            continue
        if uri in labelled_uris:
            skip(uri, "record_label_replay")
            continue
        if producers is not None and did not in producers:
            skip(uri, "not_selected_producer")
            continue
        good = [c for c in cids if c not in avatar_banner_cids]
        if len(good) < len(cids):
            counts["avatar_or_banner_cid"] += len(cids) - len(good)
        try:
            urls = [thumb_url(did, c) for c in good]
        except ValueError:
            skip(uri, "unsafe_did_or_cid")
            continue
        if not urls:
            skip(uri, "no_image_cids")
            continue
        kept.append((did, hashlib.sha256(uri.encode()).hexdigest(), uri, rkey, t, r, good, urls))

    fetch, per = [], collections.Counter()
    for did, _h, uri, rkey, t, r, good, urls in sorted(kept, key=lambda k: (k[0], k[1])):
        if per[did] + len(good) > per_producer_images:
            skip(uri, "producer_cap")
            continue
        per[did] += len(good)
        alt_by_cid = dict(zip(cids_in_order(r), list(r.get("image_alts") or [])))
        for i, (cid, url) in enumerate(zip(good, urls)):
            fetch.append({"uri": uri, "did": did, "rkey": rkey, "created_tid": t.isoformat(), "image_index": i,
                          "cid": cid, "url": url, "text": r.get("text") or "", "alt": alt_by_cid.get(cid, "")})
    counts["fetch_images"] = len(fetch)
    counts["fetch_posts"] = len({f["uri"] for f in fetch})
    counts["fetch_producers"] = len(per)
    return fetch, skips, counts


FETCH_SCHEMA = pa.schema([("uri", pa.string()), ("did", pa.string()), ("rkey", pa.string()),
                          ("created_tid", pa.string()), ("image_index", pa.int32()), ("cid", pa.string()),
                          ("url", pa.string()), ("text", pa.string()), ("alt", pa.string())])


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", help="comma-separated POST_SCHEMA parquet files (with image_cids)")
    ap.add_argument("out_dir")
    ap.add_argument("--record-labels", required=True, help="labels.py output TSV (account + record rows)")
    ap.add_argument("--now", default=datetime.now(timezone.utc).isoformat())
    ap.add_argument("--producers", help="file of DIDs allowed to be fetched (e.g. image-heavy eligible)")
    ap.add_argument("--no-scope", action="store_true", help="drop the plan §5.3 scope_community exclusion")
    ap.add_argument("--all-posts", action="store_true", help="include replies")
    a = ap.parse_args(argv)
    now = datetime.fromisoformat(a.now.replace("Z", "+00:00"))
    files = [pathlib.Path(p) for p in a.candidates.split(",")]
    cols = ["did", "rkey", "rev", "seq", "operation", "reply_parent", "self_labels", "image_cids", "text",
            "image_alts"]
    rows = []
    for f in files:
        rows += pq.read_table(f, columns=cols).to_pylist()
    excl = load_did_exclusions(account_labels_tsv=a.record_labels, scope=not a.no_scope)
    producers = _lines(a.producers) if a.producers else None
    fetch, skips, counts = gate(rows, excl, record_labelled(a.record_labels), now,
                                originals_only=not a.all_posts, producers=producers)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(fetch, schema=FETCH_SCHEMA), out / "fetch_list.parquet")
    pq.write_table(pa.Table.from_pylist(skips, schema=pa.schema([("uri", pa.string()), ("reason", pa.string())])),
                   out / "skips.parquet")
    receipt = {"now": now.isoformat(), "min_age_days": MIN_AGE.days, "per_producer_images": PER_PRODUCER_IMAGES,
               "scope_community": not a.no_scope, "originals_only": not a.all_posts,
               "did_exclusion_sizes": excl.sizes(), "counts": dict(counts),
               "inputs_sha256": {str(p): sha256_file(p) for p in files + [pathlib.Path(a.record_labels)]}}
    (out / "gate_receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps(receipt["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
