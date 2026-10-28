"""Read-only resolver for the local Bluesky labeler replays (C4 image gate, D9 policy item 2).

  labels.py OUT_TSV STORE_DIR [STORE_DIR ...] [--record-dids FILE] [--now ISO8601]

Reads stores written by `ops/backfill_moderation_labels.py` without touching them:
  - legacy stores (`journal.jsonl` + `raw/<sha>.cbor`, e.g. bluesky-v1);
  - packed stores (`manifest.json` schema catbird-public-label-replay-packed-v2 + `chunks.jsonl` +
    `chunks/<sha>.frames.gz`, e.g. bluesky-v2 and the recent tails).
Only committed data is read: a packed store's journal is read once, a torn final line (the writer is
mid-append) is ignored rather than repaired, the hash chain and every chunk's compressed and raw
sha256 are checked exactly as `PackedStore(verify=True)` does, and each frame is decoded with the
collector's own `decode_frame` (which checks the labeler DID). Nothing is written into a store, and no
lock is taken, so a running replay is unaffected. Never instantiate `PackedStore` here: on a torn
journal it renames files in the live directory.

Resolution (atproto label semantics, strict where they leave room):
  - a label is keyed by (src, uri, val); the newest by (cts, then replay order) wins;
  - a winning `neg: true` removes the label; a winning label whose `exp` is at or before NOW is expired;
  - a `cid`-scoped label is applied to the whole URI (every version), which is stricter than scoping it
    to one record version;
  - account-level subjects are a bare DID or that DID's `app.bsky.actor.profile/self` record (a label on
    the profile record is treated as a label on the account).
Output TSV (no header): subject_did, scope (account|record), uri, val, src, cts, exp. Account-level
labels are kept for every DID; record-level labels only for subject DIDs in --record-dids (all if absent,
which needs several GB of RAM on the full replay).

Runs with the collector's venv (stdlib + cbor2), e.g. ~/catbird-lake-local/label-backfill-venv.
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import sys
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "ops"))
import backfill_moderation_labels as collector  # noqa: E402

PROFILE_SUFFIX = "/app.bsky.actor.profile/self"


class StoreError(ValueError):
    pass


def subject(uri: str) -> tuple[str, str]:
    """(subject DID, 'account' | 'record'); ('', '') for a URI that names no DID."""
    if uri.startswith("did:"):
        return uri, "account"
    if uri.startswith("at://"):
        rest = uri[5:]
        did = rest.split("/", 1)[0]
        if did.startswith("did:"):
            return did, "account" if rest in (did, did + PROFILE_SUFFIX) else "record"
    return "", ""


def _legacy_frames(directory: pathlib.Path):
    journal = directory / "journal.jsonl"
    data = journal.read_bytes()
    lines = data.split(b"\n")
    if not data.endswith(b"\n"):
        lines = lines[:-1]  # torn tail: ignore, never repair
    for line in lines:
        if not line:
            continue
        entry = json.loads(line)
        raw = (directory / entry["raw_path"]).read_bytes()
        if collector.sha(raw) != entry["sha256"] or len(raw) != entry["bytes"]:
            raise StoreError(f"{directory}: legacy raw hash mismatch at {entry['raw_path']}")
        yield raw


def _packed_frames(directory: pathlib.Path, manifest: dict):
    data = (directory / "chunks.jsonl").read_bytes() if (directory / "chunks.jsonl").exists() else b""
    lines = data.split(b"\n")
    if not data.endswith(b"\n"):
        lines = lines[:-1]
    chain = collector.sha(collector.json_bytes(manifest))
    cursor = manifest["base_state"]["cursor"]
    for line in lines:
        if not line:
            continue
        entry = json.loads(line)
        if entry["previous_sha256"] != chain or entry["before_cursor"] != cursor:
            raise StoreError(f"{directory}: journal chain/cursor mismatch")
        compressed = (directory / "chunks" / (entry["compressed_sha256"] + ".frames.gz")).read_bytes()
        if len(compressed) != entry["compressed_bytes"] or collector.sha(compressed) != entry["compressed_sha256"]:
            raise StoreError(f"{directory}: committed compressed chunk hash mismatch")
        raw = gzip.decompress(compressed)
        if collector.sha(raw) != entry["raw_sha256"] or len(raw) != entry["raw_bytes"]:
            raise StoreError(f"{directory}: uncompressed chunk hash mismatch")
        yield from collector.iter_packed(raw)
        chain = collector.sha(line)
        cursor = entry["after"]["cursor"]


def store_labels(directory: pathlib.Path, stats: dict | None = None):
    """Yield (seq, label dict) for every committed label frame of one store, in replay order."""
    directory = pathlib.Path(directory).expanduser()
    identity = json.loads((directory / "identity.json").read_text())
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != collector.PACKED_SCHEMA:
            raise StoreError(f"{directory}: unknown manifest schema {manifest.get('schema')!r}")
        frames = _packed_frames(directory, manifest)
    elif (directory / "journal.jsonl").exists():
        frames = _legacy_frames(directory)
    else:
        raise StoreError(f"{directory}: neither a packed nor a legacy label store")
    st = stats if stats is not None else {}
    st.setdefault("frames", 0)
    st.setdefault("labels", 0)
    st.setdefault("max_seq", 0)
    for raw in frames:
        decoded = collector.decode_frame(raw, identity["did"])
        st["frames"] += 1
        if decoded["kind"] != "labels":
            continue
        body = decoded["body"]
        st["max_seq"] = max(st["max_seq"], body["seq"])
        for label in body["labels"]:
            st["labels"] += 1
            yield body["seq"], label


def resolve(labelled, now: str, record_dids: set[str] | None = None):
    """labelled: iterable of (seq, label). Returns {(src, uri, val): row} of labels in force at NOW."""
    latest: dict[tuple[str, str, str], tuple[str, int, dict]] = {}
    order = 0
    for _seq, label in labelled:
        order += 1
        uri = label["uri"]
        did, scope = subject(uri)
        if not did:
            continue
        if scope == "record" and record_dids is not None and did not in record_dids:
            continue
        key = (label["src"], uri, label["val"])
        prev = latest.get(key)
        rank = (label["cts"], order)
        if prev is None or rank >= (prev[0], prev[1]):
            latest[key] = (label["cts"], order, {
                "did": did, "scope": scope, "uri": uri, "val": label["val"], "src": label["src"],
                "cts": label["cts"], "exp": label.get("exp") or "", "neg": label.get("neg") is True})
    active = {}
    now_dt = parse_time(now)
    for key, (_cts, _o, row) in latest.items():
        if row["neg"]:
            continue
        if row["exp"] and parse_time(row["exp"]) <= now_dt:
            continue
        active[key] = row
    return active


def parse_time(value: str) -> datetime:
    v = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_tsv")
    ap.add_argument("stores", nargs="+")
    ap.add_argument("--record-dids", help="file of subject DIDs whose record-level labels are kept")
    ap.add_argument("--now", default=datetime.now(timezone.utc).isoformat())
    a = ap.parse_args(argv)
    record_dids = None
    if a.record_dids:
        record_dids = {l.strip() for l in open(a.record_dids) if l.strip()}
    per_store = {}

    def all_labels():
        for s in a.stores:
            st = per_store.setdefault(str(s), {})
            yield from store_labels(pathlib.Path(s), st)

    active = resolve(all_labels(), a.now, record_dids)
    out = pathlib.Path(a.out_tsv)
    tmp = out.with_name(out.name + ".partial")
    with open(tmp, "w") as f:
        for row in sorted(active.values(), key=lambda r: (r["did"], r["uri"], r["val"], r["src"])):
            f.write("\t".join([row["did"], row["scope"], row["uri"], row["val"], row["src"], row["cts"], row["exp"]]) + "\n")
    tmp.replace(out)
    summary = {"now": a.now, "stores": per_store, "active_labels": len(active),
               "active_account_dids": len({r["did"] for r in active.values() if r["scope"] == "account"}),
               "active_record_uris": len({r["uri"] for r in active.values() if r["scope"] == "record"}),
               "record_did_filter": len(record_dids) if record_dids is not None else None}
    out.with_name(out.name + ".summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
