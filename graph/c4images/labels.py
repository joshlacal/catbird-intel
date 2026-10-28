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

Several stores that split one replay into seq ranges (the parallel bluesky-v3-s* streams, tails) are read
as one union: in start-cursor order, frames already covered (seq <= the union's max) dropped as duplicates, and
a store starting beyond the union's max refused as a seq hole (`union_labels`). --label-root reads every
generation under a root; --coverage-to N fails unless the union reaches seq N.

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


def _open_frames(directory: pathlib.Path):
    identity = json.loads((directory / "identity.json").read_text())
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != collector.PACKED_SCHEMA:
            raise StoreError(f"{directory}: unknown manifest schema {manifest.get('schema')!r}")
        return identity, _packed_frames(directory, manifest), int(manifest["base_state"]["cursor"])
    if (directory / "journal.jsonl").exists():
        return identity, _legacy_frames(directory), 0
    raise StoreError(f"{directory}: neither a packed nor a legacy label store")


def start_cursor(directory) -> int:
    """The cursor a store's own stream began after (packed: manifest base_state; legacy: 0)."""
    return _open_frames(pathlib.Path(directory).expanduser())[2]


def store_frames(directory: pathlib.Path, stats: dict | None = None):
    """Yield (seq, [labels]) for every committed #labels frame of one store, in replay order, with the
    store's hash chain verified. stats gets frames, labels, first/max seq, the start cursor and the seqs the
    store's own stream skipped (gap_seqs: the labeler's own holes as this connection saw them)."""
    directory = pathlib.Path(directory).expanduser()
    identity, frames, cursor = _open_frames(directory)
    st = stats if stats is not None else {}
    for k in ("frames", "labels", "max_seq", "gap_seqs"):
        st.setdefault(k, 0)
    st["start_cursor"] = cursor
    prev = cursor
    for raw in frames:
        decoded = collector.decode_frame(raw, identity["did"])
        st["frames"] += 1
        if decoded["kind"] != "labels":
            continue
        body = decoded["body"]
        seq = body["seq"]
        st.setdefault("first_seq", seq)
        if seq > prev + 1:
            st["gap_seqs"] += seq - prev - 1
        prev = max(prev, seq)
        st["max_seq"] = max(st["max_seq"], seq)
        st["labels"] += len(body["labels"])
        yield seq, body["labels"]


def store_labels(directory: pathlib.Path, stats: dict | None = None):
    """Yield (seq, label dict) for every committed label of one store, in replay order."""
    for seq, labels in store_frames(directory, stats):
        for label in labels:
            yield seq, label


def union_labels(stores, per_store: dict, union: dict):
    """(seq, label) over several stores that split one replay into seq ranges (bluesky-v1, -v2, the parallel
    bluesky-v3-s* streams, tails). Stores are read in start-cursor order; a store may start at or below what
    the union already covers (overlaps are expected: frames with seq <= the union's max are duplicates and
    dropped), never above it: a store whose start cursor is beyond the union's max would leave seqs nobody
    fetched, so that raises StoreError. Seq gaps that remain in the union were therefore skipped by the
    labeler on a live connection (each lies inside some store's own stream); union["gap_seqs"] counts them
    for comparison with the stores' own gap counts and the collectors' missing_sequence_count."""
    order = sorted(((start_cursor(s), i, pathlib.Path(s).expanduser()) for i, s in enumerate(stores)))
    umax = 0
    union.update({"order": [str(p) for _c, _i, p in order], "duplicate_frames": 0, "gap_seqs": 0, "frames": 0})
    for cur, _i, path in order:
        if cur > umax:
            raise StoreError(f"seq hole: {path} starts after cursor {cur} but the union only reaches {umax}")
        st = per_store.setdefault(str(path), {})
        for seq, labels in store_frames(path, st):
            if seq <= umax:
                union["duplicate_frames"] += 1
                st["duplicate_frames"] = st.get("duplicate_frames", 0) + 1
                continue
            union["gap_seqs"] += seq - umax - 1
            umax = seq
            union["frames"] += 1
            for label in labels:
                yield seq, label
        cp = path / "checkpoint.json"
        if cp.exists():
            st["checkpoint_missing_sequence_count"] = json.loads(cp.read_text()).get("state", {}).get(
                "missing_sequence_count")
    union["max_seq"] = umax
    union["store_gap_seqs_sum"] = sum(v.get("gap_seqs", 0) for v in per_store.values())
    union["gaps_within_store_gaps"] = union["gap_seqs"] <= union["store_gap_seqs_sum"]
    if not union["gaps_within_store_gaps"]:
        raise StoreError(f"union has {union['gap_seqs']} missing seqs, more than the stores' own "
                         f"{union['store_gap_seqs_sum']}: a range was never fetched")


def label_stores(root) -> list[pathlib.Path]:
    """Every replay generation and tail under a moderation-labels root: bluesky-v1, bluesky-v2, bluesky-v3-s*,
    recent-tail-* (not smoke or speed-probe stores)."""
    root = pathlib.Path(root).expanduser()
    names = ["bluesky-v1", "bluesky-v2"]
    out = [root / n for n in names if (root / n).exists()]
    out += sorted(root.glob("bluesky-v3-s*")) + sorted(root.glob("recent-tail-*"))
    return [p for p in out if (p / "identity.json").exists()]


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
    ap.add_argument("stores", nargs="*")
    ap.add_argument("--label-root", help="read every store under this root (label_stores) instead of a list")
    ap.add_argument("--coverage-to", type=int, help="fail unless the union reaches at least this seq")
    ap.add_argument("--record-dids", help="file of subject DIDs whose record-level labels are kept")
    ap.add_argument("--now", default=datetime.now(timezone.utc).isoformat())
    a = ap.parse_args(argv)
    stores = label_stores(a.label_root) if a.label_root else a.stores
    if not stores:
        ap.error("no stores")
    record_dids = None
    if a.record_dids:
        record_dids = {l.strip() for l in open(a.record_dids) if l.strip()}
    per_store, union = {}, {}
    active = resolve(union_labels(stores, per_store, union), a.now, record_dids)
    if a.coverage_to and union["max_seq"] < a.coverage_to:
        raise StoreError(f"union reaches seq {union['max_seq']}, short of {a.coverage_to}")
    out = pathlib.Path(a.out_tsv)
    tmp = out.with_name(out.name + ".partial")
    with open(tmp, "w") as f:
        for row in sorted(active.values(), key=lambda r: (r["did"], r["uri"], r["val"], r["src"])):
            f.write("\t".join([row["did"], row["scope"], row["uri"], row["val"], row["src"], row["cts"], row["exp"]]) + "\n")
    tmp.replace(out)
    summary = {"now": a.now, "stores": per_store, "union": union, "coverage_to": a.coverage_to,
               "active_labels": len(active),
               "active_account_dids": len({r["did"] for r in active.values() if r["scope"] == "account"}),
               "active_record_uris": len({r["uri"] for r in active.values() if r["scope"] == "record"}),
               "record_did_filter": len(record_dids) if record_dids is not None else None}
    out.with_name(out.name + ".summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "stores"}, sort_keys=True))


if __name__ == "__main__":
    main()
