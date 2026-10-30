"""Pure helpers for S12m, the natively multimodal S12 (S1 amendment A2): input schema, the D9 image gate,
modality codes, fp16 packing, the dollar guard, the no-image-bytes audit and the projection arithmetic.

Imports only the standard library and numpy/pyarrow (lazily where possible), plus the C4 gate's TID and
URL helpers, so it runs on the Mac (tests, rehearsal) and on the GCE L4 VM.

Input rows (one per post; S12's O8 shards joined with the image-CID sidecar on (did, rkey)):
    uri          string   at://{did}/app.bsky.feed.post/{rkey}   (or `did` + `rkey` columns instead)
    text         string   O8 slab text: post text + " [image: <alt>]" per non-empty alt; no prompt
    image_cids   list<string>  blob CIDs of the post's app.bsky.embed.images (also inside
                          recordWithMedia), in embed order; null/[] when none. Never avatar/banner CIDs.
    self_labels  list<string>  optional; any value skips the images (D9)
    p            int64    optional; O8's producer integer id, carried to the output for pooling
The sidecar alone may carry (did, rkey, image_cids) or (uri, image_cids); job.py left-joins it on uri.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _d in ("c4images", "s7bench"):          # appended: sibling lanes never shadow this directory's modules
    if str(HERE.parent / _d) not in sys.path:
        sys.path.append(str(HERE.parent / _d))
from gate import thumb_url, tid_time  # noqa: E402  (D9 URL form and TID age, shared with C4)

DIM = 256
MAX_IMAGES_PER_POST = 4          # app.bsky.embed.images maxLength
MIN_AGE = timedelta(days=3)      # D9 item 3 (backfill)
IMAGE_TOKEN = "<|image|>"

# Per-post modality flag written with every vector.
TEXT, TEXT_IMAGE, IMAGE_ONLY, TEXT_FALLBACK, NONE = 0, 1, 2, 3, 4
MODALITY_NAMES = {TEXT: "text", TEXT_IMAGE: "text+image", IMAGE_ONLY: "image", TEXT_FALLBACK: "text_fallback",
                  NONE: "none"}


def did_rkey(uri: str) -> tuple[str, str]:
    if not uri.startswith("at://"):
        raise ValueError(f"not an at:// uri: {uri!r}")
    parts = uri[5:].split("/")
    if len(parts) != 3 or parts[1] != "app.bsky.feed.post":
        raise ValueError(f"not a post uri: {uri!r}")
    return parts[0], parts[2]


def post_uri(did: str, rkey: str) -> str:
    return f"at://{did}/app.bsky.feed.post/{rkey}"


# --- D9 gate --------------------------------------------------------------------------------------------------

@dataclass
class GateBundle:
    """What the VM needs to apply D9 without the Mac's per-reason files: the union of DIDs whose images are
    never fetched (purge, hide-from-recs, account labels from S5 and the replay, Claude adult/likely-minors
    profiles, adult/likely-minors communities), the post URIs with a record-level label in force, and the time
    up to which the label replay + tail is complete (the watermark)."""
    excluded_dids: set[str] = field(default_factory=set)
    labelled_uris: set[str] = field(default_factory=set)
    watermark: datetime | None = None
    sha256: dict = field(default_factory=dict)

    def lag(self, now: datetime) -> timedelta | None:
        return None if self.watermark is None else now - self.watermark

    def stale(self, now: datetime, max_lag: timedelta) -> bool:
        lag = self.lag(now)
        return lag is None or lag > max_lag or lag < timedelta(0)


def load_bundle(d) -> GateBundle:
    d = pathlib.Path(d)
    man = json.loads((d / "gate_manifest.json").read_text())
    got = {n: sha256_file(d / n) for n in ("excluded_dids.txt", "labelled_uris.txt")}
    if got != man["sha256"]:
        raise ValueError(f"gate bundle sha256 mismatch: {got} != {man['sha256']}")
    lines = lambda n: {x.strip() for x in open(d / n) if x.strip()}  # noqa: E731
    wm = datetime.fromisoformat(man["label_watermark_utc"].replace("Z", "+00:00"))
    return GateBundle(lines("excluded_dids.txt"), lines("labelled_uris.txt"), wm, got)


def gate_images(did, rkey, cids, self_labels, bundle: GateBundle, now: datetime, min_age=MIN_AGE):
    """(thumbnail URLs, None) when the post's images may be fetched, else ([], reason). A post without image
    CIDs returns ([], None): it is a text post, not a skip. The reasons are the D9 rules in order."""
    cids = [c for c in (cids or []) if c]
    if not cids:
        return [], None
    t = tid_time(rkey)
    if t is None or t > now:
        return [], "bad_tid"
    if now - t < min_age:
        return [], "too_young"
    if self_labels:
        return [], "post_self_label"
    if did in bundle.excluded_dids:
        return [], "account_excluded"
    if post_uri(did, rkey) in bundle.labelled_uris:
        return [], "record_label"
    if len(cids) > MAX_IMAGES_PER_POST:
        return [], "too_many_cids"
    try:
        return [thumb_url(did, c) for c in cids], None
    except ValueError:
        return [], "unsafe_did_or_cid"


def mm_text(prompt: str, text: str, n_images: int) -> str:
    """One interleaved input: prompt, the slab text, then one placeholder per image (C4's layout)."""
    body = (text or "").replace(IMAGE_TOKEN, " ").strip()  # a literal placeholder in a post would add an image slot
    tags = " ".join([IMAGE_TOKEN] * n_images)
    return prompt + (f"{body} {tags}" if body else tags)


# --- vectors --------------------------------------------------------------------------------------------------

def to_stored(full):
    """Matryoshka 256-d: first 256 dims, re-normalised in fp32, cast to fp16 (S7/S4 storage rule)."""
    import numpy as np
    cut = np.asarray(full, dtype=np.float32)[:, :DIM]
    return (cut / np.maximum(np.linalg.norm(cut, axis=1, keepdims=True), 1e-12)).astype(np.float16)


def pack(v16) -> list[bytes]:
    import numpy as np
    v16 = np.ascontiguousarray(v16, dtype="<f2")
    assert v16.ndim == 2 and v16.shape[1] == DIM
    return [r.tobytes() for r in v16]


def unpack(blobs):
    import numpy as np
    return np.frombuffer(b"".join(blobs), dtype="<f2").reshape(-1, DIM)


def output_schema():
    import pyarrow as pa
    return pa.schema([("uri", pa.string()), ("p", pa.int64()), ("modality", pa.int8()),
                      ("n_image_cids", pa.int8()), ("n_images", pa.int8()), ("soft_tokens", pa.int16()),
                      ("image_skip", pa.string()), ("vec", pa.binary(2 * DIM)), ("vec_text", pa.binary(2 * DIM))])


# --- money ----------------------------------------------------------------------------------------------------

# GCE Spot list prices, us-west1, read 2026-10-28 from the Cloud Billing catalog (service 6F81-5844-456A):
# SKU 59A6-9C49-BDDB G2 core $0.01495/h, BF80-650A-78E6 G2 RAM $0.001751/GiB-h, AEC2-3D5C-61BF L4 on Spot $0.3352/h.
# Spot prices move (up to once a month); the launcher passes the rate and the guard uses it with headroom.
SPOT_G2_CORE, SPOT_G2_RAM_GIB, SPOT_L4 = 0.01495, 0.001751, 0.3352
PD_BALANCED_GB_MONTH, EXT_IP_SPOT_H = 0.10, 0.0025


def g2_standard_8_spot_usd_per_hour(boot_gb=150):
    vm = 8 * SPOT_G2_CORE + 32 * SPOT_G2_RAM_GIB + SPOT_L4
    return vm + boot_gb * PD_BALANCED_GB_MONTH / 730 + EXT_IP_SPOT_H


class DollarGuard:
    """Projected spend = dollars recorded by earlier VMs of this run + this VM's wall time x rate.
    `allows(seconds)` refuses work whose projected end would cross the cap."""

    def __init__(self, cap_usd, usd_per_hour, prior_usd=0.0, t0=None, clock=time.time):
        self.cap, self.rate, self.prior, self.clock = cap_usd, usd_per_hour, prior_usd, clock
        self.t0 = clock() if t0 is None else t0

    def spent(self):
        return self.prior + (self.clock() - self.t0) / 3600 * self.rate

    def allows(self, next_seconds):
        return self.spent() + next_seconds / 3600 * self.rate <= self.cap

    def state(self):
        return {"usd_spent_est": round(self.spent(), 3), "cap_usd": self.cap, "usd_per_hour": round(self.rate, 4),
                "prior_vms_usd": round(self.prior, 3)}


def usd_per_10m(posts_per_s, usd_per_hour):
    h = 1e7 / posts_per_s / 3600
    return h * usd_per_hour, f"1e7 / {posts_per_s:,.1f} posts/s = {h:.2f} h x ${usd_per_hour:.4f}/h = ${h * usd_per_hour:.2f}"


# --- audit: no image bytes on disk ----------------------------------------------------------------------------

IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")


def has_image_header(b: bytes) -> bool:
    return b.startswith(IMAGE_MAGIC) or (b[:4] == b"RIFF" and b[8:12] == b"WEBP")


def contains_image_signature(b: bytes) -> bool:
    return any(s in b for s in IMAGE_MAGIC) or (b"RIFF" in b and b"WEBP" in b)


def _schema_ok(schema) -> bool:
    """The S12m output schema exactly, or the bench vectors file: uri, p, n_images and 512-byte vec_* columns."""
    import pyarrow as pa
    if schema.equals(output_schema()):
        return True
    fixed = {"uri": pa.string(), "p": pa.int64(), "n_images": pa.int8()}
    return all((f.name in fixed and f.type == fixed[f.name]) or (f.name.startswith("vec_") and f.type == pa.binary(2 * DIM))
               for f in schema) and "uri" in schema.names


def audit_output_table(path) -> dict:
    """An output parquet passes when its schema is an allowed one, every vector is 512 bytes of finite unit-norm
    fp16 (a thumbnail cannot hide in 512 bytes), and no string column carries an image signature. Raw-byte magic
    scans are not used on vectors: random fp16 bytes match a 3-byte JPEG marker by chance."""
    import numpy as np
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    out = {"file": str(path), "rows": t.num_rows, "schema_ok": _schema_ok(t.schema)}
    bad = 0
    for col in [n for n in t.column_names if n == "vec" or n.startswith("vec_")]:
        vals = [x for x in t[col].to_pylist() if x is not None]
        if vals:
            v = unpack(vals).astype(np.float32)
            n = np.linalg.norm(v, axis=1)
            bad += int((~np.isfinite(v).all(axis=1) | (np.abs(n - 1) > 0.01)).sum())
    out["bad_vectors"] = bad
    strs = b"".join((s or "").encode() for c in ("uri", "image_skip") if c in t.column_names for s in t[c].to_pylist())
    out["string_image_signature"] = contains_image_signature(strs)
    out["ok"] = out["schema_ok"] and not bad and not out["string_image_signature"]
    return out


def audit_files(paths, structured=frozenset()) -> dict:
    """Header check on every file, full-content signature scan on files not in `structured` (outputs that were
    validated structurally), size-capped at 256 MB per file."""
    rep = {"files": 0, "bytes": 0, "image_header": [], "image_signature": [], "unreadable": []}
    for p in map(pathlib.Path, paths):
        try:
            b = p.read_bytes() if p.stat().st_size <= 256 << 20 else open(p, "rb").read(256 << 20)
        except OSError:
            rep["unreadable"].append(str(p))
            continue
        rep["files"] += 1
        rep["bytes"] += len(b)
        if has_image_header(b):
            rep["image_header"].append(str(p))
        elif str(p) not in structured and contains_image_signature(b):
            rep["image_signature"].append(str(p))
    rep["ok"] = not rep["image_header"] and not rep["image_signature"]
    return rep


# --- misc -----------------------------------------------------------------------------------------------------

def sha256_file(p, block=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def hash_order(salt: str, keys) -> list[int]:
    """Indices of keys in salted blake2b-64 order (the S4/S7 sampling rule)."""
    h = [int.from_bytes(hashlib.blake2b(f"{salt}|{k}".encode(), digest_size=8).digest(), "big") for k in keys]
    return sorted(range(len(keys)), key=lambda i: (h[i], i))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
