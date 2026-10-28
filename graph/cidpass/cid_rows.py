"""Image-CID sidecar rows from decoded app.bsky.feed.post commits.

One decoded post commit (the decoder's JSONL payload: `operation`, `rkey`,
`rev`, `record`, `seq`, `did`) becomes:

- one `image` row per image of an `app.bsky.embed.images` embed, used directly
  (source `images`) or as `recordWithMedia`'s media (source `recordWithMedia`);
- one `image` row per item of an `app.bsky.embed.gallery` embed (upstream's newer
  multi-image embed, up to 20 `#image` items; source `gallery`, or
  `recordWithMedia` when it is the media there);
- one `external_thumb` row for an `app.bsky.embed.external` link card's thumb,
  directly (source `external`) or under `recordWithMedia` (source
  `recordWithMedia`);
- one `marker` row (no CID) for a delete, or for an update that carries no
  image or thumb. The fold needs these to tell that a post's image revision
  was superseded or deleted. Creates without media emit nothing (see
  `rows_for_event`).

CID validation and both blob forms (current `ref.$link`, legacy top-level
`cid`) are post_extract's `_blob_cid`, so the sidecar and the post extract
accept exactly the same CIDs. Video blobs, avatars and banners are never
included.
"""
from __future__ import annotations

import pathlib
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # graph/
from post_extract import POST_COLLECTION, _blob_cid  # noqa: E402

import pyarrow as pa  # noqa: E402

ALT_LIMIT = 300
MAX_IMAGES = 20          # embed.images allows 4, embed.gallery 20; past this is malformed input
MAX_ASPECT = 1 << 20     # aspectRatio is a positive integer pair; reject absurd values
_MIME_RE = re.compile(r"^[a-z]{1,16}/[a-z0-9][a-z0-9.+-]{0,47}$")
_TID_ALPHABET = "234567abcdefghijklmnopqrstuvwxyz"
_TID_LOOKUP = {ch: i for i, ch in enumerate(_TID_ALPHABET)}

# The folded sidecar (one row per image / link-card thumb of a live post's latest
# revision), and the per-segment part files the fold reads. Parts carry the TID time
# as an int plus `row_type`, `operation` and `seq` for the latest-revision fold.
_MEDIA_FIELDS = [
    ("rev", pa.string()),
    ("kind", pa.string()),             # image | external_thumb
    ("image_index", pa.int16()),       # position in embed.images (external_thumb: 0)
    ("cid", pa.string()),
    ("mime", pa.string()),
    ("alt", pa.string()),              # <= ALT_LIMIT chars; external_thumb: the card title
    ("aspect_w", pa.int32()),
    ("aspect_h", pa.int32()),
    ("source", pa.string()),           # images | gallery | recordWithMedia | external
]
SIDECAR_SCHEMA = pa.schema(
    [("did", pa.string()), ("rkey", pa.string()),
     ("post_created", pa.timestamp("us", tz="UTC"))]   # TID-decoded from rkey; null if rkey is not a TID
    + _MEDIA_FIELDS
)
PART_SCHEMA = pa.schema(
    [("did", pa.string()), ("rkey", pa.string()), ("post_created_us", pa.int64())]
    + _MEDIA_FIELDS
    + [
        ("row_type", pa.string()),         # media | marker
        ("operation", pa.string()),        # create | update | delete
        ("seq", pa.uint64()),
    ]
)


def tid_to_us(tid: object) -> int | None:
    """Microsecond timestamp of a 13-char base32-sortable TID, or None. The top bit
    must be 0 and the value must land in 2023-01-01..2100-01-01 (Bluesky's own
    sanity window for rkeys used as post times)."""
    if not isinstance(tid, str) or len(tid) != 13:
        return None
    val = 0
    for ch in tid:
        idx = _TID_LOOKUP.get(ch)
        if idx is None:
            return None
        val = (val << 5) | idx
    if val >> 63:
        return None
    us = val >> 10
    if not 1_672_531_200_000_000 <= us < 4_102_444_800_000_000:
        return None
    return us


def us_to_iso(us: int | None) -> str:
    if us is None:
        return ""
    return datetime.fromtimestamp(us / 1e6, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _mime(blob: object) -> str:
    if not isinstance(blob, dict):
        return ""
    m = blob.get("mimeType")
    return m if isinstance(m, str) and _MIME_RE.match(m) else ""


def _alt(value: object) -> str:
    return value[:ALT_LIMIT] if isinstance(value, str) else ""


def _aspect(image: dict) -> tuple[int | None, int | None]:
    ar = image.get("aspectRatio")
    if not isinstance(ar, dict):
        return None, None
    w, h = ar.get("width"), ar.get("height")
    if (isinstance(w, int) and isinstance(h, int) and not isinstance(w, bool) and not isinstance(h, bool)
            and 0 < w <= MAX_ASPECT and 0 < h <= MAX_ASPECT):
        return w, h
    return None, None


def media_items(embed: object) -> list[dict]:
    """(kind, image_index, cid, mime, alt, aspect_w, aspect_h, source) dicts for one embed,
    in record order; items whose CID fails validation are skipped (their index is kept)."""
    if not isinstance(embed, dict):
        return []
    source = None
    if embed.get("$type") == "app.bsky.embed.recordWithMedia":
        embed = embed.get("media")
        if not isinstance(embed, dict):
            return []
        source = "recordWithMedia"
    kind = embed.get("$type")
    out: list[dict] = []
    if kind == "app.bsky.embed.images" and isinstance(embed.get("images"), list):
        for idx, image in enumerate(embed["images"][:MAX_IMAGES]):
            if not isinstance(image, dict):
                continue
            blob = image.get("image")
            cid = _blob_cid(blob)
            if not cid:
                continue
            w, h = _aspect(image)
            out.append({"kind": "image", "image_index": idx, "cid": cid, "mime": _mime(blob),
                        "alt": _alt(image.get("alt")), "aspect_w": w, "aspect_h": h,
                        "source": source or "images"})
    elif kind == "app.bsky.embed.gallery" and isinstance(embed.get("items"), list):
        for idx, item in enumerate(embed["items"][:MAX_IMAGES]):
            if not isinstance(item, dict) or item.get("$type", "app.bsky.embed.gallery#image") not in (
                    "app.bsky.embed.gallery#image", "app.bsky.embed.gallery"):
                continue
            blob = item.get("image")
            cid = _blob_cid(blob)
            if not cid:
                continue
            w, h = _aspect(item)
            out.append({"kind": "image", "image_index": idx, "cid": cid, "mime": _mime(blob),
                        "alt": _alt(item.get("alt")), "aspect_w": w, "aspect_h": h,
                        "source": source or "gallery"})
    elif kind == "app.bsky.embed.external" and isinstance(embed.get("external"), dict):
        ext = embed["external"]
        cid = _blob_cid(ext.get("thumb"))
        if cid:
            out.append({"kind": "external_thumb", "image_index": 0, "cid": cid, "mime": _mime(ext.get("thumb")),
                        "alt": _alt(ext.get("title")), "aspect_w": None, "aspect_h": None,
                        "source": source or "external"})
    return out


def rows_for_event(ev: dict) -> list[dict]:
    """PART_SCHEMA rows for one decoded commit payload; [] for anything else.

    Creates without media emit no row. A no-media create can only supersede an
    earlier media revision of the same (did, rkey) if a delete came between them,
    and that delete is itself a marker, so the fold's answer is unchanged. Skipping
    them keeps the parts ~4x smaller (most posts have no image)."""
    if ev.get("collection") != POST_COLLECTION:
        return []
    op = ev.get("operation")
    did, rkey = ev.get("did"), ev.get("rkey")
    if op not in ("create", "update", "delete") or not isinstance(did, str) or not isinstance(rkey, str):
        return []
    rev = ev.get("rev") if isinstance(ev.get("rev"), str) else ""
    seq = ev.get("seq") if isinstance(ev.get("seq"), int) and ev.get("seq") >= 0 else 0
    base = {"did": did, "rkey": rkey, "post_created_us": tid_to_us(rkey), "rev": rev,
            "operation": op, "seq": seq}
    items = []
    if op != "delete":
        rec = ev.get("record")
        items = media_items(rec.get("embed")) if isinstance(rec, dict) else []
    if items:
        return [{**base, **it, "row_type": "media"} for it in items]
    if op == "create":
        return []
    return [{**base, "kind": "", "image_index": None, "cid": "", "mime": "", "alt": "",
             "aspect_w": None, "aspect_h": None, "source": "", "row_type": "marker"}]


# Byte-level prefilter for decoder JSONL: a create line without an embed can emit
# nothing, so it is skipped without a JSON parse. toWirePayload writes `operation`
# second, right after `$type`.
_CREATE_PREFIX = b'"operation":"create"'


def needs_parse(line: bytes) -> bool:
    return b'"embed"' in line or _CREATE_PREFIX not in line[:96]
