"""Post rows for archive_graph_extract.py --content posts.

One row per app.bsky.feed.post commit (create, update or delete). Deletes are
kept so downstream jobs can drop deleted posts before embedding. Text is kept
in full (posts are capped at 300 graphemes upstream); image bytes never exist
here, only counts and alt text.
"""
from __future__ import annotations

import pyarrow as pa

POST_COLLECTION = "app.bsky.feed.post"

POST_SCHEMA = pa.schema(
    [
        ("source_id", pa.string()),
        ("seq", pa.uint64()),
        ("time", pa.string()),
        ("did", pa.string()),
        ("rkey", pa.string()),
        ("rev", pa.string()),
        ("operation", pa.string()),
        ("created_at", pa.string()),
        ("text", pa.string()),
        ("langs", pa.list_(pa.string())),
        ("reply_root", pa.string()),
        ("reply_parent", pa.string()),
        ("embed_type", pa.string()),
        ("quote_uri", pa.string()),
        ("external_uri", pa.string()),
        ("image_count", pa.int32()),
        ("image_alts", pa.list_(pa.string())),
        ("facet_links", pa.list_(pa.string())),
        ("facet_tags", pa.list_(pa.string())),
        ("facet_mentions", pa.list_(pa.string())),
        ("self_labels", pa.list_(pa.string())),
        # Appended last so positional readers of the columns above keep working.
        ("image_cids", pa.list_(pa.string())),
    ]
)

_TEXT_LIMIT = 8000
_ALT_LIMIT = 2000


def _str(value: object, limit: int = 2048) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _strs(values: object, limit: int = 2048) -> list[str]:
    if not isinstance(values, list):
        return []
    return [v[:limit] for v in values if isinstance(v, str)]


def _uri(ref: object) -> str:
    return _str(ref.get("uri")) if isinstance(ref, dict) else ""


def _media(media: dict) -> tuple[int, list[str], str]:
    """Image count, alt texts and external link for an images/video/external embed."""
    kind = _str(media.get("$type"))
    if kind == "app.bsky.embed.images":
        images = [i for i in media.get("images") or [] if isinstance(i, dict)] if isinstance(media.get("images"), list) else []
        return len(images), [_str(i.get("alt"), _ALT_LIMIT) for i in images], ""
    if kind == "app.bsky.embed.video":
        return 1, [_str(media.get("alt"), _ALT_LIMIT)], ""
    if kind == "app.bsky.embed.external" and isinstance(media.get("external"), dict):
        return 0, [], _str(media["external"].get("uri"))
    return 0, [], ""


_CID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz234567")


def _blob_cid(blob: object) -> str:
    """CID of a blob ref: `ref.$link` (current form) or the legacy top-level `cid` string. Only
    base32 CIDv1 strings (`b...`, lowercase base32) up to 128 chars pass, so a CID can be placed
    in a CDN URL path verbatim; anything else is dropped."""
    if not isinstance(blob, dict):
        return ""
    ref = blob.get("ref")
    cid = ref.get("$link") if isinstance(ref, dict) else ref if isinstance(ref, str) else blob.get("cid")
    if not isinstance(cid, str) or not 8 <= len(cid) <= 128 or cid[0] != "b" or not set(cid) <= _CID_CHARS:
        return ""
    return cid


def _image_cids(embed: object) -> list[str]:
    """Blob CIDs of `app.bsky.embed.images` images, directly or as recordWithMedia's media, in
    record order; images whose blob CID is missing or malformed are skipped. Video thumbnails,
    external-link thumbnails, avatars and banners are never included."""
    if not isinstance(embed, dict):
        return []
    if embed.get("$type") == "app.bsky.embed.recordWithMedia":
        embed = embed.get("media")
        if not isinstance(embed, dict):
            return []
    if embed.get("$type") != "app.bsky.embed.images" or not isinstance(embed.get("images"), list):
        return []
    cids = [_blob_cid(i.get("image")) for i in embed["images"] if isinstance(i, dict)]
    return [c for c in cids if c]


def _embed(embed: object) -> tuple[str, str, str, int, list[str]]:
    """(embed_type, quote_uri, external_uri, image_count, image_alts)."""
    if not isinstance(embed, dict):
        return "", "", "", 0, []
    kind = _str(embed.get("$type"))
    if kind == "app.bsky.embed.record":
        return kind, _uri(embed.get("record")), "", 0, []
    if kind == "app.bsky.embed.recordWithMedia":
        record = embed.get("record")
        quote = _uri(record.get("record")) if isinstance(record, dict) else ""
        media = embed.get("media") if isinstance(embed.get("media"), dict) else {}
        count, alts, external = _media(media)
        return kind, quote, external, count, alts
    count, alts, external = _media(embed)
    return kind, "", external, count, alts


def _facets(facets: object) -> tuple[list[str], list[str], list[str]]:
    links: list[str] = []
    tags: list[str] = []
    mentions: list[str] = []
    if not isinstance(facets, list):
        return links, tags, mentions
    for facet in facets:
        features = facet.get("features") if isinstance(facet, dict) else None
        if not isinstance(features, list):
            continue
        for feat in features:
            if not isinstance(feat, dict):
                continue
            kind = feat.get("$type")
            if kind == "app.bsky.richtext.facet#link" and isinstance(feat.get("uri"), str):
                links.append(feat["uri"][:2048])
            elif kind == "app.bsky.richtext.facet#tag" and isinstance(feat.get("tag"), str):
                tags.append(feat["tag"][:640])
            elif kind == "app.bsky.richtext.facet#mention" and isinstance(feat.get("did"), str):
                mentions.append(feat["did"][:2048])
    return links, tags, mentions


def _self_labels(labels: object) -> list[str]:
    if not isinstance(labels, dict) or not isinstance(labels.get("values"), list):
        return []
    return [_str(v.get("val"), 128) for v in labels["values"] if isinstance(v, dict) and isinstance(v.get("val"), str)]


def post_row_from_event(base: dict) -> dict:
    """Build a post row from archive_graph_extract.row_from_event's common fields
    plus the raw record (base["record"])."""
    rec = base.get("record") if isinstance(base.get("record"), dict) else {}
    reply = rec.get("reply") if isinstance(rec.get("reply"), dict) else {}
    embed_type, quote_uri, external_uri, image_count, image_alts = _embed(rec.get("embed"))
    links, tags, mentions = _facets(rec.get("facets"))
    tags += [t for t in _strs(rec.get("tags"), 640) if t not in tags]
    return {
        "source_id": base["source_id"],
        "seq": base["seq"],
        "time": base["time"],
        "did": base["did"],
        "rkey": base["rkey"],
        "rev": base["rev"],
        "operation": base["operation"],
        "created_at": _str(rec.get("createdAt"), 64),
        "text": _str(rec.get("text"), _TEXT_LIMIT),
        "langs": _strs(rec.get("langs"), 64),
        "reply_root": _uri(reply.get("root")),
        "reply_parent": _uri(reply.get("parent")),
        "embed_type": embed_type,
        "quote_uri": quote_uri,
        "external_uri": external_uri,
        "image_count": image_count,
        "image_alts": image_alts,
        "facet_links": links,
        "facet_tags": tags,
        "facet_mentions": mentions,
        "self_labels": _self_labels(rec.get("labels")),
        "image_cids": _image_cids(rec.get("embed")),
    }
