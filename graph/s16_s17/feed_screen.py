"""Unattended broadcast-feed screen, shared by the S17 picker and the serving-bundle builder.

The starting lineup is followed on the user's behalf, so every account in it should be one that
people in the scene actually interact with. This screen removes accounts that only broadcast
(no replies, reposts or quotes), have no mutual ties to the scene, and draw almost no engagement:
job-listing bots, RSS mirrors, preprint mirrors and similar auto-posters.

It reads only `candidate_signals.parquet` columns and has no third-party dependencies, so both
`build_s17.py` (rev-4 data-signal screens) and `build_intel_serving_v2.py` (final serving gate)
can import it. The older screens it complements each need an extreme value on one axis
(`pq_light_engagers <= 2`, or `slab_near_dup_frac >= 0.58`); an account that is moderately bad
on every axis passed all of them.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

REASON = "unattended_broadcast_feed"

# Enough posts in the producer-quality window to judge behaviour at all.
MIN_WINDOW_POSTS = 30
# Share of window posts that are originals (not replies, reposts or quotes).
MIN_ORIGINAL_SHARE = 0.95
# Share of the account's follows that are mutual with scene members.
MAX_MUTUAL_SHARE = 0.02
# Distinct light engagers per window post: below this nobody is reading the feed.
MAX_ENGAGERS_PER_POST = 0.10
# A templated feed (near-duplicate posts) is held to a higher engagement bar.
MAX_ENGAGERS_PER_POST_TEMPLATED = 0.40
MIN_SLAB_POSTS = 12
TEMPLATED_NEAR_DUP_FRAC = 0.25
TEMPLATED_NN_MEAN_COS = 0.85


def _num(sig: Mapping[str, Any], key: str) -> float | None:
    value = sig.get(key)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def unattended_feed_reason(sig: Mapping[str, Any] | None) -> str | None:
    """Return `REASON` when the signals describe an unattended broadcast feed, else None.

    Missing signals never exclude: an account without a producer-quality window, or without a
    measured mutual share, is kept.
    """
    if not sig:
        return None
    posts = _num(sig, "pq_posts")
    originals = _num(sig, "pq_originals")
    mutual = _num(sig, "pq_mutual_share")
    engagers = _num(sig, "pq_light_engagers")
    if posts is None or originals is None or mutual is None or engagers is None:
        return None
    if posts < MIN_WINDOW_POSTS or originals / posts < MIN_ORIGINAL_SHARE:
        return None
    if mutual > MAX_MUTUAL_SHARE:
        return None

    engagers_per_post = engagers / posts
    if engagers_per_post < MAX_ENGAGERS_PER_POST:
        return REASON

    slab_posts = _num(sig, "slab_n_posts") or 0.0
    near_dup = _num(sig, "slab_near_dup_frac") or 0.0
    nn_cos = _num(sig, "slab_nn_mean_cos") or 0.0
    templated = slab_posts >= MIN_SLAB_POSTS and (
        near_dup >= TEMPLATED_NEAR_DUP_FRAC or nn_cos >= TEMPLATED_NN_MEAN_COS
    )
    if templated and engagers_per_post < MAX_ENGAGERS_PER_POST_TEMPLATED:
        return REASON
    return None


def is_broadcast_only(sig: Mapping[str, Any] | None) -> bool:
    """True when the window shows a posting pattern with effectively no replies, reposts or quotes."""
    if not sig:
        return False
    posts = _num(sig, "pq_posts")
    originals = _num(sig, "pq_originals")
    if posts is None or originals is None or posts < MIN_WINDOW_POSTS:
        return False
    return originals / posts >= MIN_ORIGINAL_SHARE
