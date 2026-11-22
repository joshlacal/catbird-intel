"""Unit tests for the unattended broadcast-feed screen (`feed_screen.py`). No lake access."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from feed_screen import REASON, is_broadcast_only, unattended_feed_reason  # noqa: E402

# Shape of the account that prompted the screen: a job-listing auto-poster that the picker
# kept because each older single-axis threshold narrowly missed it.
LISTING_BOT = {
    "pq_posts": 313,
    "pq_originals": 313,
    "pq_mutual_share": 0.0,
    "pq_light_engagers": 6,
    "slab_n_posts": 64,
    "slab_near_dup_frac": 0.47,
    "slab_nn_mean_cos": 0.90,
}


def variant(**changes: object) -> dict[str, object]:
    return {**LISTING_BOT, **changes}


def test_listing_bot_is_screened() -> None:
    assert unattended_feed_reason(LISTING_BOT) == REASON


def test_engaged_publication_is_kept() -> None:
    # Broadcast-only and isolated, but readers engage: a publication people chose to follow.
    assert unattended_feed_reason(variant(pq_light_engagers=780, slab_near_dup_frac=0.0, slab_nn_mean_cos=0.6)) is None
    # Templated and engaged (for example a daily prompt account).
    assert unattended_feed_reason(variant(pq_light_engagers=400)) is None


def test_templated_feeds_need_more_engagement() -> None:
    # 0.25 engagers per post: fine for varied posts, not for a templated feed.
    quiet = variant(pq_posts=40, pq_originals=40, pq_light_engagers=10)
    assert unattended_feed_reason(quiet) == REASON
    assert unattended_feed_reason({**quiet, "slab_near_dup_frac": 0.05, "slab_nn_mean_cos": 0.6}) is None
    # Too few embedded posts to call the feed templated.
    assert unattended_feed_reason({**quiet, "slab_n_posts": 8}) is None


def test_people_are_kept() -> None:
    # Replies or reposts in the window.
    assert unattended_feed_reason(variant(pq_originals=250)) is None
    # Mutual ties to the scene.
    assert unattended_feed_reason(variant(pq_mutual_share=0.08)) is None
    # Too few posts to judge.
    assert unattended_feed_reason(variant(pq_posts=20, pq_originals=20, pq_light_engagers=0)) is None


def test_missing_signals_never_exclude() -> None:
    assert unattended_feed_reason(None) is None
    assert unattended_feed_reason({}) is None
    for key in ("pq_posts", "pq_originals", "pq_mutual_share", "pq_light_engagers"):
        assert unattended_feed_reason(variant(**{key: None})) is None
        assert unattended_feed_reason(variant(**{key: float("nan")})) is None


def test_broadcast_only() -> None:
    assert is_broadcast_only(LISTING_BOT)
    assert not is_broadcast_only(variant(pq_originals=250))
    assert not is_broadcast_only(variant(pq_posts=10, pq_originals=10))
    assert not is_broadcast_only(None)
