"""Unanswered link-feed screen for the onboarding signals file (build_onboarding_signals.py).

The starting lineup is followed on the viewer's behalf, so it should hold accounts people in the
scene talk to. This screen catches accounts whose own posts are mostly links to one site and that
almost nobody replies to or quotes: blog and newsletter self-promotion, RSS mirrors, shortener
auto-posters (buff.ly, dlvr.it) and paper-alert bots.

`feed_screen.unattended_feed_reason` covers a similar class from the catalog's producer-quality
window, but that window holds fewer than 30 posts for about 89% of served accounts, so it abstains
for nearly everyone. This screen reads the 14-day export window instead (link per post, and replies
and quotes from other accounts), measured on 2026-11-08..21: it flags 275 of 25,297 served accounts.

The export has no likes, so "answered" means replies and quotes. An account with no posts in the
window, or fewer than MIN_TOP_LEVEL posts, is never flagged here.
"""
from __future__ import annotations

import re

import numpy as np

REASON = "unanswered_link_feed"

# Enough of the account's own (non-reply) posts in the window to judge it.
MIN_TOP_LEVEL = 10
# Share of those posts that carry a link (embed card or link facet).
MIN_LINK_SHARE = 0.70
# Share of the linked posts that point at the single most common site.
MIN_TOP_DOMAIN_SHARE = 0.60
# Distinct other accounts replying to or quoting the account, per own post in the window.
MAX_RESPONDERS_PER_POST = 0.05

_HOST = re.compile(r"^[a-z][a-z0-9+.-]*://(?:[^/?#@]*@)?([^/?#:]+)", re.IGNORECASE)


def link_host(url: str | None) -> str:
    """Lower-case host of a URL without a leading "www.", or "" when there is none."""
    if not url:
        return ""
    m = _HOST.match(url.strip())
    if not m:
        return ""
    host = m.group(1).lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def link_feed_mask(top_level: np.ndarray, linked: np.ndarray, top_domain: np.ndarray,
                   responders: np.ndarray) -> np.ndarray:
    """Element-wise: True where the account is an unanswered single-site link feed.

    top_level  own non-reply posts in the window
    linked     how many of those carry a link
    top_domain how many of the linked ones point at the most common site
    responders distinct other accounts that replied to or quoted the account in the window
    """
    top = np.asarray(top_level, dtype=np.float64)
    linked = np.asarray(linked, dtype=np.float64)
    top_domain = np.asarray(top_domain, dtype=np.float64)
    responders = np.asarray(responders, dtype=np.float64)
    return ((top >= MIN_TOP_LEVEL)
            & (linked >= MIN_LINK_SHARE * top)
            & (linked > 0)
            & (top_domain >= MIN_TOP_DOMAIN_SHARE * linked)
            & (responders < MAX_RESPONDERS_PER_POST * top))
