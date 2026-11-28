#!/usr/bin/env python3
"""The "why picked" line for core-scene accounts, and an in-place relabel of built catalog files.

The catalog used to say "Core scene anchor · strong original posts" for every account that fell
through the other roles, a quality claim nothing measured: blog self-promotion feeds whose posts are
headline links got it too. The line now states only what was measured:

- accounts whose own posts in the export window are mostly links: "Shares articles and links"
- everyone else in the role: "Core scene anchor · mostly original posts" (the role already means
  a repost share below 42%)

Link counts come from `build_onboarding_signals.py --link-stats-out` (own non-reply posts, and how
many of them carry a link, over the activity window).

Relabel usage (catalog or pool file; only `whyPicked` of `anchor_creator` members changes):
  why_picked.py --link-stats link_stats.json --in intel_serving_v2.json --out intel_serving_v2.whyfix.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

# Own posts in the window needed before the link share is trusted.
MIN_POSTS = 5
# Share of own posts carrying a link at which the account is described as a link sharer.
MIN_LINK_SHARE = 0.5
MAX_LEN = 108

_PPW = re.compile(r"~(\d+)/wk")


def load_link_stats(path: pathlib.Path | str | None) -> dict[str, tuple[int, int]]:
    """{did: (own posts, linked own posts)} from a --link-stats-out file; {} when no path."""
    if not path:
        return {}
    doc = json.loads(pathlib.Path(path).read_text())
    return {did: (int(v[0]), int(v[1])) for did, v in doc["accounts"].items()}


def is_link_sharer(stats: tuple[int, int] | None) -> bool:
    if not stats:
        return False
    posts, linked = stats
    return posts >= MIN_POSTS and linked >= MIN_LINK_SHARE * posts


def anchor_why(est_ppw: int, stats: tuple[int, int] | None) -> str:
    if is_link_sharer(stats):
        return f"Shares articles and links · ~{est_ppw}/wk"[:MAX_LEN]
    return f"Core scene anchor · mostly original posts · ~{est_ppw}/wk"[:MAX_LEN]


def relabel(doc: dict[str, Any], stats: dict[str, tuple[int, int]]) -> dict[str, int]:
    """Rewrite `whyPicked` of anchor_creator members in a catalog or pool document. Returns counts."""
    counts = {"anchors": 0, "linkSharers": 0, "changed": 0}
    for community in doc["communities"]:
        for m in community["members"]:
            if m.get("role") != "anchor_creator":
                continue
            counts["anchors"] += 1
            found = _PPW.search(str(m.get("whyPicked") or ""))
            ppw = int(found.group(1)) if found else int(m.get("estimatedPostsPerWeek") or 1)
            sharer = is_link_sharer(stats.get(m["did"]))
            counts["linkSharers"] += int(sharer)
            new = anchor_why(ppw, stats.get(m["did"]))
            if suffix := (" · Originals Only suggested" if "Originals Only suggested" in str(m.get("whyPicked")) else ""):
                new = (new + suffix)[:MAX_LEN]
            if new != m.get("whyPicked"):
                m["whyPicked"] = new
                counts["changed"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--link-stats", type=pathlib.Path, required=True)
    ap.add_argument("--in", dest="src", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(argv)
    if a.out.resolve() == a.src.resolve():
        print("refusing to overwrite the input; write beside it", file=sys.stderr)
        return 2
    doc = json.loads(a.src.read_text())
    counts = relabel(doc, load_link_stats(a.link_stats))
    a.out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(json.dumps(counts), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
