"""Tests for the deep onboarding pool: the picker's deep-list rules (`build_s17.deep_pool_pids`) and the
builder's pool assembly (`build_intel_serving_v2.assemble_pool_members`), plus a lake-backed check of the
written `intel_onboarding_pool_v1` file against the bundle built in the same run. The lake check is skipped
when the files are absent. Run: python test_onboarding_pool.py
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_intel_serving_v2 import assemble_pool_members, pool_rank_bps  # noqa: E402
from build_s17 import BUNDLE_DEFAULT, POP_LIMIT, deep_pool_pids  # noqa: E402

LAKE = pathlib.Path(os.environ.get("CATBIRD_LAKE", str(pathlib.Path.home() / "catbird-lake-local")))
POOL_RUN_DIR = pathlib.Path(os.environ.get("POOL_RUN_DIR", str(LAKE / "s17-pool-20261122")))
POOL_PATH = pathlib.Path(
    os.environ.get(
        "POOL_PATH", str(LAKE / "s17-picker-v1-20261030/intel_onboarding_pool_v1.candidate-20261122.json")
    )
)
BUNDLE_PATH = pathlib.Path(  # the shipping bundle; the pool's own run rebuilt it byte-equal except generatedAt
    os.environ.get(
        "POOL_BUNDLE_PATH", str(LAKE / "s17-picker-v1-20261030/intel_serving_v2.candidate-20261121.json")
    )
)
SIDECAR_PATH = pathlib.Path(os.environ.get("POOL_SIDECAR_PATH", str(POOL_RUN_DIR / "pool_candidates.json")))

MEMBER_KEYS = {
    "did", "followers", "role", "recommendedMode", "whyPicked", "rankScoreBps", "repostShareBps",
    "mediaShareBps", "estimatedPostsPerWeek", "samplePostUri",
}


# --- picker: deep_pool_pids -------------------------------------------------------------------------

def _deep(eff_sorted, cards, depth, sup=(), rehome=(), approved=(), excluded=()):
    return deep_pool_pids(
        eff_sorted,
        cards,
        depth,
        is_sup=lambda p: p in sup,
        is_unapproved_rehome=lambda p: p in rehome and p not in approved,
        is_excluded=lambda p: p in excluded,
    )


def test_top_25_verbatim_then_eff_order() -> None:
    eff_sorted = list(range(100))
    cards = list(reversed(range(27)))  # caps can reorder the top; the pool keeps the card order
    out = _deep(eff_sorted, cards, 60)
    assert out[:BUNDLE_DEFAULT] == cards[:BUNDLE_DEFAULT]
    assert out[BUNDLE_DEFAULT:] == [p for p in eff_sorted if p not in set(cards[:BUNDLE_DEFAULT])][: 60 - 25]
    assert len(out) == 60 and len(set(out)) == 60


def test_card_extras_26_27_compete_on_eff() -> None:
    # card_cands[25:27] are uncapped extras; deep positions re-rank them with everyone else by eff.
    eff_sorted = [50, 51, 25, 26] + list(range(25)) + list(range(27, 50))
    cards = list(range(27))
    out = _deep(eff_sorted, cards, 30)
    assert out[25:30] == [50, 51, 25, 26, 27]


def test_supplements_and_unapproved_rehomes_never_deep() -> None:
    eff_sorted = list(range(40))
    cards = [0, 1, 2]  # a sup and a rehome inside the top survive: the top is verbatim
    out = _deep(eff_sorted, cards, 40, sup={1, 10, 11}, rehome={2, 12, 13}, approved={13})
    assert out[:3] == [0, 1, 2]
    deep = out[3:]
    assert 10 not in deep and 11 not in deep and 12 not in deep
    assert 13 in deep  # an approved rehome is a normal deep candidate


def test_excluded_accounts_dropped_and_popular_kept() -> None:
    eff_sorted = list(range(30))
    followers = {p: (POP_LIMIT * 3 if p % 2 else 100) for p in eff_sorted}
    out = _deep(eff_sorted, [0], 30, excluded={5, 6})
    assert 5 not in out and 6 not in out
    assert sum(1 for p in out if followers[p] > POP_LIMIT) > 2  # the top-25 popularity cap is not applied deep


def test_depth_truncates_and_zero_is_empty() -> None:
    eff_sorted = list(range(200))
    cards = list(range(25))
    assert len(_deep(eff_sorted, cards, 150)) == 150
    assert _deep(eff_sorted, cards, 10) == cards[:10]
    assert _deep(eff_sorted, cards, 0) == []


# --- builder: assemble_pool_members -----------------------------------------------------------------

def _bundle_member(did: str, bps: int) -> dict:
    return {
        "did": did, "followers": 1000, "role": "core", "recommendedMode": "all", "whyPicked": "w",
        "rankScoreBps": bps, "repostShareBps": 0, "mediaShareBps": 0, "estimatedPostsPerWeek": 7,
        "samplePostUri": None,
    }


def _make_member(cand: dict, pos: int) -> dict:
    m = _bundle_member(cand["did"], pool_rank_bps(pos, float(cand.get("s13", 100.0))))
    m["followers"] = int(cand["followers"])
    return m


def _cand(did: str, eff: float, followers: int = 10, s13: float = 100.0) -> dict:
    return {"did": did, "eff": eff, "followers": followers, "s13": s13, "top": False}


def test_pool_keeps_bundle_then_eff_desc_and_blocks() -> None:
    bundle = [_bundle_member(f"did:plc:b{i}", 9000 - 10 * i) for i in range(3)]
    cands = [
        _cand("did:plc:b1", 9.0),  # already a member: not repeated
        _cand("did:plc:x1", 1.0),
        _cand("did:plc:x2", 3.0),
        _cand("did:plc:blocked", 5.0),
        _cand("did:plc:x3", 3.0, followers=50),  # ties on eff break on followers, then DID
        _cand("did:plc:x1", 1.0),  # duplicate candidate row
    ]
    out = assemble_pool_members(bundle, cands, 10, blocked={"did:plc:blocked"}, make_member=_make_member)
    assert out[:3] == bundle
    assert [m["did"] for m in out[3:]] == ["did:plc:x3", "did:plc:x2", "did:plc:x1"]
    dids = [m["did"] for m in out]
    assert len(dids) == len(set(dids))


def test_pool_bps_never_rises_after_bundle() -> None:
    bundle = [_bundle_member(f"did:plc:b{i}", 5200) for i in range(25)]
    # A huge s13 would score above the bundle's last member at an early deep position; it is clamped.
    cands = [_cand(f"did:plc:d{i}", 100.0 - i, s13=1e9 if i % 3 == 0 else 50.0) for i in range(80)]
    out = assemble_pool_members(bundle, cands, 100, blocked=set(), make_member=_make_member)
    assert len(out) == 100
    bps = [m["rankScoreBps"] for m in out[24:]]
    assert all(a >= b for a, b in zip(bps, bps[1:]))
    assert all(5000 <= m["rankScoreBps"] <= 9950 for m in out)
    assert set(out[30]) == MEMBER_KEYS


def test_pool_short_bundle_and_depth_cap() -> None:
    bundle = [_bundle_member("did:plc:b0", 9100)]
    cands = [_cand(f"did:plc:d{i}", 10.0 - i) for i in range(5)]
    out = assemble_pool_members(bundle, cands, 3, blocked=set(), make_member=_make_member)
    assert [m["did"] for m in out] == ["did:plc:b0", "did:plc:d0", "did:plc:d1"]
    assert assemble_pool_members([], cands, 2, blocked=set(), make_member=_make_member)[0]["did"] == "did:plc:d0"


# --- lake-backed (a)-(e) ----------------------------------------------------------------------------

def test_lake_pool_matches_bundle_and_rules() -> None:
    if not (POOL_PATH.exists() and BUNDLE_PATH.exists() and SIDECAR_PATH.exists()):
        print(f"  SKIP lake check (missing one of {POOL_PATH}, {BUNDLE_PATH}, {SIDECAR_PATH})")
        return
    import pyarrow.parquet as pq

    from feed_screen import unattended_feed_reason

    raw = POOL_PATH.read_text()
    pool = json.loads(raw)
    bundle = json.loads(BUNDLE_PATH.read_text())
    sidecar = json.loads(SIDECAR_PATH.read_text())
    eff = {int(cd["community"]): {c["did"]: float(c["eff"]) for c in cd["candidates"]} for cd in sidecar["cards"]}
    b_comms = {int(c["communityId"]): c["members"] for c in bundle["communities"]}

    assert set(pool) <= {"poolVersion", "generatedAt", "sourceCatalogVersion", "depth", "communities", "noUnauthDids"}
    assert re.fullmatch(r"pool_v1_\d{8}_[0-9a-f]{8}", pool["poolVersion"])
    assert pool["sourceCatalogVersion"] == bundle["catalogVersion"]
    assert pool["depth"] == sidecar["poolDepth"]

    # (c) only bundle communities (and every one of them)
    p_ids = [int(c["communityId"]) for c in pool["communities"]]
    assert len(p_ids) == len(set(p_ids)) and set(p_ids) == set(b_comms)

    deep_dids: set[str] = set()
    for comm in pool["communities"]:
        cid = int(comm["communityId"])
        members = comm["members"]
        top = b_comms[cid]
        n_top = min(BUNDLE_DEFAULT, len(top))
        # (a) the bundle's members verbatim
        assert members[:n_top] == top[:n_top], cid
        assert len(members) <= pool["depth"]
        # (d) no duplicates
        dids = [m["did"] for m in members]
        assert len(dids) == len(set(dids)), cid
        # (b) deeper members by descending eff; rankScoreBps never rises from the last bundle member on
        deep = members[len(top):]
        effs = [eff[cid][m["did"]] for m in deep]
        assert all(a >= b for a, b in zip(effs, effs[1:])), cid
        bps = [m["rankScoreBps"] for m in members[max(len(top) - 1, 0):]]
        assert all(a >= b for a, b in zip(bps, bps[1:])), cid
        for m in members:
            assert set(m) == MEMBER_KEYS, (cid, sorted(m))
            assert m["did"].startswith("did:")
            uri = m["samplePostUri"]
            assert uri is None or uri.startswith(f"at://{m['did']}/app.bsky.feed.post/")
        deep_dids.update(m["did"] for m in deep)

    # (d) no hideFromRecs DID and no DID failing the final feed gate (operator approvals win, as in the bundle)
    prof = pq.read_table(LAKE / "s5-sweep-20261030/profiles.parquet", columns=["did", "hide_from_recs"])
    hfr = {d for d, h in zip(prof["did"].to_pylist(), prof["hide_from_recs"].to_pylist()) if h}
    all_dids = {m["did"] for c in pool["communities"] for m in c["members"]}
    assert len(all_dids) == sum(len(c["members"]) for c in pool["communities"])  # one community per DID
    assert not (all_dids & hfr)
    approved = set(json.loads((POOL_RUN_DIR / "manual_overrides.json").read_text()).get("approved_dids", []))
    cols = ["did", "pq_posts", "pq_originals", "pq_mutual_share", "pq_light_engagers", "slab_n_posts",
            "slab_near_dup_frac", "slab_nn_mean_cos"]
    sig = pq.read_table(POOL_RUN_DIR / "candidate_signals.parquet", columns=cols).to_pylist()
    failing = {r["did"] for r in sig if r["did"] in all_dids and r["did"] not in approved and unattended_feed_reason(r)}
    assert not failing, sorted(failing)[:5]

    # (e) zero PII: only DIDs, numbers, enum/explanation strings and post URIs
    for banned in ('"handle"', '"displayName"', '"description"', '"bio"', '"name"', ".bsky.social"):
        assert banned not in raw, banned
    print(f"  lake pool OK: {len(p_ids)} communities, {len(all_dids)} distinct DIDs, {len(deep_dids)} deep")


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"ok {name}")
    print(f"{len(tests)} passed")
