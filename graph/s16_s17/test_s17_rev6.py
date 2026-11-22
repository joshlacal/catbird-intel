#!/usr/bin/env python3
"""TDD verification suite for S17 rev 6 (`intel_serving_v2.json` & `s17_rev6_experiments.json`)."""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import sys

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_intel_serving_v2 import CATALOG_VERSION  # noqa: E402
from feed_screen import unattended_feed_reason  # noqa: E402

LAKE = pathlib.Path("/Users/joshlacalamito/catbird-lake-local")
S17_DIR = LAKE / "s17-picker-v1-20261030"
# INTEL_SERVING_BUNDLE_PATH points the suite at a candidate bundle before it replaces the lake file.
BUNDLE_PATH = pathlib.Path(os.environ.get("INTEL_SERVING_BUNDLE_PATH") or S17_DIR / "intel_serving_v2.json")
EXP_PATH = S17_DIR / "s17_rev6_experiments.json"

ALLOWED_MACRO_CATEGORIES = {
    "arts_illustration",
    "games_interactive",
    "software_ai_tech",
    "science_nature",
    "music_audio",
    "film_tv_anime",
    "books_writing_comics",
    "sports_motorsport",
    "cities_craft_food",
    "news_civics_history",
    "queer_fandom_hangout",
    "world_languages",
}

ALLOWED_ROLES = {
    "anchor_creator",
    "visual_maker",
    "conversationalist",
    "scene_curator",
    "project_or_studio",
}

ALLOWED_MODES = {"follow_all", "originals_only", "topic_feed_only"}
AT_POST_RE = re.compile(r"^at://did:(?:plc|web):[^/]+/app\.bsky\.feed\.post/[a-z0-9]+$")


@pytest.fixture(scope="module")
def bundle() -> dict:
    assert BUNDLE_PATH.exists(), f"Missing {BUNDLE_PATH}"
    return json.loads(BUNDLE_PATH.read_text())


@pytest.fixture(scope="module")
def experiments() -> dict:
    assert EXP_PATH.exists(), f"Missing {EXP_PATH}"
    return json.loads(EXP_PATH.read_text())


def test_safety_and_skeleton_only_invariants(bundle: dict) -> None:
    assert bundle["catalogVersion"] == CATALOG_VERSION
    assert len(bundle["communities"]) == 1016
    assert len(bundle["scenes"]) == 104

    # Load S5 safety sets
    purge_dids = set((LAKE / "s5-sweep-20261030/purge_dids.txt").read_text().splitlines())
    non_org = set((S17_DIR / "non_organic_exclude_dids.txt").read_text().splitlines())
    overrides = json.loads((S17_DIR / "manual_overrides.json").read_text())
    manual_excl = set(overrides.get("excluded_dids", []))

    prof = pq.read_table(
        LAKE / "s5-sweep-20261030/profiles.parquet",
        columns=["did", "hide_from_recs", "hard_label", "any_mod_label", "no_unauthenticated"],
    )
    dids_col = prof["did"].to_pylist()
    hfr_col = prof["hide_from_recs"].to_pylist()
    mod_col = prof["any_mod_label"].to_pylist()
    nounauth_col = prof["no_unauthenticated"].to_pylist()

    hfr_dids = {d for d, f in zip(dids_col, hfr_col) if f}
    mod_dids = {d for d, f in zip(dids_col, mod_col) if f}
    expected_no_unauth = {d for d, f in zip(dids_col, nounauth_col) if f}

    bundle_no_unauth = set(bundle["noUnauthDids"])
    all_catalog_dids: set[str] = set()

    for comm in bundle["communities"]:
        # Ensure zero raw post text or user bios exist on community objects
        assert "rawPosts" not in comm
        for uri in comm["samplePostUris"]:
            assert AT_POST_RE.match(uri), f"Invalid post AT-URI {uri}"
            owner = uri.split("/")[2]
            assert owner not in hfr_dids, f"Sample post from hide_from_recs DID {owner}"
            assert owner not in purge_dids, f"Sample post from purged DID {owner}"
            assert owner not in mod_dids, f"Sample post from mod-labelled DID {owner}"

        for m in comm["members"]:
            did = m["did"]
            all_catalog_dids.add(did)
            assert "bio" not in m and "text" not in m
            assert did not in hfr_dids, f"hide_from_recs DID leaked: {did}"
            assert did not in purge_dids, f"purged DID leaked: {did}"
            assert did not in mod_dids, f"mod-labelled DID leaked: {did}"
            assert did not in non_org, f"non_organic DID leaked: {did}"
            assert did not in manual_excl, f"manual excluded DID leaked: {did}"
            if did in expected_no_unauth:
                assert did in bundle_no_unauth, f"!no-unauthenticated DID {did} missing from noUnauthDids"


def test_no_unattended_broadcast_feeds_are_served(bundle: dict) -> None:
    overrides = json.loads((S17_DIR / "manual_overrides.json").read_text())
    approved = set(overrides.get("approved_dids", []))
    columns = [
        "did",
        "pq_posts",
        "pq_originals",
        "pq_mutual_share",
        "pq_light_engagers",
        "slab_n_posts",
        "slab_near_dup_frac",
        "slab_nn_mean_cos",
    ]
    table = pq.read_table(S17_DIR / "candidate_signals.parquet", columns=columns)
    cols = {name: table[name].to_pylist() for name in columns}
    screened = {
        did
        for idx, did in enumerate(cols["did"])
        if did not in approved and unattended_feed_reason({name: cols[name][idx] for name in columns[1:]})
    }
    assert screened, "The screen matched nothing; candidate_signals columns may have changed"

    for comm in bundle["communities"]:
        for m in comm["members"]:
            assert m["did"] not in screened, f"Unattended broadcast feed served as a member: {m['did']}"
        for uri in comm["samplePostUris"] + comm["previewPostUris"]:
            assert uri.split("/")[2] not in screened, f"Post from an unattended broadcast feed served: {uri}"
        for did in comm["sampleAccountDids"]:
            assert did not in screened, f"Unattended broadcast feed served as a sample account: {did}"
    for duel in bundle["tasteDuels"]:
        for side in ("optionA", "optionB"):
            for uri in duel[side]["samplePostUris"]:
                assert uri.split("/")[2] not in screened, f"Duel post from an unattended broadcast feed: {uri}"


def test_all_104_scenes_have_clean_human_titles_categories_and_symbols(bundle: dict) -> None:
    slugs = set()
    titles = set()
    for sc in bundle["scenes"]:
        title = sc["sceneTitle"]
        slug = sc["sceneSlug"]
        cat = sc["macroCategory"]
        sym = sc["symbolName"]
        assert " / " not in title, f"Raw slash list-pair leaked into sceneTitle: {title!r}"
        assert "sleep paralysis" not in title.lower()
        assert "🚀" not in title
        assert len(title) >= 4
        assert slug not in slugs, f"Duplicate sceneSlug: {slug}"
        slugs.add(slug)
        titles.add(title)
        assert cat in ALLOWED_MACRO_CATEGORIES, f"Invalid macroCategory {cat}"
        assert isinstance(sym, str) and len(sym) >= 3
        assert sc["cardCount"] >= 1
    # Titles should be distinct across scenes
    assert len(titles) == 104, f"Expected 104 distinct human scene titles, got {len(titles)}"


def test_medoid_backfill_and_preview_coverage(bundle: dict) -> None:
    ge_3 = sum(1 for c in bundle["communities"] if len(c["samplePostUris"]) >= 3)
    ge_1 = sum(1 for c in bundle["communities"] if len(c["samplePostUris"]) >= 1)
    with_preview = sum(1 for c in bundle["communities"] if len(c["previewPostUris"]) >= 3)
    assert ge_3 >= 1006, f"Expected >= 1006/1016 cards with >=3 sample posts after backfill, got {ge_3}"
    assert ge_1 >= 1014, f"Expected >= 1014/1016 cards with >=1 sample post, got {ge_1}"
    assert with_preview >= 1000, f"Expected >= 1000/1016 cards with >=3 preview posts, got {with_preview}"


def test_account_roles_delivery_modes_and_explanations(bundle: dict) -> None:
    role_counts = {r: 0 for r in ALLOWED_ROLES}
    originals_only_count = 0
    total_members = 0
    for c in bundle["communities"]:
        for m in c["members"]:
            total_members += 1
            role = m["role"]
            mode = m["recommendedMode"]
            assert role in ALLOWED_ROLES
            assert mode in ALLOWED_MODES
            role_counts[role] += 1
            assert isinstance(m["whyPicked"], str) and 8 <= len(m["whyPicked"]) <= 110
            assert 0 <= m["repostShareBps"] <= 10000
            assert 0 <= m["mediaShareBps"] <= 10000
            assert 0 <= m["estimatedPostsPerWeek"] <= 500
            if m["repostShareBps"] >= 4500:
                assert mode == "originals_only", f"Heavy reposter ({m['repostShareBps']} bps) should recommend originals_only"
            if mode == "originals_only":
                originals_only_count += 1

    assert total_members >= 25000
    assert originals_only_count >= 500, f"Expected meaningful originals_only detection, got {originals_only_count}"
    # Role diversity Shannon entropy >= 1.75 bits across the 5 roles
    entropy = -sum(
        (cnt / total_members) * math.log2(cnt / total_members)
        for cnt in role_counts.values()
        if cnt > 0
    )
    assert entropy >= 1.75, f"Role entropy too degenerate: {entropy:.3f} bits ({role_counts})"


def test_taste_duels_and_hill_climb_experiments(bundle: dict, experiments: dict) -> None:
    duels = bundle["tasteDuels"]
    assert len(duels) >= 40, f"Expected >= 40 contrastive taste duels, got {len(duels)}"
    trap_ids = {d["swipeId"] for d in duels if d["kind"] == "collision_trap"}
    assert len(trap_ids) >= 8, f"Expected all 8 classic collision traps, got {trap_ids}"
    for d in duels:
        assert d["optionA"]["communityId"] != d["optionB"]["communityId"]
        assert len(d["optionA"]["samplePostUris"]) >= 1
        assert len(d["optionB"]["samplePostUris"]) >= 1

    # Verify M2-val ONLY guard and quantitative hill-climb improvements
    assert experiments["validation_split_guard"] == "M2-val_only_never_M2-test"
    solver_exp = experiments["solver_hill_climb"]
    assert solver_exp["submodular_mean_bundle_role_entropy_bits"] > solver_exp["greedy_mean_bundle_role_entropy_bits"] + 0.15
    assert solver_exp["submodular_role_entropy_bits"] > solver_exp["greedy_role_entropy_bits"]
    assert solver_exp["submodular_p95_daily_posts_with_tamer"] < solver_exp["greedy_p95_daily_posts_raw"] * 0.65
    assert solver_exp["submodular_high_hostility_pair_rate"] == 0.0
    assert solver_exp["m2_val_recall_at_25_ratio_vs_greedy"] >= 0.98

    search_exp = experiments["search_hill_climb"]
    assert search_exp["hybrid_recall_at_10"] >= 0.90
    assert search_exp["hybrid_recall_at_10"] > search_exp["title_only_recall_at_10"] + 0.15
