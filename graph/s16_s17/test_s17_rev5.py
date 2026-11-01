#!/usr/bin/env python3
"""TDD regression suite for S17 rev 5:
1. Post-eviction & post-rehoming curated-list hygiene (`k >= 2`, no junk/personal/F4F tags) & stale description cleanup
2. Chimera / loose-mix card eviction & Leiden twin-card consolidation (preserving distinct clubs/regions/engines)
3. Multi-signal proper re-homing (`S12m` clean post + multimodal fused + `S16` Tier-1 + `known_for` rank 1/2 + curated lists + global `Y3` guardrail + `approved_dids`)
"""
from __future__ import annotations

import json
import pathlib
import re
import unittest

import build_s17

LAKE = pathlib.Path.home() / "catbird-lake-local"
OUT_DIR = LAKE / "s17-picker-v1-20261030"


class TestS17Rev5(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cards_v1 = json.loads((OUT_DIR / "cards_v1.json").read_text())
        cls.descs = json.loads((OUT_DIR / "cards_descriptions.json").read_text())
        cls.insp = json.loads((OUT_DIR / "inspector_index.json").read_text())
        cls.verify = json.loads((OUT_DIR / "verify.json").read_text())
        cls.receipt = json.loads((OUT_DIR / "receipt.json").read_text())

        cls.did_to_aid: dict[str, int] = {}
        cls.handle_to_aid: dict[str, int] = {}
        for aid, row in enumerate(cls.cards_v1["accts"]):
            d = build_s17.expand_did(row[0])
            h = build_s17.expand_handle(row[1]).lower()
            cls.did_to_aid[d] = aid
            cls.handle_to_aid[h] = aid

    def test_catalog_version_budget_and_zero_violations(self) -> None:
        self.assertEqual(self.cards_v1.get("version"), "onboarding_v0_20261031_rev5")
        self.assertTrue(self.cards_v1.get("rev5_rehoming_and_consolidation"))
        self.assertTrue(self.verify.get("zero_safety_violations"))
        self.assertEqual(self.verify.get("violations_count"), 0)
        raw_size = (OUT_DIR / "cards_v1.json").stat().st_size
        self.assertLessEqual(raw_size, 2_097_152)

    def test_card_2421_and_catalog_list_and_description_hygiene(self) -> None:
        self.assertIn("2421", self.cards_v1["cards"])
        c2421 = self.cards_v1["cards"]["2421"]
        ln_names = [n.lower() for n, _k in c2421.get("ln", [])]
        self.assertNotIn("cats on bluesky", ln_names)
        self.assertNotIn("my feed", ln_names)

        d2421 = self.descs["2421"]["d"].lower()
        self.assertNotIn("pet-cat", d2421)
        self.assertNotIn("miscellaneous chatter", d2421)

        # Across entire catalog: every displayed curated list has >= 2 independent curators and no junk/F4F list names
        junk_re = re.compile(r"^(?:my feed|friends|close friends|favorites|favourites|main|01|𝕏)$|تابعني", re.I)
        for cid, cd in self.cards_v1["cards"].items():
            for nm, k in cd.get("ln", []):
                self.assertGreaterEqual(int(k), 2, f"Card {cid} kept k<2 curated list {nm!r} ({k})")
                self.assertIsNone(junk_re.search(nm), f"Card {cid} kept junk list name {nm!r}")
            for nm, k in self.insp["card_health"][cid].get("all_lists", []):
                self.assertGreaterEqual(int(k), 2, f"Card {cid} inspector list k<2: {nm!r} ({k})")

        # Stale outlier/politics clauses stripped from descriptions
        for cid, dobj in self.descs.items():
            txt = dobj["d"]
            self.assertIsNone(
                re.search(r"pet-cat accounts|unrelated construction/bim|anti-maga|^a loose mix of\b", txt, re.I),
                f"Card {cid} still has stale pre-eviction blurb: {txt}",
            )

    def test_chimera_drops_and_twin_card_consolidation(self) -> None:
        cards = self.cards_v1["cards"]
        for bad_cid in ("3189", "1264", "3670", "3114", "2536", "2594", "3633", "1690", "3635"):
            self.assertNotIn(bad_cid, cards, f"Chimera/loose-mix card {bad_cid} should be dropped")

        # Exact-title "Indie game developers" over-splits consolidated into [651]
        indie_gd_cids = [cid for cid, cd in cards.items() if cd["t"].strip().lower() == "indie game developers"]
        self.assertEqual(indie_gd_cids, ["651"])
        for absorbed in ("84", "1188", "1445", "3894", "4365", "4371"):
            self.assertNotIn(absorbed, cards, f"Twin card {absorbed} should be consolidated into its canonical card")

        # Distinct regional, engine, and rival-club cards MUST remain distinct
        for distinct_cid in ("1951", "1456", "1213", "2421", "164", "418", "901"):
            self.assertIn(distinct_cid, cards, f"Distinct community card {distinct_cid} must be preserved")

    def test_multi_signal_rehoming_and_account_hygiene(self) -> None:
        cards = self.cards_v1["cards"]
        rehome_summary = self.insp.get("rehome_summary") or {}
        self.assertGreaterEqual(int(rehome_summary.get("rehomed_eligible_total") or 0), 400)
        self.assertGreaterEqual(int(rehome_summary.get("merged_twin_cards") or 0), 15)

        # Cats evicted from [2421] and routed into [117] Cat accounts of Bluesky
        c2421_handles = {
            build_s17.expand_handle(self.cards_v1["accts"][aid][1]).lower()
            for aid in cards["2421"]["m"]
        }
        for cat_h in ("salad.acegikmo.com", "thor.acegikmo.com", "gumpy.acegikmo.com"):
            self.assertNotIn(cat_h, c2421_handles)

        rehomed_to_117 = set(self.insp["card_health"]["117"].get("rehomed_handles_sample") or [])
        self.assertTrue(
            {"salad.acegikmo.com", "gumpy.acegikmo.com", "unicouniuni3.bsky.social"} & rehomed_to_117,
            f"Expected cat accounts re-homed into [117], got {rehomed_to_117}",
        )

        # Josh's manual approvals (@vonkoz.bsky.social, @videospacegames.bsky.social) are re-homed into the catalog
        rehomed_all_dids = set(rehome_summary.get("approved_dids_rehomed") or [])
        self.assertIn("did:plc:26b2hutoxkosr7hhgb7qorek", rehomed_all_dids)
        self.assertIn("did:plc:zxta5pxqrqbz57z534dpgjp2", rehomed_all_dids)

        # Manual exclusions & newly caught adult/NSFW/celebrity interlopers absent from catalog
        for blocked_h in (
            "vanity.blue",
            "lulumutt.bsky.social",
            "myrmecophilia.bug.sex",
            "onlycosmia.bsky.social",
            "thestormydaniels.bsky.social",
            "monopolymermaid.bsky.social",
            "craigweekends.bsky.social",
        ):
            self.assertNotIn(blocked_h, self.handle_to_aid, f"Blocked/interloper handle {blocked_h} leaked into accts")

        # Zero negative post-cosine interlopers remaining in any v0 card
        for aid, meta in enumerate(self.insp["acct_meta"]):
            sp = meta.get("sp")
            if sp is not None:
                self.assertGreaterEqual(
                    float(sp),
                    0.12,
                    f"Candidate {self.cards_v1['accts'][aid][1]} still has low post cosine sp={sp} in its assigned card",
                )


if __name__ == "__main__":
    unittest.main()
