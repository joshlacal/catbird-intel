"""Numeric structural fixtures; no fabricated post text or feature-quality claims."""
import copy
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

import build_mode_influence_candidate as m


def fixture():
    names = ("p", "q", "a", "b", "c")
    account = lambda name: f"did:test:{name}"
    post_uri = lambda name, key: f"at://{account(name)}/app.bsky.feed.post/{key}"
    tables = {name: [] for name in m.TABLES}
    manifest = {
        "schema_version": "mode-influence-input-v1", "generation_id": "numeric-fixture-only",
        "cutoff_at": "2026-10-20T00:00:00Z", "window_start": "2026-10-06T00:00:00Z",
        "interests_cutoff_at": "2026-10-05T00:00:00Z", "event_clock": "validated_observed_at",
        "graph_generation": "numeric-graph", "score_generation": "numeric-scores",
        "score_dimensions": {dimension: {"model_sha256": "a" * 64, "rubric_sha256": "b" * 64,
                                        "meaning": "numeric structural fixture only"}
                             for dimension in ("politicalness", "hostility")},
    }
    tables["accounts"] = [{"did": account(name)} for name in names]
    tables["communities"] = [{"community": c, "sensitivity": "non_sensitive",
                             "permitted_for_audience_aggregation": True} for c in (0, 1)]
    tables["producers"] = [
        {"did": account("p"), "primary_community": 0, "primary_score": .5, "followers_est": 10},
        {"did": account("q"), "primary_community": 1, "primary_score": .7, "followers_est": None},
    ]
    for owner, key, day, reply, quote, politicalness, hostility in [
        ("p", "0", 7, False, False, .2, None),
        ("p", "1", 8, True, True, .8, .4),
        ("p", "2", 9, False, False, .4, .2),
        ("q", "0", 7, False, True, .9, .6),
    ]:
        uri = post_uri(owner, key)
        content_hash = m.digest(uri.encode())
        tables["posts"].append({"uri": uri, "author_did": account(owner), "event_at": f"2026-10-{day:02d}T12:00:00Z",
                                "is_reply": reply, "is_quote": quote, "content_sha256": content_hash, "is_deleted": False})
        tables["eligible_posts"].append({"uri": uri})
        values = {"politicalness": politicalness, "hostility": hostility}
        tables["post_scores"].append({"uri": uri, "content_sha256": content_hash, "score_generation": "numeric-scores",
                                      "scores": values, "unscored_reasons": {key: "fixture_unscored" for key, value in values.items()
                                                                           if value is None}})
    tables["eligible_posts"].append({"uri": post_uri("q", "unresolved")})
    for index, (actor, kind, owner, key) in enumerate([
        ("a", "like", "p", "0"), ("a", "repost", "p", "0"), ("b", "like", "p", "1"),
        ("c", "like", "p", "1"), ("p", "repost", "q", "0"),
        ("p", "repost", "q", "unresolved"), ("p", "repost", "q", "ineligible"),
    ]):
        tables["engagements"].append({
            "uri": f"at://{account(actor)}/app.bsky.feed.{kind}/{index}",
            "actor_did": account(actor), "subject_uri": post_uri(owner, key), "kind": kind,
            "event_at": "2026-10-10T12:00:00Z", "is_deleted": False,
        })
    tables["interests"] = [
        {"did": account(actor), "community": community, "score": value,
         "as_of": manifest["interests_cutoff_at"], "graph_generation": manifest["graph_generation"]}
        for actor, community, value in (("a", 0, .6), ("a", 1, .2), ("b", 1, 1.0))
    ]
    return manifest, tables


def write_inputs(directory, manifest, tables):
    manifest = copy.deepcopy(manifest)
    manifest["artifacts"] = {}
    for name in m.TABLES:
        raw = b"".join(m.encoded(row) for row in tables[name])
        path = directory / f"{name}.jsonl"
        path.write_bytes(raw)
        manifest["artifacts"][name] = {"path": path.name, "sha256": m.digest(raw), "rows": len(tables[name])}
    receipt = {
        "schema_version": "current-state-validation-v1", "status": "passed",
        "generation_id": manifest["generation_id"], "cutoff_at": manifest["cutoff_at"],
        "input_sha256": {name: manifest["artifacts"][name]["sha256"] for name in m.TABLES},
        "checks": {key: True for key in ("latest_versions_folded", "record_deletions_applied",
                                        "account_status_applied", "visibility_policy_applied")},
        "account_state_generation": "fixture-state", "visibility_policy_generation": "fixture-policy",
    }
    path = directory / "validation_receipt.json"
    path.write_bytes(m.encoded(receipt))
    manifest["artifacts"]["validation_receipt"] = {"path": path.name, "sha256": m.digest(path.read_bytes())}
    path = directory / "input_manifest.json"
    path.write_bytes(m.encoded(manifest))
    return path


class ModeInfluenceCandidateTests(unittest.TestCase):
    def test_distinct_audience_normalization_preserves_unknown_mass_and_separate_popularity(self):
        manifest, tables = fixture()
        result = m.build_candidate(manifest, tables, bootstrap_replicates=0)
        influence = result["audience_influence"][0]
        self.assertEqual(influence["distinct_engagers"], 3)
        self.assertEqual(influence["engagers_without_interest_rows"], 1)
        self.assertAlmostEqual(influence["interest_mass_coverage"], .6)
        self.assertAlmostEqual(influence["weighted_interest_mass_coverage"], .68)
        community = influence["communities"][0]
        self.assertAlmostEqual(community["raw_interest_mass"], .6)
        self.assertAlmostEqual(community["producer_normalized_observed_share"], 1 / 3)
        self.assertAlmostEqual(community["all_engager_mass_lower"], .2)
        self.assertAlmostEqual(community["all_engager_mass_upper"], .6)
        self.assertEqual(influence["exposure"]["status"], "unavailable")
        baseline = result["popularity_baseline"][0]
        self.assertAlmostEqual(baseline["value"], .5 * math.log1p(.5 * 5 + .1 * 10))
        self.assertIsNone(result["popularity_baseline"][1]["value"])
        self.assertFalse(result["qualification"]["production_publishable"])

    def test_mode_types_are_exclusive_and_reposts_use_subject_scores(self):
        manifest, tables = fixture()
        profile = m.build_candidate(manifest, tables, bootstrap_replicates=100, seed=7)["mode_profiles"][0]
        self.assertEqual(profile["events"], 6)
        self.assertEqual([profile["types"][kind]["events"] for kind in m.KINDS], [2, 1, 0, 3])
        repost = profile["types"]["repost"]["dimensions"]["politicalness"]
        self.assertEqual(repost["scored_events"], 1)
        self.assertEqual(repost["mean_model_score"], .9)
        self.assertEqual(repost["unscored_events"], 2)
        self.assertEqual(repost["missing_reasons"]["eligible_subject_not_resolved"], 1)
        self.assertEqual(repost["missing_reasons"]["subject_outside_eligible_post_universe"], 1)
        originals = profile["types"]["original"]["dimensions"]["politicalness"]
        self.assertAlmostEqual(originals["mean_model_score"], .3)
        self.assertEqual(originals["uncertainty"]["status"], "computed")
        self.assertEqual(originals["uncertainty"]["lower"], .2)
        self.assertEqual(originals["uncertainty"]["upper"], .4)
        self.assertEqual(profile["types"]["original"]["dimensions"]["hostility"]["unscored_events"], 1)
        self.assertIsNone(profile["assigned_follow_mode"])

    def test_deletions_suppress_subject_events_and_conflicting_eligibility_fails(self):
        manifest, tables = fixture()
        subject = tables["posts"][-1]["uri"]
        tables["deletions"] = [{"kind": "post", "key": subject}]
        with self.assertRaisesRegex(ValueError, "eligible post conflicts"):
            m.build_candidate(manifest, tables, bootstrap_replicates=0)
        tables["eligible_posts"] = [row for row in tables["eligible_posts"] if row["uri"] != subject]
        result = m.build_candidate(manifest, tables, bootstrap_replicates=0)
        self.assertEqual(result["mode_profiles"][0]["types"]["repost"]["events"], 2)
        self.assertEqual(result["audit"]["engagement_suppressed_by_deletion_ledger"], 1)
        self.assertEqual(result["audience_influence"][1]["distinct_engagers"], 0)

    def test_duplicate_future_and_unscored_contract_errors_fail(self):
        manifest, tables = fixture()
        bad = copy.deepcopy(tables)
        bad["engagements"].append(copy.deepcopy(bad["engagements"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate engagement"):
            m.build_candidate(manifest, bad, bootstrap_replicates=0)
        bad = copy.deepcopy(tables)
        bad["engagements"][0]["event_at"] = manifest["cutoff_at"]
        with self.assertRaisesRegex(ValueError, "cutoff"):
            m.build_candidate(manifest, bad, bootstrap_replicates=0)
        bad = copy.deepcopy(tables)
        bad["post_scores"][0]["scores"]["hostility"] = 0
        with self.assertRaisesRegex(ValueError, "null scores"):
            m.build_candidate(manifest, bad, bootstrap_replicates=0)
        bad = copy.deepcopy(tables)
        bad["post_scores"][0]["content_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "content identity"):
            m.build_candidate(manifest, bad, bootstrap_replicates=0)
        bad = copy.deepcopy(tables)
        bad["interests"][0]["score"] = 1
        with self.assertRaisesRegex(ValueError, "unit mass"):
            m.build_candidate(manifest, bad, bootstrap_replicates=0)

    def test_no_scores_or_audience_do_not_become_safe_or_zero_confidence(self):
        manifest, tables = fixture()
        tables["post_scores"] = []
        tables["engagements"] = []
        result = m.build_candidate(manifest, tables, bootstrap_replicates=0)
        original = result["mode_profiles"][0]["types"]["original"]["dimensions"]["politicalness"]
        self.assertIsNone(original["mean_model_score"])
        self.assertEqual(original["unscored_events"], 2)
        self.assertIsNone(result["audience_influence"][0]["interest_mass_coverage"])
        self.assertEqual(result["audience_influence"][0]["communities"], [])
        self.assertFalse(result["qualification"]["follow_gate_qualified"])

    def test_seals_reject_changed_bytes_and_receipt_mismatch(self):
        manifest, tables = fixture()
        directory = pathlib.Path(tempfile.mkdtemp(prefix="catbird-ri-rm-seal-"))
        path = write_inputs(directory, manifest, tables)
        (directory / "posts.jsonl").write_bytes((directory / "posts.jsonl").read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            m.load_inputs(path, 100000, 1024 * 1024)
        directory2 = pathlib.Path(tempfile.mkdtemp(prefix="catbird-ri-rm-receipt-"))
        path2 = write_inputs(directory2, manifest, tables)
        changed = json.loads(path2.read_text())
        changed["generation_id"] = "other-generation"
        path2.write_bytes(m.encoded(changed))
        with self.assertRaisesRegex(ValueError, "receipt generation"):
            m.load_inputs(path2, 100000, 1024 * 1024)

    def test_sensitive_scope_and_late_interests_are_rejected(self):
        manifest, tables = fixture()
        bad = copy.deepcopy(tables)
        bad["communities"][0]["sensitivity"] = "unreviewed"
        with self.assertRaisesRegex(ValueError, "non-sensitive"):
            m.build_candidate(manifest, bad, bootstrap_replicates=0)
        late = copy.deepcopy(manifest)
        late["interests_cutoff_at"] = "2026-10-19T00:00:00Z"
        directory = pathlib.Path(tempfile.mkdtemp(prefix="catbird-ri-rm-time-"))
        path = write_inputs(directory, late, tables)
        with self.assertRaisesRegex(ValueError, "predate"):
            m.load_inputs(path, 100000, 1024 * 1024)

    def test_actual_cli_emits_candidate_hashes_and_never_overwrites(self):
        manifest, tables = fixture()
        directory = pathlib.Path(tempfile.mkdtemp(prefix="catbird-ri-rm-cli-"))
        path = write_inputs(directory, manifest, tables)
        output = directory / "candidate"
        invocation = [sys.executable, str(pathlib.Path(m.__file__)), "--manifest", str(path),
                      "--output-dir", str(output), "--bootstrap-replicates", "100", "--seed", "7"]
        run = subprocess.run(invocation, capture_output=True, text=True,
                             env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        (directory / "stdout.txt").write_text(run.stdout)
        (directory / "stderr.txt").write_text(run.stderr)
        self.assertEqual(run.returncode, 0, run.stderr)
        seal = json.loads((output / "candidate_manifest.json").read_text())
        self.assertFalse(seal["qualification"]["feature_quality_measured"])
        for artifact in seal["artifacts"].values():
            self.assertEqual(m.digest((output / artifact["path"]).read_bytes()), artifact["sha256"])
        self.assertEqual(seal["source"]["sha256"], m.digest(pathlib.Path(m.__file__).read_bytes()))
        with self.assertRaisesRegex(ValueError, "output directory exists"):
            m.build_to_directory(path, output, bootstrap_replicates=100, seed=7)


if __name__ == "__main__":
    unittest.main()
