"""Numeric structural checks only; none are empirical graph-quality results."""
import json
import pathlib
import os
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from eval_phase1 import (
    PackRecallAccumulator,
    bulk_follow_diagnostic,
    eval_follow_edge_capture,
    eval_pack_comembership,
    primary_partition_null,
    sha256_file,
    unperformed_edge_reconstruction,
)


class EvaluationDiagnostics(unittest.TestCase):
    def test_pack_weightings_do_not_confuse_pair_micro_and_pack_macro(self):
        accumulator = PackRecallAccumulator()
        accumulator.add(150, 150, {"top1": 11175, "top3": 11175, "scene": 11175})
        accumulator.add(8, 8, {"top1": 0, "top3": 0, "scene": 0})
        report = accumulator.report({"expected_same_primary_probability": .25})
        scores = report["views"]["top1"]
        self.assertAlmostEqual(scores["micro_conditional_recall"], 11175 / 11203)
        self.assertEqual(scores["per_pack_macro_conditional_recall"], .5)
        self.assertAlmostEqual(scores["inverse_size_pair_weighted_conditional_recall"], 75 / 79)
        self.assertEqual(report["primary_only_null_comparison"]["micro_conditional_recall"]["expected"], .25)

    def test_coverage_and_zero_denominators_are_explicit(self):
        accumulator = PackRecallAccumulator()
        accumulator.add(4, 2, {"top1": 1, "top3": 1, "scene": 1})
        accumulator.add(3, 1, {"top1": 0, "top3": 0, "scene": 0})
        report = accumulator.report({"expected_same_primary_probability": .5})
        scores = report["views"]["top1"]
        self.assertEqual(scores["micro_conditional_recall"], 1)
        self.assertAlmostEqual(scores["micro_end_to_end_recall"], 1 / 9)
        self.assertAlmostEqual(scores["per_pack_macro_end_to_end_recall"], 1 / 12)
        self.assertEqual(report["packs_with_represented_pairs"], 1)
        self.assertAlmostEqual(report["primary_only_null_comparison"]["micro_end_to_end_recall"]["expected"], 1 / 18)
        empty = PackRecallAccumulator().report({"expected_same_primary_probability": None})
        self.assertIsNone(empty["views"]["top1"]["micro_conditional_recall"])
        self.assertIsNone(empty["primary_only_null_comparison"]["micro_end_to_end_recall"]["expected"])

    def test_primary_null_preserves_actual_sizes_and_ignores_secondary_memberships(self):
        memberships = {
            str(i): [(0 if i < 2 else 1, 0, 0, .7), (9, 0, 0, .3)]
            for i in range(5)
        }
        null = primary_partition_null(memberships)
        self.assertEqual(null["cluster_size_histogram"], {2: 1, 3: 1})
        self.assertEqual(null["expected_same_primary_probability"], .4)
        trimmed = primary_partition_null(memberships, {"0", "1"})
        self.assertEqual(trimmed["accounts"], 3)
        self.assertEqual(trimmed["expected_same_primary_probability"], 1)
        self.assertIsNone(primary_partition_null({})["expected_same_primary_probability"])
        singleton = primary_partition_null({"0": [(1, 0, 0, 1.0)]})
        self.assertIsNone(singleton["expected_same_primary_probability"])

    def test_dehub_uses_observed_pack_frequency_and_preserves_base_population(self):
        directory = pathlib.Path(tempfile.mkdtemp(prefix="catbird-eval-diagnostics-"))
        packs = [
            ["did:test:h", "did:test:a", "did:test:b"],
            ["did:test:h", "did:test:c", "did:test:d"],
            ["did:test:x", "did:test:y", "did:test:z"],
        ]
        rows = []
        for pack_index, members in enumerate(packs):
            list_uri = f"at://did:test:owner/app.bsky.graph.list/{pack_index}"
            common = {"did": "did:test:owner", "rev": "1", "operation": "create", "list_uri": list_uri}
            rows.append({**common, "collection": "app.bsky.graph.starterpack", "rkey": str(pack_index), "subject": ""})
            for member_index, did in enumerate(members):
                rows.append({**common, "collection": "app.bsky.graph.listitem",
                             "rkey": f"{pack_index}-{member_index}", "subject": did})
        path = directory / "events-1.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        memberships = {f"did:test:{name}": [(community, 0, 0, 1.0)]
                       for name, community in [("h", 0), ("a", 1), ("b", 1), ("c", 2), ("d", 3), ("x", 4)]}
        artifact = directory / "pack-metrics.parquet"
        report = eval_pack_comembership([path], memberships, 5, per_pack_path=artifact)
        self.assertEqual(report["starterpacks_evaluated"], 2)
        self.assertEqual(report["views"]["base"]["packs_in_base_population"], 3)
        self.assertEqual(report["views"]["base"]["raw_pair_occurrences"], 9)
        self.assertEqual(report["views"]["base"]["represented_pair_occurrences"], 6)
        self.assertEqual(report["dehub_policy"]["hubs"][0]["did"], "did:test:h")
        self.assertEqual(report["dehub_policy"]["hub_accounts"], 1)
        dehubbed = report["views"]["dehubbed"]
        self.assertEqual(dehubbed["packs_in_base_population"], 3)
        self.assertEqual(dehubbed["raw_pair_occurrences"], 5)
        self.assertEqual(dehubbed["represented_pair_occurrences"], 2)
        self.assertEqual(dehubbed["views"]["top1"]["micro_conditional_recall"], .5)
        self.assertEqual(dehubbed["views"]["top1"]["micro_end_to_end_recall"], .2)
        self.assertEqual(pq.read_table(artifact).num_rows, 6)
        self.assertEqual(report["per_pack_artifact"]["sha256"], sha256_file(artifact))
        self.assertEqual(report["bulk_follow_adoption"]["status"], "not_performed")

    def test_hub_removal_can_leave_no_pairs_without_inventing_zero_quality(self):
        accumulator = PackRecallAccumulator()
        accumulator.add(1, 1, {"top1": 0, "top3": 0, "scene": 0})
        report = accumulator.report({"expected_same_primary_probability": .25})
        self.assertEqual(report["packs_in_base_population"], 1)
        self.assertEqual(report["packs_with_raw_pairs"], 0)
        self.assertIsNone(report["views"]["top1"]["per_pack_macro_end_to_end_recall"])

    def test_follow_capture_validates_counts_and_does_not_claim_community_capture(self):
        directory = pathlib.Path(tempfile.mkdtemp(prefix="catbird-follow-capture-"))
        stats = {"follow_records": 12, "follow_edges": 10, "followers": 4,
                 "inactive_accounts": 0, "producers": 2, "producer_edges": 5}
        (directory / "stats.tsv").write_text("\t".join(stats) + "\n" + "\t".join(map(str, stats.values())) + "\n")
        (directory / "producers.tsv").write_text("0\tdid:test:0\t3\n1\tdid:test:1\t2\n")
        (directory / "edges.bin").write_bytes(bytes(40))
        (directory / "input_provenance.json").write_text(json.dumps({
            "fold_stats": stats,
            "artifacts": {name: {"sha256": sha256_file(directory / name), "bytes": (directory / name).stat().st_size}
                          for name in ("stats.tsv", "producers.tsv", "edges.bin")},
        }))
        report = eval_follow_edge_capture(directory)
        self.assertEqual(report["status"], "computed")
        self.assertEqual(report["edge_retention_fraction"], .5)
        self.assertIsNone(report["pure_producer_selection_capture"])
        self.assertEqual(report["community_bipartite_edge_capture"]["status"], "not_performed")
        (directory / "edges.bin").write_bytes(bytes(32))
        self.assertEqual(eval_follow_edge_capture(directory)["status"], "invalid_inputs")
        absent = pathlib.Path(tempfile.mkdtemp(prefix="catbird-follow-absent-"))
        self.assertEqual(eval_follow_edge_capture(absent)["status"], "unavailable")

    def test_unavailable_inputs_cannot_become_empirical_precision_or_adoption(self):
        adoption = bulk_follow_diagnostic({"did", "rev", "created_at", "operation"})
        self.assertEqual(adoption["status"], "not_performed")
        self.assertFalse(adoption["rev_only_proxy_used"])
        self.assertIn("commit_complete", adoption["missing_exact_contract_columns"])
        precision = unperformed_edge_reconstruction()
        self.assertEqual(precision["status"], "not_performed")
        self.assertNotIn("precision", {key: value for key, value in precision.items() if key != "definition"})

    def test_cli_seals_numeric_fixture_report_and_pack_artifact(self):
        directory = pathlib.Path(tempfile.mkdtemp(prefix="catbird-eval-cli-"))
        events = directory / "events"
        output = directory / "out"
        events.mkdir()
        output.mkdir()
        list_uri = "at://did:test:owner/app.bsky.graph.list/0"
        common = {"did": "did:test:owner", "rev": "1", "operation": "create",
                  "list_uri": list_uri, "name": "", "description": "", "extra": ""}
        rows = [{**common, "collection": "app.bsky.graph.starterpack", "rkey": "pack", "subject": ""}]
        rows.extend({**common, "collection": "app.bsky.graph.listitem", "rkey": str(i),
                     "subject": f"did:test:{i}"} for i in range(3))
        pq.write_table(pa.Table.from_pylist(rows), events / "events-1.parquet")
        pq.write_table(pa.Table.from_pylist([
            {"did": f"did:test:{i}", "community": 0, "scene_id": 0, "domain_id": 0, "score": 1.0, "rank": 0}
            for i in range(3)
        ]), output / "known_for.parquet")
        pq.write_table(pa.Table.from_pylist([
            {"community": 0, "scene_id": 0, "domain_id": 0, "size": 3,
             "top_dids": [f"did:test:{i}" for i in range(3)], "top_followers": [1, 1, 1]}
        ]), output / "communities.parquet")
        (output / "producers.tsv").write_text("".join(f"{i}\tdid:test:{i}\t1\n" for i in range(3)))
        (output / "cluster_stats.json").write_text(json.dumps({"test_fixture_only": True}))
        manifest = directory / "seeds.json"
        manifest.write_text(json.dumps({"subcultures": {}, "collision_traps": {}}))
        source = pathlib.Path(__file__).with_name("eval_phase1.py").resolve()
        invocation = subprocess.run(
            [sys.executable, str(source), "--events-dir", str(events), "--graph-dir", str(output),
             "--manifest", str(manifest)],
            capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        (directory / "stdout.txt").write_text(invocation.stdout)
        (directory / "stderr.txt").write_text(invocation.stderr)
        self.assertEqual(invocation.returncode, 0, invocation.stderr)
        report = json.loads((output / "phase1_eval_report.json").read_text())
        self.assertEqual(report["evaluator_source_observation"]["sha256"], sha256_file(source))
        self.assertEqual(report["seed_manifest_sha256"], sha256_file(manifest))
        self.assertEqual(report["pack_comembership"]["per_pack_artifact"]["rows"], 2)
        self.assertEqual(report["pack_comembership"]["per_pack_artifact"]["sha256"],
                         sha256_file(output / "pack_recall_diagnostics.parquet"))
        self.assertEqual(report["follow_edge_capture"]["status"], "unavailable")
        self.assertIsNone(report["netcup_ship"])


if __name__ == "__main__":
    unittest.main()
