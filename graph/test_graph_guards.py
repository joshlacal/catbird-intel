"""Structural regression tests; these are not empirical community-quality metrics."""
import unittest
import pathlib
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from cluster import community_median_size, median_size, prune_for_backbone
from eval_phase1 import eval_pack_comembership, eval_subcultures_and_traps


class GraphGuards(unittest.TestCase):
    def test_size_objective_distinguishes_community_and_producer_medians(self):
        memberships = np.repeat(np.array([0, 1, 2]), [10, 10, 1000])
        self.assertEqual(community_median_size(memberships, 8), 10)
        self.assertEqual(median_size(memberships, 8), 1000)

    def test_top_neighbors_are_sorted_within_producer(self):
        edges = np.array(
            [(0, 1, .1), (0, 2, .8), (0, 3, .2)],
            dtype=[("p", "<u4"), ("q", "<u4"), ("w", "<f4")],
        )
        self.assertEqual(prune_for_backbone(edges, top_k=1)["q"].tolist(), [2])

    def test_below_threshold_edges_are_not_reintroduced(self):
        edges = np.array([(0, 1, .001)], dtype=[("p", "<u4"), ("q", "<u4"), ("w", "<f4")])
        self.assertEqual(len(prune_for_backbone(edges)), 0)

    def test_shared_primary_is_reported_as_merge(self):
        memberships = {
            "account-1": [(7, 1, 1, 1.0), (8, 1, 1, .1)],
            "account-2": [(7, 1, 1, 1.0), (9, 1, 1, .1)],
        }
        result = eval_subcultures_and_traps(
            {"subcultures": {"cohort-1": ["account-1"], "cohort-2": ["account-2"],
                             "uncovered": ["account-3"]}, "collision_traps": {}},
            memberships,
        )
        self.assertEqual(result["distinct_primary_communities_across_26"], 1)
        self.assertEqual(result["merged_subcultures"], [
            {"community": 7, "subcultures": ["cohort-1", "cohort-2"]},
        ])
        self.assertEqual(result["pairwise_jsd_evaluable_pairs"], 1)

    def test_uncovered_pole_cannot_pass(self):
        result = eval_subcultures_and_traps(
            {"subcultures": {}, "collision_traps": {"trap": {"a": ["absent"], "b": ["absent-too"]}}},
            {},
        )
        self.assertEqual(result["collision_traps"]["trap"]["status"], "fail")
        self.assertEqual(result["collision_traps"]["trap"]["evaluated_accounts"], 0)

    def test_pack_metrics_use_actual_packs_and_fold_deletes(self):
        # Structural record fixtures contain no fabricated post/evaluation text.
        directory = pathlib.Path(tempfile.mkdtemp(prefix="catbird-pack-guards-"))
        list_uri = "at://did:test:owner/app.bsky.graph.list/list"
        rows = [
            {"collection": "app.bsky.graph.listitem", "did": "did:test:owner", "rkey": str(i),
             "rev": "1", "operation": "create", "list_uri": list_uri, "subject": f"did:test:{i}"}
            for i in range(4)
        ]
        rows.extend([
            {"collection": "app.bsky.graph.starterpack", "did": "did:test:owner", "rkey": "pack",
             "rev": "1", "operation": "create", "list_uri": list_uri, "subject": ""},
            {"collection": "app.bsky.graph.listitem", "did": "did:test:owner", "rkey": "3",
             "rev": "2", "operation": "delete", "list_uri": "", "subject": ""},
        ])
        first = directory / "events-1.parquet"
        pq.write_table(pa.Table.from_pylist(rows), first)
        memberships = {f"did:test:{i}": [(0, 0, 0, 1.0)] for i in range(4)}
        result = eval_pack_comembership([first], memberships, 1)
        self.assertEqual(result["co_member_pairs"], 3)
        self.assertEqual(result["known_for_top1_pair_hits"], 3)
        self.assertIsNone(result["beats_a_blue_start_503_partition"])
        overlapping = {f"did:test:{i}": [(i, 0, 0, .7), ((i + 1) % 3, 0, 0, .3)] for i in range(3)}
        result = eval_pack_comembership([first], overlapping, 3)
        self.assertEqual(result["known_for_top1_pair_hits"], 0)
        self.assertEqual(result["known_for_top3_pair_hits"], 3)
        self.assertEqual(result["known_for_scene_pair_hits"], 3)
        second = directory / "events-2.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"collection": "app.bsky.graph.list", "did": "did:test:owner", "rkey": "list",
             "rev": "3", "operation": "delete", "list_uri": "", "subject": ""},
        ]), second)
        result = eval_pack_comembership([first, second], memberships, 1)
        self.assertEqual(result["starterpacks_evaluated"], 0)
        self.assertIsNone(result["known_for_top1_pair_recall"])


if __name__ == "__main__":
    unittest.main()
