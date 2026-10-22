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

    def test_fold_follows_snapshot_replacement_barriers_and_coverage(self):
        import shutil
        import subprocess
        if shutil.which("clickhouse") is None:
            self.skipTest("clickhouse local not installed")

        schema = pa.schema([
            ("seq", pa.uint64()), ("time", pa.string()), ("kind", pa.string()), ("did", pa.string()),
            ("collection", pa.string()), ("rkey", pa.string()), ("rev", pa.string()), ("operation", pa.string()),
            ("created_at", pa.string()), ("subject", pa.string()), ("list_uri", pa.string()),
            ("name", pa.string()), ("description", pa.string()), ("extra", pa.string()),
            ("account_active", pa.bool_()), ("account_status", pa.string()),
        ])

        def make_row(**kw):
            base = {
                "seq": 1, "time": "2026-08-04T12:00:00Z", "kind": "commit", "did": "did:plc:u1",
                "collection": "app.bsky.graph.follow", "rkey": "f1", "rev": "3kaug00000001",
                "operation": "create", "created_at": "2026-01-01T00:00:00Z", "subject": "did:plc:p1",
                "list_uri": "", "name": "", "description": "", "extra": "",
                "account_active": None, "account_status": "",
            }
            base.update(kw)
            return base

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="catbird-fold-guards-"))
        aug_dir = tmp / "archive-graph" / "events"
        oct_dir = tmp / "gce-archive-graph" / "events"
        out_dir = tmp / "out"
        ch_dir = tmp / "ch"
        for d in (aug_dir, oct_dir, out_dir, ch_dir):
            d.mkdir(parents=True, exist_ok=True)

        pq.write_table(pa.Table.from_pylist([
            make_row(seq=100, did="did:plc:alice", rkey="f_zombie", subject="did:plc:p_zombie", rev="3kaug00000001"),
            make_row(seq=101, did="did:plc:alice", rkey="f_kept", subject="did:plc:p_kept", rev="3kaug00000002"),
            make_row(seq=200, did="did:plc:dave", rkey="f_dave_unfollow", subject="did:plc:p_zombie", rev="3kaug00000003"),
            make_row(seq=201, did="did:plc:dave", rkey="f_dave_kept", subject="did:plc:p_kept", rev="3kaug00000004"),
            make_row(seq=300, did="did:plc:frank", rkey="f_frank_old", subject="did:plc:p_zombie", rev="3zzzzzzzzzzzz"),
            make_row(seq=400, did="did:plc:grace", rkey="f_grace_old", subject="did:plc:p_zombie", rev="3kaug00000005"),
            make_row(seq=999999998, did="did:plc:heidi", rkey="f_heidi", subject="did:plc:p_zombie", rev="3kaug00000006"),
            make_row(seq=999999999, kind="account", did="did:plc:heidi", collection="", rkey="", rev="",
                     operation="", subject="", account_active=True, account_status="active"),
        ], schema=schema), aug_dir / "events-0001.parquet")

        pq.write_table(pa.Table.from_pylist([
            make_row(seq=10, time="2026-10-21T10:00:00Z", did="did:plc:alice", rkey="f_kept",
                     subject="did:plc:p_kept", rev="3loct00000001", created_at="2026-01-01T00:00:00Z"),
            make_row(seq=20, time="2026-10-21T15:00:01Z", did="did:plc:dave", rkey="f_dave_unfollow",
                     subject="", rev="3loct00000002", operation="delete", created_at=""),
            make_row(seq=21, time="2026-10-21T15:00:02Z", did="did:plc:dave", rkey="f_dave_new",
                     subject="did:plc:p_live", rev="3loct00000003", operation="create",
                     created_at="2026-10-21T15:00:02Z"),
            make_row(seq=30, time="2026-10-21T11:00:00Z", kind="sync", did="did:plc:frank",
                     collection="", rkey="", rev="", operation="", subject=""),
            make_row(seq=31, time="2026-10-21T11:00:01Z", did="did:plc:frank", rkey="f_frank_new",
                     subject="did:plc:p_kept", rev="3aaaaaaaaaaaa", created_at="2026-02-01T00:00:00Z"),
            make_row(seq=40, time="2026-10-21T12:00:00Z", kind="account", did="did:plc:grace",
                     collection="", rkey="", rev="", operation="", subject="",
                     account_active=False, account_status="deleted"),
            make_row(seq=41, time="2026-10-21T12:05:00Z", kind="account", did="did:plc:grace",
                     collection="", rkey="", rev="", operation="", subject="",
                     account_active=True, account_status="active"),
            make_row(seq=42, time="2026-10-21T12:06:00Z", did="did:plc:grace", rkey="f_grace_new",
                     subject="did:plc:p_kept", rev="3loct00000004", created_at="2026-10-21T12:06:00Z"),
            make_row(seq=5, time="2026-10-21T08:00:00Z", kind="account", did="did:plc:heidi",
                     collection="", rkey="", rev="", operation="", subject="",
                     account_active=False, account_status="suspended"),
        ], schema=schema), oct_dir / "events-0001.parquet")

        sql_path = pathlib.Path(__file__).with_name("fold_follows.sql")
        sql = (sql_path.read_text()
               .replace("__EVENTS__", str(tmp / "{archive-graph,gce-archive-graph}/events/events-*.parquet"))
               .replace("__OUT__", str(out_dir))
               .replace("__MIN_FOLLOWERS__", "1")
               .replace("__MAX_FOLLOWS__", "20000"))
        (tmp / "fold.sql").write_text(sql)

        proc = subprocess.run(
            ["clickhouse", "local", "--path", str(ch_dir), "--queries-file", str(tmp / "fold.sql"),
             "--", f"--user_files_path={tmp}"],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

        producers = [line.split("\t") for line in (out_dir / "producers.tsv").read_text().strip().splitlines()]
        self.assertEqual(producers, [["0", "did:plc:p_kept", "4"], ["1", "did:plc:p_live", "1"]])

        stats_lines = (out_dir / "stats.tsv").read_text().strip().splitlines()
        stats = dict(zip(stats_lines[0].split("\t"), map(int, stats_lines[1].split("\t"))))
        self.assertEqual(stats, {
            "follow_records": 12,
            "follow_edges": 5,
            "followers": 4,
            "inactive_accounts": 1,
            "producers": 2,
            "producer_edges": 5,
        })

        cov_lines = (out_dir / "account_coverage.tsv").read_text().strip().splitlines()
        cov_header = cov_lines[0].split("\t")
        cov_by_did = {row.split("\t")[0]: dict(zip(cov_header, row.split("\t"))) for row in cov_lines[1:]}
        self.assertEqual(cov_by_did["did:plc:alice"]["snapshot_source"], "october_gce_snapshot")
        self.assertEqual(cov_by_did["did:plc:alice"]["is_stale"], "0")
        self.assertEqual(cov_by_did["did:plc:dave"]["snapshot_source"], "august_seed_fallback")
        self.assertEqual(cov_by_did["did:plc:dave"]["is_stale"], "1")
        self.assertEqual(cov_by_did["did:plc:frank"]["has_sync_reset"], "1")
        self.assertEqual(cov_by_did["did:plc:grace"]["is_stale"], "0")

        cstats_lines = (out_dir / "coverage_stats.tsv").read_text().strip().splitlines()
        cstats = dict(zip(cstats_lines[0].split("\t"), map(int, cstats_lines[1].split("\t"))))
        self.assertEqual(cstats["zombie_follows_suppressed"], 3)
        self.assertEqual(cstats["stale_august_fallback_accounts"], 2)


if __name__ == "__main__":
    unittest.main()

