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

        cov_parquet = pq.read_table(out_dir / "coverage.parquet").to_pylist()
        cov_pq_by_did = {r["did"]: r for r in cov_parquet}
        self.assertTrue({"did", "host", "snapshot_source", "snapshot_rev", "coverage"}.issubset(cov_parquet[0].keys()))
        self.assertTrue({r["coverage"] for r in cov_parquet}.issubset({"oct", "stale_aug", "missing"}))
        self.assertEqual(cov_pq_by_did["did:plc:alice"]["coverage"], "oct")
        self.assertEqual(cov_pq_by_did["did:plc:alice"]["snapshot_rev"], "3loct00000001")
        self.assertEqual(cov_pq_by_did["did:plc:dave"]["coverage"], "stale_aug")
        self.assertEqual(cov_pq_by_did["did:plc:dave"]["snapshot_rev"], "3kaug00000004")
        self.assertEqual(cov_pq_by_did["did:plc:p_kept"]["coverage"], "missing")

    def test_archive_graph_extract_source_id_sync_and_netcup_export(self):
        import datetime
        import json
        from archive_graph_extract import (
            SCHEMA,
            SOURCE_IDS,
            convert_shards,
            infer_source_id,
            normalize_source_id,
            row_from_event,
            rows_from_netcup_export,
            should_keep_event,
            tid_from_uint64,
            uint64_from_tid,
        )

        self.assertEqual(
            SOURCE_IDS,
            ("public_archive", "gce_main", "gce_bootstrap_live", "netcup_live"),
        )
        self.assertEqual(SCHEMA.names[0], "source_id")
        for sid in SOURCE_IDS:
            self.assertEqual(normalize_source_id(sid), sid)
        self.assertEqual(infer_source_id("/tmp/archive-graph"), "public_archive")
        self.assertEqual(infer_source_id("/tmp/gce-archive-graph"), "gce_main")
        self.assertEqual(infer_source_id("/tmp/gce-bootstrap-live"), "gce_bootstrap_live")
        self.assertEqual(infer_source_id("/tmp/netcup-live"), "netcup_live")
        with self.assertRaises(ValueError):
            normalize_source_id("invalid_source")

        tid = "3mtidt27w5s27"
        tid_u64 = uint64_from_tid(tid)
        self.assertEqual(tid_u64, 1830090684623872005)
        self.assertEqual(tid_from_uint64(tid_u64), tid)

        sync_ev = {
            "$type": "network.bsky.jetstream.subscribeEvents#sync",
            "seq": 42,
            "time": "2026-10-19T20:06:16.716Z",
            "did": "did:plc:alice",
            "sync": {
                "did": "did:plc:alice",
                "rev": "3m4dmshmwwf2l",
                "time": "2026-10-19T20:06:16.716Z",
            },
        }
        self.assertTrue(should_keep_event(sync_ev))
        sync_row = row_from_event(sync_ev, source_id="gce_main")
        self.assertEqual(sync_row["source_id"], "gce_main")
        self.assertEqual(sync_row["kind"], "sync")
        self.assertEqual(sync_row["did"], "did:plc:alice")
        self.assertEqual(sync_row["rev"], "3m4dmshmwwf2l")

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="catbird-extract-guards-"))
        decoded = tmp / "decoded"
        out_dir = tmp / "events"
        decoded.mkdir(parents=True, exist_ok=True)
        shard_file = decoded / "shard-00.jsonl"
        shard_file.write_text(
            json.dumps(sync_ev)
            + "\n"
            + json.dumps(
                {
                    "$type": "network.bsky.jetstream.subscribeEvents#commit",
                    "seq": 43,
                    "time": "2026-10-19T20:06:16.716Z",
                    "did": "did:plc:alice",
                    "collection": "app.bsky.graph.follow",
                    "rkey": "3kfollow1",
                    "rev": "3m4dmshmwwf2l",
                    "operation": "create",
                    "record": {"subject": "did:plc:bob", "createdAt": "2026-01-01T00:00:00Z"},
                }
            )
            + "\n"
        )
        (decoded / "checkpoint.json").write_text(
            json.dumps({"shards": [{"path": "shard-00.jsonl", "sha256": "abc123"}]})
        )
        done: set[str] = set()
        count = convert_shards(decoded, out_dir, done, "000000-000000", source_id="gce_main")
        self.assertEqual(count, 2)
        written = pq.read_table(out_dir / "events-000000-000000-00-abc123.parquet").to_pylist()
        self.assertEqual([r["kind"] for r in written], ["sync", "commit"])
        self.assertEqual([r["source_id"] for r in written], ["gce_main", "gce_main"])
        self.assertEqual([r["rev"] for r in written], ["3m4dmshmwwf2l", "3m4dmshmwwf2l"])

        netcup_dir = tmp / "netcup-export"
        netcup_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "collection": "app.bsky.graph.follow",
                        "did": "did:plc:alice",
                        "rkey": "3kfollow1",
                        "rev": tid_u64,
                        "seq": 100,
                        "indexed_at": datetime.datetime(2026, 10, 20, 12, 0, 0, tzinfo=datetime.timezone.utc),
                    }
                ]
            ),
            netcup_dir / "deletions.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "did": "did:plc:bob",
                        "active": 0,
                        "status": "deleted",
                        "seq": 101,
                        "indexed_at": datetime.datetime(2026, 10, 20, 12, 5, 0, tzinfo=datetime.timezone.utc),
                    }
                ]
            ),
            netcup_dir / "account_status.parquet",
        )
        nc_rows = rows_from_netcup_export(netcup_dir)
        self.assertEqual(len(nc_rows), 2)
        self.assertEqual(nc_rows[0]["source_id"], "netcup_live")
        self.assertEqual(nc_rows[0]["rev"], tid)
        self.assertEqual(nc_rows[0]["operation"], "delete")
        self.assertEqual(nc_rows[1]["kind"], "account")
        self.assertEqual(nc_rows[1]["account_active"], False)
        self.assertEqual(nc_rows[1]["account_status"], "deleted")

    def test_fold_follows_w2_rev_ordering_netcup_stop_window_and_host_coverage(self):
        import datetime
        import shutil
        import subprocess
        from archive_graph_extract import SCHEMA, uint64_from_tid

        if shutil.which("clickhouse") is None:
            self.skipTest("clickhouse local not installed")

        def make_row(**kw):
            base = {
                "source_id": "public_archive",
                "seq": 1,
                "time": "2026-08-04T12:00:00Z",
                "kind": "commit",
                "did": "did:plc:u1",
                "collection": "app.bsky.graph.follow",
                "rkey": "f1",
                "rev": "3kaug00000001",
                "operation": "create",
                "created_at": "2026-01-01T00:00:00Z",
                "subject": "did:plc:p_target",
                "list_uri": "",
                "name": "",
                "description": "",
                "extra": "",
                "account_active": None,
                "account_status": "",
            }
            base.update(kw)
            return base

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="catbird-fold-w2-"))
        aug_dir = tmp / "archive-graph" / "events"
        oct_dir = tmp / "gce-archive-graph" / "events"
        out_dir = tmp / "out"
        ch_dir = tmp / "ch"
        for d in (aug_dir, oct_dir, out_dir, ch_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Public archive (Aug 2026 seed)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    # 1. Alice: Aug-only follow absent from Oct snapshot -> must be dropped
                    make_row(
                        source_id="public_archive",
                        seq=500,
                        did="did:plc:alice",
                        rkey="f_aug_zombie",
                        subject="did:plc:p_zombie",
                        rev="3kaug00000001",
                    ),
                    # 2. Stale fallback account (never downloads in Oct) -> keeps Aug snapshot + post-Oct-19 live events
                    make_row(
                        source_id="public_archive",
                        seq=600,
                        did="did:plc:stale_user",
                        rkey="f_stale_base",
                        subject="did:plc:p_target",
                        rev="3kaug00000099",
                    ),
                    # 3. Account deleted during GCE stop window on netcup -> records must vanish
                    make_row(
                        source_id="public_archive",
                        seq=700,
                        did="did:plc:netcup_deleted_acct",
                        rkey="f_del_acct",
                        subject="did:plc:p_zombie",
                        rev="3kaug00000050",
                    ),
                ],
                schema=SCHEMA,
            ),
            aug_dir / "events-0001.parquet",
        )

        # October 2026 sources (gce_main, gce_bootstrap_live, netcup_live)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    # Alice pre-snapshot live follow (rev < snapshot_rev 3loct22222232) -> must be dropped!
                    make_row(
                        source_id="gce_bootstrap_live",
                        seq=99999999,
                        time="2026-10-19T21:00:00Z",
                        did="did:plc:alice",
                        rkey="f_pre_snap_unfollowed",
                        subject="did:plc:p_zombie",
                        rev="3loct22222225",
                        created_at="2026-10-19T21:00:00Z",
                    ),
                    # Alice Oct CAR snapshot at rev=3loct22222232 (sync + follow create)
                    make_row(
                        source_id="gce_main",
                        seq=10,
                        time="2026-10-20T01:00:00Z",
                        kind="sync",
                        did="did:plc:alice",
                        collection="",
                        rkey="",
                        rev="3loct22222232",
                        operation="",
                        subject="",
                    ),
                    make_row(
                        source_id="gce_main",
                        seq=11,
                        time="2026-10-20T01:00:00Z",
                        did="did:plc:alice",
                        rkey="f_oct_snap",
                        subject="did:plc:p_target",
                        rev="3loct22222232",
                        created_at="2026-03-01T00:00:00Z",
                    ),
                    # Alice follow that will be deleted during the GCE stop window via netcup_deletions.parquet
                    make_row(
                        source_id="gce_main",
                        seq=12,
                        time="2026-10-20T01:00:00Z",
                        did="did:plc:alice",
                        rkey="f_deleted_on_netcup",
                        subject="did:plc:p_zombie",
                        rev="3loct22222232",
                        created_at="2026-04-01T00:00:00Z",
                    ),
                    # Alice post-snapshot live follow (rev > snapshot_rev) -> must be kept!
                    make_row(
                        source_id="netcup_live",
                        seq=2,
                        time="2026-10-21T14:00:00Z",
                        did="did:plc:alice",
                        rkey="f_post_snap_live",
                        subject="did:plc:p_live",
                        rev="3loct22222252",
                        created_at="2026-10-21T14:00:00Z",
                    ),
                    # Stale user: live follow after 2026-10-19 (no Oct snapshot) -> kept with coverage='stale_aug'
                    make_row(
                        source_id="netcup_live",
                        seq=3,
                        time="2026-10-20T18:00:00Z",
                        did="did:plc:stale_user",
                        rkey="f_stale_live",
                        subject="did:plc:p_live",
                        rev="3loct22222242",
                        created_at="2026-10-20T18:00:00Z",
                    ),
                    # Cross-source seq inversion on commit: gce_main has huge seq=900000000 & older rev=3loct22222232,
                    # netcup_live has tiny seq=1 & newer rev=3loct22222242 at identical timestamp -> delete must win!
                    make_row(
                        source_id="gce_main",
                        seq=900000000,
                        time="2026-10-21T12:00:00Z",
                        did="did:plc:seq_invert_commit",
                        rkey="f_keep",
                        subject="did:plc:p_target",
                        rev="3loct22222232",
                        created_at="2026-01-01T00:00:00Z",
                    ),
                    make_row(
                        source_id="gce_main",
                        seq=900000001,
                        time="2026-10-21T12:00:00Z",
                        did="did:plc:seq_invert_commit",
                        rkey="f_inverted_del",
                        subject="did:plc:p_zombie",
                        rev="3loct22222232",
                        created_at="2026-01-01T00:00:00Z",
                    ),
                    make_row(
                        source_id="netcup_live",
                        seq=1,
                        time="2026-10-21T12:00:00Z",
                        did="did:plc:seq_invert_commit",
                        rkey="f_inverted_del",
                        subject="",
                        rev="3loct22222242",
                        operation="delete",
                        created_at="",
                    ),
                    # Cross-source seq inversion on account status: gce_main has huge seq=999999999 & older rev=3loct22222232 (active),
                    # netcup_live has tiny seq=1 & newer rev=3loct22222252 (deactivated) at identical timestamp -> inactive must win!
                    make_row(
                        source_id="gce_main",
                        seq=999999998,
                        time="2026-10-21T12:00:00Z",
                        did="did:plc:seq_invert_acct",
                        rkey="f_acct_inv",
                        subject="did:plc:p_zombie",
                        rev="3loct22222232",
                        created_at="2026-01-01T00:00:00Z",
                    ),
                    make_row(
                        source_id="gce_main",
                        seq=999999999,
                        time="2026-10-21T12:00:00Z",
                        kind="account",
                        did="did:plc:seq_invert_acct",
                        collection="",
                        rkey="",
                        rev="3loct22222232",
                        operation="",
                        subject="",
                        account_active=True,
                        account_status="active",
                    ),
                    make_row(
                        source_id="netcup_live",
                        seq=1,
                        time="2026-10-21T12:00:00Z",
                        kind="account",
                        did="did:plc:seq_invert_acct",
                        collection="",
                        rkey="",
                        rev="3loct22222252",
                        operation="",
                        subject="",
                        account_active=False,
                        account_status="deactivated",
                    ),
                ],
                schema=SCHEMA,
            ),
            oct_dir / "events-0001.parquet",
        )

        # Netcup stop-window ledgers (Oct 20 -> Oct 21 10:18Z)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "collection": "app.bsky.graph.follow",
                        "did": "did:plc:alice",
                        "rkey": "f_deleted_on_netcup",
                        "rev": uint64_from_tid("3loct22222245"),
                        "seq": 55,
                        "indexed_at": datetime.datetime(2026, 10, 20, 15, 30, 0, tzinfo=datetime.timezone.utc),
                    }
                ],
                schema=pa.schema(
                    [
                        ("collection", pa.string()),
                        ("did", pa.string()),
                        ("rkey", pa.string()),
                        ("rev", pa.uint64()),
                        ("seq", pa.uint64()),
                        ("indexed_at", pa.timestamp("ms", tz="UTC")),
                    ]
                ),
            ),
            out_dir / "netcup_deletions_window.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "did": "did:plc:netcup_deleted_acct",
                        "active": 0,
                        "status": "deleted",
                        "seq": 77,
                        "indexed_at": datetime.datetime(2026, 10, 21, 9, 15, 0, tzinfo=datetime.timezone.utc),
                    }
                ],
                schema=pa.schema(
                    [
                        ("did", pa.string()),
                        ("active", pa.uint8()),
                        ("status", pa.string()),
                        ("seq", pa.uint64()),
                        ("indexed_at", pa.timestamp("ms", tz="UTC")),
                    ]
                ),
            ),
            out_dir / "netcup_account_status_window.parquet",
        )

        # PDS host directory including an account that never downloaded in either archive
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {"did": "did:plc:alice", "host": "pds-high.bsky.network"},
                    {"did": "did:plc:seq_invert_commit", "host": "pds-high.bsky.network"},
                    {"did": "did:plc:seq_invert_acct", "host": "pds-high.bsky.network"},
                    {"did": "did:plc:stale_user", "host": "pds-low.example.com"},
                    {"did": "did:plc:never_downloaded", "host": "pds-low.example.com"},
                ]
            ),
            out_dir / "did_hosts.parquet",
        )

        sql_path = pathlib.Path(__file__).with_name("fold_follows.sql")
        sql = (
            sql_path.read_text()
            .replace("__EVENTS__", str(tmp / "{archive-graph,gce-archive-graph}/events/events-*.parquet"))
            .replace("__OUT__", str(out_dir))
            .replace("__MIN_FOLLOWERS__", "1")
            .replace("__MAX_FOLLOWS__", "20000")
        )
        (tmp / "fold.sql").write_text(sql)

        proc = subprocess.run(
            [
                "clickhouse",
                "local",
                "--path",
                str(ch_dir),
                "--queries-file",
                str(tmp / "fold.sql"),
                "--",
                f"--user_files_path={tmp}",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

        # Verify p_zombie has 0 surviving followers (all zombie/deleted/inverted edges were dropped)
        producers = [line.split("\t") for line in (out_dir / "producers.tsv").read_text().strip().splitlines()]
        self.assertEqual(
            producers,
            [
                ["0", "did:plc:p_target", "3"],
                ["1", "did:plc:p_live", "2"],
            ],
        )

        cov_rows = pq.read_table(out_dir / "coverage.parquet").to_pylist()
        cov_by_did = {r["did"]: r for r in cov_rows}
        self.assertEqual(cov_by_did["did:plc:alice"]["coverage"], "oct")
        self.assertEqual(cov_by_did["did:plc:alice"]["snapshot_rev"], "3loct22222232")
        self.assertEqual(cov_by_did["did:plc:alice"]["host"], "pds-high.bsky.network")
        self.assertEqual(cov_by_did["did:plc:stale_user"]["coverage"], "stale_aug")
        self.assertEqual(cov_by_did["did:plc:stale_user"]["snapshot_rev"], "3kaug00000099")
        self.assertEqual(cov_by_did["did:plc:stale_user"]["host"], "pds-low.example.com")
        self.assertEqual(cov_by_did["did:plc:never_downloaded"]["coverage"], "missing")
        self.assertEqual(cov_by_did["did:plc:never_downloaded"]["snapshot_rev"], "")
        self.assertEqual(cov_by_did["did:plc:never_downloaded"]["inverse_coverage_weight"], 0.0)

        host_rows = {r["host"]: r for r in pq.read_table(out_dir / "host_coverage.parquet").to_pylist()}
        self.assertEqual(host_rows["pds-high.bsky.network"]["oct_coverage_rate"], 1.0)
        self.assertEqual(host_rows["pds-low.example.com"]["oct_coverage_rate"], 0.0)
        self.assertEqual(host_rows["pds-low.example.com"]["stale_aug_accounts"], 1)
        self.assertEqual(host_rows["pds-low.example.com"]["missing_accounts"], 1)
        self.assertGreater(
            host_rows["pds-low.example.com"]["inverse_coverage_weight"],
            host_rows["pds-high.bsky.network"]["inverse_coverage_weight"],
        )


if __name__ == "__main__":
    unittest.main()


