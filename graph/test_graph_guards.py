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

    def test_fold_follows_regression_cases(self):
        """Run fold_follows.sql on fold_fixtures and compare every follower's edges."""
        import shutil
        import subprocess
        import fold_fixtures
        if shutil.which("clickhouse") is None:
            self.skipTest("clickhouse local not installed")
        fixture, expected, inactive = fold_fixtures.build()
        root = pathlib.Path(tempfile.mkdtemp(prefix="catbird-fold-"))
        fixture.write(root)
        out = root / "out"
        out.mkdir()
        sql = (pathlib.Path(__file__).parent / "fold_follows.sql").read_text()
        for key, value in {"__EVENTS__": str(root / "events" / "*.parquet"),
                           "__SNAPSHOTS__": str(root / "snapshots.jsonl"),
                           "__NETCUP__": str(root / "netcup"), "__OUT__": str(out),
                           "__MIN_FOLLOWERS__": "1", "__MAX_FOLLOWS__": "1000"}.items():
            sql = sql.replace(key, value)
        (root / "fold.sql").write_text(sql)
        run = subprocess.run(["clickhouse", "local", "--path", str(root / "ch"), "--queries-file", str(root / "fold.sql"),
                              "--", f"--user_files_path={root}"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr[-3000:])

        producers = {}
        for line in (out / "producers.tsv").read_text().splitlines():
            pid, did, _ = line.split("\t")
            producers[int(pid)] = did
        edges = np.fromfile(out / "edges.bin", dtype=[("uid", "<u4"), ("pid", "<u4")])
        self.assertTrue((np.diff(edges["uid"].astype(np.int64)) >= 0).all(), "simgraph needs edges sorted by uid")
        weights = [line.split("\t") for line in (out / "follower_weights.tsv").read_text().splitlines()]
        uid_did = {int(w[0]): w[1] for w in weights}
        got = {}
        for uid, pid in edges.tolist():
            got.setdefault(uid_did[uid], set()).add(producers[pid])
        for did, follows in expected.items():
            with self.subTest(did=did):
                self.assertEqual(got.get(did, set()), follows)
        self.assertEqual(set(got) - set(expected), set())

        accounts = {r["did"]: r for r in _tsv_rows(out / "account_coverage.tsv")}
        self.assertEqual({d for d, r in accounts.items() if r["inactive"] == "1"}, inactive)
        self.assertEqual(accounts["did:plc:a"]["snapshot_source"], "collector_snapshot")
        self.assertEqual(accounts["did:plc:l"]["coverage"], "snapshot")
        self.assertEqual(accounts["did:plc:b"]["coverage"], "events_only")
        self.assertEqual(accounts["did:plc:h"]["coverage"], "events_only")
        self.assertEqual(accounts["did:plc:n"]["snapshot_source"], "sync_reset")
        self.assertEqual(accounts["did:plc:lowmiss0"]["coverage"], "missing")

        hosts = {r["host"]: r for r in _tsv_rows(out / "host_coverage.tsv")}
        self.assertAlmostEqual(float(hosts[fold_fixtures.LOW_HOST]["snapshot_coverage_rate"]), 0.1)
        self.assertEqual(float(hosts[fold_fixtures.LOW_HOST]["inverse_coverage_weight"]), 3.0)
        self.assertTrue(all(float(w[3]) <= 3.0 for w in weights))
        stats = _tsv_rows(out / "coverage_stats.tsv")[0]
        self.assertGreater(int(stats["superseded_follow_rows"]), 0)

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
            ("public_archive", "gce_main", "gce_bootstrap_live", "netcup_live", "repo_supplement"),
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

    def test_structural_hub_demotion_promotes_topical_and_retains_hub_secondary(self):
        from cluster import detect_and_demote_hubs

        # Community 0: structural mega-hub (size >= 50, median > 50k, mean > 100k)
        # Community 1: normal topical community (size >= 50)
        # Community 2: small community (< 50 members) with high followers -> must NOT be demoted
        comm = np.array([0] * 50 + [1] * 50 + [2] * 10, dtype=np.int64)
        followers = np.array([120_000] * 50 + [800] * 50 + [150_000] * 10, dtype=np.uint64)
        # Producer 0 has both hub 0 and non-hub 1 in KnownFor; Producer 1 only has hub 0 in KnownFor
        raw = np.array(
            [(0, 50, 0.25), (1, 100, 0.20), (1, 101, 0.15), (50, 51, 0.50)],
            dtype=[("p", "<u4"), ("q", "<u4"), ("w", "<f4")],
        )
        kp = np.array([0, 0, 1, 50], dtype=np.int64)
        kc = np.array([0, 1, 0, 1], dtype=np.int64)
        share = np.array([0.70, 0.20, 0.80, 0.90], dtype=np.float64)

        out_kp, out_kc, out_sh, demoted = detect_and_demote_hubs(
            n=110, n_comm=3, comm=comm, followers=followers, raw=raw, kp=kp, kc=kc, share=share
        )
        self.assertEqual(demoted, [0])
        p0_mask = out_kp == 0
        self.assertEqual(out_kc[p0_mask].tolist(), [1, 0])
        self.assertAlmostEqual(out_sh[p0_mask].tolist()[0], 0.20)
        self.assertAlmostEqual(out_sh[p0_mask].tolist()[1], 0.05)

        p1_mask = out_kp == 1
        self.assertEqual(out_kc[p1_mask].tolist(), [2, 0])
        self.assertAlmostEqual(out_sh[p1_mask].tolist()[0], 0.09)
        self.assertAlmostEqual(out_sh[p1_mask].tolist()[1], 0.05)

    def test_structural_entropy_density_hub_demotion_and_megahub_exclusion(self):
        from cluster import assign_excluded_hubs, detect_and_demote_hubs

        # Community 0: low-density hub (50 members, med=60k, mean=60k, low internal density)
        # Community 1: large low-density community (50 members, med=5k, low internal density)
        # Community 2: cohesive topical community (10 members, dense internal edges)
        comm = np.array([0] * 50 + [1] * 50 + [2] * 10, dtype=np.int64)
        followers = np.array([60_000] * 50 + [5_000] * 50 + [1_000] * 10, dtype=np.uint64)
        # Dense internal edges in community 2 (nodes 100..109) so int_dens[2] is high
        dense_c2 = [(100 + i, 100 + j, 0.40) for i in range(10) for j in range(10) if i != j]
        # Producer 0 (in hub 0) has equal raw edge weight to large low-density c1 and cohesive c2
        raw = np.array(
            [(0, 50, 0.25), (0, 100, 0.25)] + dense_c2,
            dtype=[("p", "<u4"), ("q", "<u4"), ("w", "<f4")],
        )
        kp = np.array([0, 50, 100], dtype=np.int64)
        kc = np.array([0, 1, 2], dtype=np.int64)
        share = np.array([0.85, 0.90, 0.90], dtype=np.float64)

        out_kp, out_kc, out_sh, demoted = detect_and_demote_hubs(
            n=110,
            n_comm=3,
            comm=comm,
            followers=followers,
            raw=raw,
            kp=kp,
            kc=kc,
            share=share,
            hub_mode="structural",
        )
        self.assertEqual(demoted, [0])
        p0_mask = out_kp == 0
        # Cohesive community 2 must beat large low-density community 1 ahead of demoted hub 0
        self.assertEqual(out_kc[p0_mask].tolist(), [2, 0])
        self.assertAlmostEqual(float(out_sh[p0_mask][0]), 0.09)
        self.assertAlmostEqual(float(out_sh[p0_mask][1]), 0.05)

        # Pre-Leiden excluded mega-hub (node 0 with comm=-1) is assigned to cohesive community 2
        comm_ex = comm.copy()
        comm_ex[0] = -1
        assigned = assign_excluded_hubs(
            n=110,
            n_comm=3,
            comm=comm_ex,
            followers=followers,
            raw=raw,
            excluded_hub_pids=np.array([0], dtype=np.int64),
        )
        self.assertIn(0, assigned)
        self.assertEqual(assigned[0][0][0], 2)
        self.assertGreaterEqual(assigned[0][0][1], 0.50)

    def test_compact_known_for_preserves_schema_and_keeps_hide_from_recs_in_top_dids(self):
        import json
        from cluster import write_compact_known_for
        from eval_phase1 import enrich_and_label_communities, load_known_for

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="catbird-kf-compact-"))
        kf_in = pa.table(
            {
                "pid": pa.array([0, 0, 1, 1], type=pa.uint32()),
                "did": pa.array(
                    ["did:plc:hidden1", "did:plc:hidden1", "did:plc:visible2", "did:plc:visible2"],
                    type=pa.string(),
                ),
                "community": pa.array([10, 20, 10, 20], type=pa.uint32()),
                "scene_id": pa.array([300, 400, 300, 400], type=pa.uint16()),
                "domain_id": pa.array([5, 6, 5, 6], type=pa.uint8()),
                "score": pa.array([0.85, 0.05, 0.90, 0.05], type=pa.float32()),
                "rank": pa.array([0, 1, 0, 1], type=pa.uint8()),
                "hide_from_recs": pa.array([True, True, False, False], type=pa.bool_()),
            }
        )
        write_compact_known_for(kf_in, tmp / "known_for.parquet")
        comm_in = pa.table(
            {
                "community": pa.array([10, 20], type=pa.uint32()),
                "scene_id": pa.array([300, 400], type=pa.uint16()),
                "domain_id": pa.array([5, 6], type=pa.uint8()),
                "size": pa.array([2, 2], type=pa.uint32()),
                "top_dids": pa.array(
                    [
                        ["did:plc:hidden1", "did:plc:visible2"],
                        ["did:plc:hidden1", "did:plc:visible2"],
                    ],
                    type=pa.list_(pa.string()),
                ),
                "top_followers": pa.array([[5000, 1000], [5000, 1000]], type=pa.list_(pa.uint64())),
            }
        )
        pq.write_table(comm_in, tmp / "communities.parquet")

        by_did, kf_read, comm_read = load_known_for(tmp)
        self.assertEqual(kf_read.num_rows, 4)
        self.assertEqual(kf_read.schema.field("scene_id").type, pa.uint16())
        self.assertEqual(kf_read.schema.field("domain_id").type, pa.uint8())
        self.assertEqual(kf_read.schema.field("hide_from_recs").type, pa.bool_())

        events_path = tmp / "events-1.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "collection": "app.bsky.actor.profile",
                        "operation": "create",
                        "did": "did:plc:hidden1",
                        "rev": "1",
                        "name": "Hidden User",
                        "description": "Astronomy galaxy telescope",
                        "extra": "hidden.bsky.social",
                    },
                    {
                        "collection": "app.bsky.actor.profile",
                        "operation": "create",
                        "did": "did:plc:visible2",
                        "rev": "1",
                        "name": "Visible User",
                        "description": "Astronomy galaxy telescope",
                        "extra": "visible.bsky.social",
                    },
                ]
            ),
            events_path,
        )
        enriched = enrich_and_label_communities([events_path], tmp, by_did, comm_read, kf=kf_read)
        # hide-from-recs is about posts, not profiles: flagged producers stay in top_dids.
        self.assertEqual(enriched["top_dids"].to_pylist(), [["did:plc:hidden1", "did:plc:visible2"]] * 2)
        self.assertEqual(enriched["top_followers"].to_pylist(), [[5000, 1000], [5000, 1000]])
        # Demoted hub community 20 (which only appears at rank > 0 in KnownFor) must still get TF-IDF keywords
        self.assertTrue(len(enriched["keywords"].to_pylist()[1]) > 0)
        prompt_objs = [json.loads(line) for line in (tmp / "colab_community_labeling_prompts.jsonl").read_text().splitlines()]
        self.assertEqual([p["did"] for p in prompt_objs[0]["top_producers"]], ["did:plc:hidden1", "did:plc:visible2"])
        self.assertEqual([p["did"] for p in prompt_objs[1]["top_producers"]], ["did:plc:hidden1", "did:plc:visible2"])

    def test_colab_community_labeling_gate_checkpoint_and_receipt(self):
        import json
        from colab_label_communities import (
            benchmark_one_shard,
            prepare_shards,
            run_all_shards,
            verify_and_write_receipt,
        )

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="catbird-colab-label-"))
        prompts_path = tmp / "prompts.jsonl"
        hide_path = tmp / "hide_from_recs.tsv"
        hide_path.write_text("did:plc:hidden\n")
        prompts_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "community": i,
                        "scene_id": 1,
                        "domain_id": 0,
                        "size": 100 - i,
                        "heuristic_label": f"Term{i}",
                        "top_terms": [f"term{i}", "shared"],
                        "enriched_tlds": [".dev"],
                        "top_producers": [
                            {"did": "did:plc:hidden", "handle": "hidden.dev", "name": "H", "bio": "hidden"},
                            {"did": f"did:plc:vis{i}", "handle": f"vis{i}.dev", "name": f"V{i}", "bio": "visible"},
                        ],
                    }
                )
                for i in range(5)
            )
            + "\n"
        )
        out_dir = tmp / "colab-out"
        manifest = prepare_shards(prompts_path, out_dir, shard_size=2, hide_from_recs_path=hide_path)
        self.assertEqual(manifest["total_communities"], 5)
        self.assertEqual(manifest["total_shards"], 3)
        self.assertEqual(manifest["hide_from_recs_producers_removed"], 5)

        # Live execution without --approve-spend must be blocked by the §4a gate
        with self.assertRaises(SystemExit):
            benchmark_one_shard(out_dir, approve_spend=False, dry_run_local=False)

        bench = benchmark_one_shard(out_dir, dry_run_local=True)
        self.assertTrue(bench["fits_budget"])
        self.assertFalse(bench["qualifies_for_live_production_labels"])

        ckpt = run_all_shards(out_dir, dry_run_local=True)
        self.assertEqual(len(ckpt["completed_shards"]), 3)

        receipt = verify_and_write_receipt(out_dir)
        self.assertEqual(receipt["communities_labeled"], 5)
        self.assertEqual(receipt["status"], "prepared_dry_run_only")

        # Re-preparing with updated prompt content must invalidate stale shard checkpoints
        old_s0_sha = ckpt["completed_shards"]["shard-000"]["input_sha256"]
        prompts_path.write_text(prompts_path.read_text().replace("term0", "updated_term0"))
        manifest2 = prepare_shards(prompts_path, out_dir, shard_size=2, hide_from_recs_path=hide_path)
        self.assertNotEqual(manifest2["shards"][0]["sha256"], old_s0_sha)
        benchmark_one_shard(out_dir, dry_run_local=True)
        ckpt2 = run_all_shards(out_dir, dry_run_local=True)
        self.assertEqual(ckpt2["completed_shards"]["shard-000"]["input_sha256"], manifest2["shards"][0]["sha256"])


def _tsv_rows(path):
    header, *rows = path.read_text().splitlines()
    names = header.split("\t")
    return [dict(zip(names, row.split("\t"))) for row in rows]


if __name__ == "__main__":
    unittest.main()
