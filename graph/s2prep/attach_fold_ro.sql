-- Attach the fold DB tables (graph-build-v3/ch, built by fold_follows.sql on 2026-10-24) from an
-- APFS clone of their store directories. __CLONE__ is replaced by chq.sh with $S2_OUT/chclone/.
-- Definitions are copied from graph-build-v3/ch/store/a22/a22d324d-.../<table>.sql.
-- Merges are stopped globally before any ATTACH and per table after it, so no part is rewritten.
SYSTEM STOP MERGES;
CREATE DATABASE IF NOT EXISTS g ENGINE = Atomic;

ATTACH TABLE g.follow_events UUID '5d9614eb-f5d2-4c4c-bcd7-39ef07fc219a'
(did String, rkey String, rev UInt64, is_delete UInt8, subject String)
ENGINE = MergeTree ORDER BY (did, rkey, rev)
SETTINGS index_granularity = 8192, disk = disk(name = 'g_clone', type = 'local', path = '__CLONE__');
SYSTEM STOP MERGES g.follow_events;

ATTACH TABLE g.follows UUID '205eb761-5236-4660-947d-c39b5ed13f98'
(follower String, producer String)
ENGINE = ReplacingMergeTree ORDER BY (follower, producer)
SETTINGS index_granularity = 8192, disk = disk(name = 'g_clone', type = 'local', path = '__CLONE__');
SYSTEM STOP MERGES g.follows;

ATTACH TABLE g.producers UUID 'ce7ca214-7f65-455f-abe7-b0b2ebbc2b3d'
(did String, followers UInt64, pid UInt32)
ENGINE = MergeTree ORDER BY did
SETTINGS index_granularity = 8192, disk = disk(name = 'g_clone', type = 'local', path = '__CLONE__');
SYSTEM STOP MERGES g.producers;

ATTACH TABLE g.floors UUID '44c52c46-3bf5-4804-bfe8-af7b4e6ff94f'
(did String, floor UInt64, snap_rev UInt64, sync_rev UInt64)
ENGINE = MergeTree ORDER BY did
SETTINGS index_granularity = 8192, disk = disk(name = 'g_clone', type = 'local', path = '__CLONE__');
SYSTEM STOP MERGES g.floors;

ATTACH TABLE g.inactive UUID '99dcd9ce-49fa-4059-96cc-2734161305d7'
(did String)
ENGINE = MergeTree ORDER BY did
SETTINGS index_granularity = 8192, disk = disk(name = 'g_clone', type = 'local', path = '__CLONE__');
SYSTEM STOP MERGES g.inactive;

ATTACH TABLE g.followers_ok UUID '204b3613-6b20-4db0-91ca-82388c6aef48'
(follower String)
ENGINE = MergeTree ORDER BY follower
SETTINGS index_granularity = 8192, disk = disk(name = 'g_clone', type = 'local', path = '__CLONE__');
SYSTEM STOP MERGES g.followers_ok;

ATTACH TABLE g.snapshots UUID '72fd9fa8-74c5-4024-843a-84b9d505c9ac'
(did String, status LowCardinality(String), snap_rev UInt64, completed_at DateTime64(6, 'UTC'), host String, active UInt8)
ENGINE = MergeTree ORDER BY did
SETTINGS index_granularity = 8192, disk = disk(name = 'g_clone', type = 'local', path = '__CLONE__');
SYSTEM STOP MERGES g.snapshots;

ATTACH TABLE g.coverage UUID 'b7d6a4cf-1d72-4341-b837-b9e599390372'
(did String, host String, snapshot_source String, snapshot_rev UInt64, coverage LowCardinality(String),
 host_snapshot_coverage_rate Float64, host_effective_coverage_rate Float64, inverse_coverage_weight Float64)
ENGINE = MergeTree ORDER BY did
SETTINGS index_granularity = 8192, disk = disk(name = 'g_clone', type = 'local', path = '__CLONE__');
SYSTEM STOP MERGES g.coverage;

-- rkey TID -> 64-bit integer (fold.sql's tid_u64); 0 when the rkey is not TID-shaped.
CREATE FUNCTION IF NOT EXISTS tid_u64 AS (s) -> if(
    match(s, '^[234567abcdefghij][234567abcdefghijklmnopqrstuvwxyz]{12}$'),
    toUInt64(arraySum(i -> bitShiftLeft(toUInt64(position('234567abcdefghijklmnopqrstuvwxyz', substring(s, i, 1)) - 1), 5 * (13 - i)),
                      range(1, 14))),
    toUInt64(0));
