-- One-off migration (2026-10-19): daily rkey_date partitions on the hot tables
-- broke inserts. Deleting an old like or post writes a tombstone into that
-- record's original day, so one batch touched >100 partitions and every
-- insert failed (max_partitions_per_insert_block). Monthly partitions bound
-- this to ~46 (2023-01 .. now). Existing rows are copied and the tables
-- swapped atomically; the ingest service must be stopped while this runs.

CREATE TABLE catbird.posts_m AS catbird.posts
ENGINE = ReplacingMergeTree(rev, is_deleted)
PARTITION BY toYYYYMM(rkey_date)
ORDER BY (did, rkey)
TTL rkey_date + INTERVAL 14 DAY;

INSERT INTO catbird.posts_m SELECT * FROM catbird.posts;
EXCHANGE TABLES catbird.posts AND catbird.posts_m;
DROP TABLE catbird.posts_m;

CREATE TABLE catbird.engagements_m
(
    kind               Enum8('like' = 1, 'repost' = 2),
    did                String,
    rkey               String,
    rev                UInt64,
    is_deleted         UInt8,
    rkey_date          Date,
    seq                UInt64,
    indexed_at         DateTime64(3, 'UTC'),
    subject_did        String,
    subject_collection LowCardinality(String),
    subject_rkey       String,
    via_uri            String,
    PROJECTION by_subject
    (
        SELECT * ORDER BY (subject_did, subject_rkey, indexed_at)
    )
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
PARTITION BY toYYYYMM(rkey_date)
ORDER BY (did, kind, rkey)
TTL rkey_date + INTERVAL 14 DAY
SETTINGS index_granularity = 8192, deduplicate_merge_projection_mode = 'rebuild';

INSERT INTO catbird.engagements_m SELECT * FROM catbird.engagements;
EXCHANGE TABLES catbird.engagements AND catbird.engagements_m;
DROP TABLE catbird.engagements_m;
