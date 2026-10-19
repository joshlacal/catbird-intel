-- Catbird community intelligence: Phase 0 hot store on netcup.
-- Database `catbird` is separate from the prototype `feed` database.
--
-- Record tables use ReplacingMergeTree(rev, is_deleted) keyed by (did, rkey).
--   * `rev` is the repo revision TID of the commit, decoded to UInt64. It is
--     monotonic per repository, so a re-delivered duplicate carries the same
--     version and an older create can never override a newer delete.
--   * Delete commits carry only did/collection/rkey/rev; they are written as a
--     row with is_deleted = 1 under the same key.
--   * Partitions use the month of `rkey_date` (decoded from the TID record key),
--     so a delete lands in its create's partition and merges can drop both.
--     Monthly, not daily: deletes of old records otherwise spread one insert
--     over >100 partitions (see ops/migrations/002_monthly_hot_partitions.sql).
-- Read paths must use FINAL (or the v_* views) and exclude is_deleted = 1.

CREATE DATABASE IF NOT EXISTS catbird;

-- ---------------------------------------------------------------- hot (14 d)

CREATE TABLE IF NOT EXISTS catbird.posts
(
    did              String CODEC(ZSTD(3)),
    rkey             String CODEC(ZSTD(3)),
    rev              UInt64 CODEC(Delta, ZSTD(3)),
    is_deleted       UInt8,
    rkey_date        Date,
    seq              UInt64 CODEC(Delta, ZSTD(3)),
    indexed_at       DateTime64(3, 'UTC') CODEC(DoubleDelta, ZSTD(3)),
    created_at       DateTime64(3, 'UTC') CODEC(DoubleDelta, ZSTD(3)),
    cid              String CODEC(ZSTD(3)),
    text             String CODEC(ZSTD(3)),
    langs            Array(LowCardinality(String)),
    reply_root       String CODEC(ZSTD(3)),
    reply_parent     String CODEC(ZSTD(3)),
    quote_uri        String CODEC(ZSTD(3)),
    embed_type       LowCardinality(String),
    image_count      UInt8,
    has_alt_text     UInt8,
    has_video        UInt8,
    external_uri     String CODEC(ZSTD(3)),
    facet_links      Array(String) CODEC(ZSTD(3)),
    facet_tags       Array(String) CODEC(ZSTD(3)),
    facet_mentions   Array(String) CODEC(ZSTD(3)),
    self_labels      Array(LowCardinality(String))
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
PARTITION BY toYYYYMM(rkey_date)
ORDER BY (did, rkey)
TTL rkey_date + INTERVAL 14 DAY
SETTINGS index_granularity = 8192;

-- Likes and reposts share one table; `kind` distinguishes them. Subject URIs
-- are split into did/collection/rkey. Per-subject aggregation is done by
-- Phase 2 materialized views, not a projection (a full-copy projection
-- doubled this table's disk use).
CREATE TABLE IF NOT EXISTS catbird.engagements
(
    kind               Enum8('like' = 1, 'repost' = 2),
    did                String CODEC(ZSTD(3)),
    rkey               String CODEC(ZSTD(3)),
    rev                UInt64 CODEC(Delta, ZSTD(3)),
    is_deleted         UInt8,
    rkey_date          Date,
    seq                UInt64 CODEC(Delta, ZSTD(3)),
    indexed_at         DateTime64(3, 'UTC') CODEC(DoubleDelta, ZSTD(3)),
    subject_did        String CODEC(ZSTD(3)),
    subject_collection LowCardinality(String),
    subject_rkey       String CODEC(ZSTD(3)),
    via_uri            String CODEC(ZSTD(3))
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
PARTITION BY toYYYYMM(rkey_date)
ORDER BY (did, kind, rkey)
TTL rkey_date + INTERVAL 14 DAY
SETTINGS index_granularity = 8192;

-- --------------------------------------------- current state (no TTL)

CREATE TABLE IF NOT EXISTS catbird.follows
(
    did        String,
    rkey       String,
    rev        UInt64,
    is_deleted UInt8,
    rkey_date  Date,
    seq        UInt64,
    indexed_at DateTime64(3, 'UTC'),
    subject    String
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
PARTITION BY toYYYYMM(rkey_date)
ORDER BY (did, rkey);

CREATE TABLE IF NOT EXISTS catbird.blocks
(
    did        String,
    rkey       String,
    rev        UInt64,
    is_deleted UInt8,
    rkey_date  Date,
    seq        UInt64,
    indexed_at DateTime64(3, 'UTC'),
    subject    String
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
PARTITION BY toYYYYMM(rkey_date)
ORDER BY (did, rkey);

CREATE TABLE IF NOT EXISTS catbird.lists
(
    did         String,
    rkey        String,
    rev         UInt64,
    is_deleted  UInt8,
    seq         UInt64,
    indexed_at  DateTime64(3, 'UTC'),
    purpose     LowCardinality(String),
    name        String,
    description String
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
ORDER BY (did, rkey);

CREATE TABLE IF NOT EXISTS catbird.listitems
(
    did        String,
    rkey       String,
    rev        UInt64,
    is_deleted UInt8,
    rkey_date  Date,
    seq        UInt64,
    indexed_at DateTime64(3, 'UTC'),
    list_uri   String,
    subject    String
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
PARTITION BY toYYYYMM(rkey_date)
ORDER BY (did, rkey);

CREATE TABLE IF NOT EXISTS catbird.starterpacks
(
    did         String,
    rkey        String,
    rev         UInt64,
    is_deleted  UInt8,
    seq         UInt64,
    indexed_at  DateTime64(3, 'UTC'),
    name        String,
    description String,
    list_uri    String,
    feed_uris   Array(String)
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
ORDER BY (did, rkey);

CREATE TABLE IF NOT EXISTS catbird.feedgens
(
    did          String,
    rkey         String,
    rev          UInt64,
    is_deleted   UInt8,
    seq          UInt64,
    indexed_at   DateTime64(3, 'UTC'),
    service_did  String,
    display_name String,
    description  String
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
ORDER BY (did, rkey);

CREATE TABLE IF NOT EXISTS catbird.profiles
(
    did          String,
    rev          UInt64,
    is_deleted   UInt8,
    seq          UInt64,
    indexed_at   DateTime64(3, 'UTC'),
    display_name String,
    description  String,
    self_labels  Array(LowCardinality(String)),
    pinned_post  String
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
ORDER BY did;

-- app.bsky.actor.contentVisibilityDeclaration. A missing record means false.
CREATE TABLE IF NOT EXISTS catbird.visibility
(
    did                     String,
    rev                     UInt64,
    is_deleted              UInt8,
    seq                     UInt64,
    indexed_at              DateTime64(3, 'UTC'),
    hide_from_algorithmic   UInt8
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
ORDER BY did;

CREATE TABLE IF NOT EXISTS catbird.account_status
(
    did        String,
    seq        UInt64,
    indexed_at DateTime64(3, 'UTC'),
    active     UInt8,
    status     LowCardinality(String)
)
ENGINE = ReplacingMergeTree(seq)
ORDER BY did;

CREATE TABLE IF NOT EXISTS catbird.identities
(
    did        String,
    seq        UInt64,
    indexed_at DateTime64(3, 'UTC'),
    handle     String
)
ENGINE = ReplacingMergeTree(seq)
ORDER BY did;

-- Collections we do not model yet (threadgate, postgate, verification, ...).
CREATE TABLE IF NOT EXISTS catbird.other_records
(
    collection LowCardinality(String),
    did        String,
    rkey       String,
    rev        UInt64,
    is_deleted UInt8,
    seq        UInt64,
    indexed_at DateTime64(3, 'UTC'),
    record     String CODEC(ZSTD(3))
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
PARTITION BY toYYYYMMDD(indexed_at)
ORDER BY (collection, did, rkey)
TTL toDate(indexed_at) + INTERVAL 14 DAY;

-- ------------------------------------------------ ledgers and health

-- Every delete commit, for derived-data invalidation and the cold-lake ledger.
CREATE TABLE IF NOT EXISTS catbird.deletions
(
    collection LowCardinality(String),
    did        String,
    rkey       String,
    rev        UInt64,
    seq        UInt64,
    indexed_at DateTime64(3, 'UTC')
)
ENGINE = MergeTree
PARTITION BY toYYYYMMDD(indexed_at)
ORDER BY (collection, did, rkey)
TTL toDate(indexed_at) + INTERVAL 30 DAY;

CREATE TABLE IF NOT EXISTS catbird.ingest_health
(
    ts              DateTime64(3, 'UTC'),
    service         LowCardinality(String),
    last_seq        UInt64,
    last_event_time DateTime64(3, 'UTC'),
    lag_ms          Int64,
    events          UInt32,
    rows_written    UInt32,
    reconnects      UInt32,
    flush_ms        UInt32,
    rss_mb          UInt32
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY ts
TTL toDate(ts) + INTERVAL 180 DAY;

-- -------------------------------------------------------- read views

CREATE VIEW IF NOT EXISTS catbird.v_inactive_accounts AS
SELECT did FROM catbird.account_status FINAL WHERE active = 0;

CREATE VIEW IF NOT EXISTS catbird.v_hidden_from_algorithms AS
SELECT did FROM catbird.visibility FINAL WHERE is_deleted = 0 AND hide_from_algorithmic = 1;

CREATE VIEW IF NOT EXISTS catbird.v_posts_live AS
SELECT *
FROM catbird.posts FINAL
WHERE is_deleted = 0
  AND did NOT IN (SELECT did FROM catbird.v_inactive_accounts);
