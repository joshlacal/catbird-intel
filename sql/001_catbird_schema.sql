-- Catbird community intelligence: Phase 0 hot store on netcup.
-- Database `catbird` is separate from the prototype `feed` database.
--
-- Record tables use ReplacingMergeTree(rev, is_deleted) keyed by (did, rkey).
--   * `rev` is the repo revision TID of the commit, decoded to UInt64. It is
--     monotonic per repository, so a re-delivered duplicate carries the same
--     version and an older create can never override a newer delete.
--   * Delete commits carry only did/collection/rkey/rev; they are written as a
--     row with is_deleted = 1 under the same key.
--   * Partitions use `rkey_date` (decoded from the TID record key), so a delete
--     lands in the same partition as its create and merges can drop both.
-- Read paths must use FINAL (or the v_* views) and exclude is_deleted = 1.

CREATE DATABASE IF NOT EXISTS catbird;

-- ---------------------------------------------------------------- hot (14 d)

CREATE TABLE IF NOT EXISTS catbird.posts
(
    did              String,
    rkey             String,
    rev              UInt64,
    is_deleted       UInt8,
    rkey_date        Date,
    seq              UInt64,
    indexed_at       DateTime64(3, 'UTC'),
    created_at       DateTime64(3, 'UTC'),
    cid              String,
    text             String CODEC(ZSTD(3)),
    langs            Array(LowCardinality(String)),
    reply_root       String,
    reply_parent     String,
    quote_uri        String,
    embed_type       LowCardinality(String),
    image_count      UInt8,
    has_alt_text     UInt8,
    has_video        UInt8,
    external_uri     String,
    facet_links      Array(String),
    facet_tags       Array(String),
    facet_mentions   Array(String),
    self_labels      Array(LowCardinality(String))
)
ENGINE = ReplacingMergeTree(rev, is_deleted)
PARTITION BY rkey_date
ORDER BY (did, rkey)
TTL rkey_date + INTERVAL 14 DAY
SETTINGS index_granularity = 8192;

-- Likes and reposts share one table; `kind` distinguishes them. Subject URIs
-- are split so the projection can aggregate by subject cheaply.
CREATE TABLE IF NOT EXISTS catbird.engagements
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
PARTITION BY (rkey_date, kind)
ORDER BY (did, kind, rkey)
TTL rkey_date + INTERVAL 14 DAY
SETTINGS index_granularity = 8192, deduplicate_merge_projection_mode = 'rebuild';

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
