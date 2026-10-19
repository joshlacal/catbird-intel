-- Catbird community intelligence: Phase 2 Non-neural For You v0 & Community Browse.
-- Builds on Phase 0 hot store (`catbird.*`) and Phase 1 `KnownFor` (`catbird.known_for`, `catbird.communities`).
--
-- Serving uses two top-k inverted tables (`catbird.item_communities` and
-- `catbird.community_items` + `catbird.community_directory`) so `getFeedSkeleton`
-- and `/xrpc/blue.catbird.intel.getCommunity` execute with zero ANN index and
-- sub-10ms compute latency on netcup.

CREATE TABLE IF NOT EXISTS catbird.known_for
(
    version   LowCardinality(String),
    pid       UInt32,
    did       String CODEC(ZSTD(3)),
    community UInt32,
    scene_id  UInt16,
    domain_id UInt8,
    score     Float32,
    rank      UInt8
)
ENGINE = ReplacingMergeTree
ORDER BY (version, did, rank);

CREATE TABLE IF NOT EXISTS catbird.communities
(
    version         LowCardinality(String),
    community       UInt32,
    scene_id        UInt16,
    domain_id       UInt8,
    size            UInt32,
    label           String,
    seed_subculture LowCardinality(String),
    keywords        Array(String),
    top_dids        Array(String),
    top_followers   Array(UInt64)
)
ENGINE = ReplacingMergeTree
ORDER BY (version, community);

-- User community affinity (`InterestedIn`): follows + likes/reposts joined against
-- KnownFor, requiring social_proof >= 2 distinct producers per community, top 50 per user.
CREATE TABLE IF NOT EXISTS catbird.interested_in
(
    did          String CODEC(ZSTD(3)),
    community    UInt32,
    scene_id     UInt16,
    score        Float32,
    social_proof UInt16,
    rank         UInt8,
    updated_at   DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (did, rank);

-- Producer community influence (`Influence`) + posting-mode mix + account role.
CREATE TABLE IF NOT EXISTS catbird.producer_influence
(
    did             String CODEC(ZSTD(3)),
    community       UInt32,
    influence       Float32,
    engagers        UInt32,
    followers_est   UInt32,
    role            LowCardinality(String),
    original_share  Float32,
    reply_share     Float32,
    repost_share    Float32,
    quote_share     Float32,
    political_share Float32,
    updated_at      DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (community, did);

-- Top-k inverted table 1: item -> communities (streaming post community vectors,
-- t=0 author KnownFor prior + 8h half-life time-decayed engager InterestedIn).
CREATE TABLE IF NOT EXISTS catbird.item_communities
(
    item_uri   String CODEC(ZSTD(3)),
    did        String CODEC(ZSTD(3)),
    rkey       String CODEC(ZSTD(3)),
    community  UInt32,
    score      Float32,
    created_at DateTime64(3, 'UTC') CODEC(DoubleDelta, ZSTD(3)),
    updated_at DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (item_uri, community);

-- Top-k inverted table 2: community -> ranked candidate posts using community-local
-- engagement residual (observed vs expected given author size, community, age, hour-of-week).
CREATE TABLE IF NOT EXISTS catbird.community_items
(
    community            UInt32,
    rank                 UInt16,
    item_uri             String CODEC(ZSTD(3)),
    did                  String CODEC(ZSTD(3)),
    rkey                 String CODEC(ZSTD(3)),
    created_at           DateTime64(3, 'UTC') CODEC(DoubleDelta, ZSTD(3)),
    engagers             UInt32,
    expected_engagements Float32,
    residual_score       Float32,
    community_score      Float32,
    politicalness        Float32,
    heat                 Float32,
    has_no_unauth_label  UInt8 DEFAULT 0,
    text_preview         String CODEC(ZSTD(3)),
    updated_at           DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (community, rank);

-- Community browse directory: top accounts by role, feeds, and starter packs
-- per community with hub filtering, adoption floor, and popularity ceiling.
CREATE TABLE IF NOT EXISTS catbird.community_directory
(
    community           UInt32,
    kind                Enum8('account' = 1, 'feed' = 2, 'pack' = 3),
    role                LowCardinality(String),
    rank                UInt16,
    uri_or_did          String CODEC(ZSTD(3)),
    handle_or_name      String CODEC(ZSTD(3)),
    description         String CODEC(ZSTD(3)),
    influence           Float32,
    adoption_count      UInt32,
    hub_concentration   Float32,
    original_share      Float32,
    political_share     Float32,
    has_no_unauth_label UInt8 DEFAULT 0,
    updated_at          DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY (community, kind, role, rank);

-- Telemetry log for served feeds (tracks latency + political/heat share of served items).
CREATE TABLE IF NOT EXISTS catbird.feed_serve_log
(
    ts              DateTime64(3, 'UTC') DEFAULT now64(3),
    requester_did   String CODEC(ZSTD(3)),
    persona         LowCardinality(String),
    items_served    UInt16,
    political_items UInt16,
    political_share Float32,
    mean_heat       Float32,
    compute_ms      Float32
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(ts)
ORDER BY ts
TTL toDateTime(ts) + INTERVAL 30 DAY;
