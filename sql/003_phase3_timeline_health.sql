-- Catbird community intelligence: Phase 3 Timeline Health & Phase 4 Onboarding Sessions.
-- Adds real-time ingest health scores to `catbird.posts`, creates the
-- per-author mode-mix & "Follow originals only" eligibility table, and creates
-- `catbird.onboarding_sessions` for the Phase 4/6 offline simulator.

ALTER TABLE catbird.posts
    ADD COLUMN IF NOT EXISTS politicalness   Float32 DEFAULT 0.0 CODEC(ZSTD(3)),
    ADD COLUMN IF NOT EXISTS heat            Float32 DEFAULT 0.0 CODEC(ZSTD(3)),
    ADD COLUMN IF NOT EXISTS engagement_bait Float32 DEFAULT 0.0 CODEC(ZSTD(3)),
    ADD COLUMN IF NOT EXISTS spam            Float32 DEFAULT 0.0 CODEC(ZSTD(3));

CREATE TABLE IF NOT EXISTS catbird.author_mode_mix
(
    did                            String CODEC(ZSTD(3)),
    handle                         String CODEC(ZSTD(3)),
    community                      UInt32,
    followers_est                  UInt32,
    total_events_14d               UInt32,
    orig_posts                     UInt32,
    reply_posts                    UInt32,
    quote_posts                    UInt32,
    reposts                        UInt32,
    orig_political_share           Float32,
    reply_political_share          Float32,
    quote_political_share          Float32,
    repost_political_share         Float32,
    overall_political_share        Float32,
    orig_heat_mean                 Float32,
    overall_heat_mean              Float32,
    follow_originals_only_eligible UInt8,
    recommend_mode                 LowCardinality(String),
    role                           LowCardinality(String),
    active_days_14d                UInt8,
    has_no_unauth_label            UInt8 DEFAULT 0,
    hide_from_algorithmic          UInt8 DEFAULT 0,
    updated_at                     DateTime DEFAULT now()
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY did;

CREATE TABLE IF NOT EXISTS catbird.onboarding_sessions
(
    session_id          String CODEC(ZSTD(3)),
    created_at          DateTime64(3, 'UTC') DEFAULT now64(3),
    persona             LowCardinality(String),
    unauthenticated     UInt8,
    allocation_json     String CODEC(ZSTD(3)),
    swipes_json         String CODEC(ZSTD(3)),
    controls_json       String CODEC(ZSTD(3)),
    posterior_json      String CODEC(ZSTD(3)),
    bundle_json         String CODEC(ZSTD(3)),
    predicted_mix_json  String CODEC(ZSTD(3)),
    realized_mix_json   String CODEC(ZSTD(3)),
    solve_ms            Float32
)
ENGINE = MergeTree
ORDER BY (created_at, session_id);
