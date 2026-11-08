-- Opus 5.5 + S17 rev 6 community classification metadata for cand5-full-matched communities.
-- Used by catbird-feedgen (discover lens + cold-start fallback) and catbird-intel to prevent
-- political outrage, adult content, follow-trains, and furry/roleplay leak-through.

CREATE TABLE IF NOT EXISTS catbird.community_meta (
    community        UInt32,
    label            String,
    primary_language LowCardinality(String),
    community_type   LowCardinality(String),
    onboarding_fit   LowCardinality(String),
    macro_category   LowCardinality(String),
    scene_slug       LowCardinality(String),
    is_political     UInt8,
    is_adult         UInt8,
    is_furry         UInt8,
    is_junk          UInt8,
    is_s17_clean     UInt8
)
ENGINE = ReplacingMergeTree
ORDER BY community;
