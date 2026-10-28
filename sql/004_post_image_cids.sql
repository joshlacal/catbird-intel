-- Keep each post's image blob CIDs (and mime types) so feed and embedding jobs
-- can build cdn.bsky.app feed_thumbnail URLs from (did, cid) without
-- re-hydrating posts. Metadata-only ALTER; existing rows read as empty arrays.

ALTER TABLE catbird.posts
    ADD COLUMN IF NOT EXISTS image_cids  Array(String) CODEC(ZSTD(3)) AFTER has_alt_text,
    ADD COLUMN IF NOT EXISTS image_mimes Array(LowCardinality(String)) AFTER image_cids;
