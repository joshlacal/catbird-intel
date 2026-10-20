import { installServingSnapshot, type DirectoryEntry, type PostCandidate } from "../src/api.ts";

export const FIXTURE_DIDS = {
  ordinary: "did:plc:aaaaaaaaaaaaaaaaaaaaaaaa",
  unknown: "did:plc:bbbbbbbbbbbbbbbbbbbbbbbb",
  restricted: "did:plc:cccccccccccccccccccccccc",
  hidden: "did:plc:dddddddddddddddddddddddd",
};

export function installLocalServingFixture() {
  const post = (did: string, rkey: string, score: number, rank: number): PostCandidate => ({
    did, rkey, community: 1, rank, item_uri: `at://${did}/app.bsky.feed.post/${rkey}`,
    created_at: "2026-10-20 00:00:00", engagers: 10, expected_engagements: 5,
    residual_score: 1, community_score: 1, politicalness: score, heat: score,
    has_no_unauth_label: did === FIXTURE_DIDS.restricted ? 1 : 0,
    text_preview: "__FORBIDDEN_TEXT_FIELD__",
  });
  const entry = (did: string, kind: DirectoryEntry["kind"] = "account"): DirectoryEntry => ({
    community: 1, kind, role: "creator", rank: 1,
    uri_or_did: kind === "account" ? did : `at://${did}/${kind === "feed" ? "app.bsky.feed.generator" : "app.bsky.graph.starterpack"}/fixture`,
    handle_or_name: "local-fixture.invalid", description: "__FORBIDDEN_DESCRIPTION_FIELD__",
    influence: 10, adoption_count: 25, hub_concentration: 1, original_share: 0.5, political_share: 0.9,
    // Feed/pack ownership must still be checked even when this legacy flag is zero.
    has_no_unauth_label: kind === "account" && did === FIXTURE_DIDS.restricted ? 1 : 0,
  });
  installServingSnapshot({
    version: "fixture-generation-only",
    communities: [{ community: 1, scene_id: 1, domain_id: 1, size: 4, label: "Local fixture",
      seed_subculture: "systems_rust", keywords: ["fixture"], top_dids: Object.values(FIXTURE_DIDS), top_followers: [1, 2, 3, 4] }],
    items: [
      post(FIXTURE_DIDS.ordinary, "high", 0.9, 1),
      post(FIXTURE_DIDS.unknown, "unknown", 0, 2),
      post(FIXTURE_DIDS.restricted, "restricted", 0.1, 3),
      post(FIXTURE_DIDS.hidden, "hidden", 0.1, 4),
    ],
    directory: Object.values(FIXTURE_DIDS).flatMap((did) => [entry(did), entry(did, "feed"), entry(did, "pack")]),
    hiddenDids: [FIXTURE_DIDS.hidden],
    noUnauthDids: [FIXTURE_DIDS.restricted],
  });
}
