import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { createApiHandler, flushTelemetry, pendingTelemetryForTests } from "../src/api.ts";
import { controlReport, healthEvidence, parseControls, passesControls } from "../src/serving_contract.ts";
import { FIXTURE_DIDS, installLocalServingFixture } from "./serving_fixtures.ts";

describe("honest serving contract over local HTTP", () => {
  let server: ReturnType<typeof Bun.serve>;
  beforeAll(() => { server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: createApiHandler() }); });
  beforeEach(installLocalServingFixture);
  afterAll(() => { server.stop(true); });
  const get = async (path: string, headers?: HeadersInit) => {
    const response = await fetch(new URL(path, server.url), { headers });
    return { status: response.status, body: await response.json() as any };
  };
  const post = async (path: string, body: unknown) => {
    const response = await fetch(new URL(`/xrpc/blue.catbird.onboarding.${path}`, server.url), {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
    });
    return { status: response.status, body: await response.json() as any };
  };

  test("missing token is logged out; forged boolean cannot expose restricted posts", async () => {
    const result = await get("/xrpc/app.bsky.feed.getFeedSkeleton?unauthenticated=false&debug=1");
    expect(result.status).toBe(200);
    expect(result.body.unauthenticated).toBe(true);
    expect(result.body.feed).toHaveLength(2);
    expect(result.body.feed.every((item: any) => Object.keys(item).join() === "post")).toBe(true);
    expect(JSON.stringify(result.body)).not.toContain("__FORBIDDEN_TEXT_FIELD__");
    expect(JSON.stringify(result.body)).not.toContain(FIXTURE_DIDS.restricted);
    expect(result.body.meta.control_policy.health_filtering).toBe("off");
    expect(result.body.meta.control_policy.unscored_share).toBe(0.5);
  });

  test("feed and preview default to the same off policy, including high and unknown scores", async () => {
    const feed = await get("/xrpc/app.bsky.feed.getFeedSkeleton");
    const preview = await post("previewTimeline", { allocation: { systems_rust: 6 }, unauthenticated: false });
    expect(preview.status).toBe(200);
    expect(preview.body.preview_posts.map((row: any) => row.post).sort()).toEqual(feed.body.feed.map((row: any) => row.post).sort());
    expect(preview.body.control_policy).toEqual(feed.body.meta.control_policy);
    expect(JSON.stringify(preview.body)).not.toMatch(/realized|predicted|text_preview|live_event_volume/);
  });

  test("same optional baseline cutoff removes scored high values and keeps unknown", async () => {
    const controls = { baseline_score_cutoff: { politicalness: 0.2, heat: 0.2 } };
    const feed = await get(`/xrpc/app.bsky.feed.getFeedSkeleton?baseline_score_cutoff=${encodeURIComponent(JSON.stringify(controls.baseline_score_cutoff))}`);
    const preview = await post("previewTimeline", { allocation: { systems_rust: 6 }, controls });
    expect(feed.body.feed).toHaveLength(1);
    expect(preview.body.preview_posts).toEqual(feed.body.feed);
    expect(preview.body.control_policy.unscored_share).toBe(1);
    expect(preview.body.control_policy.semantics).toBe("optional_uncalibrated_baseline_score_cutoff");
  });

  test("typed follow/hide-reposts and topic-feed actions remain distinct", async () => {
    const result = await post("solveBundle", { allocation: { systems_rust: 6 },
      preferences: { hideReposts: true }, delivery_modes: { [FIXTURE_DIDS.ordinary]: "topic_feed" } });
    expect(result.status).toBe(200);
    expect(result.body.follows).toHaveLength(1);
    expect(result.body.follows[0]).toEqual({ type: "follow_account", did: FIXTURE_DIDS.unknown,
      client_preferences: { scope: "catbird", hideReposts: true }, applied: false, executable: false });
    expect(result.body.topic_feeds).toEqual([{ type: "topic_feed_intent", did: FIXTURE_DIDS.ordinary, community: 1, follow: false, applied: false, executable: false }]);
    expect(result.body.role_summary.total_follows).toBe(1);
    expect(result.body.actions_applied).toBe(false);
    expect(JSON.stringify(result.body)).not.toMatch(/realized|predicted|live_event_volume|Follow originals only/);
    expect(JSON.stringify(pendingTelemetryForTests())).not.toMatch(/realized|predicted/);
  });

  test("classifier-free repost hiding does not promise to remove quotes or replies", async () => {
    const result = await post("solveBundle", { allocation: { systems_rust: 6 },
      delivery_modes: { [FIXTURE_DIDS.ordinary]: "hide_reposts" } });
    expect(result.body.follows.find((row: any) => row.did === FIXTURE_DIDS.ordinary).client_preferences.hideReposts).toBe(true);
    expect(result.body.follows.find((row: any) => row.did === FIXTURE_DIDS.unknown).client_preferences.hideReposts).toBe(false);
  });

  test("creator visibility applies to feed/pack URIs and metadata contains no top DIDs", async () => {
    for (const path of ["/community/1?unauthenticated=false", "/xrpc/blue.catbird.intel.listCommunities", "/xrpc/blue.catbird.onboarding.getSceneCards?unauthenticated=false"]) {
      const result = await get(path);
      expect(result.status).toBe(200);
      expect(JSON.stringify(result.body)).not.toContain(FIXTURE_DIDS.restricted);
      expect(JSON.stringify(result.body)).not.toContain(FIXTURE_DIDS.hidden);
      expect(JSON.stringify(result.body)).not.toMatch(/text_preview|top_dids|__FORBIDDEN_/);
    }
    const bundle = await post("solveBundle", { allocation: { systems_rust: 6 }, unauthenticated: false });
    expect(JSON.stringify(bundle.body)).not.toContain(FIXTURE_DIDS.restricted);
    expect(JSON.stringify(bundle.body)).not.toContain(FIXTURE_DIDS.hidden);
  });

  test("author diagnostics are aggregate only, with explicit unqualified readiness", async () => {
    const modes = await get("/xrpc/blue.catbird.intel.getAuthorModes");
    expect(modes.body.individual_profiles_available).toBe(false);
    expect(modes.body.producers).toBeUndefined();
    const health = await get("/health");
    expect(health.body.ready_for_public_promotion).toBe(false);
    expect(health.body.derived_generation).toBeNull();
    expect(health.body.source_community_version).toBe("fixture-generation-only");
  });

  test("invalid/unconfigured authorization fails closed instead of downgrading", async () => {
    const result = await get("/xrpc/blue.catbird.onboarding.getSceneCards", { authorization: "Bearer invalid" });
    expect(result.status).toBe(401);
    expect((await get(`/xrpc/app.bsky.feed.getFeedSkeleton?actor=${FIXTURE_DIDS.ordinary}`)).status).toBe(403);
  });

  test("legacy telemetry persistence is disabled and diagnostics preserve unknowns", async () => {
    await get("/xrpc/app.bsky.feed.getFeedSkeleton");
    const telemetry = pendingTelemetryForTests();
    expect(telemetry.feeds.at(-1)?.control_policy).toMatchObject({ unscored_share: 0.5, unknown_policy: "include" });
    expect(JSON.stringify(telemetry)).not.toMatch(/realized|predicted|political_share|mean_heat/);
    expect(await flushTelemetry()).toEqual({
      persisted: false, reason: "legacy_schema_incompatible", retention: "bounded_process_memory",
    });
  });

  test("obsolete controls, invalid limits and oversized allocation are rejected", async () => {
    expect((await get("/xrpc/app.bsky.feed.getFeedSkeleton?limit=NaN")).status).toBe(400);
    expect((await get("/xrpc/app.bsky.feed.getFeedSkeleton?max_heat=0.5")).status).toBe(400);
    expect((await post("previewTimeline", { controls: { politics: 0.15 } })).status).toBe(400);
    expect((await post("previewTimeline", { controls: { live_event_volume: 0.2 } })).status).toBe(400);
    expect((await post("solveBundle", { allocation: { systems_rust: 7 } })).status).toBe(400);
    expect((await post("solveBundle", { preferences: { hideReposts: "true" } })).status).toBe(400);
  });

  test("malformed JSON and mismatched swipe pairs are rejected", async () => {
    const malformed = await fetch(new URL("/xrpc/blue.catbird.onboarding.getCalibrationSwipes", server.url),
      { method: "POST", headers: { "content-type": "application/json" }, body: "{" });
    expect(malformed.status).toBe(400);
    expect((await post("solveBundle", { allocation: { systems_rust: 6 },
      swipes: [{ swipe_id: "trap_1_rust", choice: "a", community_a: 999, community_b: 1 }] })).status).toBe(400);
  });
});

describe("score uncertainty is preserved", () => {
  test("zero defaults, missing values and nonfinite scores remain unknown", () => {
    for (const value of [0, undefined, null, NaN, Infinity, -1]) {
      expect(healthEvidence({ politicalness: value, heat: value }).source).toBe("unscored");
      expect(passesControls({ politicalness: value, heat: value }, { baseline_score_cutoff: { heat: 0.1 } })).toBe(true);
    }
    expect(controlReport([], {}).unscored_share).toBeNull();
    expect(healthEvidence({ politicalness: 0.9, heat: 0.1 }).calibrated).toBe(false);
    expect(() => parseControls({ baseline_score_cutoff: { heat: NaN } })).toThrow();
  });
});
