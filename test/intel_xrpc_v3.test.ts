import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { P256Keypair, formatMultikey } from "@atproto/crypto";
import { createApiHandler } from "../src/api.ts";
import { createDidKeyResolver } from "../src/did_resolver.ts";
import { S17CatalogEngine, type IntelServingBundleV2 } from "../src/intel_catalog.ts";
import { createServiceAuthenticator } from "../src/service_auth.ts";

function makeCentVecB64(hotDim: number, secondaryDim = -1): string {
  const arr = new Int8Array(256);
  arr[hotDim] = 100;
  if (secondaryDim >= 0) arr[secondaryDim] = 65;
  return Buffer.from(arr.buffer).toString("base64");
}

const VIEWER_DID = "did:plc:synthviewer0000000000001";
const HIDDEN_DID = "did:plc:synthhidden0000000000001";
const NO_UNAUTH_DID = "did:plc:synthnounauth00000000001";
const SEED_FOLLOW_DID = "did:plc:synthseedfollow000000001";
const AUDIENCE = "did:web:intel.catbird.blue#catbird_intel";

function buildSyntheticBundle(): IntelServingBundleV2 {
  const mkMembers = (prefix: string, cid: number) => [
    {
      did: `did:plc:${prefix}anchor0000000001`,
      role: "anchor_creator" as const,
      recommendedMode: "follow_all" as const,
      whyPicked: `Core anchor voice in community ${cid}`,
      rankScoreBps: 9100,
      repostShareBps: 800,
      mediaShareBps: 1500,
      estimatedPostsPerWeek: 14,
      followers: 12000,
      samplePostUri: `at://did:plc:${prefix}anchor0000000001/app.bsky.feed.post/3l7revaaaaaaa`,
    },
    {
      did: `did:plc:${prefix}visual0000000002`,
      role: "visual_maker" as const,
      recommendedMode: "follow_all" as const,
      whyPicked: `Original visual maker in community ${cid}`,
      rankScoreBps: 8800,
      repostShareBps: 600,
      mediaShareBps: 6200,
      estimatedPostsPerWeek: 10,
      followers: 4500,
      samplePostUri: `at://did:plc:${prefix}visual0000000002/app.bsky.feed.post/3l7revaaaaaab`,
    },
    {
      did: `did:plc:${prefix}convo00000000003`,
      role: "conversationalist" as const,
      recommendedMode: "follow_all" as const,
      whyPicked: `Active conversationalist in community ${cid}`,
      rankScoreBps: 8500,
      repostShareBps: 1200,
      mediaShareBps: 900,
      estimatedPostsPerWeek: 21,
      followers: 2200,
      samplePostUri: `at://did:plc:${prefix}convo00000000003/app.bsky.feed.post/3l7revaaaaaac`,
    },
    {
      did: `did:plc:${prefix}curat00000000004`,
      role: "scene_curator" as const,
      recommendedMode: "originals_only" as const,
      whyPicked: `High-signal curator in community ${cid} (originals recommended)`,
      rankScoreBps: 8300,
      repostShareBps: 5800,
      mediaShareBps: 1100,
      estimatedPostsPerWeek: 65,
      followers: 3800,
      samplePostUri: `at://did:plc:${prefix}curat00000000004/app.bsky.feed.post/3l7revaaaaaad`,
    },
    {
      did: HIDDEN_DID,
      role: "anchor_creator" as const,
      recommendedMode: "follow_all" as const,
      whyPicked: "Should be filtered out by A1 hideFromRecsDids",
      rankScoreBps: 9999,
      repostShareBps: 0,
      mediaShareBps: 0,
      estimatedPostsPerWeek: 5,
      followers: 9000,
      samplePostUri: `at://${HIDDEN_DID}/app.bsky.feed.post/3l7revhidden1`,
    },
  ];

  return {
    catalogVersion: "s17_rev6_synth_test",
    generatedAt: "2026-10-31T22:00:00Z",
    sourceCardsSha256: "00".repeat(32),
    totalScenes: 3,
    totalCommunities: 4,
    totalUniqueAccounts: 16,
    noUnauthDids: [NO_UNAUTH_DID],
    hideFromRecsDids: [HIDDEN_DID],
    highHostilityPairs: [[101, 303]],
    warmStartIndex: {
      [SEED_FOLLOW_DID]: [
        [101, 9200],
        [102, 7400],
      ],
    },
    scenes: [
      {
        sceneId: 1,
        sceneSlug: "apple-swift-craft",
        sceneTitle: "Apple & Swift Craft",
        macroCategory: "software_ai_tech",
        symbolName: "swift",
        cardCount: 2,
        totalEligibleAccounts: 120,
        topCommunityIds: [101, 102],
      },
      {
        sceneId: 2,
        sceneSlug: "pop-fandom-hangout",
        sceneTitle: "Pop Culture & Stan Hangout",
        macroCategory: "queer_fandom_hangout",
        symbolName: "sparkles",
        cardCount: 1,
        totalEligibleAccounts: 95,
        topCommunityIds: [201],
      },
      {
        sceneId: 3,
        sceneSlug: "rival-friction-scene",
        sceneTitle: "Rival Friction Commentary",
        macroCategory: "news_civics_history",
        symbolName: "newspaper.fill",
        cardCount: 1,
        totalEligibleAccounts: 80,
        topCommunityIds: [303],
      },
    ],
    communities: [
      {
        communityId: 101,
        title: "SwiftUI & Apple Platform Craft",
        description: "iOS and macOS developers building native apps and custom Metal shaders.",
        sceneId: 1,
        sceneSlug: "apple-swift-craft",
        sceneTitle: "Apple & Swift Craft",
        macroCategory: "software_ai_tech",
        symbolName: "swift",
        eligibleCount: 68,
        postCoherenceBps: 6400,
        socialCoherenceBps: 7100,
        curatedListTags: ["iosdev", "swiftlang"],
        searchTerms: ["xcode", "liquidglass", "coreanimation"],
        centPostsInt8B64: makeCentVecB64(10, 11),
        neighborCommunityIds: [102],
        samplePostUris: [
          "at://did:plc:swftanchor0000000001/app.bsky.feed.post/3l7revaaaaaaa",
          "at://did:plc:swftvisual0000000002/app.bsky.feed.post/3l7revaaaaaab",
          `at://${NO_UNAUTH_DID}/app.bsky.feed.post/3l7revnounau1`,
          `at://${HIDDEN_DID}/app.bsky.feed.post/3l7revhidden1`,
        ],
        previewPostUris: [
          "at://did:plc:swftanchor0000000001/app.bsky.feed.post/3l7revaaaaaaa",
          "at://did:plc:swftvisual0000000002/app.bsky.feed.post/3l7revaaaaaab",
          "at://did:plc:swftconvo00000000003/app.bsky.feed.post/3l7revaaaaaac",
        ],
        sampleAccountDids: ["did:plc:swftanchor0000000001", "did:plc:swftvisual0000000002"],
        feedGeneratorUris: ["at://did:plc:swftfeed000000000001/app.bsky.feed.generator/iosdev"],
        starterPackUris: ["at://did:plc:swftpack000000000001/app.bsky.graph.starterpack/3l7revpack001"],
        members: mkMembers("swft", 101),
      },
      {
        communityId: 102,
        title: "Creative Shaders & Demoscene",
        description: "GLSL, WGSL, raymarching, and real-time procedural graphics.",
        sceneId: 1,
        sceneSlug: "apple-swift-craft",
        sceneTitle: "Apple & Swift Craft",
        macroCategory: "software_ai_tech",
        symbolName: "swift",
        eligibleCount: 52,
        postCoherenceBps: 6900,
        socialCoherenceBps: 6300,
        curatedListTags: ["graphics", "gamedev"],
        searchTerms: ["glsl", "raymarching"],
        centPostsInt8B64: makeCentVecB64(10, 12), // Semantically close to 101!
        neighborCommunityIds: [101],
        samplePostUris: [
          "at://did:plc:shdranchor0000000001/app.bsky.feed.post/3l7revaaaaaaa",
          "at://did:plc:shdrvisual0000000002/app.bsky.feed.post/3l7revaaaaaab",
          "at://did:plc:shdrconvo00000000003/app.bsky.feed.post/3l7revaaaaaac",
        ],
        previewPostUris: [
          "at://did:plc:shdranchor0000000001/app.bsky.feed.post/3l7revaaaaaaa",
          "at://did:plc:shdrvisual0000000002/app.bsky.feed.post/3l7revaaaaaab",
        ],
        sampleAccountDids: ["did:plc:shdranchor0000000001"],
        feedGeneratorUris: [],
        starterPackUris: [],
        members: mkMembers("shdr", 102),
      },
      {
        communityId: 201,
        title: "Pop Tour & Album Fandom",
        description: "Concert livestreams, lyric deep dives, and pop music chart chatter.",
        sceneId: 2,
        sceneSlug: "pop-fandom-hangout",
        sceneTitle: "Pop Culture & Stan Hangout",
        macroCategory: "queer_fandom_hangout",
        symbolName: "sparkles",
        eligibleCount: 95,
        postCoherenceBps: 5900,
        socialCoherenceBps: 7200,
        curatedListTags: ["swifties", "popmusic"],
        searchTerms: ["erastour", "popstan"],
        centPostsInt8B64: makeCentVecB64(90),
        neighborCommunityIds: [],
        samplePostUris: [
          "at://did:plc:popfanchor0000000001/app.bsky.feed.post/3l7revaaaaaaa",
          "at://did:plc:popfvisual0000000002/app.bsky.feed.post/3l7revaaaaaab",
          "at://did:plc:popfconvo00000000003/app.bsky.feed.post/3l7revaaaaaac",
        ],
        previewPostUris: ["at://did:plc:popfanchor0000000001/app.bsky.feed.post/3l7revaaaaaaa"],
        sampleAccountDids: ["did:plc:popfanchor0000000001"],
        feedGeneratorUris: [],
        starterPackUris: [],
        members: mkMembers("popf", 201),
      },
      {
        communityId: 303,
        title: "Hostile Combatant Subscene",
        description: "High cross-community friction group paired against 101 in highHostilityPairs.",
        sceneId: 3,
        sceneSlug: "rival-friction-scene",
        sceneTitle: "Rival Friction Commentary",
        macroCategory: "news_civics_history",
        symbolName: "newspaper.fill",
        eligibleCount: 80,
        postCoherenceBps: 5500,
        socialCoherenceBps: 6000,
        curatedListTags: ["debate"],
        searchTerms: ["friction"],
        centPostsInt8B64: makeCentVecB64(180),
        neighborCommunityIds: [],
        samplePostUris: [
          "at://did:plc:rivlanchor0000000001/app.bsky.feed.post/3l7revaaaaaaa",
          "at://did:plc:rivlvisual0000000002/app.bsky.feed.post/3l7revaaaaaab",
          "at://did:plc:rivlconvo00000000003/app.bsky.feed.post/3l7revaaaaaac",
        ],
        previewPostUris: ["at://did:plc:rivlanchor0000000001/app.bsky.feed.post/3l7revaaaaaaa"],
        sampleAccountDids: ["did:plc:rivlanchor0000000001"],
        feedGeneratorUris: [],
        starterPackUris: [],
        members: mkMembers("rivl", 303),
      },
    ],
    tasteDuels: [
      {
        swipeId: "trap_swift",
        kind: "collision_trap",
        prompt: "When you say 'Swift', which timeline are you looking for?",
        optionA: {
          communityId: 101,
          title: "SwiftUI & Apple Platform Craft",
          subtitle: "iOS and macOS developers building native apps.",
          sceneTitle: "Apple & Swift Craft",
          samplePostUris: ["at://did:plc:swftanchor0000000001/app.bsky.feed.post/3l7revaaaaaaa"],
        },
        optionB: {
          communityId: 201,
          title: "Pop Tour & Album Fandom",
          subtitle: "Concert livestreams and pop music chart chatter.",
          sceneTitle: "Pop Culture & Stan Hangout",
          samplePostUris: ["at://did:plc:popfanchor0000000001/app.bsky.feed.post/3l7revaaaaaaa"],
        },
      },
    ],
  };
}

describe("Catbird Intel v3 Canonical XRPC & DID Key Resolution", () => {
  let viewerKey: P256Keypair;
  let server: ReturnType<typeof Bun.serve>;

  async function signJwt(lxm: string, overrides: Record<string, unknown> = {}) {
    const nowSec = Math.floor(Date.now() / 1000);
    const header = { typ: "JWT", alg: "ES256" };
    const payload = {
      iss: VIEWER_DID,
      aud: AUDIENCE,
      lxm,
      iat: nowSec,
      exp: nowSec + 45,
      jti: crypto.randomUUID(),
      ...overrides,
    };
    const signingInput = [header, payload]
      .map((obj) => Buffer.from(JSON.stringify(obj)).toString("base64url"))
      .join(".");
    const sig = Buffer.from(await viewerKey.sign(Buffer.from(signingInput))).toString("base64url");
    return `${signingInput}.${sig}`;
  }

  beforeAll(async () => {
    viewerKey = await P256Keypair.create();
    const multikey = viewerKey.did().replace(/^did:key:/, "");
    const didResolver = createDidKeyResolver({
      fetchImpl: async (input) => {
        const url = String(input);
        if (url === `https://plc.directory/${encodeURIComponent(VIEWER_DID)}`) {
          return Response.json({
            id: VIEWER_DID,
            verificationMethod: [
              {
                id: `${VIEWER_DID}#atproto`,
                type: "Multikey",
                controller: VIEWER_DID,
                publicKeyMultibase: multikey,
              },
            ],
          });
        }
        return new Response("Not Found", { status: 404 });
      },
    });

    const authenticator = createServiceAuthenticator({
      audience: AUDIENCE,
      resolveAtprotoKey: didResolver,
    });
    const engine = new S17CatalogEngine(buildSyntheticBundle());
    server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch: createApiHandler({
        authenticate: authenticator,
        catalogEngine: engine,
        requireServiceAuth: true,
      }),
    });
  });

  afterAll(() => {
    server.stop(true);
  });

  test("rejects unauthenticated requests to canonical XRPC endpoints with 401 AuthRequired", async () => {
    const res = await fetch(new URL("/xrpc/blue.catbird.onboarding.getCatalog", server.url));
    expect(res.status).toBe(401);
    const body = (await res.json()) as any;
    expect(body.error).toBe("AuthRequired");
  });

  test("DidKeyResolver resolves did:web .well-known/did.json and caches results", async () => {
    const webKey = await P256Keypair.create();
    const multikey = formatMultikey("ES256", webKey.publicKeyBytes());
    let calls = 0;
    const resolve = createDidKeyResolver({
      fetchImpl: async (input) => {
        calls += 1;
        expect(String(input)).toBe("https://alice.example/.well-known/did.json");
        return Response.json({
          id: "did:web:alice.example",
          verificationMethod: [
            {
              id: "#atproto",
              type: "Multikey",
              controller: "did:web:alice.example",
              publicKeyMultibase: multikey,
            },
          ],
        });
      },
    });
    const k1 = await resolve("did:web:alice.example");
    const k2 = await resolve("did:web:alice.example");
    expect(k1).toBe(webKey.did());
    expect(k2).toBe(webKey.did());
    expect(calls).toBe(1);
  });

  test("getCatalog returns warm-start orbit communities, featured scenes, and enforces A1 safety", async () => {
    const jwt = await signJwt("blue.catbird.onboarding.getCatalog");
    const url = new URL("/xrpc/blue.catbird.onboarding.getCatalog", server.url);
    url.searchParams.append("seedFollowDids", SEED_FOLLOW_DID);
    const res = await fetch(url, { headers: { authorization: `Bearer ${jwt}` } });
    expect(res.status).toBe(200);
    const body = (await res.json()) as any;
    expect(body.catalogVersion).toBe("s17_rev6_synth_test");
    expect(body.scenes).toHaveLength(3);
    expect(body.warmStartCommunities.map((c: any) => c.communityId)).toEqual([101, 102]);
    // Authenticated viewer can see NO_UNAUTH_DID sample post, but NEVER HIDDEN_DID
    expect(JSON.stringify(body)).toContain(NO_UNAUTH_DID);
    expect(JSON.stringify(body)).not.toContain(HIDDEN_DID);
  });

  test("listCommunities supports hybrid concept + semantic neighbor search and getCommunity returns full details", async () => {
    const jwtList = await signJwt("blue.catbird.intel.listCommunities");
    const listUrl = new URL("/xrpc/blue.catbird.intel.listCommunities", server.url);
    listUrl.searchParams.set("q", "liquidglass");
    const listRes = await fetch(listUrl, { headers: { authorization: `Bearer ${jwtList}` } });
    expect(listRes.status).toBe(200);
    const listBody = (await listRes.json()) as any;
    // Matches 101 directly on searchTerms ("liquidglass") AND propagates to semantic neighbor 102!
    expect(listBody.communities[0].communityId).toBe(101);
    expect(listBody.communities.some((c: any) => c.communityId === 102)).toBe(true);

    const jwtDetail = await signJwt("blue.catbird.intel.getCommunity");
    const detailUrl = new URL("/xrpc/blue.catbird.intel.getCommunity?communityId=101", server.url);
    const detailRes = await fetch(detailUrl, { headers: { authorization: `Bearer ${jwtDetail}` } });
    expect(detailRes.status).toBe(200);
    const detailBody = (await detailRes.json()) as any;
    expect(detailBody.community.communityId).toBe(101);
    expect(detailBody.members.every((m: any) => m.did !== HIDDEN_DID)).toBe(true);
    expect(detailBody.neighborCommunities.map((c: any) => c.communityId)).toEqual([102]);
  });

  test("getCalibrationSwipes, previewTimeline, and solveBundle execute full 5-beat flow with hostility & repost taming", async () => {
    // 1. Calibration swipes
    const jwtSwipes = await signJwt("blue.catbird.onboarding.getCalibrationSwipes");
    const swipesRes = await fetch(new URL("/xrpc/blue.catbird.onboarding.getCalibrationSwipes", server.url), {
      method: "POST",
      headers: {
        authorization: `Bearer ${jwtSwipes}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        selections: [
          { communityId: 101, weight: 2 },
          { communityId: 303, weight: 1 },
        ],
        maxSwipes: 4,
      }),
    });
    expect(swipesRes.status).toBe(200);
    const swipesBody = (await swipesRes.json()) as any;
    expect(swipesBody.swipes.length).toBeGreaterThanOrEqual(1);
    expect(swipesBody.swipes[0].swipeId).toBe("trap_swift");

    // 2. Preview timeline
    const jwtPreview = await signJwt("blue.catbird.onboarding.previewTimeline");
    const prevRes = await fetch(new URL("/xrpc/blue.catbird.onboarding.previewTimeline", server.url), {
      method: "POST",
      headers: {
        authorization: `Bearer ${jwtPreview}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        selections: [
          { communityId: 101, weight: 2 },
          { communityId: 102, weight: 1 },
        ],
        swipes: [{ swipeId: "trap_swift", communityA: 101, communityB: 201, choice: "a" }],
        controls: { pace: "balanced", originalsOnlyForHeavyReposters: true },
        limit: 10,
      }),
    });
    expect(prevRes.status).toBe(200);
    const prevBody = (await prevRes.json()) as any;
    expect(prevBody.items.length).toBeGreaterThan(0);
    expect(prevBody.items[0]).toHaveProperty("post");
    expect(prevBody.items[0]).toHaveProperty("communityTitle");
    expect(JSON.stringify(prevBody)).not.toContain(HIDDEN_DID);

    // 3. Solve bundle: select 101 (weight 2) + 102 (weight 1) + hostile 303 (weight 1)
    const jwtSolve = await signJwt("blue.catbird.onboarding.solveBundle");
    const pinnedDid = "did:plc:shdrvisual0000000002";
    const excludedDid = "did:plc:swftconvo00000000003";
    const solveRes = await fetch(new URL("/xrpc/blue.catbird.onboarding.solveBundle", server.url), {
      method: "POST",
      headers: {
        authorization: `Bearer ${jwtSolve}`,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        selections: [
          { communityId: 101, weight: 2 },
          { communityId: 102, weight: 1 },
          { communityId: 303, weight: 1 },
        ],
        swipes: [{ swipeId: "trap_swift", communityA: 101, communityB: 201, choice: "a" }],
        controls: { pace: "balanced", originalsOnlyForHeavyReposters: true },
        targetSize: 8,
        pinnedDids: [pinnedDid],
        excludedDids: [excludedDid],
      }),
    });
    expect(solveRes.status).toBe(200);
    const solveBody = (await solveRes.json()) as any;
    const followDids = solveBody.follows.map((f: any) => f.did);
    expect(followDids).toContain(pinnedDid);
    expect(followDids).not.toContain(excludedDid);
    expect(followDids).not.toContain(HIDDEN_DID);
    // Hostile community 303 should be suppressed once 101 is selected!
    expect(solveBody.follows.every((f: any) => f.communityId !== 303)).toBe(true);
    // Heavy reposter curator has hideReposts == true when Smart Repost Tamer is enabled
    const curatorFollow = solveBody.follows.find((f: any) => f.role === "scene_curator");
    expect(curatorFollow).toBeDefined();
    expect(curatorFollow.hideReposts).toBe(true);
    expect(solveBody.summary.originalsOnlyCount).toBeGreaterThanOrEqual(1);
    expect(solveBody.feeds.map((f: any) => f.uri)).toContain(
      "at://did:plc:swftfeed000000000001/app.bsky.feed.generator/iosdev",
    );
  });
});
