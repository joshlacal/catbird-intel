import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { createHmac } from "node:crypto";
import { createApiHandler, installVectorStore } from "../src/api.ts";
import { createServiceAuthenticator } from "../src/service_auth.ts";
import {
  IntelVectorStore,
  cosineI8,
  dequantizeQ8,
  float32ToFp16Bits,
  fp16BitsToFloat32,
  quantizeFloat32ToQ8,
  quantizedBytes,
  type IntelCommunity,
} from "../src/vector_store.ts";

const VIEWER = "did:plc:v4viewer0000000000000001";
const ACTOR_A = "did:plc:v4actora00000000000000001";
const ACTOR_B = "did:plc:v4actorb00000000000000001";
const HIDDEN = "did:plc:v4hidden0000000000000001";
const RUST_ACTOR = "did:plc:v4rust000000000000000001";
const POST_A = `at://${ACTOR_A}/app.bsky.feed.post/3mwebnae6sk2p`;
const POST_B = `at://${ACTOR_B}/app.bsky.feed.post/3mwebnae6sk2q`;
const POST_HIDDEN = `at://${HIDDEN}/app.bsky.feed.post/3mwebnae6sk2r`;
const POST_ORIGIN = `at://${ACTOR_A}/app.bsky.feed.post/3mwebnae6sk2s`;
const POST_QUOTE = `at://${ACTOR_B}/app.bsky.feed.post/3mwebnae6sk2t`;
const GENERATION = "v4-fixture-generation";
const AUDIENCE = "did:web:intel.test#catbird_intel";

function unitVector(dimension: number, secondary?: number): number[] {
  const vector = Array(256).fill(0) as number[];
  vector[dimension] = 1;
  if (secondary !== undefined) vector[secondary] = 0.4;
  return vector;
}

function community(
  communityId: number,
  label: string,
  description: string,
  vector: number[],
  samplePostUris: string[],
  sampleAccountDids: string[],
): Partial<IntelCommunity> & Pick<IntelCommunity, "communityId"> {
  const quantized = quantizeFloat32ToQ8(vector);
  return {
    communityId,
    label,
    description,
    sceneSlug: label.toLowerCase().replaceAll(" ", "-"),
    macroCategory: "software_ai_tech",
    samplePostUris,
    sampleAccountDids,
    centPostsQ8B64: quantized.q8Bytes.$bytes,
    centPostsScaleFp16: quantized.scaleFp16,
    centFusedQ8B64: quantized.q8Bytes.$bytes,
    centFusedScaleFp16: quantized.scaleFp16,
    centT1Q8B64: quantized.q8Bytes.$bytes,
    centT1ScaleFp16: quantized.scaleFp16,
  };
}

const store = IntelVectorStore.fromFixtures({
  generation: GENERATION,
  actors: [
    { did: VIEWER, all: unitVector(0), preT: unitVector(0, 1), t1: unitVector(10), topCommunityIds: [1], vectorCount: 30 },
    { did: ACTOR_A, all: unitVector(0), preT: unitVector(0, 1), t1: unitVector(10), topCommunityIds: [1], qualityScoreBps: 9400, rageBaitPermille: 100, vectorCount: 24 },
    { did: ACTOR_B, all: unitVector(1), preT: unitVector(1, 0), t1: unitVector(11), topCommunityIds: [1], qualityScoreBps: 8000, rageBaitPermille: 700, vectorCount: 18 },
    { did: HIDDEN, all: unitVector(0), preT: unitVector(0), t1: unitVector(10), topCommunityIds: [1], hiddenFromRecommendations: true, qualityScoreBps: 9999, vectorCount: 50 },
    { did: RUST_ACTOR, all: unitVector(20), preT: unitVector(20), t1: unitVector(21), topCommunityIds: [2], qualityScoreBps: 8800, vectorCount: 12 },
  ],
  communities: [
    community(1, "Swift iOS developers", "Native Apple platform apps using SwiftUI and UIKit", unitVector(0),
      [POST_A, POST_B, POST_HIDDEN], [ACTOR_A, ACTOR_B, HIDDEN]),
    community(2, "Rust systems programming", "Memory safe low-level systems and async runtimes", unitVector(20),
      [], [RUST_ACTOR]),
  ],
});

function postView(uri: string, quoteUri?: string) {
  const authorDid = uri.split("/")[2]!;
  return {
    uri,
    cid: "bafyreifixture",
    author: { did: authorDid, handle: `${authorDid.slice(-5)}.test` },
    indexedAt: "2026-11-09T12:00:00.000Z",
    record: {
      $type: "app.bsky.feed.post",
      text: uri === POST_B ? "Swift debate and outrage" : "Swift iOS native application craft",
      createdAt: "2026-11-09T12:00:00.000Z",
      ...(quoteUri ? { embed: { $type: "app.bsky.embed.record", record: { uri: quoteUri, cid: "bafyreiquote" } } } : {}),
    },
    replyCount: 2,
    quoteCount: uri === POST_QUOTE ? 8 : 1,
    likeCount: 5,
    repostCount: 3,
  };
}

const posts = new Map([
  [POST_A, postView(POST_A, POST_ORIGIN)],
  [POST_B, postView(POST_B)],
  [POST_HIDDEN, postView(POST_HIDDEN)],
  [POST_ORIGIN, postView(POST_ORIGIN)],
  [POST_QUOTE, postView(POST_QUOTE, POST_A)],
]);

const appViewFetch = async (input: string | URL | Request): Promise<Response> => {
  const url = new URL(input instanceof Request ? input.url : input.toString());
  const method = url.pathname.slice("/xrpc/".length);
  if (method === "app.bsky.graph.getFollows") return Response.json({ follows: [{ did: ACTOR_A }] });
  if (method === "app.bsky.feed.getPosts") {
    return Response.json({ posts: url.searchParams.getAll("uris").map((uri) => posts.get(uri)).filter(Boolean) });
  }
  if (method === "app.bsky.feed.searchPosts") {
    return Response.json({ posts: [posts.get(POST_A), posts.get(POST_B), posts.get(POST_HIDDEN)] });
  }
  if (method === "app.bsky.feed.getQuotes") return Response.json({ posts: [posts.get(POST_QUOTE)] });
  return Response.json({ error: "NotFound" }, { status: 404 });
};

const handler = createApiHandler({
  authenticate: async () => VIEWER,
  vectorStore: store,
  queryClickHouse: async () => { throw new Error("fixture ClickHouse is offline"); },
  appViewFetch,
  now: () => Date.parse("2026-11-09T13:00:00.000Z"),
});

function apiRequest(path: string, body?: Record<string, unknown>): Promise<Response> {
  return handler(new Request(`http://intel.test${path}`, body ? {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  } : undefined));
}
interface TestWireVector { q8Bytes: { $bytes: string } }
interface TestActor {
  did: string;
  centroids: Array<{ window: string; vector: TestWireVector }>;
  health: { qualityScoreBps: number };
}
interface TestPost {
  uri: string;
  authorDid: string;
  rageBaitPermille: number;
  similarityPermille: number;
  vector: TestWireVector;
}
interface TestCluster {
  clusterId: number;
  label: string;
  centroid: TestWireVector;
  samplePosts: TestPost[];
}
interface TestApiResponse {
  generation: string;
  vector_store_generation: string;
  vector_store_producers: number;
  notModified: boolean;
  viewerCentroids: Array<{ window: string; vector: TestWireVector }>;
  actors: TestActor[];
  posts: TestPost[];
  queryVectorUsed: TestWireVector;
  originUri: string;
  catalystUri: string;
  chain: TestPost[];
  hourlyBuckets: Array<{ hourStart: string; postCount: number }>;
  clusters: TestCluster[];
}


async function responseJson(path: string, body?: Record<string, unknown>): Promise<TestApiResponse> {
  const response = await apiRequest(path, body);
  expect(response.status).toBe(200);
  return await response.json() as TestApiResponse;
}

beforeAll(() => installVectorStore(store));
afterAll(() => installVectorStore(null));

describe("IntelVectorStore binary and quantized-vector contract", () => {
  test("fp16 conversion and q8 quantization round-trip normalized vectors", () => {
    for (const value of [0, -0, 0.0001, 0.25, 1, -3.5, 65504]) {
      const roundTrip = fp16BitsToFloat32(float32ToFp16Bits(value));
      expect(Math.abs(roundTrip - value)).toBeLessThanOrEqual(Math.max(0.000001, Math.abs(value) * 0.001));
    }
    const source = unitVector(7, 31);
    const quantized = quantizeFloat32ToQ8(source);
    const decoded = dequantizeQ8(quantized);
    expect(quantized.dims).toBe(256);
    expect(Object.keys(quantized.q8Bytes)).toEqual(["$bytes"]);
    expect(quantizedBytes(quantized)).toHaveLength(256);
    expect(cosineI8(quantizedBytes(quantized), quantizedBytes(quantizeFloat32ToQ8(decoded)))).toBeGreaterThan(0.999);
    expect(Math.hypot(...decoded)).toBeCloseTo(1, 2);
  });

  test("fixture records expose raw hidden actors but exclude them from community discovery", () => {
    expect(store.signalForDid(HIDDEN)?.hiddenFromRecommendations).toBe(true);
    expect(store.vectorForDid(HIDDEN, "all")).not.toBeNull();
    expect(store.communityMembers(1, 20)).toEqual([ACTOR_A, ACTOR_B, VIEWER]);
  });

  test("lexical query fallback deterministically selects weighted community centroids", async () => {
    const previous = process.env.EG2_SIDECAR_URL;
    process.env.EG2_SIDECAR_URL = "http://127.0.0.1:1/embed";
    try {
      const first = await store.encodeQueryText("Swift iOS application development");
      const second = await store.encodeQueryText("Swift iOS application development");
      expect(first).toEqual(second);
      expect(cosineI8(quantizedBytes(first), quantizedBytes(store.communityVector(store.community(1)!)))).toBeGreaterThan(0.9);
    } finally {
      if (previous === undefined) delete process.env.EG2_SIDECAR_URL; else process.env.EG2_SIDECAR_URL = previous;
    }
  });
});

describe("Nest internal BFF service assertion", () => {
  test("authenticates exact audience/method with constant-time HMAC and rejects replay", async () => {
    const secret = "correct horse battery staple for catbird bff";
    const now = 1_800_000_000;
    const payload = Buffer.from(JSON.stringify({
      iss: VIEWER,
      aud: AUDIENCE,
      lxm: "blue.catbird.intel.getActorVectors",
      iat: now,
      exp: now + 60,
      jti: crypto.randomUUID(),
    })).toString("base64url");
    const signature = createHmac("sha256", secret).update(payload).digest("base64url");
    const token = `bff.${payload}.${signature}`;
    const authenticate = createServiceAuthenticator({
      audience: AUDIENCE,
      internalBffSecret: secret,
      now: () => now * 1000,
      resolveAtprotoKey: async () => { throw new Error("BFF assertions never resolve a DID key"); },
    });
    const request = new Request("http://intel.test/xrpc/blue.catbird.intel.getActorVectors", {
      headers: { authorization: `Bearer ${token}` },
    });
    expect(await authenticate(request, "blue.catbird.intel.getActorVectors")).toBe(VIEWER);
    await expect(authenticate(request, "blue.catbird.intel.getActorVectors")).rejects.toThrow("Invalid or expired");
    const freshPayload = Buffer.from(JSON.stringify({
      iss: VIEWER, aud: AUDIENCE, lxm: "wrong.method", iat: now, exp: now + 60, jti: crypto.randomUUID(),
    })).toString("base64url");
    const wrongMethod = `bff.${freshPayload}.${createHmac("sha256", secret).update(freshPayload).digest("base64url")}`;
    await expect(authenticate(new Request(request.url, { headers: { authorization: `Bearer ${wrongMethod}` } }),
      "blue.catbird.intel.getActorVectors")).rejects.toThrow();
  });
});

describe("Intel XRPC v4 routes", () => {
  test("health and graph bundle report generation/counts and support not-modified", async () => {
    const health = await responseJson("/health");
    expect(health.vector_store_generation).toBe(GENERATION);
    expect(health.vector_store_producers).toBe(5);
    const bundle = await responseJson("/xrpc/blue.catbird.intel.getGraphNeighborhoodBundle?maxTwoHopActors=10");
    expect(bundle.notModified).toBe(false);
    expect(bundle.viewerCentroids.map((centroid) => centroid.window)).toEqual(["7d", "30d", "90d"]);
    expect(bundle.viewerCentroids[0]!.vector.q8Bytes).toEqual({ $bytes: expect.any(String) });
    expect(bundle.actors.map((actor) => actor.did)).toEqual([ACTOR_A, ACTOR_B]);
    expect(JSON.stringify(bundle)).not.toContain(HIDDEN);
    const unchanged = await responseJson(`/xrpc/blue.catbird.intel.getGraphNeighborhoodBundle?ifNoneMatchGeneration=${GENERATION}`);
    expect(unchanged).toMatchObject({ generation: GENERATION, notModified: true, actors: [], viewerCentroids: [] });
  });

  test("getActorVectors preserves input order, raw hidden access, wire bytes, and never logs DID lists", async () => {
    const originalLog = console.log;
    const originalError = console.error;
    const logged: unknown[][] = [];
    console.log = (...values) => { logged.push(values); };
    console.error = (...values) => { logged.push(values); };
    try {
      const result = await responseJson("/xrpc/blue.catbird.intel.getActorVectors", {
        dids: [ACTOR_B, HIDDEN, ACTOR_A],
        windows: ["30d", "pre_t"],
      });
      expect(result.actors.map((actor) => actor.did)).toEqual([ACTOR_B, HIDDEN, ACTOR_A]);
      expect(result.actors[0]!.centroids[0]!.vector.q8Bytes).toEqual({ $bytes: expect.any(String) });
      expect(JSON.stringify(logged)).not.toContain(ACTOR_A);
      expect(JSON.stringify(logged)).not.toContain(ACTOR_B);
    } finally {
      console.log = originalLog;
      console.error = originalError;
    }
  });

  test("getPostVectors returns every valid URI with vectors in ATProto bytes shape", async () => {
    const result = await responseJson("/xrpc/blue.catbird.intel.getPostVectors", {
      uris: [POST_A, POST_B, POST_HIDDEN],
      includeVectors: true,
    });
    expect(result.posts.map((post) => post.uri)).toEqual([POST_A, POST_B, POST_HIDDEN]);
    expect(result.posts.every((post) => Object.keys(post.vector.q8Bytes)[0] === "$bytes")).toBe(true);
    expect(result.posts[1]!).toMatchObject({ authorDid: ACTOR_B, rageBaitPermille: 700 });
  });

  test("searchGraphPosts ranks semantic graph posts and enforces rage-bait and hidden-author filters", async () => {
    const result = await responseJson("/xrpc/blue.catbird.intel.searchGraphPosts", {
      query: "Swift iOS application",
      queryVector: quantizeFloat32ToQ8(unitVector(0)),
      maxHops: 2,
      maxRageBaitPermille: 200,
      includeVectors: true,
      limit: 20,
    });
    expect(result.queryVectorUsed.q8Bytes).toEqual({ $bytes: expect.any(String) });
    expect(result.posts.map((post) => post.uri)).toEqual([POST_A]);
    expect(result.posts[0]!.similarityPermille).toBeGreaterThan(900);
    expect(JSON.stringify(result)).not.toContain(POST_HIDDEN);
  });

  test("expandGraphNeighborhood filters hidden and high-rage actors then ranks by affinity and quality", async () => {
    const result = await responseJson(
      "/xrpc/blue.catbird.intel.expandGraphNeighborhood?communityId=1&maxRageBaitPermille=200&limit=20",
    );
    expect(result.actors.map((actor) => actor.did)).toEqual([ACTOR_A]);
    expect(result.actors[0]!.health.qualityScoreBps).toBe(9400);
  });

  test("getQuoteChain walks upstream, identifies a catalyst, and buckets chronology", async () => {
    const result = await responseJson(`/xrpc/blue.catbird.intel.getQuoteChain?anchorUri=${encodeURIComponent(POST_A)}&limit=20`);
    expect(result.originUri).toBe(POST_ORIGIN);
    expect(result.catalystUri).toBe(POST_QUOTE);
    expect(result.chain.map((post) => post.uri)).toContain(POST_ORIGIN);
    expect(result.hourlyBuckets[0]!).toMatchObject({ hourStart: expect.any(String), postCount: expect.any(Number) });
  });

  test("getTopicClusters emits quantized centroids and removes hidden samples", async () => {
    const result = await responseJson("/xrpc/blue.catbird.intel.getTopicClusters?q=Swift&limit=10");
    expect(result.clusters).toHaveLength(1);
    expect(result.clusters[0]!).toMatchObject({ clusterId: 1, label: "Swift iOS developers" });
    expect(result.clusters[0]!.centroid.q8Bytes).toEqual({ $bytes: expect.any(String) });
    expect(result.clusters[0]!.samplePosts.map((post) => post.uri)).toEqual([POST_A, POST_B]);
    expect(JSON.stringify(result)).not.toContain(HIDDEN);
  });
});
