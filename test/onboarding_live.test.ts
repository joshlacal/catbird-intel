import { describe, expect, test } from "bun:test";
import { createApiHandler } from "../src/api.ts";
import {
  S17CatalogEngine,
  type CommunityRole,
  type IntelMemberEntry,
  type IntelServingBundleV2,
} from "../src/intel_catalog.ts";
import {
  LiveOnboardingEngine,
  allocateByWeight,
  createPublicFollowsFetcher,
  type FollowsFetcher,
  type OnboardingPoolFile,
} from "../src/onboarding_live.ts";

const ROLES: CommunityRole[] = [
  "anchor_creator", "visual_maker", "conversationalist", "scene_curator", "project_or_studio",
];
const VIEWER = { did: "did:plc:viewer00000000000000001", unauthenticated: false };
const ANON = { did: null, unauthenticated: true };
const HIDDEN = "did:plc:c101m0003";
const NO_UNAUTH = "did:plc:c101m0005";
/** In both community 101 and 102, to exercise accounts shared between two picked scenes. */
const SHARED = "did:plc:shared0001";

function memberDid(cid: number, index: number): string {
  return `did:plc:c${cid}m${String(index).padStart(4, "0")}`;
}

function member(cid: number, index: number, did = memberDid(cid, index)): IntelMemberEntry {
  return {
    did,
    role: ROLES[index % ROLES.length]!,
    recommendedMode: index % 7 === 0 ? "originals_only" : "follow_all",
    whyPicked: `Member ${index} of community ${cid}`,
    rankScoreBps: 9600 - index * 90,
    repostShareBps: index % 7 === 0 ? 5200 : 900,
    mediaShareBps: 2000,
    estimatedPostsPerWeek: 10 + (index % 5) * 12,
    followers: index % 9 === 0 ? 40_000 : index % 4 === 0 ? 900 : 5_000,
    samplePostUri: `at://${did}/app.bsky.feed.post/3lpost${String(index).padStart(4, "0")}`,
  };
}

function buildBundle(): IntelServingBundleV2 {
  const cent = Buffer.from(new Int8Array(256).fill(7).buffer).toString("base64");
  const community = (cid: number, sceneId: number, neighbors: number[]) => {
    const members = Array.from({ length: 25 }, (_, i) => member(cid, i));
    if (cid === 101 || cid === 102) members[2] = member(cid, 2, SHARED);
    return {
      communityId: cid,
      title: `Community ${cid}`,
      description: `Synthetic community ${cid}`,
      sceneId,
      sceneSlug: `scene-${sceneId}`,
      sceneTitle: `Scene ${sceneId}`,
      macroCategory: "software_ai_tech",
      symbolName: "sparkles",
      eligibleCount: 60,
      postCoherenceBps: 7000,
      socialCoherenceBps: 7000,
      curatedListTags: [`tag${cid}`],
      searchTerms: [`term${cid}`],
      centPostsInt8B64: cent,
      neighborCommunityIds: neighbors,
      samplePostUris: members.slice(0, 3).map((m) => m.samplePostUri!),
      previewPostUris: members.slice(3, 9).map((m) => m.samplePostUri!),
      sampleAccountDids: members.slice(0, 6).map((m) => m.did),
      feedGeneratorUris: [`at://${memberDid(cid, 0)}/app.bsky.feed.generator/feed${cid}`],
      starterPackUris: [`at://${memberDid(cid, 1)}/app.bsky.graph.starterpack/pack${cid}`],
      members,
    };
  };
  const communities = [
    community(101, 1, [102, 103]),
    community(102, 1, [101, 104]),
    community(103, 2, [101]),
    community(104, 2, [102]),
  ];
  return {
    catalogVersion: "synthetic_live_v1",
    generatedAt: "2026-11-22T00:00:00Z",
    sourceCardsSha256: "0".repeat(64),
    totalScenes: 2,
    totalCommunities: communities.length,
    totalUniqueAccounts: 99,
    noUnauthDids: [NO_UNAUTH],
    hideFromRecsDids: [HIDDEN],
    highHostilityPairs: [[103, 104]],
    warmStartIndex: { [memberDid(101, 0)]: [[101, 9000]] },
    scenes: [1, 2].map((sceneId) => ({
      sceneId,
      sceneSlug: `scene-${sceneId}`,
      sceneTitle: `Scene ${sceneId}`,
      macroCategory: "software_ai_tech",
      symbolName: "sparkles",
      cardCount: 2,
      totalEligibleAccounts: 120,
      topCommunityIds: sceneId === 1 ? [101, 102] : [103, 104],
    })),
    communities,
    tasteDuels: [],
  };
}

/** Bundle members first and verbatim, then 35 deeper candidates per community. */
function buildPool(bundle: IntelServingBundleV2): OnboardingPoolFile {
  return {
    poolVersion: "pool_v1_synthetic",
    generatedAt: "2026-11-22T00:00:00Z",
    sourceCatalogVersion: bundle.catalogVersion,
    depth: 60,
    communities: bundle.communities.map((c) => ({
      communityId: c.communityId,
      members: [...c.members, ...Array.from({ length: 35 }, (_, i) => member(c.communityId, 25 + i))],
    })),
  };
}

function engineWith(options: {
  follows?: string[];
  fetcher?: FollowsFetcher;
  pool?: boolean;
  seed?: number;
  config?: ConstructorParameters<typeof LiveOnboardingEngine>[0]["config"];
} = {}) {
  const bundle = buildBundle();
  const follows = options.follows ?? [];
  let draws = 0;
  return new LiveOnboardingEngine({
    catalog: new S17CatalogEngine(bundle),
    pool: options.pool === false ? null : buildPool(bundle),
    followsFetcher: options.fetcher ?? (async (_did, onPage) => onPage(follows)),
    // First draw is the session seed; later draws only label session ids.
    random: () => (draws++ === 0 ? (options.seed ?? 0.123456) : 0.5),
    config: options.config,
  });
}

const pick = (communityId: number, weight = 1) => ({ communityId, weight });

describe("live onboarding: what is shown is what is followed", () => {
  test("the card is a prefix of the detail sheet, and the sheet's prefix is the lineup", async () => {
    const engine = engineWith({ follows: [memberDid(101, 1), memberDid(102, 4), memberDid(103, 0)] });
    const catalog = await engine.getCatalog({ limit: 10 }, VIEWER);
    for (const communityId of [101, 102, 103]) {
      const card = catalog.featuredCommunities.find((c) => c.communityId === communityId)!;
      const detail = (await engine.getCommunity(communityId, VIEWER))!;
      expect(card.sampleAccountDids).toHaveLength(6);
      expect(detail.members.slice(0, 6).map((m) => m.did)).toEqual(card.sampleAccountDids);
      expect(detail.community.sampleAccountDids).toEqual(card.sampleAccountDids);
      expect(detail.defaultLineupCount).toBe(6);
      expect(detail.boostedLineupCount).toBe(12);
    }

    const solved = await engine.solveBundle({ selections: [pick(103), pick(104, 2)] }, VIEWER);
    const from = (cid: number) => solved.follows.filter((f) => f.communityId === cid).map((f) => f.did);
    const sheet103 = (await engine.getCommunity(103, VIEWER))!.members.map((m) => m.did);
    const sheet104 = (await engine.getCommunity(104, VIEWER))!.members.map((m) => m.did);
    expect(from(103)).toEqual(sheet103.slice(0, 6));
    expect(from(104)).toEqual(sheet104.slice(0, 12));
    expect(solved.summary.totalFollows).toBe(18);
    // Follows come back grouped by scene, in the order the scenes were picked.
    expect(solved.follows.map((f) => f.communityId)).toEqual([...Array(6).fill(103), ...Array(12).fill(104)]);
  });

  test("an account the viewer already follows is never shown, offered or counted", async () => {
    const follows = Array.from({ length: 14 }, (_, i) => memberDid(101, i));
    const engine = engineWith({ follows });
    const catalog = await engine.getCatalog({}, VIEWER);
    const card = catalog.featuredCommunities.find((c) => c.communityId === 101)!;
    expect(card.alreadyFollowingCount).toBe(12); // of the 14: index 2 is the shared account and index 3 is hidden
    const detail = (await engine.getCommunity(101, VIEWER))!;
    const solved = await engine.solveBundle({ selections: [pick(101)] }, VIEWER);
    const shown = [...card.sampleAccountDids, ...detail.members.map((m) => m.did),
      ...solved.follows.map((f) => f.did), ...solved.bench.map((f) => f.did)];
    for (const did of shown) expect(follows).not.toContain(did);
    expect(solved.follows).toHaveLength(6); // every slot is a new follow
    expect(solved.alreadyFollowingInScenes).toBe(12);
  });

  test("the follow list the client sends is honoured even when the server cannot fetch one", async () => {
    const engine = engineWith({ fetcher: async () => { throw new Error("appview down"); } });
    const seeds = [memberDid(101, 0), memberDid(101, 6)];
    const solved = await engine.solveBundle({ selections: [pick(101)], seedFollowDids: seeds }, VIEWER);
    expect(solved.follows).toHaveLength(6);
    for (const follow of solved.follows) expect(seeds).not.toContain(follow.did);
  });

  test("a follow list that finishes loading mid-flow only shifts a list up", async () => {
    let release: () => void = () => undefined;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const engine = engineWith({
      config: { followsBudgetMs: 5 },
      fetcher: async (_did, onPage) => {
        await gate;
        onPage([]);
      },
    });
    const before = (await engine.getCommunity(101, VIEWER))!.members.map((m) => m.did);
    // The slow fetch now reports that the viewer follows the second and fourth account shown.
    const nowFollowed = [before[1]!, before[3]!];
    const late = engineWith({
      config: { followsBudgetMs: 5 },
      fetcher: async (_did, onPage) => {
        await gate;
        onPage(nowFollowed);
      },
    });
    const lateBefore = (await late.getCommunity(101, VIEWER))!.members.map((m) => m.did);
    expect(lateBefore).toEqual(before); // same seed, same order while the fetch is pending
    release();
    await new Promise((resolve) => setTimeout(resolve, 10));
    const after = (await late.getCommunity(101, VIEWER))!.members.map((m) => m.did);
    expect(after.slice(0, 20)).toEqual(before.filter((did) => !nowFollowed.includes(did)).slice(0, 20));
  });

  test("hidden accounts never appear, and no-unauth accounts never appear to signed-out viewers", async () => {
    const engine = engineWith();
    const signedIn = (await engine.getCommunity(101, VIEWER))!.members.map((m) => m.did);
    expect(signedIn).not.toContain(HIDDEN);
    const all = await engine.solveBundle({ selections: [pick(101, 2)], targetSize: 60 }, VIEWER);
    expect(all.follows.map((f) => f.did)).toContain(NO_UNAUTH);
    expect(all.follows.map((f) => f.did)).not.toContain(HIDDEN);
    const signedOut = await engine.solveBundle({ selections: [pick(101, 2)], targetSize: 60 }, ANON);
    expect(signedOut.follows.map((f) => f.did)).not.toContain(NO_UNAUTH);
  });
});

describe("live onboarding: sizing", () => {
  test("six per picked scene, twelve per boosted scene, and no duplicates across scenes", async () => {
    const engine = engineWith();
    const solved = await engine.solveBundle({ selections: [pick(101), pick(102), pick(103, 2)] }, VIEWER);
    const dids = solved.follows.map((f) => f.did);
    expect(new Set(dids).size).toBe(dids.length);
    expect(solved.follows.filter((f) => f.communityId === 101)).toHaveLength(6);
    expect(solved.follows.filter((f) => f.communityId === 102)).toHaveLength(6);
    expect(solved.follows.filter((f) => f.communityId === 103)).toHaveLength(12);
    expect(solved.summary.coveredCommunities).toBe(3);
  });

  test("an account in two picked scenes is followed once and the second scene still fills its share", async () => {
    // A seed that puts the shared account in the first six of both scenes.
    for (let seed = 0.01; seed < 1; seed += 0.01) {
      const engine = engineWith({ seed });
      const a = (await engine.getCommunity(101, VIEWER))!.members.slice(0, 6).map((m) => m.did);
      const b = (await engine.getCommunity(102, VIEWER))!.members.slice(0, 6).map((m) => m.did);
      if (!a.includes(SHARED) || !b.includes(SHARED)) continue;
      const solved = await engine.solveBundle({ selections: [pick(101), pick(102)] }, VIEWER);
      const dids = solved.follows.map((f) => f.did);
      expect(dids.filter((did) => did === SHARED)).toHaveLength(1);
      expect(solved.follows.filter((f) => f.communityId === 102)).toHaveLength(6);
      for (const did of [...a, ...b]) expect(dids).toContain(did); // everyone shown is followed
      return;
    }
    throw new Error("no seed placed the shared account in both prefixes");
  });

  test("an older client's targetSize is one total, split by weight, still a prefix of each list", async () => {
    const engine = engineWith();
    const solved = await engine.solveBundle({ selections: [pick(101), pick(103, 2), pick(104)], targetSize: 25 }, VIEWER);
    expect(solved.follows).toHaveLength(25);
    const share = (cid: number) => solved.follows.filter((f) => f.communityId === cid).length;
    expect(share(103)).toBeGreaterThan(share(101));
    const sheet = (await engine.getCommunity(101, VIEWER))!.members.map((m) => m.did);
    expect(solved.follows.filter((f) => f.communityId === 101).map((f) => f.did)).toEqual(sheet.slice(0, share(101)));
  });

  test("allocateByWeight", () => {
    expect(allocateByWeight(25, [1, 2, 1])).toEqual([7, 12, 6]);
    expect(allocateByWeight(8, [1, 1, 1]).reduce((a, b) => a + b, 0)).toBe(8);
    expect(allocateByWeight(2, [1, 2, 1])).toEqual([1, 1, 0]);
    expect(allocateByWeight(0, [1, 1])).toEqual([0, 0]);
    expect(allocateByWeight(5, [])).toEqual([]);
  });

  test("the automatic total is capped, trimming boosted extras before anything else", async () => {
    const engine = engineWith({ config: { maxAutoTotal: 20 } });
    const solved = await engine.solveBundle({ selections: [pick(101, 2), pick(103, 2)] }, VIEWER);
    expect(solved.follows).toHaveLength(20);
    expect(solved.follows.filter((f) => f.communityId === 101)).toHaveLength(10);
    expect(solved.follows.filter((f) => f.communityId === 103)).toHaveLength(10);
  });

  test("a duel won by a scene the viewer did not pick adds three follows from it", async () => {
    const engine = engineWith();
    const solved = await engine.solveBundle({
      selections: [pick(101)],
      swipes: [{ swipeId: "s1", communityA: 103, communityB: 101, choice: "a" },
        { swipeId: "s2", communityA: 104, communityB: 102, choice: "neither" }],
    }, VIEWER);
    expect(solved.follows.filter((f) => f.communityId === 101)).toHaveLength(6);
    expect(solved.follows.filter((f) => f.communityId === 103)).toHaveLength(3);
    expect(solved.follows.filter((f) => [102, 104].includes(f.communityId))).toHaveLength(0);
  });

  test("nothing picked falls back to the catalog's first scenes", async () => {
    const solved = await engineWith().solveBundle({ selections: [] }, VIEWER);
    expect(solved.follows).toHaveLength(24);
  });
});

describe("live onboarding: pins, exclusions and alternates", () => {
  test("a pinned account is added on top of the scene's share; an excluded one is replaced by the next", async () => {
    const engine = engineWith();
    const sheet = (await engine.getCommunity(101, VIEWER))!.members.map((m) => m.did);
    const base = await engine.solveBundle({ selections: [pick(101)] }, VIEWER);
    expect(base.follows.map((f) => f.did)).toEqual(sheet.slice(0, 6));
    expect(base.bench.map((f) => f.did)).toEqual(sheet.slice(6, 8));

    const pinned = await engine.solveBundle({ selections: [pick(101)], pinnedDids: [sheet[20]!] }, VIEWER);
    expect(pinned.follows.map((f) => f.did)).toEqual([...sheet.slice(0, 6), sheet[20]!]);

    const excluded = await engine.solveBundle({ selections: [pick(101)], excludedDids: [sheet[1]!] }, VIEWER);
    expect(excluded.follows.map((f) => f.did)).toEqual([sheet[0]!, ...sheet.slice(2, 7)]);

    // Pinning someone from a scene that was not picked still follows them, under their own scene.
    const outsider = memberDid(104, 4);
    const outside = await engine.solveBundle({ selections: [pick(101)], pinnedDids: [outsider] }, VIEWER);
    expect(outside.follows.at(-1)).toMatchObject({ did: outsider, communityId: 104 });
  });

  test("pace changes how accounts are delivered, never who is followed", async () => {
    const engine = engineWith();
    const calm = await engine.solveBundle({
      selections: [pick(101, 2)], controls: { pace: "calm", originalsOnlyForHeavyReposters: true },
    }, VIEWER);
    const lively = await engine.solveBundle({
      selections: [pick(101, 2)], controls: { pace: "lively", originalsOnlyForHeavyReposters: false },
    }, VIEWER);
    expect(calm.follows.map((f) => f.did)).toEqual(lively.follows.map((f) => f.did));
    expect(lively.follows.every((f) => !f.hideReposts)).toBe(true);
    expect(calm.summary.estimatedDailyPosts).toBeLessThanOrEqual(lively.summary.estimatedDailyPosts);
  });
});

describe("live onboarding: rotation and variety", () => {
  test("one session keeps its order; another session draws a different one", async () => {
    const first = engineWith({ seed: 0.11 });
    const again = (await first.getCommunity(101, VIEWER))!.members.map((m) => m.did);
    expect((await first.getCommunity(101, VIEWER))!.members.map((m) => m.did)).toEqual(again);
    const second = engineWith({ seed: 0.77 });
    const other = (await second.getCommunity(101, VIEWER))!.members.map((m) => m.did);
    expect(other).not.toEqual(again);
    expect(new Set(other).size).toBe(other.length);
  });

  test("a session expires and the next one reshuffles", async () => {
    let t = 1_000_000;
    let draws = 0;
    const bundle = buildBundle();
    const engine = new LiveOnboardingEngine({
      catalog: new S17CatalogEngine(bundle),
      pool: buildPool(bundle),
      now: () => t,
      random: () => [0.2, 0.9][Math.min(draws++, 1)]!,
    });
    const before = (await engine.getCommunity(101, VIEWER))!.members.map((m) => m.did);
    t += 60 * 60_000;
    expect((await engine.getCommunity(101, VIEWER))!.members.map((m) => m.did)).toEqual(before);
    t += 91 * 60_000;
    expect((await engine.getCommunity(101, VIEWER))!.members.map((m) => m.did)).not.toEqual(before);
  });

  test("the first six mix kinds of account and hold at most two headliners", async () => {
    for (const seed of [0.05, 0.31, 0.58, 0.83]) {
      const engine = engineWith({ seed });
      const detail = (await engine.getCommunity(103, VIEWER))!;
      const six = detail.members.slice(0, 6);
      expect(new Set(six.map((m) => m.role)).size).toBeGreaterThanOrEqual(3);
      expect(six.filter((m) => m.tier === "headliner").length).toBeLessThanOrEqual(2);
      expect(detail.members.slice(0, 12).filter((m) => m.tier === "headliner").length).toBeLessThanOrEqual(4);
    }
  });

  test("the mix holds among the accounts a viewer does not follow yet", async () => {
    // Following every mid-sized account in the catalog's own 25 must not leave headliners in a row.
    const follows = Array.from({ length: 25 }, (_, i) => i)
      .filter((i) => i % 9 !== 0 && i % 4 !== 0)
      .map((i) => memberDid(103, i));
    for (const seed of [0.05, 0.31, 0.58, 0.83]) {
      const engine = engineWith({ seed, follows });
      const detail = (await engine.getCommunity(103, VIEWER))!;
      const six = detail.members.slice(0, 6);
      expect(six.some((m) => follows.includes(m.did))).toBe(false);
      expect(six.filter((m) => m.tier === "headliner").length).toBeLessThanOrEqual(2);
      expect(new Set(six.map((m) => m.role)).size).toBeGreaterThanOrEqual(3);
    }
  });

  test("across sessions the deep pool is used, and stronger accounts lead more often", async () => {
    const appearances = new Map<string, number>();
    for (let i = 0; i < 200; i++) {
      const engine = engineWith({ seed: (i + 0.5) / 200 });
      for (const m of (await engine.getCommunity(104, VIEWER))!.members.slice(0, 6)) {
        appearances.set(m.did, (appearances.get(m.did) ?? 0) + 1);
      }
    }
    expect(appearances.size).toBeGreaterThan(25); // more than the bundle's own 25
    const top = Array.from({ length: 10 }, (_, i) => appearances.get(memberDid(104, i)) ?? 0);
    const deep = Array.from({ length: 10 }, (_, i) => appearances.get(memberDid(104, 50 + i)) ?? 0);
    expect(top.reduce((a, b) => a + b, 0)).toBeGreaterThan(3 * deep.reduce((a, b) => a + b, 0));
  });

  test("accounts close to the viewer's existing follows move up", async () => {
    // Follow-space vectors: even members point one way, odd members the other.
    const vector = (did: string): Uint8Array | null => {
      const m = /m(\d{4})$/.exec(did);
      if (!m) return null;
      const out = new Uint8Array(256);
      out[Number(m[1]) % 2] = 100;
      return out;
    };
    const bundle = buildBundle();
    const follows = Array.from({ length: 12 }, (_, i) => memberDid(104, i * 2)); // even members of another scene
    const leadShare = async (personalised: boolean) => {
      let even = 0;
      for (let i = 0; i < 60; i++) {
        let draws = 0;
        const engine = new LiveOnboardingEngine({
          catalog: new S17CatalogEngine(bundle),
          pool: buildPool(bundle),
          vectors: personalised ? { rawVectorForDid: (did) => vector(did) } : null,
          followsFetcher: async (_did, onPage) => onPage(follows),
          random: () => (draws++ === 0 ? (i + 0.5) / 60 : 0.5),
        });
        for (const m of (await engine.getCommunity(103, VIEWER))!.members.slice(0, 12)) {
          if (Number(/m(\d{4})$/.exec(m.did)![1]) % 2 === 0) even += 1;
        }
      }
      return even / (60 * 12);
    };
    expect(await leadShare(true)).toBeGreaterThan((await leadShare(false)) + 0.08);
  });
});

describe("live onboarding: fallbacks and surfaces", () => {
  test("without a pool the catalog's own members serve, with the same guarantees", async () => {
    const engine = engineWith({ pool: false, follows: [memberDid(101, 0)] });
    const detail = (await engine.getCommunity(101, VIEWER))!;
    expect(detail.members.length).toBeLessThanOrEqual(25);
    const solved = await engine.solveBundle({ selections: [pick(101)] }, VIEWER);
    expect(solved.follows.map((f) => f.did)).toEqual(detail.members.slice(0, 6).map((m) => m.did));
  });

  test("a pool built for another catalog is left out, and the catalog's own members still serve", async () => {
    const bundle = buildBundle();
    const engine = new LiveOnboardingEngine({
      catalog: new S17CatalogEngine(bundle),
      pool: { ...buildPool(bundle), sourceCatalogVersion: "some_other_catalog" },
    });
    expect(engine.poolVersion).toBeNull();
    expect(engine.poolSkipped).toContain("some_other_catalog");
    expect(engine.describe().onboarding_pool_skipped).toContain("some_other_catalog");
    const detail = (await engine.getCommunity(103, VIEWER))!;
    expect(detail.members.every((m) => Number(m.did.slice(-4)) < 25)).toBe(true);
  });

  test("a pool account that opted out of signed-out viewing is hidden from them only", async () => {
    const bundle = buildBundle();
    const deepDid = memberDid(103, 30);
    const make = () => new LiveOnboardingEngine({
      catalog: new S17CatalogEngine(bundle),
      pool: { ...buildPool(bundle), noUnauthDids: [deepDid] },
      random: () => 0.4,
      config: { sheetDepth: 80 },
    });
    const signedIn = (await make().getCommunity(103, VIEWER))!.members.map((m) => m.did);
    const signedOut = (await make().getCommunity(103, ANON))!.members.map((m) => m.did);
    expect(signedIn).toContain(deepDid);
    expect(signedOut).not.toContain(deepDid);
  });

  test("a scene with fewer candidates than its share gives what it has", async () => {
    const follows = Array.from({ length: 22 }, (_, i) => memberDid(103, i));
    const engine = engineWith({ pool: false, follows });
    const solved = await engine.solveBundle({ selections: [pick(103, 2)] }, VIEWER);
    expect(solved.follows).toHaveLength(3);
    expect(solved.bench).toHaveLength(0);
  });

  test("the preview timeline shows posts by the people the lineup follows, mixing scenes", async () => {
    const engine = engineWith();
    const solved = await engine.solveBundle({ selections: [pick(101), pick(103)] }, VIEWER);
    const preview = await engine.previewTimeline({ selections: [pick(101), pick(103)], limit: 12 }, VIEWER);
    expect(preview.items).toHaveLength(12);
    const lineup = new Set(solved.follows.map((f) => f.did));
    for (const item of preview.items) expect(lineup.has(/^at:\/\/(did:[^/]+)\//.exec(item.post)![1]!)).toBe(true);
    expect(preview.items.slice(0, 2).map((i) => i.communityId)).toEqual([101, 103]);
  });

  test("feeds, starter packs and neighbours come from the covered scenes; hostile neighbours are left out", async () => {
    const solved = await engineWith().solveBundle({ selections: [pick(101), pick(103)] }, VIEWER);
    expect(solved.feeds.map((f) => f.communityId)).toEqual([101, 103]);
    expect(solved.starterPacks).toHaveLength(2);
    // 102 neighbours 101; 104 is hostile to 103 and so is not suggested.
    expect(solved.harmoniousNeighbors.map((c) => c.communityId)).toEqual([102]);
    expect(solved.harmoniousNeighbors[0]!.sampleAccountDids).toHaveLength(6);
  });

  test("the public follows fetcher pages until the cursor ends and tolerates malformed rows", async () => {
    const pages: Record<string, unknown> = {
      "": { follows: [{ did: "did:plc:a" }, { did: 7 }, null], cursor: "c2" },
      c2: { follows: [{ did: "did:plc:b" }, { did: "not-a-did" }] },
    };
    const seen: string[] = [];
    const fetcher = createPublicFollowsFetcher({
      fetcher: async (url) => Response.json(pages[url.searchParams.get("cursor") ?? ""]),
    });
    await fetcher("did:plc:viewer", (dids) => seen.push(...dids));
    expect(seen).toEqual(["did:plc:a", "did:plc:b"]);
  });
});

describe("live onboarding: generated duels", () => {
  test("a duel offers unpicked neighbours, and choosing a side follows the accounts whose posts it showed", async () => {
    const engine = engineWith();
    const { swipes } = await engine.getCalibrationSwipes({ selections: [pick(101)], maxSwipes: 4 }, VIEWER);
    expect(swipes.length).toBe(1);
    const duel = swipes[0]!;
    expect([duel.optionA.communityId, duel.optionB.communityId].sort()).toEqual([102, 103]);
    expect(duel.swipeId).toBe("live:102:103");

    const solved = await engine.solveBundle({
      selections: [pick(101)],
      swipes: [{ swipeId: duel.swipeId, communityA: duel.optionA.communityId, communityB: duel.optionB.communityId, choice: "a" }],
    }, VIEWER);
    const added = solved.follows.filter((f) => f.communityId === duel.optionA.communityId).map((f) => f.did);
    expect(added.length).toBe(3);
    const shownOwners = duel.optionA.samplePostUris.map((uri) => uri.slice(5).split("/")[0]);
    expect(shownOwners.every((did) => added.includes(did!))).toBe(true);
    expect(solved.follows.some((f) => f.communityId === duel.optionB.communityId)).toBe(false);
  });

  test("a scene hostile to a pick is never a side of its duels", async () => {
    const engine = engineWith();
    const { swipes } = await engine.getCalibrationSwipes({ selections: [pick(103)], maxSwipes: 4 }, VIEWER);
    expect(swipes.length).toBe(1);
    expect([swipes[0]!.optionA.communityId, swipes[0]!.optionB.communityId].sort()).toEqual([101, 102]);
  });

  test("a duel never shows a post by someone the viewer already follows", async () => {
    const follows = Array.from({ length: 60 }, (_, i) => memberDid(102, i)).filter((_, i) => i % 2 === 0);
    const engine = engineWith({ follows });
    const { swipes } = await engine.getCalibrationSwipes({ selections: [pick(101)], maxSwipes: 4 }, VIEWER);
    const side = [swipes[0]!.optionA, swipes[0]!.optionB].find((o) => o.communityId === 102)!;
    for (const uri of side.samplePostUris) expect(follows).not.toContain(uri.slice(5).split("/")[0]!);
  });
});

describe("live onboarding over HTTP", () => {
  const post = (path: string, body: unknown) => new Request(`http://intel.local/xrpc/${path}`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });

  test("routes use the live engine when one is installed and the static engine otherwise", async () => {
    const bundle = buildBundle();
    const catalog = new S17CatalogEngine(bundle);
    const live = new LiveOnboardingEngine({ catalog, pool: buildPool(bundle), random: () => 0.4 });
    const authenticate = async () => null;
    const liveHandler = createApiHandler({ authenticate, catalogEngine: catalog, liveOnboarding: live, vectorStore: null });
    const staticHandler = createApiHandler({ authenticate, catalogEngine: catalog, vectorStore: null });

    const health = await (await liveHandler(new Request("http://intel.local/health"))).json() as Record<string, unknown>;
    expect(health.onboarding_engine).toBe("live");
    expect(health.onboarding_pool_version).toBe("pool_v1_synthetic");
    const staticHealth = await (await staticHandler(new Request("http://intel.local/health"))).json() as Record<string, unknown>;
    expect(staticHealth.onboarding_engine).toBe("static");

    const detail = await (await liveHandler(
      new Request("http://intel.local/xrpc/blue.catbird.intel.getCommunity?communityId=103"))).json() as {
      members: { did: string }[]; defaultLineupCount: number };
    expect(detail.defaultLineupCount).toBe(6);
    expect(detail.members.length).toBe(40);

    const solved = await (await liveHandler(post("blue.catbird.onboarding.solveBundle", {
      selections: [{ communityId: 103, weight: 1 }, { communityId: 101, weight: 2 }],
    }))).json() as { follows: { did: string; communityId: number }[]; bench: unknown[] };
    expect(solved.follows).toHaveLength(18);
    expect(solved.follows.slice(0, 6).map((f) => f.did)).toEqual(detail.members.slice(0, 6).map((m) => m.did));
    expect(solved.bench.length).toBeGreaterThan(0);

    // Limits the live engine raises stay at their old values on the static engine.
    const manyPins = Array.from({ length: 60 }, (_, i) => memberDid(103, i));
    const body = { selections: [{ communityId: 103, weight: 1 }], pinnedDids: manyPins };
    expect((await liveHandler(post("blue.catbird.onboarding.solveBundle", body))).status).toBe(200);
    expect((await staticHandler(post("blue.catbird.onboarding.solveBundle", body))).status).toBe(400);

    const staticSolved = await (await staticHandler(post("blue.catbird.onboarding.solveBundle", {
      selections: [{ communityId: 103, weight: 1 }],
    }))).json() as { follows: unknown[]; bench?: unknown };
    expect(staticSolved.follows).toHaveLength(24);
    expect(staticSolved.bench).toBeUndefined();
  });
});
