import { describe, expect, test } from "bun:test";
import type { IntelCommunityEntry } from "../src/intel_catalog.ts";
import { generateDuels, type DuelSource } from "../src/onboarding_duels.ts";

/** A small catalog: scene groups of three communities, each linked to the next few by id. */
function catalog(options: {
  hostile?: [number, number][];
  offerable?: Record<number, number>;
  followed?: Record<number, number>;
  noPosts?: number[];
  seed?: number;
} = {}) {
  const communities = new Map<number, IntelCommunityEntry>();
  const ids = Array.from({ length: 18 }, (_, i) => 200 + i);
  for (const id of ids) {
    const sceneId = Math.floor((id - 200) / 3);
    communities.set(id, {
      communityId: id,
      title: `Community ${id}`,
      description: `People who post about topic ${id} every day, with long threads, shared links and plenty of pictures from their own work and travels`,
      sceneId,
      sceneSlug: `scene-${sceneId}`,
      sceneTitle: `Scene ${sceneId}`,
      macroCategory: "arts",
      symbolName: "sparkles",
      eligibleCount: 60,
      postCoherenceBps: 7000,
      socialCoherenceBps: 7000,
      curatedListTags: [],
      searchTerms: [],
      centPostsInt8B64: "",
      // Closest first: the two other members of the scene group (also its content-nearest), then
      // the next group's first two.
      neighborCommunityIds: ids.filter((other) => other !== id && Math.floor((other - 200) / 3) === sceneId)
        .concat([200 + ((sceneId + 1) % 6) * 3, 201 + ((sceneId + 1) % 6) * 3]),
      samplePostUris: [],
      previewPostUris: [],
      sampleAccountDids: [],
      feedGeneratorUris: [],
      starterPackUris: [],
      members: [],
    });
  }
  const hostile = new Set((options.hostile ?? []).map(([a, b]) => (a < b ? `${a}:${b}` : `${b}:${a}`)));
  const source: DuelSource = {
    community: (id) => communities.get(id) ?? null,
    nearestByContent: (id) => ids.filter((other) => other !== id && Math.floor((other - 200) / 3) === Math.floor((id - 200) / 3)),
    isHostilePair: (a, b) => hostile.has(a < b ? `${a}:${b}` : `${b}:${a}`),
    offerPosts: (id) => (options.noPosts?.includes(id) ? [] : [1, 2, 3].map((n) => `at://did:plc:c${id}m${n}/app.bsky.feed.post/p${n}`)),
    offerableCount: (id) => options.offerable?.[id] ?? 20,
    followedCount: (id) => options.followed?.[id] ?? 0,
    noise: (id) => {
      let h = (Math.imul(id, 0x9e3779b1) ^ Math.imul(options.seed ?? 1, 0x85ebca6b)) >>> 0;
      h ^= h >>> 15;
      h = Math.imul(h, 0x2c1b3c6d) >>> 0;
      h ^= h >>> 12;
      h = Math.imul(h, 0x297a2d39) >>> 0;
      h ^= h >>> 15;
      return ((h >>> 0) + 0.5) / 0x1_0000_0000;
    },
  };
  return source;
}

const sides = (duels: ReturnType<typeof generateDuels>) =>
  duels.flatMap((d) => [d.optionA.communityId, d.optionB.communityId]);

describe("generated taste duels", () => {
  test("every side is a scene the viewer did not pick, and no scene appears twice", () => {
    const picks = [200, 203, 209];
    const duels = generateDuels({ picks, maxSwipes: 4, source: catalog() });
    expect(duels.length).toBe(4);
    const all = sides(duels);
    expect(all.some((id) => picks.includes(id))).toBe(false);
    expect(new Set(all).size).toBe(all.length);
    expect(duels.map((d) => d.step)).toEqual([1, 2, 3, 4]);
    for (const duel of duels) {
      expect(duel.optionA.samplePostUris.length).toBeGreaterThan(0);
      expect(duel.optionB.samplePostUris.length).toBeGreaterThan(0);
      expect(duel.optionA.subtitle.length).toBeLessThanOrEqual(121);
    }
  });

  test("each picked scene gets a duel before any gets a second", () => {
    const duels = generateDuels({ picks: [200, 206, 212], maxSwipes: 3, source: catalog(), config: { rotation: 0 } });
    const parents = duels.map((d) => d.prompt);
    expect(new Set(parents).size).toBe(3);
  });

  test("two scenes alike in content are a finer split of the pick; neighbours from elsewhere are a bridge", () => {
    const source = catalog();
    const [first] = generateDuels({ picks: [200], maxSwipes: 1, source, config: { rotation: 0 } });
    expect(first!.kind).toBe("subscene_split");
    expect(first!.prompt).toBe("You picked Community 200. Which of its neighbours belongs on your timeline too?");
    expect(sides([first!]).sort()).toEqual([201, 202]);

    const [, second] = generateDuels({ picks: [200], maxSwipes: 2, source, config: { rotation: 0 } });
    expect(second!.kind).toBe("taste_bridge");
    expect(second!.prompt).toBe("You picked Community 200. Which of its neighbours belongs on your timeline too?");
    expect(sides([second!]).sort()).toEqual([203, 204]);
  });

  test("a scene hostile to any pick is never offered, and hostile scenes are not paired", () => {
    const duels = generateDuels({
      picks: [200, 209],
      maxSwipes: 6,
      source: catalog({ hostile: [[209, 201], [203, 204]] }),
    });
    expect(sides(duels)).not.toContain(201);
    for (const duel of duels) {
      expect([duel.optionA.communityId, duel.optionB.communityId].sort()).not.toEqual([203, 204]);
    }
  });

  test("a scene that cannot add enough accounts, or has no posts to show, is left out", () => {
    const duels = generateDuels({
      picks: [200],
      maxSwipes: 6,
      source: catalog({ offerable: { 201: 2 }, noPosts: [202] }),
    });
    expect(sides(duels)).not.toContain(201);
    expect(sides(duels)).not.toContain(202);
  });

  test("a scene the viewer already follows people in is offered sooner", () => {
    const plain = generateDuels({ picks: [200], maxSwipes: 1, source: catalog(), config: { rotation: 0 } });
    expect(sides(plain)).not.toContain(204);
    const leaning = generateDuels({
      picks: [200],
      maxSwipes: 1,
      source: catalog({ followed: { 204: 6, 203: 6 } }),
      config: { rotation: 0 },
    });
    // 201 and 202 score 1.8 and 1.03; six follows lift 203 (0.33 + 0.48) and 204 (0.25 + 0.48), but not past 201.
    expect(sides(leaning)).toContain(201);
  });

  test("different sessions draw different duels; one session repeats its own", () => {
    const picks = [200, 203, 206, 209];
    const first = generateDuels({ picks, maxSwipes: 4, source: catalog({ seed: 1 }) });
    const again = generateDuels({ picks, maxSwipes: 4, source: catalog({ seed: 1 }) });
    expect(again).toEqual(first);
    const drawn = new Set<string>();
    for (let seed = 1; seed <= 12; seed++) {
      for (const duel of generateDuels({ picks, maxSwipes: 4, source: catalog({ seed }) })) drawn.add(duel.swipeId);
    }
    // One session shows four duels; a dozen sessions between them show more than that.
    expect(drawn.size).toBeGreaterThan(4);
  });

  test("picks with too few neighbours give fewer duels, down to none", () => {
    const source = catalog({ offerable: Object.fromEntries(Array.from({ length: 18 }, (_, i) => [200 + i, 0])) });
    expect(generateDuels({ picks: [200], maxSwipes: 4, source })).toEqual([]);
    expect(generateDuels({ picks: [], maxSwipes: 4, source: catalog() })).toEqual([]);
  });

  test("the duel id is the same whichever side a scene lands on", () => {
    const [duel] = generateDuels({ picks: [200], maxSwipes: 1, source: catalog(), config: { rotation: 0 } });
    expect(duel!.swipeId).toBe("live:201:202");
  });
});
