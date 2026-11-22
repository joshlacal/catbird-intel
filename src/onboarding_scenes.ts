/**
 * Which scenes a request draws from: the scenes most like a given one by content, the scenes a
 * viewer picked, and the lineup plan that sizes each. Reads the catalog and config only.
 */
import type {
  CalibrationSwipeAnswerInput,
  CommunitySelectionInput,
  S17CatalogEngine,
} from "./intel_catalog.ts";
import type { LiveOnboardingConfig } from "./onboarding_live_types.ts";
import { allocateByWeight, clamp } from "./onboarding_math.ts";

/** The scenes whose posts are most like this one's: closest post centroids first, weak matches left out. */
export function nearestByContent(
  postCentroids: ReadonlyMap<number, Int8Array>,
  nearestCache: Map<number, number[]>,
  config: LiveOnboardingConfig,
  communityId: number,
): number[] {
  const cached = nearestCache.get(communityId);
  if (cached) return cached;
  const own = postCentroids.get(communityId);
  const scored: { id: number; cosine: number }[] = [];
  if (own) {
    let ownSq = 0;
    for (let i = 0; i < own.length; i++) ownSq += own[i]! * own[i]!;
    for (const [id, other] of postCentroids) {
      if (id === communityId || other.length !== own.length) continue;
      let dot = 0;
      let otherSq = 0;
      for (let i = 0; i < own.length; i++) {
        dot += own[i]! * other[i]!;
        otherSq += other[i]! * other[i]!;
      }
      const cosine = ownSq > 0 && otherSq > 0 ? dot / Math.sqrt(ownSq * otherSq) : 0;
      if (cosine >= config.duelMinLikeness) scored.push({ id, cosine });
    }
  }
  const nearest = scored
    .sort((x, y) => y.cosine - x.cosine || x.id - y.id)
    .slice(0, config.duelContentNeighbours)
    .map((entry) => entry.id);
  nearestCache.set(communityId, nearest);
  return nearest;
}

/** The scenes a viewer picked, in order, or the catalog's first four when they picked none. */
export function pickedCommunityIds(catalog: S17CatalogEngine, selections: CommunitySelectionInput[]): number[] {
  const picks: number[] = [];
  for (const selection of selections) {
    if (!picks.includes(selection.communityId) && catalog.hasCommunity(selection.communityId)) {
      picks.push(selection.communityId);
    }
  }
  if (picks.length > 0) return picks;
  return catalog.bundle.communities.slice(0, 4).map((community) => community.communityId);
}

/**
 * The scenes a lineup draws from, in output order, with how many follows each contributes.
 * `targetSize` undefined sizes each scene on its own; a number is the older client contract
 * (one total), split across the picked scenes in proportion to their weight.
 */
export function planScenes(
  catalog: S17CatalogEngine,
  config: LiveOnboardingConfig,
  selections: CommunitySelectionInput[],
  swipes: CalibrationSwipeAnswerInput[],
  targetSize: number | undefined,
): { communityId: number; quota: number; picked: boolean }[] {
  const picked: { communityId: number; boosted: boolean }[] = [];
  const seen = new Set<number>();
  for (const selection of selections) {
    if (seen.has(selection.communityId) || !catalog.hasCommunity(selection.communityId)) continue;
    seen.add(selection.communityId);
    picked.push({ communityId: selection.communityId, boosted: selection.weight === 2 });
  }
  if (picked.length === 0) {
    // Nothing picked: the same four-scene fallback the static engine uses.
    for (const community of catalog.bundle.communities.slice(0, 4)) {
      seen.add(community.communityId);
      picked.push({ communityId: community.communityId, boosted: false });
    }
  }

  // A duel answer in favour of a scene the viewer did not pick adds a small share of it.
  const duelAdds: number[] = [];
  for (const swipe of swipes) {
    const chosen = swipe.choice === "a" ? [swipe.communityA] :
      swipe.choice === "b" ? [swipe.communityB] :
      swipe.choice === "both" ? [swipe.communityA, swipe.communityB] : [];
    for (const communityId of chosen) {
      if (seen.has(communityId) || !catalog.hasCommunity(communityId)) continue;
      seen.add(communityId);
      duelAdds.push(communityId);
    }
  }

  if (targetSize !== undefined) {
    const total = clamp(targetSize, 1, config.maxLegacyTotal);
    const weights = [...picked.map((p) => (p.boosted ? 2 : 1)), ...duelAdds.map(() => 0.5)];
    const quotas = allocateByWeight(total, weights);
    // A scene never gives more than its sheet shows, so every follow can be seen before it is made.
    const shown = (quota: number) => Math.min(quota, config.sheetDepth);
    return [
      ...picked.map((p, i) => ({ communityId: p.communityId, quota: shown(quotas[i]!), picked: true })),
      ...duelAdds.map((communityId, i) => ({
        communityId,
        quota: shown(quotas[picked.length + i]!),
        picked: false,
      })),
    ];
  }

  const plan = [
    ...picked.map((p) => ({
      communityId: p.communityId,
      quota: p.boosted ? config.boostedLineup : config.defaultLineup,
      picked: true,
    })),
    ...duelAdds.map((communityId) => ({ communityId, quota: config.duelLineup, picked: false })),
  ];
  let total = plan.reduce((acc, scene) => acc + scene.quota, 0);
  // Over the cap: duel additions give way first, then the extra half of boosted scenes.
  for (let i = plan.length - 1; i >= 0 && total > config.maxAutoTotal; i--) {
    if (plan[i]!.picked) continue;
    total -= plan[i]!.quota;
    plan[i]!.quota = 0;
  }
  while (total > config.maxAutoTotal) {
    const largest = plan.reduce((best, scene) => (scene.quota > best.quota ? scene : best), plan[0]!);
    if (largest.quota <= 1) break;
    largest.quota -= 1;
    total -= 1;
  }
  return plan;
}
