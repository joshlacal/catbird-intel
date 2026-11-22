/**
 * The weighted draw behind every scene order: each member gets a weight from its rank, any live
 * signal and the viewer's taste, then a seeded Gumbel draw sorts them. Pure function of its inputs.
 */
import type { IntelMemberEntry } from "./intel_catalog.ts";
import type {
  FollowSpaceVectors,
  LiveAccountSignals,
  LiveOnboardingConfig,
  OrderedMember,
} from "./onboarding_live_types.ts";
import { clamp, seededHash } from "./onboarding_math.ts";
import { affinity } from "./onboarding_taste.ts";
import { tierOf } from "./onboarding_slate.ts";

/** A scene's members in the viewer's seeded order, before the slate arranges the front of the list. */
export function scoreMembers(
  members: IntelMemberEntry[],
  centroid: Float32Array | null,
  seed: number,
  vectors: FollowSpaceVectors | null,
  config: LiveOnboardingConfig,
  liveSignals: LiveAccountSignals | null,
): OrderedMember[] {
  // Closeness to the viewer's taste is compared within the scene, so it reorders a scene's
  // members without favouring scenes that happen to sit near the viewer's existing follows.
  const affinities = centroid ? members.map((m) => affinity(vectors, m.did, centroid)) : null;
  let mean = 0;
  let deviation = 0;
  if (affinities) {
    const known = affinities.filter((a): a is number => a !== null);
    if (known.length >= 4) {
      mean = known.reduce((acc, a) => acc + a, 0) / known.length;
      deviation = Math.sqrt(known.reduce((acc, a) => acc + (a - mean) * (a - mean), 0) / known.length);
    }
  }

  // Rank by the picker's score rather than using the score itself: scores fall almost linearly
  // through a scene, so raw scores would make the draw close to uniform.
  const rankOf = new Map<string, number>();
  [...members]
    .sort((x, y) => y.rankScoreBps - x.rankScoreBps)
    .forEach((member, rank) => rankOf.set(member.did, rank));

  const scored: OrderedMember[] = [];
  members.forEach((member, index) => {
    const live = liveSignals?.accountMultiplier(member.did) ?? null;
    if (live !== null && live <= 0) return;
    let weight = 1 / Math.pow(rankOf.get(member.did)! + config.rotationOffset, config.rotationSharpness);
    if (live !== null) weight *= live;
    const a = affinities?.[index] ?? null;
    if (a !== null && deviation > 1e-6) {
      weight *= Math.exp(config.personalBeta * clamp((a - mean) / deviation, -2, 2));
    }
    // Gumbel-max: sorting by log(weight) + Gumbel noise draws an order in which an account's
    // chance of coming first is proportional to its weight.
    const u = (seededHash(member.did, seed) + 0.5) / 0x1_0000_0000;
    scored.push({ member, tier: tierOf(member, config), key: Math.log(weight) - Math.log(-Math.log(u)) });
  });
  scored.sort((x, y) => y.key - x.key || (x.member.did < y.member.did ? -1 : 1));
  return scored;
}
