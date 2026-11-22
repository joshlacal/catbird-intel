/**
 * A viewer's taste in follow-space: the unit-length centroid of the accounts they follow, and how
 * close a candidate account sits to it. Both read the int8 follow-space vectors and nothing else.
 */
import type { FollowSpaceVectors, LiveOnboardingConfig } from "./onboarding_live_types.ts";
import { signedByte } from "./onboarding_math.ts";

const VECTOR_DIMS = 256;

/** Unit-length mean of the follow-space vectors of the given accounts. */
export function tasteCentroid(
  vectors: FollowSpaceVectors | null,
  config: LiveOnboardingConfig,
  follows: Iterable<string>,
): Float32Array | null {
  if (!vectors) return null;
  const sum = new Float32Array(VECTOR_DIMS);
  let used = 0;
  for (const did of follows) {
    const raw = vectors.rawVectorForDid(did, "t1");
    if (!raw) continue;
    let normSq = 0;
    for (let i = 0; i < VECTOR_DIMS; i++) {
      const v = signedByte(raw[i]!);
      normSq += v * v;
    }
    if (normSq === 0) continue;
    const inv = 1 / Math.sqrt(normSq);
    for (let i = 0; i < VECTOR_DIMS; i++) sum[i] = sum[i]! + signedByte(raw[i]!) * inv;
    used += 1;
  }
  if (used < config.minFollowsForTaste) return null;
  let normSq = 0;
  for (let i = 0; i < VECTOR_DIMS; i++) normSq += sum[i]! * sum[i]!;
  if (normSq === 0) return null;
  const inv = 1 / Math.sqrt(normSq);
  for (let i = 0; i < VECTOR_DIMS; i++) sum[i] = sum[i]! * inv;
  return sum;
}

export function affinity(
  vectors: FollowSpaceVectors | null,
  did: string,
  centroid: Float32Array,
): number | null {
  const raw = vectors?.rawVectorForDid(did, "t1");
  if (!raw) return null;
  let dot = 0;
  let normSq = 0;
  for (let i = 0; i < VECTOR_DIMS; i++) {
    const v = signedByte(raw[i]!);
    dot += v * centroid[i]!;
    normSq += v * v;
  }
  return normSq === 0 ? null : dot / Math.sqrt(normSq);
}
