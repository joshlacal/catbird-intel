/** Uncalibrated legacy scores are optional filtering aids, never exposure budgets. */
export const SERVING_CONTRACT_VERSION = "catbird.intel.serving.v2";

export interface ServingControls {
  baseline_score_cutoff?: { politicalness?: number; heat?: number };
}

export class InvalidRequest extends Error {}

export function parseControls(value: unknown): ServingControls {
  if (value === undefined) return {};
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new InvalidRequest("controls must be an object");
  const object = value as Record<string, unknown>;
  if (Object.keys(object).some((key) => key !== "baseline_score_cutoff")) {
    throw new InvalidRequest("Only baseline_score_cutoff is supported; health filtering is off by default");
  }
  if (object.baseline_score_cutoff === undefined) return {};
  const cutoff = object.baseline_score_cutoff;
  if (!cutoff || typeof cutoff !== "object" || Array.isArray(cutoff)) throw new InvalidRequest("baseline_score_cutoff must be an object");
  const result: NonNullable<ServingControls["baseline_score_cutoff"]> = {};
  for (const [key, score] of Object.entries(cutoff)) {
    if (key !== "politicalness" && key !== "heat") throw new InvalidRequest("Unknown baseline score");
    if (typeof score !== "number" || !Number.isFinite(score) || score < 0 || score > 1) {
      throw new InvalidRequest("baseline score cutoffs must be finite numbers from 0 to 1");
    }
    result[key] = score;
  }
  return Object.keys(result).length ? { baseline_score_cutoff: result } : {};
}

/**
 * Legacy tables have no scorer/version columns and default missing scores to zero.
 * Preserve this uncertainty. Nonzero legacy values remain unversioned heuristics.
 */
export function healthEvidence(post: { politicalness?: number | null; heat?: number | null }) {
  const score = (n: number | null | undefined) =>
    typeof n === "number" && Number.isFinite(n) && n > 0 && n <= 1 ? n : null;
  const politicalness = score(post.politicalness);
  const heat = score(post.heat);
  return {
    source: politicalness === null && heat === null ? "unscored" : "legacy_unversioned_heuristic",
    calibrated: false,
    politicalness,
    heat,
  };
}

export function passesControls(post: { politicalness?: number | null; heat?: number | null }, controls: ServingControls): boolean {
  const evidence = healthEvidence(post);
  for (const key of ["politicalness", "heat"] as const) {
    const cutoff = controls.baseline_score_cutoff?.[key];
    if (cutoff !== undefined && evidence[key] !== null && evidence[key] > cutoff) return false;
  }
  return true; // Unknown scores never silently become low scores or a reason to exclude.
}

export function controlReport(posts: { politicalness?: number | null; heat?: number | null }[], controls: ServingControls) {
  const unscored = posts.filter((post) => {
    const evidence = healthEvidence(post);
    return evidence.politicalness === null || evidence.heat === null;
  }).length;
  return {
    controls,
    semantics: "optional_uncalibrated_baseline_score_cutoff",
    health_filtering: controls.baseline_score_cutoff ? "baseline_cutoff" : "off",
    unknown_policy: "include",
    items: posts.length,
    unscored_items: unscored,
    unscored_share: posts.length ? unscored / posts.length : null,
  };
}

export type RecommendationAction =
  | { type: "follow_account"; did: string; client_preferences: { scope: "catbird"; hideReposts: boolean }; applied: false; executable: false }
  | { type: "topic_feed_intent"; did: string; community: number; follow: false; applied: false; executable: false };

export function recommendationAction(did: string, community: number, mode: string): RecommendationAction {
  if (mode === "Topic feed only") return { type: "topic_feed_intent", did, community, follow: false, applied: false, executable: false };
  return {
    type: "follow_account",
    did,
    client_preferences: { scope: "catbird", hideReposts: mode === "Follow originals only" },
    applied: false,
    executable: false,
  };
}
