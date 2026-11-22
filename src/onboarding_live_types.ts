/**
 * Types and tunables for the live onboarding engine: the pool, vector, follow and signal
 * interfaces, the per-viewer session shape, the config type and its defaults.
 * The engine itself is in onboarding_live.ts, which re-exports every public name from here.
 */
import type { IntelMemberEntry } from "./intel_catalog.ts";

export type MemberTier = "headliner" | "core" | "rising";

/** Deep reviewed candidate pool written by `graph/s16_s17/build_intel_serving_v2.py --pool-out`. */
export interface OnboardingPoolFile {
  poolVersion: string;
  generatedAt: string;
  sourceCatalogVersion: string;
  depth: number;
  /** Pool accounts that asked not to be shown to signed-out viewers. */
  noUnauthDids?: string[];
  communities: { communityId: number; members: IntelMemberEntry[] }[];
}

/** The slice of the vector store this engine needs: a producer's follow-space vector as raw int8 bytes. */
export interface FollowSpaceVectors {
  rawVectorForDid(did: string, window: "t1"): Uint8Array | null;
}

/** Loads a viewer's follow list page by page. Rejecting leaves a partial list in place and is retried soon. */
export type FollowsFetcher = (
  viewerDid: string,
  /** Called with each page, newest follows first. Returning true stops the load early. */
  onPage: (dids: string[]) => boolean | void,
) => Promise<void>;

/** Live activity signals. A multiplier of 0 removes the account from new lists; null means "no data". */
export interface LiveAccountSignals {
  accountMultiplier(did: string): number | null;
  samplePostFor(did: string): string | null;
  version(): string | null;
  /** Extra fields for the health report. */
  describe?(): Record<string, unknown>;
}

export interface ViewerRef {
  did: string | null;
  unauthenticated: boolean;
}

export interface LiveOnboardingConfig {
  cardFaces: number;
  sheetDepth: number;
  defaultLineup: number;
  boostedLineup: number;
  /** Follows taken from a scene the viewer did not pick but chose in a taste duel. */
  duelLineup: number;
  /** How many content-nearest scenes a pick contributes to its duels. */
  duelContentNeighbours: number;
  /** A scene must be at least this close in content (cosine of post centroids) to count as alike. */
  duelMinLikeness: number;
  benchPerScene: number;
  maxAutoTotal: number;
  maxLegacyTotal: number;
  sessionTtlMs: number;
  maxSessions: number;
  /** How long the first request of a session waits for the follow list before answering with what it has. */
  followsBudgetMs: number;
  followsRefreshMs: number;
  /** How long to wait after a failed follow-list load before trying again. */
  followsRetryMs: number;
  /** Follow-list loads running at once across all viewers. */
  maxFollowLoads: number;
  /** Offerable accounts remembered as followed, per viewer. */
  maxKeptFollows: number;
  /** Scene orders held per viewer; the least recently used is dropped beyond this. */
  maxListsPerSession: number;
  /**
   * An account's draw weight is 1 / (rank + rotationOffset) ^ rotationSharpness, where rank is its
   * position in the picker's order. A higher exponent keeps the strongest accounts first more
   * often; a higher offset treats the top few as near equals. At 1.5 and 3 the top-ranked account
   * is in the first six in about three sessions of four, and about a quarter of first-six places
   * go to accounts ranked below 25.
   */
  rotationSharpness: number;
  rotationOffset: number;
  /** Strength of the "close to who you already follow" boost, per standard deviation. */
  personalBeta: number;
  minFollowsForTaste: number;
  headlinerFollowers: number;
  risingFollowers: number;
  megaFollowers: number;
  maxHeadlinersPerSix: number;
  /** A slot's wanted kind is honoured only if such an account is this close to the front of what is left. */
  slateWindow: number;
}

export const LIVE_ONBOARDING_DEFAULTS: LiveOnboardingConfig = {
  cardFaces: 6,
  sheetDepth: 40,
  defaultLineup: 6,
  boostedLineup: 12,
  duelLineup: 3,
  duelContentNeighbours: 10,
  duelMinLikeness: 0.5,
  benchPerScene: 2,
  maxAutoTotal: 180,
  maxLegacyTotal: 200,
  sessionTtlMs: 90 * 60_000,
  maxSessions: 4000,
  followsBudgetMs: 1200,
  followsRefreshMs: 10 * 60_000,
  followsRetryMs: 30_000,
  maxFollowLoads: 8,
  maxKeptFollows: 10_000,
  maxListsPerSession: 600,
  rotationSharpness: 1.5,
  rotationOffset: 3,
  personalBeta: 0.7,
  minFollowsForTaste: 8,
  headlinerFollowers: 20_000,
  risingFollowers: 1_500,
  megaFollowers: 500_000,
  maxHeadlinersPerSix: 2,
  slateWindow: 20,
};

export interface OrderedMember {
  member: IntelMemberEntry;
  tier: MemberTier;
  key: number;
}

export interface Session {
  seed: number;
  lastSeenAt: number;
  /** The accounts the viewer follows that this engine could offer. Their other follows are not kept. */
  follows: Set<string>;
  followsLoad: Promise<void> | null;
  followsLoading: boolean;
  /** When the follow list was last loaded to the end; null until that has happened once. */
  followsLoadedAt: number | null;
  /** The earliest time another load may start. */
  followsNextAt: number;
  /** The viewer's most recent follow at the last complete load. A refresh stops when it meets it. */
  newestFollow: string | null;
  /**
   * The viewer's taste, from the first page of their follows. undefined until that page arrives
   * or the first list is built without it; null when there is too little to infer taste from.
   * Set once and never changed, so every scene in a session is ordered by the same taste.
   */
  centroid: Float32Array | null | undefined;
  /** Per scene, the viewer's order as positions in that scene's member list: a few hundred bytes each. */
  lists: Map<number, Uint16Array>;
}
