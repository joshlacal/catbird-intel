/**
 * Live onboarding engine.
 *
 * The static S17 engine solves one global lineup of a fixed size, so the accounts a scene card
 * shows and the accounts the lineup follows are chosen independently. This engine keeps ONE
 * ordered candidate list per viewer and scene, and every surface reads a prefix of it:
 *
 *   scene card      first `cardFaces`
 *   scene detail    first `sheetDepth`
 *   lineup          first `defaultLineup` per picked scene (`boostedLineup` when boosted)
 *
 * A longer prefix contains a shorter one, so an account shown under a picked scene is followed.
 * Accounts the viewer already follows are filtered out when a list is read, never when it is
 * built, so a follow list that finishes loading mid-flow only shifts a list up; it never
 * reshuffles it.
 *
 * The order is a seeded weighted draw: stable for one session, different for the next.
 * Design: docs/design-docs/2026-11-22-live-onboarding.md (workspace root repository).
 */
import { readFileSync } from "node:fs";
import type {
  BundleControlsInput,
  CalibrationSwipeAnswerInput,
  CommunityRole,
  CommunitySelectionInput,
  IntelMemberEntry,
  S17CatalogEngine,
} from "./intel_catalog.ts";
import { generateDuels, type DuelSource } from "./onboarding_duels.ts";

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

/** Loads a viewer's follow list page by page. Rejecting or returning early leaves a partial list in place. */
export type FollowsFetcher = (viewerDid: string, onPage: (dids: string[]) => void) => Promise<void>;

/** Live activity signals. A multiplier of 0 removes the account from new lists; null means "no data". */
export interface LiveAccountSignals {
  accountMultiplier(did: string): number | null;
  samplePostFor(did: string): string | null;
  version(): string | null;
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
  rotationSharpness: 1.5,
  rotationOffset: 3,
  personalBeta: 0.35,
  minFollowsForTaste: 8,
  headlinerFollowers: 20_000,
  risingFollowers: 1_500,
  megaFollowers: 500_000,
  maxHeadlinersPerSix: 2,
  slateWindow: 20,
};

interface OrderedMember {
  member: IntelMemberEntry;
  tier: MemberTier;
  key: number;
}

interface Session {
  seed: number;
  lastSeenAt: number;
  follows: Set<string>;
  followsLoad: Promise<void> | null;
  followsLoading: boolean;
  followsLoadedAt: number | null;
  /** undefined until the first list is built; null when the viewer has too few follows to infer taste. */
  centroid: Float32Array | null | undefined;
  /** Per scene, the viewer's order as positions in that scene's member list: a few hundred bytes each. */
  lists: Map<number, Uint16Array>;
}

interface SlateSlot {
  roles?: readonly CommunityRole[];
  tier?: MemberTier;
}

/** The kinds of account the first six positions aim for, repeated for positions seven to twelve. */
const SLATE: readonly SlateSlot[] = [
  { roles: ["anchor_creator"] },
  { roles: ["visual_maker"] },
  { roles: ["conversationalist"] },
  { tier: "rising" },
  { roles: ["anchor_creator", "project_or_studio"] },
  { roles: ["scene_curator"] },
];
const SLATE_POSITIONS = SLATE.length * 2;
const VECTOR_DIMS = 256;

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/** FNV-1a over the string, mixed with the seed and finished with a murmur3 avalanche. */
function seededHash(text: string, seed: number): number {
  let h = (0x811c9dc5 ^ seed) >>> 0;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}

function signedByte(value: number): number {
  return value > 127 ? value - 256 : value;
}

/** Splits `total` across `weights` by largest remainder; every entry gets at least one while the total allows. */
export function allocateByWeight(total: number, weights: readonly number[]): number[] {
  const n = weights.length;
  const out = new Array<number>(n).fill(0);
  if (n === 0 || total <= 0) return out;
  const order = weights.map((w, i) => ({ w, i })).sort((a, b) => b.w - a.w || a.i - b.i);
  if (total <= n) {
    for (const { i } of order.slice(0, total)) out[i] = 1;
    return out;
  }
  const spare = total - n;
  const sum = weights.reduce((acc, w) => acc + w, 0);
  const shares = weights.map((w) => (sum > 0 ? (spare * w) / sum : spare / n));
  let given = 0;
  shares.forEach((share, i) => {
    out[i] = 1 + Math.floor(share);
    given += out[i]!;
  });
  const byRemainder = shares
    .map((share, i) => ({ r: share - Math.floor(share), i }))
    .sort((a, b) => b.r - a.r || a.i - b.i);
  for (let k = 0; given < total; k++, given++) out[byRemainder[k % n]!.i]! += 1;
  return out;
}

export class LiveOnboardingEngine {
  readonly config: LiveOnboardingConfig;
  readonly poolVersion: string | null;
  private readonly catalog: S17CatalogEngine;
  private readonly vectors: FollowSpaceVectors | null;
  private readonly followsFetcher: FollowsFetcher | null;
  private readonly liveSignals: LiveAccountSignals | null;
  private readonly now: () => number;
  private readonly random: () => number;
  private readonly orderSalt: number;
  private readonly membersByCommunity = new Map<number, IntelMemberEntry[]>();
  private readonly memberHome = new Map<string, { communityId: number; member: IntelMemberEntry }>();
  private readonly postCentroids = new Map<number, Int8Array>();
  private readonly nearestCache = new Map<number, number[]>();
  private readonly sessions = new Map<string, Session>();
  private readonly poolNoUnauth = new Set<string>();
  /** Why a supplied pool was not used, if it was not. */
  readonly poolSkipped: string | null;
  private poolMemberCount = 0;

  constructor(options: {
    catalog: S17CatalogEngine;
    pool?: OnboardingPoolFile | null;
    vectors?: FollowSpaceVectors | null;
    followsFetcher?: FollowsFetcher | null;
    liveSignals?: LiveAccountSignals | null;
    config?: Partial<LiveOnboardingConfig>;
    now?: () => number;
    random?: () => number;
    /** Mixed into every signed-in viewer's order. Instances that should agree must share it. */
    orderSalt?: number;
  }) {
    this.catalog = options.catalog;
    this.vectors = options.vectors ?? null;
    this.followsFetcher = options.followsFetcher ?? null;
    this.liveSignals = options.liveSignals ?? null;
    this.config = { ...LIVE_ONBOARDING_DEFAULTS, ...(options.config ?? {}) };
    this.now = options.now ?? Date.now;
    this.random = options.random ?? Math.random;
    this.orderSalt = (options.orderSalt ?? 0) >>> 0;
    // A pool is screened against one catalog build. Behind any other catalog its accounts have not
    // been through that catalog's exclusions, so it is left out rather than trusted.
    const supplied = options.pool ?? null;
    const pool = supplied && supplied.sourceCatalogVersion === this.catalog.bundle.catalogVersion ? supplied : null;
    this.poolSkipped = supplied && !pool
      ? `pool ${supplied.poolVersion} was built for catalog ${supplied.sourceCatalogVersion}, not ${this.catalog.bundle.catalogVersion}`
      : null;
    this.poolVersion = pool?.poolVersion ?? null;
    for (const did of pool?.noUnauthDids ?? []) this.poolNoUnauth.add(did);

    const poolByCommunity = new Map<number, IntelMemberEntry[]>();
    for (const entry of pool?.communities ?? []) poolByCommunity.set(entry.communityId, entry.members);

    for (const community of this.catalog.bundle.communities) {
      // The bundle's own members stay first and verbatim; the pool only adds depth behind them.
      const merged: IntelMemberEntry[] = [];
      const seen = new Set<string>();
      for (const member of [...community.members, ...(poolByCommunity.get(community.communityId) ?? [])]) {
        if (seen.has(member.did) || !this.catalog.isAccountAllowed(member.did, false)) continue;
        seen.add(member.did);
        merged.push(member);
      }
      this.membersByCommunity.set(community.communityId, merged);
      const centroid = Buffer.from(community.centPostsInt8B64, "base64");
      if (centroid.byteLength > 0) {
        this.postCentroids.set(community.communityId, new Int8Array(centroid.buffer, centroid.byteOffset, centroid.byteLength));
      }
      this.poolMemberCount += merged.length;
      for (const member of merged) {
        if (!this.memberHome.has(member.did)) this.memberHome.set(member.did, { communityId: community.communityId, member });
      }
    }
  }

  static loadPoolFile(path: string): OnboardingPoolFile {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as OnboardingPoolFile;
    if (!parsed || !Array.isArray(parsed.communities) || typeof parsed.poolVersion !== "string") {
      throw new Error("Onboarding pool file is not an intel_onboarding_pool_v1 document");
    }
    return parsed;
  }

  describe(): Record<string, unknown> {
    return {
      onboarding_engine: "live",
      onboarding_pool_version: this.poolVersion,
      onboarding_pool_members: this.poolMemberCount,
      onboarding_pool_skipped: this.poolSkipped,
      onboarding_live_signals: this.liveSignals?.version() ?? null,
      onboarding_personalised: this.vectors !== null,
      onboarding_sessions: this.sessions.size,
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Sessions and follow lists
  // ---------------------------------------------------------------------------------------------

  private async session(viewer: ViewerRef, seedFollowDids: readonly string[] = []): Promise<Session> {
    const key = viewer.did ?? "anonymous";
    const t = this.now();
    let session = this.sessions.get(key);
    if (session && t - session.lastSeenAt <= this.config.sessionTtlMs) {
      this.sessions.delete(key); // re-insert so eviction drops the least recently used session
    } else {
      session = {
        seed: this.seedFor(viewer, t),
        lastSeenAt: t,
        follows: new Set<string>(),
        followsLoad: null,
        followsLoading: false,
        followsLoadedAt: null,
        centroid: undefined,
        lists: new Map(),
      };
    }
    this.sessions.set(key, session);
    while (this.sessions.size > this.config.maxSessions) {
      const oldest = this.sessions.keys().next().value;
      if (oldest === undefined) break;
      this.sessions.delete(oldest);
    }
    session.lastSeenAt = t;
    for (const did of seedFollowDids) session.follows.add(did);

    if (viewer.did && this.followsFetcher) {
      const firstLoad = session.followsLoad === null;
      const stale = session.followsLoadedAt !== null && t - session.followsLoadedAt > this.config.followsRefreshMs;
      if ((firstLoad || stale) && !session.followsLoading) this.startFollowsLoad(session, viewer.did);
      // Only the first load is worth waiting for; a refresh adds the few accounts followed since.
      if (session.followsLoading && session.followsLoadedAt === null && session.followsLoad) {
        await Promise.race([
          session.followsLoad,
          new Promise<void>((resolve) => setTimeout(resolve, this.config.followsBudgetMs)),
        ]);
      }
    }
    return session;
  }

  /**
   * A signed-in viewer's order comes from who they are and which stretch of time it is, not from a
   * random draw, so a service restart or a second instance in the middle of a flow arrives at the
   * same order. A stretch is one session lifetime long and starts at a different moment for each
   * viewer, so an expired session always falls in a later stretch and reshuffles.
   */
  private seedFor(viewer: ViewerRef, t: number): number {
    if (!viewer.did) return Math.floor(this.random() * 0x1_0000_0000) >>> 0;
    const stretch = this.config.sessionTtlMs;
    const offset = seededHash(viewer.did, 0x5eed) % stretch;
    return seededHash(`${viewer.did}:${Math.floor((t + offset) / stretch)}`, this.orderSalt);
  }

  private startFollowsLoad(session: Session, viewerDid: string): void {
    session.followsLoading = true;
    session.followsLoad = this.followsFetcher!(viewerDid, (dids) => {
      for (const did of dids) session.follows.add(did);
    })
      .catch(() => undefined) // a partial follow list is still better than none
      .then(() => {
        session.followsLoading = false;
        session.followsLoadedAt = this.now();
      });
  }

  // ---------------------------------------------------------------------------------------------
  // Ordered lists
  // ---------------------------------------------------------------------------------------------

  private tierOf(member: IntelMemberEntry): MemberTier {
    if (member.followers >= this.config.headlinerFollowers) return "headliner";
    if (member.followers < this.config.risingFollowers) return "rising";
    return "core";
  }

  /** Unit-length mean of the follow-space vectors of the accounts the viewer follows. */
  private tasteCentroid(follows: ReadonlySet<string>): Float32Array | null {
    if (!this.vectors) return null;
    const sum = new Float32Array(VECTOR_DIMS);
    let used = 0;
    for (const did of follows) {
      const raw = this.vectors.rawVectorForDid(did, "t1");
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
    if (used < this.config.minFollowsForTaste) return null;
    let normSq = 0;
    for (let i = 0; i < VECTOR_DIMS; i++) normSq += sum[i]! * sum[i]!;
    if (normSq === 0) return null;
    const inv = 1 / Math.sqrt(normSq);
    for (let i = 0; i < VECTOR_DIMS; i++) sum[i] = sum[i]! * inv;
    return sum;
  }

  private affinity(did: string, centroid: Float32Array): number | null {
    const raw = this.vectors?.rawVectorForDid(did, "t1");
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

  private orderedList(session: Session, communityId: number, viewer: ViewerRef): OrderedMember[] {
    const members = this.membersByCommunity.get(communityId) ?? [];
    const cached = session.lists.get(communityId);
    if (cached) {
      const restored: OrderedMember[] = new Array(cached.length);
      for (let i = 0; i < cached.length; i++) {
        const member = members[cached[i]!]!;
        restored[i] = { member, tier: this.tierOf(member), key: 0 };
      }
      return restored;
    }
    if (session.centroid === undefined) session.centroid = this.tasteCentroid(session.follows);
    const centroid = session.centroid;

    // Closeness to the viewer's taste is compared within the scene, so it reorders a scene's
    // members without favouring scenes that happen to sit near the viewer's existing follows.
    const affinities = centroid ? members.map((m) => this.affinity(m.did, centroid)) : null;
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
      const live = this.liveSignals?.accountMultiplier(member.did) ?? null;
      if (live !== null && live <= 0) return;
      let weight = 1 / Math.pow(rankOf.get(member.did)! + this.config.rotationOffset, this.config.rotationSharpness);
      if (live !== null) weight *= live;
      const a = affinities?.[index] ?? null;
      if (a !== null && deviation > 1e-6) {
        weight *= Math.exp(this.config.personalBeta * clamp((a - mean) / deviation, -2, 2));
      }
      // Gumbel-max: sorting by log(weight) + Gumbel noise draws an order in which an account's
      // chance of coming first is proportional to its weight.
      const u = (seededHash(member.did, session.seed) + 0.5) / 0x1_0000_0000;
      scored.push({ member, tier: this.tierOf(member), key: Math.log(weight) - Math.log(-Math.log(u)) });
    });
    scored.sort((x, y) => y.key - x.key || (x.member.did < y.member.did ? -1 : 1));

    // The mix is arranged among the accounts this viewer can still be offered. Accounts already
    // known to be followed go to the back (they are filtered whenever the list is read), so a
    // viewer who follows a scene's mid-sized accounts is not left with its largest ones in a row.
    const open: OrderedMember[] = [];
    const closed: OrderedMember[] = [];
    for (const candidate of scored) {
      const offered = !session.follows.has(candidate.member.did) &&
        this.allowed(candidate.member.did, viewer);
      (offered ? open : closed).push(candidate);
    }
    const ordered = this.applySlate(open).concat(closed);
    const position = new Map<IntelMemberEntry, number>();
    members.forEach((member, index) => position.set(member, index));
    session.lists.set(communityId, Uint16Array.from(ordered, (entry) => position.get(entry.member)!));
    return ordered;
  }

  /** Gives the first twelve positions a mix of account kinds and at most a few very large accounts. */
  private applySlate(byKey: OrderedMember[]): OrderedMember[] {
    const remaining = [...byKey];
    const ordered: OrderedMember[] = [];
    let headliners = 0;
    let megas = 0;
    const withinCaps = (c: OrderedMember, position: number): boolean => {
      const headlinerCap = this.config.maxHeadlinersPerSix * (Math.floor(position / SLATE.length) + 1);
      if (c.tier === "headliner" && headliners >= headlinerCap) return false;
      if (c.member.followers > this.config.megaFollowers && megas >= 1) return false;
      return true;
    };
    for (let position = 0; position < SLATE_POSITIONS && remaining.length > 0; position++) {
      const slot = SLATE[position % SLATE.length]!;
      const wanted = (c: OrderedMember): boolean =>
        (slot.tier ? c.tier === slot.tier : true) && (slot.roles ? slot.roles.includes(c.member.role) : true);
      let pick = -1;
      const window = Math.min(remaining.length, this.config.slateWindow);
      for (let i = 0; i < window; i++) {
        if (wanted(remaining[i]!) && withinCaps(remaining[i]!, position)) {
          pick = i;
          break;
        }
      }
      if (pick < 0) pick = remaining.findIndex((c) => withinCaps(c, position));
      if (pick < 0) pick = 0;
      const [chosen] = remaining.splice(pick, 1);
      if (chosen!.tier === "headliner") headliners += 1;
      if (chosen!.member.followers > this.config.megaFollowers) megas += 1;
      ordered.push(chosen!);
    }
    return ordered.concat(remaining);
  }

  /** May this viewer be shown this account, or a post by it? Covers pool accounts the catalog does not list. */
  private allowed(uriOrDid: string, viewer: ViewerRef): boolean {
    if (!this.catalog.isAccountAllowed(uriOrDid, viewer.unauthenticated)) return false;
    if (!viewer.unauthenticated || this.poolNoUnauth.size === 0) return true;
    const owner = uriOrDid.startsWith("at://") ? uriOrDid.slice(5).split("/", 1)[0]! : uriOrDid;
    return !this.poolNoUnauth.has(owner);
  }

  /** A scene's list as this viewer may see it now: no one they follow, hid, or may not be shown. */
  private visible(
    session: Session,
    communityId: number,
    viewer: ViewerRef,
    alsoExcluded?: ReadonlySet<string>,
  ): OrderedMember[] {
    return this.orderedList(session, communityId, viewer).filter(({ member }) =>
      !session.follows.has(member.did) &&
      !(alsoExcluded?.has(member.did) ?? false) &&
      this.allowed(member.did, viewer));
  }

  private followedCount(session: Session, communityId: number): number {
    let count = 0;
    for (const member of this.membersByCommunity.get(communityId) ?? []) {
      if (session.follows.has(member.did)) count += 1;
    }
    return count;
  }

  private samplePostOf(member: IntelMemberEntry, viewer: ViewerRef): string | undefined {
    const uri = this.liveSignals?.samplePostFor(member.did) ?? member.samplePostUri;
    return uri && this.allowed(uri, viewer) ? uri : undefined;
  }

  private decorate<T extends { communityId: number }>(summary: T, session: Session, viewer: ViewerRef) {
    return {
      ...summary,
      sampleAccountDids: this.visible(session, summary.communityId, viewer)
        .slice(0, this.config.cardFaces)
        .map((o) => o.member.did),
      alreadyFollowingCount: this.followedCount(session, summary.communityId),
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Catalog surfaces
  // ---------------------------------------------------------------------------------------------

  async getCatalog(params: {
    seedFollowDids?: string[];
    q?: string | null;
    macroCategory?: string | null;
    limit?: number;
  }, viewer: ViewerRef) {
    const session = await this.session(viewer, params.seedFollowDids);
    // The static engine warms the "based on who you follow" row from the 100 follows the client
    // sends; the whole follow list gives it far more to go on.
    const base = this.catalog.getCatalog({
      ...params,
      seedFollowDids: [...session.follows],
      unauthenticated: viewer.unauthenticated,
    });
    return {
      ...base,
      featuredCommunities: base.featuredCommunities.map((c) => this.decorate(c, session, viewer)),
      warmStartCommunities: base.warmStartCommunities.map((c) => this.decorate(c, session, viewer)),
    };
  }

  async listCommunities(params: {
    q?: string | null;
    macroCategory?: string | null;
    sceneSlug?: string | null;
    limit?: number;
    cursor?: string | null;
  }, viewer: ViewerRef) {
    const session = await this.session(viewer);
    const base = this.catalog.listCommunities({ ...params, unauthenticated: viewer.unauthenticated });
    return { ...base, communities: base.communities.map((c) => this.decorate(c, session, viewer)) };
  }

  async getCommunity(communityId: number, viewer: ViewerRef) {
    const base = this.catalog.getCommunity(communityId, viewer.unauthenticated);
    if (!base) return null;
    const session = await this.session(viewer);
    const ordered = this.visible(session, communityId, viewer).slice(0, this.config.sheetDepth);

    const members = ordered.map(({ member, tier }) => {
      const samplePostUri = this.samplePostOf(member, viewer);
      return {
        did: member.did,
        role: member.role,
        recommendedMode: member.recommendedMode,
        whyPicked: member.whyPicked,
        rankScoreBps: member.rankScoreBps,
        repostShareBps: member.repostShareBps,
        mediaShareBps: member.mediaShareBps,
        estimatedPostsPerWeek: member.estimatedPostsPerWeek,
        tier,
        ...(samplePostUri ? { samplePostUri } : {}),
      };
    });

    // Posts by the people the viewer would follow come first, then the scene's own samples.
    const samplePosts: typeof base.samplePosts = [];
    const seenPosts = new Set<string>();
    for (const { member } of ordered.slice(0, this.config.boostedLineup)) {
      const uri = this.samplePostOf(member, viewer);
      if (!uri || seenPosts.has(uri)) continue;
      seenPosts.add(uri);
      samplePosts.push({
        post: uri,
        communityId,
        communityTitle: base.community.title,
        authorRole: member.role,
      });
    }
    for (const post of base.samplePosts) {
      if (samplePosts.length >= 16) break;
      if (seenPosts.has(post.post)) continue;
      seenPosts.add(post.post);
      samplePosts.push(post);
    }

    return {
      ...base,
      community: this.decorate(base.community, session, viewer),
      members,
      samplePosts: samplePosts.slice(0, 16),
      neighborCommunities: base.neighborCommunities.map((c) => this.decorate(c, session, viewer)),
      defaultLineupCount: this.config.defaultLineup,
      boostedLineupCount: this.config.boostedLineup,
    };
  }

  async getCalibrationSwipes(params: {
    selections: CommunitySelectionInput[];
    seedFollowDids?: string[];
    maxSwipes?: number;
  }, viewer: ViewerRef) {
    const session = await this.session(viewer, params.seedFollowDids);
    const picks = this.pickedCommunityIds(params.selections);
    const source: DuelSource = {
      community: (communityId) => this.catalog.community(communityId),
      nearestByContent: (communityId) => this.nearestByContent(communityId),
      isHostilePair: (a, b) => this.catalog.isHostilePair(a, b),
      offerableCount: (communityId) => this.visible(session, communityId, viewer).length,
      followedCount: (communityId) => this.followedCount(session, communityId),
      noise: (communityId) => (seededHash(`duel:${communityId}`, session.seed) + 0.5) / 0x1_0000_0000,
      // The posts a side shows are by the accounts choosing it would add, then the scene's own samples.
      offerPosts: (communityId) => {
        const posts: string[] = [];
        for (const { member } of this.visible(session, communityId, viewer).slice(0, this.config.duelLineup)) {
          const uri = this.samplePostOf(member, viewer);
          if (uri && !posts.includes(uri)) posts.push(uri);
        }
        for (const uri of this.catalog.community(communityId)?.samplePostUris ?? []) {
          if (posts.length >= 3) break;
          if (!posts.includes(uri) && this.allowed(uri, viewer)) posts.push(uri);
        }
        return posts;
      },
    };
    const swipes = generateDuels({
      picks,
      maxSwipes: clamp(params.maxSwipes ?? 5, 1, 12),
      source,
      config: { minOffer: this.config.duelLineup },
    });
    return { catalogVersion: this.catalog.bundle.catalogVersion, swipes };
  }

  /** The scenes whose posts are most like this one's: closest post centroids first, weak matches left out. */
  private nearestByContent(communityId: number): number[] {
    const cached = this.nearestCache.get(communityId);
    if (cached) return cached;
    const own = this.postCentroids.get(communityId);
    const scored: { id: number; cosine: number }[] = [];
    if (own) {
      let ownSq = 0;
      for (let i = 0; i < own.length; i++) ownSq += own[i]! * own[i]!;
      for (const [id, other] of this.postCentroids) {
        if (id === communityId || other.length !== own.length) continue;
        let dot = 0;
        let otherSq = 0;
        for (let i = 0; i < own.length; i++) {
          dot += own[i]! * other[i]!;
          otherSq += other[i]! * other[i]!;
        }
        const cosine = ownSq > 0 && otherSq > 0 ? dot / Math.sqrt(ownSq * otherSq) : 0;
        if (cosine >= this.config.duelMinLikeness) scored.push({ id, cosine });
      }
    }
    const nearest = scored
      .sort((x, y) => y.cosine - x.cosine || x.id - y.id)
      .slice(0, this.config.duelContentNeighbours)
      .map((entry) => entry.id);
    this.nearestCache.set(communityId, nearest);
    return nearest;
  }

  /** The scenes a viewer picked, in order, or the catalog's first four when they picked none. */
  private pickedCommunityIds(selections: CommunitySelectionInput[]): number[] {
    const picks: number[] = [];
    for (const selection of selections) {
      if (!picks.includes(selection.communityId) && this.catalog.hasCommunity(selection.communityId)) {
        picks.push(selection.communityId);
      }
    }
    if (picks.length > 0) return picks;
    return this.catalog.bundle.communities.slice(0, 4).map((community) => community.communityId);
  }

  // ---------------------------------------------------------------------------------------------
  // Lineup
  // ---------------------------------------------------------------------------------------------

  /**
   * The scenes a lineup draws from, in output order, with how many follows each contributes.
   * `targetSize` undefined sizes each scene on its own; a number is the older client contract
   * (one total), split across the picked scenes in proportion to their weight.
   */
  private planScenes(
    selections: CommunitySelectionInput[],
    swipes: CalibrationSwipeAnswerInput[],
    targetSize: number | undefined,
  ): { communityId: number; quota: number; picked: boolean }[] {
    const picked: { communityId: number; boosted: boolean }[] = [];
    const seen = new Set<number>();
    for (const selection of selections) {
      if (seen.has(selection.communityId) || !this.catalog.hasCommunity(selection.communityId)) continue;
      seen.add(selection.communityId);
      picked.push({ communityId: selection.communityId, boosted: selection.weight === 2 });
    }
    if (picked.length === 0) {
      // Nothing picked: the same four-scene fallback the static engine uses.
      for (const community of this.catalog.bundle.communities.slice(0, 4)) {
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
        if (seen.has(communityId) || !this.catalog.hasCommunity(communityId)) continue;
        seen.add(communityId);
        duelAdds.push(communityId);
      }
    }

    if (targetSize !== undefined) {
      const total = clamp(targetSize, 1, this.config.maxLegacyTotal);
      const weights = [...picked.map((p) => (p.boosted ? 2 : 1)), ...duelAdds.map(() => 0.5)];
      const quotas = allocateByWeight(total, weights);
      return [
        ...picked.map((p, i) => ({ communityId: p.communityId, quota: quotas[i]!, picked: true })),
        ...duelAdds.map((communityId, i) => ({ communityId, quota: quotas[picked.length + i]!, picked: false })),
      ];
    }

    const plan = [
      ...picked.map((p) => ({
        communityId: p.communityId,
        quota: p.boosted ? this.config.boostedLineup : this.config.defaultLineup,
        picked: true,
      })),
      ...duelAdds.map((communityId) => ({ communityId, quota: this.config.duelLineup, picked: false })),
    ];
    let total = plan.reduce((acc, scene) => acc + scene.quota, 0);
    // Over the cap: duel additions give way first, then the extra half of boosted scenes.
    for (let i = plan.length - 1; i >= 0 && total > this.config.maxAutoTotal; i--) {
      if (plan[i]!.picked) continue;
      total -= plan[i]!.quota;
      plan[i]!.quota = 0;
    }
    while (total > this.config.maxAutoTotal) {
      const largest = plan.reduce((best, scene) => (scene.quota > best.quota ? scene : best), plan[0]!);
      if (largest.quota <= 1) break;
      largest.quota -= 1;
      total -= 1;
    }
    return plan;
  }

  async solveBundle(params: {
    selections: CommunitySelectionInput[];
    swipes?: CalibrationSwipeAnswerInput[];
    controls?: BundleControlsInput;
    targetSize?: number;
    pinnedDids?: string[];
    excludedDids?: string[];
    seedFollowDids?: string[];
  }, viewer: ViewerRef) {
    const t0 = performance.now();
    const session = await this.session(viewer, params.seedFollowDids);
    const pace = params.controls?.pace ?? "balanced";
    const repostTamer = params.controls?.originalsOnlyForHeavyReposters ?? true;
    const excluded = new Set(params.excludedDids ?? []);
    const plan = this.planScenes(params.selections, params.swipes ?? [], params.targetSize);

    interface Pick { communityId: number; member: IntelMemberEntry }
    const used = new Set<string>();
    const groups = new Map<number, Pick[]>();
    const bench: Pick[] = [];
    for (const scene of plan) {
      const group: Pick[] = [];
      groups.set(scene.communityId, group);
      if (scene.quota === 0) continue;
      let benched = 0;
      for (const { member } of this.visible(session, scene.communityId, viewer, excluded)) {
        if (used.has(member.did)) continue; // already followed through an earlier scene
        if (group.length < scene.quota) {
          used.add(member.did);
          group.push({ communityId: scene.communityId, member });
        } else if (benched < this.config.benchPerScene) {
          benched += 1;
          bench.push({ communityId: scene.communityId, member });
        } else {
          break;
        }
      }
    }

    // Pins are accounts the viewer ticked themselves. They are added to the lineup, never traded
    // against a scene's own share.
    const extras: Pick[] = [];
    for (const did of params.pinnedDids ?? []) {
      if (used.has(did) || excluded.has(did) || session.follows.has(did)) continue;
      if (!this.allowed(did, viewer)) continue;
      let home: { communityId: number; member: IntelMemberEntry } | undefined;
      for (const scene of plan) {
        const member = this.membersByCommunity.get(scene.communityId)?.find((m) => m.did === did);
        if (member) {
          home = { communityId: scene.communityId, member };
          break;
        }
      }
      home ??= this.memberHome.get(did);
      if (!home) continue;
      used.add(did);
      const group = groups.get(home.communityId);
      if (group) group.push(home);
      else extras.push(home);
    }

    const picks: Pick[] = [...plan.flatMap((scene) => groups.get(scene.communityId) ?? []), ...extras];
    const toFollow = ({ communityId, member }: Pick) => {
      const heavyReposter = member.recommendedMode === "originals_only" ||
        (pace === "calm" && member.repostShareBps >= 2500 && member.estimatedPostsPerWeek > 35);
      const hideReposts = repostTamer && heavyReposter;
      const samplePostUri = this.samplePostOf(member, viewer);
      return {
        did: member.did,
        communityId,
        communityTitle: this.catalog.community(communityId)?.title ?? "",
        role: member.role,
        deliveryMode: hideReposts ? ("originals_only" as const) : ("follow_all" as const),
        hideReposts,
        whyPicked: member.whyPicked,
        estimatedPostsPerWeek: member.estimatedPostsPerWeek,
        ...(samplePostUri ? { samplePostUri } : {}),
      };
    };
    const follows = picks.map(toFollow);
    // An alternate another scene went on to use, or that two scenes both offered, appears once or not at all.
    const benchSeen = new Set<string>();
    const alternates = bench.filter(({ member }) => {
      if (used.has(member.did) || benchSeen.has(member.did)) return false;
      benchSeen.add(member.did);
      return true;
    });

    const roleCounts = new Map<CommunityRole, number>();
    for (const { member } of picks) roleCounts.set(member.role, (roleCounts.get(member.role) ?? 0) + 1);
    const estimatedDailyPosts = Math.round(picks.reduce((sum, { member }, i) => {
      const kept = follows[i]!.hideReposts ? 1.0 - 0.85 * (member.repostShareBps / 10_000) : 1.0;
      return sum + (member.estimatedPostsPerWeek / 7.0) * kept;
    }, 0));

    const coveredIds = [...new Set(picks.map((p) => p.communityId))];
    const auxiliary = this.catalog.auxiliaryFor(coveredIds, coveredIds, viewer.unauthenticated);
    const followedInScenes = new Set<string>();
    for (const scene of plan) {
      if (!scene.picked) continue;
      for (const member of this.membersByCommunity.get(scene.communityId) ?? []) {
        if (session.follows.has(member.did)) followedInScenes.add(member.did);
      }
    }

    return {
      sessionId: `onb_${this.now()}_${Math.floor(this.random() * 0x7fffffff).toString(36)}`,
      catalogVersion: this.catalog.bundle.catalogVersion,
      solveMs: Math.max(0, Math.round(performance.now() - t0)),
      summary: {
        totalFollows: follows.length,
        anchorCount: (roleCounts.get("anchor_creator") ?? 0) + (roleCounts.get("project_or_studio") ?? 0),
        makerCount: roleCounts.get("visual_maker") ?? 0,
        conversationalCount: roleCounts.get("conversationalist") ?? 0,
        curatorCount: roleCounts.get("scene_curator") ?? 0,
        originalsOnlyCount: follows.filter((f) => f.hideReposts).length,
        estimatedDailyPosts,
        coveredCommunities: coveredIds.length,
      },
      follows,
      feeds: auxiliary.feeds,
      starterPacks: auxiliary.starterPacks,
      harmoniousNeighbors: auxiliary.harmoniousNeighbors.map((c) => this.decorate(c, session, viewer)),
      bench: alternates.map(toFollow),
      alreadyFollowingInScenes: followedInScenes.size,
    };
  }

  async previewTimeline(params: {
    selections: CommunitySelectionInput[];
    swipes?: CalibrationSwipeAnswerInput[];
    controls?: BundleControlsInput;
    limit?: number;
  }, viewer: ViewerRef) {
    const solved = await this.solveBundle({
      selections: params.selections,
      swipes: params.swipes,
      controls: params.controls,
    }, viewer);
    const session = await this.session(viewer);
    const limit = clamp(params.limit ?? 20, 1, 40);
    const items: { post: string; communityId: number; communityTitle: string; authorRole: string }[] = [];
    const seen = new Set<string>();

    // One post per lineup account, taking a scene at a time so the preview mixes scenes the way
    // the real timeline will.
    const byScene = new Map<number, typeof solved.follows>();
    for (const follow of solved.follows) {
      const list = byScene.get(follow.communityId);
      if (list) list.push(follow);
      else byScene.set(follow.communityId, [follow]);
    }
    const queues = [...byScene.values()];
    for (let round = 0; items.length < limit; round++) {
      let any = false;
      for (const queue of queues) {
        const follow = queue[round];
        if (!follow) continue;
        any = true;
        if (!follow.samplePostUri || seen.has(follow.samplePostUri)) continue;
        seen.add(follow.samplePostUri);
        items.push({
          post: follow.samplePostUri,
          communityId: follow.communityId,
          communityTitle: follow.communityTitle,
          authorRole: follow.role,
        });
        if (items.length >= limit) break;
      }
      if (!any) break;
    }

    // Short of posts: fill from each scene's own preview posts, preferring authors the viewer
    // will follow or already follows over authors who will not be in their timeline.
    const lineup = new Set(solved.follows.map((f) => f.did));
    for (const inTimeline of [true, false]) {
      for (const communityId of byScene.keys()) {
        if (items.length >= limit) break;
        const community = this.catalog.community(communityId);
        if (!community) continue;
        const uris = community.previewPostUris.length > 0 ? community.previewPostUris : community.samplePostUris;
        for (const uri of uris) {
          if (items.length >= limit) break;
          if (seen.has(uri) || !this.allowed(uri, viewer)) continue;
          const owner = /^at:\/\/(did:[^/]+)\//.exec(uri)?.[1] ?? "";
          if ((lineup.has(owner) || session.follows.has(owner)) !== inTimeline) continue;
          seen.add(uri);
          items.push({
            post: uri,
            communityId,
            communityTitle: community.title,
            authorRole: community.members.find((m) => m.did === owner)?.role ?? "anchor_creator",
          });
        }
      }
    }

    return {
      catalogVersion: this.catalog.bundle.catalogVersion,
      items,
      estimatedDailyPosts: solved.summary.estimatedDailyPosts,
    };
  }
}

/**
 * Follow-list loader over the public AppView. One page is 100 follows; `maxPages` bounds the
 * work for accounts that follow tens of thousands.
 */
export function createPublicFollowsFetcher(options: {
  fetcher?: (input: URL, init?: RequestInit) => Promise<Response>;
  baseUrl?: string;
  maxPages?: number;
  timeoutMs?: number;
} = {}): FollowsFetcher {
  const fetcher = options.fetcher ?? fetch;
  const baseUrl = options.baseUrl ?? "https://public.api.bsky.app";
  const maxPages = options.maxPages ?? 100;
  const timeoutMs = options.timeoutMs ?? 3_000;
  return async (viewerDid, onPage) => {
    let cursor: string | undefined;
    for (let page = 0; page < maxPages; page++) {
      const url = new URL("/xrpc/app.bsky.graph.getFollows", baseUrl);
      url.searchParams.set("actor", viewerDid);
      url.searchParams.set("limit", "100");
      if (cursor) url.searchParams.set("cursor", cursor);
      const response = await fetcher(url, {
        headers: { accept: "application/json" },
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!response.ok) return;
      const body = await response.json() as { follows?: unknown; cursor?: unknown };
      const dids: string[] = [];
      for (const raw of Array.isArray(body.follows) ? body.follows : []) {
        const did = raw && typeof raw === "object" ? (raw as { did?: unknown }).did : undefined;
        if (typeof did === "string" && did.startsWith("did:")) dids.push(did);
      }
      onPage(dids);
      cursor = typeof body.cursor === "string" && body.cursor.length > 0 ? body.cursor : undefined;
      if (!cursor) return;
    }
  };
}
