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
import {
  LIVE_ONBOARDING_DEFAULTS,
  type FollowSpaceVectors,
  type FollowsFetcher,
  type LiveAccountSignals,
  type LiveOnboardingConfig,
  type MemberTier,
  type OnboardingPoolFile,
  type OrderedMember,
  type Session,
  type ViewerRef,
} from "./onboarding_live_types.ts";
import { allocateByWeight, clamp, seededHash } from "./onboarding_math.ts";
import { scoreMembers } from "./onboarding_order.ts";
import { nearestByContent, pickedCommunityIds, planScenes } from "./onboarding_scenes.ts";
import { applySlate, tierOf } from "./onboarding_slate.ts";
import { tasteCentroid } from "./onboarding_taste.ts";

export { LIVE_ONBOARDING_DEFAULTS, allocateByWeight };
export { createPublicFollowsFetcher } from "./onboarding_follows.ts";
export type {
  FollowSpaceVectors,
  FollowsFetcher,
  LiveAccountSignals,
  LiveOnboardingConfig,
  MemberTier,
  OnboardingPoolFile,
  ViewerRef,
};

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
  private followLoadsActive = 0;
  private readonly followLoadWaiters: Array<() => void> = [];
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

    // Every account belongs to one scene. The catalog's own members claim theirs first, in catalog
    // order; the pool then adds depth behind them, and never moves an account the catalog placed.
    // (An account in two scenes would be followed through one and silently missing from the other.)
    for (const community of this.catalog.bundle.communities) {
      const own: IntelMemberEntry[] = [];
      for (const member of community.members) {
        if (this.memberHome.has(member.did) || !this.catalog.isAccountAllowed(member.did, false)) continue;
        this.memberHome.set(member.did, { communityId: community.communityId, member });
        own.push(member);
      }
      this.membersByCommunity.set(community.communityId, own);
    }
    for (const community of this.catalog.bundle.communities) {
      const merged = this.membersByCommunity.get(community.communityId)!;
      for (const member of poolByCommunity.get(community.communityId) ?? []) {
        if (this.memberHome.has(member.did) || !this.catalog.isAccountAllowed(member.did, false)) continue;
        this.memberHome.set(member.did, { communityId: community.communityId, member });
        merged.push(member);
      }
      const centroid = Buffer.from(community.centPostsInt8B64, "base64");
      if (centroid.byteLength > 0) {
        this.postCentroids.set(community.communityId, new Int8Array(centroid.buffer, centroid.byteOffset, centroid.byteLength));
      }
      this.poolMemberCount += merged.length;
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
      ...(this.liveSignals?.describe?.() ?? {}),
      onboarding_personalised: this.vectors !== null,
      onboarding_sessions: this.sessions.size,
    };
  }

  // ---------------------------------------------------------------------------------------------
  // Sessions and follow lists
  // ---------------------------------------------------------------------------------------------

  private newSession(seed: number, t: number): Session {
    return {
      seed,
      lastSeenAt: t,
      follows: new Set<string>(),
      followsLoad: null,
      followsLoading: false,
      followsLoadedAt: null,
      followsNextAt: 0,
      newestFollow: null,
      centroid: undefined,
      lists: new Map(),
    };
  }

  /** Remembers which offerable accounts the viewer follows. Anything else cannot affect a list. */
  private noteFollows(session: Session, dids: Iterable<string>): void {
    for (const did of dids) {
      if (session.follows.size >= this.config.maxKeptFollows) return;
      if (this.memberHome.has(did)) session.follows.add(did);
    }
  }

  private async session(viewer: ViewerRef, seedFollowDids: readonly string[] = []): Promise<Session> {
    const t = this.now();
    if (!viewer.did) {
      // A signed-out viewer has no identity to keep anything under, so nothing is kept: one
      // request's follows can never show up in another's. The order comes from the day instead.
      const session = this.newSession(this.seedFor(viewer, t), t);
      this.noteFollows(session, seedFollowDids);
      session.centroid = tasteCentroid(this.vectors, this.config, seedFollowDids);
      return session;
    }
    const key = viewer.did;
    let session = this.sessions.get(key);
    this.sessions.delete(key); // re-inserted below, so eviction drops the least recently used
    if (!session || t - session.lastSeenAt > this.config.sessionTtlMs) {
      session = this.newSession(this.seedFor(viewer, t), t);
    }
    this.sessions.set(key, session);
    while (this.sessions.size > this.config.maxSessions) {
      const oldest = this.sessions.keys().next().value;
      if (oldest === undefined) break;
      this.sessions.delete(oldest);
    }
    session.lastSeenAt = t;
    this.noteFollows(session, seedFollowDids);

    if (!this.followsFetcher) {
      if (session.centroid === undefined && seedFollowDids.length > 0) {
        session.centroid = tasteCentroid(this.vectors, this.config, seedFollowDids);
      }
      return session;
    }
    if (!session.followsLoading && t >= session.followsNextAt) this.startFollowsLoad(session, viewer.did);
    // Only the first load is worth waiting for; a refresh adds the few accounts followed since.
    if (session.followsLoading && session.followsLoadedAt === null && session.followsLoad) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      await Promise.race([
        session.followsLoad,
        new Promise<void>((resolve) => {
          timer = setTimeout(resolve, this.config.followsBudgetMs);
        }),
      ]);
      clearTimeout(timer);
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
    // Signed-out viewers share one order per day: the same on every request and every instance.
    if (!viewer.did) return seededHash(`signed-out:${Math.floor(t / 86_400_000)}`, this.orderSalt);
    const stretch = this.config.sessionTtlMs;
    const offset = seededHash(viewer.did, 0x5eed) % stretch;
    return seededHash(`${viewer.did}:${Math.floor((t + offset) / stretch)}`, this.orderSalt);
  }

  private startFollowsLoad(session: Session, viewerDid: string): void {
    session.followsLoading = true;
    // After one complete load, a refresh only needs the follows made since: it stops at the
    // account that was newest last time.
    const stopAt = session.followsLoadedAt === null ? null : session.newestFollow;
    let newest: string | null | undefined;
    const run = async () => {
      await this.acquireFollowLoad();
      try {
        await this.followsFetcher!(viewerDid, (dids) => {
          if (newest === undefined) {
            newest = dids[0] ?? null;
            // Taste comes from the first page alone (the hundred most recent follows), so it does
            // not depend on how much of a long follow list had arrived when a list was first built.
            if (session.centroid === undefined) session.centroid = tasteCentroid(this.vectors, this.config, dids);
          }
          this.noteFollows(session, dids);
          return stopAt !== null && dids.includes(stopAt);
        });
        session.followsLoadedAt = this.now();
        session.followsNextAt = session.followsLoadedAt + this.config.followsRefreshMs;
        if (newest) session.newestFollow = newest;
      } catch {
        // What arrived is still used; the rest is tried again shortly.
        session.followsNextAt = this.now() + this.config.followsRetryMs;
      } finally {
        this.releaseFollowLoad();
        session.followsLoading = false;
      }
    };
    session.followsLoad = run();
  }

  /** Bounds how many follow lists are being loaded from the public API at once. */
  private acquireFollowLoad(): Promise<void> {
    if (this.followLoadsActive < this.config.maxFollowLoads) {
      this.followLoadsActive += 1;
      return Promise.resolve();
    }
    return new Promise<void>((resolve) => this.followLoadWaiters.push(resolve));
  }

  private releaseFollowLoad(): void {
    const next = this.followLoadWaiters.shift();
    if (next) next(); // the slot passes straight to the next waiter
    else this.followLoadsActive -= 1;
  }

  // ---------------------------------------------------------------------------------------------
  // Ordered lists
  // ---------------------------------------------------------------------------------------------

  private orderedList(session: Session, communityId: number, viewer: ViewerRef): OrderedMember[] {
    const members = this.membersByCommunity.get(communityId) ?? [];
    const cached = session.lists.get(communityId);
    if (cached) {
      session.lists.delete(communityId); // re-inserted: the least recently read scene is dropped first
      session.lists.set(communityId, cached);
      const restored: OrderedMember[] = new Array(cached.length);
      for (let i = 0; i < cached.length; i++) {
        const member = members[cached[i]!]!;
        restored[i] = { member, tier: tierOf(member, this.config), key: 0 };
      }
      return restored;
    }
    // From here on the session's taste is settled: a first page of follows that arrives later
    // does not change it, so scenes built before and after it are ordered the same way.
    if (session.centroid === undefined) session.centroid = null;
    const centroid = session.centroid;

    const scored = scoreMembers(members, centroid, session.seed, this.vectors, this.config, this.liveSignals);

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
    const ordered = applySlate(open, this.config).concat(closed);
    const position = new Map<IntelMemberEntry, number>();
    members.forEach((member, index) => position.set(member, index));
    session.lists.set(communityId, Uint16Array.from(ordered, (entry) => position.get(entry.member)!));
    while (session.lists.size > this.config.maxListsPerSession) {
      const oldest = session.lists.keys().next().value;
      if (oldest === undefined) break;
      session.lists.delete(oldest);
    }
    return ordered;
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
    const picks = pickedCommunityIds(this.catalog, params.selections);
    const source: DuelSource = {
      community: (communityId) => this.catalog.community(communityId),
      nearestByContent: (communityId) => nearestByContent(this.postCentroids, this.nearestCache, this.config, communityId),
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

  // ---------------------------------------------------------------------------------------------
  // Lineup
  // ---------------------------------------------------------------------------------------------

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
    const plan = planScenes(this.catalog, this.config, params.selections, params.swipes ?? [], params.targetSize);

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
    // against a scene's own share, up to the most one lineup may follow.
    const extras: Pick[] = [];
    for (const did of params.pinnedDids ?? []) {
      if (used.size >= this.config.maxLegacyTotal) break;
      if (used.has(did) || excluded.has(did) || session.follows.has(did)) continue;
      if (!this.allowed(did, viewer)) continue;
      const home = this.memberHome.get(did);
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
    targetSize?: number;
    pinnedDids?: string[];
    excludedDids?: string[];
    seedFollowDids?: string[];
  }, viewer: ViewerRef) {
    // Solved with everything the lineup will be solved with, so the preview is of that lineup.
    const solved = await this.solveBundle({
      selections: params.selections,
      swipes: params.swipes,
      controls: params.controls,
      targetSize: params.targetSize,
      pinnedDids: params.pinnedDids,
      excludedDids: params.excludedDids,
      seedFollowDids: params.seedFollowDids,
    }, viewer);
    const session = await this.session(viewer, params.seedFollowDids);
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
