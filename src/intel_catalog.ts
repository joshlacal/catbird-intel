import { readFileSync } from "node:fs";

export type CommunityRole =
  | "anchor_creator"
  | "visual_maker"
  | "conversationalist"
  | "scene_curator"
  | "project_or_studio";

export type DeliveryMode = "follow_all" | "originals_only" | "topic_feed_only";

export interface IntelSceneGroup {
  sceneId: number;
  sceneSlug: string;
  sceneTitle: string;
  macroCategory: string;
  symbolName: string;
  cardCount: number;
  totalEligibleAccounts: number;
  topCommunityIds: number[];
}

export interface IntelMemberEntry {
  did: string;
  role: CommunityRole;
  recommendedMode: DeliveryMode;
  whyPicked: string;
  rankScoreBps: number;
  repostShareBps: number;
  mediaShareBps: number;
  estimatedPostsPerWeek: number;
  followers: number;
  samplePostUri?: string;
}

export interface IntelCommunityEntry {
  communityId: number;
  title: string;
  description: string;
  sceneId: number;
  sceneSlug: string;
  sceneTitle: string;
  macroCategory: string;
  symbolName: string;
  eligibleCount: number;
  postCoherenceBps: number;
  socialCoherenceBps: number;
  curatedListTags: string[];
  searchTerms: string[];
  centPostsInt8B64: string;
  neighborCommunityIds: number[];
  samplePostUris: string[];
  previewPostUris: string[];
  sampleAccountDids: string[];
  feedGeneratorUris: string[];
  starterPackUris: string[];
  members: IntelMemberEntry[];
}

export interface IntelTasteDuelOption {
  communityId: number;
  title: string;
  subtitle: string;
  sceneTitle: string;
  samplePostUris: string[];
}

export interface IntelTasteDuel {
  swipeId: string;
  kind: "collision_trap" | "subscene_split" | "taste_bridge";
  prompt: string;
  optionA: IntelTasteDuelOption;
  optionB: IntelTasteDuelOption;
}

export interface IntelServingBundleV2 {
  catalogVersion: string;
  generatedAt: string;
  sourceCardsSha256: string;
  totalScenes: number;
  totalCommunities: number;
  totalUniqueAccounts: number;
  noUnauthDids: string[];
  hideFromRecsDids?: string[];
  highHostilityPairs: [number, number][];
  warmStartIndex: Record<string, [number, number][]>;
  scenes: IntelSceneGroup[];
  communities: IntelCommunityEntry[];
  tasteDuels: IntelTasteDuel[];
}

export interface CommunitySelectionInput {
  communityId: number;
  weight: number;
}

export interface CalibrationSwipeAnswerInput {
  swipeId: string;
  communityA: number;
  communityB: number;
  choice: "a" | "b" | "both" | "neither";
}

export interface BundleControlsInput {
  pace: "calm" | "balanced" | "lively";
  originalsOnlyForHeavyReposters: boolean;
}

function tokenize(text: string): string[] {
  const matches = text.toLowerCase().match(/[a-z0-9#+._-]{2,28}/g) ?? [];
  const out: string[] = [];
  for (const raw of matches) {
    const cleaned = raw.replace(/^[._-]+|[._-]+$/g, "");
    if (cleaned.length >= 2) out.push(cleaned);
  }
  return out;
}

function ownerDidFromUri(uriOrDid: string): string | null {
  if (uriOrDid.startsWith("did:")) return uriOrDid;
  const m = /^at:\/\/(did:[^/]+)\//.exec(uriOrDid);
  return m?.[1] ?? null;
}

function pairKey(a: number, b: number): string {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

export class S17CatalogEngine {
  readonly bundle: IntelServingBundleV2;
  private readonly commById = new Map<number, IntelCommunityEntry>();
  private readonly commIndexById = new Map<number, number>();
  private readonly hiddenDids: Set<string>;
  private readonly noUnauthDids: Set<string>;
  private readonly highHostilitySet = new Set<string>();
  private readonly docTermCounts: Map<string, number>[];
  private readonly titleTermSets: Set<string>[];
  private readonly listTagTermSets: Set<string>[];
  private readonly docVectors: Float32Array[];
  private readonly didToMember = new Map<string, { comm: IntelCommunityEntry; member: IntelMemberEntry }>();

  constructor(bundle: IntelServingBundleV2) {
    this.bundle = bundle;
    this.hiddenDids = new Set(bundle.hideFromRecsDids ?? []);
    this.noUnauthDids = new Set(bundle.noUnauthDids ?? []);
    for (const [a, b] of bundle.highHostilityPairs ?? []) {
      this.highHostilitySet.add(pairKey(a, b));
    }

    this.docTermCounts = [];
    this.titleTermSets = [];
    this.listTagTermSets = [];
    this.docVectors = [];

    bundle.communities.forEach((comm, idx) => {
      this.commById.set(comm.communityId, comm);
      this.commIndexById.set(comm.communityId, idx);

      this.titleTermSets.push(new Set(tokenize(comm.title)));
      this.listTagTermSets.push(new Set(tokenize(comm.curatedListTags.join(" "))));

      // Build hybrid lexical index (title + sceneTitle + description + curatedListTags + searchTerms)
      const tokens = tokenize(
        `${comm.title} ${comm.sceneTitle} ${comm.description} ${comm.curatedListTags.join(" ")} ${comm.searchTerms.join(" ")}`,
      );
      const cnt = new Map<string, number>();
      for (const t of tokens) cnt.set(t, (cnt.get(t) ?? 0) + 1);
      this.docTermCounts.push(cnt);

      // Decode 256d int8 cent_posts vector and L2-normalize
      const rawBytes = Buffer.from(comm.centPostsInt8B64 || "", "base64");
      const vec = new Float32Array(256);
      let normSq = 0;
      for (let d = 0; d < Math.min(256, rawBytes.length); d++) {
        // Interpret byte as signed int8
        const u = rawBytes[d]!;
        const s = u > 127 ? u - 256 : u;
        vec[d] = s;
        normSq += s * s;
      }
      const norm = Math.sqrt(normSq);
      if (norm > 1e-8) {
        for (let d = 0; d < 256; d++) vec[d] = vec[d]! / norm;
      }
      this.docVectors.push(vec);

      for (const m of comm.members) {
        if (this.hiddenDids.has(m.did)) continue;
        if (!this.didToMember.has(m.did)) {
          this.didToMember.set(m.did, { comm, member: m });
        }
      }
    });
  }

  static fromFile(path: string): S17CatalogEngine {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as IntelServingBundleV2;
    return new S17CatalogEngine(parsed);
  }

  hasCommunity(communityId: number): boolean {
    return this.commById.has(communityId);
  }

  private isUriAllowed(uriOrDid: string, unauthenticated: boolean): boolean {
    const owner = ownerDidFromUri(uriOrDid);
    if (!owner) return false;
    if (this.hiddenDids.has(owner)) return false;
    if (unauthenticated && this.noUnauthDids.has(owner)) return false;
    return true;
  }

  toCommunitySummary(comm: IntelCommunityEntry, unauthenticated: boolean) {
    return {
      communityId: comm.communityId,
      title: comm.title,
      description: comm.description,
      sceneId: comm.sceneId,
      sceneSlug: comm.sceneSlug,
      sceneTitle: comm.sceneTitle,
      macroCategory: comm.macroCategory,
      symbolName: comm.symbolName,
      eligibleCount: comm.eligibleCount,
      postCoherenceBps: comm.postCoherenceBps,
      socialCoherenceBps: comm.socialCoherenceBps,
      curatedListTags: comm.curatedListTags,
      neighborCommunityIds: comm.neighborCommunityIds.filter((nid) => this.commById.has(nid)),
      samplePostUris: comm.samplePostUris.filter((u) => this.isUriAllowed(u, unauthenticated)).slice(0, 6),
      sampleAccountDids: comm.sampleAccountDids.filter((d) => this.isUriAllowed(d, unauthenticated)).slice(0, 6),
      feedGeneratorUris: comm.feedGeneratorUris.filter((u) => this.isUriAllowed(u, unauthenticated)),
      starterPackUris: comm.starterPackUris.filter((u) => this.isUriAllowed(u, unauthenticated)),
    };
  }

  private hybridRankCommunities(query: string): IntelCommunityEntry[] {
    const qLower = query.trim().toLowerCase();
    const qTokens = tokenize(qLower);
    const n = this.bundle.communities.length;
    if (n === 0 || qTokens.length === 0) return this.bundle.communities;

    const df = new Map<string, number>();
    for (const qt of qTokens) {
      const singular = qt.length >= 5 && qt.endsWith("s") ? qt.slice(0, -1) : null;
      let hits = 0;
      for (const d of this.docTermCounts) {
        if (d.has(qt) || (singular && d.has(singular))) {
          hits += 1;
        } else if (qt.length >= 4) {
          for (const w of d.keys()) {
            if (w.startsWith(qt)) {
              hits += 1;
              break;
            }
          }
        }
      }
      df.set(qt, hits);
    }

    const lexScores = new Float32Array(n);
    let maxLex = 0;
    for (let i = 0; i < n; i++) {
      const comm = this.bundle.communities[i]!;
      const dCnt = this.docTermCounts[i]!;
      const tSet = this.titleTermSets[i]!;
      const lSet = this.listTagTermSets[i]!;
      let score = 0;
      if (comm.title.toLowerCase().includes(qLower)) score += 10.0;
      else if (comm.sceneTitle.toLowerCase().includes(qLower)) score += 3.5;

      for (const qt of qTokens) {
        const singular = qt.length >= 5 && qt.endsWith("s") ? qt.slice(0, -1) : null;
        const idf = Math.log((n + 1.0) / (1.0 + (df.get(qt) ?? 0))) + 1.0;
        const exact = (dCnt.get(qt) ?? 0) + (singular ? (dCnt.get(singular) ?? 0) : 0);
        if (tSet.has(qt) || (singular !== null && tSet.has(singular))) {
          score += idf * 3.6;
        }
        if (lSet.has(qt) || (singular !== null && lSet.has(singular))) {
          score += idf * 1.6;
        }
        if (exact > 0) {
          score += idf * (1.6 + Math.log1p(exact));
        } else if (qt.length >= 4) {
          let prefCount = 0;
          for (const [w, cnt] of dCnt) {
            if (w.startsWith(qt)) prefCount += cnt;
          }
          if (prefCount > 0) score += idf * 0.75;
        }
      }
      lexScores[i] = score;
      if (score > maxLex) maxLex = score;
    }

    if (maxLex <= 0) return [];

    // Semantic centroid neighborhood propagation from top-5 lexical seeds
    const indices = Array.from({ length: n }, (_, i) => i);
    indices.sort((a, b) => lexScores[b]! - lexScores[a]!);
    const seedIdx = indices.slice(0, 5).filter((i) => lexScores[i]! > 0);
    let seedSum = 0;
    for (const si of seedIdx) seedSum += lexScores[si]!;

    const finalScores = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let sem = 0;
      if (seedSum > 1e-6) {
        const vi = this.docVectors[i]!;
        for (const si of seedIdx) {
          const vs = this.docVectors[si]!;
          let dot = 0;
          for (let d = 0; d < 256; d++) dot += vi[d]! * vs[d]!;
          if (dot > 0.35) {
            sem += (lexScores[si]! / seedSum) * dot;
          }
        }
      }
      const boost = sem * maxLex * 0.35;
      const raw = lexScores[i]! + (boost > 0.15 * maxLex ? boost : 0);
      finalScores[i] = raw > 0 ? raw + 1e-4 * Math.log1p(this.bundle.communities[i]!.eligibleCount) : 0;
    }

    return indices
      .filter((i) => finalScores[i]! > 0)
      .sort((a, b) => finalScores[b]! - finalScores[a]!)
      .map((i) => this.bundle.communities[i]!);
  }

  listCommunities(params: {
    q?: string | null;
    macroCategory?: string | null;
    sceneSlug?: string | null;
    limit?: number;
    cursor?: string | null;
    unauthenticated: boolean;
  }) {
    let list = params.q && params.q.trim().length > 0
      ? this.hybridRankCommunities(params.q)
      : [...this.bundle.communities];

    if (params.macroCategory) {
      list = list.filter((c) => c.macroCategory === params.macroCategory);
    }
    if (params.sceneSlug) {
      list = list.filter((c) => c.sceneSlug === params.sceneSlug);
    }

    const offset = params.cursor ? Math.max(0, Number.parseInt(params.cursor, 10) || 0) : 0;
    const limit = Math.min(Math.max(params.limit ?? 60, 1), 200);
    const slice = list.slice(offset, offset + limit);
    const nextCursor = offset + slice.length < list.length ? String(offset + slice.length) : undefined;

    return {
      catalogVersion: this.bundle.catalogVersion,
      totalCount: list.length,
      ...(nextCursor ? { cursor: nextCursor } : {}),
      scenes: this.bundle.scenes,
      communities: slice.map((c) => this.toCommunitySummary(c, params.unauthenticated)),
    };
  }

  getCommunity(communityId: number, unauthenticated: boolean) {
    const comm = this.commById.get(communityId);
    if (!comm) return null;

    const members = comm.members
      .filter((m) => this.isUriAllowed(m.did, unauthenticated))
      .map((m) => {
        const samplePostUri = m.samplePostUri && this.isUriAllowed(m.samplePostUri, unauthenticated)
          ? m.samplePostUri
          : undefined;
        return {
          did: m.did,
          role: m.role,
          recommendedMode: m.recommendedMode,
          whyPicked: m.whyPicked,
          rankScoreBps: m.rankScoreBps,
          repostShareBps: m.repostShareBps,
          mediaShareBps: m.mediaShareBps,
          estimatedPostsPerWeek: m.estimatedPostsPerWeek,
          ...(samplePostUri ? { samplePostUri } : {}),
        };
      });

    const roleByDid = new Map(members.map((m) => [m.did, m.role]));
    const postUris = [...new Set([...comm.samplePostUris, ...comm.previewPostUris])]
      .filter((u) => this.isUriAllowed(u, unauthenticated))
      .slice(0, 16);

    const samplePosts = postUris.map((uri) => {
      const owner = ownerDidFromUri(uri) ?? "";
      return {
        post: uri,
        communityId: comm.communityId,
        communityTitle: comm.title,
        authorRole: roleByDid.get(owner) ?? "anchor_creator",
      };
    });

    const neighborCommunities = comm.neighborCommunityIds
      .map((nid) => this.commById.get(nid))
      .filter((c): c is IntelCommunityEntry => Boolean(c))
      .slice(0, 6)
      .map((c) => this.toCommunitySummary(c, unauthenticated));

    return {
      catalogVersion: this.bundle.catalogVersion,
      community: this.toCommunitySummary(comm, unauthenticated),
      members,
      samplePosts,
      neighborCommunities,
    };
  }

  getCatalog(params: {
    seedFollowDids?: string[];
    q?: string | null;
    macroCategory?: string | null;
    limit?: number;
    unauthenticated: boolean;
  }) {
    const warmScores = new Map<number, number>();
    for (const did of params.seedFollowDids ?? []) {
      const hits = this.bundle.warmStartIndex[did];
      if (!hits) continue;
      for (const [cid, bps] of hits) {
        if (this.commById.has(cid)) {
          warmScores.set(cid, (warmScores.get(cid) ?? 0) + bps / 10000.0);
        }
      }
    }
    const warmStartCommunities = [...warmScores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 16)
      .map(([cid]) => this.commById.get(cid)!)
      .map((c) => this.toCommunitySummary(c, params.unauthenticated));

    const limit = Math.min(Math.max(params.limit ?? 80, 1), 200);
    let featuredRaw: IntelCommunityEntry[];
    if (params.q && params.q.trim().length > 0) {
      featuredRaw = this.hybridRankCommunities(params.q);
      if (params.macroCategory) {
        featuredRaw = featuredRaw.filter((c) => c.macroCategory === params.macroCategory);
      }
    } else {
      // Scene-diversified round-robin across scenes so every macro-category shines immediately
      const pool = params.macroCategory
        ? this.bundle.communities.filter((c) => c.macroCategory === params.macroCategory)
        : this.bundle.communities;
      const byScene = new Map<number, IntelCommunityEntry[]>();
      for (const c of pool) {
        const arr = byScene.get(c.sceneId) ?? [];
        arr.push(c);
        byScene.set(c.sceneId, arr);
      }
      const queues = [...byScene.values()];
      featuredRaw = [];
      let round = 0;
      while (featuredRaw.length < limit) {
        let added = false;
        for (const q of queues) {
          if (round < q.length) {
            featuredRaw.push(q[round]!);
            added = true;
            if (featuredRaw.length >= limit) break;
          }
        }
        if (!added) break;
        round += 1;
      }
    }

    return {
      catalogVersion: this.bundle.catalogVersion,
      totalCommunities: this.bundle.communities.length,
      scenes: this.bundle.scenes,
      featuredCommunities: featuredRaw.slice(0, limit).map((c) => this.toCommunitySummary(c, params.unauthenticated)),
      warmStartCommunities,
    };
  }

  getCalibrationSwipes(params: {
    selections: CommunitySelectionInput[];
    seedFollowDids?: string[];
    maxSwipes?: number;
    unauthenticated: boolean;
  }) {
    const selectedCids = new Set(params.selections.map((s) => s.communityId));
    const selectedScenes = new Set(
      params.selections
        .map((s) => this.commById.get(s.communityId)?.sceneId)
        .filter((sid): sid is number => sid !== undefined),
    );
    const maxSwipes = Math.min(Math.max(params.maxSwipes ?? 5, 1), 12);

    const scored = this.bundle.tasteDuels
      .map((duel) => {
        const aCid = duel.optionA.communityId;
        const bCid = duel.optionB.communityId;
        const aScene = this.commById.get(aCid)?.sceneId;
        const bScene = this.commById.get(bCid)?.sceneId;
        let relevance = 0;
        if (selectedCids.has(aCid)) relevance += 4;
        if (selectedCids.has(bCid)) relevance += 4;
        if (aScene !== undefined && selectedScenes.has(aScene)) relevance += 2;
        if (bScene !== undefined && selectedScenes.has(bScene)) relevance += 2;
        if (duel.kind === "collision_trap") relevance += 1;
        return { duel, relevance };
      })
      .sort((x, y) => y.relevance - x.relevance);

    const swipes = [];
    for (const { duel } of scored) {
      if (swipes.length >= maxSwipes) break;
      const urisA = duel.optionA.samplePostUris.filter((u) => this.isUriAllowed(u, params.unauthenticated));
      const urisB = duel.optionB.samplePostUris.filter((u) => this.isUriAllowed(u, params.unauthenticated));
      if (urisA.length === 0 || urisB.length === 0) continue;
      swipes.push({
        swipeId: duel.swipeId,
        step: swipes.length + 1,
        kind: duel.kind,
        prompt: duel.prompt,
        optionA: { ...duel.optionA, samplePostUris: urisA.slice(0, 3) },
        optionB: { ...duel.optionB, samplePostUris: urisB.slice(0, 3) },
      });
    }

    return {
      catalogVersion: this.bundle.catalogVersion,
      swipes,
    };
  }

  private computePosteriorWeights(
    selections: CommunitySelectionInput[],
    swipes: CalibrationSwipeAnswerInput[] = [],
    seedFollowDids: string[] = [],
  ): Map<number, number> {
    const weights = new Map<number, number>();
    const add = (cid: number, delta: number) => {
      if (!this.commById.has(cid)) return;
      weights.set(cid, Math.max(0, (weights.get(cid) ?? 0) + delta));
    };

    for (const sel of selections) {
      const w = sel.weight === 2 ? 2.0 : 1.0;
      add(sel.communityId, w);
      // Gentle spill into top harmonious neighbor for coverage depth
      const comm = this.commById.get(sel.communityId);
      for (const nid of (comm?.neighborCommunityIds ?? []).slice(0, 2)) {
        if (!this.highHostilitySet.has(pairKey(sel.communityId, nid))) {
          add(nid, w * 0.22);
        }
      }
    }

    for (const seedDid of seedFollowDids) {
      for (const [cid, bps] of this.bundle.warmStartIndex[seedDid] ?? []) {
        add(cid, (bps / 10000.0) * 0.35);
      }
    }

    for (const sw of swipes) {
      if (sw.choice === "a") {
        add(sw.communityA, 0.95);
        add(sw.communityB, -0.55);
      } else if (sw.choice === "b") {
        add(sw.communityB, 0.95);
        add(sw.communityA, -0.55);
      } else if (sw.choice === "both") {
        add(sw.communityA, 0.65);
        add(sw.communityB, 0.65);
      } else if (sw.choice === "neither") {
        add(sw.communityA, -0.85);
        add(sw.communityB, -0.85);
      }
    }

    // Fallback if user selected nothing or zeroed everything
    if ([...weights.values()].every((v) => v <= 0.05)) {
      for (const c of this.bundle.communities.slice(0, 4)) {
        weights.set(c.communityId, 1.0);
      }
    }
    return weights;
  }

  solveBundle(params: {
    selections: CommunitySelectionInput[];
    swipes?: CalibrationSwipeAnswerInput[];
    controls?: BundleControlsInput;
    targetSize?: number;
    pinnedDids?: string[];
    excludedDids?: string[];
    seedFollowDids?: string[];
    unauthenticated: boolean;
  }) {
    const t0 = performance.now();
    const pace = params.controls?.pace ?? "balanced";
    const repostTamer = params.controls?.originalsOnlyForHeavyReposters ?? true;
    const targetSize = Math.min(Math.max(params.targetSize ?? 24, 4), 50);

    const excluded = new Set<string>([
      ...(params.excludedDids ?? []),
      ...(params.seedFollowDids ?? []),
    ]);

    const weights = this.computePosteriorWeights(
      params.selections,
      params.swipes ?? [],
      params.seedFollowDids ?? [],
    );
    const rankedComms = [...weights.entries()]
      .filter(([, w]) => w > 0.05)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 16);
    const maxW = Math.max(...rankedComms.map(([, w]) => w), 1e-6);

    interface Candidate {
      comm: IntelCommunityEntry;
      member: IntelMemberEntry;
      commWeight: number;
    }

    const pool: Candidate[] = [];
    const seen = new Set<string>();

    for (const [cid, w] of rankedComms) {
      const comm = this.commById.get(cid);
      if (!comm) continue;
      const wn = w / maxW;
      for (const m of comm.members) {
        if (seen.has(m.did) || excluded.has(m.did) || !this.isUriAllowed(m.did, params.unauthenticated)) continue;
        seen.add(m.did);
        pool.push({ comm, member: m, commWeight: wn });
      }
    }

    const targetRoleShare: Record<CommunityRole, number> = {
      anchor_creator: 0.34,
      visual_maker: 0.26,
      conversationalist: 0.22,
      scene_curator: 0.12,
      project_or_studio: 0.06,
    };
    const lambdaRole = 0.10;
    const lambdaPace = 0.04;
    const dailyCapPpw = pace === "calm" ? 35.0 : pace === "balanced" ? 70.0 : 140.0;

    const selected: { comm: IntelCommunityEntry; member: IntelMemberEntry; hideReposts: boolean }[] = [];
    const usedDids = new Set<string>();
    const commCounts = new Map<number, number>();
    const roleCounts = new Map<string, number>();
    const selectedCids = new Set<number>();
    let popHi = 0;

    // 1. Honor explicit user-pinned DIDs first
    for (const pinnedDid of params.pinnedDids ?? []) {
      if (selected.length >= targetSize) break;
      if (excluded.has(pinnedDid) || usedDids.has(pinnedDid) || !this.isUriAllowed(pinnedDid, params.unauthenticated)) {
        continue;
      }
      const found = pool.find((p) => p.member.did === pinnedDid) ?? this.didToMember.get(pinnedDid);
      if (!found) continue;
      usedDids.add(pinnedDid);
      if (found.member.followers > 500_000) popHi += 1;
      commCounts.set(found.comm.communityId, (commCounts.get(found.comm.communityId) ?? 0) + 1);
      roleCounts.set(found.member.role, (roleCounts.get(found.member.role) ?? 0) + 1);
      selectedCids.add(found.comm.communityId);
      const hideReposts = repostTamer && found.member.recommendedMode === "originals_only";
      selected.push({ comm: found.comm, member: found.member, hideReposts });
    }

    // 2. Hill-climbed submodular greedy portfolio solver
    while (selected.length < targetSize) {
      let bestIdx = -1;
      let bestGain = -1e9;
      for (let i = 0; i < pool.length; i++) {
        const cand = pool[i]!;
        if (usedDids.has(cand.member.did)) continue;
        if (cand.member.followers > 500_000 && popHi >= 2) continue;

        const cid = cand.comm.communityId;
        // Hard S14 cross-community hostility guard
        let hostileConflict = false;
        for (const scid of selectedCids) {
          if (scid !== cid && this.highHostilitySet.has(pairKey(cid, scid))) {
            hostileConflict = true;
            break;
          }
        }
        if (hostileConflict) continue;

        const role = cand.member.role;
        const base = cand.commWeight * (cand.member.rankScoreBps / 10000.0);
        const commMult = Math.pow(1.0 + (commCounts.get(cid) ?? 0), -0.38);

        const desired = (targetRoleShare[role] ?? 0.15) * targetSize;
        const deficit = Math.max(0.0, (desired - (roleCounts.get(role) ?? 0)) / Math.max(desired, 1.0));
        const roleBonus = lambdaRole * cand.commWeight * deficit;

        const hideRp = repostTamer && cand.member.recommendedMode === "originals_only";
        const effPpw =
          cand.member.estimatedPostsPerWeek *
          (hideRp ? 1.0 - 0.85 * (cand.member.repostShareBps / 10000.0) : 1.0);
        const pacePen = lambdaPace * Math.max(0.0, (effPpw - dailyCapPpw) / dailyCapPpw);

        const gain = base * commMult + roleBonus - pacePen;
        if (gain > bestGain) {
          bestGain = gain;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      const chosen = pool[bestIdx]!;
      usedDids.add(chosen.member.did);
      if (chosen.member.followers > 500_000) popHi += 1;
      commCounts.set(chosen.comm.communityId, (commCounts.get(chosen.comm.communityId) ?? 0) + 1);
      roleCounts.set(chosen.member.role, (roleCounts.get(chosen.member.role) ?? 0) + 1);
      selectedCids.add(chosen.comm.communityId);
      const hideReposts = repostTamer && chosen.member.recommendedMode === "originals_only";
      selected.push({ comm: chosen.comm, member: chosen.member, hideReposts });
    }

    const follows = selected.map(({ comm, member, hideReposts }) => {
      const samplePostUri =
        member.samplePostUri && this.isUriAllowed(member.samplePostUri, params.unauthenticated)
          ? member.samplePostUri
          : undefined;
      return {
        did: member.did,
        communityId: comm.communityId,
        communityTitle: comm.title,
        role: member.role,
        deliveryMode: hideReposts ? ("originals_only" as const) : ("follow_all" as const),
        hideReposts,
        whyPicked: member.whyPicked,
        estimatedPostsPerWeek: member.estimatedPostsPerWeek,
        ...(samplePostUri ? { samplePostUri } : {}),
      };
    });

    // Collect top feeds & starter packs from covered communities
    const feeds: { uri: string; communityId: number; communityTitle: string }[] = [];
    const seenFeeds = new Set<string>();
    const starterPacks: { uri: string; communityId: number; communityTitle: string }[] = [];
    const seenPacks = new Set<string>();

    for (const [cid] of rankedComms) {
      if (!selectedCids.has(cid)) continue;
      const comm = this.commById.get(cid);
      if (!comm) continue;
      for (const uri of comm.feedGeneratorUris) {
        if (!seenFeeds.has(uri) && this.isUriAllowed(uri, params.unauthenticated) && feeds.length < 6) {
          seenFeeds.add(uri);
          feeds.push({ uri, communityId: cid, communityTitle: comm.title });
        }
      }
      for (const uri of comm.starterPackUris) {
        if (!seenPacks.has(uri) && this.isUriAllowed(uri, params.unauthenticated) && starterPacks.length < 4) {
          seenPacks.add(uri);
          starterPacks.push({ uri, communityId: cid, communityTitle: comm.title });
        }
      }
    }

    // Harmonious neighbor communities not yet selected
    const neighborScores = new Map<number, number>();
    for (const cid of selectedCids) {
      const comm = this.commById.get(cid);
      if (!comm) continue;
      for (const nid of comm.neighborCommunityIds) {
        if (selectedCids.has(nid) || !this.commById.has(nid)) continue;
        let hostile = false;
        for (const scid of selectedCids) {
          if (this.highHostilitySet.has(pairKey(nid, scid))) {
            hostile = true;
            break;
          }
        }
        if (!hostile) {
          neighborScores.set(nid, (neighborScores.get(nid) ?? 0) + 1);
        }
      }
    }
    const harmoniousNeighbors = [...neighborScores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([nid]) => this.toCommunitySummary(this.commById.get(nid)!, params.unauthenticated));

    const estimatedDailyPosts = Math.round(
      selected.reduce((sum, { member, hideReposts }) => {
        const mult = hideReposts ? 1.0 - 0.85 * (member.repostShareBps / 10000.0) : 1.0;
        return sum + (member.estimatedPostsPerWeek / 7.0) * mult;
      }, 0),
    );

    const solveMs = Math.max(0, Math.round(performance.now() - t0));
    const sessionId = `onb_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    return {
      sessionId,
      catalogVersion: this.bundle.catalogVersion,
      solveMs,
      summary: {
        totalFollows: follows.length,
        anchorCount: (roleCounts.get("anchor_creator") ?? 0) + (roleCounts.get("project_or_studio") ?? 0),
        makerCount: roleCounts.get("visual_maker") ?? 0,
        conversationalCount: roleCounts.get("conversationalist") ?? 0,
        curatorCount: roleCounts.get("scene_curator") ?? 0,
        originalsOnlyCount: follows.filter((f) => f.hideReposts).length,
        estimatedDailyPosts,
        coveredCommunities: selectedCids.size,
      },
      follows,
      feeds,
      starterPacks,
      harmoniousNeighbors,
    };
  }

  previewTimeline(params: {
    selections: CommunitySelectionInput[];
    swipes?: CalibrationSwipeAnswerInput[];
    controls?: BundleControlsInput;
    limit?: number;
    unauthenticated: boolean;
  }) {
    const solved = this.solveBundle({
      selections: params.selections,
      swipes: params.swipes,
      controls: params.controls,
      targetSize: 24,
      unauthenticated: params.unauthenticated,
    });

    const limit = Math.min(Math.max(params.limit ?? 20, 1), 40);
    const items: { post: string; communityId: number; communityTitle: string; authorRole: string }[] = [];
    const seenUris = new Set<string>();

    // First interleave representative sample posts from the solved lineup's actual authors!
    for (const f of solved.follows) {
      if (items.length >= limit) break;
      if (f.samplePostUri && !seenUris.has(f.samplePostUri) && this.isUriAllowed(f.samplePostUri, params.unauthenticated)) {
        seenUris.add(f.samplePostUri);
        items.push({
          post: f.samplePostUri,
          communityId: f.communityId,
          communityTitle: f.communityTitle,
          authorRole: f.role,
        });
      }
    }

    // Backfill round-robin from covered communities' previewPostUris
    const coveredComms = [...new Set(solved.follows.map((f) => f.communityId))]
      .map((cid) => this.commById.get(cid))
      .filter((c): c is IntelCommunityEntry => Boolean(c));

    let round = 0;
    while (items.length < limit && coveredComms.length > 0) {
      let added = false;
      for (const comm of coveredComms) {
        const uris = comm.previewPostUris.length > 0 ? comm.previewPostUris : comm.samplePostUris;
        if (round < uris.length) {
          const uri = uris[round]!;
          added = true;
          if (!seenUris.has(uri) && this.isUriAllowed(uri, params.unauthenticated)) {
            seenUris.add(uri);
            const owner = ownerDidFromUri(uri) ?? "";
            const member = comm.members.find((m) => m.did === owner);
            items.push({
              post: uri,
              communityId: comm.communityId,
              communityTitle: comm.title,
              authorRole: member?.role ?? "anchor_creator",
            });
            if (items.length >= limit) break;
          }
        }
      }
      if (!added) break;
      round += 1;
    }

    return {
      catalogVersion: this.bundle.catalogVersion,
      items,
      estimatedDailyPosts: solved.summary.estimatedDailyPosts,
    };
  }
}
