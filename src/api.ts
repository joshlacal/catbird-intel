#!/usr/bin/env bun
/** Local serving candidate. Ranking remains heuristic and deployment is unqualified.
 * Public post responses contain identifiers for viewer-aware AppView hydration.
 * No invented observations, unverified authentication, or exposure-budget claims.
 */
import { existsSync } from "node:fs";
import { ClickHouseWriter } from "./clickhouse.ts";
import { createDidKeyResolver } from "./did_resolver.ts";
import {
  S17CatalogEngine,
  type BundleControlsInput,
  type CalibrationSwipeAnswerInput,
  type CommunitySelectionInput,
} from "./intel_catalog.ts";
import { LiveOnboardingEngine, createPublicFollowsFetcher } from "./onboarding_live.ts";
import { AuthenticationError, createServiceAuthenticator, type ServiceAuthenticator } from "./service_auth.ts";
import { SERVING_CONTRACT_VERSION, InvalidRequest, parseControls, passesControls, controlReport,
  recommendationAction, type ServingControls } from "./serving_contract.ts";
import { tidToBigInt } from "./tid.ts";
import {
  DEFAULT_INTEL_VECTOR_STORE_DIR,
  IntelVectorStore,
  cosineI8,
  quantizedBytes,
  type ActorIntelNode,
  type IntelCommunity,
  type QuantizedVector256,
  type WindowedCentroid,
} from "./vector_store.ts";

export interface PostCandidate {
  community: number;
  rank: number;
  item_uri: string;
  did: string;
  rkey: string;
  created_at: string;
  engagers: number;
  expected_engagements: number;
  residual_score: number;
  community_score: number;
  politicalness: number;
  heat: number;
  has_no_unauth_label: number;
  text_preview: string;
}

export interface CommunityMeta {
  community: number;
  scene_id: number;
  domain_id: number;
  size: number;
  label: string;
  seed_subculture: string;
  keywords: string[];
  top_dids: string[];
  top_followers: number[];
}

export interface DirectoryEntry {
  community: number;
  kind: "account" | "feed" | "pack";
  role: string;
  rank: number;
  uri_or_did: string;
  handle_or_name: string;
  description: string;
  influence: number;
  adoption_count: number;
  hub_concentration: number;
  original_share: number;
  political_share: number;
  has_no_unauth_label: number;
}

export interface AuthorModeMix {
  did: string;
  handle: string;
  community: number;
  followers_est: number;
  total_events_14d: number;
  orig_posts: number;
  reply_posts: number;
  quote_posts: number;
  reposts: number;
  orig_political_share: number;
  reply_political_share: number;
  quote_political_share: number;
  repost_political_share: number;
  overall_political_share: number;
  orig_heat_mean: number;
  overall_heat_mean: number;
  follow_originals_only_eligible: number;
  recommend_mode: "Follow all" | "Follow originals only" | "Topic feed only";
  role: string;
  active_days_14d: number;
  has_no_unauth_label: number;
  hide_from_algorithmic: number;
}

export interface Affinity {
  community: number;
  scene_id: number;
  score: number;
  social_proof: number;
}

const ch = new ClickHouseWriter({
  url: process.env.CLICKHOUSE_URL ?? process.env.CH_URL ?? "http://127.0.0.1:8123",
  database: process.env.CLICKHOUSE_DB ?? process.env.CH_DATABASE ?? "catbird",
  user: process.env.CH_USER,
  password: process.env.CH_PASSWORD,
  maxRows: 5000,
});

// In-memory serving state; production memory and latency qualification remain outstanding.
let communitiesById = new Map<number, CommunityMeta>();
let itemsByCommunity = new Map<number, PostCandidate[]>();
let directoryByCommunity = new Map<number, DirectoryEntry[]>();
let subcultureToCommunities = new Map<string, number[]>();
let authorModeByDid = new Map<string, AuthorModeMix>();
let hiddenAlgoDids = new Set<string>();
let noUnauthDids = new Set<string>();
const userAffinityCache = new Map<string, { ts: number; affinities: Affinity[] }>();
let defaultCommunities: number[] = [];
let sourceCommunityVersion: string | null = null;
let cacheRefreshedAt: string | null = null;
let activeCatalogEngine: S17CatalogEngine | null = null;
let activeLiveOnboarding: LiveOnboardingEngine | null = null;
let activeVectorStore: IntelVectorStore | null = null;
const publicFollowsCache = new Map<string, { expiresAt: number; dids: string[] }>();

export function installVectorStore(store: IntelVectorStore | null) {
  activeVectorStore = store;
  publicFollowsCache.clear();
}

export function installCatalogEngine(engine: S17CatalogEngine | null) {
  activeCatalogEngine = engine;
}

/** The live onboarding engine wraps the catalog engine; null serves onboarding from the static catalog. */
export function installLiveOnboardingEngine(engine: LiveOnboardingEngine | null) {
  activeLiveOnboarding = engine;
}

/** Explicit dependency seam for local fixture-backed HTTP tests and future atomic publication. */
export function installServingSnapshot(snapshot: {
  communities: CommunityMeta[]; items: PostCandidate[]; directory: DirectoryEntry[];
  modes?: AuthorModeMix[]; hiddenDids?: string[]; noUnauthDids?: string[]; version: string;
}) {
  activeCatalogEngine = null;
  communitiesById = new Map(snapshot.communities.map((row) => [row.community, row]));
  itemsByCommunity = new Map();
  directoryByCommunity = new Map();
  subcultureToCommunities = new Map();
  for (const row of snapshot.items) itemsByCommunity.set(row.community, [...(itemsByCommunity.get(row.community) ?? []), row]);
  for (const row of snapshot.directory) directoryByCommunity.set(row.community, [...(directoryByCommunity.get(row.community) ?? []), row]);
  for (const row of snapshot.communities) if (row.seed_subculture) {
    subcultureToCommunities.set(row.seed_subculture, [...(subcultureToCommunities.get(row.seed_subculture) ?? []), row.community]);
  }
  authorModeByDid = new Map((snapshot.modes ?? []).map((row) => [row.did, row]));
  hiddenAlgoDids = new Set(snapshot.hiddenDids ?? []);
  noUnauthDids = new Set(snapshot.noUnauthDids ?? []);
  defaultCommunities = [...itemsByCommunity.keys()].slice(0, 12);
  sourceCommunityVersion = snapshot.version;
  cacheRefreshedAt = new Date().toISOString();
  userAffinityCache.clear();
}

function ownerDid(uriOrDid: string): string | null {
  if (uriOrDid.startsWith("did:")) return uriOrDid;
  return /^at:\/\/(did:[^/]+)\//.exec(uriOrDid)?.[1] ?? null;
}
function eligibleOwner(uriOrDid: string, unauthenticated: boolean, noUnauthLabel = 0): boolean {
  const owner = ownerDid(uriOrDid);
  return owner !== null && !hiddenAlgoDids.has(owner) &&
    (!unauthenticated || (noUnauthLabel === 0 && !noUnauthDids.has(owner)));
}
function publicCommunity(row: CommunityMeta) {
  const { top_dids: _dids, top_followers: _followers, ...metadata } = row;
  return metadata;
}
function postSkeleton(post: PostCandidate) { return { post: post.item_uri }; }
function contractMetadata() {
  return { contract_version: SERVING_CONTRACT_VERSION, source_community_version: sourceCommunityVersion,
    derived_generation: activeVectorStore?.manifest.generation ?? null, cache_refreshed_at: cacheRefreshedAt,
    qualification: "local_candidate_unqualified", hydration: "viewer_aware_appview_required" };
}

// Built-in persona presets for hand review & cold-start testing
export const PERSONA_PRESETS: Record<string, string[]> = {
  josh_catbird_builder: ["systems_rust", "apple_swift_indie", "local_ai_ml", "hardware_homelab", "tpot_postrat_poetics"],
  systems_rust_apple_dev: ["systems_rust", "apple_swift_indie", "infosec_networking", "hardware_homelab"],
  sports_sabermetrics_f1: ["baseball_sabermetrics", "nba_f1_sickos", "soccer_fpl_tactics", "hockey_nhl_sky"],
  indie_creative_arts: ["indie_gamedev", "music_vinyl_bandcamp", "film_anime_art", "photo_birding_culinary"],
};

const serveLogBuffer: Record<string, unknown>[] = [];
const onboardingSessionBuffer: Record<string, unknown>[] = [];
const DIAGNOSTIC_BUFFER_LIMIT = 1000;
function bufferDiagnostic(buffer: Record<string, unknown>[], event: Record<string, unknown>) {
  buffer.push({ contract_version: SERVING_CONTRACT_VERSION, ...event });
  if (buffer.length > DIAGNOSTIC_BUFFER_LIMIT) buffer.splice(0, buffer.length - DIAGNOSTIC_BUFFER_LIMIT);
}

async function queryJsonEachRow<T>(sql: string): Promise<T[]> {
  const txt = await ch.query(`${sql} FORMAT JSONEachRow`);
  if (!txt.trim()) return [];
  return txt
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line) as T);
}

export async function refreshServingCache(): Promise<{
  communities: number;
  candidate_posts: number;
  directory_entries: number;
  author_modes: number;
}> {
  const versionTxt = (
    await ch.query("SELECT toString(version) FROM catbird.communities ORDER BY version DESC LIMIT 1 FORMAT TabSeparated")
  ).trim();
  const commRows = await queryJsonEachRow<CommunityMeta>(
    `SELECT community, scene_id, domain_id, size, label, seed_subculture, keywords, top_dids, top_followers
     FROM catbird.communities FINAL
     WHERE version = '${versionTxt}'
     ORDER BY size DESC`,
  );
  const nextComms = new Map<number, CommunityMeta>();
  const nextSubMap = new Map<string, number[]>();
  for (const r of commRows) {
    const cid = Number(r.community);
    nextComms.set(cid, {
      ...r,
      community: cid,
      scene_id: Number(r.scene_id),
      domain_id: Number(r.domain_id),
      size: Number(r.size),
    });
    if (r.seed_subculture) {
      const list = nextSubMap.get(r.seed_subculture) ?? [];
      list.push(cid);
      nextSubMap.set(r.seed_subculture, list);
    }
  }

  const itemRows = await queryJsonEachRow<PostCandidate>(
    `SELECT community, rank, item_uri, did, rkey, toString(created_at) AS created_at,
            engagers, expected_engagements, residual_score, community_score,
            politicalness, heat, has_no_unauth_label, text_preview
     FROM catbird.community_items FINAL
     ORDER BY community ASC, rank ASC`,
  );
  const nextItems = new Map<number, PostCandidate[]>();
  for (const it of itemRows) {
    const cid = Number(it.community);
    const list = nextItems.get(cid) ?? [];
    list.push({
      ...it,
      community: cid,
      rank: Number(it.rank),
      engagers: Number(it.engagers),
      expected_engagements: Number(it.expected_engagements),
      residual_score: Number(it.residual_score),
      community_score: Number(it.community_score),
      politicalness: Number(it.politicalness),
      heat: Number(it.heat),
      has_no_unauth_label: Number(it.has_no_unauth_label ?? 0),
    });
    nextItems.set(cid, list);
  }

  const dirRows = await queryJsonEachRow<DirectoryEntry>(
    `SELECT community, kind, role, rank, uri_or_did, handle_or_name, description,
            influence, adoption_count, hub_concentration, original_share, political_share, has_no_unauth_label
     FROM catbird.community_directory FINAL
     ORDER BY community ASC, kind ASC, role ASC, rank ASC`,
  );
  const nextDir = new Map<number, DirectoryEntry[]>();
  for (const d of dirRows) {
    const cid = Number(d.community);
    const list = nextDir.get(cid) ?? [];
    list.push({
      ...d,
      community: cid,
      rank: Number(d.rank),
      influence: Number(d.influence),
      adoption_count: Number(d.adoption_count),
      hub_concentration: Number(d.hub_concentration),
      original_share: Number(d.original_share),
      political_share: Number(d.political_share),
      has_no_unauth_label: Number(d.has_no_unauth_label ?? 0),
    });
    nextDir.set(cid, list);
  }

  let modeRows: AuthorModeMix[] = [];
  try {
    modeRows = await queryJsonEachRow<AuthorModeMix>(
      `SELECT did, handle, community, followers_est, total_events_14d,
              orig_posts, reply_posts, quote_posts, reposts,
              orig_political_share, reply_political_share, quote_political_share, repost_political_share,
              overall_political_share, orig_heat_mean, overall_heat_mean,
              follow_originals_only_eligible, recommend_mode, role, active_days_14d,
              has_no_unauth_label, hide_from_algorithmic
       FROM catbird.author_mode_mix FINAL`,
    );
  } catch (error) {
    throw new Error(`Author-mode refresh failed: ${String(error)}`);
  }
  const nextModes = new Map<string, AuthorModeMix>();
  const nextHidden = new Set<string>();
  const nextNoUnauth = new Set<string>();
  for (const m of modeRows) {
    const parsed: AuthorModeMix = {
      ...m,
      community: Number(m.community),
      followers_est: Number(m.followers_est),
      total_events_14d: Number(m.total_events_14d),
      orig_posts: Number(m.orig_posts),
      reply_posts: Number(m.reply_posts),
      quote_posts: Number(m.quote_posts),
      reposts: Number(m.reposts),
      orig_political_share: Number(m.orig_political_share),
      reply_political_share: Number(m.reply_political_share),
      quote_political_share: Number(m.quote_political_share),
      repost_political_share: Number(m.repost_political_share),
      overall_political_share: Number(m.overall_political_share),
      orig_heat_mean: Number(m.orig_heat_mean),
      overall_heat_mean: Number(m.overall_heat_mean),
      follow_originals_only_eligible: Number(m.follow_originals_only_eligible),
      active_days_14d: Number(m.active_days_14d),
      has_no_unauth_label: Number(m.has_no_unauth_label ?? 0),
      hide_from_algorithmic: Number(m.hide_from_algorithmic ?? 0),
    };
    nextModes.set(m.did, parsed);
    if (parsed.hide_from_algorithmic === 1) nextHidden.add(m.did);
    if (parsed.has_no_unauth_label === 1) nextNoUnauth.add(m.did);
  }

  // Also pull direct visibility declarations & profile !no-unauthenticated labels from Phase 0 tables
  try {
    const visRows = await queryJsonEachRow<{ did: string }>(
      `SELECT did FROM catbird.visibility FINAL WHERE is_deleted = 0 AND hide_from_algorithmic = 1`,
    );
    for (const v of visRows) nextHidden.add(v.did);
    const profNoUnauth = await queryJsonEachRow<{ did: string }>(
      `SELECT did FROM catbird.profiles FINAL WHERE is_deleted = 0 AND has(self_labels, '!no-unauthenticated')`,
    );
    for (const p of profNoUnauth) nextNoUnauth.add(p.did);
  } catch (error) {
    throw new Error(`Visibility refresh failed: ${String(error)}`);
  }

  communitiesById = nextComms;
  itemsByCommunity = nextItems;
  directoryByCommunity = nextDir;
  subcultureToCommunities = nextSubMap;
  authorModeByDid = nextModes;
  hiddenAlgoDids = nextHidden;
  noUnauthDids = nextNoUnauth;
  defaultCommunities = [...nextItems.entries()]
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 12)
    .map(([cid]) => cid);

  sourceCommunityVersion = versionTxt;
  cacheRefreshedAt = new Date().toISOString();
  userAffinityCache.clear();
  return {
    communities: communitiesById.size,
    candidate_posts: itemRows.length,
    directory_entries: dirRows.length,
    author_modes: nextModes.size,
  };
}

async function resolveAffinities(actorOrPersona: string): Promise<Affinity[]> {
  const presetSubs = PERSONA_PRESETS[actorOrPersona] ?? (subcultureToCommunities.has(actorOrPersona) ? [actorOrPersona] : null);
  if (presetSubs) {
    const out: Affinity[] = [];
    for (const sc of presetSubs) {
      const cids = subcultureToCommunities.get(sc) ?? [];
      for (const cid of cids.slice(0, 4)) {
        const meta = communitiesById.get(cid);
        out.push({
          community: cid,
          scene_id: meta?.scene_id ?? 0,
          score: 1.0 / Math.max(presetSubs.length * Math.min(cids.length, 4), 1),
          social_proof: 5,
        });
      }
    }
    if (out.length > 0) return out;
  }

  const cached = userAffinityCache.get(actorOrPersona);
  if (cached && Date.now() - cached.ts < 300_000) {
    return cached.affinities;
  }

  if (actorOrPersona.startsWith("did:")) {
    const safeDid = actorOrPersona.replace(/[^a-zA-Z0-9:._-]/g, "");
    const rows = await queryJsonEachRow<Affinity>(
      `SELECT community, scene_id, score, social_proof
       FROM catbird.interested_in FINAL
       WHERE did = '${safeDid}'
       ORDER BY rank ASC
       LIMIT 25`,
    );
    if (rows.length > 0) {
      const parsed = rows.map((r) => ({
        community: Number(r.community),
        scene_id: Number(r.scene_id),
        score: Number(r.score),
        social_proof: Number(r.social_proof),
      }));
      userAffinityCache.set(actorOrPersona, { ts: Date.now(), affinities: parsed });
      return parsed;
    }
  }

  const fallback: Affinity[] = defaultCommunities.slice(0, 8).map((cid, idx) => ({
    community: cid,
    scene_id: communitiesById.get(cid)?.scene_id ?? 0,
    score: Math.pow(0.85, idx),
    social_proof: 2,
  }));
  return fallback;
}

export async function scoreFeedForActor(
  actor: string,
  limit: number,
  cursorOffset: number,
  controls: ServingControls = {},
  unauthenticated = true,
) {
  const t0 = performance.now();
  const affinities = await resolveAffinities(actor);

  const candidateScores = new Map<
    string,
    {
      cand: PostCandidate;
      totalScore: number;
      matchedCommunities: { community: number; label: string; weight: number }[];
    }
  >();

  for (const aff of affinities) {
    const posts = itemsByCommunity.get(aff.community);
    if (!posts) continue;
    const commLabel = communitiesById.get(aff.community)?.label ?? `Community #${aff.community}`;
    for (let i = 0; i < Math.min(posts.length, 60); i++) {
      const p = posts[i]!;
      if (hiddenAlgoDids.has(p.did)) continue;
      if (unauthenticated && (p.has_no_unauth_label === 1 || noUnauthDids.has(p.did))) continue;
      if (!passesControls(p, controls)) continue;
      const contrib = aff.score * Math.max(p.residual_score, 0.05);
      const prev = candidateScores.get(p.item_uri);
      if (prev) {
        prev.totalScore += contrib * 0.65;
        prev.matchedCommunities.push({ community: aff.community, label: commLabel, weight: Number(contrib.toFixed(4)) });
      } else {
        candidateScores.set(p.item_uri, {
          cand: p,
          totalScore: contrib,
          matchedCommunities: [{ community: aff.community, label: commLabel, weight: Number(contrib.toFixed(4)) }],
        });
      }
    }
  }

  const rawList = [...candidateScores.values()].sort((a, b) => b.totalScore - a.totalScore);
  const authorCounts = new Map<string, number>();
  const commCounts = new Map<number, number>();
  const reranked = rawList
    .map((entry) => {
      const aCount = authorCounts.get(entry.cand.did) ?? 0;
      const cId = entry.matchedCommunities[0]?.community ?? entry.cand.community;
      const cCount = commCounts.get(cId) ?? 0;
      authorCounts.set(entry.cand.did, aCount + 1);
      commCounts.set(cId, cCount + 1);
      const fatigue = Math.pow(0.45, aCount) * Math.pow(0.88, cCount);
      return { ...entry, finalScore: entry.totalScore * fatigue };
    })
    .sort((a, b) => b.finalScore - a.finalScore);

  const slice = reranked.slice(cursorOffset, cursorOffset + limit);
  const computeMs = performance.now() - t0;

  const policy = controlReport(slice.map((entry) => entry.cand), controls);
  bufferDiagnostic(serveLogBuffer, {
    ts: new Date().toISOString().replace("T", " ").replace("Z", ""),
    requester_did: actor,
    persona: PERSONA_PRESETS[actor] ? actor : "live_user",
    items_served: slice.length,
    control_policy: policy,
    compute_ms: Number(computeMs.toFixed(3)),
  });

  const nextCursor = cursorOffset + slice.length < reranked.length ? String(cursorOffset + slice.length) : undefined;
  return {
    cursor: nextCursor,
    feed: slice.map((entry) => postSkeleton(entry.cand)),
    meta: { actor, compute_ms: Number(computeMs.toFixed(3)), control_policy: policy },
  };
}

// ============================================================================
// Phase 4: Active Onboarding Engine (Scene Cards, Calibration Swipes, Preview, Portfolio Solver)
// ============================================================================

export type OnboardingControls = ServingControls;

export interface SwipeChoice {
  swipe_id: string;
  choice: "a" | "b" | "both" | "neither";
  community_a: number;
  community_b: number;
}

const COLLISION_TRAP_SWIPES = [
  {
    swipe_id: "trap_1_rust",
    prompt: "Which neighborhood feels closer to what you want on your timeline?",
    subculture_a: "systems_rust",
    label_a: "Systems Programming, Rust & Compilers (cargo, tokio, async, OS internals)",
    subculture_b: "indie_gamedev",
    label_b: "Multiplayer Survival Games, Wipe Day & Base Building",
    is_consistency_probe: false,
  },
  {
    swipe_id: "trap_2_swift",
    prompt: "When you see 'Swift', which world are you looking for?",
    subculture_a: "apple_swift_indie",
    label_a: "iOS/macOS App Craft, SwiftUI, Xcode & Indie Apple Developers",
    subculture_b: "hangout_celebs_comedy",
    label_b: "Pop Culture, Eras Tour Fandom & Music Stan Timelines",
    is_consistency_probe: false,
  },
  {
    swipe_id: "trap_3_macro",
    prompt: "Pick the 'Macro' lens you actually enjoy reading:",
    subculture_a: "fintwit_econsky",
    label_a: "Macroeconomics, Fed Policy, Bond Yields & Labor Markets",
    subculture_b: "photo_birding_culinary",
    label_b: "90mm Macro Lens Photography, Entomology, Birding & Film Cameras",
    is_consistency_probe: false,
  },
  {
    swipe_id: "trap_4_lab",
    prompt: "Which 'Lab' belongs in your starter bundle?",
    subculture_a: "hardware_homelab",
    label_a: "Self-Hosted Homelabs, Proxmox, Kubernetes, Tailscale & NAS",
    subculture_b: "photo_birding_culinary",
    label_b: "Golden Retrievers, Labrador Pets & Outdoors Nature Walks",
    is_consistency_probe: false,
  },
  {
    swipe_id: "trap_5_humor",
    prompt: "What kind of humor & commentary fits your vibe?",
    subculture_a: "weird_shitposting",
    label_a: "Absurdist / Post-Ironic Shitposting & Weird Internet Humor",
    subculture_b: "tpot_postrat_poetics",
    label_b: "Reflective Essays, Poetics, Systems Theory & Meditative Notes",
    is_consistency_probe: false,
  },
  {
    swipe_id: "trap_6_espresso",
    prompt: "Morning routine check:",
    subculture_a: "photo_birding_culinary",
    label_a: "Specialty Espresso, Light-Roast Pourover, Sourdough & Culinary Craft",
    subculture_b: "urbanism_transit",
    label_b: "Transit Headways, Bike Lanes, Housing Policy & Walkable Cities",
    is_consistency_probe: false,
  },
  {
    swipe_id: "trap_7_football",
    prompt: "Matchday preference:",
    subculture_a: "soccer_fpl_tactics",
    label_a: "World Football / Soccer Tactics, xG Models & Fantasy Premier League",
    subculture_b: "nba_f1_sickos",
    label_b: "NBA Cap Nerds, F1 Telemetry & North American Pro Sports",
    is_consistency_probe: false,
  },
  {
    swipe_id: "trap_8_indie",
    prompt: "Creative indie scene:",
    subculture_a: "indie_gamedev",
    label_a: "Godot / Love2D Shaders, Pixel Art & Solo Game Development",
    subculture_b: "music_vinyl_bandcamp",
    label_b: "Bandcamp Fridays, Vinyl Records, Modular Synths & Indie Labels",
    is_consistency_probe: false,
  },
];

function buildSceneCards(unauthenticated: boolean) {
  // Coverage-aware cards across all 26 seed subcultures + top emergent communities
  const cards: Record<string, unknown>[] = [];
  for (const [subculture, cids] of subcultureToCommunities.entries()) {
    const primaryCid = cids[0];
    if (primaryCid === undefined) continue;
    const meta = communitiesById.get(primaryCid);
    if (!meta) continue;

    // Medoid / representative non-heated sample posts respecting Hard Invariant #5
    const samplePosts = (itemsByCommunity.get(primaryCid) ?? [])
      .filter((p) => {
        if (hiddenAlgoDids.has(p.did)) return false;
        if (unauthenticated && (p.has_no_unauth_label === 1 || noUnauthDids.has(p.did))) return false;
        return true;
      })
      .slice(0, 3)
      .map((p) => ({
        post: p.item_uri,
      }));

    const sampleAccounts = (directoryByCommunity.get(primaryCid) ?? [])
      .filter((d) => {
        if (d.kind !== "account") return false;
        if (hiddenAlgoDids.has(d.uri_or_did)) return false;
        if (unauthenticated && (d.has_no_unauth_label === 1 || noUnauthDids.has(d.uri_or_did))) return false;
        return true;
      })
      .slice(0, 4)
      .map((d) => ({
        did: d.uri_or_did,
        role: d.role,
      }));

    cards.push({
      card_id: subculture,
      primary_community: primaryCid,
      related_communities: cids.slice(0, 4),
      scene_id: meta.scene_id,
      domain_id: meta.domain_id,
      title: meta.label,
      keywords: meta.keywords.slice(0, 8),
      producer_count: meta.size,
      sample_posts: samplePosts,
      sample_accounts: sampleAccounts,
    });
  }
  return {
    point_budget: 6,
    unauthenticated,
    total_cards: cards.length,
    cards,
  };
}

function buildCalibrationSwipes(allocation: Record<string, number>, unauthenticated: boolean) {
  const resolveCid = (sub: string): number => subcultureToCommunities.get(sub)?.[0] ?? defaultCommunities[0] ?? 0;
  const pickMedoid = (cid: number) => {
    const posts = (itemsByCommunity.get(cid) ?? []).filter((p) => {
      if (hiddenAlgoDids.has(p.did)) return false;
      if (unauthenticated && (p.has_no_unauth_label === 1 || noUnauthDids.has(p.did))) return false;
      return true;
    });
    return posts[0]
      ? { post: posts[0].item_uri }
      : null;
  };

  // Prioritize swipes touching the user's allocated subcultures first, then fill to 12 swipes + 2 consistency probes
  const allocatedKeys = new Set(Object.keys(allocation).filter((k) => (allocation[k] ?? 0) > 0));
  const sortedTraps = [...COLLISION_TRAP_SWIPES].sort((a, b) => {
    const aHit = (allocatedKeys.has(a.subculture_a) ? 1 : 0) + (allocatedKeys.has(a.subculture_b) ? 1 : 0);
    const bHit = (allocatedKeys.has(b.subculture_a) ? 1 : 0) + (allocatedKeys.has(b.subculture_b) ? 1 : 0);
    return bHit - aHit;
  });

  const swipes = sortedTraps.map((t, idx) => {
    const cidA = resolveCid(t.subculture_a);
    const cidB = resolveCid(t.subculture_b);
    return {
      swipe_id: t.swipe_id,
      step: idx + 1,
      prompt: t.prompt,
      neither_allowed: true,
      both_allowed: true,
      is_consistency_probe: false,
      option_a: {
        community: cidA,
        subculture: t.subculture_a,
        label: t.label_a,
        sample_post: pickMedoid(cidA),
      },
      option_b: {
        community: cidB,
        subculture: t.subculture_b,
        label: t.label_b,
        sample_post: pickMedoid(cidB),
      },
    };
  });

  // Add 4 coverage-aware sub-community splitters + 2 consistency probes (total 12 swipes)
  const allocList = [...allocatedKeys];
  for (let i = 0; i < 4; i++) {
    const sub = allocList[i % Math.max(allocList.length, 1)] ?? "systems_rust";
    const cids = subcultureToCommunities.get(sub) ?? defaultCommunities;
    const cidA = cids[0] ?? 0;
    const cidB = cids[1] ?? cids[0] ?? 1;
    const isProbe = i >= 2;
    swipes.push({
      swipe_id: isProbe ? `consistency_probe_${i - 1}` : `depth_split_${i + 1}`,
      step: swipes.length + 1,
      prompt: isProbe
        ? "Consistency check: which post style would you rather see every morning?"
        : `Within ${sub}, do you prefer deep technical/craft threads or conversational hangout?`,
      neither_allowed: true,
      both_allowed: true,
      is_consistency_probe: isProbe,
      option_a: {
        community: cidA,
        subculture: sub,
        label: communitiesById.get(cidA)?.label ?? `Primary ${sub}`,
        sample_post: pickMedoid(cidA),
      },
      option_b: {
        community: cidB,
        subculture: sub,
        label: communitiesById.get(cidB)?.label ?? `Secondary ${sub}`,
        sample_post: pickMedoid(cidB),
      },
    });
  }

  return {
    total_swipes: swipes.length,
    consistency_probes: 2,
    swipes,
  };
}

function computePosteriorAffinities(
  allocation: Record<string, number>,
  swipes: SwipeChoice[],
): Affinity[] {
  const weights = new Map<number, number>();
  const addW = (cid: number, w: number) => {
    if (!communitiesById.has(cid)) return;
    weights.set(cid, Math.max((weights.get(cid) ?? 0) + w, 0));
  };

  // 1. 6-point scene card allocation prior
  let totalPoints = 0;
  for (const [key, pts] of Object.entries(allocation)) {
    const p = Math.max(Number(pts) || 0, 0);
    if (p <= 0) continue;
    totalPoints += p;
    if (subcultureToCommunities.has(key)) {
      const cids = subcultureToCommunities.get(key)!;
      cids.slice(0, 4).forEach((cid, idx) => addW(cid, p * Math.pow(0.75, idx)));
    } else if (!Number.isNaN(Number(key))) {
      addW(Number(key), p);
    }
  }
  if (totalPoints === 0) {
    for (const sc of PERSONA_PRESETS.josh_catbird_builder!) {
      const cids = subcultureToCommunities.get(sc) ?? [];
      cids.slice(0, 2).forEach((cid) => addW(cid, 1.5));
    }
  }

  // 2. Fixed additive swipe heuristic, including a negative update for neither
  for (const sw of swipes ?? []) {
    if (sw.choice === "a") {
      addW(sw.community_a, 1.25);
      addW(sw.community_b, -0.45);
    } else if (sw.choice === "b") {
      addW(sw.community_b, 1.25);
      addW(sw.community_a, -0.45);
    } else if (sw.choice === "both") {
      addW(sw.community_a, 0.75);
      addW(sw.community_b, 0.75);
    } else if (sw.choice === "neither") {
      addW(sw.community_a, -0.85);
      addW(sw.community_b, -0.85);
    }
  }

  const entries = [...weights.entries()]
    .filter(([, w]) => w > 0.05)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 16);
  const sumW = entries.reduce((acc, [, w]) => acc + w, 0) || 1.0;

  return entries.map(([cid, w]) => ({
    community: cid,
    scene_id: communitiesById.get(cid)?.scene_id ?? 0,
    score: Number((w / sumW).toFixed(4)),
    social_proof: 5,
  }));
}

export function solveOnboardingBundle(
  allocation: Record<string, number>,
  swipes: SwipeChoice[],
  controls: OnboardingControls,
  unauthenticated = true,
  targetSize = 25,
  hideReposts = false,
  deliveryModes: Record<string, "follow" | "hide_reposts" | "topic_feed"> = {},
) {
  const t0 = performance.now();
  const posterior = computePosteriorAffinities(allocation, swipes);
  const curatorShare = 0.15;
  const creatorShare = 0.40;
  const convShare = 0.45;

  const targetQuotas: Record<string, number> = {
    creator: Math.round(targetSize * creatorShare),
    conversational: Math.round(targetSize * convShare),
    curator: Math.max(targetSize - Math.round(targetSize * creatorShare) - Math.round(targetSize * convShare), 1),
  };

  interface BundleAccount {
    did: string;
    handle: string;
    community: number;
    community_label: string;
    role: string;
    mode_label: "Follow all" | "Follow originals only" | "Topic feed only";
    influence: number;
    utility_score: number;
  }

  const candidateMap = new Map<string, BundleAccount>();
  const feedsPool: DirectoryEntry[] = [];
  const packsPool: DirectoryEntry[] = [];

  for (const aff of posterior) {
    const dir = directoryByCommunity.get(aff.community) ?? [];
    const commLabel = communitiesById.get(aff.community)?.label ?? `Community #${aff.community}`;
    for (const d of dir) {
      if (!eligibleOwner(d.uri_or_did, unauthenticated, d.has_no_unauth_label)) continue;
      if (d.kind === "feed") {
        feedsPool.push(d);
        continue;
      }
      if (d.kind === "pack") {
        packsPool.push(d);
        continue;
      }
      const did = d.uri_or_did;
      // Enforce Hard Invariant #5: never include hide_from_algorithmic, and never include !no-unauthenticated when unauthenticated
      if (hiddenAlgoDids.has(did)) continue;
      if (unauthenticated && (d.has_no_unauth_label === 1 || noUnauthDids.has(did))) continue;

      const modeInfo = authorModeByDid.get(did);
      if (modeInfo) {
        if (modeInfo.hide_from_algorithmic === 1) continue;
        if (unauthenticated && modeInfo.has_no_unauth_label === 1) continue;
      }

      // Delivery choices are explicit requests, independent of unqualified health scores.
      const requestedMode = deliveryModes[did] ?? "follow";
      const modeLabel = requestedMode === "topic_feed" ? "Topic feed only"
        : requestedMode === "hide_reposts" ? "Follow originals only" : "Follow all";
      const utility = aff.score * Math.log1p(Math.max(d.influence, 0.5));

      const prev = candidateMap.get(did);
      if (!prev || utility > prev.utility_score) {
        candidateMap.set(did, {
          did,
          handle: modeInfo?.handle && !modeInfo.handle.startsWith("did:") ? modeInfo.handle : d.handle_or_name,
          community: aff.community,
          community_label: commLabel,
          role: d.role || modeInfo?.role || "creator",
          mode_label: modeLabel,
          influence: Number(d.influence.toFixed(3)),
          utility_score: Number(utility.toFixed(4)),
        });
      }
    }
  }

  // Two-pass utility heuristic with role/community caps on the first pass
  const allCands = [...candidateMap.values()].sort((a, b) => b.utility_score - a.utility_score);
  const selected: BundleAccount[] = [];
  const roleCounts: Record<string, number> = { creator: 0, conversational: 0, curator: 0 };
  const commCounts = new Map<number, number>();

  for (const pass of [1, 2]) {
    for (const c of allCands) {
      if (selected.length >= targetSize) break;
      if (selected.some((s) => s.did === c.did)) continue;
      const r = c.role in targetQuotas ? c.role : "creator";
      const cCount = commCounts.get(c.community) ?? 0;
      if (pass === 1) {
        if ((roleCounts[r] ?? 0) >= (targetQuotas[r] ?? 10)) continue;
        if (cCount >= 4) continue;
      }
      selected.push(c);
      roleCounts[r] = (roleCounts[r] ?? 0) + 1;
      commCounts.set(c.community, cCount + 1);
    }
  }

  // Deduplicate and pick top 4 feeds and top 3 starter packs
  const uniqueFeeds = [...new Map(feedsPool.map((f) => [f.uri_or_did, f])).values()]
    .sort((a, b) => b.influence - a.influence)
    .slice(0, 4)
    .map((f) => ({
      uri: f.uri_or_did,
      community: f.community,
      community_label: communitiesById.get(f.community)?.label ?? `Community #${f.community}`,
    }));

  const uniquePacks = [...new Map(packsPool.map((p) => [p.uri_or_did, p])).values()]
    .sort((a, b) => b.influence - a.influence)
    .slice(0, 3)
    .map((p) => ({
      uri: p.uri_or_did,
      community: p.community,
      community_label: communitiesById.get(p.community)?.label ?? `Community #${p.community}`,
    }));

  const actions = selected.map((account) => {
    const action = recommendationAction(account.did, account.community, account.mode_label);
    if (action.type === "follow_account" && hideReposts) action.client_preferences.hideReposts = true;
    return action;
  });
  const follows = actions.filter((action) => action.type === "follow_account");
  const topicFeeds = actions.filter((action) => action.type === "topic_feed_intent");

  const solveMs = Number((performance.now() - t0).toFixed(3));
  const sessionId = `onb_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  const bundleResult = {
    session_id: sessionId,
    solve_ms: solveMs,
    unauthenticated,
    controls,
    actions_applied: false,
    delivery_preferences_source: "explicit_request",
    posterior_communities: posterior.map((p) => ({
      ...p,
      label: communitiesById.get(p.community)?.label ?? `Community #${p.community}`,
    })),
    role_summary: {
      total_follows: follows.length,
      creators: roleCounts.creator ?? 0,
      conversational: roleCounts.conversational ?? 0,
      curators: roleCounts.curator ?? 0,
      hide_reposts_count: follows.filter((action) => action.client_preferences.hideReposts).length,
      topic_feed_only_count: topicFeeds.length,
    },
    actions,
    follows,
    topic_feeds: topicFeeds,
    feeds: uniqueFeeds,
    starter_packs: uniquePacks,
  };

  bufferDiagnostic(onboardingSessionBuffer, {
    session_id: sessionId,
    created_at: new Date().toISOString().replace("T", " ").replace("Z", ""),
    persona: Object.keys(allocation).slice(0, 3).join("+") || "default",
    unauthenticated: unauthenticated ? 1 : 0,
    allocation_json: JSON.stringify(allocation),
    swipes_json: JSON.stringify(swipes ?? []),
    controls_json: JSON.stringify(controls ?? {}),
    posterior_json: JSON.stringify(posterior),
    bundle_json: JSON.stringify({
      actions,
      feeds: uniqueFeeds.map((f) => f.uri),
      packs: uniquePacks.map((p) => p.uri),
    }),
    solve_ms: solveMs,
  });

  return bundleResult;
}

export function pendingTelemetryForTests() {
  return { feeds: [...serveLogBuffer], onboarding: [...onboardingSessionBuffer] };
}

export async function flushTelemetry() {
  // Existing database tables contain invented outcome fields and unqualified score aggregates.
  // Do not silently fill their omitted fields with defaults. A versioned migration must precede writes.
  return { persisted: false, reason: "legacy_schema_incompatible", retention: "bounded_process_memory" };
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new InvalidRequest(`${name} must be an object`);
  return value as Record<string, unknown>;
}

function integer(value: unknown, fallback: number, min: number, max: number): number {
  const n = value === null || value === undefined ? fallback : Number(value);
  if (!Number.isInteger(n) || n < min || n > max) throw new InvalidRequest(`Expected an integer from ${min} to ${max}`);
  return n;
}

function allocationInput(value: unknown): Record<string, number> {
  if (value === undefined) return {};
  const allocation = record(value, "allocation");
  if (Object.keys(allocation).length > 6) throw new InvalidRequest("At most six allocated scenes");
  const parsed: Record<string, number> = {};
  let total = 0;
  for (const [key, points] of Object.entries(allocation)) {
    if (!subcultureToCommunities.has(key) && !communitiesById.has(Number(key))) throw new InvalidRequest("Unknown scene");
    if (typeof points !== "number") throw new InvalidRequest("Allocation points must be numbers");
    parsed[key] = integer(points, 0, 0, 6);
    total += parsed[key]!;
  }
  if (total > 6) throw new InvalidRequest("Allocation exceeds the six-point budget");
  return parsed;
}

function swipeInput(value: unknown, allocation: Record<string, number>, unauthenticated: boolean): SwipeChoice[] {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > 15) throw new InvalidRequest("At most fifteen swipes");
  const offered = new Map(buildCalibrationSwipes(allocation, unauthenticated).swipes.map((swipe) => [swipe.swipe_id, swipe]));
  const seen = new Set<string>();
  return value.map((raw) => {
    const swipe = record(raw, "swipe");
    if (typeof swipe.swipe_id !== "string" || seen.has(swipe.swipe_id)) throw new InvalidRequest("Missing or repeated swipe ID");
    const original = offered.get(swipe.swipe_id);
    if (!original || swipe.community_a !== original.option_a.community || swipe.community_b !== original.option_b.community) {
      throw new InvalidRequest("Swipe does not match the offered community pair");
    }
    if (!["a", "b", "both", "neither"].includes(String(swipe.choice))) throw new InvalidRequest("Invalid swipe choice");
    seen.add(swipe.swipe_id);
    return swipe as unknown as SwipeChoice;
  });
}

async function bodyInput(request: Request): Promise<Record<string, unknown>> {
  if (!request.headers.get("content-type")?.startsWith("application/json")) throw new InvalidRequest("Expected application/json");
  const text = await request.text();
  if (Buffer.byteLength(text) > 65536) throw new InvalidRequest("Request body exceeds 64 KiB");
  try { return record(JSON.parse(text), "body"); }
  catch (error) {
    if (error instanceof InvalidRequest) throw error;
    throw new InvalidRequest("Malformed JSON");
  }
}

function deliveryInput(body: Record<string, unknown>) {
  const preferences = body.preferences === undefined ? {} : record(body.preferences, "preferences");
  if (Object.keys(preferences).some((key) => key !== "hideReposts") ||
      (preferences.hideReposts !== undefined && typeof preferences.hideReposts !== "boolean")) {
    throw new InvalidRequest("preferences supports only boolean hideReposts");
  }
  const modes = body.delivery_modes === undefined ? {} : record(body.delivery_modes, "delivery_modes");
  if (Object.keys(modes).length > 100) throw new InvalidRequest("Too many delivery modes");
  for (const [did, mode] of Object.entries(modes)) {
    if (!/^did:(plc|web):[^#\s]+$/.test(did) || !["follow", "hide_reposts", "topic_feed"].includes(String(mode))) {
      throw new InvalidRequest("Invalid delivery mode");
    }
  }
  return { hideReposts: preferences.hideReposts === true, modes: modes as Record<string, "follow" | "hide_reposts" | "topic_feed"> };
}

function parseDidArray(value: unknown, fieldName: string, maxLen = 100): string[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value) || value.length > maxLen) {
    throw new InvalidRequest(`${fieldName} must be an array of at most ${maxLen} DIDs`);
  }
  return value.map((item) => {
    if (typeof item !== "string" || !/^did:(plc|web):[^#\s]+$/.test(item)) {
      throw new InvalidRequest(`Invalid DID in ${fieldName}`);
    }
    return item;
  });
}

function parseSelectionsInput(value: unknown): CommunitySelectionInput[] {
  if (!Array.isArray(value) || value.length > 24) {
    throw new InvalidRequest("selections must be an array of at most 24 items");
  }
  return value.map((raw) => {
    const obj = record(raw, "selection");
    const communityId = integer(obj.communityId, -1, 0, 0xffffffff);
    const weight = integer(obj.weight, 1, 1, 2);
    return { communityId, weight };
  });
}

function parseCanonicalSwipesInput(value: unknown): CalibrationSwipeAnswerInput[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value) || value.length > 16) {
    throw new InvalidRequest("swipes must be an array of at most 16 items");
  }
  return value.map((raw) => {
    const obj = record(raw, "swipe");
    if (typeof obj.swipeId !== "string" || obj.swipeId.length === 0 || obj.swipeId.length > 80) {
      throw new InvalidRequest("Invalid swipeId");
    }
    const communityA = integer(obj.communityA, -1, 0, 0xffffffff);
    const communityB = integer(obj.communityB, -1, 0, 0xffffffff);
    const choice = String(obj.choice);
    if (!["a", "b", "both", "neither"].includes(choice)) {
      throw new InvalidRequest("Invalid choice");
    }
    return {
      swipeId: obj.swipeId,
      communityA,
      communityB,
      choice: choice as "a" | "b" | "both" | "neither",
    };
  });
}

function parseCanonicalControlsInput(value: unknown): BundleControlsInput | undefined {
  if (value === undefined || value === null) return undefined;
  const obj = record(value, "controls");
  const pace = String(obj.pace ?? "balanced");
  if (!["calm", "balanced", "lively"].includes(pace)) {
    throw new InvalidRequest("Invalid controls.pace");
  }
  if (typeof obj.originalsOnlyForHeavyReposters !== "boolean") {
    throw new InvalidRequest("controls.originalsOnlyForHeavyReposters must be boolean");
  }
  return {
    pace: pace as "calm" | "balanced" | "lively",
    originalsOnlyForHeavyReposters: obj.originalsOnlyForHeavyReposters,
  };
}

type JsonRow = Record<string, unknown>;
type ClickHouseQuery = (sql: string) => Promise<JsonRow[] | null>;
type AppViewFetch = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;
const V4_METHODS: Record<string, true> = {
  "blue.catbird.intel.getGraphNeighborhoodBundle": true,
  "blue.catbird.intel.getActorVectors": true,
  "blue.catbird.intel.getPostVectors": true,
  "blue.catbird.intel.searchGraphPosts": true,
  "blue.catbird.intel.expandGraphNeighborhood": true,
  "blue.catbird.intel.getQuoteChain": true,
  "blue.catbird.intel.getTopicClusters": true,
};

interface V4Dependencies {
  store: IntelVectorStore;
  viewer: string;
  query: ClickHouseQuery;
  appViewFetch: AppViewFetch;
  now: () => number;
}

interface Neighborhood {
  actors: Map<string, { hopDistance: number; mutualFollowCount: number; mutualEngagerCount: number }>;
  firstHop: string[];
  engagementWeights: Map<string, number>;
  clickHouseAvailable: boolean;
}

interface RawPost {
  uri: string;
  authorDid: string;
  indexedAt: string;
  text?: string;
  quoteUri?: string;
  replyRootUri?: string;
  hasImages?: boolean;
  replyCount?: number;
  quoteCount?: number;
  likeRepostCount?: number;
  socialEndorserCount?: number;
  rageBaitPermille?: number;
  controversyPermille?: number;
  communityId?: number;
}

interface PostIntelSkeleton {
  uri: string;
  authorDid: string;
  indexedAt: string;
  hopDistance?: number;
  socialEndorserCount?: number;
  similarityPermille?: number;
  rageBaitPermille?: number;
  controversyPermille?: number;
  replyCount?: number;
  quoteCount?: number;
  likeRepostCount?: number;
  quoteUri?: string;
  replyRootUri?: string;
  hasImages?: boolean;
  communityId?: number;
  vector?: QuantizedVector256;
}

const PUBLIC_APPVIEW = "https://public.api.bsky.app";
let clickHouseUnavailableUntil = 0;

async function optionalClickHouseQuery(sql: string): Promise<JsonRow[] | null> {
  if (process.env.INTEL_SKIP_CLICKHOUSE === "1" || Date.now() < clickHouseUnavailableUntil) return null;
  try {
    return await queryJsonEachRow<JsonRow>(sql);
  } catch {
    clickHouseUnavailableUntil = Date.now() + 30_000;
    return null;
  }
}

function sqlString(value: string): string {
  return `'${value.replaceAll("\\", "\\\\").replaceAll("'", "\\'")}'`;
}

function plainObject(value: unknown): JsonRow {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRow : {};
}

function stringValue(row: JsonRow, ...keys: string[]): string | undefined {
  for (const key of keys) if (typeof row[key] === "string") return row[key] as string;
  return undefined;
}

function numberValue(row: JsonRow, ...keys: string[]): number | undefined {
  for (const key of keys) {
    const value = Number(row[key]);
    if (row[key] !== undefined && Number.isFinite(value)) return value;
  }
  return undefined;
}

function parsePostUri(uri: string): { did: string; rkey: string } | null {
  const match = /^at:\/\/(did:[a-z0-9]+:[^/\s]+)\/app\.bsky\.feed\.post\/([^/?#\s]+)$/i.exec(uri);
  return match ? { did: match[1]!, rkey: match[2]! } : null;
}

function indexedAtFromPostUri(uri: string): string {
  const rkey = parsePostUri(uri)?.rkey;
  const tid = rkey ? tidToBigInt(rkey) : null;
  if (tid !== null) {
    const milliseconds = Number(tid >> 10n) / 1000;
    if (Number.isFinite(milliseconds) && milliseconds >= Date.UTC(2023, 0, 1)) return new Date(milliseconds).toISOString();
  }
  return new Date(0).toISOString();
}

function quantizedVectorInput(value: unknown): QuantizedVector256 {
  const input = record(value, "queryVector");
  if (typeof input.modelId !== "string" || input.modelId.length < 1 || input.modelId.length > 64) {
    throw new InvalidRequest("queryVector.modelId must contain 1..64 characters");
  }
  const q8Bytes = record(input.q8Bytes, "queryVector.q8Bytes");
  const vector: QuantizedVector256 = {
    modelId: input.modelId,
    dims: integer(input.dims, 256, 256, 256) as 256,
    scaleFp16: integer(input.scaleFp16, 0, 0, 0xffff),
    q8Bytes: { $bytes: typeof q8Bytes.$bytes === "string" ? q8Bytes.$bytes : "" },
  };
  try { quantizedBytes(vector); }
  catch { throw new InvalidRequest("queryVector must contain exactly 256 signed int8 bytes"); }
  return vector;
}

async function appViewJson(
  fetcher: AppViewFetch,
  method: string,
  params: Record<string, string | number | string[] | undefined>,
): Promise<JsonRow | null> {
  const url = new URL(`/xrpc/${method}`, PUBLIC_APPVIEW);
  for (const [name, value] of Object.entries(params)) {
    if (value === undefined) continue;
    if (Array.isArray(value)) for (const item of value) url.searchParams.append(name, item);
    else url.searchParams.set(name, String(value));
  }
  try {
    const response = await fetcher(url, { headers: { accept: "application/json" }, signal: AbortSignal.timeout(3_000) });
    if (!response.ok) return null;
    return plainObject(await response.json());
  } catch {
    return null;
  }
}

async function publicFollows(viewer: string, fetcher: AppViewFetch): Promise<string[]> {
  const cached = publicFollowsCache.get(viewer);
  if (cached && cached.expiresAt > Date.now()) return cached.dids;
  const dids: string[] = [];
  let cursor: string | undefined;
  for (let page = 0; page < 50; page++) {
    const response = await appViewJson(fetcher, "app.bsky.graph.getFollows", { actor: viewer, limit: 100, cursor });
    if (!response) break;
    for (const raw of Array.isArray(response.follows) ? response.follows : []) {
      const did = stringValue(plainObject(raw), "did");
      if (did?.startsWith("did:")) dids.push(did);
    }
    cursor = typeof response.cursor === "string" ? response.cursor : undefined;
    if (!cursor) break;
  }
  const unique = [...new Set(dids)];
  publicFollowsCache.set(viewer, { expiresAt: Date.now() + 600_000, dids: unique });
  return unique;
}

const neighborhoodCache = new Map<string, { expiresAt: number; value: Neighborhood }>();

async function resolveNeighborhood(
  deps: V4Dependencies,
  seedDid: string,
  maxTwoHop: number,
  engagementWindowHours = 336,
): Promise<Neighborhood> {
  const cacheKey = `${seedDid}:${maxTwoHop}:${engagementWindowHours}`;
  const cached = neighborhoodCache.get(cacheKey);
  if (cached && cached.expiresAt > deps.now()) return cached.value;

  const actors = new Map<string, { hopDistance: number; mutualFollowCount: number; mutualEngagerCount: number }>();
  const engagementWeights = new Map<string, number>();
  const firstRows = await deps.query(
    `SELECT any(subject) AS did FROM catbird.follows ` +
    `WHERE did = ${sqlString(seedDid)} GROUP BY did, rkey HAVING max(is_deleted) = 0 LIMIT 3000`,
  );
  if (firstRows !== null) {
    const firstHop = [...new Set(firstRows.map((row) => stringValue(row, "did", "subject")).filter((did): did is string => !!did))];
    for (const did of firstHop) actors.set(did, { hopDistance: 1, mutualFollowCount: 1, mutualEngagerCount: 0 });
    const cappedWindow = Math.min(engagementWindowHours, 336);
    const engagedRows = await deps.query(
      `SELECT subject_did AS did, count() AS weight FROM (` +
      `SELECT any(subject_did) AS subject_did FROM catbird.engagements ` +
      `WHERE did = ${sqlString(seedDid)} AND rkey_date >= toDate(now() - INTERVAL ${cappedWindow} HOUR) ` +
      `GROUP BY did, kind, rkey HAVING max(is_deleted) = 0) GROUP BY subject_did LIMIT 3000`,
    );
    for (const row of engagedRows ?? []) {
      const did = stringValue(row, "did", "subject_did");
      if (!did) continue;
      const weight = Math.max(1, Math.trunc(numberValue(row, "weight", "count") ?? 1));
      engagementWeights.set(did, weight);
      const current = actors.get(did);
      if (current) current.mutualEngagerCount += weight;
      else actors.set(did, { hopDistance: 1, mutualFollowCount: 0, mutualEngagerCount: weight });
    }
    if (firstHop.length && maxTwoHop > 0) {
      const seedHop = firstHop.slice(0, 400);
      const inList = seedHop.map(sqlString).join(",");
      const [secondRows, engagerRows] = await Promise.all([
        deps.query(
          `SELECT subject AS did, uniqExact(follower) AS mutualFollowCount FROM (` +
          `SELECT did AS follower, any(subject) AS subject FROM catbird.follows ` +
          `WHERE did IN (${inList}) GROUP BY did, rkey HAVING max(is_deleted) = 0` +
          `) WHERE subject != ${sqlString(seedDid)} GROUP BY subject ORDER BY mutualFollowCount DESC LIMIT ${maxTwoHop}`,
        ),
        deps.query(
          `SELECT subject_did AS did, uniqExact(engager) AS mutualEngagerCount FROM (` +
          `SELECT did AS engager, any(subject_did) AS subject_did FROM catbird.engagements ` +
          `WHERE did IN (${inList}) AND rkey_date >= toDate(now() - INTERVAL 168 HOUR) ` +
          `GROUP BY did, kind, rkey HAVING max(is_deleted) = 0` +
          `) GROUP BY subject_did ORDER BY mutualEngagerCount DESC LIMIT ${Math.max(maxTwoHop, 600)}`,
        ),
      ]);
      for (const row of secondRows ?? []) {
        const did = stringValue(row, "did", "subject");
        if (!did || actors.has(did)) continue;
        actors.set(did, {
          hopDistance: 2,
          mutualFollowCount: Math.max(0, Math.trunc(numberValue(row, "mutualFollowCount", "mutual_follow_count") ?? 0)),
          mutualEngagerCount: 0,
        });
      }
      for (const row of engagerRows ?? []) {
        const did = stringValue(row, "did", "subject_did");
        if (!did) continue;
        const count = Math.max(0, Math.trunc(numberValue(row, "mutualEngagerCount", "mutual_engager_count") ?? 0));
        const current = actors.get(did);
        if (current) current.mutualEngagerCount = count;
        else if (actors.size < firstHop.length + maxTwoHop) {
          actors.set(did, { hopDistance: 2, mutualFollowCount: 0, mutualEngagerCount: count });
        }
      }
    }
    actors.delete(seedDid);
    const resolved = { actors, firstHop, engagementWeights, clickHouseAvailable: true };
    neighborhoodCache.set(cacheKey, { expiresAt: deps.now() + 120_000, value: resolved });
    return resolved;
  }
  const firstHop = await publicFollows(seedDid, deps.appViewFetch);
  for (const did of firstHop) actors.set(did, { hopDistance: 1, mutualFollowCount: 1, mutualEngagerCount: 0 });
  const communityCounts = new Map<number, number>();
  for (const did of firstHop) {
    for (const communityId of deps.store.signalForDid(did)?.topCommunityIds ?? []) {
      communityCounts.set(communityId, (communityCounts.get(communityId) ?? 0) + 1);
    }
  }
  for (const [communityId] of [...communityCounts].sort((a, b) => b[1] - a[1])) {
    for (const did of deps.store.communityMembers(communityId, maxTwoHop)) {
      if (did === seedDid || actors.has(did)) continue;
      actors.set(did, {
        hopDistance: 2,
        mutualFollowCount: communityCounts.get(communityId) ?? 0,
        mutualEngagerCount: 0,
      });
      if (actors.size >= firstHop.length + maxTwoHop) break;
    }
    if (actors.size >= firstHop.length + maxTwoHop) break;
  }
  return { actors, firstHop, engagementWeights, clickHouseAvailable: false };
}

function viewerTopCommunities(store: IntelVectorStore, dids: Iterable<string>) {
  const scores = new Map<number, { score: number; socialProof: number }>();
  for (const did of dids) {
    const signal = store.signalForDid(did);
    if (!signal || signal.hiddenFromRecommendations) continue;
    for (const communityId of signal.topCommunityIds) {
      const current = scores.get(communityId) ?? { score: 0, socialProof: 0 };
      current.score += Math.max(1, signal.qualityScoreBps);
      current.socialProof++;
      scores.set(communityId, current);
    }
  }
  const total = [...scores.values()].reduce((sum, value) => sum + value.score, 0);
  return [...scores]
    .sort((a, b) => b[1].score - a[1].score || a[0] - b[0])
    .slice(0, 12)
    .map(([communityId, value]) => {
      const community = store.community(communityId);
      return {
        communityId,
        ...(community?.label ? { title: community.label } : {}),
        scoreBps: total ? Math.min(10000, Math.round(value.score / total * 10000)) : 0,
        socialProof: value.socialProof,
      };
    });
}

function viewerCentroids(store: IntelVectorStore, neighborhood: Neighborhood): WindowedCentroid[] {
  const allDids = new Set([...neighborhood.firstHop, ...neighborhood.engagementWeights.keys()]);
  const recentDids = neighborhood.engagementWeights.size ? neighborhood.engagementWeights.keys() : neighborhood.firstHop;
  const sevenDay = store.centroidForDids(recentDids, "all", neighborhood.engagementWeights);
  const thirtyDay = store.centroidForDids(allDids, "all");
  const ninetyDay = store.centroidForDids(allDids, "preT");
  if (sevenDay) sevenDay.window = "7d";
  if (thirtyDay) thirtyDay.window = "30d";
  if (ninetyDay) ninetyDay.window = "90d";
  return [sevenDay, thirtyDay, ninetyDay].filter((centroid): centroid is WindowedCentroid => centroid !== null);
}

async function actorNodesForDids(
  deps: V4Dependencies,
  dids: string[],
  windows: string[],
  includeMutualStats: boolean,
  neighborhoodStats?: ReadonlyMap<string, { hopDistance: number; mutualFollowCount: number; mutualEngagerCount: number }>,
): Promise<ActorIntelNode[]> {
  if (dids.length === 0) return [];
  const didList = dids.map(sqlString).join(",");
  const uncachedDids = dids.filter((did) => !authorModeByDid.has(did));
  const [modeRows, statusRows] = await Promise.all([
    uncachedDids.length > 0 && authorModeByDid.size === 0
      ? deps.query(
          `SELECT did, argMax(total_events_14d, updated_at) AS total_events_14d, ` +
          `argMax(orig_posts, updated_at) AS orig_posts, argMax(reply_posts, updated_at) AS reply_posts, ` +
          `argMax(quote_posts, updated_at) AS quote_posts, argMax(reposts, updated_at) AS reposts ` +
          `FROM catbird.author_mode_mix WHERE did IN (${didList}) GROUP BY did`,
        )
      : Promise.resolve(null),
    deps.query(
      `SELECT did, argMax(active, seq) AS active, argMax(status, seq) AS status ` +
      `FROM catbird.account_status WHERE did IN (${didList}) GROUP BY did`,
    ),
  ]);
  const modes = new Map<string, JsonRow>();
  for (const did of dids) {
    const cachedMode = authorModeByDid.get(did);
    if (cachedMode) modes.set(did, cachedMode as unknown as JsonRow);
  }
  for (const row of modeRows ?? []) {
    const did = stringValue(row, "did");
    if (did) modes.set(did, row);
  }
  const inactiveDids = new Set((statusRows ?? []).filter((row) => {
    const status = stringValue(row, "status")?.toLowerCase() ?? "";
    return numberValue(row, "active") === 0 || /deleted|deactivated|suspended|takendown/.test(status);
  }).map((row) => stringValue(row, "did")).filter((did): did is string => !!did));
  const mutualFollows = new Map<string, number>();
  const mutualEngagers = new Map<string, number>();
  if (includeMutualStats && !neighborhoodStats) {
    const neighborhood = await resolveNeighborhood(deps, deps.viewer, 1500, 336);
    for (const did of dids) {
      const stats = neighborhood.actors.get(did);
      if (stats) {
        mutualFollows.set(did, stats.mutualFollowCount);
        mutualEngagers.set(did, stats.mutualEngagerCount);
      }
    }
  }
  const nodes: ActorIntelNode[] = [];
  for (const did of dids) {
    if (inactiveDids.has(did)) continue;
    const stats = neighborhoodStats?.get(did);
    const mutualFollowCount = stats?.mutualFollowCount ?? mutualFollows.get(did) ?? 0;
    const mutualEngagerCount = stats?.mutualEngagerCount ?? mutualEngagers.get(did) ?? 0;
    const node = deps.store.actorNode(did, {
      hopDistance: stats?.hopDistance ?? 0,
      mutualFollowCount,
      mutualEngagerCount,
      socialAffinityPermille: Math.min(1000, stats ? 350 + mutualFollowCount * 100 + mutualEngagerCount * 40 : mutualFollowCount * 300 + mutualEngagerCount * 80),
      windows,
    });
    if (!node) continue;
    const mode = modes.get(did);
    if (mode && node.health) {
      const originals = Math.max(0, numberValue(mode, "orig_posts") ?? 0);
      const replies = Math.max(0, numberValue(mode, "reply_posts") ?? 0);
      const quotes = Math.max(0, numberValue(mode, "quote_posts") ?? 0);
      const reposts = Math.max(0, numberValue(mode, "reposts") ?? 0);
      const total = originals + replies + quotes + reposts;
      node.health.posts14d = Math.trunc(numberValue(mode, "total_events_14d") ?? total);
      if (total > 0) {
        node.health.originalSharePermille = Math.round(originals / total * 1000);
        node.health.replySharePermille = Math.round(replies / total * 1000);
        node.health.quoteSharePermille = Math.round(quotes / total * 1000);
        node.health.repostSharePermille = Math.max(0, 1000 - node.health.originalSharePermille -
          node.health.replySharePermille - node.health.quoteSharePermille);
      }
    }
    nodes.push(node);
  }
  return nodes;
}
function isoDateTime(value: string | undefined, fallback: string): string {
  if (!value) return fallback;
  const parsed = Date.parse(value.endsWith("Z") ? value : `${value.replace(" ", "T")}Z`);
  return Number.isFinite(parsed) ? new Date(parsed).toISOString() : fallback;
}

function permille(value: number | undefined): number | undefined {
  if (value === undefined) return undefined;
  return Math.max(0, Math.min(1000, Math.round(value <= 1 ? value * 1000 : value)));
}

function rawPostFromAppView(value: unknown): RawPost | null {
  const post = plainObject(value);
  const uri = stringValue(post, "uri");
  const authorDid = stringValue(plainObject(post.author), "did") ?? (uri ? parsePostUri(uri)?.did : undefined);
  if (!uri || !authorDid || !parsePostUri(uri)) return null;
  const recordValue = plainObject(post.record);
  const recordEmbed = plainObject(recordValue.embed);
  const viewEmbed = plainObject(post.embed);
  const recordRef = plainObject(recordEmbed.record);
  const viewRecord = plainObject(viewEmbed.record);
  const nestedViewRecord = plainObject(viewRecord.record);
  const quoteUri = stringValue(recordRef, "uri") ?? stringValue(viewRecord, "uri") ?? stringValue(nestedViewRecord, "uri");
  const reply = plainObject(recordValue.reply);
  const replyRootUri = stringValue(plainObject(reply.root), "uri");
  const images = Array.isArray(viewEmbed.images) ? viewEmbed.images : Array.isArray(recordEmbed.images) ? recordEmbed.images : [];
  return {
    uri,
    authorDid,
    indexedAt: isoDateTime(stringValue(post, "indexedAt") ?? stringValue(recordValue, "createdAt"), indexedAtFromPostUri(uri)),
    text: stringValue(recordValue, "text"),
    ...(quoteUri ? { quoteUri } : {}),
    ...(replyRootUri ? { replyRootUri } : {}),
    hasImages: images.length > 0,
    replyCount: Math.max(0, Math.trunc(numberValue(post, "replyCount") ?? 0)),
    quoteCount: Math.max(0, Math.trunc(numberValue(post, "quoteCount") ?? 0)),
    likeRepostCount: Math.max(0, Math.trunc((numberValue(post, "likeCount") ?? 0) + (numberValue(post, "repostCount") ?? 0))),
  };
}

function rawPostFromClickHouse(row: JsonRow): RawPost | null {
  const did = stringValue(row, "authorDid", "author_did", "did");
  const rkey = stringValue(row, "rkey");
  const uri = stringValue(row, "uri") ?? (did && rkey ? `at://${did}/app.bsky.feed.post/${rkey}` : undefined);
  const authorDid = did ?? (uri ? parsePostUri(uri)?.did : undefined);
  if (!uri || !authorDid || !parsePostUri(uri)) return null;
  return {
    uri,
    authorDid,
    indexedAt: isoDateTime(stringValue(row, "indexedAt", "indexed_at", "created_at"), indexedAtFromPostUri(uri)),
    text: stringValue(row, "text"),
    quoteUri: stringValue(row, "quoteUri", "quote_uri") || undefined,
    replyRootUri: stringValue(row, "replyRootUri", "reply_root") || undefined,
    hasImages: (numberValue(row, "imageCount", "image_count") ?? 0) > 0,
    replyCount: Math.max(0, Math.trunc(numberValue(row, "replyCount", "reply_count", "replies") ?? 0)),
    quoteCount: Math.max(0, Math.trunc(numberValue(row, "quoteCount", "quote_count", "quotes") ?? 0)),
    likeRepostCount: Math.max(0, Math.trunc(numberValue(row, "likeRepostCount", "like_repost_count") ?? 0)),
    socialEndorserCount: Math.max(0, Math.trunc(numberValue(row, "socialEndorserCount", "social_endorser_count") ?? 0)),
    rageBaitPermille: permille(numberValue(row, "rageBaitPermille", "rage_bait_permille", "engagement_bait")),
    controversyPermille: permille(numberValue(row, "controversyPermille", "controversy_permille", "heat")),
    communityId: numberValue(row, "communityId", "community_id", "community"),
  };
}

async function appViewPosts(fetcher: AppViewFetch, uris: string[]): Promise<RawPost[]> {
  const posts: RawPost[] = [];
  for (let offset = 0; offset < uris.length; offset += 25) {
    const response = await appViewJson(fetcher, "app.bsky.feed.getPosts", { uris: uris.slice(offset, offset + 25) });
    for (const value of Array.isArray(response?.posts) ? response.posts : []) {
      const post = rawPostFromAppView(value);
      if (post) posts.push(post);
    }
  }
  return posts;
}

function communityForPost(store: IntelVectorStore, post: RawPost): IntelCommunity | null {
  const explicit = post.communityId === undefined ? null : store.community(post.communityId);
  if (explicit) return explicit;
  const actorCommunity = store.signalForDid(post.authorDid)?.topCommunityIds[0];
  if (actorCommunity !== undefined) return store.community(actorCommunity);
  return store.communities.find((community) => community.samplePostUris.includes(post.uri)) ?? null;
}

async function loadRawPosts(deps: V4Dependencies, uris: string[], seedRows: JsonRow[] = []): Promise<Map<string, RawPost>> {
  const posts = new Map<string, RawPost>();
  for (const row of seedRows) {
    const post = rawPostFromClickHouse(row);
    if (post) posts.set(post.uri, post);
  }
  const missingForClickHouse = uris.filter((uri) => !posts.has(uri));
  if (missingForClickHouse.length) {
    const tuples = missingForClickHouse
      .map((uri) => parsePostUri(uri))
      .filter((parsed): parsed is { did: string; rkey: string } => parsed !== null)
      .map((parsed) => `(${sqlString(parsed.did)},${sqlString(parsed.rkey)})`);
    if (tuples.length) {
      const rows = await deps.query(
        `SELECT concat('at://', did, '/app.bsky.feed.post/', rkey) AS uri, did AS authorDid, rkey, ` +
        `toString(any(indexed_at)) AS indexedAt, any(text) AS text, any(quote_uri) AS quoteUri, any(reply_root) AS replyRootUri, ` +
        `any(image_count) AS imageCount, any(engagement_bait) AS engagement_bait, any(heat) AS heat ` +
        `FROM catbird.posts WHERE (did, rkey) IN (${tuples.join(",")}) GROUP BY did, rkey HAVING max(is_deleted) = 0`,
      );
      for (const row of rows ?? []) {
        const post = rawPostFromClickHouse(row);
        if (post) posts.set(post.uri, post);
      }
    }
  }
  if (uris.length) {
    const uriList = uris.map(sqlString).join(",");
    const [countRows, popRows] = await Promise.all([
      deps.query(
        `SELECT uri, sum(replies) AS replyCount, sum(quotes) AS quoteCount FROM catbird.post_counts ` +
        `WHERE uri IN (${uriList}) GROUP BY uri`,
      ),
      deps.query(
        `SELECT item AS uri, sum(pop) AS likeRepostCount FROM feed.item_pop ` +
        `WHERE item IN (${uriList}) GROUP BY item`,
      ),
    ]);
    for (const row of countRows ?? []) {
      const uri = stringValue(row, "uri");
      const post = uri ? posts.get(uri) : undefined;
      if (!post) continue;
      post.replyCount = Math.max(0, Math.trunc(numberValue(row, "replyCount", "replies") ?? 0));
      post.quoteCount = Math.max(0, Math.trunc(numberValue(row, "quoteCount", "quotes") ?? 0));
    }
    for (const row of popRows ?? []) {
      const uri = stringValue(row, "uri");
      const post = uri ? posts.get(uri) : undefined;
      if (!post) continue;
      post.likeRepostCount = Math.max(post.likeRepostCount ?? 0, Math.trunc(numberValue(row, "likeRepostCount") ?? 0));
    }
  }
  const appViewMissing = uris.filter((uri) => !posts.has(uri));
  for (const post of await appViewPosts(deps.appViewFetch, appViewMissing)) posts.set(post.uri, post);
  for (const uri of uris) {
    if (posts.has(uri)) continue;
    const parsed = parsePostUri(uri);
    if (parsed) posts.set(uri, { uri, authorDid: parsed.did, indexedAt: indexedAtFromPostUri(uri) });
  }
  return posts;
}

async function postSkeletons(
  deps: V4Dependencies,
  uris: string[],
  includeVectors: boolean,
  seedRows: JsonRow[] = [],
  hopByDid?: ReadonlyMap<string, { hopDistance: number }>,
): Promise<PostIntelSkeleton[]> {
  const rawPosts = await loadRawPosts(deps, uris, seedRows);
  const output: PostIntelSkeleton[] = [];
  for (const uri of uris) {
    const post = rawPosts.get(uri);
    if (!post) continue;
    const signal = deps.store.signalForDid(post.authorDid);
    const community = communityForPost(deps.store, post);
    let vector = deps.store.vectorForDid(post.authorDid, "all") ?? (community ? deps.store.communityVector(community) : null);
    if (includeVectors && !vector) vector = await deps.store.encodeQueryText(post.text ?? post.uri);
    const rage = post.rageBaitPermille ?? signal?.rageBaitPermille ?? 0;
    const controversy = post.controversyPermille ?? signal?.rageBaitPermille ?? 0;
    const hopDistance = hopByDid?.get(post.authorDid)?.hopDistance;
    output.push({
      uri: post.uri,
      authorDid: post.authorDid,
      indexedAt: post.indexedAt,
      ...(hopDistance === undefined ? {} : { hopDistance }),
      socialEndorserCount: post.socialEndorserCount ?? 0,
      rageBaitPermille: rage,
      controversyPermille: controversy,
      replyCount: post.replyCount ?? 0,
      quoteCount: post.quoteCount ?? 0,
      likeRepostCount: post.likeRepostCount ?? 0,
      ...(post.quoteUri ? { quoteUri: post.quoteUri } : {}),
      ...(post.replyRootUri ? { replyRootUri: post.replyRootUri } : {}),
      hasImages: post.hasImages ?? false,
      ...(community ? { communityId: community.communityId } : {}),
      ...(includeVectors && vector ? { vector } : {}),
    });
  }
  return output;
}
async function searchGraphPosts(
  deps: V4Dependencies,
  input: JsonRow,
): Promise<{ generation: string; cursor?: string; queryVectorUsed: QuantizedVector256; posts: PostIntelSkeleton[] }> {
  const queryText = typeof input.query === "string" ? input.query.trim() : "";
  if (!queryText || queryText.length > 300) throw new InvalidRequest("query must contain 1..300 characters");
  const maxHops = integer(input.maxHops, 2, 1, 2);
  const windowHours = integer(input.windowHours, 168, 1, 336);
  const maxRage = integer(input.maxRageBaitPermille, 1000, 0, 1000);
  const minEndorsers = integer(input.minSocialEndorsers, 0, 0, 100);
  const includeVectors = input.includeVectors === true;
  const limit = integer(input.limit, 30, 1, 100);
  const cursor = integer(input.cursor, 0, 0, 100000);
  const queryVector = input.queryVector === undefined ? await deps.store.encodeQueryText(queryText) : quantizedVectorInput(input.queryVector);
  const neighborhood = await resolveNeighborhood(deps, deps.viewer, 3000, windowHours);
  const candidateDids = [...neighborhood.actors]
    .filter(([, stats]) => stats.hopDistance <= maxHops)
    .map(([did]) => did)
    .slice(0, 4000);
  const seedRows: JsonRow[] = [];
  const candidateUris: string[] = [];
  if (neighborhood.clickHouseAvailable && candidateDids.length) {
    const queryDids = candidateDids.slice(0, 600);
    const rows = await deps.query(
      `SELECT concat('at://', did, '/app.bsky.feed.post/', rkey) AS uri, did AS authorDid, rkey, ` +
      `toString(any(indexed_at)) AS indexedAt, any(text) AS text, any(quote_uri) AS quoteUri, any(reply_root) AS replyRootUri, ` +
      `any(image_count) AS imageCount, any(engagement_bait) AS engagement_bait, any(heat) AS heat ` +
      `FROM catbird.posts WHERE did IN (${queryDids.map(sqlString).join(",")}) ` +
      `AND rkey_date >= toDate(now() - INTERVAL ${windowHours} HOUR) ` +
      `AND indexed_at >= now() - INTERVAL ${windowHours} HOUR ` +
      `GROUP BY did, rkey HAVING max(is_deleted) = 0 ORDER BY indexedAt DESC LIMIT 1200`,
    );
    for (const row of rows ?? []) {
      const post = rawPostFromClickHouse(row);
      if (!post) continue;
      seedRows.push(row);
      candidateUris.push(post.uri);
    }
  }
  if (candidateUris.length === 0) {
    const communityIds = new Set<number>();
    for (const did of candidateDids) {
      for (const communityId of deps.store.signalForDid(did)?.topCommunityIds ?? []) communityIds.add(communityId);
    }
    for (const community of deps.store.matchingCommunities(queryText, 8)) communityIds.add(community.communityId);
    for (const communityId of communityIds) {
      const community = deps.store.community(communityId);
      if (community) candidateUris.push(...community.samplePostUris);
    }
    const searched = await appViewJson(deps.appViewFetch, "app.bsky.feed.searchPosts", { q: queryText, limit: 100, sort: "latest" });
    for (const value of Array.isArray(searched?.posts) ? searched.posts : []) {
      const post = rawPostFromAppView(value);
      if (!post) continue;
      const inNeighborhood = neighborhood.actors.has(post.authorDid);
      const actorCommunities = deps.store.signalForDid(post.authorDid)?.topCommunityIds ?? [];
      if (inNeighborhood || actorCommunities.some((id) => communityIds.has(id))) candidateUris.push(post.uri);
    }
  }
  const uniqueUris = [...new Set(candidateUris)].filter((uri) => {
    const did = parsePostUri(uri)?.did;
    const stats = did ? neighborhood.actors.get(did) : undefined;
    return did !== undefined && stats !== undefined && stats.hopDistance <= maxHops &&
      !deps.store.isHiddenFromRecommendations(did);
  });
  const candidates = await postSkeletons(deps, uniqueUris, true, seedRows, neighborhood.actors);
  const queryBytes = quantizedBytes(queryVector);
  const ranked = candidates
    .filter((post) => (post.hopDistance ?? maxHops) <= maxHops &&
      (post.rageBaitPermille ?? 0) <= maxRage &&
      (post.socialEndorserCount ?? 0) >= minEndorsers)
    .map((post) => ({
      ...post,
      similarityPermille: post.vector ? Math.max(-1000, Math.min(1000,
        Math.round(cosineI8(queryBytes, quantizedBytes(post.vector)) * 1000))) : 0,
    }))
    .sort((a, b) => b.similarityPermille - a.similarityPermille ||
      (b.socialEndorserCount ?? 0) - (a.socialEndorserCount ?? 0) ||
      b.indexedAt.localeCompare(a.indexedAt));
  const page = ranked.slice(cursor, cursor + limit).map((post) => {
    if (includeVectors) return post;
    const { vector: _vector, ...withoutVector } = post;
    return withoutVector;
  });
  return {
    generation: deps.store.manifest.generation,
    ...(cursor + limit < ranked.length ? { cursor: String(cursor + limit) } : {}),
    queryVectorUsed: queryVector,
    posts: page,
  };
}

async function expandedActors(
  deps: V4Dependencies,
  seedDid: string,
  communityId: number | null,
  queryText: string | null,
  windowHours: number,
  maxRageBaitPermille: number,
  limit: number,
): Promise<ActorIntelNode[]> {
  const neighborhood = await resolveNeighborhood(deps, seedDid, Math.max(limit * 5, 300), windowHours);
  const candidates = new Map(neighborhood.actors);
  const selectedCommunities = new Set<number>();
  if (communityId !== null) selectedCommunities.add(communityId);
  if (queryText) for (const community of deps.store.matchingCommunities(queryText, 8)) selectedCommunities.add(community.communityId);
  for (const id of selectedCommunities) {
    for (const did of deps.store.communityMembers(id, Math.max(limit * 5, 300))) {
      if (did === seedDid || candidates.has(did)) continue;
      candidates.set(did, { hopDistance: 2, mutualFollowCount: 0, mutualEngagerCount: 0 });
    }
  }
  const queryVector = queryText ? await deps.store.encodeQueryText(queryText) : null;
  const queryBytes = queryVector ? quantizedBytes(queryVector) : null;
  const nodes = await actorNodesForDids(deps, [...candidates.keys()], ["30d", "90d"], true, candidates);
  for (const node of nodes) {
    if (!queryBytes) continue;
    const vector = deps.store.vectorForDid(node.did, "all");
    if (vector) node.socialAffinityPermille = Math.max(0, Math.min(1000,
      Math.round((cosineI8(queryBytes, quantizedBytes(vector)) + 1) * 500)));
  }
  return nodes
    .filter((node) => {
      const signal = deps.store.signalForDid(node.did);
      if (!signal || !signal.eligible || signal.hiddenFromRecommendations ||
          signal.rageBaitPermille > maxRageBaitPermille) return false;
      return selectedCommunities.size === 0 || signal.topCommunityIds.some((id) => selectedCommunities.has(id));
    })
    .sort((a, b) => (b.socialAffinityPermille ?? 0) - (a.socialAffinityPermille ?? 0) ||
      (b.health?.qualityScoreBps ?? 0) - (a.health?.qualityScoreBps ?? 0) ||
      a.did.localeCompare(b.did))
    .slice(0, limit);
}
async function quoteChain(
  deps: V4Dependencies,
  anchorUri: string | null,
  topicQuery: string | null,
  windowHours: number,
  limit: number,
) {
  const seedRows: JsonRow[] = [];
  const candidateUris: string[] = [];
  if (anchorUri) {
    const parsedAnchor = parsePostUri(anchorUri);
    if (parsedAnchor) {
      const anchorRows = await deps.query(
        `SELECT concat('at://', did, '/app.bsky.feed.post/', rkey) AS uri, did AS authorDid, rkey, ` +
        `toString(any(indexed_at)) AS indexedAt, any(text) AS text, any(quote_uri) AS quoteUri, any(reply_root) AS replyRootUri, any(image_count) AS imageCount ` +
        `FROM catbird.posts WHERE (did, rkey) = (${sqlString(parsedAnchor.did)}, ${sqlString(parsedAnchor.rkey)}) ` +
        `GROUP BY did, rkey HAVING max(is_deleted) = 0`,
      );
      for (const row of anchorRows ?? []) {
        const post = rawPostFromClickHouse(row);
        if (post) {
          seedRows.push(row);
          candidateUris.push(post.uri);
        }
      }
    }
  }
  if (anchorUri) {
    candidateUris.push(anchorUri);
    const quoted = await appViewJson(deps.appViewFetch, "app.bsky.feed.getQuotes", { uri: anchorUri, limit: Math.min(100, limit * 2) });
    for (const value of Array.isArray(quoted?.posts) ? quoted.posts : []) {
      const post = rawPostFromAppView(value);
      if (post) candidateUris.push(post.uri);
    }
  }
  if (topicQuery) {
    const hood = await resolveNeighborhood(deps, deps.viewer, 300, windowHours);
    const topicDids = new Set<string>(hood.actors.keys());
    for (const comm of deps.store.matchingCommunities(topicQuery, 6)) {
      for (const did of deps.store.communityMembers(comm.communityId, 120)) topicDids.add(did);
    }
    const scopedDids = [...topicDids].slice(0, 600);
    if (scopedDids.length) {
      const topicRows = await deps.query(
        `SELECT concat('at://', did, '/app.bsky.feed.post/', rkey) AS uri, did AS authorDid, rkey, ` +
        `toString(any(indexed_at)) AS indexedAt, any(text) AS text, any(quote_uri) AS quoteUri, any(reply_root) AS replyRootUri, any(image_count) AS imageCount ` +
        `FROM catbird.posts WHERE did IN (${scopedDids.map(sqlString).join(",")}) ` +
        `AND rkey_date >= toDate(now() - INTERVAL ${windowHours} HOUR) ` +
        `AND indexed_at >= now() - INTERVAL ${windowHours} HOUR ` +
        `AND positionCaseInsensitiveUTF8(text, ${sqlString(topicQuery)}) > 0 ` +
        `GROUP BY did, rkey HAVING max(is_deleted) = 0 ORDER BY indexedAt DESC LIMIT ${Math.max(limit * 3, 60)}`,
      );
      for (const row of topicRows ?? []) {
        const post = rawPostFromClickHouse(row);
        if (post) {
          seedRows.push(row);
          candidateUris.push(post.uri);
          if (post.quoteUri) candidateUris.push(post.quoteUri);
        }
      }
    }
    if (candidateUris.length === 0) {
      const searched = await appViewJson(deps.appViewFetch, "app.bsky.feed.searchPosts", {
        q: topicQuery,
        limit: Math.min(100, limit * 2),
        sort: "latest",
      });
      for (const value of Array.isArray(searched?.posts) ? searched.posts : []) {
        const post = rawPostFromAppView(value);
        if (post) candidateUris.push(post.uri);
      }
    }
  }
  const raw = await loadRawPosts(deps, [...new Set(candidateUris)], seedRows);
  const seedQuoteUri = [...raw.values()].find((post) => !!post.quoteUri)?.uri ?? null;
  let upstream = anchorUri ?? seedQuoteUri;
  let originUri = anchorUri ?? (seedQuoteUri ? raw.get(seedQuoteUri)?.quoteUri ?? seedQuoteUri : null);
  for (let depth = 0; upstream && depth < 12; depth++) {
    const post = raw.get(upstream) ?? (await loadRawPosts(deps, [upstream])).get(upstream);
    if (!post?.quoteUri || post.quoteUri === upstream) break;
    originUri = post.quoteUri;
    upstream = post.quoteUri;
    if (!raw.has(upstream)) {
      const loaded = await loadRawPosts(deps, [upstream]);
      const ancestor = loaded.get(upstream);
      if (ancestor) raw.set(upstream, ancestor);
    }
  }
  const allUris = [...new Set([...candidateUris, ...raw.keys()])];
  const chain = (await postSkeletons(deps, allUris, false, seedRows))
    .sort((a, b) => a.indexedAt.localeCompare(b.indexedAt))
    .slice(0, limit);
  if (!originUri && chain.length) originUri = chain[0]!.uri;
  const catalyst = [...chain].sort((a, b) => {
    const aHours = Math.max(1, (deps.now() - Date.parse(a.indexedAt)) / 3_600_000);
    const bHours = Math.max(1, (deps.now() - Date.parse(b.indexedAt)) / 3_600_000);
    return (b.quoteCount ?? 0) / bHours - (a.quoteCount ?? 0) / aHours ||
      (b.likeRepostCount ?? 0) - (a.likeRepostCount ?? 0);
  })[0];
  const buckets = new Map<string, { hourStart: string; postCount: number; quoteCount: number; replyCount: number }>();
  for (const post of chain) {
    const date = new Date(post.indexedAt);
    if (!Number.isFinite(date.valueOf())) continue;
    date.setUTCMinutes(0, 0, 0);
    const hourStart = date.toISOString();
    const bucket = buckets.get(hourStart) ?? { hourStart, postCount: 0, quoteCount: 0, replyCount: 0 };
    bucket.postCount++;
    if (post.quoteUri) bucket.quoteCount++;
    if (post.replyRootUri) bucket.replyCount++;
    buckets.set(hourStart, bucket);
  }
  return {
    generation: deps.store.manifest.generation,
    ...(originUri ? { originUri } : {}),
    ...(catalyst ? { catalystUri: catalyst.uri } : {}),
    chain,
    hourlyBuckets: [...buckets.values()].sort((a, b) => a.hourStart.localeCompare(b.hourStart)),
  };
}

async function topicClusters(
  deps: V4Dependencies,
  windowHours: number,
  queryText: string | null,
  onlySpiking: boolean,
  limit: number,
) {
  const cappedHours = Math.min(windowHours, 48);
  const volumeRows = await deps.query(
    `SELECT m.community AS communityId, ` +
    `countIf(p.indexed_at >= now() - INTERVAL ${cappedHours} HOUR) AS currentCount, ` +
    `countIf(p.indexed_at < now() - INTERVAL ${cappedHours} HOUR AND ` +
    `p.indexed_at >= now() - INTERVAL ${cappedHours * 2} HOUR) AS previousCount, ` +
    `uniqExactIf(p.did, p.indexed_at >= now() - INTERVAL ${cappedHours} HOUR) AS authorCount ` +
    `FROM (SELECT did, any(indexed_at) AS indexed_at FROM catbird.posts ` +
    `WHERE rkey_date >= toDate(now() - INTERVAL ${cappedHours * 2} HOUR) ` +
    `AND indexed_at >= now() - INTERVAL ${cappedHours * 2} HOUR ` +
    `GROUP BY did, rkey HAVING max(is_deleted) = 0 LIMIT 250000) AS p ` +
    `INNER JOIN (SELECT did, argMax(community, updated_at) AS community FROM catbird.author_mode_mix GROUP BY did) AS m ON p.did = m.did ` +
    `GROUP BY m.community`,
  );
  const volumes = new Map<number, JsonRow>();
  for (const row of volumeRows ?? []) {
    const communityId = numberValue(row, "communityId", "community_id", "community");
    if (communityId !== undefined) volumes.set(Math.trunc(communityId), row);
  }
  const matchingIds = queryText ? new Set(deps.store.matchingCommunities(queryText, deps.store.communities.length)
    .map((community) => community.communityId)) : null;
  const clusters = deps.store.communities.flatMap((community) => {
    if (matchingIds && !matchingIds.has(community.communityId)) return [];
    const visibleUris = community.samplePostUris.filter((uri) => {
      const did = parsePostUri(uri)?.did;
      return did !== undefined && !deps.store.isHiddenFromRecommendations(did);
    });
    const visibleAuthors = new Set([
      ...community.sampleAccountDids.filter((did) => !deps.store.isHiddenFromRecommendations(did)),
      ...visibleUris.map((uri) => parsePostUri(uri)!.did),
    ]);
    const volume = volumes.get(community.communityId);
    const currentCount = Math.max(0, Math.trunc(numberValue(volume ?? {}, "currentCount", "current_count", "postCount") ?? visibleUris.length));
    const previousCount = Math.max(0, Math.trunc(numberValue(volume ?? {}, "previousCount", "previous_count") ?? currentCount));
    const authorCount = Math.max(0, Math.trunc(numberValue(volume ?? {}, "authorCount", "author_count") ?? visibleAuthors.size));
    if (currentCount === 0 || (visibleUris.length === 0 && authorCount === 0)) return [];
    const spikeZScoreBps = Math.max(-10000, Math.min(100000,
      Math.round((currentCount - previousCount) / Math.sqrt(Math.max(1, previousCount)) * 10000)));
    if (onlySpiking && spikeZScoreBps < 20000) return [];
    const rageValues = [...visibleAuthors].map((did) => deps.store.signalForDid(did)?.rageBaitPermille)
      .filter((value): value is number => value !== undefined);
    const controversyPermille = rageValues.length ?
      Math.round(rageValues.reduce((sum, value) => sum + value, 0) / rageValues.length) : 0;
    const samplePosts: PostIntelSkeleton[] = visibleUris.slice(0, 20).map((uri) => {
      const parsed = parsePostUri(uri)!;
      const signal = deps.store.signalForDid(parsed.did);
      return {
        uri,
        authorDid: parsed.did,
        indexedAt: indexedAtFromPostUri(uri),
        rageBaitPermille: signal?.rageBaitPermille ?? 0,
        controversyPermille: signal?.rageBaitPermille ?? 0,
        communityId: community.communityId,
      };
    });
    return [{
      clusterId: community.communityId,
      label: community.label,
      ...(community.description ? { description: community.description } : {}),
      ...(community.macroCategory ? { macroCategory: community.macroCategory } : {}),
      ...(community.sceneSlug ? { sceneSlug: community.sceneSlug } : {}),
      postCount: currentCount,
      authorCount,
      spikeZScoreBps,
      controversyPermille,
      centroid: deps.store.communityVector(community),
      samplePosts,
    }];
  });
  return {
    generation: deps.store.manifest.generation,
    clusters: clusters.sort((a, b) => b.spikeZScoreBps - a.spikeZScoreBps ||
      b.postCount - a.postCount || a.clusterId - b.clusterId).slice(0, limit),
  };
}
export function createApiHandler(options: {
  authenticate?: ServiceAuthenticator;
  catalogEngine?: S17CatalogEngine | null;
  liveOnboarding?: LiveOnboardingEngine | null;
  vectorStore?: IntelVectorStore | null;
  queryClickHouse?: (sql: string) => Promise<JsonRow[]>;
  appViewFetch?: AppViewFetch;
  now?: () => number;
  requireServiceAuth?: boolean;
} = {}) {
  const authenticate = options.authenticate ?? createServiceAuthenticator();
  const getEngine = () => options.catalogEngine ?? activeCatalogEngine;
  // A handler given its own catalog engine never falls through to the process-wide live engine,
  // which wraps a different catalog.
  const getLive = () => Object.prototype.hasOwnProperty.call(options, "liveOnboarding") ?
    options.liveOnboarding ?? null : options.catalogEngine ? null : activeLiveOnboarding;
  const getVectorStore = () => Object.prototype.hasOwnProperty.call(options, "vectorStore") ?
    options.vectorStore ?? null : activeVectorStore;
  const query: ClickHouseQuery = options.queryClickHouse ? async (sql) => {
    try { return await options.queryClickHouse!(sql); } catch { return null; }
  } : optionalClickHouseQuery;
  const appViewFetch = options.appViewFetch ?? fetch;
  const now = options.now ?? Date.now;
  const json = (body: Record<string, unknown>, status = 200) =>
    Response.json({ ...body, ...contractMetadata() }, { status, headers: { "cache-control": "no-store" } });

  return async (request: Request): Promise<Response> => {
    try {
      const url = new URL(request.url);
      const path = url.pathname;
      const engine = getEngine();
      const live = engine ? getLive() : null;
      const vectorStore = getVectorStore();
      if (path === "/health") return json({
        ok: true,
        ready_for_public_promotion: false,
        blockers: ["live_deletion_overlay_unqualified", "viewer_moderation_requires_hydration"],
        telemetry_persistence: "disabled_pending_versioned_schema",
        communities: communitiesById.size,
        communities_with_posts: itemsByCommunity.size,
        author_modes_cached: authorModeByDid.size,
        ...(vectorStore ? {
          vector_store_generation: vectorStore.manifest.generation,
          vector_store_producers: vectorStore.manifest.producerCount,
          vector_store_vectorized_producers: vectorStore.manifest.vectorizedProducerCount,
          vector_store_communities: vectorStore.manifest.communityCentroidCount,
        } : {}),
        ...(engine ? {
          s17_catalog_version: engine.bundle.catalogVersion,
          s17_communities: engine.bundle.communities.length,
          s17_scenes: engine.bundle.scenes.length,
          ...(live ? live.describe() : { onboarding_engine: "static" }),
        } : {}),
      });
      const method = path.startsWith("/xrpc/") ? path.slice(6) : "blue.catbird.intel.getCommunity";
      const viewer = await authenticate(request, method);
      const unauthenticated = viewer === null; // Never infer authentication from request fields.
      const viewerRef = { did: viewer, unauthenticated };
      const v4Method = V4_METHODS[method] === true;
      if ((v4Method || (options.requireServiceAuth && path.startsWith("/xrpc/blue.catbird."))) && unauthenticated) {
        return json({ error: "AuthRequired", message: "Service authentication required" }, 401);
      }
      const postRoute = path === "/xrpc/blue.catbird.onboarding.previewTimeline" ||
        path === "/xrpc/blue.catbird.onboarding.solveBundle" ||
        path === "/xrpc/blue.catbird.onboarding.getCalibrationSwipes" ||
        path === "/xrpc/blue.catbird.intel.getActorVectors" ||
        path === "/xrpc/blue.catbird.intel.getPostVectors" ||
        path === "/xrpc/blue.catbird.intel.searchGraphPosts";
      if (request.method !== (postRoute ? "POST" : "GET")) return json({ error: "MethodNotAllowed" }, 405);
      if (v4Method && !vectorStore) {
        return json({ error: "UpstreamUnavailable", message: "Intel vector store not loaded" }, 503);
      }
      const v4 = v4Method ? {
        store: vectorStore!,
        viewer: viewer!,
        query,
        appViewFetch,
        now,
      } satisfies V4Dependencies : null;

      if (path === "/xrpc/blue.catbird.intel.getGraphNeighborhoodBundle") {
        const maxTwoHop = integer(url.searchParams.get("maxTwoHopActors"), 1000, 0, 3000);
        const generation = v4!.store.manifest.generation;
        const ifNoneMatchGeneration = url.searchParams.get("ifNoneMatchGeneration");
        if (ifNoneMatchGeneration && ifNoneMatchGeneration.length > 128) {
          throw new InvalidRequest("ifNoneMatchGeneration exceeds 128 characters");
        }
        if (ifNoneMatchGeneration === generation) {
          return json({ generation, notModified: true, viewerCentroids: [], viewerTopCommunities: [], actors: [] });
        }
        const neighborhood = await resolveNeighborhood(v4!, v4!.viewer, maxTwoHop, 168);
        const recommendableDids = [...neighborhood.actors]
          .filter(([did]) => {
            const signal = v4!.store.signalForDid(did);
            return signal?.eligible === true && !signal.hiddenFromRecommendations && signal.vecRow >= 0;
          })
          .sort((a, b) => a[1].hopDistance - b[1].hopDistance ||
            b[1].mutualEngagerCount - a[1].mutualEngagerCount ||
            b[1].mutualFollowCount - a[1].mutualFollowCount)
          .map(([did]) => did);
        const actors = await actorNodesForDids(
          v4!,
          recommendableDids,
          ["7d", "30d", "90d", "pre_t"],
          true,
          neighborhood.actors,
        );
        const affinityDids = new Set([...neighborhood.firstHop, ...neighborhood.engagementWeights.keys()]);
        return json({
          generation,
          notModified: false,
          viewerCentroids: viewerCentroids(v4!.store, neighborhood),
          viewerTopCommunities: viewerTopCommunities(v4!.store, affinityDids),
          actors,
        });
      }

      if (path === "/xrpc/blue.catbird.intel.getActorVectors") {
        const body = await bodyInput(request);
        const dids = parseDidArray(body.dids, "dids", 512);
        if (dids.length === 0) throw new InvalidRequest("dids must contain 1..512 DIDs");
        let windows = ["30d", "90d"];
        if (body.windows !== undefined) {
          if (!Array.isArray(body.windows) || body.windows.length > 5 ||
              body.windows.some((window) => typeof window !== "string" ||
                !["7d", "30d", "90d", "pre_t", "all"].includes(window))) {
            throw new InvalidRequest("windows contains an unsupported vector window");
          }
          windows = body.windows as string[];
        }
        if (body.includeMutualStatsForViewer !== undefined && typeof body.includeMutualStatsForViewer !== "boolean") {
          throw new InvalidRequest("includeMutualStatsForViewer must be boolean");
        }
        const actors = await actorNodesForDids(v4!, dids, windows, body.includeMutualStatsForViewer !== false);
        return json({ generation: v4!.store.manifest.generation, actors });
      }

      if (path === "/xrpc/blue.catbird.intel.getPostVectors") {
        const body = await bodyInput(request);
        if (!Array.isArray(body.uris) || body.uris.length < 1 || body.uris.length > 200 ||
            body.uris.some((uri) => typeof uri !== "string" || !parsePostUri(uri))) {
          throw new InvalidRequest("uris must contain 1..200 post AT-URIs");
        }
        if (body.includeVectors !== undefined && typeof body.includeVectors !== "boolean") {
          throw new InvalidRequest("includeVectors must be boolean");
        }
        const posts = await postSkeletons(v4!, body.uris as string[], body.includeVectors !== false);
        return json({ generation: v4!.store.manifest.generation, posts });
      }

      if (path === "/xrpc/blue.catbird.intel.searchGraphPosts") {
        const result = await searchGraphPosts(v4!, await bodyInput(request));
        return json({ ...result });
      }

      if (path === "/xrpc/blue.catbird.intel.expandGraphNeighborhood") {
        const seedDid = url.searchParams.get("seedDid") ?? v4!.viewer;
        if (!/^did:[a-z0-9]+:[^#\s]+$/i.test(seedDid)) throw new InvalidRequest("Invalid seedDid");
        const communityId = url.searchParams.has("communityId") ?
          integer(url.searchParams.get("communityId"), 0, 0, 0xffff_ffff) : null;
        if (communityId !== null && !v4!.store.community(communityId)) throw new InvalidRequest("Unknown communityId");
        const queryText = url.searchParams.get("q")?.trim() || null;
        if (queryText && queryText.length > 160) throw new InvalidRequest("q exceeds 160 characters");
        const windowHours = integer(url.searchParams.get("windowHours"), 168, 1, 720);
        const maxRage = integer(url.searchParams.get("maxRageBaitPermille"), 1000, 0, 1000);
        const limit = integer(url.searchParams.get("limit"), 100, 1, 300);
        const actors = await expandedActors(v4!, seedDid, communityId, queryText, windowHours, maxRage, limit);
        return json({ generation: v4!.store.manifest.generation, actors });
      }

      if (path === "/xrpc/blue.catbird.intel.getQuoteChain") {
        const anchorUri = url.searchParams.get("anchorUri");
        const topicQuery = url.searchParams.get("topicQuery")?.trim() || null;
        if (anchorUri && !parsePostUri(anchorUri)) throw new InvalidRequest("Invalid anchorUri");
        if (topicQuery && topicQuery.length > 200) throw new InvalidRequest("topicQuery exceeds 200 characters");
        if (!anchorUri && !topicQuery) throw new InvalidRequest("anchorUri or topicQuery is required");
        const result = await quoteChain(
          v4!,
          anchorUri,
          topicQuery,
          integer(url.searchParams.get("windowHours"), 72, 1, 336),
          integer(url.searchParams.get("limit"), 25, 1, 60),
        );
        return json({ ...result });
      }

      if (path === "/xrpc/blue.catbird.intel.getTopicClusters") {
        const onlySpikingValue = url.searchParams.get("onlySpiking");
        if (onlySpikingValue !== null && onlySpikingValue !== "true" && onlySpikingValue !== "false") {
          throw new InvalidRequest("onlySpiking must be boolean");
        }
        const queryText = url.searchParams.get("q")?.trim() || null;
        if (queryText && queryText.length > 160) throw new InvalidRequest("q exceeds 160 characters");
        const result = await topicClusters(
          v4!,
          integer(url.searchParams.get("windowHours"), 24, 1, 720),
          queryText,
          onlySpikingValue === "true",
          integer(url.searchParams.get("limit"), 25, 1, 100),
        );
        return json({ ...result });
      }

      if (path === "/xrpc/blue.catbird.onboarding.getCatalog") {
        if (!engine) return json({ error: "UpstreamUnavailable", message: "S17 catalog bundle not loaded" }, 503);
        const seedFollowDids = parseDidArray(url.searchParams.getAll("seedFollowDids"), "seedFollowDids", 100);
        const q = url.searchParams.get("q");
        const macroCategory = url.searchParams.get("macroCategory");
        const limit = integer(url.searchParams.get("limit"), 80, 1, 200);
        if (live) return json(await live.getCatalog({ seedFollowDids, q, macroCategory, limit }, viewerRef));
        return json(engine.getCatalog({ seedFollowDids, q, macroCategory, limit, unauthenticated }));
      }

      if (path === "/xrpc/app.bsky.feed.getFeedSkeleton") {
        for (const obsolete of ["max_politicalness", "max_heat", "live_event_volume"]) {
          if (url.searchParams.has(obsolete)) throw new InvalidRequest("Use the versioned baseline_score_cutoff contract");
        }
        const rawCutoff = url.searchParams.get("baseline_score_cutoff");
        let controls: ServingControls = {};
        if (rawCutoff !== null) {
          try { controls = parseControls({ baseline_score_cutoff: JSON.parse(rawCutoff) }); }
          catch { throw new InvalidRequest("Invalid baseline_score_cutoff"); }
        }
        const actorParam = url.searchParams.get("actor");
        if (actorParam && actorParam !== viewer) return json({ error: "ActorNotAuthorized" }, 403);
        const persona = url.searchParams.get("persona");
        if (persona && !PERSONA_PRESETS[persona]) throw new InvalidRequest("Unknown editorial persona");
        const actor = viewer ?? persona ?? "anonymous";
        const result = await scoreFeedForActor(actor,
          integer(url.searchParams.get("limit"), 30, 1, 100),
          integer(url.searchParams.get("cursor"), 0, 0, 100000),
          controls, unauthenticated);
        return json({ ...result, unauthenticated });
      }

      if (path === "/xrpc/blue.catbird.intel.getCommunity" || path.startsWith("/community/")) {
        if (engine && path === "/xrpc/blue.catbird.intel.getCommunity" && !url.searchParams.has("community")) {
          if (!url.searchParams.has("communityId")) {
            throw new InvalidRequest("Missing required parameter: communityId");
          }
          const cid = integer(url.searchParams.get("communityId"), 0, 0, 0xffffffff);
          const detail = live ? await live.getCommunity(cid, viewerRef) : engine.getCommunity(cid, unauthenticated);
          if (!detail) return json({ error: "CommunityNotFound" }, 404);
          return json(detail);
        }
        if (!engine && communitiesById.size === 0 && path === "/xrpc/blue.catbird.intel.getCommunity") {
          return json({ error: "UpstreamUnavailable", message: "Community intelligence catalog not loaded" }, 503);
        }
        const cid = integer(url.searchParams.get("community") ?? path.split("/").pop(), 0, 0, 0xffffffff);
        const community = communitiesById.get(cid);
        if (!community) return json({ error: "CommunityNotFound" }, 404);
        const entries = (directoryByCommunity.get(cid) ?? []).filter((entry) =>
          eligibleOwner(entry.uri_or_did, unauthenticated, entry.has_no_unauth_label));
        const accounts: Record<string, { did: string }[]> = {};
        for (const entry of entries.filter((entry) => entry.kind === "account")) {
          (accounts[entry.role] ??= []).push({ did: entry.uri_or_did });
        }
        return json({
          unauthenticated, community: publicCommunity(community), accounts_by_role: accounts,
          posts: (itemsByCommunity.get(cid) ?? []).filter((post) =>
            eligibleOwner(post.did, unauthenticated, post.has_no_unauth_label)).slice(0, 30).map(postSkeleton),
          feeds: entries.filter((entry) => entry.kind === "feed").map((entry) => ({ uri: entry.uri_or_did })),
          packs: entries.filter((entry) => entry.kind === "pack").map((entry) => ({ uri: entry.uri_or_did })),
        });
      }
      if (path === "/xrpc/blue.catbird.intel.listCommunities") {
        if (engine && !url.searchParams.has("scene_id") && !url.searchParams.has("domain_id")) {
          const q = url.searchParams.get("q");
          const macroCategory = url.searchParams.get("macroCategory");
          const sceneSlug = url.searchParams.get("sceneSlug");
          const limit = integer(url.searchParams.get("limit"), 60, 1, 200);
          const cursor = url.searchParams.get("cursor");
          if (live) return json(await live.listCommunities({ q, macroCategory, sceneSlug, limit, cursor }, viewerRef));
          return json(engine.listCommunities({ q, macroCategory, sceneSlug, limit, cursor, unauthenticated }));
        }
        if (!engine && communitiesById.size === 0) {
          return json({ error: "UpstreamUnavailable", message: "Community intelligence catalog not loaded" }, 503);
        }
        let communities = [...communitiesById.values()];
        for (const key of ["scene_id", "domain_id"] as const) {
          if (url.searchParams.has(key)) {
            const value = integer(url.searchParams.get(key), 0, 0, 65535);
            communities = communities.filter((community) => community[key] === value);
          }
        }
        return json({ total: communities.length, communities: communities.slice(0, 200).map(publicCommunity) });
      }
      if (path === "/xrpc/blue.catbird.intel.getAuthorModes") {
        // Aggregate operational diagnostics only; no inferred political profile of an identified author.
        return json({ total_producers_scored: authorModeByDid.size,
          active_producers_in_window: [...authorModeByDid.values()].filter((author) => author.total_events_14d > 0).length,
          scoring_status: "legacy_heuristic_unqualified", individual_profiles_available: false });
      }
      if (path === "/xrpc/blue.catbird.onboarding.getSceneCards") {
        return json(buildSceneCards(unauthenticated));
      }
      if (postRoute) {
        const body = await bodyInput(request);
        if (engine && (body.selections !== undefined || body.allocation === undefined)) {
          const selections = parseSelectionsInput(body.selections);
          const seedFollowDids = parseDidArray(body.seedFollowDids, "seedFollowDids", 100);
          if (path.endsWith(".getCalibrationSwipes")) {
            const maxSwipes = integer(body.maxSwipes, 5, 1, 12);
            if (live) return json(await live.getCalibrationSwipes({ selections, seedFollowDids, maxSwipes }, viewerRef));
            return json(engine.getCalibrationSwipes({ selections, seedFollowDids, maxSwipes, unauthenticated }));
          }
          const swipes = parseCanonicalSwipesInput(body.swipes);
          const controls = parseCanonicalControlsInput(body.controls);
          if (path.endsWith(".previewTimeline")) {
            const limit = integer(body.limit, 20, 1, 40);
            if (live) return json(await live.previewTimeline({ selections, swipes, controls, limit }, viewerRef));
            return json(engine.previewTimeline({ selections, swipes, controls, limit, unauthenticated }));
          }
          if (live) {
            // No targetSize asks the live engine to size each scene itself; a number is the older
            // client contract, one total split across the picked scenes.
            const targetSize = body.targetSize === undefined || body.targetSize === null ? undefined :
              integer(body.targetSize, 24, 8, live.config.maxLegacyTotal);
            const pinnedDids = parseDidArray(body.pinnedDids, "pinnedDids", 200);
            const excludedDids = parseDidArray(body.excludedDids, "excludedDids", 500);
            return json(await live.solveBundle(
              { selections, swipes, controls, targetSize, pinnedDids, excludedDids, seedFollowDids },
              viewerRef,
            ));
          }
          const targetSize = integer(body.targetSize, 24, 8, 50);
          const pinnedDids = parseDidArray(body.pinnedDids, "pinnedDids", 30);
          const excludedDids = parseDidArray(body.excludedDids, "excludedDids", 100);
          return json(
            engine.solveBundle({
              selections,
              swipes,
              controls,
              targetSize,
              pinnedDids,
              excludedDids,
              seedFollowDids,
              unauthenticated,
            }),
          );
        }
        if (!engine && communitiesById.size === 0) {
          return json({ error: "UpstreamUnavailable", message: "Community intelligence catalog not loaded" }, 503);
        }
        const allocation = allocationInput(body.allocation);
        const controls = parseControls(body.controls);
        if (body.live_event_volume !== undefined) throw new InvalidRequest("Event-volume control is unavailable");
        if (path.endsWith(".getCalibrationSwipes")) {
          return json({ ...buildCalibrationSwipes(allocation, unauthenticated), unauthenticated });
        }
        const swipes = swipeInput(body.swipes, allocation, unauthenticated);
        const delivery = deliveryInput(body);
        const bundle = solveOnboardingBundle(allocation, swipes, controls, unauthenticated,
          integer(body.target_size, 25, 10, 50), delivery.hideReposts, delivery.modes);
        if (path.endsWith(".solveBundle")) return json(bundle);
        const preview: PostCandidate[] = [];
        const seen = new Set<string>();
        for (const affinity of bundle.posterior_communities.slice(0, 8)) {
          for (const post of itemsByCommunity.get(affinity.community) ?? []) {
            if (seen.has(post.item_uri) || !eligibleOwner(post.did, unauthenticated, post.has_no_unauth_label) ||
                !passesControls(post, controls)) continue;
            preview.push(post); seen.add(post.item_uri);
            if (preview.length >= 25) break;
          }
          if (preview.length >= 25) break;
        }
        return json({ unauthenticated, preview_posts: preview.map(postSkeleton),
          control_policy: controlReport(preview, controls) });
      }
      // Cache publication remains internal; service authentication does not confer administration.
      if (path === "/xrpc/blue.catbird.intel.refreshCache") return json({ error: "InternalOperationOnly" }, 403);
      return json({ error: "NotFound" }, 404);
    } catch (error) {
      if (error instanceof AuthenticationError) return json({ error: "AuthRequired", message: error.message }, 401);
      if (error instanceof InvalidRequest) return json({ error: "InvalidRequest", message: error.message }, 400);
      console.error("serving request failed", error instanceof Error ? error.name : "UnknownError");
      return json({ error: "InternalServerError" }, 500);
    }
  };
}

if (import.meta.main) {
  const defaultBundlePath = "/Users/joshlacalamito/catbird-lake-local/s17-picker-v1-20261030/intel_serving_v2.json";
  const bundlePath = process.env.INTEL_SERVING_BUNDLE_PATH ?? (existsSync(defaultBundlePath) ? defaultBundlePath : null);
  if (bundlePath && existsSync(bundlePath)) {
    activeCatalogEngine = S17CatalogEngine.fromFile(bundlePath);
    console.log(
      `Loaded S17 catalog bundle ${activeCatalogEngine.bundle.catalogVersion}: ${activeCatalogEngine.bundle.communities.length} communities across ${activeCatalogEngine.bundle.scenes.length} scenes`,
    );
  }
  const vectorStoreDirectory = process.env.INTEL_VECTOR_STORE_DIR ??
    (existsSync(`${DEFAULT_INTEL_VECTOR_STORE_DIR}/manifest.json`) ? DEFAULT_INTEL_VECTOR_STORE_DIR : null);
  if (vectorStoreDirectory) {
    activeVectorStore = IntelVectorStore.fromDirectory(vectorStoreDirectory);
    console.log(
      `Loaded Intel vectors ${activeVectorStore.manifest.generation}: ` +
      `${activeVectorStore.manifest.vectorizedProducerCount}/${activeVectorStore.manifest.producerCount} producers, ` +
      `${activeVectorStore.manifest.communityCentroidCount} communities`,
    );
  }
  if (process.env.INTEL_ONBOARDING_ENGINE === "live" && activeCatalogEngine) {
    const poolPath = process.env.INTEL_ONBOARDING_POOL_PATH;
    const pool = poolPath && existsSync(poolPath) ? LiveOnboardingEngine.loadPoolFile(poolPath) : null;
    if (poolPath && !pool) console.warn(`Onboarding pool not found at ${poolPath}; using the catalog's own members`);
    const lineupSize = (name: string, fallback: number) => {
      const value = Number(process.env[name] ?? fallback);
      return Number.isInteger(value) && value >= 1 && value <= 24 ? value : fallback;
    };
    activeLiveOnboarding = new LiveOnboardingEngine({
      catalog: activeCatalogEngine,
      pool,
      vectors: activeVectorStore,
      followsFetcher: createPublicFollowsFetcher({ baseUrl: PUBLIC_APPVIEW }),
      config: {
        defaultLineup: lineupSize("INTEL_ONBOARDING_LINEUP_DEFAULT", 6),
        boostedLineup: lineupSize("INTEL_ONBOARDING_LINEUP_BOOSTED", 12),
      },
    });
    console.log(`Live onboarding engine on: ${JSON.stringify(activeLiveOnboarding.describe())}`);
  }
  if (process.env.INTEL_SKIP_CLICKHOUSE !== "1") {
    try {
      await refreshServingCache();
      setInterval(() => {
        void refreshServingCache().catch((error) => console.error("cache refresh failed", String(error)));
      }, 120000);
    } catch (error) {
      if (!activeCatalogEngine && !activeVectorStore) throw error;
      console.warn("ClickHouse cache unavailable; serving from local intelligence files:", String(error));
    }
  }
  const serviceAudience = process.env.INTEL_SERVICE_DID ?? "did:web:intel.catbird.blue#catbird_intel";
  const requireServiceAuth = process.env.INTEL_REQUIRE_AUTH !== "0";
  const authenticate = createServiceAuthenticator({
    audience: serviceAudience,
    resolveAtprotoKey: createDidKeyResolver(),
    internalBffSecret: process.env.INTEL_INTERNAL_BFF_SECRET,
  });
  const server = Bun.serve({
    hostname: process.env.HOST ?? "127.0.0.1",
    port: Number(process.env.PORT ?? 3210),
    idleTimeout: 60,
    maxRequestBodySize: 65536,
    fetch: createApiHandler({ authenticate, requireServiceAuth }),
  });
  console.log(`catbird-intel-api listening on ${server.url} (aud=${serviceAudience}, requireAuth=${requireServiceAuth})`);
}
