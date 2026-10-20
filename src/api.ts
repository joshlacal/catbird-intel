#!/usr/bin/env bun
/** Local serving candidate. Ranking remains heuristic and deployment is unqualified.
 * Public post responses contain identifiers for viewer-aware AppView hydration.
 * No invented observations, unverified authentication, or exposure-budget claims.
 */
import { ClickHouseWriter } from "./clickhouse.ts";
import { AuthenticationError, createServiceAuthenticator, type ServiceAuthenticator } from "./service_auth.ts";
import { SERVING_CONTRACT_VERSION, InvalidRequest, parseControls, passesControls, controlReport,
  recommendationAction, type ServingControls } from "./serving_contract.ts";

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

/** Explicit dependency seam for local fixture-backed HTTP tests and future atomic publication. */
export function installServingSnapshot(snapshot: {
  communities: CommunityMeta[]; items: PostCandidate[]; directory: DirectoryEntry[];
  modes?: AuthorModeMix[]; hiddenDids?: string[]; noUnauthDids?: string[]; version: string;
}) {
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
    derived_generation: null, cache_refreshed_at: cacheRefreshedAt,
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

export function createApiHandler(options: { authenticate?: ServiceAuthenticator } = {}) {
  const authenticate = options.authenticate ?? createServiceAuthenticator();
  const json = (body: Record<string, unknown>, status = 200) =>
    Response.json({ ...body, ...contractMetadata() }, { status, headers: { "cache-control": "no-store" } });

  return async (request: Request): Promise<Response> => {
    try {
      const url = new URL(request.url);
      const path = url.pathname;
      if (path === "/health") return json({
        ok: true,
        ready_for_public_promotion: false,
        blockers: ["derived_generation_unversioned", "live_deletion_overlay_unqualified", "viewer_moderation_requires_hydration"],
        telemetry_persistence: "disabled_pending_versioned_schema",
        communities: communitiesById.size,
        communities_with_posts: itemsByCommunity.size,
        author_modes_cached: authorModeByDid.size,
      });
      const method = path.startsWith("/xrpc/") ? path.slice(6) : "blue.catbird.intel.getCommunity";
      const viewer = await authenticate(request, method);
      const unauthenticated = viewer === null; // Never infer authentication from request fields.
      const postRoute = path === "/xrpc/blue.catbird.onboarding.previewTimeline" ||
        path === "/xrpc/blue.catbird.onboarding.solveBundle" ||
        path === "/xrpc/blue.catbird.onboarding.getCalibrationSwipes";
      if (request.method !== (postRoute ? "POST" : "GET")) return json({ error: "MethodNotAllowed" }, 405);

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
      if (error instanceof AuthenticationError) return json({ error: "AuthenticationRequired", message: error.message }, 401);
      if (error instanceof InvalidRequest) return json({ error: "InvalidRequest", message: error.message }, 400);
      console.error("serving request failed", error instanceof Error ? error.name : "UnknownError");
      return json({ error: "InternalServerError" }, 500);
    }
  };
}

if (import.meta.main) {
  await refreshServingCache();
  setInterval(() => { void refreshServingCache().catch((error) => console.error("cache refresh failed", String(error))); }, 120000);
  const server = Bun.serve({ hostname: process.env.HOST ?? "127.0.0.1", port: Number(process.env.PORT ?? 3210),
    maxRequestBodySize: 65536, fetch: createApiHandler() });
  console.log(`catbird-intel-api local candidate listening on ${server.url}; authenticated serving is unconfigured`);
}
