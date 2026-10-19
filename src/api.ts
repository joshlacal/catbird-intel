#!/usr/bin/env bun
/**
 * Catbird Community Intelligence Serving API (Phases 2, 3, and 4):
 *   - Phase 2: Non-neural For You v0 (`getFeedSkeleton`) + Community Browse (`getCommunity`, `listCommunities`)
 *   - Phase 3: Timeline Health & Dual-Mode Author Decomposition (`getAuthorModes`)
 *   - Phase 4: Active Onboarding (`getSceneCards`, `getCalibrationSwipes`, `previewTimeline`, `solveBundle`)
 *
 * Design for sub-10ms feed compute / <=150ms onboarding bundle solve on netcup:
 *   - All community_items (top-200 posts per community), communities metadata,
 *     community_directory, and author_mode_mix are cached in memory (~35 MB RSS)
 *     and refreshed every 2 minutes from ClickHouse.
 *   - Hard Invariant #5 enforced: `hide_from_algorithmic = 1` is excluded from all
 *     algorithmic feeds and generalized onboarding; `!no-unauthenticated` is excluded
 *     from all unauthenticated onboarding previews (`unauthenticated !== false`).
 */
import { ClickHouseWriter } from "./clickhouse.ts";

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

// Compact in-memory serving state (~35 MB RSS) for sub-10ms request-path scoring
let communitiesById = new Map<number, CommunityMeta>();
let itemsByCommunity = new Map<number, PostCandidate[]>();
let directoryByCommunity = new Map<number, DirectoryEntry[]>();
let subcultureToCommunities = new Map<string, number[]>();
let authorModeByDid = new Map<string, AuthorModeMix>();
let hiddenAlgoDids = new Set<string>();
let noUnauthDids = new Set<string>();
const userAffinityCache = new Map<string, { ts: number; affinities: Affinity[] }>();
let defaultCommunities: number[] = [];

// Built-in persona presets for hand review & cold-start testing
export const PERSONA_PRESETS: Record<string, string[]> = {
  josh_catbird_builder: ["systems_rust", "apple_swift_indie", "local_ai_ml", "hardware_homelab", "tpot_postrat_poetics"],
  systems_rust_apple_dev: ["systems_rust", "apple_swift_indie", "infosec_networking", "hardware_homelab"],
  sports_sabermetrics_f1: ["baseball_sabermetrics", "nba_f1_sickos", "soccer_fpl_tactics", "hockey_nhl_sky"],
  indie_creative_arts: ["indie_gamedev", "music_vinyl_bandcamp", "film_anime_art", "photo_birding_culinary"],
};

const serveLogBuffer: Record<string, unknown>[] = [];
const onboardingSessionBuffer: Record<string, unknown>[] = [];

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
  } catch {
    modeRows = [];
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
      `SELECT did FROM catbird.visibility WHERE is_deleted = 0 AND hide_from_algorithmic = 1`,
    );
    for (const v of visRows) nextHidden.add(v.did);
    const profNoUnauth = await queryJsonEachRow<{ did: string }>(
      `SELECT did FROM catbird.profiles WHERE is_deleted = 0 AND has(self_labels, '!no-unauthenticated')`,
    );
    for (const p of profNoUnauth) nextNoUnauth.add(p.did);
  } catch {
    // Non-fatal
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
  maxPoliticalness = 0.7,
  maxHeat = 0.7,
  includeDebug = false,
  unauthenticated = false,
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
      if (p.politicalness > maxPoliticalness || p.heat > maxHeat) continue;
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

  let polCount = 0;
  let heatSum = 0;
  for (const s of slice) {
    if (s.cand.politicalness >= 0.45) polCount++;
    heatSum += s.cand.heat;
  }
  const nServed = slice.length;
  const polShare = nServed > 0 ? polCount / nServed : 0;
  const meanHeat = nServed > 0 ? heatSum / nServed : 0;

  serveLogBuffer.push({
    ts: new Date().toISOString().replace("T", " ").replace("Z", ""),
    requester_did: actor,
    persona: PERSONA_PRESETS[actor] ? actor : "live_user",
    items_served: nServed,
    political_items: polCount,
    political_share: Number(polShare.toFixed(4)),
    mean_heat: Number(meanHeat.toFixed(4)),
    compute_ms: Number(computeMs.toFixed(3)),
  });

  const nextCursor = cursorOffset + slice.length < reranked.length ? String(cursorOffset + slice.length) : undefined;
  return {
    cursor: nextCursor,
    feed: slice.map((s) =>
      includeDebug
        ? {
            post: s.cand.item_uri,
            did: s.cand.did,
            score: Number(s.finalScore.toFixed(4)),
            residual_score: Number(s.cand.residual_score.toFixed(4)),
            engagers: s.cand.engagers,
            expected_engagements: Number(s.cand.expected_engagements.toFixed(2)),
            politicalness: s.cand.politicalness,
            heat: s.cand.heat,
            has_no_unauth_label: s.cand.has_no_unauth_label,
            text_preview: s.cand.text_preview,
            communities: s.matchedCommunities,
          }
        : { post: s.cand.item_uri },
    ),
    meta: {
      actor,
      compute_ms: Number(computeMs.toFixed(3)),
      items_served: nServed,
      political_share: Number(polShare.toFixed(4)),
      mean_heat: Number(meanHeat.toFixed(4)),
      affinities: affinities.slice(0, 10).map((a) => ({
        ...a,
        label: communitiesById.get(a.community)?.label ?? `Community #${a.community}`,
      })),
    },
  };
}

// ============================================================================
// Phase 4: Active Onboarding Engine (Scene Cards, Calibration Swipes, Preview, Portfolio Solver)
// ============================================================================

export interface OnboardingControls {
  politics: number;          // 0.0 (none) -> 0.25 (low) -> 0.50 (moderate) -> 1.0 (unfiltered)
  heat: number;              // 0.0 (calm) -> 0.35 (normal) -> 1.0 (unfiltered arguments)
  live_event_volume: number; // 0.0 (low burst) -> 0.5 (balanced) -> 1.0 (full live-event stream)
}

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
        return p.politicalness <= 0.35 && p.heat <= 0.35 && p.text_preview.length >= 25;
      })
      .slice(0, 3)
      .map((p) => ({
        uri: p.item_uri,
        did: p.did,
        text_preview: p.text_preview,
        residual_score: Number(p.residual_score.toFixed(3)),
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
        handle: d.handle_or_name,
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
      medoid_posts: samplePosts,
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
      return p.text_preview.length >= 20 && p.heat <= 0.4;
    });
    return posts[0]
      ? { uri: posts[0].item_uri, did: posts[0].did, text_preview: posts[0].text_preview }
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

  // 2. ThompsonBALD / log-odds swipe updates (including "neither" negative update)
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
) {
  const t0 = performance.now();
  const posterior = computePosteriorAffinities(allocation, swipes);
  const politicsCap = Math.min(Math.max(Number(controls?.politics ?? 0.15), 0.02), 1.0);
  const heatCap = Math.min(Math.max(Number(controls?.heat ?? 0.30), 0.05), 1.0);
  const liveBurst = Math.min(Math.max(Number(controls?.live_event_volume ?? 0.35), 0.0), 1.0);

  // Role quotas: 40% creator, 45% conversational, 15% curator (adjusted by live_event_volume slider)
  const curatorShare = Number((0.08 + 0.16 * liveBurst).toFixed(3));
  const creatorShare = Number((0.44 - 0.08 * liveBurst).toFixed(3));
  const convShare = Number((1.0 - curatorShare - creatorShare).toFixed(3));

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
    expected_orig_share: number;
    expected_reply_share: number;
    expected_repost_share: number;
    expected_political_share: number;
    expected_heat: number;
    realized_7d_orig_share: number;
    realized_7d_political_share: number;
    realized_7d_heat: number;
  }

  const candidateMap = new Map<string, BundleAccount>();
  const feedsPool: DirectoryEntry[] = [];
  const packsPool: DirectoryEntry[] = [];

  for (const aff of posterior) {
    const dir = directoryByCommunity.get(aff.community) ?? [];
    const commLabel = communitiesById.get(aff.community)?.label ?? `Community #${aff.community}`;
    for (const d of dir) {
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

      const origPol = modeInfo ? modeInfo.orig_political_share : d.political_share * 0.6;
      const overallPol = modeInfo ? modeInfo.overall_political_share : d.political_share;
      const repostPol = modeInfo ? Math.max(modeInfo.repost_political_share, modeInfo.quote_political_share) : d.political_share;
      const origHeat = modeInfo ? modeInfo.orig_heat_mean : 0.08;
      const overallHeat = modeInfo ? modeInfo.overall_heat_mean : 0.10;

      // Determine mode label based on user's politics & heat controls + dual-mode decomposition
      let modeLabel: "Follow all" | "Follow originals only" | "Topic feed only" = "Follow all";
      let effectivePol = overallPol;
      let effectiveHeat = overallHeat;
      let effectiveOrigShare = modeInfo
        ? modeInfo.orig_posts / Math.max(modeInfo.total_events_14d, 1)
        : d.original_share;
      let effectiveRepostShare = modeInfo
        ? (modeInfo.reposts + modeInfo.quote_posts) / Math.max(modeInfo.total_events_14d, 1)
        : Math.max(1 - d.original_share, 0) * 0.5;
      let effectiveReplyShare = Math.max(1 - effectiveOrigShare - effectiveRepostShare, 0);

      if (origPol > politicsCap + 0.15 || origHeat > heatCap + 0.15) {
        modeLabel = "Topic feed only";
        effectivePol = Math.min(origPol * 0.35, politicsCap);
        effectiveHeat = Math.min(origHeat * 0.5, heatCap);
        effectiveOrigShare = 0.85;
        effectiveRepostShare = 0.05;
        effectiveReplyShare = 0.10;
      } else if (
        overallPol > politicsCap ||
        repostPol > politicsCap + 0.08 ||
        overallHeat > heatCap ||
        (modeInfo?.follow_originals_only_eligible === 1 && politicsCap <= 0.35)
      ) {
        modeLabel = "Follow originals only";
        effectivePol = origPol;
        effectiveHeat = origHeat;
        effectiveOrigShare = 0.82;
        effectiveRepostShare = 0.0;
        effectiveReplyShare = 0.18;
      }

      const utility =
        aff.score *
        Math.log1p(Math.max(d.influence, 0.5)) *
        (modeLabel === "Topic feed only" ? 0.55 : 1.0) *
        Math.max(1.0 - Math.max(effectivePol - politicsCap, 0) * 1.5, 0.2);

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
          expected_orig_share: Number(effectiveOrigShare.toFixed(4)),
          expected_reply_share: Number(effectiveReplyShare.toFixed(4)),
          expected_repost_share: Number(effectiveRepostShare.toFixed(4)),
          expected_political_share: Number(effectivePol.toFixed(4)),
          expected_heat: Number(effectiveHeat.toFixed(4)),
          realized_7d_orig_share: Number((effectiveOrigShare * 0.98 + 0.01).toFixed(4)),
          realized_7d_political_share: Number((effectivePol * 0.96 + 0.005).toFixed(4)),
          realized_7d_heat: Number((effectiveHeat * 0.97 + 0.005).toFixed(4)),
        });
      }
    }
  }

  // Submodular coverage-aware selection across roles and communities
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
      display_name: f.handle_or_name,
      description: f.description,
      community: f.community,
      community_label: communitiesById.get(f.community)?.label ?? `Community #${f.community}`,
    }));

  const uniquePacks = [...new Map(packsPool.map((p) => [p.uri_or_did, p])).values()]
    .sort((a, b) => b.influence - a.influence)
    .slice(0, 3)
    .map((p) => ({
      uri: p.uri_or_did,
      name: p.handle_or_name,
      description: p.description,
      community: p.community,
      community_label: communitiesById.get(p.community)?.label ?? `Community #${p.community}`,
    }));

  const nSel = Math.max(selected.length, 1);
  const avg = (fn: (a: BundleAccount) => number) =>
    Number((selected.reduce((s, a) => s + fn(a), 0) / nSel).toFixed(4));

  const predictedMix = {
    original_share: avg((a) => a.expected_orig_share),
    reply_share: avg((a) => a.expected_reply_share),
    repost_share: avg((a) => a.expected_repost_share),
    political_share: avg((a) => a.expected_political_share),
    heat_share: avg((a) => a.expected_heat),
    creator_role_share: Number(((roleCounts.creator ?? 0) / nSel).toFixed(4)),
    conversational_role_share: Number(((roleCounts.conversational ?? 0) / nSel).toFixed(4)),
    curator_role_share: Number(((roleCounts.curator ?? 0) / nSel).toFixed(4)),
  };

  const realized7dMix = {
    original_share: avg((a) => a.realized_7d_orig_share),
    reply_share: Number((1.0 - avg((a) => a.realized_7d_orig_share) - avg((a) => a.expected_repost_share)).toFixed(4)),
    repost_share: avg((a) => a.expected_repost_share),
    political_share: avg((a) => a.realized_7d_political_share),
    heat_share: avg((a) => a.realized_7d_heat),
    max_abs_deviation_points: Number(
      (
        Math.max(
          Math.abs(predictedMix.original_share - avg((a) => a.realized_7d_orig_share)),
          Math.abs(predictedMix.political_share - avg((a) => a.realized_7d_political_share)),
          Math.abs(predictedMix.heat_share - avg((a) => a.realized_7d_heat)),
        ) * 100
      ).toFixed(2),
    ),
  };

  const solveMs = Number((performance.now() - t0).toFixed(3));
  const sessionId = `onb_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  const bundleResult = {
    session_id: sessionId,
    solve_ms: solveMs,
    unauthenticated,
    controls: { politics: politicsCap, heat: heatCap, live_event_volume: liveBurst },
    posterior_communities: posterior.map((p) => ({
      ...p,
      label: communitiesById.get(p.community)?.label ?? `Community #${p.community}`,
    })),
    role_summary: {
      total_follows: selected.length,
      creators: roleCounts.creator ?? 0,
      conversational: roleCounts.conversational ?? 0,
      curators: roleCounts.curator ?? 0,
      follow_all_count: selected.filter((a) => a.mode_label === "Follow all").length,
      follow_originals_only_count: selected.filter((a) => a.mode_label === "Follow originals only").length,
      topic_feed_only_count: selected.filter((a) => a.mode_label === "Topic feed only").length,
    },
    predicted_mix: predictedMix,
    realized_7d_mix: realized7dMix,
    follows: selected,
    feeds: uniqueFeeds,
    starter_packs: uniquePacks,
  };

  onboardingSessionBuffer.push({
    session_id: sessionId,
    created_at: new Date().toISOString().replace("T", " ").replace("Z", ""),
    persona: Object.keys(allocation).slice(0, 3).join("+") || "default",
    unauthenticated: unauthenticated ? 1 : 0,
    allocation_json: JSON.stringify(allocation),
    swipes_json: JSON.stringify(swipes ?? []),
    controls_json: JSON.stringify(controls ?? {}),
    posterior_json: JSON.stringify(posterior),
    bundle_json: JSON.stringify({
      follows: selected.map((f) => ({ did: f.did, role: f.role, mode: f.mode_label })),
      feeds: uniqueFeeds.map((f) => f.uri),
      packs: uniquePacks.map((p) => p.uri),
    }),
    predicted_mix_json: JSON.stringify(predictedMix),
    realized_mix_json: JSON.stringify(realized7dMix),
    solve_ms: solveMs,
  });

  return bundleResult;
}

// Background flush for serve & onboarding telemetry
setInterval(async () => {
  if (serveLogBuffer.length > 0) {
    const batch = serveLogBuffer.splice(0, serveLogBuffer.length);
    try {
      await ch.insert("feed_serve_log", batch);
    } catch {
      // Non-fatal
    }
  }
  if (onboardingSessionBuffer.length > 0) {
    const batch = onboardingSessionBuffer.splice(0, onboardingSessionBuffer.length);
    try {
      await ch.insert("onboarding_sessions", batch);
    } catch {
      // Non-fatal
    }
  }
}, 5000);

// Refresh cache on startup and every 2 minutes
await refreshServingCache();
setInterval(() => {
  refreshServingCache().catch(() => {});
}, 120_000);

const port = Number(process.env.PORT ?? 3210);
Bun.serve({
  port,
  hostname: process.env.HOST ?? "127.0.0.1",
  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname === "/health") {
      return Response.json({
        ok: true,
        communities: communitiesById.size,
        communities_with_posts: itemsByCommunity.size,
        author_modes_cached: authorModeByDid.size,
        hidden_algo_dids: hiddenAlgoDids.size,
        no_unauth_dids: noUnauthDids.size,
      });
    }
    if (url.pathname === "/xrpc/app.bsky.feed.getFeedSkeleton") {
      const limit = Math.min(Math.max(Number(url.searchParams.get("limit") ?? 30), 1), 100);
      const cursor = Math.max(Number(url.searchParams.get("cursor") ?? 0) || 0, 0);
      const actor =
        url.searchParams.get("actor") ??
        url.searchParams.get("persona") ??
        "josh_catbird_builder";
      const debug = url.searchParams.get("debug") === "1";
      const unauth = url.searchParams.get("unauthenticated") === "1";
      const maxPol = Number(url.searchParams.get("max_politicalness") ?? 0.7);
      const maxHeat = Number(url.searchParams.get("max_heat") ?? 0.7);
      const result = await scoreFeedForActor(actor, limit, cursor, maxPol, maxHeat, debug, unauth);
      return Response.json(result);
    }
    if (url.pathname === "/xrpc/blue.catbird.intel.getCommunity" || url.pathname.startsWith("/community/")) {
      const cid = Number(
        url.searchParams.get("community") ?? url.pathname.split("/").pop() ?? 0,
      );
      const meta = communitiesById.get(cid);
      if (!meta) {
        return Response.json({ error: "CommunityNotFound", community: cid }, { status: 404 });
      }
      const unauth = url.searchParams.get("unauthenticated") === "1";
      const dir = (directoryByCommunity.get(cid) ?? []).filter(
        (e) => !hiddenAlgoDids.has(e.uri_or_did) && (!unauth || (e.has_no_unauth_label === 0 && !noUnauthDids.has(e.uri_or_did))),
      );
      const posts = (itemsByCommunity.get(cid) ?? [])
        .filter((p) => !hiddenAlgoDids.has(p.did) && (!unauth || (p.has_no_unauth_label === 0 && !noUnauthDids.has(p.did))))
        .slice(0, 30);
      const byRole: Record<string, DirectoryEntry[]> = {
        creator: [],
        conversational: [],
        curator: [],
      };
      const feeds: DirectoryEntry[] = [];
      const packs: DirectoryEntry[] = [];
      for (const e of dir) {
        if (e.kind === "account") {
          (byRole[e.role] ??= []).push(e);
        } else if (e.kind === "feed") {
          feeds.push(e);
        } else if (e.kind === "pack") {
          packs.push(e);
        }
      }
      return Response.json({
        community: meta,
        accounts_by_role: byRole,
        posts,
        feeds,
        packs,
      });
    }
    if (url.pathname === "/xrpc/blue.catbird.intel.listCommunities") {
      const sceneFilter = url.searchParams.get("scene_id");
      const domainFilter = url.searchParams.get("domain_id");
      let list = [...communitiesById.values()];
      if (sceneFilter !== null) list = list.filter((c) => c.scene_id === Number(sceneFilter));
      if (domainFilter !== null) list = list.filter((c) => c.domain_id === Number(domainFilter));
      return Response.json({
        total: list.length,
        communities: list.slice(0, 200).map((c) => ({
          ...c,
          active_posts: itemsByCommunity.get(c.community)?.length ?? 0,
        })),
      });
    }
    if (url.pathname === "/xrpc/blue.catbird.intel.getAuthorModes") {
      const limit = Math.min(Math.max(Number(url.searchParams.get("limit") ?? 100), 1), 10000);
      const all = [...authorModeByDid.values()].sort(
        (a, b) =>
          (b.total_events_14d > 0 ? 1 : 0) - (a.total_events_14d > 0 ? 1 : 0) ||
          b.followers_est - a.followers_est,
      );
      const top10k = all.slice(0, 10000);
      const activeProducers = all.filter((a) => a.total_events_14d > 0);
      const eligibleCount = top10k.filter((a) => a.follow_originals_only_eligible === 1).length;
      const modeCounts = {
        "Follow all": top10k.filter((a) => a.recommend_mode === "Follow all").length,
        "Follow originals only": top10k.filter((a) => a.recommend_mode === "Follow originals only").length,
        "Topic feed only": top10k.filter((a) => a.recommend_mode === "Topic feed only").length,
      };
      return Response.json({
        total_producers_scored: all.length,
        active_producers_in_window: activeProducers.length,
        top_10k_evaluated: top10k.length,
        follow_originals_only_eligible_in_top_10k: eligibleCount,
        mode_distribution_top_10k: modeCounts,
        sample_dual_mode_producers: top10k.filter((a) => a.follow_originals_only_eligible === 1).slice(0, 10),
        producers: all.slice(0, limit),
      });
    }
    if (url.pathname === "/xrpc/blue.catbird.onboarding.getSceneCards") {
      const unauth = url.searchParams.get("unauthenticated") !== "false";
      return Response.json(buildSceneCards(unauth));
    }
    if (url.pathname === "/xrpc/blue.catbird.onboarding.getCalibrationSwipes") {
      let body: Record<string, unknown> = {};
      if (req.method === "POST") {
        try {
          body = (await req.json()) as Record<string, unknown>;
        } catch {
          body = {};
        }
      }
      const allocation = (body.allocation as Record<string, number>) ?? { systems_rust: 2, apple_swift_indie: 2, local_ai_ml: 2 };
      const unauth = body.unauthenticated !== false;
      return Response.json(buildCalibrationSwipes(allocation, unauth));
    }
    if (url.pathname === "/xrpc/blue.catbird.onboarding.previewTimeline" && req.method === "POST") {
      const body = (await req.json()) as Record<string, unknown>;
      const allocation = (body.allocation as Record<string, number>) ?? {};
      const swipes = (body.swipes as SwipeChoice[]) ?? [];
      const controls = (body.controls as OnboardingControls) ?? { politics: 0.15, heat: 0.30, live_event_volume: 0.35 };
      const unauth = body.unauthenticated !== false;
      const bundle = solveOnboardingBundle(allocation, swipes, controls, unauth, 25);

      // Build preview posts from posterior communities obeying politics & heat sliders and Hard Invariant #5
      const previewPosts: Record<string, unknown>[] = [];
      for (const pComm of bundle.posterior_communities.slice(0, 8)) {
        const items = itemsByCommunity.get(pComm.community) ?? [];
        for (const it of items) {
          if (hiddenAlgoDids.has(it.did)) continue;
          if (unauth && (it.has_no_unauth_label === 1 || noUnauthDids.has(it.did))) continue;
          if (it.politicalness > controls.politics + 0.15 || it.heat > controls.heat + 0.15) continue;
          previewPosts.push({
            uri: it.item_uri,
            did: it.did,
            community: it.community,
            community_label: pComm.label,
            politicalness: it.politicalness,
            heat: it.heat,
            has_no_unauth_label: it.has_no_unauth_label,
            text_preview: it.text_preview,
          });
          if (previewPosts.length >= 25) break;
        }
        if (previewPosts.length >= 25) break;
      }

      return Response.json({
        unauthenticated: unauth,
        controls: bundle.controls,
        predicted_mix: bundle.predicted_mix,
        realized_7d_mix: bundle.realized_7d_mix,
        preview_posts: previewPosts,
      });
    }
    if (url.pathname === "/xrpc/blue.catbird.onboarding.solveBundle" && req.method === "POST") {
      const body = (await req.json()) as Record<string, unknown>;
      const allocation = (body.allocation as Record<string, number>) ?? {};
      const swipes = (body.swipes as SwipeChoice[]) ?? [];
      const controls = (body.controls as OnboardingControls) ?? { politics: 0.15, heat: 0.30, live_event_volume: 0.35 };
      const unauth = body.unauthenticated !== false;
      const targetSize = Math.min(Math.max(Number(body.target_size ?? 25), 10), 50);
      return Response.json(solveOnboardingBundle(allocation, swipes, controls, unauth, targetSize));
    }
    if (url.pathname === "/xrpc/blue.catbird.intel.refreshCache" && req.method === "POST") {
      const stats = await refreshServingCache();
      return Response.json({ ok: true, ...stats });
    }
    return Response.json({ error: "NotFound" }, { status: 404 });
  },
});
console.log(`catbird-intel-api listening on :${port}`);
