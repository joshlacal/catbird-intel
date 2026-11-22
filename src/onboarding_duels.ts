/**
 * Taste duels generated from what a viewer picked.
 *
 * A duel offers two scenes the viewer did not pick, both close to something they did. Choosing a
 * side adds a few accounts from it to the lineup, so every duel can change the outcome, and the
 * posts shown for a side are by the accounts that would be added. Nothing here is a fixed list:
 * the candidates are the scenes whose people overlap with a pick (the catalog's neighbour links)
 * and the scenes that post about the most similar things (closest post centroids).
 */

import type { IntelCommunityEntry } from "./intel_catalog";

export interface GeneratedDuelOption {
  communityId: number;
  title: string;
  subtitle: string;
  sceneTitle: string;
  samplePostUris: string[];
}

export interface GeneratedDuel {
  swipeId: string;
  step: number;
  kind: "subscene_split" | "taste_bridge";
  prompt: string;
  optionA: GeneratedDuelOption;
  optionB: GeneratedDuelOption;
}

/** What the generator needs to know about the catalog and about this viewer. */
export interface DuelSource {
  community(communityId: number): IntelCommunityEntry | null;
  /** The scenes whose posts are most like this one's, closest first. */
  nearestByContent(communityId: number): readonly number[];
  isHostilePair(a: number, b: number): boolean;
  /** Posts this viewer may see, by the accounts they would follow if the scene won, lineup order first. */
  offerPosts(communityId: number): string[];
  /** How many accounts in the scene this viewer could still be offered. */
  offerableCount(communityId: number): number;
  /** How many accounts in the scene this viewer already follows. */
  followedCount(communityId: number): number;
  /** A number in (0, 1) that is stable for this viewer's session and differs between sessions. */
  noise(communityId: number): number;
}

export interface DuelConfig {
  /** A side must be able to add at least this many accounts. */
  minOffer: number;
  /** A side needs at least this many posts to show. */
  minPosts: number;
  /** How strongly the order of candidates varies between sessions (0 = fixed order). */
  rotation: number;
  /** At most this many duels grow out of one picked scene. */
  maxPerPick: number;
}

export const DUEL_DEFAULTS: DuelConfig = { minOffer: 3, minPosts: 1, rotation: 0.6, maxPerPick: 2 };

interface Candidate {
  communityId: number;
  community: IntelCommunityEntry;
  /** The picked scene this candidate is closest to. */
  parent: number;
  parentStrength: number;
  /** Whether the candidate posts about much the same things as that picked scene. */
  alike: boolean;
  score: number;
  key: number;
  posts: string[];
}

const SUBTITLE_MAX = 120;

function shorten(text: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= SUBTITLE_MAX) return clean;
  const cut = clean.slice(0, SUBTITLE_MAX);
  const lastSpace = cut.lastIndexOf(" ");
  return `${cut.slice(0, lastSpace > 60 ? lastSpace : SUBTITLE_MAX).replace(/[\s,;:.]+$/, "")}…`;
}

function option(candidate: Candidate): GeneratedDuelOption {
  return {
    communityId: candidate.communityId,
    title: candidate.community.title,
    subtitle: shorten(candidate.community.description),
    sceneTitle: candidate.community.sceneTitle,
    samplePostUris: candidate.posts.slice(0, 3),
  };
}

function gatherCandidates(picks: readonly number[], source: DuelSource, config: DuelConfig): Candidate[] {
  const picked = new Set(picks);
  // Credit from one pick: its overlap credit and its likeness credit add up, so a scene that is
  // both a listed neighbour and close in content leads.
  const perPick = new Map<number, Map<number, { strength: number; alike: boolean }>>();
  const credit = (communityId: number, parent: number, strength: number, alike: boolean) => {
    if (picked.has(communityId)) return;
    let fromPicks = perPick.get(communityId);
    if (!fromPicks) perPick.set(communityId, (fromPicks = new Map()));
    const entry = fromPicks.get(parent);
    if (entry) {
      entry.strength += strength;
      entry.alike = entry.alike || alike;
    } else {
      fromPicks.set(parent, { strength, alike });
    }
  };

  for (const pick of picks) {
    const community = source.community(pick);
    if (!community) continue;
    // Both lists come closest first; a scene close to two picks collects credit from both.
    community.neighborCommunityIds.forEach((id, position) => credit(id, pick, 1 / (1 + position), false));
    source.nearestByContent(pick).forEach((id, position) => credit(id, pick, 0.8 / (1 + 0.5 * position), true));
  }

  const found = new Map<number, { parent: number; parentStrength: number; alike: boolean; score: number }>();
  for (const [communityId, fromPicks] of perPick) {
    let best: { parent: number; strength: number; alike: boolean } | null = null;
    let score = 0;
    for (const [parent, entry] of fromPicks) {
      score += entry.strength;
      if (!best || entry.strength > best.strength) best = { parent, strength: entry.strength, alike: entry.alike };
    }
    found.set(communityId, { parent: best!.parent, parentStrength: best!.strength, alike: best!.alike, score });
  }

  const candidates: Candidate[] = [];
  for (const [communityId, entry] of found) {
    const community = source.community(communityId);
    if (!community) continue;
    if (picks.some((pick) => source.isHostilePair(pick, communityId))) continue;
    if (source.offerableCount(communityId) < config.minOffer) continue;
    const posts = source.offerPosts(communityId);
    if (posts.length < config.minPosts) continue;
    // Already following people in a scene is a sign it belongs, up to a point.
    const score = entry.score + Math.min(source.followedCount(communityId), 6) * 0.08;
    const u = Math.min(Math.max(source.noise(communityId), 1e-9), 1 - 1e-9);
    const key = Math.log(score) - config.rotation * Math.log(-Math.log(u));
    candidates.push({
      communityId,
      community,
      parent: entry.parent,
      parentStrength: entry.parentStrength,
      alike: entry.alike,
      score,
      key,
      posts,
    });
  }
  return candidates.sort((x, y) => y.key - x.key || x.communityId - y.communityId);
}

/**
 * Up to `maxSwipes` duels for these picks. Each side is a scene the viewer did not pick, and no
 * scene appears twice. Fewer duels (or none) come back when the picks have too few neighbours.
 */
export function generateDuels(params: {
  picks: readonly number[];
  maxSwipes: number;
  source: DuelSource;
  config?: Partial<DuelConfig>;
}): GeneratedDuel[] {
  const config = { ...DUEL_DEFAULTS, ...(params.config ?? {}) };
  const { source } = params;
  const candidates = gatherCandidates(params.picks, source, config);

  // Group by the picked scene each candidate grew out of, keeping the overall order.
  const byParent = new Map<number, Candidate[]>();
  for (const candidate of candidates) {
    const group = byParent.get(candidate.parent);
    if (group) group.push(candidate);
    else byParent.set(candidate.parent, [candidate]);
  }

  const pairs: { a: Candidate; b: Candidate; parent: number | null }[] = [];
  const used = new Set<number>();
  const takePair = (group: Candidate[]): [Candidate, Candidate] | null => {
    const free = group.filter((c) => !used.has(c.communityId));
    for (let i = 0; i < free.length; i++) {
      for (let j = i + 1; j < free.length; j++) {
        if (!source.isHostilePair(free[i]!.communityId, free[j]!.communityId)) return [free[i]!, free[j]!];
      }
    }
    return null;
  };

  // One duel per picked scene first, then a second round, so a single pick cannot fill the list.
  for (let round = 0; round < config.maxPerPick && pairs.length < params.maxSwipes; round++) {
    for (const [parent, group] of byParent) {
      if (pairs.length >= params.maxSwipes) break;
      const pair = takePair(group);
      if (!pair) continue;
      used.add(pair[0].communityId);
      used.add(pair[1].communityId);
      pairs.push({ a: pair[0], b: pair[1], parent });
    }
  }
  // Whatever is left over pairs across picked scenes.
  while (pairs.length < params.maxSwipes) {
    const pair = takePair(candidates);
    if (!pair) break;
    used.add(pair[0].communityId);
    used.add(pair[1].communityId);
    pairs.push({ a: pair[0], b: pair[1], parent: pair[0].parent === pair[1].parent ? pair[0].parent : null });
  }

  return pairs.map(({ a, b, parent }, index) => {
    const parentCommunity = parent === null ? null : source.community(parent);
    // Two scenes that post about much the same things as the pick are a finer split of it; anything
    // else reaches across to where its people also spend time.
    const finerSplit = parentCommunity !== null && a.alike && b.alike;
    const prompt = parentCommunity
      ? `You picked ${parentCommunity.title}. Which of its neighbours belongs on your timeline too?`
      : "Two scenes close to what you picked. Which belongs on your timeline?";
    const [low, high] = a.communityId < b.communityId ? [a.communityId, b.communityId] : [b.communityId, a.communityId];
    return {
      swipeId: `live:${low}:${high}`,
      step: index + 1,
      kind: finerSplit ? "subscene_split" as const : "taste_bridge" as const,
      prompt,
      optionA: option(a),
      optionB: option(b),
    };
  });
}
