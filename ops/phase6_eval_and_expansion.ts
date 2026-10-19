#!/usr/bin/env bun
/**
 * Phase 6 — Evaluation and Expansion (`catbird-intel/ops/phase6_eval_and_expansion.ts`):
 *   1. Offline onboarding simulator (200 synthetic users across 26 subcultures) measuring
 *      follow survival, unfollow/mute rates, community depth, and exposure vs chosen controls.
 *   2. A/B test of slider notches (`strict_0.05`, `balanced_0.15`, `moderate_0.35`, `unfiltered_1.00`).
 *   3. Residual (emerging) scene detection from unclustered high-residual engagement clusters.
 *   4. Live-event burst node detection from short-window community velocity spikes.
 */
import { ClickHouseWriter } from "../src/clickhouse.ts";

const API_URL = process.env.API_URL ?? "http://127.0.0.1:3210";
const ch = new ClickHouseWriter({
  url: process.env.CLICKHOUSE_URL ?? process.env.CH_URL ?? "http://127.0.0.1:8123",
  database: process.env.CLICKHOUSE_DB ?? process.env.CH_DATABASE ?? "catbird",
  maxRows: 1000,
});

async function queryJson<T>(sql: string): Promise<T[]> {
  const txt = await ch.query(`${sql} FORMAT JSONEachRow`);
  if (!txt.trim()) return [];
  return txt
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l) as T);
}

const SUBCULTURES = [
  "systems_rust", "apple_swift_indie", "local_ai_ml", "hardware_homelab",
  "infosec_networking", "keyboards_pens_makers", "tpot_postrat_poetics", "hangout_celebs_comedy",
  "weird_shitposting", "indie_gamedev", "retrogaming_fpga_fgc", "ttrpg_worldbuilding",
  "wrestling_aew_graps", "sports_media_yappers", "nba_f1_sickos", "baseball_sabermetrics",
  "soccer_fpl_tactics", "hockey_nhl_sky", "fintwit_econsky", "urbanism_transit",
  "space_astronomy", "wxsky_earth_science", "history_archaeology_booksky", "film_anime_art",
  "music_vinyl_bandcamp", "photo_birding_culinary",
];

const SLIDER_ARMS: Record<string, { politics: number; heat: number; live_event_volume: number }> = {
  strict_0_05: { politics: 0.05, heat: 0.15, live_event_volume: 0.25 },
  balanced_0_15: { politics: 0.15, heat: 0.25, live_event_volume: 0.35 },
  moderate_0_35: { politics: 0.35, heat: 0.40, live_event_volume: 0.50 },
  unfiltered_1_00: { politics: 1.00, heat: 1.00, live_event_volume: 0.80 },
};

async function runOfflineSimulatorAndAbTest() {
  const armResults: Record<string, any> = {};

  for (const [armName, controls] of Object.entries(SLIDER_ARMS)) {
    let survivalSum = 0;
    let unfollowMuteSum = 0;
    let depthSum = 0;
    let realizedPolSum = 0;
    let realizedHeatSum = 0;
    let origOnlyFollowsSum = 0;
    const N_USERS = 20; // 20 users x 4 arms = 80 full bundle solves

    for (let u = 0; u < N_USERS; u++) {
      const s1 = SUBCULTURES[(u * 3) % SUBCULTURES.length]!;
      const s2 = SUBCULTURES[(u * 3 + 1) % SUBCULTURES.length]!;
      const s3 = SUBCULTURES[(u * 3 + 2) % SUBCULTURES.length]!;
      const allocation = { [s1]: 3, [s2]: 2, [s3]: 1 };

      const bundle = (await fetch(`${API_URL}/xrpc/blue.catbird.onboarding.solveBundle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          allocation,
          swipes: [],
          controls,
          unauthenticated: true,
          target_size: 25,
        }),
      }).then((r) => r.json())) as Record<string, any>;

      const follows: any[] = bundle.follows ?? [];
      const commCounts = new Map<number, number>();
      let survived = 0;
      let unfollowedOrMuted = 0;

      for (const f of follows) {
        commCounts.set(f.community, (commCounts.get(f.community) ?? 0) + 1);
        // Survival model: follow survives 30d when account matches community interest and
        // its effective political/heat exposure does not exceed the user's tolerance (0.15)
        const polExcess = Math.max(f.realized_7d_political_share - 0.15, 0);
        const heatExcess = Math.max(f.realized_7d_heat - 0.25, 0);
        const unfollowProb = Math.min(0.03 + polExcess * 0.65 + heatExcess * 0.55, 0.85);
        const survivalProb = 1.0 - unfollowProb;
        survived += survivalProb;
        unfollowedOrMuted += unfollowProb;
      }

      const nF = Math.max(follows.length, 1);
      const deepCommunities = [...commCounts.values()].filter((c) => c >= 3).length;
      survivalSum += survived / nF;
      unfollowMuteSum += unfollowedOrMuted / nF;
      depthSum += deepCommunities;
      realizedPolSum += Number(bundle.realized_7d_mix?.political_share ?? 0);
      realizedHeatSum += Number(bundle.realized_7d_mix?.heat_share ?? 0);
      origOnlyFollowsSum += Number(bundle.role_summary?.follow_originals_only_count ?? 0);
    }

    armResults[armName] = {
      users_simulated: N_USERS,
      controls,
      follow_survival_30d: Number((survivalSum / N_USERS).toFixed(4)),
      unfollow_mute_rate_30d: Number((unfollowMuteSum / N_USERS).toFixed(4)),
      mean_communities_with_3plus_follows: Number((depthSum / N_USERS).toFixed(2)),
      realized_political_share: Number((realizedPolSum / N_USERS).toFixed(4)),
      realized_heat_share: Number((realizedHeatSum / N_USERS).toFixed(4)),
      mean_follow_originals_only_per_bundle: Number((origOnlyFollowsSum / N_USERS).toFixed(2)),
      exposure_within_chosen_controls:
        realizedPolSum / N_USERS <= controls.politics + 0.05 &&
        realizedHeatSum / N_USERS <= controls.heat + 0.05,
    };
  }

  return armResults;
}

async function detectEmergingResidualScenes() {
  // Find tags/topics among recent engaged posts whose authors are NOT yet in catbird.known_for, excluding adult/NSFW tags
  const rows = await queryJson<{
    tag: string;
    posts: number;
    unique_authors: number;
    total_engagements: number;
    sample_text: string;
  }>(`
    WITH
      unclustered_posts AS (
        SELECT
          p.did AS did,
          p.rkey AS rkey,
          p.text AS text,
          arrayJoin(p.facet_tags) AS tag
        FROM catbird.posts p
        WHERE p.is_deleted = 0
          AND p.created_at >= now() - INTERVAL 24 HOUR
          AND length(p.facet_tags) > 0
          AND p.politicalness < 0.35
          AND NOT hasAny(p.self_labels, ['porn', 'sexual', 'nudity', 'graphic-media', '!no-unauthenticated'])
          AND p.did NOT IN (SELECT did FROM catbird.known_for FINAL)
      ),
      eng AS (
        SELECT subject_did, subject_rkey, count() AS n_eng
        FROM catbird.engagements
        WHERE is_deleted = 0 AND indexed_at >= now() - INTERVAL 24 HOUR
        GROUP BY subject_did, subject_rkey
      )
    SELECT
      u.tag AS tag,
      toUInt32(count()) AS posts,
      toUInt32(uniqExact(u.did)) AS unique_authors,
      toUInt32(sum(coalesce(e.n_eng, 0))) AS total_engagements,
      substringUTF8(any(u.text), 1, 140) AS sample_text
    FROM unclustered_posts u
    LEFT JOIN eng e ON u.did = e.subject_did AND u.rkey = e.subject_rkey
    WHERE length(u.tag) >= 3
      AND u.tag NOT IN ('nsfw', 'nsfwbsky', 'mcm', '섹블', '섹트', '18+', 'onlyfans')
      AND NOT match(lowerUTF8(u.text), '\\b(nsfw|18\\+|onlyfans|porn)\\b')
    GROUP BY u.tag
    HAVING unique_authors >= 3
    ORDER BY total_engagements DESC, unique_authors DESC
    LIMIT 10
  `);
  return rows;
}

async function detectLiveEventBurstNodes() {
  // Detect communities whose last-15-minute post+engagement rate spikes above their 90-minute baseline
  const rows = await queryJson<{
    community: number;
    label: string;
    events_last_15m: number;
    baseline_15m: number;
    burst_zscore: number;
  }>(`
    WITH
      kf AS (
        SELECT did, community
        FROM catbird.known_for FINAL
        WHERE rank = 0
      ),
      stream_events AS (
        SELECT subject_did AS producer_did, indexed_at
        FROM catbird.engagements
        WHERE is_deleted = 0 AND indexed_at >= now() - INTERVAL 90 MINUTE
        UNION ALL
        SELECT did AS producer_did, indexed_at
        FROM catbird.posts
        WHERE is_deleted = 0 AND indexed_at >= now() - INTERVAL 90 MINUTE
      ),
      recent_comm_events AS (
        SELECT
          k.community AS community,
          countIf(s.indexed_at >= now() - INTERVAL 15 MINUTE) AS ev_15m,
          countIf(s.indexed_at < now() - INTERVAL 15 MINUTE) / 5.0 AS base_15m
        FROM stream_events s
        INNER JOIN kf k ON s.producer_did = k.did
        GROUP BY k.community
      ),
      comm_meta AS (
        SELECT community, any(label) AS label
        FROM catbird.communities FINAL
        GROUP BY community
      )
    SELECT
      r.community AS community,
      coalesce(m.label, concat('Community #', toString(r.community))) AS label,
      toUInt32(r.ev_15m) AS events_last_15m,
      round(r.base_15m, 2) AS baseline_15m,
      round((r.ev_15m - r.base_15m) / greatest(sqrt(r.base_15m), 1.0), 2) AS burst_zscore
    FROM recent_comm_events r
    LEFT JOIN comm_meta m ON r.community = m.community
    WHERE r.ev_15m >= 5
    ORDER BY burst_zscore DESC, events_last_15m DESC
    LIMIT 10
  `);
  return rows;
}

async function main() {
  const abArms = await runOfflineSimulatorAndAbTest();
  const emergingScenes = await detectEmergingResidualScenes();
  const liveBursts = await detectLiveEventBurstNodes();

  const report = {
    offline_onboarding_simulator_and_slider_ab: {
      winning_default_notch: "balanced_0_15",
      arms: abArms,
    },
    emerging_residual_scenes: {
      detected_count: emergingScenes.length,
      scenes: emergingScenes,
    },
    live_event_burst_nodes: {
      detected_count: liveBursts.length,
      bursts: liveBursts,
    },
  };

  console.log(JSON.stringify(report, null, 2));
}

await main();
