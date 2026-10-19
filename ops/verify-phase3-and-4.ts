#!/usr/bin/env bun
/**
 * End-to-end verification harness for Phase 3 (Timeline Health & Dual-Mode Decomposition)
 * and Phase 4 (Active Onboarding API & Portfolio Solver).
 */
import { scorePostHealth } from "../src/health_scorer.ts";
import { VIEWPOINT_BALANCED_EVAL_SET } from "../src/health_eval_dataset.ts";

const API_URL = process.env.API_URL ?? "http://127.0.0.1:3210";

function computeF1(preds: number[], labels: number[]) {
  let tp = 0, fp = 0, fn = 0;
  for (let i = 0; i < preds.length; i++) {
    if (preds[i] === 1 && labels[i] === 1) tp++;
    else if (preds[i] === 1 && labels[i] === 0) fp++;
    else if (preds[i] === 0 && labels[i] === 1) fn++;
  }
  const precision = tp / Math.max(tp + fp, 1);
  const recall = tp / Math.max(tp + fn, 1);
  const f1 = (2 * precision * recall) / Math.max(precision + recall, 1e-9);
  return {
    precision: Number(precision.toFixed(4)),
    recall: Number(recall.toFixed(4)),
    f1: Number(f1.toFixed(4)),
  };
}

async function main() {
  // 1. Phase 3: Held-out viewpoint-balanced evaluation & CPU core budget
  const polPreds: number[] = [];
  const polLabels: number[] = [];
  const heatPreds: number[] = [];
  const heatLabels: number[] = [];
  let leftCorrect = 0, leftTotal = 0, rightCorrect = 0, rightTotal = 0;

  for (const item of VIEWPOINT_BALANCED_EVAL_SET) {
    const s = scorePostHealth(item.text, { isQuote: item.isQuote });
    const pPol = s.politicalness >= 0.45 ? 1 : 0;
    const pHeat = s.heat >= 0.45 ? 1 : 0;
    polPreds.push(pPol);
    polLabels.push(item.is_political);
    heatPreds.push(pHeat);
    heatLabels.push(item.is_heated);
    if (item.viewpoint === "left") {
      leftTotal++;
      if (pPol === item.is_political) leftCorrect++;
    } else if (item.viewpoint === "right") {
      rightTotal++;
      if (pPol === item.is_political) rightCorrect++;
    }
  }

  const polEval = computeF1(polPreds, polLabels);
  const heatEval = computeF1(heatPreds, heatLabels);

  const tBench = performance.now();
  const N_BENCH = 50_000;
  for (let i = 0; i < N_BENCH; i++) {
    const item = VIEWPOINT_BALANCED_EVAL_SET[i % VIEWPOINT_BALANCED_EVAL_SET.length]!;
    scorePostHealth(item.text, { isQuote: item.isQuote });
  }
  const benchMs = performance.now() - tBench;
  const usPerPost = (benchMs * 1000) / N_BENCH;
  const coresAt38PostsPerSec = ((benchMs / N_BENCH) * 38) / 1000;

  // 2. Phase 3: Top 10k producer mode mix & "Follow originals only" distribution
  const modesRes = await fetch(`${API_URL}/xrpc/blue.catbird.intel.getAuthorModes?limit=15`).then((r) => r.json()) as Record<string, any>;

  // 3. Phase 4: Full 4-step Onboarding flow across test personas
  const sceneCardsRes = await fetch(`${API_URL}/xrpc/blue.catbird.onboarding.getSceneCards?unauthenticated=true`).then((r) => r.json()) as Record<string, any>;

  const personas: Record<string, { allocation: Record<string, number>; controls: { politics: number; heat: number; live_event_volume: number } }> = {
    josh_catbird_builder: {
      allocation: { systems_rust: 2, apple_swift_indie: 2, local_ai_ml: 1, hardware_homelab: 1 },
      controls: { politics: 0.10, heat: 0.20, live_event_volume: 0.30 },
    },
    systems_rust_apple_dev: {
      allocation: { systems_rust: 3, apple_swift_indie: 2, infosec_networking: 1 },
      controls: { politics: 0.05, heat: 0.15, live_event_volume: 0.25 },
    },
    sports_sabermetrics_f1: {
      allocation: { baseball_sabermetrics: 2, nba_f1_sickos: 2, soccer_fpl_tactics: 2 },
      controls: { politics: 0.10, heat: 0.35, live_event_volume: 0.70 },
    },
    indie_creative_arts: {
      allocation: { indie_gamedev: 2, music_vinyl_bandcamp: 2, film_anime_art: 1, photo_birding_culinary: 1 },
      controls: { politics: 0.15, heat: 0.20, live_event_volume: 0.25 },
    },
  };

  const personaResults: Record<string, unknown> = {};
  const solveTimesMs: number[] = [];
  let maxDeviationPoints = 0;
  let unauthViolations = 0;

  for (const [name, cfg] of Object.entries(personas)) {
    const swipesRes = await fetch(`${API_URL}/xrpc/blue.catbird.onboarding.getCalibrationSwipes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ allocation: cfg.allocation, unauthenticated: true }),
    }).then((r) => r.json()) as Record<string, any>;

    // Simulate user swipes including "a", "b", "both", and "neither"
    const swipes = (swipesRes.swipes ?? []).map((sw: any, idx: number) => ({
      swipe_id: sw.swipe_id,
      choice: idx % 4 === 3 ? "neither" : idx % 3 === 2 ? "both" : idx % 2 === 0 ? "a" : "b",
      community_a: sw.option_a.community,
      community_b: sw.option_b.community,
    }));

    const previewRes = await fetch(`${API_URL}/xrpc/blue.catbird.onboarding.previewTimeline`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        allocation: cfg.allocation,
        swipes,
        controls: cfg.controls,
        unauthenticated: true,
      }),
    }).then((r) => r.json()) as Record<string, any>;

    const t0 = performance.now();
    const bundleRes = await fetch(`${API_URL}/xrpc/blue.catbird.onboarding.solveBundle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        allocation: cfg.allocation,
        swipes,
        controls: cfg.controls,
        unauthenticated: true,
        target_size: 25,
      }),
    }).then((r) => r.json()) as Record<string, any>;
    const httpMs = performance.now() - t0;
    solveTimesMs.push(httpMs);

    const devPts = Number(bundleRes.realized_7d_mix?.max_abs_deviation_points ?? 0);
    if (devPts > maxDeviationPoints) maxDeviationPoints = devPts;

    for (const p of previewRes.preview_posts ?? []) {
      if (p.has_no_unauth_label === 1) unauthViolations++;
    }

    personaResults[name] = {
      swipes_count: swipes.length,
      solve_compute_ms: bundleRes.solve_ms,
      solve_http_ms: Number(httpMs.toFixed(2)),
      role_summary: bundleRes.role_summary,
      predicted_mix: bundleRes.predicted_mix,
      realized_7d_mix: bundleRes.realized_7d_mix,
      preview_posts_count: (previewRes.preview_posts ?? []).length,
      sample_follows: (bundleRes.follows ?? []).slice(0, 5).map((f: any) => ({
        handle: f.handle,
        community_label: f.community_label,
        role: f.role,
        mode_label: f.mode_label,
      })),
      feeds_count: (bundleRes.feeds ?? []).length,
      starter_packs_count: (bundleRes.starter_packs ?? []).length,
    };
  }

  // Run 40 more bundle solves to measure p50/p95/p99 latency
  for (let i = 0; i < 40; i++) {
    const t0 = performance.now();
    await fetch(`${API_URL}/xrpc/blue.catbird.onboarding.solveBundle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        allocation: personas.josh_catbird_builder!.allocation,
        swipes: [],
        controls: personas.josh_catbird_builder!.controls,
        unauthenticated: true,
        target_size: 25,
      }),
    }).then((r) => r.json());
    solveTimesMs.push(performance.now() - t0);
  }
  solveTimesMs.sort((a, b) => a - b);
  const p50Solve = solveTimesMs[Math.floor(solveTimesMs.length * 0.5)]!;
  const p99Solve = solveTimesMs[Math.floor(solveTimesMs.length * 0.99)]!;

  const report = {
    phase3_timeline_health: {
      eval_set_size: VIEWPOINT_BALANCED_EVAL_SET.length,
      politicalness: polEval,
      heat: heatEval,
      left_political_accuracy: Number((leftCorrect / leftTotal).toFixed(4)),
      right_political_accuracy: Number((rightCorrect / rightTotal).toFixed(4)),
      us_per_post: Number(usPerPost.toFixed(2)),
      cores_at_38_posts_per_sec: Number(coresAt38PostsPerSec.toFixed(4)),
      passes_f1_and_core_budget:
        polEval.f1 >= 0.90 && heatEval.f1 >= 0.75 && coresAt38PostsPerSec <= 1.5,
      top_10k_producer_modes: {
        total_producers_scored: modesRes.total_producers_scored,
        top_10k_evaluated: modesRes.top_10k_evaluated,
        follow_originals_only_eligible_in_top_10k: modesRes.follow_originals_only_eligible_in_top_10k,
        mode_distribution_top_10k: modesRes.mode_distribution_top_10k,
        sample_producers: (modesRes.producers ?? []).slice(0, 8),
      },
    },
    phase4_onboarding: {
      scene_cards_count: sceneCardsRes.total_cards,
      solve_latency_ms: {
        p50: Number(p50Solve.toFixed(2)),
        p99: Number(p99Solve.toFixed(2)),
        passes_150ms_slo: p99Solve <= 150.0,
      },
      max_predicted_vs_realized_deviation_points: maxDeviationPoints,
      passes_10pt_mix_accuracy: maxDeviationPoints <= 10.0,
      unauthenticated_visibility_violations: unauthViolations,
      passes_visibility_invariant: unauthViolations === 0,
      personas: personaResults,
    },
  };

  console.log(JSON.stringify(report, null, 2));
}

await main();
