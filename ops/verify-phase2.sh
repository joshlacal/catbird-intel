#!/usr/bin/env bash
# Phase 2 "Done when" verification on netcup:
#   1. Hand review of Josh's feed (`josh_catbird_builder`) + 3 test personas
#      (`systems_rust_apple_dev`, `sports_sabermetrics_f1`, `indie_creative_arts`)
#      + `/community/{id}` browse.
#   2. Synthetic load test at 20 QPS for 15 seconds (300 requests across personas)
#      verifying end-to-end p99 <= 50 ms and server compute <= 10 ms.
#   3. Political share & mean heat measured and logged in `catbird.feed_serve_log`.
set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:3210}"

echo "== 1. Health & Cache Status"
curl -sS "$API_URL/health"
echo ""

echo "== 2. Persona Feed Hand Review & Political Share"
for persona in josh_catbird_builder systems_rust_apple_dev sports_sabermetrics_f1 indie_creative_arts; do
  echo "--- Persona: $persona ---"
  curl -sS "$API_URL/xrpc/app.bsky.feed.getFeedSkeleton?persona=$persona&limit=10&debug=1" \
    | /home/josh/.bun/bin/bun -e '
      const d = JSON.parse(await new Response(Bun.stdin.stream()).text());
      console.log(JSON.stringify({
        persona: d.meta.actor,
        compute_ms: d.meta.compute_ms,
        items_served: d.meta.items_served,
        political_share: d.meta.political_share,
        mean_heat: d.meta.mean_heat,
        top_affinities: d.meta.affinities.slice(0, 4).map(a => `${a.label} (#${a.community})`),
        sample_posts: d.feed.slice(0, 4).map(p => ({
          uri: p.post,
          community: p.communities?.[0]?.label,
          residual: p.residual_score,
          engagers: p.engagers,
          expected: p.expected_engagements,
          text: (p.text_preview || "").slice(0, 100)
        }))
      }, null, 2));
    '
done

echo "== 3. 20 QPS Synthetic Load Test (300 requests over 15s)"
/home/josh/.bun/bin/bun -e '
  const api = process.env.API_URL || "http://127.0.0.1:3210";
  const personas = ["josh_catbird_builder", "systems_rust_apple_dev", "sports_sabermetrics_f1", "indie_creative_arts"];
  const total = 300;
  const intervalMs = 50; // 20 QPS
  const e2eMs = [];
  const compMs = [];
  const pending = [];
  for (let i = 0; i < total; i++) {
    const p = personas[i % personas.length];
    pending.push((async () => {
      const t0 = performance.now();
      const res = await fetch(`${api}/xrpc/app.bsky.feed.getFeedSkeleton?persona=${p}&limit=30`);
      const body = await res.json();
      e2eMs.push(performance.now() - t0);
      if (body.meta?.compute_ms !== undefined) compMs.push(body.meta.compute_ms);
    })());
    await Bun.sleep(intervalMs);
  }
  await Promise.all(pending);
  e2eMs.sort((a, b) => a - b);
  compMs.sort((a, b) => a - b);
  const q = (arr, p) => Number((arr[Math.min(Math.floor(arr.length * p), arr.length - 1)] || 0).toFixed(3));
  console.log(JSON.stringify({
    requests: e2eMs.length,
    qps: 20,
    e2e_ms: { p50: q(e2eMs, 0.50), p95: q(e2eMs, 0.95), p99: q(e2eMs, 0.99), max: q(e2eMs, 1.0) },
    compute_ms: { p50: q(compMs, 0.50), p95: q(compMs, 0.95), p99: q(compMs, 0.99), max: q(compMs, 1.0) },
    passes_p99_le_50ms: q(e2eMs, 0.99) <= 50.0,
    passes_compute_le_10ms: q(compMs, 0.99) <= 10.0
  }, null, 2));
'

sleep 6
echo "== 4. Logged Telemetry in catbird.feed_serve_log"
clickhouse-client -q "
  SELECT
    persona,
    count() AS requests,
    round(avg(items_served), 1) AS avg_items,
    round(avg(political_share), 4) AS avg_political_share,
    round(avg(mean_heat), 4) AS avg_heat,
    round(quantile(0.5)(compute_ms), 3) AS p50_compute_ms,
    round(quantile(0.99)(compute_ms), 3) AS p99_compute_ms
  FROM catbird.feed_serve_log
  GROUP BY persona
  ORDER BY requests DESC
  FORMAT PrettyCompactMonoBlock
"
