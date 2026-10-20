#!/usr/bin/env bun
/** Local fixture-only HTTP acceptance run. Never connects to ClickHouse or production. */
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { P256Keypair } from "@atproto/crypto";
import { createApiHandler, flushTelemetry, pendingTelemetryForTests } from "../src/api.ts";
import { createServiceAuthenticator } from "../src/service_auth.ts";
import { FIXTURE_DIDS, installLocalServingFixture } from "../test/serving_fixtures.ts";

installLocalServingFixture();
const started = new Date().toISOString();
const key = await P256Keypair.create();
const audience = "did:web:fixture.invalid#catbird_intel";
const authenticate = createServiceAuthenticator({ audience, resolveAtprotoKey: async (issuer) => {
  if (issuer !== FIXTURE_DIDS.ordinary) throw new Error("Fixture issuer not recognized");
  return key.did();
} });
const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: createApiHandler({ authenticate }) });
const observations: Record<string, unknown>[] = [];
let checks = 0;
function check(condition: boolean, message: string) {
  checks++;
  if (!condition) throw new Error(message);
}
async function call(name: string, path: string, body?: unknown, authorization?: string) {
  const response = await fetch(new URL(path, server.url), {
    method: body === undefined ? "GET" : "POST",
    headers: { ...(body === undefined ? {} : { "content-type": "application/json" }),
      ...(authorization ? { authorization } : {}) },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await response.json() as any;
  observations.push({ name, status: response.status, response: value });
  check(!/realized|predicted_mix|live_event_volume|text_preview|__FORBIDDEN_/.test(JSON.stringify(value)),
    `${name} leaked an obsolete field or raw content`);
  return { status: response.status, value };
}

let failure: string | null = null;
try {
  const base = "/xrpc/blue.catbird.onboarding.";
  const feed = await call("anonymous_feed_forged_flag", "/xrpc/app.bsky.feed.getFeedSkeleton?unauthenticated=false&debug=1");
  check(feed.status === 200 && feed.value.unauthenticated === true && feed.value.feed.length === 2, "Anonymous visibility failed");
  const preview = await call("default_preview", base + "previewTimeline", { allocation: { systems_rust: 6 }, unauthenticated: false });
  check(preview.value.control_policy.health_filtering === "off" && preview.value.control_policy.unscored_share === 0.5, "Default controls or unknown accounting failed");
  check(JSON.stringify(preview.value.control_policy) === JSON.stringify(feed.value.meta.control_policy), "Feed/preview default policy diverged");
  const controls = { baseline_score_cutoff: { politicalness: 0.2, heat: 0.2 } };
  const filteredFeed = await call("optional_baseline_feed", "/xrpc/app.bsky.feed.getFeedSkeleton?baseline_score_cutoff=" +
    encodeURIComponent(JSON.stringify(controls.baseline_score_cutoff)));
  const filteredPreview = await call("optional_baseline_preview", base + "previewTimeline", { allocation: { systems_rust: 6 }, controls });
  check(filteredFeed.value.feed.length === 1 && JSON.stringify(filteredFeed.value.feed) === JSON.stringify(filteredPreview.value.preview_posts),
    "Feed/preview cutoff policy diverged");
  check(filteredPreview.value.control_policy.unscored_share === 1, "Unknown item was treated as scored or excluded");
  const bundle = await call("explicit_delivery_actions", base + "solveBundle",
    { allocation: { systems_rust: 6 }, preferences: { hideReposts: true }, delivery_modes: { [FIXTURE_DIDS.ordinary]: "topic_feed" } });
  check(bundle.value.follows.length === 1 && bundle.value.follows[0].client_preferences.hideReposts === true &&
    bundle.value.topic_feeds[0].follow === false && bundle.value.role_summary.total_follows === 1, "Delivery actions are conflated");
  check(bundle.value.actions.every((action: any) => action.applied === false && action.executable === false), "Review-only actions misrepresented");
  const community = await call("creator_visibility", "/community/1?unauthenticated=false");
  check(!JSON.stringify(community.value).includes(FIXTURE_DIDS.hidden) && !JSON.stringify(community.value).includes(FIXTURE_DIDS.restricted),
    "Feed/pack creator visibility failed");
  await call("community_metadata", "/xrpc/blue.catbird.intel.listCommunities");
  const modes = await call("aggregate_author_diagnostics", "/xrpc/blue.catbird.intel.getAuthorModes");
  check(modes.value.individual_profiles_available === false && !modes.value.producers, "Individual inference leaked");
  const anonymousCards = await call("anonymous_cards", base + "getSceneCards?unauthenticated=false");
  check(anonymousCards.value.unauthenticated === true, "Caller flag authenticated");
  await call("calibration", base + "getCalibrationSwipes", { allocation: { systems_rust: 6 } });
  const invalid = await call("invalid_token", base + "getSceneCards", undefined, "Bearer invalid");
  check(invalid.status === 401, "Invalid token was silently downgraded");
  const now = Math.floor(Date.now() / 1000);
  const encoded = [{ typ: "JWT", alg: "ES256" }, { iss: FIXTURE_DIDS.ordinary, aud: audience, lxm: "blue.catbird.onboarding.getSceneCards",
    iat: now, exp: now + 60, jti: crypto.randomUUID() }].map((part) => Buffer.from(JSON.stringify(part)).toString("base64url")).join(".");
  const bearer = `Bearer ${encoded}.${Buffer.from(await key.sign(Buffer.from(encoded))).toString("base64url")}`;
  const signed = await call("verified_fixture_service_jwt", base + "getSceneCards", undefined, bearer);
  check(signed.status === 200 && signed.value.unauthenticated === false &&
    JSON.stringify(signed.value).includes(FIXTURE_DIDS.restricted), "Verified token did not reach authenticated policy");
  check((await call("replayed_fixture_service_jwt", base + "getSceneCards", undefined, bearer)).status === 401, "Replay guard failed");
  check((await call("reject_old_controls", base + "previewTimeline", { controls: { politics: 0.2 } })).status === 400,
    "Old controls were silently reinterpreted");
  check(!/realized|predicted|political_share|mean_heat/.test(JSON.stringify(pendingTelemetryForTests())), "Unqualified outcome in pending telemetry");
  check((await flushTelemetry()).persisted === false, "Legacy telemetry persistence must remain disabled");
} catch (error) {
  failure = error instanceof Error ? error.message : String(error);
} finally {
  server.stop(true);
}

const root = resolve(import.meta.dir, "..");
const validation = [];
for (const command of [
  ["bun", "test", "test/serving.test.ts", "test/service_auth.test.ts"],
  ["bun", "run", "typecheck"],
]) {
  const process = Bun.spawn(command, { cwd: root, stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(process.stdout).text(), new Response(process.stderr).text(), process.exited,
  ]);
  validation.push({ command, exit_code: exitCode, status: exitCode === 0 ? "passed" : "failed", stdout, stderr });
  if (exitCode !== 0 && failure === null) failure = `${command.join(" ")} exited ${exitCode}`;
}
const files = ["src/api.ts", "src/serving_contract.ts", "src/service_auth.ts", "src/clickhouse.ts",
  "test/serving_fixtures.ts", "test/serving.test.ts", "test/service_auth.test.ts", "ops/verify-serving-local.ts",
  "package.json", "bun.lock", "tsconfig.json"];
const hashes = Object.fromEntries(await Promise.all(files.map(async (file) =>
  [file, createHash("sha256").update(await readFile(resolve(root, file))).digest("hex")])));
const output = resolve(root, "eval/review_serving_20261020.json");
await mkdir(resolve(output, ".."), { recursive: true });
await writeFile(output, JSON.stringify({
  schema: "catbird.intel.local-serving-acceptance.v1", started_at: started, finished_at: new Date().toISOString(),
  evidence: "local_http_fixture_only", bun: Bun.version, source_hashes: hashes, production_connected: false,
  database_connected: false, client_actions_executed: false, service_stopped: true,
  checks, status: failure ? "failed" : "passed", failure, observations, validation,
  fixture_scope: "structural_identifiers_numeric_scores_and_opaque_field_sentinels_only",
  semantic_post_quality_evaluated: false,
  qualification_gates: {
    local_http_contract: failure ? "failed" : "passed",
    production_service_jwt_resolution: "unconfigured",
    distributed_nonce_protection: "unimplemented",
    consistent_derived_generation: "unimplemented",
    live_deletion_and_visibility_overlay: "unqualified",
    viewer_aware_hydration_and_moderation: "unimplemented",
    native_client_actions_and_visual_ui: "unperformed",
    recommendation_quality_and_health_calibration: "unperformed",
    versioned_telemetry_persistence: "disabled_pending_schema",
    production_promotion: "blocked",
  },
}, null, 2) + "\n");
console.log(JSON.stringify({ output, checks, status: failure ? "failed" : "passed", failure }));
if (failure) process.exitCode = 1;
