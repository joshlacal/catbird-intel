// End-to-end walk of the onboarding flow over HTTP against the local live engine.
// Usage: bun eval/onboarding/http_walk.ts <viewer did | anon> [base url]
const viewer = process.argv[2] && process.argv[2] !== "anon" ? process.argv[2] : null;
const base = process.argv[3] ?? "http://127.0.0.1:3299";
const headers: Record<string, string> = { "content-type": "application/json" };
if (viewer) headers["x-debug-viewer-did"] = viewer;

async function call(path: string, body?: unknown) {
  const t0 = performance.now();
  const response = await fetch(`${base}/xrpc/${path}`, body === undefined ? { headers } :
    { method: "POST", headers, body: JSON.stringify(body) });
  const json = await response.json() as any;
  if (!response.ok) throw new Error(`${path} -> ${response.status} ${JSON.stringify(json)}`);
  return { json, ms: Math.round(performance.now() - t0) };
}

// Independent copy of the viewer's follow list, to check the server's exclusions against.
const follows = new Set<string>();
if (viewer) {
  let cursor: string | undefined;
  for (let page = 0; page < 100; page++) {
    const url = new URL("https://public.api.bsky.app/xrpc/app.bsky.graph.getFollows");
    url.searchParams.set("actor", viewer);
    url.searchParams.set("limit", "100");
    if (cursor) url.searchParams.set("cursor", cursor);
    const body = await (await fetch(url)).json() as any;
    for (const f of body.follows ?? []) follows.add(f.did);
    cursor = body.cursor;
    if (!cursor) break;
  }
}
console.log(`viewer ${viewer ?? "anonymous"}: ${follows.size} follows (independent fetch)`);

const first = await call("blue.catbird.onboarding.getCatalog?limit=80");
const second = await call("blue.catbird.onboarding.getCatalog?limit=80");
console.log(`getCatalog: first ${first.ms} ms, repeat ${second.ms} ms; featured ${first.json.featuredCommunities.length}, warm-start ${first.json.warmStartCommunities.length}`);
for (const c of first.json.warmStartCommunities.slice(0, 5)) {
  console.log(`  warm: [${c.communityId}] ${c.title} | already following ${c.alreadyFollowingCount} | faces ${c.sampleAccountDids.length}`);
}

const cards: any[] = [...second.json.warmStartCommunities.slice(0, 3), ...second.json.featuredCommunities.slice(0, 4)]
  .filter((c, i, all) => all.findIndex((o) => o.communityId === c.communityId) === i).slice(0, 6);
const selections = cards.map((c, i) => ({ communityId: c.communityId, weight: i === 0 ? 2 : 1 }));
let shownTotal = 0;
const shown = new Map<number, string[]>();
for (const c of cards) {
  const detail = await call(`blue.catbird.intel.getCommunity?communityId=${c.communityId}`);
  const sheet: string[] = detail.json.members.map((m: any) => m.did);
  const prefixOk = JSON.stringify(sheet.slice(0, c.sampleAccountDids.length)) === JSON.stringify(c.sampleAccountDids);
  const quota = selections.find((s) => s.communityId === c.communityId)!.weight === 2
    ? detail.json.boostedLineupCount : detail.json.defaultLineupCount;
  shown.set(c.communityId, sheet.slice(0, quota));
  shownTotal += Math.min(quota, sheet.length);
  const tiers = detail.json.members.slice(0, 6).map((m: any) => `${m.role.split("_")[0]}/${m.tier}`).join(" ");
  console.log(`  sheet [${c.communityId}] ${c.title}: ${sheet.length} listed, card is prefix: ${prefixOk}, ${detail.ms} ms | first six: ${tiers}`);
}

const solved = await call("blue.catbird.onboarding.solveBundle", { selections });
const followDids: string[] = solved.json.follows.map((f: any) => f.did);
const alreadyFollowed = followDids.filter((d) => follows.has(d));
let promised = 0, kept = 0;
for (const [, dids] of shown) for (const did of dids) { promised += 1; if (followDids.includes(did)) kept += 1; }
console.log(`solveBundle: ${solved.json.follows.length} follows (${shownTotal} promised by the sheets), ${solved.ms} ms, bench ${solved.json.bench.length}, alreadyFollowingInScenes ${solved.json.alreadyFollowingInScenes}`);
console.log(`  shown in a lineup section and followed: ${kept}/${promised}; follows the viewer already had: ${alreadyFollowed.length}; duplicates: ${followDids.length - new Set(followDids).size}`);
const perScene = new Map<string, number>();
for (const f of solved.json.follows) perScene.set(f.communityTitle, (perScene.get(f.communityTitle) ?? 0) + 1);
console.log(`  per scene: ${[...perScene].map(([t, n]) => `${t}=${n}`).join("; ")}`);
console.log(`  summary: ${JSON.stringify(solved.json.summary)}`);

const preview = await call("blue.catbird.onboarding.previewTimeline", { selections, limit: 16 });
const previewOwners = preview.json.items.map((i: any) => /^at:\/\/(did:[^/]+)\//.exec(i.post)![1]);
console.log(`previewTimeline: ${preview.json.items.length} posts, ${previewOwners.filter((d: string) => followDids.includes(d)).length} by lineup accounts, ${preview.ms} ms`);

const legacy = await call("blue.catbird.onboarding.solveBundle", { selections, targetSize: 25 });
console.log(`older client (targetSize 25): ${legacy.json.follows.length} follows, already followed ${legacy.json.follows.filter((f: any) => follows.has(f.did)).length}`);
const swipes = await call("blue.catbird.onboarding.getCalibrationSwipes", { selections, maxSwipes: 4 });
console.log(`getCalibrationSwipes: ${swipes.json.swipes.length} duels: ${swipes.json.swipes.map((s: any) => s.kind).join(", ")}`);
