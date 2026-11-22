// Memory and time per onboarding session in the live engine, over the real bundle and pool.
// Each simulated viewer follows a few hundred pool accounts and runs the whole flow once.
// Usage: bun eval/onboarding/load_check.ts <bundle.json> <pool.json> [sessions=2000]
const SRC = new URL("../../src/", import.meta.url).pathname;
const { S17CatalogEngine } = await import(`${SRC}intel_catalog.ts`);
const { LiveOnboardingEngine } = await import(`${SRC}onboarding_live.ts`);

const catalog = S17CatalogEngine.fromFile(process.argv[2]!);
const pool = LiveOnboardingEngine.loadPoolFile(process.argv[3]!);
const sessions = Number(process.argv[4] ?? 2000);
const allDids: string[] = pool.communities.flatMap((c: { members: { did: string }[] }) => c.members.map((m) => m.did));
const communityIds: number[] = catalog.bundle.communities.map((c: { communityId: number }) => c.communityId);

let state = 2026;
const rng = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 0x1_0000_0000; };
const pickSome = <T>(items: T[], k: number): T[] => Array.from({ length: k }, () => items[Math.floor(rng() * items.length)]!);

// Stands in for the vector store: a fixed pseudo-random 256-byte vector per account, so the
// personal ordering costs what it costs in production without loading the real store.
const vectorCache = new Map<string, Uint8Array>();
const vectors = {
  rawVectorForDid(did: string) {
    let v = vectorCache.get(did);
    if (!v) {
      v = new Uint8Array(256);
      let h = 2166136261;
      for (let i = 0; i < did.length; i++) h = Math.imul(h ^ did.charCodeAt(i), 16777619) >>> 0;
      for (let i = 0; i < 256; i++) { h = (Math.imul(h, 1664525) + 1013904223) >>> 0; v[i] = h >>> 24; }
      vectorCache.set(did, v);
    }
    return v;
  },
};
for (const did of allDids) vectors.rawVectorForDid(did);

const followLists = new Map<string, string[]>();
const engine = new LiveOnboardingEngine({
  catalog,
  pool,
  vectors,
  followsFetcher: async (did: string, onPage: (dids: string[]) => void) => onPage(followLists.get(did) ?? []),
  config: { maxSessions: sessions + 10 },
});

Bun.gc(true);
const heapBefore = process.memoryUsage().heapUsed;
const rssBefore = process.memoryUsage().rss;
const times: number[] = [];
let follows = 0;
for (let s = 0; s < sessions; s++) {
  const viewer = { did: `did:plc:loadviewer${s}`, unauthenticated: false };
  followLists.set(viewer.did, pickSome(allDids, 400));
  const t0 = performance.now();
  await engine.getCatalog({ limit: 80 }, viewer);
  const picked = pickSome(communityIds, 6);
  for (const communityId of picked) await engine.getCommunity(communityId, viewer);
  const selections = picked.map((communityId, i) => ({ communityId, weight: i === 0 ? 2 : 1 }));
  const { swipes } = await engine.getCalibrationSwipes({ selections, maxSwipes: 4 }, viewer);
  const answers = swipes.map((d: any) => ({ swipeId: d.swipeId, communityA: d.optionA.communityId, communityB: d.optionB.communityId, choice: "a" as const }));
  const solved = await engine.solveBundle({ selections, swipes: answers }, viewer);
  await engine.previewTimeline({ selections, swipes: answers, limit: 16 }, viewer);
  follows += solved.follows.length;
  times.push(performance.now() - t0);
  followLists.delete(viewer.did);
}
Bun.gc(true);
const heapAfter = process.memoryUsage().heapUsed;
times.sort((a, b) => a - b);
const mb = (bytes: number) => (bytes / 1048576).toFixed(0);
console.log(`${sessions} sessions held at once: heap +${mb(heapAfter - heapBefore)} MB (${((heapAfter - heapBefore) / sessions / 1024).toFixed(0)} KB per session), rss ${mb(rssBefore)} -> ${mb(process.memoryUsage().rss)} MB`);
console.log(`whole flow per viewer (catalog of 80, 6 scene sheets, duels, lineup, preview): median ${times[times.length >> 1]!.toFixed(1)} ms, p95 ${times[Math.floor(times.length * 0.95)]!.toFixed(1)} ms, worst ${times[times.length - 1]!.toFixed(1)} ms; ${(follows / sessions).toFixed(1)} follows per lineup`);
console.log(`engine: ${JSON.stringify(engine.describe())}`);
