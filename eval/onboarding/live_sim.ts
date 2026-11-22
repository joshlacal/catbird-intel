// Before/after simulation: static S17 solver vs the live engine, over the real serving bundle.
// Usage: bun eval/onboarding/live_sim.ts <bundle.json> [pool.json]
import { existsSync } from "node:fs";
const SRC = new URL("../../src/", import.meta.url).pathname;
const { S17CatalogEngine } = await import(`${SRC}intel_catalog.ts`);
const { LiveOnboardingEngine } = await import(`${SRC}onboarding_live.ts`);

const bundlePath = process.argv[2]!;
const poolPath = process.argv[3];
const catalog = S17CatalogEngine.fromFile(bundlePath);
const pool = poolPath && existsSync(poolPath) ? LiveOnboardingEngine.loadPoolFile(poolPath) : null;
const communities = catalog.bundle.communities;
console.log(`bundle ${catalog.bundle.catalogVersion}: ${communities.length} communities; pool: ${pool ? pool.poolVersion : "none (bundle members only)"}`);

let rngState = 12345;
const rng = () => {
  rngState = (Math.imul(rngState, 1664525) + 1013904223) >>> 0;
  return rngState / 0x1_0000_0000;
};
const sample = <T>(items: T[], k: number): T[] => {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j]!, copy[i]!];
  }
  return copy.slice(0, k);
};
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const pct = (x: number) => `${(100 * x).toFixed(0)}%`;

const TRIALS = 200;
console.log("\nviewer | scenes | engine | follows | new follows | card faces followed | lineup section followed | scenes with 0");
for (const viewerKind of ["new account", "follows top half of each scene"] as const) {
  for (const k of [3, 5, 8, 12, 20]) {
    const rows: Record<string, { follows: number[]; fresh: number[]; card: number[]; section: number[]; empty: number[] }> = {
      static: { follows: [], fresh: [], card: [], section: [], empty: [] },
      live: { follows: [], fresh: [], card: [], section: [], empty: [] },
    };
    for (let trial = 0; trial < TRIALS; trial++) {
      const picked = sample(communities, k);
      const selections = picked.map((c: { communityId: number }) => ({ communityId: c.communityId, weight: 1 }));
      const followed = new Set<string>();
      if (viewerKind !== "new account") {
        for (const c of picked) for (const m of c.members.slice(0, Math.ceil(c.members.length / 2))) followed.add(m.did);
      }
      const followList = [...followed];

      // Static engine as the shipped app drives it: 50 most recent follows as seeds, 25 total.
      const staticSeeds = followList.slice(0, 50);
      const st = catalog.solveBundle({ selections, targetSize: 25, seedFollowDids: staticSeeds, unauthenticated: false });
      const stDids = new Set<string>(st.follows.map((f: { did: string }) => f.did));
      rows.static!.follows.push(st.follows.length);
      rows.static!.fresh.push(st.follows.filter((f: { did: string }) => !followed.has(f.did)).length);
      let stEmpty = 0;
      for (const c of picked) {
        const card = catalog.toCommunitySummary(c, false).sampleAccountDids as string[];
        rows.static!.card.push(mean(card.map((d) => (stDids.has(d) && !followed.has(d) ? 1 : 0))));
        const detail = catalog.getCommunity(c.communityId, false)!.members.slice(0, 12).map((m: { did: string }) => m.did);
        rows.static!.section.push(mean(detail.map((d: string) => (stDids.has(d) && !followed.has(d) ? 1 : 0))));
        if (!st.follows.some((f: { communityId: number }) => f.communityId === c.communityId)) stEmpty += 1;
      }
      rows.static!.empty.push(stEmpty / k);

      // Live engine: the server knows the whole follow list; sizing is per scene.
      const live = new LiveOnboardingEngine({
        catalog,
        pool,
        followsFetcher: async (_did: string, onPage: (dids: string[]) => boolean | void) => { onPage(followList); },
        random: rng,
      });
      // A signed-in viewer's order comes from their DID, so each trial is a different viewer.
      const viewer = { did: `did:plc:simviewer${Math.floor(rng() * 1e9)}`, unauthenticated: false };
      const cards = new Map<number, string[]>();
      const sections = new Map<number, string[]>();
      for (const c of picked) {
        const detail = (await live.getCommunity(c.communityId, viewer))!;
        cards.set(c.communityId, detail.community.sampleAccountDids);
        sections.set(c.communityId, detail.members.slice(0, detail.defaultLineupCount).map((m: { did: string }) => m.did));
      }
      const lv = await live.solveBundle({ selections }, viewer);
      const lvDids = new Set<string>(lv.follows.map((f: { did: string }) => f.did));
      rows.live!.follows.push(lv.follows.length);
      rows.live!.fresh.push(lv.follows.filter((f: { did: string }) => !followed.has(f.did)).length);
      let lvEmpty = 0;
      for (const c of picked) {
        const card = cards.get(c.communityId)!;
        if (card.length) rows.live!.card.push(mean(card.map((d) => (lvDids.has(d) ? 1 : 0))));
        const section = sections.get(c.communityId)!;
        if (section.length) rows.live!.section.push(mean(section.map((d) => (lvDids.has(d) ? 1 : 0))));
        if (!lv.follows.some((f: { communityId: number }) => f.communityId === c.communityId)) lvEmpty += 1;
      }
      rows.live!.empty.push(lvEmpty / k);
    }
    for (const engine of ["static", "live"]) {
      const r = rows[engine]!;
      console.log(`${viewerKind} | ${k} | ${engine} | ${mean(r.follows).toFixed(1)} | ${mean(r.fresh).toFixed(1)} | ${pct(mean(r.card))} | ${pct(mean(r.section))} | ${pct(mean(r.empty))}`);
    }
  }
}

// Variety: how many distinct accounts lead a scene across sessions, and how often the strongest does.
const probe = communities.slice(0, 40);
const distinct: number[] = [];
const topShare: number[] = [];
for (const c of probe) {
  const seen = new Map<string, number>();
  for (let s = 0; s < 100; s++) {
    const live = new LiveOnboardingEngine({ catalog, pool, random: rng });
    // One hundred different viewers: the order is derived from who is asking.
    const detail = (await live.getCommunity(c.communityId, { did: `did:plc:variety${s}`, unauthenticated: false }))!;
    for (const m of detail.members.slice(0, 6)) seen.set(m.did, (seen.get(m.did) ?? 0) + 1);
  }
  distinct.push(seen.size);
  topShare.push((seen.get(c.members[0]!.did) ?? 0) / 100);
}
console.log(`\nvariety over 100 sessions (40 scenes): distinct accounts ever in a scene's first six = ${mean(distinct).toFixed(1)} (static: 6); top-ranked account is in the first six in ${pct(mean(topShare))} of sessions`);
