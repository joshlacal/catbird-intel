// Print the duels the live engine generates for a few sets of picks, over the real catalog.
// Usage: bun eval/onboarding/duel_probe.ts <bundle.json> <pool.json>
const SRC = new URL("../../src/", import.meta.url).pathname;
const { S17CatalogEngine } = await import(`${SRC}intel_catalog.ts`);
const { LiveOnboardingEngine } = await import(`${SRC}onboarding_live.ts`);
const catalog = S17CatalogEngine.fromFile(process.argv[2]!);
const pool = LiveOnboardingEngine.loadPoolFile(process.argv[3]!);
const byTitle = (needle: string) =>
  catalog.bundle.communities.find((c: { title: string }) => c.title.toLowerCase().includes(needle.toLowerCase()))!;

const sets: string[][] = [
  ["iOS and Swift developers"],
  ["Fantasy illustrators", "Webcomic", "Bluesky joke posters"],
  ["NBA basketball", "College football", "Urbanism", "Academic philosophy"],
  ["Japanese cat owners", "Haiku"],
];
let state = 7;
const rng = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 0x1_0000_0000; };
for (const names of sets) {
  const picks = names.map(byTitle);
  console.log(`\nPICKED: ${picks.map((c: { title: string }) => c.title).join(" + ")}`);
  for (let session = 0; session < 2; session++) {
    const engine = new LiveOnboardingEngine({ catalog, pool, random: rng });
    const { swipes } = await engine.getCalibrationSwipes({
      selections: picks.map((c: { communityId: number }) => ({ communityId: c.communityId, weight: 1 })),
      maxSwipes: 4,
    }, { did: null, unauthenticated: false });
    console.log(`  session ${session + 1}:`);
    for (const s of swipes) {
      console.log(`    [${s.kind}] ${s.prompt}\n        A: ${s.optionA.title} (${s.optionA.samplePostUris.length} posts)   B: ${s.optionB.title} (${s.optionB.samplePostUris.length} posts)`);
    }
  }
}

// Coverage: how many single picks get at least one duel, and how many get four.
let none = 0, four = 0, total = 0, splits = 0, bridges = 0;
const engine = new LiveOnboardingEngine({ catalog, pool, random: rng });
for (const c of catalog.bundle.communities) {
  const { swipes } = await engine.getCalibrationSwipes({ selections: [{ communityId: c.communityId, weight: 1 }], maxSwipes: 4 }, { did: null, unauthenticated: false });
  total += 1;
  if (swipes.length === 0) none += 1;
  if (swipes.length === 4) four += 1;
  for (const s of swipes) s.kind === "subscene_split" ? splits++ : bridges++;
}
console.log(`\nsingle-pick coverage over ${total} scenes: no duel ${none}, four duels ${four}; kinds: ${splits} splits, ${bridges} bridges`);
