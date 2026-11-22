/**
 * Small pure helpers shared by the live onboarding modules: clamping, the seeded string hash that
 * drives every session order, int8 decoding, and the largest-remainder split of a lineup budget.
 */

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/** FNV-1a over the string, mixed with the seed and finished with a murmur3 avalanche. */
export function seededHash(text: string, seed: number): number {
  let h = (0x811c9dc5 ^ seed) >>> 0;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}

export function signedByte(value: number): number {
  return value > 127 ? value - 256 : value;
}

/** Splits `total` across `weights` by largest remainder; every entry gets at least one while the total allows. */
export function allocateByWeight(total: number, weights: readonly number[]): number[] {
  const n = weights.length;
  const out = new Array<number>(n).fill(0);
  if (n === 0 || total <= 0) return out;
  const order = weights.map((w, i) => ({ w, i })).sort((a, b) => b.w - a.w || a.i - b.i);
  if (total <= n) {
    for (const { i } of order.slice(0, total)) out[i] = 1;
    return out;
  }
  const spare = total - n;
  const sum = weights.reduce((acc, w) => acc + w, 0);
  const shares = weights.map((w) => (sum > 0 ? (spare * w) / sum : spare / n));
  let given = 0;
  shares.forEach((share, i) => {
    out[i] = 1 + Math.floor(share);
    given += out[i]!;
  });
  const byRemainder = shares
    .map((share, i) => ({ r: share - Math.floor(share), i }))
    .sort((a, b) => b.r - a.r || a.i - b.i);
  for (let k = 0; given < total; k++, given++) out[byRemainder[k % n]!.i]! += 1;
  return out;
}
