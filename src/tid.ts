// TID (timestamp identifier) helpers. A TID is 13 base32-sortable characters
// encoding a 64-bit integer: 53 bits of microseconds since the Unix epoch and a
// 10-bit clock id. https://atproto.com/specs/tid

const ALPHABET = "234567abcdefghijklmnopqrstuvwxyz";
const LOOKUP = new Map<string, bigint>([...ALPHABET].map((c, i) => [c, BigInt(i)]));
const TID_RE = /^[234567abcdefghij][234567abcdefghijklmnopqrstuvwxyz]{12}$/;

/** Decode a TID to its 64-bit integer value, or null when it is not a TID. */
export function tidToBigInt(tid: string): bigint | null {
  if (!TID_RE.test(tid)) return null;
  let value = 0n;
  for (const ch of tid) value = (value << 5n) | (LOOKUP.get(ch) as bigint);
  return value;
}

/** Revision version for ReplacingMergeTree. Non-TID revisions sort lowest. */
export function revVersion(rev: string | undefined): string {
  if (!rev) return "0";
  const v = tidToBigInt(rev);
  return v === null ? "0" : v.toString();
}

const MIN_MS = Date.UTC(2023, 0, 1);

/**
 * Partition date for a record key. Uses the TID timestamp when the key is a
 * plausible TID (2023-01-01 .. arrival + 1 day), otherwise the arrival date.
 * Deletes compute the same value from the same rkey, so they share a partition
 * with the create whenever the rkey is a TID.
 */
export function rkeyDate(rkey: string, arrivalMs: number): string {
  const v = tidToBigInt(rkey);
  if (v !== null) {
    const ms = Number(v >> 10n) / 1000;
    if (ms >= MIN_MS && ms <= arrivalMs + 86_400_000) return isoDate(ms);
  }
  return isoDate(arrivalMs);
}

function isoDate(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}
