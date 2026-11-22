import { describe, expect, test } from "bun:test";
import { FileAccountSignals, MAX_SIGNAL_MULTIPLIER, parseSignalsFile } from "../src/onboarding_signals.ts";

const DID_A = "did:plc:aaaaaaaaaaaaaaaaaaaaaaaa";
const DID_B = "did:plc:bbbbbbbbbbbbbbbbbbbbbbbb";
const START = Date.parse("2026-11-22T12:00:00Z");

function file(version: string, accounts: Record<string, unknown>, generatedAt = "2026-11-22T11:00:00Z"): string {
  return JSON.stringify({ signalsVersion: version, generatedAt, accounts });
}

/** A signals reader over an in-memory "disk" whose clock and contents the test controls. */
function harness(initial: string | null, options: { maxAgeMs?: number } = {}) {
  const disk = { text: initial, modified: initial === null ? null as number | null : 1, reads: 0 };
  const clock = { t: START };
  const signals = new FileAccountSignals({
    path: "/signals.json",
    maxAgeMs: options.maxAgeMs,
    recheckMs: 60_000,
    now: () => clock.t,
    read: () => {
      disk.reads += 1;
      return disk.text!;
    },
    modifiedAt: () => disk.modified,
  });
  const write = (text: string | null) => {
    disk.text = text;
    disk.modified = text === null ? null : (disk.modified ?? 0) + 1;
  };
  return { signals, disk, clock, write };
}

describe("onboarding signals file", () => {
  test("multipliers and sample posts are read per account", () => {
    const { signals } = harness(file("s1", {
      [DID_A]: { multiplier: 0.7, samplePostUri: `at://${DID_A}/app.bsky.feed.post/3abc` },
      [DID_B]: { multiplier: 0 },
    }));
    expect(signals.accountMultiplier(DID_A)).toBe(0.7);
    expect(signals.samplePostFor(DID_A)).toBe(`at://${DID_A}/app.bsky.feed.post/3abc`);
    expect(signals.accountMultiplier(DID_B)).toBe(0);
    expect(signals.samplePostFor(DID_B)).toBeNull();
    expect(signals.accountMultiplier("did:plc:unknown")).toBeNull();
    expect(signals.version()).toBe("s1");
  });

  test("bad values are dropped rather than trusted", () => {
    const loaded = parseSignalsFile(file("s1", {
      [DID_A]: { multiplier: 99, samplePostUri: `at://${DID_B}/app.bsky.feed.post/3abc` },
      [DID_B]: { multiplier: -1, samplePostUri: "https://example.com/post" },
      "not-a-did": { multiplier: 1 },
    }));
    expect(loaded.multipliers.get(DID_A)).toBe(MAX_SIGNAL_MULTIPLIER);
    expect(loaded.multipliers.has(DID_B)).toBe(false);
    expect(loaded.multipliers.has("not-a-did")).toBe(false);
    // An account can only be given one of its own posts.
    expect(loaded.samplePosts.size).toBe(0);
  });

  test("a missing file means no signal, and the file is picked up when it appears", () => {
    const { signals, clock, write } = harness(null);
    expect(signals.accountMultiplier(DID_A)).toBeNull();
    expect(signals.version()).toBeNull();
    write(file("s1", { [DID_A]: { multiplier: 1.2 } }));
    expect(signals.accountMultiplier(DID_A)).toBeNull(); // not rechecked yet
    clock.t += 61_000;
    expect(signals.accountMultiplier(DID_A)).toBe(1.2);
  });

  test("a new file replaces the old one without a restart; an unchanged file is not re-read", () => {
    const { signals, clock, write, disk } = harness(file("s1", { [DID_A]: { multiplier: 0.5 } }));
    expect(signals.accountMultiplier(DID_A)).toBe(0.5);
    clock.t += 61_000;
    expect(signals.accountMultiplier(DID_A)).toBe(0.5);
    expect(disk.reads).toBe(1);
    write(file("s2", { [DID_A]: { multiplier: 1.1 } }));
    clock.t += 61_000;
    expect(signals.accountMultiplier(DID_A)).toBe(1.1);
    expect(signals.version()).toBe("s2");
    expect(disk.reads).toBe(2);
  });

  test("a half-written file leaves the previous signals in place and is reported", () => {
    const { signals, clock, write } = harness(file("s1", { [DID_A]: { multiplier: 0.5 } }));
    expect(signals.accountMultiplier(DID_A)).toBe(0.5);
    write("{ \"signalsVersion\": \"s2\", \"accou");
    clock.t += 61_000;
    expect(signals.accountMultiplier(DID_A)).toBe(0.5);
    expect(signals.describe().onboarding_signals_error).not.toBeNull();
    write(file("s3", { [DID_A]: { multiplier: 0.9 } }));
    clock.t += 61_000;
    expect(signals.accountMultiplier(DID_A)).toBe(0.9);
    expect(signals.describe().onboarding_signals_error).toBeNull();
  });

  test("a removed file stops being used", () => {
    const { signals, clock, write } = harness(file("s1", { [DID_A]: { multiplier: 0 } }));
    expect(signals.accountMultiplier(DID_A)).toBe(0);
    write(null);
    clock.t += 61_000;
    expect(signals.accountMultiplier(DID_A)).toBeNull();
  });

  test("signals past their maximum age are ignored until a fresh file arrives", () => {
    const { signals, clock, write } = harness(file("s1", { [DID_A]: { multiplier: 0 } }), { maxAgeMs: 6 * 3_600_000 });
    expect(signals.accountMultiplier(DID_A)).toBe(0);
    clock.t += 7 * 3_600_000;
    expect(signals.accountMultiplier(DID_A)).toBeNull();
    expect(signals.describe().onboarding_signals_in_use).toBe(false);
    write(file("s2", { [DID_A]: { multiplier: 0 } }, new Date(clock.t).toISOString()));
    clock.t += 61_000;
    expect(signals.accountMultiplier(DID_A)).toBe(0);
  });
});
