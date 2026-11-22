// Account signals for the live onboarding engine, read from a file that is rebuilt on its own
// schedule (recent activity, how often follows of an account are kept, a fresher sample post).
//
// The file is separate from the catalog and the pool so that it can change many times a day
// without a deploy or a restart: the service notices a new file and swaps it in. A missing,
// unreadable or expired file means "no signal", never a failure: the engine then orders accounts
// exactly as it does without signals.
import { readFileSync, statSync } from "node:fs";
import type { LiveAccountSignals } from "./onboarding_live.ts";

export interface OnboardingSignalsFile {
  signalsVersion: string;
  /** ISO time the signals were computed. */
  generatedAt: string;
  accounts: Record<string, OnboardingAccountSignal>;
}

export interface OnboardingAccountSignal {
  /** Scales the account's chance of being drawn early. 0 keeps it out of new lists. */
  multiplier?: number;
  /** A more recent post to show for the account than the one in the catalog. */
  samplePostUri?: string;
}

export const MAX_SIGNAL_MULTIPLIER = 4;

interface Loaded {
  version: string;
  generatedAtMs: number;
  multipliers: Map<string, number>;
  samplePosts: Map<string, string>;
}

export function parseSignalsFile(text: string): Loaded {
  const parsed = JSON.parse(text) as Partial<OnboardingSignalsFile> | null;
  if (!parsed || typeof parsed.signalsVersion !== "string" || typeof parsed.accounts !== "object" || !parsed.accounts) {
    throw new Error("not an intel_onboarding_signals_v1 document");
  }
  const generatedAtMs = Date.parse(parsed.generatedAt ?? "");
  if (!Number.isFinite(generatedAtMs)) throw new Error("generatedAt is missing or not a time");
  const multipliers = new Map<string, number>();
  const samplePosts = new Map<string, string>();
  for (const [did, signal] of Object.entries(parsed.accounts)) {
    if (!did.startsWith("did:") || !signal || typeof signal !== "object") continue;
    const m = signal.multiplier;
    if (typeof m === "number" && Number.isFinite(m) && m >= 0) multipliers.set(did, Math.min(m, MAX_SIGNAL_MULTIPLIER));
    const uri = signal.samplePostUri;
    // Only the account's own posts: a signals file cannot make one account show another's post.
    if (typeof uri === "string" && uri.startsWith(`at://${did}/app.bsky.feed.post/`)) samplePosts.set(did, uri);
  }
  return { version: parsed.signalsVersion, generatedAtMs, multipliers, samplePosts };
}

export class FileAccountSignals implements LiveAccountSignals {
  private readonly path: string;
  private readonly maxAgeMs: number | null;
  private readonly recheckMs: number;
  private readonly now: () => number;
  private readonly read: (path: string) => string;
  private readonly modifiedAt: (path: string) => number | null;
  private loaded: Loaded | null = null;
  private loadedModifiedAt: number | null = null;
  private lastCheck = Number.NEGATIVE_INFINITY;
  private lastError: string | null = null;

  constructor(options: {
    path: string;
    /** Signals older than this are ignored. Leave unset to use them whatever their age. */
    maxAgeMs?: number | null;
    recheckMs?: number;
    now?: () => number;
    read?: (path: string) => string;
    modifiedAt?: (path: string) => number | null;
  }) {
    this.path = options.path;
    this.maxAgeMs = options.maxAgeMs ?? null;
    this.recheckMs = options.recheckMs ?? 60_000;
    this.now = options.now ?? Date.now;
    this.read = options.read ?? ((path) => readFileSync(path, "utf8"));
    this.modifiedAt = options.modifiedAt ?? ((path) => {
      try {
        return statSync(path).mtimeMs;
      } catch {
        return null;
      }
    });
  }

  accountMultiplier(did: string): number | null {
    return this.current()?.multipliers.get(did) ?? null;
  }

  samplePostFor(did: string): string | null {
    return this.current()?.samplePosts.get(did) ?? null;
  }

  version(): string | null {
    return this.current()?.version ?? null;
  }

  describe(): Record<string, unknown> {
    const usable = this.current();
    return {
      onboarding_signals_path: this.path,
      onboarding_signals_version: this.loaded?.version ?? null,
      onboarding_signals_in_use: usable !== null,
      onboarding_signals_age_hours: this.loaded
        ? Math.round((this.now() - this.loaded.generatedAtMs) / 360_000) / 10
        : null,
      onboarding_signals_accounts: this.loaded?.multipliers.size ?? 0,
      onboarding_signals_sample_posts: this.loaded?.samplePosts.size ?? 0,
      onboarding_signals_error: this.lastError,
    };
  }

  /** The loaded signals if there are any and they have not expired. Reloads when the file changes. */
  private current(): Loaded | null {
    const t = this.now();
    if (t - this.lastCheck >= this.recheckMs) {
      this.lastCheck = t;
      this.reloadIfChanged();
    }
    if (!this.loaded) return null;
    if (this.maxAgeMs !== null && t - this.loaded.generatedAtMs > this.maxAgeMs) return null;
    return this.loaded;
  }

  private reloadIfChanged(): void {
    const modified = this.modifiedAt(this.path);
    if (modified === null) {
      // The file was removed: stop using what was loaded from it.
      this.loaded = null;
      this.loadedModifiedAt = null;
      this.lastError = null;
      return;
    }
    if (modified === this.loadedModifiedAt) return;
    this.loadedModifiedAt = modified;
    try {
      this.loaded = parseSignalsFile(this.read(this.path));
      this.lastError = null;
    } catch (error) {
      // A half-written or malformed file leaves the previous signals in place.
      this.lastError = String(error instanceof Error ? error.message : error).slice(0, 200);
    }
  }
}
