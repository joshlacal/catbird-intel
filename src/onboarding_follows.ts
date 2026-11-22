/**
 * The default follow-list loader for live onboarding: pages through a viewer's follows on the
 * public AppView. The engine takes any `FollowsFetcher`, so tests and other hosts can swap this out.
 */
import type { FollowsFetcher } from "./onboarding_live_types.ts";

/**
 * Follow-list loader over the public AppView. One page is 100 follows; `maxPages` bounds the
 * work for accounts that follow tens of thousands.
 */
export function createPublicFollowsFetcher(options: {
  fetcher?: (input: URL, init?: RequestInit) => Promise<Response>;
  baseUrl?: string;
  maxPages?: number;
  timeoutMs?: number;
} = {}): FollowsFetcher {
  const fetcher = options.fetcher ?? fetch;
  const baseUrl = options.baseUrl ?? "https://public.api.bsky.app";
  const maxPages = options.maxPages ?? 100;
  const timeoutMs = options.timeoutMs ?? 3_000;
  return async (viewerDid, onPage) => {
    let cursor: string | undefined;
    for (let page = 0; page < maxPages; page++) {
      const url = new URL("/xrpc/app.bsky.graph.getFollows", baseUrl);
      url.searchParams.set("actor", viewerDid);
      url.searchParams.set("limit", "100");
      if (cursor) url.searchParams.set("cursor", cursor);
      const response = await fetcher(url, {
        headers: { accept: "application/json" },
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!response.ok) throw new Error(`getFollows answered ${response.status}`);
      const body = await response.json() as { follows?: unknown; cursor?: unknown };
      const dids: string[] = [];
      for (const raw of Array.isArray(body.follows) ? body.follows : []) {
        const did = raw && typeof raw === "object" ? (raw as { did?: unknown }).did : undefined;
        if (typeof did === "string" && did.startsWith("did:")) dids.push(did);
      }
      if (onPage(dids) === true) return;
      cursor = typeof body.cursor === "string" && body.cursor.length > 0 ? body.cursor : undefined;
      if (!cursor) return;
    }
  };
}
