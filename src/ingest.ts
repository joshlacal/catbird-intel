#!/usr/bin/env bun
// catbird-intel-ingest: Jetstream v2 live stream -> ClickHouse `catbird` db.
//
// At-least-once: rows are flushed in batches and the cursor is saved only
// after a flush succeeds. Every record table is keyed (did, rkey) and versioned
// by repo revision, so replayed events collapse.

import { Jetstream, websocketTransport } from "@bsky/jetstream";
import { setTimeout as delay } from "node:timers/promises";
import { ClickHouseWriter } from "./clickhouse.ts";
import { loadCursor, resumeCursor, saveCursor, type CursorState } from "./cursor.ts";
import { chTime, eventToRows, type RawEventLike } from "./extract.ts";

const env = (k: string, d: string) => process.env[k] ?? d;
const SERVICES = env("JETSTREAM_SERVICES", "https://jetstream.us-east.bsky.network,https://jetstream.us-west.bsky.network")
  .split(",")
  .map((s) => s.trim())
  .filter(Boolean);
const CURSOR_FILE = env("CURSOR_FILE", "/var/lib/catbird-intel/ingest.cursor");
const FLUSH_MS = Number(env("FLUSH_MS", "2000"));
const MAX_ROWS = Number(env("MAX_ROWS", "20000"));
const HEALTH_MS = Number(env("HEALTH_MS", "10000"));
const COLLECTIONS = ["app.bsky.feed.*", "app.bsky.graph.*", "app.bsky.actor.*"] as const;

const writer = new ClickHouseWriter({
  url: env("CH_URL", "http://127.0.0.1:8123"),
  database: env("CH_DATABASE", "catbird"),
  user: process.env.CH_USER,
  password: process.env.CH_PASSWORD,
  maxRows: MAX_ROWS,
});

const stop = new AbortController();
let serviceIndex = 0;
let lastFlushAt = Date.now();
let lastFlushMs = 0;
let lastDelivered: CursorState | null = null;
let stats = { events: 0, rows: 0, reconnects: 0 };

function log(msg: string, extra?: Record<string, unknown>) {
  console.log(JSON.stringify({ ts: new Date().toISOString(), msg, ...extra }));
}

async function flush(): Promise<void> {
  const started = Date.now();
  const written = await writer.flush();
  lastFlushMs = Date.now() - started;
  lastFlushAt = Date.now();
  stats.rows += written;
  if (lastDelivered) await saveCursor(CURSOR_FILE, lastDelivered);
}

async function flushWithRetry(): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      await flush();
      return;
    } catch (error) {
      // Keep the buffer and the old cursor; storage outages must not skip data.
      log("flush failed; retrying", { attempt, error: String(error) });
      if (stop.signal.aborted && attempt >= 3) throw error;
      await delay(Math.min(30_000, 1000 * 2 ** attempt));
    }
  }
}

async function writeHealth(): Promise<void> {
  const now = Date.now();
  const lastEventMs = lastDelivered ? lastDelivered.timeUs / 1000 : 0;
  const row = {
    ts: chTime(now),
    service: SERVICES[serviceIndex],
    last_seq: lastDelivered?.seq ?? 0,
    last_event_time: chTime(lastEventMs || now),
    lag_ms: lastEventMs ? Math.round(now - lastEventMs) : -1,
    events: stats.events,
    rows_written: stats.rows,
    reconnects: stats.reconnects,
    flush_ms: lastFlushMs,
    rss_mb: Math.round(process.memoryUsage().rss / 1048576),
  };
  stats = { events: 0, rows: 0, reconnects: 0 };
  try {
    await writer.insert("ingest_health", [row]);
  } catch (error) {
    log("health write failed", { error: String(error) });
  }
}

async function runService(service: string): Promise<void> {
  const saved = (await loadCursor(CURSOR_FILE)) ?? lastDelivered;
  const cursor = resumeCursor(saved, service);
  log("connecting", { service, cursor: cursor ?? "live-tip" });
  const js = new Jetstream({ service, validateWire: true });
  // live() only calls load(); this process persists its own cursor after
  // each successful flush, so save() is intentionally a no-op.
  const cursorStore = { load: async () => cursor, save: async () => undefined };
  const stream = js.live({
    collections: COLLECTIONS,
    cursor: cursorStore,
    raw: true,
    signal: stop.signal,
    liveTransport: websocketTransport({
      highWaterMark: 1024 * 1024,
      maxBufferedBytes: 64 * 1024 * 1024,
      onReconnect: (error, info) => {
        stats.reconnects++;
        log("transport reconnect", { attempt: info.attempt, error: error instanceof Error ? error.message : String(error) });
      },
    }),
    onInfo: (info) => log("jetstream info", { name: info.name, message: info.message }),
    onError: (err) => log("jetstream event error", { error: err.message }),
  });

  for await (const raw of stream as AsyncIterable<RawEventLike>) {
    stats.events++;
    for (const { table, row } of eventToRows(raw)) writer.add(table, row);
    const timeMs = Date.parse(raw.time);
    lastDelivered = { service, seq: raw.seq, timeUs: Number.isFinite(timeMs) ? timeMs * 1000 : Date.now() * 1000 };
    if (writer.shouldFlush() || Date.now() - lastFlushAt >= FLUSH_MS) await flushWithRetry();
  }
}

async function main(): Promise<void> {
  process.on("SIGTERM", () => stop.abort());
  process.on("SIGINT", () => stop.abort());
  const health = setInterval(() => void writeHealth(), HEALTH_MS);
  let failures = 0;
  while (!stop.signal.aborted) {
    const service = SERVICES[serviceIndex] as string;
    const started = Date.now();
    try {
      await runService(service);
      if (stop.signal.aborted) break;
      log("stream ended without error; reconnecting", { service });
    } catch (error) {
      if (stop.signal.aborted) break;
      log("stream failed", { service, error: error instanceof Error ? `${error.name}: ${error.message}` : String(error) });
    }
    // Persist what we have before reconnecting; resume from the saved cursor.
    await flushWithRetry().catch((e) => log("flush before reconnect failed", { error: String(e) }));
    failures = Date.now() - started > 120_000 ? 0 : failures + 1;
    if (failures >= 3 && SERVICES.length > 1) {
      serviceIndex = (serviceIndex + 1) % SERVICES.length;
      failures = 0;
      log("failing over", { to: SERVICES[serviceIndex] });
    }
    await delay(Math.min(30_000, 1000 * 2 ** failures)).catch(() => undefined);
  }
  clearInterval(health);
  await flushWithRetry().catch((e) => log("final flush failed; cursor unchanged", { error: String(e) }));
  await writeHealth();
  log("stopped");
}

await main();
