#!/usr/bin/env bun
// catbird-intel-ingest: Jetstream v2 live stream -> ClickHouse `catbird` db.
//
// At-least-once: rows are accumulated in memory, swapped out atomically into a
// flush batch so the WebSocket loop never blocks on HTTP I/O (preventing Bun's
// unimplemented ws.pause() from overflowing during catch-up bursts), and the
// cursor is saved only after the batch write succeeds. Every record table is
// keyed (did, rkey) and versioned by repo revision, so replayed events collapse.

import { Jetstream, websocketTransport } from "@bsky/jetstream";
import { X509Certificate } from "node:crypto";
import { readFileSync } from "node:fs";
import { isAbsolute } from "node:path";
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
const FLUSH_MS = Number(env("FLUSH_MS", "1500"));
const MAX_ROWS = Number(env("MAX_ROWS", "15000"));
const HEALTH_MS = Number(env("HEALTH_MS", "10000"));
const COLLECTIONS = ["app.bsky.feed.*", "app.bsky.graph.*", "app.bsky.actor.*"] as const;

// A stale host CA bundle must fail startup. Bun's bundled roots can still
// contain these incident roots, so --use-system-ca alone cannot remove them.
const REJECTED_ROOTS = new Set([
  "7DD78D5F4F13459A83DFF9ABBBA62EDBAF6F2D102BF257FD712F4D9F2746ED8D", // emSign Root TLS CA - G3
  "F2EDAAFF3D9B701FB4D14C9F285F8C0BAD90FC3BCC1FA07D47F4718EDABA34A5", // emSign Root CS CA - G3
  "D7A7A0FB5D7E2731D771E9484EBCDEF71D5F0C3E0A2948782BC83EE0EA699EF4", // AAA Certificate Services
  "86A1ECBA089C4A8D3BBE2734C612BA341D813E043CF9E8A862CD5C57A36BBE6B", // emSign ECC Root CA - G3
]);

export function ingestTlsOptions(
  bundlePath = env("INGEST_TRUST_BUNDLE", "/etc/ssl/certs/ca-certificates.crt"),
): { ca: string; rejectUnauthorized: true } {
  const [major = NaN, minor = NaN] = Bun.version.split(".").map(Number);
  if (!Number.isInteger(major) || !Number.isInteger(minor) || major < 1 || (major === 1 && minor < 4)) {
    throw new Error("catbird-intel ingest requires Bun 1.4 or later for explicit WebSocket TLS trust");
  }
  if (!isAbsolute(bundlePath)) throw new Error("INGEST_TRUST_BUNDLE must be an absolute path");
  if (process.env.NODE_TLS_REJECT_UNAUTHORIZED === "0") {
    throw new Error("NODE_TLS_REJECT_UNAUTHORIZED=0 disables certificate verification");
  }
  const ca = readFileSync(bundlePath, "utf8");
  const certificates = ca.match(/-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----/g) ?? [];
  if (certificates.length === 0 || ca.replace(/-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----/g, "").trim()) {
    throw new Error(`invalid PEM certificate bundle: ${bundlePath}`);
  }
  for (const pem of certificates) {
    const fingerprint = new X509Certificate(pem).fingerprint256.replaceAll(":", "").toUpperCase();
    if (REJECTED_ROOTS.has(fingerprint)) {
      throw new Error(`affected certificate remains in ${bundlePath}: ${fingerprint}`);
    }
  }
  return { ca, rejectUnauthorized: true };
}

const jetstreamTls = ingestTlsOptions();

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
let inFlightFlush: Promise<void> | null = null;
let stats = { events: 0, rows: 0, reconnects: 0 };

function log(msg: string, extra?: Record<string, unknown>) {
  console.log(JSON.stringify({ ts: new Date().toISOString(), msg, ...extra }));
}

async function flushWithRetry(): Promise<void> {
  if (inFlightFlush) {
    await inFlightFlush;
    return;
  }
  const batch = writer.takeBatch();
  const snapCursor = lastDelivered ? { ...lastDelivered } : null;
  if (batch.count === 0) return;

  const p = (async () => {
    for (let attempt = 0; ; attempt++) {
      try {
        const started = Date.now();
        const written = await writer.writeBatch(batch);
        lastFlushMs = Date.now() - started;
        lastFlushAt = Date.now();
        stats.rows += written;
        if (snapCursor) await saveCursor(CURSOR_FILE, snapCursor);
        return;
      } catch (error) {
        log("flush failed; retrying", { attempt, error: String(error) });
        if (stop.signal.aborted && attempt >= 3) throw error;
        await delay(Math.min(30_000, 1000 * 2 ** attempt));
      }
    }
  })();

  inFlightFlush = p.finally(() => {
    inFlightFlush = null;
  });
  await inFlightFlush;
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
  const js = new Jetstream({ service, validateWire: false });
  const cursorStore = { load: async () => cursor, save: async () => undefined };
  const stream = js.live({
    collections: COLLECTIONS,
    cursor: cursorStore,
    raw: true,
    signal: stop.signal,
    liveTransport: websocketTransport({
      // @atproto/ws-client forwards this to Bun's native WebSocket upgrade.
      // Setting node:tls defaults alone does not replace Bun's bundled roots.
      tls: jetstreamTls,
      highWaterMark: 64 * 1024 * 1024,
      maxBufferedBytes: 512 * 1024 * 1024,
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
    if (!inFlightFlush && (writer.shouldFlush() || Date.now() - lastFlushAt >= FLUSH_MS)) {
      // Launch flush asynchronously so the WebSocket consumer loop never pauses
      void flushWithRetry().catch((e) => log("async flush error", { error: String(e) }));
    } else if (writer.bufferedRows >= MAX_ROWS * 6 && inFlightFlush) {
      // Only apply backpressure if ClickHouse is genuinely falling behind by >90k rows
      await inFlightFlush;
    }
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
      const errName = error instanceof Error ? error.name : "";
      const errMsg = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
      log("stream failed", { service, error: errMsg });
      if (errName === "MalformedError" && lastDelivered) {
        lastDelivered = {
          ...lastDelivered,
          seq: lastDelivered.seq + 1,
          timeUs: lastDelivered.timeUs + 1000,
        };
      }
    }
    if (inFlightFlush) await inFlightFlush.catch(() => undefined);
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
  if (inFlightFlush) await inFlightFlush.catch(() => undefined);
  await flushWithRetry().catch((e) => log("final flush failed; cursor unchanged", { error: String(e) }));
  await writeHealth();
  log("stopped");
}

if (import.meta.main) await main();
