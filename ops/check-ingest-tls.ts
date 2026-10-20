#!/usr/bin/env bun
// Read-only network check for the ingest runtime and its installed CA bundle.
// Run with the same Bun version and INGEST_TRUST_BUNDLE as src/ingest.ts.
import { Jetstream, websocketTransport } from "@bsky/jetstream";
import { ClickHouseWriter } from "../src/clickhouse.ts";
import { ingestTlsOptions } from "../src/ingest.ts";

const tls = ingestTlsOptions();
const rejectedCodes = new Set([
  "UNABLE_TO_GET_ISSUER_CERT_LOCALLY",
  "UNABLE_TO_VERIFY_LEAF_SIGNATURE",
  "SELF_SIGNED_CERT_IN_CHAIN",
]);
const incidentHosts = ["dv", "ov", "ev"].flatMap((tier) =>
  ["g3b", "g3c"].flatMap((chain) =>
    ["", "-aaa"].map((suffix) => `${tier}-${chain}${suffix}-emudhra.we-have-a-problem.com`)));

function certificateErrorCode(error: unknown): string | undefined {
  let current: unknown = error;
  for (let depth = 0; depth < 6 && current && typeof current === "object"; depth++) {
    const detail = current as { code?: string; cause?: unknown };
    if (detail.code && rejectedCodes.has(detail.code)) return detail.code;
    current = detail.cause;
  }
  return undefined;
}

function isCertificateFailure(error: unknown): boolean {
  let current: unknown = error;
  for (let depth = 0; depth < 6 && current && typeof current === "object"; depth++) {
    const detail = current as { code?: string; message?: string; cause?: unknown };
    if (detail.code && rejectedCodes.has(detail.code)) return true;
    if (detail.message?.includes("TLS handshake failed")) return true;
    current = detail.cause;
  }
  return false;
}

async function rejectAffectedHttps(host: string): Promise<void> {
  try {
    const response = await fetch(`https://${host}/`, { tls, signal: AbortSignal.timeout(10_000) });
    await response.body?.cancel();
    throw new Error(`affected HTTPS chain accepted: ${host}, status=${response.status}`);
  } catch (error) {
    const code = certificateErrorCode(error);
    if (!code) throw new Error(`affected HTTPS probe did not fail certificate verification: ${host}`, { cause: error });
    console.log(`HTTPS certificate rejected: ${host}, ${code}`);
  }
}

async function rejectAffectedChain(host: string): Promise<void> {
  const stream = websocketTransport({ shouldReconnect: false, tls })
    .stream(() => `wss://${host}/`, AbortSignal.timeout(10_000))[Symbol.asyncIterator]();
  try {
    const result = await stream.next();
    throw new Error(`affected WSS chain accepted: ${host}, result=${JSON.stringify(result)}`);
  } catch (error) {
    if (!isCertificateFailure(error)) {
      throw new Error(`affected WSS probe did not fail at TLS: ${host}`, { cause: error });
    }
    console.log(`WSS TLS rejected: ${host}`);
  } finally {
    await stream.return?.();
  }
}

await Promise.all(incidentHosts.map(rejectAffectedHttps));
await Promise.all(incidentHosts.map(rejectAffectedChain));

const controlResponse = await fetch("https://jetstream.us-east.bsky.network/status", {
  tls,
  signal: AbortSignal.timeout(10_000),
});
await controlResponse.body?.cancel();
if (!controlResponse.ok) throw new Error(`normal HTTPS control failed: ${controlResponse.status}`);
console.log("normal Jetstream HTTPS control passed");

async function confirmJetstream(service: string, transportTls = tls): Promise<void> {
  const jetstream = new Jetstream({ service, validateWire: false });
  const stream = jetstream.live({
    collections: ["app.bsky.feed.*"],
    raw: true,
    signal: AbortSignal.timeout(20_000),
    liveTransport: websocketTransport({ shouldReconnect: false, tls: transportTls }),
  });
  let firstSeq: number | undefined;
  let lastSeq: number | undefined;
  for await (const event of stream) {
    firstSeq ??= event.seq;
    lastSeq = event.seq;
    if (lastSeq > firstSeq) break;
  }
  if (firstSeq === undefined || lastSeq === undefined || lastSeq <= firstSeq) {
    throw new Error(`normal Jetstream WSS did not advance sequence for ${service}: ${firstSeq} -> ${lastSeq}`);
  }
  console.log(`normal Jetstream WSS sequence advanced for ${service}: ${firstSeq} -> ${lastSeq}`);
}

await Promise.all([
  confirmJetstream("https://jetstream.us-east.bsky.network"),
  confirmJetstream("https://jetstream.us-west.bsky.network"),
]);

// Passing only an unrelated root to the same healthy WSS endpoint must fail.
// If Bun ignores the patched tls.ca option, its built-in roots will connect.
const restrictedCa = tls.ca.match(/-----BEGIN CERTIFICATE-----[\s\S]*?-----END CERTIFICATE-----/)?.[0];
if (!restrictedCa) throw new Error("host CA bundle has no certificate for restricted trust control");
let restrictedRejected = false;
try {
  await confirmJetstream("https://jetstream.us-east.bsky.network", { ca: restrictedCa, rejectUnauthorized: true });
} catch (error) {
  if (!isCertificateFailure(error)) throw new Error("restricted CA WSS probe was inconclusive", { cause: error });
  restrictedRejected = true;
  console.log("Jetstream WSS rejected unrelated CA while full host CA succeeded");
}
if (!restrictedRejected) throw new Error("Jetstream WSS accepted an unrelated restricted CA");

let clickhouseRequest = false;
const server = Bun.serve({
  hostname: "127.0.0.1",
  port: 0,
  fetch(request) {
    const url = new URL(request.url);
    if (url.protocol !== "http:" || url.hostname !== "127.0.0.1" || request.method !== "POST") {
      return new Response("unexpected loopback request", { status: 400 });
    }
    clickhouseRequest = true;
    return new Response("1\n");
  },
});
try {
  const writer = new ClickHouseWriter({
    url: `http://127.0.0.1:${server.port}`,
    database: "catbird",
    maxRows: 1,
  });
  const result = await writer.query("SELECT 1");
  if (!clickhouseRequest || result !== "1\n") throw new Error("ClickHouse loopback HTTP control failed");
  console.log("ClickHouse loopback HTTP control passed");
} finally {
  server.stop(true);
}

console.log("ingest TLS check passed: 12 HTTPS and 12 WSS chains rejected; Jetstream and loopback controls passed");
