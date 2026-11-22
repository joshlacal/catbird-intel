// Local-only development proxy for the intel service.
// The service learns who the viewer is from a signed gateway token. This proxy stands in for the
// gateway on a developer machine: it signs each forwarded request for one viewer DID, taken from
// the `x-debug-viewer-did` request header or the VIEWER_DID environment variable. With neither,
// requests are forwarded unsigned (anonymous viewer).
//
//   INTEL_INTERNAL_BFF_SECRET=<same dev secret as the service> UPSTREAM=http://127.0.0.1:3298 \
//   PORT=3299 [VIEWER_DID=did:plc:...] bun eval/onboarding/dev_viewer_proxy.ts
import { createHmac, randomUUID } from "node:crypto";

const secretText = process.env.INTEL_INTERNAL_BFF_SECRET ?? "";
const secret = secretText.length % 2 === 0 && /^[0-9a-f]+$/i.test(secretText)
  ? Buffer.from(secretText, "hex") : Buffer.from(secretText, "utf8");
if (secret.byteLength < 32) throw new Error("INTEL_INTERNAL_BFF_SECRET must be at least 32 bytes");
const upstream = process.env.UPSTREAM ?? "http://127.0.0.1:3298";
const audience = process.env.INTEL_SERVICE_DID ?? "did:web:intel.catbird.blue#catbird_intel";
const defaultViewer = process.env.VIEWER_DID ?? null;

const server = Bun.serve({
  hostname: "127.0.0.1",
  port: Number(process.env.PORT ?? 3299),
  idleTimeout: 60,
  async fetch(request) {
    const url = new URL(request.url);
    const target = new URL(url.pathname + url.search, upstream);
    const headers = new Headers({ accept: "application/json" });
    const contentType = request.headers.get("content-type");
    if (contentType) headers.set("content-type", contentType);
    const viewer = request.headers.get("x-debug-viewer-did") ?? defaultViewer;
    if (viewer && url.pathname.startsWith("/xrpc/")) {
      const iat = Math.floor(Date.now() / 1000);
      const body = Buffer.from(JSON.stringify({
        iss: viewer, aud: audience, lxm: url.pathname.slice(6), iat, exp: iat + 50, jti: randomUUID(),
      })).toString("base64url");
      const signature = createHmac("sha256", secret).update(body).digest().toString("base64url");
      headers.set("authorization", `Bearer bff.${body}.${signature}`);
    }
    const hasBody = request.method !== "GET" && request.method !== "HEAD";
    const response = await fetch(target, {
      method: request.method,
      headers,
      body: hasBody ? await request.arrayBuffer() : undefined,
    });
    return new Response(response.body, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
    });
  },
});
console.log(`dev viewer proxy on ${server.url} -> ${upstream} (viewer: ${defaultViewer ?? "from x-debug-viewer-did header, else anonymous"})`);
