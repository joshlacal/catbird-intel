import { describe, expect, test } from "bun:test";
import { P256Keypair, Secp256k1Keypair, type Keypair } from "@atproto/crypto";
import { createServiceAuthenticator } from "../src/service_auth.ts";
import { createApiHandler } from "../src/api.ts";
import { FIXTURE_DIDS, installLocalServingFixture } from "./serving_fixtures.ts";

const ISSUER = FIXTURE_DIDS.ordinary;
const AUDIENCE = "did:web:local.example#catbird_intel";
const METHOD = "blue.catbird.onboarding.getSceneCards";
const NOW = 1800000000000;
const keys = await P256Keypair.create();

async function token(body: Record<string, unknown> = {}, header: Record<string, unknown> = {}, key: Keypair = keys) {
  const encoded = [ { typ: "JWT", alg: "ES256", ...header },
    { iss: ISSUER, aud: AUDIENCE, lxm: METHOD, iat: NOW / 1000, exp: NOW / 1000 + 60, jti: crypto.randomUUID(), ...body },
  ].map((value) => Buffer.from(JSON.stringify(value)).toString("base64url")).join(".");
  return `${encoded}.${Buffer.from(await key.sign(Buffer.from(encoded))).toString("base64url")}`;
}
const request = (jwt?: string) => new Request("http://127.0.0.1/xrpc/" + METHOD,
  jwt ? { headers: { authorization: `Bearer ${jwt}` } } : {});
const verifier = () => createServiceAuthenticator({ audience: AUDIENCE, now: () => NOW,
  resolveAtprotoKey: async (issuer) => {
    if (issuer !== ISSUER) throw new Error("unknown issuer");
    return keys.did();
  } });

describe("official inter-service JWT boundary with real cryptographic signatures", () => {
  test("anonymous without a token, but no token accepted without trust configuration", async () => {
    expect(await createServiceAuthenticator()(request(), METHOD)).toBeNull();
    await expect(createServiceAuthenticator()(request(await token()), METHOD)).rejects.toThrow();
  });
  test("valid ES256 signature authenticates the exact issuer once", async () => {
    const verify = verifier();
    const jwt = await token();
    expect(await verify(request(jwt), METHOD)).toBe(ISSUER);
    await expect(verify(request(jwt), METHOD)).rejects.toThrow();
  });
  test("valid ES256K signature works with the matching trusted key", async () => {
    const k256 = await Secp256k1Keypair.create();
    const verify = createServiceAuthenticator({ audience: AUDIENCE, now: () => NOW,
      resolveAtprotoKey: async () => k256.did() });
    expect(await verify(request(await token({}, { alg: "ES256K" }, k256)), METHOD)).toBe(ISSUER);
  });
  test("wrong signature, key purpose and algorithm are rejected", async () => {
    const wrong = await P256Keypair.create();
    for (const jwt of [await token({}, {}, wrong), await token({}, { kid: "#other" }),
      await token({}, { alg: "HS256" }), await token({}, { typ: "at+jwt" }), await token({}, { alg: "ES256K" })]) {
      await expect(verifier()(request(jwt), METHOD)).rejects.toThrow();
    }
  });
  test("audience fragment, endpoint and issuer are bound", async () => {
    for (const body of [{ aud: "did:web:local.example" }, { aud: AUDIENCE + "_other" }, { lxm: "other.method" },
      { iss: ISSUER + "#atproto" }, { iss: "did:plc:unknown" }]) {
      await expect(verifier()(request(await token(body)), METHOD)).rejects.toThrow();
    }
  });
  test("expiry, issued-at, lifetime and nonce are required", async () => {
    for (const body of [{ exp: NOW / 1000 }, { iat: NOW / 1000 + 1 }, { exp: NOW / 1000 + 61 },
      { iat: undefined }, { exp: undefined }, { jti: undefined }, { jti: "" }]) {
      await expect(verifier()(request(await token(body)), METHOD)).rejects.toThrow();
    }
  });
  test("nonce cache capacity fails closed", async () => {
    const verify = createServiceAuthenticator({ audience: AUDIENCE, now: () => NOW, maxNonces: 1,
      resolveAtprotoKey: async () => keys.did() });
    expect(await verify(request(await token()), METHOD)).toBe(ISSUER);
    await expect(verify(request(await token()), METHOD)).rejects.toThrow();
  });
  test("audience configuration requires the explicit service fragment", () => {
    expect(() => createServiceAuthenticator({ audience: "did:web:local.example", resolveAtprotoKey: async () => keys.did() })).toThrow();
  });
  test("expiry is rechecked after asynchronous identity resolution", async () => {
    let now = NOW;
    const verify = createServiceAuthenticator({ audience: AUDIENCE, now: () => now,
      resolveAtprotoKey: async () => { now += 61000; return keys.did(); } });
    await expect(verify(request(await token()), METHOD)).rejects.toThrow();
  });
  test("real signed JWT reaches local HTTP route; forged auth flag does not", async () => {
    installLocalServingFixture();
    const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: createApiHandler({ authenticate: verifier() }) });
    try {
      const endpoint = new URL(`/xrpc/${METHOD}?unauthenticated=false`, server.url);
      const anonymous = await (await fetch(endpoint)).json() as any;
      expect(anonymous.unauthenticated).toBe(true);
      expect(JSON.stringify(anonymous)).not.toContain(FIXTURE_DIDS.restricted);
      const jwt = await token();
      const authenticated = await fetch(endpoint, { headers: { authorization: `Bearer ${jwt}` } });
      expect(authenticated.status).toBe(200);
      const body = await authenticated.json() as any;
      expect(body.unauthenticated).toBe(false);
      expect(JSON.stringify(body)).toContain(FIXTURE_DIDS.restricted);
      expect(JSON.stringify(body)).not.toContain(FIXTURE_DIDS.hidden);
      expect((await fetch(endpoint, { headers: { authorization: `Bearer ${jwt}` } })).status).toBe(401);
    } finally { server.stop(true); }
  });
});
