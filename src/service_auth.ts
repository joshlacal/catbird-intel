import { createHmac, timingSafeEqual } from "node:crypto";
import { verifySignature } from "@atproto/crypto";

/** https://atproto.com/specs/xrpc#inter-service-authentication-jwt */
export interface ServiceAuthOptions {
  audience: string;
  /** Trusted DID-document resolver; return the allowed #atproto key encoded as did:key. */
  resolveAtprotoKey: (issuer: string) => Promise<string>;
  /** Shared only by Nest and catbird-intel. Hex secrets are decoded before use. */
  internalBffSecret?: string;
  now?: () => number;
  maxNonces?: number;
}

export class AuthenticationError extends Error {}
export type ServiceAuthenticator = (request: Request, method: string) => Promise<string | null>;

function objectPart(part: string): Record<string, unknown> {
  if (!/^[A-Za-z0-9_-]+$/.test(part)) throw new AuthenticationError("Invalid service token");
  const value = JSON.parse(Buffer.from(part, "base64url").toString("utf8")) as unknown;
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new AuthenticationError("Invalid service token");
  return value as Record<string, unknown>;
}

function bffSecretBytes(value: string | undefined): Buffer | null {
  if (!value) return null;
  const bytes = value.length % 2 === 0 && /^[0-9a-f]+$/i.test(value) ? Buffer.from(value, "hex") : Buffer.from(value, "utf8");
  if (bytes.byteLength < 32) throw new Error("Internal BFF secret must contain at least 32 bytes");
  return bytes;
}

export function createServiceAuthenticator(options?: ServiceAuthOptions): ServiceAuthenticator {
  const usedJtis = new Map<string, number>();
  const internalBffSecret = bffSecretBytes(options?.internalBffSecret ?? process.env.INTEL_INTERNAL_BFF_SECRET);
  if (options && !/^did:[a-z]+:[^#\s]+#[A-Za-z0-9_-]+$/.test(options.audience)) {
    throw new Error("Service audience must contain a DID and explicit service fragment");
  }

  const acceptOnce = (issuer: string, jti: string, expires: number, now: number, maxNonces: number) => {
    for (const [nonce, exp] of usedJtis) if (exp <= now) usedJtis.delete(nonce);
    const nonce = `${issuer}:${jti}`;
    if (usedJtis.has(nonce) || usedJtis.size >= maxNonces) throw new Error("replay or capacity");
    usedJtis.set(nonce, expires);
  };

  return async (request, method) => {
    const authorization = request.headers.get("authorization");
    if (!authorization) return null;
    if (!options) throw new AuthenticationError("Authenticated serving is not configured");
    try {
      if (!authorization.startsWith("Bearer ") || authorization.length > 16384) throw new Error("format");
      const parts = authorization.slice(7).split(".");
      if (parts.length !== 3) throw new Error("parts");
      const [headerPart, bodyPart, signaturePart] = parts as [string, string, string];

      if (headerPart === "bff") {
        if (!internalBffSecret || !/^[A-Za-z0-9_-]+$/.test(signaturePart)) throw new Error("bff disabled");
        const supplied = Buffer.from(signaturePart, "base64url");
        const expected = createHmac("sha256", internalBffSecret).update(bodyPart).digest();
        if (supplied.byteLength !== expected.byteLength || !timingSafeEqual(supplied, expected)) throw new Error("signature");
        const body = objectPart(bodyPart);
        if (typeof body.iss !== "string" || !/^did:[a-z0-9]+:[^#\s]+$/i.test(body.iss)) throw new Error("issuer");
        if (body.aud !== options.audience || body.lxm !== method) throw new Error("audience or method");
        if (!Number.isSafeInteger(body.iat) || !Number.isSafeInteger(body.exp)) throw new Error("time");
        const issued = body.iat as number, expires = body.exp as number;
        const now = Math.floor((options.now?.() ?? Date.now()) / 1000);
        if (issued > now + 5 || expires <= now - 5 || expires <= issued || expires > issued + 60) throw new Error("lifetime");
        if (typeof body.jti !== "string" ||
            !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(body.jti)) {
          throw new Error("nonce");
        }
        acceptOnce(body.iss, body.jti, expires + 5, now, options.maxNonces ?? 10000);
        return body.iss;
      }

      const header = objectPart(headerPart);
      const body = objectPart(bodyPart);
      if (header.typ !== "JWT" || (header.alg !== "ES256" && header.alg !== "ES256K")) throw new Error("algorithm");
      if (header.kid !== undefined && header.kid !== "#atproto") throw new Error("key purpose");
      if (typeof body.iss !== "string" || !/^did:(plc|web):[^#\s]+$/.test(body.iss)) throw new Error("issuer");
      if (body.aud !== options.audience || body.lxm !== method) throw new Error("audience or method");
      const now = Math.floor((options.now?.() ?? Date.now()) / 1000);
      if (!Number.isSafeInteger(body.iat) || !Number.isSafeInteger(body.exp)) throw new Error("time");
      const issued = body.iat as number, expires = body.exp as number;
      if (issued > now || expires <= now || expires <= issued || expires - issued > 60) throw new Error("lifetime");
      if (typeof body.jti !== "string" || body.jti.length < 8 || body.jti.length > 256) throw new Error("nonce");
      if (!/^[A-Za-z0-9_-]+$/.test(signaturePart)) throw new Error("signature encoding");
      const signature = Buffer.from(signaturePart, "base64url");
      if (signature.length !== 64) throw new Error("signature length");
      const didKey = await options.resolveAtprotoKey(body.iss);
      if (!await verifySignature(didKey, Buffer.from(`${headerPart}.${bodyPart}`), signature, { jwtAlg: header.alg })) {
        throw new Error("signature");
      }
      if (expires <= Math.floor((options.now?.() ?? Date.now()) / 1000)) throw new Error("expired during resolution");
      acceptOnce(body.iss, body.jti, expires, now, options.maxNonces ?? 10000);
      return body.iss;
    } catch {
      throw new AuthenticationError("Invalid or expired service token");
    }
  };
}
