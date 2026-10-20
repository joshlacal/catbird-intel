import { verifySignature } from "@atproto/crypto";

/** https://atproto.com/specs/xrpc#inter-service-authentication-jwt */
export interface ServiceAuthOptions {
  audience: string;
  /** Trusted DID-document resolver; return the allowed #atproto key encoded as did:key. */
  resolveAtprotoKey: (issuer: string) => Promise<string>;
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

export function createServiceAuthenticator(options?: ServiceAuthOptions): ServiceAuthenticator {
  const usedNonces = new Map<string, number>();
  if (options && !/^did:[a-z]+:[^#\s]+#[A-Za-z0-9_-]+$/.test(options.audience)) {
    throw new Error("Service audience must contain a DID and explicit service fragment");
  }
  return async (request, method) => {
    const authorization = request.headers.get("authorization");
    if (!authorization) return null;
    if (!options) throw new AuthenticationError("Authenticated serving is not configured");
    try {
      if (!authorization.startsWith("Bearer ") || authorization.length > 16384) throw new Error("format");
      const parts = authorization.slice(7).split(".");
      if (parts.length !== 3) throw new Error("parts");
      const [headerPart, bodyPart, signaturePart] = parts as [string, string, string];
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
      for (const [nonce, exp] of usedNonces) if (exp <= now) usedNonces.delete(nonce);
      const nonce = `${body.iss}:${body.jti}`;
      if (usedNonces.has(nonce) || usedNonces.size >= (options.maxNonces ?? 10000)) throw new Error("replay or capacity");
      usedNonces.set(nonce, expires);
      return body.iss;
    } catch {
      throw new AuthenticationError("Invalid or expired service token");
    }
  };
}
