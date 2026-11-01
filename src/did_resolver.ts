import { formatDidKey, multibaseToBytes, parseMultikey } from "@atproto/crypto";

export interface DidKeyResolverOptions {
  plcDirectoryUrl?: string;
  ttlMs?: number;
  maxEntries?: number;
  timeoutMs?: number;
  staticKeys?: Record<string, string>;
  fetchImpl?: (input: string | URL | Request, init?: RequestInit) => Promise<Response>;
}

interface VerificationMethod {
  id?: string;
  type?: string;
  controller?: string;
  publicKeyMultibase?: string;
}

interface DidDocument {
  id?: string;
  verificationMethod?: VerificationMethod[];
}

function didWebToDocumentUrl(did: string): string {
  const rest = did.slice("did:web:".length);
  if (!rest || rest.includes("/") || rest.includes("?") || rest.includes("#")) {
    throw new Error("Invalid did:web identifier");
  }
  const segments = rest.split(":").map((part) => decodeURIComponent(part));
  const host = segments[0];
  if (!host) throw new Error("Missing did:web host");
  if (segments.length === 1) {
    return `https://${host}/.well-known/did.json`;
  }
  return `https://${host}/${segments.slice(1).join("/")}/did.json`;
}

function extractAtprotoDidKey(did: string, doc: DidDocument): string {
  const methods = Array.isArray(doc.verificationMethod) ? doc.verificationMethod : [];
  const vm = methods.find(
    (entry) => entry && (entry.id === "#atproto" || entry.id === `${did}#atproto` || entry.id?.endsWith("#atproto")),
  );
  if (!vm || typeof vm.publicKeyMultibase !== "string" || vm.publicKeyMultibase.length < 16) {
    throw new Error(`DID document for ${did} lacks #atproto verification key`);
  }
  if (vm.type === "Multikey") {
    const parsed = parseMultikey(vm.publicKeyMultibase);
    return formatDidKey(parsed.jwtAlg, parsed.keyBytes);
  }
  if (vm.type === "EcdsaSecp256k1VerificationKey2019") {
    return formatDidKey("ES256K", multibaseToBytes(vm.publicKeyMultibase));
  }
  if (vm.type === "EcdsaSecp256r1VerificationKey2019") {
    return formatDidKey("ES256", multibaseToBytes(vm.publicKeyMultibase));
  }
  // Fallback for Multikey-encoded values lacking explicit type string
  const parsed = parseMultikey(vm.publicKeyMultibase);
  return formatDidKey(parsed.jwtAlg, parsed.keyBytes);
}

export function createDidKeyResolver(options: DidKeyResolverOptions = {}): (did: string) => Promise<string> {
  const plcBase = (options.plcDirectoryUrl ?? process.env.PLC_DIRECTORY_URL ?? "https://plc.directory").replace(/\/+$/, "");
  const ttlMs = options.ttlMs ?? 600_000;
  const maxEntries = options.maxEntries ?? 20_000;
  const timeoutMs = options.timeoutMs ?? 4_000;
  const fetchFn = options.fetchImpl ?? fetch;

  const envStaticKeys: Record<string, string> = process.env.INTEL_STATIC_DID_KEYS_JSON
    ? (JSON.parse(process.env.INTEL_STATIC_DID_KEYS_JSON) as Record<string, string>)
    : {};
  const staticMap = new Map<string, string>(Object.entries({ ...envStaticKeys, ...(options.staticKeys ?? {}) }));
  const cache = new Map<string, { didKey: string; expiresAt: number }>();

  return async (did: string): Promise<string> => {
    const staticHit = staticMap.get(did);
    if (staticHit) return staticHit;

    const now = Date.now();
    const cached = cache.get(did);
    if (cached && cached.expiresAt > now) {
      return cached.didKey;
    }

    let docUrl: string;
    if (did.startsWith("did:plc:")) {
      docUrl = `${plcBase}/${encodeURIComponent(did)}`;
    } else if (did.startsWith("did:web:")) {
      docUrl = didWebToDocumentUrl(did);
    } else {
      throw new Error(`Unsupported DID method: ${did}`);
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let response: Response;
    try {
      response = await fetchFn(docUrl, {
        headers: { accept: "application/did+ld+json, application/json" },
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
    if (!response.ok) {
      throw new Error(`DID resolution failed (${response.status}) for ${did}`);
    }
    const doc = (await response.json()) as DidDocument;
    const didKey = extractAtprotoDidKey(did, doc);

    if (cache.size >= maxEntries) {
      for (const [k, v] of cache) {
        if (v.expiresAt <= now) cache.delete(k);
      }
      if (cache.size >= maxEntries) {
        const oldest = cache.keys().next().value;
        if (oldest) cache.delete(oldest);
      }
    }
    cache.set(did, { didKey, expiresAt: now + ttlMs });
    return didKey;
  };
}
