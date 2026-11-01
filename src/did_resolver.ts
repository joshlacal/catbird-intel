import { lookup } from "node:dns/promises";
import { isIP } from "node:net";
import { formatDidKey, multibaseToBytes, parseMultikey } from "@atproto/crypto";

const MAX_DID_DOC_BYTES = 65_536;
const DISALLOWED_TLD_SUFFIXES = [
  ".localhost",
  ".local",
  ".internal",
  ".arpa",
  ".corp",
  ".lan",
];

export interface DidKeyResolverOptions {
  plcDirectoryUrl?: string;
  ttlMs?: number;
  maxEntries?: number;
  timeoutMs?: number;
  maxResponseBytes?: number;
  staticKeys?: Record<string, string>;
  fetchImpl?: (input: string | URL | Request, init?: RequestInit) => Promise<Response>;
  dnsLookupImpl?: (hostname: string) => Promise<Array<{ address: string; family: number }>>;
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

function isPublicIpv4Octets(a: number, b: number, c: number, d: number): boolean {
  if ([a, b, c, d].some((n) => !Number.isInteger(n) || n < 0 || n > 255)) return false;
  if (a === 0) return false; // 0.0.0.0/8
  if (a === 10) return false; // 10.0.0.0/8 RFC1918
  if (a === 100 && b >= 64 && b <= 127) return false; // 100.64.0.0/10 CGNAT
  if (a === 127) return false; // 127.0.0.0/8 loopback
  if (a === 169 && b === 254) return false; // 169.254.0.0/16 link-local / metadata
  if (a === 172 && b >= 16 && b <= 31) return false; // 172.16.0.0/12 RFC1918
  if (a === 192 && b === 0 && (c === 0 || c === 2)) return false; // 192.0.0.0/24, 192.0.2.0/24
  if (a === 192 && b === 88 && c === 99) return false; // 192.88.99.0/24 6to4 relay
  if (a === 192 && b === 168) return false; // 192.168.0.0/16 RFC1918
  if (a === 198 && (b === 18 || b === 19)) return false; // 198.18.0.0/15 benchmark
  if (a === 198 && b === 51 && c === 100) return false; // 198.51.100.0/24 TEST-NET-2
  if (a === 203 && b === 0 && c === 113) return false; // 203.0.113.0/24 TEST-NET-3
  if (a >= 224) return false; // 224.0.0.0/4 multicast + 240.0.0.0/4 reserved
  return true;
}

function expandIpv6Hextets(rawIp: string): number[] | null {
  const zoneIdx = rawIp.indexOf("%");
  const clean = (zoneIdx >= 0 ? rawIp.slice(0, zoneIdx) : rawIp).toLowerCase();
  let v6Part = clean;
  let v4Tail: number[] | null = null;

  if (v6Part.includes(".")) {
    const lastColon = v6Part.lastIndexOf(":");
    if (lastColon < 0) return null;
    const v4Str = v6Part.slice(lastColon + 1);
    const parts = v4Str.split(".");
    if (parts.length !== 4) return null;
    const nums = parts.map((p) => (/^\d+$/.test(p) ? Number(p) : -1));
    if (nums.some((n) => n < 0 || n > 255)) return null;
    v4Tail = [((nums[0]! << 8) | nums[1]!) & 0xffff, ((nums[2]! << 8) | nums[3]!) & 0xffff];
    v6Part = v6Part.slice(0, lastColon);
    if (v6Part.endsWith(":")) v6Part += "0";
    else v6Part = `${v6Part}:0`;
  }

  const dblParts = v6Part.split("::");
  if (dblParts.length > 2) return null;
  const parseSide = (s: string): number[] | null => {
    if (!s) return [];
    const tokens = s.split(":");
    const out: number[] = [];
    for (const tok of tokens) {
      if (!/^[0-9a-f]{1,4}$/.test(tok)) return null;
      out.push(Number.parseInt(tok, 16));
    }
    return out;
  };

  let hextets: number[];
  const targetLen = v4Tail ? 6 : 8;
  if (dblParts.length === 2) {
    // If we appended ":0" for v4Tail onto a trailing "::", strip the synthetic "0"
    let rightRaw = dblParts[1]!;
    if (v4Tail && rightRaw === "0") rightRaw = "";
    else if (v4Tail && rightRaw.endsWith(":0")) rightRaw = rightRaw.slice(0, -2);
    const left = parseSide(dblParts[0]!);
    const right = parseSide(rightRaw);
    if (!left || !right || left.length + right.length >= targetLen) return null;
    const zeros = new Array<number>(targetLen - left.length - right.length).fill(0);
    hextets = [...left, ...zeros, ...right];
  } else {
    let singleRaw = dblParts[0]!;
    if (v4Tail && singleRaw.endsWith(":0")) singleRaw = singleRaw.slice(0, -2);
    const parsed = parseSide(singleRaw);
    if (!parsed || parsed.length !== targetLen) return null;
    hextets = parsed;
  }

  if (v4Tail) hextets.push(v4Tail[0]!, v4Tail[1]!);
  return hextets.length === 8 ? hextets : null;
}

export function isPublicIpAddress(ip: string): boolean {
  const family = isIP(ip);
  if (family === 4) {
    const parts = ip.split(".").map((p) => Number(p));
    return parts.length === 4 && isPublicIpv4Octets(parts[0]!, parts[1]!, parts[2]!, parts[3]!);
  }
  if (family === 6) {
    const h = expandIpv6Hextets(ip);
    if (!h) return false;
    const [h0, h1, h2, h3, h4, h5, h6, h7] = h as [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ];
    // Unspecified ::/128 and loopback ::1/128
    if (h0 === 0 && h1 === 0 && h2 === 0 && h3 === 0 && h4 === 0 && h5 === 0 && h6 === 0 && h7 <= 1) {
      return false;
    }
    // IPv4-mapped ::ffff:0:0/96 or IPv4-compatible ::/96 or NAT64 64:ff9b::/96
    const isV4Mapped = h0 === 0 && h1 === 0 && h2 === 0 && h3 === 0 && (h4 === 0 && (h5 === 0xffff || h5 === 0));
    const isNat64 = h0 === 0x0064 && h1 === 0xff9b && (h2 === 0 || h2 === 1) && h3 === 0 && h4 === 0 && h5 === 0;
    if (isV4Mapped || isNat64) {
      return isPublicIpv4Octets((h6 >> 8) & 0xff, h6 & 0xff, (h7 >> 8) & 0xff, h7 & 0xff);
    }
    if ((h0 & 0xfe00) === 0xfc00) return false; // fc00::/7 unique local
    if ((h0 & 0xffc0) === 0xfe80) return false; // fe80::/10 link-local
    if ((h0 & 0xff00) === 0xff00) return false; // ff00::/8 multicast
    if (h0 === 0x2001 && h1 === 0x0db8) return false; // 2001:db8::/32 documentation
    if (h0 === 0x0100 && h1 === 0 && h2 === 0 && h3 === 0) return false; // 100::/64 discard
    return true;
  }
  return false;
}

function didWebToDocumentTarget(did: string): { url: string; hostname: string } {
  const rest = did.slice("did:web:".length);
  // ATProto did:web only permits domain-level lowercase hostnames (no ports, percent-encoding, or path segments)
  if (
    !rest ||
    rest.length > 253 ||
    rest.includes(":") ||
    rest.includes("%") ||
    rest.includes("/") ||
    rest.includes("\\") ||
    rest.includes("?") ||
    rest.includes("#") ||
    rest.includes("@")
  ) {
    throw new Error("Invalid did:web identifier");
  }
  if (
    !/^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$/.test(rest)
  ) {
    throw new Error("Invalid did:web hostname");
  }
  if (isIP(rest) !== 0 || /^\d+(?:\.\d+)*$/.test(rest)) {
    throw new Error("IP literals are not permitted in did:web");
  }
  if (rest === "localhost" || DISALLOWED_TLD_SUFFIXES.some((suffix) => rest.endsWith(suffix))) {
    throw new Error("Non-public domain is not permitted in did:web");
  }
  const parsed = new URL(`https://${rest}/.well-known/did.json`);
  if (
    parsed.protocol !== "https:" ||
    parsed.hostname !== rest ||
    parsed.port !== "" ||
    parsed.username !== "" ||
    parsed.password !== ""
  ) {
    throw new Error("Invalid did:web document URL");
  }
  return { url: parsed.toString(), hostname: rest };
}

async function readBoundedJson<T>(response: Response, maxBytes: number): Promise<T> {
  const declaredLen = response.headers.get("content-length");
  if (declaredLen !== null) {
    const parsedLen = Number(declaredLen);
    if (!Number.isFinite(parsedLen) || parsedLen < 0 || parsedLen > maxBytes) {
      throw new Error(`DID document exceeds ${maxBytes} bytes`);
    }
  }
  if (!response.body) {
    throw new Error("Empty DID document body");
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        total += value.byteLength;
        if (total > maxBytes) {
          await reader.cancel();
          throw new Error(`DID document exceeds ${maxBytes} bytes`);
        }
        chunks.push(value);
      }
    }
  } finally {
    reader.releaseLock();
  }
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const text = new TextDecoder("utf-8", { fatal: true }).decode(merged);
  return JSON.parse(text) as T;
}

function extractAtprotoDidKey(did: string, doc: DidDocument): string {
  if (!doc || typeof doc !== "object" || doc.id !== did) {
    throw new Error(`DID document subject mismatch for ${did}`);
  }
  const methods = Array.isArray(doc.verificationMethod) ? doc.verificationMethod : [];
  const vm = methods.find(
    (entry) =>
      entry &&
      (entry.id === "#atproto" || entry.id === `${did}#atproto`) &&
      (entry.controller === undefined || entry.controller === did),
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
  const parsed = parseMultikey(vm.publicKeyMultibase);
  return formatDidKey(parsed.jwtAlg, parsed.keyBytes);
}

export function createDidKeyResolver(options: DidKeyResolverOptions = {}): (did: string) => Promise<string> {
  const plcBase = (options.plcDirectoryUrl ?? process.env.PLC_DIRECTORY_URL ?? "https://plc.directory").replace(/\/+$/, "");
  const ttlMs = options.ttlMs ?? 600_000;
  const maxEntries = options.maxEntries ?? 20_000;
  const timeoutMs = options.timeoutMs ?? 4_000;
  const maxResponseBytes = options.maxResponseBytes ?? MAX_DID_DOC_BYTES;
  const fetchFn = options.fetchImpl ?? fetch;
  const dnsLookupFn =
    options.dnsLookupImpl ??
    (options.fetchImpl
      ? null
      : (hostname: string) => lookup(hostname, { all: true, verbatim: true }));

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
      if (!/^did:plc:[a-z0-9]{16,64}$/.test(did)) {
        throw new Error(`Invalid did:plc identifier: ${did}`);
      }
      docUrl = `${plcBase}/${encodeURIComponent(did)}`;
    } else if (did.startsWith("did:web:")) {
      const target = didWebToDocumentTarget(did);
      if (dnsLookupFn) {
        const records = await dnsLookupFn(target.hostname);
        if (!records.length || records.some((rec) => !isPublicIpAddress(rec.address))) {
          throw new Error(`SSRF blocked non-public DNS resolution for ${did}`);
        }
      }
      docUrl = target.url;
    } else {
      throw new Error(`Unsupported DID method: ${did}`);
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let response: Response;
    try {
      response = await fetchFn(docUrl, {
        headers: { accept: "application/did+ld+json, application/json" },
        redirect: "error",
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
    if (!response.ok) {
      throw new Error(`DID resolution failed (${response.status}) for ${did}`);
    }
    const doc = await readBoundedJson<DidDocument>(response, maxResponseBytes);
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
