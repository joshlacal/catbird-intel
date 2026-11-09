import { readFileSync } from "node:fs";
import { join } from "node:path";

export const DEFAULT_INTEL_VECTOR_STORE_DIR = "/Users/joshlacalamito/catbird-lake-local/intel-serving-v4";
export const INTEL_VECTOR_DIMS = 256;
const SIGNAL_BYTES = 24;
const NONE_COMMUNITY = 0xffff_ffff;

export interface IntelVectorManifest {
  version: string;
  generation: string;
  builtAtUtc?: string;
  wallSeconds?: number;
  modelId: string;
  socialModelId: string;
  dims: number;
  producerCount: number;
  vectorizedProducerCount: number;
  preTProducerCount: number;
  communityCentroidCount: number;
  signalRecordBytes: number;
  muPosts: number[];
  muFused: number[];
  muT1: number[];
}

export interface IntelCommunity {
  communityId: number;
  label: string;
  description: string;
  primaryLanguage: string;
  communityType: string;
  onboardingFit: string;
  macroCategory: string;
  sceneSlug: string;
  isS17Clean: boolean;
  sensitivity: string[];
  samplePostUris: string[];
  sampleAccountDids: string[];
  centPostsQ8B64: string;
  centPostsScaleFp16: number;
  centFusedQ8B64: string;
  centFusedScaleFp16: number;
  centT1Q8B64: string;
  centT1ScaleFp16: number;
  hostileTargets: number[];
}

export interface IntelActorSignal {
  pid: number;
  flags: number;
  eligible: boolean;
  hiddenFromRecommendations: boolean;
  qualityScoreBps: number;
  rageBaitPermille: number;
  spamPermille: number;
  botPermille: number;
  originalSharePermille: number;
  topCommunityIds: number[];
  vecRow: number;
}

export interface QuantizedVector256 {
  modelId: string;
  dims: 256;
  scaleFp16: number;
  q8Bytes: { $bytes: string };
}

export interface WindowedCentroid {
  window: string;
  sampleCount: number;
  vector: QuantizedVector256;
}

export interface ActorHealthSignals {
  scorerVersion: string;
  isCalibrated: boolean;
  posts14d: number;
  originalSharePermille: number;
  replySharePermille: number;
  repostSharePermille: number;
  quoteSharePermille: number;
  rageBaitPermille: number;
  controversyPermille: number;
  spamPermille: number;
  botPermille: number;
  qualityScoreBps: number;
  topCommunityIds: number[];
}

export interface ActorIntelNode {
  did: string;
  hopDistance: number;
  mutualFollowCount?: number;
  mutualEngagerCount: number;
  socialAffinityPermille?: number;
  centroids?: WindowedCentroid[];
  socialVector?: QuantizedVector256;
  health?: ActorHealthSignals;
}

export interface IntelActorFixture {
  did: string;
  eligible?: boolean;
  hiddenFromRecommendations?: boolean;
  qualityScoreBps?: number;
  rageBaitPermille?: number;
  spamPermille?: number;
  botPermille?: number;
  originalSharePermille?: number;
  topCommunityIds?: number[];
  all?: readonly number[];
  preT?: readonly number[];
  t1?: readonly number[];
  vectorCount?: number;
}

export interface IntelVectorStoreFixtures {
  generation?: string;
  modelId?: string;
  socialModelId?: string;
  actors: IntelActorFixture[];
  communities?: Array<Partial<IntelCommunity> & Pick<IntelCommunity, "communityId">>;
}

type VectorWindow = "all" | "preT" | "t1";
interface PackedVectors { q8: Uint8Array; meta: DataView }

const floatScratch = new ArrayBuffer(4);
const floatScratchF32 = new Float32Array(floatScratch);
const floatScratchU32 = new Uint32Array(floatScratch);

export function fp16BitsToFloat32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * (fraction === 0 ? 0 : 2 ** -14 * (fraction / 1024));
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export function float32ToFp16Bits(value: number): number {
  floatScratchF32[0] = value;
  const word = floatScratchU32[0]!;
  const sign = (word >>> 16) & 0x8000;
  let exponent = ((word >>> 23) & 0xff) - 127 + 15;
  let fraction = word & 0x7f_ffff;
  if (((word >>> 23) & 0xff) === 0xff) return sign | (fraction === 0 ? 0x7c00 : 0x7e00);
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    fraction = (fraction | 0x80_0000) >>> (1 - exponent);
    return sign | ((fraction + 0x1000 + ((fraction >>> 13) & 1)) >>> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  fraction += 0x1000 + ((fraction >>> 13) & 1);
  if (fraction & 0x80_0000) {
    fraction = 0;
    exponent++;
    if (exponent >= 31) return sign | 0x7c00;
  }
  return sign | (exponent << 10) | (fraction >>> 13);
}

export function quantizeFloat32ToQ8(vector: ArrayLike<number>, modelId = "eg2-256-q8"): QuantizedVector256 {
  if (vector.length !== INTEL_VECTOR_DIMS) throw new Error(`Expected ${INTEL_VECTOR_DIMS} vector dimensions`);
  let normSquared = 0;
  let maxAbs = 0;
  for (let i = 0; i < INTEL_VECTOR_DIMS; i++) {
    const value = Number(vector[i]);
    if (!Number.isFinite(value)) throw new Error("Vector contains a non-finite component");
    normSquared += value * value;
    maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  const bytes = new Int8Array(INTEL_VECTOR_DIMS);
  if (normSquared === 0 || maxAbs === 0) {
    return { modelId, dims: INTEL_VECTOR_DIMS, scaleFp16: 0, q8Bytes: { $bytes: Buffer.from(bytes.buffer).toString("base64") } };
  }
  const scale = maxAbs / Math.sqrt(normSquared) / 127;
  for (let i = 0; i < INTEL_VECTOR_DIMS; i++) bytes[i] = Math.max(-127, Math.min(127, Math.round(Number(vector[i]) / maxAbs * 127)));
  return {
    modelId,
    dims: INTEL_VECTOR_DIMS,
    scaleFp16: float32ToFp16Bits(scale),
    q8Bytes: { $bytes: Buffer.from(bytes.buffer).toString("base64") },
  };
}

export function quantizedBytes(vector: QuantizedVector256): Uint8Array {
  if (vector.dims !== INTEL_VECTOR_DIMS || !Number.isInteger(vector.scaleFp16) || vector.scaleFp16 < 0 || vector.scaleFp16 > 0xffff) {
    throw new Error("Invalid quantized vector metadata");
  }
  const encoded = vector.q8Bytes?.$bytes;
  if (typeof encoded !== "string" || !/^[A-Za-z0-9+/]*={0,2}$/.test(encoded)) throw new Error("Invalid quantized vector bytes");
  const bytes = Buffer.from(encoded, "base64");
  if (bytes.byteLength !== INTEL_VECTOR_DIMS) throw new Error(`Expected ${INTEL_VECTOR_DIMS} quantized bytes`);
  return bytes;
}

export function dequantizeQ8(vector: QuantizedVector256): Float32Array {
  const bytes = quantizedBytes(vector);
  const scale = fp16BitsToFloat32(vector.scaleFp16);
  const result = new Float32Array(INTEL_VECTOR_DIMS);
  for (let i = 0; i < INTEL_VECTOR_DIMS; i++) result[i] = signedByte(bytes[i]!) * scale;
  return result;
}

export function cosineI8(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  aOffset = 0,
  bOffset = 0,
  dims = INTEL_VECTOR_DIMS,
): number {
  if (aOffset < 0 || bOffset < 0 || dims < 1 || aOffset + dims > a.length || bOffset + dims > b.length) {
    throw new Error("Quantized vector slice is out of bounds");
  }
  let dot = 0, aa = 0, bb = 0;
  for (let i = 0; i < dims; i++) {
    const av = signedByte(Number(a[aOffset + i]));
    const bv = signedByte(Number(b[bOffset + i]));
    dot += av * bv;
    aa += av * av;
    bb += bv * bv;
  }
  return aa === 0 || bb === 0 ? 0 : dot / Math.sqrt(aa * bb);
}

function signedByte(value: number): number {
  return value > 127 ? value - 256 : value;
}

function dataView(bytes: Uint8Array): DataView {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function normalizeCommunity(raw: Partial<IntelCommunity> & Pick<IntelCommunity, "communityId">): IntelCommunity {
  return {
    communityId: raw.communityId,
    label: raw.label ?? `Community #${raw.communityId}`,
    description: raw.description ?? "",
    primaryLanguage: raw.primaryLanguage ?? "en",
    communityType: raw.communityType ?? "organic",
    onboardingFit: raw.onboardingFit ?? "ok",
    macroCategory: raw.macroCategory ?? "",
    sceneSlug: raw.sceneSlug ?? "",
    isS17Clean: raw.isS17Clean ?? true,
    sensitivity: raw.sensitivity ?? [],
    samplePostUris: raw.samplePostUris ?? [],
    sampleAccountDids: raw.sampleAccountDids ?? [],
    centPostsQ8B64: raw.centPostsQ8B64 ?? Buffer.alloc(INTEL_VECTOR_DIMS).toString("base64"),
    centPostsScaleFp16: raw.centPostsScaleFp16 ?? 0,
    centFusedQ8B64: raw.centFusedQ8B64 ?? raw.centPostsQ8B64 ?? Buffer.alloc(INTEL_VECTOR_DIMS).toString("base64"),
    centFusedScaleFp16: raw.centFusedScaleFp16 ?? raw.centPostsScaleFp16 ?? 0,
    centT1Q8B64: raw.centT1Q8B64 ?? Buffer.alloc(INTEL_VECTOR_DIMS).toString("base64"),
    centT1ScaleFp16: raw.centT1ScaleFp16 ?? 0,
    hostileTargets: raw.hostileTargets ?? [],
  };
}

function quantizedFromParts(modelId: string, scaleFp16: number, bytes: Uint8Array): QuantizedVector256 {
  return {
    modelId,
    dims: INTEL_VECTOR_DIMS,
    scaleFp16,
    q8Bytes: { $bytes: Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString("base64") },
  };
}

export class IntelVectorStore {
  readonly didToPid: Map<string, number>;
  readonly pidToDid: string[];
  readonly communities: IntelCommunity[];
  readonly manifest: IntelVectorManifest;

  private readonly signals: DataView;
  private readonly vectors: Record<VectorWindow, PackedVectors>;
  private readonly communitiesById: Map<number, IntelCommunity>;
  private communityMemberPids: Map<number, number[]> | null = null;
  private sidecarUnavailableUntil = 0;

  private constructor(
    manifest: IntelVectorManifest,
    dids: string[],
    signals: Uint8Array,
    vectors: Record<VectorWindow, PackedVectors>,
    communities: IntelCommunity[],
  ) {
    if (manifest.version !== "catbird.intel.vectors.v4" || manifest.dims !== INTEL_VECTOR_DIMS || manifest.signalRecordBytes !== SIGNAL_BYTES) {
      throw new Error("Unsupported Intel vector store manifest");
    }
    if (dids.length !== manifest.producerCount || signals.byteLength !== dids.length * SIGNAL_BYTES) {
      throw new Error("Intel vector store producer files do not match the manifest");
    }
    for (const packed of Object.values(vectors)) {
      if (packed.q8.byteLength !== manifest.vectorizedProducerCount * INTEL_VECTOR_DIMS || packed.meta.byteLength !== manifest.vectorizedProducerCount * 4) {
        throw new Error("Intel vector files do not match the manifest");
      }
    }
    this.manifest = manifest;
    this.pidToDid = dids;
    this.didToPid = new Map();
    for (let pid = 0; pid < dids.length; pid++) this.didToPid.set(dids[pid]!, pid);
    this.signals = dataView(signals);
    this.vectors = vectors;
    this.communities = communities;
    this.communitiesById = new Map();
    for (const community of communities) this.communitiesById.set(community.communityId, community);
  }

  static fromDirectory(directory = process.env.INTEL_VECTOR_STORE_DIR ?? DEFAULT_INTEL_VECTOR_STORE_DIR): IntelVectorStore {
    const manifest = JSON.parse(readFileSync(join(directory, "manifest.json"), "utf8")) as IntelVectorManifest;
    const didText = readFileSync(join(directory, "dids.txt"), "utf8");
    const dids = didText.endsWith("\n") ? didText.slice(0, -1).split("\n") : didText.split("\n");
    const load = (name: string): Uint8Array => readFileSync(join(directory, name));
    const vectors = {
      all: { q8: load("vec_all_i8.bin"), meta: dataView(load("vec_all_meta.bin")) },
      preT: { q8: load("vec_preT_i8.bin"), meta: dataView(load("vec_preT_meta.bin")) },
      t1: { q8: load("vec_t1_i8.bin"), meta: dataView(load("vec_t1_meta.bin")) },
    };
    const communities = (JSON.parse(readFileSync(join(directory, "communities.json"), "utf8")) as IntelCommunity[]).map(normalizeCommunity);
    if (communities.length !== manifest.communityCentroidCount) throw new Error("Community count does not match the manifest");
    return new IntelVectorStore(manifest, dids, load("signals.bin"), vectors, communities);
  }

  static fromFixtures(fixtures: IntelVectorStoreFixtures): IntelVectorStore {
    const dids = fixtures.actors.map((actor) => actor.did);
    if (new Set(dids).size !== dids.length) throw new Error("Fixture actor DIDs must be unique");
    const vectorActors = fixtures.actors.filter((actor) => actor.all || actor.preT || actor.t1);
    const rowByDid = new Map(vectorActors.map((actor, row) => [actor.did, row]));
    const signals = new Uint8Array(dids.length * SIGNAL_BYTES);
    const signalView = dataView(signals);
    const q8 = { all: new Uint8Array(vectorActors.length * INTEL_VECTOR_DIMS), preT: new Uint8Array(vectorActors.length * INTEL_VECTOR_DIMS), t1: new Uint8Array(vectorActors.length * INTEL_VECTOR_DIMS) };
    const metas = { all: new Uint8Array(vectorActors.length * 4), preT: new Uint8Array(vectorActors.length * 4), t1: new Uint8Array(vectorActors.length * 4) };
    const metaViews = { all: dataView(metas.all), preT: dataView(metas.preT), t1: dataView(metas.t1) };
    for (let pid = 0; pid < fixtures.actors.length; pid++) {
      const actor = fixtures.actors[pid]!;
      const row = rowByDid.get(actor.did) ?? -1;
      let flags = actor.eligible === false ? 0 : 1;
      if (actor.all) flags |= 2;
      if (actor.preT) flags |= 4;
      if (actor.t1) flags |= 8;
      if ((actor.topCommunityIds?.length ?? 0) > 0) flags |= 16;
      if (actor.hiddenFromRecommendations) flags |= 32;
      const offset = pid * SIGNAL_BYTES;
      signalView.setUint16(offset, flags, true);
      signalView.setUint16(offset + 2, actor.qualityScoreBps ?? 7000, true);
      signalView.setUint16(offset + 4, actor.rageBaitPermille ?? 0, true);
      signalView.setUint16(offset + 6, actor.spamPermille ?? 0, true);
      signalView.setUint16(offset + 8, actor.botPermille ?? 0, true);
      signalView.setUint16(offset + 10, actor.originalSharePermille ?? 700, true);
      signalView.setUint32(offset + 12, actor.topCommunityIds?.[0] ?? NONE_COMMUNITY, true);
      signalView.setUint32(offset + 16, actor.topCommunityIds?.[1] ?? NONE_COMMUNITY, true);
      signalView.setInt32(offset + 20, row, true);
      if (row < 0) continue;
      for (const window of ["all", "preT", "t1"] as const) {
        const source = actor[window];
        if (!source) continue;
        const quantized = quantizeFloat32ToQ8(source, window === "t1" ? fixtures.socialModelId ?? "s16-t1-256-q8" : fixtures.modelId ?? "eg2-256-q8");
        q8[window].set(quantizedBytes(quantized), row * INTEL_VECTOR_DIMS);
        metaViews[window].setUint16(row * 4, quantized.scaleFp16, true);
        metaViews[window].setUint16(row * 4 + 2, actor.vectorCount ?? 1, true);
      }
    }
    const zeroMean = Array(INTEL_VECTOR_DIMS).fill(0) as number[];
    const communities = (fixtures.communities ?? []).map(normalizeCommunity);
    const manifest: IntelVectorManifest = {
      version: "catbird.intel.vectors.v4",
      generation: fixtures.generation ?? "fixture-v4",
      modelId: fixtures.modelId ?? "eg2-256-q8",
      socialModelId: fixtures.socialModelId ?? "s16-t1-256-q8",
      dims: INTEL_VECTOR_DIMS,
      producerCount: dids.length,
      vectorizedProducerCount: vectorActors.length,
      preTProducerCount: fixtures.actors.filter((actor) => actor.preT).length,
      communityCentroidCount: communities.length,
      signalRecordBytes: SIGNAL_BYTES,
      muPosts: zeroMean,
      muFused: zeroMean,
      muT1: zeroMean,
    };
    return new IntelVectorStore(manifest, dids, signals, {
      all: { q8: q8.all, meta: metaViews.all },
      preT: { q8: q8.preT, meta: metaViews.preT },
      t1: { q8: q8.t1, meta: metaViews.t1 },
    }, communities);
  }

  signalForDid(did: string): IntelActorSignal | null {
    const pid = this.didToPid.get(did);
    return pid === undefined ? null : this.signalForPid(pid);
  }

  signalForPid(pid: number): IntelActorSignal {
    if (!Number.isInteger(pid) || pid < 0 || pid >= this.pidToDid.length) throw new RangeError("Producer ID is out of bounds");
    const offset = pid * SIGNAL_BYTES;
    const flags = this.signals.getUint16(offset, true);
    const communities = [this.signals.getUint32(offset + 12, true), this.signals.getUint32(offset + 16, true)].filter((id) => id !== NONE_COMMUNITY);
    return {
      pid,
      flags,
      eligible: (flags & 1) !== 0,
      hiddenFromRecommendations: (flags & 32) !== 0,
      qualityScoreBps: this.signals.getUint16(offset + 2, true),
      rageBaitPermille: this.signals.getUint16(offset + 4, true),
      spamPermille: this.signals.getUint16(offset + 6, true),
      botPermille: this.signals.getUint16(offset + 8, true),
      originalSharePermille: this.signals.getUint16(offset + 10, true),
      topCommunityIds: communities,
      vecRow: this.signals.getInt32(offset + 20, true),
    };
  }

  isHiddenFromRecommendations(did: string): boolean {
    return this.signalForDid(did)?.hiddenFromRecommendations ?? false;
  }

  vectorForDid(did: string, window: VectorWindow): QuantizedVector256 | null {
    const signal = this.signalForDid(did);
    if (!signal || signal.vecRow < 0 || !this.hasWindow(signal, window)) return null;
    return this.vectorForRow(signal.vecRow, window);
  }

  vectorCountForDid(did: string, window: VectorWindow): number {
    const signal = this.signalForDid(did);
    if (!signal || signal.vecRow < 0 || !this.hasWindow(signal, window)) return 0;
    return this.vectors[window].meta.getUint16(signal.vecRow * 4 + 2, true);
  }

  dequantizedVectorForDid(did: string, window: VectorWindow): Float32Array | null {
    const vector = this.vectorForDid(did, window);
    return vector ? dequantizeQ8(vector) : null;
  }

  community(id: number): IntelCommunity | null {
    return this.communitiesById.get(id) ?? null;
  }

  communityVector(community: IntelCommunity, kind: "posts" | "fused" | "t1" = "posts"): QuantizedVector256 {
    const modelId = kind === "t1" ? this.manifest.socialModelId : this.manifest.modelId;
    const scaleFp16 = kind === "posts" ? community.centPostsScaleFp16 :
      kind === "fused" ? community.centFusedScaleFp16 : community.centT1ScaleFp16;
    const encoded = kind === "posts" ? community.centPostsQ8B64 :
      kind === "fused" ? community.centFusedQ8B64 : community.centT1Q8B64;
    return { modelId, dims: INTEL_VECTOR_DIMS, scaleFp16, q8Bytes: { $bytes: encoded } };
  }

  matchingCommunities(query: string, limit = 8): IntelCommunity[] {
    const terms = tokenize(query);
    if (terms.length === 0) return [];
    return this.communities
      .map((community) => ({ community, score: lexicalScore(terms, community) }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score || a.community.communityId - b.community.communityId)
      .slice(0, limit)
      .map(({ community }) => community);
  }

  communityMembers(communityId: number, limit = 3000): string[] {
    this.ensureCommunityMemberIndex();
    return (this.communityMemberPids!.get(communityId) ?? [])
      .slice(0, limit)
      .map((pid) => this.pidToDid[pid]!);
  }

  actorNode(did: string, options: {
    hopDistance?: number;
    mutualFollowCount?: number;
    mutualEngagerCount?: number;
    socialAffinityPermille?: number;
    windows?: string[];
  } = {}): ActorIntelNode | null {
    const signal = this.signalForDid(did);
    if (!signal || signal.vecRow < 0) return null;
    const requested = options.windows ?? ["30d", "90d"];
    const centroids: WindowedCentroid[] = [];
    for (const window of requested) {
      const source: VectorWindow | null = window === "7d" || window === "30d" || window === "all" ? "all" :
        window === "90d" || window === "pre_t" || window === "preT" ? "preT" : null;
      if (!source) continue;
      const vector = this.vectorForDid(did, source);
      if (vector) centroids.push({ window, sampleCount: this.vectorCountForDid(did, source), vector });
    }
    const socialVector = this.vectorForDid(did, "t1") ?? undefined;
    const remainder = Math.max(0, 1000 - signal.originalSharePermille);
    return {
      did,
      hopDistance: Math.max(0, Math.min(3, options.hopDistance ?? 0)),
      ...(options.mutualFollowCount === undefined ? {} : { mutualFollowCount: options.mutualFollowCount }),
      mutualEngagerCount: options.mutualEngagerCount ?? 0,
      ...(options.socialAffinityPermille === undefined ? {} : { socialAffinityPermille: options.socialAffinityPermille }),
      ...(centroids.length ? { centroids } : {}),
      ...(socialVector ? { socialVector } : {}),
      health: {
        scorerVersion: this.manifest.version,
        isCalibrated: false,
        posts14d: this.vectorCountForDid(did, "all"),
        originalSharePermille: signal.originalSharePermille,
        replySharePermille: 0,
        repostSharePermille: remainder,
        quoteSharePermille: 0,
        rageBaitPermille: signal.rageBaitPermille,
        controversyPermille: signal.rageBaitPermille,
        spamPermille: signal.spamPermille,
        botPermille: signal.botPermille,
        qualityScoreBps: signal.qualityScoreBps,
        topCommunityIds: signal.topCommunityIds,
      },
    };
  }

  centroidForDids(dids: Iterable<string>, window: "all" | "preT", weights?: ReadonlyMap<string, number>): WindowedCentroid | null {
    const sum = new Float64Array(INTEL_VECTOR_DIMS);
    let sampleCount = 0;
    let totalWeight = 0;
    for (const did of dids) {
      const vector = this.dequantizedVectorForDid(did, window);
      if (!vector) continue;
      const weight = Math.max(0, weights?.get(did) ?? 1);
      if (weight === 0) continue;
      for (let i = 0; i < INTEL_VECTOR_DIMS; i++) sum[i] = sum[i]! + vector[i]! * weight;
      totalWeight += weight;
      sampleCount += this.vectorCountForDid(did, window);
    }
    if (totalWeight === 0) return null;
    for (let i = 0; i < INTEL_VECTOR_DIMS; i++) sum[i] = sum[i]! / totalWeight;
    return {
      window: window === "all" ? "30d" : "90d",
      sampleCount,
      vector: quantizeFloat32ToQ8(sum, this.manifest.modelId),
    };
  }

  async encodeQueryText(query: string): Promise<QuantizedVector256> {
    const trimmed = query.trim();
    if (!trimmed) return quantizeFloat32ToQ8(new Float32Array(INTEL_VECTOR_DIMS), this.manifest.modelId);
    const sidecar = await this.trySidecar(trimmed);
    if (sidecar) return sidecar;
    const terms = tokenize(trimmed);
    const matches = this.communities
      .map((community) => ({ community, score: lexicalScore(terms, community) }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score || a.community.communityId - b.community.communityId)
      .slice(0, 8);
    const sum = new Float64Array(INTEL_VECTOR_DIMS);
    for (const { community, score } of matches) {
      const vector = dequantizeQ8(this.communityVector(community));
      for (let i = 0; i < INTEL_VECTOR_DIMS; i++) sum[i] = sum[i]! + vector[i]! * score;
    }
    return quantizeFloat32ToQ8(sum, this.manifest.modelId);
  }

  private vectorForRow(row: number, window: VectorWindow): QuantizedVector256 {
    const packed = this.vectors[window];
    const offset = row * INTEL_VECTOR_DIMS;
    const model = window === "t1" ? this.manifest.socialModelId : this.manifest.modelId;
    return quantizedFromParts(model, packed.meta.getUint16(row * 4, true), packed.q8.subarray(offset, offset + INTEL_VECTOR_DIMS));
  }

  private hasWindow(signal: IntelActorSignal, window: VectorWindow): boolean {
    return (signal.flags & (window === "all" ? 2 : window === "preT" ? 4 : 8)) !== 0;
  }

  private ensureCommunityMemberIndex(): void {
    if (this.communityMemberPids) return;
    const index = new Map<number, number[]>();
    for (let pid = 0; pid < this.pidToDid.length; pid++) {
      const offset = pid * SIGNAL_BYTES;
      const flags = this.signals.getUint16(offset, true);
      if ((flags & 1) === 0 || (flags & 32) !== 0 || this.signals.getInt32(offset + 20, true) < 0) continue;
      const first = this.signals.getUint32(offset + 12, true);
      const second = this.signals.getUint32(offset + 16, true);
      if (first !== NONE_COMMUNITY) {
        const members = index.get(first);
        if (members) members.push(pid); else index.set(first, [pid]);
      }
      if (second !== NONE_COMMUNITY && second !== first) {
        const members = index.get(second);
        if (members) members.push(pid); else index.set(second, [pid]);
      }
    }
    for (const members of index.values()) {
      members.sort((a, b) =>
        this.signals.getUint16(b * SIGNAL_BYTES + 2, true) -
        this.signals.getUint16(a * SIGNAL_BYTES + 2, true) || a - b);
    }
    this.communityMemberPids = index;
  }

  private async trySidecar(text: string): Promise<QuantizedVector256 | null> {
    if (Date.now() < this.sidecarUnavailableUntil) return null;
    const url = process.env.EG2_SIDECAR_URL ?? "http://127.0.0.1:3211/embed";
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ texts: [text] }),
        signal: AbortSignal.timeout(400),
      });
      if (!response.ok) {
        this.sidecarUnavailableUntil = Date.now() + 30_000;
        return null;
      }
      const body = await response.json() as Record<string, unknown>;
      const candidate = (Array.isArray(body.embeddings) ? body.embeddings[0] : Array.isArray(body.vectors) ? body.vectors[0] : body) as Record<string, unknown> | undefined;
      if (!candidate || !Number.isInteger(candidate.scaleFp16) || typeof candidate.q8B64 !== "string") return null;
      const vector: QuantizedVector256 = {
        modelId: this.manifest.modelId,
        dims: INTEL_VECTOR_DIMS,
        scaleFp16: candidate.scaleFp16 as number,
        q8Bytes: { $bytes: candidate.q8B64 },
      };
      quantizedBytes(vector);
      return vector;
    } catch {
      this.sidecarUnavailableUntil = Date.now() + 30_000;
      return null;
    }
  }
}

function tokenize(value: string): string[] {
  return [...new Set(value.toLocaleLowerCase("en-US").match(/[\p{L}\p{N}]{2,}/gu) ?? [])];
}

function lexicalScore(terms: string[], community: IntelCommunity): number {
  if (terms.length === 0) return 0;
  const label = new Set(tokenize(`${community.label} ${community.sceneSlug.replaceAll("-", " ")}`));
  const all = new Set(tokenize(`${community.label} ${community.description} ${community.sceneSlug.replaceAll("-", " ")}`));
  let score = 0;
  for (const term of terms) {
    if (label.has(term)) score += 3;
    else if (all.has(term)) score += 1;
  }
  return score / Math.sqrt(Math.max(1, all.size) * terms.length);
}
