// Pure mapping from Jetstream v2 raw events to ClickHouse rows. No I/O here so
// it can be unit-tested against fixtures.

import { scorePostHealth } from "./health_scorer.ts";
import { revVersion, rkeyDate } from "./tid.ts";

export type Row = Record<string, unknown>;

export type TableName =
  | "posts"
  | "engagements"
  | "follows"
  | "blocks"
  | "lists"
  | "listitems"
  | "starterpacks"
  | "feedgens"
  | "profiles"
  | "visibility"
  | "account_status"
  | "identities"
  | "other_records"
  | "deletions";

export interface RowOut {
  table: TableName;
  row: Row;
}

/** Minimal structural view of a raw v2 event (see @bsky/jetstream RawEvent). */
export interface RawEventLike {
  kind: string;
  did: string;
  seq: number;
  time: string;
  commit?: {
    operation: "create" | "update" | "delete";
    collection: string;
    rkey: string;
    rev: string;
    cid?: string;
    record?: unknown;
  };
  identity?: { handle?: string };
  account?: { active: boolean; status?: string };
}

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const str = (v: unknown, max = 4096): string => (typeof v === "string" ? v.slice(0, max) : "");
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

/** ClickHouse DateTime64(3) literal in UTC: "YYYY-MM-DD hh:mm:ss.sss". */
export function chTime(ms: number): string {
  return new Date(ms).toISOString().replace("T", " ").replace("Z", "");
}

/** Parse a record datetime, clamping garbage to the arrival time. */
function recordTime(v: unknown, arrivalMs: number): string {
  if (typeof v === "string") {
    const ms = Date.parse(v);
    if (Number.isFinite(ms) && ms >= Date.UTC(2022, 0, 1) && ms <= arrivalMs + 86_400_000) return chTime(ms);
  }
  return chTime(arrivalMs);
}

export interface ParsedUri {
  did: string;
  collection: string;
  rkey: string;
}

export function parseAtUri(uri: unknown): ParsedUri | null {
  if (typeof uri !== "string" || !uri.startsWith("at://")) return null;
  const parts = uri.slice(5).split("/");
  if (parts.length !== 3 || !parts[0] || !parts[1] || !parts[2]) return null;
  return { did: parts[0], collection: parts[1], rkey: parts[2] };
}

function selfLabels(record: Obj): string[] {
  const labels = record.labels;
  if (!isObj(labels)) return [];
  return arr(labels.values)
    .map((v) => (isObj(v) ? str(v.val, 128) : ""))
    .filter(Boolean)
    .slice(0, 16);
}

/** Pull embed-derived fields out of a post record. */
function postEmbed(embed: unknown) {
  const out = { embedType: "", imageCount: 0, hasAlt: 0, hasVideo: 0, externalUri: "", quoteUri: "" };
  if (!isObj(embed)) return out;
  const type = str(embed.$type, 128);
  out.embedType = type.replace(/^app\.bsky\.embed\./, "");
  const media = (m: unknown) => {
    if (!isObj(m)) return;
    const t = str(m.$type, 128);
    if (t === "app.bsky.embed.images") {
      const images = arr(m.images);
      out.imageCount = Math.min(images.length, 255);
      out.hasAlt = images.some((i) => isObj(i) && str(i.alt).trim().length > 0) ? 1 : 0;
    } else if (t === "app.bsky.embed.video") {
      out.hasVideo = 1;
      out.hasAlt = str(m.alt).trim().length > 0 ? 1 : 0;
    } else if (t === "app.bsky.embed.external" && isObj(m.external)) {
      out.externalUri = str(m.external.uri, 2048);
    }
  };
  if (type === "app.bsky.embed.record" && isObj(embed.record)) {
    out.quoteUri = str(embed.record.uri, 512);
  } else if (type === "app.bsky.embed.recordWithMedia") {
    const rec = isObj(embed.record) && isObj(embed.record.record) ? embed.record.record : null;
    if (rec) out.quoteUri = str(rec.uri, 512);
    media(embed.media);
  } else {
    media(embed);
  }
  return out;
}

function postFacets(record: Obj) {
  const links: string[] = [];
  const tags: string[] = [];
  const mentions: string[] = [];
  for (const facet of arr(record.facets)) {
    if (!isObj(facet)) continue;
    for (const feature of arr(facet.features)) {
      if (!isObj(feature)) continue;
      const t = str(feature.$type, 128);
      if (t === "app.bsky.richtext.facet#link") links.push(str(feature.uri, 2048));
      else if (t === "app.bsky.richtext.facet#tag") tags.push(str(feature.tag, 128).toLowerCase());
      else if (t === "app.bsky.richtext.facet#mention") mentions.push(str(feature.did, 256));
    }
  }
  return { links: links.slice(0, 32), tags: tags.slice(0, 32), mentions: mentions.slice(0, 64) };
}

const COLLECTION_TABLE: Record<string, TableName> = {
  "app.bsky.feed.post": "posts",
  "app.bsky.feed.like": "engagements",
  "app.bsky.feed.repost": "engagements",
  "app.bsky.graph.follow": "follows",
  "app.bsky.graph.block": "blocks",
  "app.bsky.graph.list": "lists",
  "app.bsky.graph.listitem": "listitems",
  "app.bsky.graph.starterpack": "starterpacks",
  "app.bsky.feed.generator": "feedgens",
  "app.bsky.actor.profile": "profiles",
  "app.bsky.actor.contentVisibilityDeclaration": "visibility",
};

/**
 * Map one raw event to zero or more rows. `arrivalMs` is the event's wire
 * `time` (the clock for every windowed feature, per the ClickHouse Bluesky
 * findings that record createdAt values are unreliable).
 */
export function eventToRows(ev: RawEventLike): RowOut[] {
  const arrivalMs = Number.isFinite(Date.parse(ev.time)) ? Date.parse(ev.time) : Date.now();
  const indexed_at = chTime(arrivalMs);
  const base = { did: ev.did, seq: ev.seq, indexed_at };

  if (ev.kind === "identity" && ev.identity) {
    return [{ table: "identities", row: { ...base, handle: str(ev.identity.handle, 256) } }];
  }
  if (ev.kind === "account" && ev.account) {
    return [{
      table: "account_status",
      row: { ...base, active: ev.account.active ? 1 : 0, status: str(ev.account.status, 64) },
    }];
  }
  if (ev.kind !== "commit" || !ev.commit) return [];

  const c = ev.commit;
  const table = COLLECTION_TABLE[c.collection] ?? "other_records";
  const isDelete = c.operation === "delete";
  const keyed = {
    ...base,
    rkey: c.rkey,
    rev: revVersion(c.rev),
    is_deleted: isDelete ? 1 : 0,
  };
  const dated = { ...keyed, rkey_date: rkeyDate(c.rkey, arrivalMs) };
  const out: RowOut[] = [];

  if (isDelete) {
    out.push({
      table: "deletions",
      row: { collection: c.collection, did: ev.did, rkey: c.rkey, rev: keyed.rev, seq: ev.seq, indexed_at },
    });
  }

  const record: Obj = !isDelete && isObj(c.record) ? c.record : {};

  switch (table) {
    case "posts": {
      if (isDelete) {
        out.push({ table, row: dated });
        break;
      }
      const reply = isObj(record.reply) ? record.reply : {};
      const embed = postEmbed(record.embed);
      const facets = postFacets(record);
      const text = str(record.text, 3000);
      const replyParent = isObj(reply.parent) ? str(reply.parent.uri, 512) : "";
      const health = scorePostHealth(text, {
        isQuote: embed.quoteUri.length > 0,
        isReply: replyParent.length > 0,
        hashtagCount: facets.tags.length,
        linkCount: facets.links.length,
      });
      out.push({
        table,
        row: {
          ...dated,
          created_at: recordTime(record.createdAt, arrivalMs),
          cid: str(c.cid, 128),
          text,
          langs: arr(record.langs).map((l) => str(l, 16)).filter(Boolean).slice(0, 4),
          reply_root: isObj(reply.root) ? str(reply.root.uri, 512) : "",
          reply_parent: replyParent,
          quote_uri: embed.quoteUri,
          embed_type: embed.embedType,
          image_count: embed.imageCount,
          has_alt_text: embed.hasAlt,
          has_video: embed.hasVideo,
          external_uri: embed.externalUri,
          facet_links: facets.links,
          facet_tags: facets.tags,
          facet_mentions: facets.mentions,
          self_labels: selfLabels(record),
          ...health,
        },
      });
      break;
    }
    case "engagements": {
      const kind = c.collection === "app.bsky.feed.like" ? "like" : "repost";
      const subject = isObj(record.subject) ? parseAtUri(record.subject.uri) : null;
      if (!isDelete && !subject) break;
      out.push({
        table,
        row: {
          ...dated,
          kind,
          subject_did: subject?.did ?? "",
          subject_collection: subject?.collection ?? "",
          subject_rkey: subject?.rkey ?? "",
          via_uri: isObj(record.via) ? str(record.via.uri, 512) : "",
        },
      });
      break;
    }
    case "follows":
    case "blocks": {
      const subject = str(record.subject, 256);
      if (!isDelete && !subject.startsWith("did:")) break;
      out.push({ table, row: { ...dated, subject } });
      break;
    }
    case "lists":
      out.push({
        table,
        row: {
          ...keyed,
          purpose: str(record.purpose, 128).replace(/^app\.bsky\.graph\.defs#/, ""),
          name: str(record.name, 256),
          description: str(record.description, 3000),
        },
      });
      break;
    case "listitems":
      out.push({ table, row: { ...dated, list_uri: str(record.list, 512), subject: str(record.subject, 256) } });
      break;
    case "starterpacks":
      out.push({
        table,
        row: {
          ...keyed,
          name: str(record.name, 256),
          description: str(record.description, 3000),
          list_uri: str(record.list, 512),
          feed_uris: arr(record.feeds).map((f) => (isObj(f) ? str(f.uri, 512) : "")).filter(Boolean).slice(0, 8),
        },
      });
      break;
    case "feedgens":
      out.push({
        table,
        row: {
          ...keyed,
          service_did: str(record.did, 256),
          display_name: str(record.displayName, 256),
          description: str(record.description, 3000),
        },
      });
      break;
    case "profiles": {
      if (c.rkey !== "self") break;
      out.push({
        table,
        row: {
          did: ev.did,
          rev: keyed.rev,
          is_deleted: keyed.is_deleted,
          seq: ev.seq,
          indexed_at,
          display_name: str(record.displayName, 256),
          description: str(record.description, 3000),
          self_labels: selfLabels(record),
          pinned_post: isObj(record.pinnedPost) ? str(record.pinnedPost.uri, 512) : "",
        },
      });
      break;
    }
    case "visibility": {
      if (c.rkey !== "self") break;
      out.push({
        table,
        row: {
          did: ev.did,
          rev: keyed.rev,
          is_deleted: keyed.is_deleted,
          seq: ev.seq,
          indexed_at,
          hide_from_algorithmic: record.hideFromAlgorithmicRecommendations === true ? 1 : 0,
        },
      });
      break;
    }
    case "other_records":
      out.push({
        table,
        row: { ...keyed, collection: c.collection, record: isDelete ? "" : JSON.stringify(record).slice(0, 20_000) },
      });
      break;
  }
  return out;
}
