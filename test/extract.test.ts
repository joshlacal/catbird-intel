import { describe, expect, test } from "bun:test";
import { eventToRows, parseAtUri } from "../src/extract.ts";
import { resumeCursor } from "../src/cursor.ts";
import { revVersion, rkeyDate, tidToBigInt } from "../src/tid.ts";

const DID = "did:plc:abcdefghijklmnopqrstuvwx";
const TIME = "2026-10-19T12:00:00.000Z";
const ARRIVAL = Date.parse(TIME);

function commit(collection: string, operation: "create" | "update" | "delete", record?: unknown, rkey = "3m3kfx3fqpk2a") {
  return {
    kind: "commit",
    did: DID,
    seq: 42,
    time: TIME,
    commit: { operation, collection, rkey, rev: "3m3kfx3fqpk2b", cid: "bafyrei", record },
  };
}

describe("tid", () => {
  test("decodes a TID and orders revisions", () => {
    expect(tidToBigInt("3m3kfx3fqpk2a")).not.toBeNull();
    expect(tidToBigInt("self")).toBeNull();
    expect(BigInt(revVersion("3m3kfx3fqpk2b")) > BigInt(revVersion("3m3kfx3fqpk2a"))).toBe(true);
    expect(revVersion(undefined)).toBe("0");
  });

  test("rkey date falls back to arrival for non-TID keys and garbage TIDs", () => {
    expect(rkeyDate("self", ARRIVAL)).toBe("2026-10-19");
    // 2222222222222 decodes to 1970 -> implausible -> arrival
    expect(rkeyDate("2222222222222", ARRIVAL)).toBe("2026-10-19");
    const d = rkeyDate("3m3kfx3fqpk2a", ARRIVAL);
    expect(d).toMatch(/^\d{4}-\d{2}-\d{2}$/);
  });
});

describe("eventToRows", () => {
  test("post with reply, quote+media, facets, labels", () => {
    const rows = eventToRows(commit("app.bsky.feed.post", "create", {
      $type: "app.bsky.feed.post",
      text: "rust macros #rustlang",
      createdAt: "2026-10-19T11:59:59.000Z",
      langs: ["en"],
      reply: { root: { uri: "at://did:plc:r/app.bsky.feed.post/1" }, parent: { uri: "at://did:plc:p/app.bsky.feed.post/2" } },
      embed: {
        $type: "app.bsky.embed.recordWithMedia",
        record: { record: { uri: "at://did:plc:q/app.bsky.feed.post/3" } },
        media: { $type: "app.bsky.embed.images", images: [
          { alt: "a crab", image: { $type: "blob", ref: { $link: "bafkreidtt757okdivxau7bl5jgwtmbaq62s7s67cd63greeaqbiq33msae" }, mimeType: "image/jpeg", size: 1 } },
          { alt: "", image: { cid: "../../etc/passwd", mimeType: "image/png" } },
        ] },
      },
      facets: [{ features: [{ $type: "app.bsky.richtext.facet#tag", tag: "RustLang" }, { $type: "app.bsky.richtext.facet#link", uri: "https://doc.rust-lang.org" }] }],
      labels: { $type: "com.atproto.label.defs#selfLabels", values: [{ val: "!no-unauthenticated" }] },
    }));
    expect(rows).toHaveLength(1);
    const r = rows[0]!.row;
    expect(rows[0]!.table).toBe("posts");
    expect(r.reply_parent).toBe("at://did:plc:p/app.bsky.feed.post/2");
    expect(r.quote_uri).toBe("at://did:plc:q/app.bsky.feed.post/3");
    expect(r.image_count).toBe(2);
    expect(r.has_alt_text).toBe(1);
    expect(r.image_cids).toEqual(["bafkreidtt757okdivxau7bl5jgwtmbaq62s7s67cd63greeaqbiq33msae"]);
    expect(r.image_mimes).toEqual(["image/jpeg"]);
    expect(r.facet_tags).toEqual(["rustlang"]);
    expect(r.facet_links).toEqual(["https://doc.rust-lang.org"]);
    expect(r.self_labels).toEqual(["!no-unauthenticated"]);
    expect(r.is_deleted).toBe(0);
  });

  test("legacy blob form and plain image posts keep their CIDs", () => {
    const rows = eventToRows(commit("app.bsky.feed.post", "create", {
      text: "wip",
      createdAt: TIME,
      embed: { $type: "app.bsky.embed.images", images: [{ alt: "", image: { cid: "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG", mimeType: "image/png" } }] },
    }));
    expect(rows[0]!.row.image_cids).toEqual(["QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"]);
    expect(rows[0]!.row.image_mimes).toEqual(["image/png"]);
    const plain = eventToRows(commit("app.bsky.feed.post", "create", { text: "no media", createdAt: TIME }));
    expect(plain[0]!.row.image_cids).toEqual([]);
  });

  test("delete writes a tombstone row in the same partition plus a ledger row", () => {
    const create = eventToRows(commit("app.bsky.feed.like", "create", { subject: { uri: "at://did:plc:x/app.bsky.feed.post/9" } }));
    const del = eventToRows(commit("app.bsky.feed.like", "delete"));
    expect(del.map((d) => d.table).sort()).toEqual(["deletions", "engagements"]);
    const tomb = del.find((d) => d.table === "engagements")!.row;
    expect(tomb.is_deleted).toBe(1);
    expect(tomb.rkey_date).toBe(create[0]!.row.rkey_date);
    expect(tomb.kind).toBe("like");
  });

  test("likes without a valid subject are skipped; follows require a DID", () => {
    expect(eventToRows(commit("app.bsky.feed.like", "create", { subject: { uri: "nope" } }))).toEqual([]);
    expect(eventToRows(commit("app.bsky.graph.follow", "create", { subject: "nope" }))).toEqual([]);
    const f = eventToRows(commit("app.bsky.graph.follow", "create", { subject: "did:plc:target" }));
    expect(f[0]!.row.subject).toBe("did:plc:target");
  });

  test("visibility declaration and missing-field default", () => {
    const yes = eventToRows(commit("app.bsky.actor.contentVisibilityDeclaration", "create", { hideFromAlgorithmicRecommendations: true }, "self"));
    const no = eventToRows(commit("app.bsky.actor.contentVisibilityDeclaration", "create", {}, "self"));
    expect(yes[0]!.row.hide_from_algorithmic).toBe(1);
    expect(no[0]!.row.hide_from_algorithmic).toBe(0);
  });

  test("account and identity events", () => {
    const acct = eventToRows({ kind: "account", did: DID, seq: 7, time: TIME, account: { active: false, status: "deleted" } });
    expect(acct[0]).toEqual({ table: "account_status", row: { did: DID, seq: 7, indexed_at: "2026-10-19 12:00:00.000", active: 0, status: "deleted" } });
    const id = eventToRows({ kind: "identity", did: DID, seq: 8, time: TIME, identity: { handle: "josh.catbird.blue" } });
    expect(id[0]!.row.handle).toBe("josh.catbird.blue");
  });

  test("unknown collections go to other_records; sync events are ignored", () => {
    const o = eventToRows(commit("app.bsky.feed.threadgate", "create", { allow: [] }));
    expect(o[0]!.table).toBe("other_records");
    expect(eventToRows({ kind: "sync", did: DID, seq: 1, time: TIME })).toEqual([]);
  });

  test("garbage createdAt is clamped to arrival", () => {
    const rows = eventToRows(commit("app.bsky.feed.post", "create", { text: "x", createdAt: "1970-01-01T00:00:00Z" }));
    expect(rows[0]!.row.created_at).toBe("2026-10-19 12:00:00.000");
  });
});

test("parseAtUri", () => {
  expect(parseAtUri("at://did:plc:a/app.bsky.feed.post/xyz")).toEqual({ did: "did:plc:a", collection: "app.bsky.feed.post", rkey: "xyz" });
  expect(parseAtUri("https://x")).toBeNull();
});

test("resumeCursor uses seq on the same service and time elsewhere", () => {
  const saved = { service: "east", seq: 100, timeUs: 1_800_000_000_000_000 };
  expect(resumeCursor(saved, "east")).toBe(100);
  expect(resumeCursor(saved, "west")).toBe(1_800_000_000_000_000 - 30_000_000);
  expect(resumeCursor(null, "east")).toBeUndefined();
});
