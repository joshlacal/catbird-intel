#!/usr/bin/env node
/**
 * Synthetic .jss segments for the CID sidecar tests and the decode benchmark.
 * Uses the decoder's own encodeJssSegment, so the bytes go through the exact
 * zstd/DAG-CBOR/XXH3 path real segments do.
 *
 *   node make_segments.mjs --decoder-dir DIR --spec spec.json --out ARCHIVE
 *   node make_segments.mjs --decoder-dir DIR --bench N --post-frac F --image-frac G --index I --out ARCHIVE
 *
 * spec.json: {"segments": [{"index": 3671, "group": "0000000e", "events": [...]}]}
 * Events are encodeJssSegment descriptors. Inside a record, {"$blob": CID,
 * "mimeType": M, "size": S} becomes a real blob ref (CID link) and
 * {"$legacyBlob": CID, "mimeType": M} becomes the legacy {cid, mimeType} form.
 * Output layout is the decoder's: ARCHIVE/segments/<group>/<seg>.jss plus
 * <seg>.jss.receipt.json.
 */
import { createHash } from 'node:crypto';
import * as fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1]]);
  return acc;
}, []));
const decoderDir = path.resolve(args['decoder-dir']);
const req = createRequire(path.join(decoderDir, 'package.json'));
const { encodeJssSegment } = await import(pathToFileURL(path.join(decoderDir, 'decode_segments.mjs')).href);
const { parseCid, cidForRawHash } = await import(pathToFileURL(req.resolve('@atproto/lex-data')).href);

function materialize(value) {
  if (Array.isArray(value)) return value.map(materialize);
  if (value && typeof value === 'object') {
    if (typeof value.$blob === 'string') {
      return { $type: 'blob', ref: parseCid(value.$blob), mimeType: value.mimeType ?? 'image/jpeg', size: value.size ?? 1000 };
    }
    if (typeof value.$legacyBlob === 'string') {
      return { cid: value.$legacyBlob, mimeType: value.mimeType ?? 'image/jpeg' };
    }
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, materialize(v)]));
  }
  return value;
}

async function writeSegment(out, group, index, events) {
  const { data, entry } = await encodeJssSegment(index, events.map((e) => ({ ...e, record: materialize(e.record) })));
  const dir = path.join(out, 'segments', group);
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(path.join(dir, entry.name), data);
  const receipt = {
    schema: 'catbird.jetstream.archive.v1', name: entry.name, checksum: entry.checksum,
    bytes: entry.sizeBytes, sha256: createHash('sha256').update(data).digest('hex'),
  };
  await fs.writeFile(path.join(dir, `${entry.name}.receipt.json`), JSON.stringify(receipt, null, 2));
  return { name: entry.name, bytes: entry.sizeBytes, events: events.length, sha256: receipt.sha256 };
}

// Deterministic pseudo-random stream (mulberry32) so benchmark inputs are reproducible.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const B32 = '234567abcdefghijklmnopqrstuvwxyz';
const WORDS = ('the a of to and in is it you that for on with this was are be at have not just like so but '
  + 'my what all can if about one out up time now day people more good new get think know really love today '
  + 'bluesky post art photo cat dog music game news politics science rust swift coffee morning night weekend').split(' ');

function benchEvents(n, postFrac, imageFrac, seed, startSeq) {
  const r = rng(seed);
  const pick = (arr) => arr[Math.floor(r() * arr.length)];
  const did = () => `did:plc:${Array.from({ length: 24 }, () => pick(B32)).join('')}`;
  const tid = () => Array.from({ length: 13 }, (_, i) => (i === 0 ? pick('234567') : pick(B32))).join('');
  // Real CIDv1 raw/sha256 strings (random digests), so they parse like upstream blob CIDs.
  const cid = () => cidForRawHash(createHash('sha256').update(String(r())).digest()).toString();
  const dids = Array.from({ length: Math.max(10, Math.floor(n / 40)) }, did);
  const events = [];
  for (let i = 0; i < n; i++) {
    const author = pick(dids);
    const u = r();
    const base = { seq: startSeq + i, kind: 'commit', operation: 'create', did: author, rkey: tid(), rev: tid() };
    if (u < postFrac) {
      const text = Array.from({ length: 4 + Math.floor(r() * 30) }, () => pick(WORDS)).join(' ');
      const record = { $type: 'app.bsky.feed.post', text, createdAt: '2026-10-20T12:00:00.000Z', langs: ['en'] };
      if (r() < 0.3) {
        const parent = { uri: `at://${pick(dids)}/app.bsky.feed.post/${tid()}`, cid: cid() };
        record.reply = { root: parent, parent };
      }
      if (r() < imageFrac) {
        const k = 1 + Math.floor(r() * 3);
        record.embed = {
          $type: 'app.bsky.embed.images',
          images: Array.from({ length: k }, () => ({
            alt: r() < 0.4 ? text.slice(0, 80) : '',
            image: { $blob: cid(), mimeType: 'image/jpeg', size: 200000 },
            aspectRatio: { width: 1200, height: 900 },
          })),
        };
      } else if (r() < 0.08) {
        record.embed = { $type: 'app.bsky.embed.external', external: {
          uri: 'https://example.com/a', title: text.slice(0, 40), description: text,
          thumb: { $blob: cid(), mimeType: 'image/jpeg', size: 50000 } } };
      }
      events.push({ ...base, collection: 'app.bsky.feed.post', record });
    } else if (u < postFrac + (1 - postFrac) * 0.7) {
      events.push({ ...base, collection: 'app.bsky.feed.like', record: {
        $type: 'app.bsky.feed.like', createdAt: '2026-10-20T12:00:00.000Z',
        subject: { uri: `at://${pick(dids)}/app.bsky.feed.post/${tid()}`, cid: cid() } } });
    } else {
      events.push({ ...base, collection: 'app.bsky.graph.follow', record: {
        $type: 'app.bsky.graph.follow', subject: pick(dids), createdAt: '2026-10-20T12:00:00.000Z' } });
    }
  }
  return events;
}

const out = path.resolve(args.out);
const written = [];
if (args.spec) {
  const spec = JSON.parse(await fs.readFile(args.spec, 'utf8'));
  for (const seg of spec.segments) {
    written.push(await writeSegment(out, seg.group ?? '0000000e', seg.index, seg.events));
  }
} else if (args.bench) {
  const n = Number(args.bench);
  const index = Number(args.index ?? 3671);
  written.push(await writeSegment(out, args.group ?? '0000000e', index,
    benchEvents(n, Number(args['post-frac'] ?? 0.3), Number(args['image-frac'] ?? 0.25), index + 1, Number(args['start-seq'] ?? 1))));
} else {
  throw new Error('give --spec or --bench');
}
process.stdout.write(JSON.stringify(written) + '\n');
