#!/usr/bin/env node
/**
 * Generate a verified catbird.jetstream.archive.v1 .receipt.json for a local
 * sealed .jss segment produced by self-hosted Jetstream v2.
 *
 * Usage: node make_receipt.mjs <path-to-seg_XXXXXXXXXX.jss>
 */
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { readSegmentHeader, verifyMetadata } from './archive.mjs';

const file = process.argv[2];
if (!file) {
  console.error('Usage: node make_receipt.mjs <path-to-seg.jss>');
  process.exit(2);
}

const name = path.basename(file);
const index = Number.parseInt(name.slice(4, -4), 36);
const { entry } = await readSegmentHeader(file, name, index);
await verifyMetadata(file, entry);

const hash = createHash('sha256');
for await (const chunk of createReadStream(file, { highWaterMark: 1024 * 1024 })) {
  hash.update(chunk);
}
const sha256 = hash.digest('hex');

const receipt = {
  schema: 'catbird.jetstream.archive.v1',
  name: entry.name,
  index: entry.index,
  checksum: entry.checksum,
  bytes: entry.sizeBytes,
  sha256,
  first_seq: entry.firstSeq,
  last_seq: entry.lastSeq,
  first_time: entry.firstTime,
  last_time: entry.lastTime,
  generation_origin: 'gce_selfhost_v0.4.4',
  verified_at: new Date().toISOString(),
};

const receiptPath = `${file}.receipt.json`;
await fs.writeFile(receiptPath, JSON.stringify(receipt, null, 2) + '\n', { mode: 0o644 });
console.log(JSON.stringify(receipt));
