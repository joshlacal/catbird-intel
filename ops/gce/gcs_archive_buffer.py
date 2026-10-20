#!/usr/bin/env python3
"""Immutable, bounded local Jetstream -> GCS spool. Never starts the collector.

Requires google-cloud-storage and google-crc32c on the VM (ADC/metadata auth).
SDK contract checked against cloud.google.com/python/docs/reference/storage/
latest/google.cloud.storage.blob.Blob: generation-match=0, checksum='crc32c',
crc32c_checksum_value; list_blobs versions and soft_deleted are separate scans.

Source epoch is an ADMISSION namespace, not proof of original Jetstream epoch.
make_receipt validates metadata XXH3 and whole-file SHA256, not block CRCs.
Budget counts every observed object generation cumulatively, including deleted
and soft-deleted generations. A dedicated bucket/single admission writer is
required: bucket byte quotas are application-enforced, not a GCS native quota.
"""
from __future__ import annotations
import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time
import uuid

TIB = 1024 ** 4
RESERVE = 32768  # deterministic receipt + release marker + metadata allowance
NAME = re.compile(r'^seg_([0-9a-z]+)\.jss$')


class Stop(RuntimeError):
    def __init__(self, reason, code=2):
        super().__init__(reason)
        self.code = code


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def atomic(path, data):
    temp = path.with_name(path.name + '.pending')
    with temp.open('wb') as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def hashes(path, crc_factory=None):
    if crc_factory is None:
        import google_crc32c
        crc_factory = google_crc32c.Checksum
    crc, md5, sha = crc_factory(), hashlib.md5(), hashlib.sha256()
    size = 0
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            size += len(chunk)
            crc.update(chunk)
            md5.update(chunk)
            sha.update(chunk)
    return {'bytes': size, 'sha256': sha.hexdigest(),
            'crc32c': base64.b64encode(crc.digest()).decode(),
            'md5_hash': base64.b64encode(md5.digest()).decode()}


def header(path):
    if path.is_symlink() or not path.is_file():
        raise Stop('source_not_regular_file:' + str(path))
    with path.open('rb') as src:
        data = src.read(256)
    if len(data) < 256 or data[:4] != b'jss0':
        raise Stop('invalid_segment_header:' + str(path))
    checksum = struct.unpack_from('<Q', data, 4)[0]
    if not checksum:
        return None  # active, unsealed file
    if struct.unpack_from('<H', data, 12)[0] != 1:
        raise Stop('unsupported_segment_version')
    footer = struct.unpack_from('<Q', data, 58)[0]
    if not 256 <= footer <= path.stat().st_size:
        raise Stop('invalid_footer_offset')
    return {'checksum': f'{checksum:016x}', 'min_seq': struct.unpack_from('<Q', data, 26)[0],
            'max_seq': struct.unpack_from('<Q', data, 34)[0],
            'min_witnessed_at': struct.unpack_from('<q', data, 42)[0],
            'max_witnessed_at': struct.unpack_from('<q', data, 50)[0]}


def local_segments(directory):
    found = []
    for path in directory.glob('seg_*.jss'):
        match = NAME.fullmatch(path.name)
        if not match:
            raise Stop('invalid_segment_name:' + path.name)
        found.append((int(match[1], 36), path))
    return sorted(found)


class Journal:
    def __init__(self, work):
        self.path = work / 'journal.jsonl'
        self.accounted, self.pairs, self.released = {}, {}, set()
        if self.path.exists():
            with self.path.open('rb') as stream:
                for line in stream:
                    if not line.endswith(b'\n'):
                        raise Stop('torn_journal_preserved_requires_recovery')
                    self.apply(json.loads(line))

    def apply(self, row):
        if row['event'] == 'account':
            key = (row['object'], str(row['generation']))
            if key in self.accounted and self.accounted[key] != row['bytes']:
                raise Stop('cloud_generation_size_changed')
            self.accounted[key] = row['bytes']
        elif row['event'] == 'durable_pair':
            old = self.pairs.get(row['key'])
            if old and old['payload']['generation'] != row['pair']['payload']['generation']:
                self.released.discard(row['key'])
            self.pairs[row['key']] = row['pair']
        elif row['event'] == 'local_release':
            self.released.add(row['key'])
        else:
            raise Stop('unknown_journal_event')

    def record(self, row):
        with self.path.open('ab') as out:
            out.write(encode(row))
            out.flush()
            os.fsync(out.fileno())
        self.apply(row)

    @property
    def admitted_bytes(self):
        return sum(self.accounted.values())

    def account(self, blob):
        key = (blob.name, str(blob.generation))
        if key not in self.accounted:
            self.record({'event': 'account', 'object': blob.name,
                         'generation': str(blob.generation), 'bytes': int(blob.size)})


def verify_blob(blob, expected):
    if blob is None:
        raise Stop('cloud_object_missing')
    if int(blob.size) != expected['bytes'] or blob.crc32c != expected['crc32c'] or blob.md5_hash != expected['md5_hash']:
        raise Stop('cloud_checksum_or_size_conflict:' + blob.name)
    if (blob.metadata or {}).get('sha256') != expected['sha256']:
        raise Stop('cloud_sha_metadata_conflict:' + blob.name)
    return blob


class Buffer:
    def __init__(self, client, args, verifier=None, hasher=hashes):
        self.client, self.args, self.hasher = client, args, hasher
        self.bucket = client.bucket(args.bucket)
        self.work = Path(args.work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.work)
        self.prefix = f'sources/{args.source_id}/{args.source_epoch}/segments/'
        self.verifier = verifier or self.node_verify
        self.cloud_bytes = 0
        self.started = time.monotonic()

    def time_check(self):
        if time.monotonic() - self.started >= self.args.run_hours * 3600:
            raise Stop('run_deadline_reached', 3)

    def node_verify(self, staged):
        result = subprocess.run(['node', self.args.receipt_script, str(staged)],
                                cwd=str(Path(self.args.receipt_script).parent), capture_output=True,
                                text=True, check=True, timeout=300)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def inventory(self, mutate=True):
        # A generation can appear in only one logical accounting slot even if
        # listings overlap. Deleted observations never reduce the admission budget.
        objects = {}
        for options in ({'versions': True}, {'soft_deleted': True}):
            for blob in self.client.list_blobs(self.bucket, **options):
                self.time_check()
                objects[(blob.name, str(blob.generation))] = blob
        self.cloud_bytes = sum(int(blob.size) for blob in objects.values())
        if mutate:
            for blob in objects.values():
                self.journal.account(blob)
        cumulative = max(self.cloud_bytes, self.journal.admitted_bytes)
        if cumulative > self.args.max_bytes:
            raise Stop('cumulative_cloud_capacity_exceeded', 4)
        return objects

    def admit(self, size):
        self.time_check()
        if self.journal.admitted_bytes + size > self.args.max_bytes:
            raise Stop('capacity_admission_blocked', 4)

    def upload(self, path, name, expected, metadata):
        existing = self.bucket.get_blob(name)
        if existing is not None:
            blob = verify_blob(existing, expected)
            self.journal.account(blob)
            return blob
        self.admit(expected['bytes'])
        blob = self.bucket.blob(name)
        blob.metadata = {**metadata, 'sha256': expected['sha256']}
        try:
            blob.upload_from_filename(str(path), if_generation_match=0, checksum='crc32c',
                                      crc32c_checksum_value=expected['crc32c'], timeout=300)
        except Exception:
            # Retry/unknown outcome may have created exactly these bytes. Never
            # overwrite a conflicting name, even after a transport failure.
            existing = self.bucket.get_blob(name)
            if existing is None:
                raise
            blob = verify_blob(existing, expected)
        else:
            blob = self.bucket.get_blob(name, generation=blob.generation)
            verify_blob(blob, expected)
        self.journal.account(blob)
        return blob

    def pair_key(self, index, sha):
        return f'{self.args.source_id}/{self.args.source_epoch}/{index}/{sha}'

    def pair_from_receipt(self, receipt, receipt_blob, receipt_hashes):
        name = receipt.get('name', '')
        match = NAME.fullmatch(name)
        if receipt.get('schema') != 'catbird.jetstream.archive.v1' or not match:
            raise Stop('invalid_cloud_receipt')
        index = int(match[1], 36)
        digest = receipt['sha256']
        if not re.fullmatch('[0-9a-f]{64}', digest) or receipt['index'] != index:
            raise Stop('invalid_cloud_receipt_index_or_hash')
        expected_name = self.prefix + f'{index // 256:08x}/{name}/{digest}.jss'
        pointer = receipt['gcs']
        if (receipt['source_id'], receipt['source_epoch']) != (self.args.source_id, self.args.source_epoch) or pointer['bucket'] != self.args.bucket or pointer['object'] != expected_name:
            raise Stop('cloud_receipt_namespace_mismatch')
        if receipt_blob.name != expected_name + f".g{pointer['generation']}.receipt.json":
            raise Stop('cloud_receipt_name_mismatch')
        expected = {k: pointer[k] for k in ['bytes', 'sha256', 'crc32c', 'md5_hash']}
        if expected['bytes'] != receipt['bytes'] or expected['sha256'] != digest:
            raise Stop('cloud_receipt_payload_mismatch')
        payload = self.bucket.get_blob(pointer['object'], generation=pointer['generation'])
        verify_blob(payload, expected)
        return {'name': name, 'index': index, 'sha256': digest, 'payload': pointer,
                'receipt': {'bucket': self.args.bucket, 'object': receipt_blob.name,
                            'generation': str(receipt_blob.generation), **receipt_hashes}}

    def reconcile(self, objects):
        known = {(p['receipt']['object'], p['receipt']['generation']) for p in self.journal.pairs.values()}
        for (_, generation), blob in sorted(objects.items()):
            if not blob.name.startswith(self.prefix) or (blob.metadata or {}).get('kind') != 'receipt' or (blob.name, generation) in known:
                continue
            # Soft-deleted receipts aren't available to ordinary generation GET;
            # leave their bytes accounted, never use them to authorize pruning.
            current = self.bucket.get_blob(blob.name, generation=generation)
            if current is None:
                continue
            if int(current.size) > RESERVE:
                raise Stop('oversized_cloud_receipt')
            raw = current.download_as_bytes(if_generation_match=int(generation), checksum='crc32c')
            if hashlib.sha256(raw).hexdigest() != (current.metadata or {}).get('sha256'):
                raise Stop('remote_receipt_sha_mismatch')
            receipt_hashes = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                              'crc32c': current.crc32c, 'md5_hash': current.md5_hash}
            receipt = json.loads(raw)
            pair = self.pair_from_receipt(receipt, current, receipt_hashes)
            self.journal.record({'event': 'durable_pair', 'key': self.pair_key(pair['index'], pair['sha256']), 'pair': pair})

    def offload(self, index, source):
        self.admit(source.stat().st_size + RESERVE)
        stage = Path(tempfile.mkdtemp(prefix=f'stage-{index}-', dir=self.work))
        staged = stage / source.name
        os.link(source, staged)  # pins original inode across atomic compaction replacement
        details = header(staged)
        if details is None:
            raise Stop('attempted_unsealed_upload')
        verified = self.verifier(staged)
        expected = self.hasher(staged)
        if verified.get('sha256') != expected['sha256'] or verified.get('bytes') != expected['bytes'] or verified.get('checksum') != details['checksum']:
            raise Stop('local_verifier_disagreement')
        name = self.prefix + f"{index // 256:08x}/{source.name}/{expected['sha256']}.jss"
        metadata = {'source_id': self.args.source_id, 'source_epoch': self.args.source_epoch,
                    'segment_name': source.name, 'index': str(index)}
        payload = self.upload(staged, name, expected, {**metadata, 'kind': 'payload'})
        if self.hasher(staged) != expected:
            raise Stop('source_changed_during_upload_preserved:' + str(stage))
        pointer = {'bucket': self.args.bucket, 'object': name, 'generation': str(payload.generation), **expected}
        receipt = {'schema': 'catbird.jetstream.archive.v1', 'name': source.name, 'index': index,
                   'bytes': expected['bytes'], 'sha256': expected['sha256'], **details,
                   'first_seq': details['min_seq'], 'last_seq': details['max_seq'],
                   'source_id': self.args.source_id, 'source_epoch': self.args.source_epoch,
                   'epoch_semantics': 'buffer_admission_namespace_not_original_source_epoch',
                   'generation_origin': 'gce_gcs_buffer_admission', 'gcs': pointer,
                   'verification': {'header_footer_xxh3': 'verified', 'whole_file_sha256': 'verified',
                                    'block_crc_validation': 'unperformed'}}
        receipt_path = stage / 'upload.receipt.json'
        atomic(receipt_path, encode(receipt))
        receipt_hashes = self.hasher(receipt_path)
        receipt_name = name + f'.g{payload.generation}.receipt.json'
        receipt_blob = self.upload(receipt_path, receipt_name, receipt_hashes,
                                   {**metadata, 'kind': 'receipt', 'payload_object': name,
                                    'payload_generation': str(payload.generation)})
        pair = self.pair_from_receipt(receipt, receipt_blob, receipt_hashes)
        pair['local_stat'] = self.fingerprint(staged)
        key = self.pair_key(index, expected['sha256'])
        self.journal.record({'event': 'durable_pair', 'key': key, 'pair': pair})
        # Only remove explicitly owned staging files after both remote objects verify.
        for path in [staged, receipt_path, Path(str(staged) + '.receipt.json')]:
            if path.exists():
                path.unlink()
        stage.rmdir()
        return key

    @staticmethod
    def fingerprint(path):
        info = path.stat()
        return {'dev': info.st_dev, 'ino': info.st_ino, 'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns}

    def cleanup_stages(self):
        for stage in self.work.glob('stage-*'):
            if stage.is_symlink() or not stage.is_dir():
                raise Stop('unexpected_stage_path_preserved:' + str(stage))
            candidates = [p for p in stage.iterdir() if NAME.fullmatch(p.name)]
            if len(candidates) != 1:
                continue
            staged = candidates[0]
            index = int(NAME.fullmatch(staged.name)[1], 36)
            digest = self.hasher(staged)['sha256']
            key = self.pair_key(index, digest)
            pair = self.journal.pairs.get(key)
            if pair is None:
                continue  # failed/uncommitted bytes remain preserved
            if any(path.name not in {staged.name, staged.name + '.receipt.json', 'upload.receipt.json'} for path in stage.iterdir()):
                raise Stop('unexpected_stage_files_preserved:' + str(stage))
            for part in ['payload', 'receipt']:
                pointer = pair[part]
                verify_blob(self.bucket.get_blob(pointer['object'], generation=pointer['generation']), pointer)
            for path in [staged, Path(str(staged) + '.receipt.json'), stage / 'upload.receipt.json']:
                if path.exists():
                    path.unlink()
            stage.rmdir()

    def scan_pending(self):
        segments = local_segments(Path(self.args.segments_dir))
        known = {(p['name'], json.dumps(p.get('local_stat'), sort_keys=True)) for key, p in self.journal.pairs.items() if key not in self.journal.released}
        pending = []
        for index, path in segments:
            details = header(path)
            if details is None:
                if path != segments[-1][1]:
                    raise Stop('nonterminal_unsealed_segment')
                continue
            # Cache only a generation that previously passed full hashes and the
            # cloud pair commit. Atomic source replacements invalidate inode/mtime.
            if (path.name, json.dumps(self.fingerprint(path), sort_keys=True)) not in known:
                pending.append((index, path))
        return segments, pending

    def status(self, reason, pending, current_index=None):
        value = {'reason': reason, 'updated_at_unix': time.time(), 'pending_sealed': pending,
                 'current_index': current_index, 'local_segments': len(local_segments(Path(self.args.segments_dir))),
                 'cloud_inventory_bytes': self.cloud_bytes, 'cumulative_admitted_bytes': self.journal.admitted_bytes,
                 'max_bytes': self.args.max_bytes, 'durable_pairs': len(self.journal.pairs),
                 'local_releases': len(self.journal.released), 'source_epoch': self.args.source_epoch,
                 'epoch_semantics': 'admission_namespace_only'}
        atomic(self.work / 'status.json', encode(value))
        return value

    def release(self, key, pair):
        if key in self.journal.released:
            return
        self.cleanup_stages()
        marker = {'schema': 'catbird.gcs.local-release.v1', 'payload_object': pair['payload']['object'],
                  'payload_generation': pair['payload']['generation'],
                  'receipt_object': pair['receipt']['object'], 'receipt_generation': pair['receipt']['generation'],
                  'sha256': pair['sha256']}
        marker_path = self.work / ('release-' + hashlib.sha256(key.encode()).hexdigest() + '.json')
        atomic(marker_path, encode(marker))
        expected = self.hasher(marker_path)
        self.upload(marker_path, pair['receipt']['object'] + '.local-release.json', expected,
                    {'kind': 'local_release', 'source_id': self.args.source_id, 'source_epoch': self.args.source_epoch})
        self.journal.record({'event': 'local_release', 'key': key})
        marker_path.unlink()

    def prune(self):
        segments = local_segments(Path(self.args.segments_dir))
        protected = {path.name for _, path in segments[-3:]}
        current = {path.name: path for _, path in segments}
        for key, pair in list(self.journal.pairs.items()):
            if key in self.journal.released or pair['name'] in protected:
                continue
            # Both immutable generations must still exist before removing a local copy.
            for part in ['payload', 'receipt']:
                pointer = pair[part]
                verify_blob(self.bucket.get_blob(pointer['object'], generation=pointer['generation']), pointer)
            source = current.get(pair['name'])
            if source and source.exists():
                if self.hasher(source)['sha256'] != pair['sha256']:
                    # Older generation was replaced; do not touch new path. Its
                    # old staging pin has already been released after verification.
                    self.release(key, pair)
                    continue
                quarantine = self.work / ('prune-' + uuid.uuid4().hex + '-' + source.name)
                os.rename(source, quarantine)
                # Atomic rename makes TOCTOU conservative: never delete bytes that
                # changed between the hash and rename; preserve them for recovery.
                if self.hasher(quarantine)['sha256'] != pair['sha256']:
                    raise Stop('source_replaced_during_prune_preserved:' + str(quarantine))
                quarantine.unlink()
            self.release(key, pair)

    def cycle(self, dry_run=False):
        self.status('inventory', -1)
        objects = self.inventory(mutate=not dry_run)
        if not dry_run:
            self.reconcile(objects)
            self.cleanup_stages()
        _, pending = self.scan_pending()
        self.status('dry_run' if dry_run else 'uploading', len(pending))
        if not dry_run:
            for position, (index, path) in enumerate(pending[:self.args.max_segments]):
                self.time_check()
                self.status('uploading', len(pending) - position, index)
                key = self.offload(index, path)
                self.cleanup_stages()
                self.prune()
                print(json.dumps({'event': 'durable_pair', 'key': key,
                                  'pending_at_scan_remaining': len(pending) - position - 1,
                                  'cumulative_admitted_bytes': self.journal.admitted_bytes}), flush=True)
            self.prune()
        _, fresh_pending = self.scan_pending()
        return self.status('dry_run' if dry_run else 'ready', len(fresh_pending))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bucket', required=True)
    p.add_argument('--source-id', default='gce-selfhost-001')
    p.add_argument('--source-epoch', required=True)
    p.add_argument('--segments-dir', default='/data/jetstream/segments')
    p.add_argument('--work-dir', default='/data/gcs-buffer')
    p.add_argument('--receipt-script', default='/opt/catbird-offloader/backfill/make_receipt.mjs')
    p.add_argument('--max-bytes', type=int, default=TIB)
    p.add_argument('--run-hours', type=float, default=12)
    p.add_argument('--poll-seconds', type=float, default=5)
    p.add_argument('--max-segments', type=int, default=6, help='maximum uploads per scan cycle; --once --max-segments 1 for smoke')
    p.add_argument('--once', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not 0 < args.max_bytes <= TIB or not 0 < args.run_hours <= 24 or args.poll_seconds <= 0 or args.max_segments <= 0:
        p.error('max-bytes must be <=1TiB; run-hours <=24; all bounds positive')
    if not all(re.fullmatch('[A-Za-z0-9_.-]+', x) for x in [args.source_id, args.source_epoch]):
        p.error('source namespace contains unsupported characters')
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    try:
        with (work / 'writer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from google.cloud import storage
            buffer = Buffer(storage.Client(), args)
            while True:
                buffer.time_check()
                print(json.dumps(buffer.cycle(args.dry_run)), flush=True)
                if args.once or args.dry_run:
                    return 0
                time.sleep(args.poll_seconds)
    except Exception as error:
        status = {'reason': str(error), 'error_type': type(error).__name__, 'updated_at_unix': time.time()}
        atomic(work / 'status.json', encode(status))
        print(json.dumps(status), flush=True)
        return error.code if isinstance(error, Stop) else 2


if __name__ == '__main__':
    raise SystemExit(main())
