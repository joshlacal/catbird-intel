#!/usr/bin/env python3
"""Download public moderation labels, never posts/media, with resumable receipts.

queryLabels supports exact URI/prefix patterns by lexicon, but mod.bsky.app
currently rejects wildcards (verified 2026-10-20). This tool uses subscribeLabels
from cursor=0 or the last durably journaled seq. This is service-retained event
replay, not proof of all historical labels or current eligible content. Negation,
expiry, CID scope and signatures remain unchanged in raw DAG-CBOR frames.

Dependencies: cbor2, websockets (sync client). No signature verification here.
Official contracts: https://atproto.com/specs/event-stream and
https://github.com/bluesky-social/atproto/tree/main/lexicons/com/atproto/label
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import gzip
import struct
import io
import importlib.metadata
import platform
import sys
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

HANDLE = 'moderation.bsky.app'
MAX_FRAME = 8 * 1024 * 1024


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    if isinstance(value, bytes):
        return {'$bytes': base64.b64encode(value).decode()}
    if isinstance(value, dict):
        return {k: encoded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [encoded(v) for v in value]
    return value


def atomic(path, data):
    temp = path.with_name(path.name + '.pending')
    with temp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def immutable(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('immutable bytes conflict: ' + str(path))
    else:
        atomic(path, data)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def http(url, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Catbird-local-label-audit/1'}), timeout=30) as response:
                data = response.read(MAX_FRAME + 1)
                if len(data) > MAX_FRAME:
                    raise ValueError('HTTP body exceeds cap')
                return data
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
            retry_after = error.headers.get('Retry-After', '')
            delay = min(60, max(2 ** attempt, int(retry_after) if retry_after.isdigit() else 0))
            time.sleep(delay)
    raise RuntimeError('retry exhausted')


def discover(directory):
    raw_handle = http('https://bsky.social/xrpc/com.atproto.identity.resolveHandle?' + urllib.parse.urlencode({'handle': HANDLE}))
    did = json.loads(raw_handle)['did']
    if not did.startswith('did:plc:'):
        raise ValueError('only resolved plc identity supported')
    raw_doc = http('https://plc.directory/' + did)
    doc = json.loads(raw_doc)
    if doc.get('id') != did or 'at://' + HANDLE not in doc.get('alsoKnownAs', []):
        raise ValueError('handle/DID document binding failed')
    services = [s for s in doc.get('service', []) if s['id'] in {'#atproto_labeler', did + '#atproto_labeler'} and s['type'] == 'AtprotoLabeler']
    if len(services) != 1:
        raise ValueError('unique label service required')
    endpoint = services[0]['serviceEndpoint'].rstrip('/')
    if not endpoint.startswith('https://') or urllib.parse.urlsplit(endpoint).username:
        raise ValueError('HTTPS label endpoint required')
    key = [k for k in doc.get('verificationMethod', []) if k['id'] == did + '#atproto_label']
    if len(key) != 1:
        raise ValueError('label signing key absent')
    immutable(directory / ('did-document-' + sha(raw_doc) + '.json'), raw_doc)
    immutable(directory / ('handle-resolution-' + sha(raw_handle) + '.json'), raw_handle)
    identity = {'did': did, 'endpoint': endpoint, 'did_document_sha256': sha(raw_doc),
                'handle_response_sha256': sha(raw_handle), 'label_key': key[0],
                'signature_verification': 'unperformed'}
    identity_path = directory / 'identity.json'
    immutable(identity_path, json_bytes(identity))
    return identity



def capture_policy(directory, identity):
    document = json.loads((directory / ('did-document-' + identity['did_document_sha256'] + '.json')).read_text())
    pds = next(s['serviceEndpoint'] for s in document['service'] if s['id'] in {'#atproto_pds', identity['did'] + '#atproto_pds'})
    if not pds.startswith('https://'):
        raise ValueError('policy record PDS must be HTTPS')
    url = pds.rstrip('/') + '/xrpc/com.atproto.repo.getRecord?' + urllib.parse.urlencode(
        {'repo': identity['did'], 'collection': 'app.bsky.labeler.service', 'rkey': 'self'})
    raw = http(url)
    body = json.loads(raw)
    if body.get('value', {}).get('$type') != 'app.bsky.labeler.service' or not body.get('cid'):
        raise ValueError('labeler self-declaration missing')
    immutable(directory / ('service-declaration-' + sha(raw) + '.json'), raw)
    return {'response_sha256': sha(raw), 'cid': body['cid'], 'uri': body['uri'], 'url': url}

def decode_frame(raw, source):
    import cbor2
    if not isinstance(raw, bytes) or len(raw) > MAX_FRAME:
        raise ValueError('binary frame required within cap')
    stream = io.BytesIO(raw)
    header, body = cbor2.load(stream), cbor2.load(stream)
    if stream.read(1) or not isinstance(header, dict) or not isinstance(body, dict):
        raise ValueError('invalid two-object event stream frame')
    if header.get('op') == -1:
        return {'kind': 'error', 'body': encoded(body)}
    if header.get('op') != 1:
        raise ValueError('unsupported frame operation')
    if header.get('t') == '#info':
        return {'kind': 'info', 'body': encoded(body)}
    if header.get('t') != '#labels' or type(body.get('seq')) is not int or body['seq'] < 0 or not isinstance(body.get('labels'), list):
        raise ValueError('invalid labels frame')
    for label in body['labels']:
        if not isinstance(label, dict) or not all(isinstance(label.get(k), str) for k in ['src', 'uri', 'val', 'cts']):
            raise ValueError('missing label fields')
        if label['src'] != source:
            raise ValueError('unexpected label source')
        if 'neg' in label and type(label['neg']) is not bool:
            raise ValueError('invalid label negation')
    return {'kind': 'labels', 'body': encoded(body)}


def replay(directory):
    state = {'cursor': 0, 'frames': 0, 'labels': 0, 'raw_bytes': 0, 'first_seq': None,
             'negated_labels': 0, 'expiry_labels': 0, 'cid_labels': 0, 'signed_labels': 0,
             'values': {}, 'notices': [], 'sequence_gaps': [], 'seq_hashes': {}}
    journal = directory / 'journal.jsonl'
    if journal.exists():
        with journal.open('rb') as f:
            for raw in f:
                if not raw.endswith(b'\n'):
                    raise ValueError('incomplete journal tail; preserve and repair explicitly')
                entry = json.loads(raw)
                path = directory / entry['raw_path']
                data = path.read_bytes()
                if sha(data) != entry['sha256'] or len(data) != entry['bytes']:
                    raise ValueError('journal raw hash mismatch')
                apply_entry(state, entry)
    return state


def apply_entry(state, entry):
    if entry['kind'] == 'labels':
        seq = entry['seq']
        if str(seq) in state['seq_hashes']:
            if state['seq_hashes'][str(seq)] != entry['sha256']:
                raise ValueError('same cursor has changed bytes')
            return
        if seq <= state['cursor']:
            raise ValueError('sequence moved backward')
        if seq != state['cursor'] + 1:
            state['sequence_gaps'].append({'after': state['cursor'], 'before': seq})
        if state['first_seq'] is None:
            state['first_seq'] = seq
        state['cursor'] = seq
        state['seq_hashes'][str(seq)] = entry['sha256']
        state['labels'] += entry['labels']
        for field in ['negated_labels', 'expiry_labels', 'cid_labels', 'signed_labels']:
            state[field] += entry[field]
        state['values'] = dict(Counter(state['values']) + Counter(entry['values']))
    else:
        state['notices'].append({'kind': entry['kind'], 'body': entry['body']})
    state['frames'] += 1
    state['raw_bytes'] += entry['bytes']


def save_frame(directory, state, raw, decoded):
    body = decoded['body']
    raw_hash = sha(raw)
    entry = {'received_at': utc(), 'kind': decoded['kind'], 'sha256': raw_hash,
             'bytes': len(raw), 'raw_path': 'raw/' + raw_hash + '.cbor'}
    if decoded['kind'] == 'labels':
        labels = body['labels']
        entry.update(seq=body['seq'], labels=len(labels), values=dict(Counter(l['val'] for l in labels)),
                     negated_labels=sum(l.get('neg') is True for l in labels),
                     expiry_labels=sum('exp' in l for l in labels), cid_labels=sum('cid' in l for l in labels),
                     signed_labels=sum('sig' in l for l in labels))
        if str(body['seq']) in state['seq_hashes']:
            if state['seq_hashes'][str(body['seq'])] != raw_hash:
                raise ValueError('replayed cursor bytes changed')
            return False
    else:
        entry['body'] = body
    # Validate sequence before persistence; raw first, journal second, checkpoint last.
    if entry['kind'] == 'labels' and entry['seq'] <= state['cursor']:
        raise ValueError('sequence moved backward')
    immutable(directory / entry['raw_path'], raw)
    with (directory / 'journal.jsonl').open('ab') as f:
        f.write((json.dumps(entry, sort_keys=True) + '\n').encode())
        f.flush()
        os.fsync(f.fileno())
    apply_entry(state, entry)
    atomic(directory / 'checkpoint.json', json_bytes(summary(state)))
    return True


def summary(state):
    return {k: v for k, v in state.items() if k != 'seq_hashes'}



def capture_execution(directory, args):
    source = Path(__file__).read_bytes()
    source_hash = sha(source)
    immutable(directory / ('executed-source-' + source_hash + '.py'), source)
    receipt = {'started_at': utc(), 'source_sha256': source_hash, 'source_path': str(Path(__file__).resolve()),
               'python': sys.version, 'platform': platform.platform(),
               'dependencies': {name: importlib.metadata.version(name) for name in ['cbor2', 'websockets']},
               'bounds': vars(args)}
    immutable(directory / ('execution-' + str(time.time_ns()) + '.json'), json_bytes(receipt))
    return receipt

def run_legacy(args):
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosed, InvalidStatus
    directory = Path(args.directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'raw').mkdir(exist_ok=True)
    with (directory / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = discover(directory)
        state = replay(directory)
        before = summary(state)
        start, count, attempts = time.monotonic(), 0, 0
        reason = 'bounded_run_complete'
        while count < args.max_frames and time.monotonic() - start < args.max_seconds:
            used = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
            if used + MAX_FRAME > args.max_total_bytes or shutil.disk_usage(directory).free < args.min_free_bytes:
                reason = 'disk_budget_stop'
                break
            url = identity['endpoint'].replace('https://', 'wss://', 1) + '/xrpc/com.atproto.label.subscribeLabels?' + urllib.parse.urlencode({'cursor': state['cursor']})
            try:
                with connect(url, open_timeout=15, max_size=MAX_FRAME, max_queue=4, close_timeout=5) as ws:
                    while count < args.max_frames and time.monotonic() - start < args.max_seconds:
                        if used + MAX_FRAME > args.max_total_bytes or shutil.disk_usage(directory).free < args.min_free_bytes:
                            reason = 'disk_budget_stop'
                            break
                        try:
                            raw = ws.recv(timeout=min(10, max(.01, args.max_seconds - (time.monotonic() - start))))
                        except TimeoutError:
                            continue
                        try:
                            decoded = decode_frame(raw, identity['did'])
                        except (ValueError, EOFError) as error:
                            rejected = raw if isinstance(raw, bytes) else raw.encode()
                            immutable(directory / ('rejected-' + sha(rejected) + '.bin'), rejected)
                            reason = 'invalid_frame_preserved:' + type(error).__name__
                            break
                        added = save_frame(directory, state, raw, decoded)
                        count += int(added)
                        used += len(raw) + 2048
                        if decoded['kind'] == 'error':
                            reason = 'server_error_preserved'
                            break
                        time.sleep(1 / args.frames_per_second)
                if reason != 'bounded_run_complete' or count >= args.max_frames:
                    break
            except (OSError, TimeoutError, ConnectionClosed, InvalidStatus) as error:
                attempts += 1
                if attempts >= 4:
                    reason = 'connection_retry_exhausted:' + type(error).__name__
                    break
                delay = min(15, 2 ** attempts)
                response = getattr(error, 'response', None)
                if response is not None:
                    retry_after = response.headers.get('Retry-After', '')
                    if retry_after.isdigit():
                        delay = min(60, max(delay, int(retry_after)))
                time.sleep(delay)
        receipt = {'completed_at': utc(), 'identity': identity, 'before': before, 'after': summary(state),
                   'elapsed_seconds': time.monotonic() - start, 'new_frames': count, 'stop_reason': reason,
                   'bounds': vars(args), 'coverage': 'public service-retained subscribeLabels replay; not a full-history or current-state guarantee',
                   'signature_verification': 'unperformed', 'no_label_means_safe': False,
                   'raw_format': 'exact binary websocket frames: DAG-CBOR header then body; signatures unchanged',
                   'retry_contract': 'raw fsync then journal fsync then checkpoint; journal replay authoritative; same seq/hash deduplicated'}
        output = directory / ('run-' + str(time.time_ns()) + '.json')
        immutable(output, json_bytes(receipt))
        print(json.dumps({'receipt': str(output), **summary(state), 'stop_reason': reason}, sort_keys=True))



PACKED_SCHEMA = 'catbird-public-label-replay-packed-v2'


def compact_initial():
    return {'cursor': 0, 'last_frame_sha256': None, 'frames': 0, 'labels': 0,
            'raw_bytes': 0, 'first_seq': None, 'minimum_cts': None, 'maximum_cts': None,
            'negated_labels': 0, 'expiry_labels': 0, 'cid_labels': 0, 'signed_labels': 0,
            'values': {}, 'notice_count': 0, 'notices_sample': [],
            'sequence_gap_count': 0, 'missing_sequence_count': 0, 'sequence_gaps_sample': []}


def compact_apply(state, raw, decoded):
    body = decoded['body']
    frame_hash = sha(raw)
    if decoded['kind'] == 'labels':
        seq = body['seq']
        if seq == state['cursor'] and frame_hash == state['last_frame_sha256']:
            return False
        if seq < state['cursor'] or (seq == state['cursor'] and (state['last_frame_sha256'] is not None or state['frames'] != 0)):
            raise ValueError('backward or changed boundary cursor')
        if seq > state['cursor'] + 1:
            state['sequence_gap_count'] += 1
            state['missing_sequence_count'] += seq - state['cursor'] - 1
            if len(state['sequence_gaps_sample']) < 32:
                state['sequence_gaps_sample'].append({'after': state['cursor'], 'before': seq})
        if state['first_seq'] is None:
            state['first_seq'] = seq
        state['cursor'], state['last_frame_sha256'] = seq, frame_hash
        labels = body['labels']
        state['labels'] += len(labels)
        state['negated_labels'] += sum(l.get('neg') is True for l in labels)
        state['expiry_labels'] += sum('exp' in l for l in labels)
        state['cid_labels'] += sum('cid' in l for l in labels)
        state['signed_labels'] += sum('sig' in l for l in labels)
        state['values'] = dict(Counter(state['values']) + Counter(l['val'] for l in labels))
        for label in labels:
            cts = label['cts']
            state['minimum_cts'] = min(cts, state['minimum_cts']) if state['minimum_cts'] else cts
            state['maximum_cts'] = max(cts, state['maximum_cts']) if state['maximum_cts'] else cts
    else:
        state['notice_count'] += 1
        if len(state['notices_sample']) < 32:
            state['notices_sample'].append({'kind': decoded['kind'], 'body': body})
    state['frames'] += 1
    state['raw_bytes'] += len(raw)
    return True


def iter_packed(raw):
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < 8:
            raise ValueError('partial frame length')
        length = struct.unpack('>Q', raw[offset:offset + 8])[0]
        offset += 8
        if not 0 < length <= MAX_FRAME or len(raw) - offset < length:
            raise ValueError('invalid packed frame length')
        yield raw[offset:offset + length]
        offset += length


def packed_manifest(directory, identity, target, parent_path=None, start_cursor=None, policy=None):
    path = directory / 'manifest.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['schema'] != PACKED_SCHEMA or result['identity'] != identity or result['target_seq'] != target or (start_cursor is not None and result.get('start_cursor', 0) != start_cursor):
            raise ValueError('immutable packed identity/target mismatch')
        return result
    base, parent = compact_initial(), None
    if start_cursor is not None and (type(start_cursor) is not int or start_cursor < 0):
        raise ValueError('start cursor must be nonnegative')
    if start_cursor and parent_path:
        raise ValueError('explicit start and parent coverage cannot be combined')
    base['cursor'] = start_cursor or 0
    if parent_path:
        parent_root = Path(parent_path).expanduser().resolve()
        with (parent_root / 'writer.lock').open('a') as parent_lock:
            fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            parent_identity = json.loads((parent_root / 'identity.json').read_text())
            if parent_identity != identity:
                raise ValueError('parent labeler identity differs')
            # Independently decode exact parent frames while its writer is excluded.
            journal_bytes = (parent_root / 'journal.jsonl').read_bytes()
            for line in journal_bytes.splitlines():
                row = json.loads(line)
                raw = (parent_root / row['raw_path']).read_bytes()
                if sha(raw) != row['sha256'] or len(raw) != row['bytes']:
                    raise ValueError('parent raw hash mismatch')
                compact_apply(base, raw, decode_frame(raw, identity['did']))
            parent = {'directory': str(parent_root), 'journal_sha256': sha(journal_bytes),
                      'identity_sha256': sha((parent_root / 'identity.json').read_bytes()),
                      'through_seq': base['cursor'], 'frames': base['frames'],
                      'verification': 'all parent raw hashes and CBOR frames independently decoded'}
    if target is None or type(target) is not int or target < base['cursor']:
        raise ValueError('finite target >= parent cursor required for new generation')
    result = {'schema': PACKED_SCHEMA, 'created_at': utc(), 'identity': identity,
              'target_seq': target, 'start_cursor': start_cursor or 0, 'base_state': base, 'parent': parent,
              'policy_at_creation': policy,
              'format': 'gzip chunks, each containing uint64 big-endian length then exact binary frame',
              'scope': 'public retained service replay; sequence gaps preserved; no full-history assertion',
              'signature_verification': 'unperformed'}
    immutable(path, json_bytes(result))
    return result


class PackedStore:
    """Raw gzip chunk -> fsynced hash-chained journal -> atomic checkpoint.

    Orphan chunks remain immutable and unreferenced. A torn final journal line
    is preserved by renaming the complete journal before restoring its valid
    prefix. Resume trusts only committed chunks; no frame-sized global map.
    """
    def __init__(self, directory, manifest, verify=False):
        self.directory = directory
        (directory / 'chunks').mkdir(exist_ok=True)
        self.manifest = manifest
        self.state = json.loads(json.dumps(manifest['base_state']))
        parent = manifest.get('parent')
        if parent:
            parent_root = Path(parent['directory'])
            parent_journal = (parent_root / 'journal.jsonl').read_bytes()
            if sha(parent_journal) != parent['journal_sha256'] or sha((parent_root / 'identity.json').read_bytes()) != parent['identity_sha256']:
                raise ValueError('frozen parent coverage changed')
            if verify:
                checked = compact_initial()
                for line in parent_journal.splitlines():
                    row = json.loads(line)
                    raw = (parent_root / row['raw_path']).read_bytes()
                    if sha(raw) != row['sha256'] or len(raw) != row['bytes']:
                        raise ValueError('parent raw hash mismatch')
                    compact_apply(checked, raw, decode_frame(raw, manifest['identity']['did']))
                if checked != manifest['base_state']:
                    raise ValueError('parent decoded state changed')
        self.chain = sha(json_bytes(manifest))
        self.chunk_count = 0
        self.compressed_bytes = 0
        self.journal = directory / 'chunks.jsonl'
        if self.journal.exists():
            data = self.journal.read_bytes()
            if data and not data.endswith(b'\n'):
                split = data.rfind(b'\n') + 1
                backup = directory / ('chunks-torn-' + str(time.time_ns()) + '.jsonl')
                os.rename(self.journal, backup)
                atomic(self.journal, data[:split])
                data = data[:split]
            for line in data.splitlines():
                entry = json.loads(line)
                if entry['previous_sha256'] != self.chain or entry['before_cursor'] != self.state['cursor']:
                    raise ValueError('journal chain/cursor mismatch')
                chunk_path = directory / 'chunks' / (entry['compressed_sha256'] + '.frames.gz')
                compressed = chunk_path.read_bytes()
                if len(compressed) != entry['compressed_bytes'] or sha(compressed) != entry['compressed_sha256']:
                    raise ValueError('committed compressed chunk hash mismatch')
                if verify:
                    raw = gzip.decompress(compressed)
                    if sha(raw) != entry['raw_sha256'] or len(raw) != entry['raw_bytes']:
                        raise ValueError('uncompressed chunk hash mismatch')
                    for frame in iter_packed(raw):
                        compact_apply(self.state, frame, decode_frame(frame, manifest['identity']['did']))
                    if self.state != entry['after']:
                        raise ValueError('independent decoded counts differ from journal')
                else:
                    self.state = entry['after']
                self.chain = sha(line)
                self.chunk_count += 1
                self.compressed_bytes += len(compressed)
        # One metadata scan at startup, never per frame. Includes orphan and torn files.
        self.logical_bytes = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())

    def commit(self, frames, after):
        if not frames:
            return
        packed = b''.join(struct.pack('>Q', len(f)) + f for f in frames)
        compressed = gzip.compress(packed, compresslevel=3, mtime=0)
        entry = {'previous_sha256': self.chain, 'before_cursor': self.state['cursor'],
                 'after': after, 'frames': len(frames), 'raw_sha256': sha(packed),
                 'raw_bytes': len(packed), 'compressed_sha256': sha(compressed),
                 'compressed_bytes': len(compressed), 'committed_at': utc()}
        chunk = self.directory / 'chunks' / (entry['compressed_sha256'] + '.frames.gz')
        existed = chunk.exists()
        immutable(chunk, compressed)
        line = json.dumps(entry, sort_keys=True, separators=(',', ':')).encode()
        with self.journal.open('ab') as out:
            out.write(line + b'\n')
            out.flush()
            os.fsync(out.fileno())
        self.state = json.loads(json.dumps(after))
        self.chain = sha(line)
        self.chunk_count += 1
        self.compressed_bytes += len(compressed)
        self.logical_bytes += len(line) + 1 + (0 if existed else len(compressed))
        atomic(self.directory / 'checkpoint.json', json_bytes(self.checkpoint()))

    def checkpoint(self):
        return {'schema': PACKED_SCHEMA, 'state': self.state, 'journal_chain_sha256': self.chain,
                'chunks': self.chunk_count, 'compressed_bytes': self.compressed_bytes,
                'target_seq': self.manifest['target_seq'], 'updated_at': utc()}


def run_packed(args):
    from websockets.sync.client import connect
    from websockets.exceptions import ConnectionClosed, InvalidStatus
    directory = Path(args.directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.verify_only:
            manifest = json.loads((directory / 'manifest.json').read_text())
            store = PackedStore(directory, manifest, verify=True)
            receipt = {'verified_at': utc(), 'verification': 'all committed gzip/raw hashes, framing, counters decoded',
                       **store.checkpoint(), 'signature_verification': 'unperformed'}
            output = directory / ('verification-' + str(time.time_ns()) + '.json')
            immutable(output, json_bytes(receipt))
            print(json.dumps({'receipt': str(output), **receipt}, sort_keys=True))
            return
        execution = capture_execution(directory, args)
        identity = discover(directory)
        policy = capture_policy(directory, identity) if not (directory / 'manifest.json').exists() else None
        manifest = packed_manifest(directory, identity, args.target_seq, args.parent_v1, args.start_cursor, policy)
        store = PackedStore(directory, manifest)
        before = json.loads(json.dumps(store.state))
        working = json.loads(json.dumps(store.state))
        frames, pending_bytes, count, attempts, total_disconnects = [], 0, 0, 0, 0
        start, last_flush, last_disk_check = time.monotonic(), time.monotonic(), 0.0
        reason = 'bounded_run_complete'
        target = manifest['target_seq']
        free = shutil.disk_usage(directory).free
        budget_reserve = 2 * MAX_FRAME  # raw pending + compressed worst-case scratch.

        def flush():
            nonlocal frames, pending_bytes, last_flush
            if frames:
                try:
                    store.commit(frames, working)
                except OSError as error:
                    raise RuntimeError('local persistence failed; preserve files and resume from journal') from error
                frames, pending_bytes = [], 0
                last_flush = time.monotonic()

        while count < args.max_frames and time.monotonic() - start < args.max_seconds:
            if working['cursor'] >= target:
                reason = 'target_reached_retention_not_proven'
                break
            if store.logical_bytes + budget_reserve >= args.max_total_bytes or free - budget_reserve < args.min_free_bytes:
                reason = 'disk_budget_stop'
                break
            url = identity['endpoint'].replace('https://', 'wss://', 1) + '/xrpc/com.atproto.label.subscribeLabels?' + urllib.parse.urlencode({'cursor': working['cursor']})
            connection_cursor = store.state['cursor']
            try:
                with connect(url, open_timeout=15, max_size=MAX_FRAME, max_queue=4, close_timeout=1) as ws:
                    while count < args.max_frames and time.monotonic() - start < args.max_seconds:
                        now = time.monotonic()
                        if now - last_disk_check >= 1:
                            free = shutil.disk_usage(directory).free
                            last_disk_check = now
                        if store.logical_bytes + pending_bytes * 2 + budget_reserve >= args.max_total_bytes or free - pending_bytes * 2 - budget_reserve < args.min_free_bytes:
                            reason = 'disk_budget_stop'
                            break
                        try:
                            raw = ws.recv(timeout=min(5, max(.01, args.max_seconds - (now - start))))
                        except TimeoutError:
                            flush()
                            continue
                        try:
                            decoded = decode_frame(raw, identity['did'])
                        except (ValueError, EOFError) as error:
                            rejected = raw if isinstance(raw, bytes) else raw.encode()
                            immutable(directory / ('rejected-' + sha(rejected) + '.bin'), rejected)
                            reason = 'invalid_frame_preserved:' + type(error).__name__
                            break
                        # The first label beyond target establishes a crossing, not
                        # target coverage. Preserve it separately, never in frozen input.
                        if decoded['kind'] == 'labels' and decoded['body']['seq'] > target:
                            immutable(directory / ('target-crossing-' + sha(raw) + '.cbor'), raw)
                            reason = 'target_crossed_target_event_unobserved'
                            break
                        try:
                            accepted = compact_apply(working, raw, decoded)
                        except ValueError:
                            immutable(directory / ('rejected-cursor-' + sha(raw) + '.cbor'), raw)
                            reason = 'invalid_cursor_preserved'
                            break
                        if accepted:
                            frames.append(raw)
                            pending_bytes += len(raw) + 8
                            count += 1
                        if decoded['kind'] == 'error':
                            reason = 'server_error_preserved'
                            break
                        if len(frames) >= args.batch_frames or pending_bytes >= MAX_FRAME or time.monotonic() - last_flush >= 5:
                            flush()
                        if working['cursor'] >= target:
                            reason = 'target_reached_retention_not_proven'
                            break
                        # Aggregate pacing includes receive/compression/fsync work.
                        ahead = count / args.frames_per_second - (time.monotonic() - start)
                        if ahead > 0:
                            time.sleep(min(ahead, .1))
                flush()
                if reason != 'bounded_run_complete' or count >= args.max_frames:
                    break
            except (OSError, TimeoutError, ConnectionClosed, InvalidStatus) as error:
                flush()
                total_disconnects += 1
                progressed = store.state['cursor'] > connection_cursor
                attempts = 0 if progressed else attempts + 1
                with (directory / 'connection-notices.jsonl').open('ab') as out:
                    out.write((json.dumps({'at': utc(), 'error_type': type(error).__name__,
                                          'cursor': store.state['cursor'], 'progress_since_connect': progressed,
                                          'consecutive_no_progress_failures': attempts,
                                          'http_status': getattr(getattr(error, 'response', None), 'status_code', None)}) + '\n').encode())
                    out.flush()
                    os.fsync(out.fileno())
                if attempts >= 4:
                    reason = 'connection_retry_exhausted:' + type(error).__name__
                    break
                delay = min(15, 2 ** attempts)
                response = getattr(error, 'response', None)
                if response is not None:
                    retry_after = response.headers.get('Retry-After', '')
                    if retry_after.isdigit():
                        delay = min(60, max(delay, int(retry_after)))
                time.sleep(delay)
        flush()
        receipt = {'completed_at': utc(), 'execution': execution, 'before': before, 'after': store.state,
                   'elapsed_seconds': time.monotonic() - start, 'new_frames': count,
                   'stop_reason': reason, 'bounds': vars(args), 'checkpoint': store.checkpoint(),
                   'total_disconnects': total_disconnects, 'consecutive_no_progress_failures': attempts,
                   'coverage': 'parent manifest plus committed chunks; public retained replay, not all-time truth',
                   'signature_verification': 'unperformed', 'no_label_means_safe': False,
                   'journal_sha256': sha(store.journal.read_bytes()) if store.journal.exists() else None,
                   'retry_contract': 'raw immutable gzip chunk fsync then chained journal fsync then checkpoint; pending chunks replayed on crash'}
        output = directory / ('run-' + str(time.time_ns()) + '.json')
        immutable(output, json_bytes(receipt))
        print(json.dumps({'receipt': str(output), 'stop_reason': reason, 'new_frames': count,
                          'cursor': store.state['cursor'], 'compressed_bytes': store.compressed_bytes,
                          'elapsed_seconds': receipt['elapsed_seconds']}, sort_keys=True))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='~/catbird-lake-local/moderation-labels/bluesky-v2')
    parser.add_argument('--storage', choices=['packed', 'legacy'], default='packed')
    parser.add_argument('--parent-v1', help='stopped, verified v1 directory to inherit coverage from')
    parser.add_argument('--target-seq', type=int, help='freeze finite catch-up target; does not assert full history')
    parser.add_argument('--start-cursor', type=int, help='new generation only: retained replay cursor; never implies earlier coverage')
    parser.add_argument('--batch-frames', type=int, default=5000)
    parser.add_argument('--verify-only', action='store_true', help='independently decode/hash every packed committed frame')
    parser.add_argument('--max-frames', type=int, default=1000)
    parser.add_argument('--max-seconds', type=float, default=60)
    parser.add_argument('--frames-per-second', type=float, default=50)
    parser.add_argument('--max-total-bytes', type=int, default=1024 * 1024 * 1024)
    parser.add_argument('--min-free-bytes', type=int, default=30 * 1024 * 1024 * 1024)
    args = parser.parse_args()
    if args.max_frames <= 0 or args.max_seconds <= 0 or not 0 < args.frames_per_second <= 1000 or args.max_total_bytes <= MAX_FRAME or args.min_free_bytes < 0:
        parser.error('positive bounds and rate <=1000 frames/s required; byte budget must exceed max frame')
    if args.batch_frames <= 0 or args.batch_frames > 20000:
        parser.error('batch frames must be between 1 and 20000')
    (run_packed if args.storage == 'packed' else run_legacy)(args)


if __name__ == '__main__':
    main()
