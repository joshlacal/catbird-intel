#!/usr/bin/env python3
"""Bounded, in-memory normalized Jetstream v2 projection validator.

This is NOT a raw .jss decoder or a network collector. Input is explicit JSONL:
common: source_id, source_epoch, seq (integer), witnessedAt (integer Unix us),
did, kind. commit adds collection,rkey,rev (13-character TID),operation and
record (object for create/update); account adds active/status; sync adds
reset=true; identity adds no trusted handle. Unknown fields are rejected.

freeze hashes explicit files before fold; no globs. Each input has a source,
epoch, sealed tip and role. Exactly one source is authoritative. Audit records
never supply authoritative records. Audit restrictions survive later active
rows and require an explicit reconciliation receipt to release them. Receipts
are operator-supplied attestations, NOT cryptographically/network verified here.

Official semantics checked 2026-10-20:
https://github.com/bluesky-social/jetstream/blob/main/docs/README.md sections
3.3, 4.5: sync and deleted-account markers clear preceding records; replacement
records follow sync. Revision ordering applies inside each reset epoch.
A frozen file inventory proves bytes, not source completeness or historical
as-of truth; archive compaction can make historical reconstruction biased.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

SCHEMA = 'catbird-normalized-jetstream-v2/1'
COMMON = {'source_id', 'source_epoch', 'seq', 'witnessedAt', 'did', 'kind'}
FIELDS = {'commit': {'collection', 'rkey', 'rev', 'operation', 'record'},
          'account': {'active', 'status'}, 'sync': {'reset'}, 'identity': set()}
STATES = {'active', 'deleted', 'deactivated', 'takendown', 'suspended',
          'desynchronized', 'unknown'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, obj):
    with Path(path).open('x') as out:
        out.write(json.dumps(obj, sort_keys=True, indent=2) + '\n')


def freeze(spec):
    require(set(spec) == {'generation', 'inputs'}, 'spec needs generation and inputs')
    require(isinstance(spec['generation'], str) and spec['generation'], 'generation required')
    require(spec['inputs'], 'empty inventory')
    result = {'schema': SCHEMA, 'generation': spec['generation'], 'inputs': []}
    seen = set()
    authorities = set()
    source_contracts = {}
    for item in spec['inputs']:
        require(set(item) == {'path', 'source_id', 'source_epoch', 'sealed_tip_seq', 'role'}, 'invalid input spec')
        require(item['role'] in {'authoritative', 'audit'}, 'invalid role')
        require(type(item['sealed_tip_seq']) is int and item['sealed_tip_seq'] >= 0, 'invalid sealed tip')
        require(all(isinstance(item[k], str) and item[k] for k in ['source_id', 'source_epoch']), 'source/epoch required')
        namespace = (item['source_id'], item['source_epoch'])
        contract = (item['role'], item['sealed_tip_seq'])
        require(namespace not in source_contracts or source_contracts[namespace] == contract, 'source role/tip mismatch')
        source_contracts[namespace] = contract
        path = Path(item['path']).resolve(strict=True)
        require(str(path) not in seen, 'duplicate input path')
        seen.add(str(path))
        data = path.read_bytes()
        result['inputs'].append({**item, 'path': str(path), 'bytes': len(data), 'sha256': digest(data)})
        if item['role'] == 'authoritative':
            authorities.add((item['source_id'], item['source_epoch'], item['sealed_tip_seq']))
    require(len(authorities) == 1, 'exactly one authority/epoch/tip required')
    return result


def validate(e, item):
    kind = e.get('kind')
    require(kind in FIELDS, 'unknown kind (raw decoder input unsupported)')
    require(COMMON <= e.keys() and e.keys() <= COMMON | FIELDS[kind], 'missing/unknown event fields')
    require((e['source_id'], e['source_epoch']) == (item['source_id'], item['source_epoch']), 'source identity mismatch')
    require(type(e['seq']) is int and 0 <= e['seq'] <= item['sealed_tip_seq'], 'seq outside sealed generation')
    require(type(e['witnessedAt']) is int and e['witnessedAt'] >= 0, 'witnessedAt must be Unix microseconds')
    require(isinstance(e['did'], str) and e['did'].startswith('did:'), 'invalid DID')
    if kind == 'commit':
        require({'collection', 'rkey', 'rev', 'operation'} <= e.keys(), 'incomplete commit')
        require(re.fullmatch('[234567abcdefghijklmnopqrstuvwxyz]{13}', e['rev']) is not None, 'rev must be TID')
        require(isinstance(e['collection'], str) and '.' in e['collection'] and '/' not in e['collection'], 'invalid collection')
        require(isinstance(e['rkey'], str) and e['rkey'] and '/' not in e['rkey'], 'invalid rkey')
        require(e['operation'] in {'create', 'update', 'delete'}, 'invalid operation')
        require(isinstance(e.get('record'), dict) if e['operation'] != 'delete' else 'record' not in e, 'invalid record payload')
    elif kind == 'account':
        require(type(e.get('active')) is bool and e.get('status') in STATES, 'invalid account state')
        require(e['active'] == (e['status'] == 'active'), 'contradictory account state')
    elif kind == 'sync':
        require(e.get('reset') is True, 'sync must be normalized verified divergence reset')


def fold(manifest, receipts=()):
    require(manifest.get('schema') == SCHEMA, 'unsupported manifest')
    # Read and verify all bytes before consuming any events, avoiding hash/read races.
    spec = {'generation': manifest['generation'], 'inputs': [{k: v for k, v in i.items() if k not in {'bytes', 'sha256'}} for i in manifest['inputs']]}
    require(freeze(spec) == manifest, 'inventory changed or malformed manifest')
    blobs = []
    for item in manifest['inputs']:
        data = Path(item['path']).read_bytes()
        require(len(data) == item['bytes'] and digest(data) == item['sha256'], 'input changed after freeze')
        blobs.append((item, data))
    events, duplicates = {}, 0
    for item, data in blobs:
        for line in data.splitlines():
            require(bool(line.strip()), 'blank input row')
            e = json.loads(line)
            validate(e, item)
            key = (e['source_id'], e['source_epoch'], e['seq'])
            if key in events:
                require(events[key][1] == e and events[key][0] == item['role'], 'conflicting sequence or role')
                duplicates += 1
            events[key] = (item['role'], e)
    records, accounts, identities, audits, conflicts, barriers = {}, {}, {}, {}, [], {}
    authoritative_events = 0
    audit_records = {}
    revision_payloads, reset_epochs = {}, {}
    for key in sorted(events):
        role, e = events[key]
        did, kind = e['did'], e['kind']
        namespace = (e['source_id'], e['source_epoch'], did)
        if kind == 'sync' or (kind == 'account' and e['status'] == 'deleted'):
            reset_epochs[namespace] = e['seq']
        elif kind == 'commit':
            uri = f"at://{did}/{e['collection']}/{e['rkey']}"
            reset_seq = reset_epochs.get(namespace)
            revision_key = (namespace, reset_seq, uri, e['rev'])
            payload_hash = digest(canonical([e['operation'], e.get('record')]).encode())
            payloads = revision_payloads.setdefault(revision_key, {})
            if payload_hash not in payloads:
                if payloads:
                    first_hash, first_seq = next(iter(payloads.items()))
                    conflicts.append({'source_id': e['source_id'], 'source_epoch': e['source_epoch'],
                                      'reset_seq': reset_seq, 'role': role, 'uri': uri, 'rev': e['rev'],
                                      'first_seq': first_seq, 'second_seq': e['seq'],
                                      'first_payload_sha256': first_hash, 'second_payload_sha256': payload_hash})
                payloads[payload_hash] = e['seq']
        if role == 'audit':
            namespace = (e['source_id'], e['source_epoch'])
            state = audit_records.setdefault(namespace, {})
            if kind == 'sync' or (kind == 'account' and e['status'] == 'deleted'):
                audit_records[namespace] = {uri: r for uri, r in state.items() if r['did'] != did}
            elif kind == 'commit':
                uri = f"at://{did}/{e['collection']}/{e['rkey']}"
                old = state.get(uri)
                if not old or (e['rev'], e['seq']) > (old['rev'], old['seq']):
                    state[uri] = e
            if (kind == 'account' and not e['active']) or kind == 'sync':
                audits[digest(canonical(e).encode())] = e
            continue
        authoritative_events += 1
        if kind == 'identity':
            identities[did] = {'requires_resolution': True, 'event': e}
        elif kind == 'account':
            accounts[did] = e
        if kind == 'sync' or (kind == 'account' and e['status'] == 'deleted'):
            records = {uri: r for uri, r in records.items() if r['did'] != did}
            barriers[did] = e
        elif kind == 'commit':
            uri = f"at://{did}/{e['collection']}/{e['rkey']}"
            old = records.get(uri)
            if not old or (e['rev'], e['seq']) > (old['rev'], old['seq']):
                records[uri] = e
    agreement = []
    for namespace, state in sorted(audit_records.items()):
        for uri, other in sorted(state.items()):
            current = records.get(uri)
            status = 'authority_missing_not_proof_of_loss'
            if current:
                status = 'different_revision_not_comparable'
                if current['rev'] == other['rev']:
                    status = 'agree' if (current['operation'], current.get('record')) == (other['operation'], other.get('record')) else 'equal_revision_conflict'
            agreement.append({'source_id': namespace[0], 'source_epoch': namespace[1], 'uri': uri, 'status': status})
    released = []
    authority = next(i for i in manifest['inputs'] if i['role'] == 'authoritative')
    for receipt in receipts:
        require(set(receipt) == {'restriction_sha256', 'did', 'source_id', 'source_epoch', 'observedAt', 'status', 'evidence_sha256', 'method'}, 'invalid reconciliation receipt')
        r = audits.get(receipt['restriction_sha256'])
        require(r is not None and r['did'] == receipt['did'], 'receipt does not identify restriction')
        require(receipt['source_id'] == authority['source_id'] and receipt['source_epoch'] == authority['source_epoch'], 'receipt authority mismatch')
        require(type(receipt['observedAt']) is int and receipt['observedAt'] > r['witnessedAt'], 'reconciliation must be newer')
        require(receipt['status'] == 'active' and receipt['method'] == 'authoritative_repo_reconciliation', 'explicit authoritative reconciliation required')
        require(re.fullmatch('[0-9a-f]{64}', receipt['evidence_sha256']) is not None, 'evidence hash required')
        released.append(receipt)
        del audits[receipt['restriction_sha256']]
    dids = {e['did'] for _, e in events.values()}
    restrictions = {did: [] for did in sorted(dids)}
    for did in dids:
        if did not in accounts or not accounts[did]['active']:
            restrictions[did].append('unknown_account' if did not in accounts else accounts[did]['status'])
        if did in barriers and barriers[did]['kind'] == 'sync':
            restrictions[did].append('sync_replacement_completeness_unverified')
    for r in audits.values():
        restrictions[r['did']].append('audit_disagreement_requires_reconciliation')
    return {'schema': SCHEMA, 'generation': manifest['generation'], 'inventory_sha256': digest(canonical(manifest).encode()),
            'records': records, 'accounts': accounts, 'identities': identities, 'reset_barriers': barriers,
            'restrictions': restrictions, 'audit_restrictions': audits, 'reconciliation_receipts': released,
            'conflicts': conflicts, 'record_agreement_audit': agreement, 'counts': {'unique_events': len(events), 'duplicate_events': duplicates,
            'authoritative_events': authoritative_events, 'current_records': sum(e['operation'] != 'delete' for e in records.values()),
            'record_tombstones': sum(e['operation'] == 'delete' for e in records.values()), 'conflicts': len(conflicts)},
            'promotion_eligible': False,
            'limitations': ['Normalized input only; no raw decoder or network completeness validation',
                           'Sync replacement completeness and identity resolution require external evidence',
                           'Reconciliation receipts are attestations; network/signatures not verified',
                           'In-memory bounded validator; not full-network production fold',
                           'Frozen bytes do not establish pre-compaction historical state']}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['freeze', 'fold']:
        p = commands.add_parser(name)
        p.add_argument('--input', required=True, help='spec JSON for freeze; frozen manifest JSON for fold')
        p.add_argument('--output', required=True, help='new file only; existing files never overwritten')
        if name == 'fold':
            p.add_argument('--reconciliations', help='JSON array of operator-attested reconciliation receipts')
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text())
    if args.command == 'freeze':
        output = freeze(source)
    else:
        receipts = json.loads(Path(args.reconciliations).read_text()) if args.reconciliations else []
        output = fold(source, receipts)
    write_new(args.output, output)


if __name__ == '__main__':
    main()
