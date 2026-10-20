#!/usr/bin/env python3
"""Seal historical editorial candidates for review, never a serving allowlist.

No network calls, inference, seed injection, identity expansion or user actions.
Every candidate must pass independent current eligibility and curator review.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

SCHEMA = "catbird.editorial-review.v1"
BLOCKERS = ["current_identity_unverified", "live_deletion_visibility_unverified",
            "moderation_and_adult_eligibility_unverified", "viewer_blocks_mutes_unverified",
            "reference_list_optout_unverified", "curator_approval_missing"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def pack_identity(url):
    p = urlparse(url)
    parts = p.path.strip('/').split('/')
    if p.scheme != 'https' or p.netloc != 'bsky.app' or len(parts) != 3 or parts[0] != 'starter-pack':
        raise ValueError(f"Unsupported pack reference: {url}")
    actor, rkey = parts[1:]
    # Handle-based references are not asserted to be canonical DID URIs.
    return {"source_url": url, "record_reference": f"at://{actor}/app.bsky.graph.starterpack/{rkey}",
            "canonical_uri": f"at://{actor}/app.bsky.graph.starterpack/{rkey}" if actor.startswith('did:') else None,
            "creator_identity": actor}


def build(source, source_path):
    catalog = json.loads(source)
    entries = []
    scenes = []
    for si, scene in enumerate(catalog['subcultures']):
        sid = scene['id']
        scenes.append({"id": sid, "historical_editorial_name": scene['name'], "curator_identity": None})
        candidates = []
        for ai, account in enumerate(scene['recommended_accounts']):
            did = account['did']
            if not isinstance(did, str) or not did.startswith('did:'):
                raise ValueError('Account candidate missing DID')
            candidates.append(('account', did, {'did': did},
                               f"/subcultures/{si}/recommended_accounts/{ai}",
                               {"is_anchor": account.get('is_anchor'),
                                "information_sources": account.get('information_sources', []),
                                "historical_role": account.get('account_role')},
                               {"type": "follow_account", "did": did}))
        for fi, uri in enumerate(scene['custom_feeds']):
            if not uri.startswith('at://did:') or '/app.bsky.feed.generator/' not in uri:
                raise ValueError(f'Invalid canonical feed URI: {uri}')
            candidates.append(('feed', uri, {'uri': uri}, f'/subcultures/{si}/custom_feeds/{fi}', {},
                               {"type": "pin_feed", "uri": uri}))
        for pi, url in enumerate(scene['starter_packs']):
            identity = pack_identity(url)
            candidates.append(('pack', url, identity, f'/subcultures/{si}/starter_packs/{pi}', {},
                               {"type": "inspect_pack", "source_url": url}))
        for kind, key, identity, pointer, evidence, action in candidates:
            entry_id = sha(f'{sid}\n{kind}\n{key}'.encode())[:24]
            entries.append({"entry_id": entry_id, "scene_id": sid, "kind": kind, "identity": identity,
                            "source_pointer": pointer, "historical_evidence": evidence,
                            "curator_identity": None, "editorial_approval": "pending",
                            "current_eligibility": "unknown", "release_eligible": False,
                            "proposed_action": action, "executable": False,
                            "blockers": BLOCKERS[:]})
    if len({e['entry_id'] for e in entries}) != len(entries):
        raise ValueError('Duplicate candidate within scene; review source before sealing')
    counts = {kind: sum(e['kind'] == kind for e in entries) for kind in ['account', 'feed', 'pack']}
    return {"schema": SCHEMA, "generation_id": 'editorial-review-v1-' + sha(source)[:16],
            "source": {"path": str(source_path), "sha256": sha(source),
                       "schema_version": catalog.get('schema_version'),
                       "historical_generated_at": catalog.get('generated_at')},
            "representation": "historical_editorial_catalog_not_independent_graph",
            "coverage": "Limited to the existing hand-selected catalog; no network representativeness or current quality claim.",
            "bundle_contract": {"default_account_count": 25, "maximum_account_count": 50,
                                "automatic_pack_follow": False, "hide_reposts_requires_explicit_user_choice": True},
            "serving_status": "review_only_not_release_eligible", "release_eligible_count": 0,
            "live_requests_performed": 0, "counts": counts,
            "unique_account_dids": len({e['identity']['did'] for e in entries if e['kind'] == 'account'}),
            "scenes": scenes, "entries": entries}


def write_new(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Refusing changed bytes at {path}')
    else:
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.source.read_bytes(), args.source)
    data = encode(manifest)
    prefix = 'editorial_baseline_v1_' + sha(data)[:16]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / (prefix + '.json')
    write_new(manifest_path, data)
    queue = [{"entry_id": e['entry_id'], "identity": e['identity'], "kind": e['kind'],
              "scene_id": e['scene_id'], "checks_required": e['blockers'], "status": "pending"}
             for e in manifest['entries']]
    queue_bytes = b''.join((json.dumps(row, sort_keys=True) + '\n').encode() for row in queue)
    queue_path = args.output_dir / (prefix + '.review.jsonl')
    write_new(queue_path, queue_bytes)
    receipt = {"schema": SCHEMA, "manifest": {"path": str(manifest_path), "sha256": sha(data)},
               "review_queue": {"path": str(queue_path), "sha256": sha(queue_bytes), "rows": len(queue)},
               "counts": manifest['counts'], "unique_account_dids": manifest['unique_account_dids'],
               "release_eligible_count": 0, "live_requests_performed": 0}
    write_new(args.output_dir / (prefix + '.receipt.json'), encode(receipt))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
