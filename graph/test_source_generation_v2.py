"""Synthetic protocol fixtures only: no model labels, post text, or quality evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from source_generation_v2 import freeze, fold, canonical, digest


class ProjectionTests(unittest.TestCase):
    def event(self, seq, kind='commit', source='east', **extra):
        e = dict(source_id=source, source_epoch='epoch1', seq=seq, witnessedAt=seq * 100,
                 did='did:test:a', kind=kind)
        if kind == 'commit':
            e.update(collection='app.bsky.graph.follow', rkey='follow1', rev=f'222222222222{seq}',
                     operation='create', record={'subject': 'did:test:b'})
        e.update(extra)
        return e

    def manifest(self, rows, audit=None):
        directory = Path(tempfile.mkdtemp(prefix='source-generation-protocol-'))
        inputs = []
        for name, events in [('east', rows), ('west', audit)]:
            if events is None:
                continue
            path = directory / f'{name}.jsonl'
            path.write_text(''.join(json.dumps(e) + '\n' for e in events))
            inputs.append(dict(path=str(path), source_id=name, source_epoch='epoch1',
                               sealed_tip_seq=100, role='authoritative' if name == 'east' else 'audit'))
        return freeze({'generation': 'fixture-v1', 'inputs': inputs})

    def test_replay_idempotence_revision_and_delete(self):
        e = self.event(3)
        deleted = self.event(4, operation='delete')
        del deleted['record']
        result = fold(self.manifest([e, self.event(2), e, deleted]))
        self.assertEqual(result['counts']['duplicate_events'], 1)
        self.assertEqual(result['counts']['current_records'], 0)
        self.assertEqual(result['counts']['record_tombstones'], 1)

    def test_sync_clears_old_records_accepts_replacement_revision(self):
        result = fold(self.manifest([self.event(5), self.event(6, 'sync', reset=True),
                                     self.event(7, rev='2222222222222')]))
        self.assertEqual(result['counts']['current_records'], 1)
        self.assertIn('sync_replacement_completeness_unverified', result['restrictions']['did:test:a'])

    def test_deleted_clears_but_suspended_preserves_restricted_state(self):
        for status, expected in [('deleted', 0), ('suspended', 1), ('takendown', 1)]:
            result = fold(self.manifest([self.event(2), self.event(3, 'account', active=False, status=status)]))
            self.assertEqual(result['counts']['current_records'], expected)
            self.assertIn(status, result['restrictions']['did:test:a'])

    def test_audit_does_not_supply_records_or_lift_restriction(self):
        audit = [self.event(2, 'account', 'west', active=False, status='suspended'),
                 self.event(3, 'account', 'west', active=True, status='active'), self.event(4, source='west')]
        result = fold(self.manifest([self.event(5, 'account', active=True, status='active')], audit))
        self.assertEqual(result['counts']['current_records'], 0)
        self.assertEqual(len(result['audit_restrictions']), 1)
        restriction = audit[0]
        receipt = dict(restriction_sha256=digest(canonical(restriction).encode()), did='did:test:a',
                       source_id='east', source_epoch='epoch1', observedAt=600, status='active',
                       evidence_sha256='a' * 64, method='authoritative_repo_reconciliation')
        manifest = self.manifest([self.event(5, 'account', active=True, status='active')], audit)
        released = fold(manifest, [receipt])
        self.assertEqual(released['audit_restrictions'], {})
        self.assertFalse(released['promotion_eligible'])
        receipt['observedAt'] = 100
        with self.assertRaises(ValueError):
            fold(manifest, [receipt])

    def test_hash_change_rejected(self):
        manifest = self.manifest([self.event(2)])
        Path(manifest['inputs'][0]['path']).write_text('{}\n')
        with self.assertRaises(ValueError):
            fold(manifest)

    def test_unknown_raw_schema_rejected(self):
        with self.assertRaises(ValueError):
            fold(self.manifest([{'kind': 'commit', 'commit': {}}]))

    def test_sequence_collision_rejected_equal_revision_conflict_reported(self):
        with self.assertRaises(ValueError):
            fold(self.manifest([self.event(2), self.event(2, record={'subject': 'did:test:c'})]))
        result = fold(self.manifest([self.event(2), self.event(3, rev='2222222222222', record={'subject': 'did:test:c'})]))
        self.assertEqual(result['counts']['conflicts'], 1)

    def test_old_revision_conflict_survives_newer_winner(self):
        rows = [self.event(2, rev='2222222222225'),
                self.event(3, rev='2222222222222'),
                self.event(4, rev='2222222222222', record={'subject': 'did:test:c'}),
                self.event(5, rev='2222222222222', record={'subject': 'did:test:c'})]
        result = fold(self.manifest(rows))
        self.assertEqual(result['counts']['conflicts'], 1)
        self.assertEqual((result['conflicts'][0]['first_seq'], result['conflicts'][0]['second_seq']), (3, 4))
        self.assertEqual(next(iter(result['records'].values()))['rev'], '2222222222225')
        self.assertFalse(result['promotion_eligible'])
        self.assertEqual(fold(self.manifest(list(reversed(rows))))['conflicts'], result['conflicts'])

    def test_revision_conflict_namespace_tracks_reset_and_source(self):
        rows = [self.event(2), self.event(3, 'sync', reset=True),
                self.event(4, rev='2222222222222', record={'subject': 'did:test:c'})]
        self.assertEqual(fold(self.manifest(rows))['counts']['conflicts'], 0)
        audit = [self.event(2, source='west'),
                 self.event(3, source='west', rev='2222222222222', record={'subject': 'did:test:c'})]
        result = fold(self.manifest(rows, audit))
        self.assertEqual(result['counts']['conflicts'], 1)
        self.assertEqual(result['conflicts'][0]['role'], 'audit')

    def test_tip_epoch_and_sync_guard(self):
        for e in [self.event(2, source_epoch='wrong'), self.event(101), self.event(2, 'sync', reset=False)]:
            with self.assertRaises(ValueError):
                fold(self.manifest([e]))

    def test_identity_invalidates_without_trusting_handle(self):
        result = fold(self.manifest([self.event(2, 'identity')]))
        self.assertTrue(result['identities']['did:test:a']['requires_resolution'])
        with self.assertRaises(ValueError):
            fold(self.manifest([self.event(2, 'identity', handle='untrusted.example')]))


if __name__ == '__main__':
    unittest.main()
