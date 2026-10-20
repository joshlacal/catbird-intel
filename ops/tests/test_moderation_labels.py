"""Structural label protocol fixtures, never moderation or model quality evidence."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import cbor2

spec = importlib.util.spec_from_file_location('label_backfill', Path(__file__).parents[1] / 'backfill_moderation_labels.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class LabelsTests(unittest.TestCase):
    def frame(self, seq=1, **fields):
        label = {'src': 'did:test:labeler', 'uri': 'did:test:account', 'val': '!hide',
                 'cts': '2026-01-01T00:00:00Z', 'neg': True, 'exp': '2027-01-01T00:00:00Z',
                 'cid': 'fixture-cid', 'sig': b'\x00\x01'}
        label.update(fields)
        return cbor2.dumps({'op': 1, 't': '#labels'}) + cbor2.dumps({'seq': seq, 'labels': [label]})

    def directory(self):
        d = Path(tempfile.mkdtemp(prefix='label-protocol-'))
        (d / 'raw').mkdir()
        return d

    def test_fields_preserved_and_replay_idempotent(self):
        raw = self.frame()
        decoded = m.decode_frame(raw, 'did:test:labeler')
        label = decoded['body']['labels'][0]
        self.assertEqual(label['sig'], {'$bytes': 'AAE='})
        self.assertTrue(label['neg'])
        self.assertIn('exp', label)
        self.assertIn('cid', label)
        d = self.directory()
        state = m.replay(d)
        self.assertTrue(m.save_frame(d, state, raw, decoded))
        self.assertFalse(m.save_frame(d, state, raw, decoded))
        self.assertEqual(m.replay(d), state)
        self.assertEqual(state['labels'], 1)
        self.assertEqual(state['negated_labels'], 1)
        self.assertEqual(next((d / 'raw').iterdir()).read_bytes(), raw)

    def test_notice_and_gap_are_retained(self):
        d = self.directory()
        state = m.replay(d)
        raw = cbor2.dumps({'op': 1, 't': '#info'}) + cbor2.dumps({'name': 'OutdatedCursor'})
        m.save_frame(d, state, raw, m.decode_frame(raw, 'did:test:labeler'))
        raw = self.frame(9)
        m.save_frame(d, state, raw, m.decode_frame(raw, 'did:test:labeler'))
        self.assertEqual(state['sequence_gaps'], [{'after': 0, 'before': 9}])
        self.assertEqual(state['notices'][0]['body']['name'], 'OutdatedCursor')

    def test_cursor_conflict_backward_and_hash_damage_fail(self):
        d = self.directory()
        state = m.replay(d)
        raw = self.frame(2)
        m.save_frame(d, state, raw, m.decode_frame(raw, 'did:test:labeler'))
        for raw in [self.frame(2, neg=False), self.frame(1)]:
            with self.assertRaises(ValueError):
                m.save_frame(d, state, raw, m.decode_frame(raw, 'did:test:labeler'))
        next((d / 'raw').iterdir()).write_bytes(b'bad')
        with self.assertRaises(ValueError):
            m.replay(d)

    def test_unknown_source_and_extra_cbor_rejected(self):
        for raw in [self.frame(src='did:test:other'), self.frame() + b'\x00']:
            with self.assertRaises(ValueError):
                m.decode_frame(raw, 'did:test:labeler')

    def test_journal_is_authority_after_checkpoint_stale(self):
        d = self.directory()
        state = m.replay(d)
        raw = self.frame()
        m.save_frame(d, state, raw, m.decode_frame(raw, 'did:test:labeler'))
        (d / 'checkpoint.json').write_text('{}')
        self.assertEqual(m.replay(d)['cursor'], 1)

    def test_partial_journal_stops_without_overwrite(self):
        d = self.directory()
        (d / 'journal.jsonl').write_bytes(b'{')
        with self.assertRaises(ValueError):
            m.replay(d)
        self.assertEqual((d / 'journal.jsonl').read_bytes(), b'{')

class PackedTests(unittest.TestCase):
    frame = LabelsTests.frame
    directory = LabelsTests.directory

    def manifest(self, d, cursor=0):
        base = m.compact_initial()
        base['cursor'] = cursor
        return {'schema': m.PACKED_SCHEMA, 'identity': {'did': 'did:test:labeler'},
                'base_state': base, 'target_seq': 100, 'parent': None}

    def test_packed_commit_resume_and_independent_decode(self):
        d = self.directory()
        manifest = self.manifest(d)
        store = m.PackedStore(d, manifest)
        after = json.loads(json.dumps(store.state))
        frames = [self.frame(1), self.frame(3)]
        for frame in frames:
            m.compact_apply(after, frame, m.decode_frame(frame, 'did:test:labeler'))
        store.commit(frames, after)
        recovered = m.PackedStore(d, manifest, verify=True)
        self.assertEqual(recovered.state, after)
        self.assertEqual(recovered.state['missing_sequence_count'], 1)
        self.assertEqual(recovered.chunk_count, 1)
        self.assertEqual(len(list((d / 'chunks').iterdir())), 1)

    def test_orphan_chunk_and_stale_checkpoint_safe(self):
        d = self.directory()
        manifest = self.manifest(d)
        store = m.PackedStore(d, manifest)
        (d / 'chunks' / 'orphan.frames.gz').write_bytes(b'preserved orphan')
        frame = self.frame()
        after = json.loads(json.dumps(store.state))
        m.compact_apply(after, frame, m.decode_frame(frame, 'did:test:labeler'))
        store.commit([frame], after)
        (d / 'checkpoint.json').write_text('{}')
        recovered = m.PackedStore(d, manifest, verify=True)
        self.assertEqual(recovered.state['cursor'], 1)
        self.assertTrue((d / 'chunks' / 'orphan.frames.gz').exists())

    def test_partial_journal_preserved_and_prefix_recovered(self):
        d = self.directory()
        manifest = self.manifest(d)
        store = m.PackedStore(d, manifest)
        frame = self.frame()
        after = json.loads(json.dumps(store.state))
        m.compact_apply(after, frame, m.decode_frame(frame, 'did:test:labeler'))
        store.commit([frame], after)
        with (d / 'chunks.jsonl').open('ab') as out:
            out.write(b'{partial')
        recovered = m.PackedStore(d, manifest, verify=True)
        self.assertEqual(recovered.state['cursor'], 1)
        self.assertEqual(len(list(d.glob('chunks-torn-*'))), 1)
        self.assertTrue(next(d.glob('chunks-torn-*')).read_bytes().endswith(b'{partial'))

    def test_packed_tamper_and_partial_lengths_fail(self):
        d = self.directory()
        manifest = self.manifest(d)
        store = m.PackedStore(d, manifest)
        frame = self.frame()
        after = json.loads(json.dumps(store.state))
        m.compact_apply(after, frame, m.decode_frame(frame, 'did:test:labeler'))
        store.commit([frame], after)
        next((d / 'chunks').iterdir()).write_bytes(b'corrupted')
        with self.assertRaises(ValueError):
            m.PackedStore(d, manifest)
        for raw in [b'x', b'\x00' * 8]:
            with self.assertRaises(ValueError):
                list(m.iter_packed(raw))

    def test_explicit_cursor_boundary_and_changed_duplicate(self):
        state = m.compact_initial()
        state['cursor'] = 10
        frame = self.frame(10)
        self.assertTrue(m.compact_apply(state, frame, m.decode_frame(frame, 'did:test:labeler')))
        self.assertFalse(m.compact_apply(state, frame, m.decode_frame(frame, 'did:test:labeler')))
        with self.assertRaises(ValueError):
            changed = self.frame(10, neg=False)
            m.compact_apply(state, changed, m.decode_frame(changed, 'did:test:labeler'))
        self.assertEqual(state['sequence_gap_count'], 0)

    def test_crash_after_raw_before_journal_replays_orphan(self):
        d = self.directory()
        manifest = self.manifest(d)
        store = m.PackedStore(d, manifest)
        frame = self.frame()
        after = json.loads(json.dumps(store.state))
        m.compact_apply(after, frame, m.decode_frame(frame, 'did:test:labeler'))
        original_open = Path.open
        def fail_journal(path, *args, **kwargs):
            if path.name == 'chunks.jsonl' and args and args[0] == 'ab':
                raise OSError('injected crash before journal')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', fail_journal):
            with self.assertRaises(OSError):
                store.commit([frame], after)
        recovered = m.PackedStore(d, manifest, verify=True)
        self.assertEqual(recovered.state['cursor'], 0)
        recovered.commit([frame], after)
        self.assertEqual(len(list((d / 'chunks').iterdir())), 1)
        self.assertEqual(m.PackedStore(d, manifest, verify=True).state['cursor'], 1)

    def test_crash_after_journal_before_checkpoint_recovers(self):
        d = self.directory()
        manifest = self.manifest(d)
        store = m.PackedStore(d, manifest)
        frame = self.frame()
        after = json.loads(json.dumps(store.state))
        m.compact_apply(after, frame, m.decode_frame(frame, 'did:test:labeler'))
        original_atomic = m.atomic
        def fail_checkpoint(path, data):
            if path.name == 'checkpoint.json':
                raise OSError('injected checkpoint failure')
            return original_atomic(path, data)
        with patch.object(m, 'atomic', fail_checkpoint):
            with self.assertRaises(OSError):
                store.commit([frame], after)
        self.assertEqual(m.PackedStore(d, manifest, verify=True).state['cursor'], 1)

    def test_target_crossing_is_not_committed(self):
        d = self.directory()
        identity = {'did': 'did:test:labeler', 'endpoint': 'https://example.invalid'}
        frames = iter([self.frame(1), self.frame(3)])
        class Socket:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def recv(self, **kwargs): return next(frames)
        args = SimpleNamespace(directory=str(d), verify_only=False, target_seq=2,
                               parent_v1=None, start_cursor=None, max_frames=10, max_seconds=10,
                               max_total_bytes=100000000, min_free_bytes=0, batch_frames=5,
                               frames_per_second=1000)
        with patch.object(m, 'discover', return_value=identity), patch.object(m, 'capture_policy', return_value={'fixture': True}), patch('websockets.sync.client.connect', return_value=Socket()), patch('builtins.print'):
            m.run_packed(args)
        receipt = json.loads(next(d.glob('run-*.json')).read_text())
        self.assertEqual(receipt['stop_reason'], 'target_crossed_target_event_unobserved')
        self.assertEqual(receipt['after']['cursor'], 1)
        self.assertEqual(receipt['after']['frames'], 1)
        self.assertEqual(len(list(d.glob('target-crossing-*.cbor'))), 1)

    def test_disconnects_with_durable_progress_do_not_exhaust_retry_budget(self):
        d = self.directory()
        identity = {'did': 'did:test:labeler', 'endpoint': 'https://example.invalid'}
        fixture = self
        next_seq = iter(range(1, 6))
        class Socket:
            def __init__(self, seq): self.seq, self.used = seq, False
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def recv(self, **kwargs):
                if self.used: raise OSError('injected transient disconnect')
                self.used = True
                return fixture.frame(self.seq)
        args = SimpleNamespace(directory=str(d), verify_only=False, target_seq=5,
                               parent_v1=None, start_cursor=None, max_frames=10, max_seconds=10,
                               max_total_bytes=100000000, min_free_bytes=0, batch_frames=1,
                               frames_per_second=1000)
        with patch.object(m, 'discover', return_value=identity), patch.object(m, 'capture_policy', return_value={'fixture': True}), patch('websockets.sync.client.connect', side_effect=lambda *a, **k: Socket(next(next_seq))), patch.object(m.time, 'sleep'), patch('builtins.print'):
            m.run_packed(args)
        receipt = json.loads(next(d.glob('run-*.json')).read_text())
        self.assertEqual(receipt['after']['cursor'], 5)
        self.assertEqual(receipt['total_disconnects'], 4)
        self.assertEqual(receipt['consecutive_no_progress_failures'], 0)


if __name__ == '__main__':
    unittest.main()
