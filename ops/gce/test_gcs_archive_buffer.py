"""Offline GCS API/transaction fixtures; no cloud allocation or source-quality claims."""
import base64
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import contextlib
import io
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('gcs_buffer', Path(__file__).with_name('gcs_archive_buffer.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReferenceCRC32C:
    def __init__(self): self.crc = 0xffffffff
    def update(self, data):
        for byte in data:
            self.crc ^= byte
            for _ in range(8):
                self.crc = (self.crc >> 1) ^ (0x82f63b78 if self.crc & 1 else 0)
    def digest(self): return (self.crc ^ 0xffffffff).to_bytes(4, 'big')


def checked(path):
    return m.hashes(path, ReferenceCRC32C)


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name, self.metadata = bucket, name, {}
        self.generation, self.size, self.crc32c, self.md5_hash = None, 0, None, None
        self.soft = False
    def upload_from_filename(self, path, **kwargs):
        assert kwargs['if_generation_match'] == 0
        assert kwargs['checksum'] == 'crc32c'
        if self.bucket.get_blob(self.name): raise RuntimeError('precondition failed')
        expected = checked(Path(path))
        assert kwargs['crc32c_checksum_value'] == expected['crc32c']
        self.data = Path(path).read_bytes()
        self.size, self.crc32c, self.md5_hash = expected['bytes'], expected['crc32c'], expected['md5_hash']
        self.generation = str(len(self.bucket.objects) + 1)
        self.bucket.objects[(self.name, self.generation)] = self
        if self.bucket.after_upload: self.bucket.after_upload(self)
    def download_as_bytes(self, **kwargs):
        assert str(kwargs['if_generation_match']) == self.generation
        return self.data


class Bucket:
    def __init__(self): self.objects, self.after_upload = {}, None
    def blob(self, name): return Blob(self, name)
    def get_blob(self, name, generation=None):
        found = [b for (n, g), b in self.objects.items() if n == name and not b.soft and (generation is None or str(generation) == g)]
        return found[-1] if found else None


class Client:
    def __init__(self): self.b = Bucket()
    def bucket(self, name): return self.b
    def list_blobs(self, bucket, **options):
        return [b for b in self.b.objects.values() if b.soft == bool(options.get('soft_deleted'))]


class BufferTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='gcs-buffer-fixture-'))
        self.segments = self.root / 'segments'; self.segments.mkdir()
        self.work = self.root / 'work'; self.work.mkdir()
        self.args = SimpleNamespace(bucket='fixture-bucket', source_id='fixture', source_epoch='admission-1',
                                    work_dir=str(self.work), segments_dir=str(self.segments), run_hours=1,
                                    receipt_script='unused', max_bytes=m.TIB, max_segments=6)
        self.client = Client()
        self.buffer = self.new_buffer()
    def new_buffer(self):
        def verifier(path): return {**checked(path), **m.header(path)}
        return m.Buffer(self.client, self.args, verifier=verifier, hasher=checked)
    def segment(self, index, active=False):
        p = self.segments / f'seg_{index:010d}.jss'
        raw = bytearray(300)
        raw[:4] = b'jss0'
        struct.pack_into('<Q', raw, 4, 0 if active else 123)
        struct.pack_into('<H', raw, 12, 1)
        struct.pack_into('<QQqqQ', raw, 26, index * 10 + 1, index * 10 + 9, 100, 200, 256)
        p.write_bytes(raw)
        return p
    def test_reference_crc_known_vector(self):
        c = ReferenceCRC32C(); c.update(b'123456789')
        self.assertEqual(c.digest().hex(), 'e3069283')
    def test_sealed_upload_corrected_bounds_and_pair_generation(self):
        p = self.segment(1)
        self.buffer.offload(1, p)
        pair = next(iter(self.buffer.journal.pairs.values()))
        receipt = json.loads(self.client.b.get_blob(pair['receipt']['object']).data)
        self.assertEqual(receipt['min_seq'], 11)
        self.assertEqual(receipt['first_seq'], 11)
        self.assertEqual(receipt['max_seq'], 19)
        self.assertEqual(receipt['verification']['block_crc_validation'], 'unperformed')
        self.assertEqual(pair['payload']['generation'], receipt['gcs']['generation'])
        self.assertTrue(p.exists())
        self.assertEqual(list(self.work.glob('stage-*')), [])
    def test_idempotent_same_bytes_no_new_cloud_objects(self):
        p = self.segment(1)
        self.buffer.offload(1, p)
        first = dict(self.client.b.objects)
        self.buffer.offload(1, p)
        self.assertEqual(self.client.b.objects, first)
    def test_only_prune_oldest_after_pair_verified_keep_three(self):
        for index in range(5): self.segment(index, active=index == 4)
        status = self.buffer.cycle()
        self.assertEqual(status['pending_sealed'], 0)
        self.assertEqual(len(list(self.segments.glob('*.jss'))), 3)
        self.assertEqual(len(self.buffer.journal.released), 2)
        markers = [b for b in self.client.b.objects.values() if b.metadata.get('kind') == 'local_release']
        self.assertEqual(len(markers), 2)
        body = json.loads(markers[0].data)
        self.assertIn('payload_generation', body)
    def test_capacity_rejects_before_any_upload(self):
        self.args.max_bytes = 100
        p = self.segment(1)
        with self.assertRaises(m.Stop): self.buffer.offload(1, p)
        self.assertEqual(self.client.b.objects, {})
        self.assertTrue(p.exists())
    def test_soft_deleted_and_unrelated_objects_count_and_never_decrement(self):
        p = self.root / 'probe'; p.write_bytes(b'probe')
        for name in ['control/probe', 'old/payload']:
            b = self.client.b.blob(name)
            b.metadata = {'sha256': checked(p)['sha256']}
            b.upload_from_filename(str(p), if_generation_match=0, checksum='crc32c', crc32c_checksum_value=checked(p)['crc32c'])
        self.client.b.get_blob('old/payload').soft = True
        self.buffer.inventory()
        self.assertEqual(self.buffer.cloud_bytes, 10)
        self.client.b.objects.clear()
        self.buffer.inventory()
        self.assertEqual(self.buffer.journal.admitted_bytes, 10)
    def test_conflicting_existing_object_never_overwritten(self):
        p = self.segment(1); self.buffer.offload(1, p)
        payload = next(b for b in self.client.b.objects.values() if b.metadata['kind'] == 'payload')
        payload.md5_hash = 'bad'
        with self.assertRaises(m.Stop): self.buffer.offload(1, p)
        self.assertTrue(p.exists())
    def test_unjournaled_pair_reconciled(self):
        p = self.segment(1)
        original = self.buffer.journal.record
        def fail(row):
            if row['event'] == 'durable_pair': raise OSError('crash after cloud pair')
            return original(row)
        self.buffer.journal.record = fail
        with self.assertRaises(OSError): self.buffer.offload(1, p)
        recovered = self.new_buffer()
        recovered.reconcile(recovered.inventory())
        self.assertEqual(len(recovered.journal.pairs), 1)
        recovered.cleanup_stages()
        self.assertEqual(list(self.work.glob('stage-*')), [])
        self.assertTrue(p.exists())
    def test_payload_only_failure_keeps_source_and_resume_uploads_receipt(self):
        p = self.segment(1)
        calls = 0
        original = self.buffer.upload
        def fail(path, name, expected, metadata):
            nonlocal calls
            calls += 1
            if calls == 2: raise OSError('receipt transport failed')
            return original(path, name, expected, metadata)
        self.buffer.upload = fail
        with self.assertRaises(OSError): self.buffer.offload(1, p)
        self.assertTrue(p.exists())
        self.new_buffer().offload(1, p)
        self.assertEqual(len(self.client.b.objects), 2)
    def test_source_replaced_does_not_prune_new_bytes(self):
        p = self.segment(0); self.buffer.offload(0, p)
        p.write_bytes(p.read_bytes() + b'new generation')
        for index in range(1, 4): self.segment(index, active=index == 3)
        self.buffer.prune()
        self.assertTrue(p.exists())
        self.assertTrue(p.read_bytes().endswith(b'new generation'))
    def test_missing_cloud_pair_blocks_prune(self):
        p = self.segment(0); self.buffer.offload(0, p)
        for index in range(1, 4): self.segment(index, active=index == 3)
        receipt = next(k for k, b in self.client.b.objects.items() if b.metadata['kind'] == 'receipt')
        del self.client.b.objects[receipt]
        with self.assertRaises(m.Stop): self.buffer.prune()
        self.assertTrue(p.exists())
    def test_torn_journal_stops_preserving_bytes(self):
        path = self.work / 'journal.jsonl'; path.write_bytes(b'{partial')
        with self.assertRaises(m.Stop): self.new_buffer()
        self.assertEqual(path.read_bytes(), b'{partial')
    def test_dry_run_no_cloud_upload_no_local_prune(self):
        self.segment(0); self.segment(1, active=True)
        status = self.buffer.cycle(dry_run=True)
        self.assertEqual(status['pending_sealed'], 1)
        self.assertEqual(self.client.b.objects, {})
        self.assertEqual(len(list(self.segments.glob('*.jss'))), 2)
    def test_bounded_cycle_keeps_remaining_pending_nonzero(self):
        self.args.max_segments = 1
        for index in range(4): self.segment(index, active=index == 3)
        status = self.buffer.cycle()
        self.assertEqual(status['pending_sealed'], 2)
        self.assertEqual(len(self.buffer.journal.pairs), 1)
        # Cached inode/mtime of the preserved sealed pair is trusted; changed
        # source content invalidates it and becomes pending again.
        path = self.segments / 'seg_0000000001.jss'
        self.buffer.offload(1, path)
        self.assertEqual(len(self.buffer.scan_pending()[1]), 1)
        path.write_bytes(path.read_bytes() + b'new')
        self.assertEqual(len(self.buffer.scan_pending()[1]), 2)

    def test_nonterminal_active_rejected(self):
        self.segment(0, active=True); self.segment(1)
        with self.assertRaises(m.Stop): self.buffer.cycle()
    def test_symlink_rejected(self):
        p = self.segment(0)
        (self.segments / 'seg_0000000001.jss').symlink_to(p)
        with self.assertRaises(m.Stop): self.buffer.cycle()

    def test_authorized_72_hour_limit_and_2_5_tib_capacity(self):
        args = m.parse_args(['--bucket', 'fixture', '--source-epoch', 'admission-1',
                             '--max-bytes', str(5 * m.TIB // 2), '--run-hours', '72'])
        self.assertEqual(args.max_bytes, 2748779069440)
        self.assertEqual(args.run_hours, 72)
        for flag, value in [('--max-bytes', str(m.MAX_BYTES + 1)), ('--run-hours', '72.001'),
                            ('--run-hours', 'nan'), ('--poll-seconds', 'nan')]:
            with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    m.parse_args(['--bucket', 'fixture', '--source-epoch', 'admission-1', flag, value])

    def test_bootstrap_live_segments_never_admitted_or_pruned(self):
        live = self.root / 'bootstrap' / 'live_segments'
        live.mkdir(parents=True)
        retained = live / 'seg_0000000000.jss'
        retained.write_bytes(b'unique unmerged bootstrap bytes')
        self.args.segments_dir = str(live)
        with self.assertRaisesRegex(m.Stop, 'bootstrap_live_segments'):
            self.new_buffer()
        self.assertEqual(retained.read_bytes(), b'unique unmerged bootstrap bytes')
        self.assertEqual(self.client.b.objects, {})

    def test_status_preserves_cumulative_accounting_and_supervisor_identity(self):
        self.args.run_id = 'run-123'
        self.segment(0)
        self.buffer.cycle()
        self.client.b.objects.clear()
        self.buffer.inventory()
        status = self.buffer.status('ready', 0)
        self.assertEqual(status['run_id'], 'run-123')
        self.assertEqual(status['cloud_inventory_bytes'], 0)
        self.assertGreater(status['cumulative_admitted_bytes'], 0)
        self.assertEqual(status['admission_accounting'], 'cumulative_observed_generations_no_refund')


if __name__ == '__main__': unittest.main()
