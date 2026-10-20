import importlib.util
import json
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('editorial', pathlib.Path(__file__).parents[1] / 'prepare_editorial_baseline.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class EditorialBaselineTests(unittest.TestCase):
    def fixture(self):
        return json.dumps({'subcultures': [{'id': 'test', 'name': 'Fixture',
            'recommended_accounts': [{'did': 'did:plc:fixture', 'bio': 'must not be copied', 'is_anchor': True}],
            'custom_feeds': ['at://did:plc:fixture/app.bsky.feed.generator/a'],
            'starter_packs': ['https://bsky.app/starter-pack/example.test/a']}]}).encode()

    def test_historical_evidence_never_grants_release(self):
        result = m.build(self.fixture(), 'fixture.json')
        self.assertEqual(result['release_eligible_count'], 0)
        self.assertTrue(all(not e['executable'] and not e['release_eligible'] for e in result['entries']))
        self.assertNotIn('must not be copied', json.dumps(result))
        self.assertEqual([e['proposed_action']['type'] for e in result['entries']], ['follow_account', 'pin_feed', 'inspect_pack'])
        self.assertIsNone(result['entries'][2]['identity']['canonical_uri'])
        self.assertEqual(result['bundle_contract']['maximum_account_count'], 50)

    def test_source_bytes_affect_generation(self):
        a = m.build(self.fixture(), 'f')
        b = m.build(self.fixture() + b' ', 'f')
        self.assertNotEqual(a['generation_id'], b['generation_id'])

    def test_changed_bytes_never_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'out'
            m.write_new(path, b'original')
            m.write_new(path, b'original')
            with self.assertRaises(FileExistsError):
                m.write_new(path, b'changed')
            self.assertEqual(path.read_bytes(), b'original')

    def test_malformed_pack_rejected(self):
        with self.assertRaises(ValueError):
            m.pack_identity('https://evil.test/starter-pack/a/b')


if __name__ == '__main__':
    unittest.main()
