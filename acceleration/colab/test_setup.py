"""Offline setup checks: no dependency installs, weight downloads, or model runs."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('colab_setup', HERE / 'setup_runtime.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def record(name, data=b'weights'):
    return {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, entries):
        path = self.root / 'models.tar.gz'
        with tarfile.open(path, 'w:gz') as bundle:
            for name, data, kind in entries:
                info = tarfile.TarInfo(name)
                info.type = kind
                if kind == tarfile.REGTYPE:
                    info.size = len(data)
                    bundle.addfile(info, io.BytesIO(data))
                else:
                    info.linkname = '../../escape'
                    bundle.addfile(info)
        return path

    def test_import_and_help_do_not_import_torch_or_setup(self):
        result = subprocess.run([sys.executable, str(HERE / 'setup_runtime.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--asset-cache', result.stdout)
        self.assertNotIn('torch', sys.modules)

    def test_explicit_run_required(self):
        with mock.patch.object(setup, 'hardware') as hardware, self.assertRaises(SystemExit):
            setup.main(['--checkout', '.', '--work-root', str(self.root / 'work'), '--asset-cache', str(self.root / 'cache')])
        hardware.assert_not_called()

    def test_selected_archive_members_only(self):
        key = 'data/pretrained/production/kcat/fold_0/model_0/model.pt'
        path = self.archive([(key[5:], b'weights', tarfile.REGTYPE),
                             ('pretrained/production/km/unselected.pt', b'no', tarfile.REGTYPE)])
        setup.extract_checkpoints(path, self.root / 'cache', [record(key)])
        self.assertTrue(setup.verified(self.root / 'cache' / key, record(key)))
        self.assertFalse((self.root / 'cache/pretrained/production/km/unselected.pt').exists())

    def test_archive_rejects_unsafe_names_and_links(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('/tmp/escape', tarfile.REGTYPE),
                           ('pretrained/link', tarfile.SYMTYPE), ('pretrained/link', tarfile.LNKTYPE)]:
            with self.subTest(name=name, kind=kind):
                path = self.archive([(name, b'x', kind)])
                with self.assertRaises(ValueError):
                    setup.extract_checkpoints(path, self.root / 'cache', [record('data/model.pt')])
        self.assertFalse((self.root / 'escape').exists())

    def test_archive_hash_failure_never_promotes(self):
        path = self.archive([('model.pt', b'badfile', tarfile.REGTYPE)])
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            setup.extract_checkpoints(path, self.root / 'cache', [record('model.pt')])
        self.assertFalse((self.root / 'cache/model.pt').exists())

    def test_archive_duplicate_checkpoint_rejected(self):
        path = self.archive([('data/model.pt', b'weights', tarfile.REGTYPE),
                             ('model.pt', b'weights', tarfile.REGTYPE)])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            setup.extract_checkpoints(path, self.root / 'cache', [record('data/model.pt')])

    def test_download_verified_cache_skips_network(self):
        target = self.root / 'model.pt'
        target.write_bytes(b'weights')
        with mock.patch.object(setup.urllib.request, 'urlopen') as open_url:
            setup.download('https://example.test/model.pt', target, record('model.pt'))
        open_url.assert_not_called()

    def test_download_resumes_partial_and_checks_hash(self):
        target = self.root / 'model.pt'
        target.with_suffix('.pt.part').write_bytes(b'wei')
        with mock.patch.object(setup.urllib.request, 'urlopen', return_value=Response(
                b'ghts', 206, {'Content-Range': 'bytes 3-6/7'})) as open_url:
            setup.download('https://example.test/model.pt', target, record('model.pt'))
        self.assertEqual(open_url.call_args.args[0].get_header('Range'), 'bytes=3-')
        self.assertEqual(target.read_bytes(), b'weights')
        self.assertFalse(target.with_suffix('.pt.part').exists())

    def test_download_restarts_if_server_ignores_range(self):
        target = self.root / 'model.pt'
        target.with_suffix('.pt.part').write_bytes(b'wei')
        with mock.patch.object(setup.urllib.request, 'urlopen', return_value=Response(b'weights')):
            setup.download('https://example.test/model.pt', target, record('model.pt'))
        self.assertEqual(target.read_bytes(), b'weights')

    def test_download_bad_hash_preserves_existing_file(self):
        target = self.root / 'model.pt'
        target.write_bytes(b'old')
        with mock.patch.object(setup.urllib.request, 'urlopen', return_value=Response(b'bad')):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                setup.download('https://example.test/model.pt', target, record('model.pt'))
        self.assertEqual(target.read_bytes(), b'old')
        self.assertFalse(target.with_suffix('.pt.part').exists())

    def test_download_bad_range_does_not_append(self):
        target = self.root / 'model.pt'
        partial = target.with_suffix('.pt.part')
        partial.write_bytes(b'wei')
        with mock.patch.object(setup.urllib.request, 'urlopen', return_value=Response(
                b'bad', 206, {'Content-Range': 'bytes 1-3/7'})):
            with self.assertRaisesRegex(ValueError, 'range'):
                setup.download('https://example.test/model.pt', target, record('model.pt'))
        self.assertEqual(partial.read_bytes(), b'wei')
        self.assertFalse(target.exists())

    def test_all_cached_assets_skip_downloads_and_preserve_hashes(self):
        accel = self.root / 'acceleration'
        (accel / 'source').mkdir(parents=True)
        cache = self.root / 'cache'
        models = [record(f'data/pretrained/production/kcat/fold_0/model_{i}/model.pt') for i in range(10)]
        for item in models:
            target = cache / 'checkpoints' / item['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'weights')
        esm = []
        for name in ('esm2_t33_650M_UR50D.pt', 'esm2_t33_650M_UR50D-contact-regression.pt'):
            item = record(name)
            item['name'] = item.pop('path')
            esm.append(item)
            target = cache / 'esm' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'weights')
        (accel / 'source/prior_checkpoint_manifest.json').write_text(json.dumps({'files': models}))
        (accel / 'source/prior_esm_checkpoint_manifest.json').write_text(json.dumps(esm))
        with mock.patch.object(setup.urllib.request, 'urlopen') as open_url:
            actual = setup.prepare_assets(accel, self.root / 'work', cache)
        open_url.assert_not_called()
        self.assertEqual(actual, (models, esm))
        self.assertTrue(all(setup.verified(self.root / 'work' / r['path'], r) for r in models))
        self.assertTrue(all(setup.verified(self.root / 'work/torch/hub/checkpoints' / r['name'], r) for r in esm))
        self.assertFalse((cache / 'downloads').exists())

    def test_symlink_parent_rejected(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (self.root / 'linked').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            setup.safe_path(self.root, 'linked/model.pt')

    def fixture_checkout(self):
        checkout = self.root / 'checkout'
        accel = checkout / 'acceleration'
        path = accel / 'source/CatPred/core.py'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'unchanged source')
        data = {'copied_files': [record('source/CatPred/core.py', b'unchanged source'),
                                 record('source/inputs/private.csv', b'do not copy')]}
        (accel / 'SOURCE_MANIFEST.json').write_text(json.dumps(data))
        return checkout

    def test_staging_preserves_bytes_and_omits_private_inputs(self):
        checkout = self.fixture_checkout()
        work = self.root / 'work'
        setup.stage_checkout(checkout, work)
        setup.stage_checkout(checkout, work)
        self.assertEqual((work / 'CatPred/core.py').read_bytes(), b'unchanged source')
        self.assertFalse((work / 'inputs').exists())
        (work / 'CatPred/core.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Existing file differs'):
            setup.stage_checkout(checkout, work)

    def test_unowned_work_root_rejected_without_changes(self):
        checkout = self.fixture_checkout()
        work = self.root / 'work'
        work.mkdir()
        (work / 'precious.txt').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'empty work root'):
            setup.stage_checkout(checkout, work)
        self.assertEqual((work / 'precious.txt').read_text(), 'keep')
        self.assertFalse((work / setup.OWNER).exists())

    def test_different_snapshot_rejected(self):
        checkout = self.fixture_checkout()
        work = self.root / 'work'
        setup.stage_checkout(checkout, work)
        (work / setup.OWNER).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'different source snapshot'):
            setup.stage_checkout(checkout, work)

    def test_invalid_source_does_not_claim_directory(self):
        checkout = self.fixture_checkout()
        (checkout / 'acceleration/source/CatPred/core.py').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
            setup.stage_checkout(checkout, self.root / 'work')
        self.assertFalse((self.root / 'work').exists())


if __name__ == '__main__':
    unittest.main()
