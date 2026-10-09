"""Offline input, evidence, scheduling and notebook checks; no model execution."""
import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('colab_comparison', HERE / 'compare.py')
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)
try:
    import numpy as np
except ImportError:
    np = None


class InputTests(unittest.TestCase):
    def test_order_duplicates_and_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'input.csv', Path(folder) / 'staged.csv'
            source.write_text('SMILES,sequence,tag,row_id\nCC,AAAA,first,old\nC,CCCC,second,old\nCC,AAAA,third,old\n')
            rows = comparison.prepare_input(source, target)
            self.assertEqual([r['tag'] for r in rows], ['first', 'second', 'third'])
            self.assertEqual([r['SMILES'] for r in rows], ['CC', 'C', 'CC'])
            self.assertEqual([r['row_id'] for r in rows], ['row_000000', 'row_000001', 'row_000002'])
            self.assertEqual(rows[0]['pdbpath'], rows[2]['pdbpath'])
            self.assertNotEqual(rows[0]['pdbpath'], rows[1]['pdbpath'])
            with target.open(newline='') as stream:
                self.assertEqual(list(csv.DictReader(stream)), rows)

    def test_malformed_inputs_rejected(self):
        cases = ['SMILES,sequence,sequence\nC,A,A\n', 'SMILES,sequence\nC\n',
                 'SMILES,sequence\nC,A,extra\n', 'SMILES,sequence\nC,\n',
                 'SMILES,sequence\n', 'SMILES,sequence\n' + 'C,A\n' * 16385]
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'input.csv', Path(folder) / 'staged.csv'
            for text in cases:
                with self.subTest(text=text[:40]):
                    source.write_text(text)
                    with self.assertRaises(ValueError):
                        comparison.prepare_input(source, target)

    def test_bom_and_supplied_pdbpath(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'input.csv', Path(folder) / 'staged.csv'
            source.write_text('\ufeffSMILES,sequence,pdbpath\nC,AAAA,original.pdb\n')
            self.assertEqual(comparison.prepare_input(source, target)[0]['pdbpath'], 'original.pdb')


@unittest.skipIf(np is None, 'NumPy is not installed; no dependency is downloaded')
class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, arrays=None, change_metadata=False):
        arrays = arrays or {'member_raw': np.zeros((10, 2, 2), dtype=np.float32),
                            'numeric_raw': np.zeros((2, 4), dtype=np.float64),
                            'numeric_processed': np.zeros((2, 4), dtype=np.float64)}
        path = self.root / (name + '.npz')
        np.savez(path, **arrays)
        manifest = self.root / (name + '.json')
        metadata = {'row_ids': ['r0', 'r1'], 'stages': {key: {'shape': list(value.shape), 'dtype': value.dtype.str} for key, value in arrays.items()}}
        if change_metadata:
            metadata['row_ids'].reverse()
        manifest.write_text(json.dumps(metadata))
        return {'precision_path': str(path), 'precision_manifest_path': str(manifest)}

    def test_identical_bits_accepted(self):
        self.assertTrue(comparison.compare_precision(self.write('a'), self.write('b'))['exact_bits'])

    def test_signed_zero_rejected(self):
        a = self.write('a')
        with np.load(a['precision_path']) as source:
            arrays = {key: source[key].copy() for key in comparison.ARRAYS}
        arrays['member_raw'][0, 0, 0] = -0.0
        self.assertFalse(comparison.compare_precision(a, self.write('b', arrays))['exact_bits'])

    def test_dtype_shape_metadata_and_nonfinite_rejected(self):
        for kind in ('dtype', 'shape', 'metadata', 'nan', 'inf'):
            with self.subTest(kind=kind):
                a = self.write('a')
                with np.load(a['precision_path']) as source:
                    arrays = {key: source[key].copy() for key in comparison.ARRAYS}
                if kind == 'dtype':
                    arrays['member_raw'] = arrays['member_raw'].astype(np.float64)
                elif kind == 'shape':
                    arrays['member_raw'] = arrays['member_raw'].reshape(10, 1, 4)
                elif kind in ('nan', 'inf'):
                    arrays['member_raw'][0, 0, 0] = float(kind)
                b = self.write('b', arrays, change_metadata=kind == 'metadata')
                self.assertFalse(comparison.compare_precision(a, b)['exact_bits'])


class SchedulingTests(unittest.TestCase):
    def test_two_arms_alternate(self):
        arms = ['Original', 'S_STREAM']
        self.assertEqual([comparison.order_for_round(arms, n) for n in range(4)], [arms, arms[::-1], arms, arms[::-1]])

    def test_three_arms_each_occupy_every_position(self):
        orders = [comparison.order_for_round(comparison.ARMS, n) for n in range(3)]
        self.assertTrue(all(set(column) == set(comparison.ARMS) for column in zip(*orders)))

    def execute_fake(self, mismatch=None, unavailable_k1=False, variants='all', k1='auto'):
        events = []
        class UnsupportedKinetics(RuntimeError):
            pass
        class Worker:
            def __init__(self, setup, source, output):
                self.out = output
                self.rows = [{}, {}]
                self.torch = types.SimpleNamespace(get_num_threads=lambda: 24, get_num_interop_threads=lambda: 1,
                                                  cuda=types.SimpleNamespace(empty_cache=lambda: None))
            def prepare(self, repeats=3):
                return {'esm_seconds': 1, 'model_load_seconds': 1, 'unique_sequences': 1}
            def run(self, arm, phase, repetition):
                events.append(('run', arm, phase, repetition))
                if unavailable_k1 and arm == 'S_STREAM_K1':
                    raise UnsupportedKinetics('No matching certificate')
                dest = self.out / f'{phase}_{arm}_{repetition}'
                dest.mkdir()
                return {'arm': arm, 'phase': phase, 'repetition': repetition, 'seconds': 2.0 if arm == 'Original' else 1.0,
                        'precision_path': str(dest / 'precision.npz'), 'prediction_path': str(dest / 'prediction.csv')}
            def reset(self):
                pass
            def close(self):
                events.append(('close',))
        def verify(left, right):
            events.append(('verify', right['arm'], right['phase'], right['repetition']))
            return {'exact_bits': mismatch != (right['arm'], right['phase'], right['repetition'])}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            setup, source, output = root / 'setup.json', root / 'input.csv', root / 'result'
            setup.write_text(json.dumps({'hardware': {}}));source.write_text('SMILES,sequence\nC,A\n')
            options = types.SimpleNamespace(setup=setup, input=source, output=output, repeats=2, k1=k1, variants=variants)
            error = None
            console = io.StringIO()
            with patch.object(comparison, 'Comparison', Worker), patch.object(comparison, 'compare_precision', verify), contextlib.redirect_stdout(console):
                try:
                    comparison.execute(options)
                except RuntimeError as failure:
                    error = failure
            self.last_stdout = console.getvalue()
            return events, json.loads((output / 'summary.json').read_text()), error

    def test_all_gates_precede_timing(self):
        events, summary, error = self.execute_fake()
        self.assertIsNone(error)
        first_warm = next(i for i, event in enumerate(events) if len(event) > 2 and event[2] == 'warm')
        gate_checks = [event for event in events[:first_warm] if event[0] == 'verify']
        self.assertEqual(len(gate_checks), 5)
        self.assertEqual(summary['status'], 'COMPLETE')
        self.assertTrue(summary['all_reported_timings_passed_exact_bits'])
        self.assertEqual(summary['arms']['S_STREAM']['speedup_vs_original'], 2)

    def test_original_discrepancy_stops_before_timing(self):
        events, summary, error = self.execute_fake(('Original', 'gate', 1))
        self.assertIsNotNone(error)
        self.assertFalse(any(len(event) > 2 and event[2] == 'warm' for event in events))
        self.assertEqual(summary['status'], 'FAILED')
        self.assertFalse(any('median_seconds' in arm for arm in summary['arms'].values()))

    def test_warm_discrepancy_publishes_no_timing_summary(self):
        events, summary, error = self.execute_fake(('S_STREAM', 'warm', 0))
        self.assertIsNotNone(error)
        self.assertEqual(summary['status'], 'FAILED')
        self.assertFalse(any('median_seconds' in arm for arm in summary['arms'].values()))

    def test_unavailable_k1_is_not_counted_as_speedup(self):
        events, summary, error = self.execute_fake(unavailable_k1=True)
        self.assertIsNone(error)
        self.assertEqual(summary['arms']['S_STREAM_K1']['status'], 'unavailable')
        self.assertNotIn('speedup_vs_original', summary['arms']['S_STREAM_K1'])
        self.assertFalse(any(event[:3] == ('run', 'S_STREAM_K1', 'warm') for event in events))


    def test_best_success_runs_only_original_and_complete_optimized(self):
        events, summary, error = self.execute_fake(variants='best')
        self.assertIsNone(error)
        self.assertEqual(summary['status'], 'COMPLETE')
        self.assertEqual(summary['variants'], 'best')
        self.assertEqual(set(summary['arms']), {'Original', 'S_STREAM_K1'})
        runs = [event for event in events if event[0] == 'run']
        self.assertFalse(any(event[1] == 'S_STREAM' for event in runs))
        first_warm = next(i for i, event in enumerate(events) if len(event) > 2 and event[2] == 'warm')
        self.assertEqual([event[1:] for event in events[:first_warm] if event[0] == 'verify'],
                         [('Original', 'gate', 1), ('S_STREAM_K1', 'gate', 0), ('S_STREAM_K1', 'gate', 1)])
        self.assertEqual([event[1] for event in runs if event[2] == 'warm'],
                         ['Original', 'S_STREAM_K1', 'S_STREAM_K1', 'Original'])
        self.assertEqual(summary['display_names'], {'Original': 'Original', 'S_STREAM_K1': 'Optimized'})
        self.assertIn('Optimized, repeat 1:', self.last_stdout)
        self.assertNotIn('S_STREAM', self.last_stdout)

    def test_best_unavailable_does_not_fall_back_or_time_original(self):
        events, summary, error = self.execute_fake(variants='best', unavailable_k1=True)
        self.assertIsNone(error)
        self.assertEqual(summary['status'], 'UNAVAILABLE')
        self.assertEqual(summary['arms']['S_STREAM_K1']['status'], 'unavailable')
        self.assertEqual(summary['arms']['Original']['status'], 'verified')
        self.assertFalse(summary['timings_run'])
        self.assertFalse(any(len(event) > 2 and (event[1] == 'S_STREAM' or event[2] == 'warm') for event in events))
        self.assertFalse(any('speedup_vs_original' in arm for arm in summary['arms'].values()))
        self.assertNotIn('S_STREAM', self.last_stdout)

    def test_best_candidate_discrepancy_stops_before_any_timing(self):
        events, summary, error = self.execute_fake(('S_STREAM_K1', 'gate', 0), variants='best')
        self.assertIsNone(error)
        self.assertEqual(summary['status'], 'REJECTED')
        self.assertEqual(summary['arms']['S_STREAM_K1']['status'], 'rejected')
        self.assertFalse(any(len(event) > 2 and event[2] == 'warm' for event in events))
        self.assertFalse(any('median_seconds' in arm for arm in summary['arms'].values()))

    def test_best_original_discrepancy_stops_before_candidate(self):
        events, summary, error = self.execute_fake(('Original', 'gate', 1), variants='best')
        self.assertIsNotNone(error)
        self.assertEqual(summary['status'], 'FAILED')
        self.assertEqual({event[1] for event in events if event[0] == 'run'}, {'Original'})
        self.assertFalse(any(len(event) > 2 and event[2] == 'warm' for event in events))

    def test_best_disabled_candidate_cannot_be_labeled_optimized(self):
        events, summary, error = self.execute_fake(variants='best', k1='off')
        self.assertIsNone(error)
        self.assertEqual(summary['status'], 'UNAVAILABLE')
        self.assertEqual(summary['arms']['S_STREAM_K1']['status'], 'disabled')
        self.assertFalse(any(len(event) > 2 and event[2] == 'warm' for event in events))
        self.assertFalse(any('speedup_vs_original' in arm for arm in summary['arms'].values()))

    def test_cli_defaults_to_best_and_all_remains_explicit(self):
        base = ['compare.py', '--setup', 'setup.json', '--input', 'input.csv', '--output', 'result']
        for extra, expected in (([], 'best'), (['--variants', 'all'], 'all')):
            with patch.object(sys, 'argv', base + extra), patch.object(comparison, 'execute') as execute:
                comparison.main()
                self.assertEqual(execute.call_args[0][0].variants, expected)


@unittest.skipIf(np is None, 'NumPy is not installed')
class ResumeTests(unittest.TestCase):
    def fixture(self, root):
        esm = root / 'esm'
        (esm / 'raw').mkdir(parents=True)
        (esm / 'records').mkdir()
        sequence = 'AA'
        array = np.zeros((2, 1280), dtype=np.float32)
        archive = esm / 'raw/features.npz'
        np.savez_compressed(archive, feature_0000=array)
        entries = [{'key': 'feature_0000', 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
                    'shape': [2, 1280], 'dtype': array.dtype.str, 'bytes': array.nbytes,
                    'sha256': hashlib.sha256(array.tobytes()).hexdigest()}]
        evidence = {'archive': 'raw/features.npz', 'archive_sha256': comparison.sha(archive),
                    'content_sha256': hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
                    'entries': entries}
        expected = [('Original', 'gate', 0), ('Original', 'gate', 1), ('Optimized', 'gate', 0), ('Optimized', 'gate', 1)]
        for repeat in range(3):
            expected.extend((arm, 'timed', repeat) for arm in comparison.order_for_round(('Original', 'Optimized'), repeat))
        names = []
        for arm, phase, repeat in expected:
            name = f'records/{phase}_{arm}_{repeat}.json'
            names.append(name)
            record = {'arm': arm, 'phase': phase, 'repetition': repeat, 'status': 'PASSED', 'exact_bits': True,
                      'seconds': 2. if arm == 'Original' else 1., 'features': evidence,
                      'restoration': {'functions_restored': True, 'model_released': True},
                      'actual_forward_batches': [{'sequences': [entries[0]['sequence_sha256']], 'decision': {'used_mode': 'off'}}],
                      'protein_backend': {'closed': True, 'model_released': True, 'loader_receipts': [
                          {'used_mode': 'meta', 'engaged': True, 'fallback_reason': None}]}}
            (esm / name).write_text(json.dumps(record))
        summary = {'status': 'COMPLETE', 'all_exact_bits': True, 'repetitions': 3, 'records': names,
                   'canonical_record': names[0], 'arms': {'Original': {'seconds': [2.] * 3, 'median_seconds': 2.},
                                                        'Optimized': {'seconds': [1.] * 3, 'median_seconds': 1.}}}
        (esm / 'summary.json').write_text(json.dumps(summary))
        return esm

    def test_verifies_all_retained_esm_samples_without_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            summary, canonical, reference = comparison.verify_saved_esm(root, ['AA'], 3)
            self.assertEqual(len(summary['records']), 10)
            self.assertEqual(canonical['arm'], 'Original')
            self.assertEqual(reference['AA'], bytes(2 * 1280 * 4))

    def test_changed_raw_receipt_order_or_times_reject_resume(self):
        for change in ('raw', 'order', 'times', 'fallback'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                esm = self.fixture(root)
                if change == 'raw':
                    (esm / 'raw/features.npz').write_bytes(b'corrupt')
                elif change == 'fallback':
                    path = esm / 'records/gate_Optimized_0.json'
                    value = json.loads(path.read_text())
                    value['protein_backend']['loader_receipts'][0]['used_mode'] = 'off'
                    path.write_text(json.dumps(value))
                else:
                    path = esm / 'summary.json'
                    value = json.loads(path.read_text())
                    if change == 'order':
                        value['records'].reverse()
                    else:
                        value['arms']['Original']['median_seconds'] = 0.1
                    path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    comparison.verify_saved_esm(root, ['AA'], 3)

    def test_archive_preserves_old_attempt_and_keeps_esm_in_place(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            esm = self.fixture(root)
            retained = comparison.sha(esm / 'summary.json')
            (root / 'summary.json').write_text('{"status":"UNAVAILABLE"}')
            gate = root / 'gate_Original_0'
            gate.mkdir()
            (gate / 'raw.npz').write_bytes(b'old raw evidence')
            (root / 'input.csv').write_text('input preserved')
            attempt = root / comparison.archive_attempt(root)
            self.assertEqual((attempt / 'gate_Original_0/raw.npz').read_bytes(), b'old raw evidence')
            self.assertEqual(comparison.sha(esm / 'summary.json'), retained)
            self.assertTrue((root / 'input.csv').is_file())
            self.assertFalse(gate.exists())
            self.assertEqual(comparison.archive_attempt(root), 'attempts/attempt_1')

    def test_resume_cli_is_explicit(self):
        args = ['compare.py', '--setup', 'setup.json', '--input', 'input.csv', '--output', 'result', '--resume-verified-esm']
        with patch.object(sys, 'argv', args), patch.object(comparison, 'execute') as execute:
            comparison.main()
            self.assertTrue(execute.call_args[0][0].resume_verified_esm)


class NotebookTests(unittest.TestCase):
    def notebook(self):
        return json.loads((HERE.parents[1] / 'colab_compare.ipynb').read_text())

    def test_code_compiles_and_public_default_exists(self):
        notebook = self.notebook()
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'<cell-{i}>', 'exec')
        source = ''.join(notebook['cells'][3]['source'])
        self.assertIn('input_mode = "Bundled example"', source)
        self.assertIn('repeat_count = 1', source)
        self.assertTrue((HERE.parents[1] / 'demo/batch_kcat.csv').is_file())

    def test_gitless_override_requires_verified_manifest(self):
        setup = ast.parse(''.join(self.notebook()['cells'][1]['source']))
        branch = next(node for node in setup.body if isinstance(node, ast.If) and 'CHECKOUT.exists() and' in ast.unparse(node.test))
        code = compile(ast.Module(body=[branch], type_ignores=[]), '<checkout-check>', 'exec')
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder);accel = checkout / 'acceleration';accel.mkdir()
            source = accel / 'source.txt';source.write_text('source')
            manifest = accel / 'SOURCE_MANIFEST.json'
            manifest.write_text(json.dumps({'copied_files': [{'path': 'source.txt', 'sha256': comparison.sha(source)}]}))
            bundle_files = []
            for name in ('colab_compare.ipynb', 'demo/batch_kcat.csv', 'acceleration/colab/setup_runtime.py', 'acceleration/colab/compare.py'):
                target = checkout / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('fixture')
            for target in checkout.rglob('*'):
                if target.is_file():
                    bundle_files.append({'path': str(target.relative_to(checkout)), 'sha256': comparison.sha(target)})
            (checkout / 'source_bundle_manifest.json').write_text(json.dumps({'schema': 1, 'checkout_commit': '0' * 40, 'files': bundle_files}))
            values = dict(CHECKOUT=checkout, checkout_override=str(checkout), commit='', Path=Path, hashlib=hashlib, json=json, re=re)
            exec(code, values)
            self.assertEqual(values['ACTUAL_COMMIT'], '0' * 40)
            self.assertEqual(values['SOURCE_MANIFEST_SHA'], comparison.sha(manifest))
            with self.assertRaises(RuntimeError):
                exec(code, dict(values, checkout_override=None))
            source.write_text('changed')
            with self.assertRaises(RuntimeError):
                exec(code, values)

    def test_import_has_no_framework_or_io_side_effects(self):
        script = '''import builtins, importlib.util, pathlib, sys
before = set(pathlib.Path('.').iterdir())
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
 if name.split('.')[0] in {'torch','numpy','pandas','catpred','google','requests','urllib'}:
  raise AssertionError('Unexpected import: ' + name)
 return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
spec=importlib.util.spec_from_file_location('compare_import',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert set(pathlib.Path('.').iterdir()) == before
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, '-B', '-c', script, str(HERE / 'compare.py')], cwd=folder,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
