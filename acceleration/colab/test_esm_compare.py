"""Offline ESM orchestration and exact evidence tests; no models or downloads."""
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('esm_comparison', HERE / 'esm_compare.py')
esm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(esm)


class FakeArray:
    def __init__(self, payload, shape):
        self.payload, self.shape = payload, shape
        self.dtype = types.SimpleNamespace(str='<f4')
    def tobytes(self, order):
        assert order == 'C'
        return self.payload


class FakeTensor:
    def __init__(self, sequence, *, negative_zero=False, nonfinite=False):
        values = [0.] * (min(len(sequence), 2047) * 1280)
        values[0] = float('nan') if nonfinite else -0. if negative_zero else 0.
        self.array = FakeArray(struct.pack('<' + str(len(values)) + 'f', *values),
                               (min(len(sequence), 2047), 1280))
        self.dtype, self.device = 'float32', types.SimpleNamespace(type='cpu')
    def numpy(self):
        return self.array


def fake_torch():
    def finite(tensor):
        valid = all(math.isfinite(value[0]) for value in struct.iter_unpack('<f', tensor.array.payload))
        return types.SimpleNamespace(all=lambda: types.SimpleNamespace(item=lambda: valid))
    return types.SimpleNamespace(float32='float32', isfinite=finite,
        cuda=types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None))


class PrecisionTests(unittest.TestCase):
    def test_signed_zero_and_metadata_are_part_of_exact_gate(self):
        torch = fake_torch()
        original = esm._snapshot({'AA': FakeTensor('AA')}, ['AA'], torch)
        self.assertTrue(esm._exact(original, esm._snapshot({'AA': FakeTensor('AA')}, ['AA'], torch)))
        candidate = esm._snapshot({'AA': FakeTensor('AA', negative_zero=True)}, ['AA'], torch)
        self.assertFalse(esm._exact(original, candidate))
        candidate = copy.deepcopy(original)
        candidate['entries'][0]['dtype'] = '>f4'
        self.assertFalse(esm._exact(original, candidate))
        candidate = copy.deepcopy(original)
        candidate['entries'][0]['shape'] = [1280, 2]
        self.assertFalse(esm._exact(original, candidate))

    def test_nonfinite_dtype_device_order_and_residue_shape_rejected(self):
        for kind in ('nonfinite', 'dtype', 'device', 'order', 'shape'):
            with self.subTest(kind=kind):
                tensor = FakeTensor('AA', nonfinite=kind == 'nonfinite')
                if kind == 'dtype':
                    tensor.dtype = 'float64'
                if kind == 'device':
                    tensor.device.type = 'cuda'
                if kind == 'shape':
                    tensor.array.shape = (1, 2560)
                with self.assertRaises(ValueError):
                    esm._snapshot({'AA': tensor}, ['BB'] if kind == 'order' else ['AA'], fake_torch())


class FakeRunner:
    def __init__(self, mismatch=None, unavailable=False):
        self.events, self.saved = [], []
        self.mismatch, self.unavailable = mismatch, unavailable
    def run(self, arm, phase, repetition):
        self.events.append((arm, phase, repetition))
        if self.unavailable and arm == 'Optimized':
            raise RuntimeError('Optimized ESM loader did not engage meta construction')
        value = b'changed' if self.mismatch == (arm, phase, repetition) else b'original'
        return {'features': {'AA': 'canonical'},
                'snapshot': {'entries': [{'shape': [2, 1280], 'dtype': '<f4'}], 'raw': [value]},
                'record': {'arm': arm, 'phase': phase, 'repetition': repetition,
                           'seconds': 4. if arm == 'Original' else 2.}}
    def persist(self, result, exact):
        self.saved.append((result['record'], exact))


class ScheduleTests(unittest.TestCase):
    def test_two_original_two_optimized_gates_then_ab_ba_ab(self):
        runner = FakeRunner()
        reference, arms, records = esm._execute(runner, 3)
        self.assertEqual(runner.events[:4], [('Original', 'gate', 0), ('Original', 'gate', 1),
                                            ('Optimized', 'gate', 0), ('Optimized', 'gate', 1)])
        self.assertEqual([r[0] for r in runner.events[4:]],
                         ['Original', 'Optimized', 'Optimized', 'Original', 'Original', 'Optimized'])
        self.assertEqual(len(records), 10)
        self.assertEqual(arms['Optimized']['speedup_vs_original'], 2.)
        self.assertEqual(arms['Original']['seconds'], [4., 4., 4.])
        self.assertEqual(reference['record']['phase'], 'gate')
        self.assertTrue(all(exact for _, exact in runner.saved))

    def test_each_gate_failure_stops_before_timing(self):
        for mismatch in (('Original', 'gate', 1), ('Optimized', 'gate', 0), ('Optimized', 'gate', 1)):
            with self.subTest(mismatch=mismatch):
                runner = FakeRunner(mismatch=mismatch)
                with self.assertRaises(RuntimeError):
                    esm._execute(runner, 3)
                self.assertFalse(any(event[1] == 'timed' for event in runner.events))
                self.assertFalse(runner.saved[-1][1])

    def test_timed_mismatch_returns_no_summary_and_preserves_failure(self):
        runner = FakeRunner(mismatch=('Optimized', 'timed', 1))
        with self.assertRaises(RuntimeError):
            esm._execute(runner, 3)
        self.assertFalse(runner.saved[-1][1])
        self.assertFalse(any(event[2] == 2 for event in runner.events))

    def test_meta_fallback_never_enters_timing(self):
        runner = FakeRunner(unavailable=True)
        with self.assertRaisesRegex(RuntimeError, 'did not engage'):
            esm._execute(runner, 3)
        self.assertFalse(any(event[1] == 'timed' for event in runner.events))

    def test_invalid_repeats_do_not_generate_features(self):
        for repeats in (0, 6, True, '3'):
            runner = FakeRunner()
            with self.assertRaises(ValueError):
                esm._execute(runner, repeats)
            self.assertEqual(runner.events, [])


class StateTests(unittest.TestCase):
    def make_runner(self, folder, failing=False):
        runner = esm._Generation.__new__(esm._Generation)
        runner.cache_root = Path(folder) / 'cache'
        runner.canonical = runner.cache_root / 'esm/proteins'
        runner.output = Path(folder) / 'evidence'
        (runner.output / 'records').mkdir(parents=True)
        runner.torch, runner.sequences = fake_torch(), ['A', 'CC', 'DDD', 'EEEE', 'FFFFF']
        runner.expected_batches = [[hashlib.sha256(s.encode()).hexdigest() for s in runner.sequences[i:i + 4]]
                                   for i in range(0, len(runner.sequences), 4)]
        runner.allow_unvalidated, runner.had_once, runner.old_once = True, False, None
        runner.cache_utils = types.SimpleNamespace(GLOBAL_RUN_RECORDS={'unrelated': True})
        module = types.SimpleNamespace(GLOBAL_VARIABLES={'model': None}, ESM_CACHE_PATH='esm/proteins',
                                       PROTEIN_REPR_CONFIG={'esm': {'batch_fn': None}})
        runner.esm = module
        loads = []
        def initialize():
            if runner.cache_utils.GLOBAL_RUN_RECORDS.get('init_esm'):
                return
            loads.append('stock')
            value = types.SimpleNamespace(dtype='float32', device=types.SimpleNamespace(type='cuda'),
                                          is_floating_point=lambda: True)
            model = types.SimpleNamespace(training=False, parameters=lambda: [value], buffers=lambda: [])
            module.GLOBAL_VARIABLES['model'] = (model, None)
            runner.cache_utils.GLOBAL_RUN_RECORDS['init_esm'] = True
        def batch(sequences):
            module.init_esm()
            if failing:
                raise RuntimeError('injected generation failure')
            return [FakeTensor(sequence) for sequence in sequences]
        def many(sequences, device, batch_size):
            assert device == 'cpu' and batch_size == 4
            result = {}
            for start in range(0, len(sequences), batch_size):
                subset = sequences[start:start + batch_size]
                for sequence, value in zip(subset, module._run_esm_batch(subset)):
                    target = Path(module.ESM_CACHE_PATH) / (hashlib.sha256(sequence.encode()).hexdigest() + '.pt')
                    assert not target.exists(), 'Reused an existing feature cache'
                    target.write_bytes(b'public-test-cache')
                    result[sequence] = value
            return result
        module._run_esm_batch, module.init_esm, module.get_many_esm_reprs = batch, initialize, many
        module.PROTEIN_REPR_CONFIG['esm']['batch_fn'] = many
        runner.old = {'batch': batch, 'many': many, 'init': initialize, 'path': 'esm/proteins',
                      'config': many, 'model': None}
        return runner, loads

    def install_meta_fixture(self, runner, fallback=False):
        @contextmanager
        def context(module, *, sequences, mode, loader_mode, allow_unvalidated, cache_root, batch_observer):
            self.assertEqual((mode, loader_mode), ('off', 'meta'))
            self.assertTrue(allow_unvalidated)
            old_batch, old_path = module._run_esm_batch, module.ESM_CACHE_PATH
            receipt = {'loader_receipts': [{'used_mode': 'off' if fallback else 'meta',
                       'engaged': not fallback, 'fallback_reason': 'fixture unsupported' if fallback else None}],
                       'application_cache_reused': False, 'forward_batches': [], 'closed': False}
            def batch(subset):
                values = runner.old['batch'](subset)
                event = {'sequences': [hashlib.sha256(s.encode()).hexdigest() for s in subset],
                         'decision': {'used_mode': 'off'}}
                batch_observer(event)
                receipt['forward_batches'].append(event)
                return values
            def release():
                module.GLOBAL_VARIABLES['model'] = None
                receipt['model_released'] = True
            module._run_esm_batch = batch
            module.ESM_CACHE_PATH = str(Path(cache_root) / 'catpred_exact')
            Path(module.ESM_CACHE_PATH).mkdir()
            handle = types.SimpleNamespace(summary=lambda: copy.deepcopy(receipt), release_model=release)
            try:
                yield handle
            finally:
                module._run_esm_batch, module.ESM_CACHE_PATH = old_batch, old_path
                receipt['closed'] = True
        runner.protein = types.SimpleNamespace(features_context=context)

    def test_original_samples_reload_model_use_distinct_caches_and_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            runner, loads = self.make_runner(folder)
            first = runner.run('Original', 'gate', 0)
            second = runner.run('Original', 'gate', 1)
            self.assertEqual(loads, ['stock', 'stock'])
            self.assertTrue(esm._exact(first['snapshot'], second['snapshot']))
            self.assertNotEqual(first['record']['cache_path'], second['record']['cache_path'])
            self.assertEqual(len(list(runner.canonical.glob('*.pt'))), 5)
            self.assertEqual(runner.esm.ESM_CACHE_PATH, 'esm/proteins')
            self.assertIs(runner.esm._run_esm_batch, runner.old['batch'])
            self.assertIsNone(runner.esm.GLOBAL_VARIABLES['model'])
            self.assertEqual(runner.cache_utils.GLOBAL_RUN_RECORDS, {'unrelated': True})
            self.assertEqual([b['sequences'] for b in first['record']['actual_forward_batches']], runner.expected_batches)

    def test_exception_restores_functions_cache_and_run_once(self):
        with tempfile.TemporaryDirectory() as folder:
            runner, loads = self.make_runner(folder, failing=True)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                runner.run('Original', 'gate', 0)
            self.assertIs(runner.esm.init_esm, runner.old['init'])
            self.assertIs(runner.esm.get_many_esm_reprs, runner.old['many'])
            self.assertEqual(runner.esm.ESM_CACHE_PATH, 'esm/proteins')
            self.assertEqual(runner.cache_utils.GLOBAL_RUN_RECORDS, {'unrelated': True})
            self.assertIsNone(runner.esm.GLOBAL_VARIABLES['model'])
            self.assertTrue((runner.output / 'records/gate_Original_0.json').is_file())

    def test_optimized_receipt_records_closed_released_and_original_forward(self):
        with tempfile.TemporaryDirectory() as folder:
            runner, loads = self.make_runner(folder)
            self.install_meta_fixture(runner)
            result = runner.run('Optimized', 'gate', 0)
            receipt = result['record']['protein_backend']
            self.assertTrue(receipt['closed'])
            self.assertTrue(receipt['model_released'])
            self.assertEqual(receipt['loader_receipts'][0]['used_mode'], 'meta')
            self.assertTrue(all(result['record']['restoration'].values()))
            self.assertEqual([e['decision']['used_mode'] for e in receipt['forward_batches']], ['off', 'off'])
            self.assertEqual(runner.cache_utils.GLOBAL_RUN_RECORDS, {'unrelated': True})

    def test_actual_meta_fallback_receipt_rejected_and_state_restored(self):
        with tempfile.TemporaryDirectory() as folder:
            runner, loads = self.make_runner(folder)
            self.install_meta_fixture(runner, fallback=True)
            with self.assertRaisesRegex(RuntimeError, 'did not engage meta'):
                runner.run('Optimized', 'gate', 0)
            self.assertIs(runner.esm._run_esm_batch, runner.old['batch'])
            self.assertIsNone(runner.esm.GLOBAL_VARIABLES['model'])
            self.assertEqual(runner.cache_utils.GLOBAL_RUN_RECORDS, {'unrelated': True})

    def test_content_addressing_deduplicates_only_storage(self):
        with tempfile.TemporaryDirectory() as folder:
            runner, loads = self.make_runner(folder)
            (runner.output / 'raw').mkdir()
            saved = []
            fake_numpy = types.SimpleNamespace(savez_compressed=lambda stream, **arrays:
                                               (saved.append(tuple(arrays)), stream.write(b'npz-fixture')))
            with patch.dict(sys.modules, {'numpy': fake_numpy}):
                for repetition in (0, 1):
                    result = runner.run('Original', 'gate', repetition)
                    runner.persist(result, True)
            self.assertEqual(loads, ['stock', 'stock'])
            self.assertEqual(len(saved), 1)
            self.assertEqual(len(list((runner.output / 'raw').glob('*.npz'))), 1)
            self.assertEqual(len(list((runner.output / 'records').glob('*.json'))), 2)


class ImportTests(unittest.TestCase):
    def test_import_no_framework_io_or_network(self):
        script = '''import builtins, importlib.util, pathlib, sys
before=set(pathlib.Path('.').iterdir())
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','numpy','catpred','catpred_accel','requests','urllib','google'}:
  raise AssertionError(name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
spec=importlib.util.spec_from_file_location('esm_compare',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert set(pathlib.Path('.').iterdir())==before
'''
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, '-B', '-c', script, str(HERE / 'esm_compare.py')],
                                    cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
