"""Same-runtime ESM loading and feature-generation comparison.

The accepted optimization changes CPU model construction only. Imports perform
no framework initialization, downloads or model execution.
"""
from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

BOUNDARY = ('Fresh ESM features with local checkpoint loading, model construction, '
            'GPU transfer, original tokenization and batches of four, FP32 forwards, '
            'CPU outputs and application-cache writes; synchronized before and after. '
            'Downloads, imports, outer preflight validation, precision checks, evidence '
            'serialization and model release are excluded. Local filesystem caches '
            'are not flushed. The accepted loader\'s internal metadata checks remain '
            'inside the timer. This is not an end-to-end cold request.')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _snapshot(features, sequences, torch):
    if list(features) != list(sequences):
        raise ValueError('ESM sequence order changed')
    arrays, entries, raw = {}, [], []
    for index, sequence in enumerate(sequences):
        tensor = features[sequence]
        if tensor.dtype != torch.float32 or tensor.device.type != 'cpu':
            raise ValueError('ESM features must be CPU FP32 tensors')
        if not torch.isfinite(tensor).all().item():
            raise ValueError('ESM features contain nonfinite values')
        array = tensor.numpy()
        if tuple(array.shape) != (min(len(sequence), 2047), 1280):
            raise ValueError('ESM residue slice shape changed')
        payload = array.tobytes(order='C')
        key = 'feature_' + str(index).zfill(4)
        arrays[key] = array
        raw.append(payload)
        entries.append({'key': key, 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
                        'shape': list(array.shape), 'dtype': array.dtype.str,
                        'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
    return {'entries': entries, 'arrays': arrays, 'raw': raw}


def _exact(reference, candidate):
    return (reference['entries'] == candidate['entries']
            and reference['raw'] == candidate['raw'])


def _execute(runner, repeats):
    """Keep all exact gates ahead of measured samples and publish no partial gain."""
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError('ESM repeats must be between one and five')
    records = []
    reference = runner.run('Original', 'gate', 0)
    runner.persist(reference, True)
    records.append(reference['record'])
    for arm, repetition in (('Original', 1), ('Optimized', 0), ('Optimized', 1)):
        result = runner.run(arm, 'gate', repetition)
        exact = _exact(reference['snapshot'], result['snapshot'])
        runner.persist(result, exact)
        records.append(result['record'])
        if not exact:
            raise RuntimeError(arm + ' ESM features differ from original raw FP32 bits')
        del result
    for repetition in range(repeats):
        order = ('Original', 'Optimized') if repetition % 2 == 0 else ('Optimized', 'Original')
        for arm in order:
            result = runner.run(arm, 'timed', repetition)
            exact = _exact(reference['snapshot'], result['snapshot'])
            runner.persist(result, exact)
            records.append(result['record'])
            if not exact:
                raise RuntimeError(arm + ' ESM features changed during measurement')
            del result
    values = {arm: [r['seconds'] for r in records if r['phase'] == 'timed' and r['arm'] == arm]
              for arm in ('Original', 'Optimized')}
    original = statistics.median(values['Original'])
    arms = {arm: {'status': 'passed', 'seconds': seconds,
                  'median_seconds': statistics.median(seconds),
                  'speedup_vs_original': original / statistics.median(seconds)}
            for arm, seconds in values.items()}
    return reference, arms, records


class _Generation:
    def __init__(self, esm_utils, sequences, cache_root, output, torch, protein, cache_utils):
        self.esm, self.sequences = esm_utils, list(sequences)
        self.cache_root, self.output = Path(cache_root).resolve(), Path(output).resolve()
        self.torch, self.protein, self.cache_utils = torch, protein, cache_utils
        self.old = {'batch': esm_utils._run_esm_batch, 'many': esm_utils.get_many_esm_reprs,
                    'init': esm_utils.init_esm, 'path': esm_utils.ESM_CACHE_PATH,
                    'config': esm_utils.PROTEIN_REPR_CONFIG['esm']['batch_fn'],
                    'model': esm_utils.GLOBAL_VARIABLES['model']}
        self.had_once = 'init_esm' in cache_utils.GLOBAL_RUN_RECORDS
        self.old_once = cache_utils.GLOBAL_RUN_RECORDS.get('init_esm')
        if self.old['model'] is not None:
            raise ValueError('ESM comparison requires an unloaded model')
        self.canonical = (cache_utils.CACHE_PATH / self.old['path']).resolve()
        if not self.canonical.is_relative_to(self.cache_root):
            raise ValueError('Original ESM cache is outside this comparison')
        if self.canonical.exists() and any(self.canonical.iterdir()):
            raise ValueError('Original ESM cache must start empty')
        checkpoints = Path(torch.hub.get_dir()) / 'checkpoints'
        paths = (checkpoints / 'esm2_t33_650M_UR50D.pt',
                 checkpoints / 'esm2_t33_650M_UR50D-contact-regression.pt')
        if _sha(esm_utils.__file__) != protein.CALLER_SHA:
            raise ValueError('Original ESM caller source changed')
        if esm_utils.DEFAULT_ESM_BATCH_SIZE != 4 or esm_utils.ESM_MAX_LENGTH != 2048 or esm_utils.PROTEIN_EMBED_USE_CPU:
            raise ValueError('Original CUDA ESM batching configuration changed')
        self.preflight = protein.preflight(*paths)
        if self.preflight['support_reason'] is not None:
            raise RuntimeError(self.preflight['support_reason'])
        self.recommended = protein.recommended_modes(*paths)
        self.allow_unvalidated = not (self.recommended['mode'] == 'off'
                                      and self.recommended['loader_mode'] == 'meta')
        self.expected_batches = [[hashlib.sha256(s.encode()).hexdigest() for s in self.sequences[i:i + 4]]
                                 for i in range(0, len(self.sequences), 4)]
        (self.output / 'raw').mkdir(parents=True)
        (self.output / 'records').mkdir()

    def _release(self):
        self.esm.GLOBAL_VARIABLES['model'] = None
        self.cache_utils.GLOBAL_RUN_RECORDS.pop('init_esm', None)
        gc.collect()
        self.torch.cuda.empty_cache()

    def restore(self):
        self._release()
        self.esm._run_esm_batch, self.esm.get_many_esm_reprs = self.old['batch'], self.old['many']
        self.esm.init_esm, self.esm.ESM_CACHE_PATH = self.old['init'], self.old['path']
        self.esm.PROTEIN_REPR_CONFIG['esm']['batch_fn'] = self.old['config']
        self.esm.GLOBAL_VARIABLES['model'] = self.old['model']
        if self.had_once:
            self.cache_utils.GLOBAL_RUN_RECORDS['init_esm'] = self.old_once

    def run(self, arm, phase, repetition):
        self._release()
        name = f'{phase}_{arm}_{repetition}'
        canonical = arm == 'Original' and phase == 'gate' and repetition == 0
        cache = self.canonical if canonical else self.cache_root / 'esm_comparison' / name
        if not cache.exists():
            cache.mkdir(parents=True)
        if any(cache.iterdir()):
            raise ValueError('ESM sample cache is not empty: ' + name)
        batches = []
        def original_batch(sequences):
            result = self.old['batch'](sequences)
            model = self.esm.GLOBAL_VARIABLES['model'][0]
            values = [t for t in tuple(model.parameters()) + tuple(model.buffers()) if t.is_floating_point()]
            if model.training or not values or any(t.dtype != self.torch.float32 or t.device.type != 'cuda' for t in values):
                raise RuntimeError('Original ESM model is not CUDA FP32 in evaluation mode')
            batches.append({'sequences': [hashlib.sha256(s.encode()).hexdigest() for s in sequences],
                            'device': 'cuda', 'dtype': 'torch.float32', 'eval': True,
                            'all_floating_parameters_buffers_cuda_fp32': True})
            return result
        self.esm._run_esm_batch, self.esm.init_esm = original_batch, self.old['init']
        self.esm.get_many_esm_reprs = self.old['many']
        self.esm.PROTEIN_REPR_CONFIG['esm']['batch_fn'] = self.old['config']
        self.esm.ESM_CACHE_PATH = str(cache)
        context = (self.protein.features_context(self.esm, sequences=self.sequences, mode='off',
                   loader_mode='meta', allow_unvalidated=self.allow_unvalidated,
                   cache_root=str(cache), batch_observer=batches.append)
                   if arm == 'Optimized' else nullcontext(None))
        record = {'arm': arm, 'phase': phase, 'repetition': repetition, 'status': 'RUNNING',
                  'cache_path': str(cache), 'feature_cache_start_empty': True,
                  'requested_feature_mode': 'off', 'requested_loader_mode': 'meta' if arm == 'Optimized' else 'stock',
                  'allow_unvalidated': self.allow_unvalidated if arm == 'Optimized' else False,
                  'boundary': BOUNDARY}
        handle = None
        try:
            # Entering the context hashes/checks sources and weights outside the timer.
            with context as handle:
                self.torch.cuda.synchronize()
                start = time.perf_counter()
                features = self.esm.get_many_esm_reprs(self.sequences, device='cpu', batch_size=4)
                self.torch.cuda.synchronize()
                record['seconds'] = time.perf_counter() - start
                record['actual_forward_batches'] = batches
                if [b['sequences'] for b in batches] != self.expected_batches:
                    raise RuntimeError('Original ESM groups or padding changed')
                if handle is not None:
                    receipt = handle.summary()
                    record['protein_backend'] = receipt
                    loaders = receipt['loader_receipts']
                    if (len(loaders) != 1 or loaders[0]['used_mode'] != 'meta'
                            or loaders[0]['engaged'] is not True or loaders[0]['fallback_reason'] is not None):
                        raise RuntimeError('Optimized ESM loader did not engage meta construction')
                    if receipt['application_cache_reused'] or any(b['decision']['used_mode'] != 'off' for b in batches):
                        raise RuntimeError('ESM forward changed or features were reused')
                    handle.release_model()
                else:
                    record['protein_backend'] = {'feature_mode': 'off', 'loader_mode': 'stock',
                                                'actual_backend': 'original_fair_esm_2',
                                                'application_cache_reused': False}
            if handle is not None:
                record['protein_backend'] = handle.summary()
            snapshot = _snapshot(features, self.sequences, self.torch)
            record['status'] = 'GENERATED'
            return {'features': features, 'snapshot': snapshot, 'record': record}
        except BaseException as error:
            if handle is not None:
                record['protein_backend'] = handle.summary()
            record.update(status='FAILED', error=type(error).__name__ + ': ' + str(error),
                          actual_forward_batches=batches)
            _save(self.output / 'records' / (name + '.json'), record)
            raise
        finally:
            self.restore()
            record['restoration'] = {
                'functions_restored': (self.esm._run_esm_batch is self.old['batch']
                                      and self.esm.get_many_esm_reprs is self.old['many']
                                      and self.esm.init_esm is self.old['init']
                                      and self.esm.PROTEIN_REPR_CONFIG['esm']['batch_fn'] is self.old['config']),
                'cache_path_restored': self.esm.ESM_CACHE_PATH == self.old['path'],
                'model_released': self.esm.GLOBAL_VARIABLES['model'] is None,
                'run_once_restored': (('init_esm' in self.cache_utils.GLOBAL_RUN_RECORDS) == self.had_once
                                     and self.cache_utils.GLOBAL_RUN_RECORDS.get('init_esm') == self.old_once)}
            if not all(record['restoration'].values()):
                raise RuntimeError('ESM comparison did not restore the original caller state')
            if record['status'] == 'FAILED':
                _save(self.output / 'records' / (name + '.json'), record)

    def persist(self, result, exact):
        import numpy as np
        snapshot, record = result['snapshot'], result['record']
        content_key = hashlib.sha256(json.dumps(snapshot['entries'], sort_keys=True).encode()).hexdigest()
        archive = self.output / 'raw' / (content_key + '.npz')
        if not archive.exists():
            temporary = archive.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **snapshot['arrays'])
            temporary.replace(archive)
        record.update(status='PASSED' if exact else 'REJECTED', exact_bits=exact,
                      features={'archive': str(archive.relative_to(self.output)),
                                'archive_sha256': _sha(archive), 'content_sha256': content_key,
                                'entries': snapshot['entries']})
        _save(self.output / 'records' / f"{record['phase']}_{record['arm']}_{record['repetition']}.json", record)


def benchmark_esm(esm_utils, *, sequences, cache_root, output, repeats=3):
    """Return canonical original CPU features and a separately measured ESM comparison."""
    import torch
    from catpred.data import cache_utils
    from catpred_accel import protein
    sequences = list(dict.fromkeys(sequences))
    if not sequences or not all(isinstance(s, str) and s for s in sequences):
        raise ValueError('ESM comparison needs nonempty sequences')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'RUNNING', 'repetitions': repeats, 'arms': {}, 'boundary': BOUNDARY,
               'feature_mode': 'off', 'optimized_loader_mode': 'meta', 'batch_size': 4,
               'unique_sequences': len(sequences), 'numeric': 'FP32'}
    _save(output / 'summary.json', summary)
    runner = None
    try:
        runner = _Generation(esm_utils, sequences, cache_root, output, torch, protein, cache_utils)
        summary.update(preflight=runner.preflight, recommended=runner.recommended,
                       allow_unvalidated=runner.allow_unvalidated, caller_sha256=protein.CALLER_SHA)
        print('Checking ESM feature precision, then measuring feature generation...', flush=True)
        reference, arms, records = _execute(runner, repeats)
        summary.update(status='COMPLETE', arms=arms, all_exact_bits=True,
                       all_reported_timings_passed_exact_bits=True,
                       records=[f"records/{r['phase']}_{r['arm']}_{r['repetition']}.json" for r in records],
                       canonical_record='records/gate_Original_0.json')
        _save(output / 'summary.json', summary)
        return {'features': reference['features'], 'esm_seconds': reference['record']['seconds'],
                'unique_sequences': len(sequences), 'esm_forward_batches': len(runner.expected_batches),
                'esm_comparison': summary}
    except BaseException as error:
        summary.update(status='FAILED', error=type(error).__name__ + ': ' + str(error))
        summary['arms'] = {}
        _save(output / 'summary.json', summary)
        raise
    finally:
        if runner is not None:
            runner.restore()
