"""Compare original kcat with the complete optimized path on one CUDA runtime.

Imports and --help are safe on a laptop. Model execution requires Linux CUDA.
The saved precision arrays, rather than rounded CSVs, decide acceptance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

ARMS = ('Original', 'S_STREAM', 'S_STREAM_K1')
ARRAYS = ('member_raw', 'numeric_raw', 'numeric_processed')
BUDGET = 2 << 30


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_input(source, destination):
    """Retain input order and every row, adding stable IDs where needed."""
    with Path(source).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        if not {'SMILES', 'sequence'}.issubset(fields):
            raise ValueError('CSV needs SMILES and sequence columns')
        if len(set(fields)) != len(fields):
            raise ValueError('CSV column names must be unique')
        rows = list(reader)
    if not 1 <= len(rows) <= 16384:
        raise ValueError('Choose between 1 and 16,384 rows')
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError('Malformed CSV row ' + str(index + 2))
        if not row['sequence'] or not row['SMILES']:
            raise ValueError('Empty sequence or SMILES at row ' + str(index + 2))
        row['row_id'] = 'row_' + str(index).zfill(6)
        if not row.get('pdbpath'):
            row['pdbpath'] = hashlib.sha256(row['sequence'].encode()).hexdigest() + '.pdb'
    for field in ('row_id', 'pdbpath'):
        if field not in fields:
            fields.append(field)
    with Path(destination).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def compare_precision(left, right):
    import numpy as np
    a_meta = json.loads(Path(left['precision_manifest_path']).read_text())
    b_meta = json.loads(Path(right['precision_manifest_path']).read_text())
    result = {}
    with np.load(left['precision_path'], allow_pickle=False) as aa, np.load(right['precision_path'], allow_pickle=False) as bb:
        for key in ARRAYS:
            a, b = aa[key], bb[key]
            structure = (a.dtype == b.dtype and a.shape == b.shape
                         and a_meta['stages'][key] == b_meta['stages'][key]
                         and a_meta['row_ids'] == b_meta['row_ids'])
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            result[key] = {'structure_equal': structure, 'finite': finite,
                           'exact_bits': structure and finite and a.tobytes(order='C') == b.tobytes(order='C')}
    return {'exact_bits': all(value['exact_bits'] for value in result.values()), 'arrays': result}


def order_for_round(arms, index):
    """Rotate the first arm so each path occupies every position in turn."""
    arms = list(arms)
    offset = index % len(arms)
    order = arms[offset:] + arms[:offset]
    return order


def summarize_times(records, arms):
    values = {arm: [r['seconds'] for r in records if r['arm'] == arm] for arm in arms}
    baseline = statistics.median(values['Original'])
    return {arm: {'status': 'passed', 'median_seconds': statistics.median(times),
                  'seconds': times, 'speedup_vs_original': baseline / statistics.median(times),
                  'prediction_path': next(r['prediction_path'] for r in reversed(records) if r['arm'] == arm)}
            for arm, times in values.items()}


def verify_saved_esm(output, sequences, repeats):
    """Read and independently verify every retained ESM tensor and sample."""
    import numpy as np
    root = (Path(output) / 'esm').resolve()
    summary = json.loads((root / 'summary.json').read_text())
    if (summary.get('status') != 'COMPLETE' or summary.get('all_exact_bits') is not True
            or summary['repetitions'] != repeats):
        raise ValueError('Resume requires a complete exact ESM comparison with unchanged repeats')
    expected = [('Original', 'gate', 0), ('Original', 'gate', 1), ('Optimized', 'gate', 0), ('Optimized', 'gate', 1)]
    for repetition in range(repeats):
        expected.extend((arm, 'timed', repetition) for arm in order_for_round(('Original', 'Optimized'), repetition))
    expected_names = [f'records/{phase}_{arm}_{repetition}.json' for arm, phase, repetition in expected]
    if summary['records'] != expected_names or summary['canonical_record'] != expected_names[0]:
        raise ValueError('Retained ESM sample schedule changed')
    batches = [[hashlib.sha256(s.encode()).hexdigest() for s in sequences[i:i + 4]]
               for i in range(0, len(sequences), 4)]
    reference, canonical, cached = None, None, {}
    for name, (arm, phase, repetition) in zip(expected_names, expected):
        record = json.loads((root / name).read_text())
        if ((record['arm'], record['phase'], record['repetition']) != (arm, phase, repetition)
                or record['status'] != 'PASSED' or record['exact_bits'] is not True
                or [b['sequences'] for b in record['actual_forward_batches']] != batches
                or not record['restoration'] or not all(record['restoration'].values())):
            raise ValueError('Invalid retained ESM receipt: ' + name)
        if arm == 'Optimized':
            backend = record['protein_backend']
            loaders = backend['loader_receipts']
            if (not backend.get('closed') or not backend.get('model_released')
                    or len(loaders) != 1 or loaders[0]['used_mode'] != 'meta'
                    or not loaders[0]['engaged'] or loaders[0]['fallback_reason'] is not None
                    or any(b['decision']['used_mode'] != 'off' for b in record['actual_forward_batches'])):
                raise ValueError('Retained optimized ESM receipt did not use the accepted loader')
        evidence = record['features']
        archive = (root / evidence['archive']).resolve()
        if not archive.is_relative_to(root) or sha(archive) != evidence['archive_sha256']:
            raise ValueError('Retained ESM archive hash mismatch')
        key = evidence['archive_sha256']
        if key not in cached:
            with np.load(archive, allow_pickle=False) as arrays:
                cached[key] = {k: arrays[k].copy() for k in arrays.files}
        arrays = cached[key]
        entries = evidence['entries']
        if len(entries) != len(sequences) or set(arrays) != {e['key'] for e in entries}:
            raise ValueError('Retained ESM archive tensor inventory changed')
        content = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        if content != evidence['content_sha256']:
            raise ValueError('Retained ESM content identity changed')
        current = {}
        for sequence, entry in zip(sequences, entries):
            array = arrays[entry['key']]
            raw = array.tobytes(order='C')
            if (entry['sequence_sha256'] != hashlib.sha256(sequence.encode()).hexdigest()
                    or array.dtype != np.dtype('float32') or array.dtype.str != entry['dtype']
                    or list(array.shape) != entry['shape'] or array.shape != (min(len(sequence), 2047), 1280)
                    or not np.isfinite(array).all() or len(raw) != entry['bytes']
                    or hashlib.sha256(raw).hexdigest() != entry['sha256']):
                raise ValueError('Retained ESM raw tensor mismatch')
            current[sequence] = raw
        if reference is None:
            reference, canonical = current, record
        elif current != reference:
            raise ValueError('Retained ESM samples are not bitwise identical')
    for arm in ('Original', 'Optimized'):
        seconds = [json.loads((root / name).read_text())['seconds']
                   for name, spec in zip(expected_names, expected) if spec[0] == arm and spec[1] == 'timed']
        if (summary['arms'][arm]['seconds'] != seconds
                or summary['arms'][arm]['median_seconds'] != statistics.median(seconds)):
            raise ValueError('Retained ESM timing summary changed')
    return summary, canonical, reference


def archive_attempt(output):
    """Preserve only the superseded kinetics attempt; keep all ESM work in place."""
    output = Path(output)
    attempts = output / 'attempts'
    attempts.mkdir(exist_ok=True)
    index = 0
    while (attempts / f'attempt_{index}').exists():
        index += 1
    destination = attempts / f'attempt_{index}'
    destination.mkdir()
    moved = []
    for path in sorted(output.iterdir()):
        if path.name == 'summary.json' or path.name.startswith(('gate_', 'warm_')):
            path.rename(destination / path.name)
            moved.append(path.name)
    save(destination / 'preservation.json', {'files': moved, 'esm_retained_in_place': True,
                                           'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    return str(destination.relative_to(output))


class Comparison:
    def __init__(self, setup, source, output, resume=False):
        if sys.platform != 'linux':
            raise RuntimeError('Run models on the Linux G4, not the Mac')
        self.setup, self.out = setup, Path(output)
        self.root = Path(setup['work_root']).resolve()
        self.repo = self.root / 'CatPred'
        self.cache = self.out / 'feature_cache'
        self.cache.mkdir(exist_ok=resume)
        self.resume_verified_esm = resume
        self.input = self.out / 'input.csv'
        if resume:
            check = self.out / 'input.resume-check.csv'
            try:
                self.rows = prepare_input(source, check)
                if sha(check) != sha(self.input):
                    raise ValueError('Normalized input changed; cannot retain ESM work')
            finally:
                check.unlink(missing_ok=True)
        else:
            self.rows = prepare_input(source, self.input)
        self.row_ids = [r['row_id'] for r in self.rows]
        os.environ.update(CATPRED_CACHE_PATH=str(self.cache), TORCH_HOME=setup['torch_home'],
                          CATPRED_PREDICTION_CACHE_SIZE='0', CATPRED_MODEL_CACHE_SIZE='2',
                          PROTEIN_EMBED_USE_CPU='0', CATPRED_ESM_BATCH_SIZE='4',
                          CATPRED_TRUSTED_DESERIALIZATION_ROOTS=os.pathsep.join((str(self.root), str(self.out))))
        for key in ('CATPRED_ESM_META_LOAD', 'CATPRED_ESM_REPR_ONLY', 'CATPRED_ESM_REPRESENTATIONS_ONLY'):
            os.environ.pop(key, None)
        sys.path[:0] = [str(self.root / 'baseline'), str(self.repo)]
        import numpy as np
        import torch
        import packing, reuse, probes, streaming, capture
        from catpred.inference import PredictionRequest, run_inprocess_prediction_pipeline, service
        from catpred.data import esm_utils
        from catpred_accel import Runtime, RuntimeConfig
        self.torch, self.np = torch, np
        if not torch.cuda.is_available():
            raise RuntimeError('A CUDA GPU is required')
        torch.set_num_threads(24)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(20261005)
        np.random.seed(20261005)
        random.seed(20261005)
        self.components = dict(packing=packing, reuse=reuse, probes=probes, streaming=streaming)
        self.packing, self.reuse, self.probes, self.streaming, self.capture = packing, reuse, probes, streaming, capture
        self.pipeline, self.service, self.esm = run_inprocess_prediction_pipeline, service, esm_utils
        self.Runtime, self.Config = Runtime, RuntimeConfig
        assert service._PREDICTION_CACHE_SIZE == 0
        # Verify actual staged bytes before any deserialization or model work.
        owner = json.loads((self.root / '.catpred-colab-owner.json').read_text())
        for record in owner['sources']:
            assert sha(self.root / record['destination']) == record['sha256'], record['destination']
        for record in setup['checkpoint_files']:
            assert sha(self.root / record['path']) == record['sha256'], record['path']
        for record in setup['esm_files']:
            assert sha(Path(setup['torch_home']) / 'hub/checkpoints' / record['name']) == record['sha256']
        self.request = PredictionRequest(parameter='kcat', input_file=str(self.input),
                                        checkpoint_dir=setup['models_root'], use_gpu=True, repo_root=str(self.repo))
        self.bundle = None
        self.consumed = {}
        self.original_build, self.original_load = service._build_predict_args, service._load_model_objects_for_prediction
        self.original_many, self.original_batch, self.original_single = esm_utils.get_many_esm_reprs, esm_utils._run_esm_batch, esm_utils.get_single_esm_repr
        self.original_config_many = esm_utils.PROTEIN_REPR_CONFIG['esm']['batch_fn']
        probes.configure(r3=False, t1=False, t2=False)
        capture.install()
        service._build_predict_args = self.build
        service._load_model_objects_for_prediction = self.load

    def build(self, *args, **kwargs):
        value = self.original_build(*args, **kwargs)
        value.batch_size = 50
        assert value.num_workers == 0
        return value

    def load(self, *args, **kwargs):
        value = self.original_load(*args, **kwargs)
        value[0].batch_size = 50
        if self.bundle is None:
            self.bundle = value
            assert len(value[2]) == 10 and list(value[5]) == ['log10kcat_max']
            assert {p.dtype for m in value[2] for p in m.parameters()} == {self.torch.float32}
            for model in value[2]:
                model.eval()
            self.reuse.inspect_models(value[2], value[3])
            self.probes.inspect_models(value[2], value[3])
            self.streaming.inspect_models(value[2], value[3])
        else:
            assert tuple(map(id, value[2])) == tuple(map(id, self.bundle[2]))
        return value

    def prepare(self, repeats=3):
        prepared = self.service.prepare_prediction_inputs('kcat', str(self.input), str(self.repo))
        self.service._write_protein_records(prepared.input_csv, prepared.records_file)
        self.raw_csv = Path(prepared.output_csv)
        self.request = replace(self.request, protein_records_file=str(Path(prepared.records_file).resolve()))
        sequences = list(dict.fromkeys(row['sequence'] for row in self.rows))
        if self.resume_verified_esm:
            preparation = self.resume_esm(sequences, repeats)
        else:
            from esm_compare import benchmark_esm
            preparation = benchmark_esm(self.esm, sequences=sequences, cache_root=self.cache,
                                        output=self.out / 'esm', repeats=repeats)
        features = preparation.pop('features')
        assert list(features) == sequences
        self.expected = {s: self.tensor_identity(t) for s, t in features.items()}
        save(self.out / 'features.json', {'backend': 'original_fair_esm_2', 'batch_size': 4,
             'sequences': {hashlib.sha256(s.encode()).hexdigest(): value for s, value in self.expected.items()},
             'actual_forward_batches': [[hashlib.sha256(s.encode()).hexdigest() for s in sequences[i:i + 4]]
                                        for i in range(0, len(sequences), 4)],
             'seconds': preparation['esm_seconds']})
        del features
        self.esm.GLOBAL_VARIABLES['model'] = None
        gc.collect()
        self.torch.cuda.empty_cache()
        self.cache_signature = {str(p.relative_to(self.cache)): sha(p) for p in sorted(self.cache.rglob('*.pt'))}
        def no_generation(*args, **kwargs):
            raise RuntimeError('ESM cache miss in a warm request')
        def many(*args, **kwargs):
            result = self.original_many(*args, **kwargs)
            self.consumed.update(result)
            return result
        self.esm._run_esm_batch = self.esm.get_single_esm_repr = no_generation
        self.esm.get_many_esm_reprs = self.esm.PROTEIN_REPR_CONFIG['esm']['batch_fn'] = many
        start = time.perf_counter()
        self.load(self.build(self.request, prepared, self.repo))
        self.torch.cuda.synchronize()
        model_seconds = time.perf_counter() - start
        return dict(preparation, model_load_seconds=model_seconds)

    def resume_esm(self, sequences, repeats):
        from catpred.data import cache_utils
        from catpred_accel._identity import package_sources, software
        summary, canonical, reference = verify_saved_esm(self.out, sequences, repeats)
        context = summary['preflight']['context']
        current = software(self.torch)
        previous = context['software']
        if (summary['caller_sha256'] != sha(self.esm.__file__)
                or context['implementation_sources'] != package_sources()):
            raise ValueError('Frozen ESM caller or acceleration library changed')
        if {k: v for k, v in previous.items() if k != 'rdkit'} != {k: v for k, v in current.items() if k != 'rdkit'}:
            raise ValueError('Resume permits the recorded RDKit repair only')
        features = {}
        for sequence in sequences:
            tensor = cache_utils.load_cache_value(path=self.esm.ESM_CACHE_PATH, cache_key=sequence,
                                                  purpose='verified retained ESM feature', map_location='cpu')
            if (tensor is None or tensor.dtype != self.torch.float32 or tensor.device.type != 'cpu'
                    or tuple(tensor.shape) != (min(len(sequence), 2047), 1280)
                    or not self.torch.isfinite(tensor).all().item()
                    or tensor.numpy().tobytes(order='C') != reference[sequence]):
                raise ValueError('Retained canonical ESM cache differs from saved raw features')
            features[sequence] = tensor
        self.dependency_repair = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                 'dependency': 'rdkit', 'old_version': previous['rdkit'], 'new_version': current['rdkit'],
                                 'esm_retained': True, 'esm_regenerated': False,
                                 'esm_software': previous, 'kinetics_software': current,
                                 'esm_summary_sha256': sha(self.out / 'esm/summary.json'),
                                 'canonical_cache_raw_bits_verified': True}
        return {'features': features, 'esm_seconds': canonical['seconds'], 'unique_sequences': len(sequences),
                'esm_forward_batches': len(canonical['actual_forward_batches']), 'esm_comparison': summary}

    def tensor_identity(self, tensor):
        assert tensor.dtype == self.torch.float32 and tensor.device.type == 'cpu'
        assert self.torch.isfinite(tensor).all().item()
        return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'sha256': hashlib.sha256(tensor.numpy().tobytes(order='C')).hexdigest()}

    def reset(self):
        self.streaming.configure(enabled=False, budget_bytes=BUDGET)
        self.packing.install_arm('A', instrument=True)
        self.reuse.configure(r1=False, r2=False)
        self.probes.configure(r3=False, t1=False, t2=False)

    def run(self, arm, phase, repetition):
        dest = self.out / (phase + '_' + arm + '_' + str(repetition))
        dest.mkdir()
        self.reset()
        runtime = adapter = state = None
        self.consumed = {}
        self.packing.reset_counts()
        assert not self.service._PREDICTION_CACHE
        with ExitStack() as activation:
            if arm != 'Original':
                runtime = self.Runtime(self.Config(numeric='exact', memory='stream', backend='auto',
                                                  input_budget_bytes=BUDGET, fallback='raise', allow_unvalidated=True),
                                       components=self.components)
                activation.enter_context(runtime.activate(self.bundle[2], self.bundle[3]))
                selected = runtime.manifest()
                assert selected['engaged'] and selected['selected_backend'] == 'accepted_stream'
                if arm == 'S_STREAM_K1':
                    from catpred_kinetics import KineticsAdapter
                    adapter = KineticsAdapter(enabled=True, allow_unvalidated=True)
                    activation.enter_context(adapter.activate(runtime, self.bundle[2]))
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with ExitStack() as request_scope:
                if runtime is not None:
                    state = request_scope.enter_context(runtime.request())
                    stream, probe = state.stream, state.probes
                else:
                    stream = request_scope.enter_context(self.streaming.request_context())
                    request_scope.enter_context(self.reuse.request_context())
                    probe = request_scope.enter_context(self.probes.request_context())
                cap = request_scope.enter_context(self.capture.request_context(self.row_ids))
                prediction = self.pipeline(self.request, results_dir=str(dest))
            self.torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            manifest = runtime.manifest() if runtime is not None else None
        # Hashing and precision serialization are deliberately outside both timers.
        assert set(self.consumed) == set(self.expected), 'Missing consumed features'
        for sequence, tensor in self.consumed.items():
            assert self.tensor_identity(tensor) == self.expected[sequence], 'Consumed feature mismatch'
        assert {str(p.relative_to(self.cache)): sha(p) for p in self.cache.rglob('*.pt')} == self.cache_signature
        probe_summary, stream_summary = probe.summary(), stream.summary()
        counts = [probe_summary['member_counters'].get(str(i), {}).get('model_forward_calls', 0) for i in range(10)]
        assert counts == [math.ceil(len(self.rows) / 50)] * 10, counts
        assert stream_summary['closed'] and stream_summary['retained_after_request_bytes'] == 0
        if state is not None:
            assert state.summary()['request_cache_empty_on_exit'] and state.summary()['failure'] is None
            assert runtime.manifest()['restored']
        if adapter is not None:
            assert adapter.summary()['member_optimized_calls'] == counts
            assert adapter.summary()['restored'] and adapter.summary()['engaged']
        record = {'arm': arm, 'phase': phase, 'repetition': repetition, 'seconds': seconds,
                  'model_forward_counts': counts, 'batch_sizes': [min(50, len(self.rows) - i) for i in range(0, len(self.rows), 50)],
                  'runtime': manifest, 'stream': stream_summary, 'probes': probe_summary,
                  'kinetics': adapter.summary() if adapter is not None else None,
                  'cuda_peak_allocated_bytes': self.torch.cuda.max_memory_allocated(),
                  'prediction_path': str(prediction), 'prediction_sha256': sha(prediction)}
        record.update(cap.save(dest))
        shutil.copyfile(self.raw_csv, dest / 'raw_predictions.csv')
        save(dest / 'receipt.json', record)
        self.consumed = {}
        return record

    def close(self):
        self.service._build_predict_args, self.service._load_model_objects_for_prediction = self.original_build, self.original_load
        self.esm.get_many_esm_reprs, self.esm._run_esm_batch, self.esm.get_single_esm_repr = self.original_many, self.original_batch, self.original_single
        self.esm.PROTEIN_REPR_CONFIG['esm']['batch_fn'] = self.original_config_many
        self.reset()


def execute(options):
    variants = getattr(options, 'variants', 'best')
    if variants not in ('best', 'all'):
        raise ValueError('Choose best or all variants')
    candidates = ARMS[1:] if variants == 'all' else ('S_STREAM_K1',)
    def label(arm):
        return 'Optimized' if variants == 'best' and arm == 'S_STREAM_K1' else arm
    setup = json.loads(options.setup.read_text())
    resume = getattr(options, 'resume_verified_esm', False)
    if resume:
        prior = json.loads((options.output / 'summary.json').read_text())
        if prior['input_sha256'] != sha(options.input):
            raise ValueError('Input changed; cannot resume verified ESM')
        if prior.get('status') == 'COMPLETE':
            raise ValueError('Completed comparisons do not need repair')
    else:
        options.output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'RUNNING', 'hardware': setup['hardware'], 'arms': {}, 'repetitions': options.repeats,
               'variants': variants, 'display_names': {arm: label(arm) for arm in ('Original',) + tuple(candidates)},
               'batch_size': 50, 'ensemble_members': 10, 'numeric': 'FP32', 'input_budget_bytes': BUDGET,
               'input_budget_scope': 'retained device inputs', 'input_sha256': sha(options.input),
               'timing_boundary': 'Warm complete request, synchronized, including preprocessing, all ten models, uncertainty, CSV and request-cache release. Model loading, ESM generation, activation checks and evidence serialization are outside this timer.'}
    if not resume:
        save(options.output / 'summary.json', summary)
    worker = None
    archived = False
    try:
        worker = Comparison(setup, options.input, options.output, resume=True) if resume else Comparison(setup, options.input, options.output)
        summary['rows'] = len(worker.rows)
        summary['preparation'] = worker.prepare(repeats=options.repeats)
        if resume:
            attempt = archive_attempt(options.output)
            archived = True
            summary['dependency_repair'] = dict(worker.dependency_repair, previous_attempt=attempt)
            save(options.output / 'dependency_repair.json', summary['dependency_repair'])
        summary['hardware'].update(torch_threads=worker.torch.get_num_threads(), interop_threads=worker.torch.get_num_interop_threads())
        print('Checking original repeatability and complete precision outputs...', flush=True)
        reference = worker.run('Original', 'gate', 0)
        again = worker.run('Original', 'gate', 1)
        baseline_gate = compare_precision(reference, again)
        save(options.output / 'gate_Original.json', baseline_gate)
        if not baseline_gate['exact_bits']:
            raise RuntimeError('Original output is not bitwise repeatable; no speedup reported')
        eligible = ['Original']
        for arm in candidates:
            if arm == 'S_STREAM_K1' and options.k1 == 'off':
                summary['arms'][arm] = {'status': 'disabled', 'reason': 'Disabled by notebook setting'}
                continue
            if variants == 'all' and arm == 'S_STREAM_K1' and 'S_STREAM' not in eligible:
                summary['arms'][arm] = {'status': 'unavailable', 'reason': 'S_STREAM did not pass'}
                continue
            try:
                gates = []
                for repeat in range(2):
                    record = worker.run(arm, 'gate', repeat)
                    gate = compare_precision(reference, record)
                    gates.append(gate)
                    save(options.output / ('gate_' + arm + '.json'), gates)
                    if not gate['exact_bits']:
                        raise ValueError('Full-precision outputs differ from Original')
                eligible.append(arm)
            except Exception as error:
                # Keep rejected artifacts and the actual reason; never count fallback as a gain.
                status = 'unavailable' if type(error).__name__ in ('UnsupportedCapability', 'UnsupportedKinetics') else 'rejected'
                summary['arms'][arm] = {'status': status, 'reason': type(error).__name__ + ': ' + str(error)}
                worker.reset()
                worker.torch.cuda.empty_cache()
                if variants == 'all':
                    print(arm + ': ' + summary['arms'][arm]['reason'], flush=True)
        if variants == 'best' and 'S_STREAM_K1' not in eligible:
            unavailable = summary['arms']['S_STREAM_K1']['status'] in ('unavailable', 'disabled')
            summary.update(status='UNAVAILABLE' if unavailable else 'REJECTED',
                           reason='Optimized is unavailable on this runtime.' if unavailable else 'Optimized could not complete validation.',
                           timings_run=False)
            summary['arms']['Original'] = {'status': 'verified', 'reason': 'Output repeatability passed; timing requires both versions.'}
            save(options.output / 'summary.json', summary)
            print(summary['reason'] + ' No speed comparison was run. Details: ' + str(options.output / 'summary.json'), flush=True)
            return
        save(options.output / 'summary.json', summary)
        print('Precision gates complete. Measuring warm requests...', flush=True)
        records = []
        for repetition in range(options.repeats):
            for arm in order_for_round(eligible, repetition):
                record = worker.run(arm, 'warm', repetition)
                gate = compare_precision(reference, record)
                save(Path(record['precision_path']).parent / 'comparison.json', gate)
                if not gate['exact_bits']:
                    raise RuntimeError(label(arm) + ' changed during timing; no speedup reported')
                records.append(record)
                print(f"{label(arm)}, repeat {repetition + 1}: {record['seconds']:.3f} s", flush=True)
        summary['arms'].update(summarize_times(records, eligible))
        summary['status'] = 'COMPLETE'
        summary['all_reported_timings_passed_exact_bits'] = True
        save(options.output / 'summary.json', summary)
        if variants == 'all':
            print(json.dumps(summary, indent=2), flush=True)
        else:
            print('Comparison complete. Results: ' + str(options.output), flush=True)
    except BaseException as error:
        summary.update(status='FAILED', error=type(error).__name__ + ': ' + str(error))
        target = (options.output / ('resume_failed_' + str(time.time_ns()) + '.json')
                  if resume and not archived else options.output / 'summary.json')
        save(target, summary)
        raise
    finally:
        if worker is not None:
            worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=3)
    parser.add_argument('--k1', choices=('auto', 'off'), default='auto')
    parser.add_argument('--resume-verified-esm', action='store_true',
                        help='Repair kinetics execution in the same output, preserving verified ESM samples')
    parser.add_argument('--variants', choices=('best', 'all'), default='best',
                        help='Compare Original and Optimized; all also includes the intermediate diagnostic path')
    options = parser.parse_args()
    for name in ('setup', 'input', 'output'):
        setattr(options, name, getattr(options, name).resolve())
    execute(options)


if __name__ == '__main__':
    main()
