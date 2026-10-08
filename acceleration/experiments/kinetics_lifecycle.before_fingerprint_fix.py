"""Actual-model K1 exception/restoration diagnostic; never allocates a server.

Run only on the already prepared Linux/CUDA machine. Scientific package files
remain unchanged. Diagnostic timings and one extra output D2H are not benchmarks.
Saved evidence can be checked separately with --verify-only using stdlib only.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import types
import weakref


class InjectedModelFailure(RuntimeError):
    pass


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checked_path(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise AssertionError('Evidence path escaped root')
    return path


def load_worker(candidate):
    path = Path(candidate) / 'experiments/kinetics_worker.py'
    spec = importlib.util.spec_from_file_location('_kinetics_lifecycle_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(models):
    from catpred_accel.validation.lifecycle import fingerprint
    value = fingerprint(models)
    value['model_instance_forward'] = [
        {'present': 'forward' in m.__dict__, 'identity': id(m.__dict__.get('forward'))}
        for m in models]
    return value


def owned_objects(adapter, tensor_type, module_type):
    """Traverse only instance-owned objects, not class/function global namespaces."""
    seen, found = set(), []
    def visit(value, path):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (tensor_type, module_type)):
            found.append({'path': path, 'type': type(value).__name__})
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + '[' + repr(key) + ']')
        elif isinstance(value, (list, tuple, set, frozenset)):
            for index, child in enumerate(value):
                visit(child, path + '[' + str(index) + ']')
        elif isinstance(value, types.MethodType):
            visit(value.__self__, path + '.__self__')
            visit(value.__func__, path + '.__func__')
        elif isinstance(value, types.FunctionType):
            for index, cell in enumerate(value.__closure__ or ()):
                try:
                    child = cell.cell_contents
                except ValueError:
                    continue
                visit(child, path + '.closure[' + str(index) + ']')
            visit(value.__defaults__, path + '.__defaults__')
            visit(value.__kwdefaults__, path + '.__kwdefaults__')
        elif hasattr(value, '__dict__') and not isinstance(value, (type, types.ModuleType)):
            visit(vars(value), path + '.__dict__')
    visit(vars(adapter), 'adapter')
    # ContextVar has no inspectable __dict__; its current value is explicit.
    visit(adapter._scope.get(), 'adapter._scope.current')
    return found


def make_runtime(base):
    from catpred_kinetics import KineticsAdapter
    class LifecycleRuntime(base.Runtime):
        def __init__(self, *args, **kwargs):
            self.kinetics = self.kinetics_manager = None
            self.adapters = []
            self.enabled = self.inject = False
            self.request_original = None
            self.last_request_summary = None
            self.last_transition = None
            self.weak_inputs = []
            self.weak_attention_weights = []
            self.failure_observation = None
            self.injection_path = None
            super().__init__(*args, **kwargs)

        def close_kinetics(self):
            if self.kinetics_manager is not None:
                manager, self.kinetics_manager = self.kinetics_manager, None
                try:
                    manager.__exit__(None, None, None)
                finally:
                    # Remove only this harness's factory injection, including its
                    # closure over the harness; the original package is unedited.
                    self.kinetics.__dict__.pop('_model_wrapper', None)
                    self.kinetics.__dict__.pop('_attention_wrapper', None)
                    after = snapshot(self.models)
                    self.last_transition = {
                        'before': self.accepted_before, 'after': after,
                        'equal': self.accepted_before == after,
                        'scope_empty': self.kinetics._scope.get() is None,
                        'call_lock_released': not self.kinetics._call_lock.locked(),
                        'guard_released': self.kinetics._guard is None,
                        'installed_entries_empty': not self.kinetics._installed,
                        'candidate_owned_science_objects': owned_objects(
                            self.kinetics, self.torch.Tensor, self.torch.nn.Module),
                    }
            if self.request_original is not None:
                current, self.request_original = self.request_original, None
                assert self.s1_runtime.__dict__.get('request') is self.observed_request
                del self.s1_runtime.request
                assert self.s1_runtime.request == current
                self.observed_request = None

        def close_runtime(self):
            try:
                self.close_kinetics()
            finally:
                super().close_runtime()

        def configure(self, arm):
            for model in self.models:
                model.eval()
            super().configure(arm)
            if arm != 'S1':
                return
            self.last_request_summary = None
            self.last_transition = None
            self.accepted_before = snapshot(self.models)
            self.kinetics = KineticsAdapter(enabled=self.enabled,
                                           allow_unvalidated=self.enabled)
            self.adapters.append(self.kinetics)
            if self.inject:
                original_attention_factory = self.kinetics._attention_wrapper
                def attention_factory(index, original):
                    def observe_unused_weights(*args, **kwargs):
                        output, weights = original(*args, **kwargs)
                        self.weak_attention_weights.append(weakref.ref(weights))
                        return output, weights
                    return original_attention_factory(index, observe_unused_weights)
                self.kinetics._attention_wrapper = attention_factory
                original_factory = self.kinetics._model_wrapper
                def factory(index, original):
                    if index != 0:
                        return original_factory(index, original)
                    def actual_forward_then_fail(*args, **kwargs):
                        result = original(*args, **kwargs)
                        assert self.failure_observation is None
                        assert len(self.weak_attention_weights) == 1
                        assert all(ref() is None for ref in self.weak_attention_weights)
                        assert self.kinetics.summary()['member_optimized_calls'] == [1] + [0] * 9
                        assert type(result) is self.torch.Tensor
                        assert result.device.type == 'cuda' and result.dtype == self.torch.float32
                        assert tuple(result.shape) == (50, 2)
                        state = self.s1_runtime.components['streaming']._ACTIVE.get()
                        chunk = state.active_chunk
                        allocated = chunk.actual_device_bytes()
                        assert allocated > 0
                        self.weak_inputs.extend(weakref.ref(t) for values in chunk.graph_device.values() for t in values[:5])
                        self.weak_inputs.extend(weakref.ref(t) for values in chunk.protein_device.values() for t in values if t is not None)
                        assert self.weak_inputs
                        # Extra D2H exists only in this diagnostic's failed request.
                        host = result.detach().cpu().contiguous().numpy()
                        assert self.np.isfinite(host).all()
                        self.np.save(self.injection_path, host, allow_pickle=False)
                        self.failure_observation = {
                            'injection': 'raise after the original member-0 first complete model forward, inside K1 owner scope',
                            'actual_device_input_bytes': allocated,
                            'weak_attention_weight_count': len(self.weak_attention_weights),
                            'weak_attention_weights_alive_at_model_return': sum(ref() is not None for ref in self.weak_attention_weights),
                            'weak_input_count': len(self.weak_inputs),
                            'weak_inputs_alive_at_injection': sum(ref() is not None for ref in self.weak_inputs),
                            'member_optimized_calls': self.kinetics.summary()['member_optimized_calls'],
                            'partial_output_path': str(self.injection_path.relative_to(self.root)),
                            'partial_output_sha256': sha(self.injection_path),
                            'partial_shape': list(host.shape),
                            'extra_diagnostic_output_D2H': 1,
                        }
                        # Do not retain CUDA tensors through the injected traceback.
                        del host, result, chunk, state
                        raise InjectedModelFailure('intentional after actual member-0 first-batch output and live GPU inputs')
                    return original_factory(index, actual_forward_then_fail)
                self.kinetics._model_wrapper = factory
            manager = self.kinetics.activate(self.s1_runtime, self.models)
            try:
                manager.__enter__()
            except BaseException:
                self.kinetics.__dict__.pop('_model_wrapper', None)
                self.kinetics.__dict__.pop('_attention_wrapper', None)
                raise
            self.kinetics_manager = manager
            original = self.s1_runtime.request
            self.request_original = original
            @contextmanager
            def observed_request(*args, **kwargs):
                state = None
                try:
                    with original(*args, **kwargs) as state:
                        yield state
                finally:
                    if state is not None:
                        self.last_request_summary = state.summary()
            self.observed_request = observed_request
            self.s1_runtime.request = observed_request

        def run(self, name, arm, dest, phase, rep, **kwargs):
            try:
                row = super().run(name, arm, dest, phase, rep, **kwargs)
            finally:
                self.close_kinetics()
            row['kinetics_candidate'] = self.kinetics.summary()
            row['kinetics_restoration'] = self.last_transition
            row['not_performance_sample'] = True
            atomic(Path(dest) / 'receipt.json', row)
            return row
    return LifecycleRuntime


def assert_candidate_restored(state, *, failure):
    summary, transition = state['kinetics'], state['kinetics_restoration']
    assert summary['restored'] and not summary['active'] and not summary['cleanup_conflicts']
    assert bool(summary['failure']) is failure
    assert transition['equal'] and transition['before'] == transition['after']
    assert transition['scope_empty'] and transition['call_lock_released']
    assert transition['guard_released'] and transition['installed_entries_empty']
    assert not transition['candidate_owned_science_objects']
    assert state['accepted_runtime']['restored'] is True
    assert state['base_fingerprint'] == state['after_runtime_fingerprint']


def release_baseline_owners(rt, model_refs, weight_refs):
    """End-of-process diagnostic, after restoration comparisons and all requests.

    Retain exited adapters while explicitly dropping baseline-owned resident
    models. This is not a claim that production model caches release per request.
    """
    identifiers = [id(model) for model in rt.models]
    before = {'models': sum(ref() is not None for ref in model_refs),
              'weights': sum(ref() is not None for ref in weight_refs)}
    assert before['models'] == 10 and before['weights'] == len(weight_refs)
    rt.service._load_cached_model_objects.cache_clear()
    # Frozen baseline inspect_models owns these registry entries even with all
    # acceleration disabled. Remove only entries for this process's ten models.
    for ident in identifiers:
        rt.probes._MODELS.pop(ident, None)
        rt.streaming._HOOKED_MODELS.pop(ident, None)
    rt.models = rt.bundle = None
    rt.last_request_summary = None
    gc.collect()
    after = {'models': sum(ref() is not None for ref in model_refs),
             'weights': sum(ref() is not None for ref in weight_refs)}
    result = {'baseline_ownership_release_is_explicit': True,
              'released_baseline_owners': ['service._load_cached_model_objects.cache_clear',
                  'harness.models', 'harness.bundle', 'probes._MODELS:owned-ten',
                  'streaming._HOOKED_MODELS:owned-ten'],
              'exited_adapters_retained': len(rt.adapters),
              'weak_model_count': len(model_refs), 'weak_weight_count': len(weight_refs),
              'before_owner_release': before, 'after_owner_release': after,
              'candidate_owned_science_objects': [owned_objects(k, rt.torch.Tensor, rt.torch.nn.Module) for k in rt.adapters]}
    return result


def execute(args):
    if sys.platform != 'linux':
        raise SystemExit('Actual model execution is Linux/CUDA only')
    root, candidate, output = map(lambda p: Path(p).resolve(), (args.root, args.candidate_root, args.output_dir))
    assert output.is_relative_to(root) and output != root
    if output.exists():
        if (output / 'lifecycle.json').exists():
            return verify(args)
        raise RuntimeError('Partial diagnostic already exists; preserve it and reconcile saved state before continuing')
    output.mkdir(parents=True)
    worker = load_worker(candidate)
    base = worker.load_baseline(root)
    sys.path.insert(0, str(candidate / 'implementation'))
    sources = worker.candidate_sources(root)
    shutil.copyfile(Path(__file__), output / 'kinetics_lifecycle.py')
    rt = make_runtime(base)(root, root / 'results/experiments/shared_feature_cache')
    owner = base.read(root / 'results/experiments/shared_features.json')
    assert owner['manifest']['identity'] == rt.feature_identity()
    for entry in owner['cache_files']:
        assert sha(root / entry['path']) == entry['sha256']
    def no_generation(*args, **kwargs):
        raise AssertionError('Lifecycle must reuse this-run verified original ESM features')
    rt.esm._run_esm_batch = no_generation
    rt.esm.get_single_esm_repr = no_generation
    rt.expected_features = {e['sequence_sha256']: e for e in owner['manifest']['features']}
    rt.preload(args.workload)
    rt.configure('A')
    base_before = snapshot(rt.models)
    model_refs = [weakref.ref(m) for m in rt.models]
    weight_refs = [weakref.ref(p) for m in rt.models for p in m.parameters()]
    steps = []
    contract = {'schema': 1, 'workload': args.workload, 'environment': rt.env,
                'kinetics_sources': sources, 'script_sha256': sha(Path(__file__)),
                'baseline_helpers': json.loads((candidate / 'experiments/BASELINE_HELPERS.json').read_text()),
                'diagnostic_only': True, 'no_performance_claim': True,
                'sequence': ['K0_reference', 'K1_injected_model_failure', 'K1_recovery', 'K0_after_teardown']}
    atomic(output / 'contract.json', contract)
    try:
        for index, (label, enabled, inject) in enumerate((
                ('K0_reference', False, False), ('K1_injected_model_failure', True, True),
                ('K1_recovery', True, False), ('K0_after_teardown', False, False)), 1):
            rt.enabled, rt.inject = enabled, inject
            dest = output / label
            rt.injection_path = dest / 'partial_member0_first_batch.npy'
            caught = None
            try:
                rt.run(args.workload, 'S1', dest, 'kinetics-lifecycle', index,
                       diagnostic=False, cold=False, profile=False)
            except InjectedModelFailure as error:
                caught = str(error)
                if not inject:
                    raise
            finally:
                rt.close_runtime()
            gc.collect()
            assert bool(caught) is inject
            step = {'label': label, 'enabled': enabled, 'injected': inject,
                    'caught_expected_exception': caught, 'kinetics': rt.kinetics.summary(),
                    'kinetics_restoration': rt.last_transition,
                    'request': rt.last_request_summary,
                    'accepted_runtime': rt.s1_runtime.manifest(),
                    'base_fingerprint': base_before, 'after_runtime_fingerprint': snapshot(rt.models)}
            assert_candidate_restored(step, failure=inject)
            if inject:
                assert not (dest / 'receipt.json').exists(), 'Failed request is not a complete prediction'
                step['failure_observation'] = dict(rt.failure_observation)
                step['failure_observation']['weak_inputs_alive_after_request'] = sum(ref() is not None for ref in rt.weak_inputs)
                step['failure_observation']['weak_attention_weights_alive_after_request'] = sum(ref() is not None for ref in rt.weak_attention_weights)
                assert step['failure_observation']['weak_inputs_alive_after_request'] == 0
                assert step['request']['closed'] and step['request']['request_cache_empty_on_exit']
                assert step['request']['stream']['retained_after_request_bytes'] == 0
                assert 'InjectedModelFailure' in step['request']['failure']
                assert step['kinetics']['member_optimized_calls'] == [1] + [0] * 9
            else:
                receipt = dest / 'receipt.json'
                step['receipt'] = {'path': str(receipt.relative_to(root)), 'sha256': sha(receipt)}
                counts = [math.ceil(len(rt.frames[args.workload]) / 50)] * 10
                assert step['kinetics']['member_optimized_calls'] == (counts if enabled else [0] * 10)
            steps.append(step)
            atomic(output / 'progress.json', {'contract': contract, 'steps': steps})
            print('LIFECYCLE_STEP_COMPLETE', label, flush=True)
        owner_release = release_baseline_owners(rt, model_refs, weight_refs)
        report = {'contract': contract, 'steps': steps, 'owner_release': owner_release,
                  'not_performance_sample': True}
        atomic(output / 'lifecycle.json', report)
        assert owner_release['after_owner_release'] == {'models': 0, 'weights': 0}, owner_release
        assert not any(owner_release['candidate_owned_science_objects'])
    finally:
        rt.close_runtime()
    return verify(args)


def verify(args):
    """Re-read full precision arrays and features; no framework imports."""
    root, candidate, output = map(lambda p: Path(p).resolve(), (args.root, args.candidate_root, args.output_dir))
    pinned = json.loads((candidate / 'experiments/BASELINE_HELPERS.json').read_text())
    for name, digest in pinned.items():
        assert sha(root / 'experiments' / name) == digest
    sys.path.insert(0, str(root / 'experiments'))
    from verify_gpu_results import Verifier, decode_npy
    report = json.loads((output / 'lifecycle.json').read_text())
    contract = report['contract']
    assert contract['script_sha256'] == sha(output / 'kinetics_lifecycle.py') == sha(Path(__file__))
    assert contract['baseline_helpers'] == pinned
    assert report['not_performance_sample'] and contract['diagnostic_only'] and contract['no_performance_claim']
    source = contract['kinetics_sources']
    saved_manifest = checked_path(root, source['manifest'])
    assert json.loads(saved_manifest.read_text()) == source['files']
    for name, digest in source['files'].items():
        assert sha(checked_path(saved_manifest.parent, name)) == digest
        assert sha(candidate / name) == digest
    v = Verifier(root, finite_ledger_root=output / 'independent_finite_checks')
    v.prepare_owner()
    assert [s['label'] for s in report['steps']] == contract['sequence']
    assert contract['sequence'] == ['K0_reference', 'K1_injected_model_failure', 'K1_recovery', 'K0_after_teardown']
    successes = []
    for step in report['steps']:
        injected = step['label'] == 'K1_injected_model_failure'
        assert step['injected'] is injected
        assert step['enabled'] is step['label'].startswith('K1_')
        assert_candidate_restored(step, failure=injected)
        assert step['accepted_runtime']['validation_status'] == 'verified'
        assert step['accepted_runtime']['selected_backend'] == 'accepted_stream'
        summary = step['kinetics']
        assert summary['enabled'] is step['enabled'] and summary['engaged'] is step['enabled']
        assert summary['optimized_calls'] == sum(summary['member_optimized_calls'])
        assert summary['passthrough_calls'] == 0 and not summary['accepted_certificate_applies_to_K1']
        if step['enabled']:
            assert summary['validation_status'] == 'experimental_unvalidated'
            assert summary['source_identity']['candidate_sources'] == {Path(name).name: digest for name, digest in source['files'].items() if name.startswith('implementation/catpred_kinetics/')}
        else:
            assert summary['validation_status'] == 'off'
        if injected:
            obs = step['failure_observation']
            assert step['caught_expected_exception'] and 'InjectedModelFailure' in step['kinetics']['failure']
            assert obs['actual_device_input_bytes'] > 0
            assert obs['weak_input_count'] > 0 and obs['weak_inputs_alive_at_injection'] == obs['weak_input_count']
            assert obs['weak_inputs_alive_after_request'] == 0
            assert obs['weak_attention_weight_count'] == 1
            assert obs['weak_attention_weights_alive_at_model_return'] == obs['weak_attention_weights_alive_after_request'] == 0
            assert obs['member_optimized_calls'] == step['kinetics']['member_optimized_calls'] == [1] + [0] * 9
            request = step['request']
            assert request['closed'] and request['request_cache_empty_on_exit']
            assert request['stream']['retained_after_request_bytes'] == 0
            assert 'InjectedModelFailure' in request['failure']
            partial_path = checked_path(root, obs['partial_output_path'])
            assert sha(partial_path) == obs['partial_output_sha256']
            partial = decode_npy(partial_path.read_bytes())
            assert partial['shape'] == (50, 2) and partial['dtype'] == '<f4'
            assert all(map(math.isfinite, partial['values']))
        else:
            row = v.receipt(step['receipt']['path'], step['receipt']['sha256'])
            assert row['workload'] == contract['workload'] and row['phase'] == 'kinetics-lifecycle'
            assert row['kinetics_candidate'] == step['kinetics']
            assert row['kinetics_restoration'] == step['kinetics_restoration']
            assert row['runtime_request'] == step['request']
            assert row['environment'] == contract['environment']
            assert row['feature_preflight_checks'] == row['_features'][0]['sequence_count']
            counts = [math.ceil(row['rows'] / 50)] * 10
            assert step['kinetics']['member_optimized_calls'] == (counts if step['enabled'] else [0] * 10)
            assert step['kinetics']['member_calls'] == (counts if step['enabled'] else [0] * 10)
            successes.append(row)
    assert len(successes) == 3
    reference = v.arrays[successes[0]['_receipt']]['member_raw']
    assert partial['payload'] == reference['payload'][:50 * 2 * 4]
    v.pair(successes[0], successes[1], 'K0_reference_vs_K1_after_model_exception')
    v.pair(successes[0], successes[2], 'K0_reference_vs_K0_after_K1_teardown')
    release = report['owner_release']
    assert release['baseline_ownership_release_is_explicit'] and release['exited_adapters_retained'] == 4
    assert release['weak_model_count'] == 10 and release['weak_weight_count'] > 0
    assert release['before_owner_release'] == {'models': 10, 'weights': release['weak_weight_count']}
    assert release['after_owner_release'] == {'models': 0, 'weights': 0}
    assert release['candidate_owned_science_objects'] == [[], [], [], []]
    result = {'passed': True, 'lifecycle_path': str((output / 'lifecycle.json').relative_to(root)),
              'lifecycle_sha256': sha(output / 'lifecycle.json'), 'comparisons': v.comparisons,
              'partial_member0_first_batch_exact': True,
              'actual_postallocation_model_exception_and_recovery': True,
              'candidate_restored_all_four_activations': True,
              'accepted_runtime_restored_all_four_activations': True,
              'candidate_models_and_weights_released_after_explicit_baseline_owner_release': True,
              'verification_uses_saved_raw_arrays': True, 'no_performance_claim': True}
    atomic(output / 'independent.json', result)
    print(json.dumps({'passed': True, 'report': str(output / 'independent.json'),
                      'report_sha256': sha(output / 'independent.json')}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Prepared accepted baseline root on the owned runtime')
    parser.add_argument('--candidate-root', required=True, help='Prepared K1 root containing implementation/ and experiments/')
    parser.add_argument('--output-dir', required=True, help='New diagnostic directory beneath --root')
    parser.add_argument('--workload', choices=('mixed_valid_2047', 'reuse_4096', 'screening_16384'), default='mixed_valid_2047')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run', action='store_true')
    action.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    return execute(args) if args.run else verify(args)


if __name__ == '__main__':
    raise SystemExit(main())
