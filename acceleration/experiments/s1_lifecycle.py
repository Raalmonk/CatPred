"""Remote-only S1 lifecycle and fail-closed diagnostics, excluded from timing."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import importlib
import math
import shutil
import sys

from evidence import atomic, digest, read, relative, utc


def _baseline_reference(rt, workload):
    found=[]
    for path in sorted((rt.root/'results/experiments/tasks').glob('*/index.json')):
        for entry in read(path).get('receipts',[]):
            p=rt.root/entry['path']
            assert digest(p)==entry['sha256']
            row=read(p)
            if row['phase']=='baseline-gate' and row['arm']=='A' and row['workload']==workload:
                assert row['environment']['runtime_boot_id']==rt.boot
                found.append((row['repetition'],p,row))
    assert len(found)==2,'Lifecycle needs the completed same-runtime original A/A gate'
    _,path,row=sorted(found,key=lambda v:v[0])[0]
    return dict(receipt_path=relative(rt.root,path),receipt_sha256=digest(path),**row)


def _exact_outputs(root, left, right, label):
    from verify_gpu_results import load_npz, compare
    for row in (left,right):
        for kind in ('precision','precision_manifest','consumed_features'):
            assert digest(root/row[kind+'_path'])==row[kind+'_sha256']
    a,b=load_npz(root/left['precision_path']),load_npz(root/right['precision_path'])
    arrays={key:compare(a[key],b[key]) for key in a}
    metadata=read(root/left['precision_manifest_path'])==read(root/right['precision_manifest_path'])
    fa,fb=(read(root/r['consumed_features_path']) for r in (left,right))
    entries=lambda f:{e['sequence_sha256']:(e['shape'],e['dtype'],e['bytes'],e['sha256']) for e in f['features']}
    features=fa['identity']==fb['identity'] and entries(fa)==entries(fb)
    for f in (fa,fb):
        for e in f['features']:assert digest(root/e['path'])==e['sha256']
    result=dict(label=label,arrays=arrays,metadata_equal=metadata,features_exact=features,
                exact=metadata and features and all(v['exact'] and v['finite'] for v in arrays.values()))
    assert result['exact'],result
    return result


@contextmanager
def _original_predictor_entries(rt):
    """Temporarily expose the true original entry points for private vendoring.

    Existing benchmark hooks remain registered and inert (their contexts are off).
    The public capture of host aggregation/CSV remains available. Every temporary
    entry here is restored even if private construction or request execution fails.
    """
    model_module=importlib.import_module('catpred.models.model')
    mpn_module=importlib.import_module('catpred.models.mpn')
    data_module=importlib.import_module('catpred.data.data')
    assert model_module.MoleculeModel.forward is rt.reuse.ORIGINAL_MODEL_FORWARD
    assert mpn_module.MPNEncoder.forward is rt.reuse.ORIGINAL_MPN_FORWARD
    assert data_module.MoleculeDataset.batch_graph is rt.reuse.ORIGINAL_BATCH_GRAPH
    predict=importlib.import_module('catpred.train.predict')
    unc=importlib.import_module('catpred.uncertainty.uncertainty_predictor')
    changes=[(predict,'predict',rt.probes._ORIGINAL_PREDICT),
             (unc,'predict',rt.probes._ORIGINAL_UNCERTAINTY_PREDICT),
             (unc.MVEPredictor,'calculate_predictions',rt.streaming._ORIGINAL_CALCULATE)]
    absent=object();saved=[]
    try:
        for obj,name,replacement in changes:
            saved.append((obj,name,getattr(obj,name),False));setattr(obj,name,replacement)
        for model in rt.models:
            entry=rt.probes._MODELS[id(model)]
            for obj,name,replacement in ((model.rotary_embedder,'rotate_queries_or_keys',entry['original_rotate']),
                                         (model.multihead_attn,'forward',entry['original_attention'])):
                saved.append((obj,name,obj.__dict__.get(name,absent),True));setattr(obj,name,replacement)
        yield
    finally:
        for obj,name,old,instance in reversed(saved):
            if instance and old is absent:obj.__dict__.pop(name,None)
            else:setattr(obj,name,old)


@contextmanager
def _private_raw_capture(rt, runtime):
    """Capture existing required D2H arrays; perform no additional transfer."""
    probes=runtime.components['probes'];old=probes._cpu_numpy
    counted=probes._COUNTED_PREDICT.__globals__;old_counted=counted['_phase2_cpu_numpy']
    def transfer(tensor,diagnostic=False,member=None):
        result=old(tensor,diagnostic=diagnostic,member=member)
        state=rt.capture._ACTIVE.get()
        if state is not None and not diagnostic:
            assert member in range(10)
            state.member_pieces[member].append(result)
        return result
    probes._cpu_numpy=transfer;counted['_phase2_cpu_numpy']=transfer
    try:yield
    finally:
        probes._cpu_numpy=old;counted['_phase2_cpu_numpy']=old_counted


def _guard_checks(rt):
    from catpred_accel import Runtime, RuntimeConfig
    from catpred_accel import _identity
    from catpred_accel.api import RuntimeStateError
    from catpred_accel.capabilities import UnsupportedCapability
    from catpred_accel.validation.lifecycle import fingerprint
    before=fingerprint(rt.models);records=[]
    injected=dict(packing=rt.packing,reuse=rt.reuse,probes=rt.probes,streaming=rt.streaming)
    config=RuntimeConfig(allow_unvalidated=True,fallback='raise')
    def rejected(label,operation,exception,fragment):
        try:operation()
        except exception as error:
            assert fragment in str(error),(label,str(error))
            records.append(dict(case=label,passed=True,error_type=type(error).__name__,reason=str(error)))
        else:raise AssertionError(label+' did not reject')
        assert fingerprint(rt.models)==before,label+' changed live caller state'
    def activate(config,scalers):
        with Runtime(config,components=injected,certifications=[]).activate(rt.models,scalers):
            raise AssertionError('Unsupported activation entered')
    # Actual stack without supplied certification must use stock by default.
    with Runtime(RuntimeConfig(),components=injected,certifications=[]).activate(rt.models,rt.bundle[3]) as stock:
        manifest=stock.manifest();assert not manifest['engaged'] and manifest['selected_backend']=='stock'
    records.append(dict(case='uncertified_default_stock',passed=True,reason=manifest['reason']))
    bad_scalers=[list(member) for member in rt.bundle[3]];bad_scalers[0][1]=object()
    rejected('non_null_input_scaler',lambda:activate(config,bad_scalers),UnsupportedCapability,'unsupported')
    expected=_identity.SCIENTIFIC_FILES.copy()
    try:
        _identity.SCIENTIFIC_FILES[next(iter(expected))]='0'*64
        rejected('pinned_source_digest_mismatch',lambda:activate(config,rt.bundle[3]),UnsupportedCapability,'unsupported')
    finally:_identity.SCIENTIFIC_FILES.clear();_identity.SCIENTIFIC_FILES.update(expected)
    rejected('invalid_device_input_budget',lambda:activate(RuntimeConfig(input_budget_bytes=123,allow_unvalidated=True,fallback='raise'),rt.bundle[3]),UnsupportedCapability,'only 1/2/4 GiB')
    cap=tuple(rt.env['capability']);wrong='h100' if cap==(12,0) else 'g4'
    rejected('explicit_wrong_hardware',lambda:activate(RuntimeConfig(backend=wrong,allow_unvalidated=True,fallback='raise'),rt.bundle[3]),UnsupportedCapability,'requires observed')
    runtime=Runtime(config,components=injected,certifications=[])
    with runtime.activate(rt.models,rt.bundle[3]):
        try:
            with runtime.request(diagnostic=True) as invalid:
                loader=SimpleNamespace(_batch_size=49,_shuffle=False,_class_balance=False,_num_workers=0)
                predictor=SimpleNamespace(models=rt.models,scalers=rt.bundle[3],test_data_loader=loader)
                rt.streaming._calculate_stream(predictor)
        except rt.streaming.StreamUnsupported as error:
            assert 'batch=50' in str(error)
            summary=invalid.summary();assert summary['request_cache_empty_on_exit']
            assert summary['stream']['retained_after_request_bytes']==0 and summary['stream']['device_cache_peak_bytes']==0
            records.append(dict(case='non_original_batch_shape',passed=True,reason=str(error),request=summary,no_device_input_allocation=True))
        else:raise AssertionError('Wrong original batch size did not reject')
        # Mutating a scalar configuration between activation and request is refused.
        args=rt.models[0].args;old=args.add_esm_feats
        try:
            args.add_esm_feats=not old
            try:
                with runtime.request():raise AssertionError('Changed model configuration entered request')
            except RuntimeStateError as error:
                assert 'parameters/scalers changed' in str(error)
                records.append(dict(case='live_input_configuration_changed',passed=True,reason=str(error)))
        finally:args.add_esm_feats=old
    assert fingerprint(rt.models)==before
    return dict(passed=True,cases=records,caller_entries_restored=True)


def run_lifecycle(rt,workload,attempt):
    assert sys.platform=='linux' and rt.torch.cuda.is_available()
    from catpred_accel import Runtime, RuntimeConfig
    from catpred_accel.validation.lifecycle import validate_lifecycle,fingerprint
    attempt=Path(attempt);destination=attempt/'lifecycle';destination.mkdir(parents=True,exist_ok=True)
    assert not (destination/'receipt.json').exists(),'Completed lifecycle evidence already exists'
    rt.configure('A');assert rt.models is not None
    reference=_baseline_reference(rt,workload);before=fingerprint(rt.models)
    all_paths={};comparisons=[];outcomes={}
    config=RuntimeConfig(allow_unvalidated=True,fallback='raise')
    for path_kind in ('injected','private'):
        successes=[];calls=[];active=[]
        def factory():
            components=(dict(packing=rt.packing,reuse=rt.reuse,probes=rt.probes,streaming=rt.streaming)
                        if path_kind=='injected' else None)
            runtime=Runtime(config,components=components,certifications=[])
            active[:] = [runtime]
            return runtime
        def request_fn():
            index=len(calls);dest=destination/path_kind/('call_'+str(index));dest.mkdir(parents=True,exist_ok=False)
            record=dict(path_kind=path_kind,call_index=index,workload=workload,rows=len(rt.frames[workload]),
                        environment=rt.env,active_sources=rt.sources,checkpoints=rt.checkpoints,
                        input_row_ids=rt.frames[workload].row_id.tolist(),excluded_from_headline=True)
            calls.append(record);rt.forward.clear();rt.consumed={};rt.feature_preflight_checks=0
            start_cache=rt.signature();esm_before=len(rt.esm_batches)
            try:
                with rt.capture.request_context(rt.frames[workload].row_id.tolist()) as capture:
                    if path_kind=='private':
                        with _private_raw_capture(rt,active[0]):output=rt.pipeline(rt.request(workload),results_dir=str(dest))
                    else:output=rt.pipeline(rt.request(workload),results_dir=str(dest))
                rt.torch.cuda.synchronize()
            except BaseException as error:
                record.update(completed=False,failure_type=type(error).__name__,failure=str(error),
                    member_pieces={str(i):sum(len(a) for a in pieces) for i,pieces in capture.member_pieces.items()},
                    model_forward_counts=[rt.forward[i] for i in range(10)])
                # Preserve only the actually produced member arrays. No absent rows
                # or aggregates are fabricated, and this is not a completed result.
                partial={str(i):rt.np.concatenate(pieces,axis=0) for i,pieces in capture.member_pieces.items() if pieces}
                record['partial_arrays']={key:dict(dtype=value.dtype.str,shape=list(value.shape),finite=bool(rt.np.isfinite(value).all())) for key,value in partial.items()}
                assert all(value['finite'] for value in record['partial_arrays'].values())
                rt.np.savez(dest/'partial_member_raw.npz',**partial)
                record.update(partial_member_path=relative(rt.root,dest/'partial_member_raw.npz'),
                              partial_member_sha256=digest(dest/'partial_member_raw.npz'))
                atomic(dest/'call.json',record)
                raise
            record.update(completed=True,model_forward_counts=[rt.forward[i] for i in range(10)],
                          feature_preflight_checks=rt.feature_preflight_checks,
                          feature_cache_unchanged=rt.signature()==start_cache,
                          actual_ESM_forward_batches=rt.esm_batches[esm_before:])
            assert record['model_forward_counts']==[math.ceil(len(rt.frames[workload])/50)]*10
            assert record['feature_cache_unchanged'] and not record['actual_ESM_forward_batches']
            record.update(capture.save(dest))
            features=rt.save_features(rt.consumed);atomic(dest/'consumed_features.json',features)
            record.update(consumed_features_path=relative(rt.root,dest/'consumed_features.json'),consumed_features_sha256=digest(dest/'consumed_features.json'))
            for key in ('precision_path','precision_manifest_path'):record[key]=relative(rt.root,record[key])
            record.update(prediction_path=relative(rt.root,output),prediction_sha256=digest(output))
            assert record['feature_preflight_checks']==features['sequence_count']
            atomic(dest/'call.json',record);record['call_path']=relative(rt.root,dest/'call.json');record['call_sha256']=digest(dest/'call.json')
            comparisons.append(_exact_outputs(rt.root,reference,record,path_kind+': original A versus call '+str(index)))
            successes.append(record);return record
        def assert_exact(first,after):
            comparisons.append(_exact_outputs(rt.root,first,after,path_kind+': successful versus post-exception'))
            return True
        if path_kind=='private':
            with _original_predictor_entries(rt):
                result=validate_lifecycle(factory,rt.models,rt.bundle[3],request_fn,assert_exact)
        else:result=validate_lifecycle(factory,rt.models,rt.bundle[3],request_fn,assert_exact)
        assert len(calls)==3 and len(successes)==2 and calls[1]['completed'] is False
        assert fingerprint(rt.models)==before,path_kind+' did not restore benchmark caller state'
        assert not any(name.startswith('catpred_accel._session_') for name in sys.modules),'Private modules survived deactivation'
        outcomes[path_kind]=result
        all_paths[path_kind]=dict(calls=[dict(path=relative(rt.root,destination/path_kind/('call_'+str(i))/'call.json'),
                                                  sha256=digest(destination/path_kind/('call_'+str(i))/'call.json')) for i in range(3)],
                                   runtime_manifest=active[0].manifest())
        atomic(destination/(path_kind+'_lifecycle.json'),result)
    # Reuse the same capture closure for two ordinary budget diagnostics. The
    # failure/restore paths above already exercise default 2 GiB; these retain
    # the identical original workload, batch composition, padding and arithmetic.
    budget_checks={}
    for gib in (1,4):
        path_kind='budget_'+str(gib)+'g';calls=[];successes=[];active=[]
        budget_config=RuntimeConfig(input_budget_bytes=gib<<30,allow_unvalidated=True,fallback='raise')
        runtime=Runtime(budget_config,components=dict(packing=rt.packing,reuse=rt.reuse,probes=rt.probes,streaming=rt.streaming),certifications=[])
        active.append(runtime)
        with runtime.activate(rt.models,rt.bundle[3]):
            with runtime.request(diagnostic=True) as receipt:record=request_fn()
        summary=receipt.summary();stream=summary['stream']
        assert summary['request_cache_empty_on_exit'] and stream['retained_after_request_bytes']==0
        assert stream['budget_bytes']==gib<<30 and 0<stream['device_cache_peak_bytes']<=gib<<30
        assert all(c['released'] and c['actual_device_cache_bytes_after_clear']==0 and c['device_tensor_references_alive_after_clear']==0 for c in stream['chunks'])
        assert fingerprint(rt.models)==before
        callpath=destination/path_kind/'call_0/call.json'
        budget_checks[str(gib)]=dict(passed=True,input_budget_bytes=gib<<30,call=dict(path=relative(rt.root,callpath),sha256=digest(callpath)),request=summary,runtime_manifest=runtime.manifest(),excluded_from_headline=True)
    guards=_guard_checks(rt);assert fingerprint(rt.models)==before
    row=dict(schema=1,passed=True,workload=workload,environment=rt.env,active_sources=rt.sources,
             reference=dict(receipt_path=reference['receipt_path'],receipt_sha256=reference['receipt_sha256']),
             paths=all_paths,outcomes=outcomes,comparisons=comparisons,guards=guards,budget_checks=budget_checks,
             caller_entry_fingerprint=before,caller_exit_fingerprint=fingerprint(rt.models),
             private_and_injected_restored_to_baseline=True,excluded_from_headline=True,completed_utc=utc())
    path=destination/'receipt.json';atomic(path,row)
    return dict(receipt_path=relative(rt.root,path),receipt_sha256=digest(path))
