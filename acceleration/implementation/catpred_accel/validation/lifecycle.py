"""Call explicitly on the owned CUDA runtime after ordinary numerical gates."""
import importlib
import weakref


class InjectedAfterAllocation(RuntimeError):
    pass


def fingerprint(models):
    data=importlib.import_module('catpred.data.data')
    feat=importlib.import_module('catpred.features.featurization')
    model=importlib.import_module('catpred.models.model')
    mpn=importlib.import_module('catpred.models.mpn')
    predict=importlib.import_module('catpred.train.predict')
    unc=importlib.import_module('catpred.uncertainty.uncertainty_predictor')
    functions=[feat.BatchMolGraph.__init__,data.MoleculeDataset.batch_graph,model.MoleculeModel.forward,
               mpn.MPNEncoder.forward,predict.predict,unc.predict,unc.MVEPredictor.calculate_predictions]
    members=[]
    for member in models:
        members.append(dict(rotary_instance_attribute='rotate_queries_or_keys' in member.rotary_embedder.__dict__,
            rotary_identity=id(member.rotary_embedder.__dict__.get('rotate_queries_or_keys')),
            attention_instance_attribute='forward' in member.multihead_attn.__dict__,
            attention_identity=id(member.multihead_attn.__dict__.get('forward')),
            prehooks={str(k):id(v) for k,v in member._forward_pre_hooks.items()},
            posthooks={str(k):id(v) for k,v in member._forward_hooks.items()}))
    return dict(function_identities=list(map(id,functions)),members=members)


def validate_lifecycle(runtime_factory, models, scalers, request_fn, assert_exact=None):
    """Run unchanged requests with one deliberate exception after real allocation.

    request_fn executes the public pipeline and may return a full-precision result
    handle. If supplied, assert_exact compares the first successful result and
    the fresh request after the failure. Saving partial exception outputs is the
    caller's responsibility; they are never presented as a completed prediction.
    This is a diagnostic, outside every headline timing.
    """
    before=fingerprint(models)
    runtime=runtime_factory()
    weak_inputs=[]
    injected=False
    allocated_before_error=0
    with runtime.activate(models,scalers):
        if not runtime.manifest()['engaged']:
            raise AssertionError('Lifecycle test requires an actually engaged CUDA backend')
        with runtime.request(diagnostic=True) as first:
            reference=request_fn()
        components=runtime.components
        original=components['probes']._predict_dispatch
        def fail_after_member(*args,**kwargs):
            nonlocal injected,allocated_before_error
            result=original(*args,**kwargs)
            if not injected:
                state=components['streaming']._ACTIVE.get()
                chunk=state.active_chunk
                allocated_before_error=chunk.actual_device_bytes()
                if allocated_before_error<=0:
                    raise AssertionError('Injected failure must follow actual device input allocation')
                weak_inputs.extend(weakref.ref(t) for values in chunk.graph_device.values() for t in values[:5])
                weak_inputs.extend(weakref.ref(t) for values in chunk.protein_device.values() for t in values if t is not None)
                injected=True
                raise InjectedAfterAllocation('Intentional failure after first member completed with live CUDA inputs')
            return result
        components['probes']._predict_dispatch=fail_after_member
        try:
            try:
                with runtime.request(diagnostic=True) as failed:
                    request_fn()
            except InjectedAfterAllocation:
                pass
            else:
                raise AssertionError('Post-allocation exception was not triggered')
        finally:
            components['probes']._predict_dispatch=original
        assert injected and allocated_before_error>0
        assert failed.summary()['request_cache_empty_on_exit']
        assert failed.summary()['stream']['retained_after_request_bytes']==0
        assert not any(ref() is not None for ref in weak_inputs), 'Device input survived failed request release'
        with runtime.request(diagnostic=True) as fresh:
            after=request_fn()
        exact_checked=False
        if assert_exact is not None:
            result=assert_exact(reference,after)
            if result is False:
                raise AssertionError('Post-exception fresh request changed full-precision output')
            exact_checked=True
        allocation_free=components['streaming'].budget_rejection_and_cleanup_checks()
    after_first=fingerprint(models)
    assert after_first==before, 'Activation changed caller functions, instance attributes or hooks'
    # No model execution needed to check a second install/uninstall cycle.
    second=runtime_factory()
    with second.activate(models,scalers):
        with second.request() as empty:
            pass
    after_second=fingerprint(models)
    assert after_second==before, 'Second activation was not reversible'
    return dict(passed=True,post_allocation_exception=True,allocated_input_bytes_before_failure=allocated_before_error,
                weak_input_tensor_count=len(weak_inputs),weak_inputs_alive_after_failure=0,
                failed_request=failed.summary(),fresh_request=fresh.summary(),first_request=first.summary(),
                original_function_instance_hook_identities_restored=True,second_activation_restored=True,
                entry_fingerprint=before,first_exit_fingerprint=after_first,second_exit_fingerprint=after_second,
                post_exception_output_exact_checked=exact_checked,allocation_free_checks=allocation_free,
                excluded_from_headline=True)
