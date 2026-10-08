"""Fake objects exercise lifecycle and call forwarding, not numerical equivalence."""
import gc
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch
import weakref
from types import SimpleNamespace
from catpred_kinetics import KineticsAdapter,KineticsStateError,RestorationConflict,UnsupportedKinetics


class Weights:pass
class Attention:
    def __init__(self):self.calls=[];self.weight_refs=[];self.output=object()
    def forward(self,*args,**kwargs):
        self.calls.append((args,kwargs));weights=Weights();self.weight_refs.append(weakref.ref(weights))
        return self.output,weights
class Model:
    def __init__(self):self.multihead_attn=Attention();self.q,self.k,self.v=object(),object(),object();self.extra=None
    def forward(self,*args,**kwargs):
        value,ignored=self.multihead_attn.forward(self.q,self.k,self.v)
        if self.extra is not None:self.extra()
        return value
class FakeGuard:
    def __init__(self,models):
        self.models=tuple(models);self.originals=tuple((m.forward,m.multihead_attn.forward) for m in models)
        self.source_identity={'numerical_validation':'not_run'};self.reject_model=False;self.reject_attention=False
    def validate_model_call(self,index):
        if self.reject_model:raise UnsupportedKinetics('fake model guard rejection')
    def validate_attention_call(self,index,args,kwargs):
        if self.reject_attention or len(args)!=3 or kwargs:raise UnsupportedKinetics('fake attention guard rejection')


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.models=[Model() for _ in range(10)];self.guard=FakeGuard(self.models)
        self.mock=patch('catpred_kinetics.adapter.guards.build_guards',return_value=self.guard);self.mock.start();self.addCleanup(self.mock.stop)
    def adapter(self):return KineticsAdapter(enabled=True,allow_unvalidated=True)
    def test_import_and_off_never_import_torch(self):
        script="import sys;from catpred_kinetics import KineticsAdapter;k=KineticsAdapter();\nwith k.activate(None,[]):pass\nassert k.summary()['restored'] and 'torch' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_requires_explicit_unvalidated(self):
        with self.assertRaises(UnsupportedKinetics):KineticsAdapter(enabled=True)
    def test_strict_bool_options(self):
        with self.assertRaises(TypeError):KineticsAdapter(enabled=1)
    def test_original_qkv_need_weights_and_unused_release(self):
        k=self.adapter()
        with k.activate(object(),self.models):
            for model in self.models:
                self.assertIs(model.forward(),model.multihead_attn.output)
                args,kwargs=model.multihead_attn.calls[-1]
                self.assertEqual(args,(model.q,model.k,model.v));self.assertEqual(kwargs,dict(need_weights=True,average_attn_weights=False))
                gc.collect();self.assertIsNone(model.multihead_attn.weight_refs[-1]())
        self.assertEqual(k.summary()['member_optimized_calls'],[1]*10);self.assertEqual(k.summary()['optimized_calls'],10)
        self.assertTrue(k.summary()['restored']);self.assertFalse(any('forward' in m.__dict__ or 'forward' in m.multihead_attn.__dict__ for m in self.models))
    def test_public_attention_call_preserves_contract(self):
        k=self.adapter();model=self.models[0]
        with k.activate(object(),self.models):
            value,weights=model.multihead_attn.forward('external',need_weights=False,average_attn_weights=True)
            self.assertIs(value,model.multihead_attn.output);self.assertIsNotNone(weights)
            self.assertEqual(model.multihead_attn.calls[-1],(('external',),dict(need_weights=False,average_attn_weights=True)))
        self.assertEqual(k.summary()['passthrough_calls'],1);self.assertEqual(k.summary()['optimized_calls'],0)
    def test_exception_restores_and_subsequent_activation(self):
        k=self.adapter();self.models[0].extra=lambda:(_ for _ in ()).throw(ValueError('injected'))
        with self.assertRaisesRegex(ValueError,'injected'):
            with k.activate(object(),self.models):self.models[0].forward()
        self.assertTrue(k.summary()['restored']);self.models[0].extra=None
        with k.activate(object(),self.models):self.models[0].forward()
        self.assertEqual(k.summary()['optimized_calls'],1);self.assertIsNone(k.summary()['failure'])
    def test_guard_failure_restores_without_original_call(self):
        k=self.adapter();self.guard.reject_model=True
        with self.assertRaises(UnsupportedKinetics):
            with k.activate(object(),self.models):self.models[0].forward()
        self.assertFalse(self.models[0].multihead_attn.calls);self.assertTrue(k.summary()['restored'])
    def test_partial_install_failure_rolls_back(self):
        k=self.adapter();original=k._assign;count=[0]
        def fail(obj,name,value):
            original(obj,name,value);count[0]+=1
            if count[0]==3:raise RuntimeError('partial install')
        with patch.object(k,'_assign',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'partial install'):
                with k.activate(object(),self.models):pass
        self.assertTrue(k.summary()['restored']);self.assertFalse(any('forward' in m.__dict__ or 'forward' in m.multihead_attn.__dict__ for m in self.models))
    def test_nested_activation_rejected(self):
        k=self.adapter()
        with k.activate(object(),self.models):
            with self.assertRaises(KineticsStateError):
                with self.adapter().activate(object(),self.models):pass
    def test_nested_model_rejected_and_cleans_scope(self):
        k=self.adapter();self.models[0].extra=self.models[1].forward
        with self.assertRaises(KineticsStateError):
            with k.activate(object(),self.models):
                self.models[0].extra=lambda:self.models[1].forward();self.models[0].forward()
        self.assertTrue(k.summary()['restored']);self.assertIsNone(k._scope.get())
    def test_cross_thread_model_call_rejected(self):
        k=self.adapter();errors=[]
        with k.activate(object(),self.models):
            def call():
                try:self.models[0].forward()
                except Exception as ex:errors.append(ex)
            worker=threading.Thread(target=call);worker.start();worker.join()
        self.assertIsInstance(errors[0],KineticsStateError);self.assertEqual(k.summary()['optimized_calls'],0)
    def test_external_replacement_preserved_and_conflict_reported(self):
        k=self.adapter();external=lambda *args:None
        with self.assertRaises(RestorationConflict):
            with k.activate(object(),self.models):self.models[0].multihead_attn.forward=external
        self.assertIs(self.models[0].multihead_attn.forward,external)
        self.assertFalse(k.summary()['restored']);self.assertTrue(k.summary()['cleanup_conflicts'])
        del self.models[0].multihead_attn.forward
    def test_existing_instance_method_restored_by_transaction(self):
        # Production guard rejects external model methods; transaction mechanics
        # must nevertheless restore an existing allowed runtime attention method.
        model=self.models[0];before=model.multihead_attn.forward;model.multihead_attn.forward=before
        k=self.adapter()
        with k.activate(object(),self.models):model.forward()
        self.assertIs(model.multihead_attn.forward,before)
    def test_context_body_error_preserved_during_cleanup_conflict(self):
        k=self.adapter();external=lambda:None
        with self.assertRaisesRegex(ValueError,'original'):
            with k.activate(object(),self.models):
                self.models[0].forward=external;raise ValueError('original')
        self.assertIs(self.models[0].forward,external);self.assertFalse(k.summary()['restored']);del self.models[0].forward

if __name__=='__main__':unittest.main()
