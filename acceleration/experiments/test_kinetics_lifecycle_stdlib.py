import ast
from contextlib import contextmanager
from contextvars import ContextVar
import gc
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

class RejectTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('No real torch import is allowed in this test')
sys.meta_path.insert(0, RejectTorch())
CANDIDATE = Path(os.environ.get('CATPRED_KINETICS_CANDIDATE', Path(__file__).resolve().parents[1] / 'source/kinetics_candidate')).resolve()
sys.path.insert(0, str(CANDIDATE / 'implementation'))
sys.path.insert(0, str(CANDIDATE / 'implementation/tests'))
from test_adapter_stdlib import Model, FakeGuard
from catpred_kinetics import KineticsAdapter
spec = importlib.util.spec_from_file_location('life', Path(__file__).resolve().with_name('kinetics_lifecycle.py'))
life = importlib.util.module_from_spec(spec); spec.loader.exec_module(life)

class Tensor:
    shape = (50, 2)
    device = SimpleNamespace(type='cuda')
    dtype = 'float32'
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def numpy(self): return self
class Module(Model):
    def __init__(self):
        super().__init__(); self.multihead_attn.output=Tensor(); self.parameter=Tensor()
    def eval(self): return self
    def parameters(self): return iter((self.parameter,))
class Chunk:
    def __init__(self):
        self.graph_device = {'g': [Tensor() for _ in range(5)]}
        self.protein_device = {'p': [Tensor(), None]}
    def actual_device_bytes(self): return 4096
class RequestState:
    def __init__(self): self.closed=False; self.failure=None
    def summary(self):
        return {'closed': self.closed, 'failure':self.failure,'request_cache_empty_on_exit':self.closed,
                'stream':{'retained_after_request_bytes':0}}
class Accelerated:
    def __init__(self):
        self.restored=False; self.components={'streaming':SimpleNamespace(_ACTIVE=ContextVar('state',default=None))}
    def manifest(self):return {'restored':self.restored}
    @contextmanager
    def request(self, **kwargs):
        state=RequestState(); active=SimpleNamespace(active_chunk=Chunk())
        tok=self.components['streaming']._ACTIVE.set(active)
        try: yield state
        except Exception as error:
            state.failure=type(error).__name__+': '+str(error);raise
        finally:
            active.active_chunk.graph_device.clear();active.active_chunk.protein_device.clear();active.active_chunk=None
            self.components['streaming']._ACTIVE.reset(tok);state.closed=True
BASELINE_ENTRY = [object()]
BASELINE_HISTORY = []

class Base:
    def __init__(self, root):
        self.root=Path(root);self.models=[Module() for _ in range(10)];self.bundle=[None,None,self.models]
        self.torch=SimpleNamespace(Tensor=Tensor,float32='float32',nn=SimpleNamespace(Module=Module))
        self.np=SimpleNamespace(isfinite=lambda x:SimpleNamespace(all=lambda:True),save=lambda p,h,allow_pickle:Path(p).write_bytes(b'fake'))
        self.probes=SimpleNamespace(_MODELS={id(m):m for m in self.models})
        self.streaming=SimpleNamespace(_HOOKED_MODELS={id(m):m for m in self.models})
        self.service=SimpleNamespace(_load_cached_model_objects=SimpleNamespace(cache_clear=lambda:None))
        self.s1_runtime=None
    def configure(self, arm):
        self.close_runtime()
        BASELINE_HISTORY.append(BASELINE_ENTRY[0])
        BASELINE_ENTRY[0] = object()  # Instrumented constructor recreated by configure.
        if arm=='S1': self.s1_runtime=self.new_accelerated_runtime()
    def new_accelerated_runtime(self): return Accelerated()
    def close_runtime(self):
        if self.s1_runtime:self.s1_runtime.restored=True
    def run(self,name,arm,dest,phase,rep,**kwargs):
        Path(dest).mkdir()
        self.configure(arm)
        with self.s1_runtime.request():
            for model in self.models:model.forward()
        return {}
def fp(models):
    return {'graph_constructor':id(BASELINE_ENTRY[0]),'members':[{'model':id(m.__dict__.get('forward')),'attention':id(m.multihead_attn.__dict__.get('forward'))} for m in models]}

class Tests(unittest.TestCase):
    def test_import_and_ast_without_framework(self):
        ast.parse(Path(life.__file__).read_text());self.assertNotIn('torch',sys.modules)
    def test_path_escape_rejected(self):
        with self.assertRaises(AssertionError):life.checked_path('/tmp/a','../other')
    def test_owned_graph_finds_tensor_and_function_closure(self):
        k=KineticsAdapter();held=Tensor()
        k.extra={'tensor':held};self.assertTrue(life.owned_objects(k,Tensor,Module))
        del k.extra
        def fn():return held
        k.extra=fn;self.assertTrue(life.owned_objects(k,Tensor,Module))
        del k.extra;self.assertEqual(life.owned_objects(k,Tensor,Module),[])
    def test_fake_injection_restoration_recovery_and_weak_release(self):
        with tempfile.TemporaryDirectory() as d, patch.object(life,'snapshot',fp), patch('catpred_kinetics.adapter.guards.build_guards',new=lambda runtime,models:FakeGuard(models)):
            rt=life.make_runtime(SimpleNamespace(Runtime=Base))(d)
            before=fp(rt.models);model_refs=[weakref.ref(m) for m in rt.models];weights=[weakref.ref(p) for m in rt.models for p in m.parameters()]
            rt.enabled=rt.inject=True;rt.injection_path=Path(d)/'failed/partial.npy'
            with self.assertRaises(life.InjectedModelFailure):rt.run('test','S1',Path(d)/'failed','lifecycle',1)
            rt.close_runtime();gc.collect()
            self.assertEqual(rt.last_transition['before'],rt.last_transition['after']);self.assertEqual(fp(rt.models),rt.current_baseline_before);self.assertNotEqual(fp(rt.models),before)
            self.assertEqual(rt.kinetics.summary()['member_optimized_calls'],[1]+[0]*9)
            self.assertEqual(rt.last_request_summary['failure'].split(':')[0],'InjectedModelFailure')
            self.assertTrue(rt.last_transition['scope_empty']);self.assertTrue(rt.last_transition['guard_released']);self.assertTrue(rt.last_transition['call_lock_released'])
            self.assertEqual(rt.last_transition['candidate_owned_science_objects'],[])
            self.assertTrue(rt.weak_inputs);self.assertTrue(all(r() is None for r in rt.weak_inputs))
            self.assertNotIn('_model_wrapper',rt.kinetics.__dict__);self.assertNotIn('_attention_wrapper',rt.kinetics.__dict__);self.assertNotIn('request',rt.s1_runtime.__dict__)
            self.assertEqual(rt.failure_observation['weak_attention_weight_count'],1);self.assertEqual(rt.failure_observation['weak_attention_weights_alive_at_model_return'],0);self.assertTrue(all(r() is None for r in rt.weak_attention_weights))
            rt.inject=False
            row=rt.run('test','S1',Path(d)/'recovered','lifecycle',2);rt.close_runtime()
            self.assertEqual(row['kinetics_candidate']['member_optimized_calls'],[1]*10);self.assertEqual(fp(rt.models),rt.current_baseline_before);self.assertNotEqual(fp(rt.models),before)
            release=life.release_baseline_owners(rt,model_refs,weights)
            self.assertEqual(release['after_owner_release'],{'models':0,'weights':0})
            self.assertEqual(release['candidate_owned_science_objects'],[[],[]])
    def test_completed_reference_reconciliation_preserves_evidence(self):
        from types import ModuleType
        old=Path(__file__).with_name('kinetics_lifecycle.before_fingerprint_fix.py')
        source={'identity':'frozen'};pinned={'helper.py':'frozen'}
        transition={'equal':True,'before':{},'after':{},'scope_empty':True,'call_lock_released':True,'guard_released':True,'installed_entries_empty':True,'candidate_owned_science_objects':[]}
        summary={'enabled':False,'restored':True,'active':False,'cleanup_conflicts':[],'failure':None}
        row={'phase':'kinetics-lifecycle','workload':'mixed_valid_2047','repetition':1,'environment':{'process_id':101},'kinetics_candidate':summary,'kinetics_restoration':transition,'runtime_request':{},'runtime_manifest':{'restored':False}}
        contract={'script_sha256':life.sha(old),'workload':'mixed_valid_2047','kinetics_sources':source,'baseline_helpers':pinned,'environment':row['environment']}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'diagnostic';(out/'K0_reference').mkdir(parents=True)
            (out/'kinetics_lifecycle.py').write_bytes(old.read_bytes());life.atomic(out/'contract.json',contract);life.atomic(out/'K0_reference/receipt.json',row)
            saved={str(p):life.sha(p) for p in out.rglob('*') if p.is_file()}
            mod=ModuleType('verify_gpu_results')
            class Verifier:
                def __init__(self,*a,**k):pass
                def prepare_owner(self):pass
                def receipt(self,name,digest):
                    self.path=root/name
                    assert life.sha(self.path)==digest
                    return json.loads(self.path.read_text())
            mod.Verifier=Verifier
            with patch.dict(sys.modules,{'verify_gpu_results':mod}):
                adopted,step=life.reconcile_reference(root,out,'mixed_valid_2047',source,pinned)
                self.assertEqual(adopted,contract);self.assertTrue(step['imported_completed_reference'])
                self.assertFalse(step['accepted_runtime_restoration_observed']);self.assertNotIn('base_fingerprint',step)
                self.assertEqual(saved,{str(p):life.sha(p) for p in out.rglob('*') if p.is_file()})
                (out/'K1_recovery').mkdir()
                with self.assertRaisesRegex(AssertionError,'Never repeat'):
                    life.reconcile_reference(root,out,'mixed_valid_2047',source,pinned)

    def test_restoration_validator_rejects_conflict(self):
        state={'kinetics':{'restored':True,'active':False,'cleanup_conflicts':[],'failure':None},
               'kinetics_restoration':{'equal':True,'before':{},'after':{},'scope_empty':True,'call_lock_released':True,'guard_released':True,'installed_entries_empty':True,'candidate_owned_science_objects':[]},
               'accepted_runtime':{'restored':True},'base_fingerprint':{},'after_runtime_fingerprint':{}}
        life.assert_candidate_restored(state,failure=False)
        state['kinetics_restoration']['candidate_owned_science_objects']=[{'path':'leak'}]
        with self.assertRaises(AssertionError):life.assert_candidate_restored(state,failure=False)

if __name__=='__main__':unittest.main(verbosity=2)
