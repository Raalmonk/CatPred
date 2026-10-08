"""Mechanical integration tests with stdlib fakes; no torch/ESM model import."""

def main():
    from pathlib import Path
    import contextlib,hashlib,importlib.util,json,sys,tempfile,types
    source=Path(__file__).with_name('protein.py')
    if not source.exists():source=Path(__file__).resolve().parents[1]/'catpred_accel'/'protein.py'
    package=types.ModuleType('test_protein_package');package.__path__=[];sys.modules[package.__name__]=package
    keys=types.ModuleType('test_protein_package.cache_keys');keys.digest_record=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest();sys.modules[keys.__name__]=keys
    spec=importlib.util.spec_from_file_location('test_protein_package.protein',source);p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
    checks=[]
    class Tensor:
     def __getitem__(self,key):return self
     def to(self,*args):return self
     def detach(self):return self
    class Param:
     device=types.SimpleNamespace(type='cuda');dtype='fp32'
     def is_floating_point(self):return True
    class Model:
     training=False
     def eval(self):return self
     def cuda(self):return self
     def parameters(self):return iter([Param()])
     def buffers(self):return iter([])
    class Alphabet:
     def get_batch_converter(self):return lambda data:(None,None,Tensor())
    class Adapter:
     def __init__(self,model,mode,allow_unvalidated):self.model=model;self.mode=mode
     def forward(self,tokens):return Tensor()
     def diagnostics(self):return dict(requested_mode=self.mode,used_mode=self.mode,engaged=self.mode!='off',fallback_reason=None,counters={})
     def close(self):self.model=None
    with tempfile.TemporaryDirectory(prefix='protein_context_') as tmp:
     root=Path(tmp);fake_torch=types.ModuleType('torch');fake_torch.Tensor=Tensor;fake_torch.float32='fp32';fake_torch.no_grad=contextlib.nullcontext;fake_torch.cuda=types.SimpleNamespace(empty_cache=lambda:None);fake_torch.hub=types.SimpleNamespace(get_dir=lambda:str(root/'hub'));sys.modules['torch']=fake_torch
     loading=types.ModuleType('test_protein_package.protein_loading');loading.preflight=lambda *a:dict(weights_checked=True);sys.modules[loading.__name__]=loading
     data=types.ModuleType('catpred.data');data.cache_utils=types.SimpleNamespace(CACHE_PATH=root);sys.modules['catpred.data']=data
     p._sha=lambda path:p.CALLER_SHA;p._source_guard=lambda:None;p._context=lambda torch:dict(context='fake-only');p.FeaturesAdapter=Adapter;p.load_esm2=lambda *a,**k:(Model(),Alphabet(),dict(engaged=False,used_mode='off'))
     esm=types.SimpleNamespace(__file__='synthetic_esm_utils.py',DEFAULT_ESM_BATCH_SIZE=4,ESM_MAX_LENGTH=2048,PROTEIN_EMBED_USE_CPU=False,ESM_CACHE_PATH='original',GLOBAL_VARIABLES={'model':None},PROTEIN_REPR_CONFIG={'esm':{}})
     esm._run_esm_batch=lambda seqs:None;esm.init_esm=lambda:None
     def old_many(seqs,device='cpu',batch_size=4):
      directory=root/esm.ESM_CACHE_PATH;directory.mkdir(parents=True,exist_ok=True)
      missing=[s for s in seqs if not (directory/(hashlib.md5(s.encode()).hexdigest()+'.pt')).exists()]
      for i in range(0,len(missing),batch_size):
       batch=missing[i:i+batch_size];esm._run_esm_batch(batch)
       for s in batch:(directory/(hashlib.md5(s.encode()).hexdigest()+'.pt')).write_text(s)
      return {s:s for s in seqs}
     esm.get_many_esm_reprs=old_many;esm.PROTEIN_REPR_CONFIG['esm']['batch_fn']=old_many
     original=(esm._run_esm_batch,esm.init_esm,esm.ESM_CACHE_PATH,esm.GLOBAL_VARIABLES['model'],esm.get_many_esm_reprs)
     sequences=['AA','B','CCC','DDDD','EEEEE'];events=[]
     with p.features_context(esm,sequences=sequences) as noop:
      assert (esm._run_esm_batch,esm.init_esm,esm.ESM_CACHE_PATH,esm.GLOBAL_VARIABLES['model'],esm.get_many_esm_reprs)==original
     assert noop.summary()['closed'];checks.append('off_is_noop')
     with p.features_context(esm,sequences=sequences,mode='representations',allow_unvalidated=True,batch_observer=events.append) as handle:
      identity=handle.identity();namespace=esm.ESM_CACHE_PATH
      result=esm.get_many_esm_reprs(sequences);assert result==dict(zip(sequences,sequences)) and len(events)==2
      assert [len(e['sequences']) for e in events]==[4,1];checks.append('original_groups_and_success_observer')
      handle.release_model();assert esm.GLOBAL_VARIABLES['model'] is None and esm.ESM_CACHE_PATH==namespace and handle.identity()==identity
      esm.get_many_esm_reprs(sequences);assert len(events)==2;checks.append('release_retains_namespace_and_consumer_cache')
      directory=root/namespace
      next(directory.glob('*.pt')).unlink()
      esm.get_many_esm_reprs(sequences);assert len(events)==4 and len(handle.summary()['incomplete_namespaces_preserved'])==1
      assert list(directory.parent.glob(directory.name+'.incomplete_*'));checks.append('partial_namespace_preserved_and_full_groups_regenerated')
      try:
       with p.features_context(esm,sequences=sequences,mode='head_only',allow_unvalidated=True):raise AssertionError('nested entered')
      except RuntimeError:pass
      assert p._ACTIVE;checks.append('nested_scope_rejected_without_changing_owner')
     assert (esm._run_esm_batch,esm.init_esm,esm.ESM_CACHE_PATH,esm.GLOBAL_VARIABLES['model'],esm.get_many_esm_reprs)==original
     assert handle.summary()['closed'] and not p._ACTIVE;checks.append('all_caller_entries_restored')
     try:
      with p.features_context(esm,sequences=sequences,mode='representations',allow_unvalidated=True) as handle:
       esm.get_many_esm_reprs(list(reversed(sequences)))
     except AssertionError:pass
     else:raise AssertionError('changed padding context accepted')
     assert (esm._run_esm_batch,esm.init_esm,esm.ESM_CACHE_PATH,esm.GLOBAL_VARIABLES['model'],esm.get_many_esm_reprs)==original and not p._ACTIVE
     checks.append('changed_request_rejected_and_exception_restored')
     try:
      with p.features_context(esm,sequences=sequences,allow_unvalidated='false'):pass
     except TypeError:pass
     else:raise AssertionError('truthy string accepted')
     checks.append('strict_boolean_experimental_optin')
    print(json.dumps(dict(all_passed=True,count=len(checks),tests=checks,real_torch_imported=False,real_model_execution=False,protein_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2))

if __name__ == '__main__':
    main()
