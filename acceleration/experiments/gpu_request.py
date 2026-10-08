"""Remote-only GPU request worker. No allocation, installation or network control."""
from __future__ import annotations
import argparse, collections, gc, hashlib, importlib, importlib.metadata, json, math, os, random, shutil, sys, threading, time, traceback
from pathlib import Path
from contextlib import ExitStack
from evidence import *

class Runtime:
 def __init__(self,root,cache):
  assert sys.platform=='linux','Inference is remote Linux only'
  self.feature_workloads=tuple(WORKLOADS);self.root=Path(root).resolve();self.out=self.root/'results/experiments';self.cache=Path(cache)
  os.environ.update(CATPRED_CACHE_PATH=str(cache),TORCH_HOME=str(self.root/'torch'),CATPRED_PREDICTION_CACHE_SIZE='0',CATPRED_MODEL_CACHE_SIZE='2',PROTEIN_EMBED_USE_CPU='0',CATPRED_ESM_BATCH_SIZE='4',CATPRED_TRUSTED_DESERIALIZATION_ROOTS=str(self.root))
  # Experimental CPU ESM modifications must never leak into the reference lane.
  for key in ('CATPRED_ESM_META_LOAD','CATPRED_ESM_REPR_ONLY','CATPRED_ESM_REPRESENTATIONS_ONLY'):os.environ.pop(key,None)
  sys.path[:0]=[str(self.root/'baseline'),str(self.root/'CatPred'),str(self.root)]
  import numpy as np,pandas as pd,torch,psutil,packing,reuse,probes,streaming,capture,catpred_rust_packing
  from catpred.inference import PredictionRequest,run_inprocess_prediction_pipeline,service
  from catpred.data import esm_utils
  from rdkit import RDLogger
  self.np,self.pd,self.torch,self.psutil=np,pd,torch,psutil
  self.packing,self.reuse,self.probes,self.streaming,self.capture=packing,reuse,probes,streaming,capture
  self.Request,self.pipeline,self.service,self.esm=PredictionRequest,run_inprocess_prediction_pipeline,service,esm_utils
  RDLogger.DisableLog('rdApp.warning')
  assert torch.cuda.is_available();assert catpred_rust_packing.BUILD_PROFILE=='release'
  torch.set_num_threads(24);torch.set_num_interop_threads(1)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
  torch.manual_seed(20261005);np.random.seed(20261005);random.seed(20261005)
  assert service._PREDICTION_CACHE_SIZE==0
  self.boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  self.sources=source_history(self.root)
  for n,h in BASELINE_HASHES.items():assert digest(self.root/'baseline'/n)==h,n
  for n,h in SCIENCE_HASHES.items():assert digest(self.root/'CatPred'/n)==h,n
  for n,h in INPUT_HASHES.items():assert digest(self.root/'inputs'/(n+'.csv'))==h,n
  prior=read(self.root/'prior_checkpoint_manifest.json')['files'];self.checkpoints=[]
  for entry in prior:
   p=self.root/entry['path'];assert p.stat().st_size==entry['bytes'] and digest(p)==entry['sha256']
   self.checkpoints.append(entry)
  for name,sha in ESM_WEIGHTS.items():
   candidates=list((self.root/'torch').rglob(name));assert len(candidates)==1,name
   assert digest(candidates[0])==sha,name
  props=torch.cuda.get_device_properties(0)
  self.env=dict(runtime_boot_id=self.boot,process_id=os.getpid(),python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),gpu_total_bytes=props.total_memory,torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,cpu_affinity=sorted(os.sched_getaffinity(0)),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),compiled_rust_profile=catpred_rust_packing.BUILD_PROFILE)
  self.fair_esm_sources={}
  for name in ('esm.model.esm2','esm.modules','esm.multihead_attention','esm.rotary_embedding','esm.data','esm.pretrained'):
   source=Path(importlib.import_module(name).__file__);sha=digest(source);target=self.out/'environment_sources'/sha/source.name
   if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
   self.fair_esm_sources[name]=dict(sha256=sha,path=relative(self.root,target))
  self.env['fair_esm_version']=importlib.metadata.version('fair-esm');self.env['fair_esm_sources']=self.fair_esm_sources
  self.frames={n:pd.read_csv(self.root/'inputs'/(n+'.csv')) for n in WORKLOADS}
  for n,count in WORKLOADS.items():assert len(self.frames[n])==count and self.frames[n].row_id.is_unique
  # A declared cold/profile panel preserves the first two entire original batches.
  self.panel='mixed_prefix_100';self.frames[self.panel]=self.frames['mixed_valid_2047'].iloc[:100].copy()
  panelpath=self.root/'inputs'/(self.panel+'.csv')
  if not panelpath.exists():self.frames[self.panel].to_csv(panelpath,index=False)
  assert pd.read_csv(panelpath).equals(self.frames[self.panel])
  self.models=None;self.bundle=None;self.forward=collections.Counter();self.current={};self.consumed={};self.esm_batches=[];self.expected_features=None;self.feature_preflight_checks=0;self.s1_runtime=None;self.s1_manager=None;self.esm_mode='off';self.esm_loader_mode='off';self.protein_identity=None
  probes.configure(r3=False,t1=False,t2=False);capture.install()
  old_build,old_load=service._build_predict_args,service._load_model_objects_for_prediction
  def build(*a,**k):
   v=old_build(*a,**k);v.batch_size=50;assert v.num_workers==0;self.current['num_workers']=v.num_workers;return v
  def load(*a,**k):
   v=old_load(*a,**k);v[0].batch_size=50
   if self.models is None:
    self.models=v[2];self.bundle=v;assert len(self.models)==10
    assert {p.dtype for m in self.models for p in m.parameters()}=={torch.float32}
    self.inspection=dict(reuse=reuse.inspect_models(v[2],v[3]),probes=probes.inspect_models(v[2],v[3]),streaming=streaming.inspect_models(v[2],v[3]))
    for i,m in enumerate(self.models):
     def hook(module,args,index=i):self.forward[index]+=1
     m.register_forward_pre_hook(hook)
   else:assert tuple(map(id,v[2]))==tuple(map(id,self.models))
   return v
  service._build_predict_args=build;service._load_model_objects_for_prediction=load
  orig_many=esm_utils.get_many_esm_reprs;orig_batch=esm_utils._run_esm_batch
  def many(*a,**k):
   values=orig_many(*a,**k)
   if self.expected_features is not None:
    for sequence,tensor in values.items():
     expected=self.expected_features[hashlib.sha256(sequence.encode()).hexdigest()]
     assert list(tensor.shape)==expected['shape'] and tensor.dtype==torch.float32 and tensor.device.type=='cpu'
     assert hashlib.sha256(tensor.numpy().tobytes(order='C')).hexdigest()==expected['sha256'],'Consumed ESM tensor failed exact gate before predictor forward'
     self.feature_preflight_checks+=1
   self.consumed.update(values);return values
  def batch(sequences):
   value=orig_batch(sequences)
   model=esm_utils.GLOBAL_VARIABLES['model'][0]
   assert not model.training
   floating=[t for t in list(model.parameters())+list(model.buffers()) if t.is_floating_point()]
   assert floating and all(t.device.type=='cuda' and t.dtype==torch.float32 for t in floating)
   self.esm_batches.append(dict(sequences=[hashlib.sha256(s.encode()).hexdigest() for s in sequences],device='cuda',dtype='torch.float32',eval=True,floating_parameter_buffer_count=len(floating),all_floating_parameters_buffers_cuda_fp32=True))
   return value
  esm_utils.get_many_esm_reprs=many;esm_utils.PROTEIN_REPR_CONFIG['esm']['batch_fn']=many;esm_utils._run_esm_batch=batch
  self.process=psutil.Process()
 def request(self,name):
  return self.Request(parameter='kcat',input_file=str(self.root/'inputs'/(name+'.csv')),checkpoint_dir=str(self.root/'data/pretrained/production/kcat'),use_gpu=True,repo_root=str(self.root/'CatPred'))
 def signature(self):return [(relative(self.root,p),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(self.cache.rglob('*.pt'))]
 def feature_identity(self):
  import esm
  identity=dict(checkpoints=ESM_WEIGHTS,esm_code=digest(self.root/'CatPred/catpred/data/esm_utils.py'),fair_esm_version=self.env['fair_esm_version'],fair_esm_sources=self.fair_esm_sources,backend='original_fair_esm_2',numeric_contract='FP32_exact',layer=33,tokenizer='original_ESM_2_alphabet',batch_size=4,sequence_order='first occurrence: '+','.join(self.feature_workloads),torch=self.env['torch'],cuda=self.env['cuda'],capability=self.env['capability'],runtime_boot_id=self.boot,matmul_tf32=False,cudnn_tf32=True)
  if self.protein_identity is not None:identity.update(backend='catpred_accel_features',protein_backend_identity=self.protein_identity)
  return identity
 def save_features(self,values):
  entries=[];base=self.out/'feature_tensors';base.mkdir(parents=True,exist_ok=True)
  for seq,t in values.items():
   assert t.device.type=='cpu' and t.dtype==self.torch.float32 and self.torch.isfinite(t).all().item()
   raw=t.numpy().tobytes(order='C');sha=hashlib.sha256(raw).hexdigest();path=base/(sha+'.f32')
   if not path.exists():
    tmp=path.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(path)
   assert path.stat().st_size==len(raw)
   entries.append(dict(sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),shape=list(t.shape),stride=list(t.stride()),dtype='<f4',bytes=len(raw),sha256=sha,path=relative(self.root,path)))
  return dict(identity=self.feature_identity(),sequence_count=len(entries),features=entries)
 def generate(self,names):
  seqs=list(dict.fromkeys(s for n in names for s in self.frames[n].sequence));before=len(self.esm_batches);start=time.perf_counter()
  values=self.esm.get_many_esm_reprs(seqs,device='cpu',batch_size=4);self.torch.cuda.synchronize()
  assert list(values)==seqs
  return values,dict(seconds=time.perf_counter()-start,sequence_count=len(seqs),actual_forward_batches=self.esm_batches[before:])
 def drop_esm(self):
  self.esm.GLOBAL_VARIABLES['model']=None;self.consumed={};gc.collect();self.torch.cuda.empty_cache()
 def preload(self,name):
  req=self.request(name);paths=self.service.prepare_prediction_inputs('kcat',req.input_file,str(self.root/'CatPred'))
  self.service._write_protein_records(paths.input_csv,paths.records_file)
  self.service._load_model_objects_for_prediction(self.service._build_predict_args(req,paths,self.root/'CatPred'));self.torch.cuda.synchronize()
 def close_runtime(self):
  if self.s1_manager is not None:
   manager=self.s1_manager;self.s1_manager=None;manager.__exit__(None,None,None)
   assert self.s1_runtime.manifest().get('restored') is True
 def new_accelerated_runtime(self):
  from catpred_accel import Runtime as AcceleratedRuntime,RuntimeConfig
  return AcceleratedRuntime(RuntimeConfig(numeric='exact',memory='stream',backend='auto',input_budget_bytes=2<<30,fallback='raise',allow_unvalidated=True),components=dict(packing=self.packing,reuse=self.reuse,probes=self.probes,streaming=self.streaming))
 def configure(self,arm):
  self.close_runtime()
  self.streaming.configure(enabled=False,budget_bytes=2<<30)
  # Every S1 activation starts from original A/off; the library restores this state.
  baseline_arm='A' if arm in ('A','S1') else 'C'
  self.packing.install_arm(baseline_arm,instrument=True)
  self.reuse.configure(r1=arm=='S0',r2=arm=='S0');self.probes.configure(r3=arm=='S0',t1=False,t2=False)
  self.streaming.configure(enabled=arm=='S0',budget_bytes=2<<30)
  if arm=='S1':
   self.s1_runtime=self.new_accelerated_runtime();manager=self.s1_runtime.activate(self.models,self.bundle[3]);manager.__enter__();self.s1_manager=manager
   selected=self.s1_runtime.manifest();assert selected['engaged'] and selected['selected_backend']=='accepted_stream',selected
 def run(self,name,arm,dest,phase,rep,diagnostic=False,cold=False,profile=False):
  dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);torch=self.torch
  if not cold:self.configure(arm)
  self.packing.reset_counts();self.forward.clear();self.consumed={};self.feature_preflight_checks=0;before=self.signature();esm_before=len(self.esm_batches)
  assert not self.service._PREDICTION_CACHE
  rss=[self.process.memory_info().rss];done=threading.Event()
  def sample():
   while not done.wait(.1):rss.append(self.process.memory_info().rss)
  monitor=threading.Thread(target=sample,daemon=True)
  prof=None;transfers={};graphs={};reuse_state=None;s1_state=None;generated=None;generation=None;load_seconds=None
  feature_stack=ExitStack();protein=None;self.protein_identity=None
  if cold and arm=='S1' and (self.esm_mode!='off' or self.esm_loader_mode!='off'):
   from catpred_accel.protein import features_context
   sequences=list(dict.fromkeys(self.frames[name].sequence))
   protein=feature_stack.enter_context(features_context(self.esm,sequences=sequences,mode=self.esm_mode,loader_mode=self.esm_loader_mode,allow_unvalidated=True,cache_root=str(self.cache),batch_observer=self.esm_batches.append))
   self.protein_identity=protein.identity()
  monitor.start()
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();epoch=time.time()
  try:
   if profile:
    prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False);prof.__enter__()
   if cold:
    assert not before,'Cold application cache must start empty'
    generated,generation=self.generate((name,))
    if protein is not None:protein.release_model()
    self.drop_esm();load_start=time.perf_counter();self.preload(name);load_seconds=time.perf_counter()-load_start;self.configure(arm)
   with ExitStack() as stack:
    if arm=='S1':
     s1_state=stack.enter_context(self.s1_runtime.request(diagnostic=diagnostic));stream=s1_state.stream;probe=s1_state.probes
    if diagnostic:
     from diagnostics import transfer_and_module_diagnostics,original_graph_checks
     transfers=stack.enter_context(transfer_and_module_diagnostics(self.models));graphs=stack.enter_context(original_graph_checks())
    if arm!='S1':
     stream=stack.enter_context(self.streaming.request_context(diagnostic=diagnostic))
     if arm=='A':reuse_state=stack.enter_context(self.reuse.request_context(diagnostic=diagnostic))
     probe=stack.enter_context(self.probes.request_context(diagnostic=diagnostic))
    cap=stack.enter_context(self.capture.request_context(self.frames[name].row_id.tolist()))
    output=self.pipeline(self.request(name),results_dir=str(dest))
   feature_stack.close()
   torch.cuda.synchronize();end=time.perf_counter()
  finally:
   feature_stack.close()
   done.set();monitor.join()
   if prof is not None:prof.__exit__(None,None,None)
  row=dict(arm=arm,workload=name,phase=phase,repetition=rep,rows=len(self.frames[name]),seconds=end-start,epoch_start=epoch,profiled=profile,diagnostic=diagnostic,environment=self.env,active_sources=self.sources,checkpoints=self.checkpoints,input_sha256=digest(self.root/'inputs'/(name+'.csv')),input_row_ids=self.frames[name].row_id.tolist(),batch_size=50,batch_sizes=[len(self.frames[name].iloc[i:i+50]) for i in range(0,len(self.frames[name]),50)],num_workers=self.current['num_workers'],model_forward_counts=[self.forward[i] for i in range(10)],constructor_counts=self.packing.counts(),stream=stream.summary(),reuse=reuse_state.summary() if reuse_state else None,probes=probe.summary(),inspection=self.inspection,transfer_diagnostics=dict(transfers),original_graph_checks=graphs,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),cuda_allocated_after_bytes=torch.cuda.memory_allocated(),cpu_peak_rss_bytes=max(rss),request_cache_initial_empty=True,final_prediction_cache_disabled=True,feature_cache_start_empty=not before,feature_preflight_checks=self.feature_preflight_checks,feature_cache_unchanged=before==self.signature(),actual_ESM_forward_batches=self.esm_batches[esm_before:],cold_generation=generation,model_load_seconds=load_seconds,boundary='Synchronized complete request including final CSV and request-cache release; extra evidence serialization excluded',completed_utc=utc())
  row['protein_backend']=dict(requested_mode='off',used_mode='off',loader_mode='off',engaged=False,actual_backend='original_fair_esm_2',generation_in_this_request=cold,consumed_source='this_request_original_generation' if cold else 'this_run_original_shared_features')
  if protein is not None:
   row['protein_backend']=protein.summary();row['protein_backend'].update(actual_backend='catpred_accel_features',generation_in_this_request=True,batch_telemetry_source='candidate_successful_batch_observer')
   assert row['protein_backend']['closed'] and row['protein_backend']['model_released'] and row['protein_backend']['engaged']
  if s1_state is not None:
   row['runtime_request']=s1_state.summary();row['runtime_manifest']=self.s1_runtime.manifest()
   assert row['runtime_request']['closed'] and row['runtime_request']['request_cache_empty_on_exit'] and row['runtime_request']['failure'] is None
  row.update(cap.save(dest));raw=self.root/'inputs'/(name+'_input_output.csv');shutil.copyfile(raw,dest/'raw_predictions.csv')
  row['prediction_path']=str(output);row['prediction_sha256']=digest(output);row['raw_prediction_path']=str(dest/'raw_predictions.csv');row['raw_prediction_sha256']=digest(dest/'raw_predictions.csv')
  features=self.save_features(self.consumed);atomic(dest/'consumed_features.json',features)
  row['consumed_features_path']=relative(self.root,dest/'consumed_features.json');row['consumed_features_sha256']=digest(dest/'consumed_features.json')
  assert {e['sequence_sha256'] for e in features['features']}=={hashlib.sha256(s.encode()).hexdigest() for s in self.frames[name].sequence}
  if generated is not None:atomic(dest/'generated_features.json',self.save_features(generated));row['generated_features_path']=relative(self.root,dest/'generated_features.json');row['generated_features_sha256']=digest(dest/'generated_features.json')
  if prof is not None:
   from diagnostics import save_torch_profile
   row['profile']=save_torch_profile(prof,dest,arm+'_'+phase)
  for key in ('precision_path','precision_manifest_path','prediction_path','raw_prediction_path'):row[key]=relative(self.root,row[key])
  assert row['model_forward_counts']==[math.ceil(len(self.frames[name])/50)]*10
  assert row['stream']['closed'] and row['stream']['retained_after_request_bytes']==0
  if not cold:assert row['feature_cache_unchanged'] and not row['actual_ESM_forward_batches']
  atomic(dest/'receipt.json',row);return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--task',required=True);p.add_argument('--attempt',required=True);p.add_argument('--arm',choices=('A','S0','S1'),default='A');p.add_argument('--workload',default='mixed_valid_2047');p.add_argument('--phase',choices=('prepare','baseline-gate','candidate-gate','warm','cold','profile-warm','profile-cold','lifecycle'),required=True);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--start-repetition',type=int,default=1);p.add_argument('--esm-mode',choices=('off','head_only','weights_only','representations'),default='off');p.add_argument('--esm-loader-mode',choices=('off','meta'),default='off');a=p.parse_args()
 root=Path(a.root).resolve();attempt=Path(a.attempt).resolve();attempt.mkdir(parents=True,exist_ok=True)
 cold=a.phase in ('cold','profile-cold');cache=attempt/'empty_feature_cache' if cold else root/'results/experiments/shared_feature_cache'
 cache.mkdir(parents=True,exist_ok=True);rt=Runtime(root,cache);rt.esm_mode=a.esm_mode;rt.esm_loader_mode=a.esm_loader_mode;atomic(attempt/'environment.json',rt.env)
 if a.esm_mode!='off' or a.esm_loader_mode!='off':assert cold and a.arm=='S1','ESM candidates apply only to explicitly selected S1 cold requests'
 if cold:rt.feature_workloads=(a.workload,)
 if a.phase=='prepare':
  owner=root/'results/experiments/shared_features.json'
  if owner.exists():
   saved=read(owner);assert saved['manifest']['identity']==rt.feature_identity()
   for entry in saved['cache_files']:assert digest(root/entry['path'])==entry['sha256']
  else:
   pending=owner.with_name('shared_features_progress.json')
   seqs=list(dict.fromkeys(s for n in WORKLOADS for s in rt.frames[n].sequence))
   state=read(pending) if pending.exists() else dict(identity=rt.feature_identity(),next_sequence=0,features=[],cache_files=[],actual_forward_batches=[],seconds=0.)
   assert state['identity']==rt.feature_identity()
   known={e['path']:e for e in state['cache_files']}
   for name,e in known.items():assert digest(root/name)==e['sha256']
   # Interrupted partial ESM groups are quarantined; completed groups stay intact.
   # Recompute only the incomplete original group of four, preserving its padding.
   for q in sorted(cache.rglob('*.pt')):
    if relative(root,q) not in known:
     target=attempt/'interrupted_cache'/q.relative_to(cache);target.parent.mkdir(parents=True,exist_ok=True);q.replace(target)
   atomic(pending,state)
   for i in range(state['next_sequence'],len(seqs),4):
    count_before=len(rt.esm_batches);start=time.perf_counter();values=rt.esm.get_many_esm_reprs(seqs[i:i+4],device='cpu',batch_size=4);rt.torch.cuda.synchronize()
    state['seconds']+=time.perf_counter()-start;state['features'].extend(rt.save_features(values)['features']);state['actual_forward_batches'].extend(rt.esm_batches[count_before:]);state['next_sequence']=min(i+4,len(seqs))
    state['cache_files']=[dict(path=relative(root,q),sha256=known[relative(root,q)]['sha256'] if relative(root,q) in known else digest(q)) for q in sorted(cache.rglob('*.pt'))]
    known={e['path']:e for e in state['cache_files']};atomic(pending,state)
   manifest=dict(identity=state['identity'],sequence_count=len(state['features']),features=state['features'])
   atomic(owner,dict(manifest=manifest,generation=dict(seconds=state['seconds'],sequence_count=len(seqs),actual_forward_batches=state['actual_forward_batches']),environment=rt.env,cache_files=state['cache_files']))
  atomic(attempt/'COMPLETE.json',dict(phase='prepare',owner=relative(root,owner),sha256=digest(owner),completed_utc=utc()));return
 if not cold:
  owner=read(root/'results/experiments/shared_features.json');assert owner['manifest']['identity']==rt.feature_identity()
  for e in owner['cache_files']:assert digest(root/e['path'])==e['sha256']
  def missing(*args,**kwargs):raise RuntimeError('Warm ESM cache miss')
  rt.esm._run_esm_batch=missing;rt.esm.get_single_esm_repr=missing
  if a.phase in ('baseline-gate','candidate-gate','lifecycle'):rt.expected_features={e['sequence_sha256']:e for e in owner['manifest']['features']}
  rt.preload(a.workload)
  if a.phase in ('warm','profile-warm'):rt.run(a.workload,a.arm,attempt/'warmup','warmup',0)
 if a.phase=='lifecycle':
  from s1_lifecycle import run_lifecycle
  rt.configure('A');result=run_lifecycle(rt,a.workload,attempt);atomic(attempt/'COMPLETE.json',dict(task=a.task,status='COMPLETE',lifecycle=result,completed_utc=utc()));return
 rows=[]
 try:
  for rep in range(a.start_repetition,a.start_repetition+a.repetitions):
   row=rt.run(a.workload,a.arm,attempt/('rep_'+str(rep)),a.phase,rep,diagnostic=a.phase in ('baseline-gate','candidate-gate') and rep==1,cold=cold,profile=a.phase.startswith('profile-'))
   rows.append(relative(root,attempt/('rep_'+str(rep))/'receipt.json'))
   atomic(attempt/'progress.json',dict(task=a.task,completed_receipts=rows,status='RUNNING'))
   print('REQUEST_COMPLETE',json.dumps(dict(task=a.task,rep=rep,seconds=row['seconds'])),flush=True)
 finally:rt.close_runtime()
 if a.arm=='S1':atomic(attempt/'activation_restored.json',rt.s1_runtime.manifest())
 atomic(attempt/'COMPLETE.json',dict(task=a.task,completed_receipts=rows,status='COMPLETE',completed_utc=utc()))

if __name__=='__main__':main()
