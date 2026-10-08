"""One real kcat request through the pinned service and private accelerated runtime.

No benchmark modules, runtime allocation, authentication, or weight downloads.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib,importlib.metadata,json,os,shutil,sys,time
from dataclasses import replace
from pathlib import Path

ESM_NAMES=('esm2_t33_650M_UR50D.pt','esm2_t33_650M_UR50D-contact-regression.pt')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 return h.hexdigest()
def save(path,value):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def protein_context(esm_utils,options,sequences,selection,cache):
 from catpred_accel.protein import features_context
 return features_context(esm_utils,sequences=sequences,mode=selection['mode'],loader_mode=selection['loader_mode'],allow_unvalidated=options.experimental_unvalidated,cache_root=str(cache))

def protein_record(handle,selection):
 return dict(selection=selection,execution=handle.summary() if handle is not None else None)

def main():
 parser=argparse.ArgumentParser(description='Run one kcat request with exact FP32 and a bounded 2 GiB input cache.')
 parser.add_argument('--repo-root',type=Path,required=True,help='Supplied pinned compatible CatPred checkout')
 parser.add_argument('--checkpoint-dir',type=Path,required=True,help='Production kcat directory containing all ten model checkpoints')
 parser.add_argument('--torch-home',type=Path,required=True,help='Local torch cache containing the two ESM2 checkpoint files')
 parser.add_argument('--input',type=Path,required=True,help='Input CSV with SMILES and sequence columns')
 parser.add_argument('--output-dir',type=Path,required=True,help='A new directory for this request and its backend manifest')
 parser.add_argument('--feature-cache',type=Path,required=True,help='Persistent application cache root; exact identities create separate namespaces')
 parser.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
 parser.add_argument('--backend',choices=('auto','g4','h100','generic','cpu'),default='auto')
 parser.add_argument('--mode',choices=('off','exact'),default='exact')
 parser.add_argument('--fallback',choices=('stock','raise'),default='stock')
 parser.add_argument('--protein-backend',choices=('auto','stock'),default='auto',help='Auto selects only a certified beneficial ESM combination for this exact stack')
 parser.add_argument('--esm-mode',choices=('auto','off','head_only','weights_only','representations'),default='auto',help='Set off to disable E1 while retaining the selected loader')
 parser.add_argument('--esm-loader-mode',choices=('auto','off','meta'),default='auto',help='Set off to disable E2 while retaining the selected forward')
 parser.add_argument('--input-budget-gib',type=int,choices=(1,2,4),default=2)
 parser.add_argument('--threads',type=int,default=24)
 parser.add_argument('--experimental-unvalidated',action='store_true',help='Explicitly permit an uncertified compatible runtime; manifest records experimental status')
 args=parser.parse_args()
 if sys.platform!='linux':parser.error('Run inference on the Linux CPU/CUDA host; this example does not run models on the Mac.')
 if args.threads<1:parser.error('--threads must be positive')
 for field in ('repo_root','checkpoint_dir','torch_home','input','output_dir','feature_cache'):setattr(args,field,getattr(args,field).expanduser().resolve())
 if not (args.repo_root/'catpred/inference/service.py').is_file():parser.error('Missing the supplied compatible CatPred service')
 if not args.input.is_file():parser.error('Input CSV is missing')
 checkpoints=sorted(args.checkpoint_dir.rglob('*.pt'))
 if len(checkpoints)!=10:parser.error('The production kcat checkpoint directory must contain exactly ten .pt files')
 esm_weights={}
 for name in ESM_NAMES:
  checkpoint=args.torch_home/'hub'/'checkpoints'/name
  if not checkpoint.is_file():parser.error('Restore the local checkpoint at '+str(checkpoint)+' before running')
  esm_weights[name]=dict(path=str(checkpoint),sha256=sha(checkpoint))
 with args.input.open(newline='') as stream:
  reader=csv.DictReader(stream)
  if not {'SMILES','sequence'}.issubset(reader.fieldnames or []):parser.error('Input CSV requires SMILES and sequence columns')
  rows=list(reader)
 if not rows:parser.error('Input CSV is empty')
 sequences=list(dict.fromkeys(row['sequence'] for row in rows))
 if not all(sequences):parser.error('Input sequences must be nonempty')
 # Import only the framework and ESM metadata before initializing CatPred's cache.
 import torch
 torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 use_gpu=args.device!='cpu' and torch.cuda.is_available()
 if args.device=='cuda' and not use_gpu:parser.error('CUDA was requested but is unavailable')
 device='cuda' if use_gpu else 'cpu'
 esm_sources={}
 for name in ('esm.model.esm2','esm.modules','esm.multihead_attention','esm.rotary_embedding','esm.data','esm.pretrained'):
  path=Path(importlib.import_module(name).__file__);esm_sources[name]=sha(path)
 from catpred_accel.protein import recommended_modes
 from catpred_accel._identity import package_sources,software as runtime_software
 recommendation=recommended_modes(esm_weights[ESM_NAMES[0]]['path'],esm_weights[ESM_NAMES[1]]['path']) if use_gpu and args.mode=='exact' and args.protein_backend=='auto' else dict(mode='off',loader_mode='off',reason='Explicit stock/off mode or CPU')
 selection=dict(recommendation)
 if args.protein_backend=='auto' and args.mode=='exact' and use_gpu:
  if args.esm_mode!='auto':selection['mode']=args.esm_mode
  if args.esm_loader_mode!='auto':selection['loader_mode']=args.esm_loader_mode
 selection['explicit_forward_override']=args.esm_mode;selection['explicit_loader_override']=args.esm_loader_mode
 feature_identity=dict(schema=1,candidate_implementation_sources=package_sources(),checkpoint_sha256={name:entry['sha256'] for name,entry in esm_weights.items()},feature_code_sha256=sha(args.repo_root/'catpred/data/esm_utils.py'),fair_esm_sources=esm_sources,fair_esm_version=importlib.metadata.version('fair-esm'),backend=dict(feature_mode=selection['mode'],loader_mode=selection['loader_mode']),numeric='exact_fp32',layer=33,batch_size=4,input_sha256=sha(args.input),ordered_sequence_sha256=[hashlib.sha256(seq.encode()).hexdigest() for seq in sequences],software=runtime_software(torch),hardware=dict(device=device,name=torch.cuda.get_device_name() if use_gpu else 'CPU',capability=list(torch.cuda.get_device_capability()) if use_gpu else None),matmul_tf32=False,cudnn_tf32=True)
 namespace=args.feature_cache/digest(feature_identity);namespace.mkdir(parents=True,exist_ok=True)
 owner=namespace/'identity.json'
 if owner.exists():assert json.loads(owner.read_text())==feature_identity,'Feature-cache identity mismatch'
 else:save(owner,feature_identity)
 # Only a completed, content-verified original batch grouping may be reused.
 # Failed or interrupted populations stay on disk and never change a later group.
 latest=namespace/'LATEST_COMPLETE.json'
 if latest.exists():
  pointer=json.loads(latest.read_text());cache=(namespace/pointer['cache']).resolve();assert cache.is_relative_to(namespace)
  complete=cache/'COMPLETE.json';assert sha(complete)==pointer['manifest_sha256']
  cache_record=json.loads(complete.read_text());assert cache_record['identity_sha256']==digest(feature_identity)
  expected={e['path']:e['sha256'] for e in cache_record['feature_files']}
  actual={str(p.relative_to(cache)):sha(p) for p in sorted(cache.rglob('*.pt'))}
  assert actual==expected,'Saved ESM cache content changed'
 else:
  cache=namespace/'populations'/str(time.time_ns());cache.mkdir(parents=True)
 args.output_dir.mkdir(parents=True,exist_ok=False)
 staged_input=args.output_dir/args.input.name;shutil.copyfile(args.input,staged_input)
 roots=os.pathsep.join(str(p) for p in (args.repo_root,args.checkpoint_dir,args.torch_home,args.feature_cache,args.output_dir))
 os.environ.update(TORCH_HOME=str(args.torch_home),CATPRED_CACHE_PATH=str(cache),CATPRED_TRUSTED_DESERIALIZATION_ROOTS=roots,CATPRED_PREDICTION_CACHE_SIZE='0',CATPRED_MODEL_CACHE_SIZE='2',CATPRED_ESM_BATCH_SIZE='4',PROTEIN_EMBED_USE_CPU='0' if use_gpu else '1')
 sys.path.insert(0,str(args.repo_root))
 from catpred.inference import PredictionRequest,run_inprocess_prediction_pipeline,service
 from catpred.data import esm_utils
 from catpred_accel import Runtime,RuntimeConfig
 request=PredictionRequest(parameter='kcat',input_file=str(staged_input),checkpoint_dir=str(args.checkpoint_dir),use_gpu=use_gpu,repo_root=str(args.repo_root))
 old_build,old_load=service._build_predict_args,service._load_model_objects_for_prediction
 bundle=None;model_ids=None;counts=[0]*10;handles=[];runtime=None;receipt=None;protein=None
 manifest=dict(status='RUNNING',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),input_path=str(args.input),input_sha256=sha(args.input),rows=len(rows),device=device,batch_size=50,ensemble_size=10,parameter='kcat',precision='FP32',esm=dict(identity=feature_identity,cache_path=str(cache)),checkpoints=[dict(path=str(p),sha256=sha(p)) for p in checkpoints],input_budget_bytes=args.input_budget_gib<<30,input_budget_scope='Retained input tensors; not total process or GPU memory')
 save(args.output_dir/'runtime_manifest.json',manifest)
 def build(*a,**k):
  value=old_build(*a,**k);value.batch_size=50
  assert value.num_workers==0
  return value
 def load(*a,**k):
  nonlocal bundle,model_ids
  value=old_load(*a,**k);value[0].batch_size=50
  if bundle is None:
   bundle=value;model_ids=tuple(map(id,value[2]));assert len(value[2])==10
   assert list(value[5])==['log10kcat_max'],'Only production kcat targets are supported'
   assert {p.dtype for model in value[2] for p in model.parameters()}=={torch.float32}
   for index,model in enumerate(value[2]):
    def hook(module,arguments,member=index):counts[member]+=1
    handles.append(model.register_forward_pre_hook(hook))
  else:assert tuple(map(id,value[2]))==model_ids
  return value
 service._build_predict_args,service._load_model_objects_for_prediction=build,load
 try:
  prepared=service.prepare_prediction_inputs('kcat',request.input_file,str(args.repo_root));service._write_protein_records(prepared.input_csv,prepared.records_file)
  # Passing the original records prevents service deduplication from changing
  # row/batch composition; no learned predictions or ensemble members are skipped.
  request=replace(request,protein_records_file=str(Path(prepared.records_file).resolve()))
  service._load_model_objects_for_prediction(service._build_predict_args(request,prepared,args.repo_root))
  runtime=Runtime(RuntimeConfig(numeric=args.mode,memory='stream',backend=args.backend,input_budget_bytes=args.input_budget_gib<<30,fallback=args.fallback,allow_unvalidated=args.experimental_unvalidated))
  with protein_context(esm_utils,args,sequences,selection,cache) as protein:
   with runtime.activate(bundle[2],bundle[3]):
    with runtime.request() as receipt:
     output=run_inprocess_prediction_pipeline(request,results_dir=str(args.output_dir))
    if use_gpu:torch.cuda.synchronize()
  manifest.update(status='COMPLETE',output_path=str(output),output_sha256=sha(output),runtime=runtime.manifest(),request=receipt.summary(),protein_backend=protein_record(protein,selection),model_forward_counts=counts,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  assert len(set(counts))==1 and counts[0]>0
  save(cache/'COMPLETE.json',dict(identity_sha256=digest(feature_identity),feature_files=[dict(path=str(p.relative_to(cache)),sha256=sha(p)) for p in sorted(cache.rglob('*.pt'))]))
  save(namespace/'LATEST_COMPLETE.json',dict(cache=str(cache.relative_to(namespace)),manifest_sha256=sha(cache/'COMPLETE.json')))
  save(args.output_dir/'runtime_manifest.json',manifest)
  print(json.dumps(dict(output=str(output),manifest=str(args.output_dir/'runtime_manifest.json'),requested_mode=args.mode,actual_backend=runtime.manifest()['selected_backend'],validation_status=runtime.manifest()['validation_status']),ensure_ascii=False))
 except BaseException as error:
  manifest.update(status='FAILED',error=type(error).__name__+': '+str(error),runtime=runtime.manifest() if runtime is not None else None,request=receipt.summary() if receipt is not None and receipt.closed else None,protein_backend=protein_record(protein,selection),model_forward_counts=counts)
  save(args.output_dir/'runtime_manifest.json',manifest);raise
 finally:
  service._build_predict_args,service._load_model_objects_for_prediction=old_build,old_load
  for handle in handles:handle.remove()

if __name__=='__main__':main()
