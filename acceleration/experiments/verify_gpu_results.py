"""Independent saved-artifact verification: Python standard library, no torch/NumPy."""
from __future__ import annotations
import argparse,array,ast,csv,hashlib,inspect,io,json,math,statistics,struct,sys,time,zipfile
from collections import defaultdict
from pathlib import Path
from evidence import *
ATOL,RTOL=1e-6,1e-5

# Accepted production members: fixed independently of editable run manifests.
PREDICTOR_CHECKPOINTS = [{'path': 'data/pretrained/production/kcat/fold_0/model_0/model.pt', 'bytes': 9975234, 'sha256': 'feec68cf1884bb14fc80abf8477aa3fc10f76781984ab2849e17e90bbfe8da90'}, {'path': 'data/pretrained/production/kcat/fold_0/model_1/model.pt', 'bytes': 9975234, 'sha256': '7a64419ddae49bf4db1b0c45ea0eb08321bab6eaf28a882e3d588bfe844bc382'}, {'path': 'data/pretrained/production/kcat/fold_0/model_2/model.pt', 'bytes': 9975234, 'sha256': 'd1ef752bfee6a51b8967fcf9fb65e45f9b976eff84e1f80f3995342ea0b15253'}, {'path': 'data/pretrained/production/kcat/fold_0/model_3/model.pt', 'bytes': 9975234, 'sha256': '3cfd167fd099612e62392ef66cf5cf06032594eac9c126f98d3fc9c30009f173'}, {'path': 'data/pretrained/production/kcat/fold_0/model_4/model.pt', 'bytes': 9975234, 'sha256': '016500999499369fc9519bc005f080eea19c80bcdc55323855c4a5ab3eba6320'}, {'path': 'data/pretrained/production/kcat/fold_0/model_5/model.pt', 'bytes': 9975234, 'sha256': '2574f3b25dbd333ab956ebe8b6bc1c8eed600a36a45d33bc914be7344f19eae0'}, {'path': 'data/pretrained/production/kcat/fold_0/model_6/model.pt', 'bytes': 9975234, 'sha256': 'ea23beb23fa40ba83666bd56aa48f0812bb9cc1ee31c3b5a927a3d0b7bab9690'}, {'path': 'data/pretrained/production/kcat/fold_0/model_7/model.pt', 'bytes': 9975234, 'sha256': 'af3f35fa25a18e9e409f8c3b9ee85341ce690fba985e277e956f421d487358c3'}, {'path': 'data/pretrained/production/kcat/fold_0/model_8/model.pt', 'bytes': 9975234, 'sha256': 'cdc03244885e6f094bbbaefca260f23447cf1667e0f8df269405ac12518821f7'}, {'path': 'data/pretrained/production/kcat/fold_0/model_9/model.pt', 'bytes': 9975234, 'sha256': 'e1639c49680a472a5655f34fb483ea4cb14da90b0e484d015e0de5874d30ea35'}]

def decode_npy(data):
 f=io.BytesIO(data);assert f.read(6)==b'\x93NUMPY';version=tuple(f.read(2));assert version in ((1,0),(2,0),(3,0))
 size=struct.unpack('<H' if version==(1,0) else '<I',f.read(2 if version==(1,0) else 4))[0];assert size<1<<20
 h=ast.literal_eval(f.read(size).decode('utf8' if version==(3,0) else 'latin1'))
 shape=tuple(h['shape']);dtype=h['descr'];assert all(type(n)==int and n>=0 for n in shape)
 assert dtype in ('<f4','<f8','>f4','>f8','=f4','=f8');width=int(dtype[-1]);payload=f.read();assert len(payload)==math.prod(shape)*width
 # Normalize *bytes* to logical C order, preserving signed zero and every mantissa bit.
 if h['fortran_order'] and len(shape)>1:
  strides=[math.prod(shape[:i]) for i in range(len(shape))];blocks=[]
  for flat in range(math.prod(shape)):
   rem=flat;offset=0
   for axis in range(len(shape)-1,-1,-1):offset+=(rem%shape[axis])*strides[axis];rem//=shape[axis]
   blocks.append(payload[offset*width:(offset+1)*width])
  payload=b''.join(blocks)
 values=array.array('f' if width==4 else 'd');values.frombytes(payload)
 if dtype[0] in '<>' and ((dtype[0]=='<')!=(sys.byteorder=='little')):values.byteswap()
 return dict(shape=shape,dtype=dtype,payload=payload,values=values)

def load_npz(path):
 with zipfile.ZipFile(path) as z:
  names=z.namelist();assert len(names)==len(set(names));assert set(names)=={'member_raw.npy','numeric_raw.npy','numeric_processed.npy'}
  return {n[:-4]:decode_npy(z.read(n)) for n in names}

def compare(a,b):
 compatible=a['shape']==b['shape'] and a['dtype']==b['dtype'];out=dict(shape_dtype_equal=compatible,exact=compatible,allclose=compatible,finite=True,mismatching_bits=0,mismatching_values=0,tolerance_failures=0,max_absolute_error=0.,max_relative_error_nonzero_reference=0.,first_mismatches=[],per_member={})
 if not compatible:out.update(exact=False,allclose=False);return out
 width=int(a['dtype'][-1]);memberwidth=math.prod(a['shape'][1:]) if len(a['shape'])==3 and a['shape'][0]==10 else 0
 for i,(x,y) in enumerate(zip(a['values'],b['values'])):
  finite=math.isfinite(x) and math.isfinite(y);bits=a['payload'][i*width:(i+1)*width]==b['payload'][i*width:(i+1)*width];delta=abs(x-y) if finite else None;close=finite and delta<=ATOL+RTOL*abs(x)
  out['finite'] &= finite;out['exact'] &= finite and bits;out['allclose'] &= close;out['mismatching_bits']+=int(not bits);out['mismatching_values']+=int(not finite or x!=y);out['tolerance_failures']+=int(not close)
  if not bits and len(out['first_mismatches'])<12:out['first_mismatches'].append(dict(flat_index=i,reference=repr(x),candidate=repr(y)))
  if finite:
   out['max_absolute_error']=max(out['max_absolute_error'],delta)
   if x:out['max_relative_error_nonzero_reference']=max(out['max_relative_error_nonzero_reference'],delta/abs(x))
  if memberwidth:
   m=out['per_member'].setdefault(str(i//memberwidth),dict(exact=True,allclose=True,mismatching_bits=0));m['exact'] &= finite and bits;m['allclose'] &= close;m['mismatching_bits']+=int(not bits)
 return out

def assert_finite_f32(path):
 """Complete bounded-memory IEEE754 scan; receipts cache only successful scans."""
 words_checked=0
 with Path(path).open('rb') as f:
  while data:=f.read(4<<20):
   assert len(data)%4==0,'Truncated float32 payload'
   words=array.array('I');words.frombytes(data)
   if sys.byteorder!='little':words.byteswap()
   assert all((w&0x7f800000)!=0x7f800000 for w in words),('Nonfinite ESM tensor',str(path))
   words_checked+=len(words)
 return words_checked

FINITE_ALGORITHM_SHA256=hashlib.sha256(inspect.getsource(assert_finite_f32).encode()).hexdigest()

def ledger_payload_sha(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class Verifier:
 def __init__(self,root,finite_ledger_root=None):self.root=Path(root).resolve();self.finite_ledger_root=Path(finite_ledger_root).resolve() if finite_ledger_root is not None else self.root/'results/experiments/finite_checks';self.errors=[];self.hashes={};self.arrays={};self.feature_files={};self.rows=[];self.comparisons=[];self.finite_ledger_hits=0;self.finite_fresh_scans=0;self.invalid_finite_receipts=0
 def path(self,name):
  p=(self.root/name).resolve();assert p.is_relative_to(self.root),'Escaped evidence root';return p
 def checked(self,name,sha):
  p=self.path(name);key=(str(p),p.stat().st_size,p.stat().st_mtime_ns)
  if key not in self.hashes:self.hashes[key]=digest(p)
  assert self.hashes[key]==sha,('SHA256',name);return p
 def features(self,row,kind='consumed'):
  return self.feature_object(read(self.checked(row[kind+'_features_path'],row[kind+'_features_sha256'])))
 def feature_object(self,obj):
  entries={e['sequence_sha256']:e for e in obj['features']};assert len(entries)==obj['sequence_count']
  for e in entries.values():
   p=self.checked(e['path'],e['sha256']);assert e['dtype']=='<f4' and p.stat().st_size==e['bytes']==math.prod(e['shape'])*4
   if e['sha256'] not in self.feature_files:
    # checked() above always recomputes the actual file SHA in this invocation.
    # A durable receipt avoids only a previously completed Python finiteness scan.
    self.finite_check(p,e['sha256'],e['bytes'])
    self.feature_files[e['sha256']]=True
  return obj,entries
 def finite_check(self,path,tensor_sha256,byte_count):
  receipt=self.finite_ledger_root/FINITE_ALGORITHM_SHA256/(tensor_sha256+'.json')
  required=dict(schema=1,algorithm_sha256=FINITE_ALGORITHM_SHA256,tensor_sha256=tensor_sha256,dtype='<f4',bytes=byte_count,finite=True,words_checked=byte_count//4)
  if receipt.exists():
   try:
    stored=read(receipt);payload=stored['payload']
    assert stored['receipt_sha256']==ledger_payload_sha(payload)
    assert all(payload.get(k)==v for k,v in required.items())
    self.finite_ledger_hits+=1;return
   except (OSError,ValueError,KeyError,TypeError,AssertionError):
    # Preserve malformed evidence while allowing a complete mechanical recheck.
    corrupt=receipt.with_name(receipt.stem+'.invalid.'+str(time.time_ns())+'.json')
    receipt.replace(corrupt);self.invalid_finite_receipts+=1
  words=assert_finite_f32(path);assert words==byte_count//4
  payload=dict(required,checked_utc=utc(),verifier_source_sha256=digest(Path(__file__)))
  atomic(receipt,dict(payload=payload,receipt_sha256=ledger_payload_sha(payload)))
  self.finite_fresh_scans+=1
 def prepare_owner(self):
  owner=read(self.root/'results/experiments/shared_features.json');identity=owner['manifest']['identity'];env=owner['environment']
  assert identity['checkpoints']==ESM_WEIGHTS and identity['esm_code']==SCIENCE_HASHES['catpred/data/esm_utils.py']
  assert identity['layer']==33 and identity['batch_size']==4 and identity['numeric_contract']=='FP32_exact'
  assert identity['matmul_tf32'] is False and identity['cudnn_tf32'] is True and identity['runtime_boot_id']==env['runtime_boot_id']
  assert identity['fair_esm_sources']==env['fair_esm_sources'] and identity['fair_esm_version']==env['fair_esm_version']
  for e in identity['fair_esm_sources'].values():self.checked(e['path'],e['sha256'])
  sequences=[];seen=set()
  for n,h in INPUT_HASHES.items():
   p=self.checked('inputs/'+n+'.csv',h)
   with p.open(newline='') as f:
    for row in csv.DictReader(f):
     seq=row['sequence']
     if seq not in seen:seen.add(seq);sequences.append(hashlib.sha256(seq.encode()).hexdigest())
  batches=owner['generation']['actual_forward_batches'];assert len(batches)==math.ceil(len(sequences)/4)
  for i,b in enumerate(batches):
   assert b['sequences']==sequences[4*i:4*i+4] and b['device']=='cuda' and b['dtype']=='torch.float32' and b['eval']
   assert b['all_floating_parameters_buffers_cuda_fp32'] and b['floating_parameter_buffer_count']>0
  assert [e['sequence_sha256'] for e in owner['manifest']['features']]==sequences
  for e in owner['cache_files']:self.checked(e['path'],e['sha256'])
  self.owner=owner;self.owner_features=self.feature_object(owner['manifest'])
 def receipt(self,name,sha):
  row=read(self.checked(name,sha));row['_receipt']=name
  assert row['checkpoints']==PREDICTOR_CHECKPOINTS,'Production checkpoint identity changed'
  env=row['environment'];assert env['torch_threads']==24 and env['interop_threads']==1 and env['matmul_tf32'] is False and env['cudnn_tf32'] is True
  assert row['batch_size']==50 and row['num_workers']==0 and row['model_forward_counts']==[math.ceil(row['rows']/50)]*10
  assert row['request_cache_initial_empty'] and row['final_prediction_cache_disabled'];assert row['seconds']>0
  assert row['batch_sizes']==[min(50,row['rows']-i) for i in range(0,row['rows'],50)]
  if row['workload'] in WORKLOADS:assert row['rows']==WORKLOADS[row['workload']] and row['input_sha256']==INPUT_HASHES[row['workload']]
  assert len(row['input_row_ids'])==row['rows'] and len(set(row['input_row_ids']))==row['rows']
  src=row['active_sources'];manifest=read(self.path(src['manifest']));assert manifest==src['files']
  snapshot=self.path(src['manifest']).parent
  for n,h in manifest.items():self.checked(relative(self.root,snapshot/n),h)
  for n,h in BASELINE_HASHES.items():assert manifest['baseline/'+n]==h
  for n,h in SCIENCE_HASHES.items():assert manifest['CatPred/'+n]==h
  for key in ('precision','precision_manifest','prediction','raw_prediction'):self.checked(row[key+'_path'],row[key+'_sha256'])
  arrays=load_npz(self.path(row['precision_path']));assert arrays['member_raw']['shape']==(10,row['rows'],2) and arrays['member_raw']['dtype']=='<f4'
  for values in arrays.values():assert all(map(math.isfinite,values['values']))
  meta=read(self.path(row['precision_manifest_path']));assert meta['row_ids']==row['input_row_ids'];assert arrays['numeric_raw']['shape']==(row['rows'],4)
  assert meta['retained_raw_member_bytes']==10*row['rows']*2*4 and meta['extra_cuda_transfers']==0
  self.arrays[name]=arrays;row['_meta']=meta;row['_features']=self.features(row)
  assert row['_features'][0]['identity']['checkpoints']==ESM_WEIGHTS,'ESM checkpoint identity changed'
  if row['phase'] in ('baseline-gate','candidate-gate'):assert row['feature_preflight_checks']==row['_features'][0]['sequence_count']
  if row['phase'] not in ('cold','profile-cold'):
   assert row['_features'][0]['identity']==self.owner['manifest']['identity']
   expected={key:self.owner_features[1][key] for key in row['_features'][1]}
   assert self.feature_comparison((None,expected),row['_features'])['exact'],'Consumed ESM differs from this-run original owner'
   assert env['runtime_boot_id']==self.owner['environment']['runtime_boot_id']
  if row['phase'] in ('cold','profile-cold'):
   assert row['feature_cache_start_empty'] and row['actual_ESM_forward_batches'];generated=self.features(row,'generated')
   assert self.feature_comparison(generated,row['_features'])['exact'],'Generated versus consumed ESM drift'
   self.cold_features(row)
  else:assert row['feature_cache_unchanged'] and not row['actual_ESM_forward_batches']
  if row['arm']=='S1':
   runtime=row['runtime_request'];manifest=row['runtime_manifest']
   assert runtime['closed'] and runtime['request_cache_initial_empty'] and runtime['request_cache_empty_on_exit'] and runtime['failure'] is None
   assert manifest['engaged'] and manifest['selected_backend']=='accepted_stream'
   assert runtime['stream']==row['stream'] and runtime['probes']==row['probes'] and runtime['reuse'] is None
   assert manifest['requested']['numeric']=='exact' and manifest['requested']['input_budget_bytes']==2<<30 and manifest['requested']['fallback']=='raise'
   assert manifest['t1'] is False and manifest['t2'] is False and manifest['feature_backend']=='external_consumed_features' and manifest['predictor_input_contract']=='original_esm2_layer33_fp32'
   for name,value in manifest['source_hashes'].items():assert src['files']['catpred_accel/'+name]==value
  stream=row['stream'];assert stream['closed'] and stream['failure'] is None and stream['retained_after_request_bytes']==0 and stream['fallback_count']==0
  if row['arm'] in ('S0','S1'):
   assert stream['enabled'] and stream['budget_bytes']==2<<30 and stream['device_cache_peak_bytes']<=stream['budget_bytes']
   assert stream['forward_counts']==row['model_forward_counts'];offset=0
   for chunk in stream['chunks']:
    assert chunk['start_row']==offset and chunk['stop_row']>offset
    assert chunk['actual_unique_device_storage_bytes']==chunk['planned_device_storage_bytes']<=stream['budget_bytes']
    assert chunk['released'] and chunk['actual_device_cache_bytes_after_clear']==0
    if row['diagnostic']:assert chunk['device_tensor_references_alive_after_clear']==0
    offset=chunk['stop_row']
   assert offset==row['rows'] and stream['chunk_count']==len(stream['chunks'])
  else:
   reuse=row['reuse'];assert reuse['closed'] and reuse['cpu_cache_retained_bytes']==reuse['device_cache_retained_bytes']==0
  self.rows.append(row);return row
 def feature_comparison(self,a,b):
  _,aa=a;_,bb=b;exact=set(aa)==set(bb);details=[];close=exact
  for seq in sorted(set(aa)&set(bb)):
   x,y=aa[seq],bb[seq];same=x['shape']==y['shape'] and x['dtype']==y['dtype'] and x['sha256']==y['sha256'];exact &= same
   if not same:
    def raw(e):
     payload=self.path(e['path']).read_bytes();v=array.array('f');v.frombytes(payload)
     if sys.byteorder!='little':v.byteswap()
     return dict(shape=tuple(e['shape']),dtype=e['dtype'],payload=payload,values=v)
    diagnostic=compare(raw(x),raw(y));close &= diagnostic['allclose'];details.append(dict(sequence_sha256=seq,diagnostic=diagnostic))
  return dict(exact=exact,allclose=close,sequence_count=len(bb),mismatches=details)
 def pair(self,a,b,label):
  assert a['environment']==dict(b['environment'],process_id=a['environment']['process_id']),'Same-runtime environment differs'
  features=self.feature_comparison(a['_features'],b['_features'])
  arrays={k:compare(self.arrays[a['_receipt']][k],self.arrays[b['_receipt']][k]) for k in self.arrays[a['_receipt']]}
  metadata=a['_meta']==b['_meta'];csv_equal={}
  for kind in ('prediction','raw_prediction'):
   with self.path(a[kind+'_path']).open(newline='') as x,self.path(b[kind+'_path']).open(newline='') as y:csv_equal[kind]=list(csv.reader(x))==list(csv.reader(y))
  result=dict(label=label,reference=a['_receipt'],candidate=b['_receipt'],features=features,arrays=arrays,metadata_equal=metadata,csv_equal=csv_equal,exact=features['exact'] and metadata and all(csv_equal.values()) and all(v['exact'] for v in arrays.values()),allclose=features['allclose'] and all(v['allclose'] for v in arrays.values()))
  self.comparisons.append(result);assert result['exact'],('Strict bitwise comparison failed',label);return result
 def cold_features(self,row):
  if row['workload']=='mixed_prefix_100':
   with self.checked('inputs/mixed_valid_2047.csv',INPUT_HASHES['mixed_valid_2047']).open(newline='') as source:original=list(csv.reader(source))[:101]
   stream=io.StringIO(newline='');csv.writer(stream,lineterminator='\n').writerows(original)
   assert hashlib.sha256(stream.getvalue().encode()).hexdigest()==row['input_sha256'],'Cold panel differs from first two original batches'
   inputs=list(csv.DictReader(io.StringIO(stream.getvalue())))
  else:
   path=self.checked('inputs/'+row['workload']+'.csv',row['input_sha256'])
   with path.open(newline='') as source:inputs=list(csv.DictReader(source))
  assert len(inputs)==row['rows'] and [r['row_id'] for r in inputs]==row['input_row_ids']
  sequences=list(dict.fromkeys(r['sequence'] for r in inputs));hashes=[hashlib.sha256(s.encode()).hexdigest() for s in sequences]
  batches=row['actual_ESM_forward_batches'];assert len(batches)==math.ceil(len(hashes)/4)
  for i,event in enumerate(batches):
   assert event['sequences']==hashes[4*i:4*i+4] and event['device']=='cuda' and event['dtype']=='torch.float32' and event['eval']
   assert event['all_floating_parameters_buffers_cuda_fp32'] and event['floating_parameter_buffer_count']>0
  assert set(row['_features'][1])==set(hashes)
  identity=row['_features'][0]['identity'];assert identity['fair_esm_sources']==row['environment']['fair_esm_sources'] and identity['fair_esm_version']==row['environment']['fair_esm_version']
  assert identity['esm_code']==SCIENCE_HASHES['catpred/data/esm_utils.py'] and identity['numeric_contract']=='FP32_exact' and identity['layer']==33 and identity['batch_size']==4
  protein=row.get('protein_backend')
  if protein is None or protein['actual_backend']=='original_fair_esm_2':
   assert identity['backend']=='original_fair_esm_2' and 'protein_backend_identity' not in identity
   return
  assert row['arm']=='S1' and protein['actual_backend']=='catpred_accel_features'
  assert protein['closed'] and protein['engaged'] and protein['model_released'] and protein['generation_in_this_request']
  assert protein['batch_telemetry_source']=='candidate_successful_batch_observer' and protein['forward_batches']==batches
  mode,loader=protein['requested_mode'],protein['requested_loader_mode'];assert mode in ('off','head_only','weights_only','representations') and loader in ('off','meta') and (mode!='off' or loader!='off')
  context=protein['context'];static=identity['protein_backend_identity'];assert identity['backend']=='catpred_accel_features'
  assert static==protein['identity'] and static['context']==context and static['weights']==ESM_WEIGHTS and static['caller_sha256']==SCIENCE_HASHES['catpred/data/esm_utils.py']
  assert static['mode']==mode and static['loader_mode']==loader and static['batch_size']==4 and static['max_tokens']==2048
  assert static['ordered_request_sha256']==ledger_payload_sha(sequences) and static['cache_namespace']==protein['cache_namespace']
  assert static['cache_namespace']==ledger_payload_sha(dict(schema=1,kind='esm_features_actual_cache',context=context,weights=ESM_WEIGHTS,caller=SCIENCE_HASHES['catpred/data/esm_utils.py'],mode=mode,loader_mode=loader,ordered_sequence_hashes=hashes,batch_size=4,max_tokens=2048,layer=33,dtype='float32',padding='original ordered group of four'))
  assert context['implementation_sources']==row['runtime_manifest']['source_hashes'] and context['fair_esm_sources']=={n:e['sha256'] for n,e in row['environment']['fair_esm_sources'].items()}
  observed=row['runtime_manifest']['hardware']
  assert observed['device_type']=='cuda' and observed['name']==row['environment']['gpu'] and observed['compute_capability']==row['environment']['capability'] and observed['total_memory']==row['environment']['gpu_total_bytes']
  assert context['hardware']==dict(name=row['environment']['gpu'],capability=row['environment']['capability'],total_memory=row['environment']['gpu_total_bytes'],multiprocessors=observed['multiprocessors'])
  software=context['software'];assert software['torch']==row['environment']['torch'] and software['cuda']==row['environment']['cuda'] and software['threads']==24 and software['interop_threads']==1 and software['matmul_tf32'] is False and software['cudnn_tf32'] is True
  for event in batches:
   decision=event['decision'];assert decision['requested_mode']==decision['used_mode']==mode and decision['fallback_reason'] is None
   assert decision['context_key']==ledger_payload_sha(context) and decision['engaged']==(mode!='off')
   if mode!='off':assert decision['counters']==dict(lm_head_calls=0 if mode in ('head_only','representations') else 1,attention_calls=33)
  assert len(protein['loader_receipts'])==1
  load=protein['loader_receipts'][0];assert load['requested_mode']==load['used_mode']==loader and load['engaged']==(loader=='meta') and load['fallback_reason'] is None
  assert load['hidden_gpu_transfers']==0 and load['output_device']=='cpu' and load['dtype']=='torch.float32'
  assert load['context_key']==ledger_payload_sha(load['context']) and load['context']['weights']==ESM_WEIGHTS and load['context']['implementation_sources']==context['implementation_sources']
 def source_object(self,source):
  manifest=read(self.path(source['manifest']));assert manifest==source['files'];snapshot=self.path(source['manifest']).parent
  for name,value in manifest.items():self.checked(relative(self.root,snapshot/name),value)
  for name,value in BASELINE_HASHES.items():assert manifest['baseline/'+name]==value
  for name,value in SCIENCE_HASHES.items():assert manifest['CatPred/'+name]==value
 def stream_request(self,request,rows,budget,failed=False):
  assert request['closed'] and request['request_cache_initial_empty'] and request['request_cache_empty_on_exit']
  assert bool(request['failure'])==failed
  stream=request['stream'];assert stream['enabled'] and stream['closed'] and bool(stream['failure'])==failed
  assert stream['budget_bytes']==budget and 0<stream['device_cache_peak_bytes']<=budget
  assert stream['retained_after_request_bytes']==0 and stream['fallback_count']==0
  offset=0
  for chunk in stream['chunks']:
   assert chunk['start_row']==offset and chunk['stop_row']>offset
   assert chunk['actual_unique_device_storage_bytes']==chunk['planned_device_storage_bytes']<=budget
   assert chunk['released'] and chunk['actual_device_cache_bytes_after_clear']==0 and chunk['device_tensor_references_alive_after_clear']==0
   offset=chunk['stop_row']
  assert stream['chunk_count']==len(stream['chunks'])
  if not failed:assert offset==rows and stream['forward_counts']==[math.ceil(rows/50)]*10
  return stream
 def lifecycle_call(self,entry,reference,expected_index,path_kind):
  row=read(self.checked(entry['path'],entry['sha256']));assert row['call_index']==expected_index and row['path_kind']==path_kind
  assert row['excluded_from_headline'] and row['workload']==reference['workload'] and row['rows']==reference['rows']
  assert row['input_row_ids']==reference['input_row_ids'] and row['checkpoints']==reference['checkpoints']
  assert row['environment']==dict(reference['environment'],process_id=row['environment']['process_id'])
  self.source_object(row['active_sources'])
  if expected_index==1:
   assert row['completed'] is False and row['failure_type']=='InjectedAfterAllocation'
   with zipfile.ZipFile(self.checked(row['partial_member_path'],row['partial_member_sha256'])) as archive:
    assert archive.namelist()==['0.npy'];partial=decode_npy(archive.read('0.npy'))
   count=partial['shape'][0];assert 0<count<=row['rows'] and partial['shape']==(count,2) and partial['dtype']=='<f4'
   assert all(map(math.isfinite,partial['values']))
   assert row['partial_arrays']=={'0':dict(dtype='<f4',shape=[count,2],finite=True)}
   assert row['member_pieces']['0']==count and all(v==0 for k,v in row['member_pieces'].items() if k!='0')
   assert row['model_forward_counts']==[math.ceil(count/50)]+[0]*9
   original=self.arrays[reference['_receipt']]['member_raw'];assert partial['payload']==original['payload'][:count*2*4]
   return dict(path=entry['path'],completed=False,partial_member0_rows=count,partial_member0_exact=True,finite=True)
  assert row['completed'] is True and row['feature_cache_unchanged'] and not row['actual_ESM_forward_batches']
  assert row['model_forward_counts']==[math.ceil(row['rows']/50)]*10
  for key in ('precision','precision_manifest','prediction'):self.checked(row[key+'_path'],row[key+'_sha256'])
  arrays=load_npz(self.path(row['precision_path']));results={key:compare(self.arrays[reference['_receipt']][key],value) for key,value in arrays.items()}
  assert all(v['exact'] and v['finite'] for v in results.values())
  metadata=read(self.path(row['precision_manifest_path']));assert metadata==reference['_meta']
  features=self.features(row);assert features[0]['identity']==self.owner['manifest']['identity']
  assert row['feature_preflight_checks']==features[0]['sequence_count']
  feature_result=self.feature_comparison(reference['_features'],features);assert feature_result['exact']
  with self.path(row['prediction_path']).open(newline='') as a,self.path(reference['prediction_path']).open(newline='') as b:assert list(csv.reader(a))==list(csv.reader(b))
  return dict(path=entry['path'],completed=True,arrays=results,metadata_equal=True,features=feature_result,prediction_csv_equal=True,exact=True)
 def lifecycle(self,entry):
  saved=read(self.checked(entry['receipt_path'],entry['receipt_sha256']));assert saved['schema']==1 and saved['passed'] and saved['excluded_from_headline']
  ref=saved['reference'];reference=next(r for r in self.rows if r['_receipt']==ref['receipt_path']);self.checked(ref['receipt_path'],ref['receipt_sha256'])
  assert reference['phase']=='baseline-gate' and reference['arm']=='A' and reference['workload']==saved['workload']
  assert saved['environment']==dict(reference['environment'],process_id=saved['environment']['process_id'])
  self.source_object(saved['active_sources'])
  assert saved['caller_entry_fingerprint']==saved['caller_exit_fingerprint'] and saved['private_and_injected_restored_to_baseline']
  assert set(saved['paths'])==set(saved['outcomes'])=={'injected','private'}
  verified={};contexts=set()
  for kind in ('injected','private'):
   path=saved['paths'][kind];out=saved['outcomes'][kind];assert len(path['calls'])==3
   verified[kind]=[self.lifecycle_call(call,reference,i,kind) for i,call in enumerate(path['calls'])]
   assert out['passed'] and out['post_allocation_exception'] and out['allocated_input_bytes_before_failure']>0
   assert out['weak_input_tensor_count']>0 and out['weak_inputs_alive_after_failure']==0
   assert out['entry_fingerprint']==out['first_exit_fingerprint']==out['second_exit_fingerprint']
   assert out['original_function_instance_hook_identities_restored'] and out['second_activation_restored'] and out['post_exception_output_exact_checked'] and out['excluded_from_headline']
   self.stream_request(out['first_request'],reference['rows'],2<<30)
   self.stream_request(out['fresh_request'],reference['rows'],2<<30)
   failed=self.stream_request(out['failed_request'],reference['rows'],2<<30,failed=True)
   assert failed['forward_counts']==[math.ceil(verified[kind][1]['partial_member0_rows']/50)]+[0]*9
   checks=out['allocation_free_checks']
   for key in ('passed','allocation_free','over_budget_reserve_rejected_before_copy','failed_request_closed','next_request_closed','context_states_cleared','configuration_unchanged'):assert checks[key] is True
   assert checks['configured_budget_bytes']==2<<30
   manifest=path['runtime_manifest'];assert manifest['restored'] and manifest['engaged'] and manifest['selected_backend']=='accepted_stream'
   contexts.add(manifest['context_key'])
   for name,value in manifest['source_hashes'].items():assert saved['active_sources']['files']['catpred_accel/'+name]==value
  budgets={}
  assert set(saved['budget_checks'])=={'1','4'}
  for gib,record in saved['budget_checks'].items():
   assert record['passed'] and record['excluded_from_headline'] and record['input_budget_bytes']==int(gib)<<30
   self.stream_request(record['request'],reference['rows'],int(gib)<<30)
   assert record['runtime_manifest']['restored'] and record['runtime_manifest']['requested']['input_budget_bytes']==int(gib)<<30
   budgets[gib]=self.lifecycle_call(record['call'],reference,0,'budget_'+gib+'g')
  guards=saved['guards'];assert guards['passed'] and guards['caller_entries_restored']
  required={'uncertified_default_stock','non_null_input_scaler','pinned_source_digest_mismatch','invalid_device_input_budget','explicit_wrong_hardware','non_original_batch_shape','live_input_configuration_changed'}
  cases={case['case']:case for case in guards['cases']};assert set(cases)==required and len(cases)==len(guards['cases'])
  assert all(case['passed'] and case['reason'] for case in cases.values())
  wrong=cases['non_original_batch_shape'];assert wrong['no_device_input_allocation'] and wrong['request']['request_cache_empty_on_exit'] and wrong['request']['stream']['device_cache_peak_bytes']==0
  # Private vendoring does not change the scientific runtime context.
  candidates=[r for r in self.rows if r['phase']=='candidate-gate' and r['arm']=='S1']
  assert candidates and all(r['runtime_manifest']['source_hashes']==saved['paths']['injected']['runtime_manifest']['source_hashes'] for r in candidates)
  assert len(contexts)==1 and contexts=={r['runtime_manifest']['context_key'] for r in candidates},'Lifecycle and packaged gate context differs'
  return dict(status='PASS',receipt_path=entry['receipt_path'],receipt_sha256=entry['receipt_sha256'],workload=saved['workload'],paths=verified,budget_checks=budgets,guard_cases=sorted(cases),function_instance_hooks_restored=True,post_allocation_exception_cleaned=True,excluded_from_headline=True)
 def verify(self):
  try:self.prepare_owner()
  except Exception as ex:
   self.errors.append(dict(stage='prepare_owner',error=repr(ex)))
   return dict(status='FAIL',all_baseline_gates_passed=False,baseline_gate={},errors=self.errors,missing=['valid prepare owner'],receipt_count=0,comparison_count=0,comparisons=[],statistics=[],created_utc=utc())
  base=self.root/'results/experiments/tasks'
  for taskpath in sorted(base.glob('*/index.json')):
   task=read(taskpath)
   for e in task.get('receipts',[]):
    try:
     row=self.receipt(e['path'],e['sha256']);assert row['repetition']==e['rep']
     for key in ('phase','workload','arm'):assert row[key]==task['task'][key]
     if 'esm_mode' in task['task']:
      assert row['protein_backend']['requested_mode']==task['task']['esm_mode']
      assert row['protein_backend'].get('requested_loader_mode',row['protein_backend'].get('loader_mode'))==task['task']['esm_loader_mode']
    except Exception as ex:self.errors.append(dict(path=e['path'],error=repr(ex)))
  by=defaultdict(list)
  for r in self.rows:by[(r['phase'],r['workload'],r['arm'])].append(r)
  gate={};missing=[]
  for n in WORKLOADS:
   aa=sorted(by['baseline-gate',n,'A'],key=lambda x:x['repetition']);ss=sorted(by['baseline-gate',n,'S0'],key=lambda x:x['repetition'])
   try:
    if len(aa)>=2:self.pair(aa[0],aa[1],n+': A versus A')
    if len(ss)>=2:self.pair(ss[0],ss[1],n+': S0 versus S0')
    if aa:
     for s in ss:self.pair(aa[0],s,n+': A versus S0')
    gate[n]=len(aa)==2 and len(ss)==2
   except Exception as ex:self.errors.append(dict(workload=n,error=repr(ex)));gate[n]=False
   if not gate[n]:missing.append('baseline-gate:'+n)
  candidate_gate={}
  for n in WORKLOADS:
   aa=sorted(by['baseline-gate',n,'A'],key=lambda x:x['repetition']);ss=sorted(by['candidate-gate',n,'S1'],key=lambda x:x['repetition'])
   try:
    if ss:
     assert aa,'S1 candidate requires the same-workload A reference'
     for r in ss:self.pair(aa[0],r,n+': A versus S1 candidate')
    if len(ss)>=2:
     self.pair(ss[0],ss[1],n+': S1 versus S1 candidate')
     assert ss[0]['runtime_manifest']['context_key']==ss[1]['runtime_manifest']['context_key'],'S1 candidate context changed between repeats'
    candidate_gate[n]=len(ss)==2
   except Exception as ex:self.errors.append(dict(stage='candidate-gate',workload=n,error=repr(ex)));candidate_gate[n]=False
  for r in self.rows:
   if r['phase'] in ('warm','profile-warm'):
    refs=by['baseline-gate',r['workload'],'A']
    if refs:
     try:
      self.pair(refs[0],r,r['phase']+':'+r['workload']+':'+r['arm']+':'+str(r['repetition']))
      if r['arm']=='S1':
       candidates=by['candidate-gate',r['workload'],'S1'];assert len(candidates)==2
       assert {x['runtime_manifest']['context_key'] for x in candidates}=={r['runtime_manifest']['context_key']},'S1 implementation/context changed after exact gate'
     except Exception as ex:self.errors.append(dict(path=r['_receipt'],error=repr(ex)))
    else:self.errors.append(dict(path=r['_receipt'],error='Missing baseline reference'))
  for n in {r['workload'] for r in self.rows if r['phase']=='cold'}:
   aa=sorted(by['cold',n,'A'],key=lambda x:x['repetition'])
   if aa:
    for arm in ('A','S0','S1'):
     for r in by['cold',n,arm]:
      if r is aa[0]:continue
      try:
       self.pair(aa[0],r,'cold:'+n+':'+arm+':'+str(r['repetition']))
       if arm=='S1':
        contexts={x['runtime_manifest']['context_key'] for x in self.rows if x['phase']=='candidate-gate' and x['arm']=='S1'}
        assert contexts=={r['runtime_manifest']['context_key']},'Cold S1 context did not pass the packaged runtime gate'
      except Exception as ex:self.errors.append(dict(path=r['_receipt'],error=repr(ex)))
  for r in self.rows:
   if r['phase']=='profile-cold':
    refs=by['cold',r['workload'],'A']
    if refs:
     try:self.pair(refs[0],r,'profile-cold:'+r['workload']+':'+r['arm'])
     except Exception as ex:self.errors.append(dict(path=r['_receipt'],error=repr(ex)))
    else:self.errors.append(dict(path=r['_receipt'],error='Missing cold A panel reference'))
  lifecycle_results=[]
  for taskpath in sorted(base.glob('*/index.json')):
   task=read(taskpath)
   if task['task']['phase']!='lifecycle':continue
   for attempt in task.get('attempts',[]):
    complete=self.path(attempt['path'])/'COMPLETE.json'
    if complete.exists():
     try:lifecycle_results.append(self.lifecycle(read(complete)['lifecycle']))
     except Exception as ex:self.errors.append(dict(stage='lifecycle',path=relative(self.root,complete),error=repr(ex)))
  summary=[]
  for (phase,n,arm),rows in sorted(by.items()):
   if not rows or phase not in ('warm','cold'):continue
   if arm=='S1':
    contexts={r['runtime_manifest']['context_key'] for r in rows}
    if len(contexts)!=1:self.errors.append(dict(stage='summary',phase=phase,workload=n,arm=arm,error='Different S1 implementation/context keys in one timing group'))
   times=sorted(r['seconds'] for r in rows);count=len(times);expected=3 if phase=='cold' or n=='screening_16384' else 5
   summary.append(dict(phase=phase,workload=n,arm=arm,n=count,expected_n=expected,process_count=len({r['environment']['process_id'] for r in rows}),median_seconds=statistics.median(times),minimum_seconds=times[0],maximum_seconds=times[-1],all_seconds=times,rows_per_second=rows[0]['rows']/statistics.median(times),max_cuda_peak_allocated_bytes=max(r['cuda_peak_allocated_bytes'] for r in rows),max_cuda_peak_reserved_bytes=max(r['cuda_peak_reserved_bytes'] for r in rows),max_cpu_peak_rss_bytes=max(r['cpu_peak_rss_bytes'] for r in rows)))
  return dict(status='PASS' if not self.errors else 'FAIL',all_baseline_gates_passed=all(gate.values()) and not self.errors,baseline_gate=gate,candidate_gate=candidate_gate,lifecycle_verification=lifecycle_results,lifecycle_passed=bool(lifecycle_results) and all(r['status']=='PASS' for r in lifecycle_results) and not self.errors,all_candidate_gates_passed=all(candidate_gate.values()) and not self.errors,errors=self.errors,missing=missing,receipt_count=len(self.rows),comparison_count=len(self.comparisons),comparisons=self.comparisons,statistics=summary,atol=ATOL,rtol=RTOL,exact_policy='Shape/dtype and logical C-order bit patterns, including signed zeros; finite values required',verified_feature_files=len(self.feature_files),finite_verification=dict(ledger_root=str(self.finite_ledger_root),algorithm_sha256=FINITE_ALGORITHM_SHA256,ledger_hits=self.finite_ledger_hits,fresh_complete_scans=self.finite_fresh_scans,invalid_receipts_rechecked=self.invalid_finite_receipts,actual_file_sha256_recomputed=True),created_utc=utc())

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output');p.add_argument('--finite-ledger-root');p.add_argument('--require-baseline',action='store_true');p.add_argument('--require-candidate',action='store_true');p.add_argument('--require-lifecycle',action='store_true');a=p.parse_args();v=Verifier(a.root,finite_ledger_root=a.finite_ledger_root).verify();atomic(a.output or Path(a.root)/'results/experiments/verification.json',v);print(json.dumps({k:v[k] for k in ('status','all_baseline_gates_passed','receipt_count','comparison_count','errors','missing')}));return 0 if v['status']=='PASS' and (not a.require_baseline or v['all_baseline_gates_passed']) and (not a.require_candidate or v.get('all_candidate_gates_passed') is True) and (not a.require_lifecycle or v.get('lifecycle_passed') is True) else 1
if __name__=='__main__':raise SystemExit(main())
