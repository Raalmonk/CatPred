"""Read-only stdlib analysis of saved K0/K1 warm requests and shape profiles.

Example: python3 analyze_kinetics.py --root SAVED_WORK --verification warm_verification.json --output analysis.json
This does not run models, replace raw verification, or issue an acceptance certificate.
"""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json,math,statistics,time
from pathlib import Path
WORKLOADS={'mixed_valid_2047':2047,'reuse_4096':4096,'screening_16384':16384}
REPETITIONS={'mixed_valid_2047':5,'reuse_4096':5,'screening_16384':3}

def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def ensure(value,message):
 if not value:raise ValueError(message)
def local(root,path,remote):
 p=Path(path)
 if p.is_absolute():
  if p.is_relative_to(root):
   p=p.resolve();ensure(p.is_relative_to(root),'Absolute path escapes saved root');return p
  p=p.relative_to(remote)
 p=(root/p).resolve();ensure(p.is_relative_to(root),'Path escapes saved root: '+str(path));return p

# Limits cover one complete JSON value, not the whole trace. Unknown metadata
# values obey the same bound and are discarded after decoding. Oversized or
# malformed values fail explicitly; no prefix of a trace is accepted as complete.
STREAM_CHUNK_CHARS = 64 << 10
STREAM_MAX_VALUE_CHARS = 8 << 20


class _TraceJSONReader:
 """Incrementally decode a top-level trace object and yield its array events."""
 def __init__(self, stream, chunk_chars=None, max_value_chars=None):
  self.stream=stream
  self.chunk_chars=STREAM_CHUNK_CHARS if chunk_chars is None else chunk_chars
  self.limit=STREAM_MAX_VALUE_CHARS if max_value_chars is None else max_value_chars
  ensure(type(self.chunk_chars) is int and self.chunk_chars>0,'Invalid JSON read size')
  ensure(type(self.limit) is int and self.limit>0,'Invalid JSON value limit')
  self.buffer='';self.position=0;self.offset=0;self.eof=False
  self.decoder=json.JSONDecoder();self.device_properties=None
  self.peak_buffer_chars=0;self.trace_seen=False;self.complete=False

 def _fill(self):
  if self.position:
   self.offset+=self.position;self.buffer=self.buffer[self.position:];self.position=0
  if self.eof:return False
  room=self.limit-len(self.buffer)
  ensure(room>0,'JSON value exceeds bounded buffer of '+str(self.limit)+' characters at '+str(self.offset))
  block=self.stream.read(min(self.chunk_chars,room))
  if not block:self.eof=True;return False
  self.buffer+=block;self.peak_buffer_chars=max(self.peak_buffer_chars,len(self.buffer))
  return True

 def _space(self):
  while True:
   while self.position<len(self.buffer) and self.buffer[self.position] in ' \t\r\n':self.position+=1
   if self.position<len(self.buffer) or not self._fill():return

 def _peek(self):
  self._space()
  return self.buffer[self.position:self.position+1]

 def _expect(self,character):
  ensure(self._peek()==character,'Expected '+repr(character)+' at JSON character '+str(self.offset+self.position))
  self.position+=1

 def _value(self):
  self._space()
  while True:
   try:value,end=self.decoder.raw_decode(self.buffer,self.position)
   except json.JSONDecodeError as error:
    if self._fill():continue
    raise ValueError('Incomplete or malformed JSON value at character '+str(self.offset+error.pos)+': '+error.msg) from error
   # A number may end exactly at a read boundary, or raw_decode may accept
   # only the prefix of an incomplete exponent. Require a real delimiter.
   if end==len(self.buffer):
    if self._fill():continue
   elif self.buffer[end] not in ' \t\r\n,]}:':
    if self._fill():continue
    raise ValueError('Invalid JSON value delimiter at character '+str(self.offset+end))
   self.position=end
   return value

 def events(self):
  self._expect('{');keys=set();first=True
  while self._peek()!='}':
   if not first:self._expect(',')
   first=False
   key=self._value();ensure(isinstance(key,str),'Top-level JSON key is not a string')
   ensure(key not in keys,'Duplicate top-level JSON key: '+key);keys.add(key)
   self._expect(':')
   if key=='traceEvents':
    self.trace_seen=True;self._expect('[');first_event=True
    while self._peek()!=']':
     if not first_event:self._expect(',')
     first_event=False
     event=self._value();ensure(isinstance(event,dict),'Trace event is not a JSON object')
     yield event
    self._expect(']')
   else:
    value=self._value()
    if key=='deviceProperties':self.device_properties=value
    del value
  self._expect('}')
  ensure(self.trace_seen,'Top-level traceEvents array is missing')
  ensure(self._peek()=='','Trailing content after complete trace object')
  self.complete=True


def _open_trace(path):
 return gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open('r',encoding='utf-8')


def trace_analysis(trace_path):
 """Stream all events; preserve original metrics, shapes and addition order."""
 trace_path=Path(trace_path)
 counts=collections.Counter();shapes={};mean_ids=set();missing=0;mean_count=0;event_count=0
 shape_observed=False;kernel_by_id={}
 markers=('scaled_dot_product','sdpa','native_multi_head_attention','native_mha','flash')
 changed_backend=collections.Counter()
 with _open_trace(trace_path) as stream:
  reader=_TraceJSONReader(stream)
  for event in reader.events():
   event_count+=1
   if event.get('cat')=='cpu_op' and event.get('ph')=='X':
    name=event['name'];counts[name]+=1;args=event.get('args',{})
    shape_observed=shape_observed or 'Input Dims' in args or 'Input Shapes' in args
    if name=='aten::mean':
     mean_count+=1;dims=args.get('Input Dims',args.get('Input Shapes'))
     if dims is None:missing+=1
     key=json.dumps(dims,sort_keys=True)
     group=shapes.setdefault(key,dict(input_dimensions=dims,count=0,cpu_total_duration_us=0.,concrete_inputs_examples=[]))
     group['count']+=1;group['cpu_total_duration_us']+=event.get('dur',0.)
     concrete=args.get('Concrete Inputs')
     if concrete is not None and concrete not in group['concrete_inputs_examples'] and len(group['concrete_inputs_examples'])<3:group['concrete_inputs_examples'].append(concrete)
     if 'External id' in args:
      external=args['External id']
      mean_ids.add(external)
   if event.get('cat')=='kernel':
    external=event.get('args',{}).get('External id');duration=event.get('dur',0.)
    group=kernel_by_id.setdefault(external,[0,0]);group[0]+=1;group[1]+=duration
   if event.get('ph')=='X' and any(s in event.get('name','').lower() for s in markers):changed_backend[event.get('name','')]+=1
  ensure(reader.complete,'Incomplete trace stream')
  device_properties=reader.device_properties
 # Resolve all mean ids before a second streaming pass. Using the built-in
 # sum over events in their original order preserves that Python version's
 # exact summation semantics (including the Python 3.12+ compensated sum).
 mean_kernel_count=sum(group[0] for external,group in kernel_by_id.items() if external in mean_ids)
 with _open_trace(trace_path) as stream:
  reader=_TraceJSONReader(stream)
  known_mean_kernel_duration=sum(event.get('dur',0.) for event in reader.events()
    if event.get('cat')=='kernel' and event.get('args',{}).get('External id') in mean_ids)
  ensure(reader.complete,'Incomplete trace replay')
 rank4=0
 for group in shapes.values():
  dims=group['input_dimensions'];first=dims[0] if isinstance(dims,list) and dims else None
  if isinstance(first,list) and len(first)==4 and first[2]==first[3]:rank4+=group['count']
 result=dict(trace_path=str(trace_path),trace_sha256=sha(trace_path),event_count=event_count,cpu_operator_counts=dict(sorted(counts.items())),
   bmm_calls=counts['aten::bmm'],softmax_calls=counts['aten::_softmax'],mean_calls=mean_count,mean_events_missing_shapes=missing,
   mean_shapes_available=(missing==0),shape_instrumentation_observed=shape_observed,mean_shape_groups=[shapes[k] for k in sorted(shapes)],
   rank4_square_mean_calls=rank4 if not missing else None,
   directly_linked_mean_kernel_count=mean_kernel_count,directly_linked_mean_kernel_duration_us=known_mean_kernel_duration,
   sdpa_native_mha_flash_events=dict(sorted(changed_backend.items())),device_properties=device_properties,
   time_interpretation='Kernel sums explain saved profiled execution only; CPU/GPU intervals can overlap and are never added or used as a predicted request speedup.')
 suffix='.json.gz' if trace_path.name.endswith('.json.gz') else '.json'
 operators=trace_path.with_name(trace_path.name[:-len(suffix)]+'_operators.json')
 if operators.exists():
  rows=read(operators);aten=[r for r in rows if r['operator'].startswith('aten::')]
  chosen=('aten::mean','aten::bmm','aten::_softmax','aten::addmm','aten::mm','aten::linear','aten::copy_','aten::tanh')
  result.update(operators_path=str(operators),operators_sha256=sha(operators),aten_self_cuda_sum_us=sum(r['self_device_time_us'] for r in aten),
    selected_operator_records=[r for r in aten if r['operator'] in chosen],
    operator_sum_rule='ATen self_device_time only; exclude separately listed CUDA leaf kernels to avoid double counting.')
 return result

def load_task(root,phase,workload,arm,remote):
 task=root/'results/kinetics/tasks'/f'{phase}_{workload}_{arm}';index_path=task/'index.json'
 if not index_path.exists():return dict(status='not_run',task=str(task.relative_to(root)),n=0,rows=[])
 idx=read(index_path);contract=idx['contract'];ensure((contract['phase'],contract['workload'],contract['arm'])==(phase,workload,arm),'Task contract mismatch')
 expected=REPETITIONS[workload] if phase=='warm' else 1
 ensure(contract['repetitions']==expected,'Unexpected declared repetitions')
 entries=idx['receipts'];seen=set();rows=[]
 for entry in entries:
  ensure(entry['repetition'] not in seen,'Duplicate indexed repetition');seen.add(entry['repetition'])
  ensure(1<=entry['repetition']<=expected,'Repetition outside declared count')
  path=local(root,entry['path'],remote);ensure(sha(path)==entry['sha256'],'Receipt SHA mismatch: '+str(path));row=read(path)
  ensure((row['phase'],row['workload'],row['kinetics_arm'],row['repetition'])==('kinetics-'+phase,workload,arm,entry['repetition']),'Receipt identity mismatch')
  ensure(row['rows']==WORKLOADS[workload] and row['arm']=='S1','Workload/source mismatch')
  ensure(row['profiled'] is (phase=='profile') and not row['diagnostic'],'Mixed timing boundaries')
  ensure(math.isfinite(row['seconds']) and row['seconds']>0,'Invalid time')
  ensure(row['model_forward_counts']==[math.ceil(row['rows']/50)]*10,'Original model count changed')
  rows.append(dict(receipt=row,path=str(path.relative_to(root)),sha256=entry['sha256']))
 complete=idx['status']=='COMPLETE' and len(rows)==expected
 return dict(status='complete' if complete else 'partial',index_status=idx['status'],task=str(task.relative_to(root)),index_path=str(index_path.relative_to(root)),index_sha256=sha(index_path),n=len(rows),expected_n=expected,rows=rows)

def warm_stats(task):
 result={k:v for k,v in task.items() if k!='rows'};rows=[x['receipt'] for x in task['rows']]
 result['receipt_sha256']={x['path']:x['sha256'] for x in task['rows']}
 if not rows:
  result.update(seconds=[],median_seconds=None,range_seconds=None,time_reduction=None,process_count=None,cuda_allocated_peak_bytes=None,cuda_allocated_peak_max_bytes=None);return result
 times=[r['seconds'] for r in rows];allocated=[r['cuda_peak_allocated_bytes'] for r in rows]
 result.update(seconds=times,median_seconds=statistics.median(times),range_seconds=[min(times),max(times)],
   process_count=len({(r['environment']['runtime_boot_id'],r['environment']['process_id']) for r in rows}),
   runtime_boot_ids=sorted({r['environment']['runtime_boot_id'] for r in rows}),
   source_identities=sorted({r['kinetics_sources']['scientific_identity'] for r in rows}),
   baseline_context_keys=sorted({r['runtime_manifest']['context_key'] for r in rows}),
   rows_per_second=rows[0]['rows']/statistics.median(times),cuda_allocated_peak_bytes=allocated,cuda_allocated_peak_max_bytes=max(allocated),
   cuda_reserved_peak_max_bytes=max(r['cuda_peak_reserved_bytes'] for r in rows),cpu_rss_peak_max_bytes=max(r['cpu_peak_rss_bytes'] for r in rows))
 return result

def analyze(root,remote,verification=None):
 root=Path(root).resolve();remote=Path(remote);errors=[];tasks={};warm=[];profiles=[];comparisons=[]
 for workload in WORKLOADS:
  pair={}
  for arm in ('K0','K1'):
   try:task=load_task(root,'warm',workload,arm,remote);tasks[('warm',workload,arm)]=task;pair[arm]=warm_stats(task)
   except Exception as e:errors.append(dict(phase='warm',workload=workload,arm=arm,error=type(e).__name__+': '+str(e)));pair[arm]=dict(status='error')
  a,b=pair['K0'],pair['K1'];both=all(x.get('median_seconds') is not None for x in (a,b));complete=all(x['status']=='complete' for x in (a,b))
  comparable=both and all(a[k]==b[k] and len(a[k])==1 for k in ('runtime_boot_ids','source_identities','baseline_context_keys'))
  row=dict(workload=workload,rows=WORKLOADS[workload],arms=pair,complete=complete,identical_runtime_source_context=comparable,
      K1_time_reduction=(1-b['median_seconds']/a['median_seconds']) if comparable else None,
      K0_over_K1_speedup=a['median_seconds']/b['median_seconds'] if comparable else None,
      K1_minus_K0_peak_allocated_bytes=b['cuda_allocated_peak_max_bytes']-a['cuda_allocated_peak_max_bytes'] if comparable else None)
  warm.append(row)
  pp={}
  for arm in ('K0','K1'):
   try:
    task=load_task(root,'profile',workload,arm,remote);tasks[('profile',workload,arm)]=task
    if not task['rows']:p=dict(status='not_run',workload=workload,arm=arm)
    else:
     saved=task['rows'][0];r=saved['receipt'];trace=local(root,r['profile']['trace_path'],remote)
     ensure(trace.stat().st_size==r['profile']['compressed_bytes'],'Profile gzip size mismatch')
     p=dict(status=task['status'],workload=workload,arm=arm,receipt_path=saved['path'],receipt_sha256=saved['sha256'],profiled_seconds_excluded_from_warm=r['seconds'],record_shapes_requested=r.get('kinetics_profile_record_shapes'),analysis=trace_analysis(trace))
    pp[arm]=p;profiles.append(p)
   except Exception as e:errors.append(dict(phase='profile',workload=workload,arm=arm,error=type(e).__name__+': '+str(e)));pp[arm]=dict(status='error')
  if all(pp[a].get('analysis') is not None for a in ('K0','K1')):
   x,y=pp['K0']['analysis'],pp['K1']['analysis']
   comparisons.append(dict(workload=workload,K0_mean_calls=x['mean_calls'],K1_mean_calls=y['mean_calls'],K0_mean_shape_groups=x['mean_shape_groups'],K1_mean_shape_groups=y['mean_shape_groups'],
     shapes_observable=x['mean_shapes_available'] and y['mean_shapes_available'] and x['shape_instrumentation_observed'] and y['shape_instrumentation_observed'],bmm_calls_unchanged=x['bmm_calls']==y['bmm_calls'],softmax_calls_unchanged=x['softmax_calls']==y['softmax_calls'],
     K0_bmm_calls=x['bmm_calls'],K1_bmm_calls=y['bmm_calls'],K0_softmax_calls=x['softmax_calls'],K1_softmax_calls=y['softmax_calls'],
     K1_sdpa_native_mha_flash_events=y['sdpa_native_mha_flash_events'],new_backend_events=sorted(set(y['sdpa_native_mha_flash_events'])-set(x['sdpa_native_mha_flash_events'])),
     rank4_square_mean_removed=(x['rank4_square_mean_calls']>0 and y['rank4_square_mean_calls']==0) if x['rank4_square_mean_calls'] is not None and y['rank4_square_mean_calls'] is not None and x['shape_instrumentation_observed'] and y['shape_instrumentation_observed'] else None))
 strict=dict(status='not_supplied',passed=False)
 if verification:
  try:
   v=read(verification);ensure(v['status']=='PASS' and v['phase']=='warm','Expected independent warm PASS')
   ensure(v['receipt_count']==38 and v['comparison_count']==35 and v['exact'] is True,'Incomplete raw validation coverage')
   ensure(len(v['comparisons'])==35 and all(x['exact'] for x in v['comparisons']),'Nonexact raw comparison')
   refs={x['path']:x['sha256'] for x in v['task_indices']}
   for path,value in refs.items():ensure(sha(local(root,path,remote))==value,'Stale independent verification index')
   for key,t in tasks.items():
    if key[0]=='warm':ensure(t.get('index_path') in refs and refs[t['index_path']]==t['index_sha256'],'Warm task not independently verified')
   strict=dict(status='PASS',passed=True,path=str(Path(verification).resolve()),sha256=sha(verification))
  except Exception as e:strict=dict(status='FAIL',passed=False,error=type(e).__name__+': '+str(e));errors.append(dict(phase='independent_verification',error=str(e)))
 eligible=all(r['complete'] and r['identical_runtime_source_context'] for r in warm)
 threshold=dict(complete_warm_matrix=eligible,formula='reduction=1-median(K1)/median(K0); speedup=median(K0)/median(K1)',
   at_least_one_reduction_ge_2pct=any(r['K1_time_reduction']>=.02 for r in warm) if eligible else None,
   none_slower_more_than_3pct=all(r['K1_time_reduction']>=-.03 for r in warm) if eligible else None,
   all_peak_allocated_nonincreasing=all(r['K1_minus_K0_peak_allocated_bytes']<=0 for r in warm) if eligible else None,
   adoption='not_decided_here',remaining_acceptance='Real exception restoration and any memory increase require separate evidence; this helper issues no certificate.')
 return dict(schema=1,root=str(root),analysis_status='error' if errors else ('complete' if eligible and len(comparisons)==3 else 'partial'),errors=errors,
  strict_numeric_verification=strict,warm=warm,profiles=profiles,profile_comparisons=comparisons,performance_thresholds=threshold,
  method='Read-only saved receipts; no torch, models, compilation or remote activity. Profile times never enter warm medians.',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--remote-root',default='/content/catpred_accel_research');p.add_argument('--verification',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 out=analyze(a.root,a.remote_root,a.verification);payload=json.dumps(out,indent=2,allow_nan=False)+'\n'
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(payload);print(json.dumps(dict(analysis_status=out['analysis_status'],errors=len(out['errors']),output=str(a.output))))
 else:print(payload,end='')
 return 1 if out['errors'] else 0
if __name__=='__main__':raise SystemExit(main())
