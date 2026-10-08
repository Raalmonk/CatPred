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

def trace_analysis(trace_path):
 """Trace events are primary evidence for counts/shapes; never infer missing shapes."""
 trace_path=Path(trace_path)
 with gzip.open(trace_path,'rt') if trace_path.suffix=='.gz' else trace_path.open() as f:trace=json.load(f)
 events=trace['traceEvents'];cpu=[e for e in events if e.get('cat')=='cpu_op' and e.get('ph')=='X']
 counts=collections.Counter(e['name'] for e in cpu)
 means=[e for e in cpu if e['name']=='aten::mean'];shapes={};mean_ids=set();missing=0
 for event in means:
  args=event.get('args',{});dims=args.get('Input Dims',args.get('Input Shapes'))
  if dims is None:missing+=1
  key=json.dumps(dims,sort_keys=True)
  group=shapes.setdefault(key,dict(input_dimensions=dims,count=0,cpu_total_duration_us=0.,concrete_inputs_examples=[]))
  group['count']+=1;group['cpu_total_duration_us']+=event.get('dur',0.)
  concrete=args.get('Concrete Inputs')
  if concrete is not None and concrete not in group['concrete_inputs_examples'] and len(group['concrete_inputs_examples'])<3:group['concrete_inputs_examples'].append(concrete)
  if 'External id' in args:mean_ids.add(args['External id'])
 mean_kernel_events=[e for e in events if e.get('cat')=='kernel' and e.get('args',{}).get('External id') in mean_ids]
 markers=('scaled_dot_product','sdpa','native_multi_head_attention','native_mha','flash')
 changed_backend=collections.Counter(e.get('name','') for e in events if e.get('ph')=='X' and any(s in e.get('name','').lower() for s in markers))
 rank4=0
 for group in shapes.values():
  dims=group['input_dimensions'];first=dims[0] if isinstance(dims,list) and dims else None
  if isinstance(first,list) and len(first)==4 and first[2]==first[3]:rank4+=group['count']
 result=dict(trace_path=str(trace_path),trace_sha256=sha(trace_path),event_count=len(events),cpu_operator_counts=dict(sorted(counts.items())),
   bmm_calls=counts['aten::bmm'],softmax_calls=counts['aten::_softmax'],mean_calls=len(means),mean_events_missing_shapes=missing,
   mean_shapes_available=(missing==0),shape_instrumentation_observed=any('Input Dims' in e.get('args',{}) or 'Input Shapes' in e.get('args',{}) for e in cpu),mean_shape_groups=[shapes[k] for k in sorted(shapes)],
   rank4_square_mean_calls=rank4 if not missing else None,
   directly_linked_mean_kernel_count=len(mean_kernel_events),directly_linked_mean_kernel_duration_us=sum(e.get('dur',0.) for e in mean_kernel_events),
   sdpa_native_mha_flash_events=dict(sorted(changed_backend.items())),device_properties=trace.get('deviceProperties'),
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
