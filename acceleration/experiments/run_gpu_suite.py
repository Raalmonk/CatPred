"""Bounded foreground suite: fresh child attempts, immutable receipts, task-level resume."""
from __future__ import annotations
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from evidence import *
from verify_gpu_results import Verifier

def proc_start_ticks(pid):
 return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])

def terminate_owned(child):
 if child.poll() is not None:return
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:child.wait(timeout=20)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait(timeout=20)

def interrupted(signum,frame):raise SystemExit(128+signum)

def tasks(stage,arms):
 if stage=='prepare':return [dict(id='prepare',phase='prepare',arm='A',workload='mixed_valid_2047',repetitions=1)]
 if stage=='baseline-gate':return [dict(id=f'gate_{n}_{arm}',phase=stage,arm=arm,workload=n,repetitions=2) for n in WORKLOADS for arm in ('A','S0')]
 if stage=='candidate-gate':return [dict(id=f'candidate_gate_{n}_S1',phase=stage,arm='S1',workload=n,repetitions=2) for n in WORKLOADS]
 if stage=='lifecycle':return [dict(id='S1_lifecycle',phase=stage,arm='S1',workload='mixed_valid_2047',repetitions=1)]
 if stage=='warm':return [dict(id=f'warm_{n}_{arm}',phase=stage,arm=arm,workload=n,repetitions=3 if n=='screening_16384' else 5) for n in WORKLOADS for arm in arms]
 if stage=='cold':return [dict(id=f'cold_mixed_prefix_100_{arm}_{rep}',phase=stage,arm=arm,workload='mixed_prefix_100',repetitions=1,repetition=rep) for rep in range(1,4) for arm in (arms if rep%2 else tuple(reversed(arms)))]
 if stage=='cold-full':return [dict(id=f'cold_full_mixed_valid_2047_{arm}_{rep}',phase='cold',arm=arm,workload='mixed_valid_2047',repetitions=1,repetition=rep) for rep in range(1,4) for arm in (arms if rep%2 else tuple(reversed(arms)))]
 if stage=='profile':return [dict(id=f'profile_{phase}' if arm=='S0' else f'profile_{arm}_{phase}',phase=phase,arm=arm,workload='mixed_valid_2047' if phase=='profile-warm' else 'mixed_prefix_100',repetitions=1) for arm in arms if arm in ('S0','S1') for phase in ('profile-warm','profile-cold')]
 raise ValueError(stage)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',required=True,choices=('prepare','baseline-gate','candidate-gate','lifecycle','warm','cold','cold-full','profile'));p.add_argument('--arms',nargs='+',choices=('A','S0','S1'),default=['A','S0','S1']);p.add_argument('--max-new-tasks',type=int,default=1);p.add_argument('--timeout-seconds',type=int,default=1800);p.add_argument('--prepare-timeout-seconds',type=int,default=3600);p.add_argument('--esm-mode',choices=('off','head_only','weights_only','representations'),default='off');p.add_argument('--esm-loader-mode',choices=('off','meta'),default='off');a=p.parse_args()
 assert sys.platform=='linux','Suite execution is remote Linux only';assert a.timeout_seconds>0 and a.prepare_timeout_seconds>0 and a.max_new_tasks>0
 root=Path(a.root).resolve();out=root/'results/experiments';out.mkdir(parents=True,exist_ok=True)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
 if a.stage in ('candidate-gate','lifecycle','warm','cold','cold-full','profile'):
  gate=Verifier(root).verify();atomic(out/'verification.json',gate);assert gate['all_baseline_gates_passed'],'Strict baseline gate must precede candidate/performance/profile stages'
  if a.stage in ('warm','cold','cold-full','profile','lifecycle') and 'S1' in a.arms:
   assert gate.get('all_candidate_gates_passed') is True,'S1 exact candidate gate must precede S1 performance/lifecycle stages'
 schedule=tasks(a.stage,tuple(a.arms));started=0
 for task in schedule:
  if task['arm']=='S1' and task['phase'] in ('cold','profile-cold'):
   task.update(esm_mode=a.esm_mode,esm_loader_mode=a.esm_loader_mode)
 atomic(out/'suite_schedule.json',dict(stage=a.stage,tasks=schedule,timeout_seconds=a.timeout_seconds,prepare_timeout_seconds=a.prepare_timeout_seconds))
 for task in schedule:
  folder=out/'tasks'/task['id'];folder.mkdir(parents=True,exist_ok=True);indexpath=folder/'index.json'
  index=read(indexpath) if indexpath.exists() else dict(task=task,receipts=[],attempts=[],status='PENDING')
  assert index['task']==task,'Task scientific identity changed'
  # Recover every completed request receipt even if the parent/child stopped before index update.
  existing={e['rep']:e for e in index['receipts']}
  for attempt in index['attempts']:
   for receipt in sorted((root/attempt['path']).glob('rep_*/receipt.json')):
    row=read(receipt);rep=row['repetition'];candidate=dict(rep=rep,path=relative(root,receipt),sha256=digest(receipt))
    if rep in existing:assert existing[rep]==candidate,'Two completed receipts for one scientific repetition'
    else:existing[rep]=candidate
  required=[task.get('repetition',1)+i for i in range(task['repetitions'])]
  for e in existing.values():assert digest(root/e['path'])==e['sha256']
  index['receipts']=sorted(existing.values(),key=lambda e:e['rep'])
  prepare_done=task['phase'] in ('prepare','lifecycle') and any((root/e['path']/'COMPLETE.json').exists() for e in index['attempts'])
  missing=[i for i in required if i not in existing]
  if prepare_done or not missing:
   index['status']='COMPLETE';atomic(indexpath,index);continue
  if started>=a.max_new_tasks:
   atomic(out/'suite_status.json',dict(stage=a.stage,status='awaiting_resume',next_task=task['id'],completed_new_tasks=started,utc=utc()));return 0
  # Each invocation starts at most one missing contiguous repetition block.
  reps=[missing[0]]
  if task['phase'] not in ('prepare','lifecycle'):
   for i in missing[1:]:
    if i==reps[-1]+1:reps.append(i)
    else:break
  attempt=folder/('attempt_'+str(len(index['attempts'])+1).zfill(3));attempt.mkdir(exist_ok=False)
  entry=dict(path=relative(root,attempt),started_utc=utc(),status='RUNNING');index['attempts'].append(entry);index['status']='RUNNING';atomic(indexpath,index)
  cmd=[sys.executable,str(root/'experiments/gpu_request.py'),'--root',str(root),'--task',task['id'],'--attempt',str(attempt),'--arm',task['arm'],'--workload',task['workload'],'--phase',task['phase'],'--repetitions',str(len(reps)),'--start-repetition',str(reps[0])]
  if 'esm_mode' in task:cmd.extend(['--esm-mode',task['esm_mode'],'--esm-loader-mode',task['esm_loader_mode']])
  timeout=a.prepare_timeout_seconds if task['phase']=='prepare' else a.timeout_seconds
  with (attempt/'stdout.log').open('ab') as log:
   child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
   # Persistent PID files let the owning controller find children after parent interruption.
   (out/f'gpu_worker_{task["id"]}_{child.pid}.pid').write_text(str(child.pid)+'\n')
   atomic(out/'worker_state.json',dict(pid=child.pid,pgid=child.pid,process_start_ticks=proc_start_ticks(child.pid),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),task=task,attempt=relative(root,attempt),timeout_seconds=timeout,start_epoch=time.time()))
   try:
    try:code=child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:terminate_owned(child);code=124
   finally:terminate_owned(child)
  entry.update(returncode=code,status='COMPLETE' if code==0 else 'FAILED',ended_utc=utc())
  for receipt in sorted(attempt.glob('rep_*/receipt.json')):
   row=read(receipt);rep=row['repetition'];assert rep not in existing
   existing[rep]=dict(rep=rep,path=relative(root,receipt),sha256=digest(receipt))
  index['receipts']=sorted(existing.values(),key=lambda e:e['rep']);index['status']='COMPLETE' if code==0 and (task['phase'] in ('prepare','lifecycle') or all(i in existing for i in required)) else 'INCOMPLETE';atomic(indexpath,index)
  # Standard-library verifier runs outside the application timer, before later task release.
  if task['phase']!='prepare':
   verified=Verifier(root).verify();atomic(out/'verification.json',verified)
   if verified['status']!='PASS':
    atomic(out/'suite_status.json',dict(stage=a.stage,status='strict_gate_failed',task=task['id'],utc=utc()));return 2
  started+=1
  atomic(out/'suite_status.json',dict(stage=a.stage,status='task_complete' if code==0 else 'task_failed',task=task['id'],completed_new_tasks=started,returncode=code,utc=utc()))
  if code:return code
 atomic(out/'suite_status.json',dict(stage=a.stage,status='COMPLETE',completed_new_tasks=started,utc=utc()));return 0

if __name__=='__main__':raise SystemExit(main())
