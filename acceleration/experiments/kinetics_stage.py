from pathlib import Path
import argparse,subprocess,sys,json,time
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['gate','warm','profile'],required=True);a=p.parse_args()
R=Path('/content/catpred_accel_research');C=R/'kinetics_candidate';out=R/'results/kinetics';out.mkdir(parents=True,exist_ok=True)
def write(j):
 path=out/(a.phase+'_progress.json');tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(j,indent=2));tmp.replace(path)
workloads=['mixed_valid_2047','reuse_4096','screening_16384'];start=time.time()
for i,w in enumerate(workloads):
 for mode in (['off','head_mean'] if i%2==0 else ['head_mean','off']):
  write({'phase':a.phase,'status':'running','workload':w,'mode':mode,'elapsed_seconds':time.time()-start})
  subprocess.run([sys.executable,str(C/'experiments/kinetics_worker.py'),'--root',str(R),'--workload',w,'--mode',mode,'--phase',a.phase,'--run'],check=True,timeout=1800)
subprocess.run([sys.executable,str(C/'experiments/verify_kinetics.py'),'--root',str(R),'--phase',a.phase,'--output',str(out/(a.phase+'_verification.json'))],check=True,timeout=600)
write({'phase':a.phase,'status':'COMPLETE','elapsed_seconds':time.time()-start})
