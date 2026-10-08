"""Idempotent Colab-only setup; weights are restored independently from Drive."""
from pathlib import Path
import concurrent.futures, hashlib, json, os, subprocess, sys, time, traceback
R=Path('/content/catpred_accel_research'); O=R/'results'; O.mkdir(exist_ok=True)
def write(name,value):
 p=O/name; t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def run(argv,timeout=600,**kw):
 p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout,**kw)
 with (O/'setup_commands.log').open('a') as f:f.write(json.dumps(argv)+'\n'+p.stdout+'\n')
 if p.returncode:raise RuntimeError(str(argv)+'\n'+p.stdout[-5000:])
 return p.stdout
def main():
 import torch
 start=time.time();write('setup_status.json',{'status':'running','started_epoch':start})
 props=torch.cuda.get_device_properties(0)
 assert torch.cuda.is_available()
 write('hardware.json',{'epoch':start,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
  'python':sys.version,'torch':torch.__version__,'torch_file':torch.__file__,'cuda':torch.version.cuda,
  'cudnn':torch.backends.cudnn.version(),'gpu':props.name,'capability':list(torch.cuda.get_device_capability()),
  'gpu_bytes':props.total_memory,'cpu_count':os.cpu_count(),'affinity':sorted(os.sched_getaffinity(0)),
  'cpu':run(['lscpu'],30),'gpu_detail':run(['nvidia-smi','-q'],30),'meminfo':Path('/proc/meminfo').read_text(),
  'cgroup_limits':{n:(Path('/sys/fs/cgroup')/n).read_text().strip() for n in ['cpu.max','memory.max','cpuset.cpus.effective'] if (Path('/sys/fs/cgroup')/n).exists()},
  'initial_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'initial_tf32_cudnn':torch.backends.cudnn.allow_tf32})
 original={'version':torch.__version__,'file':torch.__file__}
 def dependencies():
  v=R/'venv'
  if not (v/'bin/python').exists():run([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(v)],60)
  py=str(v/'bin/python');c=R/'preserve_system_torch.txt';c.write_text('torch=='+torch.__version__+'\n')
  run([py,'-m','pip','install','--disable-pip-version-check','-c',str(c),'-e',str(R/'CatPred'),
       'fair-esm==2.0.0','rotary-embedding-torch==0.9.1','ipdb==0.13.13','maturin==1.9.6','pytest','psutil'],900)
  text=run([py,'-c','import torch,json,catpred,esm,rdkit;print("IDENTITY="+json.dumps({"version":torch.__version__,"file":torch.__file__}))'],90,cwd=R/'CatPred')
  actual=json.loads(next(x.split('=',1)[1] for x in text.splitlines() if x.startswith('IDENTITY=')))
  assert actual==original
  write('system_torch_preservation.json',{'original':original,'installed':actual,'unchanged':True})
  write('installed_packages.json',json.loads(run([py,'-m','pip','list','--format=json'],60)))
 dependencies()
 py=str(R/'venv/bin/python')
 wheels=list((R/'accepted_distribution').glob('*.whl'));assert len(wheels)==1
 assert 'AMD EPYC 9B45' in run(['lscpu'],30),'Review native CPU compatibility before using accepted wheel'
 assert sys.version_info[:2]==(3,13)
 run([py,'-m','pip','install','--no-deps',str(wheels[0])],90)
 text=run([py,'-c','import catpred_rust_packing as p;assert p.BUILD_PROFILE=="release";print(p.__file__)'],60)
 write('native_build.json',{'mode':'reuse_accepted_G4_native_wheel','wheel':wheels[0].name,'sha256':hashlib.sha256(wheels[0].read_bytes()).hexdigest(),'import_result':text,'cpu':run(['lscpu'],30)})
 write('setup_status.json',{'status':'ready','seconds':time.time()-start})
if __name__=='__main__':
 try:main()
 except BaseException:
  write('setup_failure.json',{'epoch':time.time(),'traceback':traceback.format_exc()});raise
