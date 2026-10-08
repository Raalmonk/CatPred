"""Filesystem evidence helpers; standard library only, safe to use in saved analysis."""
from __future__ import annotations
import hashlib, json, os, shutil, time
from pathlib import Path

WORKLOADS = {'mixed_valid_2047':2047, 'reuse_4096':4096, 'screening_16384':16384}
# Immutable Phase 3 byte identities, independent of an editable manifest.
INPUT_HASHES = {
 'mixed_valid_2047':'dcb2a10b208496d8f040973ccdb44a6e3079c6cb980d11d823ffcec8fd50f213',
 'reuse_4096':'a9aae7750c5a2750d05f8a0710810cf2bc0b37f85b06aed11e8bf3622064c0c7',
 'screening_16384':'1034c8cde94a0d5df1e6cd32af9b9edc176a6509454d268bdf982d328aec320e'}
BASELINE_HASHES = {
 'packing.py':'38616de366710f06c61c6090576b78f4c634feed4b0d6ee92180bf08bb75b77f',
 'reuse.py':'594f14ac87387b0b8fc22df1692c4c25c7510f24cce905f748c54168c98763c0',
 'probes.py':'2588f38d4fc381f80b887f3c6fb231e25757d36944dd78aed1baccc8e879e088'}
BASELINE_HASHES.update({'streaming.py': '6871b65ae3b7cfa9798bc239ad2ca04a49df5d712e461d49baf11c91d4db0829', 'capture.py': '4080231420d2b2a6474de76074bc04a64794ae327e3b7b8de38e64bd3dfccf12', 'diagnostics.py': 'd9cdb786714de9eaf628f69479d20328372701640777443224b15e615d87e787'})
SCIENCE_HASHES = {
 'catpred/data/esm_utils.py':'80fa2786ca403bc49c442115c20e9971fa32f9d61c480c0a38028aeb7877f5c4',
 'catpred/features/featurization.py':'4b37628e1fc59e786371a780f979400e0e33faa4440c50004d6551e25a4d3006',
 'catpred/uncertainty/uncertainty_predictor.py':'87692aabf08a30d8bbc52290026abc72afad2c6bd25339671273e2875509616c',
 'catpred/train/predict.py':'0dad566c983ad95a4fc14d912da6ad713fe3ed19007ba8c719b05a83c86e9c2c',
 'catpred/models/model.py':'8cbc6ee75ff2c6526031e856612bf4384d81d3673be49902a8ac3cb708f39676',
 'catpred/models/mpn.py':'d1309fdf0c6daf2409a780ed5d85da399778f7067142e6b9099cbbf2f8143313'}
ESM_WEIGHTS = {'esm2_t33_650M_UR50D.pt':'ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c',
 'esm2_t33_650M_UR50D-contact-regression.pt':'8ffe6edbd4173dc8d45c2cd5cb27d43aad77ec26b4c768200c58ae1f96693575'}

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 tmp.replace(path)

def read(path):return json.loads(Path(path).read_text())
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def relative(root,path):return str(Path(path).resolve().relative_to(Path(root).resolve()))

def source_history(root):
 root=Path(root);out=root/'results/experiments/source_history';files=[]
 for folder in ('baseline','experiments','CatPred/catpred','catpred_accel'):
  files += list((root/folder).rglob('*.py')) if (root/folder).is_dir() else []
 files += [p for p in (root/'rust_packing').rglob('*') if p.is_file() and (p.name in ('Cargo.toml','Cargo.lock','pyproject.toml') or p.suffix=='.rs') and 'target' not in p.parts]
 entries={relative(root,p):digest(p) for p in sorted(files)}
 identity=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest();dest=out/identity
 for name,sha in entries.items():
  target=dest/name
  if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
  assert digest(target)==sha,name
 atomic(dest/'manifest.json',entries)
 return {'identity':identity,'manifest':relative(root,dest/'manifest.json'),'files':entries}
