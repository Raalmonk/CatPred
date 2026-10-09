"""Prepare the pinned comparison environment; imports and --help do no setup."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request

MODEL_URL = 'https://catpred.s3.us-east-1.amazonaws.com/pretrained_production.tar.gz'
ESM_URL = 'https://dl.fbaipublicfiles.com/fair-esm/'
OWNER = '.catpred-colab-owner.json'
TORCH_VERSION = '2.11.0+cu130'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verified(path, record):
    path = Path(path)
    return (path.is_file() and not path.is_symlink()
            and ('bytes' not in record or path.stat().st_size == record['bytes'])
            and sha256(path) == record['sha256'])


def safe_path(root, relative):
    """Do not follow a pre-existing cache symlink or an archive traversal."""
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Unsafe relative path: ' + str(relative))
    root = Path(root).absolute()
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink in destination: ' + str(current))
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError('Destination escapes its root')
    return current


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.is_symlink() or path.is_symlink():
        raise ValueError('Refusing symlink output')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def copy_verified(source, target, record):
    if verified(target, record):
        return
    target = Path(target)
    if target.exists() or target.is_symlink():
        raise ValueError('Existing file differs: ' + str(target))
    if not verified(source, record):
        raise ValueError('Source hash mismatch: ' + str(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.copying')
    if temporary.is_symlink():
        raise ValueError('Refusing symlink temporary file')
    shutil.copyfile(source, temporary)
    if not verified(temporary, record):
        raise ValueError('Copy hash mismatch: ' + str(target))
    temporary.replace(target)


def download(url, target, record=None):
    """Resume interrupted transfers; promote verified weights atomically."""
    target = Path(target)
    if record is not None and verified(target, record):
        return
    if target.is_symlink():
        raise ValueError('Refusing symlink cache file')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    if partial.is_symlink():
        raise ValueError('Refusing symlink partial file')
    if record and partial.exists() and partial.stat().st_size >= record['bytes']:
        if verified(partial, record):
            partial.replace(target)
            return
        partial.unlink()
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {'Range': f'bytes={offset}-'} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = response.status
        if status == 206:
            content_range = response.headers.get('Content-Range', '')
            if not content_range.startswith(f'bytes {offset}-'):
                raise ValueError('Unexpected partial-download range')
            mode = 'ab' if offset else 'wb'
        elif status == 200:
            mode = 'wb'  # A server may ignore Range; restart without duplicating bytes.
        else:
            raise ValueError('Unexpected download status: ' + str(status))
        with partial.open(mode) as output:
            shutil.copyfileobj(response, output, 1024 * 1024)
    if record is not None and not verified(partial, record):
        partial.unlink()
        raise ValueError('Downloaded weight hash mismatch: ' + target.name)
    partial.replace(target)


def extract_checkpoints(archive, cache, records):
    """Extract only the manifest's ten regular files; verify before promotion."""
    wanted = {}
    for record in records:
        name = record['path']
        safe_path(cache, name)
        wanted[name] = record
        if name.startswith('data/'):
            wanted[name[5:]] = record  # Public archive starts at pretrained/.
    seen = set()
    with tarfile.open(archive, mode='r|gz') as bundle:
        for member in bundle:
            name = str(PurePosixPath(member.name))
            safe_path(cache, name)
            if member.issym() or member.islnk():
                raise ValueError('Archive contains a link: ' + name)
            record = wanted.get(name)
            if record is None:
                continue
            if record['path'] in seen:
                raise ValueError('Duplicate checkpoint in archive: ' + name)
            seen.add(record['path'])
            if not member.isfile() or member.size != record['bytes']:
                raise ValueError('Unexpected checkpoint entry: ' + name)
            target = safe_path(cache, record['path'])
            if verified(target, record):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = safe_path(cache, record['path'] + '.extracting')
            with bundle.extractfile(member) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            if not verified(temporary, record):
                temporary.unlink()
                raise ValueError('Archive checkpoint hash mismatch: ' + name)
            temporary.replace(target)
    missing = [r['path'] for r in records if not verified(safe_path(cache, r['path']), r)]
    if missing:
        raise ValueError('Archive is missing expected checkpoints: ' + ', '.join(missing))


def staged_records(acceleration):
    manifest = json.loads((acceleration / 'SOURCE_MANIFEST.json').read_text())
    records = []
    prefixes = ('source/CatPred/', 'source/baseline/', 'implementation/',
                'source/kinetics_candidate/implementation/', 'source/accepted_distribution/')
    for record in manifest['copied_files']:
        name = record['path']
        if name.startswith(prefixes) or name in (
                'source/prior_checkpoint_manifest.json', 'source/prior_esm_checkpoint_manifest.json'):
            source = safe_path(acceleration, name)
            if not verified(source, record):
                raise ValueError('Published source hash mismatch: ' + name)
            records.append({**record, 'destination': name.removeprefix('source/')})
    if not records:
        raise ValueError('Empty source manifest')
    return records


def stage_checkout(checkout, work_root):
    acceleration = Path(checkout) / 'acceleration'
    work_root = Path(work_root)
    if work_root.is_symlink():
        raise ValueError('Work root must not be a symlink')
    records = staged_records(acceleration)
    identity = {'schema': 1, 'sources': records}
    owner = work_root / OWNER
    if owner.exists():
        if owner.is_symlink() or json.loads(owner.read_text()) != identity:
            raise ValueError('Work root belongs to a different source snapshot')
    elif work_root.exists() and any(work_root.iterdir()):
        raise ValueError('Choose an empty work root; this directory is not owned by this notebook')
    else:
        work_root.mkdir(parents=True, exist_ok=True)
        write_json(owner, identity)
    for record in records:
        copy_verified(safe_path(acceleration, record['path']),
                      safe_path(work_root, record['destination']), record)
    return records


def prepare_assets(acceleration, work_root, asset_cache):
    model_records = json.loads((acceleration / 'source/prior_checkpoint_manifest.json').read_text())['files']
    expected = {f'data/pretrained/production/kcat/fold_0/model_{i}/model.pt' for i in range(10)}
    if len(model_records) != 10 or {r['path'] for r in model_records} != expected:
        raise ValueError('Expected the original ten kcat checkpoints')
    cache_models = safe_path(asset_cache, 'checkpoints')
    if not all(verified(safe_path(cache_models, r['path']), r) for r in model_records):
        archive = safe_path(asset_cache, 'downloads/pretrained_production.tar.gz')
        if not archive.is_file():
            download(MODEL_URL, archive)
        try:
            extract_checkpoints(archive, cache_models, model_records)
        except (tarfile.TarError, EOFError, ValueError):
            archive.unlink(missing_ok=True)  # Keep already verified individual weights.
            raise
    for record in model_records:
        copy_verified(safe_path(cache_models, record['path']), safe_path(work_root, record['path']), record)
    esm_records = json.loads((acceleration / 'source/prior_esm_checkpoint_manifest.json').read_text())
    expected_esm = {'esm2_t33_650M_UR50D.pt', 'esm2_t33_650M_UR50D-contact-regression.pt'}
    if len(esm_records) != 2 or {r['name'] for r in esm_records} != expected_esm:
        raise ValueError('Expected the original ESM-2 650M weights')
    for record in esm_records:
        cached = safe_path(asset_cache, 'esm/' + record['name'])
        category = 'regression/' if 'contact-regression' in record['name'] else 'models/'
        download(ESM_URL + category + record['name'], cached, record)
        copy_verified(cached, safe_path(work_root, 'torch/hub/checkpoints/' + record['name']), record)
    return model_records, esm_records


def command(args, log, timeout=900, **kwargs):
    result = subprocess.run([str(arg) for arg in args], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout, **kwargs)
    with Path(log).open('a') as output:
        output.write(json.dumps([str(arg) for arg in args]) + '\n' + result.stdout + '\n')
    if result.returncode:
        raise RuntimeError(result.stdout[-5000:])
    return result.stdout


def hardware():
    if platform.system() != 'Linux' or platform.machine() != 'x86_64' or sys.version_info[:2] != (3, 13):
        raise RuntimeError('Select a Colab G4 runtime with Linux x86_64 and Python 3.13')
    import torch
    if torch.__version__ != TORCH_VERSION or not torch.cuda.is_available():
        raise RuntimeError('This comparison requires the G4 system PyTorch 2.11.0+cu130 and CUDA')
    cpu = subprocess.check_output(['lscpu'], text=True)
    if 'AMD EPYC 9B45' not in cpu:
        raise RuntimeError('The bundled native wheel was validated on AMD EPYC 9B45')
    props = torch.cuda.get_device_properties(0)
    return {'python': sys.version, 'torch': torch.__version__, 'torch_file': torch.__file__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': props.name, 'gpu_bytes': props.total_memory,
            'capability': list(torch.cuda.get_device_capability()),
            'cpu_count': os.cpu_count(), 'cpu': cpu}


def install(work_root, details):
    log = work_root / 'setup_commands.log'
    environment = work_root / 'venv'
    python = environment / 'bin/python'
    if not python.exists():
        command([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', environment], log)
    constraint = work_root / 'preserve_system_torch.txt'
    constraint.write_text('torch==' + TORCH_VERSION + '\n')
    command([python, '-m', 'pip', 'install', '--disable-pip-version-check', '-c', constraint,
             '-e', work_root / 'CatPred', 'numpy==2.1.3', 'pandas==2.2.3', 'rdkit==2026.3.6',
             'fair-esm==2.0.0', 'rotary-embedding-torch==0.9.1',
             'ipdb==0.13.13', 'maturin==1.9.6', 'pytest', 'psutil'], log)
    wheels = list((work_root / 'accepted_distribution').glob('*.whl'))
    if len(wheels) != 1 or 'cp313-cp313' not in wheels[0].name:
        raise ValueError('Expected the accepted CPython 3.13 native wheel')
    command([python, '-m', 'pip', 'install', '--no-deps', wheels[0],
             '-e', work_root / 'implementation', '-e', work_root / 'kinetics_candidate/implementation'], log)
    check = ('import torch,json,catpred,esm,rdkit,catpred_accel,catpred_kinetics;'
             'import catpred_rust_packing as p;assert p.BUILD_PROFILE=="release";'
             'print("IDENTITY="+json.dumps({"torch":torch.__version__,"torch_file":torch.__file__}))')
    output = command([python, '-c', check], log, cwd=work_root / 'CatPred')
    actual = json.loads(next(line[9:] for line in output.splitlines() if line.startswith('IDENTITY=')))
    if actual != {key: details[key] for key in ('torch', 'torch_file')}:
        raise RuntimeError('System PyTorch identity changed during setup')
    return python


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--asset-cache', type=Path, required=True)
    parser.add_argument('--run', action='store_true', help='Install dependencies and fetch missing verified weights')
    args = parser.parse_args(argv)
    if not args.run:
        parser.error('Pass --run to perform setup')
    details = hardware()  # Reject a different machine before installation or downloads.
    checkout, work_root, cache = (p.absolute() for p in (args.checkout, args.work_root, args.asset_cache))
    records = stage_checkout(checkout, work_root)
    python = install(work_root, details)
    models, esm = prepare_assets(checkout / 'acceleration', work_root, cache)
    result = {'status': 'ready', 'checkout': str(checkout), 'work_root': str(work_root),
              'asset_cache': str(cache), 'python': str(python), 'py': str(python),
              'models_root': str(work_root / 'data/pretrained/production/kcat'),
              'torch_home': str(work_root / 'torch'), 'hardware': details,
              'checkpoint_files': models, 'esm_files': esm,
              'source_files_verified': len(records), 'system_torch_preserved': True}
    result.update(models=result['models_root'], torchhome=result['torch_home'])
    write_json(work_root / 'setup.json', result)
    print(json.dumps({'status': 'ready', 'setup': str(work_root / 'setup.json')}))


if __name__ == '__main__':
    main()
