"""Run framework-free tests and verify the published source snapshot."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
for entry in manifest['copied_files']:
    path = ROOT / entry['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], entry['path']

suites = [
    (ROOT / 'implementation', 'tests', 'test*.py'),
    (ROOT / 'source/kinetics_candidate/implementation', 'tests', 'test*.py'),
    (ROOT / 'source/kinetics_candidate', 'experiments', 'test_kinetics_harness.py'),
    (ROOT, 'experiments', 'test_kinetics_lifecycle_stdlib.py'),
    (ROOT, 'experiments/tests', 'test_analyze_kinetics_streaming.py'),
]
for cwd, tests, pattern in suites:
    env = dict(os.environ, PYTHONPATH=str(cwd), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', tests, '-p', pattern],
                   cwd=cwd, env=env, check=True)
print('Source hashes and all framework-free tests passed.')
