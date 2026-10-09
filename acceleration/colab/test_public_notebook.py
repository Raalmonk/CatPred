"""Public notebook onboarding checks. No setup, network, or model execution."""
import ast
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import types
import unittest
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parents[2]


def notebook():
    return json.loads((ROOT / 'colab_compare.ipynb').read_text())


def code_cells():
    return [''.join(cell['source']) for cell in notebook()['cells'] if cell['cell_type'] == 'code']


class PublicNotebookTests(unittest.TestCase):
    def test_no_drive_import_mount_or_visible_setting(self):
        for source in code_cells():
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(node.module == 'google.colab' and any(alias.name == 'drive' for alias in node.names))
                elif isinstance(node, ast.Import):
                    self.assertFalse(any(alias.name.startswith('google.colab.drive') for alias in node.names))
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertFalse(node.func.attr == 'mount', 'Public cells must not mount storage')
            for line in source.splitlines():
                if '#@param' in line or '# @param' in line:
                    self.assertNotIn('drive', line.lower())
                    self.assertNotIn('commit', line.lower())
            self.assertNotIn('/content/drive', source)
            self.assertNotIn('use_drive_cache', source)
            self.assertNotIn('save_zip_to_drive', source)

    def cache_selection(self, environment):
        tree = ast.parse(code_cells()[0])
        relevant = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if names & {'asset_cache_override', 'ASSET_CACHE'}:
                    relevant.append(node)
            elif isinstance(node, ast.If):
                assigned = {target.id for child in ast.walk(node) if isinstance(child, ast.Assign)
                            for target in child.targets if isinstance(target, ast.Name)}
                if 'ASSET_CACHE' in assigned:
                    relevant.append(node)
        self.assertTrue(relevant, 'Cache selection must be visible in the setup cell')
        values = {'Path': Path, 'os': types.SimpleNamespace(environ=environment), 'HEADLESS': False}
        exec(compile(ast.Module(body=relevant, type_ignores=[]), '<cache-selection>', 'exec'), values)
        return values['ASSET_CACHE']

    def test_default_cache_is_local_and_needs_no_credentials(self):
        self.assertEqual(self.cache_selection({}), Path('/content/catpred-weights-cache'))

    def test_explicit_cache_override_is_preserved(self):
        custom = '/content/preloaded-public-models'
        self.assertEqual(self.cache_selection({'CATPRED_NOTEBOOK_ASSET_CACHE': custom}), Path(custom))

    def test_default_public_example_keeps_all_rows_in_order(self):
        source = code_cells()[1]
        self.assertIn('input_mode = "Bundled example"', source)
        self.assertIn('repeat_count = 1', source)
        with (ROOT / 'demo/batch_kcat.csv').open(newline='') as stream:
            original_rows = list(csv.DictReader(stream))
        self.assertEqual(len(original_rows), 14)
        with tempfile.TemporaryDirectory() as folder:
            def local_path(value):
                return Path(folder) / 'runs' if value == '/content/catpred-comparisons' else Path(value)
            values = dict(Path=local_path, CHECKOUT=ROOT, SETUP={'status': 'ready'}, ACTUAL_COMMIT=None,
                          CHECKOUT_STATUS='', SOURCE_MANIFEST_SHA=None, SOURCE_BUNDLE_SHA=None,
                          SETUP_PATH=Path(folder) / 'setup.json', datetime=datetime, timezone=timezone,
                          uuid=uuid, json=json, hashlib=hashlib)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, '<public-input>', 'exec'), values)
            with values['INPUT_CSV'].open(newline='') as stream:
                actual = list(csv.DictReader(stream))
            self.assertEqual(actual, original_rows)
            self.assertEqual(len(actual), 14)

    def test_incomplete_comparison_never_displays_timing_table(self):
        source = code_cells()[2]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_csv = root / 'input.csv'; input_csv.write_text('SMILES,sequence\nC,AAAA\n')
            displayed, commands = [], []
            def run_logged(arguments, log_path):
                commands.append(arguments)
                output = Path(arguments[arguments.index('--output') + 1])
                output.mkdir()
                (output / 'summary.json').write_text(json.dumps({'status': 'FAILED', 'reason': 'Fixture precision mismatch'}))
            values = dict(Path=Path, RUN_DIR=root, INPUT_CSV=input_csv, PYTHON='python',
                          COMPARE_SCRIPT=Path('compare.py'), SETUP_PATH=Path('setup.json'),
                          uuid=uuid, json=json, run_logged=run_logged, display=displayed.append,
                          summary={'status': 'COMPLETE', 'arms': {'stale': {'median_seconds': 1}}})
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                exec(compile(source, '<comparison>', 'exec'), values)
            self.assertEqual(displayed, [])
            self.assertNotEqual(values['summary'].get('status'), 'COMPLETE')
            self.assertEqual(len(commands), 1)
            command = commands[0]
            self.assertEqual(command[command.index('--input') + 1], input_csv)
            self.assertEqual(command[command.index('--repeats') + 1], '3')
            self.assertEqual(command[command.index('--variants') + 1], 'best')
            self.assertNotIn('--k1', command)

    def test_public_notes_and_saved_outputs_show_only_two_versions(self):
        cells = notebook()['cells']
        visible = []
        for cell in cells:
            if cell['cell_type'] == 'markdown':
                visible.append(''.join(cell['source']))
            for output in cell.get('outputs', []):
                visible.append(''.join(output.get('text', [])))
                visible.extend(''.join(value) for mime, value in output.get('data', {}).items()
                               if mime in {'text/plain', 'text/html', 'text/markdown'})
        text = '\n'.join(visible)
        self.assertNotIn('S_STREAM', text)
        self.assertNotIn('K1', text)
        labels = [''.join(output['text']).strip() for output in cells[5]['outputs']
                  if output['output_type'] == 'stream']
        self.assertEqual(labels, ['Original', 'Optimized'])
        tree = ast.parse(''.join(cells[4]['source']))
        versions = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'VERSIONS' for t in node.targets))
        self.assertEqual(versions, (('Original', 'Original'), ('S_STREAM_K1', 'Optimized')))

    def test_saved_two_version_table_uses_recorded_measurements(self):
        cells = notebook()['cells']
        table = next(''.join(output['data']['text/html']) for output in cells[4]['outputs']
                     if 'text/html' in output.get('data', {}))
        rows = [re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
                for row in re.findall(r'<tr>.*?</tr>', table, re.S)]
        rows = [row for row in rows if row]
        saved = json.loads((ROOT / 'acceleration/colab/example_results/summary.json').read_text())
        if 'esm_comparison' in saved['preparation']:
            esm = saved['preparation']['esm_comparison']['arms']
            stages = [('ESM features + loading', esm['Original'], esm['Optimized']),
                      ('Prediction (models loaded)', saved['arms']['Original'], saved['arms']['S_STREAM_K1'])]
            expected = [[label, f"{a['median_seconds']:.3f}", f"{b['median_seconds']:.3f}",
                         f"{a['median_seconds'] / b['median_seconds']:.2f}×"] for label, a, b in stages]
            self.assertTrue(any('image/png' in o.get('data', {}) for o in cells[4]['outputs']))
        else:
            # Prior executed outputs stay intact until the new run is published.
            expected = [[label, f"{saved['arms'][arm]['median_seconds']:.3f}",
                         f"{saved['arms'][arm]['speedup_vs_original']:.2f}×"]
                        for arm, label in (('Original', 'Original'), ('S_STREAM_K1', 'Optimized'))]
        self.assertEqual(rows, expected)

    def test_models_use_public_https_sources_without_auth_parameters(self):
        tree = ast.parse((ROOT / 'acceleration/colab/setup_runtime.py').read_text())
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in {'MODEL_URL', 'ESM_URL'}:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(set(constants), {'MODEL_URL', 'ESM_URL'})
        expected_hosts = {'MODEL_URL': 'catpred.s3.us-east-1.amazonaws.com', 'ESM_URL': 'dl.fbaipublicfiles.com'}
        for name, url in constants.items():
            parsed = urlparse(url)
            self.assertEqual(parsed.scheme, 'https')
            self.assertEqual(parsed.hostname, expected_hosts[name])
            self.assertIsNone(parsed.username)
            self.assertIsNone(parsed.password)
            self.assertEqual(parsed.query, '')
            self.assertEqual(parsed.fragment, '')

    def test_code_cells_remain_valid_python_and_form_view(self):
        for index, cell in enumerate(notebook()['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'<cell-{index}>', 'exec')
                self.assertEqual(cell['metadata'].get('cellView'), 'form')


class CheckoutUpdateTests(unittest.TestCase):
    """Exercise the actual setup-cell Git branch using local temporary remotes."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='catpred-checkout-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / 'origin.git'
        self.seed = self.root / 'seed'
        self.checkout = self.root / 'CatPred-compare'
        self.git('init', '--bare', '--initial-branch=g4-inference', self.origin)
        self.git('init', '--initial-branch=g4-inference', self.seed)
        (self.seed / 'core.txt').write_text('old source')
        self.git('add', 'core.txt', cwd=self.seed)
        self.git('commit', '-m', 'Initial fixture', cwd=self.seed)
        self.git('remote', 'add', 'origin', self.origin, cwd=self.seed)
        self.git('push', '-u', 'origin', 'g4-inference', cwd=self.seed)
        self.git('clone', '--branch', 'g4-inference', self.origin, self.checkout)
        self.old_head = self.git('rev-parse', 'HEAD', cwd=self.checkout).strip()
        (self.seed / 'core.txt').write_text('current source')
        self.git('add', 'core.txt', cwd=self.seed)
        self.git('commit', '-m', 'Update fixture', cwd=self.seed)
        self.git('push', 'origin', 'g4-inference', cwd=self.seed)
        self.new_head = self.git('rev-parse', 'HEAD', cwd=self.seed).strip()

    def git(self, *args, cwd=None):
        result = subprocess.run(['git', '-c', 'user.name=Notebook Test', '-c', 'user.email=notebook@example.test',
                                 *[str(arg) for arg in args]], cwd=cwd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def run_checkout_branch(self, *, override=None, commit=''):
        setup_tree = ast.parse(code_cells()[0])
        branch = next(node for node in setup_tree.body if isinstance(node, ast.If)
                      and 'CHECKOUT.exists() and' in ast.unparse(node.test))
        commands = []
        def run_logged(arguments, log_path):
            commands.append([str(arg) for arg in arguments])
            result = subprocess.run([str(arg) for arg in arguments], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        work_root = self.root / 'existing-runtime'
        values = dict(CHECKOUT=self.checkout, checkout_override=override, commit=commit,
                      REPOSITORY=str(self.origin), Path=Path, subprocess=subprocess,
                      uuid=uuid, run_logged=run_logged, SETUP_LOG=self.root / 'setup.log',
                      WORK_ROOT=work_root)
        exec(compile(ast.Module(body=[branch], type_ignores=[]), '<checkout-update>', 'exec'), values)
        self.assertEqual(values['WORK_ROOT'], work_root)
        return values, commands

    def test_clean_default_checkout_fast_forwards_in_place(self):
        values, commands = self.run_checkout_branch()
        self.assertEqual(values['CHECKOUT'], self.checkout)
        self.assertEqual(values['ACTUAL_COMMIT'], self.new_head)
        self.assertEqual((self.checkout / 'core.txt').read_text(), 'current source')
        self.assertTrue(any('fetch' in command for command in commands))
        self.assertTrue(any('merge' in command and '--ff-only' in command for command in commands))
        self.assertFalse(any('clone' in command for command in commands))

    def test_dirty_checkout_keeps_edits_and_uses_a_new_sibling(self):
        (self.checkout / 'core.txt').write_text('user edit')
        (self.checkout / 'notes.txt').write_text('user notes')
        values, commands = self.run_checkout_branch()
        self.assertNotEqual(values['CHECKOUT'], self.checkout)
        self.assertEqual(values['CHECKOUT'].parent, self.checkout.parent)
        self.assertEqual(values['ACTUAL_COMMIT'], self.new_head)
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=self.checkout).strip(), self.old_head)
        self.assertEqual((self.checkout / 'core.txt').read_text(), 'user edit')
        self.assertEqual((self.checkout / 'notes.txt').read_text(), 'user notes')
        self.assertFalse(any('fetch' in command or 'merge' in command for command in commands))

    def test_detached_checkout_is_preserved(self):
        self.git('checkout', '--detach', 'HEAD', cwd=self.checkout)
        values, commands = self.run_checkout_branch()
        self.assertNotEqual(values['CHECKOUT'], self.checkout)
        self.assertEqual(values['ACTUAL_COMMIT'], self.new_head)
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=self.checkout).strip(), self.old_head)
        self.assertEqual(self.git('rev-parse', '--abbrev-ref', 'HEAD', cwd=self.checkout).strip(), 'HEAD')
        self.assertFalse(any('merge' in command for command in commands))

    def test_diverged_checkout_keeps_local_commit(self):
        (self.checkout / 'local.txt').write_text('local work')
        self.git('add', 'local.txt', cwd=self.checkout)
        self.git('commit', '-m', 'Local change', cwd=self.checkout)
        local_head = self.git('rev-parse', 'HEAD', cwd=self.checkout).strip()
        values, commands = self.run_checkout_branch()
        self.assertNotEqual(values['CHECKOUT'], self.checkout)
        self.assertEqual(values['ACTUAL_COMMIT'], self.new_head)
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=self.checkout).strip(), local_head)
        self.assertEqual((self.checkout / 'local.txt').read_text(), 'local work')
        self.assertFalse(any('merge' in command for command in commands))

    def test_explicit_override_keeps_its_original_checkout(self):
        values, commands = self.run_checkout_branch(override=str(self.checkout))
        self.assertEqual(values['CHECKOUT'], self.checkout)
        self.assertEqual(values['ACTUAL_COMMIT'], self.old_head)
        self.assertEqual(commands, [])

    def test_explicit_pin_keeps_its_original_checkout(self):
        values, commands = self.run_checkout_branch(commit=self.old_head)
        self.assertEqual(values['CHECKOUT'], self.checkout)
        self.assertEqual(values['ACTUAL_COMMIT'], self.old_head)
        self.assertEqual(commands, [])

    def test_repeat_controls_stay_hidden_with_one_input_copy(self):
        values = {}
        for source in code_cells():
            self.assertFalse(any('repeat' in line.lower() and '#@param' in line for line in source.splitlines()))
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {'repeat_count', 'repeats'}:
                            values[target.id] = ast.literal_eval(node.value)
        self.assertEqual(values, {'repeat_count': 1, 'repeats': 3})


if __name__ == '__main__':
    unittest.main()
