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
        self.assertIn('repeat_count = 8', source)
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
            self.assertEqual(actual, original_rows * 8)
            self.assertEqual(len(actual), 112)

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
            self.assertEqual(command[command.index('--k1') + 1], 'auto')

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


if __name__ == '__main__':
    unittest.main()
