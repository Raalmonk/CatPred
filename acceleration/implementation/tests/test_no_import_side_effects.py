"""No torch/model imports, generic alias pollution or implicit acceleration."""
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from catpred_accel import _vendor
from catpred_accel.source import verify_files


class ImportTests(unittest.TestCase):
    def test_public_import_and_runtime_creation_do_not_import_torch(self):
        code="import sys; from catpred_accel import Runtime,RuntimeConfig; r=Runtime(RuntimeConfig()); assert 'torch' not in sys.modules; assert r.manifest()['engaged'] is False"
        subprocess.run([sys.executable,"-c",code],check=True)

    def test_temporary_legacy_aliases_restore_existing_and_missing_names(self):
        name="catpred_alias_test_name"
        self.assertNotIn(name,sys.modules)
        sentinel=types.ModuleType(name)
        with _vendor._aliases({name:sentinel}):self.assertIs(sys.modules[name],sentinel)
        self.assertNotIn(name,sys.modules)
        old=types.ModuleType(name);sys.modules[name]=old
        try:
            with self.assertRaises(ValueError):
                with _vendor._aliases({name:sentinel}):raise ValueError("injected")
            self.assertIs(sys.modules[name],old)
        finally:sys.modules.pop(name,None)

    def test_vendor_bytes_match_frozen_manifest(self):
        root=Path(_vendor.__file__).resolve().parent
        expected=json.loads((root/"accepted_sources.json").read_text())
        self.assertEqual(verify_files(root/"_accepted",expected),expected)


if __name__=="__main__":unittest.main(verbosity=2)
