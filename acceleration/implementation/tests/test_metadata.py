"""Pure-standard-library policy checks; no models or devices are imported."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from catpred_accel import RuntimeConfig
from catpred_accel.capabilities import Capability, resolve, UnsupportedCapability
from catpred_accel.cache_keys import feature_key, compile_key
from catpred_accel.source import verify_files, SourceMismatch


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.config = RuntimeConfig()
        self.g4 = Capability("cuda", (12,0), "observed G4", True, True, software_key="recorded-stack")
        self.feature = dict(checkpoint_sha256="1"*64,feature_code_sha256="2"*64,
                            tokenizer_sha256="3"*64,backend="stock_esm2",software={"torch":"2.11.0"},
                            batch_context={"sequences":["AC"],"padded_tokens":4})

    def test_exact_default_and_no_fast_or_full(self):
        self.assertEqual(self.config.input_budget_bytes,2<<30)
        for kwargs in ({"numeric":"fast"},{"memory":"full"},{"input_budget_bytes":True}):
            with self.assertRaises(ValueError):RuntimeConfig(**kwargs)

    def test_capability_does_not_claim_validation(self):
        result=resolve(replace(self.config,allow_unvalidated=True),self.g4)
        self.assertTrue(result.engaged)
        self.assertEqual(result.validation_status,"experimental_unvalidated")
        self.assertEqual(result.hardware_profile,"g4_sm120")

    def test_unvalidated_context_does_not_engage_by_default(self):
        self.assertFalse(resolve(self.config,self.g4).engaged)
        self.assertTrue(resolve(self.config,replace(self.g4,validated_context=True)).engaged)
        self.assertFalse(resolve(replace(self.config,allow_unvalidated=True),replace(self.g4,software_key="unknown")).engaged)

    def test_h100_never_selected_from_g4_label(self):
        result=resolve(replace(self.config,backend="h100"),self.g4)
        self.assertFalse(result.engaged)
        with self.assertRaises(UnsupportedCapability):
            resolve(replace(self.config,backend="h100",fallback="raise"),self.g4)

    def test_missing_extension_or_source_reports_stock(self):
        for cap in (replace(self.g4,rust_available=False),replace(self.g4,source_compatible=False),Capability("cpu")):
            result=resolve(self.config,cap)
            self.assertEqual(result.selected_backend,"stock")
            self.assertTrue(result.reason)

    def test_off_does_not_engage(self):
        self.assertFalse(resolve(replace(self.config,numeric="off"),self.g4).engaged)

    def test_feature_namespace_separates_numeric_hardware_padding_and_weights(self):
        original=feature_key("AC",**self.feature)
        changes=({"numeric":"fast"},{"backend":"cuda_esm2"},{"checkpoint_sha256":"4"*64},
                 {"batch_context":{"sequences":["AC","ACDE"],"padded_tokens":6}},
                 {"software":{"torch":"new"}},{"feature_code_sha256":"5"*64})
        for change in changes:self.assertNotEqual(original,feature_key("AC",**dict(self.feature,**change)))
        self.assertEqual(original,feature_key("AC",**self.feature))
        with self.assertRaises(ValueError):feature_key("AC",**dict(self.feature,batch_context=None))

    def test_compile_namespace_separates_gpu_abi_and_cpu_features(self):
        kw=dict(operation="native",source_sha256="1"*64,gpu_arch="sm_90",software={"cuda":"13"},compiler_abi="cp313",cpu_features=["avx2"])
        original=compile_key(**kw)
        for change in ({"gpu_arch":"sm_120"},{"compiler_abi":"cp312"},{"cpu_features":["avx512"]}):
            self.assertNotEqual(original,compile_key(**dict(kw,**change)))

    def test_source_mismatch_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"module.py";path.write_text("value=1\n")
            expected={"module.py":hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(verify_files(directory,expected),expected)
            path.write_text("value=2\n")
            with self.assertRaises(SourceMismatch):verify_files(directory,expected)
            with self.assertRaises(SourceMismatch):verify_files(directory,{"../escape.py":"0"*64})


if __name__=="__main__":unittest.main(verbosity=2)
