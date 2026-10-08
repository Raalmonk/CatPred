import copy, importlib.util, sys, unittest, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kinetics_worker import COUNTS, parser, recover_completed, sha
from verify_kinetics import check_summary, validate_index

class HarnessTests(unittest.TestCase):
    def row(self, arm="K1"):
        return {"rows": 2047, "phase": "kinetics-gate", "feature_preflight_checks": 1544, "_features": [{"sequence_count": 1544}], "kinetics_arm": arm, "arm": "S1", "actual_ESM_forward_batches": [], "feature_cache_unchanged": True,
            "kinetics_candidate": {"enabled": arm == "K1", "engaged": arm == "K1", "restored": True,
                "optimized_calls": 410 if arm == "K1" else 0,
                "member_optimized_calls": {str(i): 41 for i in range(10)},
                "validation_status": "experimental_unvalidated", "cleanup_conflicts": [], "failure": None}}
    def test_default_is_off_and_execution_opt_in(self):
        a=parser().parse_args(["--root","/tmp/unused","--workload","mixed_valid_2047"])
        self.assertEqual(a.mode,"off"); self.assertFalse(a.run)
    def test_k1_and_off_counter_contracts(self):
        check_summary(self.row()); check_summary(self.row("K0"))
    def test_missed_member_rejected(self):
        r=self.row(); r["kinetics_candidate"]["member_optimized_calls"]["9"]=40
        with self.assertRaises(AssertionError): check_summary(r)
    def test_silent_fallback_rejected(self):
        r=self.row(); r["kinetics_candidate"]["engaged"]=False
        with self.assertRaises(AssertionError): check_summary(r)
    def test_exception_cleanup_and_new_esm_rejected(self):
        for modify in [lambda r:r["kinetics_candidate"].update(restored=False),
                       lambda r:r["kinetics_candidate"].update(cleanup_conflicts=["forward"]),
                       lambda r:r.update(actual_ESM_forward_batches=[{}])]:
            r=self.row();modify(r)
            with self.assertRaises(AssertionError):check_summary(r)
    def test_incomplete_or_duplicate_science_samples_rejected(self):
        x={"status":"COMPLETE","contract":{"phase":"gate","workload":"mixed_valid_2047","arm":"K1","repetitions":2},
           "receipts":[{"repetition":1,"path":"a"},{"repetition":2,"path":"b"}]}
        validate_index(x,"gate","mixed_valid_2047","K1")
        for change in [lambda a:a.update(status="RUNNING"),lambda a:a["receipts"][1].update(path="a"),
                       lambda a:a["receipts"][1].update(repetition=1)]:
            y=copy.deepcopy(x);change(y)
            with self.assertRaises(AssertionError):validate_index(y,"gate","mixed_valid_2047","K1")
    def test_completed_sample_is_recovered_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);task=root/"task";out=task/"rep_1/attempt_1";out.mkdir(parents=True)
            contract={"phase":"gate","workload":"mixed_valid_2047","arm":"K1","repetitions":2,"scientific_identity":"a"}
            row=self.row();row.update(kinetics_sources={"scientific_identity":"a"},repetition=1,phase="kinetics-gate",workload="mixed_valid_2047")
            for kind in ("precision","precision_manifest","prediction","raw_prediction","consumed_features"):
                f=out/kind;f.write_bytes(b"retained science")
                row[kind+"_path"]=str(f.relative_to(root));row[kind+"_sha256"]=sha(f)
            (out/"receipt.json").write_text(json.dumps(row))
            index={"receipts":[],"contract":contract,"status":"RUNNING"}
            recover_completed(root,task,index,contract)
            self.assertEqual(len(index["receipts"]),1)
            self.assertEqual(index["receipts"][0]["repetition"],1)
            self.assertEqual(json.loads((task/"index.json").read_text()),index)
    def test_partial_teardown_receipt_is_not_silently_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);task=root/"task";out=task/"rep_1/attempt_1";out.mkdir(parents=True)
            (out/"receipt.json").write_text('{"seconds": 1.0}')
            with self.assertRaisesRegex(RuntimeError,"reconciliation"):
                recover_completed(root,task,{"receipts":[]},{"repetitions":2})
    def test_no_torch_imported(self): self.assertNotIn("torch",sys.modules)

if __name__ == "__main__": unittest.main()
