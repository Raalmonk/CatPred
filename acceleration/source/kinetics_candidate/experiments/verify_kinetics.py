"""Independent K0/K1 raw evidence verification. Standard library only."""
from __future__ import annotations
import argparse, importlib, json, math, statistics, sys
from pathlib import Path
from kinetics_worker import WORKLOADS, COUNTS, sha, atomic, load_baseline

def check_summary(row):
    arm = row["kinetics_arm"]; summary = row["kinetics_candidate"]
    if arm not in ("K0", "K1"): raise AssertionError("Unknown kinetics arm")
    expected = math.ceil(row["rows"] / 50)
    assert summary["enabled"] is (arm == "K1")
    assert summary["engaged"] is (arm == "K1")
    assert summary["restored"] is True
    assert not summary.get("cleanup_conflicts") and not summary.get("failure")
    assert summary["optimized_calls"] == (expected * 10 if arm == "K1" else 0)
    if arm == "K1":
        member = summary["member_optimized_calls"]
        values = [member[str(i)] for i in range(10)] if isinstance(member, dict) else member
        assert values == [expected] * 10
        assert summary["validation_status"] == "experimental_unvalidated"
    assert row["arm"] == "S1" and row["actual_ESM_forward_batches"] == []
    assert row["feature_cache_unchanged"] is True
    if row["phase"] == "kinetics-gate":
        assert row["feature_preflight_checks"] == row["_features"][0]["sequence_count"]

def validate_index(index, phase, workload, arm):
    assert index["status"] == "COMPLETE", "Task is incomplete"
    contract = index["contract"]
    assert (contract["phase"], contract["workload"], contract["arm"]) == (phase, workload, arm)
    count = COUNTS[phase][workload]
    assert contract["repetitions"] == count and len(index["receipts"]) == count
    assert [r["repetition"] for r in index["receipts"]] == list(range(1, count + 1))
    assert len({r["path"] for r in index["receipts"]}) == count

def verify(root, phase, finite_ledger_root=None):
    root = Path(root).resolve(); load_baseline(root)
    original = importlib.import_module("verify_gpu_results")
    expected_sources = json.loads((Path(__file__).resolve().parents[1] / "references/ACCEPTED_SOURCE_FREEZE.json").read_text())["python_sources"]
    context_key = None
    v = original.Verifier(root, finite_ledger_root=finite_ledger_root or root / "results/kinetics/finite_checks")
    v.prepare_owner()
    phases = ["gate"] + ([] if phase == "gate" else [phase])
    groups = {}; source_identity = None; contracts = []
    for ph in phases:
        for workload in WORKLOADS:
            for arm in ("K0", "K1"):
                task = root / "results/kinetics/tasks" / (ph + "_" + workload + "_" + arm)
                index_path = task / "index.json"
                index = json.loads(index_path.read_text()); validate_index(index, ph, workload, arm)
                contracts.append({"path": str(index_path.relative_to(root)), "sha256": sha(index_path)})
                restoration = json.loads((task / "restoration.json").read_text())
                assert restoration["runtime"]["restored"] is True
                assert restoration["kinetics"]["restored"] is True
                rows = []
                for entry in index["receipts"]:
                    row = v.receipt(entry["path"], entry["sha256"])
                    assert row["phase"] == "kinetics-" + ph
                    assert row["workload"] == workload and row["kinetics_arm"] == arm
                    assert row["repetition"] == entry["repetition"]
                    assert row["profiled"] is (ph == "profile")
                    check_summary(row)
                    assert row["runtime_manifest"]["source_hashes"] == expected_sources
                    if context_key is None: context_key = row["runtime_manifest"]["context_key"]
                    assert row["runtime_manifest"]["context_key"] == context_key, "Mixed baseline runtime contexts"
                    source = row["kinetics_sources"]
                    assert source["scientific_identity"] == index["contract"]["scientific_identity"]
                    assert index["contract"]["baseline_helpers"] == json.loads(Path(__file__).with_name("BASELINE_HELPERS.json").read_text())
                    manifest_path = v.path(source["manifest"])
                    manifest = json.loads(manifest_path.read_text())
                    assert manifest == source["files"]
                    import hashlib
                    assert hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == source["identity"]
                    for name, digest in manifest.items():
                        p = (manifest_path.parent / name).resolve()
                        assert p.is_relative_to(manifest_path.parent.resolve())
                        assert sha(p) == digest
                    science = {name: digest for name, digest in manifest.items() if name.startswith("implementation/catpred_kinetics/")}
                    if arm == "K1":
                        own = {name.removeprefix("implementation/catpred_kinetics/"): digest for name, digest in science.items()}
                        assert row["kinetics_candidate"]["source_identity"]["candidate_sources"] == own
                        assert row["kinetics_candidate"]["accepted_certificate_applies_to_K1"] is False
                    assert hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest() == source["scientific_identity"]
                    if source_identity is None: source_identity = source["scientific_identity"]
                    assert source_identity == source["scientific_identity"], "Mixed candidate implementation"
                    rows.append(row)
                groups[(ph, workload, arm)] = rows
    for workload in WORKLOADS:
        reference = groups[("gate", workload, "K0")][0]
        for (ph, name, arm), rows in groups.items():
            if name != workload: continue
            for row in rows:
                if row is not reference: v.pair(reference, row, ph + ":" + name + ":" + arm + ":" + str(row["repetition"]))
    stats = []
    if phase == "warm":
        for workload in WORKLOADS:
            a = [x["seconds"] for x in groups[("warm", workload, "K0")]]
            b = [x["seconds"] for x in groups[("warm", workload, "K1")]]
            stats.append({"workload": workload, "n_per_arm": len(a), "K0_seconds": a, "K1_seconds": b,
                          "K0_median": statistics.median(a), "K1_median": statistics.median(b),
                          "K0_range": [min(a), max(a)], "K1_range": [min(b), max(b)],
                          "K0_process_count": len({x["environment"]["process_id"] for x in groups[("warm", workload, "K0")]}),
                          "K1_process_count": len({x["environment"]["process_id"] for x in groups[("warm", workload, "K1")]}),
                          "speedup": statistics.median(a) / statistics.median(b)})
    return {"status": "PASS", "phase": phase, "candidate_status": "experimental_only_no_default_certificate",
            "receipt_count": len(v.rows), "comparison_count": len(v.comparisons),
            "comparisons": v.comparisons, "statistics": stats, "task_indices": contracts,
            "candidate_source_identity": source_identity, "exact": True, "rtol_for_exact": 0, "atol_for_exact": 0}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--phase", choices=tuple(COUNTS), default="gate")
    p.add_argument("--output")
    p.add_argument("--finite-ledger-root")
    a = p.parse_args(argv)
    output = Path(a.output) if a.output else Path(a.root) / "results/kinetics" / (a.phase + "_verification.json")
    try: result = verify(a.root, a.phase, a.finite_ledger_root)
    except Exception as error:
        result = {"status": "FAIL", "phase": a.phase, "error": type(error).__name__ + ": " + str(error)}
    atomic(output, result)
    print(json.dumps({k: result[k] for k in ["status", "phase", "receipt_count", "comparison_count", "error"] if k in result}))
    return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
