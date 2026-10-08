"""Finite future Linux/CUDA worker; never allocates a runtime or installs software."""
from __future__ import annotations
import argparse, hashlib, importlib, json, math, os, shutil, sys
from pathlib import Path

WORKLOADS = {"mixed_valid_2047": 2047, "reuse_4096": 4096, "screening_16384": 16384}
COUNTS = {"gate": {k: 2 for k in WORKLOADS},
          "warm": {"mixed_valid_2047": 5, "reuse_4096": 5, "screening_16384": 3},
          "profile": {k: 1 for k in WORKLOADS}}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)

def load_baseline(root):
    root = Path(root).resolve()
    pinned = json.loads(Path(__file__).with_name("BASELINE_HELPERS.json").read_text())
    for name, digest in pinned.items():
        if sha(root / "experiments" / name) != digest:
            raise RuntimeError("Baseline helper changed: " + name)
    sys.path.insert(0, str(root / "experiments"))
    return importlib.import_module("gpu_request")

def candidate_sources(root):
    candidate = Path(__file__).resolve().parents[1]
    sources = sorted(p for p in (candidate / "implementation/catpred_kinetics").rglob("*") if p.is_file() and p.suffix in (".py", ".json"))
    sources += [Path(__file__).resolve(), Path(__file__).with_name("BASELINE_HELPERS.json")]
    entries = {str(p.relative_to(candidate)): sha(p) for p in sources}
    key = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    destination = root / "results/kinetics/source_history" / key
    for p in sources:
        target = destination / p.relative_to(candidate)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copy2(p, target)
        if sha(target) != entries[str(p.relative_to(candidate))]:
            raise RuntimeError("Candidate source snapshot mismatch")
    atomic(destination / "manifest.json", entries)
    scientific = {name: value for name, value in entries.items() if name.startswith("implementation/catpred_kinetics/")}
    science_key = hashlib.sha256(json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    return {"identity": key, "scientific_identity": science_key,
            "manifest": str((destination / "manifest.json").relative_to(root)), "files": entries}

def recover_completed(root, task, index, contract):
    """Adopt a fully saved receipt after an index-write failure; never remeasure it."""
    root, task = Path(root).resolve(), Path(task).resolve()
    for rep in range(len(index["receipts"]) + 1, contract["repetitions"] + 1):
        valid = []
        for path in sorted((task / ("rep_" + str(rep))).glob("attempt_*/receipt.json")):
            row = json.loads(path.read_text())
            if "kinetics_candidate" not in row:
                # The original scientific worker finished but K1 teardown evidence
                # was not committed. Preserve and require reconciliation, not rerun.
                raise RuntimeError("Completed baseline-format receipt requires local reconciliation: " + str(path))
            if (row.get("kinetics_sources", {}).get("scientific_identity") != contract["scientific_identity"] or
                row.get("kinetics_arm") != contract["arm"] or
                row.get("repetition") != rep or row.get("workload") != contract["workload"] or
                row.get("phase") != "kinetics-" + contract["phase"]):
                raise RuntimeError("Unindexed completed receipt contract mismatch: " + str(path))
            for kind in ("precision", "precision_manifest", "prediction", "raw_prediction", "consumed_features"):
                target = (root / row[kind + "_path"]).resolve()
                if not target.is_relative_to(root) or sha(target) != row[kind + "_sha256"]:
                    raise RuntimeError("Unindexed completed evidence mismatch: " + str(path))
            summary = row["kinetics_candidate"]
            if not summary.get("restored") or summary.get("cleanup_conflicts") or summary.get("failure"):
                raise RuntimeError("Completed receipt needs cleanup reconciliation: " + str(path))
            valid.append(path)
        if not valid: break
        if len(valid) != 1: raise RuntimeError("Multiple completed receipts for one repetition; reconcile saved state")
        path = valid[0]
        index["receipts"].append({"repetition": rep, "path": str(path.relative_to(root)), "sha256": sha(path)})
        atomic(task / "index.json", index)
    return index

def make_runtime(base, mode, source_record):
    from catpred_kinetics import KineticsAdapter
    class KineticsRuntime(base.Runtime):
        def __init__(self, *args, **kwargs):
            self.kinetics = self.kinetics_manager = None
            super().__init__(*args, **kwargs)
        def close_kinetics(self):
            if self.kinetics_manager is not None:
                manager, self.kinetics_manager = self.kinetics_manager, None
                manager.__exit__(None, None, None)
        def close_runtime(self):
            try:
                self.close_kinetics()
            finally:
                super().close_runtime()
        def configure(self, arm):
            # Apply the original inference eval state equally to K0 and K1
            # before the candidate's activation guard; predict still calls eval.
            for model in self.models: model.eval()
            super().configure(arm)
            self.kinetics = KineticsAdapter(enabled=mode == "head_mean", allow_unvalidated=mode == "head_mean")
            manager = self.kinetics.activate(self.s1_runtime, self.models)
            manager.__enter__(); self.kinetics_manager = manager
        def run(self, name, arm, dest, phase, rep, **kwargs):
            original_profiler = self.torch.profiler.profile
            profiled = kwargs.get("profile", False)
            def profile_with_shapes(*args, **options):
                options["record_shapes"] = True
                return original_profiler(*args, **options)
            if profiled: self.torch.profiler.profile = profile_with_shapes
            try:
                row = super().run(name, arm, dest, phase, rep, **kwargs)
            finally:
                if profiled: self.torch.profiler.profile = original_profiler
                self.close_kinetics()
            summary = self.kinetics.summary()
            row["kinetics_arm"] = "K1" if mode == "head_mean" else "K0"
            row["kinetics_candidate"] = summary
            row["kinetics_sources"] = source_record
            row["kinetics_profile_record_shapes"] = bool(profiled)
            atomic(Path(dest) / "receipt.json", row)
            return row
    return KineticsRuntime

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, help="Prepared Linux baseline WORK directory")
    p.add_argument("--mode", choices=("off", "head_mean"), default="off")
    p.add_argument("--phase", choices=tuple(COUNTS), default="gate")
    p.add_argument("--workload", choices=tuple(WORKLOADS), required=True)
    p.add_argument("--run", action="store_true", help="Explicitly execute on existing Linux CUDA runtime")
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    if not args.run: raise SystemExit("No execution requested; use --run on an existing Linux CUDA runtime")
    if sys.platform != "linux": raise SystemExit("Model execution is Linux/CUDA only; no Mac execution")
    root = Path(args.root).resolve()
    # No network, allocation, dependency installation, or native compilation.
    if args.phase != "gate":
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).with_name("verify_kinetics.py")),
                        "--root", str(root), "--phase", "gate"], check=True, timeout=600)
    base = load_baseline(root)
    candidate = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(candidate / "implementation"))
    sources = candidate_sources(root)
    arm = "K1" if args.mode == "head_mean" else "K0"
    task = root / "results/kinetics/tasks" / (args.phase + "_" + args.workload + "_" + arm)
    task.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (task / ".worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    count = COUNTS[args.phase][args.workload]
    contract = {"schema": 1, "phase": args.phase, "workload": args.workload,
                "arm": arm, "repetitions": count, "scientific_identity": sources["scientific_identity"],
                "baseline_helpers": json.loads(Path(__file__).with_name("BASELINE_HELPERS.json").read_text()),
                "measurement_boundary": "frozen gpu_request.run; configure outside; complete request and cache release inside"}
    contract_path = task / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError("Task contract changed; preserve the existing task")
    atomic(contract_path, contract)
    index_path = task / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"status": "RUNNING", "contract": contract, "receipts": []}
    if index.get("contract") != contract: raise RuntimeError("Index contract changed")
    for i, entry in enumerate(index["receipts"], 1):
        if entry["repetition"] != i or sha(root / entry["path"]) != entry["sha256"]:
            raise RuntimeError("Saved completed receipt mismatch")
    index = recover_completed(root, task, index, contract)
    if len(index["receipts"]) == count:
        # A lost process-level restoration record is different from a lost
        # sample. Never repeat the samples to manufacture that record.
        if not (task / "restoration.json").exists():
            raise RuntimeError("All samples saved; process-restoration receipt needs reconciliation, no rerun")
        index["status"] = "COMPLETE"; atomic(index_path, index)
        print("ALREADY_COMPLETE", task.name); return 0
    if index.get("status") == "COMPLETE": raise RuntimeError("Incomplete completed task")
    Runtime = make_runtime(base, args.mode, sources)
    rt = Runtime(root, root / "results/experiments/shared_feature_cache")
    owner = base.read(root / "results/experiments/shared_features.json")
    if owner["manifest"]["identity"] != rt.feature_identity(): raise RuntimeError("Shared ESM context mismatch")
    for entry in owner["cache_files"]:
        if sha(root / entry["path"]) != entry["sha256"]: raise RuntimeError("Shared ESM cache changed")
    def missing(*args, **kwargs): raise RuntimeError("Unexpected ESM generation in kinetics comparison")
    rt.esm._run_esm_batch = missing; rt.esm.get_single_esm_repr = missing
    if args.phase == "gate":
        rt.expected_features = {e["sequence_sha256"]: e for e in owner["manifest"]["features"]}
    rt.preload(args.workload)
    try:
        if args.phase in ("warm", "profile"):
            # Warmup is excluded and separately named even after a mechanical resume.
            warmups = task / "warmups"; warmups.mkdir(exist_ok=True)
            warm = warmups / ("warmup_" + str(len(list(warmups.iterdir())) + 1))
            rt.run(args.workload, "S1", warm, "warmup", 0)
        for rep in range(len(index["receipts"]) + 1, count + 1):
            parent = task / ("rep_" + str(rep)); parent.mkdir(exist_ok=True)
            # Preserve partial attempts; only completed, indexed scientific samples are skipped.
            dest = parent / ("attempt_" + str(len(list(parent.iterdir())) + 1))
            rt.run(args.workload, "S1", dest, "kinetics-" + args.phase, rep,
                   diagnostic=False, cold=False, profile=args.phase == "profile")
            receipt = dest / "receipt.json"
            index["receipts"].append({"repetition": rep, "path": str(receipt.relative_to(root)), "sha256": sha(receipt)})
            atomic(index_path, index)
            print("REQUEST_COMPLETE", task.name, rep, flush=True)
    finally:
        rt.close_runtime()
        atomic(task / "restoration.json", {"runtime": rt.s1_runtime.manifest() if rt.s1_runtime else None,
                                          "kinetics": rt.kinetics.summary() if rt.kinetics else None})
    index["status"] = "COMPLETE"; atomic(index_path, index)
    return 0

if __name__ == "__main__": raise SystemExit(main())
