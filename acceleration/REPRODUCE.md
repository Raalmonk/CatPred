# Replay on G4

Use an existing Linux Colab G4 with Python 3.13, PyTorch 2.11.0+cu130 and an AMD EPYC 9B45 CPU. The setup checks these assumptions before installing the accepted Rust wheel. It does not allocate a runtime.

From the fork checkout:

```sh
set -e
DELIVERY="$PWD/acceleration"
WORK=/content/catpred_accel_research
test ! -e "$WORK"
mkdir -p "$WORK"
cp -a "$DELIVERY/source/." "$WORK/"
cp -a "$DELIVERY/experiments" "$WORK/experiments"
cp -a "$DELIVERY/implementation/catpred_accel" "$WORK/catpred_accel"
python3 "$WORK/experiments/ops/remote_setup.py"
PY="$WORK/venv/bin/python"
```

Restore the original checkpoints to these locations. Their hashes are recorded in `source/prior_checkpoint_manifest.json` and `source/prior_esm_checkpoint_manifest.json`.

```text
/content/catpred_accel_research/data/pretrained/production/kcat/fold_0/model_0/model.pt
... model_1/model.pt through model_9/model.pt
/content/catpred_accel_research/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
/content/catpred_accel_research/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt
```

Copy the three archived benchmark CSVs (`mixed_valid_2047.csv`, `reuse_4096.csv`, `screening_16384.csv`) into `$WORK/inputs/`. Their expected hashes are in `source/prior_input_manifest.json`; sequence data is not included in this repository.

Generate the original ESM features once, then run the numerical checks before timing:

```sh
timeout 3700 "$PY" "$WORK/experiments/run_gpu_suite.py" \
  --root "$WORK" --stage prepare --max-new-tasks 1 --arms A \
  --prepare-timeout-seconds 3600 --timeout-seconds 1800
for PHASE in gate warm profile; do
  timeout 12000 "$PY" "$WORK/experiments/kinetics_stage.py" --phase "$PHASE"
done
timeout 1800 "$PY" "$WORK/experiments/kinetics_lifecycle.py" \
  --root "$WORK" --candidate-root "$WORK/kinetics_candidate" \
  --output-dir "$WORK/results/kinetics/lifecycle_01" \
  --workload mixed_valid_2047 --run
```

The scripts preserve completed samples and resume unfinished work. They fix batch=50, all ten FP32 models, original padding, scalers, uncertainty calculations and the 2 GiB input-cache budget. This budget covers cached inputs, not total GPU memory. K1 is enabled only in its comparison arm.

Raw member arrays, predictions, feature manifests and traces are saved under `$WORK/results/`. Recheck the saved bytes independently:

```sh
for PHASE in gate warm profile; do
  "$PY" "$WORK/kinetics_candidate/experiments/verify_kinetics.py" \
    --root "$WORK" --phase "$PHASE"
done
"$PY" "$WORK/experiments/analyze_kinetics.py" \
  --root "$WORK" \
  --verification "$WORK/results/kinetics/warm_verification.json" \
  --output "$WORK/results/kinetics/analysis.json"
```

Profiling is separate from the warm timings. These commands reproduce the K0/K1 comparison; the earlier original-Python results are documented in [benchmarks](benchmarks/README.md).
