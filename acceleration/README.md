# Inference runtime

The [comparison notebook](../colab_compare.ipynb) includes saved G4 outputs from the public CatPred example. It checks full-precision outputs before timing both paths and caches weights in Google Drive.

`catpred_accel` packages the accepted S_STREAM input reuse and ESM loading changes. `catpred_kinetics` adds the optional K1 attention change. Both retain the original ten-model FP32 prediction path.

Install into an existing CatPred environment with its dependencies:

```sh
python -m pip install --no-deps ./acceleration/implementation \
  ./acceleration/source/kinetics_candidate/implementation
```

The measured setup uses Python 3.13, PyTorch 2.11.0+cu130 and the bundled Rust wheel on a Colab G4 with an AMD EPYC 9B45 CPU. The [replay instructions](REPRODUCE.md) set up that layout and install dependencies.

```python
from catpred_accel import Runtime, RuntimeConfig
from catpred_kinetics import KineticsAdapter

runtime = Runtime(RuntimeConfig())
for model in models:
    model.eval()
kinetics = KineticsAdapter(enabled=True, allow_unvalidated=True)
with runtime.activate(models, scalers):
    with kinetics.activate(runtime, models):
        with runtime.request():
            result = original_prediction_request()
```

`models`, the five scaler slots per member, and the request function come from the original service. Keep its `no_grad` inference context. K1 requires the verified exact 2 GiB runtime and the pinned model/PyTorch source; it is off by default and remains an explicit experimental option. The base runtime checks its certificate and falls back to stock when the environment does not match.

The source snapshot in `source/CatPred` keeps the replay tied to the measured upstream commit. Runtime and candidate source bytes match the tested delivery; [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records their hashes. Original model checkpoints and the benchmark CSVs must be supplied separately.

Run the tests without loading models:

```sh
python acceleration/check.py
python -m unittest discover -s acceleration/colab -p 'test_*.py'
```
