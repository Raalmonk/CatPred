# Saved Colab example

Executed on 9 October 2026 with one G4: RTX PRO 6000 Blackwell, AMD EPYC 9B45, 24 Torch threads, Python 3.13.15 and PyTorch 2.11.0+cu130.

The 14 public rows in [`demo/batch_kcat.csv`](../../../..//demo/batch_kcat.csv) were repeated eight times to make 112 rows. All paths evaluate every row with batch=50 and all ten FP32 models. Prediction caching and row deduplication are disabled.

Three warm repetitions per path, measured on the same runtime:

| Path | Median seconds | Speedup |
| --- | ---: | ---: |
| Original | 1.2023 | 1.00× |
| S_STREAM | 0.4280 | 2.81× |
| S_STREAM + K1 | 0.3926 | 3.06× |

Original ESM features took 3.849 seconds for nine unique sequences; model loading took 0.336 seconds. These preparation steps are outside the warm timings.

All 15 gate and timed requests passed independent comparisons of raw member outputs, predictions and uncertainties, including dtype, row order and exact bytes. All five notebook code cells completed, and 32 Colab helper tests passed. The G4 was released after collection.

- [Open the executed notebook](../../../colab_compare.ipynb)
- [Download CSVs, raw arrays and run receipts](results.zip)
- [Independent verification](verification.json)
