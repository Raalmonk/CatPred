# Saved Colab example

Run on 9 October 2026 in a fresh G4 session: RTX PRO 6000 Blackwell, AMD EPYC 9B45, Python 3.13.15 and PyTorch 2.11.0+cu130.

The notebook cloned the public repository, installed its dependencies and downloaded all models from public sources. It used no Drive account or preloaded files. All five code cells finished in 66.9 seconds, including setup and downloads.

The example repeats the 14 public rows in [`demo/batch_kcat.csv`](../../../demo/batch_kcat.csv) eight times, giving 112 rows. Both versions evaluate every row with batch size 50 and all ten FP32 models.

Three warm runs per version, measured on the same G4:

| Version | Median seconds | Speedup |
| --- | ---: | ---: |
| Original | 1.2033 | 1.00× |
| Optimized | 0.3933 | 3.06× |

ESM feature generation took 3.787 seconds for nine unique sequences. Model loading took 0.328 seconds. Both are measured separately from the warm requests.

All ten validation and timed requests matched exactly in raw model outputs, predictions and uncertainties. All 54 tests passed. The G4 was released after collecting the results.

- [Open the executed notebook](../../../colab_compare.ipynb)
- [Download CSVs, raw arrays and run receipts](results.zip)
- [Independent verification](verification.json)
- [Public source](source_public.json), [empty starting cache](fresh_defaults.json) and [downloaded model hashes](public_asset_downloads.json)
