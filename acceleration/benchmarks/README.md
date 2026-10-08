# G4 results

Both experiments used an RTX PRO 6000 Blackwell G4, 48 visible CPU cores and 24 PyTorch threads. Each comparison ran on one runtime. Times are medians in seconds.

The earlier experiment compared original Python with the accepted S_STREAM runtime:

| Request | Rows | Original | S_STREAM | Speedup |
|---|---:|---:|---:|---:|
| Warm | 2,047 | 31.391 | 7.371 | 4.26x |
| Warm | 4,096 | 47.598 | 6.463 | 7.37x |
| Warm | 16,384 | 173.020 | 23.748 | 7.29x |
| Cold | 2,047 | 86.207 | 61.612 | 1.40x |

The latest experiment compared that runtime (K0) with the unused head average removed (K1):

| Rows | Repeats per version | K0 | K1 | Time saved |
|---|---:|---:|---:|---:|
| 2,047 | 5 | 7.334 | 6.857 | 6.5% |
| 4,096 | 5 | 6.467 | 6.302 | 2.6% |
| 16,384 | 3 | 23.694 | 23.169 | 2.2% |

Warm requests include input construction, all ten models, CSV output and input-cache release. Cold requests also include model loading and actual ESM feature generation. Installation and weight downloads are outside both timers. The 4,096/16,384 workloads reuse fewer proteins than the mixed 2,047 workload, so row counts alone do not determine runtime.

K1 preserved every member's raw output, final predictions and uncertainties bit for bit. Profiles showed 410, 820 and 3,280 unused averages removed, with unchanged matrix-multiplication and softmax counts. Peak allocated GPU memory was unchanged. Exception recovery and cleanup also passed.

[results.json](results.json) contains the repetitions, ranges, source identities and original receipt hashes. The original Python comparison and latest K1 comparison are separate runs; their speedups are not multiplied. Cold requests and H100 were not tested in the K1 experiment. Full prediction arrays and profiler traces remain in the archived experiment data.
