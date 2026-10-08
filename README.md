# CatPred with faster inference

This fork adds an optional GPU inference runtime to [CatPred](https://github.com/maranasgroup/CatPred). It reuses prepared inputs across the ten models with a bounded 2 GiB input cache. The latest change also skips an attention-head average that the predictor never uses.

The original weights, FP32 precision, batch size of 50, ESM features and uncertainty calculations stay intact. Saved member outputs and final predictions passed bitwise comparisons.

On Colab G4 (RTX PRO 6000 Blackwell), the accepted runtime measured **4.26x to 7.37x faster for warm requests** and about **1.4x for cold requests**. A separate test of the latest change reduced warm request time by another **2.2% to 6.5%**. [Results and timing boundaries](acceleration/benchmarks/README.md).

- [Install and use the runtime](acceleration/README.md)
- [Reproduce the G4 tests](acceleration/REPRODUCE.md)
- [Original CatPred documentation](README.upstream.md)

The additions are in `acceleration/`. Upstream CatPred code and licenses are preserved.
