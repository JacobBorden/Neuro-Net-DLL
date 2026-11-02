## 2026-11-02 - Optimize ScaledDotProductAttention::forward In-Place Scaling & Mask Addition
**Learning:** In the `ScaledDotProductAttention::forward` step, the intermediate results of `scores * scale_factor` and `scaled_scores + mask` allocate temporary `Matrix` objects.
**Action:** Replace out-of-place scalar multiplication with in-place scaling (`scores *= scale_factor`) and replace out-of-place mask addition with in-place matrix addition (`scores += mask`) to reduce heap allocations during the forward pass of the transformer.
