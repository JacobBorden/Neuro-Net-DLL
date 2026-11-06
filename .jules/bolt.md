
## 2026-11-06 - Optimize ScaledDotProductAttention::forward
**Learning:** In `ScaledDotProductAttention::forward` (`src/transformer/attention.cpp`), scaling attention scores out-of-place (`scaled_scores = scores * scale_factor;`) and adding the mask out-of-place (`scaled_scores + mask`) incurs intermediate `Matrix` object allocations (`scaled_scores` and `scaled_scores + mask`).
**Action:** Use in-place operations (`scores *= scale_factor;` and `scores += mask;`) to eliminate these intermediate object allocations and speed up the attention forward pass without affecting semantics.
