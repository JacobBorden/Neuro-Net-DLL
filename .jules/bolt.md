## 2026-10-08 - Precomputing Gaussian Distribution Constants Yields No Speedup
**Learning:** Precomputing constants in GaussianDistribution::pdf provided no measurable performance gain because floating-point std::exp evaluation dominates total execution time (~120ms per 10M calls).
**Action:** Focus on algorithm-level and memory access optimizations rather than hoisting micro-constants around dominant transcendental functions.
