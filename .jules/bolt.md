## 2024-05-24 - Parallelization Overhead
**Learning:** In `src/math/extended_matrix_ops.cpp`, computation-heavy functions like `softmax`, `layer_norm`, and `gelu` benefit significantly from OpenMP `#pragma omp parallel for` on outer loops. However, data movement operations like `split_matrix_by_cols` degrade in performance with OpenMP due to parallelization overhead.
**Action:** Use `#pragma omp parallel for` for expensive arithmetic operations (gelu, layer_norm, softmax), but avoid it for simple data moving functions.
