## 2026-09-29 - OpenMP Parallelization for Matrix Operations
**Learning:** Element-wise and row/column-wise operations like `gelu`, `softmax`, and `layer_norm` in `src/math/extended_matrix_ops.cpp` are highly parallelizable but were running serially.
**Action:** Adding `#pragma omp parallel for` to outer loops speeds up these operations significantly without requiring complex refactoring, as long as aggregation variables are kept loop-local to avoid data races.
