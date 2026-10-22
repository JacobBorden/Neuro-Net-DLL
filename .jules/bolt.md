## 2024-05-19 - Parallelizing MathUtils operations
**Learning:** Adding OpenMP `#pragma omp parallel for` to outer loops in `gelu`, `layer_norm`, and `softmax` inside `extended_matrix_ops.cpp` significantly reduces execution time from ~2409ms down to ~989ms on 2000x2000 matrices.
**Action:** Always apply outer loop parallelization using OpenMP for independent matrix operations when iterating over rows or columns if computationally heavy.
