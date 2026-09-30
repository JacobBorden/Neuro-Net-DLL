## 2026-09-30 - OpenMP for Extended Matrix Ops
**Learning:** In src/math/extended_matrix_ops.cpp, computation-heavy functions like gelu, layer_norm, and softmax benefit significantly from OpenMP #pragma omp parallel for on outer loops.
**Action:** Apply #pragma omp parallel for to outer loops of independent matrix operations for large performance gains.
