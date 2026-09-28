## 2024-05-24 - OpenMP Optimization
**Learning:** In src/math/extended_matrix_ops.cpp, computation-heavy functions benefit significantly from OpenMP parallelization on outer loops.
**Action:** Apply #pragma omp parallel for to outer loops for independent matrix operations.
