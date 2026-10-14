## 2026-03-29 - OpenMP Parallelization for Extended Matrix Operations
**Learning:** Parallelizing row-wise matrix ops (`gelu`, `softmax`, `layer_norm`) on outer loops using `#pragma omp parallel for` yields ~6.7x-20x speedups for 512x512 matrices because rows are computationally independent.
**Action:** Apply OpenMP parallelization to outer loops of independent row-wise matrix calculations.
