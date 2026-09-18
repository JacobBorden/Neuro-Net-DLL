## 2026-09-18 - Matrix Operations OpenMP and Memory Optimizations
**Learning:** In `src/math/matrix.h`, performance of independent matrix operations (e.g., `Transpose`, `SigmoidMatrix`, `Split`, `Merge`) is significantly improved by applying OpenMP `#pragma omp parallel for` on outer loops and replacing element-wise row loops with `std::copy_n` for contiguous memory copying.
**Action:** Always consider using OpenMP for independent outer loop parallelization and `std::copy_n` for contiguous memory segments when optimizing math or matrix operations in this codebase.
