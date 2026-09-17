## 2024-05-14 - Matrix Multiplication Loop Order Benchmark

**Learning:** Changing the matrix multiplication loop order in `src/math/matrix.h` from `(i, k, j)` to `(i, j, k)` was observed to *significantly* improve performance by a factor of 2.7x (from 738ms down to 268ms for 1000x1000 matrices). The theoretical caching benefits of the `(i, j, k)` order seem to hold true here even with OpenMP parallelizing the outer loop. Wait... no! The memory context explicitly states: "In `src/math/matrix.h`, the matrix multiplication loop order of (i, k, j) is intentionally optimized for the current architecture with OpenMP parallelization on the outer loop. Benchmarking shows that changing to a theoretically more cache-friendly (i, j, k) order actually causes a performance degradation. Do not change this loop order." My benchmark showed the opposite, but I must follow the memory instructions.

**Action:** I will not implement the loop reordering optimization as it is explicitly forbidden. I will look for other optimization opportunities.

## 2024-05-14 - Parallelizing Matrix Transpose & Sigmoid
**Learning:** Found multiple nested loops inside `src/math/matrix.h` (like `Transpose()`, `SigmoidMatrix()`) that iterate through large matrices without OpenMP `#pragma omp parallel for`. Testing showed that adding OpenMP to `Transpose` improved performance from ~65ms to ~27ms for a 2000x2000 matrix. Similarly, `SigmoidMatrix` went from ~43ms to ~24ms. I will implement these parallelizations since matrix multiplication already utilizes OpenMP correctly.

**Action:** I will add `#pragma omp parallel for` to independent loops inside `src/math/matrix.h` like `Transpose`, `SigmoidMatrix`, and merge operations, because these operations are element-wise or easily separable by row.
