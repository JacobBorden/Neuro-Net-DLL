## 2024-09-15 - C++ OpenMP Parallel Matrix Multiplication Reordering Optimization

**Learning:** Matrix multiplication using `operator*` in `src/math/matrix.h` is currently using standard loops `for i` -> `for k` -> `for j` where `sum += A[i][j] * B[j][k]`. In C++, memory is laid out in row-major order. Consequently, accessing `B[j][k]` inside the `j` loop is highly inefficient because it causes a cache miss for every single element, since jumping by `j` strides across columns instead of linear memory access across the row. Reordering loops to `for i` -> `for j` -> `for k`, and caching `A[i][j]` outside the `k` loop, vastly improves spatial locality and cache hits. Benchmarking a 1500x1500 matrix multiplication goes from ~4175ms down to ~2112ms (nearly 2x speedup!).

**Action:** Loop reordering for matrix multiplication is a massive optimization here since we are implementing our own naive algorithms instead of using BLAS libraries.
