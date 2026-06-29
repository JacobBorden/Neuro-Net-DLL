## 2024-05-24 - Matrix Multiplication Cache Locality Optimization
**Learning:** The custom `Matrix<T>` class in `src/math/matrix.h` uses row-major layout but the naive `i-k-j` nested loops for matrix multiplication caused highly non-sequential memory access (striding by `Cols`) for the inner loop, severely degrading CPU cache hit rates.
**Action:** When working with row-major matrices, always use `i-j-k` loop interchange to ensure contiguous memory access in the inner loop.
