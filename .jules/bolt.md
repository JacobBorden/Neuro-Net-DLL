## 2024-05-24 - Cache-Friendly Matrix Multiplication Optimization
**Learning:** In row-major matrix implementations like `Matrix::Matrix<T>` in this codebase, the standard `i-k-j` or O(N^3) nested loops for matrix multiplication (where `k` is the inner-most loop iterating over columns of `c` and `b`) causes significant cache misses because it accesses memory non-sequentially.
**Action:** Always use an `i-j-k` loop interchange for matrix multiplication in C++ row-major matrix implementations to ensure sequential memory access and avoid significant cache misses.
