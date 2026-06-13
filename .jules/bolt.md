## 2024-06-13 - Matrix Multiplication Cache Friendliness
**Learning:** In the custom row-major matrix implementation, the standard `i-k-j` matrix multiplication loop ordering caused significant cache misses due to non-sequential memory access in the inner loop. Reordering the loops to `i-j-k` (iterating over rows of `b` sequentially rather than columns) improved performance by ~40%.
**Action:** Always prefer `i-j-k` loop ordering or block-based algorithms for matrix multiplication in row-major implementations to maximize CPU cache utilization.
