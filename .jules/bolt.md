## 2026-09-13 - Optimize Matrix Multiplication with Tiled Loop Iteration

**Learning:** Matrix multiplication using nested loops where the inner loop iterates over columns (j) instead of rows (k) causes poor memory cache usage when accessing elements of `b` (the second matrix), as it iterates over columns instead of rows sequentially. This degrades performance significantly due to cache misses.

**Action:** Reorder matrix multiplication loops from `i-k-j` to `i-j-k` to improve cache locality when calculating intermediate sums for row-by-row memory accesses. Furthermore, manually setting sums outside of iterations and writing to the matrix via `c.m_Data[i][k] = T(0)` prior to summation speeds up multiplication.
