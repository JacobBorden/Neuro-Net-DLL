## 2023-10-24 - Matrix Multiplication Loop Optimization
**Learning:** The default `(i, k, j)` loop order in `Matrix::operator*` is sub-optimal for caching because `j` is the innermost loop and iterates over rows of `b` (column-wise access for `b`). By changing the order to `(i, j, k)`, both `c` and `b` are accessed row-wise in the innermost loop.
**Action:** Always consider cache access patterns (row-major vs column-major) when implementing matrix math. Use `(i, j, k)` for better row-major cache locality.
