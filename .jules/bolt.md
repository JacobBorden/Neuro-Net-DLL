## 2026-07-17 - [Optimize Matrix Multiplication loop order]
**Learning:** Naive i-k-j loop in matrix multiplication causes poor cache locality for the inner loop over column indices. Switching to an i-j-k loop order significantly improves sequential memory access, reducing matrix multiplication time by an order of magnitude.
**Action:** Always prefer i-j-k loop order in row-major matrix operations instead of nested loops that skip memory indices.
