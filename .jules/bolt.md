## 2026-07-26 - Matrix Multiplication Loop Interchange
**Learning:** Naive i-k-j matrix multiplication causes massive cache misses in row-major memory layouts because inner loop accesses stride across columns.
**Action:** Always use i-j-k loop interchange for matrix multiplication to ensure sequential memory access and improve cache utilization.
