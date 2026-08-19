## 2026-08-19 - Matrix Multiplication Loop Interchange
**Learning:** In C++ row-major matrix implementations, the naive i-k-j loop ordering for matrix multiplication causes significant cache misses when traversing the right-hand matrix by column. Loop interchange to i-j-k ensures sequential memory access for both matrices.
**Action:** Always use i-j-k loop interchange for matrix multiplication in row-major implementations to optimize cache line utilization.
