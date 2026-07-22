## 2026-07-22 - [Optimize Matrix Multiplication Loop Order]
**Learning:** The naive i-k-j loop in matrix multiplication causes severe cache misses because it iterates over the columns of the right-hand matrix sequentially in the inner loop, which are not contiguous in row-major memory layouts.
**Action:** Always use i-j-k loop interchange for matrix multiplications in C++ row-major implementations to ensure sequential memory access and improve cache utilization.
