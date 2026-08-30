## 2026-08-30 - Optimize Matrix Multiplication Loop Order
**Learning:** In C++ row-major matrix implementations, the naive i-k-j loop for matrix multiplication results in significant cache misses due to non-sequential memory access on the second matrix.
**Action:** Use an i-j-k loop interchange for matrix multiplication to ensure sequential memory access and improve performance.
