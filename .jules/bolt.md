## 2026-08-08 - Matrix Multiplication Optimization
**Learning:** In C++ row-major matrix implementations, using an i-j-k loop interchange for matrix multiplication rather than the naive i-k-j significantly reduces cache misses by ensuring sequential memory access.
**Action:** Always use i-j-k loop interchange for matrix multiplication to ensure sequential memory access and avoid significant cache misses.
