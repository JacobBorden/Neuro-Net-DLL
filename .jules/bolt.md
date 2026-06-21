## 2024-06-21 - [Loop Interchange for Matrix Multiplication]
**Learning:** In C++ row-major matrix representations, an i-k-j loop order for multiplication causes non-sequential memory access on the innermost loop, resulting in cache misses.
**Action:** Always prefer an i-j-k loop order (loop interchange) for matrix multiplication to ensure cache-friendly sequential memory access.
