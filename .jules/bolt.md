## 2026-09-06 - Matrix Multiplication Cache Optimization
**Learning:** For C++ row-major Matrix implementations, using a naive i-k-j loop order for matrix multiplication causes significant cache misses as elements in matrix B are accessed column-wise. Interchanging the loops to an i-j-k order allows sequential memory access for both matrix A and matrix B, resulting in dramatic performance improvements.
**Action:** Always prefer an i-j-k loop interchange for matrix multiplication or similar multi-dimensional operations over O(N^3) nested loops to ensure sequential memory access.
