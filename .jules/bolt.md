## 2026-07-21 - Matrix Multiplication Cache Optimization
**Learning:** The naive `i-k-j` nested loops for matrix multiplication causes significant CPU cache misses due to column-major access of the second matrix in our C++ row-major `Matrix` implementation.
**Action:** Always use an `i-j-k` loop interchange for matrix multiplication to ensure sequential memory access and drastically improve cache utilization.
