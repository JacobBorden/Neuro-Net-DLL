## 2026-09-01 - [Optimize Matrix Multiplication Loop]
**Learning:** In C++ row-major matrices, i-j-k loop ordering takes much better advantage of cache locality for matrix multiplication than an i-k-j loop, leading to substantial speedups.
**Action:** Always structure nested loops for matrix operations in an i-j-k order to ensure contiguous memory accesses when working with row-major multidimensional arrays.
