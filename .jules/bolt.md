## $(date +%Y-%m-%d) - Optimize Matrix Multiplication with i-j-k Loop Interchange
**Learning:** The previous implementation used an `i-k-j` loop structure for matrix multiplication which caused significant cache misses due to non-sequential memory access on the inner loop, especially for larger matrix sizes. C++ stores multidimensional arrays in row-major order.
**Action:** When implementing matrix multiplication in row-major systems, always use an `i-j-k` loop interchange to ensure sequential memory access for improved cache locality.
