## $(date +%Y-%m-%d) - [Optimize Matrix Multiplication for Cache Locality]
**Learning:** The naive `i-k-j` loop ordering in C++ row-major matrices creates severe cache trashing due to non-sequential memory access on the inner loop.
**Action:** Always use an `i-j-k` loop interchange for matrix multiplication or similar 2D convolution tasks to ensure contiguous memory access.
