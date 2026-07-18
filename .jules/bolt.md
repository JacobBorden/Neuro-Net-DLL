## 2026-07-18 - Optimize matrix multiplication cache locality
**Learning:** In C++ row-major memory layouts, an i-k-j loop nesting for matrix multiplication results in significant cache misses for the right-hand matrix because elements are accessed column-wise. Interchanging loops to i-j-k enables sequential memory access for both matrices, significantly improving performance, even when the outer loop is parallelized with OpenMP.
**Action:** Always use i-j-k loop ordering for operations on row-major matrix structures to maximize CPU cache utilization and minimize memory access latency.
