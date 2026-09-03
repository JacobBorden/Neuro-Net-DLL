## 2026-09-03 - Optimize Matrix Multiplication loop
**Learning:** The naive matrix multiplication implementation with i-k-j loop iteration order results in significant cache misses and poor performance. Switching to an i-j-k loop interchange ensures sequential memory access for the innermost loops, drastically improving performance.
**Action:** Ensure all row-major matrix operations use an i-j-k loop interchange pattern to optimize cache performance.
