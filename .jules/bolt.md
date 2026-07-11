## 2024-05-20 - [Cache-friendly loop interchange for matrix multiplication]
**Learning:** In C++ row-major custom Matrix implementations, naively iterating columns in the inner loop during matrix multiplication causes significant cache misses, which heavily impacts performance (e.g. 79ms for 500x500).
**Action:** Always use an `i-j-k` loop interchange (iterating `k` in the innermost loop) to ensure sequential memory access and utilize the CPU cache efficiently.
