## 2026-07-12 - [Optimize Matrix Multiplication Cache Access]
**Learning:** [In C++ row-major matrix implementations, always use an i-j-k loop interchange for matrix multiplication rather than the naive i-k-j or O(N^3) nested loops to ensure sequential memory access and avoid significant cache misses.]
**Action:** [I will apply the i-j-k loop interchange when implementing matrix multiplication to maximize cache locality.]
