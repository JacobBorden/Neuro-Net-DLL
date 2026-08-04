## 2026-08-04 - [Optimize Matrix Multiplication Loop Order]
**Learning:** [Loop interchange (i-j-k) is crucial for row-major matrix multiplication to ensure sequential memory access and avoid cache misses.]
**Action:** [Always use i-j-k loop order when iterating over row-major matrices in C++.]
