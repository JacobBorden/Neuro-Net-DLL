## 2026-08-20 - Matrix multiplication cache-friendly optimization
**Learning:** Using a naive i-k-j nested loop for matrix multiplication results in significant cache misses due to non-sequential memory access.
**Action:** Always use i-j-k loop interchange for row-major matrix implementations to ensure sequential memory access and maximize performance.
