## 2026-08-28 - Matrix Multiplication Cache Optimization
**Learning:** The naive i-k-j loop order in Matrix multiplication causes significant cache misses, whereas an i-j-k loop interchange enables sequential memory access, drastically improving performance.
**Action:** Always use i-j-k loop interchange for row-major matrix multiplications.
