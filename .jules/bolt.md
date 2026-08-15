## 2026-08-15 - Matrix Multiplication Cache Locality
**Learning:** Using an i-k-j naive matrix multiplication loop structure on row-major matrices results in poor cache utilization.
**Action:** Implement an i-j-k loop interchange for O(N^3) matrix operations to ensure sequential memory access and avoid significant cache misses.
