## 2026-08-16 - Optimize Matrix Multiplication loop interchange
**Learning:** Matrix multiplication in C++ row-major format requires i-j-k loop ordering instead of i-k-j. Naive i-k-j leads to inefficient cache usage, causing severe performance degradation for large matrices due to non-sequential memory access on the inner loop.
**Action:** Always check the memory layout and loop ordering for multi-dimensional array operations. For row-major data, ensure inner loops traverse contiguous memory (rows).
