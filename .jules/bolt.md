## 2026-07-19 - Optimize Matrix Multiplication Loop Order
**Learning:** For row-major matrices, an i-k-j loop order accesses the right-hand matrix columns sequentially in the inner loop, causing significant CPU cache misses. Changing to an i-j-k loop order accesses both matrices sequentially in memory, vastly improving cache utilization and enabling SIMD vectorization.
**Action:** Always use an i-j-k loop interchange for matrix multiplication in C++ row-major implementations to ensure sequential memory access and avoid cache misses.
