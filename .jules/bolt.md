## 2026-08-27 - Matrix Multiplication Cache Optimization
**Learning:** Naive i-k-j matrix multiplication causes severe cache misses for row-major layouts in C++ (like std::vector of std::vector).
**Action:** When implementing matrix multiplication in C++, always use an i-j-k loop interchange to ensure sequential memory access and maximize CPU cache hits.
