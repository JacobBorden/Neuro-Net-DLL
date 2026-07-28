## 2026-07-28 - Matrix multiplication loop ordering optimization
**Learning:** In C++ row-major matrices, traversing loops using i-k-j naive ordering leads to poor cache performance. By restructuring matrix multiplication loops to i-j-k, memory access becomes contiguous and sequential, significantly reducing CPU cache misses.
**Action:** Always verify loop nesting order matches the memory storage format (i-j-k for row-major) when implementing O(N^3) multidimensional array traversals.
