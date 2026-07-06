## 2024-07-06 - Optimized Matrix Multiplication Loop Order
**Learning:** In C++, accessing elements sequentially in memory (row-major order) is critical for performance. The original `i-k-j` matrix multiplication inner loop caused constant cache misses. Reordering to `i-j-k` provided a massive speedup on 500x500 matrices due to improved cache locality.
**Action:** Always utilize an `i-j-k` (or equivalent cache-friendly) loop ordering for matrix and multidimensional array traversal tasks, particularly those exhibiting O(N^3) complexity.
