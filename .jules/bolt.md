## 2026-07-16 - Matrix Multiplication Optimization
**Learning:** The codebase's row-major Matrix operations suffer massive cache thrashing in O(n^3) nested loops unless properly interchanged to i-j-k.
**Action:** Always use i-j-k loop interchange for matrix multiplication.
