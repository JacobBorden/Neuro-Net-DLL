## 2024-07-24 - [Loop Interchange for Matrix Multiplication]
**Learning:** In C++ row-major matrix implementations (like `src/math/matrix.h`), the default O(N^3) nested loop for matrix multiplication (i-k-j) results in significant cache misses for the inner loop access `b.m_Data[j][k]` since the column index changes but the row index stays the same.
**Action:** Always use an `i-j-k` loop interchange for matrix multiplication where possible to ensure sequential memory access for both the result matrix and the operand matrices.
