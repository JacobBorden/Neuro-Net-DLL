## $(date +%Y-%m-%d) - Optimize Matrix Multiplication with i-j-k Loop Interchange
**Learning:** The previous implementation of matrix multiplication in `src/math/matrix.h` used an i-k-j loop order or initialized elements sequentially and accumulated them. Reordering the loops to i-j-k improves spatial locality for row-major matrices, leading to better cache utilization and significantly faster execution for large matrices.
**Action:** Always prefer sequential memory access patterns like i-j-k loop interchanges when iterating over row-major matrices in C++.
