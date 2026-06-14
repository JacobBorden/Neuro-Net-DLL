## 2024-06-14 - Matrix Multiplication Loop Interchange
**Learning:** Reordering the inner loops of matrix multiplication from i-k-j to i-j-k significantly improves cache locality because matrices are stored in row-major order.
**Action:** When working with row-major matrices, always use i-j-k loop order for matrix multiplication to avoid the O(N^2) memory overhead of transposing the matrix.
