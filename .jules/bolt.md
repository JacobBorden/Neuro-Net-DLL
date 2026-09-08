## 2024-05-24 - Cache Locality in Matrix Multiplication
**Learning:** The existing matrix multiplication loop order (i-k-j) causes poor spatial cache locality when accessing elements of matrix B, leading to significant performance degradation for large matrices.
**Action:** Always optimize matrix multiplication by changing the loop order to i-j-k, ensuring that the innermost loop accesses matrix B elements sequentially in memory.
