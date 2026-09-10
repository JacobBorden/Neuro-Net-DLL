## 2024-09-10 - Cache-friendly Matrix Multiplication Optimization
**Learning:** The default `ijk` loop order for matrix multiplication is less efficient than the `ikj` loop order due to memory access patterns. The `ikj` order accesses elements sequentially in memory for both arrays B and C, which improves cache locality and significantly speeds up matrix multiplication.
**Action:** When implementing matrix multiplication or similar algorithms, use the `ikj` loop order to maximize cache efficiency.
