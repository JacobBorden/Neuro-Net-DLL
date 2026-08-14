## 2026-08-14 - Cache-friendly i-j-k Matrix Multiplication
**Learning:** In C++ row-major matrices, sequential memory access is crucial. A naive i-k-j loop in Matrix::operator* causes a high rate of cache misses when accessing columns of the second matrix.
**Action:** Implement loop interchange to group memory accesses sequentially. In future matrix operation implementations, ensure innermost loops iterate linearly across memory (e.g., swapping inner loops to compute results linearly across columns).
