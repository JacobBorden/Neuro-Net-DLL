## 2026-09-11 - C++ Matrix Multiplication Cache Locality Optimization
**Learning:** In standard i-k-j matrix multiplication loops (`c[i][k] += a[i][j] * b[j][k]`), accessing `b[j][k]` traverses elements column-wise, which causes frequent cache misses for row-major matrices.
**Action:** Interchanged the inner loops to i-j-k, ensuring that both `a` and `b` are accessed sequentially by rows (`c[i][k] += a_ij * b[j][k]`), which significantly improved cache locality and reduced computation time.
