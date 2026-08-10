## $(date +%Y-%m-%d) - Optimize Matrix Multiplication loop pattern
**Learning:** For row-major matrix representations like `Matrix::Matrix<T>`, iterating over columns in the innermost loop (the standard i-k-j pattern) causes significant cache misses during matrix multiplication because elements are accessed with large memory strides.
**Action:** Always implement an i-j-k loop interchange for matrix multiplication where possible to ensure sequential (stride-1) memory access along rows for both the result matrix and the right-hand operand.
