## $(date +%Y-%m-%d) - Optimize Matrix Multiplication with i-j-k Loop Interchange
**Learning:** In C++ row-major matrices, the naive i-k-j loop for matrix multiplication (where the innermost loop iterates over rows of the right-hand matrix) causes severe cache thrashing, as memory is accessed non-sequentially.
**Action:** Always use the i-j-k loop interchange pattern for matrix multiplication to ensure sequential memory access (spatial locality) for both the result matrix and the right-hand matrix.
