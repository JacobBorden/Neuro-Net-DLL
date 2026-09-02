## 2026-09-02 - [Optimize Matrix Multiplication Loop]
**Learning:** Matrix multiplication originally utilized an `i-k-j` loop sequence, leading to inefficient sequential memory access in row-major architectures (cache thrashing), which severely penalized large matrices performance.
**Action:** Enforced an `i-j-k` loop interchange for `operator*` in `src/math/matrix.h`. Always analyze nested loops and optimize for sequential memory access where possible in array-based operations.
