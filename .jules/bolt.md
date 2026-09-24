## 2024-09-24 - Initial Setup
**Learning:** Initializing journal to track C++ optimizations in NeuroNet.
**Action:** Ready to find performance improvements.

## 2024-09-24 - Extended Matrix Ops Parallelization
**Learning:** Found that operations in `src/math/extended_matrix_ops.cpp` (specifically `gelu`, `softmax`, and `layer_norm`) are heavily computational and iterate over large loops. Applying OpenMP `#pragma omp parallel for` significantly speeds up these methods (roughly 3x improvement on a 2000x2000 matrix).
**Action:** When working on similar mathematical/iterative functions on large matrices, always consider outer loop parallelization via OpenMP. Ensure that OpenMP is linked in `CMakeLists.txt`.
