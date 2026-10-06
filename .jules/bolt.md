## 2024-05-24 - Optimization of Conv2DLayer::Forward

**Learning:** `Conv2DLayer::Forward` performance is optimized by hoisting invariant spatial stride (`oh * stride_ - padding_`) and channel offsets out of the inner `kh`, `kw` kernel loops, and parallelizing outer output channel and height iterations using OpenMP (`#pragma omp parallel for collapse(2) schedule(static)`).
**Action:** Always hoist invariants out of tight inner loops and consider parallelizing outer loops when working with multi-dimensional iterations.
