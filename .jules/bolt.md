## 2026-10-05 - Optimize Conv2DLayer::Forward with Loop Hoisting and OpenMP
**Learning:** In `src/neural_network/conv2d_layer.cpp`, `Conv2DLayer::Forward` had significant CPU overhead due to re-calculating invariant spatial strides and channel offsets within the innermost loops, alongside missing OpenMP parallelization on the outer loops.
**Action:** Always hoist invariant arithmetic out of inner loops and apply `#pragma omp parallel for collapse(2)` on independent outer loops (like output channel and height) for tensor operations to maximize CPU cache locality and thread utilization.
