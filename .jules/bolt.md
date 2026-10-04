## 2026-10-04 - Conv2D Forward Pass Loop Hoisting & OpenMP Parallelization
**Learning:** In Conv2D layer forward passes, repeated indexing multiplications `(ic * spatial_size)` inside 2D filter/input sliding window loops cause significant overhead. Pre-calculating spatial stride offsets and parallelizing output channel iterations with OpenMP yields ~1.8x speedup.
**Action:** When working with 2D sliding windows or convolutions, hoist fixed strides/offsets out of inner kernel loops and apply OpenMP parallelization over outer channel/height loops.
