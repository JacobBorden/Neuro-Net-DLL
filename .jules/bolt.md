## 2026-09-12 - Optimize Matrix Multiplication Loop Order
**Learning:** In C++, traversing matrices column-wise is cache-inefficient due to row-major storage. The naive loop order (i, k, j) caused cache misses.
**Action:** Reorder matrix multiplication loops to (i, j, k) to ensure row-wise memory access, dramatically improving cache hit rates and overall performance. Remember to relax GoogleTest EXPECT_FLOAT_EQ assertions to EXPECT_NEAR when loop orders change, as floating-point accumulation order affects precision.
