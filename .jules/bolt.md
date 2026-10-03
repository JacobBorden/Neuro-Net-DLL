## 2026-10-03 - Matrix Multiplication Cache Locality
**Learning:** Changing loop order from (i, k, j) to (i, j, k) significantly improves performance but requires explicitly zeroing the matrix and relaxing floating-point tests.
**Action:** When reordering floating-point math for performance, update EXPECT_FLOAT_EQ to EXPECT_NEAR in tests.
