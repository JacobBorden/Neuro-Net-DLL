## 2026-07-07 - Matrix Multiplication Cache Locality
**Learning:** In a custom row-major Matrix implementation, the standard i-k-j loop order for matrix multiplication suffers from poor cache locality because it accesses the second matrix column-wise. Changing the loop order to i-j-k allows sequential access, dramatically improving performance (e.g., 51.4ms vs 106.3ms for 500x500).
**Action:** Use loop interchange (i-j-k order) for matrix operations on row-major structures to ensure cache-friendly memory access.
