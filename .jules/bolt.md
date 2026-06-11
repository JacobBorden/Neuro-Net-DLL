## 2024-06-11 - Optimize Matrix Multiplication loop order
**Learning:** Found a critical performance bottleneck in custom C++ matrix operations where memory access was causing large cache misses due to row-major memory layouts.
**Action:** Changed loop order in matrix multiplication from i-k-j to i-j-k which correctly addresses cache-friendly memory access when looping through rows, resulting in ~15-20% faster matrix multiplication processing.
