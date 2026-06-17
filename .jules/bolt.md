## 2024-05-18 - [Matrix Multiplication Loop Interchange]
**Learning:** In the custom Matrix multiplication function, the standard nested loop order `i-k-j` (or similar inner loops that read columns sequentially) results in a non-sequential access pattern for the inner loop, suffering from significant memory overhead.
**Action:** Use a loop interchange pattern `i-j-k` instead to explicitly guarantee sequential cache-friendly access.
