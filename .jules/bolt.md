## 2026-09-05 - [Optimize Matrix Multiplication Loop Interchange]
**Learning:** The previous implementation used an i-k-j loop order for matrix multiplication which causes cache thrashing by accessing memory non-sequentially. In row-major matrices, switching to an i-j-k loop order enables sequential memory access patterns, significantly improving L1 cache hit rate.
**Action:** Always verify loop nesting order corresponds to the underlying memory layout (e.g. row-major vs col-major) to avoid performance penalties from cache misses.
