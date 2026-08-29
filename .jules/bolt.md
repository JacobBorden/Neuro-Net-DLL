## 2026-08-29 - Matrix Multiplication Loop Order Optimization
**Learning:** In C++ row-major matrix implementations, using an i-k-j naive nested loop for matrix multiplication causes significant cache misses.
**Action:** Always use an i-j-k loop interchange for matrix multiplication rather than the naive nested loops to ensure sequential memory access and avoid significant cache misses. Also, initialize the result matrix elements to zero prior to the accumulation loops inside the parallel block to maximize efficiency and thread safety.
