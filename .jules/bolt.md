## 2026-11-04 - Optimize MNISTLoader
**Learning:** Reading large datasets like MNIST byte-by-byte (e.g., 47M+ `file.read` calls) causes massive I/O overhead.
**Action:** Always batch I/O operations into block reads when possible to minimize system call overhead.
