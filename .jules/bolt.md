## 2024-05-24 - Optimizing MNIST Loader

**Learning:** Batching `file.read` single-byte operations in `LoadImages` and `LoadLabels` to load the dataset in a single bulk read using a large `std::vector<unsigned char>` buffer drastically improves performance by avoiding millions of small system I/O calls. This is a common performance bottleneck in C++ programs that process large binaries byte-by-byte.

**Action:** Look for loops performing large numbers of small reads from file streams and convert them to bulk buffer reads where possible, then cast or convert the data.
