## 2024-05-24 - [Loop Interchange Cache Hit Rate]
**Learning:** Dense matrix operations in custom array structures lacking cache layout intelligence heavily benefit from loop interchange (e.g. from i-k-j to i-j-k for multiplication). Without it, iterating jumping over the column space of a row-major array causes a massive pipeline stall waiting for RAM.
**Action:** Always observe loop nest hierarchies when traversing 2D row-major matrices, ensuring the innermost loop iterates across contiguous memory.
