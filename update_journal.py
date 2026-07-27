import datetime

entry = f"""## {datetime.date.today().strftime('%Y-%m-%d')} - Optimized Matrix Multiplication Loop Order
**Learning:** In C++ row-major matrix implementations, the naive `i-k-j` or O(N^3) nested loops for matrix multiplication lead to significant cache misses because elements in the second matrix are not accessed sequentially.
**Action:** Always use an `i-j-k` loop interchange for matrix multiplication to ensure sequential memory access, drastically improving cache utilization and enabling SIMD auto-vectorization.
"""

with open(".jules/bolt.md", "a") as f:
    f.write("\n" + entry)
