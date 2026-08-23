## $(date +%Y-%m-%d) - [Matrix Multiplication Performance Fix]
**Learning:** Matrix multiplication using naive nested loops (i-k-j) results in frequent cache misses, severely impacting performance for larger matrices. The i-k-j loop interchange ensures sequential memory access for the inner loop, minimizing cache misses.
**Action:** When implementing matrix multiplication, always use an i-j-k or i-k-j loop interchange for better cache utilization, and parallelize the outermost loop using OpenMP for further performance gains.
