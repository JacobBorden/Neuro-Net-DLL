## 2026-10-20 - [Optimize PositionalEncoding forward pass]
**Learning:** In C++ deep learning logic, relying on overloaded operators (like `Matrix::operator+`) for element-wise addition along with allocating temporary sub-matrices (`relevant_pe`) causes heavy memory allocation and traversal overhead.
**Action:** Always prefer direct element-wise computation (e.g., `output[i][j] = input[i][j] + pe_table_[i][j]`) to avoid redundant matrix allocation and extra looping passes.
