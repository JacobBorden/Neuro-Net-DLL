## 2024-07-02 - [Matrix multiplication bug]
**Learning:** Relying on implicit value-initialization of memory objects (e.g. from `std::make_unique` in the `Matrix` constructor) to accumulate sums safely during optimized operations can be brittle, and will cause bugs if memory happens to be uninitialized or if the function is ever reused in a context where the target memory is overwritten.
**Action:** When implementing algorithms that accumulate into memory (like loop-interchanged matrix multiplication where `c.m_Data[i][k] += ...`), always explicitly initialize the target row values to zero before accumulation to guarantee correctness and safety.

## 2024-07-02 - [Bash Session Directory Persistence]
**Learning:** Each call to `run_in_bash_session` starts a fresh environment from the default repository root directory. State changes such as `cd build` do not carry over to subsequent calls.
**Action:** Always combine directory changes (e.g., `cd build && ...`) or use root-relative paths in single `run_in_bash_session` calls to avoid file access errors.
