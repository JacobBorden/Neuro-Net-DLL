## 2024-05-18 - Avoid unnecessary deep memory allocations and copies
**Learning:** Activation functions in NeuroNetLayer modify the input matrix in-place and return void. This design avoids unnecessary memory allocations during matrix operations and the forward pass.
**Action:** Apply this optimization to the .cpp file and update the .h file.
