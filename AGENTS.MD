# Instructions for Agents Working on NeuroNet

## Scope
These instructions apply to the entire `NeuroNet` project repository.

## Build and Testing

The project is built using CMake and C++17. The primary build system relies on out-of-source builds.

To build the library and tests, follow these steps:
```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

To run tests after building:
```bash
cd build
ctest --output-on-failure
```

If you modify source code in `src/`, always run the test suite to ensure no regressions are introduced.

## Adding Features

*   **Header Files:** The project organizes headers and implementations separately inside `src/`. For example, `src/neural_network/neuronet.h` and `src/neural_network/neuronet.cpp`.
*   **Math Module:** Core matrix operations are contained in `src/math/matrix.h`. If making performance optimizations, verify them by building with `ENABLE_BENCHMARKING` and running the benchmark tests in `tests/test_benchmarks.cpp`.
*   **Testing:** New features should include Google Test units in the `tests/` directory.

## Code Formatting and Style
*   Follow the existing style regarding naming conventions (e.g., PascalCase for classes/methods in older files, snake_case in newer additions like the transformer module). Attempt to match the style of the file you are modifying.
*   Document new classes, structures, and functions using Doxygen-style comments `/** ... */`.

## Check before completing
*   Ensure that any new files are added to `CMakeLists.txt` or `tests/CMakeLists.txt` appropriately so they compile.
