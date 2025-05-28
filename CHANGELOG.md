# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-05-23

### Added
- **NeuroNet Model Serialization & Deserialization:**
    - Implemented `NeuroNet::save_model()` to export models (weights, architecture, parameters) to a human-readable JSON format.
    - Implemented `NeuroNet::load_model()` to import models from the JSON format.
    - Added `NeuroNetLayer::get_activation_type()` to facilitate serialization.
    - Comprehensive unit tests for saving and loading models, ensuring data integrity and error handling for invalid files/formats.
    - Documented the serialization process, JSON format, and usage examples in `README.md`.
- **JSON Library Unit Tests:**
    - Created a new test suite (`tests/test_json.cpp`) for the JSON library (`src/utilities/json/`).
    - Added extensive tests covering:
        - JSON parsing of various data types, arrays, objects, and nested structures.
        - JSON serialization of `Json::Value` objects to strings.
        - `Json::Value` API for data type handling (type checking, value retrieval, member access, array manipulation).
        - JSON parser error handling for malformed or invalid input.
    - Integrated these tests into the existing CMake build system (within the `test_neuronet` executable).
- Pluggable activation functions (None, ReLU, LeakyReLU, ELU, Softmax) for `NeuroNetLayer`. Layers now apply the selected activation function after the linear transformation.
- `ActivationFunctionType` enum to specify activation types.
- `NeuroNetLayer::SetActivationFunction()` method.
- Private helper methods in `NeuroNetLayer` for applying each activation function.
- Performance benchmarking capabilities:
    - Timer utility (`src/utilities/timer.h`) for measuring execution time.
    - Instrumentation for core operations: matrix multiplication, neural network forward pass, and genetic algorithm steps (crossover, mutation, selection, evaluation).
    - Benchmark scenarios (`tests/test_benchmarks.cpp`) to exercise instrumented operations.
    - `ENABLE_BENCHMARKING` macro to control timing code compilation.
- Optimization:
    - Parallelized matrix multiplication (`Matrix<T>::operator*`) using OpenMP for improved performance on multi-core systems.

### Changed
- Replaced `jsoncpp` library with a custom internal JSON library (`src/utilities/json/`) for handling JSON data. This affects model saving/loading, test suites, and CMake configuration. The custom library (`json.hpp`, `json.cpp`) is now compiled directly into the main `neuronet` library.
- `NeuroNetLayer::CalculateOutput()` now incorporates the selected activation function.
- Default constructor `NeuroNetLayer()` initializes with `ActivationFunctionType::None`.
- Updated Doxygen comments in `neuronet.h` and `neuronet.cpp` for new activation function features.
- Updated `README.md` with a new section explaining activation functions and providing usage examples, and updated dependency information to reflect the custom JSON library.
- Updated Doxygen comments in `src/utilities/json/json.hpp` to fully document the custom JSON library's API.

## [Previous Version - e.g., 0.2.0] - 2025-05-23 (Date of previous changes if known, or adjust as needed)

### Added
- Created `CHANGELOG.md` to track changes.

### Changed
- Modularized the codebase for improved readability and maintainability:
    - Matrix library (`Matrix` namespace) moved to `src/math/matrix.h`.
    - Neural Network components (`NeuroNet` namespace: `NeuroNet`, `NeuroNetLayer`) moved to `src/neural_network/`.
    - Genetic Algorithm (`GeneticAlgorithm`) moved to `src/optimization/` and its namespace changed to `Optimization`.
- Updated Visual Studio project files (`src/Neuro Net DLL.vcxproj`, `src/Neuro Net DLL.vcxproj.filters`) to reflect new paths for `neuronet.h` and `neuronet.cpp`.
- Identified that a `ProjectReference` in `src/Neuro Net DLL.vcxproj` to `../includes/Neuro Net Includes.vcxproj` is likely legacy for Windows builds and not relevant for the current CMake-based Linux build. This missing file was therefore not modified.
- Updated `README.md` to reflect the new directory structure, file paths, and updated include paths in code examples.
- Updated root `CMakeLists.txt` and includes in test files (`tests/test_neuronet.cpp`, `tests/test_genetic_algorithm.cpp`) to support the new modularized file structure.
- Corrected namespace qualification for `NeuroNet` type within the Optimization module (`genetic_algorithm.h` and `genetic_algorithm.cpp`) to resolve compilation errors.


