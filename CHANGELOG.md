# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-05-23

### Added
- Pluggable activation functions (None, ReLU, LeakyReLU, ELU, Softmax) for `NeuroNetLayer`. Layers now apply the selected activation function after the linear transformation.
- `ActivationFunctionType` enum to specify activation types.
- `NeuroNetLayer::SetActivationFunction()` method.
- Private helper methods in `NeuroNetLayer` for applying each activation function.

### Changed
- `NeuroNetLayer::CalculateOutput()` now incorporates the selected activation function.
- Default constructor `NeuroNetLayer()` initializes with `ActivationFunctionType::None`.
- Updated Doxygen comments in `neuronet.h` and `neuronet.cpp` for new activation function features.
- Updated `README.md` with a new section explaining activation functions and providing usage examples.

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


