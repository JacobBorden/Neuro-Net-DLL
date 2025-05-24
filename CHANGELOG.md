# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-05-23

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
