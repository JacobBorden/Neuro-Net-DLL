cmake_minimum_required(VERSION 3.0.0)
project(neuronet VERSION 0.1.0)

include(CTest)
include(FetchContent)
enable_testing()

# Catch2
FetchContent_Declare(
  Catch2
  URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.5.2.zip # Using a specific recent version
)
FetchContent_MakeAvailable(Catch2)
# catch2_INCLUDE_DIRS might not be the correct variable or might not be set as expected.
# Rely on target properties. Catch2::Catch2 should be an INTERFACE library target
# that defines the necessary include directories.

include_directories(src)

add_library(neuronet STATIC src/neural_network/neuronet.cpp src/optimization/genetic_algorithm.cpp)
# Reverted: target_link_libraries(neuronet INTERFACE Catch2::Catch2)
# Rely on tests/CMakeLists.txt to handle Catch2 linkage for test executables.

# Testing subdirectory
add_subdirectory(tests)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
