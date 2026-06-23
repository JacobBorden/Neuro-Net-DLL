cmake_minimum_required(VERSION 3.0.0)
project(neuronet VERSION 0.1.0)

include(CTest)
include(FetchContent)
enable_testing()

# Google Test
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip # Or a suitable release
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(src)

add_library(neuronet STATIC 
    src/neural_network/neuronet.cpp
    src/optimization/genetic_algorithm.cpp
    src/optimization/neural_pathfinder.cpp
    src/utilities/json/json.cpp
    src/utilities/timer.cpp
    src/utilities/vocabulary.cpp
    src/utilities/logger.cpp
    src/utilities/dataset_loader.cpp
    src/transformer/attention.cpp
    src/transformer/embedding.cpp
    src/transformer/multi_head_attention.cpp
    src/transformer/positional_encoding.cpp
    src/transformer/transformer_encoder_layer.cpp
    src/transformer/transformer_ffn.cpp
    src/transformer/transformer_model.cpp
    src/math/extended_matrix_ops.cpp
    src/math/gaussian_distribution.cpp
    src/math/least_squares.cpp
    src/classifier/naive_bayes.cpp
)

# Testing subdirectory
add_subdirectory(tests)

# Example executable
add_executable(BasicUsageExport examples/basic_usage_export.cpp)
target_link_libraries(BasicUsageExport PRIVATE neuronet)
# target_include_directories(BasicUsageExport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src) # Already covered by global include_directories(src)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# --- INSTALL & PACKAGING ---

include(GNUInstallDirs)

# 1. Install Library
install(TARGETS neuronet
    EXPORT NeuroNetTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 2. Install Headers
install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neuronet
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# 3. Packaging Settings (CPack)
set(CPACK_PACKAGE_NAME "neuronet")
set(CPACK_PACKAGE_VENDOR "NeuroNet Contributors")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++ Neural Network and Transformer Library")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "maintainer@example.com") # Required for DEB
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

# Set the target packages to build
set(CPACK_GENERATOR "DEB;RPM;TGZ")

# Debian specifics
set(CPACK_DEBIAN_PACKAGE_DEPENDS "")

# RPM specifics
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

include(CPack)
