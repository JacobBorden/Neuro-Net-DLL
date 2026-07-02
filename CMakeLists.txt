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

# Enable PIC for pybind11 shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(neuronet STATIC 
    src/neural_network/neuronet.cpp
    src/neural_network/rnn_layer.cpp
    src/neural_network/lstm_layer.cpp
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
    src/transformer/transformer_decoder_layer.cpp
    src/transformer/transformer_encoder_decoder_model.cpp
    src/transformer/transformer_ffn.cpp
    src/transformer/transformer_model.cpp
    src/math/extended_matrix_ops.cpp
    src/math/gaussian_distribution.cpp
    src/math/least_squares.cpp
    src/classifier/naive_bayes.cpp
)

# Testing subdirectory
add_subdirectory(tests)

# Pybind11 for Python bindings
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(pyneuronet src/python_bindings/pyneuronet.cpp)
target_link_libraries(pyneuronet PRIVATE neuronet)

# Example executable
add_executable(BasicUsageExport examples/basic_usage_export.cpp)
target_link_libraries(BasicUsageExport PRIVATE neuronet)
# target_include_directories(BasicUsageExport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src) # Already covered by global include_directories(src)

# --- INSTALL & PACKAGING ---

include(GNUInstallDirs)

# Install library
install(TARGETS neuronet
    EXPORT NeuroNetTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(NOT DEFINED NEURONET_PYTHON_INSTALL_DIR)
    set(NEURONET_PYTHON_EXECUTABLE "")
    foreach(PYTHON_EXECUTABLE_CANDIDATE Python3_EXECUTABLE Python_EXECUTABLE PYTHON_EXECUTABLE)
        if(DEFINED ${PYTHON_EXECUTABLE_CANDIDATE} AND NOT NEURONET_PYTHON_EXECUTABLE)
            set(NEURONET_PYTHON_EXECUTABLE "${${PYTHON_EXECUTABLE_CANDIDATE}}")
        endif()
    endforeach()

    if(NEURONET_PYTHON_EXECUTABLE)
        execute_process(
            COMMAND ${NEURONET_PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('platlib', vars={'base':'','platbase':''}).lstrip('/'))"
            OUTPUT_VARIABLE NEURONET_DETECTED_PYTHON_INSTALL_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()

    if(NEURONET_DETECTED_PYTHON_INSTALL_DIR)
        set(NEURONET_PYTHON_INSTALL_DIR "${NEURONET_DETECTED_PYTHON_INSTALL_DIR}")
    elseif(DEFINED Python3_SITEARCH)
        set(NEURONET_PYTHON_INSTALL_DIR "${Python3_SITEARCH}")
    elseif(DEFINED Python_SITEARCH)
        set(NEURONET_PYTHON_INSTALL_DIR "${Python_SITEARCH}")
    else()
        set(NEURONET_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    endif()

    if(IS_ABSOLUTE "${NEURONET_PYTHON_INSTALL_DIR}")
        file(RELATIVE_PATH NEURONET_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}" "${NEURONET_PYTHON_INSTALL_DIR}")
    endif()

    if(NOT NEURONET_PYTHON_INSTALL_DIR OR NEURONET_PYTHON_INSTALL_DIR MATCHES "^\\.\\.")
        set(NEURONET_PYTHON_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
    endif()
endif()
set(NEURONET_PYTHON_INSTALL_DIR "${NEURONET_PYTHON_INSTALL_DIR}" CACHE PATH "Install destination for the pyneuronet Python extension")

install(TARGETS pyneuronet
    LIBRARY DESTINATION ${NEURONET_PYTHON_INSTALL_DIR}
)

# Install public headers
install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/neuronet
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Packaging settings must be defined before including CPack.
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_NAME "neuronet")
set(CPACK_PACKAGE_VENDOR "NeuroNet Contributors")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++ Neural Network and Transformer Library")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT "maintainer@example.com")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_GENERATOR "DEB;RPM;TGZ")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

include(CPack)
