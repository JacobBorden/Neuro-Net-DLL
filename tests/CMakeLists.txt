# Enable CTest
include(CTest)
include(GoogleTest)

# Define the test executable
add_executable(runTests
    test_neuronet.cpp
    test_rnn_layer.cpp
    test_genetic_algorithm.cpp
    test_json.cpp
    test_vocabulary.cpp
    test_transformer_model.cpp
    test_transformer_encoder_decoder.cpp
    test_gaussian_distribution.cpp
    test_least_squares.cpp
    test_naive_bayes.cpp
    test_extended_matrix_ops.cpp
    test_timer.cpp
    test_neural_pathfinder.cpp
    test_dataset_loader.cpp
    test_logger.cpp
)

# Link libraries
target_link_libraries(runTests PRIVATE neuronet GTest::gtest_main)

# Discover and add tests
gtest_discover_tests(runTests)

set(NEURONET_PYTHON_TEST_EXECUTABLE "")
foreach(PYTHON_EXECUTABLE_CANDIDATE Python3_EXECUTABLE Python_EXECUTABLE PYTHON_EXECUTABLE)
    if(DEFINED ${PYTHON_EXECUTABLE_CANDIDATE} AND NOT NEURONET_PYTHON_TEST_EXECUTABLE)
        set(NEURONET_PYTHON_TEST_EXECUTABLE "${${PYTHON_EXECUTABLE_CANDIDATE}}")
    endif()
endforeach()

if(NEURONET_PYTHON_TEST_EXECUTABLE)
    add_test(
        NAME pyneuronet_python
        COMMAND ${NEURONET_PYTHON_TEST_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_pyneuronet.py
    )
    set_tests_properties(
        pyneuronet_python
        PROPERTIES ENVIRONMENT "PYNEURONET_BUILD_DIR=${CMAKE_BINARY_DIR}"
    )
endif()
