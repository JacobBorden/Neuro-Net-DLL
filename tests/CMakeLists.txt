# Enable CTest
include(CTest)
include(GoogleTest)

# Define the test executable
add_executable(runTests
    test_neuronet.cpp
    test_rnn_layer.cpp
    test_lstm_layer.cpp
    test_conv2d_layer.cpp
    test_genetic_algorithm.cpp
    test_json.cpp
    test_vocabulary.cpp
    test_transformer_model.cpp
    test_transformer_encoder_decoder.cpp
    test_gaussian_distribution.cpp
    test_least_squares.cpp
    test_naive_bayes.cpp
    test_extended_matrix_ops.cpp
    test_matrix_multiplication.cpp
    test_timer.cpp
    test_neural_pathfinder.cpp
    test_dataset_loader.cpp
    test_logger.cpp
)

# Link libraries
target_link_libraries(runTests PRIVATE neuronet GTest::gtest_main)

add_executable(matrix_benchmarks test_benchmarks.cpp)
target_link_libraries(matrix_benchmarks PRIVATE neuronet_benchmarks)
target_compile_definitions(matrix_benchmarks PRIVATE ENABLE_BENCHMARKING)

# Discover and add tests
gtest_discover_tests(runTests)

find_package(Python3 COMPONENTS Interpreter Development.Module)

if(Python3_FOUND AND Python3_Interpreter_FOUND)
    add_test(
        NAME pyneuronet_python
        COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/test_pyneuronet.py
    )
    set_tests_properties(
        pyneuronet_python
        PROPERTIES ENVIRONMENT "PYNEURONET_BUILD_DIR=$<TARGET_FILE_DIR:pyneuronet>"
    )
endif()

# Add test_matrix.cpp
target_sources(runTests PRIVATE test_matrix.cpp)
