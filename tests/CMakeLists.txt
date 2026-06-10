# Enable CTest
include(CTest)
include(GoogleTest)

# Define the test executable
add_executable(runTests
    test_neuronet.cpp
    test_genetic_algorithm.cpp
    test_json.cpp
    test_vocabulary.cpp
    test_transformer_model.cpp
    test_gaussian_distribution.cpp
    test_least_squares.cpp
    test_naive_bayes.cpp
    test_extended_matrix_ops.cpp
    test_timer.cpp
    test_neural_pathfinder.cpp
)

# Link libraries
target_link_libraries(runTests PRIVATE neuronet GTest::gtest_main)

# Discover and add tests
gtest_discover_tests(runTests)
