# Enable CTest
include(CTest)
include(GoogleTest)

# Define the test executable
add_executable(runTests
    test_neuronet.cpp
    test_genetic_algorithm.cpp
    test_json.cpp
)

# Link libraries
target_link_libraries(runTests PRIVATE neuronet GTest::gtest_main)

# Discover and add tests
gtest_discover_tests(runTests)
