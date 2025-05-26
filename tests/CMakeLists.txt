# Enable CTest
include(CTest)

# Define the NeuroNet test executable
add_executable(test_neuronet test_neuronet.cpp test_json.cpp)
target_link_libraries(test_neuronet PRIVATE neuronet GTest::gtest) 
add_test(NAME test_neuronet COMMAND test_neuronet)

# Define the Genetic Algorithm test executable
add_executable(test_genetic_algorithm test_genetic_algorithm.cpp)
target_link_libraries(test_genetic_algorithm PRIVATE neuronet GTest::gtest_main) 
add_test(NAME test_genetic_algorithm COMMAND test_genetic_algorithm)
