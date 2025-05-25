# Enable CTest
include(CTest)

# Define the NeuroNet test executable
add_executable(test_neuronet test_neuronet.cpp)
# Global include_directories in parent CMakeLists.txt should handle Catch2 includes.
target_link_libraries(test_neuronet PRIVATE neuronet Catch2::Catch2WithMain) 
add_test(NAME test_neuronet COMMAND test_neuronet)

# Define the Genetic Algorithm test executable
add_executable(test_genetic_algorithm test_genetic_algorithm.cpp)
# Global include_directories in parent CMakeLists.txt should handle Catch2 includes.
target_link_libraries(test_genetic_algorithm PRIVATE neuronet Catch2::Catch2WithMain) 
add_test(NAME test_genetic_algorithm COMMAND test_genetic_algorithm)
