cmake_minimum_required(VERSION 3.0.0)
project(neuronet VERSION 0.1.0)

include(CTest)
include(FetchContent)
enable_testing()

FetchContent_Populate(
    matrix
    GIT_REPOSITORY https://github.com/JacobBorden/Matrix.git
    SOURCE_DIR ../dependencies/matrix
)

include_directories(src)
include_directories(dependencies)

add_library(neuronet SHARED src/dllmain.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
