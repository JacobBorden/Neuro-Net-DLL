cmake_minimum_required(VERSION 3.0.0)
project(neuronet VERSION 0.1.0)

include(CTest)
include(FetchContent)
enable_testing()

FetchContent_Populate(
    matrix
    GIT_REPOSITORY https://github.com/JacobBorden/Matrix.git
    GIT_TAG 9ae78f85c920a1f6419b426a23f2f7c843b4ea25
    SOURCE_DIR matrix
)


add_library(neuronet SHARED neuronet.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
