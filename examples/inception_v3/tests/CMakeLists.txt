find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB_RECURSE TEST_SOURCES
    *.cpp
)

add_executable(test_inception_v3
    EXCLUDE_FROM_ALL
    ${TEST_SOURCES}
)

target_include_directories(test_inception_v3 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CUDANet_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(test_inception_v3
    ${GTEST_BOTH_LIBRARIES}
    ${CUDANet_LIBRARY}
    CUDA::cudart
    inception_v3_lib
)

add_test(NAME TestMain COMMAND test_inception_v3)