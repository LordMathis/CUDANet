find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB_RECURSE TEST_SOURCES
    *.cu
    kernels/*.cu
    layers/*.cu
    model/*.cpp
)

add_executable(test_main 
    EXCLUDE_FROM_ALL
    ${TEST_SOURCES}
)

target_link_libraries(test_main ${GTEST_BOTH_LIBRARIES} CUDANet)

add_test(NAME TestMain COMMAND test_main)