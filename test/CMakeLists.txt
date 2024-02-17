find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(test_dense layers/test_dense.cpp)

add_library(test_utils
    test_utils/test_cublas_fixture.cpp
)

target_include_directories(test_utils PUBLIC test_utils)

target_link_libraries(test_dense gtest gtest_main CUDANet test_utils)

add_test(NAME TestDense COMMAND test_dense)