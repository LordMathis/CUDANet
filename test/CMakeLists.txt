find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(test_main layers/test_dense.cu kernels/test_activations.cu)

add_library(test_utils
    test_utils/test_cublas_fixture.cu
)

target_include_directories(test_utils PUBLIC test_utils)

target_link_libraries(test_main ${GTEST_BOTH_LIBRARIES} CUDANet test_utils)

add_test(NAME TestMain COMMAND test_main)