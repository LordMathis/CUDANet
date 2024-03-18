find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(test_main 
    EXCLUDE_FROM_ALL
    layers/test_activation.cu
    layers/test_concat.cu
    layers/test_conv2d.cu
    layers/test_dense.cu
    layers/test_input.cu
    kernels/test_activation_functions.cu
    kernels/test_matmul.cu
)

target_link_libraries(test_main ${GTEST_BOTH_LIBRARIES} CUDANet)

add_test(NAME TestMain COMMAND test_main)