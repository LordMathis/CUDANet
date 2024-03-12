find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(test_main 
    layers/test_dense.cu
    layers/test_conv2d.cu
    layers/test_input.cu
    kernels/test_activations.cu
    kernels/test_padding.cu
)

target_link_libraries(test_main ${GTEST_BOTH_LIBRARIES} CUDANet)

add_test(NAME TestMain COMMAND test_main)