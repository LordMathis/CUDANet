find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB COMMON_TEST_SOURCES
    common/*.hpp
    model/*.cpp
)

if(USE_CUDA)
    file(GLOB TEST_SOURCES
        cuda/*.cu
        cuda/kernels/*.cu
        cuda/layers/*.cu
        cuda/utils/*.cu
    )
    add_executable(test_main 
        EXCLUDE_FROM_ALL
        ${COMMON_TEST_SOURCES}
        ${TEST_SOURCES}
    )
    set_target_properties(test_main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    file(GLOB TEST_SOURCES
        cpu/*.cpp
    )
    add_executable(test_main 
        EXCLUDE_FROM_ALL
        ${COMMON_TEST_SOURCES}
        ${TEST_SOURCES}
    )
endif()

target_link_libraries(test_main ${GTEST_BOTH_LIBRARIES} CUDANet)

add_test(NAME TestMain COMMAND test_main)