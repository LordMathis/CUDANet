cmake_minimum_required(VERSION 3.17)

project(CUDANet
        LANGUAGES CXX CUDA
)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Add project source files for the library
set(LIBRARY_SOURCES
    src/utils/cuda_helper.cu
    src/layers/dense.cu
)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_75)

# Build static library
add_library(${PROJECT_NAME} STATIC ${LIBRARY_SOURCES})

# Link cuBLAS library to the library
target_link_libraries(${PROJECT_NAME} CUDA::cublas CUDA::cudart)

# Set include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

# Add testing subdirectory
add_subdirectory(test)