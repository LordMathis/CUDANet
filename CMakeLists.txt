cmake_minimum_required(VERSION 3.12)

project(CUDANet)

# Find CUDA
find_package(CUDA REQUIRED)

# Add CUDA include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add project source files
set(SOURCES
    src/main.cpp
    src/utils/cuda_helper.cpp
)

# Set CUDA architecture (change according to your GPU)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_75)

# Build executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES})

# Link cuBLAS library
target_link_libraries(${PROJECT_NAME} ${CUDA_cublas_LIBRARY})

# Set include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Set C++ standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)