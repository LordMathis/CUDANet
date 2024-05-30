cmake_minimum_required(VERSION 3.20)

project(Inceptionv3
        LANGUAGES CXX CUDA
)

add_executable(inception_v3 
    inception_v3.cpp 
    inception_modules.cpp
)

find_library(CUDANet_LIBRARY NAMES CUDANet HINTS ${CMAKE_CURRENT_SOURCE_DIR}/../../build)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(CUDAToolkit REQUIRED)

set (CUDANet_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/model
)

target_include_directories(
    inception_v3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS} ${CUDANet_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(inception_v3 PRIVATE ${CUDANet_LIBRARY} ${OpenCV_LIBS} CUDA::cudart)

set_property(TARGET inception_v3 PROPERTY CXX_STANDARD 20)