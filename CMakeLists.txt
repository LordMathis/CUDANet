cmake_minimum_required(VERSION 3.20)

project(CUDANet
        LANGUAGES CXX CUDA
)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

file(GLOB_RECURSE LIBRARY_SOURCES
    src/*.cu
    src/utils/*.cu
    src/kernels/*.cu
    src/layers/*.cu
    src/model/*.cpp
)

set(LIBRARY_SOURCES
    ${LIBRARY_SOURCES}
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build static library
add_library(${PROJECT_NAME} STATIC ${LIBRARY_SOURCES})

target_link_libraries(${PROJECT_NAME} CUDA::cudart)

# Set include directories for the library
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/include/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/include/model
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

# Add testing subdirectory
add_subdirectory(test)