set(LAYER_SOURCES layers/dense.cpp)

add_library(CUDANet
    utils/cuda_helper.cpp
    ${LAYER_SOURCES}
)
