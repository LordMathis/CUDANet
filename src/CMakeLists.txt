set(LAYER_SOURCES layers/dense.cu)

add_library(CUDANet
    utils/cuda_helper.cu
    utils/functions.cu
    ${LAYER_SOURCES}
)
