#include "activation_functions.cuh"
#include "cuda_helper.cuh"

using namespace CUDANet;

__global__ void Kernels::sigmoid(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        dst[i] = 1.0 / (1.0 + exp(-src[i]));
    }
}

__global__ void Kernels::relu(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        dst[i] = src[i] < 0.0 ? 0.0 : src[i];
    }
}
