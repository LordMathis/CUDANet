#include <functional>

#include "activations.cuh"

__device__ float sigmoid(float a) {
    return 1.0 / (1.0 + exp(-a));
}

__device__ float relu(float a) {
    return a < 0.0 ? 0.0 : a;
}

__device__ float linear(float a) {
    return a;
}

__global__ void sigmoid_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int len
) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        dst[i] = sigmoid(src[i]);
    }
}

__global__ void
relu_kernel(const float* __restrict__ src, float* __restrict__ dst, int len) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        dst[i] = relu(src[i]);
    }
}

__global__ void
linear_kernel(const float* __restrict__ src, float* __restrict__ dst, int len) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        dst[i] = linear(src[i]);
    }
}
