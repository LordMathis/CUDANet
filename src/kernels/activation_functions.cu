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

__global__ void Kernels::softmax_exp(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        dst[i] = expf(src[i]);
    }
}

__global__ void Kernels::softmax_sum(
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int w
) {
    __shared__ float partial_sum[BLOCK_SIZE];
    int              i       = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    partial_sum[threadIdx.x] = d_vector[i] + d_vector[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = partial_sum[0];
    }
}

__global__ void Kernels::softmax_div(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const float* __restrict__ sum,
    const unsigned int len
) {
    int stride = gridDim.x * blockDim.x;
    int tid    = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = tid; i < len; i += stride) {
        dst[i] = src[i] / sum[0];
    }
}