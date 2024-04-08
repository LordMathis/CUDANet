#include "cuda_helper.cuh"
#include "matmul.cuh"

using namespace CUDANet;

__global__ void Kernels::mat_vec_mul(
    const float* __restrict__ d_matrix,
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int w,
    const unsigned int h
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float shared[BLOCK_SIZE];

    float temp = 0.0f;

#pragma unroll
    for (unsigned int i = 0; i < (w + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        if (i * BLOCK_SIZE + threadIdx.x < w) {
            shared[threadIdx.x] = d_vector[i * BLOCK_SIZE + threadIdx.x];
        } else {
            shared[threadIdx.x] = 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (unsigned int j = 0; j < BLOCK_SIZE; j++) {
            temp += d_matrix[tid * w + i * BLOCK_SIZE + j] * shared[j];
        }

        __syncthreads();
    }

    d_output[tid] = temp;
}

__global__ void Kernels::vec_vec_add(
    const float* __restrict__ d_vector1,
    const float* __restrict__ d_vector2,
    float* __restrict__ d_output,
    const unsigned int w
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= w) {
        return;
    }
    d_output[tid] = d_vector1[tid] + d_vector2[tid];
}

__global__ void Kernels::max_reduce(
    const float* __restrict__ d_vector,
    float* __restrict__ d_output
) {
    __shared__ float shared_max[BLOCK_SIZE];
    int i       = blockIdx.x * blockDim.x + threadIdx.x;

    shared_max[threadIdx.x] = d_vector[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = shared_max[0];
    }
}

__global__ void Kernels::vec_scalar_sub(
    const float* __restrict__ d_vector,
    const float* __restrict__ d_scalar,
    float* __restrict__ d_output,
    const unsigned int w
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= w) {
        return;
    }
    d_output[tid] = d_vector[tid] - d_scalar[0];
}