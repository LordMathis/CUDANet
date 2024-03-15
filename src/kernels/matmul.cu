#include "cuda_helper.cuh"
#include "matmul.cuh"

#define SHARED_SIZE 128 * 4

__global__ void Kernels::mat_vec_mul(
    const float* __restrict__ d_matrix,
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    int w,
    int h
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float shared[BLOCK_SIZE];

    float temp = 0.0f;

    #pragma unroll
    for (unsigned int i = 0; i < (w + BLOCK_SIZE - 1) / BLOCK_SIZE; i++)
    {
        if (i * BLOCK_SIZE + threadIdx.x < w) {
            shared[threadIdx.x] = d_vector[i * BLOCK_SIZE + threadIdx.x];
        } else {
            shared[threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (unsigned int j = 0; j < BLOCK_SIZE; j++)
        {
            temp += d_matrix[tid * w + i * BLOCK_SIZE + j] * shared[j];
        }

        __syncthreads();
    }
    
    d_output[tid] = temp;
}

__global__ void Kernels::vec_vec_add(
    const float* d_vector1,
    const float* d_vector2,
    float*       d_output,
    int          w
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= w) {
        return;
    }
    d_output[tid] = d_vector1[tid] + d_vector2[tid];
}
