#include "matmul.cuh"

__global__ void Kernels::mat_vec_mul(
    const float* d_matrix,
    const float* d_vector,
    float*       d_output,
    int          w,
    int          h
) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ float shared[];
    
    if (tid < w) {
        shared[tid] = d_vector[tid];
    }

    __syncthreads();

    if (tid < h) {
        d_output[tid] = 0.0f;

        #pragma unroll
        for (int i = 0; i < w; i++) {
            d_output[tid] += d_matrix[tid * w + i] * shared[i];
        }
    }
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
