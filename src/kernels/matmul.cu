#include "matmul.cuh"

__global__ void Kernels::mat_vec_mul(
    const float* d_matrix,
    const float* d_vector,
    float*       d_output,
    int          w,
    int          h
) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= w * h) {
        return;
    }

    d_output[tid] = 0.0f;

    for (int i = 0; i < w; i++) {
        d_output[tid] += d_matrix[tid * w + i] * d_vector[i];
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
