#ifndef CUDANET_MATMUL_H
#define CUDANET_MATMUL_H

namespace CUDANet::Kernels {

/**
 * @brief Matrix vector multiplication kernel
 *
 * @param d_matrix Device pointer to matrix
 * @param d_vector Device pointer to vector
 * @param d_output Device pointer to output vector
 * @param w Width of the matrix
 * @param h Height of the matrix
 */
__global__ void mat_vec_mul(
    const float* __restrict__ d_matrix,
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int w,
    const unsigned int h
);

/**
 * @brief Vector vector addition kernel
 *
 * @param d_vector1 Device pointer to first vector
 * @param d_vector2 Device pointer to second vector
 * @param d_output Device pointer to output vector
 * @param w Length of the vectors
 */
__global__ void vec_vec_add(
    const float* __restrict__ d_vector1,
    const float* __restrict__ d_vector2,
    float* __restrict__ d_output,
    const unsigned int w
);

}  // namespace CUDANet::Kernels

#endif  // CUDANET_MATMUL_H