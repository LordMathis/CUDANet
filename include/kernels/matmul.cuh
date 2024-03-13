#ifndef CUDANET_MATMUL_H
#define CUDANET_MATMUL_H

namespace Kernels {

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
    const float* d_matrix,
    const float* d_vector,
    float*       d_output,
    int          w,
    int          h
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
    const float* d_vector1,
    const float* d_vector2,
    float*       d_output,
    int          w
);

}  // namespace Kernels

#endif  // CUDANET_MATMUL_H