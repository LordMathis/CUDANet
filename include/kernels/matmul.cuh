#ifndef CUDANET_MATMUL_H
#define CUDANET_MATMUL_H

#include <cuda_runtime.h>

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

/**
 * @brief Vector vector subtraction kernel
 * 
 * @param d_vector1 
 * @param d_vector2 
 * @param d_output 
 * @param w 
 * @return __global__ 
 */
__global__ void vec_vec_sub(
    const float* __restrict__ d_vector1,
    const float* __restrict__ d_vector2,
    float* __restrict__ d_output,
    const unsigned int w
);

__global__ void vec_vec_mul(
    const float* __restrict__ d_vector1,
    const float* __restrict__ d_vector2,
    float* __restrict__ d_output,
    const unsigned int w
);

/**
 * @brief Sub scalar from each element of the vector
 * 
 * @param d_vector 
 * @param d_scalar 
 * @param d_output 
 * @param w 
 * @return __global__ 
 */
__global__ void vec_scalar_sub(
    const float* __restrict__ d_src,
    float* __restrict__ d_out,
    const float* __restrict__ d_scalar,
    const unsigned int len
);

/**
 * @brief Add scalar to each element of the vector
 * 
 * @param d_src 
 * @param d_out 
 * @param d_scalar 
 * @param len 
 * @return __global__ 
 */
__global__ void vec_scalar_add(
    const float* __restrict__ d_src,
    float* __restrict__ d_out,
    const float* __restrict__ d_scalar,
    const unsigned int len
);

/**
 * @brief Divide each element of the vector by a scalar
 *
 * @param src Pointer to the source array
 * @param dst Pointer to the destination array
 * @param len Length of the arrays
 */
__global__ void vec_scalar_div(
    const float* __restrict__ d_src,
    float* __restrict__ d_out,
    const float* __restrict__ d_scalar,
    const unsigned int len
);

/**
 * @brief Multiply each element of the vector by a scalar
 * 
 * @param d_src 
 * @param d_out 
 * @param d_scalar 
 * @param len 
 * @return __global__ 
 */
__global__ void vec_scalar_mul(
    const float* __restrict__ d_src,
    float* __restrict__ d_out,
    const float* __restrict__ d_scalar,
    const unsigned int len
);

/**
 * @brief Exponentiate each element of the vector
 *
 * @param src Pointer to the source array
 * @param dst Pointer to the destination array
 * @param len Length of the arrays
 */
__global__ void vec_exp(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
);

/**
 * @brief Compute the square root of each element of the vector
 * 
 * @param src Device pointer to source vector
 * @param dst Device pointer to destination vector
 * @param len Length of the vector
 * @return __global__ 
 */
__global__ void vec_sqrt(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
);

/**
 * @brief Max reduction kernel
 *
 * @param d_vector Device pointer to vector
 * @param d_output Device pointer to output vector
 */
__global__ void max_reduce(
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int len
);

/**
 * @brief 
 * 
 * @param d_vector Device pointer to vector
 * @param d_output Device pointer to output vector
 * @param len Length of the vector
 */
__global__ void sum_reduce(
    const float* __restrict__ d_vector,
    float* __restrict__ d_output,
    const unsigned int len
);

}  // namespace CUDANet::Kernels

#endif  // CUDANET_MATMUL_H