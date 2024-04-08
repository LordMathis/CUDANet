#ifndef CUDANET_ACTIVATION_FUNCTIONS_H
#define CUDANET_ACTIVATION_FUNCTIONS_H

namespace CUDANet::Kernels {

/**
 * @brief Sigmoid activation function kernel
 *
 * @param src Pointer to the source array
 * @param dst Pointer to the destination array
 * @param len Length of the arrays
 */
__global__ void sigmoid(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
);

/**
 * @brief Relu activation function kernel
 *
 * @param src Pointer to the source array
 * @param dst Pointer to the destination array
 * @param len Length of the arrays
 */
__global__ void relu(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
);

/**
 * @brief Softmax activation exponentiation kernel
 *
 * @param src Pointer to the source array
 * @param dst Pointer to the destination array
 * @param len Length of the arrays
 */
__global__ void softmax_exp(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int len
);

/**
 * @brief 
 * 
 * @param d_vector Device pointer to vector
 * @param d_output Device pointer to output vector
 * @param w Length of the vector
 */
__global__ void softmax_sum(
    const float* __restrict__ d_vector,
    float* __restrict__ d_output
);

/**
 * @brief Softmax activation function kernel
 *
 * @param src Pointer to the source array
 * @param dst Pointer to the destination array
 * @param len Length of the arrays
 */
__global__ void softmax_div(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const float* __restrict__ sum,
    const unsigned int len
);

}  // namespace CUDANet::Kernels

#endif  // CUDANET_ACTIVATION_FUNCTIONS_H