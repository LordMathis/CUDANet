#ifndef CUDANET_ACTIVATIONS_H
#define CUDANET_ACTIVATIONS_H

namespace Kernels {

/**
 * @brief Sigmoid activation function kernel
 * 
 * @param src Pointer to the source array
 * @param dst Pointer to the destination array
 * @param len Length of the arrays
 */
__global__ void
sigmoid(const float* __restrict__ src, float* __restrict__ dst, int len);

/**
 * @brief Relu activation function kernel
 * 
 * @param src Pointer to the source array
 * @param dst Pointer to the destination array
 * @param len Length of the arrays
 */
__global__ void
relu(const float* __restrict__ src, float* __restrict__ dst, int len);

}  // namespace Kernels

#endif  // CUDANET_ACTIVATIONS_H