#ifndef CUDANET_ACTIVATION_FUNCTIONS_H
#define CUDANET_ACTIVATION_FUNCTIONS_H

#include <cuda_runtime.h>

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

}  // namespace CUDANet::Kernels

#endif  // CUDANET_ACTIVATION_FUNCTIONS_H