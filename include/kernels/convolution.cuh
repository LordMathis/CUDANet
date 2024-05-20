#ifndef CUDANET_CONVOLUTION_H
#define CUDANET_CONVOLUTION_H

#include <cuda_runtime.h>
#include "layer.cuh"

namespace CUDANet::Kernels {

/**
 * @brief Convolution kernel
 *
 * @param d_input Device pointer to the input matrix
 * @param d_kernel Device pointer to the convolution kernel
 * @param d_bias Device pointer to the bias
 * @param d_output Device pointer to the output matrix
 * @param inputSize Width and height of the input matrix
 * @param nChannels Number of channels in the input matrix
 * @param kernelSize Width and height of the convolution kernel
 * @param stride Convolution stride
 * @param nFilters Number of output filters
 * @param outputSize Width and height of the output matrix
 */
__global__ void convolution(
    const float* __restrict__ d_input,
    const float* __restrict__ d_kernel,
    const float* __restrict__ d_bias,
    float* __restrict__ d_output,
    const dim2d inputSize,
    const int nChannels,
    const dim2d paddingSize,
    const dim2d kernelSize,
    const dim2d stride,
    const int nFilters,
    const dim2d outputSize
);

}  // namespace CUDANet::Kernels

#endif  // CUDANET_CONVOLUTION_H