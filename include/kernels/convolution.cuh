#ifndef CUDANET_CONVOLUTION_H
#define CUDANET_CONVOLUTION_H

namespace CUDANet::Kernels {

/**
 * @brief Convolution kernel
 *
 * @param d_input Device pointer to the input matrix
 * @param d_kernel Device pointer to the convolution kernel
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
    float* __restrict__ d_output,
    const int inputSize,
    const int nChannels,
    const int paddingSize,
    const int kernelSize,
    const int stride,
    const int nFilters,
    const int outputSize
);

}  // namespace CUDANet::Kernels

#endif  // CUDANET_CONVOLUTION_H