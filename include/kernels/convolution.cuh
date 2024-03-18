#ifndef CUDANET_CONVOLUTION_H
#define CUDANET_CONVOLUTION_H

namespace CUDANet::Kernels {

/**
 * @brief Kernel that pads the input matrix with zeros
 *
 * @param d_input Device pointer to the input matrix (as vector)
 * @param d_padded Device pointer to the padded matrix (as vector)
 * @param w Width of the input matrix
 * @param h Height of the input matrix
 * @param n Number of input channels
 * @param p Padding size
 */
__global__ void padding(
    const float* __restrict__ d_input,
    float* __restrict__ d_padded,
    const unsigned int w,
    const unsigned int h,
    const unsigned int n,
    const unsigned int p
);

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