#ifndef CONVOLUTION_H
#define CONVOLUTION_H

namespace Kernels {

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
    const float* d_input,
    float*       d_padded,
    int          w,
    int          h,
    int          n,
    int          p
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
    const float* d_input,
    const float* d_kernel,
    float*       d_output,
    int          inputSize,
    int          nChannels,
    int          kernelSize,
    int          stride,
    int          nFilters,
    int          outputSize
);

}  // namespace Kernels

#endif  // CONVOLUTION_H