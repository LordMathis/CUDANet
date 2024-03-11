#ifndef CONVOLUTION_H
#define CONVOLUTION_H

namespace Kernels {

__global__ void padding(
    const float* d_input,
    float*       d_padded,
    int          w,
    int          h,
    int          n,
    int          p
);

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