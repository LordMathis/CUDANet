#ifndef CONVOLUTION_H
#define CONVOLUTION_H

__global__ void convolution_kernel(
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

#endif  // CONVOLUTION_H