#include <iostream>

#include "convolution.cuh"

using namespace CUDANet;

__global__ void Kernels::convolution(
    const float* __restrict__ d_input,
    const float* __restrict__ d_kernel,
    const float* __restrict__ d_bias,
    float* __restrict__ d_output,
    const shape2d inputSize,
    const int   nChannels,
    const shape2d paddingSize,
    const shape2d kernelSize,
    const shape2d stride,
    const int   nFilters,
    const shape2d outputSize
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int f = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= outputSize.first || j >= outputSize.second || f >= nFilters) {
        return;
    }

    float sum = 0.0f;

    // Iterate over kernel and input matrix
    for (int c = 0; c < nChannels; c++) {
        for (int k = 0; k < kernelSize.first; k++) {
            for (int l = 0; l < kernelSize.second; l++) {
                // if i, j is in the padding region
                if (i * stride.first + k < paddingSize.first ||
                    i * stride.first + k >=
                        (inputSize.first + paddingSize.first) ||
                    j * stride.second + l < paddingSize.second ||
                    j * stride.second + l >=
                        (inputSize.second + paddingSize.second)) {
                    continue;
                }

                int kernelIndex =
                    f * kernelSize.first * kernelSize.second * nChannels +
                    c * kernelSize.first * kernelSize.second +
                    k * kernelSize.second + l;
                int inputIndex = c * inputSize.first * inputSize.second +
                                 (i * stride.first + k - paddingSize.first) *
                                     inputSize.second +
                                 (j * stride.second + l - paddingSize.second);

                sum += d_kernel[kernelIndex] * d_input[inputIndex];
            }
        }
    }

    d_output[f * outputSize.first * outputSize.second + i * outputSize.second + j] =
        sum + d_bias[f];
}