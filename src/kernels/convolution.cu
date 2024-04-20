#include <iostream>

#include "convolution.cuh"

using namespace CUDANet;

__global__ void Kernels::convolution(
    const float* __restrict__ d_input,
    const float* __restrict__ d_kernel,
    const float* __restrict__ d_bias,
    float* __restrict__ d_output,
    const int inputSize,
    const int nChannels,
    const int paddingSize,
    const int kernelSize,
    const int stride,
    const int nFilters,
    const int outputSize
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int f = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= outputSize || j >= outputSize || f >= nFilters) {
        return;
    }

    float sum = 0.0f;

    // Iterate over kernel and input matrix
    for (int c = 0; c < nChannels; c++) {
        for (int k = 0; k < kernelSize; k++) {
            for (int l = 0; l < kernelSize; l++) {

                // if i, j is in the padding region
                if (i * stride + k < paddingSize ||
                    i * stride + k >= (inputSize + paddingSize) ||
                    j * stride + l < paddingSize ||
                    j * stride + l >= (inputSize + paddingSize)) {
                    continue;
                }

                int kernelIndex = f * kernelSize * kernelSize * nChannels +
                                  c * kernelSize * kernelSize + k * kernelSize +
                                  l;
                int inputIndex = c * inputSize * inputSize +
                                 (i * stride + k - paddingSize) * inputSize +
                                 (j * stride + l - paddingSize);

                sum += d_kernel[kernelIndex] * d_input[inputIndex];
            }
        }
    }

    d_output[f * outputSize * outputSize + i * outputSize + j] = sum + d_bias[f];
}