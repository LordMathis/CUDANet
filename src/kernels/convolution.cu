#include <iostream>

#include "convolution.cuh"

using namespace CUDANet::Kernels;

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
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= outputSize * outputSize * nFilters) {
        return;
    }

    // Get output index
    int f = tid / (outputSize * outputSize);
    int i = tid % (outputSize * outputSize) / outputSize;
    int j = tid % outputSize;

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

    d_output[tid] = sum;
}