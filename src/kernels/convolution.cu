#include "convolution.cuh"
#include <iostream>

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
    for (int k = 0; k < kernelSize; k++) {
        for (int l = 0; l < kernelSize; l++) {
            for (int c = 0; c < nChannels; c++) {
                int kernelIndex = f * kernelSize * kernelSize * nChannels +
                                  c * kernelSize * kernelSize + k * kernelSize +
                                  l;
                int inputIndex = c * inputSize * inputSize +
                                 (i * stride + k) * inputSize +
                                 (j * stride + l);

                sum += d_kernel[kernelIndex] * d_input[inputIndex];
            }
        }
    }

    d_output[tid] = sum;
}