#include "convolution.cuh"

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
    int i = (tid % (outputSize * outputSize)) / outputSize;
    int j = (tid % (outputSize * outputSize)) % outputSize;

    float sum = 0.0f;

    // std::cout << "f: " << f << ", i: " << i << ", j: " << j << std::endl;

    // Iterate over kernel and input matrix
    for (int k = 0; k < kernelSize; k++) {
        for (int l = 0; l < kernelSize; l++) {
            for (int c = 0; c < nChannels; c++) {
                int kernelIndex =
                    k * (kernelSize * nChannels * nFilters) +
                    l * (nChannels * nFilters) + c * (nFilters) + f;
                int inputIndex =
                    (i * stride + k) * (inputSize * nChannels) +
                    (j * stride + l) * (nChannels) + c;

                // std::cout << "kernelIndex: " << kernelIndex << ", kernel
                // value: " << kernels[kernelIndex] << ", inputIndex: " <<
                // inputIndex << ", input value: " << input[inputIndex] <<
                // std::endl;

                sum += d_kernel[kernelIndex] * d_input[inputIndex];
            }
        }
    }

    // std::cout << "sum: " << sum << std::endl;

    d_output[i * (outputSize * nFilters) + j * (nFilters) + f] = sum;
}