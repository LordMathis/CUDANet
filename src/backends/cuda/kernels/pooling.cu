#include "cuda_helper.cuh"
#include "layer.cuh"
#include "pooling.cuh"

using namespace CUDANet;

__global__ void Kernels::max_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const shape2d inputSize,
    const shape2d outputSize,
    const int   nChannels,
    const shape2d poolingSize,
    const shape2d stride,
    const shape2d padding
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= outputSize.first || j >= outputSize.second || c >= nChannels) {
        return;
    }

    float max = 0.0f;

    for (int k = 0; k < poolingSize.first; k++) {
        for (int l = 0; l < poolingSize.second; l++) {
            int inputRow = i * stride.first + k - padding.first;
            int inputCol = j * stride.second + l - padding.second;

            if (inputRow >= 0 && inputRow < inputSize.first && inputCol >= 0 &&
                inputCol < inputSize.second) {
                int inputIndex = c * inputSize.first * inputSize.second +
                                 inputRow * inputSize.second + inputCol;
                if (d_input[inputIndex] > max) {
                    max = d_input[inputIndex];
                }
            }
        }
    }

    d_output
        [c * outputSize.first * outputSize.second + i * outputSize.second + j] =
            max;
}

__global__ void Kernels::avg_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const shape2d inputSize,
    const shape2d outputSize,
    const int   nChannels,
    const shape2d poolingSize,
    const shape2d stride,
    const shape2d padding
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= outputSize.first || j >= outputSize.second || c >= nChannels) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < poolingSize.first; k++) {
        for (int l = 0; l < poolingSize.second; l++) {
            int inputRow = i * stride.first + k - padding.first;
            int inputCol = j * stride.second + l - padding.second;

            if (inputRow >= 0 && inputRow < inputSize.first && inputCol >= 0 &&
                inputCol < inputSize.second) {
                int inputIndex = c * inputSize.first * inputSize.second +
                                 inputRow * inputSize.second + inputCol;
                sum += d_input[inputIndex];
            }
        }
    }

    d_output
        [c * outputSize.first * outputSize.second + i * outputSize.second + j] =
            sum / (poolingSize.first * poolingSize.second);
}