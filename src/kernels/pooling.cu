#include "cuda_helper.cuh"
#include "layer.cuh"
#include "pooling.cuh"

using namespace CUDANet;

__global__ void Kernels::max_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const dim2d inputSize,
    const dim2d outputSize,
    const int   nChannels,
    const dim2d poolingSize,
    const dim2d stride
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
            int inputIndex = c * inputSize.first * inputSize.second +
                             (i * stride.first + k) * inputSize.second +
                             (j * stride.second + l);

            if (d_input[inputIndex] > max) {
                max = d_input[inputIndex];
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
    const dim2d inputSize,
    const dim2d outputSize,
    const int   nChannels,
    const dim2d poolingSize,
    const dim2d stride
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
            int inputIndex = c * inputSize.first * inputSize.second +
                             (i * stride.first + k) * inputSize.second +
                             (j * stride.second + l);

            sum += d_input[inputIndex];
        }
    }

    d_output
        [c * outputSize.first * outputSize.second + i * outputSize.second + j] =
            sum / (poolingSize.first * poolingSize.second);
}