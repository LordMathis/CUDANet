#include "cuda_helper.cuh"
#include "pooling.cuh"

using namespace CUDANet;

__global__ void Kernels::max_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const int inputSize,
    const int outputSize,
    const int nChannels,
    const int poolingSize,
    const int stride
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= outputSize || j >= outputSize || c >= nChannels) {
        return;
    }

    float max = 0.0f;

    for (int k = 0; k < poolingSize; k++) {
        for (int l = 0; l < poolingSize; l++) {
            int inputIndex = c * inputSize * inputSize +
                             (i * stride + k) * inputSize + (j * stride + l);

            if (d_input[inputIndex] > max) {
                max = d_input[inputIndex];
            }
        }
    }

    d_output[c * outputSize * outputSize + i * outputSize + j] = max;
}

__global__ void Kernels::avg_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const int inputSize,
    const int outputSize,
    const int nChannels,
    const int poolingSize,
    const int stride
) {
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= outputSize || j >= outputSize || c >= outputSize) {
        return;
    }

    float sum = 0.0f;

    for (int k = 0; k < poolingSize; k++) {
        for (int l = 0; l < poolingSize; l++) {
            int inputIndex = c * inputSize * inputSize +
                             (i * stride + k) * inputSize + (j * stride + l);

            sum += d_input[inputIndex];
        }
    }

    d_output[c * outputSize * outputSize + i * outputSize + j] =
        sum / (poolingSize * poolingSize);
}