#include "pooling.cuh"

#include "cuda_helper.cuh"

using namespace CUDANet;

__global__ void Kernels::max_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const int inputSize,
    const int nChannels,
    const int poolingSize,
    const int stride,
    const int paddingSize
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= inputSize * inputSize * nChannels) {
        return;
    }

    // Get output index
    int c = tid / (inputSize * inputSize);
    int i = tid % (inputSize * inputSize) / inputSize;
    int j = tid % inputSize;

    float max = 0.0f;

    for (int k = 0; k < poolingSize; k++) {
        for (int l = 0; l < poolingSize; l++) {

            if (i * stride + k < paddingSize ||
                i * stride + k >= (inputSize + paddingSize) ||
                j * stride + l < paddingSize ||
                j * stride + l >= (inputSize + paddingSize)) {
                continue;
            }


            int inputIndex = c * inputSize * inputSize +
                             (i * stride + k - paddingSize) * inputSize +
                             (j * stride + l - paddingSize);

            if (d_input[inputIndex] > max) {
                max = d_input[inputIndex];
            }
        }
    }

    d_output[tid] = max;
}

__global__ void Kernels::avg_pooling(
    const float* __restrict__ d_input,
    float* __restrict__ d_output,
    const int inputSize,
    const int nChannels,
    const int poolingSize,
    const int stride,
    const int paddingSize
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= inputSize * inputSize * nChannels) {
        return;
    }

    // Get output index
    int c = tid / (inputSize * inputSize);
    int i = tid % (inputSize * inputSize) / inputSize;
    int j = tid % inputSize;

    float sum = 0.0f;

    for (int k = 0; k < poolingSize; k++) {
        for (int l = 0; l < poolingSize; l++) {

            if (i * stride + k < paddingSize ||
                i * stride + k >= (inputSize + paddingSize) ||
                j * stride + l < paddingSize ||
                j * stride + l >= (inputSize + paddingSize)) {
                continue;
            }

            int inputIndex = c * inputSize * inputSize +
                             (i * stride + k - paddingSize) * inputSize +
                             (j * stride + l - paddingSize);

            sum += d_input[inputIndex];
        }
    }

    d_output[tid] = sum / (poolingSize * poolingSize);
}