#include "avg_pooling.hpp"
#include "cuda_helper.cuh"
#include "pooling.cuh"

using namespace CUDANet::Layers;

void AvgPooling2d::initCUDA() {
    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output,
        sizeof(float) * outputSize.first * outputSize.second * nChannels
    ));
}

void AvgPooling2d::delCUDA() {
    cudaFree(d_output);
}

float* AvgPooling2d::forwardCUDA(const float* d_input) {
    dim3 block(8, 8, 8);
    dim3 grid(
        (outputSize.first + block.x - 1) / block.x,
        (outputSize.second + block.y - 1) / block.y,
        (nChannels + block.z - 1) / block.z
    );

    Kernels::avg_pooling<<<grid, block>>>(
        d_input, d_output, inputSize, outputSize, nChannels, poolingSize,
        stride, padding
    );
    CUDA_CHECK(cudaGetLastError());

    activation->activate(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}

void AdaptiveAvgPooling2d::initCUDA() {
    cudaFree(d_output);
    cudaMalloc(
        (void**)&d_output,
        sizeof(float) * outputSize.first * outputSize.second * nChannels
    );
}
