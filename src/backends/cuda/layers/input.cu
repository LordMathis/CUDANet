#include "cuda_helper.cuh"
#include "input.hpp"

using namespace CUDANet::Layers;

void Input::initCUDA() {
    d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(float) * inputSize));
}

void Input::delCUDA() {
    cudaFree(d_output);
}

float* Input::forwardCUDA(const float* input) {
    CUDA_CHECK(cudaMemcpy(
        d_output, input, sizeof(float) * inputSize, cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}