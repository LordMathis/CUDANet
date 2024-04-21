#include "cuda_helper.cuh"
#include "input.cuh"

using namespace CUDANet::Layers;

Input::Input(int inputSize) : inputSize(inputSize) {
    d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(float) * inputSize));
}

Input::~Input() {
    cudaFree(d_output);
}

float* Input::forward(const float* input) {
    CUDA_CHECK(cudaMemcpy(
        d_output, input, sizeof(float) * inputSize, cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}
