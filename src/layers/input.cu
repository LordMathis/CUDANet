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

/*
Copies host input to device d_output

Args
    const float* input Host pointer to input data
    float* d_output Device pointer to input data copied to device
*/
float* Input::forward(const float* input) {
    CUDA_CHECK(cudaMemcpy(
        d_output, input, sizeof(float) * inputSize, cudaMemcpyHostToDevice
    ));

    return d_output;
}
