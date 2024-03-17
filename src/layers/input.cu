#include "cuda_helper.cuh"
#include "input.cuh"

using namespace CUDANet;

Layers::Input::Input(int inputSize) : inputSize(inputSize) {
    d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(float) * inputSize));
}

Layers::Input::~Input() {
    cudaFree(d_output);
}

/*
Copies host input to device d_output

Args
    const float* input Host pointer to input data
    float* d_output Device pointer to input data copied to device
*/
float* Layers::Input::forward(const float* input) {
    CUDA_CHECK(cudaMemcpy(
        d_output, input, sizeof(float) * inputSize, cudaMemcpyHostToDevice
    ));

    return d_output;
}

void Layers::Input::setWeights(const float* weights) {}
void Layers::Input::setBiases(const float* biases) {}

void Layers::Input::initializeWeights() {}
void Layers::Input::initializeBiases() {}

void Layers::Input::toCuda() {}