#include <iostream>
#include <vector>

#include "activation.cuh"
#include "conv2d.cuh"
#include "convolution.cuh"
#include "cuda_helper.cuh"
#include "layer.cuh"
#include "matmul.cuh"
#include "vector.cuh"

using namespace CUDANet::Layers;

Conv2d::Conv2d(
    shape2d          inputSize,
    int            inputChannels,
    shape2d          kernelSize,
    shape2d          stride,
    int            numFilters,
    shape2d          paddingSize,
    ActivationType activationType
)
    : inputSize(inputSize),
      inputChannels(inputChannels),
      kernelSize(kernelSize),
      stride(stride),
      numFilters(numFilters),
      paddingSize(paddingSize) {

    outputSize = {
        (inputSize.first - kernelSize.first + 2 * paddingSize.first) /
                stride.first + 1,
        (inputSize.second - kernelSize.second + 2 * paddingSize.second) /
                stride.second + 1
    };

    activation =
        new Activation(activationType, outputSize.first * outputSize.second * numFilters);

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output, sizeof(float) * outputSize.first * outputSize.second * numFilters
    ));

    weights.resize(kernelSize.first * kernelSize.second * inputChannels * numFilters);
    initializeWeights();

    d_weights = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_weights,
        sizeof(float) * kernelSize.first * kernelSize.second * inputChannels * numFilters
    ));

    biases.resize(numFilters);
    initializeBiases();

    d_biases = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_biases, sizeof(float) * numFilters));

    toCuda();
}

Conv2d::~Conv2d() {
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
    delete activation;
}

void Conv2d::initializeWeights() {
    std::fill(weights.begin(), weights.end(), 0.0f);
}

void Conv2d::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
}

void Conv2d::setWeights(const float* weights_input) {
    std::copy(weights_input, weights_input + weights.size(), weights.begin());
    toCuda();
}

std::vector<float> Conv2d::getWeights() {
    return weights;
}

void Conv2d::setBiases(const float* biases_input) {
    std::copy(biases_input, biases_input + biases.size(), biases.begin());
    toCuda();
}

std::vector<float> Conv2d::getBiases() {
    return biases;
}

void Conv2d::toCuda() {
    CUDA_CHECK(cudaMemcpy(
        d_weights, weights.data(),
        sizeof(float) * kernelSize.first * kernelSize.second * inputChannels * numFilters,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_biases, biases.data(), sizeof(float) * numFilters,
        cudaMemcpyHostToDevice
    ));
}

float* Conv2d::forward(const float* d_input) {
    // Convolve
    dim3 block(8, 8, 8);
    dim3 grid(
        (outputSize.first + block.x - 1) / block.x,
        (outputSize.second + block.y - 1) / block.y,
        (numFilters + block.z - 1) / block.z
    );

    CUDANet::Utils::clear(d_output, outputSize.first * outputSize.second * numFilters);

    Kernels::convolution<<<grid, block>>>(
        d_input, d_weights, d_biases, d_output, inputSize, inputChannels,
        paddingSize, kernelSize, stride, numFilters, outputSize
    );
    CUDA_CHECK(cudaGetLastError());

    // Apply activation
    activation->activate(d_output);

    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}

int Conv2d::getOutputSize() {
    return outputSize.first * outputSize.second * numFilters;
}

int Conv2d::getInputSize() {
    return inputSize.first * inputSize.second * inputChannels;
}

shape2d Conv2d::getOutputDims() {
    return outputSize;
}