#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>

#include "activations.cuh"
#include "cuda_helper.cuh"
#include "dense.cuh"
#include "matmul.cuh"

using namespace CUDANet;

Layers::Dense::Dense(
    int                inputSize,
    int                outputSize,
    Layers::Activation activation
)
    : inputSize(inputSize), outputSize(outputSize), activation(activation) {
    // Allocate memory for weights and biases
    weights.resize(outputSize * inputSize);
    biases.resize(outputSize);

    initializeWeights();
    initializeBiases();

    d_output = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(float) * outputSize));

    d_weights = nullptr;
    d_biases  = nullptr;

    // Allocate GPU memory for weights and biases
    CUDA_CHECK(
        cudaMalloc((void**)&d_weights, sizeof(float) * inputSize * outputSize)
    );
    CUDA_CHECK(cudaMalloc((void**)&d_biases, sizeof(float) * outputSize));
    toCuda();

    // Calculate block and grid sizes
    forwardGridSize =
        (std::max(inputSize, outputSize) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    biasGridSize = (outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

Layers::Dense::~Dense() {
    // Free GPU memory
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}

void Layers::Dense::initializeWeights() {
    std::fill(weights.begin(), weights.end(), 0.0f);
}

void Layers::Dense::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
}

float* Layers::Dense::forward(const float* d_input) {
    Kernels::mat_vec_mul<<<forwardGridSize, BLOCK_SIZE>>>(
        d_weights, d_input, d_output, inputSize, outputSize
    );

    Kernels::vec_vec_add<<<biasGridSize, BLOCK_SIZE>>>(
        d_biases, d_output, d_output, outputSize
    );

    switch (activation) {
        case SIGMOID:
            Kernels::sigmoid<<<biasGridSize, BLOCK_SIZE>>>(
                d_output, d_output, outputSize
            );
            break;

        case RELU:
            Kernels::relu<<<biasGridSize, BLOCK_SIZE>>>(
                d_output, d_output, outputSize
            );
            break;

        default:
            break;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}

void Layers::Dense::toCuda() {
    CUDA_CHECK(cudaMemcpy(
        d_weights, weights.data(), sizeof(float) * inputSize * outputSize,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_biases, biases.data(), sizeof(float) * outputSize,
        cudaMemcpyHostToDevice
    ));
}

void Layers::Dense::setWeights(const float* weights_input) {
    std::copy(weights_input, weights_input + weights.size(), weights.begin());
    toCuda();
}

void Layers::Dense::setBiases(const float* biases_input) {
    std::copy(biases_input, biases_input + biases.size(), biases.begin());
    toCuda();
}