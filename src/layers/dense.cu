#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>

#include "activations.cuh"
#include "cuda_helper.cuh"
#include "dense.cuh"
#include "matrix_math.cuh"

Layers::Dense::Dense(
    int            inputSize,
    int            outputSize,
    Activation     activation
)
    : inputSize(inputSize),
      outputSize(outputSize),
      activation(activation) {
    // Allocate memory for weights and biases
    weights.resize(outputSize * inputSize);
    biases.resize(outputSize);

    initializeWeights();
    initializeBiases();

    d_weights = nullptr;
    d_biases  = nullptr;

    // Allocate GPU memory for weights and biases
    CUDA_CHECK(
        cudaMalloc((void**)&d_weights, sizeof(float) * inputSize * outputSize)
    );
    CUDA_CHECK(cudaMalloc((void**)&d_biases, sizeof(float) * outputSize));

    toCuda();
}

Layers::Dense::~Dense() {
    // Free GPU memory
    cudaFree(d_weights);
    cudaFree(d_biases);
}

void Layers::Dense::initializeWeights() {
    std::fill(weights.begin(), weights.end(), 0.0f);
}

void Layers::Dense::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
}

void Layers::Dense::forward(const float* d_input, float* d_output) {

    mat_vec_mul_kernel<<<1, outputSize>>>(
        d_weights, d_input, d_output, inputSize, outputSize
    );

    vec_vec_add_kernel<<<1, outputSize>>>(
        d_biases, d_output, d_output, outputSize
    );

    switch (activation) {
        case SIGMOID:
            sigmoid_kernel<<<1, outputSize>>>(
                d_output, d_output, outputSize
            );
            break;

        case RELU:
            relu_kernel<<<1, outputSize>>>(
                d_output, d_output, outputSize
            );
            break;

        default:
            break;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
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

void Layers::Dense::setWeights(
    const std::vector<std::vector<float>>& weights_input
) {
    int numWeights = inputSize * outputSize;

    if (weights.size() != numWeights) {
        std::cerr << "Invalid number of weights" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            weights[i * inputSize + j] = weights_input[i][j];
        }
    }

    toCuda();
}

void Layers::Dense::setBiases(const std::vector<float>& biases_input) {
    std::copy(biases_input.begin(), biases_input.end(), biases.begin());
    toCuda();
}