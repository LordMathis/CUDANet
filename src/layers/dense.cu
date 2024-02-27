#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>

#include "activations.cuh"
#include "cuda_helper.cuh"
#include "dense.cuh"

Layers::Dense::Dense(
    int            inputSize,
    int            outputSize,
    std::string    activation,
    cublasHandle_t cublasHandle
)
    : inputSize(inputSize),
      outputSize(outputSize),
      cublasHandle(cublasHandle),
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
    const float alpha = 1.0f;
    const float beta  = 1.0f;

    CUBLAS_CHECK(cublasSgemv(
        cublasHandle, CUBLAS_OP_N, inputSize, outputSize, &alpha, d_weights,
        inputSize, d_input, 1, &beta, d_output, 1
    ));
    CUBLAS_CHECK(
        cublasSaxpy(cublasHandle, outputSize, &alpha, d_biases, 1, d_output, 1)
    );

    int threadsPerBlock = 256;
    int blocksPerGrid   = (outputSize + threadsPerBlock - 1) / threadsPerBlock;

    if (activation == "sigmoid") {
        sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_output, d_output, outputSize
        );
    } else if (activation == "relu") {
        relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_output, d_output, outputSize
        );
    } else {
        linear_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_output, d_output, outputSize
        );
    }
}

void Layers::Dense::toCuda() {
    CUBLAS_CHECK(cublasSetMatrix(
        outputSize, inputSize, sizeof(float), weights.data(), outputSize,
        d_weights, outputSize
    ));
    CUBLAS_CHECK(cublasSetVector(
        biases.size(), sizeof(float), biases.data(), 1, d_biases, 1
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

    for (int j = 0; j < inputSize; ++j) {
        for (int i = 0; i < outputSize; ++i) {
            int idx      = IDX2C(i, j, outputSize);
            weights[idx] = weights_input[i][j];
        }
    }

    toCuda();
}

void Layers::Dense::setBiases(const std::vector<float>& biases_input) {
    std::copy(biases_input.begin(), biases_input.end(), biases.begin());
    toCuda();
}