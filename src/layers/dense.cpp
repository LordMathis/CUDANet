#include "dense.h"
#include "cuda_helper.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <random>

Layers::Dense::Dense(int inputSize, int outputSize, cublasHandle_t cublasHandle)
    : inputSize(inputSize), outputSize(outputSize), cublasHandle(cublasHandle) {

    // Allocate memory for weights and biases
    weights.resize(outputSize * inputSize);
    biases.resize(outputSize);

    initializeWeights();
    initializeBiases();

    // Allocate GPU memory for weights and biases
    CUDA_CHECK(cudaMalloc((void**)&d_weights, sizeof(float) * inputSize * outputSize));
    CUDA_CHECK(cudaMalloc((void**)&d_biases, sizeof(float) * biases.size()));

    toCuda();
}

Layers::Dense::~Dense() {
    // Free GPU memory
    cudaFree(d_weights);
    cudaFree(d_biases);
}

void Layers::Dense::initializeWeights() {
    int numWeights = inputSize * outputSize;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.01f); // Xavier initialization

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            int idx = IDX2C(i, j, inputSize);
            weights[idx] = dist(gen);
        }
    }
}

void Layers::Dense::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.1f);
}

void Layers::Dense::forward(const float* d_input, float* d_output) {
    const float alpha = 1.0f;
    const float beta = 1.0f;

    cublasSgemv(cublasHandle, CUBLAS_OP_N, inputSize, outputSize, &alpha, d_weights, inputSize, d_input, 1, &beta, d_output, 1);
    cublasSaxpy(cublasHandle, outputSize, &alpha, d_biases, 1, d_output, 1);
}

void Layers::Dense::toCuda() {
    CUBLAS_CHECK(cublasSetMatrix(outputSize, inputSize, sizeof(float), weights.data(), inputSize, d_weights, outputSize));
    CUBLAS_CHECK(cublasSetVector(biases.size(), sizeof(float), biases.data(), 1, d_biases, 1));
}

void Layers::Dense::setWeights(const std::vector<std::vector<float>>& weights_input) {
    int numWeights = inputSize * outputSize;

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            int idx = IDX2C(i, j, inputSize);
            weights[idx] = weights_input[i][j];
        }
    }

    toCuda();
}

void Layers::Dense::setBiases(const std::vector<float>& biases_input) {
    std::copy(biases_input.begin(), biases_input.end(), biases.begin());
    toCuda();
}