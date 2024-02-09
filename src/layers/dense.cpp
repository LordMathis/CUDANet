#include "dense.h"
#include <cublas_v2.h>


Layers::Dense::Dense(int inputSize, int outputSize, cublasHandle_t cublasHandle)
    : inputSize(inputSize), outputSize(outputSize), cublasHandle(cublasHandle) {

    // Allocate memory for weights and biases
    weights.resize(outputSize, std::vector<float>(inputSize));
    biases.resize(outputSize);

    initializeWeights();
    initializeBiases();

    // Allocate GPU memory for weights and biases
    cudaMalloc((void**)&d_weights, sizeof(float) * inputSize * outputSize);
    cudaMalloc((void**)&d_biases, sizeof(float) * biases.size());

    // Copy weights and biases to GPU
    cudaMemcpy(d_weights, weights.data(), sizeof(float) * inputSize * outputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases.data(), sizeof(float) * biases.size(), cudaMemcpyHostToDevice);
}

Layers::Dense::~Dense() {
    // Free GPU memory
    cudaFree(d_weights);
    cudaFree(d_biases);
}

void Layers::Dense::initializeWeights() {
    for (auto& row : weights) {
        for (float& weight : row) {
            weight = 0.0f;
        }
    }
}

void Layers::Dense::initializeBiases() {
    for (float& bias : biases) {
        bias = 0.0f;
    }
}

void Layers::Dense::forward(const float* input, float* output) {
    // Perform matrix multiplication: output = weights * input + biases
    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasSgemv(cublasHandle, CUBLAS_OP_N, inputSize, outputSize, &alpha, d_weights, inputSize, input, 1, &beta, output, 1);

    // Add biases
    cublasSaxpy(cublasHandle, outputSize, &alpha, d_biases, 1, output, 1);
}