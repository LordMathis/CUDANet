#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>

#include "vector.cuh"
#include "activation.hpp"
#include "cuda_helper.cuh"
#include "dense.hpp"
#include "matmul.cuh"

using namespace CUDANet::Layers;

void Dense::initCUDA() {
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

void Dense::delCUDA() {
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}

void Dense::toCuda() {
    CUDA_CHECK(cudaMemcpy(
        d_weights, weights.data(), sizeof(float) * inputSize * outputSize,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_biases, biases.data(), sizeof(float) * outputSize,
        cudaMemcpyHostToDevice
    ));
}

float* Dense::forwardCUDA(const float* d_input) {
    Kernels::mat_vec_mul<<<forwardGridSize, BLOCK_SIZE>>>(
        d_weights, d_input, d_output, inputSize, outputSize
    );
    CUDA_CHECK(cudaGetLastError());

    Kernels::vec_vec_add<<<biasGridSize, BLOCK_SIZE>>>(
        d_biases, d_output, d_output, outputSize
    );
    CUDA_CHECK(cudaGetLastError());

    activation->activate(d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}
