#include <iostream>
#include <vector>

#include "activation.cuh"
#include "activation_functions.cuh"
#include "cuda_helper.cuh"
#include "matmul.cuh"
#include "vector.cuh"

using namespace CUDANet::Layers;

Activation::Activation(ActivationType activation, const int length)
    : activationType(activation), length(length) {
    if (activationType == SOFTMAX) {
        d_max = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_max, sizeof(float) * length));

        d_softmax_sum = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_softmax_sum, sizeof(float) * length));
    }

    gridSize = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

Activation::~Activation() {
    if (activationType == SOFTMAX) {
        cudaFree(d_softmax_sum);
        cudaFree(d_max);
    }
}

void Activation::activate(float* d_input) {

    // float sum = 0.0f;

    switch (activationType) {
        case SIGMOID:
            Kernels::sigmoid<<<gridSize, BLOCK_SIZE>>>(
                d_input, d_input, length
            );
            CUDA_CHECK(cudaGetLastError());
            break;

        case RELU:
            Kernels::relu<<<gridSize, BLOCK_SIZE>>>(d_input, d_input, length);
            CUDA_CHECK(cudaGetLastError());
            break;
        case SOFTMAX:

            // Find max value
            Utils::max(d_input, d_max, length);

            // Subtract max value to improve numerical stability
            Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
                d_input, d_input, d_max, length
            );
            CUDA_CHECK(cudaGetLastError());

            // Compute exponentials
            Kernels::vec_exp<<<gridSize, BLOCK_SIZE>>>(
                d_input, d_input, length
            );
            CUDA_CHECK(cudaGetLastError());

            // Find sum
            Utils::sum(d_input, d_softmax_sum, length);

            Kernels::vec_scalar_div<<<gridSize, BLOCK_SIZE>>>(
                d_input, d_input, d_softmax_sum, length
            );
            CUDA_CHECK(cudaGetLastError());

            break;

        default:
            break;    
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

