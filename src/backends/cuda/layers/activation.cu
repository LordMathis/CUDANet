#include <vector>

#include "activation.hpp"
#include "activation_functions.cuh"
#include "cuda_helper.cuh"
#include "matmul.cuh"
#include "vector.cuh"

using namespace CUDANet::Layers;

void Activation::initCUDA() {
    if (activationType == SOFTMAX) {
        d_softmax_sum = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_softmax_sum, sizeof(float) * length));

        d_max = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_max, sizeof(float) * length));
    }

    gridSize = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

void Activation::delCUDA() {
    if (activationType == SOFTMAX) {
        CUDA_CHECK(cudaFree(d_softmax_sum));
        CUDA_CHECK(cudaFree(d_max));
    }
}

void Activation::activateCUDA(float* d_input) {

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
                d_input, d_input, &d_max[0], length
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
                d_input, d_input, &d_softmax_sum[0], length
            );
            CUDA_CHECK(cudaGetLastError());
            break;

        default:
            break;    
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}
