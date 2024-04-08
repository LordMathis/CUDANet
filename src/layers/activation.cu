#include "activation.cuh"

#include "cuda_helper.cuh"
#include "activation_functions.cuh"

using namespace CUDANet::Layers;

Activation::Activation(ActivationType activation, const unsigned int length)
    : activationType(activation), length(length) {

    if (activationType == SOFTMAX) {
        d_softmax_sum = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_softmax_sum, sizeof(float) * length));
    }

    gridSize = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

Activation::~Activation() {
    if (activationType == SOFTMAX) {
        cudaFree(d_softmax_sum);
    }
}

void Activation::activate(float* __restrict__ d_input) {

    switch (activationType) {
        case SIGMOID:
            Kernels::sigmoid<<<gridSize, BLOCK_SIZE>>>(
                d_input, d_input, length
            );
            break;

        case RELU:
            Kernels::relu<<<gridSize, BLOCK_SIZE>>>(
                d_input, d_input, length
            );
            break;
        case SOFTMAX:
            Kernels::softmax_exp<<<gridSize, BLOCK_SIZE>>>(
                d_input, d_input, length
            );

            Kernels::softmax_sum<<<gridSize, BLOCK_SIZE>>>(
                d_input, d_softmax_sum
            );

            Kernels::softmax_sum<<<1, BLOCK_SIZE>>>(
                d_softmax_sum, d_softmax_sum
            ); 

            Kernels::softmax_div<<<gridSize, BLOCK_SIZE>>>(
                d_input, d_input, d_softmax_sum, length
            );
            break;

        default:
            break;
    }
}