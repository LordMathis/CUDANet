#include <vector>

#include "activation.hpp"
#include "batch_norm.hpp"
#include "cuda_helper.cuh"
#include "layer.hpp"
#include "matmul.cuh"
#include "vector.cuh"

using namespace CUDANet::Layers;

void BatchNorm2d::initCUDA() {
    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void **)&d_output,
        sizeof(float) * inputSize.first * inputSize.second * inputChannels
    ));

    d_running_mean = nullptr;
    CUDA_CHECK(
        cudaMalloc((void **)&d_running_mean, sizeof(float) * inputChannels)
    );

    d_running_var = nullptr;
    CUDA_CHECK(
        cudaMalloc((void **)&d_running_var, sizeof(float) * inputChannels)
    );

    d_weights = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_weights, sizeof(float) * inputChannels));

    d_biases = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_biases, sizeof(float) * inputChannels));

    d_length     = nullptr;
    float length = (float)inputSize.first * inputSize.second;
    CUDA_CHECK(cudaMalloc((void **)&d_length, sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(d_length, &length, sizeof(float), cudaMemcpyHostToDevice)
    );

    d_epsilon = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_epsilon, sizeof(float)));
    CUDA_CHECK(
        cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice)
    );

    gridSize =
        (inputSize.first * inputSize.second + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

void BatchNorm2d::delCUDA() {
    cudaFree(d_output);
    cudaFree(d_running_mean);
    cudaFree(d_running_var);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_length);
    cudaFree(d_epsilon);
}

void BatchNorm2d::toCuda() {
    CUDA_CHECK(cudaMemcpy(
        d_weights, weights.data(), sizeof(float) * inputChannels,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_biases, biases.data(), sizeof(float) * inputChannels,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_running_mean, running_mean.data(), sizeof(float) * inputChannels,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_running_var, running_var.data(), sizeof(float) * inputChannels,
        cudaMemcpyHostToDevice
    ));
}

float *BatchNorm2d::forwardCUDA(const float *d_input) {
    // Compute per-channel batch normalization
    for (int i = 0; i < inputChannels; i++) {
        // Subtract mean from input
        Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
            d_input + i * inputSize.first * inputSize.second,
            d_output + i * inputSize.first * inputSize.second,
            &d_running_mean[i], inputSize.first * inputSize.second
        );
        CUDA_CHECK(cudaGetLastError());

        // Divide by sqrt(running_var + epsilon)
        Kernels::vec_scale<<<gridSize, BLOCK_SIZE>>>(
            d_output + i * inputSize.first * inputSize.second,
            d_output + i * inputSize.first * inputSize.second,
            &d_running_var[i], d_epsilon, inputSize.first * inputSize.second
        );
        CUDA_CHECK(cudaGetLastError());

        // Multiply by weights
        Kernels::vec_scalar_mul<<<gridSize, BLOCK_SIZE>>>(
            d_output + i * inputSize.first * inputSize.second,
            d_output + i * inputSize.first * inputSize.second, &d_weights[i],
            inputSize.first * inputSize.second
        );
        CUDA_CHECK(cudaGetLastError());

        // Add biases
        Kernels::vec_scalar_add<<<gridSize, BLOCK_SIZE>>>(
            d_output + i * inputSize.first * inputSize.second,
            d_output + i * inputSize.first * inputSize.second, &d_biases[i],
            inputSize.first * inputSize.second
        );
        CUDA_CHECK(cudaGetLastError());
    }

    activation->activate(d_output);

    return d_output;
}