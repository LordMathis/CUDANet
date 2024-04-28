#include <vector>

#include "activation.cuh"
#include "batch_norm.cuh"
#include "cuda_helper.cuh"
#include "layer.cuh"
#include "matmul.cuh"

using namespace CUDANet::Layers;

BatchNorm::BatchNorm(
    int            inputSize,
    int            inputChannels,
    ActivationType activationType
)
    : inputSize(inputSize), inputChannels(inputChannels) {
    activation =
        new Activation(activationType, inputSize * inputSize * inputChannels);

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void **)&d_output,
        sizeof(float) * inputSize * inputSize * inputChannels
    ));

    d_mean = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_mean, sizeof(float) * inputChannels));

    d_sqrt_var = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_sqrt_var, sizeof(float) * inputChannels));

    d_weights = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_weights, sizeof(float) * inputChannels));

    d_biases = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_biases, sizeof(float) * inputChannels));

    weights.resize(inputChannels);
    biases.resize(inputChannels);
    mean.resize(inputChannels);
    sqrt_var.resize(inputChannels);

    initializeWeights();
    initializeBiases();
    initializeMean();
    initializeSqrtVar();

    toCuda();

    gridSize =
        (inputSize * inputSize * inputChannels + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

BatchNorm::~BatchNorm() {
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_sqrt_var);
    cudaFree(d_weights);
    cudaFree(d_biases);
}

void BatchNorm::initializeWeights() {
    std::fill(weights.begin(), weights.end(), 1.0f);
}

void BatchNorm::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
}

void BatchNorm::initializeMean() {
    std::fill(mean.begin(), mean.end(), 0.0f);
}

void BatchNorm::initializeSqrtVar() {
    std::fill(sqrt_var.begin(), sqrt_var.end(), 1.0f);
}

void BatchNorm::setWeights(const float *weights_input) {
    std::copy(weights_input, weights_input + weights.size(), weights.begin());
    toCuda();
}

std::vector<float> BatchNorm::getWeights() {
    return weights;
}

void BatchNorm::setBiases(const float *biases_input) {
    std::copy(biases_input, biases_input + biases.size(), biases.begin());
    toCuda();
}

std::vector<float> BatchNorm::getBiases() {
    return biases;
}

void BatchNorm::toCuda() {
    CUDA_CHECK(cudaMemcpy(
        d_weights, weights.data(), sizeof(float) * inputChannels,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_biases, biases.data(), sizeof(float) * inputChannels,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_mean, mean.data(), sizeof(float) * inputChannels,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_sqrt_var, sqrt_var.data(), sizeof(float) * inputChannels,
        cudaMemcpyHostToDevice
    ));
}

int BatchNorm::getInputSize() {
    return inputSize * inputSize * inputChannels;
}

int BatchNorm::getOutputSize() {
    return inputSize * inputSize * inputChannels;
}

float *BatchNorm::forward(const float *d_input) {
    
    for (int i = 0; i < inputChannels; i++) {
        Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
            d_input + i * inputSize * inputSize,
            d_output + i * inputSize * inputSize,
            &d_mean[i],
            inputSize * inputSize
        );
        CUDA_CHECK(cudaGetLastError());

        Kernels::vec_scalar_div<<<gridSize, BLOCK_SIZE>>>(
            d_output + i * inputSize * inputSize,
            d_output + i * inputSize * inputSize,
            &d_sqrt_var[i],
            inputSize * inputSize
        );
        CUDA_CHECK(cudaGetLastError());

        Kernels::vec_scalar_mul<<<gridSize, BLOCK_SIZE>>>(
            d_output + i * inputSize * inputSize,
            d_output + i * inputSize * inputSize,
            &d_weights[i],
            inputSize * inputSize
        );
        CUDA_CHECK(cudaGetLastError());

        Kernels::vec_scalar_add<<<gridSize, BLOCK_SIZE>>>(
            d_output + i * inputSize * inputSize,
            d_output + i * inputSize * inputSize,
            &d_biases[i],
            inputSize * inputSize
        );
        CUDA_CHECK(cudaGetLastError());
    }

    return d_output;
}