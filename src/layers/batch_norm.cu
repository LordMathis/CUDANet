#include <vector>

#include "activation.cuh"
#include "batch_norm.cuh"
#include "cuda_helper.cuh"
#include "layer.cuh"
#include "matmul.cuh"
#include "vector.cuh"

using namespace CUDANet::Layers;

BatchNorm2d::BatchNorm2d(
    shape2d          inputSize,
    int            inputChannels,
    float          epsilon,
    ActivationType activationType
)
    : inputSize(inputSize), inputChannels(inputChannels) {
    activation = new Activation(
        activationType, inputSize.first * inputSize.second * inputChannels
    );

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void **)&d_output,
        sizeof(float) * inputSize.first * inputSize.second * inputChannels
    ));

    d_running_mean = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void **)&d_running_mean, sizeof(float) * inputChannels
    ));

    d_running_var = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void **)&d_running_var, sizeof(float) * inputChannels
    ));

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

    weights.resize(inputChannels);
    biases.resize(inputChannels);

    running_mean.resize(inputChannels);
    running_var.resize(inputChannels);

    initializeWeights();
    initializeBiases();
    initializeRunningMean();
    initializeRunningVar();

    toCuda();

    gridSize =
        (inputSize.first * inputSize.second + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

BatchNorm2d::~BatchNorm2d() {
    cudaFree(d_output);
    cudaFree(d_running_mean);
    cudaFree(d_running_var);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_length);
    cudaFree(d_epsilon);
}

void BatchNorm2d::initializeWeights() {
    std::fill(weights.begin(), weights.end(), 1.0f);
}

void BatchNorm2d::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
}

void BatchNorm2d::initializeRunningMean() {
    std::fill(running_mean.begin(), running_mean.end(), 0.0f);
}

void BatchNorm2d::initializeRunningVar() {
    std::fill(running_var.begin(), running_var.end(), 1.0f);
}

void BatchNorm2d::setWeights(const float *weights_input) {
    std::copy(weights_input, weights_input + weights.size(), weights.begin());
    toCuda();
}

std::vector<float> BatchNorm2d::getWeights() {
    return weights;
}

void BatchNorm2d::setBiases(const float *biases_input) {
    std::copy(biases_input, biases_input + biases.size(), biases.begin());
    toCuda();
}

std::vector<float> BatchNorm2d::getBiases() {
    return biases;
}

void BatchNorm2d::setRunningMean(const float* running_mean_input) {
    std::copy(running_mean_input, running_mean_input + inputChannels, running_mean.begin());
    toCuda();
}

std::vector<float> BatchNorm2d::getRunningMean() {
    return running_mean;
}

void BatchNorm2d::setRunningVar(const float* running_var_input) {
    std::copy(running_var_input, running_var_input + inputChannels, running_var.begin());
    toCuda();
}

std::vector<float> BatchNorm2d::getRunningVar() {
    return running_var;
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

int BatchNorm2d::getInputSize() {
    return inputSize.first * inputSize.second * inputChannels;
}

int BatchNorm2d::getOutputSize() {
    return inputSize.first * inputSize.second * inputChannels;
}

shape2d BatchNorm2d::getOutputDims() {
    return inputSize;
}

float *BatchNorm2d::forward(const float *d_input) {
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
            &d_running_var[i],
            d_epsilon,
            inputSize.first * inputSize.second
        );
        CUDA_CHECK(cudaGetLastError());

        // Multiply by weights
        Kernels::vec_scalar_mul<<<gridSize, BLOCK_SIZE>>>(
            d_output + i * inputSize.first * inputSize.second,
            d_output + i * inputSize.first * inputSize.second,
            &d_weights[i],
            inputSize.first * inputSize.second
        );
        CUDA_CHECK(cudaGetLastError());

        // Add biases
        Kernels::vec_scalar_add<<<gridSize, BLOCK_SIZE>>>(
            d_output + i * inputSize.first * inputSize.second,
            d_output + i * inputSize.first * inputSize.second,
            &d_biases[i],
            inputSize.first * inputSize.second
        );
        CUDA_CHECK(cudaGetLastError());
    }

    activation->activate(d_output);

    return d_output;
}