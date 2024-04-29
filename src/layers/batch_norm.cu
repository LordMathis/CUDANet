#include <vector>

#include "activation.cuh"
#include "batch_norm.cuh"
#include "cuda_helper.cuh"
#include "layer.cuh"
#include "matmul.cuh"
#include "vector.cuh"

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
    CUDA_CHECK(cudaMalloc((void **)&d_mean, sizeof(float) * inputSize * inputSize));

    d_mean_sub = nullptr;
    CUDA_CHECK(
        cudaMalloc((void **)&d_mean_sub, sizeof(float) * inputSize * inputSize)
    );

    d_sqrt_var = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_sqrt_var, sizeof(float) * inputSize * inputSize));

    d_weights = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_weights, sizeof(float) * inputChannels));

    d_biases = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_biases, sizeof(float) * inputChannels));

    d_length = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_length, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_length, inputSize * inputSize, sizeof(float)));

    d_epsilon = nullptr;
    float epsilon = 1e-5f;
    CUDA_CHECK(cudaMalloc((void **)&d_epsilon, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice));

    weights.resize(inputChannels);
    biases.resize(inputChannels);

    initializeWeights();
    initializeBiases();

    toCuda();

    gridSize =
        (inputSize * inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

BatchNorm::~BatchNorm() {
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_mean_sub);
    cudaFree(d_sqrt_var);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_length);
    cudaFree(d_epsilon);
}

void BatchNorm::initializeWeights() {
    std::fill(weights.begin(), weights.end(), 1.0f);
}

void BatchNorm::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
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
}

int BatchNorm::getInputSize() {
    return inputSize * inputSize * inputChannels;
}

int BatchNorm::getOutputSize() {
    return inputSize * inputSize * inputChannels;
}

float *BatchNorm::forward(const float *d_input) {
    
    // Compute per-channel batch normalization
    for (int i = 0; i < inputChannels; i++) {

        // Compute mean
        // Sum over all values
        Utils::sum(
            d_input + i * inputSize * inputSize,
            d_mean,
            inputSize * inputSize
        );

        // Divide sum by length -> mean
        Kernels::vec_scalar_div<<<gridSize, BLOCK_SIZE>>>(
            d_mean,
            d_mean,
            d_length,
            inputSize * inputSize
        );
        CUDA_CHECK(cudaGetLastError());

        // Subtract mean from input
        Kernels::vec_scalar_sub<<<gridSize, BLOCK_SIZE>>>(
            d_input + i * inputSize * inputSize,
            d_mean_sub,
            &d_mean[0],
            inputSize * inputSize
        );
        CUDA_CHECK(cudaGetLastError());

        // Compute variance
        // Square differences of input - mean
        Kernels::vec_vec_mul<<<gridSize, BLOCK_SIZE>>>(
            d_mean_sub,
            d_mean_sub,
            d_sqrt_var,
            inputSize * inputSize
        );
        CUDA_CHECK(cudaGetLastError());

        // Sum over all differences
        Utils::sum(
            d_sqrt_var,
            d_sqrt_var,
            inputSize * inputSize
        );

        // Divide by difference sum / length -> variance
        Kernels::vec_scalar_div<<<gridSize, BLOCK_SIZE>>>(
            d_sqrt_var,
            d_sqrt_var,
            d_length,
            inputSize * inputSize
        );
        CUDA_CHECK(cudaGetLastError());

        // Add epsilon to variance to avoid division by zero
        Kernels::vec_scalar_add<<<gridSize, BLOCK_SIZE>>>(
            d_sqrt_var,
            d_sqrt_var,
            &d_epsilon[0],
            inputSize * inputSize
        );
        CUDA_CHECK(cudaGetLastError());

        // Compute squared root of variance
        Kernels::vec_sqrt<<<gridSize, BLOCK_SIZE>>>(
            d_sqrt_var,
            d_sqrt_var,
            inputSize * inputSize
        );
        CUDA_CHECK(cudaGetLastError());

        // Divide by squared root of variance
        Kernels::vec_scalar_div<<<gridSize, BLOCK_SIZE>>>(
            d_mean_sub,
            d_output + i * inputSize * inputSize,
            &d_sqrt_var[0],
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