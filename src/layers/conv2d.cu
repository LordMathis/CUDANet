#include <iostream>
#include <string>

#include "activations.cuh"
#include "conv2d.cuh"
#include "convolution.cuh"
#include "cuda_helper.cuh"
#include "matmul.cuh"

Layers::Conv2d::Conv2d(
    int                inputSize,
    int                inputChannels,
    int                kernelSize,
    int                stride,
    Layers::Padding    padding,
    int                numFilters,
    Layers::Activation activation
)
    : inputSize(inputSize),
      inputChannels(inputChannels),
      kernelSize(kernelSize),
      stride(stride),
      numFilters(numFilters),
      activation(activation) {
    switch (padding) {
        case SAME:
            outputSize  = inputSize;
            paddingSize = ((stride - 1) * inputSize - stride + kernelSize) / 2;
            break;

        case VALID:
            paddingSize = 0;
            outputSize  = (inputSize - kernelSize) / stride + 1;
            break;

        default:
            break;
    }

    weights.resize(kernelSize * kernelSize * inputChannels * numFilters);
    initializeWeights();

    d_weights = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_weights,
        sizeof(float) * kernelSize * kernelSize * inputChannels * numFilters
    ));

    biases.resize(outputSize * outputSize * numFilters);
    initializeBiases();

    d_biases = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_biases, sizeof(float) * outputSize * outputSize * numFilters
    ));

    d_padded = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_padded, sizeof(float) * (inputSize + 2 * paddingSize) *
                               (inputSize + 2 * paddingSize) * inputChannels
    ));

    toCuda();
}

Layers::Conv2d::~Conv2d() {
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_padded);
}

void Layers::Conv2d::initializeWeights() {
    std::fill(weights.begin(), weights.end(), 0.0f);
}

void Layers::Conv2d::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
}

void Layers::Conv2d::setWeights(const float* weights_input) {
    std::copy(weights_input, weights_input + weights.size(), weights.begin());
    toCuda();
}

void Layers::Conv2d::setBiases(const float* biases_input) {
    std::copy(biases_input, biases_input + biases.size(), biases.begin());
    toCuda();
}

void Layers::Conv2d::toCuda() {
    CUDA_CHECK(cudaMemcpy(
        d_weights, weights.data(),
        sizeof(float) * kernelSize * kernelSize * inputChannels * numFilters,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_biases, biases.data(),
        sizeof(float) * outputSize * outputSize * numFilters,
        cudaMemcpyHostToDevice
    ));
}

void Layers::Conv2d::forward(const float* d_input, float* d_output) {
    // Pad input
    int THREADS_PER_BLOCK = (inputSize + 2 * paddingSize) *
                            (inputSize + 2 * paddingSize) * inputChannels;

    Kernels::padding<<<1, THREADS_PER_BLOCK>>>(
        d_input, d_padded, inputSize, inputSize, inputChannels, paddingSize
    );

    // Convolve
    THREADS_PER_BLOCK = outputSize * outputSize * numFilters;
    Kernels::convolution<<<1, THREADS_PER_BLOCK>>>(
        d_padded, d_weights, d_output, inputSize + (2 * paddingSize),
        inputChannels, kernelSize, stride, numFilters, outputSize
    );

    // Add bias
    Kernels::vec_vec_add<<<1, biases.size()>>>(
        d_biases, d_output, d_output, biases.size()
    );

    CUDA_CHECK(cudaDeviceSynchronize());
}

/*
Convolves input vector with kernel and stores result in output

input: matrix (inputSize + paddingSize) x (inputSize + paddingSize) x
inputChannels represented as a vector output: output matrix outputSize x
outputSize x numFilters

*/
void Layers::Conv2d::host_conv(const float* input, float* output) {
    // Iterate over output matrix
    for (int tid = 0; tid < outputSize * outputSize * numFilters; tid++) {
        // Get output index
        int f = tid / (outputSize * outputSize);
        int i = tid % (outputSize * outputSize) / outputSize;
        int j = tid % outputSize;

        float sum = 0.0f;

        // Iterate over kernel and input matrix
        for (int k = 0; k < kernelSize; k++) {
            for (int l = 0; l < kernelSize; l++) {
                for (int c = 0; c < inputChannels; c++) {
                    int kernelIndex =
                        f * kernelSize * kernelSize * inputChannels +
                        c * kernelSize * kernelSize + k * kernelSize + l;
                    int inputIndex = c * inputSize * inputSize +
                                     (i * stride + k) * inputSize +
                                     (j * stride + l);

                    sum += weights[kernelIndex] * input[inputIndex];
                }
            }
        }

        int outputIndex = f * outputSize * outputSize + i * outputSize + j;

        output[outputIndex] = sum;
    }
}