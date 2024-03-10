#include <iostream>
#include <string>

#include "activations.cuh"
#include "conv2d.cuh"
#include "convolution.cuh"
#include "cuda_helper.cuh"
#include "matrix_math.cuh"
#include "padding.cuh"

Layers::Conv2d::Conv2d(
    int         inputSize,
    int         inputChannels,
    int         kernelSize,
    int         stride,
    Padding padding,
    int         numFilters,
    Activation  activation
)
    : inputSize(inputSize),
      inputChannels(inputChannels),
      kernelSize(kernelSize),
      stride(stride),
      numFilters(numFilters),
      activation(activation) {
    // Allocate memory for kernels

    switch (padding)
    {
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

    kernels.resize(kernelSize * kernelSize * inputChannels * numFilters);
    initializeKernels();

    d_kernels = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_kernels,
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
    cudaFree(d_kernels);
    cudaFree(d_biases);
    cudaFree(d_padded);
}

void Layers::Conv2d::initializeKernels() {
    std::fill(kernels.begin(), kernels.end(), 0.0f);
}

void Layers::Conv2d::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
}

void Layers::Conv2d::setKernels(const std::vector<float>& kernels_input) {
    std::copy(kernels_input.begin(), kernels_input.end(), kernels.begin());
    toCuda();
}

void Layers::Conv2d::toCuda() {
    CUDA_CHECK(cudaMemcpy(
        d_kernels, kernels.data(),
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

    pad_matrix_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_input, d_padded, inputSize, inputSize, inputChannels, paddingSize
    );

    // Convolve
    THREADS_PER_BLOCK = outputSize * outputSize * numFilters;
    convolution_kernel<<<1, THREADS_PER_BLOCK>>>(
        d_padded, d_kernels, d_output, inputSize + (2 * paddingSize),
        inputChannels, kernelSize, stride, numFilters, outputSize
    );

    // Add bias
    vec_vec_add_kernel<<<1, biases.size()>>>(
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
    for (int tid = 0; tid < outputSize * outputSize * numFilters; tid++)
    {  
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
                        c * kernelSize * kernelSize + k * kernelSize +
                        l;
                    int inputIndex = c * inputSize * inputSize +
                                        (i * stride + k) * inputSize +
                                        (j * stride + l);

                    sum += kernels[kernelIndex] * input[inputIndex];
                }
            }
        }

        int outputIndex =
            f * outputSize * outputSize + i * outputSize + j;

        output[outputIndex] = sum;
    }
}