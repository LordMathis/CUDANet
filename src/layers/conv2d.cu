#include <cublas_v2.h>

#include <string>

#include "activations.cuh"
#include "conv2d.cuh"
#include "cuda_helper.cuh"
#include "padding.cuh"

Layers::Conv2d::Conv2d(
    int            inputSize,
    int            inputChannels,
    int            kernelSize,
    int            stride,
    std::string    padding,
    int            numFilters,
    Activation     activation,
    cublasHandle_t cublasHandle
)
    : inputSize(inputSize),
      inputChannels(inputChannels),
      kernelSize(kernelSize),
      stride(stride),
      numFilters(numFilters),
      cublasHandle(cublasHandle),
      activation(activation) {
    // Allocate memory for kernels

    if (padding == "SAME") {
        outputSize  = inputSize;
        paddingSize = ((stride - 1) * inputSize - stride + kernelSize) / 2;
    } else if (padding == "VALID") {
        paddingSize = 0;
        outputSize  = (inputSize - kernelSize) / stride + 1;
    }

    kernels.resize(kernelSize * kernelSize);
    initializeKernels();

    d_kernels = nullptr;

    CUDA_CHECK(
        cudaMalloc((void**)&d_kernels, sizeof(float) * kernelSize * kernelSize)
    );
    toCuda();

    d_padded = nullptr;

    if (paddingSize > 0) {
        CUDA_CHECK(
            cudaMalloc((void**)&d_padded,
                       sizeof(float) * (inputSize + 2 * paddingSize) *
                           (inputSize + 2 * paddingSize) * inputChannels)
        );
    }
}

Layers::Conv2d::~Conv2d() {
    cudaFree(d_kernels);
    cudaFree(d_padded);
}

void Layers::Conv2d::initializeKernels() {
    std::fill(kernels.begin(), kernels.end(), 0.0f);
}

void Layers::Conv2d::toCuda() {
    CUDA_CHECK(cudaMemcpy(
        d_kernels, kernels.data(), sizeof(float) * kernelSize * kernelSize,
        cudaMemcpyHostToDevice
    ));
}

void Layers::Conv2d::forward(const float* d_input, float* d_output) {

    // Padd input
    int THREADS_PER_BLOCK = 256;
    int BLOCKS            = (outputSize * outputSize * inputChannels) / THREADS_PER_BLOCK + 1;

    pad_matrix_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
        d_input, d_padded, inputSize, inputSize, inputChannels, paddingSize
    );

    // TODO: Implement 2D convolution

}