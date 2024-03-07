#include <string>

#include "activations.cuh"
#include "conv2d.cuh"
#include "cuda_helper.cuh"
#include "padding.cuh"

Layers::Conv2d::Conv2d(
    int         inputSize,
    int         inputChannels,
    int         kernelSize,
    int         stride,
    std::string padding,
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
        CUDA_CHECK(cudaMalloc(
            (void**)&d_padded, sizeof(float) * (inputSize + 2 * paddingSize) *
                                   (inputSize + 2 * paddingSize) * inputChannels
        ));
    }
}

Layers::Conv2d::~Conv2d() {
    cudaFree(d_kernels);
    cudaFree(d_padded);
}

void Layers::Conv2d::initializeKernels() {
    std::fill(kernels.begin(), kernels.end(), 0.0f);
}

void Layers::Conv2d::setKernels(const std::vector<float>& kernels_input) {
    std::copy(kernels_input.begin(), kernels_input.end(), kernels.begin());
    toCuda();
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
    int BLOCKS =
        (outputSize * outputSize * inputChannels) / THREADS_PER_BLOCK + 1;

    pad_matrix_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
        d_input, d_padded, inputSize, inputSize, inputChannels, paddingSize
    );

    // TODO: Implement 2D convolution
}

/*
Convolves input vector with kernel and stores result in output

input: matrix (inputSize + paddingSize) x (inputSize + paddingSize) x
inputChannels represented as a vector output: output matrix outputSize x
outputSize x numFilters

*/
void Layers::Conv2d::host_conv(const float* input, float* output) {
    // Iterate over output matrix
    for (int f = 0; f < numFilters; f++) {
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                
                float sum = 0.0f;

                // Iterate over kernel and input matrix
                for (int k = 0; k < kernelSize; k++) {
                    for (int l = 0; l < kernelSize; l++) {
                        for (int c = 0; c < inputChannels; c++) {

                            // For now stride = 1
                            
                            int kernelIndex = k * (kernelSize * inputChannels * numFilters) + l * (inputChannels * numFilters) + c * (numFilters) + f;
                            int inputIndex  = (i * stride + k) * (inputSize * inputChannels) + (j + stride + l) * (inputChannels) + c;

                            sum += kernels[kernelIndex] * input[inputIndex];
                        }                      
                    }
                }

                output[i * (outputSize * numFilters) + j * (numFilters) + f] = sum;
            }
        }
    }
}