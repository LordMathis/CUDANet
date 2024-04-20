#include "activation.cuh"
#include "conv2d.cuh"
#include "convolution.cuh"
#include "cuda_helper.cuh"
#include "matmul.cuh"

using namespace CUDANet::Layers;

Conv2d::Conv2d(
    int                    inputSize,
    int                    inputChannels,
    int                    kernelSize,
    int                    stride,
    int                    numFilters,
    int        paddingSize,
    ActivationType activationType
)
    : inputSize(inputSize),
      inputChannels(inputChannels),
      kernelSize(kernelSize),
      stride(stride),
      numFilters(numFilters),
      paddingSize(paddingSize) {

    outputSize = (inputSize - kernelSize + 2 * paddingSize) / stride + 1;

    activation = Activation(
        activationType, outputSize * outputSize * numFilters
    );

    d_output = nullptr;
    CUDA_CHECK(cudaMalloc(
        (void**)&d_output, sizeof(float) * outputSize * outputSize * numFilters
    ));

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

    toCuda();
}

Conv2d::~Conv2d() {
    cudaFree(d_output);
    cudaFree(d_weights);
    cudaFree(d_biases);
}

void Conv2d::initializeWeights() {
    std::fill(weights.begin(), weights.end(), 0.0f);
}

void Conv2d::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
}

void Conv2d::setWeights(const float* weights_input) {
    std::copy(weights_input, weights_input + weights.size(), weights.begin());
    toCuda();
}

std::vector<float> Conv2d::getWeights() {
    return weights;
}

void Conv2d::setBiases(const float* biases_input) {
    std::copy(biases_input, biases_input + biases.size(), biases.begin());
    toCuda();
}

std::vector<float> Conv2d::getBiases() {
    return biases;
}

void Conv2d::toCuda() {
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

float* Conv2d::forward(const float* d_input) {
    // Convolve
    dim3 block(8,8,8);
    dim3 grid(
        (outputSize + block.x - 1) / block.x,
        (outputSize + block.y - 1) / block.y,
        (numFilters + block.z - 1) / block.z
    );

    Kernels::convolution<<<grid, block>>>(
        d_input, d_weights, d_output, inputSize, inputChannels, paddingSize,
        kernelSize, stride, numFilters, outputSize
    );

    // Add bias
    Kernels::vec_vec_add<<<1, biases.size()>>>(
        d_biases, d_output, d_output, biases.size()
    );

    // Apply activation
    activation.activate(d_output);

    CUDA_CHECK(cudaDeviceSynchronize());

    return d_output;
}
