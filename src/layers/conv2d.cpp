#include <stdexcept>
#include <vector>

#include "activation.hpp"
#include "conv2d.hpp"
#include "layer.hpp"

using namespace CUDANet::Layers;

Conv2d::Conv2d(
    shape2d        inputSize,
    int            inputChannels,
    shape2d        kernelSize,
    shape2d        stride,
    int            numFilters,
    shape2d        paddingSize,
    ActivationType activationType
)
    : inputSize(inputSize),
      inputChannels(inputChannels),
      kernelSize(kernelSize),
      stride(stride),
      numFilters(numFilters),
      paddingSize(paddingSize) {
    outputSize = {
        (inputSize.first - kernelSize.first + 2 * paddingSize.first) /
                stride.first +
            1,
        (inputSize.second - kernelSize.second + 2 * paddingSize.second) /
                stride.second +
            1
    };

    activation = new Activation(
        activationType, outputSize.first * outputSize.second * numFilters
    );

    weights.resize(
        kernelSize.first * kernelSize.second * inputChannels * numFilters
    );
    initializeWeights();

    biases.resize(numFilters);
    initializeBiases();

#ifdef USE_CUDA
    initCUDA();
    toCuda();
#endif
}

Conv2d::~Conv2d() {
#ifdef USE_CUDA
    delCUDA();
#endif
    delete activation;
}

void Conv2d::initializeWeights() {
    std::fill(weights.begin(), weights.end(), 0.0f);
}

void Conv2d::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
}

void Conv2d::setWeights(const float* weights_input) {
    std::copy(weights_input, weights_input + weights.size(), weights.begin());
#ifdef USE_CUDA
    toCuda();
#endif
}

std::vector<float> Conv2d::getWeights() {
    return weights;
}

void Conv2d::setBiases(const float* biases_input) {
    std::copy(biases_input, biases_input + biases.size(), biases.begin());
#ifdef USE_CUDA
    toCuda();
#endif
}

std::vector<float> Conv2d::getBiases() {
    return biases;
}

float* Conv2d::forwardCPU(const float* input) {
    throw std::logic_error("Not implemented");
}

float* Conv2d::forward(const float* input) {
#ifdef USE_CUDA
    return forwardCUDA(input);
#else
    return forwardCPU(input);
#endif
}

int Conv2d::getOutputSize() {
    return outputSize.first * outputSize.second * numFilters;
}

int Conv2d::getInputSize() {
    return inputSize.first * inputSize.second * inputChannels;
}

shape2d Conv2d::getOutputDims() {
    return outputSize;
}