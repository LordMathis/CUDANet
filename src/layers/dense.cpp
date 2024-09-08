#include "dense.hpp"

#include <stdexcept>

#include "activation.hpp"

using namespace CUDANet::Layers;

Dense::Dense(int inputSize, int outputSize, ActivationType activationType)
    : inputSize(inputSize), outputSize(outputSize) {
    // Allocate memory for weights and biases
    weights.resize(outputSize * inputSize);
    biases.resize(outputSize);

    initializeWeights();
    initializeBiases();

    activation = new Activation(activationType, outputSize);

#ifdef USE_CUDA
    initCUDA();
#endif
}

Dense::~Dense() {
    delete activation;
#ifdef USE_CUDA
    delCUDA();
#endif
}

void Dense::initializeWeights() {
    std::fill(weights.begin(), weights.end(), 0.0f);
}

void Dense::initializeBiases() {
    std::fill(biases.begin(), biases.end(), 0.0f);
}

float* Dense::forwardCPU(const float* input) {
    throw std::logic_error("Not implemented");
}

float* Dense::forward(const float* input) {
#ifdef USE_CUDA
    return forwardCUDA(input);
#else
    return forwardCPU(input);
#endif
}

void Dense::setWeights(const float* weights_input) {
    std::copy(weights_input, weights_input + weights.size(), weights.begin());
#ifdef USE_CUDA
    toCuda();
#endif
}

std::vector<float> Dense::getWeights() {
    return weights;
}

void Dense::setBiases(const float* biases_input) {
    std::copy(biases_input, biases_input + biases.size(), biases.begin());
#ifdef USE_CUDA
    toCuda();
#endif
}

std::vector<float> Dense::getBiases() {
    return biases;
}

int Dense::getOutputSize() {
    return outputSize;
}

int Dense::getInputSize() {
    return inputSize;
}