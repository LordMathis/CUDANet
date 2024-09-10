#include "batch_norm.hpp"

#include <stdexcept>
#include <vector>

#include "activation.hpp"
#include "layer.hpp"

using namespace CUDANet::Layers;

BatchNorm2d::BatchNorm2d(
    shape2d        inputSize,
    int            inputChannels,
    float          epsilon,
    ActivationType activationType
)
    : inputSize(inputSize), inputChannels(inputChannels), epsilon(epsilon) {
    activation = new Activation(
        activationType, inputSize.first * inputSize.second * inputChannels
    );

    weights.resize(inputChannels);
    biases.resize(inputChannels);

    running_mean.resize(inputChannels);
    running_var.resize(inputChannels);

    initializeWeights();
    initializeBiases();
    initializeRunningMean();
    initializeRunningVar();

#ifdef USE_CUDA
    initCUDA();
    toCuda();
#endif
}

BatchNorm2d::~BatchNorm2d() {
#ifdef USE_CUDA
    delCUDA();
#endif
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

void BatchNorm2d::setWeights(const float* weights_input) {
    std::copy(weights_input, weights_input + weights.size(), weights.begin());
#ifdef USE_CUDA
    toCuda();
#endif
}

std::vector<float> BatchNorm2d::getWeights() {
    return weights;
}

void BatchNorm2d::setBiases(const float* biases_input) {
    std::copy(biases_input, biases_input + biases.size(), biases.begin());
#ifdef USE_CUDA
    toCuda();
#endif
}

std::vector<float> BatchNorm2d::getBiases() {
    return biases;
}

void BatchNorm2d::setRunningMean(const float* running_mean_input) {
    std::copy(
        running_mean_input, running_mean_input + inputChannels,
        running_mean.begin()
    );
#ifdef USE_CUDA
    toCuda();
#endif
}

std::vector<float> BatchNorm2d::getRunningMean() {
    return running_mean;
}

void BatchNorm2d::setRunningVar(const float* running_var_input) {
    std::copy(
        running_var_input, running_var_input + inputChannels,
        running_var.begin()
    );
#ifdef USE_CUDA
    toCuda();
#endif
}

std::vector<float> BatchNorm2d::getRunningVar() {
    return running_var;
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

float* BatchNorm2d::forwardCPU(const float* input) {
    throw std::logic_error("Not implemented");
}

float* BatchNorm2d::forward(const float* input) {
#ifdef USE_CUDA
    return forwardCUDA(input);
#else
    return forwardCPU(input);
#endif
}