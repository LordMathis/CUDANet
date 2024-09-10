#include <stdexcept>

#include "input.hpp"

using namespace CUDANet::Layers;

Input::Input(int inputSize) : inputSize(inputSize) {
#ifdef USE_CUDA
    initCUDA();
#endif
}

Input::~Input() {
#ifdef USE_CUDA
    delCUDA();
#endif
}

float* Input::forwardCPU(const float* input) {
    throw std::logic_error("Not implemented");
}

float* Input::forward(const float* input) {
#ifdef USE_CUDA
    return forwardCUDA(input);
#else
    return forwardCPU(input);
#endif
}

int Input::getOutputSize() {
    return inputSize;
}

int Input::getInputSize() {
    return inputSize;
}