#include <stdexcept>
#include <vector>

#include "activation.hpp"

using namespace CUDANet::Layers;

Activation::Activation(ActivationType activation, const int length)
    : activationType(activation), length(length) {
#ifdef USE_CUDA
    initCUDA();
#endif
}

Activation::~Activation() {
#ifdef USE_CUDA
    delCUDA();
#endif
}

void Activation::activateCPU(float* input) {
    throw std::logic_error("Not implemented");
}

void Activation::activate(float* input) {
#ifdef USE_CUDA
    activateCUDA(input);
#else
    activateCPU(input);
#endif
}