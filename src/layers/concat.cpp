#include <stdexcept>

#include "concat.hpp"

using namespace CUDANet::Layers;

Concat::Concat(const int inputASize, const int inputBSize)
    : inputASize(inputASize), inputBSize(inputBSize) {
#ifdef USE_CUDA
    initCUDA();
#endif
}

Concat::~Concat() {
#ifdef USE_CUDA
    delCUDA();
#endif
}

float* Concat::forwardCPU(const float* input_A, const float* input_B) {
    throw std::logic_error("Not implemented");
}

float* Concat::forward(const float* input_A, const float* input_B) {
#ifdef USE_CUDA
    return forwardCUDA(input_A, input_B);
#else
    return forwardCPU(input_A, input_B);
#endif
}

int Concat::getOutputSize() {
    return inputASize + inputBSize;
};
