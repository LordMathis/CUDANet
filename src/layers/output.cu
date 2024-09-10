#include "output.hpp"
#include <stdexcept>

using namespace CUDANet::Layers;


Output::Output(int inputSize) : inputSize(inputSize) {
    h_output = (float*) malloc(sizeof(float) * inputSize);
}

Output::~Output() {
    free(h_output);
}

float* Output::forwardCPU(const float* input) {
    throw std::logic_error("Not implemented");
}

float* Output::forward(const float* input) {
#ifdef USE_CUDA
    return forwardCUDA(input);
#else
    return forwardCPU(input);
#endif
}

int Output::getOutputSize() {
    return inputSize;
}


int Output::getInputSize() {
    return inputSize;
}