#include "add.hpp"

#include <stddef.h>

using namespace CUDANet::Layers;


Add::Add(int inputSize)
    : inputSize(inputSize) {

    output = new float[inputSize];

#ifdef USE_CUDA
    initCUDA();
#endif
    
}


Add::~Add() {
#ifdef USE_CUDA
    delCUDA();
#endif
}


float* Add::forward(const float* inputA, const float* inputB) {

#ifdef USE_CUDA
    return forwardCUDA(inputA, inputB);
#else
    return forwardCPU(inputA, inputB);
#endif

}

float* Add::forwardCPU(const float* inputA, const float* inputB) {
    for (size_t i = 0; i < inputSize; i++)
    {
        output[i] = inputA[i] + inputB[i];
    }

    return output;    
}