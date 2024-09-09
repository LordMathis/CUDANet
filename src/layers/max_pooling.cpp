#include "max_pooling.hpp"
#include <stdexcept>

using namespace CUDANet::Layers;

MaxPooling2d::MaxPooling2d(
    shape2d          inputSize,
    int            nChannels,
    shape2d          poolingSize,
    shape2d          stride,
    shape2d          padding,
    ActivationType activationType
)
    : inputSize(inputSize),
      nChannels(nChannels),
      poolingSize(poolingSize),
      stride(stride),
      padding(padding) {
    outputSize = {
        (inputSize.first + 2 * padding.first - poolingSize.first) /
                stride.first +
            1,
        (inputSize.second + 2 * padding.second - poolingSize.second) /
                stride.second +
            1
    };

    activation = new Activation(
        activationType, outputSize.first * outputSize.second * nChannels
    );

    #ifdef USE_CUDA
    initCUDA();
#endif
}

MaxPooling2d::~MaxPooling2d() {
#ifdef USE_CUDA
    delCUDA();
#endif
    delete activation;
}

float* MaxPooling2d::forwardCPU(const float* input) {
    throw std::logic_error("Not implemented");
}

float* MaxPooling2d::forward(const float* input) {
#ifdef USE_CUDA
    return forwardCUDA(input);
#else
    return forwardCPU(input);
#endif
}


int MaxPooling2d::getOutputSize() {
    return outputSize.first * outputSize.second * nChannels;
}

int MaxPooling2d::getInputSize() {
    return inputSize.first * inputSize.second * nChannels;
}

shape2d MaxPooling2d::getOutputDims() {
    return outputSize;
}