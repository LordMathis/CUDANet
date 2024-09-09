#include <stdexcept>

#include "avg_pooling.hpp"

using namespace CUDANet::Layers;

AvgPooling2d::AvgPooling2d(
    shape2d        inputSize,
    int            nChannels,
    shape2d        poolingSize,
    shape2d        stride,
    shape2d        padding,
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

AvgPooling2d::~AvgPooling2d() {
#ifdef USE_CUDA
    delCUDA();
#endif
    delete activation;
}

float* AvgPooling2d::forwardCPU(const float* input) {
    throw std::logic_error("Not implemented");
}

float* AvgPooling2d::forward(const float* input) {
#ifdef USE_CUDA
    return forwardCUDA(input);
#else
    return forwardCPU(input);
#endif
}

int AvgPooling2d::getOutputSize() {
    return outputSize.first * outputSize.second * nChannels;
}

int AvgPooling2d::getInputSize() {
    return inputSize.first * inputSize.second * nChannels;
}

shape2d AvgPooling2d::getOutputDims() {
    return outputSize;
}

AdaptiveAvgPooling2d::AdaptiveAvgPooling2d(
    shape2d        inputShape,
    int            nChannels,
    shape2d        outputShape,
    ActivationType activationType
)
    : AvgPooling2d(
          inputShape,
          nChannels,
          {1, 1},
          {1, 1},
          {0, 0},
          activationType
      ) {
    stride = {
        inputShape.first / outputShape.first,
        inputShape.second / outputShape.second
    };
    poolingSize = {
        inputShape.first - (outputShape.first - 1) * stride.first,
        inputShape.second - (outputShape.second - 1) * stride.second
    };
    padding    = {(poolingSize.first - 1) / 2, (poolingSize.second - 1) / 2};
    outputSize = outputShape;

    activation = new Activation(
        activationType, outputSize.first * outputSize.second * nChannels
    );

#ifdef USE_CUDA
    initCUDA();
#endif
}