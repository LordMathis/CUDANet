#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <iostream>

#include "conv2d.cuh"

TEST(Conv2dTest, ValidPadding) {

    int inputSize = 3;
    int inputChannels = 1;
    int kernelSize = 3;
    int stride = 1;
    std::string padding = "VALID";
    int numFilters = 1;
    Activation activation = LINEAR;

    Layers::Conv2d conv2d(
        inputSize,
        inputChannels,
        kernelSize,
        stride,
        padding,
        numFilters,
        activation
    );

    int outputSize = (inputSize - kernelSize) / stride + 1;
    EXPECT_EQ(outputSize, conv2d.outputSize);

    std::vector<float> input(inputSize * inputSize * inputChannels);
    std::vector<float> output(outputSize * outputSize * numFilters);
    std::vector<float> kernels(kernelSize * kernelSize * inputChannels * numFilters);

}
